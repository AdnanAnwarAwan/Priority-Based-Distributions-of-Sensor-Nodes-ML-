"""
train_rl_agent.py — RL training loop for the redistribution/hibernation agent.

Usage
-----
python src/ml/train_rl_agent.py --config config/default.yaml
python src/ml/train_rl_agent.py --nodes 100 --episodes 500
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from src.simulation.wsn_env import PriorityWSNEnv
from src.ml.rl_agent import DQNAgent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TrainRL")


def train(cfg: dict) -> list:
    env_cfg  = cfg["env"]
    ag_cfg   = cfg["agent"]
    tr_cfg   = cfg["training"]

    env = PriorityWSNEnv(
        n_nodes   = env_cfg["n_nodes"],
        area_size = env_cfg["area_size"],
        tx_radius = env_cfg["tx_radius"],
        max_steps = env_cfg["max_steps"],
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim   = state_dim,
        action_dim  = action_dim,
        hidden      = ag_cfg["hidden_dim"],
        lr          = ag_cfg["lr"],
        gamma       = ag_cfg["gamma"],
        tau         = ag_cfg["tau"],
        buffer_size = ag_cfg["buffer_size"],
        batch_size  = ag_cfg["batch_size"],
        eps_start   = ag_cfg["eps_start"],
        eps_end     = ag_cfg["eps_end"],
        eps_decay   = ag_cfg["eps_decay"],
    )

    ckpt_dir = Path(tr_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_ep       = tr_cfg["episodes"]
    log_every  = tr_cfg.get("log_every", 10)
    save_every = tr_cfg.get("save_every", 50)
    upd_every  = tr_cfg.get("update_every", 4)

    window    = deque(maxlen=50)
    best_avg  = -float("inf")
    records   = []
    t0        = time.time()

    log.info(f"Training RL agent: {n_ep} episodes | state={state_dim} | actions={action_dim}")

    for ep in range(1, n_ep + 1):
        obs, _    = env.reset()
        ep_reward = 0.0
        ep_loss   = 0.0
        ep_steps  = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done)
            if ep_steps % upd_every == 0:
                ep_loss += agent.update()

            obs        = next_obs
            ep_reward += reward
            ep_steps  += 1
            if done:
                break

        window.append(ep_reward)
        avg = float(np.mean(window))

        rec = {
            "episode":          ep,
            "reward":           round(ep_reward, 4),
            "avg50":            round(avg, 4),
            "critical_coverage":round(info.get("critical_coverage", 0), 3),
            "alive":            info.get("alive", 0),
            "hibernating":      info.get("hibernating", 0),
            "energy_saved":     round(info.get("energy_saved", 0), 4),
            "epsilon":          round(agent.epsilon, 4),
            "loss":             round(ep_loss / max(1, ep_steps // upd_every), 6),
            "elapsed_s":        round(time.time() - t0, 1),
        }
        records.append(rec)

        if avg > best_avg and ep > 20:
            best_avg = avg
            agent.save(str(ckpt_dir / "rl_agent.pt"))

        if ep % save_every == 0:
            agent.save(str(ckpt_dir / f"rl_ep{ep}.pt"))

        if ep % log_every == 0:
            log.info(
                f"ep {ep:4d}/{n_ep} | r={ep_reward:+.3f} | avg50={avg:+.3f} | "
                f"crit_cov={info.get('critical_coverage', 0):.2f} | "
                f"alive={info.get('alive', 0)} | ε={agent.epsilon:.3f}"
            )

    csv_path = ckpt_dir / "rl_training.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
    log.info(f"Done. Best avg={best_avg:.4f} | Results → {csv_path}")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train RL redistribution agent")
    ap.add_argument("--config",   default="config/default.yaml")
    ap.add_argument("--nodes",    type=int)
    ap.add_argument("--episodes", type=int)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.nodes:    cfg["env"]["n_nodes"]       = args.nodes
    if args.episodes: cfg["training"]["episodes"] = args.episodes
    train(cfg)
