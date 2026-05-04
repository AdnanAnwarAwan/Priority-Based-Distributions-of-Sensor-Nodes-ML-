"""
run_pipeline.py — End-to-end WSN Priority-ML pipeline.

Steps:
  1. Train zone classifier (supervised, synthetic maps)
  2. Run federated learning (nodes share observations)
  3. Train RL redistribution agent
  4. Run full lifetime simulation (ML method)
  5. Run all baselines (random, PACR, QoS-NRT)
  6. Print and export comparison table

Usage
-----
python scripts/run_pipeline.py
python scripts/run_pipeline.py --nodes 100 --rounds 300 --skip-rl
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("Pipeline")


def step_train_classifier(cfg):
    log.info("=== Step 1: Train Zone Classifier ===")
    from src.ml.train_zone_classifier import train
    train(cfg)


def step_federated(cfg):
    log.info("=== Step 2: Federated Learning ===")
    import numpy as np
    from src.ml.federated_trainer import FederatedTrainer
    from src.network.activity_map import ActivityMap

    area    = cfg["env"]["area_size"]
    n_nodes = cfg["env"]["n_nodes"]
    rounds  = cfg["federated"]["comm_rounds"]

    pos     = np.random.uniform(0, area, (n_nodes, 2)).astype("float32")
    amap    = ActivityMap(area_size=area, n_hotspots=5)
    trainer = FederatedTrainer(n_nodes=n_nodes,
                               local_epochs=cfg["federated"]["local_epochs"])
    results = trainer.run(n_rounds=rounds, activity_map=amap, positions=pos)
    acc     = trainer.global_accuracy(amap)
    log.info(f"Federated training complete. Global accuracy: {acc:.3f}")
    trainer.save_global_model(cfg["training"]["checkpoint_dir"] + "/federated_zone_clf.pt")


def step_train_rl(cfg):
    log.info("=== Step 3: Train RL Redistribution Agent ===")
    from src.ml.train_rl_agent import train
    train(cfg)


def step_lifetime_ml(cfg) -> dict:
    log.info("=== Step 4: ML Lifetime Simulation ===")
    from src.simulation.lifetime_sim import LifetimeSimulator
    sim = LifetimeSimulator(
        n_nodes    = cfg["env"]["n_nodes"],
        area_size  = cfg["env"]["area_size"],
        tx_radius  = cfg["env"]["tx_radius"],
        fed_rounds = cfg["simulation"]["fed_rounds"],
    )
    res = sim.run(max_rounds=cfg["simulation"]["max_rounds"], verbose=True)
    sim.export_csv(cfg["output"]["results_dir"] + "ml_lifetime.csv")
    return res


def step_baselines(cfg) -> dict:
    log.info("=== Step 5: Baseline Simulations ===")
    from src.simulation.baseline_sim import RandomBaseline, PACRBaseline, QoSNRTBaseline
    kw = dict(n_nodes=cfg["env"]["n_nodes"], area_size=cfg["env"]["area_size"])
    mr = cfg["simulation"]["max_rounds"]
    results = {}
    for name, cls in [("Random", RandomBaseline), ("PACR", PACRBaseline), ("QoS-NRT", QoSNRTBaseline)]:
        log.info(f"Running {name}...")
        sim = cls(**kw)
        results[name] = sim.run(max_rounds=mr)
    return results


def print_comparison(ml_res: dict, baselines: dict, cfg: dict):
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, res in baselines.items():
        rows.append({
            "Method":           name,
            "FND":              res["FND"],
            "HND":              res["HND"],
            "END":              res["END"],
            "Avg Critical Cov": f"{res['avg_crit_cov']:.1%}",
            "Avg Coverage":     f"{res['avg_coverage']:.1%}",
        })
    rows.append({
        "Method":           "ML Priority (ours)",
        "FND":              ml_res["FND"],
        "HND":              ml_res["HND"],
        "END":              ml_res["END"],
        "Avg Critical Cov": f"{ml_res['avg_crit_cov']:.1%}",
        "Avg Coverage":     f"{ml_res['avg_coverage']:.1%}",
    })

    with open(out_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"\n{'='*70}")
    print(f"  PRIORITY-AWARE WSN ML — PIPELINE RESULTS")
    print(f"  Nodes: {cfg['env']['n_nodes']}  |  Area: {cfg['env']['area_size']}×{cfg['env']['area_size']} m")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'FND':>6} {'HND':>6} {'END':>6} {'CritCov':>9} {'Coverage':>9}")
    print(f"  {'-'*64}")
    for r in rows:
        print(f"  {r['Method']:<22} {str(r['FND']):>6} {str(r['HND']):>6} "
              f"{str(r['END']):>6} {r['Avg Critical Cov']:>9} {r['Avg Coverage']:>9}")
    print(f"{'='*70}")
    log.info(f"Comparison saved to {out_dir}/comparison.csv")


def main():
    ap = argparse.ArgumentParser(description="WSN Priority-ML Pipeline")
    ap.add_argument("--config",       default="config/default.yaml")
    ap.add_argument("--nodes",        type=int)
    ap.add_argument("--rounds",       type=int)
    ap.add_argument("--skip-clf",     action="store_true")
    ap.add_argument("--skip-fed",     action="store_true")
    ap.add_argument("--skip-rl",      action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.nodes:  cfg["env"]["n_nodes"]              = args.nodes
    if args.rounds: cfg["simulation"]["max_rounds"]    = args.rounds

    Path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["results_dir"]).mkdir(parents=True, exist_ok=True)

    if not args.skip_clf:       step_train_classifier(cfg)
    if not args.skip_fed:       step_federated(cfg)
    if not args.skip_rl:        step_train_rl(cfg)

    ml_res    = step_lifetime_ml(cfg)
    baselines = {} if args.skip_baselines else step_baselines(cfg)

    print_comparison(ml_res, baselines, cfg)


if __name__ == "__main__":
    main()
