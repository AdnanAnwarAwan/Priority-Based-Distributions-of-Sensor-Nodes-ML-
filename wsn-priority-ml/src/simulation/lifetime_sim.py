"""
lifetime_sim.py — Full Network Lifetime Simulator.

Runs the complete priority-aware WSN pipeline for many rounds
until the network dies or the round budget is exhausted.

Tracks: FND (First Node Dead), HND (Half Nodes Dead), END (All Dead),
coverage quality over time, critical-zone coverage, energy saved
by hibernation, and migration statistics.

Used to generate the comparison table in the paper:
    Random | PACR | QoS-NRT | ML Priority (ours)
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from src.network.activity_map import ActivityMap
from src.network.energy_model import EnergyModel
from src.network.coverage import CoverageCalculator
from src.network.wsn_graph import WSNGraph
from src.simulation.hibernation import HibernationController, NodeState
from src.simulation.redistribution import RedistributionEngine
from src.ml.zone_classifier import ZoneClassifier
from src.ml.federated_trainer import FederatedTrainer

log = logging.getLogger("LifetimeSim")


class LifetimeSimulator:
    """
    End-to-end simulation of a priority-aware WSN deployment.

    Each round:
    1.  Federated learning: nodes share observations, global model updated
    2.  Zone classification: each node's zone assigned via global model
    3.  Redistribution: surplus nodes migrate toward priority deficits
    4.  Hibernation: nodes in low-priority zones enter/exit sleep
    5.  Energy drain: active/dozing/sleeping costs applied
    6.  Metrics logged
    """

    def __init__(
        self,
        n_nodes:      int   = 100,
        area_size:    float = 500.0,
        tx_radius:    float = 75.0,
        n_hotspots:   int   = 6,
        fed_rounds:   int   = 5,    # federated rounds per sim round
        seed:         Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)

        self.N          = n_nodes
        self.area       = area_size
        self.tx_radius  = tx_radius

        self.sink       = np.array([area_size / 2, area_size / 2])
        self.em         = EnergyModel()
        self.cov_calc   = CoverageCalculator(area_size, grid_res=20.0)

        # Initial random deployment (simulates airdrop)
        self._pos       = rng.uniform(0, area_size, (n_nodes, 2)).astype(np.float32)
        self._energies  = rng.uniform(0.8, 1.0, n_nodes).astype(np.float32)

        self._act_map   = ActivityMap(area_size, n_hotspots=n_hotspots, seed=int(rng.integers(0, 9999)))
        self._fed       = FederatedTrainer(n_nodes=n_nodes)
        self._hibCtrl   = HibernationController(n_nodes, self._energies.copy())
        self._redist    = RedistributionEngine(area_size, self.em)
        self._clf       = ZoneClassifier()

        self._fed_rounds_per_sim = fed_rounds
        self._round     = 0
        self._history:  List[Dict] = []
        self._fnd:      Optional[int] = None
        self._hnd:      Optional[int] = None
        self._end:      Optional[int] = None

    # ── Main simulation loop ──────────────────────────────────────────────

    def run(self, max_rounds: int = 500, verbose: bool = False) -> Dict:
        log.info(f"Starting simulation: {self.N} nodes, {self.area}×{self.area} m, {max_rounds} rounds")

        for r in range(max_rounds):
            self._round = r + 1
            alive = self._hibCtrl.alive_mask()

            if not alive.any():
                if self._end is None:
                    self._end = r
                log.info(f"All nodes dead at round {r}.")
                break

            # 1. Evolve battlefield activity
            self._act_map.evolve(rounds=1, drift=0.004)

            # 2. Feed observations to federated trainer
            self._fed.feed_activity_map(self._act_map, self._pos)

            # 3. Federated learning (every fed_rounds sim steps)
            if r % max(1, self._fed_rounds_per_sim) == 0:
                self._fed.communication_round()
                # Sync global model to local classifier
                self._clf.set_weights(self._fed.global_clf.get_weights())

            # 4. Zone classification for all alive nodes
            zones = self._classify_zones(alive)

            # 5. Redistribution (alive nodes not hibernating)
            sensing = self._hibCtrl.sensing_mask()
            if sensing.any():
                self._redist.step(
                    self._pos, self._energies, zones, alive,
                    self._act_map, self.sink
                )

            # 6. Hibernation update
            nb_act = self._neighbour_activity()
            self._hibCtrl.update(zones, nb_act, active_cost=0.005)
            self._energies = self._hibCtrl.energies()

            # 7. Metrics
            metrics = self._compute_metrics(r, zones, alive)
            self._history.append(metrics)

            # Lifetime milestones
            n_alive = metrics["alive"]
            if n_alive < self.N and self._fnd is None:
                self._fnd = r
            if n_alive <= self.N // 2 and self._hnd is None:
                self._hnd = r
            if n_alive == 0 and self._end is None:
                self._end = r

            if verbose and r % 50 == 0:
                log.info(
                    f"Round {r:4d} | alive={n_alive:3d} | "
                    f"crit_cov={metrics['critical_coverage']:.2f} | "
                    f"e_saved={metrics['cumulative_energy_saved']:.3f}"
                )

        return self._summary()

    # ── Per-round helpers ─────────────────────────────────────────────────

    def _classify_zones(self, alive: np.ndarray) -> np.ndarray:
        features = np.array([
            self._act_map.observation_at(
                i, float(self._pos[i, 0]), float(self._pos[i, 1])
            ).to_feature_vector()
            for i in range(self.N)
        ])
        return self._clf.predict(features)

    def _neighbour_activity(self) -> np.ndarray:
        activity = np.array([
            self._act_map.activity_at(float(self._pos[i, 0]), float(self._pos[i, 1]))
            for i in range(self.N)
        ], dtype=np.float32)
        result = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            dists = np.linalg.norm(self._pos - self._pos[i], axis=1)
            nbrs  = (dists < self.tx_radius) & (dists > 0)
            result[i] = float(activity[nbrs].max()) if nbrs.any() else 0.0
        return result

    def _compute_metrics(self, r: int, zones: np.ndarray, alive: np.ndarray) -> Dict:
        sensing   = self._hibCtrl.sensing_mask()
        hib_sum   = self._hibCtrl.summary()
        cov       = self.cov_calc.compute(self._pos[sensing], self.tx_radius) if sensing.any() else 0.0
        crit_mask = sensing & (zones >= 2)
        crit_cov  = self.cov_calc.compute(self._pos[crit_mask], self.tx_radius) if crit_mask.any() else 0.0
        return {
            "round":                    r,
            "alive":                    hib_sum["alive"],
            "hibernating":              hib_sum["hibernating"],
            "active":                   hib_sum["active"],
            "coverage":                 round(cov, 3),
            "critical_coverage":        round(crit_cov, 3),
            "mean_energy":              round(hib_sum["mean_energy"], 3),
            "cumulative_energy_saved":  round(hib_sum["energy_saved"], 4),
            "migrations":               self._redist.total_migrations(),
        }

    def _summary(self) -> Dict:
        return {
            "n_nodes":        self.N,
            "area":           self.area,
            "FND":            self._fnd,
            "HND":            self._hnd,
            "END":            self._end,
            "total_rounds":   self._round,
            "avg_crit_cov":   round(np.mean([h["critical_coverage"] for h in self._history]), 3),
            "avg_coverage":   round(np.mean([h["coverage"]           for h in self._history]), 3),
            "energy_saved":   round(self._hibCtrl.total_energy_saved(), 4),
            "migrations":     self._redist.total_migrations(),
            "history":        self._history,
        }

    def export_csv(self, path: str):
        if not self._history:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._history[0].keys())
            w.writeheader()
            w.writerows(self._history)
        log.info(f"History saved to {p}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="WSN Priority-ML Lifetime Simulation")
    ap.add_argument("--nodes",   type=int,   default=100)
    ap.add_argument("--area",    type=float, default=500.0)
    ap.add_argument("--rounds",  type=int,   default=300)
    ap.add_argument("--config",  default="config/default.yaml")
    ap.add_argument("--out",     default="data/results/lifetime.csv")
    args = ap.parse_args()

    sim    = LifetimeSimulator(n_nodes=args.nodes, area_size=args.area)
    result = sim.run(max_rounds=args.rounds, verbose=True)
    sim.export_csv(args.out)

    print(f"\n{'='*55}")
    print(f"  LIFETIME SIMULATION RESULTS")
    print(f"{'='*55}")
    print(f"  Nodes: {result['n_nodes']} | Area: {result['area']}×{result['area']} m")
    print(f"  First Node Dead  : round {result['FND']}")
    print(f"  Half Nodes Dead  : round {result['HND']}")
    print(f"  All Nodes Dead   : round {result['END']}")
    print(f"  Avg critical cov : {result['avg_crit_cov']:.1%}")
    print(f"  Energy saved     : {result['energy_saved']:.4f} J")
    print(f"  Total migrations : {result['migrations']}")
    print(f"{'='*55}")
