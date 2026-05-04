"""
baseline_sim.py — Baseline simulations for comparison.

Implements:
  - Random deployment (no redistribution, no hibernation)
  - PACR-inspired reactive recovery (from prior work)
  - QoS-NRT-inspired relocation (from prior work)

All baselines use the same energy model and activity map
as the ML system to ensure fair comparison.

Usage
-----
python src/simulation/baseline_sim.py --nodes 100 --mode random
python src/simulation/baseline_sim.py --nodes 100 --mode pacr
python src/simulation/baseline_sim.py --nodes 100 --mode qos
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional

import numpy as np

from src.network.activity_map import ActivityMap
from src.network.energy_model import EnergyModel
from src.network.coverage import CoverageCalculator
from src.network.wsn_graph import WSNGraph

log = logging.getLogger("BaselineSim")


class BaselineSimulator:
    """
    Shared infrastructure for all baseline simulations.
    Subclasses override `_round_action` to apply their policy.
    """

    def __init__(
        self,
        n_nodes:   int   = 100,
        area_size: float = 500.0,
        tx_radius: float = 75.0,
        n_hotspots: int  = 6,
        seed:      Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        self.N         = n_nodes
        self.area      = area_size
        self.tx_radius = tx_radius
        self.em        = EnergyModel()
        self.cov_calc  = CoverageCalculator(area_size, grid_res=20.0)
        self.sink      = np.array([area_size / 2, area_size / 2])

        self._pos      = rng.uniform(0, area_size, (n_nodes, 2)).astype(np.float32)
        self._energies = rng.uniform(0.8, 1.0, n_nodes).astype(np.float32)
        self._alive    = np.ones(n_nodes, dtype=bool)
        self._act_map  = ActivityMap(area_size, n_hotspots=n_hotspots, seed=int(rng.integers(0, 9999)))

        self._round    = 0
        self._history: List[Dict] = []
        self._fnd: Optional[int] = None
        self._hnd: Optional[int] = None
        self._end: Optional[int] = None

    def run(self, max_rounds: int = 300, verbose: bool = False) -> Dict:
        for r in range(max_rounds):
            self._round = r
            if not self._alive.any():
                if self._end is None: self._end = r
                break

            self._act_map.evolve(rounds=1)
            self._round_action()
            self._drain_energy()
            self._alive = self._energies > 0

            metrics = self._metrics(r)
            self._history.append(metrics)

            n = self._alive.sum()
            if n < self.N and self._fnd is None:     self._fnd = r
            if n <= self.N // 2 and self._hnd is None: self._hnd = r
            if n == 0 and self._end is None:         self._end = r

            if verbose and r % 50 == 0:
                log.info(f"Round {r}: alive={n} cov={metrics['critical_coverage']:.2f}")

        return self._summary()

    def _round_action(self):
        pass  # Subclasses override

    def _drain_energy(self, active_cost: float = 0.005):
        for i in range(self.N):
            if not self._alive[i]: continue
            d = float(np.linalg.norm(self._pos[i] - self.sink))
            e = self.em.tx_energy(bits=4000, distance=d) * 1e6 * 0.001
            self._energies[i] = max(0.0, self._energies[i] - active_cost - e)

    def _metrics(self, r: int) -> Dict:
        sensing   = self._alive
        cov       = self.cov_calc.compute(self._pos[sensing], self.tx_radius) if sensing.any() else 0.0
        zones     = np.array([self._act_map.zone_at(float(x), float(y)) for x, y in self._pos])
        crit_mask = sensing & (zones >= 2)
        crit_cov  = self.cov_calc.compute(self._pos[crit_mask], self.tx_radius) if crit_mask.any() else 0.0
        return {
            "round":             r,
            "alive":             int(self._alive.sum()),
            "coverage":          round(float(cov), 3),
            "critical_coverage": round(float(crit_cov), 3),
            "mean_energy":       round(float(self._energies[self._alive].mean()) if self._alive.any() else 0.0, 3),
        }

    def _summary(self) -> Dict:
        return {
            "FND":          self._fnd,
            "HND":          self._hnd,
            "END":          self._end,
            "rounds":       self._round,
            "avg_crit_cov": round(np.mean([h["critical_coverage"] for h in self._history]), 3),
            "avg_coverage": round(np.mean([h["coverage"] for h in self._history]), 3),
        }


class RandomBaseline(BaselineSimulator):
    """No redistribution, no hibernation. Pure random deployment."""
    pass   # _round_action does nothing


class PACRBaseline(BaselineSimulator):
    """
    PACR-inspired: reactive recovery only.
    When a node dies, nearest alive neighbour moves to fill its position.
    No proactive redistribution, no hibernation.
    """

    def _round_action(self):
        dead_this_round = (~self._alive) & (self._energies <= 0)
        for i in np.where(dead_this_round)[0]:
            dists = np.linalg.norm(self._pos - self._pos[i], axis=1)
            dists[~self._alive] = np.inf
            dists[i] = np.inf
            nearest = int(np.argmin(dists))
            if dists[nearest] < np.inf:
                # Move nearest node toward dead node
                direction = self._pos[i] - self._pos[nearest]
                dist      = float(np.linalg.norm(direction))
                step      = min(10.0, dist)
                self._pos[nearest] += (direction / (dist + 1e-8)) * step
                self._energies[nearest] -= 0.002   # movement cost


class QoSNRTBaseline(BaselineSimulator):
    """
    QoS-NRT inspired: node relocation based on energy threshold.
    Nodes below 30% energy stop transmitting to conserve battery.
    No zone-priority awareness.
    """

    def _round_action(self):
        for i in range(self.N):
            if not self._alive[i]: continue
            if self._energies[i] < 0.30:
                # Save energy: reduce tx power (simplified as half cost)
                self._energies[i] += 0.001  # skip one tx round


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="WSN Baseline Simulation")
    ap.add_argument("--nodes",  type=int,   default=100)
    ap.add_argument("--area",   type=float, default=500.0)
    ap.add_argument("--rounds", type=int,   default=300)
    ap.add_argument("--mode",   choices=["random", "pacr", "qos"], default="random")
    args = ap.parse_args()

    cls = {"random": RandomBaseline, "pacr": PACRBaseline, "qos": QoSNRTBaseline}[args.mode]
    sim = cls(n_nodes=args.nodes, area_size=args.area)
    res = sim.run(max_rounds=args.rounds, verbose=True)

    print(f"\n{args.mode.upper()} Baseline — {args.nodes} nodes, {args.area}×{args.area} m")
    print(f"  FND: round {res['FND']}  |  HND: round {res['HND']}  |  END: round {res['END']}")
    print(f"  Avg critical coverage: {res['avg_crit_cov']:.1%}")
