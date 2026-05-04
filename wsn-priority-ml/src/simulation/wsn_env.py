"""
wsn_env.py — Priority-Aware WSN Gymnasium Environment.

The RL agent learns when and where to migrate nodes, optimising
for network lifetime and critical-zone coverage simultaneously.

Observation per node (6 features, normalised):
    [activity_score, zone_label/3, energy, hop_count/10,
     neighbour_density/10, sensing_quality]

Action space: Discrete(4) per node
    0 = STAY         (do nothing)
    1 = MIGRATE      (move toward nearest high-priority under-served cell)
    2 = HIBERNATE    (enter deep sleep)
    3 = WAKE         (exit hibernation if dozing/sleeping)

Global action = one action per node → MultiDiscrete([4] × N)
RL agent selects one node per step (simplified): Discrete(N × 4)

Reward:
    R = α·critical_coverage + β·energy_efficiency + γ·lifetime_bonus
      + δ·priority_bonus − penalty·coverage_holes
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from src.network.activity_map import ActivityMap, ZONE_MOST_CRITICAL, ZONE_CRITICAL
from src.network.wsn_graph import WSNGraph
from src.network.energy_model import EnergyModel
from src.network.coverage import CoverageCalculator
from src.simulation.hibernation import HibernationController, NodeState
from src.simulation.redistribution import RedistributionEngine
from src.ml.zone_classifier import ZoneClassifier


class PriorityWSNEnv(gym.Env):
    """
    Priority-Aware WSN Environment.

    Combines zone classification, hibernation, and redistribution
    into a single Gymnasium-compatible environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        n_nodes:    int   = 100,
        area_size:  float = 500.0,
        tx_radius:  float = 75.0,
        max_steps:  int   = 300,
        n_hotspots: int   = 5,
        alpha:      float = 0.40,   # critical coverage weight
        beta:       float = 0.30,   # energy efficiency weight
        gamma:      float = 0.20,   # lifetime weight
        delta:      float = 0.10,   # priority bonus weight
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.N          = n_nodes
        self.area       = area_size
        self.tx_radius  = tx_radius
        self.max_steps  = max_steps
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.delta      = delta
        self.render_mode = render_mode

        self.sink = np.array([area_size / 2, area_size / 2])
        self.em   = EnergyModel()
        self.coverage_calc = CoverageCalculator(area_size, grid_res=20.0)
        self.zone_clf      = ZoneClassifier()

        # State
        self._pos:       np.ndarray    = None
        self._energies:  np.ndarray    = None
        self._zones:     np.ndarray    = None
        self._act_map:   ActivityMap   = None
        self._hibernation: HibernationController  = None
        self._redistrib:   RedistributionEngine   = None
        self._step:      int   = 0
        self._prev_r:    float = 0.0

        # Spaces: observation = [activity, zone_norm, energy, hop_norm, density_norm, sq] × N
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_nodes * 6,), dtype=np.float32
        )
        # Action: pick node_i and apply action_a (4 choices)
        self.action_space = spaces.Discrete(n_nodes * 4)

    # ── Gym interface ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        rng = self.np_random

        self._pos      = rng.uniform(0, self.area, (self.N, 2)).astype(np.float32)
        self._energies = rng.uniform(0.7, 1.0, self.N).astype(np.float32)
        self._act_map  = ActivityMap(
            area_size=self.area,
            n_hotspots=5,
            seed=int(rng.integers(0, 10000)),
        )
        self._hibernation = HibernationController(self.N, self._energies.copy())
        self._redistrib   = RedistributionEngine(self.area, self.em)
        self._step        = 0
        self._prev_r      = 0.0

        self._update_zones()
        return self._obs(), self._info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        node_i   = action // 4
        action_a = action %  4

        alive = self._hibernation.alive_mask()

        # Apply chosen action for node_i
        if alive[node_i]:
            mgr = self._hibernation.managers[node_i]
            if action_a == 1:   # MIGRATE
                self._redistrib.step(
                    self._pos, self._energies, self._zones, alive,
                    self._act_map, self.sink
                )
            elif action_a == 2:   # HIBERNATE force
                if mgr.state != NodeState.DEAD:
                    mgr._transition(NodeState.HIBERNATE)
            elif action_a == 3:   # WAKE force
                if mgr.state == NodeState.HIBERNATE:
                    mgr._transition(NodeState.DOZING)
            # action_a == 0: STAY — do nothing

        # Evolve activity map slightly
        self._act_map.evolve(rounds=1, drift=0.005)
        self._update_zones()

        # Update hibernation states
        nb_activity = self._neighbour_activity()
        self._hibernation.update(
            zones=self._zones,
            neighbour_activity=nb_activity,
            active_cost=0.005,
        )
        self._energies = self._hibernation.energies()

        raw_r    = self._reward()
        shaped_r = raw_r - self._prev_r
        self._prev_r = raw_r
        self._step  += 1

        terminated = not self._hibernation.alive_mask().any()
        truncated  = self._step >= self.max_steps

        return self._obs(), shaped_r, terminated, truncated, self._info()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _update_zones(self):
        features = np.array([
            self._act_map.observation_at(i, float(self._pos[i, 0]), float(self._pos[i, 1])).to_feature_vector()
            for i in range(self.N)
        ])
        self._zones = self.zone_clf.predict(features)

    def _neighbour_activity(self) -> np.ndarray:
        """Max activity score among alive neighbours within tx_radius."""
        activity = np.array([
            self._act_map.activity_at(float(self._pos[i, 0]), float(self._pos[i, 1]))
            for i in range(self.N)
        ], dtype=np.float32)
        result = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            dists = np.linalg.norm(self._pos - self._pos[i], axis=1)
            nbrs  = (dists < self.tx_radius) & (dists > 0)
            if nbrs.any():
                result[i] = float(activity[nbrs].max())
        return result

    def _obs(self) -> np.ndarray:
        alive   = self._hibernation.alive_mask()
        sq      = self._hibernation.sensing_quality()
        g       = WSNGraph(self._pos, self.sink, self.tx_radius, self._energies)
        g.build()
        hops    = np.array([g.hop_count(i) / 10.0 for i in range(self.N)], dtype=np.float32)

        # Node density: alive neighbours / 10
        density = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            dists = np.linalg.norm(self._pos - self._pos[i], axis=1)
            density[i] = ((dists < self.tx_radius) & alive).sum() / 10.0

        activity = np.array([
            self._act_map.activity_at(float(x), float(y))
            for x, y in self._pos
        ], dtype=np.float32)

        obs = np.column_stack([
            activity,
            self._zones / 3.0,
            self._energies,
            hops,
            np.clip(density, 0, 1),
            sq,
        ]).flatten().astype(np.float32)
        return obs

    def _info(self) -> Dict[str, Any]:
        alive      = self._hibernation.alive_mask()
        sensing    = self._hibernation.sensing_mask()
        n_conn     = sum(
            WSNGraph(self._pos, self.sink, self.tx_radius, self._energies).build() or True
            for _ in [1]
        )
        cov        = self.coverage_calc.compute(self._pos[sensing], self.tx_radius)
        crit_nodes = sensing & (self._zones >= ZONE_CRITICAL)
        crit_cov   = self.coverage_calc.compute(
            self._pos[crit_nodes], self.tx_radius
        ) if crit_nodes.any() else 0.0

        return {
            "step":              self._step,
            "alive":             int(alive.sum()),
            "hibernating":       int((self._hibernation.states() == NodeState.HIBERNATE).sum()),
            "coverage":          round(float(cov), 3),
            "critical_coverage": round(float(crit_cov), 3),
            "mean_energy":       round(float(self._energies[alive].mean()) if alive.any() else 0.0, 3),
            "energy_saved":      round(self._hibernation.total_energy_saved(), 4),
            "migrations":        self._redistrib.total_migrations(),
        }

    def _reward(self) -> float:
        info      = self._info()
        crit_cov  = info["critical_coverage"]
        e_eff     = info["mean_energy"]
        alive_f   = info["alive"] / self.N
        hi_zones  = (self._zones >= ZONE_CRITICAL) & self._hibernation.sensing_mask()
        p_bonus   = float(hi_zones.sum()) / max(1, int((self._zones >= ZONE_CRITICAL).sum()))
        return (self.alpha * crit_cov + self.beta * e_eff
                + self.gamma * alive_f + self.delta * p_bonus)
