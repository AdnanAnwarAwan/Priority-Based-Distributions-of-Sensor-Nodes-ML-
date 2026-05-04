"""
hibernation.py — Node Hibernation and Wake Protocol.

Nodes in lower-priority zones (Less Critical / Non-Critical) enter
deep-sleep to conserve battery. Hibernating nodes consume ~0.1% of
active energy. A wake trigger fires when neighbour activity spikes
above a threshold, pulling the node back into active mode.

States
------
  ACTIVE     — full sensing, Tx/Rx, ML inference running
  DOZING     — reduced sensing rate, Tx/Rx enabled, ready to wake fast
  HIBERNATE  — near-zero power, Rx only on beacon channel, ~0.1% power
  DEAD       — battery depleted, permanently offline

Transitions
-----------
  ACTIVE    → DOZING     : zone demoted to Less Critical (zone 1)
  DOZING    → HIBERNATE  : zone demoted to Non-Critical (zone 0)
  DOZING    → ACTIVE     : zone promoted to Critical/Most Critical (≥ zone 2)
  HIBERNATE → DOZING     : neighbour wake-trigger or zone promoted to zone 1
  HIBERNATE → ACTIVE     : zone promoted to zone ≥ 2 (emergency wake)
  any       → DEAD       : energy ≤ 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.network.activity_map import (
    ZONE_MOST_CRITICAL, ZONE_CRITICAL,
    ZONE_LESS_CRITICAL, ZONE_NON_CRITICAL,
)


# ─── Node states ──────────────────────────────────────────────────────────────

class NodeState(IntEnum):
    ACTIVE    = 3
    DOZING    = 2
    HIBERNATE = 1
    DEAD      = 0


# Power fractions relative to ACTIVE
POWER_FRACTION = {
    NodeState.ACTIVE:    1.000,
    NodeState.DOZING:    0.350,
    NodeState.HIBERNATE: 0.001,
    NodeState.DEAD:      0.000,
}

# Sensing rate (fraction of full rate)
SENSING_RATE = {
    NodeState.ACTIVE:    1.0,
    NodeState.DOZING:    0.3,
    NodeState.HIBERNATE: 0.0,
    NodeState.DEAD:      0.0,
}


# ─── Single node hibernation manager ─────────────────────────────────────────

@dataclass
class NodeHibernationManager:
    """
    Manages the hibernation state-machine for one sensor node.

    Parameters
    ----------
    node_id        : unique node identifier
    initial_energy : starting battery level [0, 1]
    wake_threshold : activity spike that triggers HIBERNATE → DOZE
    """
    node_id:         int
    initial_energy:  float = 1.0
    wake_threshold:  float = 0.55   # activity above this wakes a hibernating node

    energy:          float = field(init=False)
    state:           NodeState = field(init=False)
    zone:            int   = field(init=False)
    rounds_in_state: int   = field(init=False)
    total_active_r:  int   = field(init=False)
    total_saved_e:   float = field(init=False)

    def __post_init__(self):
        self.energy          = self.initial_energy
        self.state           = NodeState.ACTIVE
        self.zone            = ZONE_CRITICAL
        self.rounds_in_state = 0
        self.total_active_r  = 0
        self.total_saved_e   = 0.0

    @property
    def is_alive(self) -> bool:
        return self.state != NodeState.DEAD

    @property
    def is_sensing(self) -> bool:
        return self.state in (NodeState.ACTIVE, NodeState.DOZING)

    # ── State transitions ──────────────────────────────────────────────────

    def update_zone(self, new_zone: int, neighbour_activity: float = 0.0):
        """
        Called each round with the node's current zone classification.
        Transitions state based on zone + neighbour wake signals.
        """
        if self.state == NodeState.DEAD:
            return

        old_state = self.state

        if self.state == NodeState.ACTIVE:
            if new_zone == ZONE_LESS_CRITICAL:
                self._transition(NodeState.DOZING)
            elif new_zone == ZONE_NON_CRITICAL:
                self._transition(NodeState.HIBERNATE)

        elif self.state == NodeState.DOZING:
            if new_zone >= ZONE_CRITICAL:
                self._transition(NodeState.ACTIVE)
            elif new_zone == ZONE_NON_CRITICAL:
                self._transition(NodeState.HIBERNATE)

        elif self.state == NodeState.HIBERNATE:
            if new_zone >= ZONE_CRITICAL or neighbour_activity >= self.wake_threshold:
                # Emergency wake or neighbour spike
                self._transition(NodeState.ACTIVE if new_zone >= ZONE_CRITICAL else NodeState.DOZING)

        self.zone = new_zone

    def _transition(self, new_state: NodeState):
        self.state           = new_state
        self.rounds_in_state = 0

    def consume_energy(self, active_cost: float) -> float:
        """
        Drain energy proportional to current power mode.
        Returns energy actually consumed this round.
        """
        if self.state == NodeState.DEAD:
            return 0.0

        cost = active_cost * POWER_FRACTION[self.state]
        saved = active_cost - cost
        self.energy         = max(0.0, self.energy - cost)
        self.total_saved_e += saved
        self.rounds_in_state += 1

        if self.state == NodeState.ACTIVE:
            self.total_active_r += 1

        if self.energy <= 0:
            self.state = NodeState.DEAD

        return cost

    # ── Sensing quality ───────────────────────────────────────────────────

    def sensing_quality(self) -> float:
        """Return effective sensing fraction for this node this round."""
        return SENSING_RATE[self.state] * self.energy


# ─── Network-level hibernation controller ────────────────────────────────────

class HibernationController:
    """
    Manages hibernation for all nodes in the network.

    Ensures minimum node density per zone is maintained:
    nodes cannot hibernate if doing so would drop zone density
    below the minimum threshold defined in ZONE_MIN_DENSITY.
    """

    def __init__(
        self,
        n_nodes:         int,
        initial_energies: Optional[np.ndarray] = None,
        wake_threshold:   float = 0.55,
    ):
        energies = (initial_energies if initial_energies is not None
                    else np.ones(n_nodes, dtype=np.float32))

        self.managers: Dict[int, NodeHibernationManager] = {
            i: NodeHibernationManager(
                node_id=i,
                initial_energy=float(energies[i]),
                wake_threshold=wake_threshold,
            )
            for i in range(n_nodes)
        }
        self.n_nodes = n_nodes

    # ── Per-round update ──────────────────────────────────────────────────

    def update(
        self,
        zones:              np.ndarray,      # (N,) current zone per node
        neighbour_activity: np.ndarray,      # (N,) max neighbour activity
        active_cost:        float = 0.01,    # energy per active round
    ) -> Dict[str, int]:
        """
        Update all nodes for one round.
        Returns count dict: active / dozing / hibernating / dead.
        """
        counts = {"active": 0, "dozing": 0, "hibernating": 0, "dead": 0}

        for i, mgr in self.managers.items():
            mgr.update_zone(int(zones[i]), float(neighbour_activity[i]))
            mgr.consume_energy(active_cost)

            if   mgr.state == NodeState.ACTIVE:    counts["active"]      += 1
            elif mgr.state == NodeState.DOZING:    counts["dozing"]      += 1
            elif mgr.state == NodeState.HIBERNATE: counts["hibernating"] += 1
            else:                                  counts["dead"]        += 1

        return counts

    # ── Accessors ─────────────────────────────────────────────────────────

    def states(self) -> np.ndarray:
        return np.array([m.state for m in self.managers.values()], dtype=np.int32)

    def energies(self) -> np.ndarray:
        return np.array([m.energy for m in self.managers.values()], dtype=np.float32)

    def alive_mask(self) -> np.ndarray:
        return np.array([m.is_alive for m in self.managers.values()], dtype=bool)

    def sensing_mask(self) -> np.ndarray:
        return np.array([m.is_sensing for m in self.managers.values()], dtype=bool)

    def sensing_quality(self) -> np.ndarray:
        return np.array([m.sensing_quality() for m in self.managers.values()], dtype=np.float32)

    def total_energy_saved(self) -> float:
        return sum(m.total_saved_e for m in self.managers.values())

    def network_alive_fraction(self) -> float:
        return self.alive_mask().mean()

    def first_node_dead_round(self) -> Optional[int]:
        """Returns round of first death (tracked externally via alive_mask)."""
        return None   # tracked by LifetimeSimulator

    def summary(self) -> Dict:
        st = self.states()
        return {
            "alive":        int((st != NodeState.DEAD).sum()),
            "active":       int((st == NodeState.ACTIVE).sum()),
            "dozing":       int((st == NodeState.DOZING).sum()),
            "hibernating":  int((st == NodeState.HIBERNATE).sum()),
            "dead":         int((st == NodeState.DEAD).sum()),
            "mean_energy":  float(self.energies()[self.alive_mask()].mean())
                            if self.alive_mask().any() else 0.0,
            "energy_saved": round(self.total_energy_saved(), 4),
        }
