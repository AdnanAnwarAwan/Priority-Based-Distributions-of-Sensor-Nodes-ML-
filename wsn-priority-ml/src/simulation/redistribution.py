"""
redistribution.py — Priority-Driven Node Redistribution Engine.

Implements the core idea from the future research:
"More sensor nodes must monitor important areas and fewer to the
lesser important areas. With the passage of time, nodes may move
from less priority to higher priority levels."

Algorithm
---------
Each round:
1.  Count current node density per zone cell.
2.  Identify over-served cells (surplus) and under-served cells (deficit).
3.  Surplus nodes in low-priority zones are candidates for relocation.
4.  Each candidate moves toward the nearest under-served high-priority cell.
5.  Movement cost is subtracted from node energy (radio + locomotion).
6.  If a node's energy drops below MIGRATION_THRESHOLD, it is ineligible
    to migrate (preserve remaining life for sensing).

This engine is used directly by the WSN Gym environment and the
lifetime simulator. The RL agent wraps it to learn optimal migration
policies beyond the greedy heuristic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.network.activity_map import (
    ActivityMap, ZONE_MIN_DENSITY, ZONE_NAMES,
    ZONE_MOST_CRITICAL, ZONE_CRITICAL, ZONE_LESS_CRITICAL, ZONE_NON_CRITICAL,
)
from src.network.energy_model import EnergyModel

log = logging.getLogger("Redistribution")

MIGRATION_THRESHOLD = 0.20      # min energy fraction to be eligible to move
MIGRATION_SPEED     = 10.0      # metres per round (physical movement speed)
BITS_PER_BEACON     = 256       # beacon message size for coordination


@dataclass
class MigrationEvent:
    """Records one node migration for logging and replay."""
    round:       int
    node_id:     int
    from_x:      float
    from_y:      float
    to_x:        float
    to_y:        float
    distance:    float
    from_zone:   int
    to_zone:     int
    energy_cost: float


class RedistributionEngine:
    """
    Greedy priority-driven node redistribution.

    Operates on the current node positions, zone map, and energy levels.
    Returns updated positions and energy levels after one round of migration.
    """

    def __init__(
        self,
        area_size:    float = 500.0,
        energy_model: Optional[EnergyModel] = None,
        speed:        float = MIGRATION_SPEED,
        min_energy:   float = MIGRATION_THRESHOLD,
    ):
        self.area      = area_size
        self.em        = energy_model or EnergyModel()
        self.speed     = speed
        self.min_energy = min_energy
        self._log: List[MigrationEvent] = []
        self._round = 0

    # ── Zone density analysis ──────────────────────────────────────────────

    def _zone_density(
        self,
        positions: np.ndarray,     # (N, 2)
        zones:     np.ndarray,     # (N,) current zone per node
        alive:     np.ndarray,     # (N,) bool
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Compute surplus and deficit counts per zone.
        Returns (surplus_dict, deficit_dict) keyed by zone label.
        """
        counts = {z: 0 for z in range(4)}
        for i in range(len(positions)):
            if alive[i]:
                counts[int(zones[i])] += 1

        total_alive = sum(counts.values())
        if total_alive == 0:
            return {}, {}

        # Desired distribution: weight by zone priority
        zone_weights = {
            ZONE_MOST_CRITICAL: 0.40,
            ZONE_CRITICAL:      0.30,
            ZONE_LESS_CRITICAL: 0.20,
            ZONE_NON_CRITICAL:  0.10,
        }
        desired = {z: max(ZONE_MIN_DENSITY[z], int(zone_weights[z] * total_alive))
                   for z in range(4)}

        surplus = {z: max(0, counts[z] - desired[z]) for z in range(4)}
        deficit = {z: max(0, desired[z] - counts[z]) for z in range(4)}
        return surplus, deficit

    # ── Target selection ──────────────────────────────────────────────────

    def _find_target_position(
        self,
        activity_map: ActivityMap,
        target_zone:  int,
        current_pos:  np.ndarray,   # (2,)
        occupied:     np.ndarray,   # (K, 2) already-occupied target positions
    ) -> np.ndarray:
        """
        Find the nearest under-served cell of target_zone that is not
        already a migration target for another node this round.
        """
        # Sample candidate positions in the target zone
        n_samples = 200
        candidates = []
        rng = np.random.default_rng()
        xs  = rng.uniform(0, self.area, n_samples)
        ys  = rng.uniform(0, self.area, n_samples)
        for x, y in zip(xs, ys):
            if activity_map.zone_at(x, y) == target_zone:
                candidates.append(np.array([x, y]))

        if not candidates:
            # Fallback: move toward field centre
            return np.array([self.area / 2, self.area / 2], dtype=float)

        # Pick nearest candidate not already claimed
        best_pos  = None
        best_dist = float("inf")
        for cand in candidates:
            # Skip if another migrating node is heading here
            if len(occupied) > 0:
                min_occ = np.linalg.norm(occupied - cand, axis=1).min()
                if min_occ < 20.0:  # 20 m exclusion radius
                    continue
            d = float(np.linalg.norm(cand - current_pos))
            if d < best_dist:
                best_dist = d
                best_pos  = cand

        return best_pos if best_pos is not None else candidates[0]

    # ── Main redistribution step ──────────────────────────────────────────

    def step(
        self,
        positions:    np.ndarray,     # (N, 2) — modified in-place
        energies:     np.ndarray,     # (N,)   — modified in-place
        zones:        np.ndarray,     # (N,)   current zone per node
        alive:        np.ndarray,     # (N,)   bool
        activity_map: ActivityMap,
        sink_pos:     Optional[np.ndarray] = None,
    ) -> int:
        """
        Execute one redistribution round.
        Returns number of nodes migrated.
        """
        self._round += 1
        surplus, deficit = self._zone_density(positions, zones, alive)

        # Determine which nodes are candidates (in surplus zones, enough energy)
        candidates: List[Tuple[int, int]] = []   # (node_idx, source_zone)
        for z in [ZONE_NON_CRITICAL, ZONE_LESS_CRITICAL]:
            if surplus.get(z, 0) <= 0:
                continue
            zone_nodes = [
                i for i in range(len(positions))
                if alive[i] and int(zones[i]) == z and energies[i] >= self.min_energy
            ]
            # Sort by energy descending (move richest nodes)
            zone_nodes.sort(key=lambda i: -energies[i])
            for i in zone_nodes[:surplus[z]]:
                candidates.append((i, z))

        if not candidates:
            return 0

        # Determine target zones (priority order)
        target_zones = [
            z for z in [ZONE_MOST_CRITICAL, ZONE_CRITICAL, ZONE_LESS_CRITICAL]
            if deficit.get(z, 0) > 0
        ]
        if not target_zones:
            return 0

        migrated  = 0
        occupied  = np.empty((0, 2))

        for node_idx, src_zone in candidates:
            target_zone = target_zones[0]   # always fill highest deficit first

            target_pos = self._find_target_position(
                activity_map, target_zone, positions[node_idx], occupied
            )

            # Move toward target by at most self.speed metres
            direction = target_pos - positions[node_idx]
            dist      = float(np.linalg.norm(direction))

            if dist < 1.0:
                continue

            move_dist = min(self.speed, dist)
            new_pos   = positions[node_idx] + (direction / dist) * move_dist
            new_pos   = np.clip(new_pos, 0, self.area)

            # Energy cost: tx beacon + locomotion (simplified)
            if sink_pos is not None:
                d_sink = float(np.linalg.norm(new_pos - sink_pos))
                e_cost = self.em.tx_energy(BITS_PER_BEACON, d_sink)
            else:
                e_cost = self.em.tx_energy(BITS_PER_BEACON, move_dist)

            # Locomotion cost modelled as proportional to distance
            e_loco = move_dist * 1e-4   # simplified mechanical energy

            total_cost = e_cost * 1e6 + e_loco   # normalised to [0,1] scale
            energies[node_idx] = max(0.0, energies[node_idx] - total_cost * 0.001)

            self._log.append(MigrationEvent(
                round=self._round,
                node_id=node_idx,
                from_x=float(positions[node_idx][0]),
                from_y=float(positions[node_idx][1]),
                to_x=float(new_pos[0]),
                to_y=float(new_pos[1]),
                distance=move_dist,
                from_zone=src_zone,
                to_zone=target_zone,
                energy_cost=total_cost * 0.001,
            ))

            positions[node_idx] = new_pos
            occupied = np.vstack([occupied, new_pos[None]])
            migrated += 1

            # Update deficit
            deficit[target_zone] = max(0, deficit[target_zone] - 1)
            if deficit[target_zone] == 0 and target_zones:
                target_zones.pop(0)
            if not target_zones:
                break

        if migrated:
            log.debug(f"Round {self._round}: {migrated} nodes migrated.")
        return migrated

    @property
    def migration_log(self) -> List[MigrationEvent]:
        return self._log

    def total_migrations(self) -> int:
        return len(self._log)
