"""
activity_map.py — Battlefield Activity Score Generator.

Each grid cell in the AOI is assigned an activity score:
    activity(x,y) = w_troops·troops + w_vehicles·vehicles
                  + w_weapons·weaponry + w_movement·movement

All observations are normalised to [0,1]. The combined score
drives zone classification (Most Critical → Non-Critical).

This module generates both synthetic scenarios for simulation
and provides the data structure each physical sensor node would
populate from its on-board sensing modalities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Observation data class ──────────────────────────────────────────────────

@dataclass
class EnvironmentObservation:
    """
    Raw observation captured by a single sensor node.
    All values normalised to [0, 1].
    """
    node_id:      int
    x:            float          # position in field (metres)
    y:            float
    troops:       float = 0.0   # detected personnel density
    vehicles:     float = 0.0   # detected vehicle count / max
    weaponry:     float = 0.0   # detected weaponry signatures
    movement:     float = 0.0   # motion intensity (doppler / PIR)
    timestamp:    float = 0.0   # simulation round

    def activity_score(
        self,
        w_troops:   float = 0.35,
        w_vehicles: float = 0.25,
        w_weapons:  float = 0.25,
        w_movement: float = 0.15,
    ) -> float:
        return (w_troops   * self.troops
              + w_vehicles * self.vehicles
              + w_weapons  * self.weaponry
              + w_movement * self.movement)

    def to_feature_vector(self) -> np.ndarray:
        """Return [troops, vehicles, weaponry, movement, x_norm, y_norm]."""
        return np.array([
            self.troops, self.vehicles, self.weaponry, self.movement,
        ], dtype=np.float32)


# ─── Zone label constants ─────────────────────────────────────────────────────

ZONE_MOST_CRITICAL = 3     # activity > 0.75
ZONE_CRITICAL      = 2     # 0.50 – 0.75
ZONE_LESS_CRITICAL = 1     # 0.25 – 0.50
ZONE_NON_CRITICAL  = 0     # < 0.25

ZONE_NAMES = {
    ZONE_MOST_CRITICAL: "Most Critical",
    ZONE_CRITICAL:      "Critical",
    ZONE_LESS_CRITICAL: "Less Critical",
    ZONE_NON_CRITICAL:  "Non-Critical",
}

ZONE_COLORS = {
    ZONE_MOST_CRITICAL: "#E24B4A",   # red
    ZONE_CRITICAL:      "#EF9F27",   # amber
    ZONE_LESS_CRITICAL: "#639922",   # green
    ZONE_NON_CRITICAL:  "#378ADD",   # blue
}

# Minimum node density per zone (nodes per 10,000 m²)
ZONE_MIN_DENSITY = {
    ZONE_MOST_CRITICAL: 4,
    ZONE_CRITICAL:      3,
    ZONE_LESS_CRITICAL: 2,
    ZONE_NON_CRITICAL:  1,
}


def activity_to_zone(score: float) -> int:
    if score > 0.75: return ZONE_MOST_CRITICAL
    if score > 0.50: return ZONE_CRITICAL
    if score > 0.25: return ZONE_LESS_CRITICAL
    return ZONE_NON_CRITICAL


# ─── Activity Map ─────────────────────────────────────────────────────────────

class ActivityMap:
    """
    2-D spatial map of enemy activity across the AOI.

    In real deployment each cell is populated from observations
    shared by the sensor nodes via federated learning. In simulation
    we generate synthetic maps using Gaussian hotspot clusters.
    """

    def __init__(
        self,
        area_size:   float = 500.0,
        grid_res:    float = 25.0,    # metres per cell
        n_hotspots:  int   = 5,       # enemy concentration points
        seed:        Optional[int] = None,
    ):
        self.area     = area_size
        self.res      = grid_res
        self.n_cells  = int(np.ceil(area_size / grid_res))
        self.rng      = np.random.default_rng(seed)

        # Grid arrays (n_cells × n_cells)
        self.troops   = np.zeros((self.n_cells, self.n_cells), dtype=np.float32)
        self.vehicles = np.zeros_like(self.troops)
        self.weaponry = np.zeros_like(self.troops)
        self.movement = np.zeros_like(self.troops)
        self._activity: Optional[np.ndarray] = None
        self._zones:    Optional[np.ndarray] = None

        self._hotspots: List[Dict] = []
        self._generate_hotspots(n_hotspots)

    # ── Generation ─────────────────────────────────────────────────────────

    def _generate_hotspots(self, n: int):
        """Place Gaussian activity hotspots across the field."""
        self._hotspots = []
        for _ in range(n):
            cx = self.rng.uniform(0.1, 0.9) * self.area
            cy = self.rng.uniform(0.1, 0.9) * self.area
            sigma = self.rng.uniform(0.05, 0.20) * self.area
            intensity = self.rng.uniform(0.5, 1.0)
            self._hotspots.append(dict(cx=cx, cy=cy, sigma=sigma, intensity=intensity))

        self._recompute()

    def _recompute(self):
        """Recompute all activity layers from hotspot parameters."""
        xs = (np.arange(self.n_cells) + 0.5) * self.res
        ys = (np.arange(self.n_cells) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys)

        combined = np.zeros((self.n_cells, self.n_cells), dtype=np.float32)
        for hs in self._hotspots:
            d2 = (gx - hs["cx"])**2 + (gy - hs["cy"])**2
            combined += hs["intensity"] * np.exp(-d2 / (2 * hs["sigma"]**2))

        combined = np.clip(combined, 0, 1).astype(np.float32)

        # Decompose into sensing modalities with noise
        noise = lambda: self.rng.uniform(-0.05, 0.05, combined.shape).astype(np.float32)
        self.troops   = np.clip(combined * 0.9  + noise(), 0, 1)
        self.vehicles = np.clip(combined * 0.75 + noise(), 0, 1)
        self.weaponry = np.clip(combined * 0.80 + noise(), 0, 1)
        self.movement = np.clip(combined * 0.85 + noise(), 0, 1)
        self._activity = None
        self._zones    = None

    def evolve(self, rounds: int = 1, drift: float = 0.02):
        """
        Simulate enemy movement: hotspots drift slightly each round.
        Models realistic temporal dynamics in a battlefield scenario.
        """
        for _ in range(rounds):
            for hs in self._hotspots:
                hs["cx"] = np.clip(
                    hs["cx"] + self.rng.normal(0, drift * self.area), 0, self.area
                )
                hs["cy"] = np.clip(
                    hs["cy"] + self.rng.normal(0, drift * self.area), 0, self.area
                )
                # Intensity also fluctuates
                hs["intensity"] = np.clip(
                    hs["intensity"] + self.rng.normal(0, 0.05), 0.2, 1.0
                )
        self._recompute()

    # ── Queries ─────────────────────────────────────────────────────────────

    @property
    def activity(self) -> np.ndarray:
        if self._activity is None:
            self._activity = (
                0.35 * self.troops + 0.25 * self.vehicles
                + 0.25 * self.weaponry + 0.15 * self.movement
            )
        return self._activity

    @property
    def zones(self) -> np.ndarray:
        """Return (n_cells, n_cells) integer zone map."""
        if self._zones is None:
            self._zones = np.vectorize(activity_to_zone)(self.activity)
        return self._zones

    def cell_of(self, x: float, y: float) -> Tuple[int, int]:
        """Return grid cell (row, col) for position (x, y)."""
        col = int(np.clip(x / self.res, 0, self.n_cells - 1))
        row = int(np.clip(y / self.res, 0, self.n_cells - 1))
        return row, col

    def activity_at(self, x: float, y: float) -> float:
        r, c = self.cell_of(x, y)
        return float(self.activity[r, c])

    def zone_at(self, x: float, y: float) -> int:
        r, c = self.cell_of(x, y)
        return int(self.zones[r, c])

    def observation_at(
        self,
        node_id: int,
        x: float,
        y: float,
        t: float = 0.0,
        noise_std: float = 0.03,
    ) -> EnvironmentObservation:
        """Generate noisy observation for a node at position (x, y)."""
        r, c = self.cell_of(x, y)
        def noisy(v: float) -> float:
            return float(np.clip(v + self.rng.normal(0, noise_std), 0, 1))
        return EnvironmentObservation(
            node_id=node_id, x=x, y=y,
            troops=noisy(self.troops[r, c]),
            vehicles=noisy(self.vehicles[r, c]),
            weaponry=noisy(self.weaponry[r, c]),
            movement=noisy(self.movement[r, c]),
            timestamp=t,
        )

    def feature_grid(self) -> np.ndarray:
        """
        Return (n_cells×n_cells, 4) feature matrix for training
        the zone classifier.
        """
        return np.stack([
            self.troops.ravel(),
            self.vehicles.ravel(),
            self.weaponry.ravel(),
            self.movement.ravel(),
        ], axis=1).astype(np.float32)

    def zone_labels(self) -> np.ndarray:
        """Return (n_cells×n_cells,) zone label array."""
        return self.zones.ravel().astype(np.int64)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "area": self.area, "res": self.res,
            "hotspots": self._hotspots,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "ActivityMap":
        data = json.loads(Path(path).read_text())
        m = cls(area_size=data["area"], grid_res=data["res"], n_hotspots=0)
        m._hotspots = data["hotspots"]
        m._recompute()
        return m
