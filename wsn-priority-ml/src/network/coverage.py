"""coverage.py — Vectorised k-coverage spatial calculator."""
from __future__ import annotations
import numpy as np


class CoverageCalculator:
    def __init__(self, area_size: float, grid_res: float = 10.0, k: int = 1):
        self.area = area_size
        self.k    = k
        xs = np.arange(0, area_size + grid_res, grid_res)
        ys = np.arange(0, area_size + grid_res, grid_res)
        gx, gy = np.meshgrid(xs, ys)
        self._grid   = np.column_stack([gx.ravel(), gy.ravel()])
        self._n_grid = len(self._grid)

    def compute(self, positions: np.ndarray, tx_radius: float) -> float:
        if len(positions) == 0: return 0.0
        diff  = self._grid[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return float(((dists <= tx_radius).sum(axis=1) >= self.k).mean())

    def coverage_holes(self, positions: np.ndarray, tx_radius: float) -> np.ndarray:
        if len(positions) == 0: return self._grid.copy()
        diff  = self._grid[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask  = (dists <= tx_radius).sum(axis=1) < self.k
        return self._grid[mask]

    def n_holes(self, positions: np.ndarray, tx_radius: float) -> int:
        return len(self.coverage_holes(positions, tx_radius))
