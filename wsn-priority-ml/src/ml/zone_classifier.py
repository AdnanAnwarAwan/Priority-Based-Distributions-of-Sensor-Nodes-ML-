"""
zone_classifier.py — Zone Classification Model.

Classifies each spatial cell into one of four priority zones:
    0 = Non-Critical      (activity < 0.25)
    1 = Less Critical     (0.25 – 0.50)
    2 = Critical          (0.50 – 0.75)
    3 = Most Critical     (> 0.75)

Each sensor node runs this model locally (federated learning).
Input features: [troops, vehicles, weaponry, movement]  (4 dims)
Optional extended: + [x_norm, y_norm, neighbour_avg_activity]  (7 dims)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

from src.network.activity_map import (
    ActivityMap, EnvironmentObservation,
    ZONE_MOST_CRITICAL, ZONE_NAMES, activity_to_zone,
)


# ─── Network architecture ─────────────────────────────────────────────────────

class ZoneNet(nn.Module):
    """
    4-class MLP zone classifier.
    Designed to run on constrained hardware (ATmega, ARM Cortex-M).
    Kept deliberately small: 4 → 32 → 32 → 4.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),    # 4 priority zones
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Classifier wrapper ───────────────────────────────────────────────────────

class ZoneClassifier:
    """
    Wraps ZoneNet with training, inference, and federated helpers.

    Federated usage
    ---------------
    Each node trains locally:
        clf.train_local(observations)
    Then shares gradients (not raw data) with coordinator:
        grads = clf.extract_gradients()
    After aggregation the coordinator pushes averaged gradients back:
        clf.apply_gradients(aggregated_grads)
    """

    def __init__(
        self,
        in_dim:  int   = 4,
        hidden:  int   = 32,
        lr:      float = 1e-3,
        device:  str   = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ZoneNet(in_dim, hidden).to(self.device)
        self.opt   = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()

    # ── Training ──────────────────────────────────────────────────────────────

    def train_local(
        self,
        observations: list[EnvironmentObservation],
        epochs: int = 5,
    ) -> float:
        """Train on local node observations. Returns final epoch loss."""
        if not observations:
            return 0.0

        X = torch.FloatTensor(
            np.stack([o.to_feature_vector() for o in observations])
        ).to(self.device)
        y = torch.LongTensor(
            [activity_to_zone(o.activity_score()) for o in observations]
        ).to(self.device)

        self.model.train()
        last_loss = 0.0
        for _ in range(epochs):
            self.opt.zero_grad()
            logits = self.model(X)
            loss   = self.loss_fn(logits, y)
            loss.backward()
            self.opt.step()
            last_loss = float(loss.item())

        self.model.eval()
        return last_loss

    def train_supervised(
        self,
        activity_map: ActivityMap,
        epochs:       int   = 50,
        batch_size:   int   = 256,
        verbose:      bool  = False,
    ) -> list[float]:
        """
        Train on a full activity map (centralized, for baseline comparison).
        Returns per-epoch loss list.
        """
        X_np = activity_map.feature_grid()
        y_np = activity_map.zone_labels()

        X = torch.FloatTensor(X_np).to(self.device)
        y = torch.LongTensor(y_np).to(self.device)

        history = []
        self.model.train()
        n = len(X)

        for ep in range(epochs):
            idx  = torch.randperm(n)
            ep_loss = 0.0
            for i in range(0, n, batch_size):
                batch_x = X[idx[i:i+batch_size]]
                batch_y = y[idx[i:i+batch_size]]
                self.opt.zero_grad()
                loss = self.loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                self.opt.step()
                ep_loss += loss.item() * len(batch_x)
            ep_loss /= n
            history.append(ep_loss)
            if verbose and (ep + 1) % 10 == 0:
                print(f"  Epoch {ep+1}/{epochs}  loss={ep_loss:.4f}")

        self.model.eval()
        return history

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict zone labels for (N, 4) feature array."""
        self.model.eval()
        with torch.no_grad():
            x   = torch.FloatTensor(features).to(self.device)
            out = self.model(x).argmax(1).cpu().numpy()
        return out

    def predict_observation(self, obs: EnvironmentObservation) -> int:
        """Predict zone for a single observation."""
        feat = obs.to_feature_vector()[None]
        return int(self.predict(feat)[0])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return softmax probabilities (N, 4)."""
        self.model.eval()
        with torch.no_grad():
            x   = torch.FloatTensor(features).to(self.device)
            out = torch.softmax(self.model(x), dim=1).cpu().numpy()
        return out

    def accuracy(self, activity_map: ActivityMap) -> float:
        """Compute accuracy on a full activity map."""
        X  = activity_map.feature_grid()
        y  = activity_map.zone_labels()
        yp = self.predict(X)
        return float((yp == y).mean())

    # ── Federated helpers ────────────────────────────────────────────────────

    def extract_gradients(self) -> list[np.ndarray]:
        """Return list of parameter gradient arrays (post-backward)."""
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.cpu().numpy().copy())
            else:
                grads.append(np.zeros_like(p.data.cpu().numpy()))
        return grads

    def apply_gradients(self, aggregated_grads: list[np.ndarray]):
        """Apply federated-averaged gradients to local model."""
        self.opt.zero_grad()
        for p, g in zip(self.model.parameters(), aggregated_grads):
            p.grad = torch.FloatTensor(g).to(self.device)
        self.opt.step()

    def get_weights(self) -> list[np.ndarray]:
        return [p.data.cpu().numpy().copy() for p in self.model.parameters()]

    def set_weights(self, weights: list[np.ndarray]):
        for p, w in zip(self.model.parameters(), weights):
            p.data.copy_(torch.FloatTensor(w).to(self.device))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "opt":   self.opt.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.model.eval()
