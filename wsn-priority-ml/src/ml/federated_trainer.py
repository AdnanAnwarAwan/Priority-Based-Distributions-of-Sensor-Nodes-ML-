"""
federated_trainer.py — Federated Learning Coordinator.

Each WSN sensor node trains a local ZoneClassifier on its own
observations. Only gradient updates (never raw surveillance data)
are shared with the coordinator. The coordinator averages gradients
(FedAvg) and broadcasts the global model back.

This preserves sensor data confidentiality while allowing all nodes
to benefit from network-wide battlefield observations.

Architecture
------------
  Coordinator
    ├── Broadcasts global model weights
    ├── Receives local gradients from each node
    ├── Applies FedAvg aggregation
    └── Sends updated global weights back

  Each Node
    ├── Receives global weights
    ├── Trains locally on local_observations (epochs=E)
    ├── Computes gradient delta
    └── Sends compressed gradient to coordinator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.ml.zone_classifier import ZoneClassifier
from src.network.activity_map import EnvironmentObservation, ActivityMap

log = logging.getLogger("FederatedTrainer")


# ─── Per-node local model ─────────────────────────────────────────────────────

@dataclass
class LocalNode:
    """Represents one sensor node's local ML state."""
    node_id:        int
    classifier:     ZoneClassifier = field(default_factory=ZoneClassifier)
    observations:   List[EnvironmentObservation] = field(default_factory=list)
    rounds_trained: int = 0
    last_loss:      float = 0.0

    def add_observation(self, obs: EnvironmentObservation):
        self.observations.append(obs)
        # Keep a sliding window of recent observations
        if len(self.observations) > 200:
            self.observations = self.observations[-200:]

    def train_local(self, epochs: int = 5) -> float:
        loss = self.classifier.train_local(self.observations, epochs=epochs)
        self.rounds_trained += 1
        self.last_loss = loss
        return loss

    def get_weights(self) -> List[np.ndarray]:
        return self.classifier.get_weights()

    def set_weights(self, weights: List[np.ndarray]):
        self.classifier.set_weights(weights)

    def predict_zone(self, obs: EnvironmentObservation) -> int:
        return self.classifier.predict_observation(obs)


# ─── FedAvg Aggregation ───────────────────────────────────────────────────────

def fedavg(
    weights_list: List[List[np.ndarray]],
    sample_counts: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """
    Federated Averaging (McMahan et al., 2017).
    Weighted average of local model weights proportional to
    number of local training samples.
    """
    n_nodes = len(weights_list)
    if n_nodes == 0:
        raise ValueError("No weights to aggregate.")

    if sample_counts is None:
        weights = [1.0 / n_nodes] * n_nodes
    else:
        total = sum(sample_counts)
        weights = [s / total for s in sample_counts]

    aggregated = []
    for layer_idx in range(len(weights_list[0])):
        layer_avg = sum(
            weights_list[i][layer_idx] * weights[i]
            for i in range(n_nodes)
        )
        aggregated.append(layer_avg)
    return aggregated


# ─── Coordinator ──────────────────────────────────────────────────────────────

class FederatedTrainer:
    """
    Central coordinator for federated learning across WSN nodes.

    Each communication round:
    1. Broadcast global model to all participating nodes
    2. Each node trains locally for E epochs
    3. Nodes send back updated weights
    4. Coordinator applies FedAvg
    5. Repeat

    In a real deployment, step 2 happens on-device; steps 1,3,4,5
    happen at the sink/gateway node.
    """

    def __init__(
        self,
        n_nodes:       int   = 50,
        local_epochs:  int   = 5,
        min_nodes_pct: float = 0.8,   # min fraction of nodes needed per round
        device:        str   = "auto",
    ):
        self.n_nodes       = n_nodes
        self.local_epochs  = local_epochs
        self.min_nodes_pct = min_nodes_pct

        # Global model at coordinator
        self.global_clf = ZoneClassifier(device=device)

        # Per-node local models
        self.nodes: Dict[int, LocalNode] = {
            i: LocalNode(node_id=i, classifier=ZoneClassifier(device=device))
            for i in range(n_nodes)
        }

        self._round = 0
        self._history: List[Dict] = []

    # ── Data ingestion ────────────────────────────────────────────────────────

    def feed_observation(self, node_id: int, obs: EnvironmentObservation):
        """Add an observation to a node's local dataset."""
        if node_id in self.nodes:
            self.nodes[node_id].add_observation(obs)

    def feed_activity_map(
        self,
        activity_map: ActivityMap,
        positions:    np.ndarray,     # (N, 2) node positions
    ):
        """
        Populate each node's observations from a simulated activity map.
        Used in simulation when we don't have real sensor readings.
        """
        for i, (x, y) in enumerate(positions):
            if i >= self.n_nodes:
                break
            obs = activity_map.observation_at(node_id=i, x=float(x), y=float(y), t=self._round)
            self.nodes[i].add_observation(obs)

    # ── Training round ────────────────────────────────────────────────────────

    def communication_round(
        self,
        participating: Optional[List[int]] = None,
    ) -> Dict:
        """
        Execute one federated communication round.

        Returns metrics dict with loss, accuracy, and participation rate.
        """
        self._round += 1

        # Default: all nodes participate
        if participating is None:
            participating = list(self.nodes.keys())

        # Filter nodes with sufficient data
        eligible = [
            nid for nid in participating
            if len(self.nodes[nid].observations) >= 5
        ]

        if len(eligible) < max(1, int(self.n_nodes * self.min_nodes_pct)):
            log.warning(
                f"Round {self._round}: only {len(eligible)} eligible nodes "
                f"(need {int(self.n_nodes * self.min_nodes_pct)}). Skipping."
            )
            return {"round": self._round, "skipped": True}

        # Step 1: broadcast global weights to participating nodes
        global_weights = self.global_clf.get_weights()
        for nid in eligible:
            self.nodes[nid].set_weights([w.copy() for w in global_weights])

        # Step 2: local training
        losses = []
        for nid in eligible:
            loss = self.nodes[nid].train_local(epochs=self.local_epochs)
            losses.append(loss)

        # Step 3: collect local weights + sample counts
        all_weights    = [self.nodes[nid].get_weights() for nid in eligible]
        sample_counts  = [len(self.nodes[nid].observations) for nid in eligible]

        # Step 4: FedAvg
        aggregated = fedavg(all_weights, sample_counts)

        # Step 5: update global model
        self.global_clf.set_weights(aggregated)

        metrics = {
            "round":            self._round,
            "participating":    len(eligible),
            "avg_loss":         float(np.mean(losses)),
            "min_loss":         float(np.min(losses)),
            "max_loss":         float(np.max(losses)),
            "skipped":          False,
        }
        self._history.append(metrics)
        log.info(
            f"Round {self._round}: {len(eligible)} nodes | "
            f"avg_loss={metrics['avg_loss']:.4f}"
        )
        return metrics

    def run(
        self,
        n_rounds:    int,
        activity_map: Optional[ActivityMap] = None,
        positions:   Optional[np.ndarray]   = None,
    ) -> List[Dict]:
        """
        Run n_rounds of federated training.
        If activity_map is provided, refreshes node observations each round.
        """
        results = []
        for r in range(n_rounds):
            if activity_map is not None and positions is not None:
                activity_map.evolve(rounds=1)
                self.feed_activity_map(activity_map, positions)

            metrics = self.communication_round()
            results.append(metrics)

        log.info(f"Federated training complete. {n_rounds} rounds finished.")
        return results

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_zone(self, obs: EnvironmentObservation) -> int:
        """Use global model to predict zone for an observation."""
        return self.global_clf.predict_observation(obs)

    def predict_zones_grid(self, activity_map: ActivityMap) -> np.ndarray:
        """Predict zones for entire activity map grid."""
        features = activity_map.feature_grid()
        return self.global_clf.predict(features).reshape(
            activity_map.n_cells, activity_map.n_cells
        )

    def global_accuracy(self, activity_map: ActivityMap) -> float:
        return self.global_clf.accuracy(activity_map)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_global_model(self, path: str):
        self.global_clf.save(path)

    def load_global_model(self, path: str):
        self.global_clf.load(path)

    @property
    def history(self) -> List[Dict]:
        return self._history
