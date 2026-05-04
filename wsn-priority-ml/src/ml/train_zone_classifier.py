"""
train_zone_classifier.py — Supervised training of the ZoneClassifier.

Generates synthetic activity maps, extracts grid features + labels,
trains the MLP, and saves the checkpoint.

Usage
-----
python src/ml/train_zone_classifier.py --config config/default.yaml
python src/ml/train_zone_classifier.py --maps 20 --epochs 100 --area 500
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from src.network.activity_map import ActivityMap
from src.ml.zone_classifier import ZoneClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TrainZoneClf")


def train(cfg: dict):
    env_cfg  = cfg["env"]
    ml_cfg   = cfg.get("zone_classifier", {})

    n_maps   = ml_cfg.get("n_training_maps", 10)
    epochs   = ml_cfg.get("epochs", 80)
    area     = env_cfg["area_size"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Generating {n_maps} activity maps, area={area}×{area} m")

    # Collect training data from multiple synthetic maps
    all_X, all_y = [], []
    for i in range(n_maps):
        amap = ActivityMap(area_size=area, n_hotspots=5, seed=i)
        all_X.append(amap.feature_grid())
        all_y.append(amap.zone_labels())

    import numpy as np
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    log.info(f"Dataset: {len(X)} samples, class distribution: "
             f"{np.bincount(y.astype(int))}")

    clf = ZoneClassifier(in_dim=4, hidden=32)
    log.info(f"Training ZoneClassifier for {epochs} epochs...")

    # Build a one-shot ActivityMap for training (pass feature arrays directly)
    from src.network.activity_map import ActivityMap
    import torch

    # Train directly on numpy arrays
    device = clf.device
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)

    clf.model.train()
    history = []
    bs = 256
    for ep in range(epochs):
        idx     = torch.randperm(len(X_t))
        ep_loss = 0.0
        for i in range(0, len(X_t), bs):
            bx = X_t[idx[i:i+bs]]
            by = y_t[idx[i:i+bs]]
            clf.opt.zero_grad()
            loss = clf.loss_fn(clf.model(bx), by)
            loss.backward()
            clf.opt.step()
            ep_loss += loss.item() * len(bx)
        ep_loss /= len(X_t)
        history.append(ep_loss)
        if (ep + 1) % 20 == 0:
            preds = clf.predict(X)
            acc   = (preds == y).mean()
            log.info(f"Epoch {ep+1}/{epochs} | loss={ep_loss:.4f} | acc={acc:.3f}")

    clf.model.eval()

    # Evaluate on a held-out map
    test_map = ActivityMap(area_size=area, n_hotspots=5, seed=999)
    test_acc = clf.accuracy(test_map)
    log.info(f"Test accuracy (held-out map): {test_acc:.3f}")

    ckpt_path = ckpt_dir / "zone_classifier.pt"
    clf.save(str(ckpt_path))
    log.info(f"Checkpoint saved to {ckpt_path}")
    return history, test_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train WSN Zone Classifier")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--maps",   type=int, help="Override n_training_maps")
    ap.add_argument("--epochs", type=int, help="Override epochs")
    ap.add_argument("--area",   type=float, help="Override area_size")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.maps:   cfg.setdefault("zone_classifier", {})["n_training_maps"] = args.maps
    if args.epochs: cfg.setdefault("zone_classifier", {})["epochs"]          = args.epochs
    if args.area:   cfg["env"]["area_size"]                                   = args.area

    train(cfg)
