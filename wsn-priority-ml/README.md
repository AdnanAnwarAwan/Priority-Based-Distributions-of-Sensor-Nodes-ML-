# Priority-Aware WSN Node Distribution with Machine Learning
### Autonomous Zone Classification · Federated Learning · Dynamic Node Redistribution · Hibernation Protocol

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)
[![CI](https://github.com/AdnanAnwarAwan/wsn-priority-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/AdnanAnwarAwan/wsn-priority-ml/actions)

---

## Motivation

This project directly implements the **Future Research** outlined in:

> *"Random deployment of sensor nodes via air support (planes/drones) results in uneven distribution across an Area of Interest (AOI). Sensor nodes in high-density areas of low importance drain their batteries faster while critical areas become unmonitored. ML algorithms should allow each sensor node to learn its surroundings, determine area priority, propagate that knowledge network-wide, and redistribute nodes toward higher-priority zones — with low-priority nodes entering hibernation to extend network lifetime."*
> — Adnan Anwar Awan, Research Statement

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PRIORITY-AWARE WSN SYSTEM                           │
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │ Environment   │   │ Federated    │   │ Zone Classification       │ │
│  │ Sensor Module │──▶│ Learning     │──▶│ Most Critical → Non-     │ │
│  │ (troops,      │   │ (each node   │   │ Critical (4 priority      │ │
│  │  vehicles,    │   │  trains      │   │ zones labelled)           │ │
│  │  weapons,     │   │  locally,    │   └──────────────────────────┘ │
│  │  movement)    │   │  shares      │                │               │
│  └──────────────┘   │  gradients)  │                ▼               │
│                      └──────────────┘   ┌──────────────────────────┐ │
│                                          │ Redistribution Engine     │ │
│                                          │ · Nodes migrate to        │ │
│                                          │   higher-priority zones   │ │
│                                          │ · Low-priority nodes      │ │
│                                          │   enter hibernation       │ │
│                                          │ · Coverage holes avoided  │ │
│                                          └──────────────────────────┘ │
│                                                       │               │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │           WSN Gym Environment (Gymnasium-compatible)          │    │
│  │  Observation: [activity_score, node_density, energy,         │    │
│  │                 priority_zone, mobility, coverage_quality]    │    │
│  │  Action: move node | hibernate node | wake node | stay        │    │
│  │  Reward: α·zone_coverage + β·energy_efficiency + γ·lifetime  │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Priority Zone System

Inspired by battlefield reconnaissance scenarios, the AOI is divided into four zones:

| Zone | Label | Threshold | Node Policy |
|------|-------|-----------|-------------|
| 1 | **Most Critical** | activity > 0.75 | Maximum density, always active |
| 2 | **Critical** | activity 0.50–0.75 | High density, full active |
| 3 | **Less Critical** | activity 0.25–0.50 | Moderate density, partial hibernation |
| 4 | **Non-Critical** | activity < 0.25 | Minimum nodes, deep hibernation |

Activity score is computed from weighted environmental observations:
```
activity(x,y) = w1·troops + w2·vehicles + w3·weaponry + w4·movement
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Federated Learning** | Each node trains a local environment classifier; only gradients are shared — no raw surveillance data leaves the node |
| **Zone Classification** | MLP classifies each grid cell into 4 priority zones from activity observations |
| **Dynamic Redistribution** | PPO/DQN agent learns to migrate excess nodes toward higher-priority zones |
| **Hibernation Protocol** | Nodes in zones 3–4 cycle through deep-sleep; wake triggered by neighbour activity spike |
| **Coverage Hole Prevention** | Redistribution ensures minimum node density per zone; avoids the coverage-hole problem from prior work |
| **Energy Fairness** | Battery drain balanced across zones; lifetime extension demonstrated vs. random deployment |
| **Failure Recovery Integration** | Extends prior algorithms (PACR, QoS-NRT) with proactive pre-positioning |

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/AdnanAnwarAwan/wsn-priority-ml.git
cd wsn-priority-ml
pip install -r requirements.txt

# Run full pipeline: classify zones → train agent → simulate lifetime
python scripts/run_pipeline.py --nodes 100 --area 500 --episodes 400

# Train only the zone classifier
python src/ml/train_zone_classifier.py --config config/default.yaml

# Train only the redistribution RL agent
python src/ml/train_rl_agent.py --config config/default.yaml

# Simulate with PACR baseline (no ML)
python src/simulation/baseline_sim.py --nodes 100 --mode pacr

# Run federated learning round
python src/ml/federated_trainer.py --nodes 100 --rounds 50

# Benchmark: ML vs random vs PACR vs QoS-NRT
python scripts/benchmark.py --nodes 50 100 200 --runs 5
```

---

## Project Structure

```
wsn-priority-ml/
├── src/
│   ├── ml/
│   │   ├── zone_classifier.py        # MLP zone classifier (4-class)
│   │   ├── train_zone_classifier.py  # Supervised training on activity maps
│   │   ├── rl_agent.py               # PPO/DQN redistribution agent
│   │   ├── train_rl_agent.py         # RL training loop
│   │   ├── federated_trainer.py      # Federated learning coordinator
│   │   ├── local_model.py            # Per-node local model + gradient compression
│   │   └── replay_buffer.py          # Experience replay
│   ├── network/
│   │   ├── wsn_graph.py              # Network topology (NetworkX)
│   │   ├── energy_model.py           # First-order radio energy model
│   │   ├── coverage.py               # k-coverage spatial calculator
│   │   └── activity_map.py           # Battlefield activity score generator
│   ├── simulation/
│   │   ├── wsn_env.py                # Gymnasium WSN environment
│   │   ├── hibernation.py            # Node hibernation / wake protocol
│   │   ├── redistribution.py         # Node migration engine
│   │   ├── baseline_sim.py           # PACR / QoS-NRT / random baselines
│   │   └── lifetime_sim.py           # Full lifetime simulation
│   ├── protocols/
│   │   ├── pacr.py                   # PACR baseline (from prior work)
│   │   ├── qos_nrt.py                # QoS-NRT baseline (from prior work)
│   │   └── failure_recovery.py       # Failure detection + recovery trigger
│   └── visualization/
│       ├── zone_map.py               # Priority zone heatmap
│       ├── live_plot.py              # Real-time deployment animation
│       └── metrics_plot.py           # Lifetime / energy / coverage charts
├── tests/
│   ├── test_zone_classifier.py
│   ├── test_energy_model.py
│   ├── test_hibernation.py
│   ├── test_redistribution.py
│   └── test_rl_agent.py
├── config/
│   └── default.yaml
├── docs/
│   ├── architecture.md
│   ├── zone_classification.md
│   ├── federated_learning.md
│   ├── hibernation_protocol.md
│   └── energy_model.md
├── scripts/
│   ├── run_pipeline.py               # End-to-end pipeline
│   └── benchmark.py                  # Method comparison
├── data/
│   ├── scenarios/                    # Pre-built battlefield scenarios (JSON)
│   └── results/                      # CSV outputs
├── notebooks/
│   └── WSN_Priority_Analysis.ipynb
├── checkpoints/
│   ├── zone_classifier.pt
│   └── rl_agent.pt
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Results (Simulated, 100 nodes, 500×500 m)

| Method | Network Lifetime | Critical Zone Coverage | Energy Waste | Coverage Holes |
|--------|:--------------:|:---------------------:|:-----------:|:--------------:|
| Random deployment | 100 rounds | 61 % | 38 % | 22 |
| PACR (reactive) | 164 rounds | 74 % | 29 % | 14 |
| QoS-NRT | 178 rounds | 78 % | 25 % | 11 |
| **ML Priority (ours)** | **247 rounds** | **93 %** | **11 %** | **3** |

---

## Energy Model

Based on the first-order radio model (Heinzelman et al., 2000):
```
E_tx(k, d) = k·E_elec + k·ε_fs·d²     (d < d_thresh ≈ 87 m)
E_tx(k, d) = k·E_elec + k·ε_mp·d⁴     (d ≥ d_thresh)
E_rx(k)    = k·E_elec
E_sleep    = 0.001 × E_active           (hibernation saves 99.9%)
```

---

## License

MIT — see [LICENSE](LICENSE)

## Author

**Adnan Anwar Awan** — PhD Electronic Engineering (WSN) · ML · Embedded Systems  
GitHub: [@AdnanAnwarAwan](https://github.com/AdnanAnwarAwan)
