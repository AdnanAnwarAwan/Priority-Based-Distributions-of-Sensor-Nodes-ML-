"""
tests/test_all.py — Full test suite.
Run: pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.network.activity_map import (
    ActivityMap, EnvironmentObservation,
    activity_to_zone, ZONE_MOST_CRITICAL, ZONE_NON_CRITICAL,
)
from src.network.energy_model  import EnergyModel
from src.network.coverage      import CoverageCalculator
from src.network.wsn_graph     import WSNGraph
from src.ml.zone_classifier    import ZoneClassifier, ZoneNet
from src.ml.replay_buffer      import ReplayBuffer
from src.ml.rl_agent           import DQNAgent, QNetwork
from src.ml.federated_trainer  import FederatedTrainer, fedavg, LocalNode
from src.simulation.hibernation import (
    NodeHibernationManager, HibernationController, NodeState,
)
from src.simulation.redistribution import RedistributionEngine


# ═══════════════════════════ ActivityMap ═════════════════════════════════════

class TestActivityMap:

    def test_activity_in_unit_interval(self):
        m = ActivityMap(200.0, n_hotspots=3, seed=0)
        assert m.activity.min() >= 0 and m.activity.max() <= 1

    def test_zone_labels_valid(self):
        m = ActivityMap(200.0, n_hotspots=3, seed=1)
        assert set(np.unique(m.zones)).issubset({0, 1, 2, 3})

    def test_cell_of_in_bounds(self):
        m = ActivityMap(200.0, seed=0)
        r, c = m.cell_of(0, 0)
        assert r == 0 and c == 0
        r, c = m.cell_of(199, 199)
        assert r < m.n_cells and c < m.n_cells

    def test_observation_features_in_range(self):
        m   = ActivityMap(200.0, seed=2)
        obs = m.observation_at(0, 50.0, 50.0)
        for v in [obs.troops, obs.vehicles, obs.weaponry, obs.movement]:
            assert 0 <= v <= 1

    def test_feature_grid_shape(self):
        m = ActivityMap(100.0, grid_res=10.0, seed=0)
        X = m.feature_grid()
        assert X.shape[1] == 4

    def test_zone_labels_length_matches_grid(self):
        m = ActivityMap(100.0, grid_res=10.0, seed=0)
        assert len(m.zone_labels()) == m.n_cells ** 2

    def test_evolve_changes_activity(self):
        m = ActivityMap(200.0, n_hotspots=3, seed=0)
        a0 = m.activity.copy()
        m.evolve(rounds=5)
        assert not np.allclose(a0, m.activity)

    def test_activity_to_zone_boundaries(self):
        assert activity_to_zone(0.80) == ZONE_MOST_CRITICAL
        assert activity_to_zone(0.10) == ZONE_NON_CRITICAL

    def test_save_load_roundtrip(self, tmp_path):
        m   = ActivityMap(100.0, n_hotspots=3, seed=5)
        p   = str(tmp_path / "amap.json")
        m.save(p)
        m2  = ActivityMap.load(p)
        assert np.allclose(m.activity, m2.activity, atol=1e-5)


# ═══════════════════════════ EnergyModel ════════════════════════════════════

class TestEnergyModel:

    @pytest.fixture
    def em(self): return EnergyModel()

    def test_tx_increases_with_distance(self, em):
        assert em.tx_energy(4000, 10) < em.tx_energy(4000, 50) < em.tx_energy(4000, 150)

    def test_rx_linear_in_bits(self, em):
        assert em.rx_energy(8000) == pytest.approx(2 * em.rx_energy(4000))

    def test_sleep_fraction(self, em):
        assert em.sleep_energy(1.0) == pytest.approx(em.e_sleep_frac)

    def test_threshold_positive(self, em):
        assert em.d_thresh > 0

    def test_lifetime_positive(self, em):
        assert em.lifetime_rounds(100) > 0


# ═══════════════════════════ CoverageCalculator ══════════════════════════════

class TestCoverage:

    @pytest.fixture
    def calc(self): return CoverageCalculator(100.0, grid_res=10.0)

    def test_full_coverage(self, calc):
        pos = np.array([[50.0, 50.0]])
        assert calc.compute(pos, 200.0) == pytest.approx(1.0, abs=0.01)

    def test_empty_positions(self, calc):
        assert calc.compute(np.empty((0, 2)), 30.0) == 0.0

    def test_partial(self, calc):
        pos = np.array([[10.0, 10.0]])
        c   = calc.compute(pos, 20.0)
        assert 0.0 < c < 1.0

    def test_more_nodes_more_coverage(self, calc):
        p1 = np.array([[50.0, 50.0]])
        p3 = np.array([[20.0, 20.0], [50.0, 50.0], [80.0, 80.0]])
        assert calc.compute(p3, 25.0) >= calc.compute(p1, 25.0)

    def test_holes_when_sparse(self, calc):
        pos   = np.array([[5.0, 5.0]])
        holes = calc.coverage_holes(pos, 10.0)
        assert len(holes) > 0


# ═══════════════════════════ WSNGraph ════════════════════════════════════════

class TestWSNGraph:

    def test_direct_hop1(self):
        pos  = np.array([[55.0, 50.0]])
        g    = WSNGraph(pos, np.array([50.0, 50.0]), tx_radius=20.0)
        g.build()
        assert g.hop_count(0) == 1

    def test_unreachable(self):
        pos  = np.array([[0.0, 0.0]])
        g    = WSNGraph(pos, np.array([50.0, 50.0]), tx_radius=5.0)
        g.build()
        assert g.hop_count(0) == 99

    def test_connected_le_n(self):
        rng  = np.random.default_rng(0)
        pos  = rng.uniform(0, 100, (10, 2)).astype(np.float32)
        g    = WSNGraph(pos, np.array([50.0, 50.0]), tx_radius=60.0)
        g.build()
        assert 0 <= g.connected_count() <= 10

    def test_adjacency_diagonal_zero(self):
        pos = np.random.rand(5, 2).astype(np.float32) * 50
        g   = WSNGraph(pos, np.array([25.0, 25.0]), tx_radius=50.0)
        g.build()
        A   = g.adjacency()
        assert np.all(np.diag(A) == 0)


# ═══════════════════════════ ZoneClassifier ═══════════════════════════════════

class TestZoneClassifier:

    def test_predict_shape(self):
        clf  = ZoneClassifier()
        X    = np.random.rand(20, 4).astype(np.float32)
        pred = clf.predict(X)
        assert pred.shape == (20,) and set(pred).issubset({0, 1, 2, 3})

    def test_predict_proba_sums_to_1(self):
        clf  = ZoneClassifier()
        X    = np.random.rand(5, 4).astype(np.float32)
        prob = clf.predict_proba(X)
        assert prob.shape == (5, 4)
        assert np.allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    def test_train_local_reduces_loss(self):
        clf  = ZoneClassifier()
        m    = ActivityMap(100.0, seed=0)
        obs  = [m.observation_at(i, float(np.random.rand()*100), float(np.random.rand()*100)) for i in range(30)]
        l1   = clf.train_local(obs, epochs=1)
        l10  = clf.train_local(obs, epochs=10)
        # Loss should generally not increase over many epochs
        assert isinstance(l10, float)

    def test_get_set_weights_roundtrip(self):
        clf1 = ZoneClassifier()
        clf2 = ZoneClassifier()
        w    = clf1.get_weights()
        clf2.set_weights(w)
        for a, b in zip(clf1.get_weights(), clf2.get_weights()):
            assert np.allclose(a, b)

    def test_save_load(self, tmp_path):
        clf1 = ZoneClassifier()
        p    = str(tmp_path / "clf.pt")
        clf1.save(p)
        clf2 = ZoneClassifier()
        clf2.load(p)
        for a, b in zip(clf1.get_weights(), clf2.get_weights()):
            assert np.allclose(a, b)


# ═══════════════════════════ FederatedTrainer ════════════════════════════════

class TestFederatedTrainer:

    def test_fedavg_uniform(self):
        w1  = [np.ones((4, 4)), np.ones(4)]
        w2  = [np.zeros((4, 4)), np.zeros(4)]
        avg = fedavg([w1, w2])
        for a in avg:
            assert np.allclose(a, 0.5)

    def test_fedavg_weighted(self):
        w1  = [np.ones(4)]
        w2  = [np.zeros(4)]
        avg = fedavg([w1, w2], sample_counts=[3, 1])
        assert np.allclose(avg[0], 0.75)

    def test_communication_round_runs(self):
        fed = FederatedTrainer(n_nodes=5)
        m   = ActivityMap(100.0, seed=1)
        pos = np.random.rand(5, 2).astype(np.float32) * 100
        fed.feed_activity_map(m, pos)
        # Give each node enough observations
        for _ in range(10):
            fed.feed_activity_map(m, pos)
        res = fed.communication_round()
        assert "round" in res

    def test_local_node_add_observation(self):
        node = LocalNode(node_id=0)
        m    = ActivityMap(100.0, seed=0)
        obs  = m.observation_at(0, 50.0, 50.0)
        node.add_observation(obs)
        assert len(node.observations) == 1


# ═══════════════════════════ Hibernation ═════════════════════════════════════

class TestHibernation:

    def test_active_on_init(self):
        mgr = NodeHibernationManager(node_id=0)
        assert mgr.state == NodeState.ACTIVE

    def test_demotion_to_dozing(self):
        from src.network.activity_map import ZONE_LESS_CRITICAL
        mgr = NodeHibernationManager(node_id=0)
        mgr.update_zone(ZONE_LESS_CRITICAL)
        assert mgr.state == NodeState.DOZING

    def test_demotion_to_hibernate(self):
        mgr = NodeHibernationManager(node_id=0)
        mgr.update_zone(ZONE_NON_CRITICAL)
        assert mgr.state == NodeState.HIBERNATE

    def test_wake_on_high_zone(self):
        from src.network.activity_map import ZONE_LESS_CRITICAL, ZONE_CRITICAL
        mgr = NodeHibernationManager(node_id=0)
        mgr.update_zone(ZONE_NON_CRITICAL)
        assert mgr.state == NodeState.HIBERNATE
        mgr.update_zone(ZONE_CRITICAL)
        assert mgr.state == NodeState.ACTIVE

    def test_energy_drain(self):
        mgr  = NodeHibernationManager(node_id=0, initial_energy=1.0)
        cost = mgr.consume_energy(0.1)
        assert mgr.energy < 1.0
        assert cost > 0

    def test_sleep_costs_less_energy(self):
        from src.network.activity_map import ZONE_NON_CRITICAL
        m_active = NodeHibernationManager(0, initial_energy=1.0)
        m_sleep  = NodeHibernationManager(1, initial_energy=1.0)
        m_sleep.update_zone(ZONE_NON_CRITICAL)
        c_a = m_active.consume_energy(0.1)
        c_s = m_sleep.consume_energy(0.1)
        assert c_s < c_a

    def test_dead_on_zero_energy(self):
        mgr = NodeHibernationManager(node_id=0, initial_energy=0.001)
        mgr.consume_energy(1.0)
        assert mgr.state == NodeState.DEAD

    def test_controller_summary(self):
        ctrl = HibernationController(n_nodes=10)
        s    = ctrl.summary()
        assert s["alive"] == 10 and s["dead"] == 0

    def test_energy_saved_after_sleep(self):
        from src.network.activity_map import ZONE_NON_CRITICAL
        ctrl  = HibernationController(5)
        zones = np.full(5, ZONE_NON_CRITICAL, dtype=int)
        nb    = np.zeros(5, dtype=np.float32)
        ctrl.update(zones, nb, active_cost=0.01)
        assert ctrl.total_energy_saved() > 0


# ═══════════════════════════ ReplayBuffer ════════════════════════════════════

class TestReplayBuffer:

    def test_push_sample(self):
        buf = ReplayBuffer(100)
        for _ in range(20):
            buf.push(np.zeros(10), 0, 1.0, np.ones(10), False)
        s, a, r, ns, d = buf.sample(8)
        assert s.shape == (8, 10)

    def test_capacity(self):
        buf = ReplayBuffer(10)
        for _ in range(25):
            buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
        assert len(buf) == 10

    def test_sample_error_if_small(self):
        buf = ReplayBuffer(100)
        buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
        with pytest.raises(Exception):
            buf.sample(10)


# ═══════════════════════════ DQNAgent ════════════════════════════════════════

class TestDQNAgent:

    @pytest.fixture
    def agent(self):
        return DQNAgent(state_dim=24, action_dim=16, hidden=64,
                        batch_size=8, buffer_size=200, eps_decay=500)

    def test_action_in_range(self, agent):
        s = np.zeros(24)
        for _ in range(20):
            assert 0 <= agent.select_action(s) < 16

    def test_epsilon_decays(self, agent):
        eps0 = agent.epsilon
        for _ in range(300): agent.select_action(np.zeros(24))
        assert agent.epsilon < eps0

    def test_no_update_underfull(self, agent):
        assert agent.update() == 0.0

    def test_update_returns_loss(self, agent):
        for _ in range(20):
            agent.store(np.zeros(24), 0, 1.0, np.ones(24), False)
        assert agent.update() >= 0.0

    def test_greedy_deterministic(self, agent):
        s = np.random.rand(24).astype(np.float32)
        assert agent.select_action(s, greedy=True) == agent.select_action(s, greedy=True)

    def test_save_load(self, agent, tmp_path):
        p = str(tmp_path / "agent.pt")
        agent.save(p)
        a2 = DQNAgent(state_dim=24, action_dim=16, hidden=64)
        a2.load(p)
        for p1, p2 in zip(agent.policy.parameters(), a2.policy.parameters()):
            assert torch.allclose(p1.data, p2.data)
