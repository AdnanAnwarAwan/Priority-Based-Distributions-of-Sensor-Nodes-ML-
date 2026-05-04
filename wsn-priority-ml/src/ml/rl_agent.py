"""
rl_agent.py — DQN Agent for Priority-Aware Node Redistribution.

Learns when to migrate, hibernate, or wake nodes to maximise
network lifetime and critical-zone coverage.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from src.ml.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Double DQN for WSN node action selection (STAY/MIGRATE/HIBERNATE/WAKE)."""

    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        hidden:      int   = 256,
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        tau:         float = 0.005,
        buffer_size: int   = 10_000,
        batch_size:  int   = 64,
        eps_start:   float = 1.0,
        eps_end:     float = 0.01,
        eps_decay:   int   = 50_000,
        device:      str   = "auto",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self._steps     = 0

        self.policy = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt    = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    @property
    def epsilon(self) -> float:
        t = min(1.0, self._steps / self.eps_decay)
        return self.eps_start + t * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        self._steps += 1
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy(s).argmax(1).item())

    def store(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_curr   = self.policy(s).gather(1, a)
        with torch.no_grad():
            a_next   = self.policy(ns).argmax(1, keepdim=True)
            q_target = r + self.gamma * self.target(ns).gather(1, a_next) * (1 - d)

        loss = nn.functional.smooth_l1_loss(q_curr, q_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        for p, tp in zip(self.policy.parameters(), self.target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return float(loss.item())

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(),
                    "target": self.target.state_dict(),
                    "opt":    self.opt.state_dict(),
                    "steps":  self._steps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.opt.load_state_dict(ckpt["opt"])
        self._steps = ckpt.get("steps", 0)
