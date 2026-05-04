"""replay_buffer.py — Uniform experience replay buffer."""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Tuple:
        idx   = np.random.choice(len(self._buf), batch_size, replace=False)
        batch = [self._buf[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(ns), np.array(d, dtype=np.float32)

    def __len__(self): return len(self._buf)
