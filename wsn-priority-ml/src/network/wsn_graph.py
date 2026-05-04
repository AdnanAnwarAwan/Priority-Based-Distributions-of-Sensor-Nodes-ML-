"""wsn_graph.py — NetworkX WSN topology (BFS hops, Dijkstra, adjacency)."""
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List, Optional


class WSNGraph:
    SINK = "sink"

    def __init__(self, positions, sink, tx_radius, energies=None):
        self.positions = positions
        self.sink_pos  = sink
        self.r         = tx_radius
        self.energies  = energies if energies is not None else np.ones(len(positions))
        self.N         = len(positions)
        self.G         = nx.DiGraph()
        self._hops: Dict[int, int] = {}

    def build(self):
        self.G.clear()
        self.G.add_node(self.SINK, pos=self.sink_pos)
        for i in range(self.N):
            self.G.add_node(i, pos=self.positions[i], energy=float(self.energies[i]))
        for i in range(self.N):
            d = float(np.linalg.norm(self.positions[i] - self.sink_pos))
            if d <= self.r:
                w = d / max(self.energies[i], 1e-8)
                self.G.add_edge(self.SINK, i, weight=w)
                self.G.add_edge(i, self.SINK, weight=w)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d <= self.r:
                    self.G.add_edge(i, j, weight=d / max(self.energies[i], 1e-8))
                    self.G.add_edge(j, i, weight=d / max(self.energies[j], 1e-8))
        self._bfs()

    def _bfs(self):
        self._hops = {}
        try:
            for node, h in nx.single_source_shortest_path_length(self.G, self.SINK).items():
                if isinstance(node, int):
                    self._hops[node] = h
        except nx.NetworkXError:
            pass

    def hop_count(self, i): return self._hops.get(i, 99)
    def is_connected(self, i): return i in self._hops
    def connected_count(self): return len(self._hops)
    def all_degrees(self): return np.array([self.G.degree(i) for i in range(self.N)], dtype=np.float32)
    def adjacency(self):
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.G.has_edge(i, j):
                    A[i, j] = 1.0
        return A
