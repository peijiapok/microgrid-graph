"""Score policies for the H1 graph-necessity experiment (torch-only).

All policies map per-node LOCAL features (+ adjacency for graph models) to a
per-node score in [0,1] (interpreted as desired served fraction). The hard
flow-aware greedy allocator (flow_projection.greedy_flow_allocation, order by
score) turns scores into a feasible allocation at eval — identical for every
policy, so the comparison isolates decision quality.

H1 fairness: node features are LOCAL only (demand, criticality, min-service,
prev-alloc, outage, priority, global budget). Topology is available to graph
models ONLY through message passing over the adjacency — never as a precomputed
node feature — so DeepSets genuinely cannot perceive structure.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_FEAT = 7


def node_features(d, prev_a, outage, minfrac, priorities, critical, budget) -> torch.Tensor:
    """(n, N_FEAT) local features for one step; everything normalized per-feeder."""
    d = np.asarray(d, float)
    tot = max(float(d.sum()), 1e-9)
    pmax = max(float(np.max(priorities)), 1e-9)
    f = np.stack([
        d / tot,
        np.asarray(minfrac, float),
        np.asarray(critical, float),
        np.asarray(prev_a, float) / tot,
        np.asarray(outage, float),
        np.asarray(priorities, float) / pmax,
        np.full_like(d, float(budget) / tot),
    ], axis=1)
    return torch.tensor(f, dtype=torch.float32)


class _Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, h):
        return torch.sigmoid(self.net(h)).squeeze(-1)


class DeepSetsPolicy(nn.Module):
    """No-graph, size-generalizing: per-node MLP + mean-pooled global context."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = _Head(2 * hidden)

    def forward(self, x, adj=None):
        h = self.phi(x)                       # (n,H)
        g = h.mean(dim=0, keepdim=True).expand(h.shape[0], -1)  # pooled context
        return self.head(torch.cat([h, g], dim=-1))


class _SAGELayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.self_w = nn.Linear(din, dout)
        self.neigh_w = nn.Linear(din, dout)

    def forward(self, h, adj):
        return torch.relu(self.self_w(h) + self.neigh_w(adj @ h))  # adj row-normalized


class GraphSAGEPolicy(nn.Module):
    """Topology via message passing over the (row-normalized) load adjacency."""
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        dims = [N_FEAT] + [hidden] * layers
        self.layers = nn.ModuleList(_SAGELayer(dims[i], dims[i + 1]) for i in range(layers))
        self.head = _Head(hidden)

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return self.head(h)


def build_policy(kind: str, hidden: int = 64, layers: int = 3) -> nn.Module:
    if kind == "deepsets":
        return DeepSetsPolicy(hidden)
    if kind in ("graphsage", "gnn"):
        return GraphSAGEPolicy(hidden, layers)
    raise ValueError(f"unknown policy kind: {kind}")
