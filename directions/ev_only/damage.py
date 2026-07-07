"""Damaged IEEE-feeder scenario generator.

A disaster removes feeder line segments; on a radial feeder, removing a line
de-energizes its whole downstream subtree. The surviving, source-connected feeder
is a NEW topology (different each scenario) — this is the topology family for the
zero-shot generalization / readiness study.
"""
from __future__ import annotations

import copy

import numpy as np

from sg_resilience.topology import build_radial_tree


def damage_feeder(net, n_remove: int, rng: np.random.Generator, line_cap_mw: float | None = 6.0):
    """Return (damaged_tree, surviving_load_ids). Removes `n_remove` random
    in-service lines; the surviving feeder = source-connected component.
    `line_cap_mw` overrides placeholder ratings (e.g. case33bw)."""
    net = copy.deepcopy(net)
    in_svc = net.line.index[net.line["in_service"]].tolist() if "in_service" in net.line \
        else net.line.index.tolist()
    k = min(n_remove, max(len(in_svc) - 1, 0))
    if k > 0:
        remove = rng.choice(in_svc, size=k, replace=False)
        net.line.loc[remove, "in_service"] = False
    tree = build_radial_tree(net, line_capacity_mw=line_cap_mw)
    surviving_buses = {tree.root_bus} | {v for (_, v) in tree.edges}
    surviving_loads = [nid for nid, bus in tree.load_bus.items() if bus in surviving_buses]
    return tree, surviving_loads
