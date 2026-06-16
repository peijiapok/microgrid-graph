"""Tests for the reference Delta_grid projection and flow-constrained oracle."""
from __future__ import annotations

import math

import numpy as np
import pandapower as pp
import pytest

from sg_resilience import metrics_v1 as M
from sg_resilience.flow_projection import make_flow_oracle, project_onto_delta_grid
from sg_resilience.topology import branch_flows, build_radial_tree, flow_violations

cp = pytest.importorskip("cvxpy")


def _line_chain(max_i_ka=0.5):
    """ext_grid@0 -> 1 -> 2 -> 3, loads at 1,2,3."""
    net = pp.create_empty_network()
    for _ in range(4):
        pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, bus=0)
    for f, t in [(0, 1), (1, 2), (2, 3)]:
        pp.create_line_from_parameters(
            net, from_bus=f, to_bus=t, length_km=0.1,
            r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, max_i_ka=max_i_ka,
        )
    for b in (1, 2, 3):
        pp.create_load(net, bus=b, p_mw=0.2)
    return net


def test_projection_is_feasible():
    tree = build_radial_tree(_line_chain())
    order = ["load_0", "load_1", "load_2"]
    d = np.array([0.2, 0.2, 0.2])
    z = np.array([0.2, 0.2, 0.2])  # wants full service, but root cap ~0.346 < 0.6
    outage = np.zeros(3, dtype=bool)
    a = project_onto_delta_grid(z, d, power=10.0, outage=outage, tree=tree, node_order=order)
    alloc = dict(zip(order, a))
    # all hard constraints hold
    assert np.all(a >= -1e-6) and np.all(a <= d + 1e-6)
    assert a.sum() <= 10.0 + 1e-6
    assert not flow_violations(tree, alloc)           # flow caps respected
    # root edge saturates near its cap (0.6 demanded, ~0.346 deliverable)
    cap = tree.edge_capacity_mw[(0, 1)]
    assert branch_flows(tree, alloc)[(0, 1)] <= cap + 1e-6


def test_outage_forces_zero():
    tree = build_radial_tree(_line_chain())
    order = ["load_0", "load_1", "load_2"]
    d = np.array([0.2, 0.2, 0.2])
    z = np.array([0.2, 0.2, 0.2])
    outage = np.array([False, True, False])  # load_1 outaged
    a = project_onto_delta_grid(z, d, power=10.0, outage=outage, tree=tree, node_order=order)
    assert a[1] <= 1e-6


def test_greedy_allocation_feasible_by_construction():
    from sg_resilience.flow_projection import greedy_flow_allocation, node_ancestor_edges
    from sg_resilience.topology import flow_violations
    tree = build_radial_tree(_line_chain())  # caps ~0.346 MW/edge
    order = ["load_0", "load_1", "load_2"]
    anc, caps = node_ancestor_edges(tree, order)
    d = np.array([0.2, 0.2, 0.2])
    pr = np.array([3.0, 2.0, 1.0]); mf = np.full(3, 0.5); crit = np.array([True, True, True])
    outage = np.array([False, True, False])  # load_1 outaged
    a = greedy_flow_allocation(d, power=10.0, outage=outage, anc=anc, caps=caps,
                               priorities=pr, minfrac=mf, critical=crit)
    assert a[1] == 0.0                                   # outaged -> 0
    assert np.all(a >= -1e-12) and np.all(a <= d + 1e-9)  # box
    assert not flow_violations(tree, dict(zip(order, a)))  # branch-flow respected
    # root edge cap ~0.346 binds: total served below cap
    assert a.sum() <= tree.edge_capacity_mw[(0, 1)] + 1e-9


def test_flow_oracle_caps_bind():
    """Root cap ~0.346 MW; three critical loads each needing 0.2 min-service
    cannot all be continuously served -> oracle commits a strict subset."""
    tree = build_radial_tree(_line_chain())
    crit = ["load_0", "load_1", "load_2"]
    oracle = make_flow_oracle(tree, crit)
    T = 5
    d = np.full((T, 3), 0.2)
    m = np.full(3, 1.0)              # min-service = full demand
    w = np.array([3.0, 2.0, 1.0])
    power = np.full(T, 10.0)         # budget is loose; flow cap is the binder
    outage = np.zeros((T, 3), dtype=bool)
    cont = oracle(d, m, w, power, outage)
    # root edge cap ~0.346 -> at most one 0.2 load fits its whole-feeder flow
    assert cont.sum() < 3
    assert cont[0] == 1.0           # highest weight committed first


def test_flow_oracle_plugs_into_metrics():
    tree = build_radial_tree(_line_chain())
    crit = ["load_0", "load_1", "load_2"]
    oracle = make_flow_oracle(tree, crit)
    T = 5
    d = np.full((T, 3), 0.2)
    m = np.full(3, 1.0)
    w = np.array([3.0, 2.0, 1.0])
    power = np.full(T, 10.0)
    outage = np.zeros((T, 3), dtype=bool)
    a = np.full((T, 3), 0.2)        # policy serves all fully (flow-infeasible, but metric just scores served)
    res = M.rollout_metrics(a, d, m, w, power, outage, windows=(2, 4), oracle=oracle)
    assert 0.0 <= res["C"] <= 1.0
    # denominator reflects the flow-limited oracle, not all three nodes
    assert res["C_den"] < float(np.sum(w))
