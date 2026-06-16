"""Tests for topology.build_radial_tree on a tiny deterministic feeder."""
from __future__ import annotations

import math

import pandapower as pp
import pytest

from sg_resilience.topology import branch_flows, build_radial_tree, flow_violations


def _line_chain():
    """ext_grid@0 -> 1 -> 2 -> 3, loads at 1,2,3. A 3-edge radial chain."""
    net = pp.create_empty_network()
    for _ in range(4):
        pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, bus=0)
    for f, t in [(0, 1), (1, 2), (2, 3)]:
        pp.create_line_from_parameters(
            net, from_bus=f, to_bus=t, length_km=0.1,
            r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, max_i_ka=0.5,
        )
    for b in (1, 2, 3):
        pp.create_load(net, bus=b, p_mw=0.05)
    return net


def test_radial_tree_structure():
    tree = build_radial_tree(_line_chain())
    assert tree.root_bus == 0
    assert tree.edges == ((0, 1), (1, 2), (2, 3))
    # subtree below the root edge (0,1) contains all three loads;
    # below (2,3) only the deepest load.
    assert set(tree.subtree_loads[(0, 1)]) == {"load_0", "load_1", "load_2"}
    assert set(tree.subtree_loads[(2, 3)]) == {"load_2"}


def test_branch_flows_accumulate_downstream():
    tree = build_radial_tree(_line_chain())
    alloc = {"load_0": 0.05, "load_1": 0.05, "load_2": 0.05}  # buses 1,2,3
    flows = branch_flows(tree, alloc)
    assert math.isclose(flows[(0, 1)], 0.15)   # carries all three
    assert math.isclose(flows[(1, 2)], 0.10)   # carries two
    assert math.isclose(flows[(2, 3)], 0.05)   # carries one


def test_flow_capacity_and_violation():
    tree = build_radial_tree(_line_chain())
    # cap = sqrt(3) * 0.4 kV * 0.5 kA  ~= 0.3464 MW per edge
    cap = tree.edge_capacity_mw[(0, 1)]
    assert math.isclose(cap, math.sqrt(3) * 0.4 * 0.5, rel_tol=1e-9)
    # allocation whose root flow (0.45) exceeds cap (0.3464) -> violation on (0,1)
    alloc = {"load_0": 0.15, "load_1": 0.15, "load_2": 0.15}
    viol = flow_violations(tree, alloc)
    assert (0, 1) in viol
    assert viol[(0, 1)] > 0


def test_capacity_override():
    # default cap = sqrt(3)*0.4*0.5 ~= 0.346; override to 0.1 MW
    tree = build_radial_tree(_line_chain(), line_capacity_mw=0.1)
    assert all(abs(c - 0.1) < 1e-12 for c in tree.edge_capacity_mw.values())
    alloc = {"load_0": 0.05, "load_1": 0.05, "load_2": 0.05}  # root flow 0.15 > 0.1
    assert (0, 1) in flow_violations(tree, alloc)


def test_nonradial_rejected():
    net = _line_chain()
    # add a loop edge 3->0 to break radiality
    pp.create_line_from_parameters(
        net, from_bus=3, to_bus=0, length_km=0.1,
        r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, max_i_ka=0.5,
    )
    with pytest.raises(ValueError, match="not a radial tree"):
        build_radial_tree(net)
