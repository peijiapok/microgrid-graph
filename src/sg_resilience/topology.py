"""Radial-tree extraction for flow-constrained allocation (ADR-0001 / ADR-0002).

Builds a rooted radial tree from a pandapower/SimBench net, *honoring* line
in_service and switch states (the current benchmark_loader does NOT — see
ADR-0002 §4.1), and exposes per-edge thermal capacity F_e and downstream-load
membership so the branch-flow constraint

    |f_e(a)| = | sum_{j in subtree(e)} a_j | <= F_e

is computable. This is the data foundation for the flow-constrained feasible set
Delta_grid; it does NOT implement the differentiable projection (that is C1/C3,
Fanchen). It is reusable as the correctness oracle his fast projection matches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class RadialTree:
    root_bus: int
    edges: tuple[tuple[int, int], ...]          # (parent_bus, child_bus), oriented from root
    edge_capacity_mw: dict[tuple[int, int], float]  # F_e per oriented edge
    edge_kind: dict[tuple[int, int], str]       # "line" | "trafo"
    subtree_loads: dict[tuple[int, int], tuple[str, ...]]  # load node ids below each edge
    load_bus: dict[str, int]                    # load node id -> bus


def active_bus_graph(net) -> nx.Graph:
    """Bus graph honoring line.in_service and switch states.

    Shared by benchmark_loader so adjacency/features are built on the true
    active topology (the old loader ignored switch/in_service — ADR-0002 §4.1).
    """
    g = nx.Graph()
    for b in net.bus.index:
        g.add_node(int(b))

    open_line_sw: set[int] = set()
    closed_bb: list[tuple[int, int]] = []
    if hasattr(net, "switch") and len(net.switch):
        for _, r in net.switch.iterrows():
            if r["et"] == "l" and not bool(r["closed"]):
                open_line_sw.add(int(r["element"]))
            elif r["et"] == "b" and bool(r["closed"]):
                closed_bb.append((int(r["bus"]), int(r["element"])))

    for idx, r in net.line.iterrows():
        if "in_service" in net.line.columns and not bool(r["in_service"]):
            continue
        if int(idx) in open_line_sw:
            continue
        g.add_edge(int(r["from_bus"]), int(r["to_bus"]), kind="line", line_idx=int(idx))

    if hasattr(net, "trafo") and len(net.trafo):
        for idx, r in net.trafo.iterrows():
            if "in_service" in net.trafo.columns and not bool(r["in_service"]):
                continue
            g.add_edge(int(r["hv_bus"]), int(r["lv_bus"]), kind="trafo", trafo_idx=int(idx))

    for a, b in closed_bb:  # fuse closed bus-bus switches as zero-impedance edges
        g.add_edge(a, b, kind="switch")

    return g


def _line_capacity_mw(net, line_idx: int) -> float:
    """S = sqrt(3) * Vn_kv * Imax_kA  (MVA ~= MW at unity PF)."""
    row = net.line.loc[line_idx]
    imax = float(row.get("max_i_ka", 0.0) or 0.0)
    vn = float(net.bus.loc[int(row["from_bus"]), "vn_kv"])
    return math.sqrt(3.0) * vn * imax


def build_radial_tree(
    net,
    line_capacity_mw: float | None = None,
    trafo_capacity_mw: float | None = None,
) -> RadialTree:
    """Rooted radial tree with per-edge capacities F_e.

    `line_capacity_mw` / `trafo_capacity_mw` override the values derived from the
    net when set — use this for feeders whose ratings are non-physical
    placeholders (e.g. pandapower case33bw, whose `max_i_ka` yields F_e ~ 2e6 MW).
    When None, line caps come from `sqrt(3)·Vn·Imax` and trafo caps from `sn_mva`.
    """
    g = active_bus_graph(net)
    if len(net.ext_grid) == 0:
        raise ValueError("net has no ext_grid (slack) to root the tree")
    root = int(net.ext_grid["bus"].iloc[0])

    comp = nx.node_connected_component(g, root)
    sub = g.subgraph(comp)
    n, e = sub.number_of_nodes(), sub.number_of_edges()
    if e != n - 1:
        raise ValueError(
            f"feeder is not a radial tree on the slack component: |V|={n}, |E|={e} "
            f"(E-(V-1)={e - (n - 1)}). Honor switch states or pick a radial config."
        )

    # orient edges from root via BFS; record parent
    parent: dict[int, int] = {root: root}
    oriented: list[tuple[int, int]] = []
    edge_kind: dict[tuple[int, int], str] = {}
    edge_cap: dict[tuple[int, int], float] = {}
    for u, v in nx.bfs_edges(sub, root):
        parent[v] = u
        oriented.append((u, v))
        data = sub.get_edge_data(u, v)
        kind = data.get("kind", "line")
        edge_kind[(u, v)] = kind
        if kind == "line":
            edge_cap[(u, v)] = (
                line_capacity_mw if line_capacity_mw is not None
                else _line_capacity_mw(net, data["line_idx"])
            )
        elif kind == "trafo":
            edge_cap[(u, v)] = (
                trafo_capacity_mw if trafo_capacity_mw is not None
                else float(net.trafo.loc[data["trafo_idx"], "sn_mva"])
            )
        else:  # fused switch: effectively unconstrained
            edge_cap[(u, v)] = math.inf

    # load -> bus map (load node ids match benchmark_loader: "load_{idx}")
    load_bus = {f"load_{idx}": int(r["bus"]) for idx, r in net.load.iterrows()}

    # for each oriented edge (u->v), the subtree is everything reachable from v
    # without going back through u. Compute via the tree's child sets.
    children: dict[int, list[int]] = {b: [] for b in comp}
    for u, v in oriented:
        children[u].append(v)

    def subtree_buses(start: int) -> set[int]:
        seen, stack = set(), [start]
        while stack:
            b = stack.pop()
            if b in seen:
                continue
            seen.add(b)
            stack.extend(children[b])
        return seen

    subtree_loads: dict[tuple[int, int], tuple[str, ...]] = {}
    for (u, v) in oriented:
        buses = subtree_buses(v)
        loads = tuple(sorted(nid for nid, bus in load_bus.items() if bus in buses))
        subtree_loads[(u, v)] = loads

    return RadialTree(
        root_bus=root,
        edges=tuple(oriented),
        edge_capacity_mw=edge_cap,
        edge_kind=edge_kind,
        subtree_loads=subtree_loads,
        load_bus=load_bus,
    )


def branch_flows(tree: RadialTree, allocation: dict[str, float]) -> dict[tuple[int, int], float]:
    """f_e(a) = sum of allocations to loads in the subtree below e."""
    return {
        e: float(sum(allocation.get(nid, 0.0) for nid in loads))
        for e, loads in tree.subtree_loads.items()
    }


def flow_violations(
    tree: RadialTree, allocation: dict[str, float], tol: float = 1e-6
) -> dict[tuple[int, int], float]:
    """Edges where |f_e(a)| exceeds F_e, with the overage amount.

    Default tol matches the protocol's absolute feasibility tolerance
    (docs/05 §2.6 epsilon_feas = 1e-6); QP solvers leave residuals at this scale.
    """
    flows = branch_flows(tree, allocation)
    out = {}
    for e, f in flows.items():
        cap = tree.edge_capacity_mw[e]
        if math.isfinite(cap) and abs(f) > cap + tol:
            out[e] = abs(f) - cap
    return out
