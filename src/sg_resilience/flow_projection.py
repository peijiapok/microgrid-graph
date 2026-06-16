"""Reference (non-differentiable) flow-constrained feasible set Delta_grid.

This is the eval-time / correctness-oracle implementation of ADR-0001's
feasible set

    Delta_grid(G,s_t) = { a : 0<=a_i<=d_i, a_i=0 on outage, sum a<=P,
                          |f_e(a)|<=F_e  for every tree edge e }.

`project_onto_delta_grid` solves the Euclidean projection as a QP (OSQP via
cvxpy). It is NOT the differentiable training projection (that is C3, Fanchen) —
it exists so the metric denominator, feasibility audits, and Fanchen's fast
projection all have a ground-truth reference to match.

`make_flow_oracle` returns a metric-compatible oracle (signature
`(d,m,w,power,outage) -> cont(k,)`) that plugs into metrics_v1.rollout_metrics.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .topology import RadialTree

EPS = 1e-9


def _subtree_matrix(
    tree: RadialTree, node_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """S (|E| x n) with S[e,j]=1 if load node_order[j] is downstream of edge e,
    and finite capacity vector cap (|E|,). Infinite-capacity (fused-switch)
    edges are dropped."""
    idx = {nid: j for j, nid in enumerate(node_order)}
    rows, caps = [], []
    for e in tree.edges:
        cap = tree.edge_capacity_mw[e]
        if not np.isfinite(cap):
            continue
        row = np.zeros(len(node_order))
        for nid in tree.subtree_loads[e]:
            if nid in idx:
                row[idx[nid]] = 1.0
        rows.append(row)
        caps.append(cap)
    if not rows:
        return np.zeros((0, len(node_order))), np.zeros(0)
    return np.vstack(rows), np.array(caps)


def project_onto_delta_grid(
    z: np.ndarray,
    d: np.ndarray,
    power: float,
    outage: np.ndarray,
    tree: RadialTree,
    node_order: Sequence[str],
) -> np.ndarray:
    """Euclidean projection of raw scores z onto Delta_grid for one timestep."""
    import cvxpy as cp

    z = np.asarray(z, dtype=float)
    d = np.asarray(d, dtype=float)
    outage = np.asarray(outage, dtype=bool)
    n = z.shape[0]
    upper = np.where(outage, 0.0, d)  # box upper = demand, 0 if outaged

    S, cap = _subtree_matrix(tree, node_order)
    a = cp.Variable(n)
    cons = [a >= 0, a <= upper, cp.sum(a) <= power]
    if S.shape[0]:
        cons.append(S @ a <= cap)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(a - z)), cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, polishing=True,
               max_iter=20000, verbose=False)
    if a.value is None:
        # degenerate fallback: clip to box+budget (drops flow caps)
        out = np.clip(z, 0.0, upper)
        if out.sum() > power:
            out *= power / max(out.sum(), EPS)
        return out
    return np.clip(a.value, 0.0, upper)  # clip tiny solver overshoot back into box


def make_flow_oracle(
    tree: RadialTree, critical_node_ids: Sequence[str]
) -> Callable[..., np.ndarray]:
    """Build a metrics-compatible oracle that respects budget AND flow caps.

    Greedy priority-ordered: commit critical nodes in descending weight; a node
    is committed iff serving its min-service m_i*d_{t,i} at every non-outaged
    step is feasible given already-committed nodes, on every ancestor edge and
    the global budget. Non-critical loads are assumed shed (set to 0) by the
    oracle, so only critical allocations consume budget/flow.
    Returns `oracle(d,m,w,power,outage) -> cont(k,)` in {0,1}.
    """
    # ancestor edges per critical node (edges whose subtree contains the node)
    finite_edges = [e for e in tree.edges if np.isfinite(tree.edge_capacity_mw[e])]
    anc: list[list[int]] = []
    for nid in critical_node_ids:
        anc.append([ei for ei, e in enumerate(finite_edges) if nid in tree.subtree_loads[e]])
    caps = np.array([tree.edge_capacity_mw[e] for e in finite_edges])

    def oracle(d: np.ndarray, m: np.ndarray, w: np.ndarray,
               power: np.ndarray, outage: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        m = np.asarray(m, dtype=float)
        w = np.asarray(w, dtype=float)
        power = np.asarray(power, dtype=float)
        outage = np.asarray(outage, dtype=bool)
        T, k = d.shape
        need = m[None, :] * d  # (T,k)
        edge_load = np.zeros((T, len(finite_edges)))  # committed flow per edge/step
        budget_used = np.zeros(T)
        committed = np.zeros(k, dtype=float)
        for i in np.argsort(-w):
            active = ~outage[:, i]
            req = np.where(active, need[:, i], 0.0)  # (T,)
            ok = np.all(budget_used + req <= power + EPS)
            if ok and anc[i]:
                for ei in anc[i]:
                    if not np.all(edge_load[:, ei] + req <= caps[ei] + EPS):
                        ok = False
                        break
            if ok:
                budget_used = budget_used + req
                for ei in anc[i]:
                    edge_load[:, ei] += req
                committed[i] = 1.0
        return committed

    return oracle
