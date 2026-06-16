"""Diagnostic: is topology load-bearing in the DECISION (not just the executor)?

Per GPT-5.5: before rewriting the learner, test whether the optimal critical-set
selection departs from naive priority-greedy and whether that gap grows as flow
caps tighten. If the gap is ~0, priority-greedy is near-optimal and no learner
(graph or not) can show a topology benefit -> honest negative result.

For each feeder at several flow-cap tightness levels, on the peak-demand step:
  - greedy_W  = critical weight servable at min-service by PRIORITY-order greedy
                (respecting budget + branch-flow caps);
  - lp_W      = LP-relaxed max servable critical weight (upper bound on any set);
  - room      = lp_W - greedy_W   (fraction of total critical weight).
Large/growing room => topology-dependent set-selection exists => graph can help.

Run: PYTHONPATH=src python scripts/diagnose_decision_room.py
"""
from __future__ import annotations

import numpy as np

from sg_resilience.eval_harness import _FeederCtx, calibrate_budget_oracle


def _critical_subtree_matrix(ctx):
    finite = [e for e in ctx.tree.edges if np.isfinite(ctx.tree.edge_capacity_mw[e])]
    caps = np.array([ctx.tree.edge_capacity_mw[e] for e in finite])
    cids = ctx.critical_ids
    S = np.zeros((len(finite), len(cids)))
    for ei, e in enumerate(finite):
        ds = ctx.tree.subtree_loads[e]
        for j, cid in enumerate(cids):
            if cid in ds:
                S[ei, j] = 1.0
    return S, caps


def greedy_priority_weight(need, w, power, S, caps):
    """Commit criticals in priority(=w) order if min-service fits budget+flow."""
    order = np.argsort(-w)
    budget = power
    edge_rem = caps.copy()
    served = 0.0
    for j in order:
        r = need[j]
        if r <= budget + 1e-12 and np.all(S[:, j] * r <= edge_rem + 1e-12):
            budget -= r
            edge_rem -= S[:, j] * r
            served += w[j]
    return served


def lp_upper_bound(need, w, power, S, caps):
    import cvxpy as cp
    k = need.shape[0]
    x = cp.Variable(k)
    cons = [x >= 0, x <= 1, need @ x <= power, S @ cp.multiply(need, x) <= caps]
    prob = cp.Problem(cp.Maximize(w @ x), cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
    return float(w @ x.value) if x.value is not None else float("nan")


def main() -> int:
    import warnings, yaml
    warnings.simplefilter("ignore")
    split = yaml.safe_load(open("configs/topology_split_v1.yaml", encoding="utf-8"))
    feeders = [dict(e, role="train") for e in split["G_train"]] + \
              [dict(e, role="ood") for e in split["G_ood"]]
    print(f"{'feeder':24} {'role':5} {'capx':>5} {'greedyW':>8} {'lpW':>6} {'room':>6}")
    print("-" * 60)
    for e in feeders:
        ctx = _FeederCtx(e, horizon=32)
        ps = calibrate_budget_oracle(ctx, tuple(range(100, 108)))["power_scale"]
        # peak-demand step (seed 0)
        D, OUT, P = ctx.arrays(0, 0.05, 0.85, ps)
        t = int(np.argmax(D.sum(axis=1)))
        dc = D[t, ctx.cidx]; oc = OUT[t, ctx.cidx]
        need = ctx.cm * dc * (~oc)          # min-service demand, 0 if outaged
        w = ctx.cw.copy()
        S0, caps0 = _critical_subtree_matrix(ctx)
        Wtot = float(w.sum()) + 1e-9
        for capx in (1.0, 0.5, 0.25, 0.125):
            caps = caps0 * capx
            gW = greedy_priority_weight(need, w, P[t], S0, caps) / Wtot
            lW = lp_upper_bound(need, w, P[t], S0, caps) / Wtot
            print(f"{e['id'][:24]:24} {e['role']:5} {capx:5.3f} {gW:8.3f} {lW:6.3f} {lW-gW:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
