"""H1 graph-necessity experiment: train learned policies by imitation of the
flow-aware greedy rule, evaluate via the hard score-driven greedy allocator, and
compare real-graph GNN vs no-graph DeepSets on cross-topology continuity C.

Methodology: ADR-0003 (oracle-calibrated matched scarcity). Plan: notes/h1_experiment_plan.md.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from . import metrics_v1 as M
from .eval_harness import _FeederCtx, calibrate_budget_oracle
from .flow_projection import greedy_flow_allocation
from .policies import build_policy, node_features

EPS = 1e-9


def _adj_tensor(ctx: _FeederCtx) -> torch.Tensor:
    a = np.array(ctx.scenario.adjacency_matrix, dtype=np.float32)
    if a.size == 0:  # no adjacency -> identity (degenerate)
        a = np.eye(len(ctx.node_order), dtype=np.float32)
    return torch.tensor(a)


def _rule_alloc(ctx: _FeederCtx, d, power, outage) -> np.ndarray:
    return greedy_flow_allocation(d, power, outage, ctx.anc, ctx.caps,
                                  ctx.priorities, ctx.minfrac, ctx.critical_mask)


def build_dataset(ctx: _FeederCtx, seeds, power_scale, p_out=0.05, p_stay=0.85):
    """Per-step (features, adjacency, target served-fraction) with teacher-forced
    previous allocation from the rule. Target = rule allocation / demand."""
    adj = _adj_tensor(ctx)
    samples = []
    for seed in seeds:
        D, OUT, P = ctx.arrays(seed, p_out, p_stay, power_scale)
        T = D.shape[0]
        prev = np.zeros(D.shape[1])
        for t in range(T):
            x = node_features(D[t], prev, OUT[t], ctx.minfrac, ctx.priorities,
                              ctx.critical_mask, P[t])
            a_rule = _rule_alloc(ctx, D[t], P[t], OUT[t])
            target = torch.tensor(a_rule / (D[t] + EPS), dtype=torch.float32).clamp(0, 1)
            samples.append((x, adj, target))
            prev = a_rule  # teacher forcing
    return samples


def eval_policy_C(model, ctx: _FeederCtx, seeds, power_scale, windows=(4, 8, 16)) -> dict:
    """Rollout the policy via the hard score-driven greedy allocator; mean C."""
    adj = _adj_tensor(ctx)
    model.eval()
    per_seed = []
    with torch.no_grad():
        for seed in seeds:
            D, OUT, P = ctx.arrays(seed, p_out=0.05, p_stay=0.85, power_scale=power_scale)
            T, n = D.shape
            A = np.zeros((T, n)); prev = np.zeros(n)
            for t in range(T):
                x = node_features(D[t], prev, OUT[t], ctx.minfrac, ctx.priorities,
                                  ctx.critical_mask, P[t])
                score = model(x, adj).numpy()
                a = greedy_flow_allocation(D[t], P[t], OUT[t], ctx.anc, ctx.caps,
                                           ctx.priorities, ctx.minfrac, ctx.critical_mask,
                                           score=score)
                A[t] = a; prev = a
            res = M.rollout_metrics(A[:, ctx.cidx], D[:, ctx.cidx], ctx.cm, ctx.cw, P,
                                    OUT[:, ctx.cidx],
                                    windows=tuple(L for L in windows if L <= T),
                                    oracle=ctx.oracle)
            per_seed.append(res["C"])
    return {"C": float(np.mean(per_seed)), "sdC": float(np.std(per_seed))}


def train_policy(kind: str, train_ctxs, scales, val_seeds, epochs=40, lr=2e-3,
                 hidden=64, seed=0) -> tuple[Any, float]:
    """Train by imitation; select the checkpoint by best mean validation hard-C."""
    torch.manual_seed(seed)
    model = build_policy(kind, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    data = []
    for ctx, sc in zip(train_ctxs, scales):
        data += build_dataset(ctx, val_seeds, sc)  # reuse seeds for train pool
    best_state, best_val = None, -1.0
    for ep in range(epochs):
        model.train()
        np.random.default_rng(ep).shuffle(data)
        tot = 0.0
        for x, adj, target in data:
            opt.zero_grad()
            loss = lossf(model(x, adj), target)
            loss.backward(); opt.step(); tot += float(loss)
        if ep % 5 == 4 or ep == epochs - 1:
            val = np.mean([eval_policy_C(model, c, val_seeds, s)["C"]
                           for c, s in zip(train_ctxs, scales)])
            if val > best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def run_h1(split_yaml: str, horizon=64, calib_seeds=tuple(range(100, 112)),
           train_seeds=tuple(range(8)), eval_seeds=tuple(range(8, 20)),
           kinds=("deepsets", "graphsage"), band=(0.65, 0.80)) -> dict:
    import yaml
    split = yaml.safe_load(open(split_yaml, encoding="utf-8"))
    train_entries = [dict(e, role="train") for e in split["G_train"]]
    ood_entries = [dict(e, role="ood") for e in split["G_ood"]]
    train_ctxs = [_FeederCtx(e, horizon) for e in train_entries]
    ood_ctxs = [_FeederCtx(e, horizon) for e in ood_entries]
    scales = [calibrate_budget_oracle(c, calib_seeds, band=band)["power_scale"] for c in train_ctxs]
    ood_scales = [calibrate_budget_oracle(c, calib_seeds, band=band)["power_scale"] for c in ood_ctxs]

    out: dict[str, Any] = {"methods": {}}
    for kind in kinds:
        model, val = train_policy(kind, train_ctxs, scales, train_seeds)
        per = {"train": [], "ood": []}
        for c, s in zip(train_ctxs, scales):
            per["train"].append({"feeder": c.scenario.name, **eval_policy_C(model, c, eval_seeds, s)})
        for c, s in zip(ood_ctxs, ood_scales):
            per["ood"].append({"feeder": c.scenario.name, **eval_policy_C(model, c, eval_seeds, s)})
        mt = float(np.mean([r["C"] for r in per["train"]]))
        mo = float(np.mean([r["C"] for r in per["ood"]]))
        out["methods"][kind] = {"val_C": val, "mu_train_C": mt, "mu_ood_C": mo,
                                "transfer_gap_C": mt - mo, "per_feeder": per}
    g, d = out["methods"].get("graphsage"), out["methods"].get("deepsets")
    if g and d:
        out["H1_graph_minus_nograph_OOD_C"] = g["mu_ood_C"] - d["mu_ood_C"]
    return out
