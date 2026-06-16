"""End-to-end evaluation harness — cross-topology continuity transfer.

Methodology (ADR-0003, GPT-5.5-backed):
  - long horizon (T>=64) so temporal continuity is measurable past outage
    persistence (~10 steps);
  - per-feeder budget calibrated to the OFFLINE ORACLE's critical serviceable
    fraction (policy-independent "physical opportunity"), NOT a rule baseline;
  - calibration seeds disjoint from eval seeds; seeds paired across policies;
  - feasible-by-construction flow-aware GREEDY allocator (no per-step QP), so
    eval scales to long horizons and many seeds. The cvxpy projection
    (flow_projection.project_onto_delta_grid) remains the correctness oracle and
    the projection layer for learned policies.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import yaml

from . import metrics_v1 as M
from .benchmark_loader import _load_pandapower_benchmark, _load_simbench_benchmark
from .flow_projection import greedy_flow_allocation, make_flow_oracle, node_ancestor_edges
from .outages import apply_markov_outages
from .scenario_schema import Scenario
from .topology import build_radial_tree


def _daily_profile(T: int) -> list[float]:
    return [round(0.8 + 0.2 * math.cos(2 * math.pi * t / max(T, 1)), 4) for t in range(T)]


def _feeder_config(entry: dict[str, Any], horizon: int) -> dict[str, Any]:
    common = {"default_min_service_fraction": 0.2, "critical_min_service_fraction": 0.55}
    if entry["source"] == "pandapower":
        bench = {"source": "pandapower", "name": entry["code"], "horizon": horizon,
                 "demand_profile": _daily_profile(horizon),
                 "supply_ratio": entry.get("supply_ratio", 0.75)}
    else:
        bench = {"source": "simbench", "code": entry["code"], "horizon": horizon,
                 "start_index": 0, "stride": 1,
                 "dispatchable_supply_ratio": entry.get("dispatchable_supply_ratio", 0.55)}
    bench.update(common)
    return {"benchmark": bench, "metadata": {"role": entry.get("role", "")}}


def _net_for(entry: dict[str, Any]):
    import pandapower.networks as ppn
    import simbench
    return (getattr(ppn, entry["code"])() if entry["source"] == "pandapower"
            else simbench.get_simbench_net(entry["code"]))


def _load_scenario(entry: dict[str, Any], horizon: int) -> Scenario:
    raw = _feeder_config(entry, horizon)
    return (_load_pandapower_benchmark(raw) if entry["source"] == "pandapower"
            else _load_simbench_benchmark(raw))


class _FeederCtx:
    """Loaded-once per-feeder context reused across seeds/calibration.

    cap_scale scales every branch-flow capacity F_e (1.0 = real ratings); used to
    study where topology becomes load-bearing under congestion (see
    notes/h1_negative_finding.md)."""
    def __init__(self, entry: dict[str, Any], horizon: int, cap_scale: float = 1.0):
        self.scenario = _load_scenario(entry, horizon)
        net = _net_for(entry)
        self.tree = build_radial_tree(net, line_capacity_mw=entry.get("line_capacity_mw_override"))
        if cap_scale != 1.0:
            for e in self.tree.edge_capacity_mw:
                self.tree.edge_capacity_mw[e] *= cap_scale
        self.node_order = [n.node_id for n in self.scenario.nodes]
        self.priorities = np.array([n.priority for n in self.scenario.nodes])
        self.minfrac = np.array([n.min_service_fraction for n in self.scenario.nodes])
        self.critical_mask = np.array([n.is_critical for n in self.scenario.nodes])
        self.critical_ids = [n.node_id for n in self.scenario.nodes if n.is_critical]
        self.cm = np.array([self.minfrac[j] for j, n in enumerate(self.scenario.nodes) if n.is_critical])
        self.cw = np.array([self.priorities[j] for j, n in enumerate(self.scenario.nodes) if n.is_critical])
        self.anc, self.caps = node_ancestor_edges(self.tree, self.node_order)
        self.oracle = make_flow_oracle(self.tree, self.critical_ids)
        self.idx = {nid: j for j, nid in enumerate(self.node_order)}
        self.cidx = [self.idx[c] for c in self.critical_ids]

    def arrays(self, seed: int, p_out: float, p_stay: float, power_scale: float):
        sc = apply_markov_outages(self.scenario, p_out=p_out, p_stay=p_stay, seed=seed)
        T, n = len(sc.states), len(self.node_order)
        D = np.zeros((T, n)); OUT = np.zeros((T, n), dtype=bool); P = np.zeros(T)
        for t, st in enumerate(sc.states):
            outaged = set(st.outages)
            for nid in self.node_order:
                D[t, self.idx[nid]] = float(st.demands[nid])
            OUT[t] = np.array([nid in outaged for nid in self.node_order], dtype=bool)
            P[t] = float(st.available_power) * power_scale
        return D, OUT, P


def _oracle_fraction(ctx: _FeederCtx, seeds: Sequence[int], p_out, p_stay, ps) -> float:
    """Mean (over seeds) weighted fraction of critical load the OFFLINE ORACLE can
    keep continuously serviceable, excluding fully-outaged critical nodes."""
    fracs = []
    for seed in seeds:
        D, OUT, P = ctx.arrays(seed, p_out, p_stay, ps)
        dc, oc = D[:, ctx.cidx], OUT[:, ctx.cidx]
        cont = ctx.oracle(dc, ctx.cm, ctx.cw, P, oc)
        has_active = ~oc.all(axis=0)  # node has >=1 non-outaged step
        denom = float(np.sum(ctx.cw * has_active))
        if denom > 0:
            fracs.append(float(np.sum(ctx.cw * cont * has_active)) / denom)
    return float(np.mean(fracs)) if fracs else 0.0


def calibrate_budget_oracle(
    ctx: _FeederCtx, calib_seeds: Sequence[int], band=(0.65, 0.80),
    p_out=0.05, p_stay=0.85, lo=0.02, hi=4.0, max_iter=18,
) -> dict[str, Any]:
    """Bisect budget scale so the oracle serviceable fraction lands in `band`
    (monotone in budget). Policy-independent; uses calibration seeds only."""
    low_b, high_b = band
    f = lambda ps: _oracle_fraction(ctx, calib_seeds, p_out, p_stay, ps)
    if f(hi) < low_b:
        return {"power_scale": hi, "oracle_frac": f(hi), "status": "below_band_at_max"}
    if f(lo) > high_b:
        return {"power_scale": lo, "oracle_frac": f(lo), "status": "above_band_at_min"}
    ps = hi
    for _ in range(max_iter):
        ps = 0.5 * (lo + hi); v = f(ps)
        if v > high_b: hi = ps
        elif v < low_b: lo = ps
        else: return {"power_scale": ps, "oracle_frac": v, "status": "in_band"}
    return {"power_scale": ps, "oracle_frac": f(ps), "status": "max_iter"}


def evaluate_feeder(
    ctx: _FeederCtx, seeds: Sequence[int], power_scale: float,
    p_out=0.05, p_stay=0.85, windows=(4, 8, 16),
) -> dict[str, Any]:
    """Flow-aware greedy allocator through the pipeline; mean metrics over seeds."""
    per_seed = []
    for seed in seeds:
        D, OUT, P = ctx.arrays(seed, p_out, p_stay, power_scale)
        T = D.shape[0]
        A = np.zeros((T, len(ctx.node_order)))
        for t in range(T):
            A[t] = greedy_flow_allocation(D[t], P[t], OUT[t], ctx.anc, ctx.caps,
                                          ctx.priorities, ctx.minfrac, ctx.critical_mask)
        res = M.rollout_metrics(A[:, ctx.cidx], D[:, ctx.cidx], ctx.cm, ctx.cw, P,
                                OUT[:, ctx.cidx],
                                windows=tuple(L for L in windows if L <= T),
                                oracle=ctx.oracle)
        per_seed.append({k: float(v) for k, v in res.items() if not isinstance(v, dict)})
    agg = lambda k: float(np.mean([s[k] for s in per_seed]))
    return {"C": agg("C"), "critical_load_adequacy": agg("critical_load_adequacy"),
            "critical_coverage": agg("critical_coverage"),
            "weighted_starvation": agg("weighted_starvation"),
            "per_seed_C": [s["C"] for s in per_seed]}


def run_split_calibrated(
    split_yaml: str, horizon: int = 64,
    calib_seeds: Sequence[int] = tuple(range(100, 116)),
    eval_seeds: Sequence[int] = tuple(range(24)),
    band=(0.65, 0.80), windows=(4, 8, 16), **kw,
) -> dict[str, Any]:
    """Oracle-calibrated, long-horizon cross-topology transfer on the frozen split."""
    split = cast(dict[str, Any], yaml.safe_load(open(split_yaml, encoding="utf-8")))
    results: dict[str, list[dict[str, Any]]] = {"train": [], "ood": []}
    for role, key in (("train", "G_train"), ("ood", "G_ood")):
        for entry in split.get(key, []):
            e = dict(entry); e["role"] = role
            ctx = _FeederCtx(e, horizon)
            cal = calibrate_budget_oracle(ctx, calib_seeds, band=band, **kw)
            r = evaluate_feeder(ctx, eval_seeds, cal["power_scale"], windows=windows, **kw)
            r.update({"feeder": e["id"], "n_loads": len(ctx.node_order),
                      "n_critical": len(ctx.critical_ids),
                      "calibrated_power_scale": cal["power_scale"],
                      "calib_oracle_frac": cal["oracle_frac"], "calib_status": cal["status"]})
            results[role].append(r)
    mC = lambda rows: float(np.mean([x["C"] for x in rows])) if rows else 0.0
    mu_t, mu_o = mC(results["train"]), mC(results["ood"])
    return {"metric": "continuity_C", "horizon": horizon,
            "calibration": {"target_band": list(band), "basis": "offline_oracle_serviceable_fraction"},
            "mu_train_C": mu_t, "mu_ood_C": mu_o,
            "transfer_gap_C": M.transfer_gap(mu_t, mu_o, higher_is_better=True),
            "per_feeder": results, "policy": "flow_aware_greedy"}
