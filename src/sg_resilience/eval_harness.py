"""End-to-end evaluation harness — first cross-topology transfer result.

Wires the v1 components into one pipeline for a non-learned baseline (priority
allocator), producing per-feeder and transfer-gap numbers on the frozen split:

    load feeder -> Markov outages -> priority baseline -> project onto Delta_grid
    -> continuity metric C (flow oracle) -> per-feeder aggregate -> transfer gap.

This establishes the pipeline and a baseline transfer profile WITHOUT training or
Fanchen's differentiable projection (it uses the reference projection). Learned
policies (GraphSAGE/C1) and the differentiable projection plug into the same
`policy` / projection slots later.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import yaml

from . import metrics_v1 as M
from .baseline_rule import priority_first_allocation
from .benchmark_loader import _load_pandapower_benchmark, _load_simbench_benchmark
from .flow_projection import make_flow_oracle, project_onto_delta_grid
from .outages import apply_markov_outages
from .scenario_schema import Scenario
from .topology import build_radial_tree


def _feeder_config(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a loader raw-config from a split-file feeder entry."""
    src = entry["source"]
    common = {
        "horizon": entry.get("horizon", 8),
        "default_min_service_fraction": 0.2,
        "critical_min_service_fraction": 0.55,
        "critical_top_k": entry.get("critical_top_k", 0) or None,
    }
    if src == "pandapower":
        bench = {"source": "pandapower", "name": entry["code"],
                 "horizon": entry.get("horizon", 4),
                 "demand_profile": entry.get("demand_profile", [1.0, 0.95, 1.05, 1.0]),
                 "supply_ratio": entry.get("supply_ratio", 0.75)}
    else:
        bench = {"source": "simbench", "code": entry["code"],
                 "horizon": entry.get("horizon", 8), "start_index": 0, "stride": 4,
                 "dispatchable_supply_ratio": entry.get("dispatchable_supply_ratio", 0.55)}
    for k, v in common.items():
        if v is not None and k not in bench:
            bench.setdefault(k, v)
    return {"benchmark": bench, "metadata": {"role": entry.get("role", "")}}


def _net_for(entry: dict[str, Any]):
    import pandapower.networks as ppn
    import simbench
    if entry["source"] == "pandapower":
        return getattr(ppn, entry["code"])()
    return simbench.get_simbench_net(entry["code"])


def _load_scenario(entry: dict[str, Any]) -> Scenario:
    raw = _feeder_config(entry)
    if entry["source"] == "pandapower":
        return _load_pandapower_benchmark(raw)
    return _load_simbench_benchmark(raw)


def evaluate_feeder(
    entry: dict[str, Any],
    seeds: Sequence[int] = (0, 1, 2),
    p_out: float = 0.05,
    p_stay: float = 0.85,
    windows: tuple[int, ...] = (2, 4),
    power_scale: float = 1.0,
) -> dict[str, Any]:
    """Run the priority baseline through the full pipeline on one feeder.

    power_scale < 1 tightens scarcity (scales available power per step) so the
    policy — not exogenous outages — becomes the binding factor for continuity.
    """
    scenario = _load_scenario(entry)
    net = _net_for(entry)
    line_cap = entry.get("line_capacity_mw_override")
    tree = build_radial_tree(net, line_capacity_mw=line_cap)

    node_order = [n.node_id for n in scenario.nodes]
    priorities = {n.node_id: n.priority for n in scenario.nodes}
    minfrac = {n.node_id: n.min_service_fraction for n in scenario.nodes}
    critical_ids = [n.node_id for n in scenario.nodes if n.is_critical]
    crit_set = set(critical_ids)
    m_arr = np.array([minfrac[i] for i in critical_ids])
    w_arr = np.array([priorities[i] for i in critical_ids])
    oracle = make_flow_oracle(tree, critical_ids)

    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        sc = apply_markov_outages(scenario, p_out=p_out, p_stay=p_stay, seed=seed)
        T = len(sc.states)
        A = np.zeros((T, len(critical_ids)))
        D = np.zeros((T, len(critical_ids)))
        OUT = np.zeros((T, len(critical_ids)), dtype=bool)
        power = np.zeros(T)
        for t, st in enumerate(sc.states):
            outaged = set(st.outages)
            budget = float(st.available_power) * power_scale
            # baseline: zero demand for outaged nodes so it doesn't waste budget
            dem = {nid: (0.0 if nid in outaged else float(st.demands[nid])) for nid in node_order}
            raw_alloc = priority_first_allocation(
                budget, dem, priorities, minfrac, crit_set
            )
            z = np.array([raw_alloc[nid] for nid in node_order])
            d_full = np.array([float(st.demands[nid]) for nid in node_order])
            out_full = np.array([nid in outaged for nid in node_order], dtype=bool)
            a = project_onto_delta_grid(z, d_full, budget, out_full, tree, node_order)
            idx = {nid: j for j, nid in enumerate(node_order)}
            for ci, cid in enumerate(critical_ids):
                A[t, ci] = a[idx[cid]]
                D[t, ci] = float(st.demands[cid])
                OUT[t, ci] = cid in outaged
            power[t] = budget
        res = M.rollout_metrics(A, D, m_arr, w_arr, power, OUT,
                                windows=tuple(L for L in windows if L <= T),
                                oracle=oracle)
        per_seed.append({k: float(v) for k, v in res.items() if not isinstance(v, dict)})

    def agg(key: str) -> float:
        return float(np.mean([s[key] for s in per_seed]))

    return {
        "feeder": entry["id"],
        "role": entry.get("role"),
        "n_loads": len(node_order),
        "n_critical": len(critical_ids),
        "seeds": list(seeds),
        "C": agg("C"),
        "critical_load_adequacy": agg("critical_load_adequacy"),
        "critical_coverage": agg("critical_coverage"),
        "weighted_starvation": agg("weighted_starvation"),
        "per_seed_C": [s["C"] for s in per_seed],
    }


def run_split(split_yaml: str, **kwargs: Any) -> dict[str, Any]:
    """Evaluate the baseline across the frozen split; compute the transfer gap on C."""
    split = cast(dict[str, Any], yaml.safe_load(open(split_yaml, encoding="utf-8")))
    results = {"train": [], "ood": []}
    for role, key in (("train", "G_train"), ("ood", "G_ood")):
        for entry in split.get(key, []):
            e = dict(entry)
            e["role"] = role
            results[role].append(evaluate_feeder(e, **kwargs))

    def mean_C(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([r["C"] for r in rows])) if rows else 0.0

    mu_train, mu_ood = mean_C(results["train"]), mean_C(results["ood"])
    return {
        "metric": "continuity_C",
        "mu_train_C": mu_train,
        "mu_ood_C": mu_ood,
        "transfer_gap_C": M.transfer_gap(mu_train, mu_ood, higher_is_better=True),
        "per_feeder": results,
        "policy": "priority_baseline+Delta_grid_projection",
    }
