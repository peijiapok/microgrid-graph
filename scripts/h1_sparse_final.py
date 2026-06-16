"""Definitive H1 with SPARSE tree topology + LP-optimal training.
Triple at the congested point (real vs wrong-graph vs no-graph) + realistic-cap check.
Run: PYTHONPATH=src:. python scripts/h1_sparse_final.py
"""
from __future__ import annotations
import json, warnings, numpy as np, yaml
warnings.simplefilter("ignore")
from pathlib import Path
from sg_resilience.eval_harness import _FeederCtx, calibrate_budget_oracle
from sg_resilience.h1_train import eval_policy_C
from scripts.h1_lp_target_test import train

CS = tuple(range(100, 106)); TSEEDS, ESEEDS = tuple(range(5)), tuple(range(5, 13))
MODEL_SEEDS = (0, 1)
split = yaml.safe_load(open("configs/topology_split_v1.yaml"))


def ctxs(cap):
    tr = [_FeederCtx(dict(e), 48, cap_scale=cap) for e in split["G_train"]]
    oo = [_FeederCtx(dict(e), 48, cap_scale=cap) for e in split["G_ood"]]
    return tr, oo


def run(cap, kind, rewire=None):
    vals = []
    for ms in MODEL_SEEDS:
        tr, oo = ctxs(cap)
        if rewire is not None:
            for c in tr + oo:
                c.rewire_seed = rewire
        trs = [calibrate_budget_oracle(c, CS)["power_scale"] for c in tr]
        oos = [calibrate_budget_oracle(c, CS)["power_scale"] for c in oo]
        m = train(kind, tr, trs, TSEEDS, seed=ms)
        vals.append(float(np.mean([eval_policy_C(m, c, ESEEDS, s)["C"] for c, s in zip(oo, oos)])))
    return float(np.mean(vals)), float(np.std(vals))


out = {}
print("=== Definitive H1, SPARSE tree topology, LP-optimal training ===")
for cap in (1.0, 0.03):
    ng = run(cap, "deepsets")
    rg = run(cap, "graphsage")
    row = {"nograph": ng, "real_graph": rg}
    line = f"cap_scale={cap:5.3f}: nograph={ng[0]:.3f}±{ng[1]:.3f}  real_graph={rg[0]:.3f}±{rg[1]:.3f}  real-nograph={rg[0]-ng[0]:+.3f}"
    if cap == 0.03:
        wg = run(cap, "graphsage", rewire=123)
        row["wrong_graph"] = wg
        line += f"  wrong_graph={wg[0]:.3f}±{wg[1]:.3f}  real-wrong={rg[0]-wg[0]:+.3f}"
    out[f"cap_{cap}"] = row
    print(line)
Path("results").mkdir(exist_ok=True)
Path("results/h1_sparse_final.json").write_text(json.dumps(out, indent=2))
print("\nH1 decisive: real-nograph>0 (graph helps) AND real-wrong>0 (REAL topology, not just message passing), under congestion.")
