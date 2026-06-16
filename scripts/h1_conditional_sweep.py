"""Claim-bearing confirmation: graph-necessity vs congestion under LP-optimal
training, multi-seed. Produces the paper's central figure data — graph-nograph
OOD C as flow caps tighten, with error bars over model seeds.

Run: PYTHONPATH=src:. python scripts/h1_conditional_sweep.py
"""
from __future__ import annotations
import json, warnings, numpy as np, yaml
warnings.simplefilter("ignore")
from pathlib import Path

from sg_resilience.eval_harness import _FeederCtx, calibrate_budget_oracle
from sg_resilience.h1_train import eval_policy_C
from scripts.h1_lp_target_test import train  # LP-target trainer

CAP_SCALES = (1.0, 0.06, 0.03)
MODEL_SEEDS = (0, 1, 2)


def main():
    split = yaml.safe_load(open("configs/topology_split_v1.yaml"))
    cs = tuple(range(100, 106))
    tseeds, eseeds = tuple(range(5)), tuple(range(5, 13))
    rows = []
    for capx in CAP_SCALES:
        tr = [_FeederCtx(dict(e), 48, cap_scale=capx) for e in split["G_train"]]
        oo = [_FeederCtx(dict(e), 48, cap_scale=capx) for e in split["G_ood"]]
        trs = [calibrate_budget_oracle(c, cs)["power_scale"] for c in tr]
        oos = [calibrate_budget_oracle(c, cs)["power_scale"] for c in oo]
        diffs = []
        for ms in MODEL_SEEDS:
            res = {}
            for kind in ("deepsets", "graphsage"):
                m = train(kind, tr, trs, tseeds, seed=ms)
                res[kind] = float(np.mean([eval_policy_C(m, c, eseeds, s)["C"] for c, s in zip(oo, oos)]))
            diffs.append(res["graphsage"] - res["deepsets"])
        row = {"cap_scale": capx, "graph_minus_nograph_mean": float(np.mean(diffs)),
               "std": float(np.std(diffs)), "per_seed": diffs}
        rows.append(row)
        print(f"cap_scale={capx:5.3f}  graph-nograph OOD C = {row['graph_minus_nograph_mean']:+.3f} "
              f"± {row['std']:.3f}  (seeds {[round(d,3) for d in diffs]})")
    Path("results").mkdir(exist_ok=True)
    Path("results/h1_conditional_sweep.json").write_text(json.dumps(rows, indent=2))
    print("\nCentral claim: graph advantage should grow as cap_scale shrinks (congestion).")


if __name__ == "__main__":
    main()
