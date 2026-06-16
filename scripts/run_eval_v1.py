"""First cross-topology transfer result (baseline) on the frozen v1 split.

Runs the priority baseline through the full Delta_grid pipeline and reports
per-feeder continuity C plus the train->OOD transfer gap. Writes a result
bundle to results/eval_v1_baseline.json.

Run: PYTHONPATH=src python scripts/run_eval_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sg_resilience.eval_harness import run_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    split = str(ROOT / "configs" / "topology_split_v1.yaml")
    print("[eval] running priority baseline + Delta_grid projection on frozen split...")
    out = run_split(split, seeds=(0, 1, 2))

    print("\n=== Per-feeder continuity C (priority baseline) ===")
    hdr = f"{'feeder':26} {'role':6} {'loads':>5} {'crit':>4} {'C':>6} {'adeq':>6} {'cover':>6} {'starv':>6}"
    print(hdr)
    print("-" * len(hdr))
    for role in ("train", "ood"):
        for r in out["per_feeder"][role]:
            print(f"{r['feeder']:26} {role:6} {r['n_loads']:5d} {r['n_critical']:4d} "
                  f"{r['C']:6.3f} {r['critical_load_adequacy']:6.3f} "
                  f"{r['critical_coverage']:6.3f} {r['weighted_starvation']:6.3f}")

    print(f"\nmean C  train={out['mu_train_C']:.3f}  OOD={out['mu_ood_C']:.3f}  "
          f"transfer_gap[C]={out['transfer_gap_C']:.3f}  (positive => OOD worse)")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    dest = results_dir / "eval_v1_baseline.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[eval] wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
