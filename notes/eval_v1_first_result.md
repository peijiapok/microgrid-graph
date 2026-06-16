# First cross-topology evaluation result (baseline) — 2026-06-16

> **Superseded by ADR-0003 for methodology + final numbers.** This note records the *exploratory* findings (loose-scarcity ≈0 gap, policy-controllability, uniform-scarcity confound) that motivated the methodology fix. The trustworthy result uses oracle-calibrated, T=64, disjoint paired seeds (`notes/adr/0003-continuity-eval-methodology.md`): mean C train 0.97 / OOD 0.99, transfer_gap[C] ≈ 0 for the flow-aware greedy reference. Additional finding from that work: **branch-flow caps (topology), not budget, dominate critical-service difficulty in real LV/MVLV feeders** — strong graph-necessity evidence.

Pipeline: `src/sg_resilience/eval_harness.py`, runner `scripts/run_eval_v1.py`.
Policy: priority allocator → projection onto `Δ_grid` (reference). Metric: capacity-normalized continuity `C` (`metrics_v1`) with the flow-constrained oracle. 3 seeds, Markov outages (p_out=0.05, p_stay=0.85).

## 1. The pipeline works end-to-end ✅
load feeder → Markov outages → priority baseline → `Δ_grid` projection → continuity `C` + transfer gap, across the full frozen split. First numbers obtained. Result bundles: `results/eval_v1_baseline.json` (loose), `results/eval_v1_baseline_tight.json` (tight).

## 2. Loose scarcity (default budgets): transfer gap ≈ 0 — and it's outage-driven
| | mean C | adequacy |
|---|---|---|
| train | 0.810 | 1.000 |
| OOD | 0.873 | 1.000 |
| transfer_gap[C] | **−0.063** (OOD slightly better) | |

A rule policy has no training, so ~0 transfer gap is the **expected, correct baseline** — it sets the bar a learned graph policy must beat. BUT **adequacy = 1.000 everywhere**: the budget is loose enough that critical min-service is always met, so `C` is invariant to the budget and driven by *exogenous outages*, not policy.

## 3. Policy-controllability finding (scarcity sweep, rural2)
| power_scale | C | adequacy |
|---|---|---|
| 1.00 / 0.60 / 0.40 | 0.885 | 1.000 |
| 0.25 | 0.578 | 0.777 |

Continuity `C` is **constant** until the budget is cut to ~0.25×, where adequacy finally drops below 1 and the policy is *forced* to shed critical loads. **Continuity is only a policy-controllable metric once in-family critical adequacy < 1** (ADR-0001's policy-controllability requirement, now empirical). At default supply ratios it is not.

## 4. Uniform scarcity CONFOUNDS topology with feeder scale (key methodological finding)
At a uniform power_scale = 0.25 across the split:
| feeder | role | C | adequacy |
|---|---|---|---|
| case33bw | train | 0.157 | 0.474 |
| rural2 | train | 0.578 | 0.777 |
| rural3 | train | 0.834 | 1.000 |
| urban6 | ood | 0.466 | 0.615 |
| mvlv-5303 | ood | 0.001 | 0.133 |
| mvlv-6305 | ood | 0.001 | 0.008 |

transfer_gap[C] = **+0.367** — but this is **not a topology-transfer signal**. A uniform budget cut lands each feeder in a *different* scarcity regime (rural3 untouched at 1.0; the MVLV feeders starved to ~0) because feeders differ in critical-demand-to-total ratio. The gap measures demand/budget heterogeneity, not topology generalization. This is exactly the workload confound `docs/05 §9` warns about.

## 5. Required fix before any claim-bearing continuity-transfer result
**Per-feeder scarcity calibration.** Tune each feeder's budget so in-family critical adequacy lands in a fixed target band (proposal: 0.65–0.80) *before* claim-bearing runs; report the per-feeder budget ratio `rho` in the workload audit. Do NOT use a uniform power scale across feeders. Without this, the transfer table is confounded by feeder scale, not topology. (Protocol updated: `docs/05 §9`.)

## 6. What this de-risks
Caught *before* any learned-policy runs: (a) continuity needs tight scarcity to be policy-meaningful; (b) scarcity must be per-feeder-calibrated or topology transfer is confounded. Both would have silently produced a meaningless transfer table. The fix is a calibration step, now specified in the protocol.

## 7. Next
- Implement per-feeder budget calibration to the adequacy band; re-run the baseline as the calibrated reference.
- Then DeepSets / GCN / GAT / GIN baselines (C5) and the GraphSAGE/C1 learned policy on the same calibrated split → the real H1/H2 result.
