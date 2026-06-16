# ADR-0003 — Continuity-transfer evaluation methodology

> Status: **ACCEPTED** 2026-06-16. Decided by Jia after the first baseline runs exposed that naive (short-horizon, uniform-budget, rule-calibrated) evaluation produces confounded, high-variance transfer numbers. Cross-checked with GPT-5.5. Updates `docs/05` §2.5/§9.1.

## Context (what went wrong with the naive approach)
- **Uniform budget ratio** across feeders lands each feeder at wildly different difficulty (rural3 adequacy 1.0, MVLV ~0.01) → transfer gap measures scale heterogeneity, not topology.
- **Short horizon (T=4–8)** can't measure temporal continuity: outage persistence is ~1/(1−p_stay)=10 steps, and windows up to 8 are degenerate on an 8-step rollout.
- **Rule-baseline budget calibration** confounds feeder difficulty with the heuristic's weakness; 1-seed calibration is far too noisy (cal adequacy 0.87 vs eval 0.23) → "max_iter", and the transfer gap flipped sign artifactually.
- **Reference cvxpy QP projection does not scale** to long horizons × many seeds × large feeders.

## Decision
1. **Horizon T ≥ 64** (default 64; sensitivity at 96/128). ~6–13 outage correlation lengths; windows {4,8,16}.
2. **Per-feeder budget calibrated to the OFFLINE ORACLE's critical serviceable fraction**, target band **[0.65, 0.80]** (pre-declared). Oracle = "physical opportunity," policy-independent, fast (pure numpy, no QP). NOT rule-baseline adequacy. Fixed-budget-ratio is a robustness appendix only.
3. **Seeds:** ≥16 calibration / ≥24 eval for development; **claim-bearing: ≥32 calib / ≥50 eval**. Seeds **paired** across policies within a feeder; **calibration seeds disjoint from eval seeds**.
4. **Feasible-by-construction flow-aware GREEDY allocator** for non-learned baseline eval (`flow_projection.greedy_flow_allocation`) — no per-step QP, so eval scales. The cvxpy projection stays the correctness oracle and the projection layer for *learned* policies (whose differentiable version is C3, Fanchen).
5. **Legitimacy:** per-feeder calibration is defensible because it is oracle-based, pre-declared, and calibration-seed-disjoint. Framing: *"cross-topology quality under matched oracle-normalized scarcity."* Report per-feeder budget ratio, oracle fraction, critical-load fraction, flow-tightness as covariates.

## Validated result (baseline, T=64, oracle-calibrated, 24 eval seeds)
All feeders calibrated **in band** (oracle_frac 0.697–0.773); variance small (sd C 0.004–0.08).
mean C: **train 0.966, OOD 0.992, transfer_gap[C] = −0.026 (≈ 0).**
Interpretation: the flow-aware greedy rule (with global tree knowledge) is near the oracle ceiling and generalizes trivially across topologies — the correct **reference ceiling**. The ML question (H1/H2) is whether *learned* policies match it and need the graph to do so on unseen feeders.

## Consequences
- `docs/05 §9.1` scarcity gate updated: oracle-based per-feeder calibration, T≥64, disjoint paired seeds; uniform-ratio forbidden for claim-bearing runs.
- Code: `eval_harness.py` rewritten around `_FeederCtx` + `calibrate_budget_oracle` + greedy allocator. `metrics_v1` / `flow_projection` reference projection unchanged (latter gains `greedy_flow_allocation`, `node_ancestor_edges`).
- A strong flow-aware rule baseline means the learned-policy contribution must be argued on *graph-necessity for generalization*, not on beating the rule in-distribution (already the v1 framing).
