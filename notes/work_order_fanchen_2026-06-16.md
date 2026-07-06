# Work order — Fanchen (C1, C3). Direction set by lead (Jia).

> **SUPERSEDED by `notes/work_order_fanchen_2026-07-06.md`** — this version wrongly framed Fanchen around a GNN control operator (C1). His actual expertise (ICDM 2025, random graph models / structural graph mining) puts him on the STRUCTURE side (d_struct, feeder atlas, RGM-based feeder generation), not the control operator. Kept for history only.

> Supersedes the discussion-framed `notes/message_to_fanchen_2026-06-16.md`. This is directive: the design is decided; these are your two implementation tasks with acceptance gates. Specs cross-checked with GPT-5.5. Date 2026-06-16.

## P0 — Investigate the topology-non-causal diagnostic FIRST (added 2026-06-17)
Before building C1/C3, look at `notes/h1_negative_finding.md`. A preliminary
diagnostic from Jia's side (quick GraphSAGE/DeepSets, NOT your C1) found: under
congestion + LP-optimal training, real-graph ≈ wrong-graph (degree-preserving
rewire) ≈ +0.002, while message-passing beats no-graph by +0.09 — i.e. the
*actual feeder topology may be non-causal* for the learned decision because
feasibility is delegated to a shared topology-aware allocator. This is YOUR
domain to adjudicate: is there a graph operator/representation/decision where the
real topology is genuinely load-bearing, or does this kill C1/C2/C6 and we pivot
(reconfiguration/AC, or a negative-result paper)? Your call — but engage with
this before investing in C1/C3, because if topology is non-causal the operator
work is moot. Confound checklist (harsher wrong-graph controls, headroom-to-oracle,
feature-only predictor, multi-seed CIs) is in the finding note.

## 0. Direction (decided — not open for re-scoping)
- **Primary objective = temporal continuity** of critical-load service (ADR-0001). Adequacy is a guard, not the target.
- **Feasible set is flow-constrained** `Δ_grid` (radial subtree-sum capacity limits). This is **Path A** (full design in v1; quality over the 2026 workshop deadline).
- Rationale and the continuity↔topology coupling are in `notes/adr/0001-continuity-primary.md`. Read it once for context; the tasks below are what to build.

## 1. What is already done (do NOT rebuild — build on these)
| Asset | File | Use to you |
|---|---|---|
| Frozen feeder split | `configs/topology_split_v1.yaml`, `notes/adr/0002-v1-feeder-split.md` | the feeders you train/eval on |
| Rooted radial tree + `F_e` + subtree membership | `src/sg_resilience/topology.py` | edge orientation, capacities, `subtree_loads` for C1 features and C3 constraints |
| **Reference** projection onto `Δ_grid` (cvxpy/OSQP) | `src/sg_resilience/flow_projection.py::project_onto_delta_grid` | the **correctness oracle your C3 must match** |
| Flow-aware oracle, metrics, continuity loss, Markov outages | `metrics_v1.py`, `continuity_loss.py`, `outages.py` | evaluation + training signal; do not reimplement |
| Loader (now switch-state correct) | `benchmark_loader.py` | feeder → `Scenario` |

Test suite is green (20 tests). Your additions must keep it green and add their own.

## 2. TASK C3 — differentiable feasibility projection

### C3-main (ship this first)
Implement a **differentiable** projection onto `Δ_grid` using a differentiable convex-optimization layer (`cvxpylayers`, OptNet-style; KKT/implicit-function backward). Same feasible set as the reference: `0≤a≤d`, `a=0` on outage, `Σa≤P`, and `f_e(a)=Σ_{j∈subtree(e)} a_j ≤ F_e` ∀ tree edge `e`.
- Build the subtree constraint matrix `S` from `topology.py` (`_subtree_matrix` in `flow_projection.py` already does this — reuse).
- Refs: OptNet (arXiv:1703.00443), differentiable convex layers (arXiv:1910.12430).

**Acceptance gates (C3-main):**
1. **Correctness:** forward output matches `flow_projection.project_onto_delta_grid` within `1e-6` (normalized by feeder nominal demand) on ≥3 feeders (case33bw + 2 SimBench) across random `z`, demands, outages.
2. **Feasibility:** `topology.flow_violations` returns empty and budget/box/outage hold at `epsilon_feas=1e-6` (docs/05 §2.6) on all test rollouts.
3. **Gradients:** finite-difference check of `∂a/∂z` on small cases (≤6 nodes) within `1e-4`; gradients finite on all feeders.
4. **Integrates** as the projection in the training loop; no feasibility regression vs the iterative water-filler.

### C3-bonus (after C3-main passes — potential standalone contribution)
A **custom near-linear tree-DP projection** exploiting the laminar structure (subtree-sum rows are nested along root→leaf paths). Sketch (GPT-5.5, adopt):
- Root at source; `u_i = d_i·1[online]`. Per subtree `v`, a monotone demand curve under uniform price τ: leaf `a_i(τ)=clip(z_i−τ,0,u_i)`; internal `R_v(τ)=a_v(τ)+Σ_{c} Q_c(τ)`, `Q_v(τ)=min(R_v(τ), C_v)` with `C_v` the edge/subtree capacity (root capacity = budget `P`).
- Top-down price recovery: where `R_v(τ)>C_v`, find local multiplier `η_v≥0` with `R_v(τ+η_v)=C_v`; pass `τ+η_v` to children. Store curves as PWL breakpoints, merge small-to-large → `O(n log n)`.
- Differentiable: solution is affine on a fixed active set → implicit diff through the active KKT system (or unroll the dual price search); subgradients at breakpoints.
- **Gate:** matches C3-main within `1e-6`; demonstrate near-linear scaling vs the QP layer on the MVLV-urban feeders.

## 3. TASK C1 — flow-aware message-passing operator
A **directed Graph-Network/MPNN block** (Battaglia arXiv:1806.01261, Gilmer arXiv:1704.01212), drop-in replacement for `GraphSAGEBlock` in `controller_model.py`. Two directed arcs per physical line (downstream `p→c`, upstream `c→p`). Edge features `e_{uv}=[y_{uv}, log F_{uv}, r_{uv}]` (admittance, log thermal capacity, direction∈{up,down}). Per layer:
```
g_{uv} = σ( φ_g^{(r)}(e_{uv}, h_u, h_v) )                  # capacity/direction gate
m_{uv} = g_{uv} ⊙ φ_m^{(r)}(h_u, h_v, e_{uv})             # gated message
m̄_v   = Σ_{u:(u,v)∈E→} m_{uv}                            # sum aggregation
h_v'   = GRU( h_v, [m̄_v, x_v] )                           # update
```
Separate parameters (or a direction embedding) for up/down. Asymmetric along the feeder, capacity-aware via `g`, permutation-equivariant via sum aggregation, size-generalizing via local shared params.

**Acceptance gates (C1):**
1. **Edge-feature wiring smoke** (docs/05 §2 C1 gate): on a fixed smoke batch, zeroing edge features changes pre-projection scores with `‖y_true−y_zero‖/‖y_true‖ > 1e-2`.
2. Drop-in: runs on case33bw, passes the existing smoke metrics within tolerance, composes with the C3 projection.
3. Shares the feature extractor, projection, loss, and training budget with all baselines (docs/05 §3) — only the operator differs.
4. (Claim-bearing, later) edge-feature ablation in main results shows zeroing `F_e`/admittance degrades OOD continuity.

## 4. Process & ownership
- Branch per contribution: `c3-diff-projection`, `c1-flow-aware-mp`. PR review before merge to `main`. Repo: github.com/peijiapok/microgrid-graph.
- You also still own C4 (CVaR — now well-motivated: continuity is heavy-tailed/path-dependent) and C6 (d_struct, v2) per `docs/04`.
- Open the C1 design PR and the C3-main PR first; C3-bonus and C4/C6 follow.
- Question for me only if a gate is wrong or infeasible — otherwise build to spec.

## 5. Note on case33bw (Jia handling, FYI)
case33bw's `max_i_ka` is a pandapower placeholder → `F_e≈2.2M MW`, so flow constraints won't bind there. I'll assign physical conductor ratings (or we lean on the SimBench feeders for the topology-binding story). Does not affect your C1/C3 work; just don't be surprised if case33bw flow caps look slack.
