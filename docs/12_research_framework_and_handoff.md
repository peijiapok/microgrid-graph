# 12 — Research Framework & Collaborator Handoff

> Last updated 2026-06-17. Single source of truth for the *current* state of the
> project: the framework as it now stands, what Jia has built, the pivotal
> (preliminary) finding, the open strategic decision, and exactly what Fanchen is
> expected to do. Supersedes scattered status across docs where they conflict;
> defers to `docs/01` (problem) and `docs/10` (formulation) for formal detail.
> Decisions: `notes/adr/0001` (continuity-primary + Δ_grid), `notes/adr/0002`
> (frozen split), `notes/adr/0003` (eval methodology). Key finding:
> `notes/h1_negative_finding.md`. All code on branch `reframe-continuity-flow`.

---

## 1. The research framework (as it now stands)

### 1.1 Problem
A learned **graph policy** allocates scarce power across the nodes of a **radial
distribution feeder**, step by step, under hard feasibility constraints, and must
**generalize zero-shot to feeder topologies never seen in training**. The
motivating regime is crisis operation (catastrophe / scarce power), where supply
falls below demand and critical infrastructure must stay served.

### 1.2 Primary objective & metric (ADR-0001)
**Temporal continuity of critical-load service** is primary (not adequacy).
Metric = **capacity-normalized, priority-weighted windowed continuity `C`**
(`src/sg_resilience/metrics_v1.py`): of the critical load an offline oracle
*could* keep continuously served under the constraints, what fraction did the
policy keep continuously served? Reported per-feeder with anti-gaming
diagnostics (coverage, weighted starvation). Adequacy is a secondary guard;
mean-max-run is a gameable diagnostic only.

### 1.3 Feasible set — flow-constrained `Δ_grid` (ADR-0001)
```
Δ_grid(G, s_t) = { a ≥ 0 : a_i ≤ d_{t,i}; a_i = 0 if i∈O_t; Σ_i a_i ≤ P_t;
                   |f_e(a)| ≤ F_e ∀ tree edge e,  f_e(a)=Σ_{j∈subtree(e)} a_j }
```
Box + global budget + **radial branch-flow capacity** (the term that makes the
feasible set topology-dependent). Linear capacity on a radial tree — **not** AC
power flow ("flow-constrained," never "physics-aware").

### 1.4 Training objective
Continuity surrogate (soft windowed-product, `continuity_loss.py`) as the primary
temporal driver; critical-shortfall guard; weighted served energy; small
anti-chatter switching term. Hard constraints live in the projection, never the loss.

### 1.5 Evaluation methodology (ADR-0003)
- Horizon **T ≥ 64** (outage persistence ~10 steps; windows {4,8,16}).
- Per-feeder budget **calibrated to the offline-oracle serviceable fraction**,
  band [0.65, 0.80] (policy-independent "matched scarcity"); rule-baseline or
  uniform-ratio calibration forbidden for claim-bearing runs.
- Calibration seeds **disjoint** from eval seeds; seeds **paired** across policies.
- Feasibility enforced identically for all policies by a shared allocator.
- Frame: "cross-topology quality under matched oracle-normalized scarcity."

### 1.6 Frozen feeder split (ADR-0002, validated)
| Set | Feeders (loads) |
|---|---|
| G_train | case33bw (32), 1-LV-rural2 (99), 1-LV-rural3 (118) |
| G_ood | 1-LV-urban6 (111), 1-MVLV-urban-5.303 (242), 1-MVLV-urban-6.305 (249) |
All radial (switch/in-service honored); 100% line thermal-capacity coverage;
MVLV feeders give size-gen (>train). case33bw line ratings are a documented
assumption (6 MVA/line; pandapower placeholder otherwise).

### 1.7 The central open question (P0)
A preliminary diagnostic (see §3) suggests that, **because feasibility is
delegated to a shared topology-aware allocator, the learned policy only ranks
nodes and the *actual feeder topology may be non-causal* for that decision.** This
is the hinge of the whole project and the first thing Fanchen must adjudicate.

---

## 2. What Jia has built (resilience / empirics side) — DONE, tested, committed

22 tests pass; everything on branch `reframe-continuity-flow`.

| Component | File | Status |
|---|---|---|
| Reframing of docs to continuity + Δ_grid | `docs/01,02,05,06,10,11` | done |
| Rooted radial tree, edge caps F_e, subtree membership, flows/violations | `src/sg_resilience/topology.py` | done, tested |
| Sparse tree-based load adjacency (real topology) | `eval_harness._FeederCtx.adjacency` | done |
| Reference Δ_grid projection (cvxpy/OSQP) + flow oracle + greedy allocator | `src/sg_resilience/flow_projection.py` | done, tested |
| Capacity-normalized continuity metric `C` + diagnostics | `src/sg_resilience/metrics_v1.py` | done, tested |
| Per-node Markov outage process | `src/sg_resilience/outages.py` | done, tested |
| Differentiable continuity surrogate loss | `src/sg_resilience/continuity_loss.py` | done, tested |
| Oracle-calibrated eval harness (matched scarcity) | `src/sg_resilience/eval_harness.py` | done |
| Loader fix (honor switch/in-service → correct radial graph) | `src/sg_resilience/benchmark_loader.py` | done |
| Feeder validation, decision-room + congestion diagnostics | `scripts/validate_feeders_v1.py`, `scripts/diagnose_decision_room.py` | done |
| **Preliminary** learned-H1 harness (DeepSets/GraphSAGE, imitation) | `src/sg_resilience/{policies,h1_train}.py`, `scripts/h1_*` | done (see §3 caveat) |

**Reference baseline result:** flow-aware greedy rule, oracle-calibrated, mean C
train 0.97 / OOD 0.99, transfer_gap ≈ 0 — the ceiling learned policies must match.

---

## 3. The pivotal finding (PRELIMINARY — for Fanchen to adjudicate)

> Ownership: the graph-learning side (C1/C2/C6) is **Fanchen's**. The H1
> experiments below were run from Jia's side with a **quick GraphSAGE/DeepSets**,
> NOT Fanchen's designed C1. Treat as a **diagnostic to investigate**, not a
> verdict on his contributions. Full detail + confound checklist:
> `notes/h1_negative_finding.md`.

Congested regime (cap_scale 0.03), sparse real topology, LP-optimal training, multi-seed:
| policy | OOD C |
|---|---|
| no-graph (DeepSets) | 0.886 |
| real-graph (GraphSAGE) | 0.977 |
| wrong-graph (degree-preserving rewire) | 0.975 |

- real − no-graph = **+0.091** → message-passing helps under congestion.
- real − wrong-graph = **+0.002** → the **actual feeder topology appears non-causal**;
  a randomly-rewired graph does as well.
- At realistic capacity: real − no-graph ≈ **+0.01** (negligible).

Decision-room diagnostic: priority-greedy = LP-optimal (room ≈ 0) at realistic
ratings; room appears only at ~0.03–0.125× of real line ratings (8–30× over-congestion).

**Interpretation (GPT-5.5-confirmed):** once feasibility is a shared
topology-aware allocator, the policy only ranks candidates; topology enters as a
constraint module downstream, so the learned decision's marginal topology need is
~0. This is *largely architectural*, not an implementation flaw — but Fanchen
owns whether a different operator/representation/decision changes it.

---

## 4. Strategic fork (LEAD decision pending)

GPT-5.5 ranking: **(A) write the rigorous negative result** [recommended] >
(B) reformulate the *decision* to be genuinely topological (network
reconfiguration / restoration / AC feasibility owned by the policy) — a new
paper > (C) drop graphs.

- **A** — "shared topology-aware allocators can make graph policies look useful
  while the actual feeder graph is non-causal for the learned decision." Clean,
  publishable cautionary result; reuses all infrastructure.
- **B** — positive graph paper, but a different problem formulation.
- **C** — engineering conclusion, weak alone.

Jia (lead) to decide A vs B vs C — *after* Fanchen engages with P0 (§5.1).

---

## 5. What Fanchen is expected to do (in detail)

His domain (per `docs/04` / collaborator brief): C1 (flow-aware operator), C2
(topology-generalization protocol), C3 (differentiable projection, joint), C4
(CVaR theory), C6 (d_struct atlas). **But P0 comes first.**

### 5.1 P0 — adjudicate the topology-causality question (do this FIRST)
Before building operators, determine whether the real feeder topology is
load-bearing for the learned decision, or whether the shared-allocator
formulation makes it non-causal (§3).
- Reproduce: `PYTHONPATH=src:. python scripts/h1_sparse_final.py` (real vs
  wrong-graph vs no-graph at cap 1.0 and 0.03).
- Close the confounds (`notes/h1_negative_finding.md` checklist): harsher
  wrong-graph controls (random tree / star / kNN-on-features / shuffled-per-episode),
  report headroom-to-oracle (not raw C), feature-only-predictor vs oracle
  selection, multi-seed CIs for real−wrong.
- **Decide:** is there an operator / graph representation / *decision* where the
  real topology is genuinely causal (real ≫ wrong-graph)? If yes → C1 etc. are
  alive, proceed. If no → recommend pivot (B) or negative result (A) to the lead.
- Deliverable: a short note in `notes/` + a recommendation at the next sync.

### 5.2 C1 — flow-aware message-passing operator (if P0 keeps the graph alive)
Directed MPNN/Graph-Network block (spec: `notes/work_order_fanchen_2026-06-16.md` §3):
two arcs per line (up/down), edge features `[admittance, log F_e, direction]`,
capacity-gated messages `g=σ(φ_g(e,h_u,h_v))`, sum aggregation, GRU update.
Drop-in for the score head in `policies.py`; shares features/allocator/loss.
**Gates:** edge-feature wiring smoke (zeroing edge features changes scores
>1e-2 rel); runs on case33bw; passes smoke; composes with C3. Branch `c1-flow-aware-mp`.

### 5.3 C3 — differentiable Δ_grid projection
Differentiable projection onto `Δ_grid` (box+budget+outage+branch-flow).
- **Primary:** cvxpylayers/OptNet QP layer; **gate:** matches the reference
  `flow_projection.project_onto_delta_grid` within 1e-6 + finite-diff gradient check.
- **Stretch (real contribution):** near-linear **laminar tree-DP** projection
  (subtree-sum constraints are a laminar family). Algorithm sketch in the work
  order §2 / `docs/06`. Note: cvxpylayers is NOT installed — add to env.
- Branch `c3-diff-projection`. The reference (non-diff) solver already exists as
  the correctness oracle.

### 5.4 C4 — CVaR objective + regret bound; C6 — d_struct atlas
- C4: replace lexicographic checkpointing with CVaR_α; regret bound (`docs/06` Thm 3).
- C6 (v2, DEFERRED): populate `configs/feeder_atlas.yaml` (currently empty
  skeleton) with structural fingerprints; define d_struct; show it predicts
  transfer. **Do NOT start until P0 confirms topology is causal** — C6's premise
  is exactly what P0 questions. Needs ≥20 train-test topology pairs (v2 gate).

### 5.5 How to start (the graph is loadable in 3 lines)
```python
from sg_resilience.eval_harness import _FeederCtx
ctx = _FeederCtx({"id":"x","source":"simbench","code":"1-LV-rural3--0-sw"}, horizon=64)
ctx.node_order          # load node ids
ctx.tree                # rooted radial tree: .edges, .edge_capacity_mw (F_e), .subtree_loads
ctx.adjacency()         # sparse row-normalized real topology (n×n); adjacency(rewire_seed=k) for wrong-graph
ctx.cidx, ctx.cw, ctx.cm   # critical indices, weights, min-service
```
Plug a new operator into `policies.build_policy` (score head), train via
`h1_train.train_policy` (imitation) or against LP-optimal targets
(`scripts/h1_lp_target_test.py`), eval via `h1_train.eval_policy_C` (hard greedy → C).

---

## 6. Division of labor (current)
| Area | Owner |
|---|---|
| Resilience env, feasibility/flow infra, metrics, eval methodology, baselines, diagnostics | Jia (done) |
| C1 flow-aware operator, C2 transfer protocol, C6 atlas/d_struct | Fanchen |
| C3 differentiable projection | Joint (reference done by Jia; differentiable = Fanchen) |
| C4 CVaR theory | Fanchen lead |
| P0 topology-causality adjudication | Fanchen (uses Jia's diagnostics) |
| Paper direction (A/B/C) | Jia (lead), after P0 |

---

## 7. Open infra items (Jia side, non-blocking for Fanchen)
- Canonical result schema `schemas/canonical_results_v1.schema.json` (`docs/11` #6).
- Report scripts (`docs/05` §6).
- Harsher wrong-graph controls + headroom-to-oracle reporting (support P0).
- `docs/01/02/05/10` already reframed; reconcile after the A/B/C decision.

## 8. Repro / environment
`pandapower 3.4, simbench 1.6.2, networkx 3.6, torch 2.6, cvxpy 1.8, numpy`.
Missing for C3-primary: `cvxpylayers`. Tests: `PYTHONPATH=src python -m pytest tests/ -q` (22 pass).
Smoke / diagnostics: `scripts/validate_feeders_v1.py`, `scripts/diagnose_decision_room.py`,
`scripts/h1_sparse_final.py`. Repo: github.com/peijiapok/microgrid-graph, branch `reframe-continuity-flow`.
