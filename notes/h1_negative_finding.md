# H1 — topology load-bearing is CONDITIONAL on constraint-binding (2026-06-17)

> RESOLVED into a conditional result (stronger than pure negative). Pilot + decision-room diagnostics + congestion sweep + B-viability test (GPT-5.5-guided).

## SCOPE NOTE (read first — corrected 2026-07-06)
This is a **control-side probe** about whether a *GNN control policy* needs
topology, run from Jia's side with a quick GraphSAGE/DeepSets. It is **NOT** a
verdict on Fanchen's research. Fanchen's actual work (ICDM 2025, random graph
models / structural graph mining) is on the **structure** side — `d_struct`,
feeder structural fingerprints, and RGM-based feeder generation (F1–F3 in
`notes/work_order_fanchen_2026-07-06.md`) — a **different question this
experiment does not touch.** "Does feeder structure predict transfer?" (his) is
wide open; "does a GNN *control policy* need topology?" (this note) is the narrow
thing probed here. The substantive, operator-somewhat-independent finding is
formulation-level: delegating feasibility to a shared topology-aware allocator
makes the policy's *decision* topology-light (GPT-5.5: "largely inevitable"). Do
NOT read this as "the graph research is infeasible" — it says nothing about the
structural approach.

## PRELIMINARY DIAGNOSTIC (2026-06-17, sparse topology + wrong-graph control)
The wrong-graph control overturns the earlier optimistic read. With the **real
sparse feeder topology** and LP-optimal training:

| cap_scale | no-graph | real-graph | wrong-graph | real−nograph | real−wrong |
|---|---|---|---|---|---|
| 1.0 (realistic) | 0.979 | 0.990 | — | +0.011 | — |
| 0.03 (crisis) | 0.886 | 0.977 | 0.975 | **+0.091** | **+0.002** |

**A degree-preserving WRONG graph does as well as the real one (+0.002).** So the
congestion benefit comes from a **generic relational/message-passing inductive
bias, NOT from the actual feeder topology.** The strong claims —
"topology-generalizing graph control" (C2) and "structural distance predicts
transfer" (C6) — are **NOT supported**: real feeder structure is not load-bearing,
even under crisis congestion. Topology-blind control (DeepSets) is near-optimal
and more OOD-robust on realistic feeders.

**This is the honest, rigorous result of the project:** a negative result with
clean no-graph AND wrong-graph controls. It is publishable as a cautionary
ML-for-power finding, but it guts the original graph-necessity thesis (C1/C2/C6
lose their load-bearing role). **Lead decision needed** — see updated options
below; the strongest honest paper is the negative result; a positive graph paper
would require a genuinely topology-coupled DECISION (network reconfiguration /
AC feasibility), i.e. a different problem.

### GPT-5.5 sign-off + hardening checklist (to make the negative result claim-bearing)
GPT-5.5 confirms the conclusion (phrase narrowly: *in this architecture/objective, once feasibility is a shared topology-aware allocator, feeder topology is not load-bearing for the learned policy*). Recommended paper: **(a) rigorous negative result** > (b) pivot decision to topological (reconfiguration/AC = new paper) > (c) drop graphs. Headline: *"shared topology-aware allocators can make graph policies look useful while the actual feeder graph is non-causal for the learned decision."* Confounds to close before final claim:
1. Harsher wrong-graph controls: random tree, star/path, kNN-on-features, shuffled-adjacency-per-episode (not only degree-preserving rewire).
2. State allocator leakage explicitly (topology enters as the constraint module; the point, not a flaw).
3. Report **headroom-to-oracle**, not just raw C (cap 1.0 is saturated; cap 0.03 is the diagnostic).
4. Check whether LP-target selection is feature-only predictable (feature-only predictor vs oracle; do selected nodes cluster topologically?).
5. Multi-seed CIs for real−wrong: the claim needs "indistinguishable," not just close point estimates.

## RESOLUTION (read first) [SUPERSEDED by FINAL VERDICT above — the +0.092 was generic message-passing, not real topology]
Graph-necessity is **conditional on flow constraints binding**:
- **Realistic feeders (capacity headroom):** decision-room ≈ 0; priority-greedy is optimal; the learned graph policy ties or *underperforms* no-graph (overfits). Topology-blind control suffices and is more OOD-robust.
- **Congested/crisis regime (cap_scale≈0.03) + training against the LP-OPTIMAL set-selection:** GraphSAGE OOD C **0.982 vs DeepSets 0.898 (+0.084)** — graph is genuinely load-bearing.

**Crucial:** the congested regime where topology matters is exactly the **catastrophe / scarce-power regime that motivates this project** (docs/01–02). So this is not unrealistic engineering — it is the crisis the paper is about. Both conditions (congestion AND optimal-target training) are necessary; neither alone gives a graph benefit.

**CONFIRMED central curve** (LP-optimal training, 3 model seeds, `scripts/h1_conditional_sweep.py`):
| cap_scale | graph−nograph OOD C |
|---|---|
| 1.0 (realistic) | +0.009 ± 0.003 (≈0) |
| 0.06 | +0.024 ± 0.030 |
| 0.03 (crisis) | **+0.092 ± 0.005** (all seeds 0.084–0.097) |
Monotonic in congestion, tight at the congested end — the conditional graph-necessity claim holds with multi-seed error bars.

**Recommended paper claim:** "Topology is load-bearing for continuity control precisely in the flow-constrained crisis regime; under normal headroom, topology-blind control suffices and transfers better. Capturing the graph benefit requires optimal-target training, not greedy imitation. We characterize the decision-room→congestion relationship that governs when graph structure helps." (Caveat: +0.084 is a single modest-seed run — confirm with a multi-seed cap-scale sweep before claim-bearing.)

---

## Original framing (the path to the resolution)

## Evidence
**1. H1 pilot (DeepSets vs GraphSAGE, oracle-calibrated, imitation of flow-aware greedy):**
- DeepSets (no-graph) OOD C = **0.993**; GraphSAGE (graph) OOD C = **0.935**.
- H1 (graph − no-graph) OOD = **−0.058** → the graph policy is *worse*, overfitting training topology while topology-blind DeepSets transfers fine.

**2. Decision-room diagnostic** (priority-greedy vs LP-optimal critical-set selection, per feeder × flow-cap tightness; `scripts/diagnose_decision_room.py`):
- **room ≈ 0.000 for nearly all feeders at all tightness levels** (min-service serving). Tightening caps 8× changes nothing. Only MVLV feeders show a tiny constant 0.01–0.02.
- **Full-demand serving:** still room ≈ 0 everywhere except case33bw at the extreme 0.125× cap (0.109).

## Diagnosis (confirmed by GPT-5.5)
*"The graph is in the executor, not in the decision."* Feasibility is delegated to a shared flow-aware allocator (topology-aware for everyone). The learned policy only controls the **ordering**, and the optimal ordering is **priority-dominated**: serve highest-priority criticals until blocked. On realistic radial feeders, line capacities carry the feeder with headroom, so flow caps rarely bind on critical service → priority-greedy is LP-optimal → topology adds no decision value, only OOD overfitting risk. This is a structural property of priority-weighted continuity on capacitated radial trees, not a tuning issue (it survives 8× cap tightening and full-demand serving).

## Open methodological item — adjacency is dense (affects how "graph" the GNN is)
The loader's load adjacency (`benchmark_loader._build_load_adjacency_matrix`) is a
distance-weighted **complete** graph (every load pair has weight 1/(1+dist) > 0).
So the GNN message-passes over a near-dense distance kernel, not sparse feeder
topology — which (a) weakens the topology signal (partly why GraphSAGE was only
marginally better) and (b) makes degree-preserving rewire ill-posed (the
wrong-graph H1 test crashed: can't rewire a complete graph). **Next step:** build
a SPARSE tree-based load adjacency (loads adjacent iff their buses are tree-
adjacent) so the GNN uses real sparse topology, then redo the conditional curve +
the wrong-graph (rewire) test. The conditional result (+0.092 under congestion)
stands on the distance-weighted graph (a legitimate but dense representation);
sparse topology is expected to sharpen it.

## Where topology DOES become load-bearing (congestion sweep + perturbation)
Full-demand decision room vs flow-cap scale (fraction of real line ratings):
| feeder | capx 0.25 | 0.125 | 0.06 | 0.03 |
|---|---|---|---|---|
| case33bw | 0.000 | 0.109 | 0.115 | 0.130 |
| rural2 | 0.000 | 0.000 | 0.000 | 0.104 |
| mvlv-5303 | 0.000 | 0.000 | 0.001 | 0.070 |

Decision room (priority-greedy sub-optimality, where topology can help) only
appears at **~0.03–0.125× of real line ratings** — i.e. ~8–30× more congested
than realistic feeders. Perturbation test (case33bw, capx=0.06): halving the
single most-loaded edge shifts the LP-optimal critical set (served-weight
0.55→0.42, L1 set change 0.90), confirming the optimal decision is genuinely
topology/capacity-dependent **in that severe-congestion regime only**.

## Congestion-stratified LEARNED H1 (DeepSets vs GraphSAGE vs cap-scale)
| cap_scale | graph OOD C | no-graph OOD C | graph−nograph |
|---|---|---|---|
| 1.0 (realistic) | 0.968 | 0.926 | +0.042 |
| 0.125 | 0.968 | 0.926 | +0.042 |
| 0.06 | 0.916 | 0.931 | −0.016 |
| 0.03 (severe) | 0.864 | 0.913 | **−0.049** |

The learned graph advantage **decreases** with congestion (opposite of the hoped
curve), and the realistic-cap value (+0.042) is within run-to-run noise (pilot
was −0.058) → net ≈ 0. Critical insight: decision-room exists under congestion,
but **both policies imitate the priority-greedy rule, which is itself suboptimal
there** — so the GNN cannot exploit the topology-dependent room and its capacity
overfits. **A positive graph result requires BOTH (a) congestion (room exists)
AND (b) training against the OPTIMAL set-selection (LP/oracle), not the greedy
rule. Neither alone is sufficient** (decisive B-viability test, not yet run:
train against LP-optimal targets at cap_scale≈0.03).

## Implication
The C2/H1 claim as written — *"graph structure is load-bearing for transferable continuity control"* — is **not supported** in this formulation. Continuing to tune the learner will not fix a formulation in which the decision is topology-independent.

## Options (lead decision)
**A. Honest negative / conditional result (recommended default).** Reframe the paper around the rigorously-supported claim (GPT-5.5's fallback):
> *Once feasibility is delegated to a shared flow-aware allocator, topology contributes to constraint projection, not policy choice. For priority-dominated continuity objectives on capacitated radial feeders, topology-blind policies are near-optimal and more OOD-robust; graph learning helps only when the policy must make topology-dependent commitment/sacrifice decisions — a regime these feeders rarely enter.*
This is publishable (a cautionary/negative ML-for-power result with clean diagnostics) and defensible. It repurposes all the built infrastructure.

**B. Reformulate so the decision is topology-dependent.** Bigger scope; candidates:
- Policy controls **network reconfiguration / switching** (inherently topological) rather than just allocation ordering.
- **Congestion-heavy** synthetic feeders engineered with real bottlenecks (departs from realistic SimBench feeders — must be justified).
- **AC/voltage-constrained** feasibility (non-separable, topology-coupled) — but this is the explicitly de-scoped "physics-aware" path.
- Objective where **clusters beat isolated high-priority loads** (e.g., partially de-emphasize priority) — changes the application semantics.

**C. Drop the graph framing** and pivot to the part that *is* real: the flow-constrained continuity *evaluation suite* + the finding that topology-blind control is robust. (Weakest as a NeurIPS "graph" paper.)

## What's solid regardless
- The eval methodology (ADR-0003: oracle-calibrated matched scarcity, T≥64, paired seeds) and the full pipeline (metrics, greedy allocator, calibration, diagnostics) are sound and reusable.
- The diagnostics themselves are a contribution: a clean test for *whether topology is load-bearing in the decision* of a constrained graph-control problem.

## Recommendation
Take **A** as the spine (it's the truth the experiments support) and consider **B-reconfiguration** only if a positive graph result is required and the scope is acceptable. Do NOT force a positive H1 by engineering unrealistic bottlenecks (option B-congestion) without flagging it as synthetic.
