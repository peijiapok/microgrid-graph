# H1 negative finding — topology is not load-bearing for the *decision* (2026-06-17)

> Decisive result from the H1 pilot + decision-room diagnostics (GPT-5.5-guided). This challenges the paper's central C2/H1 hypothesis and needs a lead decision on direction.

## Evidence
**1. H1 pilot (DeepSets vs GraphSAGE, oracle-calibrated, imitation of flow-aware greedy):**
- DeepSets (no-graph) OOD C = **0.993**; GraphSAGE (graph) OOD C = **0.935**.
- H1 (graph − no-graph) OOD = **−0.058** → the graph policy is *worse*, overfitting training topology while topology-blind DeepSets transfers fine.

**2. Decision-room diagnostic** (priority-greedy vs LP-optimal critical-set selection, per feeder × flow-cap tightness; `scripts/diagnose_decision_room.py`):
- **room ≈ 0.000 for nearly all feeders at all tightness levels** (min-service serving). Tightening caps 8× changes nothing. Only MVLV feeders show a tiny constant 0.01–0.02.
- **Full-demand serving:** still room ≈ 0 everywhere except case33bw at the extreme 0.125× cap (0.109).

## Diagnosis (confirmed by GPT-5.5)
*"The graph is in the executor, not in the decision."* Feasibility is delegated to a shared flow-aware allocator (topology-aware for everyone). The learned policy only controls the **ordering**, and the optimal ordering is **priority-dominated**: serve highest-priority criticals until blocked. On realistic radial feeders, line capacities carry the feeder with headroom, so flow caps rarely bind on critical service → priority-greedy is LP-optimal → topology adds no decision value, only OOD overfitting risk. This is a structural property of priority-weighted continuity on capacitated radial trees, not a tuning issue (it survives 8× cap tightening and full-demand serving).

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
