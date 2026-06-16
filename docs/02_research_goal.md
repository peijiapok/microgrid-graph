# 02 — Research Goal

> Last revised 2026-06-16 (was 2026-04-22). Reads as the abstract-plus-limits of the target paper. If a claim does not appear here, we do not make it.
>
> **2026-06-16 reframing (ADR-0001):** primary target is **temporal continuity of critical service** under a **flow-constrained** feasible set `Δ_grid`. "Flow-constrained," not "physics-aware" — we enforce linear radial branch-flow capacities, not AC power flow.

## 1. One-sentence pitch

When a distribution grid is in crisis, a **flow-constrained** graph policy — trained on a family of feeder topologies with a CVaR-minimax objective and a differentiable projection onto the branch-flow-capacitated allocation polytope `Δ_grid` — must generalize zero-shot to unseen topologies and keep critical infrastructure **continuously online**; we provide such a policy, a graph-mining-native structural-similarity metric that predicts how well it will transfer *before training*, and a bound on the worst-case **continuity** gap in terms of GNN expressivity, structural coverage, and adversary budget. The primary metric is capacity-normalized windowed continuity `C`; topology is load-bearing precisely because `Δ_grid` makes an unseen feeder change what is continuously deliverable.

## 1a. Why this matters

Catastrophic events — weather disasters, cascading outages, coordinated strikes on infrastructure, extreme-heat overloads — are when grid control becomes both high-stakes and hardest. Supply collapses, playbooks break, and the feeders that matter may not be the ones the operator has trained a model on. A controller worth trusting in those conditions must therefore work on graphs it has never seen, and we must be able to predict how well it will work before we deploy it. Existing learned controllers neither evaluate this nor bound it. This paper does both.

## 2. The five contributions

The contributions are organized so each **kills one specific NeurIPS-review-killer** in the current draft.

### C1. Flow-aware message passing

**Review-killer it addresses:** "graph structure is decorative; an MLP would work."

**What.** Replace the plain symmetric SAGE operator in `controller_model.py` with a **directed MPNN / Graph-Network block** (decided 2026-06-16; spec in `notes/work_order_fanchen_2026-06-16.md`): two arcs per line (up/downstream), edge features [admittance, log F_e, direction], **capacity-gated messages** `g_uv=σ(φ_g(e,h_u,h_v))`, sum aggregation, GRU update. Asymmetric along the feeder, capacity-aware, permutation-equivariant, size-generalizing.

**Done-when.** (i) An MLP ablation loses ≥ Δ on critical continuity on at least one in-family feeder. (ii) A scrambled-edge ablation loses ≥ Δ on the same metric. (iii) The flow-aware operator matches or beats SAGE on every completed feeder.

### C2. Topology-generalizing policy (with structural atlas)

**Review-killer it addresses:** "you trained and tested on the same feeder; this is not a graph-generalization paper."

**What.** Train once on a curated family G_train of feeders. Evaluate zero-shot on a held-out family G_ood. Report the transfer gap for every metric. In parallel, produce a **structural feeder atlas** cataloguing every candidate feeder by graph-mining-native properties (degree-distribution moments, clustering, spectral gap, algebraic connectivity, diameter, motif / graphlet counts up to k=4, community signature) so that splits are principled, not accidental.

**Done-when.** (i) A single policy evaluated on the held-out feeders (frozen split, `configs/topology_split_v1.yaml` / ADR-0002) has capacity-normalized continuity `C` within Δ_transfer of the in-family value. (ii) The same policy is tested at larger feeder sizes (MVLV-urban OOD, |V| > all train feeders) to demonstrate size-generalization. (iii) `configs/feeder_atlas.yaml` lists every evaluated feeder with its structural fingerprint (v2). (iv) Every split referenced in the paper is justified structurally, not by convenience.

### C3. Differentiable feasibility projection

**Review-killer it addresses:** "your 'projection' is an iterative water-filler with a loop bound; prove feasibility exactly, end-to-end differentiable."

**What.** Differentiable projection onto the **flow-constrained** polytope `Δ_grid = {a ≥ 0, a ≤ d, a=0 on outage, Σa ≤ P, |f_e(a)| ≤ F_e ∀ tree edge e}`. Decided 2026-06-16: ship a cvxpylayers/OptNet differentiable QP layer first (gate: match the reference `flow_projection.project_onto_delta_grid` within 1e-6 + finite-diff gradient check), then a near-linear **laminar tree-DP** projection as a stretch contribution (subtree-sum rows form a laminar family). Spec: `notes/work_order_fanchen_2026-06-16.md`.

**Done-when.** (i) Outputs satisfy all four hard constraints (incl. branch-flow) within `epsilon_feas`. (ii) Gradient flow verified on small cases. (iii) No performance regression vs. the reference projection.

### C4. CVaR-minimax objective with regret bound

**Review-killer it addresses:** "'worst-case-aware selection' means picking the best checkpoint; there is no theory."

**What.** Replace lexicographic checkpoint selection with a proper CVaR_α objective at training time. Give a regret bound linking training worst-case to deployment worst-case (Theorem 3 in `06_theory_sketch.md`).

**Done-when.** (i) The CVaR-trained policy attains strictly lower worst-case unserved critical demand than the lexicographic-selected policy on at least one in-family feeder. (ii) The bound is non-vacuous on our empirical setup (the bound's numerical value is less than the trivial upper bound of 1).

### C5. Graph-baseline zoo

**Review-killer it addresses:** "the only baselines are rule and classical optimization; what about GCN, GAT, GIN, EGNN, Graph Transformer?"

**What.** Run five standard GNN baselines with the same objective + projection + training loop. Report per-baseline transfer gaps.

**Done-when.** All five baselines complete on the in-family and OOD protocols, with a single table comparing them to the flow-aware operator.

### C6. Topology-similarity metric that predicts transfer

**Review-killer it addresses:** "your transfer bound is abstract; the d_WL metric is not computable on our feeders. Why should we believe your generalization claim a priori?"

**What.** Define a concrete, computable topology-similarity metric d_struct(G, G') from graph-mining primitives (graphlet counts, spectral signatures, degree-distribution distance, or a learned composite). Show empirically that d_struct between the training-topology distribution and each test feeder predicts the observed transfer gap, with positive slope and non-trivial R². Use d_struct to replace the abstract d_WL in Theorem 1 (`docs/06_theory_sketch.md`).

**Done-when.** (i) d_struct is implemented and computable on every feeder in the atlas in under one second. (ii) Across a sweep of (G_train, G_test) pairs, the correlation between d_struct and observed transfer gap has R² > 0.5 on at least one metric. (iii) Theorem 1 is restated in terms of d_struct and the constant in the bound is numerically evaluated on our data. (iv) The restated Theorem 1 is non-vacuous (numerical bound < trivial upper bound 1) on our empirical setup.

**Why this matters.** If d_struct works, the paper's headline becomes *"we predict graph generalization from structure alone"* — a contribution that elevates the paper from competent to signature. This is the Gate-4 "best paper" move, pulled forward to a proper contribution rather than deferred.

## 3. What we will NOT claim

- Universal dominance over rule/optimization baselines on every metric on every feeder.
- Physics-constrained learning with an AC/DC power-flow solver in the loop. (We DO enforce linear radial branch-flow *capacity* constraints in `Δ_grid`; we do NOT solve power flow. Call it "flow-constrained," never "physics-aware.")
- Real-world deployment readiness.
- Cross-grid-type transfer (transmission ↔ distribution).
- Safety in the formal control-theoretic sense beyond the stated feasibility theorem.

These exclusions are contractual. Any draft sentence that strays into one of them gets cut in review.

## 4. Success criteria (hierarchical)

- **Gate 1 — Viability.** C1 and C2 both achieve their done-when clauses on the frozen split (train: case33bw, LV-rural2, LV-rural3; OOD: LV-urban6, MVLV-urban-5.303, MVLV-urban-6.305 — ADR-0002), measured on the primary continuity metric `C`. No theory yet.
- **Gate 2 — Workshop-ready (NeurIPS workshop 2026, target July–August).** Gate 1 + C5 complete + draft theory for C3.
- **Gate 3 — Main-conference-ready (NeurIPS 2027).** All five contributions complete; Theorems 1–3 proved; ablations; writing.
- **Gate 4 — "Best paper" posture.** C6 delivers on its done-when in full: d_struct is non-vacuous in Theorem 1, R² > 0.5 on at least two metrics (not one), and the atlas covers IEEE 123 plus two additional OOD families. Further extensions (expressivity characterizations specific to constrained allocation, a learned d_struct variant) evaluated at Gate 3.

## 5. Non-goals for v1

Do not add, in the first pass:
- RL fine-tuning (current training is supervised + imitation; stay there).
- Multi-agent / decentralized policies.
- Communication or coordination constraints.
- Stochastic demand forecasting as a separate module.

Each of these is a paper on its own. Keeping scope narrow is how this project finishes.

## 6. Explicit link to current code

- C1 touches: `controller_model.py` (GraphSAGEBlock, build_adjacency_tensor).
- C2 touches: `benchmark_loader.py`, `scenario_generator.py`, new `topology_splits.py`, new `topology_atlas.py`.
- C3 touches: `controller_model.py` (project_allocation).
- C4 touches: `training.py` (loss, checkpoint selection).
- C5 touches: new `baselines_gnn.py`.
- C6 touches: new `topology_similarity.py`, `configs/feeder_atlas.yaml`, tests in `tests/theory/`.

This gives the collaborator a concrete file-level map on day 1.
