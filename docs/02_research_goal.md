# 02 — Research Goal

> Last revised 2026-04-22. Reads as the abstract-plus-limits of the target paper. If a claim does not appear here, we do not make it.

## 1. One-sentence pitch

A physics-aware graph policy, trained on a family of distribution-grid topologies with a CVaR-minimax objective and a differentiable feasibility projection, generalizes zero-shot to unseen topologies, and the worst-case continuity gap is bounded by a term capturing GNN expressivity, training-topology coverage, and adversarial budget.

## 2. The five contributions

The contributions are organized so each **kills one specific NeurIPS-review-killer** in the current draft.

### C1. Flow-aware message passing

**Review-killer it addresses:** "graph structure is decorative; an MLP would work."

**What.** Replace the plain symmetric SAGE operator in `controller_model.py` with an operator that respects the directed, weighted, capacity-constrained nature of power flow. Edge features include line admittance, thermal capacity, and (optional) distance-to-source.

**Done-when.** (i) An MLP ablation loses ≥ Δ on critical continuity on at least one in-family feeder. (ii) A scrambled-edge ablation loses ≥ Δ on the same metric. (iii) The flow-aware operator matches or beats SAGE on every completed feeder.

### C2. Topology-generalizing policy

**Review-killer it addresses:** "you trained and tested on the same feeder; this is not a graph-generalization paper."

**What.** Train once on a curated family G_train of feeders. Evaluate zero-shot on a held-out family G_ood. Report the transfer gap for every metric.

**Done-when.** (i) A single policy evaluated on at least two held-out feeder families has mean critical continuity within Δ_max of the in-family value. (ii) The same policy is tested at multiple feeder sizes (|V| ratios) to demonstrate size-generalization.

### C3. Differentiable feasibility projection

**Review-killer it addresses:** "your 'projection' is an iterative water-filler with a loop bound; prove feasibility exactly, end-to-end differentiable."

**What.** Replace the iterative `project_allocation()` with a closed-form (or fixed-iteration) projection onto the allocation polytope {a ≥ 0, a ≤ d, Σa ≤ P}. Give a formal feasibility certificate. Keep it differentiable for end-to-end training.

**Done-when.** (i) Theorem 2 (`06_theory_sketch.md`) holds: outputs satisfy all three hard constraints exactly. (ii) Gradient flow through the projection is verified on small cases. (iii) No performance regression vs. the iterative version.

### C4. CVaR-minimax objective with regret bound

**Review-killer it addresses:** "'worst-case-aware selection' means picking the best checkpoint; there is no theory."

**What.** Replace lexicographic checkpoint selection with a proper CVaR_α objective at training time. Give a regret bound linking training worst-case to deployment worst-case (Theorem 3 in `06_theory_sketch.md`).

**Done-when.** (i) The CVaR-trained policy attains strictly lower worst-case unserved critical demand than the lexicographic-selected policy on at least one in-family feeder. (ii) The bound is non-vacuous on our empirical setup (the bound's numerical value is less than the trivial upper bound of 1).

### C5. Graph-baseline zoo

**Review-killer it addresses:** "the only baselines are rule and classical optimization; what about GCN, GAT, GIN, EGNN, Graph Transformer?"

**What.** Run five standard GNN baselines with the same objective + projection + training loop. Report per-baseline transfer gaps.

**Done-when.** All five baselines complete on the in-family and OOD protocols, with a single table comparing them to the flow-aware operator.

## 3. What we will NOT claim

- Universal dominance over rule/optimization baselines on every metric on every feeder.
- Physics-constrained learning (power-flow solver in the loop).
- Real-world deployment readiness.
- Cross-grid-type transfer (transmission ↔ distribution).
- Safety in the formal control-theoretic sense beyond the stated feasibility theorem.

These exclusions are contractual. Any draft sentence that strays into one of them gets cut in review.

## 4. Success criteria (hierarchical)

- **Gate 1 — Viability.** C1 and C2 both achieve their done-when clauses on at least the four existing feeders (case33bw, rural1, rural2, LV-urban) plus one newly added OOD feeder. No theory yet.
- **Gate 2 — Workshop-ready (NeurIPS workshop 2026, target July–August).** Gate 1 + C5 complete + draft theory for C3.
- **Gate 3 — Main-conference-ready (NeurIPS 2027).** All five contributions complete; Theorems 1–3 proved; ablations; writing.
- **Gate 4 — "Best paper" posture.** A genuinely new graph-theoretic idea (e.g., a topology-metric that predicts transfer gap a priori, or an expressivity characterization specific to constrained allocation) added as a sixth contribution. Scope decided at Gate 3.

## 5. Non-goals for v1

Do not add, in the first pass:
- RL fine-tuning (current training is supervised + imitation; stay there).
- Multi-agent / decentralized policies.
- Communication or coordination constraints.
- Stochastic demand forecasting as a separate module.

Each of these is a paper on its own. Keeping scope narrow is how this project finishes.

## 6. Explicit link to current code

- C1 touches: `controller_model.py` (GraphSAGEBlock, build_adjacency_tensor).
- C2 touches: `benchmark_loader.py`, `scenario_generator.py`, new `topology_splits.py`.
- C3 touches: `controller_model.py` (project_allocation).
- C4 touches: `training.py` (loss, checkpoint selection).
- C5 touches: new `baselines_gnn.py`.

This gives the collaborator a concrete file-level map on day 1.
