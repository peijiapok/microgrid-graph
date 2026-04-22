# 01 — Problem Statement

> Last revised 2026-04-22. Authoritative statement of the problem this project solves. Every other doc defers to this file.

## 1. Informal statement

We study **constrained sequential resource allocation on physical graphs under topology shift**. Concretely: one learned graph-structured policy must decide, at each time step and for each node of an electrical distribution feeder, how much of a scarce power budget to route there — respecting hard budget, non-negativity, and demand caps; preserving continuity of service to critical nodes; and doing so on feeder topologies **never seen during training**.

The ML kernel is *not* "resilient load allocation". The ML kernel is:

> *Can a graph policy trained on topology family **G**ᵗʳᵃⁱⁿ produce zero-shot feasible, constraint-respecting, continuity-preserving decisions on an unseen topology family **G**ᵒᵒᵈ — and can we bound the worst-case continuity gap as a function of (GNN expressivity) + (topology coverage) + (adversary budget)?*

Grid resilience is the instance. Topology-generalizing graph control is the paper.

## 2. Formal setup

**Graph and attributes.** A feeder is a graph G = (V, E, X, W) with:
- V: load nodes (|V| = n, variable across feeders).
- E ⊆ V × V: feeder connectivity (directed from source to loads).
- X: edge features (admittance, capacity, impedance) — **currently unused in code; see C1 in `02_research_goal.md`**.
- W: per-node attributes (priority wᵢ ∈ ℕ, critical flag cᵢ ∈ {0,1}, min-service fraction mᵢ ∈ [0,1], static topology features).

**Time.** Horizon T. Per step t ∈ {1, …, T}:
- dₜ ∈ ℝ₊ⁿ: per-node demands.
- Pₜ ∈ ℝ₊: global available power.
- Oₜ ⊆ V: outage set.
- sₜ: state bundle (dₜ, Pₜ, Oₜ, derived context).

**Policy.** π_θ : (G, sₜ, aₜ₋₁) → aₜ ∈ ℝ₊ⁿ parameterized by a GNN θ.

**Hard constraints (must hold for every output at every step).**
1. Budget:  Σᵢ aₜ,ᵢ ≤ Pₜ.
2. Demand cap: 0 ≤ aₜ,ᵢ ≤ dₜ,ᵢ for all i.
3. Outage: aₜ,ᵢ = 0 for i ∈ Oₜ.

**Soft objectives (traded off in the training loss).** Existing code implements seven terms — see `src/sg_resilience/training.py`. The paper-level objective is:
- maximize weighted served energy  Σ_t wᵢ · aₜ,ᵢ;
- minimize unserved critical demand  Σ_t Σ_{i critical} (dₜ,ᵢ − aₜ,ᵢ)₊;
- maximize critical continuity  mean over t,i∈critical of  𝟙[aₜ,ᵢ > 0];
- minimize switching  Σ_t |aₜ − aₜ₋₁|₁.

**Evaluation regime.** Held-out scenarios *within* a topology give the current benchmarks. The project adds a **held-out topology** regime: G_test ∉ G_train. This is the new axis.

## 3. Why this is graph-native (not an MLP dressed up)

An MLP cannot answer the transfer question at all — its parameter count binds to n. Any policy that must act on feeders of different sizes and connectivities is a function on graphs, not on vectors. More sharply:

- **Permutation-equivariance.** Any relabeling of nodes that preserves G must produce the same allocation up to the same relabeling. This rules out MLP + positional encoding without graph message passing.
- **Size-generalization.** The policy must act on |V| = n_train and |V| = n_test ≠ n_train.
- **Topology sensitivity.** Two feeders with identical node features but different edge sets produce different correct allocations, because electrical locality matters.
- **Physics is directed and weighted.** Power flows along admittances from source to load. Symmetric unweighted adjacency (the current code) throws away the signal. A physics-aware graph operator is the primary C1 contribution.

## 4. Inputs the method must consume

| Input | Shape | Source |
|---|---|---|
| node features | n × 14 | `featurize_state()` in `controller_model.py` |
| adjacency (current) | n × n, {0,1} | `build_adjacency_tensor()` — **to be upgraded with edge features** |
| edge features (target) | |E| × d_e | new: admittance, capacity, distance |
| previous allocation | n | rolling state |
| demands, power budget, outage mask | n, scalar, subset | `TimeStepState` in `scenario_schema.py` |

## 5. Outputs and acceptance criteria

Per step the policy returns aₜ ∈ ℝ₊ⁿ. Acceptance for a full rollout:

1. **Feasibility invariant.** Budget, non-negativity, and demand cap all satisfied *exactly* (violations = 0, not ε > 0 stochastic).
2. **Transfer invariant.** On held-out topology G_test, mean critical continuity ≥ (mean on in-family) − Δ_max, with Δ_max pinned in `05_experimental_protocol.md`.
3. **Reproducibility invariant.** Smoke evaluation runs in < 10 min on CPU with a fresh clone.

## 6. What this project is *not* solving

Explicit non-goals, so reviewers do not read expectations into the paper:

- **Not** full physics-constrained learning. A power-flow solver is *not* in the training loop. Feasibility here means the allocation constraint polytope, not the AC power-flow manifold.
- **Not** real-time deployment. Wall-clock per step is a metric, not a deployment target.
- **Not** cross-grid-type transfer (transmission ↔ distribution). Topology generalization within distribution feeders only.
- **Not** a claim that the learned policy beats rule/optimization on every metric on every feeder. The claim is zero-shot transfer + bounded worst-case gap.

## 7. Pointers

- Canonical code: `src/sg_resilience/`
- Authoritative objective: `src/sg_resilience/training.py`
- Metrics: `src/sg_resilience/priority_metrics.py`
- Reframed goal: `docs/02_research_goal.md`
- Collaborator brief: `docs/03_collaborator_brief.md`
