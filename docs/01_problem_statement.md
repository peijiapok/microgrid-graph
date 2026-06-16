# 01 — Problem Statement

> Last revised 2026-06-16 (was 2026-04-22). Authoritative statement of the problem this project solves. Every other doc defers to this file.
>
> **2026-06-16 reframing (ADR-0001, ADR-0002):** temporal continuity of critical-load service is now the **primary** objective and metric (not adequacy); the feasible set is **flow-constrained** (`Δ_grid`, radial branch-flow capacities), so topology enters the math, not just the policy. These two changes are coupled — continuity is the headline only because `Δ_grid` makes an unseen topology change what is continuously deliverable.

## 1. Informal statement

### The real-world setting: catastrophic grid events

When a distribution grid experiences a catastrophic event — a hurricane severs a feeder, a wildfire forces generation offline, a coordinated outage takes out substations, an extreme-heat day drives demand past supply — there is no longer enough power to serve every load. A controller must decide, step by step, how to allocate the scarce power that remains: hospitals and data centers stay on, lower-priority loads are shed, and the decisions must not flip loads on and off erratically. This is not an academic problem. Operators make this decision under pressure, on feeders whose exact topology their playbooks never anticipated, and the cost of getting it wrong is measured in critical-infrastructure downtime — not in regret.

### The ML setting: topology-generalizing constrained allocation on graphs

We formalize the catastrophic-grid problem as an ML problem. Concretely: one learned graph-structured policy must decide, at each time step and for each node of an electrical distribution feeder, how much of a scarce power budget to route there — respecting hard budget, non-negativity, and demand caps; preserving continuity of service to critical nodes; and doing so on feeder topologies **never seen during training**. That last requirement is the hinge of the whole project. During a real crisis no one has time to retrain a model on the specific feeder that just failed; either the policy generalizes zero-shot across topologies with formal guarantees, or it is useless when it matters.

The ML kernel is *not* "resilient load allocation" in the generic sense. The ML kernel is:

> *Can a graph policy trained on topology family **G**ᵗʳᵃⁱⁿ produce zero-shot feasible, constraint-respecting, continuity-preserving decisions on an unseen topology family **G**ᵒᵒᵈ — and can we bound the worst-case continuity gap as a function of (GNN expressivity) + (topology coverage) + (adversary budget)?*

Catastrophic grid events are the high-stakes setting that makes this the right question. Topology-generalizing graph control under hard constraints is the paper.

## 2. Formal setup

**Graph and attributes.** A feeder is a graph G = (V, E, X, W) with:
- V: load nodes (|V| = n, variable across feeders).
- E ⊆ V × V: feeder connectivity (directed from source to loads).
- X: edge features (admittance, thermal capacity F_e, direction-to-source). **Now load-bearing in two places: the C1 flow-aware operator and the hard branch-flow constraint #4.** Exposed by `src/sg_resilience/topology.py`.
- W: per-node attributes (priority wᵢ ∈ ℕ, critical flag cᵢ ∈ {0,1}, min-service fraction mᵢ ∈ [0,1], static topology features).

**Time.** Horizon T. Per step t ∈ {1, …, T}:
- dₜ ∈ ℝ₊ⁿ: per-node demands.
- Pₜ ∈ ℝ₊: global available power.
- Oₜ ⊆ V: outage set.
- sₜ: state bundle (dₜ, Pₜ, Oₜ, derived context).

**Policy.** π_θ : (G, sₜ, aₜ₋₁) → aₜ ∈ ℝ₊ⁿ parameterized by a GNN θ.

**Hard constraints (must hold for every output at every step), enforced by projection onto `Δ_grid`, never by penalty.**
1. Budget:  Σᵢ aₜ,ᵢ ≤ Pₜ.
2. Demand cap: 0 ≤ aₜ,ᵢ ≤ dₜ,ᵢ for all i.
3. Outage: aₜ,ᵢ = 0 for i ∈ Oₜ.
4. **Branch-flow capacity (new, ADR-0001):** for every tree edge e, |f_e(aₜ)| ≤ F_e, where f_e(aₜ) = Σ_{j ∈ subtree(e)} aₜ,ⱼ and F_e is the edge thermal capacity. This is what makes the feasible set topology-dependent. It is a *linear* branch-flow capacity on a radial tree, **not** an AC power-flow solution (see §6).

**Objective — primary is temporal continuity (ADR-0001).** The paper-level objective, in priority order:
- **(primary) maximize temporal continuity of critical service:** keep each critical load served at ≥ its min-service fraction for *uninterrupted* runs over time, weighted by priority and normalized for scarcity. Measured by the capacity-normalized windowed-continuity metric `C` (`src/sg_resilience/metrics_v1.py`; defined in `docs/05 §2`, `docs/10 §9`); trained via the differentiable soft windowed-continuity surrogate (`src/sg_resilience/continuity_loss.py`).
- (guard) maintain critical-load adequacy  mean over t,i∈critical of 𝟙[aₜ,ᵢ ≥ mᵢdₜ,ᵢ] — secondary; prevents trading away all adequacy for continuity.
- maximize weighted served energy  Σ_t wᵢ · aₜ,ᵢ.
- minimize switching  Σ_t |aₜ − aₜ₋₁|₁ — kept small as anti-chatter regularization only, **not** the continuity mechanism (switching is magnitude-smoothness, not continuity).

**Evaluation regime.** Held-out scenarios *within* a topology give the current benchmarks. The project adds a **held-out topology** regime: G_test ∉ G_train. This is the new axis.

## 3. Why this is graph-native (not an MLP dressed up)

An MLP cannot answer the transfer question at all — its parameter count binds to n. Any policy that must act on feeders of different sizes and connectivities is a function on graphs, not on vectors. More sharply:

- **Permutation-equivariance.** Any relabeling of nodes that preserves G must produce the same allocation up to the same relabeling. This rules out MLP + positional encoding without graph message passing.
- **Size-generalization.** The policy must act on |V| = n_train and |V| = n_test ≠ n_train.
- **Topology sensitivity.** Two feeders with identical node features but different edge sets produce different correct allocations, because electrical locality matters.
- **Flow is directed and weighted.** Power flows along admittances from source to load. Symmetric unweighted adjacency (the old code) throws away the signal. A flow-aware directed graph operator is the C1 contribution.

## 4. Inputs the method must consume

| Input | Shape | Source |
|---|---|---|
| node features | n × 14 | `featurize_state()` in `controller_model.py` |
| rooted radial tree + edge capacities F_e | n nodes, |E| edges | `topology.build_radial_tree()` — honors switch/in_service state |
| edge features | |E| × d_e | admittance, log F_e, direction-to-source — consumed by C1 and by branch-flow constraint #4 |
| previous allocation | n | rolling state |
| demands, power budget, outage mask | n, scalar, subset | `TimeStepState` in `scenario_schema.py` |

## 5. Outputs and acceptance criteria

Per step the policy returns aₜ ∈ ℝ₊ⁿ. Acceptance for a full rollout:

1. **Feasibility invariant.** Budget, non-negativity, demand cap, **and branch-flow capacity** all satisfied within `epsilon_feas = 1e-6` (normalized) — `Δ_grid` membership, not soft penalties.
2. **Transfer invariant.** On held-out topology G_test, capacity-normalized windowed continuity `C` ≥ (in-family `C`) − Δ_transfer, with Δ_transfer pinned in `05_experimental_protocol.md §2.5` (now on `C`, the primary metric).
3. **Reproducibility invariant.** Smoke evaluation runs in < 10 min on CPU with a fresh clone.

## 6. What this project is *not* solving

Explicit non-goals, so reviewers do not read expectations into the paper:

- **Not** full physics-constrained learning. A power-flow solver is *not* in the training loop. Feasibility means membership in the flow-constrained allocation polytope `Δ_grid` (box + global budget + **linear radial branch-flow capacities**), not the AC power-flow manifold. We say "flow-constrained," not "physics-aware": capacity limits on a radial tree are enforced; Kirchhoff/Ohm voltage relations are not.
- **Not** real-time deployment. Wall-clock per step is a metric, not a deployment target.
- **Not** cross-grid-type transfer (transmission ↔ distribution). Topology generalization within distribution feeders only.
- **Not** a claim that the learned policy beats rule/optimization on every metric on every feeder. The claim is zero-shot transfer + bounded worst-case gap.

## 7. Pointers

- Canonical code: `src/sg_resilience/`
- Authoritative objective: `src/sg_resilience/training.py`
- Metrics: `src/sg_resilience/priority_metrics.py`
- Reframed goal: `docs/02_research_goal.md`
- Collaborator brief: `docs/03_collaborator_brief.md`
