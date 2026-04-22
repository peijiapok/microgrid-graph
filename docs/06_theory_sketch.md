# 06 — Theory Sketch

> Last revised 2026-04-22. Owned by the collaborator. Target statements of the theorems we want in the paper, with proof directions and known obstacles. This file is intentionally a sketch — formalization happens in LaTeX once the statements stabilize.

## 1. Notation

- G = (V, E, X, W): feeder graph with edge features X and node attributes W.
- 𝒢: space of feeders. D_train, D_ood: distributions over 𝒢.
- Π_Θ: hypothesis class of graph policies π_θ, parameterized over Θ.
- k: number of message-passing layers; h: hidden width.
- n = |V|, m = |E|, Δ = max degree.
- Loss ℓ(π, G, ξ) for scenario ξ drawn from scenario distribution S_G.
- J(π; G) = 𝔼_{ξ ~ S_G}[ℓ(π, G, ξ)].
- CVaR_α(Z) = 𝔼[Z | Z ≥ VaR_α(Z)].
- d_WL(G, G'): k-WL-based graph distance (see Chen et al., "Can Graph Neural Networks Count Substructures?"-style formulations). Used as a theoretical baseline; replaced in practice by d_struct below.
- d_struct(G, G'): concrete, computable topology-similarity metric defined in C6 (`02_research_goal.md`) over the feeder atlas. Computed in seconds on every feeder in our setup.

## 2. Target Theorem 1 — Transfer bound

**Statement (target).** Let π_θ ∈ Π_Θ be trained to minimize 𝔼_{G ~ D_train}[J(π_θ; G)]. Let d_struct : 𝒢 × 𝒢 → ℝ₊ be the computable topology-similarity metric from C6, with per-feature definitions in `configs/feeder_atlas.yaml`. Under the GNN-Lipschitz assumption
‖π_θ(G) − π_θ(G')‖ ≤ L_Π · d_struct(G, G'),
and bounded loss ℓ ≤ B, with probability ≥ 1 − δ over the training sample of size N,

  𝔼_{G ~ D_ood}[J(π_θ; G)] ≤ 𝔼_{G ~ D_train}[J(π_θ; G)] + L_Π · W_1(D_train, D_ood; d_struct) + B·√( log(1/δ) / (2N) ).

**Reading.** The transfer gap is bounded by three interpretable terms: (i) training risk, (ii) topology-distribution shift measured in a *computable* structural metric, (iii) sample-size slack. The headline advantage over a d_WL formulation: every term can be numerically evaluated on our atlas, so the bound is not rhetorical.

**Proof directions.**
- Step 1: Rademacher-complexity bound for Π_Θ on the fixed-topology loss.
- Step 2: Lipschitzness of π_θ w.r.t. d_struct. Two paths: (a) bound L_Π directly by bounding the sensitivity of each message-passing layer to the atlas features; (b) learn L_Π empirically via finite differences across the atlas and verify the bound is tight.
- Step 3: Kantorovich–Rubinstein duality to convert Lipschitz + distribution-shift into W_1(·; d_struct).
- Step 4 (empirical backbone): regress observed transfer gap on d_struct across splits; the slope gives the *effective* L_Π used in the bound.

**Known obstacles.**
- The Lipschitz constant L_Π can be loose for deep GNNs. Mitigation above (empirical measurement) closes this.
- d_struct is a vector feature reduced to a scalar — the functional form (weighted sum / learned combiner / Wasserstein-on-features) is a design choice; we report robustness to it.
- The loss ℓ in our setting is *not* bounded uniformly when `unserved_critical_demand` is reported on large-P scenarios; either cap it or use a conditional bound.

**Non-vacuity check (hard gate).** Plug the empirical L_Π, measured W_1(D_train, D_ood; d_struct), and N into the bound. The resulting number must be less than the trivial upper bound of 1 on critical continuity. If it is not, the bound is rhetorical and is reworked or cut. This check is a required test in `tests/theory/`.

## 3. Target Theorem 2 — Feasibility guarantee for the projection

**Statement (target).** Let f_proj : ℝⁿ × ℝ₊ⁿ × ℝ₊ → ℝ₊ⁿ be the projection layer taking (weights w, demands d, budget P). Then for any input, the output a = f_proj(w, d, P) satisfies exactly:
  (i) a ≥ 0, elementwise;
  (ii) a ≤ d, elementwise;
  (iii) Σ a ≤ P.

**Reading.** The policy cannot output infeasible allocations, by construction.

**Proof direction.** Express the projection as the solution to
  min_a ‖a − g(w, d, P)‖²  s.t.  0 ≤ a ≤ d, Σ a ≤ P
for some scoring map g. The feasible set is a compact convex polytope; projection onto it is unique and satisfies the three inequalities by the KKT conditions. The remaining work is to show a closed-form or fixed-number-of-iteration scheme and prove it returns the true projection.

**Candidate schemes.**
- Dual-variable bisection on the budget Lagrangian (classic water-filling; O(n log n)).
- Sort-based closed form with a threshold τ (O(n log n)).
- Dykstra's algorithm (converges; needs iteration bound for differentiable training).

Collaborator chooses one; we prove correctness.

## 4. Target Theorem 3 — CVaR-minimax regret bound

**Statement (target).** Let π̂ = argmin_{π ∈ Π_Θ} max_{ξ ∈ Ξ} CVaR_α[ℓ(π, G, ξ)] over an adversarial scenario set Ξ. Then for deployment scenarios drawn from a distribution S̃ with S̃ stochastically dominated by the adversary-generated distribution,

  CVaR_α[ℓ(π̂, G, ξ) ; ξ ~ S̃] ≤ min_{π ∈ Π_Θ} max_{ξ ∈ Ξ} CVaR_α[ℓ(π, G, ξ)] + ε(Π_Θ, Ξ, N).

**Reading.** The deployment worst-case is bounded by the minimax training worst-case plus a hypothesis-class / adversary-set complexity term.

**Proof direction.** Minimax stochastic optimization with CVaR objective; standard techniques from Shapiro, Dentcheva, Ruszczyński. The ε term combines Rademacher complexity of Π_Θ with adversarial coverage of Ξ.

**Non-vacuity requirement.** The numerical value of ε must be smaller than the trivial upper bound of 1 on our empirical setup. If it is not, the theorem is not useful for the paper and we must tighten the hypothesis class or the adversary set.

## 5. Nice-to-have — Theorem 4 (expressivity)

A statement along the lines of: *the class of constrained allocations expressible by a k-layer flow-aware GNN with hidden width h includes all maps that depend only on k-hop neighborhoods in the weighted adjacency.*

This lets us argue that the GNN class is "just expressive enough" for the problem — bigger classes would overfit, smaller classes would miss structure. If it proves hard, drop it. The paper does not need this to accept.

## 6. What the theory does NOT promise

- No physical / AC-power-flow safety guarantee. Feasibility here means the allocation polytope, not the grid manifold.
- No guarantee on continuity under arbitrary adversaries; only under the distribution-shift metric d_WL.
- No guarantee on cross-grid-type transfer.

These are called out in the paper's Limitations section.

## 7. Working conventions

- Proof sketches live in this file.
- Full proofs go in `paper/theory_appendix.tex` once the statements are finalized.
- Empirical checks of theorem bounds (non-vacuity in particular) live under `tests/theory/` — the bound is also an invariant to test.
