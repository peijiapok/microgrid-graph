# ADR-0001 — Temporal continuity becomes the primary objective and metric

> Status: **ACCEPTED — Path A (all-in v1).** Decided 2026-06-16 by Jia: build flow constraints + differentiable QP projection up front; v1 is the full continuity+topology paper; quality over the 2026 workshop deadline ("we don't have to submit if it's not good"). The §6 scope fork is resolved to **Path A**. Authors: Jia + Claude + GPT-5.5 cross-check. Supersedes the adequacy-primary stance in `docs/05` H1/H2 and the continuity-as-secondary framing in `docs/10`. **Requires revision of `docs/01`, `docs/02`, `docs/05`, `docs/10` to match** — pending Fanchen sync on C1/C3 ownership details.

## 1. Decision
Promote **temporal continuity of critical-load service** from a secondary metric to **(a) the primary evaluation metric and (b) the primary trained objective.** Rationale: adequacy (fraction of timesteps a critical load got its minimum) is gameable by flickering service and does not capture the real resilience value. A hospital powered 80% of the time in an on–off pattern is operationally useless; one powered 80% in a single block is not. If the paper is about resilience, continuity is the meaningful target. Adequacy is retained only as a guard/diagnostic.

## 2. Forced consequence — topology must enter the math (couples to ADR-B)
**Key finding (Claude + GPT-5.5, independent agreement):** with continuity as the headline, the central claim becomes *"zero-shot topology generalization for continuity."* That claim is only non-trivial if an unseen topology actually changes **what can be kept continuously deliverable.** Today the only hard limit is a global scalar budget `Σa_i ≤ P_t`; edges enter only the GNN. Under a global budget, a new *topology* does not change the feasible-continuity frontier any more than a new demand pattern does — so topology generalization is decorative *for continuity*.

Therefore, continuity-primary **requires** topology-coupled feasibility — branch-flow / edge-capacity constraints:
```
Δ_grid(G,s_t) = { a ∈ Δ(G,s_t) : |f_e(a)| ≤ F_e  ∀ e ∈ E }
```
where for a radial feeder `f_e(a)` = total allocation in the subtree below edge `e` (linear in `a`), `F_e` from edge features `X_E`. This makes the feasible set topology-dependent so unseen feeders have genuinely different continuity frontiers.

**Caveat (GPT-5.5):** persistent (Markov) *spatially/edge-correlated* outages can make topology matter via risk forecasting — but that is a *stochastic-forecasting* claim, not a *deliverability* claim. We must NOT sell forecasting as feeder-topology generalization. If we defer flow constraints (see §6), the v1 claim must be scoped to temporal continuity under stochastic outages, not topology-deliverability generalization.

## 3. New primary metric — capacity-normalized weighted continuity
Replace `mean_i max_run(served_i)/T` (gameable) with a windowed, priority-weighted, scarcity-normalized score. For window lengths `L ∈ 𝓛`:
```
C = (1/(|𝓛|·T)) Σ_{L∈𝓛} Σ_t  [ Σ_{i∈C} w_i · 1[i served continuously over (t−L+1..t)] ] / B*_{t,L}
```
- `w_i` = priority-weighted critical demand (MW), not raw node count.
- `B*_{t,L}` = oracle maximum critical weight that *could* be kept continuously served over that realized window under the same budget/outages/constraints (the feasible-continuity frontier). Normalizes for scarcity and makes scores comparable across feeders.
- Reported **per-feeder** (not just aggregate), with **coverage/fairness diagnostics** to catch "favorite-node" starvation.

## 4. New training objective — differentiable windowed-continuity surrogate
Replace the magnitude-smoothness term `λ_s‖a_t−a_{t-1}‖₁` as the *temporal* driver with:
```
soft served:     s_{t,i} = σ( β (a_{t,i} − m_i d_{t,i}) )
soft continuity: q_{t,i,L} = Π_{τ=t−L+1..t} s_{τ,i}     (compute as exp(Σ log s) for stability)
loss term:       −  Σ_{L,t} Σ_{i∈C} w_i q_{t,i,L}  / (normalizer)
```
- Calibrated relaxation of the windowed-continuity AUC; → exact metric as β→∞ except at the threshold. For a true lower bound use a margin: `σ(β(a − threshold − ε))` lower-bounds ε-robust service.
- Keep a small `λ_s` only as anti-chatter regularization, not as the continuity mechanism.
- Critical-shortfall and weighted-served terms remain; their λ's re-tuned. Hard constraints stay in the projection, never the loss.

## 5. Downstream impacts
- **H1/H2 (docs/05, docs/10):** primary decision metric switches from `critical_load_adequacy` to capacity-normalized continuity `C`. `Δ_transfer` re-pinned on `C`. Adequacy becomes a secondary guard (so the policy can't trade away all adequacy for continuity). This metric switch is exactly the "pre-run ADR" the protocol requires — must be frozen before claim-bearing runs.
- **Contributions:** C1 (flow-aware message passing) and C3 (differentiable feasibility projection) move from "nice-to-have" to **central** — the projection now targets the flow-constrained polytope `Δ_grid`, which is a per-step convex QP needing a differentiable opt layer (OptNet / cvxpylayers / implicit gradients). These are Fanchen's contributions, so this *strengthens* his ownership, not adds orphan work.
- **Scenario generator:** outage process must have **persistence** (two-state Markov, already in `docs/10`) tuned so continuity is policy-controllable, not exogenous noise. Add edge/spatial outage correlation as a named stress condition.
- **Evaluation additions (GPT-5.5):** per-feeder normalized scores, fairness/coverage diagnostics, stress tests over outage duration, budget tightness, and critical-node spatial distribution. Watch for "oracle denominator shift" when unseen feeders have different feasible frontiers.
- **Transfer is genuinely harder than adequacy:** continuity is path-dependent (early mistakes cast long shadows); risk of overfitting to training-feeder outage persistence / node ordering / critical-load placement. This is a feature (more meaningful) but demands the diagnostics above.

## 6. Open scope fork (decide at Fanchen sync)
The only remaining decision is **timing**, not direction:
- **Path A (coherent/ambitious):** continuity-primary **+ flow constraints in v1.** The honest, strongest paper. Cost: QP/differentiable-projection engineering (C1+C3) before any claim-bearing run; likely slips the 2026 workshop target.
- **Path B (staged):** continuity-primary in v1 on the **global-budget** setting, scoped honestly as *temporal continuity under persistent stochastic outages* (NOT topology-deliverability generalization); add flow constraints for the 2027 main-conference version. Ships sooner; weaker topology-for-continuity claim in v1; must avoid overselling.
- **Path C (de-risk):** Path B for the main v1 result **+ a one-feeder flow-constrained prototype** to measure QP cost/feasibility before committing the main conference to Path A.

Recommendation was Path C. **DECIDED 2026-06-16: Path A** — Jia chose to build the full continuity + flow-constrained design in v1 and not be bound by the 2026 workshop deadline ("I want the best outcome; we don't have to submit if it's not good"). Flow-constrained QP projection (C3) and flow-aware MP (C1) are now on the v1 critical path.

## 7. Blockers unchanged
None of this runs until the `docs/11` blockers clear: feeder train/test split frozen, `metrics_v1.py` implemented (now must implement `C`, not just adequacy), projection tests, result schema. The metric/objective change above must be reflected when those are built.
