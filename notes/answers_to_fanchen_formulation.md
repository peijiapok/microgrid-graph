# Answers to Fanchen's formulation questions

> Date 2026-06-16. Drafted by Jia (with Claude + GPT-5.5 cross-check). Authoritative sources remain `docs/01_problem_statement.md` and `docs/10_research_formulation.md`; this note answers Fanchen's three questions and flags two that should become ADRs.

Fanchen's three questions:
1. **Continuity:** Temporal graph? How to formulate it formally in the objective?
2. **Microgrid:** What does that mean mathematically?
3. **Formulation:** What are the specific inputs, constraints, and actions?

Short version: **Q3 is fully specified already** (docs 01 §2/§4/§5, doc 10 §3/§8) — answer below is a restatement. **Q1 and Q2 are real open design points** — the current setup has a genuine weakness in each, and we propose decisions.

---

## Q3 (do first — it's the anchor): inputs / constraints / actions

This is the canonical statement; everything else hangs off it.

**Inputs** (per rollout step `t`):
| Symbol | Shape | Meaning |
|---|---|---|
| `X_V` | n×d_v | node features (incl. **priority `w_i`, critical flag `c_i`, min-service `m_i`** as features, not just metadata) |
| `X_E` | \|E\|×d_e | edge features (admittance, thermal capacity, distance-to-source) — *target; currently unused* |
| `d_t` | n | per-node demand |
| `P_t` | scalar | global available power budget |
| `O_t` | ⊆V | outage mask |
| `a_{t-1}` | n | previous allocation (passed explicitly so switching/continuity is in policy state) |

**Hard constraints** (enforced by projection, never by penalty):
```
(1) budget:      sum_i a_{t,i} <= P_t
(2) demand cap:  0 <= a_{t,i} <= d_{t,i}
(3) outage:      a_{t,i} = 0   for i in O_t
```

**Action:** per-node allocation `a_t = P_Δ(π_θ(G, s_t, a_{t-1})) ∈ ℝ_+^n`, where `P_Δ` is the Euclidean projection onto the box-plus-budget polytope `Δ(G,s_t)` (sorted-breakpoint / bisection water-filling).

**What's missing / to make explicit** (GPT-5.5 + our read):
- Confirm `c_i, m_i, w_i` are wired in as **node features**, not only used at eval.
- **Clarify outage semantics**: does `i ∈ O_t` mean *load unavailable* (node present, `a_i=0`) or *node disconnected* (changes reachability/topology)? This matters for Q1 and Q2 — pick one and write it down.
- A **scalar `P_t` ignores where power injects**; with a single substation source this is fine, but it means network bottlenecks are invisible (see Q2).

---

## Q1 (Continuity / "temporal graph") — open lever #1

**Is it a temporal graph?** No, not in the dynamic-graph sense. `G=(V,E,X_V,X_E,Z)` is **fixed within a rollout**; what varies is the *node signal* `(d_t, P_t, O_t, a_t)`. So the correct name is a **discrete-time decision process over a static attributed graph** (a spatio-temporal signal on a fixed graph), not a temporal/evolving graph. The *only* time-varying topology element is the outage mask, which induces a **time-varying active subgraph** `G \ O_t` — that's the one honest sense in which "temporal" applies, and only if outage = disconnection (see Q3 clarification).

**The weakness Fanchen is pointing at:** the objective does **not** optimize continuity. The `λ_s ||a_t − a_{t-1}||_1` term is **magnitude smoothness** (anti-chatter on allocation size) — symmetric, and a load can still flicker served→unserved→served while keeping small magnitude steps. The eval metric `temporal_continuity` (max uninterrupted served-run / T) is **non-differentiable and not in the loss**. Doc 10 §8 explicitly forbids claiming the loss optimizes continuity. So today: continuity is **eval-only**.

**To put continuity into the objective formally** — add an explicit differentiable **interruption surrogate** (distinct from switching):
```
soft-served:    s̃_{t,i} = σ( (a_{t,i} − m_i d_{t,i}) / τ )          # smooth 1[served], temp τ→0
interruption:   I_{t,i} = ReLU( s̃_{t-1,i} − s̃_{t,i} )               # penalizes served→unserved DROPS only
continuity loss: L_cont = Σ_{t≥2} Σ_{i∈C} I_{t,i}
```
- **What it upper-bounds:** as `τ→0`, `Σ_t I_{t,i}` → the **number of service interruptions** for critical node `i` (count of falling edges in the served sequence). Minimizing it lengthens uninterrupted runs ⇒ directly improves the eval metric, unlike `λ_s`.
- It is **asymmetric** (only penalizes losing service, not restoring it) — the key difference from the switching term.
- GPT-5.5's alternative: a softmax-over-time run aggregator `Σ_{i∈C} softmax_t(run-score)/T` approximates max-run more tightly but is more stateful/less clean. Prefer the drop-penalty surrogate for v1.

**Recommendation (matches protocol):** keep **adequacy** as primary (per doc 05), add `L_cont` as an **optional objective term ablated against** `λ_s`-only — so we can show "continuity-in-loss improves the continuity metric without hurting adequacy." Don't promote continuity to primary unless a pre-run ADR says so. → **ADR candidate A.**

---

## Q2 (What is the "microgrid" mathematically?) — open lever #2

**Honest answer:** mathematically, the v1 "microgrid" is **not** a power-flow model. It is:
- an attributed directed graph `G` (radial feeder), plus
- a per-step **feasible allocation polytope** `Δ(G,s_t) = {a≥0, a≤d_t, a_i=0 on O_t, Σa≤P_t}`, plus
- node priorities `(w_i, c_i, m_i)`.

No Kirchhoff/Ohm law, no voltage, no line flows. **Critically: the edge set `E` and edge features `X_E` do NOT enter `Δ` or `J` at all** — the budget is global and caps are per-node. Edges enter *only* through the GNN policy's message passing. GPT-5.5 confirms: "If edges affect only message passing, the graph is an inductive bias, not a physical constraint."

This is exactly why **H1 (is graph structure load-bearing?) is the make-or-break test** of the whole paper — and why a no-graph DeepSets baseline is a credible threat: if the feasible set and objective are topology-independent, the graph can only help insofar as the *optimal allocation pattern* correlates with local neighborhood structure via features.

**Minimal principled way to make topology enter the MATH** (GPT-5.5-aligned): add **branch-flow / edge-capacity constraints** so served load must be *deliverable* through the feeder. For a radial feeder with substation injection, each edge `e` carries the downstream served load; impose thermal limits:
```
Δ_grid(G,s_t) = { a ∈ Δ(G,s_t) : |f_e(a)| ≤ F_e  ∀ e ∈ E }
```
where `f_e(a)` = sum of allocations in the subtree below edge `e` (radial ⇒ flows are a linear function of `a`), and `F_e` from `X_E`. Now the **feasible set is topology-dependent** and "microgrid" has real mathematical content; the scalar-`P_t` bottleneck blindness (Q3) is also fixed.
- **Cost:** the projection is no longer the simple capped-simplex water-filling — it becomes a **convex network-flow QP per step**, requiring a differentiable optimization layer (OptNet/cvxpylayers) or implicit gradients. Slower, more complex, harder to keep exact + differentiable.
- **Scope call:** this is a **v1.5/v2 expansion**, not v1. For v1, keep the box-plus-budget polytope but **state plainly in the paper that topology enters via the policy, not the constraints** (doc 10 §11 "Brutal Honesty" already commits to this honesty). Adding flow constraints is the cleanest single upgrade that would make the microgrid framing physically real and is a natural home for C1 (flow-aware MP) + C3 (differentiable projection). → **ADR candidate B.**

---

## Net: what changes vs. the docs
- Q3: no change — point Fanchen to docs 01/10; just wire `c_i,m_i,w_i` as features and pin outage semantics.
- Q1: **ADR-A** — add optional differentiable continuity surrogate `L_cont`, ablate vs `λ_s`; adequacy stays primary.
- Q2: **ADR-B** — decide whether v1 stays "topology via policy only" (recommended) or adds branch-flow capacity constraints (v1.5/v2); either way state it explicitly. This is the deepest question and the one most worth a sync with Fanchen.
