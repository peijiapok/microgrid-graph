> **SUPERSEDED** by `notes/work_order_fanchen_2026-06-16.md` — Jia is lead and sets direction, so the collaboration is directive (work order with acceptance gates), not "let's discuss." Kept for the rationale/context narrative only.

# Draft message to Fanchen — 2026-06-16

> Draft for Jia to send (Slack/email/GitHub issue). Tone: peer-to-peer; Fanchen leads ML/theory. Two attachments: `notes/answers_to_fanchen_formulation.md` and `notes/adr/0001-continuity-primary.md`.

---

Hi Fanchen,

Your three questions (continuity / "is it a temporal graph", what the microgrid is mathematically, and the inputs/constraints/actions) turned out to be the most useful thing anyone's said about this project — they pushed us to a decision that reframes the whole paper for the better. Two short notes are attached; here's the summary.

**Short answers to your questions first.**
- **Inputs/constraints/actions (Q3)** were already pinned down — it's in `docs/01 §2/§4/§5` and `docs/10 §3/§8`. Canonical: inputs `{X_V, X_E, d_t, P_t, O_t, a_{t-1}}`; hard constraints `{budget Σaᵢ≤P_t, demand-cap 0≤aᵢ≤dᵢ, outage aᵢ=0}`; action = projected per-node allocation. Two cleanups: wire `c_i,m_i,w_i` in as node features (not just eval metadata), and pin outage semantics (load-off vs node-disconnect).
- **"Temporal graph?" (Q1)** — no, not in the dynamic-graph sense. `G` is fixed within a rollout; only the node signal `(d_t,P_t,O_t,a_t)` moves over time. The one honest "temporal topology" element is the outage mask inducing a time-varying active subgraph.
- **"What is the microgrid, mathematically?" (Q2)** — and this is the one that changed everything: in the current setup, **the wiring `E`/`X_E` does not enter the feasible set or the objective at all.** It's a global budget + per-node caps; edges only feed the GNN. Mathematically it's "split a budget among priority nodes," and the graph is, as you'd put it, decorative for the actual problem.

**The decision this forced.** We're promoting **temporal continuity of critical-load service to the primary trained objective and primary metric** (adequacy was gameable by flickering and doesn't capture the resilience value). The key consequence — which I checked independently and it held up — is that **continuity-as-headline makes "topology generalization" trivial unless topology actually constrains what's deliverable.** Under a global budget, an unseen topology doesn't change the feasible-continuity frontier any more than a new demand pattern does. So continuity-primary *requires* putting topology into the math: **branch-flow / edge-capacity constraints**

  `Δ_grid(G,s_t) = { a ∈ Δ(G,s_t) : |f_e(a)| ≤ F_e ∀ e }`,  `f_e(a)` = subtree allocation below edge `e` (linear in `a` for a radial feeder), `F_e` from `X_E`.

Jia's call is to go **all-in on this in v1** — full continuity + flow-constrained design — and not be hostage to the 2026 workshop date. Quality over deadline; we ship when it's genuinely good.

**What this means for your contributions — it strengthens them, doesn't add orphan work:**
- **C3 (differentiable projection) becomes central, and more interesting.** The projection target is no longer the cheap box+budget water-filling — it's the flow-constrained polytope `Δ_grid`, i.e. a per-step convex QP. Open question I'd love your read on: the flow caps are subtree-sum constraints, which form a *laminar* family on a radial feeder — is there an efficient specialized projection (tree-structured, generalizing water-filling) instead of a generic OptNet/cvxpylayers QP? If so, that's arguably a contribution in its own right.
- **C1 (flow-aware MP)** now has teeth: edge capacities `F_e` appear in the constraints, not just as features, so a capacity-aware operator is doing real work.
- **C4 (CVaR-minimax)** gets better-motivated: continuity is path-dependent and heavy-tailed, so worst-case (CVaR) continuity is the natural robust objective.
- **C6 (d_struct)** sharpens: the structural features that should predict continuity transfer are exactly the deliverability ones — bottleneck edges, depth-from-source, capacity margins. That's a cleaner atlas than generic graph-mining moments.

**Metric & objective specifics** (details + rationale in the ADR):
- Primary metric = **capacity-normalized, priority-weighted windowed continuity** (oracle feasible-frontier denominator), *not* mean max-run-length — max-run is gameable by serving one easy node forever.
- Training surrogate = soft-served `s_{t,i}=σ(β(a_{t,i}−m_i d_{t,i}))`, windowed product `q=Πs` (log-sum form), priority-weighted — a calibrated relaxation of the windowed-continuity metric. The `‖a_t−a_{t-1}‖₁` term drops to a small anti-chatter regularizer.

**What I'd like from you at the next sync:**
1. Sanity-check the continuity↔topology coupling argument — if you think global-budget + persistent outages alone *can* carry a topology-for-continuity claim without flow constraints, that's worth a real argument before we commit the engineering.
2. Your design instinct on the flow-constrained differentiable projection (generic QP vs. exploiting radial/laminar structure) — this is the long pole.
3. Whether the outage process needs spatial/edge correlation (not just per-node Markov) for topology to genuinely matter to continuity.

Two notes attached: the per-question answers, and ADR-0001 with the full design and downstream impacts. Heads-up that this means revising `docs/01/02/05/10`, which I'll do after we align so I'm not rewriting them twice.

This is a better paper than what we had. Thanks for asking the right questions.

— Jia
