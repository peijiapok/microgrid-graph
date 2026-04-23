# Research Proposal

**Title.** Topology-Generalizing Graph Control for Catastrophic Grid Events.
**Authors.** Peijia Pok (lead; resilience + empirics). Collaborator (graph learning + mining + theory, TBA).
**Target venue.** NeurIPS Workshop 2026 → NeurIPS 2027 Main Conference → iterate until the paper is the best we can make.
**Repository.** https://github.com/peijiapok/microgrid-graph
**Proposal date.** 2026-04-23. Kickoff 2026-04-22.

---

## TL;DR

When a distribution grid is hit by a catastrophic event — a hurricane, a cascading outage, a wildfire-induced generation loss, an extreme-heat overload — supply collapses below demand and a controller must allocate the scarce power that remains so that critical infrastructure (hospitals, data centers, communications, cold-chain pharmacy) stays online. Operators cannot retrain a model on the specific feeder that just failed; the policy must generalize zero-shot to feeder topologies never seen during training, with formal guarantees.

We propose:

1. A **physics-aware graph policy** that generalizes zero-shot across distribution feeders.
2. A **graph-mining-native topology-similarity metric `d_struct`** that predicts how well the policy will transfer — *before training*.
3. A **bound on the worst-case continuity gap** in terms of GNN expressivity, structural coverage of the training topology family, and adversarial budget.

The combined contribution turns resilient grid control from a per-feeder engineering task into a learning problem with *a priori* transfer estimates.

---

## 1. Problem statement

### 1.1 The real-world setting: catastrophic grid events

A distribution feeder serves dozens to hundreds of load nodes with varying priority. Under nominal conditions, supply meets demand. Under catastrophic conditions, supply may fall to 30–50% of demand and specific substations may be offline entirely. A controller must, at each time step:

- respect a hard global power budget `Σ_i a_{t,i} ≤ P_t`,
- respect per-node demand caps `0 ≤ a_{t,i} ≤ d_{t,i}`,
- respect outages `a_{t,i} = 0` for nodes in the outage set `O_t`,
- preserve service continuity on critical nodes (avoid flipping hospitals off and back on),
- minimize unserved critical demand,
- avoid unnecessary switching.

Operators make this decision under time pressure, often on feeders whose exact topology their playbooks never anticipated, and the cost of error is measured in critical-infrastructure downtime — not in regret.

### 1.2 The ML problem

Cast this as a graph-structured sequential decision problem with policy `π_θ : (G, s_t, a_{t-1}) → a_t`, parameterized by a GNN over the feeder graph `G = (V, E, X, W)`. The research question, stated sharply:

> *Can a policy trained on a topology family `G_train` produce zero-shot feasible, constraint-respecting, continuity-preserving decisions on an unseen topology family `G_ood` — and can we bound the worst-case continuity gap as a function of GNN expressivity, training-topology coverage, and adversary budget?*

Grid resilience is the application. Topology-generalizing graph control under hard constraints is the paper.

---

## 2. Why now

- **Benchmark landscape.** Pandapower + SimBench now provide enough topological diversity (urban/rural, small/large, radial/meshed) to form a principled train/OOD split without needing private utility data.
- **Theory maturity.** Graph-learning theory on size-generalization and OOD transfer has matured to the point where non-vacuous bounds are writable.
- **Computational cheapness of graph-mining primitives.** Motifs, spectral signatures, community detection — all computable on any distribution feeder in under a second on CPU. This makes them viable as the basis for a computable transfer-prediction metric.

---

## 3. Research questions

- **RQ1.** Does a single graph policy, trained on a curated family of distribution feeders with a continuity-aware objective, achieve acceptable zero-shot performance on unseen feeder topologies?
- **RQ2.** Can we define a computable topology-similarity metric `d_struct(G, G')` that, computed *before* training, predicts the observed transfer gap?
- **RQ3.** What is a non-vacuous upper bound on the worst-case continuity gap, in terms of `d_struct(D_train, D_ood)`, the GNN class's expressivity, and adversarial budget?
- **RQ4.** Under the CVaR-minimax objective, can we guarantee worst-case critical continuity in deployment up to a bounded penalty term?

---

## 4. Prior art and the gaps this fills

Six buckets of related work, detailed in `docs/07_related_work_map.md`:

1. **Resilient distribution-grid operation** — Gao 2016, Aghajan 2024, Lorca 2016.
2. **Graph / multi-agent RL for grid control** — Jacob 2024, Grid2Op literature.
3. **Flow-aware and physics-informed GNNs for power systems.**
4. **OOD / topology generalization in GNNs** — WL framework (Xu 2019), size-generalization literature.
5. **Differentiable optimization layers** — OptNet, CVXPY Layers.
6. **CVaR / distributionally robust learning** — Rockafellar & Uryasev 2000, Namkoong & Duchi.

**The three gaps this proposal fills:**

- Existing resilient-control papers train and test on the *same* feeder. None evaluate zero-shot topology transfer.
- Existing graph-generalization work uses abstract distances (e.g., `d_WL`) that are rarely computable on realistic graphs. No bound is numerically evaluable on a distribution-grid benchmark.
- Existing learned grid controllers use plain symmetric adjacency. None use directed, weighted flow structure as a first-class graph object.

---

## 5. Proposed methodology — six contributions

Each contribution addresses a specific reviewer-killer weakness of the current pipeline (see `docs/02_research_goal.md` for the one-line pitch of each).

### C1. Flow-aware message passing

Replace the plain symmetric GraphSAGE operator with an operator that consumes edge features (admittance, thermal capacity) and respects the directed nature of power flow from source to load. Adapted from the MPNN / directional-message-passing literature.

### C2. Topology-generalizing policy (with structural atlas)

Train one policy on `G_train = {case33bw, rural1, rural2}`. Evaluate zero-shot on `G_ood = {lv_urban, ieee_34, ieee_123}`. Report the transfer gap per metric. In parallel, build a **structural feeder atlas** (`configs/feeder_atlas.yaml`) cataloguing every feeder by degree-distribution moments, spectral gap, diameter, clustering coefficient, graphlet counts (k≤4), and community signature.

### C3. Differentiable feasibility projection

Replace the iterative water-filling projection with a closed-form (or fixed-iteration) projection onto the allocation polytope `{a : a ≥ 0, a ≤ d, Σa ≤ P}`, with a formal feasibility certificate and end-to-end differentiability.

### C4. CVaR-minimax objective with regret bound

Replace lexicographic checkpoint selection with a proper CVaR_α objective. Prove a regret bound linking training worst-case continuity to deployment worst-case continuity.

### C5. Graph-baseline zoo

Run GCN, GAT, GIN, EGNN, and Graph Transformer through the same loss + projection + training loop. Isolate the contribution of the flow-aware operator from operator complexity generally.

### C6. Topology-similarity metric that predicts transfer — signature contribution

Define `d_struct(G, G')` from graph-mining primitives (graphlets, spectral signatures, degree-distribution distance, or a learned composite). Show empirically that `d_struct` between the training-topology distribution and each test feeder predicts the observed transfer gap. Use `d_struct` to replace the abstract `d_WL` in the transfer-bound theorem.

If C6 works, the paper's headline is: *"we predict graph generalization from structure alone"* — a result that elevates the paper from competent to signature.

---

## 6. Theoretical targets

Three theorems, detailed in `docs/06_theory_sketch.md`:

- **Theorem 1 (transfer bound).** Under a GNN-Lipschitz assumption in `d_struct`, the expected transfer gap is bounded by training risk + `L_Π · W_1(D_train, D_ood; d_struct)` + sample-size slack. Every term is numerically evaluable on our atlas.
- **Theorem 2 (feasibility).** The projection layer's output satisfies budget, non-negativity, and demand caps *exactly* — a closed-form guarantee, not a stochastic bound.
- **Theorem 3 (CVaR regret).** Under the minimax CVaR_α objective, the deployment worst-case continuity is within `ε` of the minimax-optimal policy in the hypothesis class, with an explicit sample-complexity bound.

The non-vacuity of Theorem 1 on our empirical setup is a hard invariant: if the numerical bound exceeds the trivial upper bound of 1 on critical continuity, the theorem is rhetorical and is reworked or cut (decision gate G-D).

---

## 7. Experimental protocol

Detailed in `docs/05_experimental_protocol.md`. Summary:

- **Feeders.** Six across train and OOD: case33bw, rural1, rural2 (train); lv_urban, ieee_34, ieee_123 (OOD). Spans radial-small, rural-LV, urban-LV, radial-medium, radial-large.
- **Metrics.** Four existing core metrics (weighted served energy, unserved critical demand, switching count, critical continuity ratio) + transfer gap + CVaR_0.05 of continuity + feasibility-violation count + `d_struct` + R²(`d_struct`, transfer_gap).
- **Ablations.** Graph-necessity (MLP, scrambled-edge, identity adjacency). Objective-component (no continuity, no imitation, no switching, no CVaR). Projection (softmax, iterative water-filler, closed-form).
- **Baselines.** Rule, classical optimization, and five learned GNNs (GCN, GAT, GIN, EGNN, Graph Transformer) + no-graph MLP.
- **Seeds.** Three random seeds minimum for any reported claim.

---

## 8. Timeline

Full plan in `docs/08_timeline.md`. Six decision gates:

| Gate | Target date | Question |
|---|---|---|
| G-A | 2026-05-end | Is the graph load-bearing (MLP ablation fails)? |
| G-B | 2026-06-end | Does cross-topology transfer work at all? |
| G-C | 2026-07-end | Does `d_struct` predict transfer with R² > 0.5 on ≥1 metric? Workshop draft coherent? |
| G-D | 2026-12-end | Is Theorem 1 non-vacuous on empirical data? |
| G-E | 2027-03-end | Is the main-conference draft at the quality bar? |
| G-F | 2027-05 | Submit. |

Posture: "work until this is the best paper we can make" — not "ship by the next deadline." Deadlines are gates, not panic points.

---

## 9. Team and division of labor

- **Peijia Pok (lead; resilience + empirics).** Problem definition, resilience metrics, SimBench/pandapower harness, scenario generation, physics evaluation, C5 baseline implementation, final writing integration, paper submission.
- **Collaborator (graph learning + mining + theory).** Graph operator design (C1), topology-split and structural-atlas (C2), differentiable-projection theory (C3), CVaR-minimax + regret bound (C4), GNN-baseline specs (C5), topology-similarity metric (C6).

Full ownership table in `docs/04_division_of_labor.md`. Authorship will be settled in writing before first submission via `notes/adr/0001-authorship.md`.

---

## 10. Success criteria (hierarchical)

- **Viability** (end May 2026). C1 and C2 done-when clauses satisfied on four in-family + one OOD feeder.
- **Workshop-ready** (end July 2026). Viability + C5 + `d_struct` v0 + draft theory for C3.
- **Main-conference-ready** (March 2027). All six contributions complete; Theorems 1–3 proved or empirically justified; ablations; writing.
- **Best-paper posture.** C6 non-vacuous in Theorem 1 with R² > 0.5 on at least two metrics; atlas covers IEEE 123 plus two additional OOD families.

---

## 11. Non-goals

Explicit, contractual:

- **Not** full physics-constrained learning (no power-flow solver in the training loop).
- **Not** real-time deployment claims.
- **Not** cross-grid-type transfer (transmission ↔ distribution).
- **Not** universal dominance over rule/optimization baselines on every metric on every feeder.
- **Not** safety in the formal control-theoretic sense beyond the stated feasibility theorem.

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Transfer is uniformly catastrophic across feeders | Medium | Re-scope to C1+C3+C4 narrative; C2 becomes a reported negative result |
| Theorem 1 proves intractable for our GNN class | Low-medium | Drop to a weaker empirical form; keep Theorems 2 and 3 |
| `d_struct` fails to predict transfer (R² < 0.5) | Medium | C6 becomes a reported negative result; paper's headline shifts to C1+C4 |
| Compute bottleneck on large OOD feeders | Medium | Start OOD runs in 2026-06, not later; use CPU-friendly architectures |
| Collaborator bandwidth unpredictable | Always | Monthly check on milestones; re-scope rather than silent slippage |

---

## 13. Resources required

- **Compute.** ~100 GPU-hours for workshop target; ~300 for main conference. Scoping in `docs/05 §5`.
- **Data.** SimBench and pandapower are public; no private utility data needed.
- **Infrastructure.** GitHub, shared cloud compute (provider TBD), standard ML libraries (torch, pandapower, simbench, torch-geometric for C5).

---

## 14. Broader impact

**Positive.** Learning-based resilient control could support priority service to hospitals, data centers, and communications during catastrophic grid events — domains where downtime translates to measurable human harm. The topology-similarity metric is reusable outside grid control: any graph-structured sequential decision problem with OOD topologies (supply chains, transportation networks, communication networks) could apply the same framework.

**Negative.** Deploying learned control in critical infrastructure is inherently risky. Nominal average-case success can hide tail collapse. We therefore explicitly frame near-term use as **decision support** alongside rule/optimization references, not as a replacement. Transparent reporting of incomplete benchmarks and explicit stress auditing are contributions, not afterthoughts.

---

## 15. Pointers

- Current-state snapshot: `PROGRESS.md`
- Problem statement: `docs/01_problem_statement.md`
- Research goal: `docs/02_research_goal.md`
- Collaborator brief: `docs/03_collaborator_brief.md`
- Division of labor: `docs/04_division_of_labor.md`
- Experimental protocol: `docs/05_experimental_protocol.md`
- Theory sketch: `docs/06_theory_sketch.md`
- Related work map: `docs/07_related_work_map.md`
- Timeline: `docs/08_timeline.md`
- Reproducibility & environment: `docs/09_repro_and_env.md`
