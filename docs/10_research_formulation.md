# 10 — Research Formulation

> **2026-06-16 reframing (ADR-0001, ADR-0002).** Primary objective and metric = **temporal continuity** of critical service (was adequacy). Feasible set is **flow-constrained** `Δ_grid` (radial branch-flow capacities), so topology enters the constraints, not only the policy. Sections 3, 6, 8, 9, 11, 12 updated accordingly.

## 1. Setting

The defensible setting is **constrained scarce-resource allocation on graphs
under structural distribution shift**, where feasibility includes **radial
branch-flow capacity limits** so that graph topology constrains what is
deliverable.

Version terms: v1 is the current workshop scope with the frozen six-feeder
split and three OOD feeders; v2 is the expanded main-conference scope after the
protocol's structural-association pair-count gate is met. The current working
gate is 20 structural pairs, subject to the protocol ADR; one pair means one
frozen training-feeder distribution compared with one held-out feeder.

A distribution feeder is represented as an attributed graph. At each time step,
the controller observes node demand, available global power, outage masks,
previous allocation, and graph structure. It outputs a feasible per-node power
allocation. The allocation must respect hard budget, demand-cap, non-negativity,
and outage constraints. The controller is trained on one finite family of
feeders and evaluated zero-shot on held-out feeder topologies.

Catastrophic grid events are the motivating regime: they justify scarce power,
outages, critical loads, and asymmetric service priorities. They are not the
formal claim. Unless the simulator includes validated hurricane, wildfire,
cascading-failure, or adversarial-attack models, the formal sections should use
the narrower phrase **scarce-budget allocation under stochastic outages**.

This is still not full physics-constrained grid control: the training loop does
not solve AC or DC power flow, so the method is **not** "physics-aware." It **is**
**flow-constrained** — the feasible set enforces linear radial branch-flow
*capacity* limits |f_e(a)| ≤ F_e (§3). Accurate phrases: **flow-constrained
graph policy** (feasibility) and **flow-aware directed message-passing policy**
(operator, when edge admittance/capacity/direction are consumed). Never
"physics-aware."

## 2. Research Problem

Given a family of attributed distribution-feeder graphs and a per-step
constrained-allocation problem with hard feasibility and asymmetric
critical-load value, train a single graph policy on feeders sampled from a
training topology distribution. The policy must, without retraining or
fine-tuning, produce feasible per-node allocations on held-out feeder
topologies outside the training family. The v1 research question is whether
graph structure is necessary for this transfer problem. The
structural-distance association question is deferred until enough train-test
topology pairs exist.

Short version:

> Can one feasible graph policy trained on some feeder topologies allocate scarce
> power on unseen feeder topologies, and is graph structure necessary for that
> transfer?

## 3. Definitions

This section pins down the mathematical objects before making hypotheses.

### Graph Sample

A rollout sample is:

```text
(G, xi) ~ D
G = (V, E, X_V, X_E, Z)
xi = (d_{1:T}, P_{1:T}, O_{1:T})
```

- `V`: feeder nodes, with `n = |V|` variable across feeders.
- `E`: directed feeder edges.
- `X_V in R^{n x d_v}`: node features.
- `X_E in R^{|E| x d_e}`: edge features. If edge features are absent, the
  model is not flow-feature-aware.
- `Z`: static node attributes, including priority `w_i`, critical flag
  `c_i`, and minimum service fraction `m_i`.
- `d_t in R_+^n`: demand.
- `P_t in R_+`: available global power budget.
- `O_t subset V`: outage mask.
- `s_t = (d_t, P_t, O_t)`: observed step state. `G`, including `X_V` and
  `X_E`, is fixed within a rollout. The previous allocation `a_{t-1}` is passed
  separately to make switching dependence explicit.
- `T`: rollout horizon. Claim-bearing temporal-continuity comparisons require
  the same `T` for train and test.
- `S_G`: feeder-conditioned scenario distribution for `xi`.
- `D_train`, `D_ood`: finite distributions over training and held-out feeder
  graphs, respectively.

`m_i in [0, 1]` is defined for every node. It is used for critical-load service
metrics; non-critical `m_i` values are retained for schema consistency but do
not enter v1 claim-bearing metrics. `w_i` is a frozen node service weight; v1
defaults to positive weights for all served-energy accounting, with critical
nodes additionally receiving the shortfall penalty below.

### Feasible Set And Projection

At each time step, the feasible allocation polytope is the **flow-constrained**
set `Delta_grid` (ADR-0001):

```text
Delta_grid(G, s_t) = {
  a in R_+^n :
    0 <= a_i <= d_{t,i} for all i,            # box / demand cap
    a_i = 0 for all i in O_t,                  # outage
    sum_i a_i <= P_t,                          # global budget
    | f_e(a) | <= F_e   for every tree edge e  # branch-flow capacity (NEW)
}
f_e(a) = sum_{j in subtree(e)} a_j            # radial subtree sum
```

`F_e` is the edge thermal capacity (line: sqrt(3)·Vn·Imax; trafo: sn_mva),
exposed by `src/sg_resilience/topology.py`. The branch-flow term is what makes
the feasible set **topology-dependent**: an unseen feeder changes which critical
loads can be simultaneously (and continuously) delivered. Without it, only a
global budget binds and topology generalization is decorative for continuity.

The model may produce raw scores `z_t in R^n`, but the action is:

```text
z_t = pi_theta(G, s_t, a_{t-1})
a_t = P_Delta(z_t)
P_Delta(z_t) = argmin_{a in Delta_grid(G, s_t)} ||a - z_t||_2^2
```

Feasibility is an invariant only if the projection implementation is correct.
Projection-enforced feasibility may be claimed only after proving or testing
that the implementation lands in `Delta(G, s_t)` for every valid input within
the stated numerical tolerance.

Implementation target: the Euclidean projection onto `Delta_grid`. With branch-flow
capacities this is a convex QP, not a 1-D water-fill. C3 ships it as a
differentiable QP layer (cvxpylayers/OptNet) first, with a near-linear laminar
tree-DP projection as a stretch contribution (`notes/work_order_fanchen_2026-06-16.md`);
the reference solver is `src/sg_resilience/flow_projection.py`. The
**box-plus-budget** water-fill below remains valid only as the special case when
no branch-flow constraint binds (e.g. case33bw with non-physical line ratings),
and is the inner step of the tree-DP:

1. Set `u_i = d_{t,i}` for non-outage nodes and `u_i = 0` for outage nodes.
2. Define `box_i(x) = min(max(x, 0), u_i)` and
   `f(tau) = sum_i box_i(z_i - tau) - P_t`.
3. If `f(0) <= 0`, set `tau = 0` and skip root-finding; otherwise find the
   smallest `tau >= 0` with `f(tau) <= 0`.
4. Return `a_i = box_i(z_i - tau)`.

```text
a_i = min(max(z_i - tau, 0), u_i)
```

The budget is an inequality, so the projection never increases allocations just
to spend unused budget. `tau` is a uniform price subtracted from every raw
score; raise it only when box-clipping alone still exceeds the budget. The
formal proof that this operator is the Euclidean projection onto the
box-plus-budget polytope belongs in the projection ADR/test suite, not in this
formulation. Feasibility violations are
measured with max absolute constraint violation tolerance `epsilon_feas = 1e-6`
after dividing every feeder's power quantities by that feeder's total nominal
demand, defined as `sum_i nominal_load_i` from the feeder case before scenario
scaling. The root-finder tolerance must be at least as strict as
`epsilon_feas / 10` in normalized power units.

### Risk

For a rollout return `J`, define:

```text
L(pi; G, xi) = -J(pi; G, xi)
R(pi; D) = E_{G ~ D} E_{xi ~ S_G}[ L(pi; G, xi) ]
R(pi; D_train) = E_{G ~ D_train} E_{xi ~ S_G}[ L(pi; G, xi) ]
R(pi; D_ood) = E_{G ~ D_ood} E_{xi ~ S_G}[ L(pi; G, xi) ]
```

Training minimizes rollout risk over the training rollout distribution.
Evaluation reports both train and OOD rollout risk plus task metrics.

### Workload Processes

The scenario distribution `S_G` must be specified separately from graph
structure. To test topology transfer, the demand, budget, and outage
generators must be invariant across train and OOD feeders after normalization
by total demand and node features:

```text
xi ~ S_G = S(demand_scale, budget_ratio, outage_process | G)
```

Default operational scheme:

```text
d_{t,i} = base_profile_t * demand_scale_i
demand_scale_i ~ q_scale(. | X_{V,i})
P_t = rho_t * sum_i d_{t,i}
rho_t ~ q_budget
1[i in O_t] follows a two-state Markov outage process with outage probability
p_out and persistence p_stay
```

Here the conditional laws `q_scale(. | X_V)`, `q_budget`, and `p_out` are
shared across train and OOD feeders. The demand scale may depend on node
features, so marginal demand distributions can still differ when node-feature
distributions differ; those marginal differences are audited. The budget
denominator is total realized demand at time `t`. Topology-correlated outages
may be added only as a separately named stress-test condition.

If demand or outage processes shift together with topology, topology transfer
is confounded with workload shift and any structural-association claim must be
weakened.

Workload-invariance audit applies in v1 before claiming topology transfer and
in v2 before claiming structural association. The numeric gates are specified
in `docs/05_experimental_protocol.md` Section 9.

- report per-feeder distributions of normalized demand scale, budget ratio
  `rho_t`, and outage rate;
- report realized normalized demand scale after applying `q_scale(. | X_V)`.
  This is the nontrivial audit target because it can shift when `X_V` differs
  across feeders even if the conditional law is shared. KS checks on `rho_t`
  mostly verify the sampler and are secondary diagnostics, while realized
  demand-scale and outage-rate checks are the gating workload audits;
- run two-sample KS checks on realized demand-scale and `rho_t` distributions,
  plus an absolute outage-rate difference check, before reporting structural
  association;
- if normalized workloads are distinguishable, label the result as mixed
  topology/workload shift rather than structural transfer.

## 4. Core Observation

The key observation is structural, not rhetorical:

> Under a fixed graph-policy architecture and shared constrained-allocation
> objective, zero-shot allocation quality should degrade when the k-hop
> neighborhoods seen at test time differ from those seen in training. The local
> mechanism is degree, edge-feature, and priority/demand-feature mismatch inside
> message-passing neighborhoods. Global size or path-length shifts matter only
> insofar as they change the distribution of those local neighborhoods or the
> scarce-budget allocation patterns induced by them.

This is a falsifiable mechanism. It may be false. If the size-generalizing
no-graph baseline or degree-preserving rewire baseline transfers as well as the
graph model, graph structure is not carrying the v1 claim. Structural-distance
association is a v2 question and must not be used as a v1 contribution.

## 5. Gap

### 5.1 Application Gap

Operationally, many grid-control methods are tuned, optimized, or trained for a
specific feeder. In crisis conditions, an operator may not have a validated
model, tuned policy, or retraining window for the exact feeder and damage
pattern that appears.

This motivates one-policy-across-feeders control. But it is not enough for a
machine-learning paper by itself. A reviewer can reasonably ask: why not train a
separate controller offline for every feeder of interest? The application gap is
therefore motivation, not the core novelty.

### 5.2 ML / Graph-Learning Gap

The core gap is:

1. Graph OOD evaluation suites usually study classification or regression, not
   constrained continuous allocation with projection-enforced feasibility.
2. Power-system learning papers often evaluate on scenarios from the same
   topology rather than holding out entire feeder families.
3. Existing graph-transfer theory often uses distances that are abstract or not
   directly evaluated on the actual graphs used in experiments.
4. There is no clean evaluation story tying together constrained-output graph
   policies, feeder-topology shift, projection-enforced allocation feasibility,
   and graph-necessity tests under held-out feeder families.

The contribution should be framed as that combination. Do not frame it as a new
AI problem class unless the evaluation suite and evidence are strong enough to
support that claim.

## 6. Hypotheses

All thresholds must be chosen before the corresponding run and recorded in
`docs/05_experimental_protocol.md` or a dated ADR before claim-bearing
experiments begin. The single source of truth is
`docs/05_experimental_protocol.md` Section 2.5: default service-quality
thresholds are measured in absolute metric points on the native 0-1 scale
unless explicitly defined as normalized risk gaps. The v2 pair-count gate is
defined in that protocol file.

### H1. Graph Structure Is Load-Bearing

A graph policy achieves better critical-load service than no-graph and
wrong-graph baselines under topology shift.

Do not accept H1 if any of the following hold:

- `C(graph) - C(primary_no_graph) < Delta_graph` on held-out feeders;
- `C(graph) - C(degree_preserving_rewire) < Delta_rewire` on held-out feeders;

where `C` is the primary capacity-normalized continuity metric (ADR-0001).

Accept H1 as a v1 workshop result only if both OOD continuity point-estimate gaps
clear their pre-registered thresholds. Confidence bounds are reported by the
protocol file; stronger labels require those bounds to clear the thresholds.

Primary H1 decision metric is **capacity-normalized continuity `C`** (changed
from `critical_load_adequacy` by ADR-0001, the required dated pre-run ADR).
Adequacy is reported as a secondary guard and cannot by itself rescue a failure
on `C`.

Baseline discipline: a flat padded MLP is only a diagnostic. The main no-graph
baseline must be a size-generalizing set model, such as DeepSets or a
permutation-invariant transformer, with the same node features and the same
projection layer. Otherwise H1 tests input-shape convenience, not graph
inductive bias.

Feature-ablation discipline: include a feature-permutation graph baseline that
keeps topology fixed but permutes node features within each feeder. If this
matches the graph policy, the result is feature-driven rather than
graph-structure-driven and H1 must be weakened.

Wrong-graph baseline discipline: the primary wrong-graph baseline is a
degree-preserving random rewire of each feeder. Draw at least three rewires per
feeder before training, evaluate each across the same seeds, and report
rewire-sample variance. Each rewire preserves node features, node count, edge
count, and degree sequence while destroying actual connectivity. Identity
adjacency and same-density Erdos-Renyi rewires are secondary diagnostics, not
the primary falsifier.

### H2. Zero-Shot Transfer Is Nontrivial

A single policy trained on `D_train` retains useful critical-load service on
`D_ood` without retraining.

H2 decision rule:

- the OOD degradation relative to in-family performance exceeds the
  pre-registered `Delta_transfer` threshold for **continuity `C`**:
  reject H2. `C` is the deciding H2 metric (per ADR-0001); adequacy is a
  secondary guard that may trigger a limitation note but does not decide H2.
  The degradation uses the continuity gap:

  ```text
  gap_C(pi) = E_train[C] - E_ood[C]
  ```

  Positive `gap_C` means OOD continuity is worse;
- if the observed point-estimate degradation is below `Delta_transfer`, v1
  treats H2 as a workshop viability result. Confidence intervals are reported
  for uncertainty, but H2 viability in v1 is a point-estimate rule; stronger
  retention claims require the protocol's confidence-bound label.
- if the observed point-estimate degradation is equal to `Delta_transfer`, H2 is
  reported as borderline viability, not a clean pass.

Priority allocation and continuity-aware LP/QP baselines are reported as
context for practical usefulness, but they are not the H2 transfer-retention
decision rule. They belong to H1 or main-result comparisons.

### H3. Structural Distance Is Associated With Transfer Gap (V2 Only)

A computable feeder-structure distance is positively associated with observed
transfer degradation. This is not claim-bearing in v1.

No v1 decision is made on H3. After the v2 pair-count gate is reached, the v2
protocol must define the association test, including the exact Spearman /
permutation decision rule, before any transfer results are inspected. Changing
the distance form after seeing transfer results is a protocol violation, not an
empirical rejection.

Constraint: three OOD feeders are not enough for a serious structural
association claim. Until the v2 pair-count gate is reached, H3 is not testable
and no `d_struct` or beta-slope result is reported in v1.

For H3, one pair means one frozen training-feeder distribution compared against
one held-out feeder `G_j`; pair count is
`|frozen training distributions| x |held-out feeders|`. V1 has exactly one
frozen training distribution and three OOD feeders, so it provides only three
H3 pairs. Claim-bearing H3 requires expanded splits or multiple frozen training
distributions that reach the v2 pair-count gate.

### Feasibility Invariant

The policy output should satisfy allocation constraints at every step because a
projection maps raw scores into the feasible set. This is an implementation
invariant under a correct projection implementation. Paper-facing language
should say "projection-enforced feasibility" until the projection proof and
adversarial tests pass.

The implementation fails the invariant if:

- any reported rollout has budget, demand-cap, non-negativity, or outage-mask
  violations;
- feasibility depends on a soft penalty rather than the projection;
- the projection cannot be tested independently on adversarial small cases.

## 7. Methodological Commitments

### 7.1 Temporal Continuity Is Actually Temporal

If the paper uses the word continuity, the metric must measure uninterrupted
service over time.

Do not call a metric continuity if:

- the only metric is `mean_{t,i} 1[a_{t,i} > 0]`;
- a load that flickers on and off can score the same as a load served in one
  uninterrupted run;
- the paper calls adequacy "continuity."

### 7.2 Evaluation Suite, Not Benchmark

Until the repository has reproducible loaders, baseline scripts, result schemas,
and table generators, call the artifact an **evaluation suite**, not a
benchmark. "Benchmark" invites expectations about packaging, leaderboards,
standardized splits, and external reuse that the current project does not yet
meet.

### 7.3 Learning Setup

Use "policy" only to mean a parametric allocation function. Do not call the
method reinforcement learning unless training optimizes through environment
interaction. If the method is supervised, imitation-based, or ERM over generated
rollouts, name it that way. ERM means empirical risk minimization. In this
document, "rollout" means a simulator-generated trajectory used for
ERM/evaluation unless explicitly marked as on-policy interaction.

## 8. Objective Function

Let `G = (V, E, X_V, X_E, Z)` be a feeder graph with node set `V`, edge set `E`,
node features `X_V`, edge features `X_E`, and static node attributes `Z`.
Critical nodes are `C = {i in V : c_i = 1}`.

At time `t`, the policy outputs allocation `a_t in Delta(G, s_t)` using the
feasible set and projection defined in Section 3. Hard constraints are enforced
by projection, not by penalty.

For a scenario `xi = (d_{1:T}, P_{1:T}, O_{1:T})`, use the rollout return:

```text
J(pi; G, xi) =
    lambda_cont sum_{L in W} sum_t sum_{i in C} w_i q_{t,i,L}     # PRIMARY: continuity surrogate
  - lambda_c    sum_{t} sum_{i in C \ O_t} max(m_i d_{t,i} - a_{t,i}, 0)   # adequacy guard
  + lambda_e    sum_{t} sum_{i in V} w_i a_{t,i}                  # weighted served energy
  - lambda_s    sum_{t=2}^T ||a_t - a_{t-1}||_1                   # anti-chatter only

where  s_{t,i} = sigmoid(beta (a_{t,i} - m_i d_{t,i})),
       q_{t,i,L} = prod_{tau=t-L+1..t} s_{tau,i}   (= exp(sum log s); outaged steps transparent)
```

The continuity surrogate `q` is the differentiable relaxation of the windowed
critical-continuity metric `C` (§9); as `beta -> inf` it converges to the hard
metric. It is the **primary** driver (`lambda_cont` dominant). Implemented in
`src/sg_resilience/continuity_loss.py`. The training loss is
`L(pi; G, xi) = -J(pi; G, xi)`.

Critical nodes receive served-energy reward like all nodes, plus an additional
shortfall penalty up to the minimum-service target `m_i d_{t,i}`. This encodes
asymmetric critical-load value without capping critical service at the minimum
target. The loss and primary adequacy metric exclude outaged critical node-time
pairs. The exact
`lambda_c`/`lambda_s` grid and validation split are protocol/config items. They
must be frozen before claim-bearing runs and cannot use OOD feeder results.

The smoothness term `lambda_s` penalizes allocation magnitude changes; it is
**not** temporal continuity (a flicker can be smooth). The explicit continuity
surrogate `q` above IS the term that optimizes the temporal-continuity metric, so
the loss now directly (softly) targets `C`. Keep `lambda_s` small — anti-chatter
regularization only.

Then train:

```text
theta* = argmin_theta E_{G ~ D_train} E_{xi ~ S_G}
          [ L(pi_theta; G, xi) ]
```

`lambda_c` and `lambda_s` are nonnegative hyperparameters fixed in the training
configuration before claim-bearing runs.

Do not put hard constraints into the loss. They belong in the projection and
the feasibility tests.

## 9. Metrics

Separate training objectives from paper metrics. **Primary metric (ADR-0001):**
capacity-normalized windowed continuity `C` (below). Adequacy and mean max-run
are secondary; mean max-run is **gameable** (serve one easy node forever) and is
not claim-bearing.

### Capacity-Normalized Windowed Continuity `C` (PRIMARY)

Full spec: `notes/continuity_metric_spec.md`; code: `src/sg_resilience/metrics_v1.py`.
For window lengths `L in W`, a critical node is continuously served over a length-L
window iff `served` holds for every step in it (on its outage-conditioned
sequence). Weighted by priority `w_i` and **normalized by the oracle frontier**
`B*_L` (max critical weight an offline allocator could keep continuously served
under the same `Delta_grid`, budget, and outages):

```text
C = mean_{L in W}  [ sum_{i in C} w_i cont(i,L) ] / [ sum_{i in C} w_i cont_oracle(i,L) ]
```

`C in [0,1]` = fraction of the *achievable* continuity the policy captures.
Report per-feeder, with **coverage/fairness diagnostics** (critical coverage,
weighted starvation, worst-served node) to expose favorite-node gaming. The
oracle is injected (`flow_projection.make_flow_oracle`) so it respects `Delta_grid`.

### Critical-Load Adequacy (guard, secondary)

The existing binary served metric should be renamed adequacy. `m_i` is the
minimum fraction of node `i`'s demand that must be served for that load to count
as adequately supplied:

```text
adequacy(pi; G, xi) =
  mean_{t, i in C} 1[ a_{t,i} >= m_i d_{t,i} ]
```

The mean is over the joint set of critical node-time pairs. This says how often
critical loads receive at least their minimum service fraction. It does not
measure temporal continuity.
Primary adequacy excludes node-time pairs where the critical node itself is in
the outage mask; outage-inclusive adequacy is reported separately when needed.

### Mean max-run continuity (secondary, gameable — not claim-bearing)

Retained for backward-compatibility/reporting only. **Gameable** (serve one easy
node forever), so it is NOT the primary metric — `C` above is. Use it only as a
diagnostic beside `C`.

```text
served_{t,i} = 1[ a_{t,i} >= m_i d_{t,i} ]

temporal_continuity(pi; G, xi) =
  mean_{i in C} max_run_length(served_{1:T,i}) / T
```

If `O_t` includes a critical node itself, this metric counts the outage as an
end-to-end service interruption. To isolate allocation-policy behavior from
exogenous node outages, report an outage-conditioned variant separately:
compute the same `max_run_length / T` after removing timesteps where `i in O_t`
from that node's served sequence. If no non-outage timestep remains for a
critical node, exclude that node from the outage-conditioned denominator.

Example: over a four-step horizon, served pattern `1,0,1,0` and `1,1,0,0` both
have adequacy `0.5`; their temporal-continuity scores are `0.25` and `0.5`.

### Transfer Gap

For any metric `M`:

```text
Gap_M(pi) =
  E_train[M] - E_ood[M]       for higher-is-better metrics
  E_ood[M] - E_train[M]       for lower-is-better metrics
```

Positive always means OOD is worse. Risk gaps are lower-is-better, so they use
`R(pi; D_ood) - R(pi; D_train)`. Do not compute `Gap_M` for invariants such as
feasibility violations; report those directly. Report per-feeder gaps, not only
aggregate means.

For v1, per-feeder gaps are the canonical result. Any aggregate gap is labeled
as a summary statistic only.

### Structural Distance

No v1 structural-distance metric is defined. V2 may introduce a feeder
fingerprint and structural-distance protocol after the expanded split exists.

## 10. Formal Problem Formulation

Sections 3 and 8 are the source of truth for spaces, distributions, the policy
class, projection, and training objective. This section only states the
evaluation problem and v2 association analysis to avoid duplicate definitions.

### Evaluation Problem

Evaluate `pi_{theta*}` without retraining on `G ~ D_ood`.

Primary evaluations:

- in-family adequacy and temporal continuity;
- OOD adequacy and temporal continuity;
- transfer gap for each metric;
- feasibility violation count;
- comparison to MLP, wrong-graph, rule, optimization, and GNN baselines;
- no v1 structural-distance table and no beta-slope regression.

Reserve the word guarantee for feasibility only if the projection theorem is
actually proved.

## 11. Brutal Honesty

### Claims To Remove Or Weaken

- Say "flow-constrained" (capacity limits enforced in `Delta_grid`) and
  "flow-aware" (operator). Never "physics-aware" — no power-flow equations are solved.
- Temporal continuity `C` is now implemented (`metrics_v1.py`) and is the PRIMARY
  metric and objective; it is no longer a deferred/eval-only quantity.
- Replace "formal guarantees for topology generalization" with
  "projection-enforced feasibility plus empirical transfer-risk reporting."
- Replace "predicts transfer before training" with "is associated with observed
  transfer gaps across frozen feeder splits."
- Keep "catastrophic grid events" in motivation only; use "scarce-budget
  stochastic outage regimes" in the formal problem.
- Rename the existing continuity ratio to adequacy unless the temporal-run
  metric is implemented.
- Do not claim broad zero-shot generalization from three OOD feeders.
- Do not use R2 as the primary structural-correlation claim for small N. Prefer
  per-feeder plots, Spearman correlation, and a permutation null.
- Do not describe the current training as reinforcement learning unless the
  method actually optimizes through environment interaction. If training is
  supervised/imitation-style, call it a learned graph policy.

### Reviewer Attacks To Pre-Answer

| Attack | Honest response |
|---|---|
| Why not train one model per feeder? | That is operationally plausible. The v1 scientific question is whether graph structure is load-bearing when one policy transfers across feeders. |
| Six feeders total, with three OOD feeders, is not broad graph generalization. | Correct. Treat small-feeder experiments as workshop evidence and expand before main-conference claims. |
| Your bound uses empirical constants. | Correct. V1 has no transfer bound; only the projection can become a guarantee after proof and tests. |
| Your continuity metric is not continuity. | Fix it with temporal-run continuity or stop using the word. |
| Random outage masks are not catastrophes. | Correct. Catastrophe motivates the regime; the formal setup is stochastic scarce-budget allocation. |
| Edge features are not physics. | Correct. They are flow features unless a power-flow model is enforced. |
| Your no-graph baseline is weak. | Use DeepSets or a permutation-invariant transformer with identical node features and projection; treat flat MLP as a diagnostic only. |
| Demand and outage shift confound topology shift. | Keep normalized workload processes invariant across train and OOD feeders, or report workload-shift experiments separately. |

## 12. Clean Paper Claim

The strongest honest claim is:

> We introduce a flow-constrained graph-allocation evaluation suite for
> scarce-power distribution-feeder control under topology shift. A graph policy
> with projection-enforced feasibility on the branch-flow-capacitated polytope
> `Delta_grid` is trained on one feeder family and evaluated zero-shot on
> held-out feeders. Our primary outcome is capacity-normalized temporal
> continuity of critical service `C`; we test whether graph structure is
> load-bearing through size-generalizing no-graph and wrong-graph ablations, with
> coverage/fairness diagnostics, under held-out topology shift.

For v1, no `d_struct` or beta-slope result is reported. A claim that
feeder-structure distance is associated with transfer gaps is v2 work after the
pair-count gate is met and the feeder atlas is frozen.
