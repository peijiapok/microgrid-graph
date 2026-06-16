# 05 — Experimental Protocol

> Last revised 2026-06-16 (was 2026-05-08). How we test the claims in
> `docs/10_research_formulation.md`. Every table in the paper must correspond
> to a section here, or the section is missing.
>
> **2026-06-16 reframing (ADR-0001, ADR-0002):** primary metric is
> capacity-normalized windowed continuity `C` (was `critical_load_adequacy`);
> feasibility includes branch-flow capacity (`Delta_grid`); §2.5 thresholds are
> re-pinned on `C`; §1 split is the frozen ADR-0002 split.

## 1. Topology splits

The new axis. Existing evidence trains and tests on the same feeder; that gives no transfer signal.

A result is **claim-bearing** if it appears in a paper table, figure, abstract,
or contribution claim.
ADR means architectural decision record: a dated pre-run note that freezes a
protocol choice before claim-bearing experiments.
H1 means graph structure is load-bearing under OOD topology shift. H2 means a
single policy retains useful critical-load adequacy on held-out feeders without
retraining.

Version terms: thresholds are versioned by this file; split-v1 means the
workshop-scope split with up to six feeders; v2 means expanded splits with
enough train-test topology pairs for structural-association claims.

**Scenario.** A scenario is one sampled horizon
`xi = (d_{1:T}, P_{1:T}, O_{1:T})` for a fixed feeder: demand trajectory,
available-power trajectory, and outage-mask trajectory. Scenario train/test
splits are drawn independently within each feeder and referenced by seed.
Default split: 70% train scenarios, 15% in-family validation scenarios for
hyperparameter selection, and 15% test scenarios per training feeder. OOD
feeders contribute test scenarios only; they do not contribute to training or
hyperparameter selection.

**D_train / G_train (in-family).** `D_train` is the training feeder
distribution; `G_train = support(D_train)` is the feeder set used during
training. Held-out *scenarios* per feeder.

**D_ood / G_ood (held-out topology).** `D_ood` is the OOD feeder distribution;
`G_ood = support(D_ood)` is the feeder set never seen during training;
held-out scenarios within each.

### V1 contribution legend

- **C1:** flow-feature-aware message passing.
- **C2:** preliminary v1 topology-transfer probe on held-out feeder families.
- **C3:** differentiable projection onto the flow-constrained polytope `Delta_grid`
  (box + budget + outage + branch-flow capacity) for projection-enforced feasibility.
- **C4:** optional CVaR training objective.
- **C5:** graph-baseline suite.

V2 work, not a v1 contribution: feeder atlas, `d_struct`, structural-distance
association tests, and label-permutation nulls.

### Initial split (v1)

**Frozen split (ADR-0002, validated 2026-06-16 by `scripts/validate_feeders_v1.py`).**
All feeders are radial (required for branch-flow constraints) and carry line
thermal capacity. Config: `configs/topology_split_v1.yaml`.

| Set | Feeder | Source | Loads | Topology class |
|---|---|---|---|---|
| G_train | case33bw | pandapower | 32 | `(radial, semiurban)` |
| G_train | 1-LV-rural2--0-sw | SimBench | 99 | `(radial, rural)` |
| G_train | 1-LV-rural3--0-sw | SimBench | 118 | `(radial, rural)` |
| G_ood  | 1-LV-urban6--0-sw | SimBench | 111 | `(radial, urban)` |
| G_ood  | 1-MVLV-urban-5.303-0-sw | SimBench | 242 | `(radial, urban)` |
| G_ood  | 1-MVLV-urban-6.305-0-sw | SimBench | 249 | `(radial, urban)` |

Rationale: three radial G_train feeders span semiurban+rural and sizes 32–118;
the OOD set holds out the urban feeder_type and includes two MVLV-urban feeders
larger than every train feeder (size-gen). IEEE 34/123 OpenDSS conversion is
**dropped from the v1 critical path** (the MVLV-urban feeders give a validated
larger radial OOD without conversion risk); kept for v2 C6 pair count. All v1
results retrained from scratch; legacy runs are not evidence. Note: case33bw line
ratings are a pandapower placeholder (flow caps slack there) — physical ratings
to be assigned, or rely on the SimBench feeders for the topology-binding story.

IEEE feeders that require OpenDSS-to-pandapower or multi-phase-to-single-phase
conversion must be validated before training. IEEE 34-bus is an OOD candidate:
if validated it enters claim-bearing `G_ood`; if not, it is excluded and the
ADR-selected fallback enters `G_ood` instead. IEEE 123-bus is claim-bearing
only after the same conversion gates pass:

- **Conversion procedure.** Convert each phase conductor to a separate
  pandapower branch/load first. For the single-phase graph used by the learning
  model, merge phase nodes with the same bus id; node demand is the sum of
  phase real-power loads, and edge capacity is the sum of phase ampacity
  proxies. Ampacity proxy means OpenDSS `NormAmps` when available, otherwise
  the pandapower line current limit field, otherwise an ADR-documented thermal
  limit proxy. Phase admittances combine in parallel:
  `Y_total = sum_phase Y_phase`; the single-phase impedance feature is
  `Z_total = 1 / Y_total` when `Y_total != 0`. Missing phase quantities are
  logged and zero-filled with an explicit missing-feature indicator.
- **Conversion gates.** The converted case must pass base-case convergence,
  preserve radiality, and keep total real load within 5% of the OpenDSS source
  after phase aggregation. Degree-histogram distance is reported descriptively
  with bins fixed by `configs/topology_split_v1.yaml`, not used as a hard gate.
- **Fallback rule.** If conversion is not validated, the pre-committed fallback
  must be a non-urban SimBench or pandapower distribution feeder with more than
  60 load nodes and a topology class that differs from the other OOD feeders in
  `radiality` or `feeder_type`. The fallback is selected
  through an ADR before training. Any substitute must preserve every split rule,
  including `|V_ood| > max |V_train|`. If LV-urban is invalidated, the fallback
  must have a `feeder_type` absent from `G_train`; otherwise C2 is reframed as
  size-transfer only.

### Split Rules

Any accepted split must satisfy:

1. No feeder appears in both G_train and G_ood.
2. At least one OOD feeder has |V| > max |V| in G_train (size-generalization).
3. At least one OOD feeder has topology class different from every G_train
   feeder (structural-generalization). For v1, topology class is assigned
   directly in `configs/topology_split_v1.yaml` and justified by an ADR before
   training. The ADR must record the source fields or script used to assign
   each label; hand-edited labels without provenance are not claim-bearing. The
   v1 label tuple is `(radiality, feeder_type)`, where `radiality` is `radial`
   or `meshed`, and `feeder_type` is one of `rural`, `urban`, `semiurban`, or
   `industrial`. In v1 this mostly tests rural-to-urban transfer plus size
   shift; do not overstate it as
   broad structural generalization.
   Differing in either coordinate is sufficient for the v1 rule; in practice
   v1 is expected to satisfy this mainly through `feeder_type=urban`.
4. The split records topology-class labels for every feeder.

Splits are versioned in `configs/topology_split_vN.yaml` and referenced by hash in every result artifact.

V1 topology-class assignments must be listed in `configs/topology_split_v1.yaml`
for every feeder using the tuple `(radiality, feeder_type)`.
If an OOD feeder is invalidated and no ADR-selected fallback lands before
claim-bearing runs, reducing `G_ood` below three feeders, C2 must be reframed
as a two-feeder stress test rather than a topology-transfer probe.
If LV-urban is invalidated, C2 loses its only v1 feeder-type generalization
case unless the fallback has a `feeder_type` absent from `G_train`; otherwise
C2 must be described as size-transfer evidence only.

### 1.5 Feeder atlas and topology-similarity metric

V1 uses only topology-class labels in `configs/topology_split_v1.yaml`.
`d_struct`, the feeder atlas, Spearman association tests, and label-permutation
nulls are v2-only and not used in v1 gates, contribution claims, or split
rejection. They require at least 20 train-test topology pairs and a frozen
`configs/feeder_atlas.yaml`.

## 2. Metrics

Symbol glossary: `pi` is the learned allocation policy; `i` indexes nodes; `t`
indexes time; `T` is the rollout horizon; `a_t,i` is allocation to node `i` at
time `t`; `d_t,i` is demand; `m_i` is the minimum service fraction for node
`i`; `w_i` is the node service weight frozen in the feeder config; `P_t` is
available power; `O_t` is the outage mask; `R(pi; G_j)` is expected rollout
loss over held-out scenarios on feeder `G_j`; `R(pi; D_train)` is the uniform
mean of `R(pi; G)` over `G in G_train`.
`m_i`, `w_i`, and critical flag `c_i` are frozen in
`configs/feeder_attributes_v1.yaml`; default `m_i = 0.8` for critical loads
unless that file specifies otherwise.

**Core rollout metrics.**
- `weighted_served_energy`: `sum_t sum_i w_i a_t,i`, normalized by total
  nominal demand unless a report states otherwise.
- `unserved_critical_demand`: `sum_t sum_{i:c_i=1} max(d_t,i - a_t,i, 0)`,
  normalized by total critical demand.
- `switching_count`
- **`continuity_C` (PRIMARY, ADR-0001):** capacity-normalized, priority-weighted
  windowed continuity — `mean_{L in W} [sum_{i in C} w_i cont(i,L)] / [sum_{i in C}
  w_i cont_oracle(i,L)]`, where `cont(i,L)` is the fraction of length-L windows
  critical node i is continuously served (outage-conditioned) and `cont_oracle`
  is the `Delta_grid` oracle frontier. Spec: `notes/continuity_metric_spec.md`;
  code: `src/sg_resilience/metrics_v1.py`. Reported per-feeder with coverage,
  weighted-starvation, and worst-served-node diagnostics.
- `critical_load_adequacy` (GUARD, secondary): mean over non-outaged critical
  node-time pairs of `1[a_t,i >= m_i d_t,i]`. Prevents trading all adequacy for continuity.
- `temporal_continuity` (secondary, **gameable** — diagnostic only): mean over
  critical nodes of `max_uninterrupted_run(1[a_t,i >= m_i d_t,i]) / T`.

**New, required for the paper.**
- **`transfer_gap[metric]`**: compute the scenario mean for each
  `(method, seed, feeder)`, then average equally over seeds and feeders within
  `D_train` or `D_ood`. For higher-is-better metrics, use train minus OOD. For
  lower-is-better metrics such as `unserved_critical_demand` or
  `switching_count`, use OOD minus train. Positive always means OOD is worse.
  Formula:

  ```text
  mu(seed, G, M) = mean_{scenario on G} M(seed, G, scenario)
  mu_train(M) = mean_{G in G_train} mean_seed mu(seed, G, M)
  mu_ood(M)   = mean_{G in G_ood}   mean_seed mu(seed, G, M)
  transfer_gap[M] =
      mu_train(M) - mu_ood(M)   if higher is better
      mu_ood(M) - mu_train(M)   if lower is better

  per_feeder_gap[M, G_j] =
      mu_train(M) - mean_seed mu(seed, G_j, M)   if higher is better
      mean_seed mu(seed, G_j, M) - mu_train(M)   if lower is better
  ```

  Example: adequacy `0.82` in-family and `0.70` OOD gives gap `0.12`.
  Unserved critical demand `0.10` in-family and `0.18` OOD gives gap `0.08`.
  With the v1 split, per-feeder gaps are the headline and the equal-weighted
  average is a summary only. V1 transfer numbers are limited evidence across
  three OOD feeders, not broad topology-generalization evidence.
  Feeder-equal weighting is deliberate: each feeder is one topology case,
  regardless of node count or scenario count. Scenario-weighted summaries may
  be reported only as secondary diagnostics.
- **`transfer_risk_gap`**: `transfer_gap` applied to expected rollout loss,
  reported per held-out feeder as `R(pi; G_j) - R(pi; D_train)`. Positive
  means OOD risk is worse.
- **`wall_clock_per_step`**: CPU evaluation-time rollout latency, not training
  time. It is reported because emergency feeder operation may run on ordinary
  utility/substation compute rather than an accelerator; GPU latency can be
  added as a secondary metric if deployment assumptions change.

**Optional if CVaR is included.**
- **`cvar_05[loss]`** = upper-tail conditional value at risk at α=0.05 over
  held-out scenarios.

`critical_continuity_ratio` is legacy naming. If retained in code for backward
compatibility, map it to `critical_load_adequacy` in reports. Do not call it
continuity in paper-facing text unless it uses the temporal-run definition. The
mapping must live in the report-generation layer, not in manuscript text.

Outage semantics: if a critical node is in the outage mask at time `t`, it is
excluded from the primary adequacy denominator because the allocation policy
cannot serve it. Report the outage-inclusive adequacy variant separately, where
outaged critical node-time pairs count as failures.

## 2.5 Pre-registered thresholds

These values are pre-registered by this protocol. **Per ADR-0001 the primary
gated metric is now `continuity_C`** (was `critical_load_adequacy`); thresholds
below are read on `continuity_C`, both on the native 0-1 scale. The numeric
values are carried over from the adequacy calibration and are **provisional for
`C`** until pilot variance is measured. Any change requires a dated ADR or
protocol revision before claim-bearing runs.

- **`Delta_graph = 0.05`**: minimum graph-vs-baseline improvement in absolute
  metric points on OOD `continuity_C`, the H1 primary metric. Adequacy is
  reported beside it as a guard but does not rescue a `C` failure.
- **`Delta_transfer = 0.20`**: maximum acceptable train-to-OOD degradation in
  absolute metric points for per-feeder `continuity_C`. Adequacy is reported with
  the same gap convention but is non-binding for H2 viability. `Delta_transfer`
  is not applied to `unserved_critical_demand`, `switching_count`, or latency.
- **`Delta_rewire = 0.05`**: minimum graph-vs-degree-preserving-rewire
  improvement in absolute OOD `critical_load_adequacy` points.

Rationale: `0.05` is a five-percentage-point service-quality effect, intended
to clear numerical/reporting noise and be visible in per-feeder tables. The
same effect size is used for `Delta_graph` and `Delta_rewire` because both ask
whether graph structure changes service quality enough to matter on the same
0-1 metrics. `0.20` is a viability threshold for workshop evidence: larger
degradation means the zero-shot story must be reframed. It is intentionally a
loose non-rejection bar for v1, not a strong-transfer success bar. These are
current thresholds, not evidence that the effect exists.

Pass/fail semantics for claim-bearing comparisons:

All paired-bootstrap confidence intervals use `B = 10,000` resamples unless an
ADR changes this before training.

- For H1 graph-vs-baseline and rewire comparisons, pass requires the observed
  mean improvement on `critical_load_adequacy` to exceed the relevant threshold
  and the paired-bootstrap 95% lower confidence bound on that improvement to be
  greater than the same threshold.
- If the observed improvement exceeds the threshold but the lower confidence
  bound is below the threshold, report the comparison as positive but
  inconclusive.
- If the observed improvement is below the threshold, the corresponding H1
  comparison fails regardless of confidence interval.
- For H2 transfer retention, v1 viability requires each claim-bearing
  per-feeder `critical_load_adequacy` transfer gap to be below
  `Delta_transfer`. The paired-bootstrap 95% upper confidence bound is reported:
  if it also falls below `Delta_transfer`, label the result strong retention;
  otherwise label it retention with unresolved statistical uncertainty. If the
  point gap exceeds threshold, H2 fails for that feeder. Temporal continuity is
  secondary and reported with the same gap convention.

V2 structural-association gate: at least 20 train-test topology pairs are
required before any `d_struct` association is claim-bearing. The v1 split cannot
reach that gate and therefore cannot make a structural-distance association
claim. The value 20 is a minimum design rule for exploratory rank association:
it prevents v2 from reporting a correlation from only a handful of feeder
pairs; it is not a power guarantee and should be replaced by a power analysis
once pilot variance is available.

## 2.6 Feasibility invariant

`feasibility_violations` counts budget, demand-cap, non-negativity, outage
mask, **and branch-flow capacity** (`|f_e(a)| <= F_e`, ADR-0001) breaches under
absolute tolerance `epsilon_feas = 1e-6` in normalized power units, after
dividing all feeder power quantities by that feeder's total nominal demand.
Branch flows and violations are computed by `src/sg_resilience/topology.py`. Total nominal demand means `sum_i nominal_load_i` from the
feeder case file before scenario scaling. Claim-bearing rollouts require
`feasibility_violations = 0`.
Projection ablations that intentionally remove the feasibility guarantee are
reported only in the projection-ablation table and are excluded from
claim-bearing method comparisons.
If a non-ablation method has any feasibility violation, that method fails the
feasibility invariant for the affected run before missing-data handling. The
report must show the violation count and mark the corresponding claim-bearing
comparison as failed or infeasible; no paired-bootstrap claim is reported for
that comparison.

## 3. Baselines

| Family | Baseline | Implemented | Owner |
|---|---|---|---|
| Rule | Priority allocator | Yes (`baseline_rule.py`) | Jia |
| Optimization | Continuity-aware LP/QP | Yes (`baseline_opt.py`) | Jia |
| Learned | GraphSAGE (current) | Yes | Jia |
| Learned | Flow-aware MP (C1) | No | Collaborator |
| Learned | GCN | No | Jia (C5) |
| Learned | GAT | No | Jia (C5) |
| Learned | GIN | No | Jia (C5) |
| Learned | EGNN | No | Jia (C5) |
| Learned | Graph Transformer | No | Jia (C5) |
| Learned | DeepSets (primary no-graph) | No | Jia |
| Learned | Set transformer (stretch no-graph) | No | Jia |
| Learned | MLP (diagnostic only) | No | Jia |

All learned baselines share: the same feature extractor, the same projection,
the same loss, the same training budget. Only the core operator varies. This
isolates the contribution of the operator.
Hyperparameters are frozen before claim-bearing runs in a dated config or ADR;
no OOD feeder results may be used for hyperparameter selection.

Workshop claim-bearing subset: GraphSAGE, Flow-aware MP, GCN, GAT, GIN, and
DeepSets. Set transformer, EGNN, Graph Transformer, and MLP are stretch or
diagnostic unless compute permits. If Flow-aware MP misses its implementation
gate, the fallback workshop subset is GraphSAGE, GCN, GAT, GIN, and DeepSets;
C1 is then removed from the workshop claim and the paper narrative becomes
"graph-necessity evaluation suite with projection-enforced feasibility," not a
new flow-feature-aware architecture paper.

C1 implementation gate: a branch or PR provides a flow-feature-aware operator
that consumes directed edge features, runs on `case33bw`, and passes the smoke
suite before claim-bearing workshop runs begin. The edge-feature check is only a
wiring smoke check: on a fixed smoke batch, zeroing edge features must change
pre-projection allocation scores with
`||y_true - y_zero||_2 / max(||y_true||_2, 1e-6) > 1e-2`. Evidence that edge
features are useful requires the claim-bearing edge-feature ablation in the
main results; the smoke check alone cannot support C1.

## 4. Ablations

### 4.1 Graph-necessity ablations

Goal: prove the graph is load-bearing, not decorative.

| Ablation | What is removed | Prediction | Reviewer concern addressed |
|---|---|---|---|
| DeepSets | edges removed, size-generalizing set model retained | on OOD feeders, graph policy exceeds DeepSets by at least `Delta_graph` on `critical_load_adequacy` | "why is this a graph paper" |
| Degree-preserving rewire | actual adjacency → configuration-model rewire preserving degree sequence | on OOD feeders, rewire degrades vs graph policy by at least `Delta_rewire` on `critical_load_adequacy` | "is exact topology load-bearing" |
| Edge-feature zeroing | edge features set to zero after training | if C1 is claimed, zeroing edge features degrades OOD `critical_load_adequacy` by at least `Delta_graph` | "are flow features used" |
| MLP diagnostic | graph → padded / pooled MLP | diagnostic only; not primary evidence | input-shape sanity check |
| Identity adjacency diagnostic | adjacency → I | diagnostic only; not primary evidence | message-passing sanity check |

Degree-preserving rewires for radial feeders must preserve connectedness and
edge count `|E| = |V|-1`, so they remain radial trees with the same degree
sequence. Rewires must pass the same feasibility and physics-audit checks as
the original feeder. Invalid rewire samples are resampled before training; the
number of rejected samples is logged.

### 4.2 Objective ablations

| Ablation | What is removed |
|---|---|
| No continuity surrogate | primary windowed-continuity term off → falls back to adequacy/served-energy only (tests whether the continuity surrogate drives the continuity metric `C`) |
| No imitation | SmoothL1 teacher term off |
| No switching penalty | switching term off |
| No CVaR | replace CVaR-minimax (C4) with lexicographic selection |

### 4.3 Projection ablations

| Ablation | What is used |
|---|---|
| Softmax + normalization | no feasibility guarantee |
| Box+budget projection only (no flow caps) | drops branch-flow constraints — tests whether topology-coupled feasibility is what makes topology matter for continuity |
| `Delta_grid` differentiable projection (C3) | full projection-enforced feasibility incl. branch-flow |

## 5. Compute budget

- **CPU smoke suite.** End-to-end on a single feeder (case33bw), single seed, reduced epochs. Target: <10 min on a laptop. Used as a CI-gate and as the collaborator's onboarding test.
- **Workshop-paper budget.** One training run means one `(method, seed,
  training split)` fit followed by evaluation on every in-family and OOD feeder
  in the split. The workshop claim-bearing subset has six learned methods ×
  five seeds = 30 fits if Flow-aware MP passes C1. If C1 fails, the fallback
  subset has five learned methods × five seeds = 25 fits. Add rule/LP
  evaluations and ablations on top of those fits. Target: ~160 GPU-hours total
  on an A100 equivalent for the six-method subset, including evaluation passes
  but excluding broad hyperparameter search; fallback should be lower.
- **Hyperparameter search budget.** Maximum 40 A100-equivalent GPU-hours,
  restricted to `G_train` train/validation scenarios. No OOD feeder result may
  influence hyperparameters.
- **Main-conference budget.** Above + expanded OOD families + v2
  structural-association pair count + theory-matching experiments
  ≈ 300 GPU-hours.

If compute is tight, reduce in this order: stretch baselines first, then
diagnostic ablations, then seeds. Never reduce below a valid OOD topology set
that satisfies the split rules.

## 6. Reporting

Target state: the harness writes one canonical result bundle under `results/`
containing method, feeder, seed, scenario, and metric records. Report scripts
consume that same bundle and emit table-specific `.tex` fragments. No
hand-typed numbers. A failing pipeline means the table is blank. V1 freeze
requires at least the main-result and transfer-gap report scripts to be
runnable on the canonical result bundle.

| Paper table | Generator |
|---|---|
| Main result: methods × feeders × metrics | `scripts/report_main.py` (to write) |
| Transfer gap: methods × in-family vs OOD | `scripts/report_transfer.py` (to write) |
| Ablations: components × metrics | `scripts/report_ablations.py` (to write) |
| Feasibility: methods × violations | `scripts/report_feasibility.py` (to write) |

Scripts live in `scripts/` (to create). Tables land in `paper/tables/` as `.tex` fragments.

## 7. Seeds and determinism

- Each experiment seeded on (method, feeder, seed_idx). Five seeds minimum for
  claim-bearing workshop comparisons.
- Report paired bootstrap confidence intervals over matched scenarios and seeds
  for claim-bearing method comparisons. Matched means resampling aligned
  `(feeder, scenario_id, seed_idx)` tuples shared by the compared methods.
  For each pairwise comparison, if either of the two compared methods lacks an
  entry for a tuple because of physics non-convergence or a failed run, drop
  that tuple for both methods and report the dropped fraction. If more than 5%
  of tuples are dropped for that pairwise comparison, the comparison is
  inconclusive unless an ADR pre-registers a different missing-data threshold
  and rationale before training.
  Feasibility violations are handled by §2.6 first; they count as method
  failures and stop the claim-bearing comparison rather than entering the
  ordinary missing-tuple drop set.
- PyTorch and numpy seeded at process start. Non-determinism from pandapower
  solvers is logged. For claim-bearing service-quality comparisons, repeated
  solver/evaluation calls must have 95% CI half-width <= 0.02 absolute metric
  points on `critical_load_adequacy` and `temporal_continuity`; use at least
  three repeated calls for this check, then increase repeats or report the
  affected comparison as inconclusive.
  This is intentionally strict relative to `Delta_graph = 0.05`; if the noise
  band cannot be held below 0.02, five-point effects are not stable enough to
  claim.

## 8. Physics audit

Post hoc. For every reported method-feeder-seed rollout, a fixed audit sample
of scenarios is checked for pandapower convergence. Default audit sample:
`min(50, number_of_eval_scenarios)` scenarios selected by the evaluation seed.
A rollout that fails physics convergence is reported separately, not silently
dropped.

## 9. Workload audit

V1 requires a descriptive workload audit before making even the preliminary
topology-transfer probe claim. This audit does not prove workload equivalence;
it tells readers whether transfer gaps may be partly workload-driven. V2 uses
the same quantities as gates before reporting any structural association
between `d_struct` and transfer gap.

- report per-feeder distributions of normalized demand scale;
- report per-feeder budget ratio `rho_t = P_t / sum_i d_t,i`;
- report per-feeder outage rate;
- compute normalized mean shift for demand scale and budget ratio, and compute
  absolute outage-rate difference. Default v2 gate: normalized mean shift must
  be below 0.1 pooled training standard deviations for demand scale and budget
  ratio, and outage-rate difference must be below 0.02. Two-sample KS tests at
  `alpha = 0.05` are reported as flags, not pass/fail evidence of equivalence.
  Pooled training standard deviation means the standard deviation over all
  training-feeder scenario samples used in the corresponding workload
  comparison.

For v1, the same numeric thresholds define "distinguishable" but act as a
reframing trigger, not a rejection gate: if demand-scale or budget-ratio
normalized mean shift is >= 0.1 pooled training standard deviations, or
outage-rate difference is >= 0.02, keep C2 as a "distribution-shift transfer
probe" rather than a topology-transfer probe. If normalized workloads are
distinguishable in v2, label the result as mixed topology/workload shift and do
not claim a structural association.

### 9.1 Scarcity calibration gate (required for continuity, added 2026-06-16)

Empirically established by the first baseline run (`notes/eval_v1_first_result.md`):
the continuity metric `C` is **policy-controllable only when in-family critical
adequacy < 1**. At default supply ratios, the budget meets all critical
min-service every step (adequacy = 1.0), so `C` reflects exogenous outages, not
allocation skill, and is invariant to the budget.

Two requirements before any claim-bearing continuity-transfer result:

1. **Per-feeder budget calibration.** Tune each feeder's budget so in-family
   critical adequacy lands in the band **[0.65, 0.80]** (pre-registered). A
   uniform power scale across feeders is **forbidden** for claim-bearing runs: it
   lands each feeder in a different scarcity regime (a uniform 0.25× cut left
   rural3 at adequacy 1.0 while the MVLV feeders collapsed to ~0.01), so the
   resulting transfer gap measures demand/budget-ratio heterogeneity, not
   topology. Report the calibrated per-feeder budget ratio `rho` in the workload
   audit.
2. **Adequacy-band check.** If a feeder's calibrated in-family adequacy falls
   outside [0.65, 0.80], its continuity number is not claim-bearing until
   recalibrated; record the achieved band per feeder.
