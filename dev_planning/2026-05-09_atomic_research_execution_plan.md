# Atomic Research Execution Plan - Microgrid Graph v1

Date: 2026-05-09

This is an execution roadmap, not research evidence. Do not load it into
training, evaluation, retrieval, runtime memory, scenario generation, or paper
evidence. ADR means Architecture Decision Record: a dated pre-run note that
freezes a protocol choice.

Cut A is the minimum evidence path before any full training. Cut B is the full
workshop sweep. OOD means out-of-distribution held-out feeder. MP means message
passing. H1 and H2 are hypothesis gates; C1 and C2 are contribution labels.

## Done Definition

V1 is executable when the repo can:

1. Load the frozen feeder split.
2. Build train/validation/test scenario bundles with frozen node attributes.
3. Train the claim-bearing method subset with fixed hyperparameters and seeds.
4. Run OOD evaluation and required baselines.
5. Generate a single canonical result bundle and tables for H1/H2.
6. Report feasibility, workload-audit, physics-audit, and transfer-gap outcomes.

V1 claims only a limited radial-distribution held-out-feeder transfer probe:
train on the frozen `G_train` feeders, evaluate without retraining on the
frozen `G_ood` feeders, and report per-feeder transfer gaps. V1 does not claim
structural-distance association, beta-regression, broad topology generalization
across voltage classes, industrial feeders, meshed systems, or catastrophe
simulation.

Claim-bearing method subset for v1:

- learned graph methods: GraphSAGE/current, GCN, GAT, GIN;
- no-graph learned baseline: DeepSets;
- non-learned baselines: priority allocator, continuity-aware LP/QP.

Flow-aware MP / C1 is deferred from the V1 atomic path. It can be added by a
separate ADR only after Cut A passes and after a task-anchored edge-feature
ablation protocol is written. This avoids a conditional architecture claim
blocking the basic transfer experiment.

## Inline Gate Definitions

The atomic checklist must be executable without rereading the protocol docs.
Use these v1 definitions in every PASS/FAIL decision:

- Symbol glossary: `c_i` is the critical-load flag, `m_i` is minimum service
  fraction, `d_t,i` is demand, `a_t,i` is allocation, and `P_t` is available
  power at time `t`. `w_i` is a frozen service weight used only by loss or
  secondary weighted-energy reports, not by H1/H2 gate metrics.
- `critical_load_adequacy`: mean over non-outaged critical node-time pairs of
  `1[a_t,i >= m_i d_t,i]`, where `a_t,i` is allocation, `d_t,i` is demand, and
  `m_i` is the frozen minimum service fraction.
- `unserved_critical_demand`: `sum_t sum_{i:c_i=1} max(d_t,i - a_t,i, 0)`,
  normalized by total critical demand.
- `outage-inclusive_adequacy`: same indicator as `critical_load_adequacy`, but
  outaged critical node-time pairs are counted as failures instead of removed
  from the denominator.
- `temporal_continuity`: for each critical node, build the binary sequence
  `s_t = 1[a_t,i >= m_i d_t,i]` over non-outaged evaluable times, score
  `max_uninterrupted_run_of_ones(s) / T`, then average over critical nodes.
  `T` is the full rollout horizon, not the count of non-outaged times; outaged
  times are not successes and break uninterrupted runs. Example with `T=4`:
  `1,0,1,0 -> 0.25`; `1,1,0,0 -> 0.5`.
- `transfer_gap[M]`: for higher-is-better metrics, `mu_train(M)-mu_ood(M)`;
  for lower-is-better metrics, `mu_ood(M)-mu_train(M)`. Positive means OOD is
  worse. For each `(method, seed_idx, feeder)`, first average over scenarios
  on that feeder. Then for each feeder, average equally over the five seeds.
  Then average equally over feeders within `G_train` or `G_ood`. Large feeders
  do not get extra weight.
- `per_feeder_gap[M, G_j]`: same sign convention as `transfer_gap`, comparing
  train mean against one OOD feeder `G_j`.
- `rho_t`: budget ratio `P_t / sum_i d_t,i`, where `P_t` is available power and
  the denominator is realized demand at time `t`, including demand at outaged
  nodes before applying the outage mask. This is not nominal demand.
- `claim-bearing`: any method, metric, table, figure, or claim that appears in
  the paper as evidence. Diagnostics and projection ablations are not
  claim-bearing unless an ADR says otherwise before training.
- Paired bootstrap: for H1 method-vs-method comparisons, resample matched
  `(feeder, scenario_id, seed_idx)` tuples shared by the compared methods
  within the same evaluation split. For H2 train-vs-OOD transfer gaps, do not
  pretend train and OOD feeders are paired; bootstrap train tuples and OOD
  tuples separately inside each resample, then compute the gap using the same
  scenario-to-seed-to-feeder nested averaging order as the reported point
  estimate. If a method-vs-method comparison lacks a tuple for either method,
  drop it for both and report the dropped fraction.
- H1: graph structure is load-bearing. H1 graph-vs-baseline gate passes only
  if the paired-bootstrap 95% lower confidence bound on OOD
  `critical_load_adequacy` improvement exceeds `Delta_graph = 0.05`.
- H1 graph-vs-rewire gate passes only if the paired-bootstrap 95% lower
  confidence bound on OOD `critical_load_adequacy` improvement exceeds
  `Delta_rewire = 0.05`.
- H2: zero-shot held-out-feeder transfer is viable for v1. This is not a strong
  retention claim; `Delta_transfer = 0.20` is a loose workshop viability bar
  meaning gaps larger than 20 percentage points force reframing.
- H2 transfer-retention gate: every claim-bearing per-feeder
  `critical_load_adequacy` transfer gap must be below
  `Delta_transfer = 0.20`. The 95% upper confidence bound is reported; if it
  also falls below `0.20`, label as strong retention. If the point gap passes
  but the upper confidence bound is too wide to pass, label as viable but
  statistically unresolved, not strong retention. One OOD feeder with a point
  gap `>= 0.20` downgrades C2 from topology-transfer probe to stress-test
  evidence.
- C1: optional flow-feature-aware message passing contribution. C2: preliminary
  v1 topology-transfer probe on held-out feeder families.
- H2 and C2 use the same held-out feeder data. H2 is the method-level viability
  rule based on transfer gaps; C2 is the paper contribution label for limited
  radial-distribution held-out-feeder probing and must be downgraded if
  workload or OOD-feeder gates fail.
- Workload reframing trigger: if demand-scale or `rho_t` normalized mean shift
  is `>= 0.1` pooled training standard deviations, or outage-rate difference is
  `>= 0.02`, C2 is only a "distribution-shift transfer probe", not a
  topology-transfer probe. These are descriptive v1 tripwires: large normalized
  workload shifts make topology and workload effects inseparable.
- Feasibility invariant: claim-bearing non-ablation methods must have zero
  budget, demand-cap, non-negativity, and outage-mask violations at
  `epsilon_feas = 1e-6` after normalizing all feeder powers by total nominal
  demand of that feeder, `sum_i nominal_load_i`, before scenario scaling.
  Projection ablations are excluded from claim-bearing comparisons. Allowed
  feasibility statuses are `pass`, `infeasible`, `physics_failed`,
  `missing_tuple`, and `run_failed`. `infeasible` means allocation constraints
  were violated; `physics_failed` means pandapower/OpenDSS audit did not
  converge; `run_failed` means the method did not finish; `missing_tuple` means
  a result row was absent for an otherwise completed run.
- Model selection: tune and early-stop learned methods only on in-family
  validation scenarios from the training feeders, using
  `critical_load_adequacy`. Ties use validation
  `temporal_continuity`, then lower `unserved_critical_demand`, then earliest
  epoch/checkpoint. OOD results are never used for hyperparameters,
  checkpointing, or early stopping.
- Seed/power rule: five seeds is the V1 minimum, not a power guarantee. MDE
  means minimum detectable effect. Before final claim labels, write an advisory
  pilot/MDE note using the first full seed evaluated across the frozen train
  and OOD feeders. The pilot/MDE note is read-only and post-training for that
  seed: it cannot change hyperparameters, checkpoints, method inclusion, or
  early stopping. If 95% intervals are too wide to resolve
  `Delta_graph = 0.05`, label H1 comparisons statistically inconclusive.
  Label demotion is allowed; changing method inclusion, checkpoints,
  hyperparameters, seeds, or early stopping is not. The first full seed is
  `seed_idx=0`, not whichever run finishes first. If H2 upper confidence
  bounds cannot resolve `Delta_transfer = 0.20`, label H2 viable only by point
  estimate with unresolved uncertainty, not strong retention.

## Source Data Matrix

| Artifact | Source | How to obtain | Local target | Gate |
|---|---|---|---|---|
| `case33bw` train feeder | pandapower `pandapower.networks.case33bw()`; data from MATPOWER/Baran-Wu | Python package dependency `pandapower` | `data/raw/pandapower/case33bw/metadata.json` plus derived scenario bundle | Must load, run base power flow, count 33 buses / 32 non-slack buses |
| SimBench rural train feeder 1 | SimBench package `simbench.get_simbench_net(code)` | Install optional dependency `simbench`; enumerate codes with `collect_all_simbench_codes()` | `data/raw/simbench/<rural_code_1>/metadata.json` | Placeholder code is resolved by item 2; must include profiles in `net.profiles`; class `(radial, rural)` |
| SimBench rural train feeder 2 | SimBench package | Same as above | `data/raw/simbench/<rural_code_2>/metadata.json` | Must be distinct from rural 1 and satisfy train size target |
| SimBench LV urban OOD feeder | SimBench package | Same as above | `data/raw/simbench/<urban_code>/metadata.json` | Must be `(radial, urban)` and larger than all train feeders |
| IEEE 123 OOD feeder | OpenDSS IEEE 123 DSS files, e.g. `IEEE123Master.dss` | Download/pin from OpenDSS test-case source; convert to pandapower/single-phase graph | `data/raw/opendss/ieee123/`, `data/processed/ieee123/` | Conversion gates: convergence, radiality, real-load within 5%, admittance aggregation implemented |
| IEEE 34 OOD candidate | OpenDSS IEEE 34 DSS files, e.g. `ieee34Mod2.dss` | Download/pin from OpenDSS source; convert if possible | `data/raw/opendss/ieee34/`, `data/processed/ieee34/` | If conversion fails, replace via ADR fallback; not allowed to silently shrink OOD. IEEE 34 is not the size-generalization feeder; LV-urban or fallback carries that role |
| Fallback OOD feeder | SimBench or pandapower distribution feeder | Select by ADR only if IEEE 34/123/LV-urban invalidates | `configs/adr_fallback_feeder_v1.md` and data bundle | Required for full v1 if an OOD slot invalidates; must preserve split rules: size > max train and unseen `feeder_type` if replacing LV-urban |
| Node attributes | Derived from feeder loads + frozen policy | Generate once into config | `configs/feeder_attributes_v1.yaml` | Every node has `c_i`, `m_i`, `w_i`; default critical `m_i=0.8`; no OOD tuning |
| Scenario bundles | Generated by repo code from each feeder | Deterministic seeded generator | `data/scenarios/v1/{train,val,test,ood}/...jsonl` | 70/15/15 scenario split within each training feeder; OOD test only; no OOD validation |
| Canonical results | Experiment runner output | New report schema | `results/v1/canonical_results.jsonl` | Every row has method, feeder, scenario_id, seed_idx, metric values, feasibility status |

## Atomic Checklist

### Cut A - Minimum Viable Evidence Path

1. **Dependency gate**
   - Action: install/lock `simbench`, `pandapower`, `networkx`, `torch`, `pyyaml`; create the `tests/` skeleton if only `.gitkeep` exists.
   - Artifact: lock note in `docs/09_repro_and_env.md` or `requirements-lock.txt`.
   - PASS: `python -c "import pandapower, simbench, networkx, torch, yaml"` succeeds and versions are logged.
   - FAIL: any required package missing.

2. **Feeder enumeration**
   - Action: enumerate available SimBench codes and choose two rural LV train feeders plus one urban LV OOD feeder.
   - Artifact: `configs/topology_split_v1.yaml` rewritten with actual SimBench codes and `(radial, feeder_type)` labels.
   - PASS: every listed feeder has a callable loader and class provenance.
   - FAIL: any code is placeholder, unavailable, or label is hand-edited without ADR.

3. **case33bw loader validation**
   - Action: run `pandapower.networks.case33bw()`, build graph, count buses/loads, run base power flow.
   - Artifact: `data/raw/pandapower/case33bw/metadata.json`.
   - PASS: load succeeds, base power flow converges, metadata matches protocol count convention.
   - FAIL: no convergence or count mismatch without explanation.

4. **SimBench loader validation**
   - Action: load the three selected SimBench feeders with `simbench.get_simbench_net(code)`.
   - Artifact: `data/raw/simbench/<code>/metadata.json` for each.
   - PASS: each net loads, includes `net.profiles`, has loads, and graph is connected.
   - FAIL: missing profiles or disconnected graph.

5. **IEEE 123 conversion decision**
   - Action: download/pin OpenDSS IEEE 123 files; implement or select converter.
   - Artifact: `data/raw/opendss/ieee123/SOURCE.txt`, conversion script, processed metadata.
   - PASS: converted graph passes convergence/radiality/load-preservation gates; total real load after phase aggregation is within 5% of the OpenDSS base-snapshot total real load.
   - FAIL: conversion fails; trigger ADR fallback before training.

6. **IEEE 34 candidate decision**
   - Action: download/pin OpenDSS IEEE 34 files; attempt same conversion gates.
   - Artifact: `data/raw/opendss/ieee34/SOURCE.txt`, conversion metadata or fallback ADR.
   - PASS: validated IEEE 34 or validated fallback occupies third OOD slot.
   - FAIL: no validated IEEE 34 and no validated fallback occupies the third OOD slot. Full v1 is blocked; only a separately defined stress-test ADR may proceed, and it cannot claim C2 topology transfer.

7. **Node-attribute freeze**
   - Action: generate `c_i`, `m_i`, `w_i` for every feeder node.
   - Artifact: `configs/feeder_attributes_v1.yaml`.
   - PASS: all nodes covered; default critical `m_i=0.8`; weights documented; no OOD-based tuning.
   - FAIL: missing node attributes or non-reproducible assignment.

8. **Scenario generator v1**
   - Action: implement seeded scenario generation with persistent outage process, budget ratio, demand scaling, and split labels.
   - Artifact: `data/scenarios/v1/*.jsonl`; generator config.
   - PASS: each training feeder's scenarios are split 70/15/15 into train/validation/test; OOD feeders have test scenarios only, and OOD test scenarios are never used for hyperparameter selection or early stopping.
   - FAIL: OOD appears in training/validation or scenario IDs are not reproducible.

9. **Workload audit**
   - Action: compute realized demand-scale distributions, `rho_t`, and outage rates across train/OOD.
   - Artifact: `results/v1/audits/workload_audit.json`, `results/v1/run_manifest.json`, and launcher check in `scripts/launch_full_run.py`.
   - PASS: audit and manifest are produced before the first claim-bearing training launch; manifest contains workload-audit file hash, split hash, scenario-config hash, and audit creation timestamp. If the inline workload reframing trigger fires, C2 is downgraded before results are interpreted.
   - FAIL: missing audit, missing manifest hashes, or audit generated after the first claim-bearing run.

10. **Projection feasibility tests**
    - Action: add adversarial unit tests for box-plus-budget projection.
    - Artifact: `tests/test_projection.py`.
    - Test runner: `pytest tests/test_projection.py`.
    - PASS: budget, cap, non-negativity, and outage constraints pass at `epsilon_feas = 1e-6` on normalized power units, meaning all feeder power quantities are divided by total nominal demand.
    - FAIL: any violation in non-ablation method.

11. **Metric implementation**
    - Action: implement `critical_load_adequacy`, `temporal_continuity`, outage-inclusive adequacy, `transfer_gap`, and `per_feeder_gap`.
    - Artifact: `src/sg_resilience/metrics_v1.py`, tests.
    - Test runner: `pytest tests/test_metrics_v1.py`.
    - PASS: metrics match the inline definitions above; temporal continuity uses each critical node's adequate-service binary sequence, then averages over critical nodes.
    - FAIL: legacy `critical_continuity_ratio` appears in paper-facing tables.

12. **Canonical result schema**
    - Action: define one row schema for every method/feeder/scenario/seed metric.
    - Artifact: `schemas/canonical_results_v1.schema.json` and writer.
    - PASS: all report scripts consume only this schema.
    - FAIL: hand-typed tables or method-specific result formats.

13. **Smoke run**
    - Action: run end-to-end on `case33bw`, one seed, reduced epochs, across every claim-bearing method in the V1 subset.
    - Artifact: `results/v1/smoke/case33bw_smoke.json`.
    - Hyperparameters: use `configs/smoke_case33bw.yaml` plus method-default smoke settings; these are not claim-bearing hyperparameters and do not replace the Cut B freeze.
    - PASS: finishes under CPU smoke target of <10 minutes; feasibility zero; result schema valid.
    - FAIL: before any full training.

### Cut B - Full Workshop Sweep

14. **Baseline suite**
    - Action: implement/verify priority rule, continuity-aware LP/QP, GraphSAGE/current, GCN, GAT, GIN, and DeepSets.
    - Artifact: method registry/config.
    - PASS: learned baselines consume the same feature extractor, projection, loss, and training budget; non-learned baselines use the same scenario inputs, node attributes, projection/feasibility checks where applicable, and evaluation metrics.
    - FAIL: any baseline gets unequal data or tuned OOD settings.

15. **Graph ablations**
    - Action: implement degree-preserving radial rewire, edge-feature zeroing, and feature permutation.
    - Artifact: ablation configs and tests.
    - PASS: rewire remains connected radial tree and passes audit; edge-feature zeroing runs; feature permutation keeps topology fixed and permutes node features within each feeder.
    - FAIL: invalid rewire silently used.

16. **Hyperparameter freeze**
    - Action: tune only on train/validation scenarios within 40 A100-hour budget, using the inline model-selection rule.
    - Artifact: `configs/hparams_v1.yaml`, tuning log.
    - PASS: one shared config per method, validation metric and tiebreakers logged, no OOD use. The 40 A100-hour cap applies to this Cut B tuning stage only.
    - FAIL: per-OOD or per-feeder tuning.

17. **Claim-bearing training**
    - Action: train the enumerated v1 method subset, five seeds, frozen split; `scripts/launch_full_run.py` must refuse launch unless the item 9 manifest check passes; produce the pilot/MDE note before final claim labels.
    - Artifact: `results/v1/runs/<method>/<seed>/...`.
    - PASS: all runs complete or are marked failed with reason; if intervals cannot resolve five-point H1 effects, H1 is labeled inconclusive.
    - FAIL: missing run without reason.

18. **OOD evaluation**
    - Action: evaluate every trained method on in-family test and all OOD feeders.
    - Artifact: canonical result bundle.
    - PASS: every method/seed/feeder/scenario tuple has feasibility status and metrics.
    - FAIL: missing tuples not accounted by protocol.

19. **Physics audit**
    - Action: run pandapower audit sample per reported method/feeder/seed.
    - Artifact: `results/v1/audits/physics_audit.json`.
    - PASS: convergence/failure counts reported; failed rollout not silently dropped.
    - FAIL: no audit for reported method.

20. **Statistical report**
    - Action: compute H1/H2 point gaps, paired bootstrap CIs, drop fractions, and labels.
    - Artifact: `results/v1/reports/statistical_summary.json`.
    - PASS: H1/H2 labels follow protocol; no hidden gates.
    - FAIL: table label cannot be traced to formula.

21. **Table generation**
    - Action: implement `scripts/report_main.py`, `scripts/report_transfer.py`, `scripts/report_ablations.py`, `scripts/report_feasibility.py`.
    - Artifact: `paper/tables/*.tex`.
    - PASS: tables generated from canonical result bundle only.
    - FAIL: hand-edited result values.

22. **Claim audit**
    - Action: compare generated outcomes against `docs/10` and `docs/05`.
    - Artifact: `results/v1/reports/claim_audit.md`.
    - PASS: every claim has data row, metric, threshold, and caveat.
    - FAIL: any claim uses v2 structural-distance association.

## Parallelization

Parallelizable:

- Data-source verification for pandapower, SimBench, OpenDSS.
- Report-schema design while loaders are being validated.
- Unit tests for projection/metrics while feeder conversion is being attempted.

Serialized:

- Freeze split before scenario generation.
- Freeze node attributes before training.
- Freeze hyperparameters before OOD evaluation.
- Run claim audit only after canonical results and reports exist.

## Cut Point Recommendation

Start with Cut A. Do not launch any multi-seed training until items 1-13 pass.
Cut B is blocked unless LV-urban, IEEE 123, and the third OOD slot, either IEEE
34 or an ADR-selected fallback, pass validation.

## Open Risks

- SimBench is not currently installed in the local environment.
- IEEE 34/123 conversion is the largest data risk.
- Current `topology_split_v1.yaml` still uses old class labels and placeholder loader assumptions.
- Existing training metrics still use legacy `critical_continuity_ratio`; metric implementation must be updated before claim-bearing runs.
- No tests exist beyond `.gitkeep`.
- Flow-aware MP/C1, V2 feeder atlas, `d_struct`, structural-distance association, and main-conference claims require separate plans after Cut B.
