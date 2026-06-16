# 11 - Current Progress Summary

Last updated: 2026-06-16 (was 2026-05-13)

> **2026-06-16 reframing + progress.** Direction changed (ADR-0001): primary
> objective/metric is **temporal continuity `C`** (not adequacy); feasible set is
> **flow-constrained `Delta_grid`**. Current state lives in
> `notes/progress_2026-06-16.md`; decisions in `notes/adr/0001-continuity-primary.md`
> and `notes/adr/0002-v1-feeder-split.md`. The §3/§4/§5/§7 formulas below are
> superseded by `docs/10` (continuity-primary, `Delta_grid`); §8/§10 updated in
> place. Built since: topology/flow foundation, reference projection, flow oracle,
> continuity metric, Markov outages, continuity loss, loader fix — 20 tests pass.

This file summarizes the current research state. It is a project brief, not
paper evidence. The source-of-truth technical documents remain
`docs/05_experimental_protocol.md`, `docs/10_research_formulation.md`, and
`dev_planning/2026-05-09_atomic_research_execution_plan.md`.

## 1. Current Research Direction

The defensible research setting is:

> Constrained scarce-resource allocation on distribution-feeder graphs under
> structural distribution shift.

The operational motivation is emergency or scarce-power operation on
distribution feeders, where critical loads must be served under limited
available power and stochastic outages. The formal claim is narrower than
"catastrophic grid control": the current formal setting is scarce-budget
allocation under stochastic outages.

The project should not currently claim full physics-aware grid control. The
learning loop does not solve AC/DC power flow as part of the optimization.
Use "projection-feasible graph policy" or "flow-feature-aware graph policy"
only when the implementation actually consumes edge/flow features.

## 2. Problem Statement

Given a family of attributed distribution-feeder graphs and a constrained
per-step allocation problem, train one policy on a set of training feeders. The
policy must allocate power on held-out feeder topologies without retraining or
fine-tuning, while always respecting hard feasibility constraints.

Short version:

> Can one feasible graph policy trained on some feeder topologies allocate
> scarce power on unseen feeder topologies, and is graph structure necessary for
> that transfer?

The v1 question is not whether a structural-distance metric predicts transfer
gap. That is deferred to v2 after enough train-test topology pairs exist.

## 3. Mathematical Formulation

A rollout sample is:

```text
(G, xi) ~ D
G = (V, E, X_V, X_E, Z)
xi = (d_{1:T}, P_{1:T}, O_{1:T})
```

where:

- `V` is the feeder node set.
- `E` is the feeder edge set.
- `X_V` and `X_E` are node and edge features.
- `Z` contains frozen node attributes such as critical flag `c_i`, minimum
  service fraction `m_i`, and service weight `w_i`.
- `d_t` is node demand.
- `P_t` is available global power.
- `O_t` is the outage mask.

The controller outputs raw scores, then projection enforces feasibility:

```text
z_t = pi_theta(G, s_t, a_{t-1})
a_t = P_Delta(z_t)
```

The feasible set is:

```text
Delta(G, s_t) = {
  a in R_+^n :
    0 <= a_i <= d_{t,i},
    a_i = 0 if i in O_t,
    sum_i a_i <= P_t
}
```

The projection target is the Euclidean projection onto the box-plus-budget
polytope:

```text
P_Delta(z_t) = argmin_{a in Delta(G, s_t)} ||a - z_t||_2^2
```

Feasibility is a hard invariant only after projection tests pass. Do not claim
feasibility from soft penalties.

## 4. Objective Function

For scenario `xi`, use rollout return:

```text
J(pi; G, xi) =
    sum_t sum_i w_i a_{t,i}
  - lambda_c sum_t sum_{i in C \ O_t} max(m_i d_{t,i} - a_{t,i}, 0)
  - lambda_s sum_{t=2}^T ||a_t - a_{t-1}||_1
```

Training minimizes:

```text
theta* = argmin_theta E_{G ~ D_train} E_{xi ~ S_G}
          [ -J(pi_theta; G, xi) ]
```

Hard allocation constraints belong in the projection, not in the loss.

## 5. Current Hypotheses

### H1 - Graph Structure Is Necessary

The graph policy must outperform:

- a no-graph size-generalizing baseline such as DeepSets;
- a wrong-graph degree-preserving rewire baseline.

The primary H1 metric is OOD `critical_load_adequacy`. Temporal continuity is
secondary and cannot rescue a failed adequacy result.

### H2 - Zero-Shot Held-Out Feeder Transfer Is Viable

A single policy trained on `D_train` should retain useful service quality on
`D_ood` without retraining. The v1 transfer gate uses per-feeder
`critical_load_adequacy` transfer gaps.

The current threshold is:

```text
Delta_transfer = 0.20
```

This is a loose workshop viability threshold, not a strong generalization
claim.

### H3 - Structural Distance Predicts Transfer Gap

Deferred to v2. V1 has too few held-out topology pairs for a credible
structural-association claim.

## 6. Claim-Bearing Method Set For V1

Current v1 claim-bearing methods:

- GraphSAGE/current implementation.
- GCN.
- GAT.
- GIN.
- DeepSets no-graph baseline.
- Priority allocator.
- Continuity-aware LP/QP.

Flow-aware MP / C1 is deferred from the v1 atomic path. It should only be added
after Cut A passes and after a task-anchored edge-feature ablation protocol is
written.

## 7. Metrics And Gates

Primary metric:

```text
critical_load_adequacy =
  mean over non-outaged critical node-time pairs of
  1[a_t,i >= m_i d_t,i]
```

Secondary temporal metric:

```text
temporal_continuity =
  mean over critical nodes of max_uninterrupted_run_of_ones / T
```

Transfer gap:

```text
Gap_M =
  train_mean(M) - ood_mean(M)       for higher-is-better metrics
  ood_mean(M) - train_mean(M)       for lower-is-better metrics
```

Positive gap means OOD performance is worse.

Feasibility gate:

- zero budget violations;
- zero demand-cap violations;
- zero non-negativity violations;
- zero outage-mask violations;
- tolerance `epsilon_feas = 1e-6` after normalizing by feeder total nominal
  demand.

Workload reframing trigger:

- demand-scale or `rho_t` normalized mean shift `>= 0.1` pooled training
  standard deviations; or
- outage-rate difference `>= 0.02`.

If this trigger fires, C2 becomes a distribution-shift transfer probe rather
than a topology-transfer probe.

## 8. Data And Split Status

The original intended v1 split was:

- train: `case33bw`;
- train: two SimBench rural feeders;
- OOD: SimBench urban feeder;
- OOD: IEEE 123;
- OOD: IEEE 34 or fallback.

Current blocker: the easy pure-LV SimBench split is not credible.

Observed SimBench candidates:

| Code | Role considered | Buses | Loads | Status |
|---|---:|---:|---:|---|
| `1-LV-rural1--0-sw` | train candidate | 15 | 13 | Reject; too small for serious training evidence |
| `1-LV-rural2--0-sw` | train candidate | 97 | 99 | Plausible |
| `1-LV-rural3--0-sw` | train candidate | 129 | 118 | Plausible |
| `1-LV-urban6--0-sw` | OOD candidate | 59 | 111 | Connected, but not larger than rural3 |
| `1-MVLV-urban-5.303-0-sw` | larger OOD candidate | 254 | 242 | Connected under pandapower topology |
| `1-MVLV-urban-6.305-0-sw` | larger OOD candidate | 202 | 249 | Connected under pandapower topology |
| `1-MVLV-urban-6.309-0-sw` | larger OOD candidate | 202 | 249 | Connected under pandapower topology |

Recommended split direction:

- reject `1-LV-rural1--0-sw`;
- use `1-LV-rural2--0-sw` and `1-LV-rural3--0-sw` as the SimBench rural train
  feeders;
- replace the pure LV-urban OOD slot with a larger validated MVLV urban OOD
  feeder, or explicitly revise the v1 claim scope.

This means the empirical claim should be phrased as limited radial distribution
held-out-feeder transfer, not pure LV-to-LV transfer.

## 9. Progress Completed

Completed so far:

- Reframed the research away from overbroad catastrophe/physics claims.
- Wrote the current research formulation in `docs/10_research_formulation.md`.
- Wrote the experimental protocol in `docs/05_experimental_protocol.md`.
- Wrote the atomic execution checklist in
  `dev_planning/2026-05-09_atomic_research_execution_plan.md`.
- Installed and verified the documented SimBench optional dependency.
- Recorded dependency gate versions in `requirements-lock.txt`.
- Created a non-empty `tests/` skeleton with `tests/README.md`.
- Enumerated SimBench feeder candidates and found the current split blocker.

Verified dependency versions:

```text
pandapower==3.4.0
simbench==1.6.2
networkx==3.6.1
torch==2.6.0+cu124
PyYAML==6.0.3
```

## 10. Current Blockers (status 2026-06-16)

1. ~~**Feeder split is not frozen.**~~ **RESOLVED** — frozen split in
   `configs/topology_split_v1.yaml` (ADR-0002), validated by
   `scripts/validate_feeders_v1.py`.

2. ~~**Pure LV split is weak.**~~ **RESOLVED** — train = case33bw + LV-rural2/3;
   OOD adds two MVLV-urban feeders (larger, radial) for size-gen. rural1 rejected.

3. ~~**IEEE 123 / IEEE 34 conversion.**~~ **DEFERRED to v2** — MVLV-urban gives a
   validated larger radial OOD without OpenDSS conversion risk.

4. **Metrics — PARTIAL.** Primary continuity metric `C` + diagnostics implemented
   in `src/sg_resilience/metrics_v1.py`; transfer_gap implemented. Still to wire
   into a full report harness.

5. ~~**Projection and metric tests missing.**~~ **PARTIAL** — `tests/test_metrics_v1.py`,
   `tests/test_topology.py`, `tests/test_flow_projection.py`, `tests/test_outages_and_loss.py`
   (20 tests). Differentiable C3 projection test pending Fanchen's implementation.

6. **Canonical result schema** still to do (`schemas/canonical_results_v1.schema.json`).

7. **NEW (Fanchen):** differentiable `Delta_grid` projection (C3) + flow-aware
   operator (C1) — work order at `notes/work_order_fanchen_2026-06-16.md`.

8. **NEW (Jia):** case33bw physical line ratings (placeholder caps don't bind);
   full docs/01-02-05-06-10 already reframed.

## 11. Next Atomic Actions

Recommended next steps:

1. Write an ADR revising the v1 feeder split.
2. Rewrite `configs/topology_split_v1.yaml` with actual validated feeder codes.
3. Validate `case33bw` metadata and base power flow.
4. Validate selected SimBench feeders with `simbench.get_simbench_net`.
5. Decide whether IEEE 123/34 conversion is feasible or whether fallback
   feeders are needed.
6. Implement `metrics_v1.py` and tests.
7. Implement projection adversarial tests.
8. Define the canonical result schema.
9. Only after Cut A passes, run any multi-seed training.

## 12. Claims To Avoid

Do not currently claim:

- broad topology generalization;
- structural-distance prediction of transfer gap;
- physics-aware control;
- catastrophe simulation;
- reinforcement learning, unless the training loop actually optimizes through
  environment interaction;
- benchmark status, until loaders, schemas, tests, and report generators are
  reproducible.

The honest claim target is:

> A reproducible evaluation suite for projection-feasible graph policies under
> limited radial distribution feeder transfer, with graph-necessity tests and
> explicit workload/feasibility audits.

