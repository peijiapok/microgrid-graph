# Public Benchmark Comparison Note

## Direct Comparison Standard

All direct claims in this project are based on re-run baselines under identical conditions:

- same benchmark case
- same held-out split
- same four core metrics
- same rule / optimization / learned evaluation pipeline

The four core metrics are:

- `weighted_served_energy`
- `unserved_critical_demand`
- `switching_count`
- `critical_continuity_ratio`

## Public Benchmark Coverage So Far

### SimBench Rural 1

- Config: `configs/simbench_rural1_benchmark.yaml`
- Result root: `results/simbench_fair_run/`
- This is currently the strongest public benchmark evidence because it uses a fair held-out protocol and produces a stable aggregate comparison.

Headline pattern:

- Learned is slightly behind on allocation-quality metrics.
- Learned is clearly better on switching.
- This mirrors the held-out synthetic result and therefore supports the core continuity claim on public data.

### SimBench Rural 2

- Config: `configs/simbench_rural2_benchmark.yaml`
- Result root: `results/simbench_rural2_fair_run_fast/`
- This acts as family-level evidence that the behavior of the controller is not unique to one public LV case.

Headline pattern:

- Learned remains competitive on weighted served energy.
- Switching is much higher in this benchmark family for all methods.
- This benchmark is useful as a second public case, but SimBench Rural 1 is currently cleaner for headline presentation.

## Mapping to Prior Baseline Paper Families

### Priority-Aware Restoration Papers

- Closest benchmark family in our current pipeline: `SimBench` + held-out stress adaptation
- Closest metric mapping:
  - restored critical service -> `unserved_critical_demand`
  - restoration quality -> `weighted_served_energy`
- Current claim:
  - We can compare our own re-run baselines fairly under a public benchmark family.
  - We should not yet claim one-to-one replication of every IEEE/OpenDSS restoration study.

### Outage-Aware RL / Graph RL Papers

- Closest benchmark family in our current pipeline: held-out synthetic stress + SimBench
- Closest metric mapping:
  - energy not supplied -> `unserved_critical_demand`
  - stability / sequential quality -> `switching_count`
- Current claim:
  - Our learned controller now approaches non-learned baselines on allocation quality while strongly reducing switching.
  - We do not yet claim native equivalence to full RL benchmark environments such as Grid2Op leaderboards.

### Robust / Stochastic Power Control Papers

- Closest benchmark family in our current pipeline: `pandapower` and `SimBench` public cases
- Closest metric mapping:
  - operational quality -> `weighted_served_energy`
  - unmet service -> `unserved_critical_demand`
- Current claim:
  - The optimization baseline already provides a strong local teacher and fair comparator.
  - We should not claim direct parity with OPF-cost or feasibility papers until those metrics are added explicitly.

## Reviewer-Safe Statement

The safe paper statement is:

> Direct comparisons in this work are made only against re-run baselines under identical held-out conditions. Results reported in prior papers are used as contextual references unless full benchmark and metric parity is established.

## Immediate Use in the Paper

This note supports two manuscript moves:

1. In the main results section, use our re-run public benchmark tables as the primary evidence.
2. In the related-work or discussion section, describe prior-paper benchmark families as comparison targets rather than as directly beaten systems unless full parity is achieved.
