# Paper Comparison Execution Plan

## Target Baseline Papers and What They Actually Defend

### 1. Critical-Load Restoration / Priority-Aware Resilience

- Anchor papers: `Gao et al. (IEEE TSG 2016)`, `Aghajan et al. (2024)`
- What they implicitly defend:
  - critical-load preservation
  - priority-aware load shedding / restoration quality
  - resilience under constrained operation
- Our matching benchmark families:
  - held-out synthetic stress family
  - public `SimBench` rural and urban cases
- Our matching metrics:
  - `unserved_critical_demand`
  - `weighted_served_energy`
  - `critical_continuity_ratio`

### 2. Outage-Aware Learning / Graph Control

- Anchor papers: `Jacob et al. (Nature Communications 2024)` and graph-control literature summarized in `hassouna2024`
- What they implicitly defend:
  - sequential robustness under topology stress
  - graph-aware operating policy quality
  - outage management under repeated decisions
- Our matching benchmark families:
  - public `SimBench` urban benchmark
  - stress sweep under supply collapse and outage persistence
- Our matching metrics:
  - `unserved_critical_demand`
  - `switching_count`
  - regime-specific `weighted_served_energy`

### 3. Robust / Stochastic Control

- Anchor paper: `Lorca et al. (Operations Research 2016)`
- What it defends:
  - strong operational behavior under uncertainty
- Our matching benchmark families:
  - same public cases with held-out stress generation
- Our matching metrics:
  - `weighted_served_energy`
  - `unserved_critical_demand`
  - stress degradation curves

## What We Already Have

### Strongest Existing Evidence

- Held-out synthetic near-tie with zero switching:
  - `results/fair_protocol_run_hd96/`
- Public rural near-tie with much lower switching:
  - `results/simbench_fair_run/`
- Public urban Pareto set:
  - `results/urban_pareto_tradeoff/`
- Extreme-stress regime win:
  - `results/urban_stress_sweep/`

### Most Important Specific Win

In the urban public stress sweep at supply collapse `35%`, the learned method beats both rule and optimization baselines on both weighted served energy and unserved critical demand.

This is currently the single best regime-specific result to emphasize against baseline papers, because it directly supports the claim that learned resilience control becomes valuable when stress is severe enough that simple priority heuristics stop being adequate.

## Recommended Comparison Structure in the Paper

### Main Paper

1. Held-out synthetic comparison
2. Public rural SimBench comparison
3. Urban Pareto frontier
4. Extreme stress sweep with explicit callout of the `35%` supply-collapse regime win

### Appendix / Supplementary

1. Second rural public family evidence (`simbench_rural2_fair_run_fast`)
2. Negative ablations (`sparse`, `low-switch`, weak graph-policy runs)
3. Physics spot-check note

## Concrete Next-Run Recipe for Stronger Paper-to-Paper Positioning

The next experiment cycle should not be a broad search. It should be a narrow attempt to turn the urban stress-regime advantage into a stronger paper-level claim.

### Run A: More urban data for the current best learned family

- same benchmark: `configs/simbench_lv_urban_benchmark.yaml`
- increase train random variants from `24` to `64` or `96`
- keep held-out test seed fixed
- keep main metric contract fixed
- goal: determine whether the current learned family improves under greater urban data diversity

### Run B: Stress-sweep-first training selection

- select model on a weighted score over urban stress dev scenarios, not only average dev scenarios
- goal: favor methods that win in the extreme-stress regime that matters for the paper

### Run C: Public rural + urban side-by-side final figure set

- keep the same core metrics
- present where learned is near-tied, where it forms a Pareto set, and where it outright wins under stress

## Reviewer-Safe Position

The paper should not claim universal dominance over all prior work. It should claim the following:

1. direct comparisons are re-run under identical conditions,
2. the learned method is competitive on public benchmarks,
3. the learned method yields a meaningful Pareto operating set in urban settings,
4. the learned method becomes superior in at least one severe stress regime.
