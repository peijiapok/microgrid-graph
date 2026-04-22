# Baseline Dataset and Metric Attack Map

## What We Can Compare Directly Right Now

The current pipeline supports direct and fair comparison when all three methods are evaluated under the same locally generated or adapted benchmark scenarios, the same held-out split, and the same four core metrics:

- `weighted_served_energy`
- `unserved_critical_demand`
- `switching_count`
- `critical_continuity_ratio`

This means the strongest direct comparison claim in the current manuscript should be against our own re-run baselines, not against reported numbers copied from prior papers.

## Baseline Paper Families and How They Connect to Our Current Benchmarks

### 1. Priority-Aware Restoration / Critical Load Restoration

- Representative references: critical-load restoration and priority-based load-shedding papers using IEEE distribution feeders and restoration scenarios.
- Typical benchmark families: IEEE 13-bus, IEEE 34-bus, IEEE 123-bus, OpenDSS-based restoration cases.
- Typical win metrics: restored active power, energy not supplied, critical-load restoration, restoration time.
- Our current nearest comparison path:
  - `SimBench` public LV benchmark adaptation
  - `pandapower` benchmark adaptation
  - same priority-weighted resilience metrics
- Current claim we can make:
  - We reproduce the spirit of priority-aware restoration comparison under a fair held-out protocol, but we do not yet claim one-to-one reproduction of every feeder-specific restoration environment.

### 2. Outage-Aware Learning-Based Distribution Operation

- Representative references: graph RL or multi-agent RL for outage management in active distribution networks.
- Typical benchmark families: modified IEEE 13-bus, 34-bus, 123-bus feeders with DERs and switching actions.
- Typical win metrics: energy not supplied, cumulative reward, restored load, voltage violations.
- Our current nearest comparison path:
  - fair held-out synthetic stress family
  - `SimBench` rural LV benchmark
  - same controller vs rule/optimization structure
- Current claim we can make:
  - Our learned controller now approaches strong non-learned baselines on allocation quality while sharply reducing switching.
  - We do not yet claim native equivalence to the full RL environment metrics in those papers.

### 3. Robust/Stochastic Power Control

- Representative references: robust OPF, chance-constrained OPF, stochastic AC or DC OPF.
- Typical benchmark families: IEEE 14-bus, 30-bus, 57-bus, 118-bus, MATPOWER or pandapower systems.
- Typical win metrics: expected cost, feasibility, constraint satisfaction, violation rates, runtime.
- Our current nearest comparison path:
  - public pandapower standard cases
  - same held-out stress protocol
  - same allocation-quality metrics
- Current claim we can make:
  - Our optimization baseline already provides a strong teacher and fair direct reference under our metric contract.
  - We should not claim direct comparison to OPF-cost papers until we add their cost/feasibility metrics.

### 4. Sequential Public Control Benchmarks

- Representative references: Grid2Op / L2RPN and PowerGym families.
- Typical benchmark families: IEEE 118-derived transmission tasks, Volt-Var control environments, sequential grid control tasks.
- Typical win metrics: robustness score, overload avoidance, voltage violations, leaderboard score, control cost.
- Our current nearest comparison path:
  - none yet in native environment terms
- Current claim we can make:
  - These papers are relevant as motivation and future benchmark targets, but not yet direct benchmark peers under identical conditions.

## Reviewer-Safe Claim Boundary

The current work can safely claim the following:

1. The controller is compared fairly against rule and optimization baselines on identical held-out scenario splits.
2. The learned controller nearly matches the strongest baselines on allocation-quality metrics while strongly reducing switching.
3. This pattern holds on both a held-out synthetic stress family and at least one public benchmark family (`SimBench`).

The current work should not yet claim the following:

1. Direct superiority over every paper that used IEEE/OpenDSS/Grid2Op environments.
2. Native equivalence to full RL environment leaderboards.
3. Cross-topology generalization beyond the current benchmark families.

## Immediate Next Comparison Targets

1. Add a second `SimBench` case and report family-level consistency.
2. Add one classic `pandapower` standard case with the same fair protocol.
3. Only then start mapping the public-benchmark figures to specific paper-level comparison paragraphs.

## Manuscript Use

This note should support two statements in the paper:

- Direct comparison claims are based on re-run baselines under identical conditions.
- Reported numbers from prior work are used only as contextual references unless full experimental parity is established.
