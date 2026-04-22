# Physics Validation Note

## Current Scope

The current controller and baseline comparison pipeline is still an abstract allocation framework. It does not yet enforce full network physics inside training or inside the optimization baseline itself. However, a post hoc physical feasibility check is now available for public benchmark runs using `pandapower`.

## Rural SimBench Spot Check

Artifact:

- `results/simbench_fair_run/physics_eval_spotcheck_gs.json`

What was checked:

- first 4 held-out scenarios
- first 4 time steps per scenario
- all three baselines (`rule`, `optimization`, `learned`)

Observed result:

- all sampled steps converged after adding a practical `pandapower` fallback chain ending in `gs`
- representative loading levels were below `90%`
- sampled voltage magnitudes were in a narrow range around `1.025` to `1.032` p.u.

## Interpretation

This does **not** mean the current method is fully physics-faithful.

It does mean the following:

1. The selected rural public benchmark allocations are at least compatible with a successful power-flow solve in the sampled cases.
2. The benchmark adapter is no longer purely conceptual; it now supports an actual physical feasibility audit.
3. Physical validation should be reported as a `spot-check` or `post hoc feasibility audit`, not as full physics-constrained control.

## Urban Benchmark Status

The urban SimBench case has been integrated and evaluated under the same fair protocol:

- `results/simbench_lv_urban_fair_run/`

The current learned controller underperforms strongly on this urban benchmark. This is useful evidence, because it shows that the current method is not yet robust across denser public urban settings. The correct paper interpretation is that urban public benchmarks remain a hard target and an important next step.
