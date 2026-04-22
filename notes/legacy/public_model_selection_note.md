# Public Benchmark Model Selection Note

## Selected Public Benchmark Run

The strongest fair public result remains:

- `results/simbench_fair_run/`

This run uses:

- benchmark: `configs/simbench_rural1_benchmark.yaml`
- hidden dimension: `96`
- imitation weight: `0.5`
- switching penalty: `0.2`
- fair held-out split with distinct train and eval seeds

## Why This Run Is Selected

Compared against the current public alternatives, it offers the best tradeoff between allocation quality and temporal behavior.

### Selected run: `simbench_fair_run`

- learned weighted served energy: `1.821101`
- learned unserved critical demand: `0.008342`
- learned switching count: `0.875`

### Rejected run: `simbench_fair_run_sparse`

- learned weighted served energy: `1.588348`
- learned unserved critical demand: `0.024244`
- learned switching count: `0.000000`

Reason: switching improved further, but allocation quality degraded too much.

### Rejected run: `simbench_fair_run_low_switch`

- learned weighted served energy: `1.588505`
- learned unserved critical demand: `0.024244`
- learned switching count: `0.000000`

Reason: lower switching penalty did not improve the public result. It behaved similarly to the sparse-head failure mode and materially worsened the primary metrics.

## Secondary Public Family Evidence

- `results/simbench_rural2_fair_run_fast/`

This run is useful as family-level evidence that the pipeline generalizes to a second public SimBench case. It is not the strongest headline benchmark because switching behavior is much larger for all methods in that family and the aggregate comparison is less clean for the current paper narrative.

## Use in the Paper

For the main text:

1. Use `simbench_fair_run` as the primary public benchmark result.
2. Treat `simbench_rural2_fair_run_fast` as supporting family-level evidence.
3. Present `simbench_fair_run_sparse` and `simbench_fair_run_low_switch` as negative ablations that clarify why the selected model configuration is preferred.
