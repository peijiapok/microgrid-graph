# Activity-Loss Benchmark Regeneration Summary

This report summarizes the regenerated benchmark runs after the training/generalization update in `src/sg_resilience/training.py`.

## Training changes applied

- worst-case-aware checkpoint selection
- stress-aware scenario weighting
- per-epoch teacher refresh with scheduled switching penalty
- `SmoothL1` imitation loss
- critical continuity-drop penalty
- critical activity penalty aligned to the continuity metric

## Verification status

- full test suite: `30 passed`
- modified Python files: no LSP errors
- completed regenerated runs include comparison, training metrics, physics evaluation, and stress sweeps for the configs listed below

## Regenerated benchmark results

| Config | Output directory | Learned weighted served | Learned unserved critical | Learned switching | Learned continuity | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `configs/simbench_lv_urban_benchmark.yaml` | `results/simbench_lv_urban_fair_run_activityloss` | 1.799338 | 0.009600 | 79 | 0.968750 | Major improvement over prior smoke run (`0.136110`, `0.110600`, `30`, `0.0`) |
| `configs/case33bw_benchmark.yaml` | `results/case33bw_fair_run_activityloss` | 81.297321 | 0.000000 | 0 | 1.000000 | Learned preserves critical load perfectly on benchmark comparison |
| `configs/simbench_rural1_benchmark.yaml` | `results/simbench_rural1_fair_run_activityloss` | 1.568336 | 0.000000 | 2 | 1.000000 | Learned matches perfect critical continuity |
| `configs/simbench_rural2_benchmark.yaml` | `results/simbench_rural2_fair_run_activityloss` | 0.679320 | 0.000000 | 33 | 1.000000 | Learned preserves critical load with much lower switching than baselines |

## Physics evaluation summary

All completed regenerated runs converged in physics evaluation.

- `simbench_lv_urban_fair_run_activityloss/physics_eval.json`: converged ratios 1.0 across baselines on the compared scenario.
- `case33bw_fair_run_activityloss/physics_eval.json`: converged ratios 1.0 across baselines.
- `simbench_rural1_fair_run_activityloss/physics_eval.json`: converged ratios 1.0 across baselines.
- `simbench_rural2_fair_run_activityloss/physics_eval.json`: converged ratios 1.0 across baselines.

## Stress-sweep summary

- **LV urban:** learned continuity is mostly `0.875` to `1.0` across stress families, with large gains over the earlier collapsed behavior.
- **case33:** learned continuity remains `1.0` across listed stress scenarios.
- **rural1:** learned continuity is `1.0` throughout almost all listed stress scenarios, with only one mild degradation to `0.96875` under a supply-collapse case.
- **rural2:** learned continuity is `1.0` throughout the listed stress scenarios.

## Incomplete benchmark due to runtime ceiling

The following config was attempted multiple times but did not finish within the environment wall-clock budget:

- `configs/simbench_urban_benchmark.yaml`

Attempts made:

1. bounded full `run_fair_experiment.py` run
2. fallback manual training + lean comparison path

Both attempts timed out without evidence of code failure, so this is currently classified as an **environment/runtime limitation**, not a known bug in the improved training pipeline.

## Recommended next step

Run `configs/simbench_urban_benchmark.yaml` on higher-budget compute or with a longer wall-clock allowance to confirm that the same trainer improvements scale to the largest MVLV urban benchmark.
