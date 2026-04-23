"""Smoke test for the sg_resilience pipeline.

Contract (docs/09_repro_and_env.md §3):
    python -m sg_resilience.smoke

must run end-to-end on CPU in under 10 minutes and return exit 0 on
success. On failure, it prints a diagnostic and exits non-zero.

The script:
  1. Loads case33bw via the public-benchmark loader.
  2. Generates a tiny scenario set (persistent variants + 2 random).
  3. Trains the current compact GraphSAGE controller for a handful of epochs.
  4. Evaluates on the base scenario and prints the four core metrics.
  5. Asserts feasibility exactly: 0 <= a_i <= d_i and sum(a) <= P, per step.

This is intentionally small. Anything larger belongs in a proper
benchmark run under `results/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

from .benchmark_loader import load_public_benchmark_scenario
from .scenario_generator import (
    generate_persistent_stress_variants,
    generate_random_stress_batch,
)
from .scenario_schema import Scenario
from .training import evaluate_controller_on_scenario, train_compact_controller


FEASIBILITY_EPS = 1e-4


def _assert_feasibility(scenario: Scenario, per_step: list[dict[str, Any]]) -> None:
    for step_state, step_out in zip(scenario.states, per_step):
        alloc = cast(dict[str, float], step_out["allocation"])
        budget = float(step_state.available_power)
        total = sum(alloc.values())
        assert total <= budget + FEASIBILITY_EPS, (
            f"[{scenario.name}] budget violated at t={step_state.time_index}: "
            f"sum(a)={total:.6f} > P={budget:.6f}"
        )
        for node_id, a_i in alloc.items():
            assert a_i >= -FEASIBILITY_EPS, (
                f"[{scenario.name}] negative allocation {a_i} at {node_id}"
            )
            d_i = float(step_state.demands[node_id])
            assert a_i <= d_i + FEASIBILITY_EPS, (
                f"[{scenario.name}] demand cap violated at t={step_state.time_index} "
                f"{node_id}: a={a_i:.6f} > d={d_i:.6f}"
            )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "smoke_case33bw.yaml"
    if not config_path.exists():
        print(f"ERROR: smoke config missing at {config_path}", file=sys.stderr)
        return 1

    print(f"[smoke] loading benchmark from {config_path}")
    base = load_public_benchmark_scenario(config_path)
    print(
        f"[smoke] scenario={base.name} nodes={len(base.nodes)} "
        f"steps={len(base.states)}"
    )

    train_scenarios = list(generate_persistent_stress_variants(base))  # 3
    train_scenarios += generate_random_stress_batch(base, num_variants=2, seed=0)  # 2
    eval_scenarios = [base]
    print(
        f"[smoke] train_scenarios={len(train_scenarios)} "
        f"eval_scenarios={len(eval_scenarios)}"
    )

    print("[smoke] training compact controller (10 epochs, CPU, tiny)")
    model, train_metrics = train_compact_controller(
        scenarios=train_scenarios,
        epochs=10,
        learning_rate=1e-3,
        switching_penalty=0.2,
        critical_penalty=2.0,
        floor_penalty=20.0,
        imitation_weight=0.0,
        hidden_dim=16,
        num_layers=2,
        eval_scenarios=eval_scenarios,
        eval_every=5,
        device="cpu",
    )
    final_loss = train_metrics.get("final_loss")
    print(f"[smoke] final_loss={final_loss}")

    print("[smoke] evaluating + asserting feasibility")
    for scenario in eval_scenarios:
        result = evaluate_controller_on_scenario(model, scenario, device="cpu")
        summary = cast(dict[str, float], result["summary"])
        per_step = cast(list[dict[str, Any]], result["per_step"])
        print(f"[smoke]   {scenario.name}")
        for key, value in summary.items():
            print(f"[smoke]     {key}: {value}")
        _assert_feasibility(scenario, per_step)
        print(f"[smoke]   feasibility OK ({len(scenario.states)} steps)")

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
