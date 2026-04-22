from __future__ import annotations

import random
from dataclasses import replace

from .scenario_schema import Scenario, TimeStepState


def generate_persistent_stress_variants(base_scenario: Scenario) -> list[Scenario]:
    variants: list[Scenario] = []

    severity_specs = (
        ("mild", 1.0, 0.0, (1,)),
        ("moderate", 0.85, 0.1, (1, 2)),
        ("severe", 0.7, 0.2, (1, 2, 3)),
    )

    for label, power_scale, demand_scale, outage_indices in severity_specs:
        states: list[TimeStepState] = []
        for state in base_scenario.states:
            scaled_demands = {
                node_id: round(value * (1.0 + demand_scale), 4)
                for node_id, value in state.demands.items()
            }
            outages = state.outages
            if state.time_index in outage_indices:
                outages = tuple(sorted(set(outages) | {f"persistent_{label}_outage"}))

            states.append(
                replace(
                    state,
                    available_power=round(state.available_power * power_scale, 4),
                    demands=scaled_demands,
                    outages=outages,
                    supply_scale=round(state.supply_scale * power_scale, 4),
                )
            )

        variants.append(
            Scenario(
                name=f"{base_scenario.name}_{label}",
                nodes=base_scenario.nodes,
                states=tuple(states),
                metadata={**base_scenario.metadata, "variant": label},
            )
        )

    return variants


def generate_random_stress_batch(
    base_scenario: Scenario,
    num_variants: int,
    seed: int = 0,
) -> list[Scenario]:
    rng = random.Random(seed)
    variants: list[Scenario] = []

    for index in range(num_variants):
        power_scale = rng.uniform(0.55, 1.05)
        demand_scale = rng.uniform(0.0, 0.35)
        outage_start = rng.randint(0, max(len(base_scenario.states) - 2, 0))
        outage_length = rng.randint(1, min(3, len(base_scenario.states)))
        outage_name = f"rand_outage_{index}"

        states: list[TimeStepState] = []
        for state in base_scenario.states:
            in_outage_window = outage_start <= state.time_index < outage_start + outage_length
            scaled_demands = {
                node_id: round(value * (1.0 + demand_scale), 4)
                for node_id, value in state.demands.items()
            }
            outages = state.outages
            if in_outage_window:
                outages = tuple(sorted(set(outages) | {outage_name}))

            state_power_scale = power_scale
            if in_outage_window:
                state_power_scale *= rng.uniform(0.85, 0.95)

            states.append(
                replace(
                    state,
                    available_power=round(state.available_power * state_power_scale, 4),
                    demands=scaled_demands,
                    outages=outages,
                    supply_scale=round(state.supply_scale * state_power_scale, 4),
                )
            )

        variants.append(
            Scenario(
                name=f"{base_scenario.name}_random_{index:03d}",
                nodes=base_scenario.nodes,
                states=tuple(states),
                metadata={
                    **base_scenario.metadata,
                    "variant": "random",
                    "seed": str(seed),
                    "index": str(index),
                },
            )
        )

    return variants


def generate_extreme_stress_scenarios(base_scenario: Scenario) -> list[Scenario]:
    scenarios: list[Scenario] = []

    collapse_levels = (0.5, 0.35, 0.2)
    for level in collapse_levels:
        states = []
        for state in base_scenario.states:
            states.append(
                replace(
                    state,
                    available_power=round(state.available_power * level, 6),
                    supply_scale=round(state.supply_scale * level, 6),
                )
            )
        scenarios.append(
            Scenario(
                name=f"{base_scenario.name}_supplycollapse_{str(level).replace('.', '')}",
                nodes=base_scenario.nodes,
                states=tuple(states),
                metadata={**base_scenario.metadata, "stress_family": "supply_collapse", "stress_level": str(level)},
            )
        )

    persistence_lengths = (2, 4, 6)
    for length in persistence_lengths:
        states = []
        for state in base_scenario.states:
            in_window = state.time_index < length
            states.append(
                replace(
                    state,
                    outages=tuple(sorted(set(state.outages) | ({f"persistent_len_{length}"} if in_window else set()))),
                    available_power=round(state.available_power * (0.75 if in_window else 1.0), 6),
                    supply_scale=round(state.supply_scale * (0.75 if in_window else 1.0), 6),
                )
            )
        scenarios.append(
            Scenario(
                name=f"{base_scenario.name}_outagepersist_{length}",
                nodes=base_scenario.nodes,
                states=tuple(states),
                metadata={**base_scenario.metadata, "stress_family": "outage_persistence", "stress_level": str(length)},
            )
        )

    overload_levels = (1.2, 1.35, 1.5)
    for level in overload_levels:
        states = []
        for state in base_scenario.states:
            states.append(
                replace(
                    state,
                    demands={node_id: round(value * level, 6) for node_id, value in state.demands.items()},
                    outages=tuple(sorted(set(state.outages) | {f"urban_overload_{str(level).replace('.', '')}"})),
                )
            )
        scenarios.append(
            Scenario(
                name=f"{base_scenario.name}_urbanoverload_{str(level).replace('.', '')}",
                nodes=base_scenario.nodes,
                states=tuple(states),
                metadata={**base_scenario.metadata, "stress_family": "urban_overload", "stress_level": str(level)},
            )
        )

    return scenarios
