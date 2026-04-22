from __future__ import annotations

from .baseline_rule import priority_first_allocation
from .priority_metrics import (
    critical_continuity_ratio,
    switching_count,
    unserved_critical_demand,
    weighted_served_energy,
)
from .scenario_schema import Scenario


def run_rule_baseline_on_scenario(scenario: Scenario) -> dict[str, object]:
    priorities = {node.node_id: node.priority for node in scenario.nodes}
    min_service_fractions = {node.node_id: node.min_service_fraction for node in scenario.nodes}
    critical_nodes = {node.node_id for node in scenario.nodes if node.is_critical}

    service_history = {node.node_id: [] for node in scenario.nodes}
    per_step = []
    total_weighted_served = 0.0
    total_unserved_critical = 0.0

    for state in scenario.states:
        allocation = priority_first_allocation(
            available_power=state.available_power,
            demands=state.demands,
            priorities=priorities,
            min_service_fractions=min_service_fractions,
            critical_nodes=critical_nodes,
        )
        for node_id, served in allocation.items():
            service_history[node_id].append(served)

        weighted_value = weighted_served_energy(allocation, priorities)
        unserved_value = unserved_critical_demand(allocation, state.demands, critical_nodes)
        total_weighted_served += weighted_value
        total_unserved_critical += unserved_value

        per_step.append(
            {
                "time_index": state.time_index,
                "available_power": state.available_power,
                "outages": list(state.outages),
                "allocation": allocation,
                "weighted_served_energy": weighted_value,
                "unserved_critical_demand": unserved_value,
            }
        )

    return {
        "scenario": scenario.name,
        "metadata": scenario.metadata,
        "per_step": per_step,
        "summary": {
            "weighted_served_energy": total_weighted_served,
            "unserved_critical_demand": total_unserved_critical,
            "switching_count": switching_count(service_history),
            "critical_continuity_ratio": critical_continuity_ratio(service_history, critical_nodes),
        },
    }
