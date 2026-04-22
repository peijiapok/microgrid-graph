from __future__ import annotations

from .floor_targets import compute_adaptive_floor_targets


def priority_first_allocation(
    available_power: float,
    demands: dict[str, float],
    priorities: dict[str, float],
    min_service_fractions: dict[str, float] | None = None,
    critical_nodes: set[str] | None = None,
) -> dict[str, float]:
    remaining = max(available_power, 0.0)
    allocation = {node_id: 0.0 for node_id in demands}
    min_service_fractions = min_service_fractions or {}
    critical_nodes = critical_nodes or set()

    floor_targets = compute_adaptive_floor_targets(demands, min_service_fractions, critical_nodes, available_power)
    total_floor = sum(floor_targets.values())

    if total_floor > 0.0:
        if remaining >= total_floor:
            for node_id, floor_value in floor_targets.items():
                allocation[node_id] = floor_value
            remaining -= total_floor
        else:
            ratio = remaining / total_floor
            for node_id, floor_value in floor_targets.items():
                allocation[node_id] = floor_value * ratio
            return allocation

    ordered_nodes = sorted(
        demands,
        key=lambda node_id: (-priorities.get(node_id, 0.0), node_id),
    )

    for node_id in ordered_nodes:
        residual_demand = max(demands[node_id], 0.0) - allocation[node_id]
        served = min(residual_demand, remaining)
        allocation[node_id] += served
        remaining -= served
        if remaining <= 0.0:
            break

    return allocation
