from __future__ import annotations


def switching_count(service_history: dict[str, list[float]], threshold: float = 1e-9) -> int:
    count = 0
    for values in service_history.values():
        for prev, curr in zip(values, values[1:]):
            prev_on = prev > threshold
            curr_on = curr > threshold
            if prev_on != curr_on:
                count += 1
    return count


def unserved_critical_demand(
    service: dict[str, float],
    demands: dict[str, float],
    critical_nodes: set[str],
) -> float:
    total = 0.0
    for node_id in critical_nodes:
        demand = demands.get(node_id, 0.0)
        served = service.get(node_id, 0.0)
        total += max(demand - served, 0.0)
    return total


def weighted_served_energy(service: dict[str, float], priorities: dict[str, float]) -> float:
    return sum(service.get(node_id, 0.0) * priorities.get(node_id, 0.0) for node_id in priorities)


def critical_continuity_ratio(
    service_history: dict[str, list[float]],
    critical_nodes: set[str],
    threshold: float = 1e-9,
) -> float:
    total_steps = 0
    active_steps = 0
    for node_id in critical_nodes:
        for value in service_history.get(node_id, []):
            total_steps += 1
            if value > threshold:
                active_steps += 1
    if total_steps == 0:
        return 0.0
    return active_steps / total_steps
