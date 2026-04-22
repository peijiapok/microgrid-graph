from __future__ import annotations


def compute_adaptive_floor_targets(
    demands: dict[str, float],
    min_service_fractions: dict[str, float],
    critical_nodes: set[str],
    available_power: float,
) -> dict[str, float]:
    critical_targets = {
        node_id: max(demands[node_id], 0.0) * max(min_service_fractions.get(node_id, 0.0), 0.0)
        for node_id in demands
        if node_id in critical_nodes
    }
    noncritical_targets = {
        node_id: max(demands[node_id], 0.0) * max(min_service_fractions.get(node_id, 0.0), 0.0)
        for node_id in demands
        if node_id not in critical_nodes
    }

    critical_total = sum(critical_targets.values())
    noncritical_total = sum(noncritical_targets.values())
    available = max(available_power, 0.0)

    if available >= critical_total + noncritical_total:
        return {**critical_targets, **noncritical_targets}

    if available >= critical_total:
        remaining = available - critical_total
        scale = remaining / max(noncritical_total, 1e-9)
        scaled_noncritical = {node_id: value * scale for node_id, value in noncritical_targets.items()}
        return {**critical_targets, **scaled_noncritical}

    critical_scale = available / max(critical_total, 1e-9)
    scaled_critical = {node_id: value * critical_scale for node_id, value in critical_targets.items()}
    zero_noncritical = {node_id: 0.0 for node_id in noncritical_targets}
    return {**scaled_critical, **zero_noncritical}
