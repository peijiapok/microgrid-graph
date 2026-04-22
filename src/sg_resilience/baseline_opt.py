from __future__ import annotations

import importlib

from .floor_targets import compute_adaptive_floor_targets


def continuity_aware_allocation(
    available_power: float,
    demands: dict[str, float],
    priorities: dict[str, float],
    min_service_fractions: dict[str, float] | None = None,
    critical_nodes: set[str] | None = None,
    previous_allocation: dict[str, float] | None = None,
    switching_penalty: float = 0.2,
    floor_penalty: float = 100.0,
    surplus_weight: float = 1.0,
    zero_threshold: float = 1e-4,
) -> dict[str, float]:
    cp = importlib.import_module("cvxpy")
    node_ids = list(demands)
    served = cp.Variable(len(node_ids), nonneg=True)
    shortfall = cp.Variable(len(node_ids), nonneg=True)
    surplus = cp.Variable(len(node_ids), nonneg=True)
    min_service_fractions = min_service_fractions or {}
    critical_nodes = critical_nodes or set()
    adaptive_floor_targets = compute_adaptive_floor_targets(demands, min_service_fractions, critical_nodes, available_power)
    floor_targets = [adaptive_floor_targets[node_id] for node_id in node_ids]
    constraints = [served <= [max(demands[node_id], 0.0) for node_id in node_ids]]
    constraints.append(cp.sum(served) <= max(available_power, 0.0))
    constraints.extend(
        [
            served + shortfall - surplus == cp.Constant(floor_targets),
        ]
    )

    objective = -floor_penalty * cp.sum(cp.multiply([priorities.get(node_id, 0.0) for node_id in node_ids], shortfall))
    objective += surplus_weight * cp.sum(cp.multiply([priorities.get(node_id, 0.0) for node_id in node_ids], cp.log1p(surplus)))

    if previous_allocation is not None:
        prev = [previous_allocation.get(node_id, 0.0) for node_id in node_ids]
        delta = cp.Variable(len(node_ids), nonneg=True)
        constraints.extend(
            [
                delta >= served - prev,
                delta >= prev - served,
            ]
        )
        objective -= switching_penalty * cp.sum(delta)

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    if served.value is None:
        raise RuntimeError("Optimization baseline failed to produce a solution")

    allocation = {}
    for idx, node_id in enumerate(node_ids):
        value = max(float(served.value[idx]), 0.0)
        if value < zero_threshold:
            value = 0.0
        demand_cap = max(demands[node_id], 0.0)
        allocation[node_id] = min(value, demand_cap)
    return allocation
