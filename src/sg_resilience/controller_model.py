from __future__ import annotations

import importlib
from typing import Any

from .floor_targets import compute_adaptive_floor_targets
from .scenario_schema import Scenario, TimeStepState


def _torch_modules() -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    return torch, nn


def sparsemax(logits: Any) -> Any:
    torch, _ = _torch_modules()
    z = logits - torch.max(logits)
    sorted_z, _ = torch.sort(z, descending=True)
    cssv = torch.cumsum(sorted_z, dim=0) - 1
    k = torch.arange(1, z.shape[0] + 1, device=z.device, dtype=z.dtype)
    support = sorted_z - cssv / k > 0
    k_z = torch.sum(support).to(dtype=torch.int64)
    tau = cssv[k_z - 1] / k[k_z - 1]
    return torch.clamp(z - tau, min=0.0)


class GraphSAGEBlock(importlib.import_module("torch.nn").Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        _, nn = _torch_modules()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()

    def forward(self, features: Any, adjacency: Any) -> Any:
        aggregated = adjacency @ features
        output = self.self_linear(features) + self.neigh_linear(aggregated)
        output = self.norm(output)
        return self.activation(output)


class GraphSAGEPolicy(importlib.import_module("torch.nn").Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        _, nn = _torch_modules()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(GraphSAGEBlock(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, features: Any, adjacency: Any) -> Any:
        hidden = self.activation(self.input_proj(features))
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        return self.output(hidden)


def build_compact_controller(input_dim: int = 14, hidden_dim: int = 32, num_layers: int = 3) -> Any:
    return GraphSAGEPolicy(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)


def build_adjacency_tensor(scenario: Scenario, device: str = "cpu") -> Any:
    torch, _ = _torch_modules()
    if scenario.adjacency_matrix:
        adjacency = torch.tensor(scenario.adjacency_matrix, dtype=torch.float32, device=device)
    else:
        node_count = len(scenario.nodes)
        adjacency = torch.eye(node_count, dtype=torch.float32, device=device)
    return adjacency


def featurize_state(
    scenario: Scenario,
    state: TimeStepState,
    previous_allocation: dict[str, float] | None = None,
    device: str = "cpu",
) -> tuple[Any, Any, list[str]]:
    torch, _ = _torch_modules()
    previous_allocation = previous_allocation or {}

    node_ids = [node.node_id for node in scenario.nodes]
    priorities = {node.node_id: node.priority for node in scenario.nodes}
    critical_flags = {node.node_id: 1.0 if node.is_critical else 0.0 for node in scenario.nodes}
    min_service_fractions = {node.node_id: node.min_service_fraction for node in scenario.nodes}
    static_features_map = {node.node_id: node.static_features for node in scenario.nodes}
    total_demand = sum(float(state.demands.get(node_id, 0.0)) for node_id in node_ids)
    max_demand = max((float(state.demands.get(node_id, 0.0)) for node_id in node_ids), default=1.0)
    if max_demand <= 0.0:
        max_demand = 1.0
    demand_supply_ratio = total_demand / max(float(state.available_power), 1e-6)
    outage_indicator = 1.0 if len(state.outages) > 0 else 0.0
    critical_nodes = {node.node_id for node in scenario.nodes if node.is_critical}
    adaptive_floor_targets = compute_adaptive_floor_targets(state.demands, min_service_fractions, critical_nodes, float(state.available_power))

    rows: list[list[float]] = []
    for node_id in node_ids:
        node_demand = float(state.demands.get(node_id, 0.0))
        rows.append(
            [
                node_demand / max_demand,
                float(priorities[node_id]) / 15.0,
                float(critical_flags[node_id]),
                float(previous_allocation.get(node_id, 0.0)) / max_demand,
                float(state.supply_scale),
                float(state.available_power) / max(total_demand, 1e-6),
                float(demand_supply_ratio),
                outage_indicator,
                float(adaptive_floor_targets.get(node_id, 0.0)) / max_demand,
                *list(static_features_map.get(node_id, (0.0, 0.0, 0.0, 0.0, 0.0))),
            ]
        )

    features = torch.tensor(rows, dtype=torch.float32, device=device)
    adjacency = build_adjacency_tensor(scenario, device=device)
    return features, adjacency, node_ids


def project_allocation(weights: Any, demand_tensor: Any, available_power: float) -> Any:
    torch, _ = _torch_modules()
    remaining = torch.tensor(float(available_power), dtype=demand_tensor.dtype, device=demand_tensor.device)
    residual_demand = demand_tensor.clone()
    served = torch.zeros_like(demand_tensor)
    active = torch.ones_like(demand_tensor)

    for _ in range(demand_tensor.shape[0]):
        active_weights = weights * active
        total_active_weight = torch.sum(active_weights)
        if float(total_active_weight.detach().cpu().item()) <= 1e-8:
            break

        normalized = active_weights / total_active_weight
        proposed = normalized * remaining
        increment = torch.minimum(proposed, residual_demand)
        served = served + increment
        residual_demand = torch.clamp(residual_demand - increment, min=0.0)
        remaining = torch.clamp(remaining - torch.sum(increment), min=0.0)
        active = torch.where(residual_demand > 1e-6, torch.ones_like(active), torch.zeros_like(active))
        if float(remaining.detach().cpu().item()) <= 1e-8:
            break

    return served


def allocate_from_model(
    model: Any,
    scenario: Scenario,
    state: TimeStepState,
    previous_allocation: dict[str, float] | None = None,
    device: str = "cpu",
) -> tuple[dict[str, float], Any, list[str]]:
    torch, _ = _torch_modules()
    features, adjacency, node_ids = featurize_state(scenario, state, previous_allocation, device=device)
    logits = model(features, adjacency).squeeze(-1)
    weights = sparsemax(logits)
    demand_tensor = torch.tensor(
        [float(state.demands[node_id]) for node_id in node_ids],
        dtype=torch.float32,
        device=device,
    )
    served = project_allocation(weights, demand_tensor, float(state.available_power))
    allocation = {node_id: float(served[idx].detach().cpu().item()) for idx, node_id in enumerate(node_ids)}
    return allocation, served, node_ids
