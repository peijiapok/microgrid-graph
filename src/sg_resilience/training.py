from __future__ import annotations

import importlib
from typing import Any, cast

from .baseline_opt import continuity_aware_allocation
from .controller_model import allocate_from_model, build_compact_controller, featurize_state, project_allocation, sparsemax
from .floor_targets import compute_adaptive_floor_targets
from .priority_metrics import critical_continuity_ratio, switching_count
from .scenario_schema import Scenario


def _torch_modules() -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    optim = importlib.import_module("torch.optim")
    return torch, nn, optim


def evaluate_controller_on_scenario(
    model: Any,
    scenario: Scenario,
    device: str = "cpu",
) -> dict[str, object]:
    torch, _, _ = _torch_modules()
    priorities = {node.node_id: node.priority for node in scenario.nodes}
    critical_nodes = {node.node_id for node in scenario.nodes if node.is_critical}
    service_history = {node.node_id: [] for node in scenario.nodes}
    previous_allocation = {node.node_id: 0.0 for node in scenario.nodes}
    per_step = []
    total_weighted = 0.0
    total_unserved_critical = 0.0

    model.eval()
    with torch.no_grad():
        for state in scenario.states:
            allocation, served_tensor, node_ids = allocate_from_model(
                model,
                scenario,
                state,
                previous_allocation=previous_allocation,
                device=device,
            )
            previous_allocation = allocation
            step_weighted = 0.0
            step_unserved_critical = 0.0
            for idx, node_id in enumerate(node_ids):
                served_value = float(served_tensor[idx].cpu().item())
                service_history[node_id].append(served_value)
                step_weighted += served_value * priorities[node_id]
                if node_id in critical_nodes:
                    step_unserved_critical += max(state.demands[node_id] - served_value, 0.0)

            total_weighted += step_weighted
            total_unserved_critical += step_unserved_critical
            per_step.append(
                {
                    "time_index": state.time_index,
                    "available_power": state.available_power,
                    "outages": list(state.outages),
                    "allocation": allocation,
                    "weighted_served_energy": step_weighted,
                    "unserved_critical_demand": step_unserved_critical,
                }
            )

    return {
        "scenario": scenario.name,
        "metadata": scenario.metadata,
        "per_step": per_step,
        "summary": {
            "weighted_served_energy": total_weighted,
            "unserved_critical_demand": total_unserved_critical,
            "switching_count": switching_count(service_history),
            "critical_continuity_ratio": critical_continuity_ratio(service_history, critical_nodes),
        },
    }


def _linear_schedule(start: float, end: float, epoch_index: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return end
    progress = epoch_index / max(total_epochs - 1, 1)
    return start + (end - start) * progress


def _build_teacher_cache(
    scenarios: list[Scenario],
    switching_penalty: float,
) -> dict[str, list[dict[str, float]]]:
    cache: dict[str, list[dict[str, float]]] = {}
    for scenario in scenarios:
        priorities = {node.node_id: node.priority for node in scenario.nodes}
        critical_nodes = {node.node_id for node in scenario.nodes if node.is_critical}
        min_service_fractions = {node.node_id: node.min_service_fraction for node in scenario.nodes}
        previous_allocation = {node.node_id: 0.0 for node in scenario.nodes}
        teacher_steps: list[dict[str, float]] = []
        for state in scenario.states:
            teacher_allocation = continuity_aware_allocation(
                available_power=state.available_power,
                demands=state.demands,
                priorities=priorities,
                min_service_fractions=min_service_fractions,
                critical_nodes=critical_nodes,
                previous_allocation=previous_allocation,
                switching_penalty=switching_penalty,
            )
            teacher_steps.append(teacher_allocation)
            previous_allocation = teacher_allocation
        cache[scenario.name] = teacher_steps
    return cache


def _aggregate_eval_summary(evaluations: dict[str, dict[str, object]]) -> dict[str, float]:
    summaries = [cast(dict[str, float], payload["summary"]) for payload in evaluations.values()]
    count = max(len(summaries), 1)
    return {
        "mean_weighted_served_energy": sum(float(summary["weighted_served_energy"]) for summary in summaries) / count,
        "mean_unserved_critical_demand": sum(float(summary["unserved_critical_demand"]) for summary in summaries) / count,
        "worst_unserved_critical_demand": max(
            (float(summary["unserved_critical_demand"]) for summary in summaries),
            default=0.0,
        ),
        "mean_switching_count": sum(float(summary["switching_count"]) for summary in summaries) / count,
        "mean_critical_continuity_ratio": sum(float(summary["critical_continuity_ratio"]) for summary in summaries) / count,
        "worst_critical_continuity_ratio": min(
            (float(summary["critical_continuity_ratio"]) for summary in summaries),
            default=0.0,
        ),
    }


def _is_better_eval(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        candidate["mean_unserved_critical_demand"],
        candidate["worst_unserved_critical_demand"],
        -candidate["mean_critical_continuity_ratio"],
        -candidate["worst_critical_continuity_ratio"],
        -candidate["mean_weighted_served_energy"],
        candidate["mean_switching_count"],
    )
    incumbent_key = (
        incumbent["mean_unserved_critical_demand"],
        incumbent["worst_unserved_critical_demand"],
        -incumbent["mean_critical_continuity_ratio"],
        -incumbent["worst_critical_continuity_ratio"],
        -incumbent["mean_weighted_served_energy"],
        incumbent["mean_switching_count"],
    )
    return candidate_key < incumbent_key


def _scenario_loss_weight(scenario: Scenario) -> float:
    weight = 1.0
    variant = str(scenario.metadata.get("variant", ""))
    stress_family = str(scenario.metadata.get("stress_family", ""))
    stress_level = str(scenario.metadata.get("stress_level", ""))

    if variant == "random":
        weight += 0.15
    elif variant == "mild":
        weight += 0.05
    elif variant == "moderate":
        weight += 0.15
    elif variant == "severe":
        weight += 0.3

    if stress_family == "supply_collapse":
        weight += 0.35
    elif stress_family == "outage_persistence":
        weight += 0.25
    elif stress_family == "urban_overload":
        weight += 0.3

    try:
        weight += 0.1 * max(float(stress_level) - 1.0, 0.0)
    except ValueError:
        pass

    return weight


def train_compact_controller(
    scenarios: list[Scenario],
    epochs: int = 100,
    learning_rate: float = 1e-3,
    switching_penalty: float = 0.2,
    critical_penalty: float = 2.0,
    floor_penalty: float = 20.0,
    imitation_weight: float = 0.0,
    hidden_dim: int = 32,
    num_layers: int = 3,
    eval_scenarios: list[Scenario] | None = None,
    eval_every: int = 1,
    switching_penalty_end: float | None = None,
    imitation_weight_end: float | None = None,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    torch, nn, optim = _torch_modules()
    model = build_compact_controller(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    imitation_criterion = nn.SmoothL1Loss()

    history: list[float] = []
    component_history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    best_eval_summary: dict[str, float] | None = None
    best_epoch: int | None = None
    best_state_dict: dict[str, Any] | None = None

    total_steps = sum(len(scenario.states) for scenario in scenarios)
    switching_penalty_end = switching_penalty if switching_penalty_end is None else switching_penalty_end
    imitation_weight_end = imitation_weight if imitation_weight_end is None else imitation_weight_end

    for epoch_index in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        epoch_weighted_loss = 0.0
        epoch_critical_loss = 0.0
        epoch_switching_loss = 0.0
        epoch_imitation_loss = 0.0
        epoch_continuity_loss = 0.0
        epoch_critical_activity_loss = 0.0
        current_switching_penalty = _linear_schedule(switching_penalty, switching_penalty_end, epoch_index, epochs)
        current_imitation_weight = _linear_schedule(imitation_weight, imitation_weight_end, epoch_index, epochs)
        teacher_cache = (
            _build_teacher_cache(scenarios, current_switching_penalty) if current_imitation_weight > 0.0 else {}
        )

        for scenario in scenarios:
            priorities = {node.node_id: node.priority for node in scenario.nodes}
            critical_nodes = {node.node_id for node in scenario.nodes if node.is_critical}
            min_service_fractions = {node.node_id: node.min_service_fraction for node in scenario.nodes}
            scenario_weight = _scenario_loss_weight(scenario)
            previous_served = None
            previous_allocation = {node.node_id: 0.0 for node in scenario.nodes}

            for state in scenario.states:
                features, adjacency, node_ids = featurize_state(
                    scenario,
                    state,
                    previous_allocation=previous_allocation,
                    device=device,
                )
                logits = model(features, adjacency).squeeze(-1)
                weights = sparsemax(logits)
                demand_tensor = torch.tensor(
                    [float(state.demands[node_id]) for node_id in node_ids],
                    dtype=torch.float32,
                    device=device,
                )
                served = project_allocation(weights, demand_tensor, float(state.available_power))

                priority_tensor = torch.tensor(
                    [float(priorities[node_id]) for node_id in node_ids],
                    dtype=torch.float32,
                    device=device,
                )
                critical_mask = torch.tensor(
                    [1.0 if node_id in critical_nodes else 0.0 for node_id in node_ids],
                    dtype=torch.float32,
                    device=device,
                )
                adaptive_floor_targets = compute_adaptive_floor_targets(
                    state.demands,
                    min_service_fractions,
                    critical_nodes,
                    float(state.available_power),
                )
                floor_fraction_tensor = torch.tensor(
                    [
                        float(adaptive_floor_targets[node_id])
                        / max(float(state.demands[node_id]), 1e-6)
                        for node_id in node_ids
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                floor_target = demand_tensor * floor_fraction_tensor
                shortfall = torch.relu(floor_target - served)
                surplus = torch.relu(served - floor_target)
                critical_activity_target = torch.minimum(demand_tensor * 0.05, floor_target)
                critical_activity_shortfall = torch.relu(critical_activity_target - served) * critical_mask
                weighted_surplus = torch.sum(priority_tensor * torch.sqrt(surplus + 1e-6))
                weighted_loss_term = -weighted_surplus
                floor_loss_term = floor_penalty * torch.sum(priority_tensor * shortfall)
                critical_floor_loss_term = critical_penalty * torch.sum(critical_mask * shortfall)
                critical_activity_term = critical_penalty * torch.sum(priority_tensor * critical_activity_shortfall)
                step_loss = weighted_loss_term + floor_loss_term + critical_floor_loss_term + critical_activity_term
                epoch_weighted_loss += float(weighted_loss_term.detach().cpu().item())
                epoch_critical_loss += float((floor_loss_term + critical_floor_loss_term).detach().cpu().item())
                epoch_critical_activity_loss += float(critical_activity_term.detach().cpu().item())

                if current_imitation_weight > 0.0:
                    teacher_allocation = teacher_cache[scenario.name][state.time_index]
                    teacher_tensor = torch.tensor(
                        [float(teacher_allocation[node_id]) for node_id in node_ids],
                        dtype=torch.float32,
                        device=device,
                    )
                    imitation_loss = imitation_criterion(served, teacher_tensor)
                    imitation_term = current_imitation_weight * imitation_loss
                    step_loss = step_loss + imitation_term
                    epoch_imitation_loss += float(imitation_term.detach().cpu().item())

                if previous_served is not None:
                    switching_term = current_switching_penalty * torch.sum(torch.abs(served - previous_served))
                    step_loss = step_loss + switching_term
                    epoch_switching_loss += float(switching_term.detach().cpu().item())

                    continuity_margin = 0.05 * floor_target
                    critical_drop = torch.relu(previous_served - served - continuity_margin) * critical_mask
                    continuity_term = critical_penalty * torch.sum(priority_tensor * critical_drop)
                    step_loss = step_loss + continuity_term
                    epoch_continuity_loss += float(continuity_term.detach().cpu().item())

                weighted_step_loss = scenario_weight * step_loss
                total_loss = total_loss + weighted_step_loss
                previous_served = served
                previous_allocation = {
                    node_id: float(served[idx].cpu().item()) for idx, node_id in enumerate(node_ids)
                }

        total_loss = total_loss / max(total_steps, 1)
        total_loss.backward()
        optimizer.step()
        history.append(float(total_loss.detach().cpu().item()))
        component_history.append(
            {
                "epoch": float(epoch_index),
                "total_loss": float(total_loss.detach().cpu().item()),
                "weighted_loss": epoch_weighted_loss / max(total_steps, 1),
                "critical_loss": epoch_critical_loss / max(total_steps, 1),
                "switching_loss": epoch_switching_loss / max(total_steps, 1),
                "imitation_loss": epoch_imitation_loss / max(total_steps, 1),
                "continuity_loss": epoch_continuity_loss / max(total_steps, 1),
                "critical_activity_loss": epoch_critical_activity_loss / max(total_steps, 1),
                "switching_penalty": current_switching_penalty,
                "imitation_weight": current_imitation_weight,
            }
        )

        if eval_scenarios is not None and ((epoch_index + 1) % max(eval_every, 1) == 0 or epoch_index == epochs - 1):
            evaluations = {
                scenario.name: evaluate_controller_on_scenario(model, scenario, device=device)
                for scenario in eval_scenarios
            }
            eval_summary = _aggregate_eval_summary(evaluations)
            eval_history.append({"epoch": float(epoch_index), **eval_summary})
            if _is_better_eval(eval_summary, best_eval_summary):
                best_eval_summary = eval_summary
                best_epoch = epoch_index
                best_state_dict = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "switching_penalty": switching_penalty,
        "switching_penalty_end": switching_penalty_end,
        "critical_penalty": critical_penalty,
        "floor_penalty": floor_penalty,
        "imitation_weight": imitation_weight,
        "imitation_weight_end": imitation_weight_end,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "loss_history": history,
        "component_history": component_history,
        "eval_history": eval_history,
        "best_epoch": best_epoch,
        "best_eval_summary": best_eval_summary,
        "final_loss": history[-1] if history else None,
        "device": device,
    }
    return model, metrics
