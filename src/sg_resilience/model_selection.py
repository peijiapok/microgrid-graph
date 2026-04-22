from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateConfig:
    hidden_dim: int
    imitation_weight: float
    imitation_weight_end: float
    switching_penalty: float
    switching_penalty_end: float
    floor_penalty: float
    epochs: int
    learning_rate: float


def aggregate_evaluations(evaluations: dict[str, dict[str, Any]]) -> dict[str, float]:
    summaries = [payload["summary"] for payload in evaluations.values()]
    count = max(len(summaries), 1)
    mean_weighted = sum(float(summary["weighted_served_energy"]) for summary in summaries) / count
    mean_unserved = sum(float(summary["unserved_critical_demand"]) for summary in summaries) / count
    mean_switching = sum(float(summary["switching_count"]) for summary in summaries) / count
    mean_continuity = sum(float(summary["critical_continuity_ratio"]) for summary in summaries) / count
    worst_unserved = max(float(summary["unserved_critical_demand"]) for summary in summaries)

    return {
        "mean_weighted_served_energy": mean_weighted,
        "mean_unserved_critical_demand": mean_unserved,
        "mean_switching_count": mean_switching,
        "mean_critical_continuity_ratio": mean_continuity,
        "worst_unserved_critical_demand": worst_unserved,
    }


def candidate_sort_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metrics["mean_unserved_critical_demand"],
        -metrics["mean_weighted_served_energy"],
        metrics["mean_switching_count"],
        -metrics["mean_critical_continuity_ratio"],
    )


def pareto_dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    left_tuple = (
        left["mean_unserved_critical_demand"],
        left["worst_unserved_critical_demand"],
        -left["mean_weighted_served_energy"],
        left["mean_switching_count"],
    )
    right_tuple = (
        right["mean_unserved_critical_demand"],
        right["worst_unserved_critical_demand"],
        -right["mean_weighted_served_energy"],
        right["mean_switching_count"],
    )
    return all(l <= r for l, r in zip(left_tuple, right_tuple)) and any(l < r for l, r in zip(left_tuple, right_tuple))


def pareto_front(items: list[dict[str, Any]], metrics_key: str = "metrics") -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for item in items:
        metrics = item[metrics_key]
        dominated = False
        for other in items:
            if other is item:
                continue
            if pareto_dominates(other[metrics_key], metrics):
                dominated = True
                break
        if not dominated:
            front.append(item)
    return front
