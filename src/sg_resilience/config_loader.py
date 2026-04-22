from __future__ import annotations

from pathlib import Path

import yaml

from .benchmark_loader import load_public_benchmark_scenario
from .scenario_schema import LoadNode, Scenario, TimeStepState


def load_scenario_from_yaml(path: str | Path) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if "benchmark" in raw:
        return load_public_benchmark_scenario(path)

    nodes = tuple(
        LoadNode(
            node_id=node["node_id"],
            priority=float(node["priority"]),
            is_critical=bool(node["is_critical"]),
            min_service_fraction=float(node.get("min_service_fraction", 0.0)),
            bus_idx=int(node.get("bus_idx", -1)),
            static_features=tuple(float(value) for value in node.get("static_features", (0.0, 0.0, 0.0, 0.0, 0.0))),
        )
        for node in raw["nodes"]
    )

    states = tuple(
        TimeStepState(
            time_index=int(state["time_index"]),
            available_power=float(state["available_power"]),
            demands={key: float(value) for key, value in state["demands"].items()},
            outages=tuple(state.get("outages", [])),
            supply_scale=float(state.get("supply_scale", 1.0)),
        )
        for state in raw["states"]
    )

    return Scenario(
        name=raw["name"],
        nodes=nodes,
        states=states,
        adjacency_matrix=tuple(tuple(float(value) for value in row) for row in raw.get("adjacency_matrix", [])),
        metadata={key: str(value) for key, value in raw.get("metadata", {}).items()},
    )
