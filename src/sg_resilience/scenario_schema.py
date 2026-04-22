from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadNode:
    node_id: str
    priority: float
    is_critical: bool
    min_service_fraction: float = 0.0
    bus_idx: int = -1
    static_features: tuple[float, ...] = ()


@dataclass(frozen=True)
class TimeStepState:
    time_index: int
    available_power: float
    demands: dict[str, float]
    outages: tuple[str, ...] = ()
    supply_scale: float = 1.0


@dataclass(frozen=True)
class Scenario:
    name: str
    nodes: tuple[LoadNode, ...]
    states: tuple[TimeStepState, ...]
    adjacency_matrix: tuple[tuple[float, ...], ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
