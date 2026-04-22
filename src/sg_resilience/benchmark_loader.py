from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import yaml

from .scenario_schema import LoadNode, Scenario, TimeStepState


def _normalized_static_features(
    graph: Any,
    slack_buses: set[int],
    load_bus_counts: dict[int, int],
    sgen_bus_counts: dict[int, int],
    bus_idx: int,
) -> tuple[float, float, float, float, float]:
    nx = cast(Any, importlib.import_module("networkx"))
    node_count = max(graph.number_of_nodes(), 1)
    degree_map = dict(graph.degree())
    max_degree = max(degree_map.values()) if degree_map else 1
    degree = float(degree_map.get(bus_idx, 0)) / max(max_degree, 1)

    if slack_buses:
        path_lengths = [nx.shortest_path_length(graph, source=slack, target=bus_idx) for slack in slack_buses if nx.has_path(graph, slack, bus_idx)]
        min_distance = min(path_lengths) if path_lengths else float(node_count)
    else:
        min_distance = 0.0
    normalized_distance = float(min_distance) / max(node_count - 1, 1)

    load_density = float(load_bus_counts.get(bus_idx, 0)) / max(max(load_bus_counts.values(), default=1), 1)
    sgen_density = float(sgen_bus_counts.get(bus_idx, 0)) / max(max(sgen_bus_counts.values(), default=1), 1)
    is_slack_adjacent = 1.0 if any(graph.has_edge(bus_idx, slack) for slack in slack_buses) else 0.0
    return degree, normalized_distance, load_density, sgen_density, is_slack_adjacent


def _build_network_graph(net: Any) -> Any:
    nx = cast(Any, importlib.import_module("networkx"))
    graph = nx.Graph()
    for bus_idx in net.bus.index:
        graph.add_node(int(bus_idx))

    if hasattr(net, "line"):
        for _, row in net.line.iterrows():
            graph.add_edge(int(row["from_bus"]), int(row["to_bus"]))

    if hasattr(net, "trafo"):
        for _, row in net.trafo.iterrows():
            graph.add_edge(int(row["hv_bus"]), int(row["lv_bus"]))

    return graph


def _build_load_adjacency_matrix(net: Any, graph: Any, sorted_nodes: list[tuple[str, float]]) -> tuple[tuple[float, ...], ...]:
    node_bus = {f"load_{idx}": int(row["bus"]) for idx, row in net.load.iterrows()}
    slack_buses = {int(bus) for bus in net.ext_grid["bus"].tolist()} if len(net.ext_grid) > 0 else set()
    load_ids = [node_id for node_id, _ in sorted_nodes]
    adjacency_rows: list[tuple[float, ...]] = []
    for source_id in load_ids:
        source_bus = node_bus[source_id]
        row = []
        for target_id in load_ids:
            target_bus = node_bus[target_id]
            if source_bus == target_bus:
                weight = 1.0
            else:
                try:
                    distance = graph.shortest_path_length(source_bus, target_bus)
                except Exception:
                    nx = cast(Any, importlib.import_module("networkx"))
                    distance = nx.shortest_path_length(graph, source=source_bus, target=target_bus)
                weight = 1.0 / (1.0 + float(distance))

                if slack_buses and (source_bus in slack_buses or target_bus in slack_buses):
                    weight += 0.1
            row.append(weight)
        row_sum = sum(row)
        adjacency_rows.append(tuple(value / max(row_sum, 1e-9) for value in row))
    return tuple(adjacency_rows)


def load_public_benchmark_scenario(path: str | Path) -> Scenario:
    raw = cast(dict[str, Any], yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    benchmark = cast(dict[str, Any], raw["benchmark"])
    if benchmark["source"] == "pandapower":
        return _load_pandapower_benchmark(raw)
    if benchmark["source"] == "simbench":
        return _load_simbench_benchmark(raw)
    raise ValueError(f"Unsupported benchmark source: {benchmark['source']}")


def _load_pandapower_benchmark(raw: dict[str, Any]) -> Scenario:
    benchmark = cast(dict[str, Any], raw["benchmark"])
    metadata = cast(dict[str, Any], raw.get("metadata", {}))

    networks = cast(Any, importlib.import_module("pandapower.networks"))
    network_builder = getattr(networks, benchmark["name"])
    net = network_builder()

    if len(net.load) == 0:
        raise ValueError("Benchmark network has no loads")

    demand_profile = [float(value) for value in benchmark.get("demand_profile", [1.0])]
    horizon = int(benchmark.get("horizon", len(demand_profile)))
    default_floor = float(benchmark.get("default_min_service_fraction", 0.2))
    critical_floor = float(benchmark.get("critical_min_service_fraction", 0.55))
    if horizon != len(demand_profile):
        raise ValueError("horizon must match demand_profile length")

    base_demands = {}
    for idx, row in net.load.iterrows():
        node_id = f"load_{idx}"
        base_demands[node_id] = float(row["p_mw"])

    sorted_nodes = sorted(base_demands.items(), key=lambda item: (-item[1], item[0]))
    critical_top_k = int(benchmark.get("critical_top_k", max(1, len(sorted_nodes) // 5)))
    critical_nodes = {node_id for node_id, _ in sorted_nodes[:critical_top_k]}
    max_demand = max(base_demands.values())
    graph = _build_network_graph(net)
    slack_buses = {int(bus) for bus in net.ext_grid["bus"].tolist()} if len(net.ext_grid) > 0 else set()
    load_bus_counts = {int(bus): int((net.load["bus"] == bus).sum()) for bus in net.bus.index}
    sgen_bus_counts = {int(bus): int((net.sgen["bus"] == bus).sum()) for bus in net.bus.index} if len(net.sgen) > 0 else {}

    nodes = []
    for node_id, demand in sorted_nodes:
        load_idx = int(node_id.split("_")[1])
        bus_idx = int(net.load.at[load_idx, "bus"])
        normalized = demand / max_demand if max_demand > 0.0 else 0.0
        priority = 1.0 + 9.0 * normalized
        if node_id in critical_nodes:
            priority += 5.0
        static_features = _normalized_static_features(graph, slack_buses, load_bus_counts, sgen_bus_counts, bus_idx)
        min_service_fraction = critical_floor if node_id in critical_nodes else default_floor
        nodes.append(
            LoadNode(
                node_id=node_id,
                priority=priority,
                is_critical=node_id in critical_nodes,
                min_service_fraction=min_service_fraction,
                bus_idx=bus_idx,
                static_features=static_features,
            )
        )

    total_base_demand = sum(base_demands.values())
    supply_ratio = float(benchmark.get("supply_ratio", 0.85))

    states = []
    for time_index, multiplier in enumerate(demand_profile):
        demands = {node_id: round(value * multiplier, 6) for node_id, value in base_demands.items()}
        available_power = round(total_base_demand * multiplier * supply_ratio, 6)
        states.append(
            TimeStepState(
                time_index=time_index,
                available_power=available_power,
                demands=demands,
                outages=(),
                supply_scale=supply_ratio,
            )
        )

    name = f"{benchmark['name']}_public_benchmark"
    adjacency_matrix = _build_load_adjacency_matrix(net, graph, sorted_nodes)
    return Scenario(
        name=name,
        nodes=tuple(nodes),
        states=tuple(states),
        adjacency_matrix=adjacency_matrix,
        metadata={
            **{key: str(value) for key, value in metadata.items()},
            "benchmark_source": benchmark["source"],
            "benchmark_name": benchmark["name"],
            "critical_top_k": str(critical_top_k),
        },
    )


def _load_simbench_benchmark(raw: dict[str, Any]) -> Scenario:
    benchmark = cast(dict[str, Any], raw["benchmark"])
    metadata = cast(dict[str, Any], raw.get("metadata", {}))
    simbench = cast(Any, importlib.import_module("simbench"))
    net = simbench.get_simbench_net(benchmark["code"])

    if len(net.load) == 0:
        raise ValueError("SimBench network has no loads")

    horizon = int(benchmark.get("horizon", 8))
    start_index = int(benchmark.get("start_index", 0))
    stride = int(benchmark.get("stride", 4))
    dispatchable_supply_ratio = float(benchmark.get("dispatchable_supply_ratio", 0.55))
    critical_top_k = int(benchmark.get("critical_top_k", max(1, len(net.load) // 5)))
    default_floor = float(benchmark.get("default_min_service_fraction", 0.2))
    critical_floor = float(benchmark.get("critical_min_service_fraction", 0.55))

    base_demands = {f"load_{idx}": float(row["p_mw"]) for idx, row in net.load.iterrows()}
    sorted_nodes = sorted(base_demands.items(), key=lambda item: (-item[1], item[0]))
    critical_nodes = {node_id for node_id, _ in sorted_nodes[:critical_top_k]}
    max_demand = max(base_demands.values())
    graph = _build_network_graph(net)
    slack_buses = {int(bus) for bus in net.ext_grid["bus"].tolist()} if len(net.ext_grid) > 0 else set()
    load_bus_counts = {int(bus): int((net.load["bus"] == bus).sum()) for bus in net.bus.index}
    sgen_bus_counts = {int(bus): int((net.sgen["bus"] == bus).sum()) for bus in net.bus.index} if len(net.sgen) > 0 else {}

    nodes = []
    for node_id, demand in sorted_nodes:
        load_idx = int(node_id.split("_")[1])
        bus_idx = int(net.load.at[load_idx, "bus"])
        normalized = demand / max_demand if max_demand > 0.0 else 0.0
        priority = 1.0 + 9.0 * normalized
        if node_id in critical_nodes:
            priority += 5.0
        static_features = _normalized_static_features(graph, slack_buses, load_bus_counts, sgen_bus_counts, bus_idx)
        min_service_fraction = critical_floor if node_id in critical_nodes else default_floor
        nodes.append(
            LoadNode(
                node_id=node_id,
                priority=priority,
                is_critical=node_id in critical_nodes,
                min_service_fraction=min_service_fraction,
                bus_idx=bus_idx,
                static_features=static_features,
            )
        )

    load_profiles = net.profiles["load"]
    renewable_profiles = net.profiles.get("renewables")
    total_base_demand = sum(base_demands.values())

    states = []
    for step in range(horizon):
        profile_row = load_profiles.iloc[start_index + step * stride]
        demands = {}
        total_demand = 0.0
        for idx, row in net.load.iterrows():
            node_id = f"load_{idx}"
            profile_name = str(row["profile"])
            demand_multiplier = float(profile_row[f"{profile_name}_pload"])
            demand = round(float(row["p_mw"]) * demand_multiplier, 6)
            demands[node_id] = demand
            total_demand += demand

        renewable_supply = 0.0
        if renewable_profiles is not None and len(net.sgen) > 0:
            renewable_row = renewable_profiles.iloc[start_index + step * stride]
            for _, row in net.sgen.iterrows():
                profile_name = str(row["profile"])
                renewable_supply += float(row["p_mw"]) * float(renewable_row[profile_name])

        dispatchable_supply = total_base_demand * dispatchable_supply_ratio
        available_power = round(min(total_demand, dispatchable_supply + renewable_supply), 6)
        supply_scale = available_power / max(total_demand, 1e-6)
        states.append(
            TimeStepState(
                time_index=step,
                available_power=available_power,
                demands=demands,
                outages=(),
                supply_scale=round(supply_scale, 6),
            )
        )

    name = f"{benchmark['code']}_simbench"
    adjacency_matrix = _build_load_adjacency_matrix(net, graph, sorted_nodes)
    return Scenario(
        name=name,
        nodes=tuple(nodes),
        states=tuple(states),
        adjacency_matrix=adjacency_matrix,
        metadata={
            **{key: str(value) for key, value in metadata.items()},
            "benchmark_source": benchmark["source"],
            "benchmark_name": benchmark["code"],
            "critical_top_k": str(critical_top_k),
            "dispatchable_supply_ratio": str(dispatchable_supply_ratio),
        },
    )
