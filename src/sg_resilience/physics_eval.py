from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import yaml


def load_benchmark_net(config_path: str | Path) -> Any:
    raw = cast(dict[str, Any], yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    benchmark = cast(dict[str, Any], raw["benchmark"])

    if benchmark["source"] == "pandapower":
        networks = cast(Any, importlib.import_module("pandapower.networks"))
        return getattr(networks, benchmark["name"])()

    if benchmark["source"] == "simbench":
        simbench = cast(Any, importlib.import_module("simbench"))
        return simbench.get_simbench_net(benchmark["code"])

    raise ValueError(f"Unsupported benchmark source: {benchmark['source']}")


def evaluate_allocation_feasibility(
    config_path: str | Path,
    comparison_full_path: str | Path,
    max_scenarios: int | None = None,
    max_steps: int | None = None,
) -> dict[str, object]:
    pandapower = cast(Any, importlib.import_module("pandapower"))
    raw = json.loads(Path(comparison_full_path).read_text(encoding="utf-8"))
    results: dict[str, object] = {}

    for scenario_index, (scenario_name, baselines) in enumerate(raw.items()):
        if max_scenarios is not None and scenario_index >= max_scenarios:
            break
        scenario_results: dict[str, object] = {}
        for baseline_name, payload in cast(dict[str, Any], baselines).items():
            per_step_results = []
            for step_index, step in enumerate(cast(list[dict[str, Any]], payload["per_step"])):
                if max_steps is not None and step_index >= max_steps:
                    break
                net = load_benchmark_net(config_path)
                allocation = cast(dict[str, float], step["allocation"])
                for idx in net.load.index:
                    node_id = f"load_{idx}"
                    if node_id in allocation:
                        net.load.at[idx, "p_mw"] = allocation[node_id]

                converged = False
                for run_kwargs in (
                    {"init": "dc", "algorithm": "nr", "numba": True, "check_connectivity": True},
                    {"init": "flat", "algorithm": "bfsw", "numba": True, "check_connectivity": True},
                    {"init": "flat", "algorithm": "nr", "numba": False, "check_connectivity": True},
                    {"init": "flat", "algorithm": "gs", "numba": False, "check_connectivity": True},
                ):
                    try:
                        pandapower.runpp(net, **run_kwargs)
                        converged = True
                        break
                    except Exception:
                        continue

                max_loading = None
                min_vm = None
                max_vm = None
                if converged:
                    if len(net.res_line) > 0 and "loading_percent" in net.res_line:
                        max_loading = float(net.res_line["loading_percent"].max())
                    if len(net.res_trafo) > 0 and "loading_percent" in net.res_trafo:
                        trafo_loading = float(net.res_trafo["loading_percent"].max())
                        max_loading = trafo_loading if max_loading is None else max(max_loading, trafo_loading)
                    if len(net.res_bus) > 0 and "vm_pu" in net.res_bus:
                        min_vm = float(net.res_bus["vm_pu"].min())
                        max_vm = float(net.res_bus["vm_pu"].max())

                per_step_results.append(
                    {
                        "time_index": step["time_index"],
                        "converged": converged,
                        "max_loading_percent": max_loading,
                        "min_vm_pu": min_vm,
                        "max_vm_pu": max_vm,
                    }
                )

            converged_steps = sum(1 for step in per_step_results if step["converged"])
            max_loading_values = [step["max_loading_percent"] for step in per_step_results if step["max_loading_percent"] is not None]
            min_vm_values = [step["min_vm_pu"] for step in per_step_results if step["min_vm_pu"] is not None]
            max_vm_values = [step["max_vm_pu"] for step in per_step_results if step["max_vm_pu"] is not None]
            scenario_results[baseline_name] = {
                "per_step": per_step_results,
                "summary": {
                    "converged_ratio": converged_steps / max(len(per_step_results), 1),
                    "max_loading_percent": max(max_loading_values) if max_loading_values else None,
                    "min_vm_pu": min(min_vm_values) if min_vm_values else None,
                    "max_vm_pu": max(max_vm_values) if max_vm_values else None,
                },
            }

        results[scenario_name] = scenario_results

    return results
