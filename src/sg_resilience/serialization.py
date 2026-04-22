from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .scenario_schema import Scenario


def write_json(data: object, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def scenario_to_json_dict(scenario: Scenario) -> dict[str, object]:
    return asdict(scenario)
