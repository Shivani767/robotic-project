"""Regenerate the example scenario files under ``scenarios/``.

Run with:
    .venv/bin/python scripts/generate_scenarios.py

The files are generated rather than hand-written so that they can never drift
from the catalogue definitions. ``robot-ci scenario validate`` checks them.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robot_ci.core.workspace import Workspace
from robot_ci.models.scenario import VariableSpec
from robot_ci.scenarios import (
    get_family,
    save_scenario,
    sweep_scenario,
)

# family -> subdirectory under scenarios/
PLACEMENT = {
    "nominal": "nominal",
    "depth_noise": "depth",
    "depth_bias": "depth",
    "occlusion": "perception",
    "latency": "camera",
    "joint_noise": "robot",
    "pose_perturbation": "environment",
    "lighting": "environment",
    "clutter": "environment",
    "combined": "environment",
}


def main() -> int:
    """Entry point. Returns a POSIX exit code."""
    workspace = Workspace(root=REPO)
    written = 0
    for family_name, subdir in PLACEMENT.items():
        family = get_family(family_name)
        out_dir = workspace.scenarios_dir / subdir
        if family_name == "nominal":
            scenarios = [family.build(family.values[0], seed=42, scenario_id="nominal")]
        else:
            variable = VariableSpec(
                name=family.variable_name,
                path=family.variable_path,
                values=family.values,
                unit=family.unit,
            )
            scenarios = list(sweep_scenario(family.build(family.values[0], seed=42), variable))
        for scenario, value in zip(scenarios, family.values, strict=True):
            # Filenames carry only the swept value: family__VALUE.yaml. Keeping
            # them minimal makes the directory listing readable and lets tests
            # recover the level from the name.
            token = f"{value:g}".replace(".", "p")
            save_scenario(scenario, out_dir / f"{family_name}__{token}.yaml")
            written += 1
        print(f"{family_name:<18} {len(scenarios):>2} files -> {workspace.relative(out_dir)}")
    print(f"\ntotal: {written} scenario files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
