"""The scenario files committed under ``scenarios/`` must match the catalogue.

These tests are what stop the on-disk examples from drifting away from the
catalogue definitions after a refactor.
"""

from __future__ import annotations

import pytest

from robot_ci.core.workspace import Workspace
from robot_ci.models.scenario import VariableSpec
from robot_ci.scenarios import FAMILIES, get_family, load_scenario, sweep_scenario

SUBDIRS = ("nominal", "perception", "depth", "camera", "robot", "environment")


@pytest.fixture(scope="module")
def workspace() -> Workspace:
    return Workspace.discover()


def test_scenarios_directory_has_all_subdirs(workspace: Workspace) -> None:
    for name in SUBDIRS:
        assert (workspace.scenarios_dir / name).is_dir(), f"missing scenarios/{name}"


def test_every_scenario_file_loads(workspace: Workspace) -> None:
    files = sorted(workspace.scenarios_dir.rglob("*.yaml"))
    assert files, "no scenario files found; run scripts/generate_scenarios.py"
    for path in files:
        scenario = load_scenario(path)
        assert scenario.scenario_id
        assert scenario.config_hash().startswith("sha256:")


def test_nominal_file_exists_and_is_uncorrupted(workspace: Workspace) -> None:
    files = sorted((workspace.scenarios_dir / "nominal").glob("*.yaml"))
    assert files, "run scripts/generate_scenarios.py"
    for path in files:
        scenario = load_scenario(path)
        assert scenario.sensors.is_nominal, "the control condition must be uncorrupted"


def test_generated_files_match_the_catalogue(workspace: Workspace) -> None:
    """Every family must have exactly as many files as it has sweep values."""
    for family_name, family in FAMILIES.items():
        subdir = {
            "nominal": "nominal", "depth_noise": "depth", "depth_bias": "depth",
            "occlusion": "perception", "latency": "camera", "joint_noise": "robot",
            "pose_perturbation": "environment", "lighting": "environment",
            "clutter": "environment", "combined": "environment",
        }[family_name]
        directory = workspace.scenarios_dir / subdir
        files = sorted(directory.glob(f"{family_name}__*.yaml"))
        assert files, f"no files for family {family_name}"
        if family_name == "nominal":
            assert len(files) == 1
            continue
        variable = VariableSpec(
            name=family.variable_name, path=family.variable_path,
            values=family.values, unit=family.unit,
        )
        expected = {
            s.config_hash()
            for s in sweep_scenario(family.build(family.values[0], seed=42), variable)
        }
        actual = {load_scenario(f).config_hash() for f in files}
        assert actual == expected, f"family {family_name} files are stale"


def test_corrupted_families_really_are_corrupted(workspace: Workspace) -> None:
    """A corruption family that produces clean observations measures nothing.

    Every family's sweep includes its own zero level, which is the family's
    internal control point and is therefore expected to be uncorrupted. The
    check is that *every non-zero* level actually activates its source.
    """
    checks = {
        "depth_noise": ("depth", "depth"),
        "occlusion": ("perception", "rgb"),
        "latency": ("camera", "camera"),
        "joint_noise": ("robot", "robot_state"),
    }
    for family_name, (subdir, source) in checks.items():
        family = get_family(family_name)
        files = sorted((workspace.scenarios_dir / subdir).glob(f"{family_name}__*.yaml"))
        assert files, family_name
        del family
        for path in files:
            # Filenames are family__VALUE.yaml, so the level is recoverable.
            level = float(path.stem.rsplit("__", 1)[-1].replace("p", "."))
            if level > 0.0:
                scenario = load_scenario(path)
                assert source in scenario.sensors.active_sources(), path.name


def test_every_family_is_representative(workspace: Workspace) -> None:
    """At least one file per family, so no benchmark is silently missing."""
    names = {p.name.split("__")[0] for p in workspace.scenarios_dir.rglob("*.yaml")}
    assert names == set(FAMILIES)
