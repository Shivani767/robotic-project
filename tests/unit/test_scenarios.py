"""Scenario catalogue, loading and grid expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_ci.core.errors import ScenarioError
from robot_ci.models.scenario import ScenarioConfig, VariableSpec
from robot_ci.scenarios import (
    FAMILIES,
    expand_grid,
    get_family,
    list_families,
    load_scenario,
    save_scenario,
    scenario_from_dict,
    set_by_path,
    sweep_scenario,
)


def test_catalogue_covers_b1_to_b10() -> None:
    benchmarks = {f.benchmark for f in FAMILIES.values()}
    assert benchmarks == {f"B{i}" for i in range(1, 11)}


def test_every_family_builds() -> None:
    for name, family in FAMILIES.items():
        scenario = family.build(family.values[0], seed=3)
        assert scenario.scenario_id.startswith(name)


def test_family_values_are_sorted() -> None:
    for family in FAMILIES.values():
        assert list(family.values) == sorted(family.values), family.name


def test_unknown_family_raises() -> None:
    with pytest.raises(ScenarioError, match="unknown scenario family"):
        get_family("not_a_family")


def test_set_by_path_nested() -> None:
    scenario = set_by_path(ScenarioConfig(), "sensors.depth.sigma_mm", 7.5)
    assert scenario.sensors.depth.sigma_mm == 7.5


def test_set_by_path_does_not_mutate_the_original() -> None:
    original = ScenarioConfig()
    set_by_path(original, "sensors.depth.sigma_mm", 7.5)
    assert original.sensors.depth.sigma_mm == 0.0


def test_set_by_path_unknown_path_raises() -> None:
    with pytest.raises(ScenarioError, match="does not exist"):
        set_by_path(ScenarioConfig(), "sensors.depth.not_a_field", 1.0)


def test_set_by_path_empty_raises() -> None:
    with pytest.raises(ScenarioError, match="empty variable path"):
        set_by_path(ScenarioConfig(), "", 1.0)


def test_set_by_path_indexed() -> None:
    scenario = set_by_path(ScenarioConfig(), "objects.0.size_m", 0.05)
    assert scenario.target.size_m == 0.05


def test_expand_grid_size() -> None:
    cells = expand_grid(
        ScenarioConfig(),
        (
            VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(0.0, 5.0, 10.0)),
            VariableSpec(name="o", path="sensors.rgb.occlusion_percent", values=(0.0, 20.0)),
        ),
    )
    assert len(cells) == 6


def test_expand_grid_ids_are_self_describing() -> None:
    cells = expand_grid(
        ScenarioConfig(),
        (VariableSpec(name="depth_mm", path="sensors.depth.sigma_mm", values=(0.0, 10.0)),),
    )
    assert cells[0].scenario_id.endswith("depth_mm=0")
    assert cells[1].scenario_id.endswith("depth_mm=10")


def test_expand_grid_is_deterministic() -> None:
    variable = VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(0.0, 5.0))
    first = [c.config_hash() for c in expand_grid(ScenarioConfig(), (variable,))]
    second = [c.config_hash() for c in expand_grid(ScenarioConfig(), (variable,))]
    assert first == second


def test_expand_grid_with_no_variables_returns_base() -> None:
    base = ScenarioConfig()
    assert expand_grid(base, ()) == [base]


def test_sweep_yields_ascending_values() -> None:
    variable = VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(0.0, 2.0, 5.0))
    values = [s.sensors.depth.sigma_mm for s in sweep_scenario(ScenarioConfig(), variable)]
    assert values == [0.0, 2.0, 5.0]


def test_sweep_with_seed_advances_seeds() -> None:
    variable = VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(0.0, 2.0, 5.0))
    seeds = [s.seed for s in sweep_scenario(ScenarioConfig(), variable, seed=100)]
    assert seeds == [100, 101, 102]


def test_roundtrip_yaml(tmp_path: Path) -> None:
    original = get_family("depth_noise").build(10.0, seed=9)
    path = save_scenario(original, tmp_path / "s.yaml")
    loaded = load_scenario(path)
    assert loaded.config_hash() == original.config_hash()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    from robot_ci.core.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="not found"):
        load_scenario(tmp_path / "missing.yaml")


def test_load_invalid_scenario_raises_scenario_error(tmp_path: Path) -> None:
    """Parses fine, but the content is not a scenario."""
    path = tmp_path / "bad.yaml"
    path.write_text("scenario_id: x\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="invalid scenario"):
        load_scenario(path)


def test_load_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    """The file itself cannot be parsed, which is a different failure mode."""
    from robot_ci.core.errors import ConfigurationError

    path = tmp_path / "broken.yaml"
    path.write_text("a: [1, 2\nb: {\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_scenario(path)


def test_scenario_from_dict_reports_source() -> None:
    with pytest.raises(ScenarioError, match="my_source"):
        scenario_from_dict({"bogus": 1}, source="my_source")


def test_list_families_is_serializable() -> None:
    entries = list_families()
    assert len(entries) == len(FAMILIES)
    for entry in entries:
        assert {"name", "benchmark", "values", "unit"} <= set(entry)


def test_variable_spec_rejects_unsorted_values() -> None:
    with pytest.raises(Exception, match="sorted"):
        VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(10.0, 0.0, 5.0))


def test_variable_spec_rejects_duplicates() -> None:
    with pytest.raises(Exception, match="duplicate"):
        VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(0.0, 5.0, 5.0))


def test_variable_spec_span() -> None:
    variable = VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(0.0, 5.0, 20.0))
    assert variable.span == (0.0, 20.0)


def test_default_scenario_is_a_task_that_can_be_solved() -> None:
    """The fallback condition must be solvable, or every result drawn from it is void.

    Commands with an optional ``--scenario`` used to fall back to a bare
    ``ScenarioConfig()``. That object has no objects placed and none of the
    control settings a grasp depends on, so every episode on it fails -- and
    ``robot-ci counterexample search`` reported, with no hedging:

        no counterexample found
        note: both arms fail at nominal, so no baseline success exists

    which reads as a measured null result and is in fact an empty scene. A
    false negative is worse than a crash here, because the search is a CI gate:
    it would pass a release while having evaluated nothing.
    """
    from robot_ci.models.scenario import ScenarioConfig
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario()
    assert scenario.objects, "the default condition must contain an object to grasp"
    assert scenario.sensors.is_nominal, "the default condition must be uncorrupted"
    # The placement goal has to be on the table. An off-table goal is a nominal
    # condition in name only: the arm cannot place the object there, so every
    # episode fails for a reason that has nothing to do with the policy.
    goal = scenario.task.target_pose.position
    extent = scenario.environment.table_extent_m
    assert abs(goal.x) < extent[0], f"goal x={goal.x} is off a {extent[0]} m table"
    assert abs(goal.y) < extent[1], f"goal y={goal.y} is off a {extent[1]} m table"

    # And reproducible, since it is what a bare invocation searches.
    assert default_scenario().seed == scenario.seed
    assert default_scenario(seed=7).seed == 7

    # The real defect was the seed, not the scene: a bare ScenarioConfig is
    # perfectly well-formed, but its default seed 0 is a nominal condition this
    # arm cannot solve (measured: 55 mm placement error, 44 mm lift, against
    # 5.2 mm and 50 mm on the pinned default). A search on it stopped after a
    # single evaluation and reported a false "both arms fail at nominal".
    # Solvability itself is a simulator property and is covered by
    # tests/integration, not asserted here.

    assert ScenarioConfig().seed != default_scenario().seed, (
        "if the bare default ever becomes the pinned seed, the reason this "
        "helper exists needs re-checking"
    )
