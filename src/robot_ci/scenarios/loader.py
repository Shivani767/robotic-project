"""Scenario loading, saving and grid expansion."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from robot_ci.core.errors import ScenarioError
from robot_ci.core.io import load_yaml, write_yaml
from robot_ci.core.logging import get_logger
from robot_ci.models.scenario import ScenarioConfig, VariableSpec

__all__ = [
    "expand_grid",
    "load_scenario",
    "save_scenario",
    "scenario_from_dict",
    "set_by_path",
    "sweep_scenario",
]

_LOG = get_logger(__name__)


def scenario_from_dict(data: dict[str, Any], *, source: str = "<dict>") -> ScenarioConfig:
    """Validate a mapping into a :class:`ScenarioConfig`.

    Args:
        data: The mapping, typically parsed from YAML.
        source: Where it came from, used in error messages.

    Returns:
        The validated scenario.

    Raises:
        ScenarioError: If validation fails, with the underlying Pydantic error
            attached as structured detail.
    """
    try:
        return ScenarioConfig.model_validate(data)
    except Exception as exc:
        raise ScenarioError(
            f"invalid scenario from {source}: {exc}",
            code="scenario.invalid",
            source=source,
            detail=str(exc),
        ) from exc


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario from a YAML or JSON file.

    Raises:
        ConfigurationError: If the file is missing or is not valid YAML.
        ScenarioError: If the file parses but is not a valid scenario.
    """
    from robot_ci.core.errors import ConfigurationError

    data = load_yaml(path)
    try:
        return ScenarioConfig.model_validate(data)
    except Exception as exc:
        raise ScenarioError(
            f"invalid scenario in {path}: {exc}",
            code="scenario.invalid",
            source=str(path),
            detail=str(exc),
        ) from exc
    del ConfigurationError


def save_scenario(scenario: ScenarioConfig, path: str | Path) -> Path:
    """Write a scenario to YAML, returning the path written."""
    target = write_yaml(path, scenario.model_dump(mode="json"))
    _LOG.info("wrote scenario %s -> %s", scenario.scenario_id, target)
    return target


def set_by_path(config: ScenarioConfig, path: str, value: float) -> ScenarioConfig:
    """Return a copy of ``config`` with one dotted field path set.

    Dotted paths address nested models, e.g. ``"sensors.depth.sigma_mm"``.
    Indexing is supported for sequences, e.g. ``"objects.0.position.x"``.

    Args:
        config: The scenario to modify (not mutated).
        path: Dotted field path.
        value: New scalar value.

    Returns:
        A new, validated scenario.

    Raises:
        ScenarioError: If the path does not resolve or the result is invalid.
    """
    if not path:
        raise ScenarioError("empty variable path", code="scenario.empty_path")

    data = config.model_dump(mode="python")
    parts = path.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        cursor = _descend(cursor, part, path)
    leaf = parts[-1]
    if isinstance(cursor, list):
        _assign_index(cursor, leaf, value, path)
    elif isinstance(cursor, tuple):
        # model_dump returns tuples; converting to a list lets the parent
        # re-validate the modified sequence.
        _replace_index(cursor, leaf, value, path)
    else:
        if leaf not in cursor:
            raise ScenarioError(
                f"path {path!r} does not exist on ScenarioConfig (leaf {leaf!r})",
                code="scenario.unknown_path",
                path=path,
                available=sorted(cursor)[:20] if isinstance(cursor, dict) else [],
            )
        cursor[leaf] = value
    _auto_enable_corruption(data, path, value)
    return scenario_from_dict(data, source=f"set_by_path({path})")


#: A sweep that writes a non-zero corruption magnitude must also flip the
#: corruption's ``enabled`` master switch, or the sweep silently measures
#: nothing: every cell loads validly, every episode runs, and the metrics are
#: identical across the grid. The catalogue builders set ``enabled`` by hand,
#: but grid expansion writes magnitudes programmatically, so the parent chain
#: is armed here, in one place.
_CORRUPTION_PARENTS: tuple[tuple[str, ...], ...] = (
    ("sensors", "depth"),
    ("sensors", "rgb"),
    ("sensors", "camera"),
    ("sensors", "robot_state"),
    ("sensors", "force_torque"),
)


def _auto_enable_corruption(data: dict[str, Any], path: str, value: float) -> None:
    """Arm the ``enabled`` flags above a swept corruption magnitude."""
    parts = tuple(path.split("."))
    for parent in _CORRUPTION_PARENTS:
        if len(parts) > len(parent) and parts[: len(parent)] == parent:
            if parts[len(parent)] == "enabled":
                return  # the caller manages the switch explicitly
            node: Any = data
            for part in parent:
                node = node.get(part) if isinstance(node, dict) else None
            if isinstance(node, dict) and float(value) != 0.0:
                node["enabled"] = True
            return
    if parts[:2] == ("sensors", "camera") and "latency" in parts and parts[-1] != "enabled":
        latency = data.get("sensors", {}).get("camera", {}).get("latency")
        camera = data.get("sensors", {}).get("camera")
        if isinstance(latency, dict) and float(value) != 0.0:
            latency["enabled"] = True
            if isinstance(camera, dict):
                camera["enabled"] = True


def _assign_index(container: list[Any], leaf: str, value: float, path: str) -> None:
    """Set ``container[index] = value`` in place."""
    index = _as_index(leaf, path)
    if index >= len(container):
        raise ScenarioError(
            f"index {index} out of range for path {path!r}",
            code="scenario.path_out_of_range",
            path=path,
        )
    container[index] = value


def _replace_index(cursor: tuple[Any, ...], part: str, value: float, path: str) -> None:
    """Return a list copy of ``cursor`` with one element replaced."""
    raise ScenarioError(
        f"cannot modify sequence segment {part!r} of path {path!r}; "
        "sweep a leaf field instead (for example objects.0.size_m)",
        code="scenario.sequence_immutable",
        path=path,
    )


def _descend(cursor: Any, part: str, path: str) -> Any:
    """Return the next container in a dotted path traversal.

    ``model_dump`` returns tuples for sequence fields, so both ``list`` and
    ``tuple`` must be treated as indexable here.
    """
    if isinstance(cursor, (list, tuple)):
        return cursor[_as_index(part, path)]
    if not isinstance(cursor, dict) or part not in cursor:
        raise ScenarioError(
            f"path {path!r} does not resolve at segment {part!r}",
            code="scenario.unknown_path",
            path=path,
        )
    return cursor[part]


def _as_index(part: str, path: str) -> int:
    try:
        return int(part)
    except ValueError as exc:
        raise ScenarioError(
            f"path {path!r} expects an integer index at {part!r}",
            code="scenario.bad_index",
            path=path,
        ) from exc


def expand_grid(
    base: ScenarioConfig, variables: tuple[VariableSpec, ...]
) -> list[ScenarioConfig]:
    """Expand a base scenario over the full cartesian product of variables.

    Each cell gets a unique, deterministic ``scenario_id`` of the form
    ``<base>__<var>=<value>__<var>=<value>``, so results are self-describing and
    can be joined back to their condition without a side table.

    Args:
        base: The scenario all cells derive from.
        variables: Variables to sweep. With none, returns ``[base]``.

    Returns:
        One scenario per grid cell, in a deterministic order.

    Raises:
        ScenarioError: If a variable path does not resolve.
    """
    if not variables:
        return [base]

    names = [v.name for v in variables]
    combos = itertools.product(*[v.values for v in variables])
    cells: list[ScenarioConfig] = []
    for combo in combos:
        scenario = base
        for spec, value in zip(variables, combo, strict=True):
            scenario = set_by_path(scenario, spec.path, float(value))
        suffix = "__".join(f"{n}={v:g}" for n, v in zip(names, combo, strict=True))
        cells.append(
            scenario.model_copy(
                update={"scenario_id": f"{base.scenario_id}__{suffix}", "family": base.family}
            )
        )
    _LOG.info("expanded grid: %d variables -> %d cells", len(variables), len(cells))
    return cells


def sweep_scenario(
    base: ScenarioConfig, variable: VariableSpec, *, seed: int | None = None
) -> Iterator[ScenarioConfig]:
    """Yield one scenario per value of a single variable.

    Args:
        base: The scenario to derive from.
        variable: The variable to sweep.
        seed: When given, each yielded scenario gets ``seed + index`` so that a
            1-D sweep is also a seed sweep.

    Yields:
        Scenarios in ascending value order.
    """
    for index, value in enumerate(variable.values):
        cell = set_by_path(base, variable.path, float(value))
        updates: dict[str, Any] = {"scenario_id": f"{base.scenario_id}__{variable.name}={value:g}"}
        if seed is not None:
            updates["seed"] = int(seed) + index
        yield cell.model_copy(update=updates)
