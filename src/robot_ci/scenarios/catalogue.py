"""Built-in scenario catalogue.

Each entry is a *named family* of experimental conditions. They map onto the B1-B10
benchmark suite in the project specification:

======  ========================  =========================================
Benchmark  Family                  Perturbs
======  ========================  =========================================
B1      ``nominal``               nothing (control condition)
B2      ``depth_noise``           ``sensors.depth.sigma_mm``
B3      ``depth_bias``            ``sensors.depth.bias_mm``
B4      ``occlusion``             ``sensors.rgb.occlusion_percent``
B5      ``latency``               ``sensors.camera.latency.delay_ms``
B6      ``joint_noise``           ``sensors.robot_state.joint_noise_std_rad``
B7      ``pose_perturbation``     explicit object position offset
B8      ``lighting``              ``environment.lighting_lux``
B9      ``clutter``               ``environment.clutter_count``
B10     ``combined``              several axes at once
======  ========================  =========================================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

from robot_ci.core.errors import ScenarioError
from robot_ci.models.scenario import ScenarioConfig

__all__ = ["FAMILIES", "FamilyInfo", "ScenarioFamily", "family_names", "get_family", "list_families"]


@dataclass(frozen=True)
class ScenarioFamily:
    """A named family of conditions and the axis it perturbs.

    Attributes:
        name: Family key, also used as the default ``scenario_id`` prefix.
        benchmark: Which benchmark in the project specification this covers.
        description: One-line description for reports and the CLI.
        variable_name: Human-readable name of the swept variable.
        variable_path: Dotted path of the field inside ``ScenarioConfig``.
        values: Default sweep values, ascending.
        unit: Unit label for the swept variable.
        apply: Callable that produces a scenario for one value.
    """

    name: str
    benchmark: str
    description: str
    variable_name: str
    variable_path: str
    values: tuple[float, ...]
    unit: str
    apply: Callable[..., ScenarioConfig] = field(repr=False)

    def build(self, value: float, *, seed: int = 0, scenario_id: str | None = None) -> ScenarioConfig:
        """Return the scenario for one value of this family's axis.

        Args:
            value: Position along this family's swept axis.
            seed: Scenario seed.
            scenario_id: Override for the generated identifier.

        Returns:
            A validated scenario.

        Raises:
            ScenarioError: If the family has no builder registered.
        """
        builder = self.apply
        if not callable(builder):  # pragma: no cover - dataclass misuse
            raise ScenarioError(
                f"family {self.name!r} has no builder", code="scenario.family_invalid"
            )
        return builder(
            float(value),
            seed=seed,
            scenario_id=scenario_id or f"{self.name}_{value:g}".replace(".", "p"),
        )


def _build(
    scenario_id: str,
    family: str,
    seed: int,
    description: str,
    **overrides: object,
) -> ScenarioConfig:
    """Construct a scenario from partial nested overrides.

    Pydantic accepts nested mappings for model-typed fields, so the builders
    below can write ``sensors={"depth": {...}}`` and stay readable. Routing every
    construction through this helper keeps that ergonomic call in one place and
    gives a type checker a single, honest signature to inspect.
    """
    return ScenarioConfig.model_validate(
        {
            "scenario_id": scenario_id,
            "family": family,
            "seed": seed,
            "description": description,
            **overrides,
        }
    )


def _nominal(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """The control condition: no corruption of any kind.

    ``value`` is ignored on purpose. The nominal family exists to give every
    sweep a defined zero point, not to perturb anything.
    """
    del value
    return _build(scenario_id, "nominal", seed, "Uncorrupted control condition.")


def _depth_noise(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Increasing zero-mean Gaussian depth noise (B2)."""
    return _build(
        scenario_id,
        "depth_noise",
        seed,
        f"Gaussian depth noise, sigma={value:g} mm.",
        sensors={"depth": {"enabled": True, "sigma_mm": value}},
    )


def _depth_bias(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Systematic depth offset (B3)."""
    return _build(
        scenario_id,
        "depth_bias",
        seed,
        f"Systematic depth bias, {value:g} mm.",
        sensors={"depth": {"enabled": True, "bias_mm": value}},
    )


def _occlusion(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Increasing synthetic occluder coverage (B4)."""
    return _build(
        scenario_id,
        "occlusion",
        seed,
        f"RGB occlusion covering {value:g}% of the image.",
        sensors={"rgb": {"enabled": True, "occlusion_percent": value}},
    )


def _latency(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Increasing camera delivery latency (B5)."""
    return _build(
        scenario_id,
        "latency",
        seed,
        f"Camera latency of {value:g} ms.",
        sensors={"camera": {"enabled": True, "latency": {"enabled": True, "delay_ms": value}}},
    )


def _joint_noise(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Increasing joint-encoder noise (B6)."""
    return _build(
        scenario_id,
        "joint_noise",
        seed,
        f"Joint encoder noise sigma={value:g} rad.",
        sensors={"robot_state": {"enabled": True, "joint_noise_std_rad": value}},
    )


def _pose_perturbation(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Initial object pose error (B7): the object starts displaced by ``value`` cm.

    The swept axis is the object's world X coordinate, so a grid expansion
    displaces the object along X by a known amount. The nominal pose is pinned
    explicitly; otherwise the object would be sampled from the seed and the
    displacement would be relative to an unknown reference.
    """
    from robot_ci.models.geometry import Vec3
    from robot_ci.models.scenario import ObjectSpec

    x = 0.30 + value / 100.0
    return _build(
        scenario_id,
        "pose_perturbation",
        seed,
        f"Object displaced to x={x:.3f} m from its nominal pose.",
        objects=(
            ObjectSpec(id="target_cube", is_target=True, position=Vec3.of(x, 0.0, 0.025)),
        ),
    )


def _lighting(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Different illuminance levels (B8)."""
    return _build(
        scenario_id, "lighting", seed, f"Illuminance {value:g} lux.",
        environment={"lighting_lux": value},
    )


def _clutter(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Increasing distractor count (B9)."""
    from robot_ci.models.environment import ObjectCategory
    from robot_ci.models.scenario import ObjectSpec

    count = int(round(value))
    extras = tuple(
        ObjectSpec(
            id=f"clutter_{index}",
            category=ObjectCategory.CUBE,
            is_target=False,
            is_clutter=True,
            color_rgb=(0.4, 0.4, 0.45),
        )
        for index in range(count)
    )
    return _build(
        scenario_id,
        "clutter",
        seed,
        f"{count} distractor objects on the table.",
        environment={"clutter_count": count, "clutter_separation_m": 0.09},
        objects=(ObjectSpec(id="target_cube", is_target=True), *extras),
    )


def _combined(value: float, *, seed: int, scenario_id: str) -> ScenarioConfig:
    """Several perturbation axes raised together by one stress level (B10).

    The swept axis is ``environment.lighting_lux``, which decreases monotonically
    with stress so the variable values stay ascending; every other axis is scaled
    by the same derived stress level. ``0`` stress reproduces the nominal
    condition and the maximum reproduces the hardest condition defined here.
    """
    stress = float(np.clip(value / 100.0, 0.0, 1.0))
    blur = 1 + 2 * int(round(3.0 * stress))  # odd kernel size in [1, 7]
    return _build(
        scenario_id,
        "combined",
        seed,
        f"Combined stress: depth noise, occlusion, blur, latency, joint noise, "
        f"clutter and lighting at {value:g} lux.",
        environment={"lighting_lux": value, "clutter_count": int(6 * stress)},
        sensors={
            "depth": {"enabled": True, "sigma_mm": 20.0 * stress, "bias_mm": 5.0 * stress},
            "rgb": {
                "enabled": True,
                "occlusion_percent": 50.0 * stress,
                "blur_kernel_px": blur,
            },
            "camera": {
                "enabled": True,
                "latency": {"enabled": True, "delay_ms": 120.0 * stress},
            },
            "robot_state": {"enabled": True, "joint_noise_std_rad": 0.05 * stress},
        },
    )



FAMILIES: dict[str, ScenarioFamily] = {
    f.name: f
    for f in (
        ScenarioFamily("nominal", "B1", "Uncorrupted control condition.", "none", "", (0.0,), "", _nominal),
        ScenarioFamily("depth_noise", "B2", "Gaussian depth noise (mm).", "depth_noise_mm", "sensors.depth.sigma_mm", (0.0, 2.0, 5.0, 10.0, 15.0, 20.0), "mm", _depth_noise),
        ScenarioFamily("depth_bias", "B3", "Systematic depth bias (mm).", "depth_bias_mm", "sensors.depth.bias_mm", (0.0, 2.0, 5.0, 10.0, 20.0), "mm", _depth_bias),
        ScenarioFamily("occlusion", "B4", "RGB occluder coverage (%).", "occlusion_percent", "sensors.rgb.occlusion_percent", (0.0, 10.0, 20.0, 40.0, 60.0), "%", _occlusion),
        ScenarioFamily("latency", "B5", "Camera delivery latency (ms).", "latency_ms", "sensors.camera.latency.delay_ms", (0.0, 20.0, 50.0, 100.0, 200.0), "ms", _latency),
        ScenarioFamily("joint_noise", "B6", "Joint encoder noise (rad).", "joint_noise_std_rad", "sensors.robot_state.joint_noise_std_rad", (0.0, 0.005, 0.01, 0.02, 0.05), "rad", _joint_noise),
        ScenarioFamily("pose_perturbation", "B7", "Initial object X displacement (cm).", "object_x_cm", "objects.0.position.x", (0.0, 1.0, 2.0, 5.0, 10.0), "cm", _pose_perturbation),
        ScenarioFamily("lighting", "B8", "Scene illuminance (lux).", "lighting_lux", "environment.lighting_lux", (40.0, 100.0, 200.0, 500.0, 2000.0), "lux", _lighting),
        ScenarioFamily("clutter", "B9", "Distractor object count.", "clutter_count", "environment.clutter_count", (0.0, 2.0, 5.0, 9.0), "count", _clutter),
        ScenarioFamily("combined", "B10", "Multiple simultaneous perturbations.", "lighting_lux", "environment.lighting_lux", (100.0, 200.0, 300.0, 400.0, 500.0), "lux", _combined),
    )
}


def family_names() -> tuple[str, ...]:
    """Return every catalogue family name, in benchmark order."""
    return tuple(FAMILIES)


class FamilyInfo(TypedDict):
    """Serializable description of one catalogue family."""

    name: str
    benchmark: str
    description: str
    variable: str
    path: str
    values: list[float]
    unit: str


def list_families() -> tuple[FamilyInfo, ...]:
    """Return a serializable description of every family, for the CLI and docs."""
    return tuple(
        FamilyInfo(
            name=f.name,
            benchmark=f.benchmark,
            description=f.description,
            variable=f.variable_name,
            path=f.variable_path,
            values=list(f.values),
            unit=f.unit,
        )
        for f in FAMILIES.values()
    )


def get_family(name: str) -> ScenarioFamily:
    """Return one family by name.

    Args:
        name: Family key.

    Returns:
        The :class:`ScenarioFamily`.

    Raises:
        ScenarioError: If the family is unknown.
    """
    try:
        return FAMILIES[name]
    except KeyError as exc:
        raise ScenarioError(
            f"unknown scenario family {name!r}",
            code="scenario.unknown_family",
            requested=name,
            available=family_names(),
        ) from exc
