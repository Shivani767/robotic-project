"""Scenario, policy and experiment configuration.

A :class:`ScenarioConfig` is the *complete, self-describing* description of one
experimental condition. Given only a scenario file and a seed, any backend can
reproduce the episode bit-for-bit. That property is what makes counterexample
reproduction meaningful.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import Field, model_validator

from robot_ci.core.hashing import hash_payload
from robot_ci.models.base import VersionedModel
from robot_ci.models.environment import (
    EnvironmentConfig,
    ObjectCategory,
    TaskConfig,
)
from robot_ci.models.geometry import Quaternion, Vec3
from robot_ci.models.robot import RobotConfig
from robot_ci.models.sensors import SensorBundleConfig

__all__ = [
    "ExperimentConfig",
    "ObjectSpec",
    "PolicyComponent",
    "PolicyConfig",
    "ScenarioConfig",
    "VariableSpec",
]


class PolicyComponent(str, Enum):
    """A swappable subsystem inside a policy stack.

    These are the units the ablation engine (Phase 9) swaps independently.
    """

    PERCEPTION = "perception"
    SENSOR_FUSION = "sensor_fusion"
    PLANNER = "planner"
    CONTROLLER = "controller"
    GRIPPER = "gripper"
    RECOVERY = "recovery"


class PolicyConfig(VersionedModel):
    """Which policy to run, and which version of each component it uses.

    Example — a perception-only change::

        baseline  = PolicyConfig(name="policy_v1", components={PERCEPTION: "perception_v1"})
        candidate = PolicyConfig(name="policy_v2", components={PERCEPTION: "perception_v2"})

    The ablation engine can then build the two intermediate hybrids directly
    from these component maps.
    """

    name: str = Field(default="policy_v1", min_length=1, description="Policy identifier, e.g. 'policy_v1'.")
    version: str = Field(default="1.0.0", description="Policy version string.")
    implementation: str = Field(default="rule_based", description="Factory key in robot_ci.policies.")
    components: dict[PolicyComponent, str] = Field(
        default_factory=dict, description="Component name -> version map, for ablations."
    )
    parameters: dict[str, float] = Field(default_factory=dict, description="Numeric hyperparameters.")
    max_recovery_attempts: int = Field(default=2, ge=0, le=20, description="Recovery retries before aborting.")
    action_repeat: int = Field(default=1, ge=1, le=100, description="Low-level steps per policy action.")
    seed_offset: int = Field(default=0, description="Added to the episode seed; keeps policies from sharing tie-breaks.")

    def component_version(self, component: PolicyComponent) -> str:
        """Version of one component, falling back to the policy version."""
        return self.components.get(component, self.version)

    def with_component(self, component: PolicyComponent, version: str) -> PolicyConfig:
        """Return a copy with one component swapped — the ablation primitive."""
        merged = {**self.components, component: version}
        return self.model_copy(update={"components": merged})

    def differs_from(self, other: PolicyConfig) -> tuple[PolicyComponent, ...]:
        """Components whose versions differ between two policies.

        This is what the diagnosis engine uses to decide which ablations are
        informative, so it must consider *only* declared components.
        """
        names = set(self.components) | set(other.components)
        return tuple(
            sorted(
                (c for c in names if self.component_version(c) != other.component_version(c)),
                key=lambda c: c.value,
            )
        )


class ObjectSpec(VersionedModel):
    """One object in the scene.

    Positions are sampled from the scenario seed unless given explicitly, so a
    scenario with explicit positions is byte-reproducible across machines.
    """

    id: str = Field(default="target_cube", min_length=1, description="Object identifier.")
    category: ObjectCategory = Field(default=ObjectCategory.CUBE, description="Object shape class.")
    position: Vec3 | None = Field(default=None, description="Explicit position; None means sample from seed.")
    orientation: Quaternion = Field(default_factory=Quaternion.identity, description="Explicit orientation.")
    size_m: float = Field(default=0.04, gt=0.0, le=1.0, description="Characteristic half-size in metres.")
    is_target: bool = Field(default=True, description="Whether this is the task target.")
    is_clutter: bool = Field(default=False, description="Whether this object is a distractor.")
    color_rgb: tuple[float, float, float] = Field(default=(0.85, 0.15, 0.15), description="Nominal RGB in [0,1].")
    friction: float | None = Field(default=None, ge=0.0, le=2.0, description="Override for object friction.")

    @property
    def display_name(self) -> str:
        """Human-readable label, e.g. ``"red_cube"``."""
        names = {0.85: "red", 0.15: "dark", 0.35: "blue", 0.2: "green", 0.9: "yellow"}
        hue = names.get(round(self.color_rgb[0], 2), "grey")
        if self.color_rgb[0] < 0.3 and self.color_rgb[2] > 0.4:
            hue = "blue"
        elif self.color_rgb[1] > 0.4 and self.color_rgb[0] < 0.6:
            hue = "green"
        elif self.color_rgb[0] > 0.7 and self.color_rgb[1] > 0.6:
            hue = "yellow"
        return f"{hue}_{self.category.value}"


class ScenarioConfig(VersionedModel):
    """A complete, self-describing experimental condition.

    Attributes:
        scenario_id: Stable identifier, unique within an experiment.
        family: Benchmark family (nominal, depth, occlusion, latency, ...).
        description: Human-readable description.
        environment: Scene, lighting, clutter, obstacles, friction.
        sensors: Sensor intrinsics and the full corruption configuration.
        robot: Embodiment and control mode.
        task: Task definition, success criteria and budgets.
        objects: Objects present in the scene.
        seed: Master seed for every stochastic element of this scenario.
    """

    scenario_id: str = Field(default="nominal", min_length=1, description="Unique scenario identifier.")
    family: str = Field(default="nominal", min_length=1, description="Benchmark family name.")
    description: str = Field(default="", description="Human-readable description.")
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Scene configuration.")
    sensors: SensorBundleConfig = Field(default_factory=SensorBundleConfig, description="Sensors and corruption.")
    robot: RobotConfig = Field(default_factory=RobotConfig, description="Embodiment.")
    task: TaskConfig = Field(default_factory=TaskConfig, description="Task and success criteria.")
    objects: tuple[ObjectSpec, ...] = Field(default=(), description="Objects; empty means a single default target.")
    seed: int = Field(default=0, ge=0, description="Master seed for this scenario.")
    camera_position: Vec3 = Field(
        default_factory=lambda: Vec3.of(0.55, -0.15, 0.50), description="Camera position in the world frame."
    )
    camera_look_at: Vec3 = Field(
        default_factory=lambda: Vec3.of(0.0, 0.05, 0.04), description="Camera aim point; the workspace centre."
    )
    render_rgb: bool = Field(default=True, description="Render the colour image.")
    render_depth: bool = Field(default=True, description="Render the depth image.")
    render_segmentation: bool = Field(default=False, description="Render the segmentation image.")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.objects:
            # A single default target keeps hand-written scenario files short.
            object.__setattr__(
                self, "objects", (ObjectSpec(id="target_cube", is_target=True),)
            )
        targets = [o for o in self.objects if o.is_target]
        if len(targets) != 1:
            raise ValueError(
                f"a scenario needs exactly one target object, found {len(targets)}"
            )
        if self.sensors.intrinsics.width_px < 16 or self.sensors.intrinsics.height_px < 16:
            raise ValueError("image resolution is too small to be useful")
        return self

    @property
    def target(self) -> ObjectSpec:
        """The single target object."""
        return next(obj for obj in self.objects if obj.is_target)

    @property
    def clutter(self) -> tuple[ObjectSpec, ...]:
        """All distractor objects."""
        return tuple(obj for obj in self.objects if obj.is_clutter)

    @property
    def resolution(self) -> tuple[int, int]:
        """Rendered image size as ``(height, width)``."""
        return self.sensors.intrinsics.shape

    def config_hash(self) -> str:
        """Return ``"sha256:..."`` over the experimental conditions.

        Excludes :attr:`schema_version` so that a compatible schema bump does
        not invalidate previously recorded hashes.
        """
        return hash_payload(self.fingerprint_payload())

    def describe_conditions(self) -> dict[str, Any]:
        """Flat summary of the conditions, used for reports and CSV columns."""
        conditions: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "task": self.task.name.value,
            "seed": self.seed,
            "lighting_lux": self.environment.lighting_lux,
            "clutter_count": self.environment.clutter_count + len(self.clutter),
            "obstacle_count": self.environment.obstacle_count,
            "object_friction": self.environment.object_friction,
            "depth_noise_mm": self.sensors.depth.sigma_mm,
            "depth_bias_mm": self.sensors.depth.bias_mm,
            "depth_dropout_percent": self.sensors.depth.dropout_percent,
            "rgb_occlusion_percent": self.sensors.rgb.occlusion_percent,
            "rgb_blur_px": float(self.sensors.rgb.blur_kernel_px),
            "rgb_brightness_percent": self.environment.lighting_lux / 5.0,
            "camera_latency_ms": self.sensors.camera.latency.total_delay_ms,
            "joint_noise_std_rad": self.sensors.robot_state.joint_noise_std_rad,
            "joint_offset_rad": self.sensors.robot_state.joint_offset_rad,
        }
        return conditions

    def effective_clutter_count(self) -> int:
        """Total distractor count: configured plus explicitly listed."""
        return self.environment.clutter_count + len(self.clutter)


#: Maps the variable names used in the project specification's experiment files
#: to the schema field they sweep. Keeping the mapping here means an experiment
#: file stays short and readable while still addressing an exact field.
_WELL_KNOWN_VARIABLE_PATHS: dict[str, str] = {
    "depth_noise_mm": "sensors.depth.sigma_mm",
    "depth_bias_mm": "sensors.depth.bias_mm",
    "depth_dropout_percent": "sensors.depth.dropout_percent",
    "occlusion_percent": "sensors.rgb.occlusion_percent",
    "latency_ms": "sensors.camera.latency.delay_ms",
    "joint_noise_rad": "sensors.robot_state.joint_noise_std_rad",
    "joint_offset_rad": "sensors.robot_state.joint_offset_rad",
    "lighting_lux": "environment.lighting_lux",
    "lighting_percent": "environment.lighting_lux",
    "clutter_count": "environment.clutter_count",
    "obstacle_count": "environment.obstacle_count",
    "object_friction": "environment.object_friction",
    "object_x_cm": "objects.0.position.x",
    "rgb_blur_px": "sensors.rgb.blur_kernel_px",
}


class VariableSpec(VersionedModel):
    """One swept variable in an experiment grid.

    Example::

        variables:
          depth_noise_mm:
            values: [0, 2, 5, 10, 20]

    The dotted path addresses a nested field, e.g.
    ``"sensors.depth.sigma_mm"`` or ``"environment.lighting_lux"``.
    """

    name: str = Field(min_length=1, description="Human-readable variable name.")
    path: str = Field(default="", description="Dotted field path within ScenarioConfig.")
    values: tuple[float, ...] = Field(min_length=1, description="Values to sweep.")

    @model_validator(mode="before")
    @classmethod
    def _infer_path(cls, data: Any) -> Any:
        """Infer ``path`` from a well-known variable name.

        Experiment files in the project specification name swept variables
        (``depth_noise_mm``) without repeating the schema path. Requiring both
        would make every hand-written file verbose and error-prone, so the
        standard names resolve automatically. A name outside the table must
        supply ``path`` explicitly, and the error says so.
        """
        if not isinstance(data, dict):
            return data
        if data.get("path"):
            return data
        name = data.get("name")
        inferred = _WELL_KNOWN_VARIABLE_PATHS.get(name) if isinstance(name, str) else None
        return {**data, "path": inferred} if inferred else data
    unit: str = Field(default="", description="Unit label, e.g. 'mm'.")
    description: str = Field(default="", description="What this variable perturbs.")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.path:
            raise ValueError(
                f"variable {self.name!r} has no path; either use one of "
                f"{sorted(_WELL_KNOWN_VARIABLE_PATHS)} or add an explicit "
                "'path' naming the field inside ScenarioConfig"
            )
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"variable {self.name!r} has duplicate values")
        if list(self.values) != sorted(self.values):
            raise ValueError(
                f"variable {self.name!r} values must be sorted ascending so that "
                "boundary search has a well-defined direction"
            )
        return self

    @property
    def span(self) -> tuple[float, float]:
        """``(minimum, maximum)`` of the swept values."""
        return (float(self.values[0]), float(self.values[-1]))


class ExperimentConfig(VersionedModel):
    """A full experiment: two policies, one scenario family, a variable grid.

    Both policies run under byte-identical conditions: the same scenario, the
    same seed, the same corruption, the same task. That is enforced by
    :func:`robot_ci.experiments.runner` re-deriving the scenario for each
    policy rather than mutating it.
    """

    experiment_id: str = Field(default="exp_000", min_length=1, description="Unique experiment id.")
    description: str = Field(default="", description="What question this experiment asks.")
    baseline: PolicyConfig = Field(default_factory=lambda: PolicyConfig(name="policy_v1"), description="Baseline policy.")
    candidate: PolicyConfig = Field(default_factory=lambda: PolicyConfig(name="policy_v2", version="2.0.0"), description="Candidate policy.")
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig, description="Base scenario; variables are overlaid on it.")
    variables: tuple[VariableSpec, ...] = Field(default=(), description="Swept variables.")
    repetitions: int = Field(default=5, ge=1, le=1000, description="Independent seeds per cell.")
    seed: int = Field(default=42, ge=0, description="Master experiment seed.")
    backend: str = Field(default="mujoco", min_length=1, description="Default simulator backend name.")
    store_sensors: bool = Field(default=False, description="Persist raw sensor frames (large).")
    notes: str = Field(default="", description="Free-form notes recorded in artifacts.")

    @model_validator(mode="before")
    @classmethod
    def _accept_spec_yaml(cls, data: Any) -> Any:
        """Accept the compact YAML form used in experiment files.

        The project specification writes experiments as::

            experiment:
              id: depth_regression_001
            baseline:
              policy: policy_v1
            scenario:
              task: pick_and_place
            variables:
              depth_noise_mm:
                values: [0, 2, 5]

        while the internal model is flatter. Accepting both here means an
        experiment file matches the documented format exactly, and the
        translation lives in one place instead of being repeated by every
        caller.
        """
        if not isinstance(data, dict):
            return data
        payload = dict(data)

        # ``experiment: {id: ...}`` -> ``experiment_id``
        block = payload.pop("experiment", None)
        if isinstance(block, dict) and "id" in block:
            payload.setdefault("experiment_id", block["id"])
        elif isinstance(block, str):
            payload.setdefault("experiment_id", block)

        # ``{policy: v1}`` -> ``{name: v1}``
        for role in ("baseline", "candidate"):
            entry = payload.get(role)
            if isinstance(entry, dict) and "policy" in entry and "name" not in entry:
                payload[role] = {"name": entry["policy"]}

        # ``scenario: {task: pick_and_place}`` -> a TaskConfig mapping
        scenario = payload.get("scenario")
        if isinstance(scenario, dict) and isinstance(scenario.get("task"), str):
            payload["scenario"] = {**scenario, "task": {"name": scenario["task"]}}

        # ``scenario: {task: pick_and_place}`` -> a TaskConfig mapping
        scenario = payload.get("scenario")
        if isinstance(scenario, dict) and isinstance(scenario.get("task"), str):
            payload["scenario"] = {**scenario, "task": {"name": scenario["task"]}}

        # ``{name: {values: [...]}}`` -> tuple[VariableSpec, ...]
        variables = payload.get("variables")
        if isinstance(variables, dict):
            specs: list[dict[str, Any]] = []
            for name, spec in variables.items():
                if isinstance(spec, dict):
                    entry = {"name": name, **spec}
                else:  # a bare list of values
                    entry = {"name": name, "values": list(spec)}
                specs.append(entry)
            payload["variables"] = specs

        return payload

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.baseline.name == self.candidate.name:
            raise ValueError(
                f"baseline and candidate must differ in name "
                f"(both are {self.baseline.name!r}); otherwise the comparison is vacuous"
            )
        return self

    @property
    def grid_size(self) -> int:
        """Number of distinct variable cells (1 when no variables are swept)."""
        size = 1
        for variable in self.variables:
            size *= len(variable.values)
        return size

    @property
    def total_episodes(self) -> int:
        """Episodes per policy: cells x repetitions."""
        return self.grid_size * self.repetitions

    @property
    def total_episodes_both_policies(self) -> int:
        """Episodes across both policies."""
        return 2 * self.total_episodes

    def config_hash(self) -> str:
        """Return ``"sha256:..."`` over the whole experiment definition."""
        return hash_payload(self.fingerprint_payload())
