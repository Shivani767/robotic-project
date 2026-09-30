r"""The simulator backend interface.

This module is the single most important boundary in the project. Everything
above it (policies, regression detection, counterexample search, boundary
mapping, statistics) imports **only** from here and never imports ``mujoco``.
Everything below it is an interchangeable implementation.

    MockBackend  |  SyntheticBackend  |  MuJoCoBackend   [Gazebo: later]
        \\              |                 /
         \\             |                /
          +-- SimulatorBackend (ABC) --+

Design rules
------------
* **No global state.** A backend instance owns at most one active episode; the
  caller owns the lifecycle. A backend can be reused across episodes but never
  shares state between them.
* **Determinism.** ``reset`` fully determines the episode from
  ``(scenario, episode_seed)``. Two identical calls must produce identical
  observations.
* **Ground-truth separation.** :meth:`SimulatorBackend.observe` returns an
  :class:`Observation` (no ground truth). Ground truth is retrieved separately
  through :meth:`SimulatorBackend.ground_truth`, which the policy layer never
  calls. This split is what makes perception evaluation honest.
* **Lazily-imported implementations.** The registry maps names to import paths
  so that ``import robot_ci.regression`` does not import ``mujoco``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robot_ci.core.errors import BackendUnavailableError, SimulationError
from robot_ci.core.version import SIMULATOR_VERSION
from robot_ci.models.environment import TaskName
from robot_ci.models.geometry import Pose
from robot_ci.models.robot import RobotConfig
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import Action, GroundTruthState, Observation, RobotState, SensorFrame

__all__ = [
    "BackendCapabilities",
    "BackendInfo",
    "EpisodeHandle",
    "SimulatorBackend",
    "TerminationReason",
    "Transition",
    "create_backend",
    "list_backends",
    "register_backend",
]


class TerminationReason(str, Enum):
    """Why an episode stopped."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    MAX_STEPS = "max_steps"
    ERROR = "error"
    ABORTED = "aborted"


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can and cannot do.

    The experiment engine uses this to refuse to run a configuration that the
    backend cannot honour, rather than silently degrading.
    """

    name: str
    physics: bool = False
    """True when the backend runs a contact-dynamics model (friction, compliance)."""
    collision: bool = False
    rgb: bool = False
    depth: bool = False
    segmentation: bool = False
    force_torque: bool = False
    joint_efforts: bool = False
    grasp_latch: bool = False
    deterministic: bool = True
    max_dof: int = 12
    supported_tasks: tuple[TaskName, ...] = field(default_factory=tuple)
    requires_gpu: bool = False
    notes: str = ""

    def supports(self, task: TaskName) -> bool:
        """Whether this backend can execute ``task``."""
        return not self.supported_tasks or task in self.supported_tasks

    def missing_for(self, scenario: ScenarioConfig) -> tuple[str, ...]:
        """Return *optional* capabilities ``scenario`` requests but this lacks.

        These are degradable: the experiment runner downgrades the request
        (e.g. disables RGB) and records the downgrade in the run metadata, rather
        than refusing to run. Hard incompatibilities come from
        :meth:`incompatible_for`.
        """
        missing: list[str] = []
        if scenario.render_rgb and not self.rgb:
            missing.append("rgb")
        if scenario.render_depth and not self.depth:
            missing.append("depth")
        if scenario.render_segmentation and not self.segmentation:
            missing.append("segmentation")
        if scenario.robot.has_force_torque and not self.force_torque:
            missing.append("force_torque")
        return tuple(missing)

    def incompatible_for(self, scenario: ScenarioConfig) -> tuple[str, ...]:
        """Return capabilities that make the scenario impossible to run at all.

        Unlike :meth:`missing_for`, these are never silently downgraded.
        """
        fatal: list[str] = []
        if scenario.robot.dof > self.max_dof:
            fatal.append(f"dof>{self.max_dof}")
        if not self.supports(scenario.task.name):
            fatal.append(f"task:{scenario.task.name.value}")
        if scenario.task.requires_grasping and not (self.physics or self.grasp_latch):
            fatal.append("task_requires_grasping")
        return tuple(fatal)


@dataclass(frozen=True)
class BackendInfo:
    """Identity of a backend implementation, recorded in every artifact."""

    name: str
    version: str = SIMULATOR_VERSION
    implementation: str = "robot_ci"
    capabilities: BackendCapabilities | None = None
    extras: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for ``summary.json``."""
        return {
            "name": self.name,
            "version": self.version,
            "implementation": self.implementation,
            "extras": dict(self.extras),
        }


@dataclass
class EpisodeHandle:
    """Opaque handle identifying one running episode.

    The handle is deliberately opaque: policies and evaluators must not be able
    to reach into simulator internals through it, because that would reintroduce
    exactly the coupling the backend interface exists to prevent.
    """

    episode_id: str
    scenario: ScenarioConfig
    seed: int
    step_index: int = 0
    sim_time_s: float = 0.0
    done: bool = False
    termination: TerminationReason = TerminationReason.RUNNING
    _opaque: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def dof(self) -> int:
        """Actuated degrees of freedom of the episode's robot."""
        return self.scenario.robot.dof


@dataclass(frozen=True)
class Transition:
    """The result of applying one :class:`Action`.

    Attributes:
        observation: What the policy may see next.
        robot_state: Proprioception after the step.
        ground_truth: Hidden state after the step. **Evaluator only.**
        reward: Optional scalar shaping reward; never used for success.
        terminated: Whether the episode ended.
        truncation: Whether the episode ended because a budget was exhausted.
        info: Backend diagnostics (collision flags, planning failures, ...).
    """

    observation: Observation
    robot_state: RobotState
    ground_truth: GroundTruthState
    reward: float = 0.0
    terminated: bool = False
    truncation: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def collision(self) -> bool:
        """Whether the backend reported a collision this step."""
        return bool(self.info.get("collision", False))

    @property
    def planning_failed(self) -> bool:
        """Whether the backend reported a planning failure this step."""
        return bool(self.info.get("planning_failed", False))


class SimulatorBackend(ABC):
    """Abstract base class for every simulator implementation.

    Subclasses must be usable through this interface alone. If an experiment
    needs backend-specific behaviour, that is a signal to add it to
    :class:`BackendCapabilities` rather than to branch on the backend name.
    """

    #: Short, stable identifier used in configuration and reports.
    name: str = "abstract"
    #: Human-readable one-line description.
    description: str = ""

    def __init__(self, **options: Any) -> None:
        """Store backend options.

        Args:
            **options: Backend-specific options. Unknown options are an error in
                concrete implementations rather than being silently ignored.
        """
        self._options: dict[str, Any] = dict(options)

    # -- introspection -----------------------------------------------------
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Report what this backend supports."""

    def is_operational(self) -> tuple[bool, str]:
        """Whether this backend can actually *run* here, and why not if it cannot.

        Separate from construction, and the distinction matters. A backend that
        needs an external runtime -- Isaac Sim's app, a ROS 2 graph, a GPU
        context -- can be constructible on a machine that cannot run it, because
        construction is cheap and loading the runtime is not. Reporting such a
        backend as available is worse than omitting it: ``robot-ci validate``
        would say Isaac works on a Mac, and the failure would arrive later, on
        the first episode, after the user has committed to a plan.

        The default is operational, because a backend with no external
        requirements always is. Backends with requirements override this.

        Returns:
            ``(operational, reason)``; ``reason`` is empty when operational.
        """
        return True, ""

    def info(self) -> BackendInfo:
        """Return identity and capability metadata for artifact records."""
        return BackendInfo(
            name=self.name,
            version=SIMULATOR_VERSION,
            implementation=type(self).__module__,
            capabilities=self.capabilities(),
        )

    # -- episode lifecycle -------------------------------------------------
    @abstractmethod
    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> EpisodeHandle:
        """Start a new episode.

        Args:
            scenario: The complete experimental condition.
            seed: Episode seed. Defaults to ``scenario.seed``.

        Returns:
            A handle used for all subsequent calls on this episode.

        Raises:
            SimulationError: If the scenario requests a capability this backend
                lacks, or if the episode cannot be constructed.
        """

    @abstractmethod
    def step(self, handle: EpisodeHandle, action: Action) -> Transition:
        """Apply one action and advance the simulation.

        Args:
            handle: The active episode handle.
            action: The command to apply.

        Returns:
            The resulting :class:`Transition`.

        Raises:
            SimulationError: If the handle is closed, or the action is
                incompatible with the episode's control mode.
        """

    @abstractmethod
    def observe(self, handle: EpisodeHandle) -> Observation:
        """Return the current observation without advancing time."""

    @abstractmethod
    def render(self, handle: EpisodeHandle) -> SensorFrame:
        """Render the current sensor frame (RGB, depth, segmentation).

        Returns:
            A :class:`SensorFrame`. Backends without rendering return a frame
            with ``valid=False`` and no arrays.
        """

    def ground_truth(self, handle: EpisodeHandle) -> GroundTruthState:
        """Return the hidden state; the policy layer must never call this.

        Provided as a separate method (rather than folded into
        :meth:`observe`) so that the ground-truth boundary is explicit and
        auditable.

        Raises:
            SimulationError: If called on a closed episode.
        """
        raise SimulationError(
            f"backend {self.name!r} does not expose ground truth",
            code="simulation.no_ground_truth",
        )

    def robot_config(self, handle: EpisodeHandle) -> RobotConfig:
        """Return the embodiment used by this episode."""
        return handle.scenario.robot

    def end_effector_pose(self, handle: EpisodeHandle) -> Pose:
        """Return the true end-effector pose (evaluator convenience)."""
        return self.ground_truth(handle).true_ee_pose

    def close(self, handle: EpisodeHandle) -> None:
        """Release resources for one episode.

        Optional: a stateless backend needs nothing to release. Implementations
        that own per-episode resources must override it and be idempotent.
        """

    def close_all(self) -> None:
        """Release every resource the backend holds.

        Optional, for the same reason as :meth:`close`. Implementations that
        cache models or renderers must override it to avoid leaking GL contexts.
        """

    # -- optional capabilities --------------------------------------------
    def jacobian_provider(self, handle: EpisodeHandle) -> Any:
        """Return a Jacobian source for the end effector, or ``None``.

        This is how a policy gets inverse kinematics without the framework
        depending on any particular simulator: a backend that has analytic
        kinematics returns a provider, and one that does not returns ``None`` so
        the policy can degrade explicitly rather than crash.

        Args:
            handle: The active episode.
        """
        return None

    def save_episode_artifacts(self, handle: EpisodeHandle, directory: Any) -> list[str]:
        """Optionally write debug artifacts (frames, video) for one episode.

        The default writes nothing and returns an empty list.

        Args:
            handle: The episode to dump.
            directory: Target directory; the backend creates it if needed.

        Returns:
            Paths of the files written. Empty when the backend writes nothing.
        """
        return []

    def __enter__(self) -> SimulatorBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close_all()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} name={self.name!r}>"


# --- registry ------------------------------------------------------------
#: Registry of backend name -> (module path, class name). Imports are lazy so
#: that the research engine never pulls in a heavy simulator it does not use.
_REGISTRY: dict[str, tuple[str, str]] = {
    "mock": ("robot_ci.simulation.mock", "MockBackend"),
    "synthetic": ("robot_ci.simulation.synthetic", "SyntheticBackend"),
    "mujoco": ("robot_ci.simulation.mujoco", "MuJoCoBackend"),
    "gazebo": ("robot_ci.simulation.gazebo", "GazeboBackend"),
    "isaac_sim": ("robot_ci.simulation.isaac_sim", "IsaacSimBackend"),
}


def register_backend(name: str, module_path: str, class_name: str) -> None:
    """Register an additional backend implementation.

    Args:
        name: Backend key used in configuration files.
        module_path: Dotted import path of the module.
        class_name: Class name within that module.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"backend {name!r} is already registered")
    _REGISTRY[name] = (module_path, class_name)


def list_backends() -> tuple[str, ...]:
    """Return every registered backend name."""
    return tuple(sorted(_REGISTRY))


def _import_backend(name: str) -> type[SimulatorBackend]:
    """Import and return the backend class registered under ``name``."""
    import importlib

    try:
        module_path, class_name = _REGISTRY[name]
    except KeyError as exc:
        raise BackendUnavailableError(
            f"unknown simulator backend {name!r}; available: {', '.join(list_backends())}",
            code="simulation.unknown_backend",
            requested=name,
            available=list(list_backends()),
        ) from exc

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"backend {name!r} could not be imported: {exc}. "
            f"Install its dependencies, or choose another backend "
            f"({', '.join(list_backends())}).",
            code="simulation.backend_import_failed",
            backend=name,
            module=module_path,
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise BackendUnavailableError(
            f"module {module_path!r} does not define {class_name!r}",
            code="simulation.backend_class_missing",
            backend=name,
        ) from exc


def create_backend(name: str, **options: Any) -> SimulatorBackend:
    """Instantiate a backend by name.

    Args:
        name: Registered backend name, e.g. ``"mujoco"``.
        **options: Passed to the backend constructor.

    Returns:
        A ready-to-use backend instance.

    Raises:
        BackendUnavailableError: If the backend is unknown, its dependencies
            are missing, or it cannot run on this machine. The message is written
            to be actionable (it names the missing package and the alternatives).
    """
    backend_class = _import_backend(name)
    try:
        return backend_class(**options)
    except BackendUnavailableError:
        raise
    except Exception as exc:
        raise BackendUnavailableError(
            f"backend {name!r} could not be constructed: {exc}",
            code="simulation.backend_construction_failed",
            backend=name,
        ) from exc


def probe_backend(name: str) -> dict[str, Any]:
    """Report whether a backend can be constructed here, without running it.

    Used by ``robot-ci validate`` to tell the user *before* an experiment starts
    which backends are usable on their machine.

    Returns:
        ``{"name", "available", "usable", "reason", "capabilities"}``.

        ``available`` means the backend could be constructed; ``usable`` means it
        can actually run an episode here. They differ for backends with external
        runtime requirements, and a caller that checks only ``available`` will
        report a backend as usable on a machine that cannot run it.
    """
    try:
        backend = create_backend(name)
    except BackendUnavailableError as exc:
        return {
            "name": name,
            "available": False,
            "usable": False,
            "reason": exc.message,
            "capabilities": None,
        }
    try:
        operational, reason = backend.is_operational()
        capabilities = backend.capabilities()
        return {
            "name": name,
            # Construction succeeded, so the backend is *available*; whether it
            # can run is reported separately and folds into ``usable``.
            "available": True,
            "usable": operational,
            "reason": reason,
            "capabilities": capabilities,
        }
    finally:
        backend.close_all()
