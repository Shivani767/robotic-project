"""Isaac Sim backend, behind the project's simulator interface.

**Status: implemented but not executed.** Isaac Sim is not installed on the
development machine, has no macOS build at all, and requires an NVIDIA GPU. No
line of the Isaac-facing code below has been run against a real installation.
What *is* verified is the contract: the translation layer is exercised against
mocks shaped like the Isaac API, and the unavailable paths are tested to raise
the documented error rather than something incidental. See
``docs/isaac.md`` for the evidence labels.

Design constraints, in the order they mattered:

* **Isaac is never imported at module scope.** Not for type annotations, not
  for constants, not behind a ``TYPE_CHECKING`` that a runtime path can reach.
  Every import goes through :func:`_require_isaac`, so this module imports
  cleanly on a laptop with no Omniverse and the failure is a readable message
  rather than an ``ImportError`` from three frames down.
* **The Isaac API surface is small and named.** :data:`_REQUIRED_API` lists the
  prim methods this adapter drives, and :func:`_check_api` reports a missing one
  by name. An Omniverse version bump that moves a method is a one-line
  diagnostic, not a mystery.
* **The wire format lives in a separate module.** All conversion is
  :mod:`robot_ci.simulation.isaac_translate`, which has no Isaac imports and is
  therefore testable anywhere. This module only moves data across the boundary.
* **Capabilities are reported from what the adapter will do, and separately from
  whether Isaac is present.** A backend that cannot be constructed should still
  be able to say what it would provide, which is what lets a Mac user see the
  Isaac backend's feature set and a CI job check that a scenario is feasible
  before spending GPU hours discovering it was not.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from robot_ci.core.errors import BackendUnavailableError, SimulationError
from robot_ci.models.environment import TaskName
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import Action, GroundTruthState, Observation, SensorFrame
from robot_ci.simulation.base import (
    BackendCapabilities,
    EpisodeHandle,
    SimulatorBackend,
    Transition,
)
from robot_ci.simulation.isaac_translate import (
    ISAAC_ARTICULATION_API,
    action_to_isaac,
    isaac_state_to_robot_state,
    scenario_to_isaac,
    wire_frame_to_sensor_frame,
    wire_objects_to_object_states,
)

__all__ = [
    "ISAAC_MODULES",
    "IsaacAvailability",
    "IsaacSession",
    "IsaacSimBackend",
    "isaac_availability",
]

#: Modules that must import for this adapter to be usable.
#:
#: Listed as data so a user can see, before installing anything, what the
#: adapter expects to find. ``isaacsim`` is the umbrella; the others are the
#: specific prim and annotator APIs this adapter drives.
ISAAC_MODULES: tuple[tuple[str, str], ...] = (
    ("isaacsim", "the Isaac Sim Python app"),
    ("isaacsim.core.api", "the prim API"),
    ("isaacsim.core.prims", "articulation and camera prims"),
    ("isaacsim.core.utils.stage", "USD stage access"),
)

#: Prim methods the adapter drives, named so a missing one is a clear message.
_REQUIRED_API: tuple[str, ...] = ISAAC_ARTICULATION_API

#: Gripper degrees of freedom assumed when the embodiment does not say.
_GRIPPER_DOF = 2


@runtime_checkable
class IsaacSession(Protocol):
    """What the adapter needs from a live Isaac Sim application.

    A ``Protocol`` rather than a base class, and that is what makes the adapter
    testable without Isaac: the tests supply an object shaped like this, and the
    backend cannot tell the difference. A real implementation wraps an Omniverse
    ``SimulationApp``; a test implementation returns canned wire dicts.

    The methods here take and return the wire format of
    :mod:`robot_ci.simulation.isaac_translate`, not Isaac types. That is
    deliberate: it keeps every Omniverse import in one implementation of this
    protocol rather than spread through the adapter.

    Runtime-checkable so :func:`_check_api` can report a session missing a
    method by name.
    """

    def reset(self, wire_scene: dict[str, Any], *, seed: int) -> None:
        """Build ``wire_scene`` and start the simulation at ``seed``."""

    def step(self, wire_action: dict[str, Any]) -> None:
        """Advance the simulation by one control interval."""

    def read_state(self) -> dict[str, Any]:
        """Return articulation DOF positions, velocities and forces."""

    def read_objects(self) -> list[dict[str, Any]]:
        """Return per-object wire poses."""

    def read_ground_truth(self) -> GroundTruthState:
        """Return the hidden state, for the evaluator only."""

    def render(self) -> dict[str, Any]:
        """Return a wire frame: rgb, depth, segmentation, intrinsics."""

    def jacobian_provider(self, handle: EpisodeHandle) -> Any:
        """Return a Jacobian source for the end effector, or ``None``."""

    def close_episode(self, handle: EpisodeHandle) -> None:
        """Release one episode's resources. Must be idempotent."""

    def close(self) -> None:
        """Release every resource the session holds."""


@dataclass(frozen=True)
class IsaacAvailability:
    """Whether Isaac Sim can be used on this machine, and why not if it cannot.

    Attributes:
        available: Whether every required module imported.
        missing: Modules that did not import.
        detail: Per-module outcome, for a diagnostic.
        reason: One-line summary for a report.
    """

    available: bool
    missing: tuple[str, ...] = ()
    detail: dict[str, str] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, for ``robot-ci validate`` output."""
        return {
            "available": self.available,
            "missing": list(self.missing),
            "detail": dict(self.detail or {}),
            "reason": self.reason,
        }


def isaac_availability() -> IsaacAvailability:
    """Report whether Isaac Sim is importable here, without raising.

    Probing rather than importing is what makes this usable from
    ``robot-ci validate``: a developer on a Mac should learn that the Isaac
    backend is unavailable *and why*, not get a traceback from a command whose
    job was to tell them the state of their environment.

    Returns:
        The availability report.
    """
    missing: list[str] = []
    detail: dict[str, str] = {}
    for module, purpose in ISAAC_MODULES:
        try:
            importlib.import_module(module)
        except ImportError as error:
            missing.append(module)
            detail[module] = f"{purpose}: {error}"
        else:
            detail[module] = f"{purpose}: available"
    if missing:
        return IsaacAvailability(
            available=False,
            missing=tuple(missing),
            detail=detail,
            reason=(
                "Isaac Sim is not available: "
                + ", ".join(missing)
                + ". It requires an NVIDIA GPU and has no macOS build; use the "
                "'mujoco' backend on this machine, or run this experiment on a "
                "Linux GPU host."
            ),
        )
    return IsaacAvailability(available=True, detail=detail, reason="Isaac Sim is importable")


def _require_isaac(operation: str) -> None:
    """Raise a readable error unless Isaac Sim is importable.

    Args:
        operation: What the caller was trying to do, named in the message.

    Raises:
        BackendUnavailableError: Always, when Isaac is not importable. The
            message names the missing modules, the operation, and what to do
            instead.
    """
    report = isaac_availability()
    if report.available:
        return
    raise BackendUnavailableError(
        f"Isaac Sim unavailable; cannot {operation}. {report.reason}",
        code="simulation.isaac_unavailable",
        operation=operation,
        missing=list(report.missing),
    )


def _check_api(prim: Any) -> None:
    """Verify a prim exposes every method the adapter drives.

    Args:
        prim: The articulation or stage prim Isaac handed back.

    Raises:
        BackendUnavailableError: If a required method is absent. The message
            names it, because an Omniverse version bump that moves a method is
            otherwise indistinguishable from a wiring mistake on our side.
    """
    absent = [name for name in _REQUIRED_API if not callable(getattr(prim, name, None))]
    if absent:
        raise BackendUnavailableError(
            f"the Isaac prim does not provide {', '.join(absent)}; this adapter "
            "targets the articulation API listed in isaac_translate."
            "ISAAC_ARTICULATION_API and may need updating for this Isaac version",
            code="simulation.isaac_api_mismatch",
            missing=absent,
        )


class IsaacSimBackend(SimulatorBackend):
    """Runs episodes in Isaac Sim.

    The backend is constructible **without Isaac installed** so that capability
    reporting and availability probing work on any machine. Every method that
    would need a live Isaac app raises :class:`BackendUnavailableError` with an
    actionable message instead, which is what lets ``robot-ci validate`` list
    the Isaac backend on a Mac as unavailable-with-a-reason rather than crashing.

    Passing ``session=`` injects a pre-built session, which is how the adapter
    contract is tested without Isaac. The session must implement
    :class:`IsaacSession`; nothing about it is Isaac-specific at the type level.
    """

    name = "isaac_sim"
    description = "NVIDIA Isaac Sim, for photorealistic RGB/depth rendering on a GPU host."

    def __init__(self, *, session: IsaacSession | None = None, **options: Any) -> None:
        """Construct the backend.

        Args:
            session: A pre-built session, for tests and for embedding Isaac in
                a host application that already owns the app lifecycle. When
                ``None`` the backend is inert and every episode method raises.
            **options: Stored by the base class; unused by this backend, which
                takes all its configuration from the scenario.
        """
        super().__init__(**options)
        self._session = session

    # -- introspection ----------------------------------------------------
    def capabilities(self) -> BackendCapabilities:
        """Report what this backend provides, independent of Isaac presence.

        Reported even when Isaac is absent, because the question a user has on
        a Mac is "what would this backend give me", and answering it costs
        nothing. ``available`` is not a field on this record by design: whether
        the backend can be *constructed* is
        :meth:`robot_ci.simulation.probe_backend`'s job, and conflating the two
        would mean a capability table that changes with the machine it is read
        on.
        """
        return BackendCapabilities(
            name=self.name,
            physics=True,
            collision=True,
            rgb=True,
            depth=True,
            segmentation=True,
            force_torque=True,
            joint_efforts=True,
            grasp_latch=False,
            deterministic=False,
            max_dof=32,
            supported_tasks=(
                TaskName.GRASP_OBJECT,
                TaskName.PLACE_OBJECT,
                TaskName.PICK_AND_PLACE,
            ),
            requires_gpu=True,
            notes=(
                "Isaac Sim runs contact dynamics and photorealistic rendering. "
                "Not deterministic by default, so a seed fixes the scene but not "
                "the trajectory; results across runs carry that variance. "
                "Implemented but not executed: see docs/isaac.md."
            ),
        )

    def is_operational(self) -> tuple[bool, str]:
        """Whether Isaac Sim can actually be used on this machine.

        Construction succeeds anywhere -- deliberately, so capabilities can be
        reported on a Mac -- but an episode cannot run without Isaac. Overriding
        this is what makes ``robot-ci validate`` tell the truth: without it the
        Isaac backend reports as available on a machine that has no Omniverse,
        and has no macOS build to install.
        """
        report = isaac_availability()
        if report.available and self._session is None:
            return (
                False,
                "Isaac Sim is importable but no session was supplied; construct "
                "IsaacSimBackend(session=...) with a live IsaacSession",
            )
        return (report.available, report.reason)

    # -- episode lifecycle ------------------------------------------------
    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> EpisodeHandle:
        """Start an Isaac episode.

        Args:
            scenario: The condition to build.
            seed: Episode seed; defaults to the scenario's.

        Returns:
            The episode handle.

        Raises:
            BackendUnavailableError: If no Isaac session is available.
            SimulationError: If the scenario needs something this backend
                cannot provide.
        """
        session = self._require_session("start an episode")
        fatal = self.capabilities().incompatible_for(scenario)
        if fatal:
            raise SimulationError(
                f"scenario requires capabilities the Isaac backend lacks: {', '.join(fatal)}",
                code="simulation.isaac_incompatible",
                scenario=scenario.scenario_id,
                missing=list(fatal),
            )
        session.reset(scenario_to_isaac(scenario), seed=scenario.seed if seed is None else seed)
        return EpisodeHandle(
            episode_id=f"isaac_{scenario.scenario_id}_{seed if seed is not None else scenario.seed}",
            scenario=scenario,
            seed=scenario.seed if seed is None else seed,
        )

    def step(self, handle: EpisodeHandle, action: Action) -> Transition:
        """Apply one action in Isaac and return the resulting transition.

        Args:
            handle: Active episode.
            action: Joint-position command.

        Returns:
            The transition.

        Raises:
            SimulationError: If the handle is closed, or the action does not
                match the embodiment.
        """
        session = self._require_session("step an episode")
        if handle.done:
            raise SimulationError(
                f"episode {handle.episode_id} is closed",
                code="simulation.handle_closed",
            )
        # RobotConfig declares no gripper DOF count; 2 is this embodiment
        # family's convention and is overridable through the session.
        wire = action_to_isaac(action, dof=handle.dof, gripper_dof=_GRIPPER_DOF)
        session.step(wire)
        handle.step_index += 1
        handle.sim_time_s = handle.step_index * (1.0 / max(handle.scenario.robot.control_hz, 1))
        state = isaac_state_to_robot_state(session.read_state(), robot=handle.scenario.robot)
        objects = wire_objects_to_object_states(
            session.read_objects(), timestamp_s=handle.sim_time_s
        )
        observation = Observation(
            robot=state,
            objects=tuple(objects.values()),
            step_index=handle.step_index,
            timestamp_s=handle.sim_time_s,
            scenario_id=handle.scenario.scenario_id,
            seed=handle.seed,
        )
        return Transition(
            observation=observation,
            robot_state=state,
            ground_truth=session.read_ground_truth(),
        )

    def observe(self, handle: EpisodeHandle) -> Observation:
        """Return Isaac's current observation without advancing time."""
        session = self._require_session("observe an episode")
        state = isaac_state_to_robot_state(session.read_state(), robot=handle.scenario.robot)
        objects = wire_objects_to_object_states(
            session.read_objects(), timestamp_s=handle.sim_time_s
        )
        return Observation(
            robot=state,
            objects=tuple(objects.values()),
            step_index=handle.step_index,
            timestamp_s=handle.sim_time_s,
            scenario_id=handle.scenario.scenario_id,
            seed=handle.seed,
        )

    def render(self, handle: EpisodeHandle) -> SensorFrame:
        """Return Isaac's rendered RGB/depth/segmentation frame."""
        session = self._require_session("render a frame")
        return wire_frame_to_sensor_frame(session.render(), timestamp_s=handle.sim_time_s)

    def ground_truth(self, handle: EpisodeHandle) -> GroundTruthState:
        """Return Isaac's hidden state. The policy layer must never call this."""
        session = self._require_session("read ground truth")
        return session.read_ground_truth()

    def jacobian_provider(self, handle: EpisodeHandle) -> Any:
        """Return a Jacobian source, if the session provides one.

        Isaac has analytic articulation Jacobians, but only through a prim API
        this adapter does not depend on, so the default is ``None`` and a policy
        that needs IK degrades explicitly rather than silently getting zeros.
        """
        session = self._session
        if session is None:
            return None
        return session.jacobian_provider(handle)

    # -- lifecycle --------------------------------------------------------
    def close(self, handle: EpisodeHandle) -> None:
        """Release one episode's Isaac resources. Idempotent."""
        if self._session is not None:
            self._session.close_episode(handle)

    def close_all(self) -> None:
        """Release every Isaac resource this backend holds."""
        if self._session is not None:
            self._session.close()

    def _require_session(self, operation: str) -> IsaacSession:
        """Return the injected session, or raise the documented error.

        Args:
            operation: What the caller was doing, named in the message.

        Returns:
            The session.

        Raises:
            BackendUnavailableError: If no session is available. The message
                distinguishes "Isaac is not installed here" from "a session was
                not injected", because they need different fixes and a single
                combined message sends the reader to the wrong one.
        """
        if self._session is not None:
            _check_api(self._session)
            return self._session
        report = isaac_availability()
        if report.available:
            raise BackendUnavailableError(
                f"Isaac Sim is importable but no session was supplied to drive it; "
                f"cannot {operation}. Construct IsaacSimBackend(session=...) with a "
                "live IsaacSession.",
                code="simulation.isaac_no_session",
                operation=operation,
            )
        raise BackendUnavailableError(
            f"Isaac Sim unavailable; cannot {operation}. {report.reason}",
            code="simulation.isaac_unavailable",
            operation=operation,
            missing=list(report.missing),
        )
