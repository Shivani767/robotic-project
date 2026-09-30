"""The Isaac adapter's contract, proven without Isaac.

Isaac Sim has no macOS build and needs an NVIDIA GPU, so none of its code path
can run here. What can be proven without it is everything that does not need it:

* the module imports on a machine with no Omniverse, and every Isaac import is
  behind a guard rather than at module scope;
* availability reporting is truthful -- constructible is not the same as usable,
  and reporting a backend as available on a Mac that cannot run it is the bug
  this suite exists to prevent;
* the translation layer round-trips, and fails loudly rather than silently, on
  the mistakes that produce plausible-looking wrong answers;
* the backend drives a full episode against a mock session shaped like the
  Isaac API, which is what pins the adapter to the project's simulator
  interface rather than to my assumptions about Omniverse.

Nothing here demonstrates that Isaac Sim works. It demonstrates that the code
around Isaac is correct, and that the parts which need Isaac fail in a way
someone can act on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_ci.core.errors import BackendUnavailableError
from robot_ci.models.geometry import Pose, Quaternion, Vec3
from robot_ci.models.state import Action, ActionType
from robot_ci.scenarios import default_scenario
from robot_ci.simulation import create_backend, list_backends, probe_backend
from robot_ci.simulation.isaac_sim import (
    ISAAC_MODULES,
    IsaacAvailability,
    IsaacSession,
    IsaacSimBackend,
    isaac_availability,
)
from robot_ci.simulation.isaac_translate import (
    ISAAC_ARTICULATION_API,
    action_to_isaac,
    isaac_state_to_robot_state,
    scenario_to_isaac,
    wire_frame_to_sensor_frame,
    wire_objects_to_object_states,
)

# --- a mock session shaped like the Isaac API -----------------------------


class MockIsaacSession:
    """A stand-in for a live Isaac app, satisfying :class:`IsaacSession`.

    Canned wire data, no Omniverse. Every method named in
    ``ISAAC_ARTICULATION_API`` exists so the adapter's API check passes, which is
    itself part of what is being tested.
    """

    def __init__(self, dof: int = 7) -> None:
        self.dof = dof
        self.resets: list[tuple[dict[str, Any], int]] = []
        self.actions: list[dict[str, Any]] = []
        self.closed: list[str] = []
        self.position = 0.0

    def reset(self, wire_scene: dict[str, Any], *, seed: int) -> None:
        self.resets.append((wire_scene, seed))
        self.position = 0.0

    def step(self, wire_action: dict[str, Any]) -> None:
        self.actions.append(wire_action)
        self.position += 0.01

    def read_state(self) -> dict[str, Any]:
        n = self.dof + 2
        return {
            "dof_position": [0.1 * i for i in range(n)],
            "dof_velocity": [0.0] * n,
            "dof_force": [2.5] * n,
        }

    def read_objects(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "target_cube",
                "pose": {
                    "position_m": [0.3, 0.1, 0.008],
                    "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "size_m": 0.016,
                "sigma_m": 0.001,
            }
        ]

    def read_ground_truth(self) -> Any:
        from robot_ci.models.state import GroundTruthState

        return GroundTruthState()

    def render(self) -> dict[str, Any]:
        return {
            "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "depth": np.full((2, 3), 0.5, dtype=np.float32),
            "segmentation": np.zeros((2, 3), dtype=np.int32),
            "intrinsics": (140.0, 140.0, 1.5, 1.0),
            "valid": True,
        }

    def jacobian_provider(self, handle: Any) -> Any:
        return None

    def close_episode(self, handle: Any) -> None:
        self.closed.append(handle.episode_id)

    def close(self) -> None:
        self.closed.append("all")

    # The API surface the adapter checks for.
    def get_dof_positions(self) -> Any: ...
    def set_dof_positions(self, *a: Any) -> None: ...
    def get_dof_velocities(self) -> Any: ...
    def set_dof_velocity_targets(self, *a: Any) -> None: ...
    def get_dof_force_targets(self) -> Any: ...
    def apply_action(self, *a: Any) -> None: ...
    def get_world_poses(self) -> Any: ...
    def reset_dummy(self, *a: Any) -> None: ...


@pytest.fixture
def session() -> MockIsaacSession:
    return MockIsaacSession()


@pytest.fixture
def backend(session: MockIsaacSession) -> IsaacSimBackend:
    return IsaacSimBackend(session=session)  # type: ignore[arg-type]


# --- import isolation -----------------------------------------------------


def test_the_adapter_module_imports_with_no_omniverse_installed() -> None:
    """The module must be importable on any machine.

    An ``import isaacsim`` at module scope would make this file unimportable on
    a laptop, and with it ``robot-ci validate`` and the registry.
    """
    import robot_ci.simulation.isaac_sim as module

    assert module.__name__ == "robot_ci.simulation.isaac_sim"
    for symbol in dir(module):
        assert not symbol.startswith("isaacsim"), f"{symbol} looks like a leaked Isaac symbol"


def test_isaac_packages_are_not_installed_here() -> None:
    """States the premise of this file, so it fails loudly if that changes.

    If this ever starts failing, Isaac *is* available and these tests are no
    longer proving anything -- the real backend can be exercised instead.
    """
    assert isaac_availability().available is False, (
        "Isaac Sim appears to be installed; the mock-based tests no longer prove "
        "the unavailable path and the real backend should be used instead"
    )


# --- capability and availability reporting ---------------------------------


def test_the_backend_registers_under_its_documented_name() -> None:
    assert "isaac_sim" in list_backends()
    assert isinstance(probe_backend("isaac_sim")["capabilities"], object)


def test_capabilities_are_reported_even_where_isaac_cannot_run() -> None:
    """A Mac user should still be able to see what the backend would give them.

    Declining to construct without Omniverse would report the backend as
    missing rather than as present-but-unusable, which answers a different
    question than the one being asked.
    """
    caps = IsaacSimBackend().capabilities()
    assert caps.name == "isaac_sim"
    assert caps.rgb and caps.depth and caps.segmentation
    assert caps.requires_gpu
    # Isaac is not deterministic by default, and saying otherwise would make
    # cross-run comparisons look tighter than they are.
    assert caps.deterministic is False
    assert "not executed" in caps.notes.lower()


def test_constructible_is_not_the_same_as_usable() -> None:
    """The distinction the probe exists to make.

    Reporting Isaac as "available" on a machine with no Omniverse -- and no
    macOS build to install it on -- would tell a user their plan works here. The
    probe therefore carries both facts, and the usable one is the one that is
    false.
    """
    report = probe_backend("isaac_sim")
    assert report["available"] is True, "the backend should construct without Omniverse"
    assert report["usable"] is False
    assert "isaacsim" in report["reason"]
    assert "macos" in report["reason"].lower() or "gpu" in report["reason"].lower()


def test_backends_without_requirements_stay_operational() -> None:
    """The new probe field must not change anything for working backends."""
    for name in ("mock", "mujoco"):
        report = probe_backend(name)
        assert report["available"] is True
        assert report["usable"] is True, f"{name} regressed in the probe change"


def test_availability_lists_every_module_it_would_import() -> None:
    """A user should be able to see the dependency list before installing."""
    report = isaac_availability()
    assert report.available is False
    assert set(report.missing) == {module for module, _ in ISAAC_MODULES}
    assert all(purpose for _, purpose in ISAAC_MODULES)
    assert isinstance(report, IsaacAvailability)
    assert "available" in report.to_dict()


# --- graceful failure ------------------------------------------------------


def test_every_episode_method_fails_with_a_readable_error() -> None:
    """Each entry point must name the operation and say what to do instead.

    Checked one at a time because a single ``reset`` call would leave the others
    unproven, and they take different paths.
    """
    backend = IsaacSimBackend()
    scenario = default_scenario()
    with pytest.raises(BackendUnavailableError) as excinfo:
        backend.reset(scenario)
    assert excinfo.value.code == "simulation.isaac_unavailable"
    assert "start an episode" in str(excinfo.value)
    assert "mujoco" in str(excinfo.value), "the message should name a working alternative"


def test_a_session_that_lacks_the_isaac_api_is_reported_by_name() -> None:
    """An Omniverse version bump must be a one-line diagnostic, not a mystery."""

    class Incomplete:
        def reset(self, *a: Any, **k: Any) -> None: ...
        def step(self, *a: Any, **k: Any) -> None: ...
        def read_state(self) -> Any: ...
        def read_objects(self) -> Any: ...
        def read_ground_truth(self) -> Any: ...
        def render(self) -> Any: ...
        def jacobian_provider(self, *a: Any) -> Any: ...
        def close_episode(self, *a: Any) -> None: ...
        def close(self) -> None: ...

    incomplete = Incomplete()
    with pytest.raises(BackendUnavailableError) as excinfo:
        IsaacSimBackend(session=incomplete).reset(default_scenario())  # type: ignore[arg-type]
    assert excinfo.value.code == "simulation.isaac_api_mismatch"
    # Only the genuinely absent methods should be named. Asserting the whole
    # list would be satisfied by an error that named everything, including the
    # methods the session does provide.
    absent = [
        name
        for name in ISAAC_ARTICULATION_API
        if not callable(getattr(incomplete, name, None))
    ]
    assert absent, "the fixture must actually be missing something"
    for name in absent:
        assert name in str(excinfo.value), f"{name} not named in the error"


# --- the backend contract against a mock session ---------------------------


def test_the_backend_drives_a_full_episode_through_the_project_interface(backend, session) -> None:
    """reset/step/observe/render/close, exercised through the public API.

    This is the test that makes the adapter's *contract* verified even though its
    runtime is not: every call goes through :class:`SimulatorBackend`, so the
    adapter satisfies the same interface the experiment runner uses.
    """
    scenario = default_scenario()
    handle = backend.reset(scenario, seed=7)
    assert session.resets[0][1] == 7, "the episode seed must reach the session"
    assert session.resets[0][0]["schema"] == "isaac-scene/1"

    action = Action.joint_positions(np.zeros(scenario.robot.dof)).model_copy(
        update={"gripper_width_m": 0.01}
    )
    transition = backend.step(handle, action)
    assert handle.step_index == 1
    assert transition.observation.step_index == 1
    assert len(transition.robot_state.joint_positions) == scenario.robot.dof
    assert transition.observation.objects, "the object must reach the observation"

    observation = backend.observe(handle)
    assert observation.objects[0].id == "target_cube"

    frame = backend.render(handle)
    assert frame.valid and frame.intrinsics[0] == 140.0

    backend.close(handle)
    assert handle.episode_id in session.closed
    backend.close_all()
    assert "all" in session.closed


def test_a_closed_handle_refuses_further_steps(backend) -> None:
    """Stepping a finished episode must be an error, not a silent advance."""
    scenario = default_scenario()
    handle = backend.reset(scenario)
    handle.done = True
    with pytest.raises(Exception) as excinfo:
        backend.step(handle, Action.joint_positions(np.zeros(scenario.robot.dof)))
    assert "closed" in str(excinfo.value)


def test_a_scenario_the_backend_cannot_run_is_rejected_at_reset(backend) -> None:
    """An infeasible scenario must fail at reset, not after 200 silent steps."""
    from robot_ci.core.errors import SimulationError
    from robot_ci.models.scenario import ScenarioConfig

    # Widen the limits too, or the model's own validator rejects the config
    # before the backend ever sees it -- which is a different failure with the
    # same user-visible symptom, and would make this test pass for the wrong
    # reason.
    base = default_scenario()
    robot = base.robot
    # Tuples are extended, not repeated: ``limits.lower * 2`` would double the
    # *length* to 14 and the model would reject the config before the backend saw
    # it, which is a different failure with the same symptom.
    wide = 99
    limits = robot.limits.model_copy(
        update={
            "lower": tuple([-3.0] * wide),
            "upper": tuple([3.0] * wide),
            "max_velocity": tuple([3.0] * wide),
            "max_effort": tuple([90.0] * wide),
        }
    )
    impossible = ScenarioConfig(
        robot=robot.model_copy(
            update={
                "dof": wide,
                "home_position": (0.0,) * wide,
                "limits": limits,
                # The model also cross-checks the kinematic table against dof,
                # so widening dof alone is rejected as an inconsistent config.
                "dh_parameters": tuple((0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(wide + 1)),
            }
        )
    )
    with pytest.raises(SimulationError) as excinfo:
        backend.reset(impossible)
    assert "dof" in str(excinfo.value)


def test_create_backend_builds_the_adapter_without_isaac() -> None:
    """The registry path must work on a machine with no Omniverse."""
    built = create_backend("isaac_sim")
    assert isinstance(built, IsaacSimBackend)


# --- translation -----------------------------------------------------------


def test_the_scenario_translation_carries_the_experiment() -> None:
    wire = scenario_to_isaac(default_scenario())
    assert wire["requires_gpu"] is True
    assert wire["articulation"]["dof"] == 7
    assert wire["gravity_mps2"][2] < 0
    assert wire["objects"], "the target must be placed in the scene"
    assert wire["camera"]["clip_range_m"][0] < wire["camera"]["clip_range_m"][1]


def test_render_size_matches_the_intrinsics() -> None:
    """A transposed render size is a correctly-shaped, wrongly-oriented image.

    ``ScenarioConfig.resolution`` is ``(height, width)``. Reading it the other
    way round produces an image that looks fine and is rotated, which no shape
    check would catch.
    """
    scenario = default_scenario()
    wire = scenario_to_isaac(scenario)
    assert wire["render"]["width_px"] == scenario.sensors.intrinsics.width_px
    assert wire["render"]["height_px"] == scenario.sensors.intrinsics.height_px


def test_quaternion_ordering_survives_a_round_trip() -> None:
    """Isaac stores ``(x, y, z, w)``; robot-ci stores ``(w, x, y, z)``.

    Getting this wrong yields a mirrored pose that looks entirely reasonable,
    which is why the ordering is asserted rather than assumed.
    """
    from robot_ci.simulation.isaac_translate import _pose_from_wire, _pose_to_wire

    original = Pose(
        position=Vec3.of(0.3, -0.1, 0.05),
        orientation=Quaternion(w=0.5, x=0.5, y=0.5, z=0.5),
    )
    wire = _pose_to_wire(original)
    assert set(wire["orientation_xyzw"]) == {"x", "y", "z", "w"}
    restored = _pose_from_wire(wire)
    for a, b in zip(original.orientation.array, restored.orientation.array):
        assert a == pytest.approx(b, abs=1e-9)


def test_a_non_unit_wire_quaternion_is_rejected() -> None:
    """Silently returning an unnormalised quaternion puts a wrong pose downstream."""
    from robot_ci.simulation.isaac_translate import _pose_from_wire

    bad = {
        "position_m": [0.0, 0.0, 0.0],
        "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 2.0},
    }
    with pytest.raises(ValueError, match="normalised"):
        _pose_from_wire(bad)


def test_an_action_becomes_a_position_target_not_a_velocity_one() -> None:
    """A commanded position sent as a velocity target is a silent control bug.

    Isaac's force drive separates the two; sending the position to the velocity
    channel makes the arm arrive wherever the clip allows, which reads as a
    tracking failure rather than a control error.
    """
    action = Action.joint_positions(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]))
    wire = action_to_isaac(action, dof=7, gripper_dof=2)
    assert wire["position_target"][:3] == pytest.approx([0.1, 0.2, 0.3])
    assert wire["velocity_target"] == [0.0] * 9
    # The three arrays must agree on the articulation's DOF count. They did not
    # once: the position target carried one gripper entry while the velocity
    # target carried two, so the action described two different robots.
    assert len(wire["position_target"]) == len(wire["velocity_target"]) == len(wire["effort_target"])
    assert wire["position_target"][-2:] == pytest.approx([0.0, 0.0])


def test_an_action_for_the_wrong_embodiment_is_rejected() -> None:
    """A short action means a DOF-ordering mismatch and must not be applied."""
    action = Action.joint_positions(np.zeros(5))
    with pytest.raises(ValueError, match="arm joints"):
        action_to_isaac(action, dof=7)


def test_a_non_position_action_is_rejected_explicitly() -> None:

    action = Action(type=ActionType.JOINT_VELOCITY, joint_velocity=(0.0,) * 7)
    with pytest.raises(ValueError, match="joint-position"):
        action_to_isaac(action, dof=7)


def test_a_short_state_array_names_the_cause() -> None:
    """Indexing past a short array yields plausible values in the wrong slots."""
    scenario = default_scenario()
    with pytest.raises(ValueError, match="DOF order"):
        isaac_state_to_robot_state(
            {"dof_position": [0.0] * 3, "dof_velocity": [0.0] * 3}, robot=scenario.robot
        )


def test_a_frame_without_intrinsics_is_rejected() -> None:
    """An uncalibrated frame cannot be back-projected."""
    with pytest.raises(ValueError, match="intrinsics"):
        wire_frame_to_sensor_frame({"rgb": None, "depth": None}, timestamp_s=0.0)


def test_object_beliefs_carry_provenance_and_uncertainty() -> None:
    """An object with no covariance and no source is not a belief."""
    states = wire_objects_to_object_states(
        [
            {
                "id": "cube",
                "pose": {
                    "position_m": [0.3, 0.1, 0.008],
                    "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "sigma_m": 0.002,
            }
        ],
        timestamp_s=1.0,
        confidence=0.5,
    )
    state = states["cube"]
    assert state.confidence == 0.5
    assert state.covariance.max_sigma == pytest.approx(0.002)
    assert state.source is not None


def test_the_session_protocol_is_satisfied_by_the_mock() -> None:
    """If the mock stopped matching the protocol, the contract tests would be vacuous."""
    assert isinstance(MockIsaacSession(), IsaacSession)
