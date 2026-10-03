"""The backend contract, told apart from a test that just exercises code.

Part 1 asks whether ``GazeboBackend`` really implements ``SimulatorBackend``.
That is a question about the *boundary*, not about the maths inside it, so
these tests check the boundary: readiness and startup failure, joint-state
parsing and command mapping, reset/episode isolation, timeouts and shutdown,
observation freshness, the success criteria, and -- the one that matters most
for the project's honesty claim -- that no path can fall back to mock data.

They run on a machine with no Gazebo by putting a deterministic stand-in
*underneath* the backend: the process, the topics and the clock are faked, but
the backend's own code paths (``reset``, ``step``, ``ground_truth``, ``close``)
are the real ones. Anything that would need real physics is deliberately not
claimed here; ``tests/integration/test_gazebo_runtime.py`` is what claims that,
and only inside the container.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import numpy as np
import pytest

import robot_ci.simulation.gazebo as gazebo_module
from robot_ci.core.errors import BackendUnavailableError, SimulationError
from robot_ci.gazebo.live import JOINT_NAMES, BridgeHealth, JointStateReader
from robot_ci.models.state import (
    Action,
    GripperState,
    GroundTruthObject,
    GroundTruthState,
    Observation,
    RobotState,
)
from robot_ci.scenarios import default_scenario
from robot_ci.simulation import SimulatorBackend, create_backend, probe_backend
from robot_ci.simulation.gazebo import GazeboBackend
from robot_ci.simulation.mock import MockBackend
from robot_ci.tasks import TaskStatus

PERIOD_S = 0.05  # 1/20 Hz, the default scenario's control rate


class _FakePublisher:
    """Records every ``Double`` a topic received, in order."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.data: list[float] = []

    def publish(self, message: Any) -> None:
        self.data.append(float(message.data))


class _FakeJoints:
    """Stands in for Gazebo's joint-state topic.

    ``latest()`` advances the simulation clock by one control period, which is
    what makes the backend's synchronous ``step`` over an asynchronous simulator
    take the same code path it takes live: ``step`` blocks until the clock has
    moved by one period, and would raise ``simulation.gazebo_clock_stalled`` if
    it never did.
    """

    def __init__(self, *, dt: float = PERIOD_S, frozen: bool = False) -> None:
        self.positions: dict[str, float] = dict.fromkeys(JOINT_NAMES, 0.0)
        self.sim_time = 0.0
        self.dt = 0.0 if frozen else dt
        self.messages = 0
        self.stopped = False

    def latest(self) -> tuple[dict[str, float], float]:
        self.sim_time += self.dt
        self.messages += 1
        return dict(self.positions), self.sim_time

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.stopped = True


class _FakePoses:
    """Stands in for ``/world/<name>/pose/info`` (position *and* orientation)."""

    def __init__(self, _world: str) -> None:
        self.data: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}

    def pose(self, name: str) -> Any:
        return self.data.get(name)

    def position(self, name: str) -> tuple[float, float, float] | None:
        entry = self.data.get(name)
        return None if entry is None else entry[0]


class _FakeDepth:
    def __init__(self, _world: str) -> None:
        pass

    def latest(self) -> tuple[np.ndarray | None, None, int, int]:
        return None, None, 0, 0


class _FakeProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        return 0


@pytest.fixture
def fake_gazebo(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Put deterministic stand-ins under the backend's real code paths."""
    created: dict[str, Any] = {"processes": [], "publishers": {}}

    monkeypatch.setattr(gazebo_module, "_require_gz", lambda: None)
    monkeypatch.setattr(gazebo_module, "world_sdf", lambda *_a, **_k: "<sdf/>")

    def _popen(argv: Any, **kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(str(argv[0]))
        created["processes"].append(process)
        return process

    monkeypatch.setattr(gazebo_module.subprocess, "Popen", _popen)
    monkeypatch.setattr(GazeboBackend, "_wait_for_joint_state", lambda *_a, **_k: True)
    monkeypatch.setattr(GazeboBackend, "_drive_to", lambda *_a, **_k: True)

    def _make_publishers(self: GazeboBackend) -> dict[str, Any]:
        publishers = {
            joint: _FakePublisher(f"/model/panda/joint/{joint}/0/cmd_pos")
            for joint in JOINT_NAMES
        }
        created["publishers"] = publishers
        return publishers

    monkeypatch.setattr(GazeboBackend, "_make_publishers", _make_publishers)
    monkeypatch.setattr(
        gazebo_module, "JointStateReader", lambda *_a, **_k: _FakeJoints()
    )
    monkeypatch.setattr(gazebo_module, "_PoseStream", _FakePoses)
    monkeypatch.setattr(gazebo_module, "_DepthStream", _FakeDepth)
    monkeypatch.setattr(gazebo_module, "time", _FakeClock())

    # ``_publish`` imports Gazebo's protobuf at call time. The binding only
    # exists inside the container, so a stand-in is installed for the duration
    # of the test; without it every path that sends a command would fail on a
    # machine that has no Gazebo, which is the opposite of what a contract test
    # is for.
    import sys
    import types

    double_module = types.ModuleType("gz.msgs10.double_pb2")

    class Double:
        def __init__(self) -> None:
            self.data = 0.0

    double_module.Double = Double
    monkeypatch.setitem(sys.modules, "gz", types.ModuleType("gz"))
    monkeypatch.setitem(sys.modules, "gz.msgs10", types.ModuleType("gz.msgs10"))
    monkeypatch.setitem(sys.modules, "gz.msgs10.double_pb2", double_module)
    created["Double"] = Double
    return created


class _FakeClock:
    """``time`` shim so the backend's waits are instantaneous and deterministic."""

    class _Monotonic:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            self.now += 0.001
            return self.now

    def __init__(self) -> None:
        self.monotonic = self._Monotonic()
        self._nanos = 1_000_000_000

    def sleep(self, seconds: float) -> None:
        return None

    def time_ns(self) -> int:
        self._nanos += 1
        return self._nanos


# ------------------------------------------------------- contract compliance
def test_no_abstract_methods_remain_on_gazebo_backend() -> None:
    """Every method ``SimulatorBackend`` leaves abstract must be implemented.

    This is the contract in its strictest, least gameable form: if ``reset``,
    ``step``, ``observe``, ``ground_truth`` or ``close`` were left abstract the
    class could not be instantiated at all, and a test that only checked
    behaviour would pass while the boundary itself was unimplemented.
    """
    backend = GazeboBackend()
    try:
        assert isinstance(backend, SimulatorBackend)
        # The ABC's own abstract set is what a concrete subclass must clear.
        # Asserting on ``SimulatorBackend.__abstractmethods__`` would be
        # backwards: an ABC is *supposed* to carry abstract methods, and the
        # set being non-empty there proves only that it is an ABC.
        assert GazeboBackend.__abstractmethods__ == frozenset(), (
            f"{sorted(GazeboBackend.__abstractmethods__)} are still abstract"
        )
        for name in (
            "reset",
            "step",
            "observe",
            "render",
            "ground_truth",
            "close",
            "close_all",
            "capabilities",
            "is_operational",
        ):
            assert callable(getattr(backend, name)), f"missing contract method {name!r}"
        assert type(backend).name == "gazebo"
    finally:
        backend.close_all()


def test_probe_reports_backend_and_its_capabilities_without_running_it() -> None:
    """``probe`` must distinguish constructible from runnable."""
    result = probe_backend("gazebo")
    assert result["name"] == "gazebo"
    assert result["available"] is True
    capabilities = result["capabilities"]
    assert capabilities is not None
    assert capabilities.physics is True
    assert capabilities.collision is True
    # Nothing in the contract may claim determinism it does not have: Gazebo
    # integrates against wall-clock scheduling.
    assert capabilities.deterministic is False


def test_gazebo_is_not_a_masked_mock() -> None:
    """A Gazebo request must never be answered by the mock backend.

    ``create_backend("gazebo")`` could satisfy a naive caller by returning
    something that merely *behaves* like a simulator. It must return this class,
    and ``probe_backend`` must say so when the runtime is absent rather than
    reporting a usable backend that would silently fabricate every reading.
    """
    backend = create_backend("gazebo")
    try:
        assert type(backend) is GazeboBackend
        assert not isinstance(backend, MockBackend)
        assert type(backend) is not MockBackend
    finally:
        backend.close_all()


def test_missing_gz_is_an_actionable_error_not_a_silent_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no Gazebo the backend must refuse, naming what is missing."""

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("[Errno 2] No such file or directory: 'gz'")

    monkeypatch.setattr(gazebo_module.subprocess, "run", _explode)
    with pytest.raises(BackendUnavailableError) as excinfo:
        gazebo_module._require_gz()
    assert "gz" in excinfo.value.message
    assert excinfo.value.code == "simulation.gazebo_missing"


def test_backend_declares_itself_unusable_when_gz_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``is_operational`` must say no, and say why, instead of degrading."""

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("[Errno 2] No such file or directory: 'gz'")

    monkeypatch.setattr(gazebo_module.subprocess, "run", _explode)
    backend = GazeboBackend()
    try:
        usable, reason = backend.is_operational()
        assert usable is False
        assert reason and "gz" in reason
    finally:
        backend.close_all()


# ------------------------------------------------------ joint-state parsing
def test_joint_state_parser_reads_the_nested_axis1_layout_gazebo_emits() -> None:
    """Gazebo nests the angle under ``joint[].axis1.position``.

    An earlier flat ``name``/``position`` read returned zeros for every joint
    while looking perfectly healthy, so the nested layout is pinned here.
    """
    reader = JointStateReader("/world/w/model/panda/joint_state", BridgeHealth())
    document = json.dumps(
        {
            "header": {"stamp": {"sec": 12, "nsec": 500_000_000}},
            "joint": [
                {"name": "joint1", "axis1": {"position": 0.9239}},
                {"name": "left_finger", "axis1": {"position": 0.0377}},
            ],
        }
    )
    reader._consume(document)
    positions, sim_time = reader.latest()
    assert positions["joint1"] == pytest.approx(0.9239)
    assert positions["left_finger"] == pytest.approx(0.0377)
    assert sim_time == pytest.approx(12.5)
    # Joints the message did not mention must keep their previous value rather
    # than being reset to zero: a partial message is not an absence of state.
    assert positions["joint7"] == 0.0
    health = reader._health
    assert health.joint_messages == 1 and health.joint_stream is True
    # Seeing joint state is the evidence that the simulation is running.
    assert health.gazebo_started is True


def test_joint_state_parser_ignores_malformed_documents() -> None:
    """A corrupt document must be dropped, never applied as a partial state."""
    reader = JointStateReader("/topic", BridgeHealth())
    reader._consume("{ not json")
    positions, sim_time = reader.latest()
    assert positions["joint1"] == 0.0 and sim_time == 0.0
    assert reader._health.joint_messages == 0, "unparseable input is not a message"

    # A well-formed document whose joint entries carry no position must not
    # write a default: a partial message updates nothing and the simulation
    # clock must not be advanced by a stamp that is not there.
    reader._consume(json.dumps({"header": {}, "joint": [{"name": "joint1"}]}))
    reader._consume(json.dumps({"joint": [{"name": "joint1", "axis1": {}}]}))
    positions, sim_time = reader.latest()
    assert positions["joint1"] == 0.0
    assert sim_time == 0.0
    # Documents without any joint entry fall through to the flat layout, which
    # is also empty here, so no state moved in either branch.
    reader._consume(json.dumps({"joint": "not a list"}))
    assert reader.latest()[0]["joint1"] == 0.0


def test_command_mapping_targets_each_of_the_nine_topics_exactly_once(
    fake_gazebo: dict[str, Any],
) -> None:
    """One action must reach all nine joints, arms with their own target."""
    backend = GazeboBackend()
    try:
        scenario = default_scenario()
        handle = backend.reset(scenario, seed=3)
        publishers = fake_gazebo["publishers"]
        assert set(publishers) == set(scenario.robot.all_joint_names)

        targets = [0.11, -0.22, 0.33, -2.1, 0.44, 1.5, -0.55]
        backend._publish(handle, targets, 0.06)
        for name, value in zip(scenario.robot.arm_joint_names, targets, strict=True):
            assert publishers[name].data[-1] == pytest.approx(value)
        # The aperture is half the width per finger, clamped to the finger's
        # own travel: an aperture larger than the jaw can open must saturate
        # rather than be commanded past the stop.
        for name in scenario.robot.gripper_joint_names:
            assert publishers[name].data[-1] == pytest.approx(0.03)
        backend._publish(handle, targets, 9.9)
        assert publishers[scenario.robot.gripper_joint_names[0]].data[-1] == pytest.approx(0.04)
    finally:
        backend.close_all()


# ------------------------------------------------- reset and episode isolation
def test_two_resets_produce_independent_episodes(fake_gazebo: dict[str, Any]) -> None:
    """A backend may host several episodes, but they must never share state."""
    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        first = backend.reset(scenario, seed=3)
        second = backend.reset(scenario, seed=4)
        assert first.episode_id != second.episode_id
        assert set(backend._episodes) >= {first.episode_id, second.episode_id}
        assert first.seed != second.seed
        # Episode identity is what the step path keys on: closing one must not
        # be able to stop the other.
        backend.close(first)
        assert first.episode_id not in backend._episodes
        assert second.episode_id in backend._episodes
        # Driving the closed episode must be an error, not a silent operation
        # against someone else's simulation.
        with pytest.raises(SimulationError) as excinfo:
            backend.step(first, Action.noop())
        assert excinfo.value.code == "simulation.closed_episode"
    finally:
        backend.close_all()


def test_reset_is_idempotent_for_close_and_leaves_nothing_behind(
    fake_gazebo: dict[str, Any],
) -> None:
    """Shutdown must be safe to call twice and must actually terminate."""
    backend = GazeboBackend()
    scenario = default_scenario(seed=3)
    handle = backend.reset(scenario, seed=3)
    reader = backend._state(handle)["joints"]
    process = backend._state(handle)["process"]
    backend.close(handle)
    assert reader.stopped is True
    assert process.terminated is True
    # Closing again is a no-op rather than an exception: teardown paths run in
    # ``finally`` blocks where a second failure would mask the first.
    backend.close(handle)
    backend.close_all()
    backend.close_all()
    assert backend._episodes == {}
    assert backend._processes == []


def test_terminate_survives_a_child_that_ignores_the_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stubborn child must be killed, and that must not raise."""
    backend = GazeboBackend()

    class _Stubborn(_FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            raise subprocess.TimeoutExpired(cmd="gz", timeout=timeout or 0)

    stubborn = _Stubborn("gz")
    backend._terminate(stubborn)  # type: ignore[arg-type]
    assert stubborn.terminated is True
    assert stubborn.killed is True, "SIGTERM was ignored, so SIGKILL must follow"
    backend.close_all()


# --------------------------------------------------------- timeouts on step
def test_step_reports_a_stalled_clock_instead_of_hanging_forever(
    fake_gazebo: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gazebo runs asynchronously; ``step`` must bound how long it waits.

    The backend's synchronisation adapter blocks until the simulation clock has
    advanced by one control period. If the clock never moves -- a wedged solver,
    a dead process -- the contract demands an error, not a hang, and the code
    must name that cause so it is distinguishable from a slow machine.
    """
    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        handle = backend.reset(scenario, seed=3)
        reader = backend._state(handle)["joints"]
        reader.dt = 0.0  # the clock stops
        with pytest.raises(SimulationError) as excinfo:
            backend.step(handle, Action.noop())
        assert excinfo.value.code == "simulation.gazebo_clock_stalled"
    finally:
        backend.close_all()


def test_step_advances_the_episode_clock_and_step_index(
    fake_gazebo: dict[str, Any],
) -> None:
    """Observations must be stamped with the episode's own time and index."""
    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        handle = backend.reset(scenario, seed=3)
        start_time = handle.sim_time_s
        transitions = [backend.step(handle, Action.noop()) for _ in range(4)]
        assert handle.step_index == 4
        assert handle.sim_time_s > start_time
        stamps = [tr.observation.timestamp_s for tr in transitions]
        for index, transition in enumerate(transitions, start=1):
            observation = transition.observation
            # Freshness: the observation carries this step's index, is stamped
            # on the episode clock, and is never ahead of the episode's own
            # clock. Exact equality is not asserted because the simulator clock
            # is sampled more than once per step, so the stamp is the value at
            # the read that produced the observation.
            assert observation.step_index == index
            assert 0.0 < observation.timestamp_s <= handle.sim_time_s
            assert transition.robot_state.timestamp_s == observation.timestamp_s
            # Each step must advance the clock, or "fresh" would be a synonym
            # for "stuck".
            if index > 1:
                assert stamps[index - 1] > stamps[index - 2], (
                    f"step {index} did not advance the episode clock: {stamps}"
                )
            # Ground truth is carried on the transition, never inside the
            # observation the policy reads.
            assert observation.is_leaking_ground_truth() is False
    finally:
        backend.close_all()


def test_step_rejects_an_action_type_it_cannot_execute(
    fake_gazebo: dict[str, Any],
) -> None:
    """An unsupported action must fail loudly rather than be half-applied."""
    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        handle = backend.reset(scenario, seed=3)
        with pytest.raises(SimulationError) as excinfo:
            backend.step(handle, Action.joint_velocities([0.0] * 7))
        assert excinfo.value.code == "simulation.gazebo_unsupported_action"
    finally:
        backend.close_all()


# -------------------------------------------- ground truth must be measured
def _object_pose_stream(
    monkeypatch: pytest.MonkeyPatch,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    """Make ``ground_truth`` read a known pose from the simulator's stream."""
    scenario_stream = _FakePoses.__new__(_FakePoses)
    scenario_stream.data = {"target_cube": (position, orientation)}
    monkeypatch.setattr(gazebo_module, "_PoseStream", lambda *_a, **_k: scenario_stream)


def test_ground_truth_is_the_simulators_own_pose_not_a_reconstruction(
    fake_gazebo: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever the stream says is exactly what ground truth reports.

    The pre-Part-1 backend reconstructed the object's pose as *end effector
    plus a stored offset* whenever it believed the object was grasped, which
    meant a commanded close could produce a carried object without any physics.
    A pose the simulator did not produce must never appear here.
    """
    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        handle = backend.reset(scenario, seed=3)
        resting = (
            float(scenario.environment.table_height_m) + float(scenario.target.size_m),
        )
        # A unit quaternion: 0.2 rad about z. Passing an unnormalised one would
        # make the assertion a test of Quaternion's validator instead of of
        # where the orientation came from.
        half = 0.1
        _object_pose_stream(
            monkeypatch,
            (0.11, 0.22, resting[0]),
            (float(np.cos(half)), 0.0, 0.0, float(np.sin(half))),
        )
        # Rebuild the episode's pose stream with the stub.
        backend._state(handle)["poses"] = gazebo_module._PoseStream(backend._world_name)
        truth = backend.ground_truth(handle)
        target = truth.object_by_id("target_cube")
        assert target is not None
        assert target.pose.position.x == pytest.approx(0.11)
        assert target.pose.position.y == pytest.approx(0.22)
        assert target.pose.position.z == pytest.approx(resting[0])
        # Orientation is the simulator's, not an invented identity quaternion.
        assert target.pose.orientation.w == pytest.approx(float(np.cos(0.1)), abs=1e-9)
        assert target.pose.orientation.z == pytest.approx(float(np.sin(0.1)), abs=1e-9)
        # Still on the table with an open gripper: not grasped.
        assert target.grasped is False
    finally:
        backend.close_all()


def test_a_closed_gripper_over_a_resting_object_is_not_a_grasp() -> None:
    """The three conditions must all hold; any one alone is insufficient."""
    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        table = float(scenario.environment.table_height_m)
        half = float(scenario.target.size_m)
        on_table = (0.10, 0.20, table + half)
        raised = (0.10, 0.20, table + half + 0.06)
        elsewhere = (0.40, 0.40, table + half + 0.06)
        object_width = 2.0 * half

        # Closed on the object but still resting on the table: the jaws have
        # not lifted anything, so nothing has been grasped.
        assert (
            backend._supported_by_gripper(scenario, on_table, half, (0.10, 0.20, table + half), 0.001)
            is False
        )
        # Raised, but the jaws are still wide open: it was not lifted by them.
        assert (
            backend._supported_by_gripper(scenario, raised, half, (0.10, 0.20, table + half), 0.04)
            is False
        )
        # Raised with the jaws closed, but nowhere near the gripper.
        assert backend._supported_by_gripper(scenario, elsewhere, half, (0.1, 0.2, 0.1), 0.001) is False
        # Raised, jaws closed to within the object's width, object inside the
        # jaw envelope: this is what holding looks like.
        assert (
            backend._supported_by_gripper(
                scenario, raised, half, (0.10, 0.20, table + half + 0.06), object_width - 0.002
            )
            is True
        )
    finally:
        backend.close_all()


def test_pose_beliefs_are_empty_until_the_simulator_reports_a_pose(
    fake_gazebo: dict[str, Any],
) -> None:
    """No pose yet must mean no belief, never a defaulted one."""
    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        handle = backend.reset(scenario, seed=3)
        assert backend._pose_beliefs(handle) == ()
        observation = backend.observe(handle)
        assert observation.objects == ()
    finally:
        backend.close_all()


# ------------------------------------------------- task success and failure
def _ground_truth_for(
    *, position: tuple[float, float, float], grasped: bool
) -> tuple[GroundTruthState, RobotState]:
    """Build the evaluator's inputs at a chosen measured object position."""
    from robot_ci.models.geometry import Pose, Quaternion, Vec3

    object_state = GroundTruthObject(
        id="target_cube",
        is_target=True,
        pose=Pose(position=Vec3(x=position[0], y=position[1], z=position[2]),
                  orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
        grasped=grasped,
    )
    truth = GroundTruthState(objects=(object_state,), sim_time_s=1.0)
    robot_state = RobotState(
        joint_positions=(0.0,) * 7,
        gripper=GripperState(width_m=0.0, max_width_m=0.08),
    )
    return truth, robot_state


def test_pick_and_place_requires_every_criterion_not_just_a_position() -> None:
    """Position alone must not be enough: lift, release and budget matter."""
    from robot_ci.tasks import create_task

    scenario = default_scenario(seed=3)
    task = create_task(scenario)
    target = scenario.task.target_pose.position
    placed = (float(target.x), float(target.y), float(target.z))

    truth, robot_state = _ground_truth_for(position=placed, grasped=False)
    outcome = task.evaluate(Observation(), truth, robot_state, collision=False)
    # Not yet grasped and not yet lifted, so this cannot be a success even
    # though the object is already on the target.
    assert outcome.status is not TaskStatus.SUCCESS
    assert outcome.satisfied["position"] is True
    assert outcome.satisfied["grasped_at_some_point"] is False

    # Now simulate the same episode having genuinely grasped, lifted and
    # released, which is what the criteria are there to require.
    lifted, _ = _ground_truth_for(position=placed, grasped=False)
    task.track_grasped(lifted)
    held, _ = _ground_truth_for(
        position=(placed[0], placed[1], placed[2] + 0.08), grasped=True
    )
    task.track_grasped(held)
    after, _ = _ground_truth_for(position=placed, grasped=False)
    task.track_grasped(after)
    outcome = task.evaluate(Observation(), after, robot_state, collision=False)
    assert outcome.satisfied["grasped_at_some_point"] is True
    assert outcome.satisfied["lifted"] is True
    assert outcome.satisfied["released"] is True


def test_pick_and_place_fails_when_the_object_is_far_from_the_target() -> None:
    """The measured position decides, not the arm having reached a waypoint."""
    from robot_ci.tasks import create_task

    scenario = default_scenario(seed=3)
    task = create_task(scenario)
    truth, robot_state = _ground_truth_for(position=(0.10, 0.20, 0.008), grasped=False)
    outcome = task.evaluate(Observation(), truth, robot_state, collision=False)
    assert outcome.satisfied["position"] is False
    assert outcome.status.value in {"running", "failure"}


def test_collision_budget_is_enforced_by_the_task() -> None:
    """Collisions are counted and must eventually fail the episode."""
    from robot_ci.tasks import create_task

    scenario = default_scenario(seed=3)
    task = create_task(scenario)
    truth, robot_state = _ground_truth_for(position=(0.1, 0.2, 0.008), grasped=False)
    for _ in range(task.criteria.max_collisions + 2):
        outcome = task.evaluate(Observation(), truth, robot_state, collision=True)
    assert outcome.status.value == "failure"
    assert "collision" in outcome.message


# ----------------------------------------------------- frames and provenance
def test_jacobian_provider_reports_poses_in_the_world_not_the_base(
    fake_gazebo: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The IK provider and the observations must share one frame.

    The URDF tree is relative to the robot's base, but the task's waypoints, the
    object's pose and MuJoCo's provider are all world-frame. A provider that
    answered in the base frame made the policy solve for a world target in the
    wrong frame: a live episode measured the tool settling exactly the base's
    height (20 mm) above the waypoint while the IK reported a 0.26 mm residual.
    """
    import numpy as np

    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        handle = backend.reset(scenario, seed=3)
        stream = _FakePoses("world")
        monkeypatch.setattr(gazebo_module, "_PoseStream", lambda *_a, **_k: stream)
        backend._state(handle)["poses"] = stream

        q = np.asarray(scenario.robot.home_position[:7], dtype=np.float64)
        # ``forward`` returns ``(position, rotation)``, in that order.
        before_position, before_rotation = backend.jacobian_provider(handle).forward(q)

        # A pure world translation must appear verbatim in the tool position.
        stream.data["panda"] = ((0.31, 0.0, 0.02), (1.0, 0.0, 0.0, 0.0))
        shifted, _ = backend.jacobian_provider(handle).forward(q)
        assert float(np.linalg.norm(shifted - before_position)) == pytest.approx(
            float(np.linalg.norm((0.31, 0.0, 0.02))), abs=1e-9
        ), "the base's world translation must appear in the reported tool position"

        # A pure base rotation must rotate the tool with it: the tool sits ~0.5 m
        # off the base axis, so a 90 deg yaw moves it by about a metre.
        stream.data["panda"] = ((0.0, 0.0, 0.0), (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)))
        rotated, after_rotation = backend.jacobian_provider(handle).forward(q)
        assert float(np.linalg.norm(rotated - before_position)) > 0.5
        assert after_rotation @ after_rotation.T == pytest.approx(np.eye(3), abs=1e-9)
        # The Jacobian is a velocity map and a rigid base offset cannot change it.
        assert backend.jacobian_provider(handle).jacobian(q).shape == (6, 7)
        assert before_rotation.shape == (3, 3)
    finally:
        backend.close_all()


def test_observations_and_ground_truth_agree_on_the_tool_pose(
    fake_gazebo: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One TCP, two consumers: they must not disagree by a frame offset."""
    import numpy as np

    backend = GazeboBackend()
    try:
        scenario = default_scenario(seed=3)
        handle = backend.reset(scenario, seed=3)
        stream = _FakePoses("world")
        monkeypatch.setattr(gazebo_module, "_PoseStream", lambda *_a, **_k: stream)
        backend._state(handle)["poses"] = stream
        stream.data["panda"] = ((0.07, 0.0, 0.02), (1.0, 0.0, 0.0, 0.0))

        transition = backend.step(handle, Action.noop())
        observed = transition.observation.robot.end_effector_pose.position.array
        truth = transition.ground_truth.true_ee_pose.position.array
        assert truth == pytest.approx(observed, abs=1e-9), (
            "the pose the policy is shown and the pose the evaluator reads must "
            "be the same point, or the task scores something the robot never did"
        )
        # And neither is the bare base-frame number.
        q = np.asarray(
            [transition.robot_state.joint_positions[i] for i in range(7)], dtype=np.float64
        )
        bare, _ = backend._state(handle)["jacobian"]._provider.forward(q)
        assert truth[0] != pytest.approx(float(bare[0]))
    finally:
        backend.close_all()
