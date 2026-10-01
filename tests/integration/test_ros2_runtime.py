"""Runtime tests for the ROS 2 integration.

These require a real ROS 2 installation. When one is absent they **skip with an
explicit reason** rather than passing quietly or falling back to an in-process
fake: a mocked node would prove only that the mocks agree with each other, and
the whole point of this layer is that real messages carry real state.

What is verified when ROS 2 is present:

* real ``rclpy`` nodes can be created against the project's scenario,
* an observation published on ``/robot_ci/observation`` reaches a subscriber and
  the command it produces comes back on ``/robot_ci/command``,
* the command that arrives is the one applied, and is not computed locally,
* message stamps advance monotonically and match the simulation clock,
* a short episode completes over the graph and yields the project's metrics.

Run them with::

    pytest tests/integration/test_ros2_runtime.py -v
"""

from __future__ import annotations

import time

import pytest

from robot_ci.ros2.dependencies import detect

pytestmark = pytest.mark.integration

#: The reason shown when ROS 2 is missing. Reported verbatim in the skip so a CI
#: log says *why* the coverage is absent, never just "skipped".
_SKIP_REASON = (
    "ROS 2 is not installed ({reason}). "
    "Run inside the container: scripts/ros2_docker.sh pytest "
    "tests/integration/test_ros2_runtime.py"
)

_ENVIRONMENT = detect()
requires_ros = pytest.mark.skipif(
    not _ENVIRONMENT.available,
    reason=_SKIP_REASON.format(reason=_ENVIRONMENT.reason or "rclpy unavailable"),
)


@pytest.fixture
def ros_context():
    """Initialise and tear down a ROS 2 context for one test."""
    rclpy = pytest.importorskip("rclpy")
    rclpy.init()
    try:
        yield rclpy
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture
def scenario_json() -> str:
    """The project's own nominal scenario, serialized for the nodes."""
    from robot_ci.ros2.execution import build_scenario

    return build_scenario().model_dump_json()


@requires_ros
def test_simulation_and_policy_nodes_come_up(ros_context, scenario_json) -> None:
    """Both real nodes construct against the validated scenario."""
    from robot_ci.ros2.nodes import PolicyNode, SimulationNode

    simulation = SimulationNode(scenario_json=scenario_json, seed=3)
    policy = PolicyNode(policy_name="policy_v1", scenario_json=scenario_json, seed=3)
    try:
        assert simulation.node.get_name() == "robot_ci_simulation"
        assert policy.node.get_name() == "robot_ci_policy"
        assert simulation.scenario.config_hash() == policy.scenario.config_hash()
    finally:
        policy.destroy()
        simulation.destroy()


@requires_ros
def test_observation_reaches_the_policy_node_and_a_command_returns(
    ros_context, scenario_json
) -> None:
    """The observation -> command round trip crosses real middleware.

    If this fails, commands are not travelling and any "live run" would have been
    theatre.
    """
    import threading

    from robot_ci.models.state import Observation, RobotState
    from robot_ci.ros2.nodes import PolicyNode, SimulationNode

    simulation = SimulationNode(scenario_json=scenario_json, seed=3, command_timeout_s=30.0)
    policy_node = PolicyNode(policy_name="policy_v1", scenario_json=scenario_json, seed=3)
    spinner = threading.Thread(target=ros_context.spin, args=(policy_node.node,), daemon=True)
    spinner.start()
    time.sleep(1.0)  # discovery only
    try:
        observation = Observation(
            step_index=0,
            timestamp_s=0.0,
            robot=RobotState(joint_positions=tuple([0.0] * 7)),
            seed=3,
            scenario_id="nominal",
        )
        received: list[object] = []

        def driver() -> None:
            # The driver publishes and blocks; the policy node's executor thread
            # answers. This is the lock-step the episode uses, with two real
            # nodes.
            received.append(simulation.exchange(observation, timeout_s=30.0))

        thread = threading.Thread(target=driver)
        thread.start()
        thread.join(timeout=30.0)
        assert not thread.is_alive(), "the command never arrived over the graph"

        command, diagnostics = received[0]
        assert command.type.value in {"joint_position", "arm_and_gripper"}
        # Diagnostics must survive the wire: the runner's stall detector reads
        # them to decide the failure mode.
        assert isinstance(diagnostics, dict)
        assert policy_node.commands_sent >= 1
    finally:
        policy_node.destroy()
        simulation.destroy()


@requires_ros
def test_stamps_follow_the_simulation_clock_and_advance_monotonically(
    ros_context, scenario_json
) -> None:
    """Published stamps come from MuJoCo time and never go backwards."""
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState

    from robot_ci.models.state import Observation, RobotState
    from robot_ci.ros2 import encoding
    from robot_ci.ros2.nodes import SimulationNode
    from robot_ci.ros2.topics import TOPICS

    simulation = SimulationNode(scenario_json=scenario_json, seed=3)
    seen: list[int] = []

    def on_joints(message: JointState) -> None:
        seen.append(message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec)

    node = simulation.node
    node.create_subscription(JointState, TOPICS.joint_states, on_joints, 10)
    node.create_subscription(PoseStamped, TOPICS.ee_pose, lambda _message: None, 10)
    # A short fixed sleep for discovery, not a timing assumption about the run.
    time.sleep(0.5)

    try:
        for step in range(5):
            observation = Observation(
                step_index=step,
                timestamp_s=step * 0.05,
                robot=RobotState(
                    joint_positions=tuple([0.1 * step] * 7),
                    joint_names=tuple(f"panda_joint{i + 1}" for i in range(7)),
                ),
                seed=3,
                scenario_id="nominal",
            )
            simulation.publish_state(observation, observation.timestamp_s)
            simulation.publish_clock(observation.timestamp_s)
            simulation.spin_once(timeout_sec=0.05)

        assert seen, "no joint state was received on the topic"
        assert seen == sorted(seen), "stamps must be monotonic"
        assert seen[-1] > seen[0], "stamps must advance with the episode clock"
        # Delivery is not guaranteed to be complete within a fixed spin budget,
        # and pretending otherwise would make the test flaky. What must hold is
        # that every stamp that *did* arrive is an exact episode time and that
        # the sequence never regresses -- a wall-clock stamp would fail both.
        expected = {encoding.sim_time_to_ns(step * 0.05) for step in range(5)}
        assert set(seen) <= expected, "a stamp did not come from the simulation clock"
        assert seen[-1] == max(seen)
        assert simulation.published["joint_states"] == 5
    finally:
        simulation.destroy()


@requires_ros
@pytest.mark.slow
def test_a_short_episode_runs_end_to_end_over_the_graph(ros_context, scenario_json) -> None:
    """A real pick-and-place episode completes with the policy in a peer node.

    Real MuJoCo physics, real ROS messages, the real policy, and the project's
    own metrics at the end.
    """
    import threading

    from robot_ci.experiments.runner import run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.ros2.execution import build_scenario
    from robot_ci.ros2.nodes import PolicyNode, PublishingBackend, Ros2Policy, SimulationNode
    from robot_ci.simulation import create_backend

    scenario = build_scenario()
    simulation = SimulationNode(
        scenario_json=scenario_json, seed=3, publish_images=False, command_timeout_s=60.0
    )
    policy_node = PolicyNode(policy_name="policy_v1", scenario_json=scenario_json, seed=3)
    inner = create_backend("mujoco")
    backend = PublishingBackend(inner, simulation, publish_images=False)
    policy = Ros2Policy(
        PolicyConfig(name="policy_v1"), scenario, simulation, command_timeout_s=60.0
    )

    # The peer node needs its own executor thread while the driver blocks on it.
    spinner = threading.Thread(target=ros_context.spin, args=(policy_node.node,), daemon=True)
    spinner.start()
    time.sleep(1.0)  # discovery only
    try:
        result = run_episode(backend, policy, scenario, seed=3, experiment_id="ros2_test")
    finally:
        inner.close_all()
        policy_node.destroy()
        simulation.destroy()

    assert policy.round_trips > 0, "no command crossed the graph"
    assert simulation.published["joint_states"] == policy.round_trips
    assert simulation.commands_received >= policy.round_trips
    assert result.config_hash == scenario.config_hash()
    assert result.steps > 0


@requires_ros
@pytest.mark.slow
def test_ros2_episode_matches_the_direct_mujoco_episode(ros_context) -> None:
    """A ROS 2 episode must reproduce the direct MuJoCo episode exactly.

    This is the equivalence claim, and it is the test that would have caught the
    republished-observation bug: that version sent several commands per tick, the
    policy's state machine ran ahead of the arm, and the episode still *completed*
    -- reporting a clean failure rather than an error. Comparing against the
    direct run is what distinguishes "the ROS 2 path is slower" from "the ROS 2
    path changed the experiment".

    The direct run is executed here too, in the same process and on the same
    machine, so the two numbers cannot be compared across environments.
    """
    import threading

    from robot_ci.experiments.runner import run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.ros2.execution import build_scenario
    from robot_ci.ros2.nodes import PolicyNode, PublishingBackend, Ros2Policy, SimulationNode
    from robot_ci.simulation import create_backend

    scenario = build_scenario()

    direct_engine = create_backend("mujoco")
    try:
        direct = run_episode(
            direct_engine,
            create_policy(PolicyConfig(name="policy_v1"), scenario),
            scenario,
            seed=3,
            experiment_id="direct_reference",
        )
    finally:
        direct_engine.close_all()

    simulation = SimulationNode(
        scenario_json=scenario.model_dump_json(),
        seed=3,
        publish_images=False,
        command_timeout_s=60.0,
    )
    policy_node = PolicyNode(
        policy_name="policy_v1", scenario_json=scenario.model_dump_json(), seed=3
    )
    inner = create_backend("mujoco")
    backend = PublishingBackend(inner, simulation, publish_images=False)
    policy = Ros2Policy(
        PolicyConfig(name="policy_v1"), scenario, simulation, command_timeout_s=60.0
    )
    spinner = threading.Thread(target=ros_context.spin, args=(policy_node.node,), daemon=True)
    spinner.start()
    assert simulation.wait_for_peer(timeout_s=20.0), "policy node never subscribed"
    time.sleep(1.0)  # discovery only
    try:
        over_ros = run_episode(backend, policy, scenario, seed=3, experiment_id="ros2_compare")
    finally:
        inner.close_all()
        policy_node.destroy()
        simulation.destroy()

    assert over_ros.config_hash == direct.config_hash
    assert over_ros.success is direct.success, "success differs across the transport"
    assert over_ros.status == direct.status, "terminal status differs across the transport"
    assert over_ros.steps == direct.steps, "step count differs across the transport"
    # One command per tick: a surplus means the policy and the physics disagree
    # about how many steps have happened.
    assert simulation.commands_received == policy.round_trips
    assert policy_node.duplicate_observations == 0
