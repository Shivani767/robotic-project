r"""The real ROS 2 nodes.

There are two processes and one seam, and the seam is chosen so that the
existing experiment runner does not have to change:

* :class:`SimulationNode` owns the real ``MuJoCoBackend`` and publishes state.
* :class:`PolicyNode` runs the real ROBOT CI policy and publishes commands.
* :class:`Ros2Policy` is a :class:`~robot_ci.policies.RobotPolicy` whose
  ``act`` publishes the observation, then **blocks until a command arrives on
  the command topic**, returning that command.

Because ``Ros2Policy`` is an ordinary policy, the episode is still run by
:func:`robot_ci.experiments.runner.run_episode` -- the same function that
produced every number in this repository. Corruption, the stall detector, task
evaluation, the metric vector and the config hash are therefore computed by
unchanged code, which is what makes a ROS 2 run comparable to a direct one.

What actually travels over ROS while an episode runs::

    MuJoCo --observe--> SimulationNode --/robot_ci/observation--> PolicyNode
                                                                      |
                                                            real policy.act()
                                                                      |
    MuJoCo <--step()-- Ros2Policy <--/robot_ci/command <--------------+

The action applied to MuJoCo is the object that arrived on the command topic.
It is not recomputed locally, so the command path cannot silently diverge from
what the graph carried.

``rclpy`` is imported inside this module's functions rather than at module
scope, so ``import robot_ci`` stays free of ROS.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from typing import Any

from robot_ci.core.errors import SimulationError
from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
from robot_ci.models.state import Action, Observation
from robot_ci.policies import create_policy
from robot_ci.policies.base import PolicyDiagnostics, PolicyOutput, RobotPolicy
from robot_ci.ros2 import encoding
from robot_ci.ros2.topics import FRAMES, TOPICS
from robot_ci.simulation.base import SimulatorBackend

__all__ = [
    "DEFAULT_COMMAND_TIMEOUT_S",
    "PolicyNode",
    "PublishingBackend",
    "Ros2Policy",
    "SimulationNode",
    "main_policy",
    "main_simulation",
]

#: Wall-clock seconds to wait for a command before declaring the graph dead.
#:
#: This is the **only** place wall-clock time is consulted, and it is a
#: liveness check on another process -- "did the policy node answer" -- not a
#: determinant of robot state. Robot state and every message stamp come from
#: MuJoCo time. A timeout therefore cannot make a run non-deterministic; it can
#: only fail a run whose peer process is absent.
DEFAULT_COMMAND_TIMEOUT_S = 30.0


def _stamp(node: Any, sim_time_s: float) -> Any:
    """Build a ``builtin_interfaces/Time`` from MuJoCo episode seconds."""
    from builtin_interfaces.msg import Time

    nanoseconds = encoding.sim_time_to_ns(sim_time_s)
    return Time(sec=nanoseconds // encoding.NS_PER_S, nanosec=nanoseconds % encoding.NS_PER_S)


def _header(node: Any, sim_time_s: float, frame_id: str) -> Any:
    """Build a ``std_msgs/Header`` stamped with the simulation clock."""
    from std_msgs.msg import Header

    return Header(stamp=_stamp(node, sim_time_s), frame_id=frame_id)


class SimulationNode:
    """The node that owns MuJoCo and publishes robot state.

    It is deliberately not the thing that runs the episode loop: the episode is
    run by the existing runner, and this node is bound to the backend the
    runner is using through :class:`PublishingBackend`. That is what keeps one
    definition of an episode in the repository.

    Args:
        scenario_json: The scenario, as the exact JSON the driver validated.
            Passing the payload rather than a path means the node cannot load a
            different file than the driver used.
        seed: Episode seed, from the command line -- never from time or PID.
        publish_images: Whether to render and publish camera frames.
        image_stride: Publish a frame every Nth control tick.
    """

    def __init__(
        self,
        *,
        scenario_json: str,
        seed: int,
        node_name: str = "robot_ci_simulation",
        publish_images: bool = True,
        image_stride: int = 1,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
    ) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from rclpy.executors import SingleThreadedExecutor
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Bool, String

        self.scenario: ScenarioConfig = ScenarioConfig.model_validate_json(scenario_json)
        self.seed = int(seed)
        self.publish_images = bool(publish_images)
        self.image_stride = max(1, int(image_stride))
        self.command_timeout_s = float(command_timeout_s)

        self._node = rclpy.create_node(node_name)
        # This node owns its executor rather than using rclpy's global one. Two
        # nodes in one process -- the driver and its peer -- would otherwise
        # collide on the default executor ("Executor is already spinning"), and
        # library code reaching for global state is a bug waiting for the next
        # process that runs two of them.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._node.declare_parameter("seed", self.seed)
        self._node.declare_parameter("scenario_id", self.scenario.scenario_id)

        self._pub_clock = self._node.create_publisher(Clock, TOPICS.clock, 10)
        self._pub_joints = self._node.create_publisher(JointState, TOPICS.joint_states, 10)
        self._pub_ee = self._node.create_publisher(PoseStamped, TOPICS.ee_pose, 10)
        self._pub_tf = self._node.create_publisher(TransformStamped, TOPICS.ee_tf, 10)
        self._pub_rgb = self._node.create_publisher(Image, TOPICS.camera_rgb, 5)
        self._pub_depth = self._node.create_publisher(Image, TOPICS.camera_depth, 5)
        self._pub_contact = self._node.create_publisher(Bool, TOPICS.contact_state, 10)
        self._pub_observation = self._node.create_publisher(String, TOPICS.observation, 10)
        self._pub_result = self._node.create_publisher(String, TOPICS.episode_result, 10)

        self._commands: deque[tuple[Action, dict[str, Any]]] = deque()
        self._command_count = 0
        self._command_sub = self._node.create_subscription(
            String, TOPICS.command, self._on_command, 10
        )

        self.handle: Any = None
        self.published: dict[str, int] = {
            "joint_states": 0,
            "ee_pose": 0,
            "tf": 0,
            "rgb": 0,
            "depth": 0,
            "observation": 0,
            "command": 0,
            "clock": 0,
        }

    # -- lifecycle ---------------------------------------------------------
    @property
    def node(self) -> Any:
        """The underlying ``rclpy`` node, for executors and tests."""
        return self._node

    def bind_handle(self, handle: Any) -> None:
        """Record the episode handle the runner just created."""
        self.handle = handle

    def destroy(self) -> None:
        """Tear the node down, ignoring a second call."""
        with contextlib.suppress(Exception):  # shutdown races are not failures
            self._executor.remove_node(self._node)
            self._executor.shutdown()
            self._node.destroy_node()

    # -- command path ------------------------------------------------------
    def _on_command(self, message: Any) -> None:
        """Receive a command from the policy node.

        Commands are queued rather than stored in a single slot. The episode
        loop is lock-step -- one command per observation -- so a FIFO is
        sufficient, while a single slot would let a duplicate delivery
        overwrite the command the current tick is waiting for.
        """
        try:
            action, diagnostics = encoding.command_from_payload(
                encoding.decode_payload(message.data)
            )
        except Exception as exc:
            self._node.get_logger().error(f"rejected malformed command: {exc}")
            return
        self._commands.append((action, diagnostics))
        self._command_count += 1
        self.published["command"] += 1

    def take_command(self, timeout_s: float | None = None) -> tuple[Action, dict[str, Any]]:
        """Block until the policy node sends the next command.

        Args:
            timeout_s: Wall-clock liveness budget; the node default is used
                when ``None``.

        Returns:
            The ``(action, diagnostics)`` received on the command topic.

        Raises:
            SimulationError: If no command arrives, naming the topic and the
                likely cause. Silence is reported as the graph failing to
                answer, never smoothed over by reusing the previous action.
        """
        budget = self.command_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + budget
        while not self._commands:
            if time.monotonic() > deadline:
                raise SimulationError(
                    f"no command on {TOPICS.command} within {budget:.1f}s; "
                    "is the policy node running and subscribed to "
                    f"{TOPICS.observation}?",
                    code="ros2.command_timeout",
                    topic=TOPICS.command,
                    timeout_s=budget,
                )
            self._executor.spin_once(timeout_sec=0.05)
        return self._commands.popleft()

    @property
    def pending_commands(self) -> int:
        """How many commands are queued but not yet consumed."""
        return len(self._commands)

    @property
    def commands_received(self) -> int:
        """How many commands have arrived on the command topic in total."""
        return self._command_count

    def spin_once(self, timeout_sec: float = 0.05) -> None:
        """Pump this node's executor once.

        Exposed for tests and for interactive tooling that subscribe to the
        topics in-process. It drives the node's own executor, so calling it
        never contends with another node's executor over rclpy's global one.
        """
        self._executor.spin_once(timeout_sec=timeout_sec)

    # -- observation path --------------------------------------------------
    def publish_clock(self, sim_time_s: float) -> None:
        """Publish the MuJoCo clock. This is the authoritative time source."""
        from rosgraph_msgs.msg import Clock

        message = Clock()
        message.clock = _stamp(self._node, sim_time_s)
        self._pub_clock.publish(message)
        self.published["clock"] += 1

    def publish_state(self, observation: Observation, sim_time_s: float) -> None:
        """Publish proprioception, end-effector pose, TF and contact state.

        Every stamp comes from ``sim_time_s`` -- the MuJoCo time the
        observation was taken at -- so all four topics describe one instant.
        """
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        robot = observation.robot
        fields = encoding.joint_state_fields(robot)

        joints = JointState()
        joints.header = _header(self._node, sim_time_s, FRAMES["base"])
        joints.name = fields["name"]
        joints.position = fields["position"]
        joints.velocity = fields["velocity"]
        joints.effort = fields["effort"]
        self._pub_joints.publish(joints)
        self.published["joint_states"] += 1

        pose = robot.end_effector_pose
        qx, qy, qz, qw = encoding.quaternion_to_ros(pose.orientation)

        stamped = PoseStamped()
        stamped.header = _header(self._node, sim_time_s, FRAMES["base"])
        stamped.pose.position.x = float(pose.position.x)
        stamped.pose.position.y = float(pose.position.y)
        stamped.pose.position.z = float(pose.position.z)
        stamped.pose.orientation.x = qx
        stamped.pose.orientation.y = qy
        stamped.pose.orientation.z = qz
        stamped.pose.orientation.w = qw
        self._pub_ee.publish(stamped)
        self.published["ee_pose"] += 1

        transform = TransformStamped()
        transform.header = _header(self._node, sim_time_s, FRAMES["base"])
        transform.child_frame_id = FRAMES["end_effector"]
        transform.transform.translation.x = float(pose.position.x)
        transform.transform.translation.y = float(pose.position.y)
        transform.transform.translation.z = float(pose.position.z)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self._pub_tf.publish(transform)
        self.published["tf"] += 1

        contact = Bool()
        contact.data = bool(robot.contact_detected)
        self._pub_contact.publish(contact)

    def publish_observation(self, observation: Observation) -> None:
        """Publish the full observation the policy is about to act on."""
        self._publish_observation_payload(
            encoding.encode_payload(encoding.observation_to_payload(observation))
        )

    def _publish_observation_payload(self, payload: str) -> None:
        """Publish a pre-encoded observation payload."""
        from std_msgs.msg import String

        message = String()
        message.data = payload
        self._pub_observation.publish(message)
        self.published["observation"] += 1

    def exchange(
        self,
        observation: Observation,
        timeout_s: float | None = None,
    ) -> tuple[Action, dict[str, Any]]:
        """Publish an observation once and block until a command comes back.

        Publishing happens **once** per observation, which is what makes this a
        lock step: one observation in, one command out, one physics step. An
        earlier version republished the observation every 200 ms while waiting
        for the reply, on the theory that a volatile message sent before
        discovery matched would be lost. That was correct about the transport and
        wrong about the protocol -- every republication re-entered the policy
        node's ``act``, advancing the policy's state machine several times per
        tick and leaving surplus commands in the FIFO to be applied to later
        steps. The arm and the policy desynchronised and every episode ended in
        a failed grasp. Discovery is handled once, by :meth:`wait_for_peer`,
        before the episode starts.

        Args:
            observation: The observation to publish.
            timeout_s: Overall wall-clock budget; the node default when None.

        Returns:
            The ``(action, diagnostics)`` received on the command topic.

        Raises:
            SimulationError: If no command arrives within the budget.
        """
        budget = self.command_timeout_s if timeout_s is None else float(timeout_s)
        self._publish_observation_payload(
            encoding.encode_payload(encoding.observation_to_payload(observation))
        )
        deadline = time.monotonic() + budget
        while not self._commands:
            if time.monotonic() > deadline:
                raise SimulationError(
                    f"no command on {TOPICS.command} within {budget:.1f}s; "
                    "is the policy node running and subscribed to "
                    f"{TOPICS.observation}?",
                    code="ros2.command_timeout",
                    topic=TOPICS.command,
                    timeout_s=budget,
                )
            self._executor.spin_once(timeout_sec=0.01)
        return self._commands.popleft()

    def wait_for_peer(self, timeout_s: float = 20.0) -> bool:
        """Block until something is subscribed to the observation topic.

        Called once before an episode so the first observation is not the one
        lost to discovery. Returns False on timeout, which the caller reports
        rather than proceeding into a guaranteed stall.

        Args:
            timeout_s: Wall-clock budget for discovery.

        Returns:
            True if a subscriber appeared within the budget.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._node.count_subscribers(TOPICS.observation) > 0:
                return True
            self._executor.spin_once(timeout_sec=0.05)
        return False

    def publish_frame(self, frame: Any) -> None:
        """Publish an RGB and/or depth frame as standard Image messages."""
        from sensor_msgs.msg import Image

        if frame is None or not getattr(frame, "valid", False):
            return
        stamp = _stamp(self._node, float(getattr(frame, "timestamp_s", 0.0)))

        if frame.rgb is not None:
            rgb = Image()
            rgb.header.stamp = stamp
            rgb.header.frame_id = FRAMES["camera"]
            rgb.height = int(frame.height_px)
            rgb.width = int(frame.width_px)
            rgb.encoding = "rgb8"
            rgb.is_bigendian = False
            rgb.step = int(frame.width_px) * 3
            rgb.data = encoding.rgb_to_bytes(frame.rgb)
            self._pub_rgb.publish(rgb)
            self.published["rgb"] += 1

        if frame.depth is not None:
            depth = Image()
            depth.header.stamp = stamp
            depth.header.frame_id = FRAMES["camera"]
            depth.height = int(frame.height_px)
            depth.width = int(frame.width_px)
            depth.encoding = "32FC1"
            depth.is_bigendian = False
            depth.step = int(frame.width_px) * 4
            depth.data = encoding.depth_to_float32_bytes(frame.depth)
            self._pub_depth.publish(depth)
            self.published["depth"] += 1

    def publish_result(self, payload: dict[str, Any]) -> None:
        """Publish the episode result so a subscriber sees the run end."""
        from std_msgs.msg import String

        message = String()
        message.data = encoding.encode_payload(payload)
        self._pub_result.publish(message)


class PublishingBackend(SimulatorBackend):
    """A backend decorator that mirrors every episode event onto ROS topics.

    This is a transparent wrapper: ``reset``, ``observe``, ``step``,
    ``ground_truth``, ``render`` and ``jacobian_provider`` all delegate to the
    real backend, so the physics, the task evaluation and the metrics are the
    ones the project already validates. What it adds is publication:

    * ``observe`` republishes state and the observation payload,
    * ``step`` publishes the camera frame for the state it just produced.

    It does **not** enforce that the action came from ROS. That is
    :class:`Ros2Policy`'s job, and the separation is deliberate: this class can
    be used to observe an ordinary local run, while a run whose commands came
    over the graph is the one whose policy is a :class:`Ros2Policy`.

    ``name`` reports ``mujoco+ros2`` so an artifact records which path produced
    it. The name is execution metadata: the scenario, and therefore the
    configuration hash, are untouched.
    """

    name = "mujoco+ros2"
    description = "MuJoCo physics with state and commands carried over a ROS 2 graph"

    def __init__(
        self,
        inner: SimulatorBackend,
        node: SimulationNode,
        *,
        publish_images: bool = True,
        image_stride: int = 1,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._sim = node
        self._publish_images = bool(publish_images)
        self._image_stride = max(1, int(image_stride))
        self._ticks = 0

    # -- introspection -----------------------------------------------------
    def capabilities(self) -> Any:
        """Report the wrapped backend's capabilities, unchanged."""
        return self._inner.capabilities()

    def is_operational(self) -> tuple[bool, str]:
        """Defer to the wrapped backend's own runtime check."""
        return self._inner.is_operational()

    # -- lifecycle ---------------------------------------------------------
    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> Any:
        """Start the wrapped episode and bind its handle to the node."""
        handle = self._inner.reset(scenario, seed=seed)
        self._sim.bind_handle(handle)
        return handle

    def observe(self, handle: Any) -> Observation:
        """Observe, then publish that state on the graph.

        Only the proprioceptive topics are published here. The observation
        *payload* is published by :class:`Ros2Policy`, because it has to be
        published in the same operation that waits for the command -- see
        :meth:`SimulationNode.exchange` -- and publishing it from both places
        would put two copies of every observation on the topic.

        Publication happens after the wrapped backend has produced the value,
        so what goes on the wire is what the policy will see, at the MuJoCo
        time it was measured.
        """
        observation = self._inner.observe(handle)
        sim_time = float(observation.timestamp_s)
        self._sim.publish_state(observation, sim_time)
        self._sim.publish_clock(sim_time)
        return observation

    def step(self, handle: Any, action: Action) -> Any:
        """Advance the wrapped backend, then publish the resulting frame."""
        transition = self._inner.step(handle, action)
        self._ticks += 1
        if self._publish_images and self._ticks % self._image_stride == 0:
            self._sim.publish_frame(self._inner.render(handle))
        return transition

    def render(self, handle: Any) -> Any:
        """Render the wrapped backend and publish the frame."""
        frame = self._inner.render(handle)
        self._sim.publish_frame(frame)
        return frame

    def ground_truth(self, handle: Any) -> Any:
        """Return the wrapped backend's hidden state, unpublished by design."""
        return self._inner.ground_truth(handle)

    def close(self, handle: Any) -> None:
        """Close the wrapped episode."""
        self._inner.close(handle)

    def close_all(self) -> None:
        """Close every wrapped episode."""
        self._inner.close_all()

    def jacobian_provider(self, handle: Any) -> Any:
        """Provide the wrapped backend's Jacobian, for the policy's IK."""
        return self._inner.jacobian_provider(handle)


class Ros2Policy(RobotPolicy):
    """A policy whose action is computed in another process and sent over ROS.

    ``act`` publishes nothing itself -- the observation was published by
    :class:`PublishingBackend` at observe time -- and blocks until the command
    topic delivers the action. The returned action is therefore the object that
    crossed the graph, and the runner applies exactly that to MuJoCo.

    This class holds no decision logic, which is the point: the decision is made
    by the real ROBOT CI policy in the policy node, and this is the adapter that
    carries its output into the control path. There is no fallback that
    recomputes an action locally, because a fallback would make the graph
    optional while still reporting that ROS 2 was used.
    """

    implementation = "ros2"

    def __init__(
        self,
        config: PolicyConfig,
        scenario: ScenarioConfig,
        node: SimulationNode,
        *,
        command_timeout_s: float | None = None,
    ) -> None:
        super().__init__(config, scenario)
        self._sim = node
        self._timeout = command_timeout_s
        self.round_trips = 0

    def reset(
        self, config: PolicyConfig, seed: int, scenario: ScenarioConfig | None = None
    ) -> None:
        """Record the episode identity; the peer node resets from the same seed."""
        self.config = config
        self.seed = int(seed)
        self.step_count = 0
        self.round_trips = 0
        if scenario is not None:
            self.scenario = scenario

    def act(self, observation: Observation) -> PolicyOutput:
        """Wait for the action the policy node computed for this observation.

        The observation is published here rather than by
        :class:`PublishingBackend`, because publishing and waiting for the
        reply are one operation over a lossy transport: see
        :meth:`SimulationNode.exchange`.

        Raises:
            SimulationError: If the graph does not answer, raised by
                :meth:`SimulationNode.exchange`. Never substituted with a
                locally computed action.
        """
        action, diagnostics = self._sim.exchange(observation, self._timeout)
        self.round_trips += 1
        self.step_count += 1
        # The diagnostics must survive the round trip: the runner's stall
        # detector decides the *failure mode* from them, so an empty mapping here
        # turns a detected stall into a step-cap timeout.
        return PolicyOutput(action, PolicyDiagnostics(diagnostics))


class PolicyNode:
    """The node that runs the real ROBOT CI policy and publishes commands.

    It receives observations on ``/robot_ci/observation``, decodes them, asks
    the project's own policy what to do, and publishes the resulting
    :class:`~robot_ci.models.state.Action` on ``/robot_ci/command``.

    Determinism rules honoured here:

    * the policy name, the seed and the exact scenario JSON all arrive as ROS
      **parameters**, set by the launch or the CLI -- never from a node name, a
      process id, a timestamp or the wall clock;
    * the policy is reset from those parameters when the episode starts, which
      is detected by the first observation (``step_index == 0``) rather than by
      wall-clock timing;
    * nothing here mutates the scenario, and in particular ``scenario_id`` is
      left alone. ``scenario_id`` seeds the scene's object placement, so
      rewriting it would move the target and quietly invalidate the comparison
      against a direct run.

    Args:
        policy_name: Factory key, e.g. ``policy_v1``.
        scenario_json: The exact scenario JSON the driver validated.
        seed: Episode seed, used for the policy's own tie-breaking.
        use_sim_time: Subscribe to ``/clock`` instead of the wall clock.
    """

    def __init__(
        self,
        *,
        policy_name: str,
        scenario_json: str,
        seed: int,
        node_name: str = "robot_ci_policy",
        use_sim_time: bool = True,
    ) -> None:
        import rclpy
        from rclpy.parameter import Parameter
        from std_msgs.msg import String

        self.scenario: ScenarioConfig = ScenarioConfig.model_validate_json(scenario_json)
        self.seed = int(seed)
        self.config = PolicyConfig(name=policy_name)
        self.policy_name = policy_name
        self._sim_time = bool(use_sim_time)

        # Parameter setup is done after creation, not through NodeOptions: those
        # do not exist in Jazzy (they arrived in a later distro), and this is the
        # pattern Jazzy supports. `use_sim_time` is special-cased by rclcpp and
        # can be set undeclared; every other parameter must be declared with a
        # plain default value rather than a Parameter object. Both facts were
        # established by probing this distribution rather than by reading docs.
        self._node = rclpy.create_node(node_name)
        self._node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, self._sim_time)])
        self._node.declare_parameter("seed", self.seed)
        self._node.declare_parameter("policy", policy_name)
        self._node.declare_parameter("scenario_id", self.scenario.scenario_id)

        # Private kinematic-only backend, used solely as a Jacobian source. It
        # is never stepped and never read for sensing, so it cannot influence the
        # simulation. Declared before the policy so `_attach_planner` can use it.
        self._kinematics: Any = None
        self._kinematics_handle: Any = None

        self._policy = create_policy(self.config, self.scenario)
        self._policy.reset(self.config, self.seed, self.scenario)
        self._active_scenario_id = self.scenario.scenario_id
        self._attach_planner()

        self._pub_command = self._node.create_publisher(String, TOPICS.command, 10)
        self._commands_sent = 0
        self._last_step_index = -1
        self.duplicate_observations = 0
        self._sub = self._node.create_subscription(
            String, TOPICS.observation, self._on_observation, 10
        )

    @property
    def node(self) -> Any:
        """The underlying ``rclpy`` node."""
        return self._node

    @property
    def commands_sent(self) -> int:
        """How many commands this node has published."""
        return self._commands_sent

    def _attach_planner(self) -> None:
        """Give the policy the Jacobian source it needs to solve IK.

        ``run_episode`` calls ``backend.jacobian_provider(handle)`` and hands the
        result to ``policy.attach_planner``. Over this architecture the policy
        lives in *this* process while the backend lives in the other one, so the
        runner's call lands on the driver-side :class:`Ros2Policy`, which has no
        ``attach_planner`` at all. The effect was invisible and total: the policy
        solved inverse kinematics with no Jacobian, could not reach the grasp
        pose, and every episode ran to the step cap and reported a clean
        *failure* rather than an error.

        The provider is rebuilt here from a private, never-stepped backend built
        from the same scenario. That is sound because the Jacobian depends only
        on the robot model and the queried configuration -- not on simulation
        state -- so the two processes compute identical values. Rebuilding from
        the same MuJoCo model, rather than substituting a finite-difference
        provider from a URDF, is what keeps IK numerics identical to a direct
        run; a different Jacobian source would quietly change the experiment.
        """
        if not hasattr(self._policy, "attach_planner"):
            return
        from robot_ci.simulation import create_backend

        if self._kinematics is None:
            self._kinematics = create_backend("mujoco")
            self._kinematics_handle = self._kinematics.reset(self.scenario, seed=self.seed)
        provider = self._kinematics.jacobian_provider(self._kinematics_handle)
        self._policy.attach_planner(provider)

    def destroy(self) -> None:
        """Tear the node down, ignoring a second call."""
        if self._kinematics is not None:
            with contextlib.suppress(Exception):
                self._kinematics.close_all()
        with contextlib.suppress(Exception):  # shutdown races are not failures
            self._node.destroy_node()

    def _on_observation(self, message: Any) -> None:
        """Decode an observation, run the policy, publish the command."""
        from std_msgs.msg import String

        try:
            payload = encoding.decode_payload(message.data)
            observation = encoding.observation_from_payload(payload)
        except Exception as exc:
            self._node.get_logger().error(f"rejected malformed observation: {exc}")
            return

        # A new episode is identified by the scenario changing or the step
        # counter restarting, both of which are properties of the data.
        if observation.scenario_id != self._active_scenario_id or observation.step_index == 0:
            self._active_scenario_id = observation.scenario_id
            self._last_step_index = observation.step_index
            self._policy.reset(self.config, self.seed, self.scenario)
            self._attach_planner()
        elif observation.step_index == self._last_step_index:
            # Idempotence guard. The driver publishes each observation exactly
            # once, so a repeat means the transport duplicated it. Acting on it
            # again would advance the policy's state machine without the arm
            # having moved, and the second command would be applied to a later
            # step. Dropping the repeat keeps policy and physics in step.
            self.duplicate_observations += 1
            self._node.get_logger().warning(
                f"ignored duplicate observation for step {observation.step_index}"
            )
            return
        self._last_step_index = observation.step_index

        try:
            output = self._policy.act(observation)
        except Exception as exc:
            self._node.get_logger().error(f"policy failed at step {observation.step_index}: {exc}")
            return

        command = String()
        command.data = encoding.encode_payload(
            encoding.command_to_payload(output.action, dict(output.diagnostics))
        )
        self._pub_command.publish(command)
        self._commands_sent += 1


def _parameter(node: Any, name: str, default: Any) -> Any:
    """Read a launch-supplied parameter, falling back to ``default``.

    Declaring first means an unset parameter is a known default rather than an
    exception, which is what makes `ros2 run` usable without a launch file. The
    default is passed as a plain value: Jazzy rejects a `Parameter` object here,
    which is a different failure from an undeclared parameter and cost a build
    cycle to find.
    """
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).get_parameter_value()


def main_policy() -> int:
    """Entry point for the policy node.

    Reads ``policy``, ``seed`` and ``scenario_json`` from ROS parameters and
    spins until interrupted. The seed is a parameter, not something derived
    from the environment, so two runs with the same parameters behave
    identically.
    """
    import rclpy

    from robot_ci.ros2.dependencies import require_rclpy

    require_rclpy()
    rclpy.init()
    node: PolicyNode | None = None
    try:
        probe = rclpy.create_node("robot_ci_policy_probe")
        policy_name = _parameter(probe, "policy", "policy_v1").string_value
        seed = int(_parameter(probe, "seed", 3).integer_value)
        scenario_json = _parameter(probe, "scenario_json", "").string_value
        probe.destroy_node()
        if not scenario_json:
            raise SystemExit(
                "the policy node needs scenario_json; launch the bringup package "
                "or pass -p scenario_json:='{...}'"
            )

        node = PolicyNode(policy_name=policy_name, scenario_json=scenario_json, seed=seed)
        node.node.get_logger().info(
            f"policy={policy_name} seed={seed}; subscribed to {TOPICS.observation}"
        )
        rclpy.spin(node.node)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        if node is not None:
            node.destroy()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main_simulation() -> int:
    """Entry point for the simulation/driver node.

    Runs one episode with the policy reached over the graph, then writes the
    execution artifact. Exits non-zero if the episode failed, so a launch or a
    CI job can gate on the outcome rather than on the process starting.
    """
    import rclpy

    from robot_ci.ros2.dependencies import require_rclpy
    from robot_ci.ros2.execution import ros2_artifact, run_episode_over_ros

    require_rclpy()
    rclpy.init()
    success = False
    try:
        probe = rclpy.create_node("robot_ci_simulation_probe")
        policy_name = _parameter(probe, "policy", "policy_v1").string_value
        seed = int(_parameter(probe, "seed", 3).integer_value)
        noise = float(_parameter(probe, "depth_noise_mm", -1.0).double_value)
        publish_images = bool(_parameter(probe, "publish_images", True).bool_value)
        image_stride = int(_parameter(probe, "image_stride", 1).integer_value)
        timeout = float(_parameter(probe, "command_timeout_s", 30.0).double_value)
        probe.destroy_node()

        record = run_episode_over_ros(
            policy_name=policy_name,
            seed=seed,
            depth_noise_mm=None if noise < 0 else noise,
            publish_images=publish_images,
            image_stride=image_stride,
            command_timeout_s=timeout,
            node_name="robot_ci_simulation",
        )
        written = ros2_artifact(record)
        result = record["result"]
        success = bool(result["success"])
        print(
            f"wrote {written}\n"
            f"episode: {result['status']} success={success} steps={result['steps']}\n"
            f"config_hash: {record['execution']['config_hash']}\n"
            f"commands over ROS: {record['graph']['commands_applied']}",
            flush=True,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if success else 1
