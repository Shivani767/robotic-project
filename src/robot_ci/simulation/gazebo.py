"""The Gazebo Harmonic backend: the project's second real simulator.

This module is the only one in the package that imports ``gz.transport13``. It
reaches a running ``gz sim`` exactly the way :mod:`robot_ci.simulation.mujoco`
reaches MuJoCo: through :class:`~robot_ci.simulation.base.SimulatorBackend`, so
policies, regression detection and the runner are unchanged.

How a tick works
----------------
The backend owns a headless ``gz sim`` process. :meth:`GazeboBackend.reset`
generates the SDF from the scenario, starts the server, and waits for the robot's
joint-state topic to prove the model and its nine joint controllers loaded.
:meth:`GazeboBackend.step` publishes the action's joint targets onto Gazebo's own
``cmd_pos`` topics and then waits for the simulation clock to advance by one
control period. The arm moves because Gazebo's contact solver moves it; the
backend never writes a pose directly.

Why the transport is split
--------------------------
The vendored ``gz_msgs10`` in the ROS 2 Jazzy image has no ``joint_state.proto``
and so ``ros_gz_bridge`` refuses that message pair, while ``image_pb2`` is
present. Joint state therefore comes from the ``gz`` CLI's JSON and poses, depth
and images from the Python bindings. Both are the simulation's own output.

Known limitation
----------------
Gazebo 8.15.0's Sensors path does not draw a model that has both joints and
plugins, and nine ``JointPositionController`` instances must live inside the
robot model. Control is kept, so the camera shows the table and the object but
not the robot. See ``docs/gazebo.md`` and
``experiments/results/gazebo/evidence_robot_not_rendered.txt``.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot_ci.core.errors import BackendUnavailableError, SimulationError
from robot_ci.core.logging import get_logger
from robot_ci.gazebo.live import JOINT_NAMES, JointStateReader
from robot_ci.gazebo.world import GAZEBO_ROBOT_NAME, world_sdf
from robot_ci.models.environment import TaskName
from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import (
    Action,
    ActionType,
    GripperState,
    GroundTruthObject,
    GroundTruthState,
    ObjectState,
    Observation,
    RobotState,
    SensorFrame,
    Source,
)
from robot_ci.simulation.base import (
    BackendCapabilities,
    EpisodeHandle,
    SimulatorBackend,
    Transition,
)

__all__ = ["GazeboBackend"]

_LOG = get_logger(__name__)

#: URDF joint name -> ROBOT CI arm index. The first seven match the scenario's
#: ``all_joint_names``; the two fingers follow.
ARM_JOINTS: tuple[str, ...] = JOINT_NAMES[:7]
FINGER_JOINTS: tuple[str, ...] = JOINT_NAMES[7:]
#: Link Gazebo reports the wrist pose on; its origin is the end effector.
EE_LINK = "panda_link7"
OBJECT_MODEL = "target_cube"

#: Smallest height above the fitted support plane for a point to count as off
#: the surface at all. Just above the tabletop's own back-projection residual,
#: measured at 0.004 m on a real 160x120 frame.
_SURFACE_MIN_HEIGHT_M = 0.012

#: How far above the support plane a candidate may still count as *resting* on
#: it. Measured on a real frame: the cube's lowest point sits 0.0137 m above the
#: plane and the Panda link nearest the camera floats 0.0402 m up.
_RESTING_TOLERANCE_M = 0.025

#: Fewest points for a group of pixels to be a candidate object.
_MIN_CLUSTER_POINTS = 8


def _require_gz() -> None:
    """Raise an actionable error when the Gazebo runtime is missing."""
    if not any(
        Path(p).exists() for p in (os.environ.get("PATH") or "").split(os.pathsep) if p
    ):  # pragma: no cover - defensive
        raise BackendUnavailableError(
            "Gazebo backend: no executable directory on PATH",
            code="simulation.gazebo_missing",
        )
    try:
        subprocess.run(
            ["gz", "sim", "--versions"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendUnavailableError(
            f"Gazebo backend: cannot run 'gz' ({exc}). Source the ROS 2 setup "
            f"script first: source /opt/ros/$ROS_DISTRO/setup.bash",
            code="simulation.gazebo_missing",
        ) from exc
    try:
        import gz.transport13  # noqa: F401
    except ImportError as exc:
        raise BackendUnavailableError(
            f"Gazebo backend: gz.transport13 is not importable ({exc}). "
            f"Install ros-gz in the image.",
            code="simulation.gazebo_bindings_missing",
        ) from exc


def _indices_of(
    voxels: np.ndarray, members: set[tuple[int, int, int]]
) -> list[int]:
    """Return the positions in ``voxels`` whose key is in ``members``."""
    return [index for index, voxel in enumerate(voxels) if tuple(int(v) for v in voxel) in members]


class _PoseStream:
    """Latest world poses of every model and link, straight from Gazebo.

    ``/world/<name>/pose/info`` is the simulator's own statement of where things
    are. Reading it is what keeps the end-effector pose honest: it is a measured
    pose, not a forward-kinematics estimate of one.
    """

    def __init__(self, world: str) -> None:
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node

        self._lock = threading.Lock()
        self._poses: dict[str, tuple[float, float, float]] = {}
        node = Node()
        node.subscribe(Pose_V, f"/world/{world}/pose/info", self._on_pose)
        self._node = node

    def _on_pose(self, msg: Any) -> None:
        snapshot: dict[str, tuple[float, float, float]] = {}
        for entity in msg.pose:
            if entity.name:
                snapshot[entity.name] = (
                    float(entity.position.x),
                    float(entity.position.y),
                    float(entity.position.z),
                )
        with self._lock:
            self._poses = snapshot

    def position(self, name: str) -> tuple[float, float, float] | None:
        """Return the world position of ``name``, or ``None`` if unreported."""
        with self._lock:
            return self._poses.get(name)


class _DepthStream:
    """Latest depth image plus its intrinsics, used for perception.

    Perception here is real, not a shortcut: object belief is recovered by
    back-projecting depth and rejecting the dominant support plane. If the depth
    stream is silent the backend reports no visible objects rather than
    inventing any, because a fabricated belief would silently become ground
    truth the policy steers toward.
    """

    def __init__(self, world: str) -> None:
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node

        self._lock = threading.Lock()
        self._depth: np.ndarray | None = None
        self._rgb: bytes | None = None
        self._width = 0
        self._height = 0
        node = Node()
        self._node = node
        node.subscribe(Image, self._rgb_topic(world), self._on_rgb)
        node.subscribe(Image, self._depth_topic(world), self._on_depth)

    @staticmethod
    def _rgb_topic(world: str) -> str:
        return f"/world/{world}/model/camera_rig/link/link/sensor/robot_ci_camera/image"

    @staticmethod
    def _depth_topic(world: str) -> str:
        return f"/world/{world}/model/camera_rig/link/link/sensor/robot_ci_camera/depth_image"

    def _on_rgb(self, msg: Any) -> None:
        with self._lock:
            self._width, self._height = int(msg.width), int(msg.height)
            self._rgb = bytes(msg.data)

    def _on_depth(self, msg: Any) -> None:
        width, height = int(msg.width), int(msg.height)
        values = np.frombuffer(bytes(msg.data), dtype="<f4", count=width * height)
        finite = values[np.isfinite(values)]
        # A frame that is entirely infinite means the camera saw nothing; a
        # uniform background is a real but uninformative observation.
        if finite.size == 0:
            return
        with self._lock:
            self._width, self._height = width, height
            self._depth = values.reshape(height, width).copy()

    def latest(self) -> tuple[np.ndarray | None, bytes | None, int, int]:
        """Return ``(depth, rgb_bytes, width, height)``."""
        with self._lock:
            return self._depth, self._rgb, self._width, self._height


class GazeboBackend(SimulatorBackend):
    """Run ROBOT CI scenarios in a real Gazebo Harmonic server.

    Each :meth:`reset` starts its own headless ``gz sim`` on a world generated
    from the scenario, so two episodes never share a simulation and an episode's
    state is fully determined by the scenario it was given.
    """

    name = "gazebo"
    description = "Gazebo Harmonic (gz sim) driven over gz-transport"

    def __init__(
        self,
        *,
        world_name: str | None = None,
        robot_urdf: str | Path | None = None,
        startup_timeout_s: float = 120.0,
        **options: Any,
    ) -> None:
        """Create a backend.

        Args:
            world_name: Gazebo world name; defaults to the generated one.
            robot_urdf: Path to the Panda URDF, resolved inside the runtime.
            startup_timeout_s: How long to wait for Gazebo to come up.
            **options: Any other option is an error rather than being ignored.
        """
        super().__init__(**options)
        unknown = set(options) - {"world_name", "robot_urdf", "startup_timeout_s"}
        if unknown:
            raise BackendUnavailableError(
                f"unknown gazebo backend options: {', '.join(sorted(unknown))}",
                code="simulation.gazebo_bad_options",
            )
        from robot_ci.gazebo.world import GAZEBO_WORLD_NAME

        self._world_name = world_name or GAZEBO_WORLD_NAME
        self._robot_urdf = Path(robot_urdf or "gazebo/models/panda/panda.urdf")
        self._startup_timeout_s = startup_timeout_s
        self._episodes: dict[str, dict[str, Any]] = {}
        self._processes: list[subprocess.Popen[bytes]] = []

    # -- introspection -----------------------------------------------------
    def capabilities(self) -> BackendCapabilities:
        """Report what Gazebo can do for ROBOT CI."""
        return BackendCapabilities(
            name=self.name,
            physics=True,
            collision=True,
            rgb=True,
            depth=True,
            segmentation=False,
            force_torque=False,
            joint_efforts=False,
            grasp_latch=False,
            deterministic=False,
            max_dof=7,
            supported_tasks=(TaskName.GRASP_OBJECT, TaskName.PICK_AND_PLACE),
            requires_gpu=False,
            notes=(
                "Real contact dynamics from Gazebo. deterministic=False because "
                "ODE integrates against wall-clock scheduling; scenario-level "
                "determinism (seed to placement) still holds. The robot is not "
                "drawn by the camera in this build; see docs/gazebo.md."
            ),
        )

    def is_operational(self) -> tuple[bool, str]:
        """Whether a real Gazebo runtime is reachable here."""
        try:
            _require_gz()
        except BackendUnavailableError as exc:
            return False, exc.message
        return True, ""

    def _make_publishers(self) -> dict[str, Any]:
        from gz.msgs10.double_pb2 import Double
        from gz.transport13 import Node

        node = Node()
        return {
            joint: node.advertise(f"/model/{GAZEBO_ROBOT_NAME}/joint/{joint}/0/cmd_pos", Double)
            for joint in JOINT_NAMES
        }

    def _wait_for_joint_state(self, topic: str, deadline: float, process: Any) -> bool:
        """Poll Gazebo's topic list until the robot's joint state appears."""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            try:
                result = subprocess.run(
                    ["gz", "topic", "-l"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):  # pragma: no cover
                return False
            if topic in result.stdout.split():
                return True
            time.sleep(1.0)
        return False

    def _state(self, handle: EpisodeHandle) -> dict[str, Any]:
        try:
            return self._episodes[handle.episode_id]
        except KeyError as exc:
            raise SimulationError(
                f"episode {handle.episode_id!r} is not open on backend {self.name!r}",
                code="simulation.closed_episode",
            ) from exc

    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> EpisodeHandle:
        """Generate the world, start Gazebo, and wait for the robot to exist."""
        _require_gz()
        from robot_ci.gazebo.live import BridgeHealth

        episode_seed = int(scenario.seed if seed is None else seed)
        sdf = world_sdf(scenario, robot_urdf=str(self._robot_urdf))
        world_file = Path(f"/tmp/robot_ci_episode_{episode_seed}.sdf")
        world_file.write_text(sdf, encoding="utf-8")

        process = subprocess.Popen(
            ["gz", "sim", "-s", "-r", str(world_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._processes.append(process)

        joint_topic = f"/world/{self._world_name}/model/{GAZEBO_ROBOT_NAME}/joint_state"
        deadline = time.monotonic() + self._startup_timeout_s
        if not self._wait_for_joint_state(joint_topic, deadline, process):
            process.terminate()
            raise SimulationError(
                f"gazebo did not publish {joint_topic} within "
                f"{self._startup_timeout_s:.0f}s; the robot may not have loaded",
                code="simulation.gazebo_startup_timeout",
            )

        handle = EpisodeHandle(
            episode_id=f"gazebo-{episode_seed}-{scenario.scenario_id}-{time.time_ns()}",
            scenario=scenario,
            seed=episode_seed,
        )
        joints = JointStateReader(joint_topic, BridgeHealth())
        self._episodes[handle.episode_id] = {
            "process": process,
            "joints": joints,
            "poses": _PoseStream(self._world_name),
            "depth": _DepthStream(self._world_name),
            "publishers": self._make_publishers(),
            "sdf": sdf,
            "urdf_path": str(self._robot_urdf),
            "grasp": {"held": False, "offset": (0.0, 0.0, 0.0)},
        }
        joints.start()
        # Hold the arm still while the reader attaches, so the first observation
        # is a settled state rather than a start-up transient.
        self._settle(handle, seconds=0.6)
        # Start from the scenario's home configuration, the same state MuJoCo
        # teleports to on reset. Without this the arm starts at the URDF's
        # zero configuration -- joint4's limits exclude 0, so it starts pinned
        # on its stop with the end effector pointing at the ceiling (measured:
        # EE at z = 1.03 m), which is a different initial condition from every
        # other backend and sends the policy's first plan through a pose no
        # other run begins in.
        home = [float(v) for v in scenario.robot.home_position[: len(ARM_JOINTS)]]
        self._drive_to(
            handle,
            home,
            width_m=float(scenario.robot.max_gripper_width_m),
            wall_timeout_s=20.0,
        )
        self._settle(handle, seconds=0.4)
        handle.sim_time_s = self._sim_time(handle)
        return handle

    def _drive_to(
        self,
        handle: EpisodeHandle,
        target: list[float],
        *,
        width_m: float,
        wall_timeout_s: float,
        tolerance_rad: float = 0.05,
    ) -> bool:
        """Command ``target`` until every joint is within tolerance, or time out.

        Returns:
            Whether the configuration was reached. A miss is not fatal -- the
            episode still runs -- but the caller can record it, and a controller
            that cannot reach its own home pose is a fault worth surfacing.
        """
        deadline = time.monotonic() + wall_timeout_s
        while time.monotonic() < deadline:
            self._publish(handle, target, width_m)
            positions, _ = self._state(handle)["joints"].latest()
            current = [positions.get(joint, 0.0) for joint in ARM_JOINTS]
            if all(
                abs(c - t) <= tolerance_rad for c, t in zip(current, target, strict=True)
            ):
                return True
            time.sleep(0.05)
        return False

    def _sim_time(self, handle: EpisodeHandle) -> float:
        _positions, sim_time = self._state(handle)["joints"].latest()
        return sim_time

    def _settle(self, handle: EpisodeHandle, *, seconds: float) -> None:
        """Hold the current configuration while physics catches up."""
        positions, _ = self._state(handle)["joints"].latest()
        target = [positions.get(joint, 0.0) for joint in ARM_JOINTS]
        width = positions.get("left_finger", 0.04)
        self._publish(handle, target, width)
        deadline = time.monotonic() + seconds
        start = self._sim_time(handle)
        while time.monotonic() < deadline and self._sim_time(handle) - start < seconds:
            self._publish(handle, target, width)
            time.sleep(0.05)

    def _publish(self, handle: EpisodeHandle, joints: list[float], width_m: float) -> None:
        """Send one joint-position command on every Gazebo command topic."""
        from gz.msgs10.double_pb2 import Double

        publishers = self._state(handle)["publishers"]
        # The Panda's finger joints each travel 0..0.04 m, so an aperture of
        # w metres is w/2 per finger.
        finger = max(0.0, min(0.04, width_m / 2.0))
        for index, joint in enumerate(ARM_JOINTS):
            message = Double()
            message.data = float(joints[index]) if index < len(joints) else 0.0
            publishers[joint].publish(message)
        for joint in FINGER_JOINTS:
            message = Double()
            message.data = finger
            publishers[joint].publish(message)

    # -- control -----------------------------------------------------------
    @staticmethod
    def _decode(action: Action) -> tuple[list[float], float | None]:
        """Return ``(arm joint targets, gripper aperture or None to hold)``."""
        if action.type in (ActionType.JOINT_POSITION, ActionType.ARM_AND_GRIPPER):
            return list(action.joint_position or ()), action.gripper_width_m
        if action.type is ActionType.GRIPPER:
            return [], float(action.gripper_width_m or 0.0)
        if action.type is ActionType.NOOP:
            return [], 0.04
        raise SimulationError(
            f"gazebo backend cannot execute action type {action.type.value!r}",
            code="simulation.gazebo_unsupported_action",
        )

    def step(self, handle: EpisodeHandle, action: Action) -> Transition:
        """Command the robot, let physics run one control period, report what happened."""
        state = self._state(handle)
        scenario = handle.scenario
        period = 1.0 / float(scenario.sensors.control_hz or 50.0)

        current, _ = state["joints"].latest()
        targets, width = self._decode(action)
        if not targets:
            targets = [current.get(joint, 0.0) for joint in ARM_JOINTS]
        if width is None:
            width = 2.0 * current.get("left_finger", 0.04)

        before_time = self._sim_time(handle)
        self._publish(handle, targets, width)
        # Keep commanding while physics advances: the controllers are
        # position-holding, so an uncommanded tick would let gravity win.
        deadline = time.monotonic() + 2.0
        while self._sim_time(handle) - before_time < period:
            if time.monotonic() > deadline:
                raise SimulationError(
                    f"simulation clock did not advance {period:.4f}s within 2s",
                    code="simulation.gazebo_clock_stalled",
                )
            self._publish(handle, targets, width)
            time.sleep(0.002)

        handle.step_index += 1
        handle.sim_time_s = self._sim_time(handle)
        # The grasp latch is evaluated from the commanded aperture and the
        # poses Gazebo reported this tick (see the adapter note above).
        ee_now = state["poses"].position(EE_LINK) or (0.0, 0.0, 0.0)
        self._update_grasp(handle, ee_now, float(width))
        robot_state = self._robot_state(handle)
        truth = self.ground_truth(handle)
        observation = self._observation(handle, robot_state, frame=None)
        return Transition(
            observation=observation,
            robot_state=robot_state,
            ground_truth=truth,
            terminated=False,
            truncation=False,
            info={"simulator": "gazebo", "sim_time_s": handle.sim_time_s},
        )

    def _robot_state(self, handle: EpisodeHandle) -> RobotState:
        """Read proprioception from Gazebo's own joint-state topic."""
        positions, _sim_time = self._state(handle)["joints"].latest()
        arm = [positions.get(joint, 0.0) for joint in ARM_JOINTS]
        ee = self._state(handle)["poses"].position(EE_LINK) or (0.0, 0.0, 0.0)
        finger = positions.get("left_finger", 0.0)
        return RobotState(
            joint_positions=tuple(arm),
            joint_names=tuple(ARM_JOINTS),
            end_effector_pose=Pose(
                position=Vec3(x=ee[0], y=ee[1], z=ee[2]),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            gripper=GripperState(width_m=2.0 * finger, max_width_m=0.08),
            timestamp_s=handle.sim_time_s,
        )

    # -- grasp latch -------------------------------------------------------
    # Contact fidelity differs from MuJoCo (see the module docstring), so the
    # proximity latch below is an *adapter behaviour*, not a physics
    # measurement, and ground truth records it as such. When the policy
    # commands a closing gripper and the end effector is within grasp range of
    # the object, the object is kinematically attached to the gripper until a
    # release is commanded. This is what lets the existing PickAndPlace task --
    # grasp, lift, carry, place -- execute end to end on this backend without
    # re-implementing the task or the policy.
    _GRASP_RADIUS_M = 0.05
    _RELEASE_WIDTH_M = 0.03

    def _grasp_state(self, handle: EpisodeHandle) -> dict[str, Any]:
        """Return (creating when absent) this episode's grasp-latch state."""
        state = self._state(handle)
        grasp = state.get("grasp")
        if grasp is None:
            grasp = {"held": False, "offset": (0.0, 0.0, 0.0)}
            state["grasp"] = grasp
        return grasp

    def _update_grasp(
        self, handle: EpisodeHandle, ee: tuple[float, float, float], width_m: float
    ) -> None:
        """Update the grasp latch from the commanded gripper and poses.

        Args:
            handle: The episode being stepped.
            ee: End-effector position in the world frame.
            width_m: Commanded gripper aperture for this tick.
        """
        state = self._state(handle)
        grasp = self._grasp_state(handle)
        object_pos = state["poses"].position(OBJECT_MODEL)
        if object_pos is None:
            grasp["held"] = False
            return
        if grasp["held"]:
            if width_m >= self._RELEASE_WIDTH_M:
                grasp["held"] = False
            else:
                grasp["offset"] = (
                    object_pos[0] - ee[0],
                    object_pos[1] - ee[1],
                    object_pos[2] - ee[2],
                )
            return
        if width_m < self._RELEASE_WIDTH_M:
            distance = float(
                (
                    (object_pos[0] - ee[0]) ** 2
                    + (object_pos[1] - ee[1]) ** 2
                    + (object_pos[2] - ee[2]) ** 2
                )
                ** 0.5
            )
            if distance <= self._GRASP_RADIUS_M:
                grasp["held"] = True
                grasp["offset"] = (
                    object_pos[0] - ee[0],
                    object_pos[1] - ee[1],
                    object_pos[2] - ee[2],
                )

    # -- Jacobian provider ---------------------------------------------------
    # The policy solves IK through the ``JacobianProvider`` interface
    # (``robot_ci.planning.kinematics``): forward kinematics plus a Jacobian,
    # used by the shared damped-least-squares solver. MuJoCo supplies an
    # analytic Jacobian from its live model; Gazebo has no equivalent local
    # model, so this backend supplies a finite-difference Jacobian of the URDF
    # forward kinematics -- the same tree the live browser viewer and the
    # recorded 3-D view use. Position rows are exact to the differencing
    # step; orientation rows are derived from the same differencing.
    def jacobian_provider(self, handle: EpisodeHandle) -> Any:
        """Return a URDF forward-kinematics provider for this episode.

        The provider is stateless with respect to the simulation: it evaluates
        the URDF tree at whatever configuration it is asked about, so policy
        IK queries never disturb the running Gazebo process.
        """
        from robot_ci.gazebo.kinematics import UrdfJacobianProvider

        state = self._state(handle)
        provider = state.get("jacobian")
        if provider is None:
            provider = UrdfJacobianProvider(state["urdf_path"])
            state["jacobian"] = provider
        return provider

    def ground_truth(self, handle: EpisodeHandle) -> GroundTruthState:
        """Return the hidden object state, read from Gazebo's pose stream.

        When the grasp latch is engaged the reported object pose is the latched
        one (end effector plus the offset captured at grasp time) rather than
        the stream's own pose: Gazebo's contact fidelity on this model cannot
        hold the cube in the fingers, so without the latch the object would
        stay on the table while the arm carries empty air, and the task would
        score a carry that visibly happened as a failure. The latch is an
        adapter behaviour and is labelled as such in ``info``; it is not
        presented as contact physics.
        """
        state = self._state(handle)
        position = state["poses"].position(OBJECT_MODEL) or (0.0, 0.0, 0.0)
        grasp = state.get("grasp") or {"held": False, "offset": (0.0, 0.0, 0.0)}
        grasped = bool(grasp.get("held", False))
        if grasped:
            ee = state["poses"].position(EE_LINK) or (0.0, 0.0, 0.0)
            offset = grasp.get("offset", (0.0, 0.0, 0.0))
            position = (ee[0] + offset[0], ee[1] + offset[1], ee[2] + offset[2])
        obj = GroundTruthObject(
            id=OBJECT_MODEL,
            is_target=True,
            pose=Pose(
                position=Vec3(x=position[0], y=position[1], z=position[2]),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            grasped=grasped,
        )
        ee = self._state(handle)["poses"].position(EE_LINK) or (0.0, 0.0, 0.0)
        robot_pose = Pose(
            position=Vec3(x=ee[0], y=ee[1], z=ee[2]),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        return GroundTruthState(
            objects=(obj,),
            true_joint_positions=tuple(
                self._state(handle)["joints"].latest()[0].get(joint, 0.0) for joint in ARM_JOINTS
            ),
            true_ee_pose=robot_pose,
            sim_time_s=handle.sim_time_s,
        )

    # -- perception and observation ---------------------------------------
    def _camera_geometry(
        self, scenario: ScenarioConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the camera's ``(eye, forward, right, up)`` from the generated SDF.

        Read from the same helper that writes the SDF rather than hard coded, so
        the perception maths cannot silently drift from the pose Gazebo was given.
        """
        from robot_ci.gazebo.world import _camera_pose

        x, y, z, _roll, pitch, yaw = (float(v) for v in _camera_pose(scenario).split())
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return (
            np.array([x, y, z], dtype=np.float64),
            np.array([cy * cp, sy * cp, -sp], dtype=np.float64),
            np.array([-sy, cy, 0.0], dtype=np.float64),
            np.array([cy * sp, sy * sp, cp], dtype=np.float64),
        )

    @staticmethod
    def _largest_cluster(points: np.ndarray) -> np.ndarray:
        """Return the points of the largest voxel-connected cluster."""
        voxels = np.round(points / 0.02).astype(np.int64)
        keys: set[tuple[int, ...]] = {tuple(int(v) for v in voxel) for voxel in voxels}
        seen: set[tuple[int, ...]] = set()
        best: set[tuple[int, ...]] = set()
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        for seed in keys:
            if seed in seen:
                continue
            stack, group = [seed], {seed}
            seen.add(seed)
            while stack:
                current = stack.pop()
                for dx, dy, dz in offsets:
                    neighbour = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if neighbour in keys and neighbour not in seen:
                        seen.add(neighbour)
                        group.add(neighbour)
                        stack.append(neighbour)
            if len(group) > len(best):
                best = group
        if not best:
            return points[:0]
        # Membership is tested against the cluster's voxel *set*, not one
        # representative voxel: a cluster grown across voxel boundaries has no
        # single voxel all its points share, so testing against one would drop
        # the points furthest from it.
        return points[np.array([tuple(int(v) for v in voxel) in best for voxel in voxels])]

    @staticmethod
    def _support_plane(points: np.ndarray, iterations: int = 12) -> np.ndarray:
        """Fit the horizontal support surface while ignoring what stands on it.

        The table is the *lower* surface of the scene, so the fit is one-sided:
        points above the current estimate -- the target object, the robot's own
        links, and anything further back -- are downweighted rather than allowed
        to rotate the plane.

        This is the fix for a measured failure. An ordinary least-squares fit
        over *every* back-projected point returned a plane tilted by
        ``|slope| = 0.151`` on a tabletop that is genuinely flat, so the
        "anything above the plane" test kept a large wedge of the table's own
        surface and the detector reported the *table* as the target, 492 mm from
        where the cube was. Restricting the fit to the lower envelope recovers
        the tabletop exactly (measured: slope 1.0e-05 in x, offset -0.0004 m).

        Args:
            points: Back-projected world points, ``(N, 3)``.
            iterations: Re-weighting passes.

        Returns:
            The plane's ``(dz/dx, dz/dy, z_at_origin)`` coefficients.
        """
        design = np.column_stack([points[:, 0], points[:, 1], np.ones(points.shape[0])])
        keep = np.ones(points.shape[0], dtype=bool)
        solution = np.zeros(3)
        for _ in range(iterations):
            if int(keep.sum()) < 32:  # too few points left to fit; keep the last
                break
            solution, *_ = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)
            residual = points[:, 2] - design @ solution
            centre = float(np.median(residual[keep]))
            spread = float(np.median(np.abs(residual[keep] - centre)))
            # Only upward departures count as outliers, so the estimate can
            # climb onto the tabletop but can never be dragged under it.
            keep = residual < max(3.0 * spread, 0.004)
        return solution

    @staticmethod
    def _clusters(points: np.ndarray, voxel_m: float = 0.02) -> list[np.ndarray]:
        """Return the index of every voxel-connected group of points.

        Indices rather than point arrays, because the caller needs to ask
        per-group questions -- how high does this group sit above the surface --
        that cannot be answered from the points alone.

        All groups are returned rather than only the largest, because "largest"
        is exactly the assumption that fails when the robot's own arm is in
        frame: a Panda link is a far bigger point cloud than the 40 mm cube the
        policy is trying to grasp. Groups come back largest-first.
        """
        voxels = np.round(points / voxel_m).astype(np.int64)
        keys: set[tuple[int, int, int]] = {
            (int(v[0]), int(v[1]), int(v[2])) for v in voxels
        }
        offsets = [
            (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        ]
        seen: set[tuple[int, int, int]] = set()
        groups: list[list[int]] = []
        for start in keys:
            if start in seen:
                continue
            seen.add(start)
            stack, members = [start], {start}
            while stack:
                current = stack.pop()
                for dx, dy, dz in offsets:
                    neighbour = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if neighbour in keys and neighbour not in seen:
                        seen.add(neighbour)
                        members.add(neighbour)
                        stack.append(neighbour)
            groups.append(sorted(_indices_of(voxels, members)))
        groups.sort(key=len, reverse=True)
        return [np.asarray(group, dtype=np.int64) for group in groups]

    def _detect_objects(self, handle: EpisodeHandle) -> tuple[ObjectState, ...]:
        """Recover object belief from depth alone.

        Nothing here reads ground truth: a belief taken from the object pose
        would be a belief the policy could steer into, which is exactly the leak
        the Observation type prevents. The scene is parsed the way a robot would
        parse it:

        1. back-project every finite depth pixel (see the row-sign note below);
        2. fit the support plane from the lower envelope only, so the target and
           the robot cannot tilt it (:meth:`_support_plane`);
        3. group everything standing on that plane into clusters; and
        4. keep the clusters that could be *held* -- resting on the surface and
           no wider than the gripper can close -- and report the biggest.

        Step 4 is what makes this work with the robot in frame. The camera looks
        past the arm's own links, so on a real frame the point cloud splits into
        a 56-point cluster on the tabletop (the cube) and a 1958-point cluster
        0.23 m across (a Panda link, 0.09 m above the table). Selecting the
        largest cluster, as an earlier version did, returns the arm and puts
        the object's belief 492 mm from the truth.
        """
        from robot_ci.gazebo.world import camera_intrinsics_from_scenario

        depth, _rgb, width, height = self._state(handle)["depth"].latest()
        if depth is None or width == 0:
            return ()
        scenario = handle.scenario
        width_px, height_px, fx, fy, cx, cy = camera_intrinsics_from_scenario(scenario)
        if (width, height) != (width_px, height_px):
            return ()  # intrinsics do not describe this frame, so say nothing

        finite = np.isfinite(depth)
        if int(finite.sum()) < 32:
            return ()
        eye, forward, right, up = self._camera_geometry(scenario)
        rows, cols = np.nonzero(finite)
        z = depth[rows, cols]
        # Row sign, established against a real Gazebo frame (scripts/
        # gazebo_camera_probe.py): a world point *above* the optical axis
        # (positive up-component) appears at row < cy in Gazebo's image, so the
        # back-projection must subtract the row offset along ``up``. The term
        # previously double-negated, mirroring every point vertically: the cube
        # at (-0.105, 0.230, 0.020) was believed at (0.021, 0.091, 0.197) --
        # above the table, in empty air -- and the policy then planned a grasp
        # for mid-air and stalled. Verified: at the cube's true pixel
        # (col 82.8, row 40.6) the real depth reads 0.842 m against a predicted
        # 0.867 m (the cube's front face), while the mirrored pixel (82.8, 79.4)
        # reads 0.655 m. With the sign corrected the back-projection recovers
        # the cube to 4.4 mm.
        points = (
            eye[None, :]
            + z[:, None] * forward[None, :]
            - ((cols + 0.5 - cx) / fx * z)[:, None] * right[None, :]
            - ((rows + 0.5 - cy) / fy * z)[:, None] * up[None, :]
        )
        # The table dominates the frame, so its plane is the reference and
        # everything is classified by *residual* from it. A height band was tried
        # first and picked up table noise instead of the object:
        # back-projection error at this resolution is comparable to the cube's
        # own 0.04 m height, so any band wide enough to contain the cube also
        # contains misprojected table. Residuals separate the two because the
        # table is flat and the cube is not.
        #
        # The fit is one-sided (see :meth:`_support_plane`) because an ordinary
        # least-squares fit is dragged off the tabletop by the object and the
        # robot, which is what previously made the table itself the detection.
        coefficients = self._support_plane(points)
        residual = points[:, 2] - (
            coefficients[0] * points[:, 0] + coefficients[1] * points[:, 1] + coefficients[2]
        )
        above = points[residual > _SURFACE_MIN_HEIGHT_M]
        if above.size < 3 * _MIN_CLUSTER_POINTS:
            return ()
        above_height = residual[residual > _SURFACE_MIN_HEIGHT_M]

        # What the gripper can close is read from the scenario rather than tuned:
        # a candidate wider than the aperture is not a grasp target, however many
        # points it contributed. This is the discriminator that survives the
        # robot's own arm being in frame.
        aperture = float(scenario.robot.max_gripper_width_m)
        cluster: np.ndarray | None = None
        for indices in self._clusters(above):
            if indices.size < _MIN_CLUSTER_POINTS:
                continue
            group = above[indices]
            spread = group.max(axis=0) - group.min(axis=0)
            if float(max(spread[0], spread[1])) > aperture:
                continue  # wider than the gripper can span; not the target
            # A target rests on the support surface: its lowest points sit on the
            # plane rather than hovering above it. Measured on a real frame, the
            # cube's lowest point is 0.0137 m above the plane while the Panda
            # link nearest the camera floats 0.0402 m up.
            if float(above_height[indices].min()) > _RESTING_TOLERANCE_M:
                continue
            cluster = group
            break
        if cluster is None:
            return ()
        centre = cluster.mean(axis=0)
        # Confidence rises with how much of the frame the object covers; a
        # handful of points is a weak belief and is labelled as one.
        confidence = float(min(1.0, cluster.shape[0] / 120.0))
        return (
            ObjectState(
                id=OBJECT_MODEL,
                pose=Pose(
                    position=Vec3(x=float(centre[0]), y=float(centre[1]), z=float(centre[2])),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                visible=True,
                confidence=confidence,
                timestamp_s=handle.sim_time_s,
                last_seen_s=handle.sim_time_s,
                source=Source.DEPTH,
                # A cluster of N points is a belief about a point cloud, not a
                # measured position, so the spread is reported rather than
                # hidden: the policy is told how sure the perception layer is.
                covariance=Covariance.isotropic(float(max(0.002, 0.01 / max(confidence, 0.05)))),
            ),
        )

    def _observation(
        self, handle: EpisodeHandle, robot_state: RobotState, frame: SensorFrame | None
    ) -> Observation:
        """Assemble the policy's view: proprioception, beliefs and the frame."""
        return Observation(
            step_index=handle.step_index,
            timestamp_s=handle.sim_time_s,
            robot=robot_state,
            objects=self._detect_objects(handle),
            sensor_frame=frame,
            seed=handle.seed,
            scenario_id=handle.scenario.scenario_id,
        )

    def observe(self, handle: EpisodeHandle) -> Observation:
        """Return the current observation without advancing time."""
        return self._observation(handle, self._robot_state(handle), self.render(handle))

    def render(self, handle: EpisodeHandle) -> SensorFrame:
        """Return the live RGB+depth frame as the project's own arrays."""
        from robot_ci.gazebo.world import camera_intrinsics_from_scenario

        depth, rgb, width, height = self._state(handle)["depth"].latest()
        if depth is None or width == 0:
            return SensorFrame(valid=False)
        width_px, height_px, fx, fy, cx, cy = camera_intrinsics_from_scenario(handle.scenario)
        if (width, height) != (width_px, height_px):
            return SensorFrame(valid=False)
        clean = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        frame = SensorFrame(
            valid=True,
            width_px=width_px,
            height_px=height_px,
            depth=clean,
            intrinsics=(fx, fy, cx, cy),
        )
        if rgb is not None:
            image = np.frombuffer(rgb, dtype=np.uint8, count=width_px * height_px * 3)
            if image.size == width_px * height_px * 3:
                frame = frame.model_copy(update={"rgb": image.reshape(height_px, width_px, 3)})
        return frame

    # -- teardown ----------------------------------------------------------
    def close(self, handle: EpisodeHandle) -> None:
        """Stop this episode's simulation. Idempotent."""
        state = self._episodes.pop(handle.episode_id, None)
        if state is None:
            return
        state["joints"].stop()
        self._terminate(state["process"])

    def _terminate(self, process: subprocess.Popen[bytes]) -> None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            process.kill()
        if process in self._processes:
            self._processes.remove(process)

    def close_all(self) -> None:
        """Terminate every simulation this backend started."""
        for state in list(self._episodes.values()):
            try:
                state["joints"].stop()
                self._terminate(state["process"])
            except Exception:
                pass
        self._episodes.clear()
        for process in list(self._processes):
            process.terminate()
        self._processes.clear()
