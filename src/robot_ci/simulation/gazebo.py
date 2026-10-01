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
        }
        joints.start()
        # Hold the arm still while the reader attaches, so the first observation
        # is a settled state rather than a start-up transient.
        self._settle(handle, seconds=0.6)
        handle.sim_time_s = self._sim_time(handle)
        return handle

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

    def ground_truth(self, handle: EpisodeHandle) -> GroundTruthState:
        """Return the hidden object state, read from Gazebo's pose stream."""
        position = self._state(handle)["poses"].position(OBJECT_MODEL) or (0.0, 0.0, 0.0)
        obj = GroundTruthObject(
            id=OBJECT_MODEL,
            is_target=True,
            pose=Pose(
                position=Vec3(x=position[0], y=position[1], z=position[2]),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
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
        best: list[tuple[int, ...]] = []
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        for seed in keys:
            if seed in seen:
                continue
            stack, group = [seed], []
            seen.add(seed)
            while stack:
                current = stack.pop()
                group.append(current)
                for dx, dy, dz in offsets:
                    neighbour = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if neighbour in keys and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            if len(group) > len(best):
                best = group
        chosen = np.asarray(best, dtype=np.int64)
        return points[np.array([bool(np.all(voxel == chosen, axis=1)) for voxel in voxels])]

    def _detect_objects(self, handle: EpisodeHandle) -> tuple[ObjectState, ...]:
        """Recover object belief from depth alone.

        The support plane is rejected by height, and the largest remaining
        cluster is reported as the object. Nothing here reads ground truth: a
        belief taken from the object pose would be a belief the policy could
        steer into, which is exactly the leak the Observation type prevents.
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
        points = (
            eye[None, :]
            + z[:, None] * forward[None, :]
            - ((cols + 0.5 - cx) / fx * z)[:, None] * right[None, :]
            - (-(rows + 0.5 - cy) / fy * z)[:, None] * up[None, :]
        )
        table_z = float(scenario.environment.table_height_m)
        above = points[points[:, 2] > table_z + 0.008]
        if above.size < 3 * 8:
            return ()
        cluster = self._largest_cluster(above)
        if cluster.shape[0] < 8:
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
