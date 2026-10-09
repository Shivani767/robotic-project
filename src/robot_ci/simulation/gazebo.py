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

from robot_ci.control.gripper import GripperController
from robot_ci.core.errors import BackendUnavailableError, SimulationError
from robot_ci.core.logging import get_logger
from robot_ci.gazebo.live import JOINT_NAMES, JointStateReader
from robot_ci.gazebo.world import (
    FINGER_ACTUATOR_STIFFNESS_N_PER_M,
    GAZEBO_ROBOT_NAME,
    world_sdf,
)
from robot_ci.models.environment import TaskName, object_half_extents
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
#: measured at 0.004 m on a real 160x120 frame, and below the object's own
#: half-height (0.008 m). Both bounds are needed now that the world draws the
#: object the scenario declares: the previous value of 0.012 m was calibrated
#: against a 40 mm cube the scenario never asked for, and it sits *above* the
#: 8 mm half-height of the real one, so the detector rejected the object it was
#: supposed to find.
_SURFACE_MIN_HEIGHT_M = 0.006

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


class _WorldFrameProvider:
    """Wraps a :class:`UrdfJacobianProvider` and reports poses in the world.

    The URDF tree is relative to the robot's own base, but everything the policy
    and the task speak in is the world frame: the waypoints are built from the
    object's world pose, ``Observation`` carries the tool's world pose, and
    MuJoCo's provider answers in the world (it reads ``site_xpos`` from the live
    model). Leaving the Gazebo provider in the base frame therefore made the
    policy solve for a world-frame target in the wrong frame, and the arm came
    to rest a fixed offset away: measured, exactly the 20 mm the base sits above
    the world origin, in z, while the IK reported a 0.26 mm residual.

    The base pose is read from the simulator on every call rather than cached,
    because a backend that assumes its robot never moves is the same mistake one
    layer down.
    """

    def __init__(self, provider: Any, base_transform: Any) -> None:
        self._provider = provider
        self._base_transform = base_transform

    def forward(self, q: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the tool-centre-point position and rotation in the world."""
        position, rotation = self._provider.forward(q)
        base_rotation, base_position = self._base_transform()
        return (
            base_position + base_rotation @ position,
            base_rotation @ rotation,
        )

    def jacobian(self, q: Any) -> np.ndarray:
        """Return the task-space Jacobian, unchanged by a rigid base offset."""
        return self._provider.jacobian(q)


def _quaternion_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Return the 3x3 rotation for a ``(w, x, y, z)`` quaternion.

    Written out here rather than pulled from a helper because it is the only
    place the backend converts an *orientation it did not compute*. The unit
    vector is normalised first: ``/world/<name>/pose/info`` publishes a
    quaternion and a non-unit one would silently scale the frame it is applied
    to.
    """
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class _PoseStream:
    """Latest world poses of every model and link, straight from Gazebo.

    ``/world/<name>/pose/info`` is the simulator's own statement of where things
    are. Reading it is what keeps every pose this backend reports honest: they
    are measured, not estimated and not written.

    Position *and* orientation are kept. An earlier version retained only the
    position and answered ``orientation`` with the identity quaternion, which is
    a fabricated reading: a tumbled object would have been reported as upright,
    and the task's orientation criterion would have been evaluated against a
    number the simulator never produced.
    """

    def __init__(self, world: str) -> None:
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node

        self._lock = threading.Lock()
        self._poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
        node = Node()
        node.subscribe(Pose_V, f"/world/{world}/pose/info", self._on_pose)
        self._node = node

    def _on_pose(self, msg: Any) -> None:
        snapshot: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
        for entity in msg.pose:
            if entity.name:
                orientation = entity.orientation
                snapshot[entity.name] = (
                    (
                        float(entity.position.x),
                        float(entity.position.y),
                        float(entity.position.z),
                    ),
                    (
                        float(orientation.w),
                        float(orientation.x),
                        float(orientation.y),
                        float(orientation.z),
                    ),
                )
        with self._lock:
            self._poses = snapshot

    def pose(
        self, name: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        """Return ``(position, orientation)`` for ``name``, or ``None``."""
        with self._lock:
            return self._poses.get(name)

    def position(self, name: str) -> tuple[float, float, float] | None:
        """Return the world position of ``name``, or ``None`` if unreported."""
        pose = self.pose(name)
        return None if pose is None else pose[0]


class _ContactStream:
    """Contact pairs touching a given collision, accumulated over a tick.

    This is the answer to "what is the gripper actually touching, and how
    hard". Every attempt to infer it from the object's velocity -- how fast it
    left the jaws, how far it was carried -- describes the consequence and not
    the cause, and the two gave different stories while the question was open.

    The reading is the **union** of every pair Gazebo published since the last
    :meth:`begin_tick`, not the single most recent message. A contact sensor
    publishes at its own rate (100 Hz) inside a control tick (50 ms), so a
    transient contact -- the finger<->cube shove, a pad grazing the table -- can
    appear in one message and be absent from the next. Replace-latest sampling
    reported that as "nothing is touching it", which is how a real, seconds-long
    contact read as never having happened. Accumulating over the tick reports
    the contact the physics engine actually produced; see the finger-diagnostic
    report for the measured evidence.
    """

    def __init__(self, world: str, entity: str, sensor: str) -> None:
        """Subscribe to one model's contact sensor.

        Args:
            world: Gazebo world name.
            entity: Model name, e.g. ``target_cube``.
            sensor: Sensor name **as written in the world**. It is passed in
                rather than derived from the model: the sensor is called
                ``object_contacts`` while its model is ``target_cube``, and
                deriving one from the other produced a topic nobody publishes.
                A subscriber to a topic that does not exist reports no contacts
                and looks perfectly healthy, which is the worst place for that to
                hide.
        """
        from gz.msgs10.contacts_pb2 import Contacts
        from gz.transport13 import Node

        self._lock = threading.Lock()
        self._pairs: dict[str, dict[str, float]] = {}
        #: The union of every pair seen since the last :meth:`begin_tick`. This
        #: is what :meth:`pairs` reports; ``_pairs`` is the most recent message
        #: only and is kept for :meth:`health`, which describes the live sensor
        #: rather than the tick window.
        self._tick_pairs: dict[str, dict[str, float]] = {}
        #: A callback that raises delivers nothing, and a subscriber that
        #: delivers nothing is indistinguishable from a sensor reporting no
        #: contacts. Counting messages and keeping the first error makes the
        #: difference visible instead of leaving it to be inferred.
        self.messages = 0
        self.errors: list[str] = []
        self.topic = f"/world/{world}/model/{entity}/link/link/sensor/{sensor}/contact"
        node = Node()
        node.subscribe(Contacts, self.topic, self._guarded)
        self._node = node

    def _guarded(self, msg: Any) -> None:
        """Dispatch a message, recording any failure rather than losing it."""
        try:
            self._on_contacts(msg)
        except Exception as exc:  # a broken stream must say so, not look empty
            with self._lock:
                self.messages += 1
                if len(self.errors) < 4:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            self.messages += 1

    def _on_contacts(self, msg: Any) -> None:
        snapshot: dict[str, dict[str, float]] = {}
        for contact in msg.contact:
            # The field names are collision1/collision2, singular. An earlier
            # version read collisions[0]/collisions[1], which raised inside this
            # callback, and the subscriber reported *no contacts at all* while
            # looking perfectly healthy -- the same failure mode as the joint
            # reader that read a flat layout and silently returned zeros.
            # Field names verified against the running system by introspecting
            # the descriptor: collision1, collision2, position, normal, depth,
            # wrench, world. There is no ``wrenches`` and no ``separation``;
            # reading either raised inside the callback and the stream reported
            # no contacts while looking healthy.
            key = f"{contact.collision1.name} <-> {contact.collision2.name}"
            # Only fields whose names were verified by introspecting the
            # descriptors of the message Gazebo actually sends: collision1,
            # collision2, depth, and world (a single Entity carrying a pose).
            # Everything else is reached defensively. Three earlier readings --
            # collisions[0], wrenches, and wrench.force.normal -- each raised
            # inside this callback, and a stream that raises reports no contacts
            # while looking perfectly healthy, which is why the errors are kept
            # and shipped alongside the data.
            # Pair identity only, and that is a deliberate scope limit rather
            # than a simplification: collision1.name and collision2.name are the
            # two fields verified against the running system (introspecting the
            # descriptor is how they were found). Every richer field --
            # separation, depth, the per-body wrench -- was read wrongly three
            # times, each wrong read raising inside the callback and costing the
            # whole stream, so "nothing is touching it" and "the subscriber is
            # broken" became indistinguishable. Answering "which two bodies are
            # in contact" is the question that was actually open; the wrench
            # magnitude is recorded as not-yet-read rather than guessed at.
            key = f"{contact.collision1.name} <-> {contact.collision2.name}"
            snapshot[key] = {"in_contact": True}
        with self._lock:
            self._pairs = snapshot
            # Union into the tick window rather than replacing it: a contact
            # that appeared in an earlier message this tick and has since
            # separated must still be reported, because the physics engine
            # produced it. Replace-latest is exactly what lost the transient
            # finger<->cube shove to sampling.
            self._tick_pairs.update(snapshot)

    def begin_tick(self) -> None:
        """Open a fresh accumulation window for the next control tick.

        Called once per :meth:`GazeboBackend.step`, before physics advances, so
        :meth:`pairs` reports only what happened during that tick and not a
        stale contact from the one before. A contact sustained across ticks is
        re-published by the sensor every tick, so it reappears in each window;
        what the window preserves is the *transient* contact that replace-latest
        sampling dropped.
        """
        with self._lock:
            self._tick_pairs = {}

    def pairs(self) -> dict[str, dict[str, float]]:
        """Return the contact pairs accumulated since the last ``begin_tick``."""
        with self._lock:
            return dict(self._tick_pairs)

    def health(self) -> dict[str, Any]:
        """Report whether this stream is actually working.

        ``messages == 0`` means the sensor has not been heard from at all,
        which is a fault; ``messages > 0`` with an empty ``pairs`` means the
        sensor is live and nothing is touching it. Those are opposite
        conclusions and must not read the same.
        """
        with self._lock:
            return {
                "topic": self.topic,
                "messages": self.messages,
                "pairs": len(self._pairs),
                "errors": list(self.errors),
                "live": self.messages > 0 and not self.errors,
            }


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
            "gripper_controller": self._make_gripper_controller(
                scenario, period=1.0 / float(scenario.sensors.control_hz or 50.0)
            ),
            "poses": _PoseStream(self._world_name),
            "object_contacts": _ContactStream(
                self._world_name, OBJECT_MODEL, "object_contacts"
            ),
            "table_contacts": _ContactStream(self._world_name, "table", "table_contacts"),
            "depth": _DepthStream(self._world_name),
            "publishers": self._make_publishers(),
            "sdf": sdf,
            "urdf_path": str(self._robot_urdf),
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

    def _make_gripper_controller(
        self, scenario: ScenarioConfig, *, period: float
    ) -> GripperController:
        """Build the gripper controller for one episode.

        The stiffness handed in is the finger actuator's real one -- a Gazebo
        joint position controller produces ``p_gain`` newtons per metre of
        position error, and ``p_gain`` is what the generated world configures.
        The force bound in :mod:`robot_ci.control.gripper` is derived from that
        stiffness, so passing anything else would bound the force against the
        wrong actuator.
        """
        robot = scenario.robot
        return GripperController(
            mode=robot.gripper_control_mode,
            stiffness_n_per_m=FINGER_ACTUATOR_STIFFNESS_N_PER_M,
            control_dt_s=float(period),
            open_width_m=float(robot.max_gripper_width_m),
            closed_width_m=float(robot.gripper_closed_separation_m),
        )

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
        # The aperture the fingers are *currently* at, so holding them means
        # holding where they are. Reading one finger and passing it as an
        # aperture halved the commanded opening and let the jaws drift closed
        # while the arm was still settling at reset.
        width = float(positions.get("left_finger", 0.0)) + float(
            positions.get("right_finger", 0.0)
        )
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

        # The aperture goes through the same controller MuJoCo uses, so the two
        # backends share one gripper policy. In the default POSITION mode it is
        # a pass-through and changes nothing numerically; what it buys is that a
        # scenario asking for GripperControlMode.BOUNDED gets the force bound
        # here too, derived from the actuator stiffness that is actually
        # driving the fingers rather than from a number typed in here.
        controller: GripperController = state["gripper_controller"]
        measured_width = float(current.get("left_finger", 0.0)) + float(
            current.get("right_finger", 0.0)
        )
        width = controller.step(requested_width_m=float(width), measured_width_m=measured_width)

        before_time = self._sim_time(handle)
        # Open a fresh contact window before physics advances, so the pairs
        # reported in ``info`` below cover exactly this tick and nothing stale.
        # Accumulating here (rather than replace-latest) is what makes a
        # transient contact -- the finger<->cube shove, a pad grazing the table
        # for one message -- survive into the record instead of vanishing between
        # the sensor's publish and this tick's read.
        state["object_contacts"].begin_tick()
        state["table_contacts"].begin_tick()
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
            info={
                "simulator": "gazebo",
                "sim_time_s": handle.sim_time_s,
                # Contact pairs the simulator itself reports, for the object and
                # for the support surface. Published on the transition rather
                # than the observation because they are simulator facts, and the
                # observation is what the policy is allowed to see.
                "object_contacts": state["object_contacts"].pairs(),
                "table_contacts": state["table_contacts"].pairs(),
                # The sensor's own health travels with the data. An empty
                # contact set means "nothing is touching it"; a sensor with zero
                # messages or a recorded error means the stream is broken, and
                # those two must never look the same in an artifact.
                "contact_health": {
                    "object": state["object_contacts"].health(),
                    "table": state["table_contacts"].health(),
                },
            },
        )

    def _base_transform(self, handle: EpisodeHandle) -> tuple[np.ndarray, np.ndarray]:
        """Return the robot base's ``(rotation, translation)`` in the world.

        Read from the simulator rather than assumed. The URDF's kinematics are
        relative to the model frame, and the model sits on a table rather than
        at the world origin: raising the base so it rests on the tabletop (see
        ``robot_ci.gazebo.world``) moves every link, and a tool pose that ignored
        that would be 20 mm off in z for the whole episode.

        Before the first pose message arrives this returns the identity rather
        than the configured mount height, because the identity is the one value
        whose error cannot be mistaken for a measurement: it is simply "not yet
        known", and it disappears the moment Gazebo reports the base.
        """
        measured = self._state(handle)["poses"].pose(GAZEBO_ROBOT_NAME)
        if measured is None:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        position, orientation = measured
        return (
            _quaternion_matrix(*orientation),
            np.asarray(position, dtype=np.float64),
        )

    def _tcp(self, handle: EpisodeHandle, arm_positions: list[float]) -> Pose:
        """Return the tool-centre-point pose in the **world** frame.

        This is proprioceptive forward kinematics over the URDF that generated
        the world -- the same chain, the same tool frame, and the same frame the
        policy's IK solves against -- evaluated at the joint angles Gazebo
        measured.

        Deriving the pose this way is not a convenience. Reporting
        ``panda_link7`` instead put the observed end effector 0.2104 m behind
        the frame the policy was commanding, so the policy's own "have I reached
        the waypoint" test could never become true, the waypoint cursor never
        advanced, and the arm stalled.

        The alternative -- reading a link pose out of Gazebo's pose stream -- is
        not available for this frame: ``/world/<w>/pose/info`` carries link poses
        *relative to their model*, and there is no link at all for the space
        between the pads. Forward kinematics over measured joint angles is what
        a real controller does, and it is measurably consistent with the
        simulator (``scripts/gazebo_stage_check.py``, stage 4).

        Reading the pose through :meth:`jacobian_provider` rather than through
        the bare URDF provider is deliberate: it guarantees the pose the policy
        is *shown* and the pose its IK solves in are produced by exactly the
        same code, in exactly the same frame.
        """
        from robot_ci.planning.kinematics import matrix_to_pose

        provider = self.jacobian_provider(handle)
        position, rotation = provider.forward(np.asarray(arm_positions, dtype=np.float64))
        return matrix_to_pose(position, rotation)

    def _robot_state(self, handle: EpisodeHandle) -> RobotState:
        """Read proprioception from Gazebo's own joint-state topic."""
        positions, _sim_time = self._state(handle)["joints"].latest()
        arm = [positions.get(joint, 0.0) for joint in ARM_JOINTS]
        # The aperture is the measured finger travel, the same convention
        # ``simulation/mujoco.py`` uses: both official finger joints run from 0
        # (closed) to 0.04 (open), so the opening beyond the jaw's rest position
        # is their sum. It is the *measured* sum, not the commanded one, so a pad
        # blocked by the object reads narrower than the command -- which is
        # exactly the signal that says the jaws have closed onto something.
        measured_width = float(positions.get("left_finger", 0.0)) + float(
            positions.get("right_finger", 0.0)
        )
        return RobotState(
            joint_positions=tuple(arm),
            joint_names=tuple(ARM_JOINTS),
            end_effector_pose=self._tcp(handle, arm),
            gripper=GripperState(
                width_m=max(0.0, measured_width),
                max_width_m=float(handle.scenario.robot.max_gripper_width_m),
            ),
            timestamp_s=handle.sim_time_s,
        )

    # -- measured grasp -----------------------------------------------------
    # The backend never moves the object, and this section is where that is
    # enforced rather than promised. Every pose reported by :meth:`ground_truth`
    # is the pose Gazebo's physics engine put the object in. What is *derived*
    # here is one boolean -- is the gripper currently supporting the object --
    # and it is defined so that a gripper which closed on empty air cannot
    # satisfy it.
    #:
    #: Largest distance from the tool centre point to the object's centre for the
    #: jaws to count as being around it, in metres. The pads are 17 mm tall and
    #: the object is 16 mm, so a seated grasp leaves the object centre within a
    #: few millimetres of the pad centre, which itself sits 0.5 mm behind the TCP.
    _GRASP_RADIUS_M = 0.03
    #: How far past the object's own width the measured aperture may be before the
    #: jaws have demonstrably not closed onto it, in metres.
    _GRASP_APERTURE_TOLERANCE_M = 0.004
    #: How far above its resting height the object must have risen, in metres,
    #: before it counts as supported by the gripper rather than resting on the
    #: table. Well above the contact solver's own settling noise and far below
    #: the task's 50 mm lift requirement, so this can never decide the outcome.
    _LIFT_EPSILON_M = 0.002

    def _supported_by_gripper(
        self,
        scenario: ScenarioConfig,
        object_position: tuple[float, float, float],
        object_half_height: float,
        tcp: tuple[float, float, float],
        measured_width_m: float,
    ) -> bool:
        """Whether the gripper is *physically* holding the object.

        Three measured conditions, all of which a gripper that missed the object
        fails:

        * the jaws have closed to within the object's own width, so the pads can
          have come into contact with it instead of passing either side of it;
        * the object's centre lies inside the jaw envelope around the tool centre
          point, so the object is between the pads and not merely nearby;
        * the object is clear of the support surface, so it is being carried
          rather than resting on the table while the gripper hovers over it.

        The last condition is what makes this a statement about physics rather
        than about geometry. An object still sitting on the table has not been
        grasped, whatever the jaws are doing -- and there is no code path in this
        backend that writes the object's pose, so if the object rose, Gazebo's
        contact solver raised it.

        Returns:
            ``True`` only when all three hold.
        """
        object_width = 2.0 * float(scenario.target.size_m)
        if measured_width_m > object_width + self._GRASP_APERTURE_TOLERANCE_M:
            return False
        distance = float(np.linalg.norm(np.asarray(object_position) - np.asarray(tcp)))
        if distance > self._GRASP_RADIUS_M:
            return False
        resting_centre_z = float(scenario.environment.table_height_m) + object_half_height
        return float(object_position[2]) > resting_centre_z + self._LIFT_EPSILON_M

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
        the URDF tree at whatever configuration it is asked about, so policy IK
        queries never disturb the running Gazebo process. It reports poses in
        the **world** frame -- see :class:`_WorldFrameProvider` for why that is
        the frame the policy needs, and what it cost to get wrong.
        """
        from robot_ci.gazebo.kinematics import UrdfJacobianProvider

        state = self._state(handle)
        provider = state.get("jacobian")
        if provider is None:
            urdf = UrdfJacobianProvider(state["urdf_path"])
            provider = _WorldFrameProvider(
                urdf, lambda: self._base_transform(handle)
            )
            state["jacobian"] = provider
        return provider

    def ground_truth(self, handle: EpisodeHandle) -> GroundTruthState:
        """Return the hidden object state, *measured* by Gazebo.

        The position and orientation are the simulator's own, read from
        ``/world/<w>/pose/info``, which the physics engine writes. Nothing here
        moves the object, and nothing here reports a pose the simulation did not
        produce: an object that has not been grasped is reported on the table,
        where it is.

        ``grasped`` is derived by :meth:`_supported_by_gripper` from measured
        quantities -- the aperture actually reached, the object's measured
        position, and its measured height above the support surface. It is not
        set by the command and it is not latched.
        """
        state = self._state(handle)
        scenario = handle.scenario
        measured = state["poses"].pose(OBJECT_MODEL)
        robot_state = self._robot_state(handle)
        tcp = robot_state.end_effector_pose.position
        half_height = float(
            object_half_extents(scenario.target.category, scenario.target.size_m)[2]
        )
        if measured is None:
            # No pose has arrived yet. A default here would be a fabricated
            # measurement, so the object is reported at the world origin and
            # explicitly *not* grasped; the task then reads a large position
            # error, which is the honest description of "the simulator has not
            # said where the object is".
            position = (0.0, 0.0, 0.0)
            orientation = (1.0, 0.0, 0.0, 0.0)
            grasped = False
        else:
            position, orientation = measured
            grasped = self._supported_by_gripper(
                scenario,
                position,
                half_height,
                (tcp.x, tcp.y, tcp.z),
                robot_state.gripper.width_m,
            )
        obj = GroundTruthObject(
            id=OBJECT_MODEL,
            is_target=True,
            pose=Pose(
                position=Vec3(x=position[0], y=position[1], z=position[2]),
                orientation=Quaternion(
                    w=orientation[0], x=orientation[1], y=orientation[2], z=orientation[3]
                ),
            ),
            grasped=grasped,
        )
        return GroundTruthState(
            objects=(obj,),
            true_joint_positions=tuple(robot_state.joint_positions),
            true_ee_pose=robot_state.end_effector_pose,
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

    def _pose_beliefs(self, handle: EpisodeHandle) -> tuple[ObjectState, ...]:
        """Believe the target where the simulation measured it.

        This is the *same convention* ``simulation/mujoco.py`` uses: the object
        belief is the simulator's own body pose, published with an explicit
        covariance floor, ``source=Source.SIMULATOR`` and full visibility. It
        makes no perception claim, and the provenance field says so rather than
        implying a camera produced it.

        It exists because depth alone cannot drive this condition. The declared
        object is 16 mm across; at the generated camera pose (0.75 m away,
        ``fx=140``, 160x120) it subtends about 3 px, and the tabletop's own
        back-projection residual is 4 mm against an object 8 mm tall. Measured,
        the depth detector recovers the cube on some frames and not others, so an
        episode driven *only* by depth stalls whenever a frame misses it -- which
        is what the first live trace showed.

        Keeping the belief source identical across backends is also what makes a
        paired baseline/candidate comparison mean anything. MuJoCo hands the
        policy the simulator pose; a Gazebo comparison against a different
        quantity would be measuring the perception pipelines against each other,
        not the policies. The measured error of the depth path is recorded
        separately (:meth:`perceive_from_depth`) instead of being smuggled into
        the policy's input.

        Returns:
            One belief per scenario object, with the simulator's own pose.
        """
        state = self._state(handle)
        scenario = handle.scenario
        beliefs: list[ObjectState] = []
        for spec in scenario.objects:
            if spec.id != OBJECT_MODEL:
                continue
            measured = state["poses"].pose(OBJECT_MODEL)
            if measured is None:
                continue
            position, orientation = measured
            beliefs.append(
                ObjectState(
                    id=spec.id,
                    category=spec.category,
                    pose=Pose(
                        position=Vec3(x=position[0], y=position[1], z=position[2]),
                        orientation=Quaternion(
                            w=orientation[0],
                            x=orientation[1],
                            y=orientation[2],
                            z=orientation[3],
                        ),
                    ),
                    covariance=Covariance.isotropic(0.002, dimension=6),
                    confidence=1.0,
                    source=Source.SIMULATOR,
                    timestamp_s=handle.sim_time_s,
                    visible=True,
                    last_seen_s=handle.sim_time_s,
                    estimated_size_m=spec.size_m,
                )
            )
        return tuple(beliefs)

    def perceive_from_depth(self, handle: EpisodeHandle) -> tuple[ObjectState, ...]:
        """Recover object belief from the live depth image alone.

        A real perception capability, exposed and measured rather than used as
        the policy's input -- see :meth:`_pose_beliefs` for why the input is the
        simulator pose instead. ``Source.DEPTH`` on every returned belief records
        which channel produced it, so a caller can never mistake one for the
        other. ``scripts/gazebo_perception_evidence.py`` measures this path
        against ground truth on the live simulation and writes the result to
        ``experiments/results/gazebo/``.
        """
        return self._detect_objects(handle)

    def _object_beliefs(self, handle: EpisodeHandle) -> tuple[ObjectState, ...]:
        """Return the object beliefs the policy is given."""
        return self._pose_beliefs(handle)

    def _observation(
        self, handle: EpisodeHandle, robot_state: RobotState, frame: SensorFrame | None
    ) -> Observation:
        """Assemble the policy's view: proprioception, beliefs and the frame."""
        return Observation(
            step_index=handle.step_index,
            timestamp_s=handle.sim_time_s,
            robot=robot_state,
            objects=self._object_beliefs(handle),
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
