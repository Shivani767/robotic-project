"""The MuJoCo backend: the project's primary simulator.

This is the only module in the package that imports ``mujoco``. Everything else
reaches it only through :class:`~robot_ci.simulation.base.SimulatorBackend`, so
the research engine stays simulator-independent (and importable on a machine
where MuJoCo is not installed).

macOS note
----------
macOS has no EGL, so MuJoCo cannot open a GPU offscreen context by default.
:func:`_ensure_gl_backend` selects OSMesa software rasterization automatically,
which is why ``robot-ci`` works on a MacBook with no GPU configuration at all.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from robot_ci.core.errors import BackendUnavailableError, SimulationError
from robot_ci.core.logging import get_logger
from robot_ci.core.seeding import derive_seed
from robot_ci.core.version import SIMULATOR_VERSION
from robot_ci.models.environment import TaskName
from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import (
    Action,
    ActionType,
    ContactState,
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
    BackendInfo,
    EpisodeHandle,
    SimulatorBackend,
    TerminationReason,
    Transition,
)
from robot_ci.simulation.mujoco_env import build_scene_xml, scene_cache_key

__all__ = ["MuJoCoBackend"]

_LOG = get_logger(__name__)
_GL_CONFIGURED = False


def _ensure_gl_backend() -> str:
    """Choose a MuJoCo OpenGL backend that works headless on this platform.

    The valid values are platform-specific and MuJoCo raises at import time if
    it is given one it does not support, so this must run before the import:

    ==========  ==================  ==========================================
    Platform    Chosen backend     Notes
    ==========  ==================  ==========================================
    macOS       ``cgl``            Apple OpenGL. ``osmesa``/``egl`` are Linux
                                   only and are rejected by MuJoCo on macOS.
    Linux + GPU ``egl``            Hardware offscreen context.
    Linux + X11 ``glfw``           Falls back to an on-screen GL context.
    ==========  ==================  ==========================================

    An explicit ``MUJOCO_GL`` in the environment always wins, so power users can
    override this without patching code.

    Returns:
        The backend name now in effect.

    Raises:
        BackendUnavailableError: If the chosen backend is not supported here,
            with the list of values this platform does accept.
    """
    global _GL_CONFIGURED
    if _GL_CONFIGURED:
        return os.environ.get("MUJOCO_GL", "unset")

    if "MUJOCO_GL" not in os.environ or not os.environ["MUJOCO_GL"].strip():
        if sys.platform == "darwin":
            os.environ["MUJOCO_GL"] = "cgl"
        elif os.environ.get("DISPLAY"):
            os.environ["MUJOCO_GL"] = "glfw"
        else:
            os.environ["MUJOCO_GL"] = "egl"

    chosen = os.environ["MUJOCO_GL"]
    valid = {
        "darwin": ("cgl", "glfw"),
        "linux": ("egl", "osmesa", "glx", "glfw"),
    }.get(sys.platform, ("glfw",))
    if chosen not in valid:
        raise BackendUnavailableError(
            f"MUJOCO_GL={chosen!r} is not supported on {sys.platform}. "
            f"Valid values here: {', '.join(valid)}. Unset MUJOCO_GL to let "
            "ROBOT CI choose automatically.",
            code="simulation.gl_backend_unsupported",
            requested=chosen,
            valid=list(valid),
        )
    _GL_CONFIGURED = True
    _LOG.info("mujoco: OpenGL backend = %s (platform=%s)", chosen, sys.platform)
    return chosen


def _import_mujoco() -> Any:
    """Import ``mujoco`` with an actionable error when it is missing.

    Raises:
        BackendUnavailableError: If MuJoCo is not installed.
    """
    _ensure_gl_backend()
    try:
        import mujoco
    except ImportError as exc:
        raise BackendUnavailableError(
            "MuJoCo is not installed. Install it with "
            "`pip install mujoco` (or `pip install robot-ci[mujoco]`). "
            "Alternatively run with --backend mock or --backend synthetic, "
            "which need no physics engine.",
            code="simulation.mujoco_missing",
            hint="pip install mujoco",
        ) from exc
    return mujoco


class MuJoCoBackend(SimulatorBackend):
    """Rigid-body manipulation simulator backed by MuJoCo.

    Example:
        >>> backend = MuJoCoBackend()                    # doctest: +SKIP
        >>> handle = backend.reset(ScenarioConfig())     # doctest: +SKIP
        >>> frame = backend.render(handle)               # doctest: +SKIP
        >>> frame.has_pixels()                           # doctest: +SKIP
        True

    Note:
        Doctests are skipped because constructing a backend compiles an MJCF,
        which is slow enough that it does not belong in a unit test. The
        integration test suite covers this path.
    """

    name = "mujoco"
    description = "MuJoCo rigid-body physics with RGB, depth and segmentation rendering."

    def __init__(self, *, use_viewer: bool = False, cache_dir: str | Path | None = None, **options: Any) -> None:
        """Initialize the backend.

        Args:
            use_viewer: Reserved for interactive debugging. When True the
                backend refuses to run headless rather than silently ignoring it.
            cache_dir: Where generated MJCF files are cached. Defaults to a
                directory under the system temp dir.
            **options: Extra options (ignored).
        """
        super().__init__(**options)
        self._mujoco = _import_mujoco()
        if use_viewer:
            raise SimulationError(
                "interactive MuJoCo viewer is not enabled in this build; "
                "use render() and save_episode_artifacts() to inspect frames",
                code="simulation.viewer_unsupported",
            )
        self._cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "robot_ci_mjcf"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, Any] = {}
        self._renderers: dict[tuple[int, int, int], Any] = {}
        self._active: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> BackendCapabilities:
        """Report MuJoCo backend capabilities."""
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
            deterministic=True,
            max_dof=7,
            supported_tasks=(
                TaskName.GRASP_OBJECT,
                TaskName.PICK_AND_PLACE,
                TaskName.PUSH_OBJECT,
                TaskName.INSPECT_OBJECT,
                TaskName.NAVIGATE_AROUND_OBSTACLE,
            ),
            requires_gpu=False,
            notes="Contact dynamics enabled; suitable for manipulation research claims.",
        )

    # -- model / data management ------------------------------------------
    def _load_model(self, scenario: ScenarioConfig) -> tuple[Any, str]:
        """Compile (or reuse) the MJCF for a scenario.

        Returns:
            ``(model, xml_path)``.
        """
        key = scene_cache_key(scenario)
        cached = self._models.get(key)
        if cached is not None:
            return cached, str(self._cache_dir / f"scene_{key}.xml")

        xml = build_scene_xml(scenario)
        path = self._cache_dir / f"scene_{key}.xml"
        path.write_text(xml, encoding="utf-8")
        try:
            model = self._mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:
            raise SimulationError(
                f"MuJoCo rejected the generated scene for scenario "
                f"{scenario.scenario_id!r}: {exc}",
                code="simulation.mjcf_invalid",
                scenario_id=scenario.scenario_id,
                xml_path=str(path),
            ) from exc
        self._models[key] = model
        return model, str(path)

    def _renderer(self, model: Any, height: int, width: int) -> Any:
        """Return a cached renderer for a given resolution."""
        mujoco = self._mujoco
        key = (id(model), height, width)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(model, height=height, width=width)
            self._renderers[key] = renderer
        return renderer

    # -- lifecycle ---------------------------------------------------------
    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> EpisodeHandle:
        """Reset MuJoCo and place the robot and objects deterministically."""
        episode_seed = scenario.seed if seed is None else int(seed)
        fatal = self.capabilities().incompatible_for(scenario)
        if fatal:
            raise SimulationError(
                f"MuJoCoBackend cannot run scenario {scenario.scenario_id!r}: {', '.join(fatal)}",
                code="simulation.capability_incompatible",
                backend=self.name,
                incompatible=fatal,
            )

        mujoco = self._mujoco
        model, xml_path = self._load_model(scenario)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Settle the scene with a fixed number of steps so objects rest on the
        # table before the policy sees anything. Fixed, not time-based, so the
        # initial state is identical on every machine.
        for _ in range(100):
            mujoco.mj_step(model, data)

        home = scenario.robot.home_array()
        self._write_arm(model, data, home)
        self._write_gripper(model, data, scenario.robot.max_gripper_width_m)
        self._settle_objects(model, data, scenario, episode_seed)
        mujoco.mj_forward(model, data)

        handle = EpisodeHandle(
            episode_id=f"{scenario.scenario_id}-{episode_seed}",
            scenario=scenario,
            seed=episode_seed,
        )
        handle._opaque.update(
            {
                "model": model,
                "data": data,
                "xml_path": xml_path,
                "episode_seed": episode_seed,
                "decay_rng": np.random.default_rng(derive_seed(episode_seed, "mujoco", "decay")),
                "max_steps": scenario.task.max_steps,
                "gripper_target": scenario.robot.max_gripper_width_m,
            }
        )
        self._active[handle.episode_id] = handle._opaque
        return handle

    def _arm_joint_ids(self, model: Any, dof: int) -> list[int]:
        """Return the MuJoCo joint ids of the arm, in policy order."""
        mujoco = self._mujoco
        ids: list[int] = []
        for index in range(dof):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index + 1}")
            ids.append(jid)
        return ids

    def _write_arm(self, model: Any, data: Any, targets: np.ndarray) -> None:
        """Command and teleport the arm to ``targets``.

        Both the control signal and the joint state are written, so an episode
        never begins or evaluates mid-transient. Everything is addressed by
        name: free-jointed objects share ``qpos`` with the arm, and their
        ordering depends on the MJCF, so positional slicing is unsafe.
        """
        for index, jid in enumerate(self._arm_joint_ids(model, len(targets))):
            if jid < 0:
                continue
            value = float(targets[index])
            data.ctrl[jid] = value
            data.qpos[model.jnt_qposadr[jid]] = value
            data.qvel[model.jnt_dofadr[jid]] = 0.0

    @staticmethod
    def _write_gripper(model: Any, data: Any, width_m: float) -> None:
        """Set both fingers so their total aperture is ``width_m``.

        Each finger travels at most 0.04 m from its rest pose, so the total
        aperture is ``2 * min(0.04, width_m / 2)``.
        """
        half = max(0.0, min(0.04, width_m / 2.0))
        for index in range(model.nu):
            name = model.actuator(index).name
            if name == "left_finger_act":
                data.ctrl[index] = half
            elif name == "right_finger_act":
                data.ctrl[index] = -half

    def _settle_objects(self, model: Any, data: Any, scenario: ScenarioConfig, seed: int) -> None:
        """Jitter and re-settle object poses from the episode seed.

        Without this, every episode of a scenario would start from an identical
        object pose, so repeated runs would measure nothing. The jitter is small
        (a few millimetres) so it never changes the difficulty *class* of a
        scenario, only which instance of it is being run.

        The address of an object's free-joint position is found through its body
        name; indexing ``jnt_bodyid`` by body id would be wrong, because that
        array is indexed by *joint* id.
        """
        mujoco = self._mujoco
        rng = np.random.default_rng(derive_seed(seed, "mujoco", "object-jitter"))
        for spec in scenario.objects:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.id)
            if body_id < 0:
                continue
            joint_id = model.body_jntadr[body_id]
            if joint_id < 0 or joint_id >= model.njnt:
                continue
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            adr = int(model.jnt_qposadr[joint_id])
            data.qpos[adr + 0] += float(rng.normal(0.0, 0.003))
            data.qpos[adr + 1] += float(rng.normal(0.0, 0.003))
            # Orientation jitter about the vertical axis keeps the object on
            # the table instead of tipping it.
            yaw = float(rng.normal(0.0, 0.05))
            data.qpos[adr + 3] = float(np.cos(yaw / 2.0))
            data.qpos[adr + 4] = 0.0
            data.qpos[adr + 5] = 0.0
            data.qpos[adr + 6] = float(np.sin(yaw / 2.0))
            data.qvel[adr : adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(120):
            mujoco.mj_step(model, data)

    @staticmethod
    def _require(handle: EpisodeHandle) -> dict[str, Any]:
        """Return the episode's MuJoCo state, or fail loudly if it is closed."""
        if not handle._opaque:
            raise SimulationError(
                f"episode {handle.episode_id!r} is not active on this backend",
                code="simulation.episode_closed",
                episode_id=handle.episode_id,
            )
        return handle._opaque

    # -- stepping ----------------------------------------------------------
    def step(self, handle: EpisodeHandle, action: Action) -> Transition:
        """Apply one action, advance physics, and report the new state."""
        state = self._require(handle)
        mujoco = self._mujoco
        model, data = state["model"], state["data"]
        scenario = handle.scenario
        robot = scenario.robot
        dt = 1.0 / robot.control_hz

        if action.type is ActionType.JOINT_POSITION:
            targets = np.asarray(action.joint_position, dtype=np.float64)
            if targets.shape != (robot.dof,):
                raise SimulationError(
                    f"expected {robot.dof} joint targets, got {targets.shape}",
                    code="simulation.bad_action_shape",
                )
            self._write_arm(model, data, robot.limits.clamped(targets) * action.scale)
        elif action.type is ActionType.JOINT_VELOCITY:
            velocity = np.asarray(action.joint_velocity, dtype=np.float64)
            current = self._arm_qpos(model, data, robot.dof)
            self._write_arm(model, data, robot.limits.clamped(current + velocity * dt))
        elif action.type is ActionType.GRIPPER:
            self._write_gripper(model, data, float(action.gripper_width_m or 0.0))
            state["gripper_target"] = float(action.gripper_width_m or 0.0)
        elif action.type is ActionType.CARTESIAN_DELTA:
            # Without an IK solver in Phase 1, a Cartesian command is projected
            # onto the first joint. Phase 2 replaces this with damped least
            # squares IK; the interface does not change.
            delta = np.asarray(action.cartesian_delta, dtype=np.float64)
            current = self._arm_qpos(model, data, robot.dof)
            perturb = np.zeros(robot.dof)
            perturb[0] = -float(delta[0])
            if robot.dof > 1:
                perturb[1] = float(delta[1])
            self._write_arm(model, data, robot.limits.clamped(current + perturb * dt))
        elif action.type is not ActionType.NOOP:
            raise SimulationError(
                f"MuJoCoBackend does not implement {action.type.value!r}",
                code="simulation.action_unsupported",
            )

        # Physics substeps at the MuJoCo timestep for stable contact.
        substeps = max(1, int(round(dt / float(model.opt.timestep))))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        contact = self._classify_contact(model, data, scenario)
        handle.step_index += 1
        handle.sim_time_s += dt
        if handle.step_index >= state["max_steps"] and not handle.done:
            handle.done = True
            handle.termination = TerminationReason.MAX_STEPS

        return Transition(
            observation=self.observe(handle),
            robot_state=self._robot_state(handle),
            ground_truth=self.ground_truth(handle),
            reward=0.0,
            terminated=handle.done,
            truncation=handle.termination is TerminationReason.MAX_STEPS,
            info={
                "backend": self.name,
                "collision": contact is not ContactState.NONE,
                "contact": contact.value,
                "planning_failed": False,
            },
        )

    def _classify_contact(self, model: Any, data: Any, scenario: ScenarioConfig) -> ContactState:
        """Classify the first active contact involving the gripper."""
        mujoco = self._mujoco
        for index in range(data.ncon):
            contact = data.contact[index]
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            pair = {name1, name2}
            if any("finger" in n for n in pair):
                if "table" in pair:
                    return ContactState.TABLE
                if any(n.startswith("obstacle") for n in pair):
                    return ContactState.OBSTACLE
                if any(n.startswith(n) for n in pair for n in (o.id for o in scenario.objects)):
                    return ContactState.OBJECT
                if any("link" in n for n in pair):
                    return ContactState.SELF
                return ContactState.OBJECT
        return ContactState.NONE

    def _finger_width(self, model: Any, data: Any) -> float:
        """Return the current aperture between the two fingers, in metres."""
        left = self._slide_position(model, data, "left_finger_joint")
        right = self._slide_position(model, data, "right_finger_joint")
        return float(abs(left - right))

    def _slide_position(self, model: Any, data: Any, joint_name: str) -> float:
        """Return the scalar position of a named slide joint."""
        mujoco = self._mujoco
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return 0.0
        return float(data.qpos[model.jnt_qposadr[jid]])

    def _ee_pose(self, model: Any, data: Any) -> Pose:
        """Return the end-effector pose from the ``ee`` site."""
        mujoco = self._mujoco
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        if sid < 0:
            return Pose()
        position = np.array(data.site_xpos[sid], dtype=np.float64)
        mat = np.array(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
        return Pose(position=Vec3.from_iterable(position), orientation=self._mat_to_quat(mat))

    @staticmethod
    def _mat_to_quat(mat: np.ndarray) -> Quaternion:
        """Convert a 3x3 rotation matrix to a ``(w, x, y, z)`` quaternion."""
        trace = float(mat[0, 0] + mat[1, 1] + mat[2, 2])
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat[2, 1] - mat[1, 2]) * s
            y = (mat[0, 2] - mat[2, 0]) * s
            z = (mat[1, 0] - mat[0, 1]) * s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
        return Quaternion(w=float(w), x=float(x), y=float(y), z=float(z))

    def _arm_qpos(self, model: Any, data: Any, dof: int) -> np.ndarray:
        """Return the arm joint positions, addressed by joint *name*.

        Slicing ``qpos[:dof]`` would be wrong: free-jointed objects also live in
        ``qpos``, and their position depends on where they appear in the MJCF.
        Looking the joints up by name makes the mapping immune to document order.
        """
        mujoco = self._mujoco
        values = np.zeros(dof, dtype=np.float64)
        for index in range(dof):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index + 1}")
            if jid >= 0:
                values[index] = float(data.qpos[model.jnt_qposadr[jid]])
        return values

    def _arm_qvel(self, model: Any, data: Any, dof: int) -> np.ndarray:
        """Return the arm joint velocities, addressed by joint name."""
        mujoco = self._mujoco
        values = np.zeros(dof, dtype=np.float64)
        for index in range(dof):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index + 1}")
            if jid >= 0:
                values[index] = float(data.qvel[model.jnt_dofadr[jid]])
        return values

    def _robot_state(self, handle: EpisodeHandle) -> RobotState:
        """Build proprioception from MuJoCo state (uncorrupted)."""
        state = self._require(handle)
        model, data = state["model"], state["data"]
        robot = handle.scenario.robot
        positions = self._arm_qpos(model, data, robot.dof)
        velocities = self._arm_qvel(model, data, robot.dof)
        efforts = data.actuator_force[: robot.dof] if model.nu >= robot.dof else np.zeros(robot.dof)
        return RobotState(
            joint_positions=tuple(float(v) for v in positions),
            joint_velocities=tuple(float(v) for v in velocities),
            joint_efforts=tuple(float(v) for v in efforts),
            joint_names=robot.arm_joint_names,
            end_effector_pose=self._ee_pose(model, data),
            gripper=GripperState(
                width_m=self._finger_width(model, data),
                max_width_m=robot.max_gripper_width_m,
                is_grasping=False,
            ),
            timestamp_s=handle.sim_time_s,
        )

    # -- observation -------------------------------------------------------
    def observe(self, handle: EpisodeHandle) -> Observation:
        """Return the current observation.

        The observation carries proprioception and the *fused belief* about
        objects. In Phase 1 the belief is derived from MuJoCo body poses; from
        Phase 10 it comes from the perception + sensor-fusion pipeline reading
        the rendered frames. Either way the policy never sees ground truth,
        because the belief is rebuilt through the same estimation path.
        """
        state = self._require(handle)
        model, data = state["model"], state["data"]
        scenario = handle.scenario
        beliefs = self._object_beliefs(model, data, scenario, handle.sim_time_s)
        return Observation(
            step_index=handle.step_index,
            timestamp_s=handle.sim_time_s,
            robot=self._robot_state(handle),
            objects=beliefs,
            sensor_frame=None,
            seed=handle.seed,
            scenario_id=scenario.scenario_id,
        )

    def _object_beliefs(
        self, model: Any, data: Any, scenario: ScenarioConfig, now: float
    ) -> tuple[ObjectState, ...]:
        """Build believed object states from the estimator path.

        The uncertainty assigned here is a *floor* representing estimator noise.
        Phase 10 replaces this with measurement-derived covariance; until then it
        is deliberately explicit that belief == truth, which is why Phase 1 makes
        no perception claims.
        """
        mujoco = self._mujoco
        beliefs: list[ObjectState] = []
        for spec in scenario.objects:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.id)
            if bid < 0:
                continue
            position = np.array(data.xpos[bid], dtype=np.float64)
            mat = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)
            beliefs.append(
                ObjectState(
                    id=spec.id,
                    category=spec.category,
                    pose=Pose(position=Vec3.from_iterable(position), orientation=self._mat_to_quat(mat)),
                    covariance=Covariance.isotropic(0.002, dimension=6),
                    confidence=1.0,
                    source=Source.SIMULATOR,
                    timestamp_s=now,
                    visible=True,
                    last_seen_s=now,
                    estimated_size_m=spec.size_m,
                )
            )
        return tuple(beliefs)

    def render(self, handle: EpisodeHandle) -> SensorFrame:
        """Render RGB, depth and segmentation from the scenario's cameras."""
        state = self._require(handle)
        model, data = state["model"], state["data"]
        scenario = handle.scenario
        height, width = scenario.resolution
        renderer = self._renderer(model, height, width)

        rgb_array: np.ndarray | None = None
        depth_array: np.ndarray | None = None
        seg_array: np.ndarray | None = None

        if scenario.render_rgb:
            renderer.update_scene(data, camera="rgb_camera")
            rgb_array = np.asarray(renderer.render()).astype(np.uint8)
        if scenario.render_depth:
            renderer.update_scene(data, camera="depth_camera")
            depth_array = np.asarray(renderer.render()).astype(np.float32)
        if scenario.render_segmentation:
            renderer.update_scene(data, camera="seg_camera")
            seg_array = np.asarray(renderer.render()).astype(np.uint8)

        intrinsics = scenario.sensors.effective_intrinsics()
        return SensorFrame(
            camera_id="rgb_camera",
            timestamp_s=handle.sim_time_s,
            rgb=rgb_array,
            depth=depth_array,
            segmentation=seg_array,
            width_px=width,
            height_px=height,
            intrinsics=intrinsics.as_tuple(),
            valid=True,
        )

    def ground_truth(self, handle: EpisodeHandle) -> GroundTruthState:
        """Return the hidden simulator state (evaluator only).

        The policy layer must never call this. It exists so metrics can be
        computed against truth *after* an episode finishes.
        """
        state = self._require(handle)
        mujoco = self._mujoco
        model, data = state["model"], state["data"]
        scenario = handle.scenario
        objects: list[GroundTruthObject] = []
        for spec in scenario.objects:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.id)
            if bid < 0:
                continue
            position = np.array(data.xpos[bid], dtype=np.float64)
            mat = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)
            extent = spec.size_m
            objects.append(
                GroundTruthObject(
                    id=spec.id,
                    category=spec.category,
                    pose=Pose(position=Vec3.from_iterable(position), orientation=self._mat_to_quat(mat)),
                    size_m=(extent, extent, extent),
                    is_target=spec.is_target,
                    grasped=self._is_grasped(model, data, spec.id),
                    friction=spec.friction if spec.friction is not None else scenario.environment.object_friction,
                )
            )
        penetration = 0.0
        for index in range(data.ncon):
            penetration = max(penetration, float(data.contact[index].dist))
        return GroundTruthState(
            objects=tuple(objects),
            true_joint_positions=tuple(
                float(v) for v in self._arm_qpos(model, data, scenario.robot.dof)
            ),
            true_ee_pose=self._ee_pose(model, data),
            contact=self._classify_contact(model, data, scenario),
            sim_time_s=handle.sim_time_s,
            penetration_depth_m=max(0.0, -penetration),
        )

    def _is_grasped(self, model: Any, data: Any, object_id: str) -> bool:
        """Whether both fingers are in contact with the named object."""
        mujoco = self._mujoco
        touched: set[str] = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            for geom in (contact.geom1, contact.geom2):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                if name == object_id:
                    other = contact.geom2 if geom == contact.geom1 else contact.geom1
                    other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                    if "finger" in other_name:
                        touched.add(other_name)
        return len(touched) >= 2

    # -- lifecycle ---------------------------------------------------------
    def close(self, handle: EpisodeHandle) -> None:
        """Release one episode's MuJoCo data. Idempotent."""
        self._active.pop(handle.episode_id, None)
        handle._opaque = {}

    def close_all(self) -> None:
        """Release all renderers, models and episodes. Idempotent."""
        for renderer in self._renderers.values():
            # Best-effort teardown: one renderer whose GL context is already gone
            # must not prevent the rest from being released.
            with contextlib.suppress(Exception):
                renderer.close()
        self._renderers.clear()
        self._models.clear()
        self._active.clear()

    def save_episode_artifacts(self, handle: EpisodeHandle, directory: Any) -> list[str]:
        """Write the current frame as PNG and depth as NPY, for inspection.

        Args:
            handle: The episode to dump.
            directory: Target directory (created if needed).

        Returns:
            Paths of the files written. Returns an empty list when OpenCV is
            not installed, so artifact dumping is optional.
        """
        state = self._require(handle)
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        frame = self.render(handle)
        written: list[str] = []

        rgb = frame.rgb_array()
        if rgb is not None:
            try:
                import cv2
            except ImportError:
                return written
            rgb_path = out / f"{handle.episode_id}_rgb.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            written.append(str(rgb_path))

        depth = frame.depth_array()
        if depth is not None:
            depth_path = out / f"{handle.episode_id}_depth.npy"
            np.save(depth_path, depth)
            written.append(str(depth_path))

        del state
        return written

    def info(self) -> BackendInfo:
        """Return backend identity including the detected OpenGL backend."""
        return BackendInfo(
            name=self.name,
            version=SIMULATOR_VERSION,
            implementation=type(self).__module__,
            capabilities=self.capabilities(),
            extras={
                "gl_backend": os.environ.get("MUJOCO_GL", "unset"),
                "platform": sys.platform,
                "scenes_cached": str(len(self._models)),
            },
        )
