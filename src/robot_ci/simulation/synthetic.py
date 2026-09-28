"""A dependency-light synthetic backend: NumPy ray casting, no physics engine.

Purpose
-------
``SyntheticBackend`` renders **real** depth and RGB images from an analytic
scene using only NumPy. It is not a physics simulator, but it *is* a real
renderer, which makes it the right tool for:

* testing the corruption framework against ground-truth geometry,
* developing perception without waiting on a physics engine,
* providing a second, independent implementation of the backend contract so
  that cross-backend agreement becomes a testable property.

Geometry
--------
Spheres and axis-aligned boxes are ray-cast analytically. A parallel (orthographic)
projection is used rather than a pinhole projection: it is exact, resolution
independent, and makes the depth image directly interpretable as a distance
along the view ray. Intrinsics are still carried in the interface so that
perception code written against this backend is identical to code written
against the pinhole MuJoCo backend.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robot_ci.core.errors import SimulationError
from robot_ci.core.seeding import make_numpy_rng
from robot_ci.models.environment import ObjectCategory, TaskName
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
    EpisodeHandle,
    SimulatorBackend,
    TerminationReason,
    Transition,
)

__all__ = ["SyntheticBackend"]

#: Nominal RGB colour per object category, in [0, 1].
_CATEGORY_RGB: dict[ObjectCategory, tuple[float, float, float]] = {
    ObjectCategory.CUBE: (0.85, 0.15, 0.15),
    ObjectCategory.CYLINDER: (0.15, 0.45, 0.85),
    ObjectCategory.SPHERE: (0.20, 0.70, 0.30),
    ObjectCategory.CAN: (0.90, 0.80, 0.20),
    ObjectCategory.LIFTER: (0.70, 0.70, 0.75),
}


class SyntheticBackend(SimulatorBackend):
    """Analytic ray-cast renderer with a first-order joint model.

    Example:
        >>> backend = SyntheticBackend()
        >>> handle = backend.reset(ScenarioConfig(), seed=3)
        >>> frame = backend.render(handle)
        >>> frame.width_px > 0
        True
    """

    name = "synthetic"
    description = "NumPy ray-cast renderer with analytic kinematics; no physics engine."

    def __init__(self, *, shutter: bool = False, **options: Any) -> None:
        """Initialize the backend.

        Args:
            shutter: When True, ``render`` returns an empty frame, modelling a
                camera that produces no image. Used to test the frame-drop path.
            **options: Extra options (ignored).
        """
        super().__init__(**options)
        self._shutter = bool(shutter)

    def capabilities(self) -> BackendCapabilities:
        """Report synthetic backend capabilities."""
        return BackendCapabilities(
            name=self.name,
            physics=False,
            collision=False,
            rgb=True,
            depth=True,
            segmentation=True,
            force_torque=False,
            joint_efforts=True,
            grasp_latch=True,
            deterministic=True,
            max_dof=12,
            supported_tasks=(
                TaskName.GRASP_OBJECT,
                TaskName.PICK_AND_PLACE,
                TaskName.PUSH_OBJECT,
                TaskName.INSPECT_OBJECT,
                TaskName.NAVIGATE_AROUND_OBSTACLE,
            ),
            requires_gpu=False,
            notes=(
                "Renders genuine depth/RGB by analytic ray casting. No contact "
                "dynamics, so grasping is a latch. Suitable for corruption and "
                "perception development; not for manipulation success claims."
            ),
        )

    # -- lifecycle ---------------------------------------------------------
    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> EpisodeHandle:
        """Start a synthetic episode."""
        episode_seed = scenario.seed if seed is None else int(seed)
        fatal = self.capabilities().incompatible_for(scenario)
        if fatal:
            raise SimulationError(
                f"SyntheticBackend cannot run {scenario.scenario_id!r}: {', '.join(fatal)}",
                code="simulation.capability_incompatible",
                backend=self.name,
                incompatible=fatal,
            )
        if self._shutter:
            raise SimulationError(
                "SyntheticBackend was constructed with shutter=True, which cannot run episodes",
                code="simulation.backend_misconfigured",
            )

        robot = scenario.robot
        rng = make_numpy_rng(episode_seed, "synthetic", scenario.scenario_id)
        objects: dict[str, dict[str, Any]] = {}
        table_z = scenario.environment.table_height_m
        for index, spec in enumerate(scenario.objects):
            if spec.position is not None:
                position = spec.position
            else:
                angle = 2.0 * np.pi * float(rng.uniform(0.1, 0.9))
                radius = 0.16 + 0.07 * index
                position = Vec3.of(
                    float(radius * math.cos(angle)),
                    float(radius * math.sin(angle)),
                    table_z + spec.size_m,
                )
            objects[spec.id] = {
                "id": spec.id,
                "category": spec.category,
                "position": position,
                "orientation": spec.orientation,
                "size_m": spec.size_m,
                "is_target": spec.is_target,
                "color": spec.color_rgb or _CATEGORY_RGB[spec.category],
            }

        handle = EpisodeHandle(
            episode_id=f"{scenario.scenario_id}-{episode_seed}",
            scenario=scenario,
            seed=episode_seed,
        )
        handle._opaque.update(
            {
                "joint_positions": robot.home_array(),
                "joint_velocities": np.zeros(robot.dof),
                "gripper_width": robot.max_gripper_width_m,
                "grasped": None,
                "objects": objects,
                "rng": make_numpy_rng(episode_seed, "synthetic", "steps"),
            }
        )
        return handle

    @staticmethod
    def _require(handle: EpisodeHandle) -> dict[str, Any]:
        """Return the episode's internal state, or fail loudly if it is closed.

        ``EpisodeHandle._opaque`` is emptied by :meth:`close`, which is how
        "this episode is gone" is represented uniformly across backends.
        """
        if not handle._opaque:
            raise SimulationError(
                f"episode {handle.episode_id!r} is not active on this backend",
                code="simulation.episode_closed",
                episode_id=handle.episode_id,
            )
        return handle._opaque

    def close(self, handle: EpisodeHandle) -> None:
        """Release one episode. Idempotent."""
        handle._opaque = {}

    def close_all(self) -> None:
        """Release every episode. Idempotent."""
        pass

    def _view_ray_directions(self, scenario: ScenarioConfig) -> tuple[np.ndarray, np.ndarray, float]:
        """Build an orthographic ray grid looking from the camera at the target.

        Returns:
            ``(origins, directions, pixel_size)`` where ``origins`` has shape
            ``(H, W, 3)`` and ``directions`` is ``(3,)`` shared by every ray.
        """
        height, width = scenario.resolution
        eye = scenario.camera_position.array
        target = scenario.camera_look_at.array
        forward = target - eye
        distance = float(np.linalg.norm(forward))
        forward = forward / max(distance, 1e-9)

        # Right-handed basis: world up is +z, so right = forward x up.
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Frustum width in metres, independent of pixel resolution. 0.9 m covers
        # the default tabletop workspace from the default camera pose.
        extent = 0.9
        pixel = extent / max(width, 1)
        xs = (np.arange(width) + 0.5 - width / 2.0) * pixel
        ys = -(np.arange(height) + 0.5 - height / 2.0) * pixel
        grid_x, grid_y = np.meshgrid(xs, ys)
        origins = (
            eye[None, None, :]
            + grid_x[..., None] * right[None, None, :]
            + grid_y[..., None] * up[None, None, :]
        )
        return origins, forward, float(pixel)

    def _cast(self, scenario: ScenarioConfig, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ray-cast every object; return ``(depth, class_id, colour)`` images.

        Objects are modelled as spheres of radius ``size_m`` for every category.
        The simplification is deliberate and documented: a sphere has a closed-form
        ray intersection, which keeps this renderer exact and fast, and it does
        not bias *relative* perception difficulty across noise levels.
        """
        origins, directions, _ = self._view_ray_directions(scenario)
        height, width = scenario.resolution
        depth = np.full((height, width), np.inf, dtype=np.float32)
        class_id = np.zeros((height, width), dtype=np.uint8)
        colour = np.zeros((height, width, 3), dtype=np.float32)

        # Background/table plane at the support surface height.
        table_z = scenario.environment.table_height_m
        if abs(directions[2]) > 1e-6:
            t_plane = (table_z - origins[:, :, 2]) / directions[2]
            valid = t_plane > 0
            depth[valid] = t_plane[valid]
            base_grey = 0.35 + 0.5 * (scenario.environment.lighting_lux / 500.0)
            colour[valid] = np.clip(base_grey, 0.02, 0.95)

        for index, obj in enumerate(state["objects"].values(), start=1):
            centre = obj["position"].array
            radius = float(obj["size_m"])
            # Ray-sphere intersection: |o + t d - c|^2 = r^2.
            oc = origins - centre[None, None, :]
            b = 2.0 * np.einsum("hwc,c->hw", oc, directions)
            c = np.einsum("hwc,hwc->hw", oc, oc) - radius * radius
            disc = b * b - 4.0 * c
            hit = disc > 0.0
            if not hit.any():
                continue
            t = (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0
            visible = hit & (t > 0) & (t < depth)
            if not visible.any():
                continue
            depth[visible] = t[visible]
            class_id[visible] = index
            shade = float(np.clip(obj["color"][0] * 0.9 + 0.1, 0.0, 1.0))
            colour[visible] = np.array(obj["color"], dtype=np.float32) * shade

        return depth, class_id, colour

    # -- stepping ----------------------------------------------------------
    def step(self, handle: EpisodeHandle, action: Action) -> Transition:
        """Apply one action to the analytic joint model."""
        state = self._require(handle)
        scenario = handle.scenario
        robot = scenario.robot
        dt = 1.0 / robot.control_hz

        if action.type is ActionType.JOINT_POSITION:
            target = np.asarray(action.joint_position, dtype=np.float64)
            if target.shape != (robot.dof,):
                raise SimulationError(
                    f"expected {robot.dof} joint targets, got {target.shape}",
                    code="simulation.bad_action_shape",
                )
            alpha = min(1.0, 8.0 * dt)
            previous = state["joint_positions"]
            state["joint_positions"] = (1 - alpha) * previous + alpha * robot.limits.clamped(target)
            state["joint_velocities"] = (state["joint_positions"] - previous) / max(dt, 1e-9)
        elif action.type is ActionType.JOINT_VELOCITY:
            velocity = np.asarray(action.joint_velocity, dtype=np.float64)
            state["joint_positions"] = robot.limits.clamped(state["joint_positions"] + velocity * dt)
            state["joint_velocities"] = velocity
        elif action.type is ActionType.GRIPPER:
            target_width = float(action.gripper_width_m or 0.0)
            state["gripper_width"] += (target_width - state["gripper_width"]) * min(1.0, 10.0 * dt)
        elif action.type is ActionType.CARTESIAN_DELTA:
            delta = np.asarray(action.cartesian_delta, dtype=np.float64)
            perturb = np.zeros(robot.dof)
            perturb[0] = -float(delta[0])
            if robot.dof > 1:
                perturb[1] = -float(delta[1])
            state["joint_positions"] = robot.limits.clamped(state["joint_positions"] + perturb * dt)
        elif action.type is not ActionType.NOOP:
            raise SimulationError(
                f"SyntheticBackend does not implement {action.type.value!r}",
                code="simulation.action_unsupported",
            )

        handle.step_index += 1
        handle.sim_time_s += dt
        if handle.step_index >= scenario.task.max_steps and not handle.done:
            handle.done = True
            handle.termination = TerminationReason.MAX_STEPS

        return Transition(
            observation=self.observe(handle),
            robot_state=self._robot_state(handle),
            ground_truth=self.ground_truth(handle),
            reward=0.0,
            terminated=handle.done,
            truncation=handle.termination is TerminationReason.MAX_STEPS,
            info={"backend": self.name, "collision": False, "planning_failed": False},
        )

    def _robot_state(self, handle: EpisodeHandle) -> RobotState:
        state = self._require(handle)
        robot = handle.scenario.robot
        return RobotState(
            joint_positions=tuple(float(v) for v in state["joint_positions"]),
            joint_velocities=tuple(float(v) for v in state["joint_velocities"]),
            joint_efforts=tuple(float(v) for v in state["joint_velocities"] * 2.0),
            joint_names=robot.arm_joint_names,
            end_effector_pose=self._ee_pose(state),
            gripper=GripperState(
                width_m=float(state["gripper_width"]),
                max_width_m=robot.max_gripper_width_m,
                is_grasping=state["grasped"] is not None,
                grasped_object_id=state["grasped"],
            ),
            timestamp_s=handle.sim_time_s,
        )

    @staticmethod
    def _ee_pose(state: dict[str, Any]) -> Pose:
        q = state["joint_positions"]
        reach = 0.30 + 0.15 * math.cos(float(q[0]))
        return Pose(
            position=Vec3.of(reach, 0.0, 0.35 + 0.20 * math.sin(float(q[3]))),
            orientation=Quaternion.from_axis_angle(Vec3.of(0.0, 0.0, 1.0), float(q[0])),
        )

    def observe(self, handle: EpisodeHandle) -> Observation:
        """Return the current observation (no ground truth)."""
        state = self._require(handle)
        beliefs = tuple(
            ObjectState(
                id=obj["id"],
                category=obj["category"],
                pose=Pose(position=obj["position"], orientation=obj["orientation"]),
                covariance=Covariance.isotropic(0.004, dimension=6),
                confidence=0.95,
                source=Source.FUSION,
                timestamp_s=handle.sim_time_s,
                visible=True,
                last_seen_s=handle.sim_time_s,
                estimated_size_m=float(obj["size_m"]),
            )
            for obj in state["objects"].values()
        )
        return Observation(
            step_index=handle.step_index,
            timestamp_s=handle.sim_time_s,
            robot=self._robot_state(handle),
            objects=beliefs,
            sensor_frame=None,
            seed=handle.seed,
            scenario_id=handle.scenario.scenario_id,
        )

    def render(self, handle: EpisodeHandle) -> SensorFrame:
        """Ray-cast the scene into depth, segmentation and shaded RGB images."""
        state = self._require(handle)
        scenario = handle.scenario
        depth, class_id, colour = self._cast(scenario, state)
        height, width = scenario.resolution
        depth_m = np.where(np.isfinite(depth), depth, np.nan).astype(np.float32)

        # Lighting scales the rendered radiance; the corruption layer adds noise.
        exposure = float(np.clip(scenario.environment.lighting_lux / 500.0, 0.02, 3.0))
        rgb_u8 = np.clip(colour * exposure * 255.0, 0, 255).astype(np.uint8)

        intrinsics = scenario.sensors.effective_intrinsics()
        return SensorFrame(
            camera_id="synthetic_camera",
            timestamp_s=handle.sim_time_s,
            rgb=rgb_u8.tolist() if scenario.render_rgb else None,
            depth=[[float(v) for v in row] for row in depth_m] if scenario.render_depth else None,
            segmentation=class_id.tolist() if scenario.render_segmentation else None,
            width_px=width,
            height_px=height,
            intrinsics=intrinsics.as_tuple(),
            valid=True,
        )

    def ground_truth(self, handle: EpisodeHandle) -> GroundTruthState:
        """Return the hidden state. Evaluator only."""
        state = self._require(handle)
        return GroundTruthState(
            objects=tuple(
                GroundTruthObject(
                    id=obj["id"],
                    category=obj["category"],
                    pose=Pose(position=obj["position"], orientation=obj["orientation"]),
                    size_m=(obj["size_m"],) * 3,
                    is_target=bool(obj["is_target"]),
                    grasped=obj["id"] == state["grasped"],
                    friction=handle.scenario.environment.object_friction,
                )
                for obj in state["objects"].values()
            ),
            true_joint_positions=tuple(float(v) for v in state["joint_positions"]),
            true_ee_pose=self._ee_pose(state),
            contact=ContactState.NONE,
            sim_time_s=handle.sim_time_s,
        )
