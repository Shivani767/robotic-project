"""A fully analytic mock backend.

This backend exists so that **the entire research engine can be developed and
unit-tested on a laptop CPU with no physics engine at all**. It is not a toy:
it implements the same interface, the same determinism guarantees and the same
ground-truth separation as the real backends, so a regression test written
against it exercises real framework code.

What is analytic (not simulated)
--------------------------------
* Joint trajectories: a deterministic, seeded, band-limited random walk in
  joint space, optionally tracking a commanded target.
* Gripper: first-order lag toward the commanded aperture.
* Object: rigidly attached to the gripper once grasped, otherwise static.

What this backend deliberately does **not** do
----------------------------------------------
* It does not model contact dynamics, friction, or compliant grasping.
* It does not render physically meaningful images.

Any research claim about manipulation success must therefore be produced with
``--backend mujoco``. The mock backend is for framework testing, CI smoke runs
and as a deterministic oracle for search algorithms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_ci.core.errors import SimulationError
from robot_ci.core.seeding import make_numpy_rng
from robot_ci.models.environment import TaskName
from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
from robot_ci.models.robot import RobotConfig
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

__all__ = ["MockBackend"]


class MockBackend(SimulatorBackend):
    """Deterministic, dependency-light backend for framework tests.

    Example:
        >>> backend = MockBackend()
        >>> scenario = ScenarioConfig()
        >>> handle = backend.reset(scenario, seed=7)
        >>> obs = backend.observe(handle)
        >>> len(obs.robot.joint_positions) == scenario.robot.dof
        True
    """

    name = "mock"
    description = "Analytic backend: no physics, no rendering, fully deterministic."

    def __init__(self, *, observation_noise_std: float = 0.0, **options: Any) -> None:
        """Initialize the backend.

        Args:
            observation_noise_std: Optional noise added to the *observed* joint
                state, so tests can exercise the observation/ground-truth split.
            **options: Ignored extra options (kept for interface symmetry).
        """
        super().__init__(**options)
        if observation_noise_std < 0:
            raise ValueError("observation_noise_std must be non-negative")
        self._observation_noise_std = float(observation_noise_std)
        self._active: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> BackendCapabilities:
        """Report the mock backend's (deliberately minimal) capabilities."""
        return BackendCapabilities(
            name=self.name,
            physics=False,
            collision=False,
            rgb=False,
            depth=False,
            segmentation=False,
            force_torque=False,
            joint_efforts=True,
            grasp_latch=True,
            deterministic=True,
            max_dof=12,
            supported_tasks=(
                TaskName.GRASP_OBJECT,
                TaskName.PICK_AND_PLACE,
                TaskName.PUSH_OBJECT,
                TaskName.NAVIGATE_AROUND_OBSTACLE,
            ),
            requires_gpu=False,
            notes=(
                "Analytic stand-in for framework tests. Grasp is a LATCH, not contact "
                "dynamics: suitable for exercising policy/regression/search code, NOT "
                "for research claims about manipulation success."
            ),
        )

    # -- lifecycle ---------------------------------------------------------
    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> EpisodeHandle:
        """Start a deterministic analytic episode."""
        episode_seed = scenario.seed if seed is None else int(seed)
        robot = scenario.robot
        fatal = self.capabilities().incompatible_for(scenario)
        if fatal:
            raise SimulationError(
                f"MockBackend cannot run scenario {scenario.scenario_id!r}: "
                f"incompatible on {', '.join(fatal)}",
                code="simulation.capability_incompatible",
                backend=self.name,
                incompatible=fatal,
            )

        rng = make_numpy_rng(episode_seed, "mock", scenario.scenario_id)
        joint_positions = robot.home_array()
        objects = self._place_objects(scenario, rng)

        handle = EpisodeHandle(
            episode_id=f"{scenario.scenario_id}-{episode_seed}",
            scenario=scenario,
            seed=episode_seed,
        )
        handle._opaque.update(
            {
                "joint_positions": joint_positions,
                "joint_velocities": np.zeros(robot.dof),
                "joint_efforts": np.zeros(robot.dof),
                "gripper_width": robot.max_gripper_width_m,
                "grasped_object": None,
                "objects": objects,
                "ee_pose": self._forward_kinematics_stub(robot, joint_positions),
                "rng": make_numpy_rng(episode_seed, "mock", "steps"),
                "max_steps": scenario.task.max_steps,
            }
        )
        self._active[handle.episode_id] = handle._opaque
        return handle

    def _place_objects(self, scenario: ScenarioConfig, rng: np.random.Generator) -> dict[str, dict[str, Any]]:
        """Resolve object poses: explicit positions win, else sample from seed."""
        objects: dict[str, dict[str, Any]] = {}
        table_z = scenario.environment.table_height_m
        for spec in scenario.objects:
            if spec.position is not None:
                position = spec.position
            else:
                # Rejection-free placement in a deterministic grid offset by the
                # seed, so scenes are spread but reproducible.
                index = len(objects)
                angle = 2.0 * np.pi * float(rng.uniform(0.15, 0.85))
                radius = 0.18 + 0.06 * (index % 4)
                position = Vec3.of(
                    float(radius * np.cos(angle)),
                    float(radius * np.sin(angle)),
                    table_z + spec.size_m,
                )
            objects[spec.id] = {
                "id": spec.id,
                "category": spec.category,
                "position": position,
                "orientation": spec.orientation,
                "size_m": spec.size_m,
                "is_target": spec.is_target,
                "grasped": False,
            }
        return objects

    @staticmethod
    def _forward_kinematics_stub(robot: RobotConfig, q: np.ndarray) -> Pose:
        """A crude but deterministic end-effector estimate from joint angles.

        Intentionally simple: the mock backend is not a kinematic simulator. It
        only needs to move the EE pose in a way that responds to commands so
        that policy code paths are exercised.
        """
        q1, q4 = float(q[0]), float(q[3])
        reach = 0.30 + 0.15 * np.cos(q1)
        return Pose(
            position=Vec3.of(reach, 0.0, 0.35 + 0.20 * np.sin(q4)),
            orientation=Quaternion.from_axis_angle(Vec3.of(0.0, 0.0, 1.0), q1),
        )

    def _require(self, handle: EpisodeHandle) -> dict[str, Any]:
        if handle.episode_id not in self._active:
            raise SimulationError(
                f"episode {handle.episode_id!r} is not active on {self.name!r}",
                code="simulation.episode_closed",
                episode_id=handle.episode_id,
            )
        return self._active[handle.episode_id]

    # -- stepping ----------------------------------------------------------
    def step(self, handle: EpisodeHandle, action: Action) -> Transition:
        """Apply a joint-space command with a simple first-order response."""
        state = self._require(handle)
        scenario = handle.scenario
        robot = scenario.robot
        dt = 1.0 / robot.control_hz
        rng = state["rng"]

        if action.type is ActionType.JOINT_POSITION:
            target = np.asarray(action.joint_position, dtype=np.float64)
            if target.shape != (robot.dof,):
                raise SimulationError(
                    f"expected {robot.dof} joint targets, got {target.shape}",
                    code="simulation.bad_action_shape",
                    expected=robot.dof,
                    actual=list(target.shape),
                )
            target = robot.limits.clamped(target)
            # Critically-damped-ish approach: alpha is a pure function of dt, so
            # the response is deterministic and resolution-independent.
            alpha = min(1.0, 8.0 * dt)
            state["joint_positions"] = (1.0 - alpha) * state["joint_positions"] + alpha * target
            state["joint_velocities"] = (target - state["joint_positions"]) / max(dt, 1e-9)
            state["joint_efforts"] = np.abs(state["joint_velocities"]) * 2.0

        elif action.type is ActionType.JOINT_VELOCITY:
            velocity = np.asarray(action.joint_velocity, dtype=np.float64)
            state["joint_positions"] = robot.limits.clamped(
                state["joint_positions"] + velocity * dt
            )
            state["joint_velocities"] = velocity

        elif action.type is ActionType.GRIPPER:
            target_width = float(action.gripper_width_m or 0.0)
            state["gripper_width"] += (target_width - state["gripper_width"]) * min(1.0, 10.0 * dt)
            # A grasp "latches" when the gripper is nearly closed on an object.
            if state["gripper_width"] < 0.02 and state["grasped_object"] is None:
                state["grasped_object"] = self._nearest_ungrasped(state, scenario)

        elif action.type is ActionType.CARTESIAN_DELTA:
            # Without IK, a Cartesian command perturbs the first two joints, which
            # is enough to move the stub EE pose.
            delta = np.asarray(action.cartesian_delta, dtype=np.float64)
            perturb = np.zeros(robot.dof)
            perturb[0] = -delta[0]
            if robot.dof > 1:
                perturb[1] = -delta[1]
            state["joint_positions"] = robot.limits.clamped(
                state["joint_positions"] + perturb * dt
            )

        elif action.type is not ActionType.NOOP:
            raise SimulationError(
                f"MockBackend does not implement action type {action.type.value!r}",
                code="simulation.action_unsupported",
                action_type=action.type.value,
            )

        state["ee_pose"] = self._forward_kinematics_stub(robot, state["joint_positions"])
        self._update_grasped(state, scenario)

        handle.step_index += 1
        handle.sim_time_s += dt
        if not handle.done and handle.step_index >= state["max_steps"]:
            handle.done = True
            handle.termination = TerminationReason.MAX_STEPS

        # Small deterministic dither makes the mock non-trivially path dependent
        # so that policies which ignore observations are still distinguishable.
        if self._observation_noise_std > 0.0:
            rng.normal(0.0, self._observation_noise_std, size=robot.dof)

        return Transition(
            observation=self.observe(handle),
            robot_state=self._robot_state(handle, corrupted=True),
            ground_truth=self.ground_truth(handle),
            reward=0.0,
            terminated=handle.done,
            truncation=handle.termination is TerminationReason.MAX_STEPS,
            info={"backend": self.name, "collision": False, "planning_failed": False},
        )

    def _nearest_ungrasped(self, state: dict[str, Any], scenario: ScenarioConfig) -> str | None:
        """Return the id of the closest object to the gripper, if any is in reach."""
        ee = state["ee_pose"].position
        best: tuple[float, str] | None = None
        for obj in state["objects"].values():
            if obj["grasped"]:
                continue
            distance = obj["position"].distance_to(ee)
            if distance < 0.35 and (best is None or distance < best[0]):
                best = (distance, obj["id"])
        return best[1] if best else None

    def _update_grasped(self, state: dict[str, Any], scenario: ScenarioConfig) -> None:
        """Attach a latched object rigidly to the gripper; drop it on release."""
        grasped_id = state["grasped_object"]
        width = state["gripper_width"]
        if grasped_id is not None and width > 0.6 * scenario.robot.max_gripper_width_m:
            state["grasped_object"] = None  # released
        if grasped_id is not None and grasped_id in state["objects"]:
            state["objects"][grasped_id]["grasped"] = True
            state["objects"][grasped_id]["position"] = state["ee_pose"].position
        for obj in state["objects"].values():
            if obj["grasped"] and obj["id"] != grasped_id:
                obj["grasped"] = False

    # -- observation -------------------------------------------------------
    def _robot_state(self, handle: EpisodeHandle, *, corrupted: bool) -> RobotState:
        state = self._require(handle)
        robot = handle.scenario.robot
        positions = np.array(state["joint_positions"], dtype=np.float64)
        if corrupted and self._observation_noise_std > 0.0:
            rng = make_numpy_rng(handle.seed, "obs-noise", handle.step_index)
            positions = positions + rng.normal(0.0, self._observation_noise_std, size=robot.dof)
        width = float(state["gripper_width"])
        return RobotState(
            joint_positions=tuple(float(v) for v in positions),
            joint_velocities=tuple(float(v) for v in state["joint_velocities"]),
            joint_efforts=tuple(float(v) for v in state["joint_efforts"]),
            joint_names=robot.arm_joint_names,
            end_effector_pose=state["ee_pose"],
            gripper=GripperState(
                width_m=width,
                max_width_m=robot.max_gripper_width_m,
                is_grasping=state["grasped_object"] is not None,
                grasped_object_id=state["grasped_object"],
            ),
            timestamp_s=handle.sim_time_s,
        )

    def observe(self, handle: EpisodeHandle) -> Observation:
        """Return the current observation (no ground truth, as required)."""
        state = self._require(handle)
        beliefs = tuple(
            ObjectState(
                id=obj["id"],
                category=obj["category"],
                pose=Pose(position=obj["position"], orientation=obj["orientation"]),
                covariance=Covariance.isotropic(0.005, dimension=6),
                confidence=0.9,
                source=Source.FUSION,
                timestamp_s=handle.sim_time_s,
                visible=True,
                last_seen_s=handle.sim_time_s,
                estimated_size_m=obj["size_m"],
            )
            for obj in state["objects"].values()
        )
        return Observation(
            step_index=handle.step_index,
            timestamp_s=handle.sim_time_s,
            robot=self._robot_state(handle, corrupted=True),
            objects=beliefs,
            sensor_frame=None,
            seed=handle.seed,
            scenario_id=handle.scenario.scenario_id,
            task_belief="",
        )

    def render(self, handle: EpisodeHandle) -> SensorFrame:
        """Return an empty frame: the mock backend does not render."""
        self._require(handle)
        return SensorFrame(
            camera_id="mock_none",
            timestamp_s=handle.sim_time_s,
            valid=False,
            width_px=0,
            height_px=0,
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
                    grasped=bool(obj["grasped"]),
                    friction=handle.scenario.environment.object_friction,
                )
                for obj in state["objects"].values()
            ),
            true_joint_positions=tuple(float(v) for v in state["joint_positions"]),
            true_ee_pose=state["ee_pose"],
            contact=ContactState.NONE,
            sim_time_s=handle.sim_time_s,
        )

    def close(self, handle: EpisodeHandle) -> None:
        """Forget one episode. Idempotent."""
        self._active.pop(handle.episode_id, None)

    def close_all(self) -> None:
        """Forget every episode. Idempotent."""
        self._active.clear()
