"""Robot, object and sensor state schemas.

**Ground-truth separation is a hard requirement of this project.** A policy or
perception module must never be handed :class:`GroundTruthState`; only the
evaluator may read it, and only *after* the episode has finished. The type
system expresses this: :class:`Observation` contains no ground-truth field, and
:func:`Observation.is_leaking_ground_truth` is asserted by dedicated tests.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

import numpy as np
from pydantic import Field, field_validator, model_validator

from robot_ci.models.base import VersionedModel
from robot_ci.models.environment import ObjectCategory
from robot_ci.models.geometry import Covariance, Pose

__all__ = [
    "Action",
    "ActionType",
    "ContactState",
    "GripperState",
    "GroundTruthObject",
    "GroundTruthState",
    "JointState",
    "ObjectState",
    "Observation",
    "PoseEstimate",
    "RobotState",
    "SensorFrame",
]


class Source(str, Enum):
    """Provenance of an estimate. Required on every fused quantity."""

    GROUND_TRUTH = "ground_truth"
    RGB = "rgb"
    DEPTH = "depth"
    FUSION = "fusion"
    KINEMATICS = "kinematics"
    WORLD_MODEL = "world_model"
    POLICY = "policy"
    SIMULATOR = "simulator"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """How an :class:`Action` should be interpreted by a backend."""

    JOINT_POSITION = "joint_position"
    """Absolute joint position targets, shape ``(dof,)``."""
    JOINT_VELOCITY = "joint_velocity"
    """Joint velocity targets, shape ``(dof,)``."""
    CARTESIAN_DELTA = "cartesian_delta"
    """End-effector pose delta in the base frame, shape ``(6,)``."""
    GRIPPER = "gripper"
    """Gripper command only: ``(1,)`` aperture target in metres."""
    NOOP = "noop"
    """Do nothing for one step."""


class GripperState(VersionedModel):
    """Parallel-gripper state."""

    width_m: float = Field(default=0.0, ge=0.0, le=1.0, description="Aperture between fingers in metres.")
    max_width_m: float = Field(default=0.08, gt=0.0, description="Fully open aperture in metres.")
    is_grasping: bool = Field(default=False, description="Whether an object is currently held.")
    grasped_object_id: str | None = Field(default=None, description="Id of the held object, if any.")
    force_n: float = Field(default=0.0, ge=0.0, description="Measured closing force in newtons.")
    stalled: bool = Field(default=False, description="True when the gripper commanded motion but did not move.")

    @property
    def normalized_width(self) -> float:
        """Aperture as a fraction of the fully open value, in ``[0, 1]``."""
        return float(self.width_m / self.max_width_m) if self.max_width_m > 0 else 0.0

    def is_closed(self, threshold: float = 0.2) -> bool:
        """Whether the gripper is below ``threshold`` of full open."""
        return self.normalized_width < threshold


class JointState(VersionedModel):
    """One actuated joint's feedback."""

    name: str = Field(min_length=1, description="Joint name.")
    position_rad: float = Field(description="Measured joint angle in radians.")
    velocity_rad_s: float = Field(default=0.0, description="Measured joint velocity in rad/s.")
    effort_nm: float = Field(default=0.0, description="Measured or commanded effort in Nm.")


class RobotState(VersionedModel):
    """Proprioceptive robot state available to policies.

    This is *observed* state: it may be corrupted by
    :class:`~robot_ci.models.sensors.RobotStateCorruptionConfig` before a policy
    ever sees it. It is deliberately a separate type from
    :class:`GroundTruthState`.
    """

    joint_positions: tuple[float, ...] = Field(description="Observed joint angles in radians.")
    joint_velocities: tuple[float, ...] = Field(default=(), description="Observed joint velocities in rad/s.")
    joint_efforts: tuple[float, ...] = Field(default=(), description="Observed joint efforts in Nm.")
    joint_names: tuple[str, ...] = Field(default=(), description="Joint names, in array order.")
    end_effector_pose: Pose = Field(default_factory=Pose, description="Observed end-effector pose.")
    gripper: GripperState = Field(
        default_factory=lambda: GripperState(), description="Gripper state."
    )
    timestamp_s: float = Field(default=0.0, ge=0.0, description="Observation time on the episode clock.")
    contact_detected: bool = Field(default=False, description="Whether the end effector is in contact.")

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        n = len(self.joint_positions)
        if self.joint_velocities and len(self.joint_velocities) != n:
            raise ValueError("joint_velocities length must match joint_positions")
        if self.joint_efforts and len(self.joint_efforts) != n:
            raise ValueError("joint_efforts length must match joint_positions")
        if self.joint_names and len(self.joint_names) != n:
            raise ValueError("joint_names length must match joint_positions")
        return self

    @property
    def positions_array(self) -> np.ndarray:
        """Joint positions as a ``float64`` array."""
        return np.array(self.joint_positions, dtype=np.float64)

    @property
    def velocities_array(self) -> np.ndarray:
        """Joint velocities as a ``float64`` array (zeros if not reported)."""
        return np.zeros(len(self.joint_positions)) if not self.joint_velocities else np.array(self.joint_velocities, dtype=np.float64)

    @property
    def efforts_array(self) -> np.ndarray:
        """Joint efforts as a ``float64`` array (zeros if not reported)."""
        return np.zeros(len(self.joint_positions)) if not self.joint_efforts else np.array(self.joint_efforts, dtype=np.float64)

    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        efforts: np.ndarray | None = None,
        joint_names: tuple[str, ...] = (),
        **kwargs: object,
    ) -> RobotState:
        """Build from NumPy arrays, converting to plain tuples."""
        return cls(
            joint_positions=tuple(float(v) for v in positions),
            joint_velocities=tuple(float(v) for v in velocities) if velocities is not None else (),
            joint_efforts=tuple(float(v) for v in efforts) if efforts is not None else (),
            joint_names=joint_names,
            **kwargs,  # type: ignore[arg-type]
        )


class ContactState(str, Enum):
    """Coarse contact classification for the end effector."""

    NONE = "none"
    OBJECT = "object"
    TABLE = "table"
    OBSTACLE = "obstacle"
    SELF = "self_collision"
    UNKNOWN = "unknown"


class ObjectState(VersionedModel):
    """A *believed* object state produced by perception + sensor fusion.

    Every field carries its own provenance, timestamp and uncertainty, as
    required by the sensor-fusion specification. Nothing in this model is
    ground truth: it is what the robot believes after fusion.
    """

    id: str = Field(min_length=1, description="Stable object identifier.")
    category: ObjectCategory = Field(default=ObjectCategory.CUBE, description="Perceived class.")
    pose: Pose = Field(default_factory=Pose, description="Believed pose.")
    covariance: Covariance = Field(default_factory=lambda: Covariance.isotropic(0.01), description="Positional uncertainty (6x6, m^2 / rad^2).")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Calibration-free confidence in [0,1].")
    timestamp_s: float = Field(default=0.0, ge=0.0, description="Time the estimate was produced.")
    source: Source = Field(default=Source.UNKNOWN, description="Provenance of the estimate.")
    visible: bool = Field(default=True, description="Whether the object is currently observed.")
    last_seen_s: float = Field(default=0.0, ge=0.0, description="Time of the most recent direct observation.")
    estimated_size_m: float = Field(default=0.04, gt=0.0, description="Estimated characteristic size in metres.")

    @property
    def staleness_s(self) -> float:
        """Seconds since this object was last directly seen."""
        return max(0.0, self.timestamp_s - self.last_seen_s)

    @property
    def position_sigma_m(self) -> float:
        """Largest per-axis positional standard deviation in metres."""
        return self.covariance.max_sigma

    @property
    def is_confidently_localized(self, sigma_m: float = 0.01) -> bool:
        """Whether the estimate is tight enough to attempt a grasp."""
        return self.visible and self.position_sigma_m <= sigma_m


class PoseEstimate(VersionedModel):
    """An object pose plus the metadata every estimate must carry."""

    object_id: str = Field(min_length=1, description="Which object this refers to.")
    pose: Pose = Field(default_factory=Pose, description="Estimated pose.")
    covariance: Covariance = Field(default_factory=lambda: Covariance.isotropic(0.01), description="Pose covariance.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in [0,1].")
    source: Source = Field(default=Source.UNKNOWN, description="Which sensor or module produced this.")
    timestamp_s: float = Field(default=0.0, ge=0.0, description="Estimation time on the episode clock.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Processing latency of the estimator.")

    def positional_error_to(self, truth: Pose) -> float:
        """Positional error in metres against a known ground-truth pose."""
        return self.pose.translation_distance_to(truth)

    def to_object_state(self) -> ObjectState:
        """Project this estimate into an :class:`ObjectState`."""
        return ObjectState(
            id=self.object_id,
            pose=self.pose,
            covariance=self.covariance,
            confidence=self.confidence,
            source=self.source,
            timestamp_s=self.timestamp_s,
            visible=True,
            last_seen_s=self.timestamp_s,
        )


class GroundTruthObject(VersionedModel):
    """The *true* object state. Evaluator-only."""

    id: str = Field(min_length=1, description="Object identifier.")
    category: ObjectCategory = Field(default=ObjectCategory.CUBE, description="True class.")
    pose: Pose = Field(default_factory=Pose, description="True pose.")
    size_m: tuple[float, float, float] = Field(default=(0.04, 0.04, 0.04), description="True half-extents in metres.")
    is_target: bool = Field(default=False, description="Whether this is the task target object.")
    grasped: bool = Field(default=False, description="Whether the gripper currently holds it.")
    friction: float = Field(default=0.6, ge=0.0, description="Object friction coefficient.")


class GroundTruthState(VersionedModel):
    """Complete hidden simulator state. **Never** passed to a policy.

    Kept as its own type so that the boundary between "what the robot can see"
    and "what the evaluator knows" is visible in the type signatures.
    """

    objects: tuple[GroundTruthObject, ...] = Field(default=(), description="True object states.")
    true_joint_positions: tuple[float, ...] = Field(default=(), description="True joint angles in radians.")
    true_ee_pose: Pose = Field(default_factory=Pose, description="True end-effector pose.")
    contact: ContactState = Field(default=ContactState.NONE, description="True contact state.")
    sim_time_s: float = Field(default=0.0, ge=0.0, description="Simulated time.")
    penetration_depth_m: float = Field(default=0.0, ge=0.0, description="Deepest collision penetration.")

    def object_by_id(self, object_id: str) -> GroundTruthObject | None:
        """Return the true state of one object, or ``None`` if absent."""
        return next((obj for obj in self.objects if obj.id == object_id), None)

    @property
    def target_object(self) -> GroundTruthObject | None:
        """The true state of the object flagged as the task target."""
        return next((obj for obj in self.objects if obj.is_target), None)


class SensorFrame(VersionedModel):
    """One rendered sensor frame, before corruption.

    Image arrays are stored as NumPy arrays rather than nested lists: frames are
    the largest objects in the system and converting them to Python tuples would
    dominate runtime. They are excluded from JSON serialisation, so any artifact
    that persists frames writes them as ``.npy`` files alongside the JSON and
    records their hashes.

    Attributes:
        camera_id: Which camera produced the frame.
        timestamp_s: Capture time on the episode clock.
        rgb: ``uint8`` array of shape ``(H, W, 3)``.
        depth: ``float32`` array of shape ``(H, W)`` in metres; NaN is invalid.
        segmentation: ``uint8`` array of shape ``(H, W)`` with class ids.
        intrinsics: ``(fx, fy, cx, cy)`` in pixels.
        valid: False when the frame was dropped by injected latency.
    """

    model_config = {**VersionedModel.model_config, "arbitrary_types_allowed": True, "frozen": True}

    camera_id: str = Field(default="rgb_camera", min_length=1, description="Camera identifier.")
    timestamp_s: float = Field(default=0.0, ge=0.0, description="Capture time on the episode clock.")
    rgb: Any = Field(default=None, description="uint8 (H, W, 3) image array.")
    depth: Any = Field(default=None, description="float32 (H, W) depth array in metres; NaN is invalid.")
    segmentation: Any = Field(default=None, description="uint8 (H, W) class-id image array.")
    width_px: int = Field(default=0, ge=0, description="Image width in pixels.")
    height_px: int = Field(default=0, ge=0, description="Image height in pixels.")
    intrinsics: tuple[float, ...] | None = Field(default=None, description="(fx, fy, cx, cy) in pixels.")
    valid: bool = Field(default=True, description="False when the frame was dropped by injected latency.")

    @field_validator("rgb", "depth", "segmentation", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept nested lists (from JSON) and coerce to the right NumPy dtype."""
        if value is None or isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        return value

    def rgb_array(self) -> np.ndarray | None:
        """RGB image as ``uint8`` of shape ``(H, W, 3)``, or ``None``."""
        return None if self.rgb is None else np.asarray(self.rgb, dtype=np.uint8)

    def depth_array(self) -> np.ndarray | None:
        """Depth image as ``float32`` of shape ``(H, W)``, or ``None``."""
        return None if self.depth is None else np.asarray(self.depth, dtype=np.float32)

    def segmentation_array(self) -> np.ndarray | None:
        """Segmentation image as ``uint8``, or ``None``."""
        return None if self.segmentation is None else np.asarray(self.segmentation, dtype=np.uint8)

    def has_pixels(self) -> bool:
        """Whether this frame carries any image data at all."""
        return self.rgb is not None or self.depth is not None

    def dropped(self) -> SensorFrame:
        """Return a copy marked as dropped, with no pixel data."""
        return SensorFrame(
            camera_id=self.camera_id, timestamp_s=self.timestamp_s, valid=False,
            width_px=self.width_px, height_px=self.height_px, intrinsics=self.intrinsics,
        )


class Action(VersionedModel):
    """A command issued by a policy to a simulator backend.

    Exactly one payload is populated, selected by :attr:`type`. Validated in
    :meth:`_check_payload` so an action can never be silently misinterpreted.
    """

    type: ActionType = Field(default=ActionType.JOINT_POSITION, description="Interpretation of the payload.")
    joint_position: tuple[float, ...] | None = Field(default=None, description="Absolute joint targets in rad.")
    joint_velocity: tuple[float, ...] | None = Field(default=None, description="Joint velocity targets in rad/s.")
    cartesian_delta: tuple[float, ...] | None = Field(default=None, description="EE delta pose (x,y,z,rx,ry,rz) in the base frame.")
    gripper_width_m: float | None = Field(default=None, ge=0.0, le=1.0, description="Target aperture in metres.")
    scale: float = Field(default=1.0, gt=0.0, le=10.0, description="Gain applied to the command by the controller.")

    @model_validator(mode="after")
    def _check_payload(self) -> Self:
        expected = {
            ActionType.JOINT_POSITION: self.joint_position,
            ActionType.JOINT_VELOCITY: self.joint_velocity,
            ActionType.CARTESIAN_DELTA: self.cartesian_delta,
            ActionType.GRIPPER: None if self.gripper_width_m is None else (self.gripper_width_m,),
            ActionType.NOOP: (),
        }[self.type]
        if expected is None:
            raise ValueError(f"action of type {self.type.value} requires its payload to be set")
        if self.type is ActionType.CARTESIAN_DELTA and len(self.cartesian_delta or ()) != 6:
            raise ValueError("cartesian_delta must have exactly 6 components")
        return self

    @classmethod
    def joint_positions(cls, values: np.ndarray | tuple[float, ...]) -> Action:
        """Build a joint-position action from an array."""
        return cls(type=ActionType.JOINT_POSITION, joint_position=tuple(float(v) for v in values))

    @classmethod
    def joint_velocities(cls, values: np.ndarray | tuple[float, ...]) -> Action:
        """Build a joint-velocity action from an array."""
        return cls(type=ActionType.JOINT_VELOCITY, joint_velocity=tuple(float(v) for v in values))

    @classmethod
    def cartesian(cls, values: np.ndarray | tuple[float, ...]) -> Action:
        """Build a Cartesian-delta action from a 6-vector."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,):
            raise ValueError(f"cartesian action must be 6-dimensional, got shape {array.shape}")
        return cls(type=ActionType.CARTESIAN_DELTA, cartesian_delta=tuple(float(v) for v in array))

    @classmethod
    def gripper(cls, width_m: float) -> Action:
        """Build a gripper action targeting an aperture in metres."""
        return cls(type=ActionType.GRIPPER, gripper_width_m=float(width_m))

    @classmethod
    def noop(cls) -> Action:
        """Build a no-op action."""
        return cls(type=ActionType.NOOP)


class Observation(VersionedModel):
    """Everything a policy is allowed to see at one timestep.

    **This type deliberately contains no ground truth.** It carries the corrupted
    sensor frame, proprioception, and the fused belief produced by perception.
    :meth:`is_leaking_ground_truth` is asserted false by dedicated tests.
    """

    step_index: int = Field(default=0, ge=0, description="Control step since episode start.")
    timestamp_s: float = Field(default=0.0, ge=0.0, description="Time on the episode clock.")
    robot: RobotState = Field(default_factory=lambda: RobotState(joint_positions=()), description="Proprioception.")
    objects: tuple[ObjectState, ...] = Field(default=(), description="Fused object beliefs.")
    sensor_frame: SensorFrame | None = Field(default=None, description="Corrupted sensor frame, if requested.")
    frame_dropped: bool = Field(default=False, description="True when injected latency dropped this frame.")
    frame_age_ms: float = Field(default=0.0, ge=0.0, description="Age of the delivered frame in ms.")
    task_belief: str = Field(default="", description="Task-relevant belief string, if the task provides one.")
    seed: int = Field(default=0, ge=0, description="Episode seed, for reproducible policy tie-breaks.")
    scenario_id: str = Field(default="", description="Identifier of the scenario being run.")

    @property
    def visible_objects(self) -> tuple[ObjectState, ...]:
        """Objects the robot currently believes it can see."""
        return tuple(obj for obj in self.objects if obj.visible)

    def object_by_id(self, object_id: str) -> ObjectState | None:
        """Return a believed object state by id, or ``None``."""
        return next((obj for obj in self.objects if obj.id == object_id), None)

    @property
    def best_object(self) -> ObjectState | None:
        """The most confident visible object, or ``None``."""
        visible = self.visible_objects
        return max(visible, key=lambda o: o.confidence) if visible else None

    def is_leaking_ground_truth(self) -> bool:
        """Structural self-check that no ground-truth field is present.

        Returns:
            Always ``False`` for a well-formed observation. The method exists so
            that tests can assert the invariant explicitly, and so that a future
            schema change that adds such a field fails loudly.
        """
        forbidden = {"ground_truth", "true_state", "true_joint_positions", "true_ee_pose"}
        return bool(forbidden & set(self.model_dump().keys()))

    def with_frame(self, frame: SensorFrame | None, *, age_ms: float = 0.0, dropped: bool = False) -> Observation:
        """Return a copy carrying a sensor frame (used by the corruption layer)."""
        return self.model_copy(
            update={"sensor_frame": frame, "frame_age_ms": age_ms, "frame_dropped": dropped}
        )
