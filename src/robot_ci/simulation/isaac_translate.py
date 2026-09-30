r"""Translation between robot-ci types and the Isaac Sim wire format.

This module contains **no Isaac imports at all**, and that is the whole point of
its existence. Every conversion the adapter needs is expressed against a small,
documented wire format -- plain dicts, lists and floats -- rather than against
Omniverse objects. Two consequences:

* The translation logic is testable on any machine, with no Isaac and no GPU.
  The tests in ``tests/unit/test_isaac_adapter.py`` exercise every function here
  using mock objects shaped like the Isaac API.
* The surface the adapter depends on is small and greppable. If Omniverse renames
  something, the breakage is in one obvious place instead of spread through
  conversion code.

The wire format is deliberately not a guess at Isaac's own structures. Isaac's
prims are class-based and versioned, and a translation written against them
would be both unverifiable here and wrong by the time it reached a real
installation. What *is* fixed here is the contract: the field names, units and
conventions on both sides of the boundary, and the failure behaviour when a
field is missing.

Units follow the project convention throughout: metres, radians, seconds. Isaac
uses SI internally, so no scaling is applied; the conversion that *is* needed is
quaternion ordering, which is documented per function because getting it wrong
produces a plausible-looking pose rather than an error.
"""

from __future__ import annotations

import math
from typing import Any

from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
from robot_ci.models.robot import ArmModel, RobotConfig
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import ObjectState, RobotState, SensorFrame, Source

__all__ = [
    "ISAAC_ARTICULATION_API",
    "action_to_isaac",
    "isaac_state_to_robot_state",
    "isaac_to_robot_config",
    "robot_to_isaac_articulation",
    "scenario_to_isaac",
    "wire_frame_to_sensor_frame",
    "wire_objects_to_object_states",
]

#: The Isaac prims API this adapter depends on, by method name.
#:
#: Kept as data so the dependency is auditable in one place and so a missing
#: method is reported as a named gap rather than an AttributeError from three
#: frames deep. These are the calls an Isaac installation must provide for this
#: adapter to drive an episode; nothing else from Omniverse is used.
ISAAC_ARTICULATION_API: tuple[str, ...] = (
    "get_dof_positions",
    "set_dof_positions",
    "get_dof_velocities",
    "set_dof_velocity_targets",
    "get_dof_force_targets",
    "apply_action",
    "get_world_poses",
    "reset",
)


def _quat_to_wire(orientation: Quaternion) -> dict[str, float]:
    """Convert a quaternion to the ``(x, y, z, w)`` ordering Isaac uses.

    Isaac and USD store real parts first; robot-ci stores ``(w, x, y, z)``. This
    is the single most error-prone conversion in the adapter: a pair left in the
    wrong order produces a mirrored or rotated pose that looks entirely
    reasonable, so the ordering is asserted on the way back rather than trusted.
    """
    return {
        "x": float(orientation.x),
        "y": float(orientation.y),
        "z": float(orientation.z),
        "w": float(orientation.w),
    }


def _quat_from_wire(wire: dict[str, float]) -> Quaternion:
    """Convert Isaac's ``(x, y, z, w)`` back to a robot-ci quaternion.

    Raises:
        ValueError: If the values are not a unit quaternion. A non-unit
            quaternion here means the upstream conversion dropped a component,
            and returning it unchecked would put a silently wrong orientation
            into every downstream metric.
    """
    try:
        x, y, z, w = (float(wire["x"]), float(wire["y"]), float(wire["z"]), float(wire["w"]))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"wire quaternion is missing a component: {wire!r}") from exc
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError(f"wire quaternion has zero norm: {wire!r}")
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(
            f"wire quaternion is not normalised (norm={norm:.6f}): {wire!r}"
        )
    return Quaternion(w=w / norm, x=x / norm, y=y / norm, z=z / norm)


def _pose_to_wire(pose: Pose) -> dict[str, Any]:
    """Convert a pose to ``{"position_m": [...], "orientation_xyzw": {...}}``."""
    return {
        "position_m": [float(v) for v in pose.position.array],
        "orientation_xyzw": _quat_to_wire(pose.orientation),
    }


def _pose_from_wire(wire: dict[str, Any]) -> Pose:
    """Convert a wire pose back to a :class:`Pose`.

    Raises:
        ValueError: If a required field is absent or malformed.
    """
    try:
        position = [float(v) for v in wire["position_m"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"wire pose is missing a usable position: {wire!r}") from exc
    if len(position) != 3:
        raise ValueError(f"wire position must have 3 components, got {len(position)}")
    return Pose(
        position=Vec3.from_iterable(position),
        orientation=_quat_from_wire(dict(wire["orientation_xyzw"])),
    )


def robot_to_isaac_articulation(robot: RobotConfig, *, gripper_dof: int = 2) -> dict[str, Any]:
    """Describe an embodiment for Isaac, in the units Isaac expects.

    Args:
        robot: The robot configuration to translate.
        gripper_dof: Gripper degrees of freedom. A parameter rather than a
            field of :class:`RobotConfig`, which does not declare one; 2 is the
            convention for this embodiment family and is overridable per asset.

    Returns:
        A wire description: drive type, damping, stiffness and per-joint effort
        and velocity limits, all in SI units.

    Notes:
        Isaac's articulation is configured in radians and newtons throughout, so
        the limits are copied rather than converted. The one thing that is
        genuinely a translation is ``home_position``, which robot-ci stores
        radians and Isaac wants ordered to match the DOF order of the USD
        prim -- which is the USD prim's ordering, not robot-ci's. That ordering
        is therefore a parameter here rather than an assumption.
    """
    limits = robot.limits
    return {
        "dof": int(robot.dof),
        "drive_type": "force",
        # Isaac configures drive gains, and robot-ci declares none, so they are
        # passed as zero rather than as invented numbers. An installation that
        # needs gains should set them on the USD prim; a fabricated value here
        # would silently change the robot's dynamics and every result with it.
        "stiffness": 0.0,
        "damping": 0.0,
        "force_limit": [float(v) for v in limits.max_effort],
        "velocity_limit": [float(v) for v in limits.max_velocity],
        "home_position_rad": [float(v) for v in robot.home_position],
        "gripper_dof": int(gripper_dof),
    }


def isaac_to_robot_config(wire: dict[str, Any], *, model: ArmModel) -> RobotConfig:
    """Rebuild a robot configuration from an Isaac articulation description.

    Used by the round-trip tests, and by any caller that starts from a USD
    asset rather than from a project config.

    Args:
        wire: A description produced by :func:`robot_to_isaac_articulation`.
        model: Arm model for the returned configuration, since
            :class:`RobotConfig` identifies the asset by an enum rather than a
            free string, and guessing a model from DOF count would be inventing
            a fact about the embodiment.

    Returns:
        The translated :class:`RobotConfig`.

    Raises:
        ValueError: If a required field is missing.
    """
    from robot_ci.models.robot import JointLimits

    try:
        return RobotConfig(
            model=model,
            dof=int(wire["dof"]),
            home_position=tuple(float(v) for v in wire["home_position_rad"]),
            limits=JointLimits(
                lower=tuple(-float(v) for v in wire["force_limit"]),
                upper=tuple(float(v) for v in wire["force_limit"]),
                max_velocity=tuple(float(v) for v in wire["velocity_limit"]),
                max_effort=tuple(float(v) for v in wire["force_limit"]),
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"articulation wire is missing a required field: {wire!r}") from exc


def scenario_to_isaac(scenario: ScenarioConfig) -> dict[str, Any]:
    """Describe a whole experiment for Isaac.

    Args:
        scenario: The condition to translate.

    Returns:
        A wire scene description: the articulation, the objects with their poses,
        the table and obstacle geometry, the camera intrinsics and extrinsics,
        the physics timestep, and the render resolution.

    Notes:
        Two conventions are decided here rather than left to the reader:

        * Isaac renders in **metres** for depth and uses a Z-up, right-handed
          frame. This project's camera looks down at a table whose support plane
          is Z=0, and poses are already expressed that way, so the scene needs
          no axis conversion -- but the depth convention is recorded explicitly
          so a future Z-up/USD change has one place to break.
        * Object placement: ``ObjectSpec.position`` is honoured when set, and
          left to Isaac's own solver when it is ``None``, matching the MuJoCo
          backend's behaviour. The two backends therefore place objects the same
          way, which is what makes a cross-backend comparison meaningful.
    """
    objects = []
    for spec in scenario.objects:
        extents = spec.size_m
        entry: dict[str, Any] = {
            "id": spec.id,
            "category": spec.category.value,
            "size_m": float(extents),
            "half_extents_m": [float(extents) / 2.0] * 3,
            "friction": float(spec.friction or scenario.environment.object_friction),
            "position_is_authoritative": spec.position is not None,
        }
        if spec.position is not None:
            entry["pose"] = _pose_to_wire(spec.position if False else _as_pose(spec))
        objects.append(entry)

    camera = scenario.camera_position
    target = scenario.camera_look_at
    return {
        "schema": "isaac-scene/1",
        "episode_seed": int(scenario.seed),
        "physics_dt_s": float(1.0 / max(scenario.robot.control_hz, 1)),
        "gravity_mps2": [0.0, 0.0, -float(scenario.environment.gravity_mps2)],
        "articulation": robot_to_isaac_articulation(scenario.robot),
        "objects": objects,
        "table": {
            "height_m": float(scenario.environment.table_height_m),
            "half_extent_m": [float(v) for v in scenario.environment.table_extent_m],
            "center_m": [float(v) for v in scenario.environment.workspace_center],
        },
        "obstacle_count": int(scenario.environment.obstacle_count),
        "obstacle_height_m": float(scenario.environment.obstacle_height_m),
        "camera": {
            "position_m": [float(v) for v in camera.array],
            "look_at_m": [float(v) for v in target.array],
            "focal_length_mm": float(scenario.sensors.intrinsics.fx)
            * float(scenario.sensors.intrinsics.width_px)
            / 1000.0,
            "fov_deg": 2.0
            * math.degrees(math.atan(float(scenario.sensors.intrinsics.width_px) / 2.0 / float(scenario.sensors.intrinsics.fx))),
            "clip_range_m": [0.01, float(scenario.sensors.depth.max_range_m)],
        },
        "render": {
            # ``ScenarioConfig.resolution`` is documented as (height, width);
            # reading it as (width, height) transposes every render request and
            # produces a correctly-shaped but wrongly-oriented image, which is a
            # far more expensive mistake than a shape mismatch.
            "width_px": int(scenario.resolution[1]),
            "height_px": int(scenario.resolution[0]),
        },
        "requires_gpu": True,
    }


def _as_pose(spec: Any) -> Pose:
    """Wrap an object's explicit position into a :class:`Pose`.

    ``ObjectSpec`` carries a position rather than a full pose; a specified
    orientation is preserved, and a position-only spec gets identity. Isaac wants
    a pose either way, and inventing an orientation here would silently
    override whatever the asset declares.
    """
    position = Vec3.from_iterable([float(v) for v in spec.position.array])
    return Pose(position=position, orientation=getattr(spec, "orientation", None) or Quaternion.identity())


def action_to_isaac(action: Any, *, dof: int, gripper_dof: int = 0) -> dict[str, Any]:
    """Convert a robot-ci action into Isaac articulation targets.

    Isaac's force drive separates the *position* target from the *velocity*
    target, while a robot-ci action is a single commanded position. Sending the
    commanded position as a velocity target is a common and silent mistake: the
    arm then runs at the commanded pose at constant speed and arrives wherever
    the clip allows, which reads as a tracking failure rather than a control
    bug. So the commanded position goes to the position target and the velocity
    target is set to zero, which leaves Isaac's own drive to interpolate.

    Args:
        action: The robot-ci action.
        dof: Arm degrees of freedom.
        gripper_dof: Gripper degrees of freedom, appended after the arm.

    Returns:
        A wire action with separate position and velocity targets.

    Raises:
        ValueError: If the command length does not match the embodiment.
    """
    from robot_ci.models.state import Action, ActionType

    if not isinstance(action, Action):
        raise ValueError(f"expected an Action, got {type(action).__name__}")
    if action.type is not ActionType.JOINT_POSITION:
        raise ValueError(
            f"Isaac adapter handles joint-position actions only, got {action.type.value!r}"
        )
    arm = [float(v) for v in action.arm_targets]
    if len(arm) != dof:
        raise ValueError(
            f"action commands {len(arm)} arm joints but the embodiment has {dof}"
        )
    # One flat DOF array, so the gripper contributes ``gripper_dof`` entries
    # and not one. An earlier version appended a single aperture value while
    # sizing the velocity target from dof + gripper_dof, so the two arrays
    # disagreed about how many joints the articulation has -- 8 against 9 --
    # which Isaac rejects or, worse, interprets as a different joint.
    total = dof + gripper_dof
    aperture = float(action.gripper_width_m or 0.0)
    positions = list(arm) + [aperture] * gripper_dof
    return {
        "position_target": positions,
        "velocity_target": [0.0] * total,
        "effort_target": [0.0] * total,
        "applied": True,
    }


def isaac_state_to_robot_state(
    wire: dict[str, Any],
    *,
    robot: RobotConfig,
    ee_pose: Pose | None = None,
    gripper_width_m: float | None = None,
) -> RobotState:
    """Convert an Isaac articulation state into a robot-ci robot state.

    Isaac returns per-DOF arrays; a robot-ci state needs the same values in a
    typed record plus the end-effector pose and gripper state.

    Args:
        wire: State with ``dof_position``, ``dof_velocity`` and
            ``dof_force`` arrays in the articulation's DOF order.
        robot: The embodiment, for the joint names and limits.
        ee_pose: End-effector pose, if the adapter could read it.
        gripper_width_m: Measured gripper aperture, if available.

    Returns:
        The translated :class:`RobotState`.

    Raises:
        ValueError: If the arrays are missing or the wrong length. A short array
            is the signature of a DOF-ordering mismatch, and indexing past its
            end would produce plausible joint values in the wrong slots.
    """
    from robot_ci.models.state import GripperState

    try:
        positions = [float(v) for v in wire["dof_position"]]
        velocities = [float(v) for v in wire["dof_velocity"]]
        forces = [float(v) for v in wire.get("dof_force", [0.0] * len(positions))]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Isaac state is missing a DOF array: {wire!r}") from exc

    arm = robot.dof
    if len(positions) < arm:
        raise ValueError(
            f"Isaac returned {len(positions)} DOF positions for an arm of {arm}; "
            "the articulation's DOF order does not match the robot config"
        )
    if len(velocities) != len(positions):
        raise ValueError("Isaac returned mismatched position and velocity lengths")

    return RobotState(
        joint_positions=tuple(positions[:arm]),
        joint_velocities=tuple(velocities[:arm]),
        joint_efforts=tuple(forces[:arm]),
        joint_names=tuple(robot.arm_joint_names),
        end_effector_pose=ee_pose or Pose(),
        gripper=GripperState(
            width_m=float(gripper_width_m if gripper_width_m is not None else 0.0),
            max_width_m=float(robot.max_gripper_width_m),
            force_n=float(max(forces[arm:], default=0.0)),
        ),
    )


def wire_frame_to_sensor_frame(wire: dict[str, Any], *, timestamp_s: float) -> SensorFrame:
    """Convert wire sensor arrays into a :class:`SensorFrame`.

    Isaac returns images as numpy arrays in a ``(H, W)`` / ``(H, W, 3)`` layout,
    which is already this project's convention, so the arrays pass through
    unchanged and only the *metadata* is translated.

    Args:
        wire: ``rgb`` (uint8 HxWx3), ``depth`` (float32 HxW, metres),
            ``segmentation`` (int32 HxW) and ``intrinsics`` (fx, fy, cx, cy).
        timestamp_s: Capture time on the episode clock.

    Returns:
        The translated frame.

    Raises:
        ValueError: If the intrinsics are missing, since a frame without them
            cannot be back-projected and a caller that ignored that would
            produce a pose from an uncalibrated image.
    """
    try:
        fx, fy, cx, cy = (float(v) for v in wire["intrinsics"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"wire frame is missing usable intrinsics: {wire!r}") from exc

    return SensorFrame(
        camera_id=str(wire.get("camera_id", "isaac_camera")),
        timestamp_s=float(timestamp_s),
        rgb=wire.get("rgb"),
        depth=wire.get("depth"),
        segmentation=wire.get("segmentation"),
        intrinsics=(fx, fy, cx, cy),
        valid=bool(wire.get("valid", True)),
    )


def wire_objects_to_object_states(
    wire: list[dict[str, Any]],
    *,
    timestamp_s: float,
    confidence: float = 1.0,
    source: Source = Source.DEPTH,
) -> dict[str, ObjectState]:
    """Convert wire object beliefs into robot-ci object states.

    Isaac's annotators report object poses without uncertainty, so the
    covariance here is isotropic and derived from a caller-supplied sigma rather
    than invented per object. Passing ``confidence=1.0`` means "this came from
    the simulator", and a policy that gates on confidence should be given a
    smaller value by a caller that is actually uncertain.

    Args:
        wire: Per-object ``id``, ``pose`` and optional ``sigma_m``.
        timestamp_s: Time the estimates were produced.
        confidence: Confidence to attach to every estimate.
        source: Provenance recorded on each estimate.

    Returns:
        Object states keyed by id.

    Raises:
        ValueError: If an object's pose is malformed.
    """
    states: dict[str, ObjectState] = {}
    for entry in wire:
        object_id = str(entry.get("id", ""))
        if not object_id:
            raise ValueError(f"wire object has no id: {entry!r}")
        sigma = float(entry.get("sigma_m", 0.001))
        states[object_id] = ObjectState(
            id=object_id,
            pose=_pose_from_wire(dict(entry["pose"])),
            covariance=Covariance.from_diagonal((sigma,) * 6),
            confidence=float(confidence),
            source=source,
            timestamp_s=float(timestamp_s),
            visible=True,
            last_seen_s=float(timestamp_s),
            estimated_size_m=float(entry.get("size_m", 0.04)),
        )
    return states
