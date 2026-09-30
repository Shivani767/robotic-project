"""Robot embodiment configuration.

The default embodiment is a Panda-class 7-DOF arm with a parallel gripper. Link
geometry is stored as simple capsules and the kinematics as DH parameters so
that the local simulator can do fast analytic collision checks and analytic IK
without importing MuJoCo, and so the same description can be cross-checked
against MoveIt 2 / ros2_control on a remote machine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

import numpy as np
from pydantic import Field, model_validator

from robot_ci.models.base import VersionedModel
from robot_ci.models.geometry import Vec3

__all__ = [
    "ArmModel",
    "ControllerType",
    "JointLimits",
    "LinkPrimitive",
    "RobotConfig",
]


class ArmModel(str, Enum):
    """Supported arm embodiments."""

    PANDA = "panda"
    """Franka Emika Panda: 7 DOF + parallel gripper (primary)."""
    UR5E = "ur5e"
    """Universal Robots UR5e: 6 DOF + gripper (alternate, for embodiment ablations)."""


class ControllerType(str, Enum):
    """Low-level control mode exposed to policies."""

    POSITION = "position"
    """Joint position targets (most common)."""
    VELOCITY = "velocity"
    """Joint velocity targets."""
    IMPEDANCE = "impedance"
    """Cartesian impedance control; requires a compliant model."""


class GripperControlMode(str, Enum):
    """How a requested aperture is turned into finger actuator commands.

    The distinction matters because the fingers are driven by *position*
    actuators with a finite stiffness. Asking such an actuator for a separation
    smaller than the object is asking for a position error, and a position error
    times a stiffness is a force -- an unbounded one, up to the actuator's force
    limit, with no relationship to how much force the object needs.
    """

    POSITION = "position"
    """Write the requested aperture straight through, as before.

    Simple and faithful to the request, but the contact force is whatever the
    resulting position error produces. For an object this light that can be two
    orders of magnitude more than is needed to hold it.
    """
    BOUNDED = "bounded"
    """Bound the commanded engagement so contact force cannot exceed a target.

    The fingers are still position-actuated and the collision model is
    untouched; only the requested aperture is floored, so that the position
    error can never exceed ``force / stiffness``.
    """


class JointLimits(VersionedModel):
    """Symmetric per-joint limits in radians."""

    lower: tuple[float, ...] = Field(description="Lower limits, one per actuated joint.")
    upper: tuple[float, ...] = Field(description="Upper limits, one per actuated joint.")
    max_velocity: tuple[float, ...] = Field(default=(), description="Per-joint velocity limits in rad/s.")
    max_effort: tuple[float, ...] = Field(default=(), description="Per-joint effort limits in Nm.")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.lower) != len(self.upper):
            raise ValueError(
                f"joint limit arrays must match: {len(self.lower)} vs {len(self.upper)}"
            )
        for index, (low, high) in enumerate(zip(self.lower, self.upper, strict=True)):
            if low >= high:
                raise ValueError(f"joint {index}: lower {low} must be below upper {high}")
        return self

    def clamped(self, positions: np.ndarray) -> np.ndarray:
        """Clip a position vector into the limit box."""
        return np.clip(
            np.asarray(positions, dtype=np.float64),
            np.array(self.lower, dtype=np.float64),
            np.array(self.upper, dtype=np.float64),
        )

    def within(self, positions: np.ndarray, atol: float = 1e-9) -> bool:
        """Whether a position vector lies inside the limit box."""
        array = np.asarray(positions, dtype=np.float64)
        return bool(
            np.all(array >= np.array(self.lower) - atol)
            and np.all(array <= np.array(self.upper) + atol)
        )


class LinkPrimitive(VersionedModel):
    """A capsule approximation of one robot link, used for collision checks.

    Attributes:
        name: Link name, following the ROS 2 / MoveIt naming convention.
        parent: Index of the parent link (``-1`` for the base).
        offset: Translation from the parent link frame to this link frame.
        radius: Capsule radius in metres.
        length: Capsule length along local z in metres.
    """

    name: str = Field(min_length=1, description="Link name.")
    parent: int = Field(ge=-1, description="Parent link index, or -1 for the base.")
    offset: Vec3 = Field(default_factory=lambda: Vec3.of(0.0, -0.30, 0.0), description="Frame offset from parent.")
    radius: float = Field(gt=0.0, le=0.5, description="Capsule radius in metres.")
    length: float = Field(ge=0.0, le=3.0, description="Capsule length along local z in metres.")


# --- Panda-class (Franka Emika Panda) -------------------------------------
_PANDA_DOF = 7
#: Denavit-Hartenberg (a, d, alpha) per joint, plus a flange row.
_PANDA_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -1.5707963267948966),
    (0.0, 0.316, 1.5707963267948966),
    (0.0825, 0.0, 1.5707963267948966),
    (-0.0825, 0.384, -1.5707963267948966),
    (0.0, 0.0, 1.5707963267948966),
    (0.088, 0.0, 1.5707963267948966),
    (0.0, 0.107, 0.0),
)
_PANDA_LIMITS = JointLimits(
    lower=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
    upper=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
    max_velocity=(2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100),
    max_effort=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
)
#: (name, parent index, offset, capsule radius, capsule length).
_PANDA_LINKS = (
    ("link0", -1, (0.0, 0.0, 0.0), 0.080, 0.140),
    ("link1", 0, (0.0, 0.0, 0.070), 0.060, 0.120),
    ("link2", 1, (0.0, 0.0, 0.150), 0.060, 0.130),
    ("link3", 2, (0.0, 0.0, 0.150), 0.055, 0.120),
    ("link4", 3, (0.0, 0.0, 0.100), 0.050, 0.100),
    ("link5", 4, (0.0, 0.0, 0.100), 0.045, 0.090),
    ("link6", 5, (0.0, 0.0, 0.080), 0.040, 0.060),
    ("hand", 6, (0.0, 0.0, 0.060), 0.035, 0.050),
    ("left_finger", 7, (0.0, 0.020, 0.045), 0.012, 0.030),
    ("right_finger", 7, (0.0, -0.020, 0.045), 0.012, 0.030),
)
#: Home configuration of the vendored Franka Emika Panda MJCF.
#:
#: Chosen by exhaustive search over the *arm* joint ranges (resolved by name,
#: since jnt_range[:7] is the object free joint, not the arm) for the
#: pose that maximises joint-limit margin while keeping the tool centre point
#: over the workspace and producing **no arm-vs-scene contact**. It carries
#: 1.03 rad of margin. Margin matters because the IK warm-start and the joint
#: limits both depend on it; contact-freedom matters because an arm resting in
#: the tabletop at reset generates constraint forces that launch the object.
_PANDA_HOME = (0.9239, 0.3503, 1.3375, -2.1517, -0.8058, 1.4429, -0.8909)

#: Half-height of the Panda's finger pad along the vertical, in metres.
#:
#: Measured from the compiled MJCF in the grasp configuration by
#: ``scripts/pad_target_ik_experiment.py`` (8.50 mm). It is an *embodiment*
#: property, not a tuning constant, and it bounds where the jaw can go: a pad
#: centred at height z has its lower edge at ``z - pad_half_height_m``, so the
#: jaw cannot descend below ``table_top + pad_half_height_m`` without the pad
#: sinking into the support surface. The generic default is a conservative
#: value for an unspecified gripper.
_PANDA_PAD_HALF_HEIGHT_M = 0.0085
#: How far the pads sit *behind* the tool centre point, along the approach axis.
#:
#: Measured from the compiled model, not assumed: the injected ``tcp`` site is
#: at z = 0.1034 in the hand frame and the pad's collision box centre is at
#: 0.0584 + 0.0445 = 0.1029, so the pads grip 0.5 mm behind the point the
#: planner commands. A grasp that ignores this closes the pads that far off the
#: object centre, which on a 16 mm cube means gripping its upper 12 mm and
#: levering it out of the jaws on the lift.
_PANDA_PAD_OFFSET_FROM_TCP_M = 0.0005

#: Pad separation with both fingers fully closed, in metres.
#:
#: Measured from the compiled MJCF: the fingers travel 0 to 0.04 m and the pads
#: are 11.0 mm apart at qpos 0, so the jaws cannot close past 11 mm. This
#: matters when choosing a grasp force: commanding a zero aperture against an
#: object does not "hold" it, it drives the position actuators to squeeze the
#: object out of the jaws.
_PANDA_CLOSED_SEPARATION_M = 0.011

# --- UR5e (alternate embodiment) ------------------------------------------
_UR5E_DOF = 6
#: UR5e has 6 actuated joints, so the DH table needs dof+1 = 7 rows: one per
#: joint plus a final flange row. The Panda table above follows the same rule.
_UR5E_DH = (
    (0.0, 0.1625, 1.5707963267948966),
    (0.4250, 0.0, 0.0),
    (0.3922, 0.0, 0.0),
    (0.0, 0.1333, 1.5707963267948966),
    (0.0995, 0.0, -1.5707963267948966),
    (0.0996, 0.0, 0.0),
    (0.0, 0.0, 0.0),          # flange
)
_UR5E_LIMITS = JointLimits(
    lower=(-6.283185307179586,) * 6,
    upper=(6.283185307179586,) * 6,
    max_velocity=(3.1416,) * 6,
    max_effort=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)
_UR5E_LINKS = (
    ("base_link", -1, (0.0, 0.0, 0.0), 0.075, 0.100),
    ("shoulder_link", 0, (0.0, 0.0, 0.080), 0.060, 0.120),
    ("upper_arm_link", 1, (0.0, 0.0, 0.150), 0.055, 0.200),
    ("forearm_link", 2, (0.0, 0.0, 0.120), 0.050, 0.180),
    ("wrist_1_link", 3, (0.0, 0.0, 0.100), 0.045, 0.080),
    ("wrist_2_link", 4, (0.0, 0.0, 0.060), 0.040, 0.060),
    ("wrist_3_link", 5, (0.0, 0.0, 0.050), 0.040, 0.040),
    ("left_finger", 6, (0.0, 0.020, 0.040), 0.012, 0.030),
    ("right_finger", 6, (0.0, -0.020, 0.040), 0.012, 0.030),
)
_UR5E_HOME = (0.0, -1.571, 1.571, -1.571, -1.571, 0.0)

_EMBODIMENTS: dict[ArmModel, tuple[int, tuple, JointLimits, tuple, tuple]] = {
    ArmModel.PANDA: (_PANDA_DOF, _PANDA_DH, _PANDA_LIMITS, _PANDA_LINKS, _PANDA_HOME),
    ArmModel.UR5E: (_UR5E_DOF, _UR5E_DH, _UR5E_LIMITS, _UR5E_LINKS, _UR5E_HOME),
}


def _build_links(rows: tuple) -> tuple[LinkPrimitive, ...]:
    """Convert compact link rows into :class:`LinkPrimitive` models."""
    return tuple(
        LinkPrimitive(
            name=name, parent=parent, offset=Vec3.of(*offset), radius=radius, length=length
        )
        for name, parent, offset, radius, length in rows
    )


class RobotConfig(VersionedModel):
    """Embodiment: kinematics, limits, control mode and gripper.

    This is the single source of truth for *what robot is under test*. Changing
    the embodiment changes the ``config_hash``, so a Panda run can never be
    silently compared against a UR5e run.
    """

    model: ArmModel = Field(default=ArmModel.PANDA, description="Arm model.")
    controller: ControllerType = Field(default=ControllerType.POSITION, description="Control mode.")
    dof: int = Field(default=_PANDA_DOF, ge=1, le=12, description="Actuated arm joints.")
    dh_parameters: tuple[tuple[float, float, float], ...] = Field(
        default=_PANDA_DH, description="DH (a, d, alpha) per joint plus a flange row."
    )
    limits: JointLimits = Field(default=_PANDA_LIMITS, description="Joint limits.")
    links: tuple[LinkPrimitive, ...] = Field(
        default_factory=lambda: _build_links(_PANDA_LINKS), description="Capsule links for collision."
    )
    home_position: tuple[float, ...] = Field(default=_PANDA_HOME, description="Home configuration in radians.")
    gripper_closed_separation_m: float = Field(
        default=_PANDA_CLOSED_SEPARATION_M,
        gt=0.0,
        le=0.5,
        description=(
            "Pad separation with the fingers fully closed, in metres.\n\n"
            "Bounds the narrowest graspable object and sets the correct holding "
            "command: an aperture of `object_width - closed_separation` brings "
            "the pads onto the object without driving them past it."
        ),
    )
    pad_half_height_m: float = Field(
        default=_PANDA_PAD_HALF_HEIGHT_M,
        gt=0.0,
        le=0.5,
        description=(
            "Half-height of the finger pad along the vertical, in metres.\n\n"
            "Bounds the lowest usable jaw height: the pad's lower edge is at "
            "`jaw_z - pad_half_height_m`, so the jaw must stay at or above "
            "`table_top + pad_half_height_m` or the pad sinks into the support "
            "surface and the jaws jam open without touching the object. "
            "Measured from the compiled model, not assumed."
        ),
    )
    pad_offset_from_tcp_m: float = Field(
        default=_PANDA_PAD_OFFSET_FROM_TCP_M,
        ge=-0.2,
        le=0.2,
        description=(
            "Distance from the tool centre point back to the pad centre, along "
            "the approach axis, in metres.\n\n"
            "The planner commands the tool centre point but the pads are what "
            "grip, so a grasp aimed at the object's centre closes the pads "
            "this far off it. Measured from the compiled model: the injected "
            "`tcp` site sits at z = 0.1034 in the hand frame while the pad "
            "collision box is centred at 0.1029. On a 16 mm cube that error "
            "puts the pads on the upper 12 mm of the face, and the lift then "
            "levers the object out of the jaws rather than carrying it."
        ),
    )
    base_position: Vec3 = Field(
        default_factory=lambda: Vec3.of(0.0, -0.30, 0.0),
        description=(
            "Base position in the world frame. The arm is mounted on a pedestal "
            "behind the table, as in a real cell.\n\n"
            "The 0.60 m separation from the workspace centre is a measured "
            "requirement, not a choice: objects spawn within 0.20 m of the "
            "workspace centre, and a closer base spawns them inside its own "
            "collision geometry, which launches them at reset. It is also "
            "inside the arm's measured reach envelope."
        ),
    )
    control_hz: float = Field(default=20.0, gt=0.0, le=1000.0, description="Control loop rate in Hz.")
    max_gripper_width_m: float = Field(default=0.08, gt=0.0, le=0.5, description="Fully open aperture in metres.")
    max_gripper_force_n: float = Field(default=20.0, gt=0.0, le=500.0, description="Peak closing force in N.")
    gripper_control_mode: GripperControlMode = Field(
        default=GripperControlMode.POSITION,
        description=(
            "How a requested aperture becomes finger actuator commands. "
            "POSITION writes the request through unchanged; BOUNDED floors the "
            "requested aperture so the resulting contact force cannot exceed "
            "`gripper_target_contact_force_n`. POSITION is the default so that "
            "existing results are reproduced exactly unless a run asks for "
            "bounded contact control."
        ),
    )
    gripper_target_contact_force_n: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description=(
            "Contact force the BOUNDED mode is allowed to apply per finger, in "
            "newtons. Converted to an engagement using the finger actuator's "
            "measured stiffness, so this is a force limit rather than a "
            "position limit: `engagement = force / stiffness`. The object in "
            "the default scene weighs 0.049 N, so holding it needs about "
            "1/40th of the default; the value is a controller parameter and is "
            "swept in the bounded-contact experiment rather than assumed."
        ),
    )
    gripper_max_closure_speed_m_s: float = Field(
        default=0.05,
        gt=0.0,
        le=10.0,
        description=(
            "Slew rate for BOUNDED mode when `gripper_closure_ramp_enabled` is "
            "set, in metres per second. Full travel of ~69 mm therefore takes "
            "about 28 steps at the default 20 Hz, which bounds the impulse "
            "delivered to the object. Unused by POSITION mode, which writes a "
            "step."
        ),
    )
    gripper_closure_ramp_enabled: bool = Field(
        default=False,
        description=(
            "Whether BOUNDED mode ramps the commanded aperture instead of "
            "stepping it. Off by default, and deliberately so: ramping and "
            "bounding are two different mechanisms -- one limits the impulse at "
            "contact, the other limits the steady-state force -- and changing "
            "both at once makes the result unattributable. The bounded-contact "
            "experiment holds the ramp off so that the force bound is measured "
            "on its own. Turning it on is a separate experiment."
        ),
    )
    arm_max_joint_step_rad: float = Field(
        default=0.0,
        ge=0.0,
        le=3.15,
        description=(
            "Upper bound on how far a single arm joint command may move in one "
            "control period, in radians. 0 disables the bound, which is the "
            "default so recorded results are reproduced exactly. Setting it "
            "clips the command, not the joint: the servo still drives freely "
            "toward the clipped target. Measured at 0.15 rad on this arm: the "
            "uncapped waypoint transit reaches 2.95 rad/step (a 214 mm "
            "end-effector jump in 50 ms) and carried the object 23.6 m away on "
            "one seed; the same seed stays within 0.48 m capped. Also removes "
            "the early-grasp force spike (4.6 N -> 1.1 N over t=0..29). It is "
            "not a universal improvement: one seed never reaches the object "
            "within the step budget, so it is a measured trade-off rather than "
            "a default-on fix."
        ),
    )
    has_force_torque: bool = Field(default=True, description="Whether a wrist F/T sensor is modelled.")

    @classmethod
    def panda(cls, **overrides: object) -> RobotConfig:
        """Return a Panda-class configuration with documented defaults.

        Overrides go through normal validation, so an out-of-range
        ``home_position`` is rejected rather than silently accepted.
        """
        return cls._nominal_for(ArmModel.PANDA, **overrides)

    @classmethod
    def ur5e(cls, **overrides: object) -> RobotConfig:
        """Return a UR5e-class configuration with documented defaults."""
        return cls._nominal_for(ArmModel.UR5E, **overrides)

    @classmethod
    def nominal(cls) -> RobotConfig:
        """Return the canonical Panda configuration."""
        return cls._nominal_for(ArmModel.PANDA)

    @classmethod
    def _nominal_for(cls, model: ArmModel, **overrides: object) -> RobotConfig:
        """Build the canonical configuration for ``model``, then apply overrides.

        The model is constructed through ``cls(**data)`` rather than
        ``model_copy``, because ``model_copy`` skips validation and would let an
        invalid override (a home pose outside the joint limits, a DOF that
        disagrees with the limit arrays) through undetected.
        """
        dof, dh, limits, links, home = _EMBODIMENTS[model]
        data: dict[str, Any] = {
            "model": model,
            "dof": dof,
            "dh_parameters": dh,
            "limits": limits,
            "links": _build_links(links),
            "home_position": home,
        }
        data.update(overrides)
        return cls(**data)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if len(self.limits.lower) != self.dof:
            raise ValueError(
                f"joint limits define {len(self.limits.lower)} joints but dof={self.dof}"
            )
        if len(self.home_position) != self.dof:
            raise ValueError(
                f"home_position has {len(self.home_position)} entries but dof={self.dof}"
            )
        if len(self.dh_parameters) != self.dof + 1:
            raise ValueError(
                f"dh_parameters must have dof+1={self.dof + 1} entries, got {len(self.dh_parameters)}"
            )
        if not self.limits.within(self.home_array()):
            raise ValueError(
                f"home_position {self.home_position} violates joint limits "
                f"{self.limits.lower}..{self.limits.upper}"
            )
        return self

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        """Actuated arm joint names, e.g. ``("joint1", ..., "joint7")``."""
        return tuple(f"joint{i + 1}" for i in range(self.dof))

    @property
    def all_joint_names(self) -> tuple[str, ...]:
        """Arm joint names plus the two gripper fingers."""
        return (*self.arm_joint_names, "left_finger", "right_finger")

    def home_array(self) -> np.ndarray:
        """Home configuration as a ``float64`` array of shape ``(dof,)``."""
        return np.array(self.home_position, dtype=np.float64)

    def random_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a uniformly random configuration within the joint limits."""
        lower = np.array(self.limits.lower, dtype=np.float64)
        upper = np.array(self.limits.upper, dtype=np.float64)
        return rng.uniform(lower, upper)

    @property
    def gripper_joint_names(self) -> tuple[str, str]:
        """The two finger joint names."""
        return ("left_finger", "right_finger")
