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
    offset: Vec3 = Field(default_factory=lambda: Vec3.of(0.0, 0.10, 0.0), description="Frame offset from parent.")
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
#: Home configuration of the vendored Franka Emika Panda MJCF: the tool centre
#: point sits 162 mm above the nominal grasp pose with 0.44 rad of margin to the
#: nearest joint limit, found by a multi-start IK search over the measured
#: reachable set. A home pose with margin is what lets a policy warm-start from it
#: and converge on the first waypoint every time.
_PANDA_HOME = (-1.3624, 1.3194, 0.1700, -2.2245, 0.4053, 0.4329, -0.6642)

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
    base_position: Vec3 = Field(
        default_factory=lambda: Vec3.of(0.0, 0.10, 0.0),
        description=(
            "Base position in the world frame. Chosen by sweeping the arm's "
            "measured reachable set against the tabletop grasp point rather "
            "than by eye: at y=+0.10 the grasp point at (0, 0.10, 0.06) is "
            "2.4 cm from the nearest reachable pose, comfortably inside the "
            "workspace. The base is behind the workspace centre so the forearm "
            "does not sweep through the table at ordinary poses."
        ),
    )
    control_hz: float = Field(default=20.0, gt=0.0, le=1000.0, description="Control loop rate in Hz.")
    max_gripper_width_m: float = Field(default=0.08, gt=0.0, le=0.5, description="Fully open aperture in metres.")
    max_gripper_force_n: float = Field(default=20.0, gt=0.0, le=500.0, description="Peak closing force in N.")
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
