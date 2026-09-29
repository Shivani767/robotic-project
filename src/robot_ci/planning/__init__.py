"""Motion planning and kinematics.

``MotionPlanner`` is the interface a policy uses to turn a Cartesian goal into
joint targets. Two implementations exist: :class:`SimplePlanner` (numerical IK,
works everywhere) and, from Phase 11, a MoveIt-backed planner on a remote
machine. The research engine depends only on the interface.
"""

from robot_ci.planning.kinematics import (
    IKOptions,
    IKResult,
    JacobianProvider,
    MuJoCoJacobianProvider,
    damped_least_squares_step,
    rotation_error,
    solve_ik,
)

__all__ = [
    "IKOptions",
    "IKResult",
    "JacobianProvider",
    "MuJoCoJacobianProvider",
    "damped_least_squares_step",
    "rotation_error",
    "solve_ik",
]
