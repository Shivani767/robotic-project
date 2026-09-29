r"""Damped-least-squares inverse kinematics.

Why this exists
---------------
A manipulation policy must be able to say "put the gripper at this pose", not
"set these seven joint numbers". Solving IK numerically keeps the research
engine independent of MoveIt 2, which is not installable on macOS.

The solver is the standard damped least squares (Levenberg-Marquardt) update

.. math::

    \\Delta q = J^\\top (J J^\\top + \\lambda^2 I)^{-1} e

where :math:`J` is the positional Jacobian of the end-effector site, ``e`` is
the pose error and :math:`\\lambda` is the damping term. Damping is what makes
this well-behaved near singularities and when the target is unreachable: the
step shrinks smoothly instead of exploding.

The Jacobian comes from MuJoCo when the solver is attached to a live model, and
from a finite-difference Jacobian of the DH forward kinematics otherwise. The
interface is the same either way, which is what lets a policy be written once
and run against either backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_ci.models.geometry import Pose, Quaternion, Vec3

__all__ = [
    "IKOptions",
    "IKResult",
    "JacobianProvider",
    "damped_least_squares_step",
    "rotation_error",
    "solve_ik",
    "solve_ik_refined",
]


@dataclass(frozen=True)
class IKOptions:
    """Solver settings.

    Attributes:
        damping: Levenberg-Marquardt damping. Larger values trade tracking
            accuracy for robustness near singularities.
        max_iterations: Iterations before giving up.
        position_tolerance: Convergence threshold on position error, metres.
        orientation_tolerance: Convergence threshold on orientation error, rad.
        step_clip: Maximum joint change per iteration, radians. Prevents the
            solver from taking a physically impossible jump.
        step_gain: Fraction of the computed step actually applied, in (0, 1].
            Below 1.0 this becomes a damped iteration, which converges more
            reliably on redundant 7-DOF chains.
    """

    damping: float = 0.08
    max_iterations: int = 120
    position_tolerance: float = 2.0e-3
    orientation_tolerance: float = 0.05
    step_clip: float = 0.25
    step_gain: float = 0.7
    nullspace_gain: float = 0.06
    """Weight of the null-space term that pulls the arm toward a mid-range posture.

    A redundant 7-DOF arm has a 1-D null space even when solving a 6-DOF task.
    Without a posture bias, damping and numerical noise let that free direction
    drift into folded or near-singular configurations, from which the solver
    cannot recover. Pulling gently toward the middle of the joint range keeps
    the arm in a basin the next solve can improve on. This is a standard
    redundancy-resolution term, not a hack, and it directly improves the
    success rate of sequential waypoint tracking.
    """
    posture_bias: tuple[float, ...] | None = None
    """Target posture for the null-space term. Defaults to the midpoint of the
    supplied joint limits."""


@dataclass
class IKResult:
    """Outcome of one IK solve.

    Attributes:
        joint_positions: The best configuration found.
        converged: Whether both tolerances were met.
        position_error: Final position error in metres.
        orientation_error: Final orientation error in radians.
        iterations: Iterations actually run.
        jacobian_condition: Condition number of ``J J^T + lambda^2 I`` at the
            final step. A large value means the target is near-singular.
    """

    joint_positions: np.ndarray
    converged: bool
    position_error: float
    orientation_error: float
    iterations: int
    jacobian_condition: float = math.inf

    @property
    def is_reachable(self) -> bool:
        """Whether the final error is small enough to be worth attempting."""
        return self.position_error < 0.01

    def describe(self) -> str:
        """One-line summary for logs."""
        return (
            f"IK(converged={self.converged}, pos_err={self.position_error * 1000:.1f}mm, "
            f"rot_err={math.degrees(self.orientation_error):.1f}deg, iters={self.iterations})"
        )


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the world-frame rotation vector taking ``current`` to ``target``.

    The returned vector's magnitude is the geodesic angle between the two
    orientations, and its direction is the world-frame axis to rotate about. It
    is the quantity the angular rows of a MuJoCo Jacobian expect, which is why
    it is expressed in the world frame rather than the body frame.

    Args:
        current: 3x3 rotation matrix of the current orientation.
        target: 3x3 rotation matrix of the desired orientation.

    Returns:
        A 3-vector in radians.

    Notes:
        For ``R = R_target @ R_current.T`` the standard identity is
        ``R - R.T = 2 sin(theta) * K``, where ``K`` is the skew matrix of the
        rotation axis. Reading the off-diagonal terms with consistent row/column
        indices matters: using the transposed pair silently returns zero for
        rotations about the other axis.
    """
    relative = target @ current.T
    cos_theta = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_theta)
    if angle < 1e-9:
        return np.zeros(3)

    # vec(skew(R - R.T)) / 2 gives sin(theta) * axis.
    delta = relative - relative.T
    axis_times_sin = 0.5 * np.array([delta[2, 1], delta[0, 2], delta[1, 0]])

    if angle > math.pi - 1e-6:
        # Near-pi the sin(theta) form degenerates; recover the axis from the
        # symmetric part instead, which is well conditioned there.
        symmetric = 0.5 * (relative + relative.T)
        axis = np.sqrt(np.clip(np.diag(symmetric), 0.0, None))
        # Disambiguate the sign using the antisymmetric residual.
        if float(np.dot(axis, axis_times_sin)) < 0.0:
            axis = -axis
        norm = float(np.linalg.norm(axis))
        return (axis / norm * angle) if norm > 1e-12 else np.zeros(3)

    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-12:  # pragma: no cover - guarded by the checks above
        return np.zeros(3)
    return axis_times_sin / sin_angle * angle


def damped_least_squares_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
    step_clip: float = math.inf,
) -> tuple[np.ndarray, float]:
    """Compute one damped-least-squares joint step.

    Args:
        jacobian: Task-space Jacobian, shape ``(6, dof)`` (rows 0-2 position).
        error: Desired task-space correction, shape ``(6,)``.
        damping: Levenberg-Marquardt damping factor.
        step_clip: Optional per-joint limit on the step.

    Returns:
        ``(delta_q, condition_number)``.
    """
    jjt = jacobian @ jacobian.T
    regularized = jjt + (damping**2) * np.eye(jjt.shape[0])
    try:
        condition = float(np.linalg.cond(regularized))
        delta = jacobian.T @ np.linalg.solve(regularized, error)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by damping
        delta = jacobian.T @ np.linalg.pinv(regularized) @ error
        condition = math.inf
    if not np.all(np.isfinite(delta)):
        return np.zeros(jacobian.shape[1]), condition
    if step_clip < math.inf:
        delta = np.clip(delta, -step_clip, step_clip)
    return delta, condition


class JacobianProvider:
    """Supplies the end-effector Jacobian for the current configuration.

    This is the seam that lets one policy run against any backend: a MuJoCo
    model provides an analytic Jacobian, while a purely kinematic description
    provides a finite-difference one.
    """

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """Return ``(position, rotation_matrix)`` for configuration ``q``."""
        raise NotImplementedError

    def jacobian(self, q: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Return the 6xN task-space Jacobian at ``q``."""
        raise NotImplementedError


@dataclass
class MuJoCoJacobianProvider(JacobianProvider):
    """Analytic Jacobian from a live MuJoCo model.

    Attributes:
        mujoco: The imported ``mujoco`` module.
        model: The compiled model.
        data: The simulation state (mutated by reads; restored by the caller).
        site_name: Name of the end-effector site.
        forward_fn: Callable that advances kinematics for a candidate ``q``.
    """

    mujoco: Any
    model: Any
    data: Any
    site_name: str = "ee"
    def _evaluate(self, q: np.ndarray, want_jacobian: bool) -> Any:
        """Evaluate kinematics at ``q`` without disturbing the live simulation.

        The state is saved and restored **on every call**. Caching the saved
        state across calls would be a subtle and severe bug: the simulation
        advances between policy steps, so restoring a snapshot taken at the
        start of the episode would silently rewind the arm to its initial pose
        every time the policy asks a question.
        """
        mujoco = self.mujoco
        saved_qpos = np.array(self.data.qpos, copy=True)
        saved_qvel = np.array(self.data.qvel, copy=True)
        try:
            for index, value in enumerate(q):
                self.data.qpos[index] = float(value)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name
            )
            if not want_jacobian:
                return (
                    np.array(self.data.site_xpos[site_id], copy=True),
                    np.array(self.data.site_xmat[site_id], copy=True).reshape(3, 3),
                )
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            return np.vstack([jacp[:, : len(q)], jacr[:, : len(q)]])
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the end-effector position and rotation at ``q``."""
        return self._evaluate(q, want_jacobian=False)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return the 6xN task-space Jacobian at ``q``."""
        return self._evaluate(q, want_jacobian=True)


def solve_ik(
    provider: JacobianProvider,
    target: Pose,
    q_init: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    options: IKOptions | None = None,
) -> IKResult:
    """Solve for a joint configuration that places the end effector at ``target``.

    Args:
        provider: Supplies forward kinematics and the Jacobian.
        target: Desired end-effector pose.
        q_init: Starting configuration (usually the current one).
        lower: Optional lower joint limits; steps are clipped into them.
        upper: Optional upper joint limits.
        options: Solver settings.

    Returns:
        An :class:`IKResult`. ``converged`` is False when the target could not
        be reached within the iteration budget, which is a normal outcome and not
        an error: the caller decides whether to give up.
    """
    opts = options or IKOptions()
    q = np.array(q_init, dtype=np.float64).copy()
    target_position = target.position.array
    target_rotation = _quaternion_to_matrix(target.orientation)

    position_error = math.inf
    orientation_error = math.inf
    condition = math.inf
    iterations = 0

    for step in range(1, opts.max_iterations + 1):
        iterations = step
        position, rotation = provider.forward(q)
        linear = target_position - position
        angular = rotation_error(rotation, target_rotation)
        position_error = float(np.linalg.norm(linear))
        orientation_error = float(np.linalg.norm(angular))
        if position_error < opts.position_tolerance and orientation_error < opts.orientation_tolerance:
            break
        jacobian = provider.jacobian(q)
        task_error = np.concatenate([linear, angular])
        delta, condition = damped_least_squares_step(
            jacobian, task_error, opts.damping, opts.step_clip
        )
        if opts.nullspace_gain > 0.0 and jacobian.shape[0] < jacobian.shape[1]:
            # Project the posture bias into the null space of the task Jacobian
            # so it cannot pull the end effector off target.
            damping_matrix = np.eye(jacobian.shape[0]) * (opts.damping**2)
            jjt_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping_matrix)
            projector = np.eye(jacobian.shape[1]) - jjt_pinv @ jacobian
            target_posture = (
                np.asarray(opts.posture_bias, dtype=np.float64)
                if opts.posture_bias is not None
                else 0.5 * (
                    (lower if lower is not None else q) + (upper if upper is not None else q)
                )
            )
            posture_error = np.clip(target_posture - q, -0.5, 0.5)
            delta = delta + projector @ (opts.nullspace_gain * posture_error)
        q = q + opts.step_gain * delta
        if lower is not None and upper is not None:
            q = np.clip(q, lower, upper)

    return IKResult(
        joint_positions=q,
        converged=position_error < opts.position_tolerance
        and orientation_error < opts.orientation_tolerance,
        position_error=position_error,
        orientation_error=orientation_error,
        iterations=iterations,
        jacobian_condition=condition,
    )


def solve_ik_refined(
    provider: JacobianProvider,
    target: Pose,
    q_init: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    options: IKOptions | None = None,
    restarts: int = 3,
) -> IKResult:
    """Solve IK, retrying from perturbed seeds when the first attempt stalls.

    Damped least squares converges quickly near a good seed and stagnates near
    joint limits or kinematic singularities. A warm-started policy calls this
    every control step, so a partial result that is a few millimetres out is
    worse than useless: the gripper lands beside the object and the grasp
    silently fails. Perturbing the seed and re-running escapes the stagnation
    basin for a negligible cost compared with a wasted episode.

    Args:
        provider: Supplies forward kinematics and the Jacobian.
        target: Desired end-effector pose.
        q_init: Starting configuration.
        lower: Optional lower joint limits.
        upper: Optional upper joint limits.
        options: Solver settings.
        restarts: How many perturbed retries to attempt after the first solve.

    Returns:
        The best :class:`IKResult` found. ``converged`` reflects whether *any*
        attempt met the tolerances.
    """
    best = solve_ik(provider, target, q_init, lower=lower, upper=upper, options=options)
    if best.converged or restarts <= 0:
        return best

    span = 0.35
    if lower is not None and upper is not None:
        span = float(np.min(upper - lower)) * 0.12
    for attempt in range(1, restarts + 1):
        seed = np.array(q_init, dtype=np.float64)
        # Deterministic perturbation: the same attempt index always produces the
        # same seed, so a run is reproducible.
        direction = np.arange(1, len(seed) + 1, dtype=np.float64) + attempt
        seed = seed + span * np.sin(direction * (attempt + 1))
        if lower is not None and upper is not None:
            seed = np.clip(seed, lower, upper)
        candidate = solve_ik(provider, target, seed, lower=lower, upper=upper, options=options)
        if candidate.position_error < best.position_error:
            best = candidate
        if best.converged:
            break
    return best


def _quaternion_to_matrix(quaternion: Quaternion) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    q = quaternion.array
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_pose(position: np.ndarray, rotation: np.ndarray) -> Pose:
    """Convert a position and rotation matrix into a :class:`Pose`."""
    return Pose(position=Vec3.from_iterable(position), orientation=_matrix_to_quaternion(rotation))


def _matrix_to_quaternion(mat: np.ndarray) -> Quaternion:
    """Convert a 3x3 rotation matrix to a ``(w, x, y, z)`` quaternion."""
    trace = float(mat[0, 0] + mat[1, 1] + mat[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return Quaternion(w=float(w), x=float(x), y=float(y), z=float(z))
