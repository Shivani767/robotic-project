"""The arm must obey the embodiment's velocity limits.

`JointLimits.max_velocity` was declared on the robot model and read by nothing.
The IK solution was sent straight to the position actuators, which asks the arm
to arrive at the next waypoint within one control period. Measured on the
pick-and-place waypoint sequence that is about 1 m/s of end-effector travel --
fast enough that the pads knock the object out of the grip on the lift, so the
grasp registers and is then lost. Nothing in the episode reported a control
fault: every waypoint was reached on time and the object simply was not held.

These tests pin the properties that fix depends on, using a toy analytic arm so
the limit is checked exactly rather than inferred from a simulator run.
"""

from __future__ import annotations

import numpy as np

from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
from robot_ci.models.robot import JointLimits
from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
from robot_ci.models.state import (
    GripperState,
    ObjectState,
    Observation,
    RobotState,
    Source,
)
from robot_ci.planning.kinematics import JacobianProvider
from robot_ci.policies import create_policy

TARGET_ID = "target_cube"
GRASP_ROTATION = np.diag([1.0, -1.0, -1.0])


class ToyCartesianProvider(JacobianProvider):
    """Linear arm: ``x <- q0``, ``z <- q1``, ``y <- q2``."""

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the end-effector position and a constant rotation."""
        q = np.asarray(q, dtype=np.float64)
        return np.array([q[0], q[2], q[1]], dtype=np.float64), GRASP_ROTATION

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Constant Jacobian mapping the first three joints to ``x, z, y``."""
        dof = int(np.asarray(q).size)
        jac = np.zeros((6, dof), dtype=np.float64)
        jac[0, 0] = 1.0
        jac[2, 1] = 1.0
        jac[1, 2] = 1.0
        return jac


def _with_limits(limits: JointLimits) -> ScenarioConfig:
    """A scenario whose robot declares ``limits``."""
    base = ScenarioConfig(scenario_id="rate_limit", seed=1)
    return base.model_copy(update={"robot": base.robot.model_copy(update={"limits": limits})})


def _observation(scenario: ScenarioConfig, *, joints: np.ndarray, sigma_m: float = 0.002):
    """One observation reporting the arm exactly at ``joints``."""
    dof = scenario.robot.dof
    position, _ = ToyCartesianProvider().forward(joints)
    return Observation(
        robot=RobotState(
            joint_positions=tuple(float(v) for v in joints),
            joint_velocities=tuple(0.0 for _ in range(dof)),
            joint_efforts=tuple(0.0 for _ in range(dof)),
            joint_names=scenario.robot.arm_joint_names,
            end_effector_pose=Pose(
                position=Vec3.from_iterable(position), orientation=Quaternion.identity()
            ),
            gripper=GripperState(width_m=0.04, max_width_m=scenario.robot.max_gripper_width_m),
        ),
        objects=(
            ObjectState(
                id=TARGET_ID,
                category=scenario.target.category,
                pose=Pose(
                    position=Vec3.of(0.30, 0.0, 0.025), orientation=Quaternion.identity()
                ),
                covariance=Covariance.isotropic(sigma_m, dimension=6),
                confidence=0.9,
                source=Source.FUSION,
                visible=True,
                estimated_size_m=scenario.target.size_m,
            ),
        ),
        scenario_id=scenario.scenario_id,
    )




def _drive(scenario: ScenarioConfig, *, steps: int = 40) -> list[float]:
    """Run the baseline against a perfect plant; return per-tick joint travel."""
    provider = ToyCartesianProvider()
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    policy.reset(PolicyConfig(name="policy_v1"), 1, scenario)
    policy.attach_planner(provider)

    joints = np.zeros(scenario.robot.dof, dtype=np.float64)
    travel: list[float] = []
    for _ in range(steps):
        observation = _observation(scenario, joints=joints)
        target = np.asarray(policy.act(observation).action.arm_targets, dtype=np.float64)
        travel.append(float(np.max(np.abs(target - joints))))
        joints = target
    return travel


def test_no_joint_travel_exceeds_the_velocity_limit() -> None:
    """No command may move a joint further than the limit allows in one tick."""
    scenario = ScenarioConfig(scenario_id="rate_limit", seed=1)
    limits = scenario.robot.limits
    assert len(limits.max_velocity) == len(limits.lower), (
        "the fixture needs a fully specified embodiment to be a real test"
    )
    per_tick = np.asarray(limits.max_velocity, dtype=np.float64) / float(
        scenario.robot.control_hz
    )
    for step, moved in enumerate(_drive(scenario)):
        assert moved <= float(per_tick.max()) + 1e-12, (
            f"step {step} moved {moved:.4f} rad, above the {per_tick.max():.4f} rad bound"
        )


def test_tighter_velocity_limits_produce_slower_motion() -> None:
    """The bound must come from the robot, not from a constant in the policy."""
    base = ScenarioConfig(scenario_id="rate_limit", seed=1)
    dof = base.robot.dof
    slow = JointLimits(
        lower=base.robot.limits.lower,
        upper=base.robot.limits.upper,
        max_velocity=tuple(0.05 for _ in range(dof)),
        max_effort=base.robot.limits.max_effort,
    )
    fast_travel = _drive(base)
    slow_travel = _drive(_with_limits(slow))

    assert max(slow_travel) < max(fast_travel), (
        "a robot declaring 0.05 rad/s moved at the same rate as one declaring 2.175"
    )
    assert max(slow_travel) <= 0.05 / float(base.robot.control_hz) + 1e-12


def test_unconstrained_embodiment_is_passed_through_unchanged() -> None:
    """A scenario with no velocity limits must not be silently re-limited.

    Inventing a bound for an embodiment that declared none would change results
    for anyone who left the field empty -- a silent change to every existing
    scenario rather than an opt-in behaviour.
    """
    base = ScenarioConfig(scenario_id="rate_limit", seed=1)
    dof = base.robot.dof
    free = _with_limits(
        JointLimits(
            lower=base.robot.limits.lower,
            upper=base.robot.limits.upper,
            max_velocity=(),
            max_effort=base.robot.limits.max_effort,
        )
    )
    policy = create_policy(PolicyConfig(name="policy_v1"), free)
    assert policy.max_joint_step is None, (
        "an embodiment with no declared velocity limit was given one anyway"
    )
    goal = np.linspace(0.1, 0.5, dof, dtype=np.float64)
    observation = _observation(free, joints=np.zeros(dof))
    assert np.allclose(policy._rate_limit(observation, goal), goal)


def test_the_rate_limit_is_symmetric() -> None:
    """A bound that only slowed approach would be a half-measure.

    The arm must also be limited when retreating, or a policy unwinding quickly
    would throw the object at the very speed the bound exists to prevent.
    """
    scenario = ScenarioConfig(scenario_id="rate_limit", seed=1)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    dof = scenario.robot.dof
    here = np.full(dof, 0.5, dtype=np.float64)
    observation = _observation(scenario, joints=here)

    outward = policy._rate_limit(observation, here + 1.0)
    inward = policy._rate_limit(observation, here - 1.0)
    assert np.allclose(outward - here, -(inward - here)), (
        "the bound is not symmetric about the current configuration"
    )
