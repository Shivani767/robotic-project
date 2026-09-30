"""The candidate's confidence gate must reach a branch, not a diagnostic.

Until this change ``policy_v2`` differed from ``policy_v1`` only in parameters
that no branch read: ``clearance_bias_m`` was never used and the gate fed a
diagnostic string. On the simulator the two arms therefore agreed on
``grasp_error_m`` to sixteen digits, which is a vacuous comparison presenting
itself as a null result.

These tests drive the policy through a *toy analytic arm* whose IK converges,
so the gate can be observed in milliseconds without MuJoCo. The provider is a
linear map ``x <- q0``, ``z <- q1``, ``y <- q2`` -- trivially solvable, which is
the point: the test is about which waypoint the policy chooses, not about the
solver.
"""

from __future__ import annotations

import numpy as np

from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
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

#: A 180-degree rotation about x -- the grasp orientation the task uses, so the
#: toy arm can satisfy the orientation tolerance and IK converges in one step.
GRASP_ROTATION = np.diag([1.0, -1.0, -1.0])

TARGET_ID = "target_cube"


class ToyCartesianProvider(JacobianProvider):
    """Linear, fully controllable arm: ``x <- q0``, ``z <- q1``, ``y <- q2``."""

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the end-effector position and (constant) rotation."""
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


def _observation(
    scenario: ScenarioConfig,
    *,
    sigma_m: float,
    ee: np.ndarray,
    joints: np.ndarray | None = None,
) -> Observation:
    """One observation with a believed object and a perfect-tracking arm.

    Args:
        scenario: The condition being driven.
        sigma_m: Uncertainty to report on the object belief.
        ee: Where the end-effector actually is.
        joints: Where the joints actually are. A real plant integrates the
            commanded positions; defaulting this to zero would model a frozen
            arm, and a rate-limited policy would then command the same first
            increment forever and never arrive.
    """
    dof = scenario.robot.dof
    if joints is None:
        joints = np.zeros(dof, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    return Observation(
        robot=RobotState(
            joint_positions=tuple(float(v) for v in joints),
            joint_velocities=tuple(0.0 for _ in range(dof)),
            joint_efforts=tuple(0.0 for _ in range(dof)),
            joint_names=scenario.robot.arm_joint_names,
            end_effector_pose=Pose(position=Vec3.from_iterable(ee), orientation=Quaternion.identity()),
            gripper=GripperState(width_m=0.04, max_width_m=scenario.robot.max_gripper_width_m),
        ),
        objects=(
            ObjectState(
                id=TARGET_ID,
                category=scenario.target.category,
                pose=Pose(position=Vec3.of(0.30, 0.0, 0.025), orientation=Quaternion.identity()),
                covariance=Covariance.isotropic(sigma_m, dimension=6),
                confidence=0.9,
                source=Source.FUSION,
                visible=True,
                estimated_size_m=scenario.target.size_m,
            ),
        ),
        scenario_id=scenario.scenario_id,
    )


def _drive(
    policy_name: str,
    *,
    sigma_schedule: list[float],
    scenario: ScenarioConfig | None = None,
):
    """Run one policy for ``len(sigma_schedule)`` steps with perfect tracking.

    Returns the list of ``PolicyOutput`` so a test can compare the two policies
    step by step.
    """
    scenario = scenario or ScenarioConfig(scenario_id="gate_probe", seed=1)
    provider = ToyCartesianProvider()
    policy = create_policy(PolicyConfig(name=policy_name), scenario)
    policy.reset(PolicyConfig(name=policy_name), 1, scenario)
    policy.attach_planner(provider)

    ee = np.array([0.0, 0.0, 0.50], dtype=np.float64)
    joints = np.zeros(scenario.robot.dof, dtype=np.float64)
    outputs = []
    for sigma in sigma_schedule:
        observation = _observation(scenario, sigma_m=sigma, ee=ee, joints=joints)
        output = policy.act(observation)
        outputs.append(output)
        # Perfect tracking: the arm is exactly where it was told to go, in both
        # the joint and the Cartesian sense.
        joints = np.asarray(output.action.arm_targets, dtype=np.float64)
        ee = np.asarray(provider.forward(joints)[0], dtype=np.float64)
    return outputs

#: Confidence gate on policy_v2, read from its declared default rather than
#: restated here.
#:
#: It used to be hard-coded to 0.05 and the test still passed after the default
#: moved to 0.004, because the fixtures happened to sit on either side of both
#: values. A test that keeps passing while the number it claims to pin has
#: changed is worse than no test: it certifies a constant the code no longer
#: holds. The fixtures below are now derived from this, so moving the default
#: moves the test with it and a value outside the range fails loudly.
GATE_SIGMA_M = create_policy(
    PolicyConfig(name="policy_v2"), ScenarioConfig(scenario_id="gate_probe", seed=1)
).max_position_sigma_m

#: A sigma the baseline always tolerates but the candidate does not.
ABOVE_GATE_SIGMA_M = GATE_SIGMA_M * 1.5
BELOW_GATE_SIGMA_M = GATE_SIGMA_M * 0.5
#: Steps needed to reach the grasp waypoint: 6 dwell steps to leave waypoint 0.
#:
#: The toy arm is rate-limited like every other embodiment, so "reaching" the
#: first waypoint now costs a few ticks of travel before the dwell can start.
#: The constant is therefore derived rather than guessed: the policy reports the
#: distance it has left, and the schedule runs until the arm has actually
#: arrived, which keeps the test meaningful if the rate limit or the dwell
#: changes later.
def _steps_to_reach_grasp(scenario: ScenarioConfig) -> int:
    """Longest run of steps that reliably parks the arm on the grasp waypoint."""
    schedule = [BELOW_GATE_SIGMA_M] * 60
    outputs = _drive("policy_v1", sigma_schedule=schedule, scenario=scenario)
    for step, output in enumerate(outputs):
        if int(output.diagnostics["waypoint"]) == 1:
            return step + 1
    raise AssertionError("the baseline never reached the grasp waypoint")


REACH_GRASP_STEPS = _steps_to_reach_grasp(ScenarioConfig(scenario_id="gate_probe", seed=1))


def test_gate_is_inert_below_the_threshold() -> None:
    """Below the threshold the candidate must behave exactly like the baseline.

    This is the property that makes the comparison meaningful: any difference
    measured under low noise would have to be attributed to something other
    than the gate.
    """
    schedule = [BELOW_GATE_SIGMA_M] * REACH_GRASP_STEPS
    baseline = _drive("policy_v1", sigma_schedule=schedule)
    candidate = _drive("policy_v2", sigma_schedule=schedule)
    for step, (b, c) in enumerate(zip(baseline, candidate, strict=True)):
        assert b.action.arm_targets == c.action.arm_targets, f"arms diverged at step {step}"
        assert b.action.gripper_width_m == c.action.gripper_width_m
        assert c.diagnostics["gated"] is False


def test_gate_suspends_the_descent_above_the_threshold() -> None:
    """Above the threshold the candidate must hold the approach pose, not descend.

    The check is on the *action stream*, because that is what the simulator
    sees. A gate that only sets a flag would pass a diagnostics assertion while
    changing nothing.
    """
    settle = [BELOW_GATE_SIGMA_M] * REACH_GRASP_STEPS
    schedule = [*settle, *([ABOVE_GATE_SIGMA_M] * 3)]
    baseline = _drive("policy_v1", sigma_schedule=schedule)
    candidate = _drive("policy_v2", sigma_schedule=schedule)

    # The baseline, which never gates, has advanced to the grasp waypoint.
    assert baseline[-1].diagnostics["waypoint"] == 1
    # The candidate is holding, and says so.
    last = candidate[-1]
    assert last.diagnostics["gated"] is True
    assert last.diagnostics["drive_waypoint"] == 0
    assert last.diagnostics["waypoint"] == 1
    # Holding means the two arms no longer command the same configuration.
    assert last.action.arm_targets != baseline[-1].action.arm_targets
    # And the gripper stays open: the gate exists to avoid committing the grasp.
    assert last.action.gripper_width_m == max(b.action.gripper_width_m for b in baseline)


def test_gate_budget_is_bounded_and_clears() -> None:
    """A gate that never clears would deadlock the episode.

    The hold is bounded by ``max_gate_holds``; once spent, the policy must
    resume descending so that an uncertainty which never resolves costs time
    rather than failing the episode outright.
    """
    scenario = ScenarioConfig(scenario_id="gate_budget", seed=1)
    budget = int(
        create_policy(PolicyConfig(name="policy_v2"), scenario).max_gate_holds
    )
    assert budget > 0, "a zero budget makes the candidate's one change inert"
    settle = [BELOW_GATE_SIGMA_M] * REACH_GRASP_STEPS
    schedule = [*settle, *([ABOVE_GATE_SIGMA_M] * (budget + 3))]
    candidate = _drive("policy_v2", sigma_schedule=schedule, scenario=scenario)

    holds = [int(o.diagnostics["gate_holds"]) for o in candidate]
    assert max(holds) == budget, "the hold counter must stop at the budget"
    assert candidate[-1].diagnostics["gated"] is False
    assert candidate[-1].diagnostics["gate_cleared"] is True
    # Spending exactly the budget is what makes the cost measurable.
    assert sum(1 for o in candidate if o.diagnostics["gated"]) == budget


# --- Plan acceptance -------------------------------------------------------
#
# The IK solver reports ``converged`` when its iteration stops improving, which
# for a pose it cannot quite reach means it returns its best attempt and calls
# that success. On the far transit waypoint that accepted 4.97 mm of position
# error *and* 0.85 rad of orientation error, and the arm then spent the rest of
# the episode rotating 43 degrees towards an orientation it was blocked from
# reaching -- dragging the rigidly-held object with it and failing a placement
# that had otherwise succeeded within 13 mm.
#
# These tests pin the acceptance rule and the excursion gate directly, with no
# simulator involved.


def _ik_result(position_error: float, orientation_error: float, joints: np.ndarray):
    """An ``IKResult`` reporting exactly the residuals under test."""
    from robot_ci.planning.kinematics import IKResult

    return IKResult(
        joint_positions=np.asarray(joints, dtype=np.float64),
        converged=True,
        position_error=position_error,
        orientation_error=orientation_error,
        iterations=200,
    )


def test_a_solve_that_missed_the_orientation_is_not_accepted() -> None:
    """``converged`` alone must not certify a solve.

    This is the exact shape of the defect: a solver that stopped improving on an
    unreachable orientation sets ``converged``, and accepting that is what let
    a 0.85 rad wrist error through.
    """
    policy = create_policy(PolicyConfig(name="policy_v1"), ScenarioConfig(scenario_id="ik"))
    tolerance = policy.ik_options.orientation_tolerance
    joints = np.zeros(7)

    # Converged, and the position is well inside tolerance -- only the
    # orientation betrays it.
    assert not policy._pose_achieved(_ik_result(1e-4, 10 * tolerance, joints))
    # Both residuals inside tolerance: accepted.
    assert policy._pose_achieved(_ik_result(1e-4, tolerance / 2, joints))


def test_a_far_but_exact_plan_is_preferred_against() -> None:
    """A coordinated nearby plan beats an exact one across the room.

    Driving to a distant basin is a sweep through the scene: measured, it sent
    the gripper to 374 mm on the way to a 58 mm waypoint and knocked the object
    off the table. A correct plan that destroys the episode is not a correct
    plan, so a far solution is only taken when nothing within reach is better.
    """
    from robot_ci.policies.rule_based import _IK_MAX_PLANNED_TRAVEL_RAD

    scenario = ScenarioConfig(scenario_id="ik", seed=1)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    dof = scenario.robot.dof
    here = np.zeros(dof)
    observation = _observation(scenario, sigma_m=0.002, ee=np.array([0.0, 0.0, 0.5]), joints=here)

    # A coordinated move: the same total travel spread across every joint.
    coordinated = np.full(dof, _IK_MAX_PLANNED_TRAVEL_RAD / dof)
    assert policy._fallback_reachable(coordinated, observation), (
        "a coordinated move within the budget must be allowed; rejecting it "
        "stalled the arm at 50 mm and never carried the object at all"
    )

    # The same total travel concentrated in one joint is a different basin.
    reversed_joint = np.zeros(dof)
    reversed_joint[0] = _IK_MAX_PLANNED_TRAVEL_RAD + 1.0
    assert not policy._fallback_reachable(reversed_joint, observation)


def test_a_failed_solve_always_has_a_fallback_available() -> None:
    """The travel bound must never leave the arm with nowhere to go.

    It began as a hard veto, and that froze the arm: every solution able to
    reach a distant waypoint needed about 7 rad, so seed 5 held a constant
    55.7 mm IK residual for all 300 ticks and never moved. A veto that prevents
    a bad plan by preventing every plan is not a safety property -- it is a
    guaranteed failure wearing one. The bound is now a preference, and this
    test pins the property that makes that safe: a failed solve always has
    candidates to fall back to.
    """
    scenario = ScenarioConfig(scenario_id="ik", seed=1)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    dof = scenario.robot.dof
    observation = _observation(
        scenario, sigma_m=0.002, ee=np.array([0.0, 0.0, 0.5]), joints=np.zeros(dof)
    )
    assert policy._fallback_seeds(observation), (
        "a failed solve must always have somewhere to go, or the bound below is "
        "acting as a veto"
    )


def test_fallback_seeds_are_deterministic_and_exclude_the_current_pose() -> None:
    """Seeds must be reproducible, and must not repeat the attempt already made.

    Reproducibility is the reason these are a fixed lattice rather than random
    draws: an experiment that re-planned differently on a re-run could not be
    compared against itself.
    """
    scenario = ScenarioConfig(scenario_id="ik", seed=1)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    dof = scenario.robot.dof
    joints = np.zeros(dof)
    observation = _observation(scenario, sigma_m=0.002, ee=np.array([0.0, 0.0, 0.5]), joints=joints)

    first = policy._fallback_seeds(observation)
    second = policy._fallback_seeds(observation)
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b), "fallback seeds must be deterministic"

    # None of them may be the configuration we are already in: re-solving from
    # the same seed is the attempt that just failed.
    for seed in first:
        assert not np.allclose(seed, joints, atol=1e-3)

    # Every seed must lie inside the joint limits it will be solved against.
    lower = policy.joint_lower
    upper = policy.joint_upper
    for seed in first:
        assert np.all(seed >= lower - 1e-9) and np.all(seed <= upper + 1e-9)
