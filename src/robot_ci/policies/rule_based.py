"""Rule-based pick-and-place policies.

Two policies are provided, and they differ in **exactly one documented way**.
That is the whole point: a regression test can only attribute a behavioural
difference to a change if the change is isolated.

===========  ==========================  ==================================
Policy       Perception handling          Expected behaviour
===========  ==========================  ==================================
``policy_v1`` trusts the fused belief     Robust baseline
``policy_v2`` ignores the belief's       Slightly more conservative; expected
             uncertainty                    to be *equal or better* overall
===========  ==========================  ==================================

Under this project's hypothesis the candidate should look the same or better
on aggregate. Whether that is true, and under which conditions it stops being
true, is a Phase 6 question and is **not** asserted here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_ci.core.errors import PolicyError
from robot_ci.models.geometry import Pose, Vec3
from robot_ci.models.scenario import PolicyComponent, PolicyConfig, ScenarioConfig
from robot_ci.models.state import Action, Observation
from robot_ci.planning.kinematics import (
    IKOptions,
    JacobianProvider,
    solve_ik_refined,
)
from robot_ci.policies.base import PolicyDiagnostics, PolicyOutput, RobotPolicy
from robot_ci.tasks import manipulation
from robot_ci.tasks.base import Task, create_task

__all__ = [
    "MockPolicy",
    "PickAndPlacePolicy",
    "create_policy",
    "policy_v1",
    "policy_v2",
]

#: Gripper aperture command for "open" and "closed", in metres.
#:
#: ``OPEN_WIDTH_M`` is the vendored Panda's maximum useful opening. The finger
#: joints travel 0.04 m each, giving a measured pad separation of 0.0262 m;
#: commanding more simply saturates the actuator.
OPEN_WIDTH_M = 0.04
CLOSED_WIDTH_M = 0.0

#: Largest aperture the commanded width maps to. ``_write_gripper`` converts a
#: requested aperture to a per-finger actuator command, and the measured pad
#: separation is ``requested + closed_separation``, so a width above this opens
#: no further.
MAX_WIDTH_M = 0.08

#: How the gripper decides it is holding something. A fixed dwell is the
#: simplest honest signal available without force sensing.
GRASP_DWELL_STEPS = 12
RELEASE_DWELL_STEPS = 10


class PickAndPlacePolicy(RobotPolicy):
    """Execute a task's waypoint sequence using inverse kinematics.

    The policy is a finite state machine over the task's waypoints:

    ``approach -> descend -> grasp -> lift -> transit -> place -> release``

    Each transition is driven by what the policy *believes* (the observation),
    never by ground truth. Where the baseline and the candidate differ is in
    how much they trust that belief -- see :mod:`robot_ci.policies.rule_based`.
    """

    implementation = "rule_based"

    def __init__(self, config: PolicyConfig, scenario: ScenarioConfig) -> None:
        super().__init__(config, scenario)
        self.task: Task = create_task(scenario)
        self.jacobian_provider: JacobianProvider | None = None
        # 200 iterations, not 90: the grasp target sits ~12 mm above a table,
        # and measured against this arm a 90-iteration budget stalls at 7.3 mm
        # while 200 converges to 2.0 mm. The shortfall is not a tolerance the
        # solver ignores -- the damped step simply has not converged yet.
        self.ik_options = IKOptions(
            damping=float(config.parameters.get("ik_damping", 0.05)),
            max_iterations=int(config.parameters.get("ik_iterations", 200)),
            position_tolerance=float(config.parameters.get("ik_pos_tol", 2.0e-3)),
            orientation_tolerance=float(config.parameters.get("ik_rot_tol", 0.08)),
        )
        self.home_seed = np.asarray(scenario.robot.home_position, dtype=np.float64)
        # Multi-start count. Measured on this arm for the derived grasp target:
        # the same target solves to 1.1-1.7 mm from a good seed, but home plus
        # 3 deterministic restarts stalls at 5.9-7.6 mm and 4 restarts reach
        # 2.0 mm. The grasp pose sits low and near the base, where several
        # distinct basins exist, so the policy must sample enough of them.
        self.ik_restarts = int(config.parameters.get("ik_restarts", 6))
        # Grasp aperture that brings the pads onto the object without passing
        # it. Derived from two measured quantities: the object's declared width
        # and the gripper's closed-separation envelope. Commanding zero instead
        # does not hold the object -- the position actuators keep driving and
        # squeeze it out of the jaws, which is what the episode trace showed.
        #
        # A grasp needs *preload*: commanding a separation exactly equal to the
        # object width leaves the pads merely touching, and touching does not
        # hold anything when the arm lifts. The pads are therefore commanded to
        # a fraction of the object width, so the position actuator drives them
        # into the object and maintains a squeeze.
        #
        # Note the direction: a *smaller* fraction is a *tighter* grip, because
        # the commanded separation is fraction * width and the pads then
        # interfere with the object by the difference. 1.0 just touches.
        #
        # This is a policy hyperparameter -- a real gripper exposes the same
        # thing as a force limit -- not a hidden constant, and it is recorded in
        # the config so an experiment can sweep it.
        # The default is measured, not chosen. Sweeping the fraction across
        # 0.55-0.90 and scoring the object height reached *while both pads are
        # still in contact*: any value up to 0.70 holds the object on every seed
        # tried, while 0.80 and 0.90 lose it outright on the seed where they
        # fail. Values below 0.70 buy nothing -- the commanded width has already
        # bottomed out at the measured closed-separation envelope, so 0.55, 0.65
        # and 0.70 are numerically identical.
        self.grasp_width_fraction = float(config.parameters.get("grasp_width_fraction", 0.7))
        object_width = 2.0 * float(scenario.target.size_m)
        self.hold_width_m = max(
            0.0,
            self.grasp_width_fraction * object_width
            - float(scenario.robot.gripper_closed_separation_m),
        )
        # Difference between the two policies: how much positional uncertainty
        # the policy tolerates before it stops trusting its object estimate and
        # falls back to a slower, more conservative approach.
        limits = scenario.robot.limits
        self.joint_lower = np.asarray(limits.lower, dtype=np.float64)
        self.joint_upper = np.asarray(limits.upper, dtype=np.float64)
        self.max_position_sigma_m = float(config.parameters.get("max_position_sigma_m", 1e9))
        #: Extra clearance added to every waypoint above the support surface.
        self.clearance_bias_m = float(config.parameters.get("clearance_bias_m", 0.0))
        #: Grasp offset applied along the approach axis, to sit between the
        #: fingers rather than on the object centre.
        self.grasp_bias_m = float(config.parameters.get("grasp_bias_m", 0.0))
        #: Confidence-gate budget, in control steps.
        #:
        #: This is the candidate's documented behavioural change. While the
        #: fused object pose is *less* certain than :attr:`max_position_sigma_m`
        #: the policy refuses to commit the descent: it holds the approach pose
        #: and re-observes, because a grasp executed from an untrusted estimate
        #: is how a manipulator damages itself or the part. The budget is
        #: bounded so that an uncertainty which never resolves delays the
        #: episode by at most this many steps instead of deadlocking it.
        #:
        #: The baseline is unaffected by this code path for a structural reason
        #: rather than a conventional one: its threshold is ``1e9``, so
        #: ``uncertain`` is never true and the gate can never engage. That is
        #: what keeps the two policies separated by exactly one documented
        #: change -- the threshold -- with no duplicated policy code.
        self.max_gate_holds = int(config.parameters.get("max_gate_holds", 10))
        self._init_motion_limits(scenario)

    def _init_motion_limits(self, scenario: ScenarioConfig) -> None:
        """Derive the joint slew bound and the joint-limit arrays.

        Per-joint slew bound, in radians per control tick.

        The IK solution is a *pose* target, and sending it straight to the
        position actuators asks the arm to arrive at the next waypoint in one
        control period. Measured on the waypoint sequence that is ~1 m/s of
        end-effector travel, which is not a slow approach, it is a flick: the
        pads arrive at the object with enough momentum to throw it back out, so
        the grasp registers and is then lost during the lift. Nothing in the
        episode reports this as a control fault -- the arm reaches every
        waypoint, on time, and the object is simply no longer held.

        The bound is the embodiment's own declared velocity limit over the
        control period, so it is a property of the robot rather than a tuning
        constant. A scenario that declares no velocity limits is treated as
        unconstrained and keeps the previous behaviour, because inventing a
        limit would silently change results for anyone who set none.
        """
        limits = scenario.robot.limits
        self.joint_lower = np.asarray(limits.lower, dtype=np.float64)
        self.joint_upper = np.asarray(limits.upper, dtype=np.float64)
        self.max_joint_step: np.ndarray | None = None
        if len(limits.max_velocity) == len(limits.lower):
            self.max_joint_step = np.asarray(limits.max_velocity, dtype=np.float64) / float(
                scenario.robot.control_hz
            )

    # -- lifecycle ---------------------------------------------------------
    def reset(self, config: PolicyConfig, seed: int, scenario: ScenarioConfig | None = None) -> None:
        """Begin a new episode."""
        self.config = config
        if scenario is not None:
            self.scenario = scenario
        self.seed = seed
        self.step_count = 0
        self._state = {
            "waypoint_index": 0,
            "drive_index": 0,
            "dwell": 0,
            "close_dwell": 0,
            "phase": "approach",
            "planned_joints": None,
            "waypoints": [],
            "ik_converged": True,
            "last_ik_error_m": 0.0,
            "belief_sigma_m": 0.0,
            "gate_holds": 0,
            "gate_cleared": False,
        }
        self.task = create_task(self.scenario)
        self.jacobian_provider = None
        # A reset may arrive with a different embodiment, and the slew bound and
        # the joint limits are properties of that embodiment rather than of the
        # policy, so both are re-derived whenever the scenario changes.
        self._init_motion_limits(self.scenario)

    def attach_planner(self, provider: JacobianProvider | None) -> None:
        """Attach a Jacobian source so the policy can solve IK.

        Called by the experiment runner, which owns the backend. Keeping the
        provider injectable is what lets this same policy run against MuJoCo
        (analytic Jacobian) and against a kinematic-only backend
        (finite-difference Jacobian).
        """
        self.jacobian_provider = provider

    # -- policy ------------------------------------------------------------
    def act(self, observation: Observation) -> PolicyOutput:
        """Advance the state machine and emit a joint-position command."""
        step = self._next()
        diagnostics = self.get_diagnostics()
        state = self._state

        target = self._resolve_target(observation)
        if target is None:
            diagnostics["reason"] = "no object believed visible"
            state["phase"] = "search"
            return PolicyOutput(
                Action.joint_positions(observation.robot.positions_array), diagnostics
            )

        belief = observation.best_object
        sigma = float(belief.position_sigma_m) if belief is not None else 0.0
        state["belief_sigma_m"] = sigma
        uncertain = sigma > self.max_position_sigma_m

        if not state["waypoints"]:
            state["waypoints"] = [w.array.copy() for w in self.task.waypoints(target)]
            self._apply_clearance_bias(state["waypoints"])
            state["waypoint_index"] = 0

        waypoints = state["waypoints"]
        index = int(state["waypoint_index"])

        # -- confidence gate (the candidate's behavioural change) -----------
        # While the object estimate is too uncertain to trust, refuse to
        # commit the descent: hold the approach pose and re-observe. Only the
        # grasp waypoint is gated -- once the object is held, an uncertain
        # estimate is no longer a reason to stop, and gating the carry would
        # command the arm back toward a pose the object has left.
        gated = self._gate_active(index, uncertain, state)
        drive = 0 if gated else index
        state["drive_index"] = drive

        joints = self._solve_for(observation, target)
        if joints is None:
            diagnostics["reason"] = "IK unavailable; holding position"
            return PolicyOutput(Action.joint_positions(observation.robot.positions_array), diagnostics)

        current_waypoint = waypoints[min(drive, len(waypoints) - 1)]

        gripper = self._gripper_command(observation, state, gated=gated)
        action = self._compose(joints, gripper)

        # Advance the waypoint cursor once the arm has held the commanded pose
        # for a dwell. Leaving the grasp waypoint additionally requires the
        # fingers to have finished closing, so the lift never starts before the
        # grasp is physically complete. A gated step holds its dwell at zero:
        # otherwise the hold would bank progress and the grasp would be
        # committed on the very first step after the gate expired, which is the
        # opposite of waiting for a better estimate.
        ee = observation.robot.end_effector_pose.position.array
        reached = float(np.linalg.norm(ee - current_waypoint)) < 0.02
        state["dwell"] = 0 if gated else (state["dwell"] + 1 if reached else 0)
        grasp_settled = index != 1 or int(state.get("close_dwell", 0)) >= 6
        if (
            not gated
            and state["dwell"] >= 6
            and grasp_settled
            and index < len(waypoints) - 1
        ):
            state["waypoint_index"] = index + 1
            state["dwell"] = 0
            state["close_dwell"] = 0
            state["planned_joints"] = None
            # Drop the plan's waypoint tag with the plan itself, so the next
            # step re-plans instead of inheriting a configuration that was
            # solved for a waypoint the arm has already left.
            state["planned_waypoint"] = None

        diagnostics.update(
            {
                "waypoint": index,
                "drive_waypoint": drive,
                "waypoint_count": len(waypoints),
                "reached_waypoint": reached,
                "dwell": state["dwell"],
                "belief_sigma_m": round(sigma, 5),
                "uncertain_belief": uncertain,
                "gated": gated,
                "gate_holds": int(state.get("gate_holds", 0)),
                "gate_cleared": bool(state.get("gate_cleared", False)),
                "gripper_open": gripper > 0.5 * OPEN_WIDTH_M,
                "ik_converged": state["ik_converged"],
                "ik_error_mm": round(state["last_ik_error_m"] * 1000.0, 2),
                "waypoint_target": [round(float(v), 4) for v in current_waypoint],
                "step": step,
            }
        )
        return PolicyOutput(action, diagnostics)

    def _gate_active(self, index: int, uncertain: bool, state: dict[str, Any]) -> bool:
        """Whether the confidence gate should suspend the descent this step.

        Args:
            index: The logical waypoint index (independent of the gate).
            uncertain: Whether the fused pose is beyond the trust threshold.
            state: Mutable policy state; the hold counter is advanced here.

        Returns:
            ``True`` when the policy should hold the approach pose instead of
            descending to the grasp.
        """
        if self.max_gate_holds <= 0 or bool(state.get("gate_cleared")):
            return False
        if not uncertain:
            return False
        # Only the grasp commit is gated; see the call site for why.
        if index != 1:
            return False
        holds = int(state.get("gate_holds", 0))
        if holds >= self.max_gate_holds:
            # Budget exhausted. Clear the gate for the rest of the episode so
            # the policy degrades to baseline behaviour rather than holding
            # until the episode times out.
            state["gate_cleared"] = True
            return False
        state["gate_holds"] = holds + 1
        return True

    def _apply_clearance_bias(self, waypoints: list[np.ndarray]) -> None:
        """Raise the carry waypoints by ``clearance_bias_m``.

        Applies to the lift and the transit-over-target waypoints only. The
        release pose is deliberately left alone: raising it would make the
        object be released above the target, which fails placement instead of
        protecting it. A bias of ``0.0`` -- the default for both compared
        policies -- is a no-op, so this cannot silently confound the
        baseline/candidate comparison.
        """
        if self.clearance_bias_m == 0.0:
            return
        for position in waypoints[2:4]:
            position[2] = position[2] + self.clearance_bias_m

    # -- internals ---------------------------------------------------------
    def _resolve_target(self, observation: Observation) -> Pose | None:
        """Return the pose of the task's target object, as believed."""
        state = self._state
        target_id = self.task.target_object_id()
        for obj in observation.objects:
            if obj.id == target_id:
                return obj.pose
        # Fall back to the most confident object, and remember it: a policy that
        # loses track mid-episode should not thrash between candidates.
        if state.get("locked_id") is None:
            best = observation.best_object
            if best is not None:
                state["locked_id"] = best.id
                return best.pose
            return None
        for obj in observation.objects:
            if obj.id == state["locked_id"]:
                return obj.pose
        return None

    def _solve_for(self, observation: Observation, target: Pose) -> np.ndarray | None:
        """Solve IK for the current waypoint, warm-started from the last solution."""
        state = self._state
        if self.jacobian_provider is None:
            return None
        waypoints = state["waypoints"]
        # The drive index, not the logical index: when the confidence gate is
        # holding the approach pose the logical index has already advanced to
        # the grasp, and solving for it would descend anyway.
        index = int(state.get("drive_index", state["waypoint_index"]))
        if not waypoints:
            return None
        goal = waypoints[min(index, len(waypoints) - 1)]

        # The gripper approaches from above, so the waypoint is expressed as a
        # position and a fixed top-down orientation.
        command_pose = Pose(position=Vec3.from_iterable(goal), orientation=TOP_DOWN)

        # Plan once per waypoint, then execute that plan.
        #
        # Re-solving every control tick looks natural and is wrong twice over.
        # It costs a full IK solve per step, and -- because the solver is
        # seeded from the previous *solution* rather than from the arm -- each
        # solve may land in a different basin, so the commanded configuration
        # moves while the arm is still travelling toward the old one. Under a
        # velocity limit that becomes a visible excursion: measured, the
        # end-effector wandered 40 mm sideways while carrying the object and
        # shook it out of the gripper. A plan is a commitment: it is computed
        # when the waypoint changes and held until then, which is also what a
        # real motion controller does.
        if state.get("planned_waypoint") == index and state.get("planned_joints") is not None:
            return self._rate_limit(observation, np.asarray(state["planned_joints"]))

        seed_q = state.get("planned_joints")
        if seed_q is None:
            seed_q = observation.robot.positions_array
        # Joint limits MUST be passed. Without them the solver happily returns a
        # configuration outside the Panda's range, the backend clamps it on the
        # way to the actuators, and the arm silently stops where it was
        # clamped -- which looks exactly like a control failure.
        result = solve_ik_refined(
            self.jacobian_provider,
            command_pose,
            np.asarray(seed_q),
            lower=self.joint_lower,
            upper=self.joint_upper,
            options=self.ik_options,
            restarts=self.ik_restarts,
        )
        if not result.converged:
            # Retry from the home configuration.
            #
            # A sequential waypoint policy warm-starts each solve from the arm's
            # current configuration, which for the low grasp target lands in a
            # basin the damped solver cannot leave: measured 13.7 mm from the
            # pre-grasp pose versus 1.97 mm from home, for the same target. The
            # deterministic restarts only perturb the failing seed, so they
            # cannot escape it; offering home as a fresh basin can.
            seeded = solve_ik_refined(
                self.jacobian_provider,
                command_pose,
                self.home_seed,
                lower=self.joint_lower,
                upper=self.joint_upper,
                options=self.ik_options,
                restarts=self.ik_restarts,
            )
            if seeded.position_error < result.position_error:
                result = seeded
        state["planned_joints"] = result.joint_positions
        state["planned_waypoint"] = index
        state["ik_converged"] = result.converged
        state["last_ik_error_m"] = result.position_error
        del target
        return self._rate_limit(observation, result.joint_positions)

    def _rate_limit(self, observation: Observation, goal: np.ndarray) -> np.ndarray:
        """Bound how far the joints may travel on this control tick.

        The returned value is a *command*, not the IK solution: it is the point
        on the segment from the arm's current configuration to the solution that
        the velocity limits allow this tick. When no limit is declared the
        solution is passed through unchanged, so an unconstrained scenario
        behaves exactly as it did before this existed.

        Args:
            observation: Supplies the arm's measured configuration.
            goal: The IK solution for the current waypoint.

        Returns:
            The joint command to send this tick.
        """
        if self.max_joint_step is None:
            return goal
        current = observation.robot.positions_array
        delta = goal - current
        # A per-joint bound is a box, not a ball: clamp each axis independently.
        # Scaling the whole vector by a single factor would satisfy the limit
        # only by slowing the slowest joint to the fastest one's rate.
        step = np.clip(delta, -self.max_joint_step, self.max_joint_step)
        return current + step

    def _gripper_command(
        self, observation: Observation, state: dict[str, Any], *, gated: bool = False
    ) -> float:
        """Decide the gripper aperture for this step.

        The gripper has its own dwell, separate from the waypoint dwell. Advancing
        to the lift waypoint while the fingers are still travelling would lift
        an object the gripper never actually closed on, and the failure is
        invisible in the diagnostics: the arm reaches every waypoint on time and
        the object simply never moves.

        While the confidence gate is holding, the aperture is forced open: the
        gate exists to avoid committing a grasp from an untrusted estimate, and
        closing the fingers during the hold would commit it anyway.
        """
        index = int(state["waypoint_index"])
        close_dwell = int(state.get("close_dwell", 0))
        state["close_dwell"] = close_dwell

        if gated:
            return OPEN_WIDTH_M
        if index == 0:
            # Above the object, still open.
            return OPEN_WIDTH_M
        if index == 1:
            # At the object. Close, then hold until the fingers have actually
            # stopped moving before the lift is allowed to start.
            if state["dwell"] < 3:
                return OPEN_WIDTH_M
            state["close_dwell"] = close_dwell + 1
            return self.hold_width_m
        if index >= 4:
            return OPEN_WIDTH_M if state["dwell"] >= 4 else self.hold_width_m
        # Lifting, in transit, or placing: keep holding the grasp.
        return self.hold_width_m

    def _compose(self, joints: np.ndarray, gripper_width_m: float) -> Action:
        """Combine the arm and gripper commands into one backend action.

        Both channels must be commanded on the same control tick: closing the
        gripper a step after arriving at the grasp pose is a real, measurable
        source of grasp failure, and separating them would hide that effect.
        """
        return Action.arm_and_gripper(
            joints, gripper_width_m, arm_joint_count=self.scenario.robot.dof
        )

    def get_diagnostics(self) -> PolicyDiagnostics:
        """Return the policy's internal state as structured diagnostics."""
        return PolicyDiagnostics(
            policy=self.config.name,
            version=self.config.version,
            perception=self.config.component_version(PolicyComponent.PERCEPTION),
            planner=self.config.component_version(PolicyComponent.PLANNER),
            controller=self.config.component_version(PolicyComponent.CONTROLLER),
            step_count=self.step_count,
            phase=self._state.get("phase", "unknown"),
            waypoint=self._state.get("waypoint_index", 0),
            belief_sigma_m=round(float(self._state.get("belief_sigma_m", 0.0)), 5),
            max_position_sigma_m=self.max_position_sigma_m,
        )


#: The policy uses the task's grasp orientation, so the frame a policy commands
#: and the frame the task plans are guaranteed to be the same. They diverged
#: once, when each defined its own constant, and the resulting silent failure
#: cost an entire debugging cycle.
TOP_DOWN = manipulation.TOP_DOWN

def policy_v1(
    config: PolicyConfig | None = None, scenario: ScenarioConfig | None = None
) -> PickAndPlacePolicy:
    """Baseline: trusts its fused object estimate completely."""
    return PickAndPlacePolicy(
        config or _DEFAULT_V1, scenario or ScenarioConfig()
    )


def policy_v2(
    config: PolicyConfig | None = None, scenario: ScenarioConfig | None = None
) -> PickAndPlacePolicy:
    """Candidate: refuses to act on a high-uncertainty estimate.

    The single change from :func:`policy_v1` is a confidence gate on the fused
    object pose. A regression run will show whether that gate helps on average
    and whether it introduces a conditional regression; the framework asserts
    neither, and measures both.
    """
    return PickAndPlacePolicy(
        config or _DEFAULT_V2, scenario or ScenarioConfig()
    )


class MockPolicy(RobotPolicy):
    """A policy that does nothing useful, for exercising the harness.

    Useful for measuring the *floor* of a benchmark: any task the mock policy
    solves is being solved by luck, and any task it fails is not evidence about
    the framework.
    """

    implementation = "mock"

    def reset(self, config: PolicyConfig, seed: int, scenario: ScenarioConfig | None = None) -> None:
        """Begin a new episode holding the home configuration."""
        self.config = config
        self.seed = seed
        self.step_count = 0
        self._state = {"reason": "mock policy performs no action"}

    def act(self, observation: Observation) -> PolicyOutput:
        """Hold the current configuration."""
        self._next()
        return PolicyOutput(
            Action.joint_positions(observation.robot.positions_array),
            PolicyDiagnostics(policy=self.config.name, reason="idle"),
        )


#: Default configurations for the two compared policies.
#:
#: Keyed by the name an experiment file refers to. A config that names a policy
#: inherits that policy's defaults, which is what makes ``policy: policy_v2`` in
#: a YAML file actually mean policy_v2 -- see :func:`resolve_policy_config`.
_DEFAULT_V1 = PolicyConfig(
    name="policy_v1",
    version="1.0.0",
    components={
        PolicyComponent.PERCEPTION: "fusion_v1",
        PolicyComponent.PLANNER: "waypoint_v1",
        PolicyComponent.CONTROLLER: "ik_dls_v1",
    },
    parameters={"max_position_sigma_m": 1e9},
)

_DEFAULT_V2 = PolicyConfig(
    name="policy_v2",
    version="2.0.0",
    components={
        PolicyComponent.PERCEPTION: "fusion_v1_gated",
        PolicyComponent.PLANNER: "waypoint_v1",
        PolicyComponent.CONTROLLER: "ik_dls_v1",
    },
    # ``clearance_bias_m`` is deliberately 0.0, not 0.01. Raising the carry
    # waypoints is a *second* behavioural change, and the whole point of a
    # two-arm comparison is that a measured difference is attributable to one
    # change. With the bias non-zero the version-2 regression could be caused by
    # either the gate or the clearance, and this grid could not separate them.
    # The parameter stays implemented (``_apply_clearance_bias``) and declared
    # here so an ablation can exercise it as its own factor.
    parameters={"max_position_sigma_m": 0.05, "clearance_bias_m": 0.0},
)

#: Registry of policy factories, keyed by :attr:`PolicyConfig.implementation`.
_REGISTRY: dict[str, Any] = {
    "rule_based": policy_v1,
    "policy_v1": policy_v1,
    "policy_v2": policy_v2,
    "mock": lambda config, scenario: MockPolicy(config, scenario),
}

#: Default configuration per *named* policy, so a config that only carries a
#: name can be completed into the policy that name denotes.
_NAMED_DEFAULTS: dict[str, PolicyConfig] = {
    "policy_v1": _DEFAULT_V1,
    "policy_v2": _DEFAULT_V2,
}


def resolve_policy_config(config: PolicyConfig) -> PolicyConfig:
    """Fill in the defaults of the policy ``config`` names.

    A policy config that arrives from an experiment file carries only a name --
    ``candidate: {policy: policy_v2}``. The parameters that actually make
    policy_v2 *behave* differently from policy_v1 (its confidence gate and the
    associated clearance bias) live in that policy's default config, and the
    factories only apply them when they are handed ``None``.

    The result was that a config naming ``policy_v2`` instantiated a policy that
    was behaviourally ``policy_v1`` under a ``policy_v2`` label. Both arms ran,
    both produced plausible metrics, and the harness reported no difference
    between two policies it had never actually distinguished. Nothing about the
    output looked wrong.

    So: the named policy's defaults are the base, and anything the caller set
    explicitly wins. Naming a policy now means getting that policy.

    Args:
        config: A policy configuration, possibly carrying only a name.

    Returns:
        An equivalent config with the named policy's defaults merged in, or the
        input unchanged when the name is not a known policy.
    """
    base = _NAMED_DEFAULTS.get(config.name) or _NAMED_DEFAULTS.get(config.implementation)
    if base is None:
        return config
    return config.model_copy(
        update={
            "components": {**base.components, **config.components},
            "parameters": {**base.parameters, **config.parameters},
        }
    )


def create_policy(config: PolicyConfig, scenario: ScenarioConfig) -> RobotPolicy:
    """Instantiate the policy a configuration names.

    Args:
        config: The policy configuration. Defaults belonging to the named
            policy are merged in before construction.
        scenario: The condition the policy will be evaluated under.

    Returns:
        A ready policy instance.

    Raises:
        PolicyError: If the implementation is not registered.
    """
    factory = _REGISTRY.get(config.implementation)
    if factory is None:
        raise PolicyError(
            f"unknown policy implementation {config.implementation!r}",
            code="policy.unknown_implementation",
            requested=config.implementation,
            available=sorted(_REGISTRY),
        )
    return factory(resolve_policy_config(config), scenario)
