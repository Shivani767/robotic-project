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

from itertools import pairwise
from typing import Any

import numpy as np

from robot_ci.core.errors import PolicyError
from robot_ci.models.geometry import Pose, Vec3
from robot_ci.models.scenario import PolicyComponent, PolicyConfig, ScenarioConfig
from robot_ci.models.state import Action, Observation
from robot_ci.planning.kinematics import (
    IKOptions,
    IKResult,
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

#: How many alternative IK seeds to try when a solve does not reach the pose.
#:
#: A workspace scan for the failing waypoint found a 0.68 mm / 0.005 rad
#: solution within a few hundred random seeds, so the pose is comfortably
#: reachable and only the seed was wrong. Twelve keeps the worst case bounded
#: and costs nothing on the common path, since the fallback is only entered when
#: the warm start actually failed.
_IK_FALLBACK_SEED_COUNT = 12

#: Step between successive fallback seeds, as a fraction of the joint range.
#:
#: Irrational so the walk does not resonate with the joint count and revisit the
#: same configurations.
_IK_SEED_STRIDE = 0.7548776662466927

#: Largest total joint rotation a fallback plan may require of the arm, in radians.
#:
#: Not a velocity or acceleration limit -- the rate limiter already bounds how
#: fast the arm moves. This is a bound on *where the plan ends up*: a solution
#: in a distant basin may satisfy the pose exactly and still be wrong, because
#: driving there takes the arm through the scene.
#:
#: Summed over the joints rather than taken as the largest single rotation,
#: because the two cases are not separable per joint. The real transit of this
#: scene needs about 2.8 rad on one joint, so a per-joint bound tight enough to
#: reject a wild sweep also rejects the genuine move. Summing distinguishes
#: them: a coordinated move spreads its travel, while a different basin
#: reverses several joints at once, and it is the reversal that sweeps the
#: scene. The Panda's joints span roughly 17 rad in total, so this admits a
#: substantial repositioning while excluding the across-the-workspace plans
#: that wreck the episode.
_IK_MAX_PLANNED_TRAVEL_RAD = 6.0

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


#: Aperture stability gate, in metres per tick.
#:
#: The lift must not fire while the fingers are still travelling. The previous
#: gate compared the measured aperture against an absolute on-cube mark
#: (``2*size - closed_separation + band``) derived from the *live Gazebo*
#: backend's width semantics. That mark is falsified by MuJoCo: measured on
#: this model, a successful 16 mm grasp settles at a ``width_m`` of 11.6 mm
#: (fingers squeezing the cube ~2 mm per side under the position actuators),
#: never at the 5 mm the Gazebo formula predicts, so the gate never opened and
#: every nominal MuJoCo episode stalled at the grasp waypoint (0/12 seeds at
#: commit d7026af; see ``tests/integration/test_default_scenario_solvable.py``).
#:
#: The physical fact the gate needs is *convergence*, and convergence is
#: measurable on any backend without knowing its width semantics: the close is
#: settled when the measured aperture stops changing. Measured: MuJoCo's
#: steady-state drift after settling is ~1e-7 m/tick; the live Gazebo close
#: creeps at ~4 mm/s = 1.2 mm/tick at the 0.3 s control period, which this
#: threshold refuses. 0.2 mm/tick sits two orders of magnitude above the
#: settled drift and an order below the creep.
_APERTURE_STABLE_EPS_M = 0.0002

#: Consecutive ticks the aperture must hold within ``_APERTURE_STABLE_EPS_M``
#: before the grasp counts as settled. Three ticks is one second of silence at
#: the control period -- long enough that a momentary contact bounce cannot
#: masquerade as convergence, short enough not to dominate the step budget.
_APERTURE_STABLE_TICKS = 3


class PickAndPlacePolicy(RobotPolicy):
    """Execute a task's waypoint sequence using inverse kinematics.

    The policy is a finite state machine over the task's waypoints:

    ``approach -> descend -> grasp -> lift -> transit -> place -> release``

    Each transition is driven by what the policy *believes* (the observation),
    never by ground truth. Where the baseline and the candidate differ is in
    how much they trust that belief -- see :mod:`robot_ci.policies.rule_based`.
    """

    #: Grasp aim clearance floor tolerance, in metres.
    #:
    #: The jaw may not descend so low that the pad's lower edge bites into the
    #: support surface. Measured, the task's derived aim placed the pad's lower
    #: edge 0.5-0.9 mm *below* the tabletop, and the ~4.3 N of pad-on-table
    #: Coulomb friction arrested the close at ~19.8 mm pad separation -- 2.1 mm
    #: short of cube contact, with zero preload by construction.
    #:
    #: This is an embodiment-derived floor, computed from the robot's measured
    #: pad half-height and the table height, never calibrated. It is applied on
    #: top of the task's own aim derivation, so the grasp and approach
    #: waypoints are shifted together and their relative geometry is unchanged.
    _GRASP_AIM_CLEARANCE_TOLERANCE_M = 0.001

    #: Measured aim execution shortfall, in metres (opt-in per configuration).
    #:
    #: The live Gazebo evidence (five seed-3/0 mm runs at commit 7a9014e) showed
    #: the commanded grasp waypoint is not the achieved pose on that backend:
    #: the policy commanded jaw_z = 0.0095 m on every run while the end-effector
    #: settled at 0.0080 m on every run (IK residual 0.77 mm the rest of the way
    #: down as execution tracking), so the *achieved* pad bottom sat at -0.5 mm
    #: and the table friction persisted -- the command was above the table but
    #: the pads were not. Where a backend exhibits such a deficit, the
    #: clearance floor must be raised by the measured amount so the *achieved*
    #: pose clears the table.
    #:
    #: This is a backend property, not a policy truth: adding it unconditionally
    #: (as commit d7026af did) overshoots on MuJoCo, which executes the command
    #: accurately -- the pads then grip 1.5 mm above the object centre and the
    #: placement orientation error grows past tolerance (measured: 0.485 rad,
    #: nominal seeds failing). It is therefore a configuration parameter,
    #: default 0.0, and the Gazebo evidence scripts set it to their measured
    #: 0.0015. If a backend shows a different deficit, this parameter -- not
    #: the floor itself -- is what moves.
    DEFAULT_SETTLE_SHORTFALL_M = 0.0

    implementation = "rule_based"

    def __init__(self, config: PolicyConfig, scenario: ScenarioConfig) -> None:
        super().__init__(config, scenario)
        self.task: Task = create_task(scenario)
        self.jacobian_provider: JacobianProvider | None = None
        # 200 iterations, not 90: the grasp target sits ~12 mm above a table,
        # and measured against this arm a 90-iteration budget stalls at 7.3 mm
        # while 200 converges to 2.0 mm. The shortfall is not a tolerance the
        # solver ignores -- the damped step simply has not converged yet.
        # 0.5 mm, not 2 mm. The grasp is the precision bottleneck: measured
        # across seeds, the end-effector arrives within 2.6-3.0 mm of the object
        # on the seeds that succeed and 3.9-4.6 mm on the ones that fail, so the
        # usable band is about +/-1 mm wide. A solver that is content to stop at
        # 2 mm is therefore no more accurate than the thing it has to hit, and it
        # lands inside the band roughly half the time.
        #
        # Measured over 8 seeds, changing nothing else:
        #
        #     ik_pos_tol = 2e-3   0.500
        #     ik_pos_tol = 1e-3   0.625
        #     ik_pos_tol = 5e-4   0.750
        #
        # A loose tolerance was not a free choice. `converged` means "stopped
        # improving", so a 2 mm tolerance silently accepted plans the arm could
        # not execute to the precision the grasp needs, and the episode failed at
        # the object with no diagnostic pointing at the planner.
        self.ik_options = IKOptions(
            damping=float(config.parameters.get("ik_damping", 0.05)),
            max_iterations=int(config.parameters.get("ik_iterations", 200)),
            position_tolerance=float(config.parameters.get("ik_pos_tol", 5.0e-4)),
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
        #: Backend-measured execution shortfall compensation, in metres. See
        #: :attr:`DEFAULT_SETTLE_SHORTFALL_M` for why this is opt-in: only the
        #: backend whose tracking deficit was actually measured (live Gazebo,
        #: 0.0015 m) should set it, and a backend that executes commands
        #: accurately (MuJoCo) must leave it at zero.
        self.settle_shortfall_m = float(
            config.parameters.get("settle_shortfall_m", self.DEFAULT_SETTLE_SHORTFALL_M)
        )
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
        #: Exponential-smoothing factor for the believed object position, in
        #: ``[0, 1)``. Zero disables it.
        #:
        #: This is the change that makes ``policy_v3`` a usable *positive
        #: control* for the regression machinery, so its trade-off is deliberate
        #: and documented rather than accidental. Smoothing trades latency for
        #: variance: it suppresses the per-tick jitter of a noisy estimate, and
        #: it lags behind a genuinely moved object by the same amount. So it
        #: helps when depth noise is the dominant error and hurts when sensor
        #: latency is, which is exactly the shape of conditional regression the
        #: framework exists to find.
        #: Control ticks the candidate spends collecting pose samples before it
        #: commits to a plan. Zero disables settling.
        #:
        #: The average taken over those samples is a plain mean. An exponential
        #: weighting parameter was declared here and removed: it was validated
        #: and stored but never read, because the settle window is a fixed
        #: length and a mean is what that window produces. A dead knob is worse
        #: than no knob, because an ablation that varies it measures nothing and
        #: reports the absence of an effect as a finding.
        self.settle_steps = int(config.parameters.get("settle_steps", 0))
        if self.settle_steps < 0:
            raise ValueError(f"settle_steps must be >= 0, got {self.settle_steps}")
        #: Running sum of observed positions, and how many have been seen.
        self._smoothed: np.ndarray | None = None
        self._settle = 0
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
        self._reset_estimate()
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
            # Rolling aperture trace for the stability gate; reset with the
            # episode so a previous episode's convergence cannot leak in.
            "aperture_trace": [],
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
            planned = target.position.array
            if self.settle_steps > 0:
                averaged = self._settled_estimate(target.position.array)
                if averaged is None:
                    # Still collecting. Hold position rather than plan on one
                    # noisy sample: the whole point of the candidate is not to
                    # do that.
                    diagnostics["settling"] = self._settle
                    return PolicyOutput(
                        Action.joint_positions(observation.robot.positions_array), diagnostics
                    )
                planned = averaged
            state["waypoints"] = [w.array.copy() for w in self.task.waypoints(target)]
            if self.settle_steps > 0:
                # Shift the grasp waypoint onto the averaged estimate. Only the
                # grasp is re-aimed: the place waypoint is a property of the
                # task, not of where the object was seen.
                offset = planned - np.asarray(target.position.array, dtype=np.float64)
                state["waypoints"][1] = state["waypoints"][1] + offset
            self._apply_clearance_bias(state["waypoints"])
            self._apply_aim_clearance_floor(state["waypoints"])
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
        # The grasp commit is complete once the fingers have closed and held
        # that separation for a sustained dwell: the lift must not start
        # before the jaw is physically holding the object. Measured, the task's
        # derived aim placed the pad's lower edge 0.5-0.9 mm *below* the
        # tabletop with the ~4.3 N of pad-on-table friction arresting the close
        # at ~19.8 mm pad separation; the jaw then sat ~21.3 mm with the cube
        # face still 2.1 mm away, so a policy that lifted on the old six-tick
        # dwell never applied preload by construction. The dwell is now twice
        # that long, which is comfortably above the 0.3 s per-tick control
        # period and below the episode's 300 tick budget, so a successful
        # grasp still commits in a few seconds instead of hanging the episode.
        #
        # Dwell alone proved insufficient a second time: with the aim fixed,
        # the close still creeps from open to the cube at ~4 mm/s under the
        # overdamped finger PD, so at close_dwell 12 the aperture still sits
        # ~4 mm wide of the object and the lift fires into a flick. The lift
        # therefore additionally requires the measured aperture to have
        # *stopped moving*: the grasp is not "settled" until the fingers have
        # demonstrably finished travelling, not merely been commanded to. The
        # convergence test (rather than an absolute on-cube mark) is what makes
        # it valid on every backend -- see ``_aperture_settled``.
        grasp_settled = index != 1 or (
            int(state.get("close_dwell", 0)) >= 12
            and self._aperture_settled(observation, state)
        )
        if (
            not gated
            and state["dwell"] >= 12
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
                "ik_orientation_error_rad": round(
                    float(state.get("last_ik_orientation_error_rad", 0.0)), 4
                ),
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

    def _apply_aim_clearance_floor(self, waypoints: list[np.ndarray]) -> None:
        """Raise the grasp centre waypoint so the pads clear the support surface.

        The task derives the aim at ``obj.z + pad_offset``, putting the pad
        centre on the object's centre, and the table clearance is then a *check*
        rather than the aim. Measured against the compiled model, that aim put
        the pad's lower edge 0.5-0.9 mm *below* the tabletop, and the ~4.3 N of
        pad-on-table Coulomb friction arrested the close at ~19.8 mm pad
        separation -- 2.1 mm short of cube contact, with zero preload by
        construction.

        This floor is derived from the embodiment and the environment (never
        calibrated): the pad's lower edge must sit at least
        ``_GRASP_AIM_CLEARANCE_TOLERANCE_M`` above the table, so the jaw must be
        no lower than ``table_z + pad_half + _GRASP_AIM_CLEARANCE_TOLERANCE_M``.
        On a backend whose tracking deficit has been measured (live Gazebo:
        command 0.0095 m, achieved 0.0080 m, five for five runs), the floor also
        adds ``settle_shortfall_m`` so the *achieved* pose -- not the command --
        is what sits above the table; on a backend that executes commands
        accurately the parameter stays at its default of zero. Lifting the whole
        aim shifts grasp and approach waypoints together, so their relative
        geometry is unchanged.

        Raising the grasp aim also shifts where the object rides in the grip by
        the same amount: the pads close on the object at a higher point, so the
        object hangs lower relative to the tool centre point by exactly
        ``raise_by``. The lift and transit waypoints are derived as
        ``obj.z + clearance`` under the unreraised assumption, so they must be
        raised by ``raise_by`` too or the carried object silently travels
        ``raise_by`` below its clearance. Measured: at seed 8 that deficit was
        the whole margin (object peak 49.3 mm against a 50 mm lift criterion,
        after the 1 mm floor), so this compensation is load-bearing, not
        cosmetic. The place waypoint is deliberately untouched, mirroring
        ``_apply_clearance_bias``: the object is released from the place pose,
        so raising it would change placement, not protection.
        """
        table_z = float(self.scenario.environment.table_height_m)
        pad_half = float(self.scenario.robot.pad_half_height_m)
        min_jaw_z = (
            table_z
            + pad_half
            + self._GRASP_AIM_CLEARANCE_TOLERANCE_M
            + self.settle_shortfall_m
        )
        grasp_z = float(waypoints[1][2])
        if grasp_z < min_jaw_z:
            raise_by = min_jaw_z - grasp_z
            waypoints[1][2] = min_jaw_z
            # The approach waypoint is relative to the grasp: lift it by the
            # same amount so the jaws stay clear while descending and closing.
            waypoints[0][2] = waypoints[0][2] + raise_by
            # The carry waypoints are relative to the object's unreraised
            # height: raise them by the same amount so the object, now riding
            # lower in the grip, still travels at its intended clearance.
            for position in waypoints[2:4]:
                position[2] = position[2] + raise_by

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
        # A solve is only usable if it actually achieved the commanded pose.
        #
        # `converged` is not that test. The damped solver reports convergence
        # once it stops improving, which for a pose it cannot quite reach from
        # the given seed means it returns the best configuration it found and
        # calls it done. The far transit waypoint behaved exactly that way:
        # 4.97 mm of position error, 0.85 rad of orientation error, the wrist
        # driven onto the joint-7 stop, and `converged` set anyway.
        #
        # That matters far more than the 5 mm. With the orientation unreachable
        # the arm keeps rotating towards it as it travels, measured at 43
        # degrees over the transit, and because the grasp is rigid the held
        # object rotates with it. The episode then failed on orientation having
        # satisfied every other criterion -- a failed placement that looks like
        # a control fault and is really a planning one.
        #
        # The pose is not the problem: scanning 200 seeds for that same
        # waypoint finds 0.68 mm / 0.005 rad, so it is comfortably reachable and
        # only the seed is wrong. Warm-starting from the arm therefore has to be
        # backed by seeds that are not derived from it.
        # Recorded before any candidate is ranked: `_pose_key` needs the arm's
        # current configuration to break ties on travel distance.
        state["current_joints"] = np.asarray(
            observation.robot.positions_array, dtype=np.float64
        ).copy()
        # Preference order: the best pose among the candidates the arm can
        # plausibly drive to, and only if there are none, the best pose at all.
        #
        # The fallback exists because the warm start can sit in a basin the
        # damped step cannot leave -- the far transit waypoint returned 4.97 mm
        # and 0.85 rad with `converged` set, and the arm then spent the episode
        # rotating 43 degrees toward an orientation it was blocked from
        # reaching. The pose was never the problem: 200 seeds for that waypoint
        # find 0.68 mm / 0.005 rad, so only the seed was wrong.
        #
        # The travel bound is a *preference*, not a veto. As a veto it froze the
        # arm: every solution able to reach a distant waypoint needs about 7
        # rad, so seed 5 held a constant 55.7 mm residual for all 300 ticks and
        # never moved. A veto that prevents a bad plan by preventing every plan
        # is not a safety property, it is a guaranteed failure wearing one.
        #
        # The two candidate sets are ranked *independently*, which is the part
        # that is easy to get wrong. Comparing an in-bounds candidate against
        # the best candidate seen so far -- including out-of-bounds ones -- lets
        # a distant solution become the yardstick, after which no merely
        # in-bounds solution can ever look good enough to be chosen. Measured,
        # that put the object at 148 mm on a carry waypoint planned for 58 mm.
        candidates = [result]
        if not self._pose_achieved(result):
            for candidate in self._fallback_seeds(observation):
                seeded = solve_ik_refined(
                    self.jacobian_provider,
                    command_pose,
                    candidate,
                    lower=self.joint_lower,
                    upper=self.joint_upper,
                    options=self.ik_options,
                    restarts=self.ik_restarts,
                )
                candidates.append(seeded)
                if self._pose_achieved(seeded):
                    break
        in_bounds = [c for c in candidates if self._fallback_reachable(c.joint_positions, observation)]
        result = min(in_bounds or candidates, key=self._pose_key)
        state["planned_joints"] = result.joint_positions
        state["planned_waypoint"] = index
        state["ik_converged"] = result.converged
        state["last_ik_error_m"] = result.position_error
        # Reported alongside the position residual because a large orientation
        # error is what makes a rigid grasp rotate the held object, and it is
        # invisible in the position error alone.
        state["last_ik_orientation_error_rad"] = result.orientation_error
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

    def _settled_estimate(self, position: np.ndarray) -> np.ndarray | None:
        """Accumulate observations and return the mean once enough have arrived.

        Smoothing cannot be applied at plan time, because at plan time there is
        exactly one observation and averaging a single sample with itself is a
        no-op that looks like a working filter. A variance reduction has to
        *defer* the decision in order to collect the samples it averages, and
        that deferral is the entire cost of the technique: the policy spends
        ticks holding position instead of moving, and its plan is built from an
        estimate that is ``settle_steps`` ticks stale.

        Returns:
            The mean of the accumulated positions once the settle window is
            full, or ``None`` while still collecting -- callers treat ``None``
            as "keep waiting".
        """
        current = np.asarray(position, dtype=np.float64)
        if self._smoothed is None:
            self._smoothed = current.copy()
        else:
            self._smoothed = self._smoothed + current
        self._settle += 1
        if self._settle < max(1, self.settle_steps):
            return None
        return self._smoothed / self._settle

    def _reset_estimate(self) -> None:
        """Forget accumulated samples, at the start of an episode."""
        self._smoothed = None
        self._settle = 0

    def _pose_achieved(self, result: IKResult) -> bool:
        """Whether a solve actually reached the commanded pose.

        Both residuals must be inside the solver's own tolerances. The
        ``converged`` flag is deliberately not used on its own: it reports that
        the iteration stopped improving, which for an unreachable pose means the
        solver returns its best attempt and calls that success.
        """
        return (
            result.converged
            and result.position_error <= self.ik_options.position_tolerance
            and result.orientation_error <= self.ik_options.orientation_tolerance
        )

    def _pose_key(self, result: IKResult) -> tuple[float, float]:
        """Rank candidate solves by the pose they actually achieve.

        Pose first and only: a candidate is compared on how well it reaches the
        commanded pose, and the *reachability* of a candidate is enforced
        separately by :meth:`_fallback_reachable`, which refuses configurations
        the arm would have to sweep through the scene to get to. Ranking on
        travel as a tie-break instead was tried and does not work, because a
        genuinely better pose from across the room always wins the comparison
        and the arm duly drives there.
        """
        return (
            round(result.orientation_error, 9),
            round(result.position_error, 9),
        )

    def _fallback_reachable(self, goal: np.ndarray, observation: Observation) -> bool:
        """Whether the arm may plan to ``goal`` from where it currently is.

        A seed drawn from across the workspace will return an excellent pose
        from a completely different configuration, and the arm then drives
        there through the scene. Measured: a 58 mm waypoint planned from a
        far-away seed sent the gripper to 374 mm on the way and knocked the
        object off the table. A correct plan is still the wrong plan if getting
        there destroys the episode.

        The test is the *total* joint travel, not the largest single joint. A
        coordinated transit of this scene genuinely needs about 2.8 rad of
        rotation on one joint, so any per-joint bound tight enough to exclude a
        wild sweep also rejects the real move -- measured, that stalled the arm
        at 50 mm and never carried the object. Summing instead measures what the
        bound is actually for: a coordinated move distributes travel across the
        arm, while a different basin requires several joints to reverse
        together, and it is the reversal that sweeps the scene.

        The bound applies only while something is being carried. That is the
        condition under which it does any good: the damage it prevents is an
        arm sweeping a held object out of the workspace. Enforcing it on an
        empty gripper buys no safety and has a measured cost -- with the bound
        unconditional, seed 5 sat at its first waypoint with a constant 55.7 mm
        IK residual for all 300 ticks, because every solution that could have
        reached the pose required about 7 rad and the arm refused all of them.
        Standing still is not a safe plan; it is a guaranteed failure, and a
        guaranteed failure is worse than any excursion.
        """
        current = observation.robot.positions_array
        travel = float(np.sum(np.abs(np.asarray(goal, dtype=np.float64) - current)))
        return travel <= _IK_MAX_PLANNED_TRAVEL_RAD

    def _fallback_seeds(self, observation: Observation) -> list[np.ndarray]:
        """Return alternative IK seeds, ordered from most to least plausible.

        Home comes first because a deployed arm rests there, and it is the
        single best guess when the warm start has failed. The rest spread the arm
        across its workspace on a fixed lattice so that repeated runs of the
        same episode plan the same way; the seeds are deterministic rather than
        random precisely so an experiment is reproducible.

        Args:
            observation: Supplies the arm's current configuration, so a seed
                that would merely repeat the attempt already made is skipped.

        Returns:
            Candidate seeds.
        """
        seeds: list[np.ndarray] = [np.asarray(self.home_seed, dtype=np.float64)]
        current = observation.robot.positions_array
        size = int(self.joint_lower.size)
        span = self.joint_upper - self.joint_lower
        for index in range(_IK_FALLBACK_SEED_COUNT):
            # A deterministic low-discrepancy walk. The golden-ratio stride
            # keeps successive candidates far apart, which is what matters: the
            # failure mode is a whole neighbourhood of the joint box being in
            # the wrong basin, so nearby seeds would all fail together.
            fraction = np.mod(
                (index + 1) * _IK_SEED_STRIDE + np.arange(size) * 0.6180339887498949,
                1.0,
            )
            candidate = self.joint_lower + fraction * span
            if np.allclose(candidate, current, atol=1e-3):
                continue
            seeds.append(candidate)
        return seeds

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

    # -- grasp contact gate ------------------------------------------------
    def _aperture_settled(self, observation: Observation, state: dict[str, Any]) -> bool:
        """Whether the measured jaw aperture has stopped changing.

        The physical requirement for lifting is that the fingers have
        *finished travelling* -- either they closed onto the object and the
        object holds them still, or they are blocked and will go no further.
        Convergence is observable on any backend without assuming anything
        about its width semantics, which is the whole point: the previous gate
        compared against an absolute on-cube mark (``2*size -
        closed_separation``) derived from live-Gazebo readings, and MuJoCo
        falsified that mark -- a successful 16 mm grasp settles at ``width_m``
        = 11.6 mm there, so the gate never opened and every nominal episode
        stalled at the grasp (commit d7026af, 0/12 seeds, caught by
        ``tests/integration/test_default_scenario_solvable.py``).

        The rule: keep a rolling trace of the measured aperture and declare it
        settled when the last ``_APERTURE_STABLE_TICKS`` consecutive changes
        each fall within ``_APERTURE_STABLE_EPS_M``. Measured separation of
        the two regimes: MuJoCo's settled drift is ~1e-7 m/tick; the live
        Gazebo close creeps ~1.2 mm/tick (4 mm/s at 0.3 s/tick) while still
        travelling. The 0.2 mm/tick threshold sits between them by two orders
        of magnitude on the settled side.

        A closed-out early (blocked wide) jaw also passes this gate -- that is
        correct: a finger that cannot move will not move during the lift
        either. Whether the *object* is actually held is the grasp metric's
        job at episode end, not the lift gate's.
        """
        try:
            measured = float(observation.robot.gripper.width_m)
        except (AttributeError, TypeError, ValueError):
            return False
        trace: list[float] = state.setdefault("aperture_trace", [])
        trace.append(measured)
        # Keep only the most recent window: N stable changes need N+1 samples.
        excess = len(trace) - (_APERTURE_STABLE_TICKS + 1)
        if excess > 0:
            del trace[:excess]
        if len(trace) < _APERTURE_STABLE_TICKS + 1:
            return False
        changes = [abs(b - a) for a, b in pairwise(trace)]
        return all(delta <= _APERTURE_STABLE_EPS_M for delta in changes)

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


def policy_v3(
    config: PolicyConfig | None = None, scenario: ScenarioConfig | None = None
) -> PickAndPlacePolicy:
    """Positive control: averages pose samples before committing to a plan.

    See :data:`_DEFAULT_V3` for why this policy exists. Summarised: it should
    help where depth noise dominates and hurt where sensor latency does, which
    is a falsifiable prediction the experiment suite tests rather than assumes.
    """
    return PickAndPlacePolicy(
        config or _DEFAULT_V3, scenario or ScenarioConfig()
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
    # change. The parameter stays implemented (``_apply_clearance_bias``) and
    # declared here so an ablation can exercise it as its own factor.
    #
    # ``max_position_sigma_m`` is derived from the grasp, not chosen. A parallel
    # gripper has to centre the object between the pads; the error budget for
    # that is half the object width, since past that the pads close on one face
    # and lever the object out. For the nominal 16 mm cube that budget is 8 mm,
    # and requiring it to hold at 2 sigma puts the trip point at sigma = 4 mm.
    #
    # The previous value of 0.05 was 12x larger than the budget it is supposed to
    # protect, which made the gate inert rather than conservative: measured over
    # a depth-noise grid, v1 and v2 were identical at every level, and the gate
    # only engaged at 50 mm of depth noise -- where the arm had already failed
    # to place the object at 10 mm for an unrelated reason. A safety interlock
    # that cannot engage is not a conservative policy, it is no policy at all,
    # and a two-arm comparison of two identical policies measures nothing.
    #
    # Note this makes the candidate *worse* under noise, which is the expected
    # and interesting direction: refusing to act on an untrusted estimate is
    # exactly what costs success rate on the episodes where the estimate would
    # have been good enough. That trade is the thing worth measuring.
    parameters={"max_position_sigma_m": 0.004, "clearance_bias_m": 0.0},
)

#: Positive control: a candidate with a *documented, deliberate* trade-off.
#:
#: This policy exists so the detection machinery can be validated against a
#: regression that is known to exist. Demonstrating that a regression detector
#: works requires a positive control; without one, "no regression found" and
#: "the detector is blind" are indistinguishable, and the null results recorded
#: in ``experiments/results/README.md`` would be uninterpretable.
#:
#: It is not a pretend candidate. The change is a real technique with a real
#: cost: the policy averages several pose samples before committing to a plan,
#: which suppresses depth noise but spends control ticks holding position and
#: builds its plan from an estimate that is ``settle_steps`` ticks stale. The
#: prediction is therefore falsifiable and specific -- it should help where
#: depth noise dominates and hurt where latency does, and RQ2 tests exactly
#: that. If it does not, the hypothesis is wrong and is reported as wrong.
_DEFAULT_V3 = PolicyConfig(
    name="policy_v3",
    version="3.0.0",
    components={
        PolicyComponent.PERCEPTION: "fusion_v2_settled",
        PolicyComponent.PLANNER: "waypoint_v1",
        PolicyComponent.CONTROLLER: "ik_dls_v1",
    },
    parameters={"settle_steps": 6, "clearance_bias_m": 0.0},
)

#: Registry of policy factories, keyed by :attr:`PolicyConfig.implementation`.
_REGISTRY: dict[str, Any] = {
    "rule_based": policy_v1,
    "policy_v1": policy_v1,
    "policy_v2": policy_v2,
    "policy_v3": policy_v3,
    "mock": lambda config, scenario: MockPolicy(config, scenario),
}

#: Default configuration per *named* policy, so a config that only carries a
#: name can be completed into the policy that name denotes.
_NAMED_DEFAULTS: dict[str, PolicyConfig] = {
    "policy_v1": _DEFAULT_V1,
    "policy_v2": _DEFAULT_V2,
    "policy_v3": _DEFAULT_V3,
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
