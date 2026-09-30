"""Concrete manipulation tasks: pick-and-place, grasp and push."""

from __future__ import annotations

import numpy as np

from robot_ci.models.environment import TaskName
from robot_ci.models.geometry import Pose, Quaternion, Vec3
from robot_ci.models.state import GroundTruthState, Observation, RobotState
from robot_ci.tasks.base import Phase, StepOutcome, Task, TaskStatus

__all__ = ["GraspObjectTask", "PickAndPlaceTask", "PushObjectTask"]

#: Grasp orientation for the vendored Panda: a **top-down** grasp.
#:
#: The hand's local +z is its approach axis and +y is the finger separation
#: axis, so a top-down grasp needs +z -> world -z, which is a **half turn about
#: x**. The quarter turn that is more natural to write is wrong: it points the
#: approach axis along world +y, giving a *side* grasp whose jaws separate
#: **vertically**.
#:
#: That failure is geometric, not a matter of IK accuracy. With a vertical
#: separation axis the lower pad sits ``open_separation/2`` below the jaw
#: centre; the measured open separation is 91 mm, so centring the jaws on an
#: object resting on the table puts the lower pad **35-37 mm inside the
#: tabletop**. Measured on the compiled model over the six cardinal
#: orientations (scripts/pad_target_ik_experiment.py):
#:
#:   orientation   table clearance   feasible
#:   -----------   ---------------   ---------
#:   x+90 (side)      -35.5 mm        no
#:   x-90 (side)      -37.1 mm        no
#:   x180 (top-down)   +8.0 mm        yes
#:   y180              +9.9 mm        yes
#:   z180             +51.2 mm        no (IK residual 49 mm)
#:   identity         +47.4 mm        no (IK residual 48 mm)
#:
#: The chosen orientation reaches the grasp at 1.7 mm with the object 1.1 mm
#: from the pad midpoint along the separation axis. Pinned by
#: tests/integration/test_pad_targeted_ik.py.
TOP_DOWN = Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)

#: How far the pad's lower edge may dip below the support surface before the
#: grasp is called infeasible. Contact resolution tolerates a small overlap
#: without the pads jamming, so this is a reporting threshold, not a hard stop.
_GRASP_CLEARANCE_TOLERANCE_M = 0.001

#: Lowest carry height used when the scenario declares no obstacles.
#:
#: Enough to clear the object itself and satisfy the task's lift requirement,
#: and no more. A carry height is a clearance requirement, not a preference: the
#: faster the object travels, the more likely the grasp is to shed it.
_MIN_CARRY_CLEARANCE_M = 0.03


class PickAndPlaceTask(Task):
    """Grasp the target object, carry it to the target zone, and release it.

    Success requires all of:

    * the object was grasped at some point,
    * it was lifted clear of the support surface,
    * it ended within :attr:`SuccessCriteria.position_tolerance_m` of the target,
    * its orientation is within tolerance of the target orientation,
    * the gripper released it, and
    * the collision budget was not exceeded.
    """

    name = TaskName.PICK_AND_PLACE

    def waypoints(self, object_pose: Pose) -> list[Vec3]:
        """Return the pick-and-place waypoint sequence.

        The sequence is deliberately explicit rather than a continuous spline:
        each waypoint is a checkable milestone, so when an episode fails it is
        obvious *which* stage went wrong.

        Args:
            object_pose: The object's pose as believed by the policy.

        Returns:
            Waypoints in order: pre-grasp, grasp, lift, transit, place,
            release, retreat.
        """
        table_z = self.scenario.environment.table_height_m
        obj = object_pose.position
        target = self.target_position()
        approach = float(self.scenario.task.approach_height_m)
        clearance = self._carry_clearance()

        # Grasp height, derived rather than calibrated.
        #
        # The quantity the task actually commands is the *tool centre point*,
        # and the pads grip around it. So the grasp is right when the pad
        # straddles the object's centre, and the aim height follows from where
        # the pads sit relative to that point.
        #
        # The previous derivation aimed the jaw at the midpoint of the band in
        # which the pad's *lower edge* clears the table:
        #
        #   lo = table + pad_half,  hi = object_top,  aim = (lo + hi) / 2
        #
        # Both bounds are real, but the midpoint of that band is not the
        # object's centre, and the two differ by several millimetres here.
        # Measured on the Panda: the aim landed 3.8 mm above the object centre,
        # so the pads closed on the upper 12 mm of a 16 mm face with their
        # centre 4.3 mm above the object's. The grip held -- 1.7 N of contact,
        # both pads engaged -- and was then levered open by the lift, because
        # the object pivots on the pad edge rather than being carried by the
        # pad face. Measured: contacts go from 1.7 N to exactly zero in one
        # step, the fingers retreat to their 12.71 mm rest, and the object
        # falls 100 mm to the table. Nothing else in the episode changes.
        #
        # The correct aim puts the *pad centre* on the object centre, offset by
        # the distance from the tool centre point back to the pads:
        #
        #   aim = object_centre + pad_offset_from_tcp
        #
        # That offset is an embodiment property, measured from the compiled
        # model, so it lives on RobotConfig beside the pad height rather than
        # being written down here. The table clearance is then a *check*, not
        # the aim: if the pads cannot straddle the object without the lower
        # edge touching the support surface, the grasp is infeasible for this
        # object and is reported as such instead of silently producing a grip
        # that the lift will tear open.
        pad_half = float(self.scenario.robot.pad_half_height_m)
        pad_offset = float(self.scenario.robot.pad_offset_from_tcp_m)
        object_half = float(self.scenario.target.size_m)
        grasp_z = obj.z + pad_offset
        # The pad's lower edge at that aim, used only to report feasibility.
        pad_bottom = grasp_z - pad_half
        self.last_grasp_window = (table_z + pad_half, obj.z + object_half, grasp_z)
        self.grasp_feasible = pad_bottom >= table_z - _GRASP_CLEARANCE_TOLERANCE_M
        return [
            # 1. Above the object, gripper open.
            Vec3.of(obj.x, obj.y, grasp_z + approach),
            # 2. At the derived grasp height over the object.
            Vec3.of(obj.x, obj.y, grasp_z),
            # 3. Lifted clear of the support surface.
            Vec3.of(obj.x, obj.y, obj.z + clearance),
            # 4. Above the target zone, ready to lower.
            Vec3.of(target.x, target.y, target.z + clearance),
            # 5. At the target zone.
            Vec3.of(target.x, target.y, target.z),
        ]

    def _carry_clearance(self) -> float:
        """How high above the support surface the object must be carried.

        The carry height is the tallest thing the gripper has to clear between
        the pick and the place, not a constant. It is derived from the tallest
        obstacle the scenario declares, with the task's own lift requirement as
        a floor, because the object has to reach that height to count as lifted
        at all.

        This was a hard-coded 120 mm. That clears everything in the default
        scene and is far higher than the 16 mm cube needs, and the cost was
        measured: the transit ran at 0.24 m/s and the object was torn out of
        the jaws around 60% of the way across. Carrying an object higher than
        the obstacles require buys nothing and costs grip, because it is the
        speed of the traverse that shakes the object loose.

        Returns:
            Clearance above the support surface, in metres.
        """
        environment = self.scenario.environment
        table_z = float(environment.table_height_m)
        floor = max(self.criteria.min_lift_height_m, _MIN_CARRY_CLEARANCE_M)
        if environment.obstacle_count <= 0:
            return floor
        obstacle_top = table_z + float(environment.obstacle_height_m)
        return max(floor, obstacle_top - table_z + float(self.scenario.target.size_m))

    def evaluate(
        self,
        observation: Observation,
        ground_truth: GroundTruthState,
        robot_state: RobotState,
        *,
        collision: bool,
    ) -> StepOutcome:
        """Assess progress and decide whether the episode ends."""
        if collision:
            self.collisions += 1
        self.track_grasped(ground_truth)

        position_error = self.position_error(ground_truth)
        satisfied = self.check_success(ground_truth, robot_state)

        # Phase inference from the hidden state: the task is an evaluator, so it
        # is allowed to look at what actually happened.
        ee = ground_truth.true_ee_pose.position
        target = ground_truth.object_by_id(self.target_object_id())
        if target is not None and target.grasped:
            phase = Phase.TRANSIT if position_error > 0.08 else Phase.PLACE
        elif self.ever_grasped and self.released:
            phase = Phase.RETREAT
        elif self.ever_grasped:
            phase = Phase.LIFT
        else:
            table_z = self.scenario.environment.table_height_m
            gap = ee.z - (table_z + self.scenario.target.size_m)
            phase = Phase.DESCEND if gap < 0.05 else Phase.APPROACH
        self.progress.record(phase)

        if self.safety_violation(ground_truth):
            return StepOutcome(
                status=TaskStatus.SAFETY,
                phase=phase,
                progress=0.0,
                satisfied=satisfied,
                message=(
                    f"safety violation: {ground_truth.penetration_depth_m * 1000:.1f} mm penetration"
                ),
            )

        if all(satisfied.values()):
            return StepOutcome(
                status=TaskStatus.SUCCESS,
                phase=Phase.DONE,
                progress=1.0,
                satisfied=satisfied,
                message=f"placed within {position_error * 1000:.1f} mm of the target",
            )

        if self.collisions > self.criteria.max_collisions:
            return StepOutcome(
                status=TaskStatus.FAILURE,
                phase=phase,
                progress=0.0,
                satisfied=satisfied,
                message=f"collision budget exceeded ({self.collisions})",
            )

        if self.budget_exhausted():
            return self._finish(ground_truth, robot_state)

        # Progress heuristic for reporting only; it is never used for success.
        progress = float(np.clip(1.0 - position_error / 0.5, 0.0, 0.95))
        return StepOutcome(
            status=TaskStatus.RUNNING,
            phase=phase,
            progress=progress,
            satisfied=satisfied,
            message=f"position error {position_error * 1000:.0f} mm",
        )


class GraspObjectTask(PickAndPlaceTask):
    """Grasp the object and hold it; placement is not required."""

    name = TaskName.GRASP_OBJECT

    def waypoints(self, object_pose: Pose) -> list[Vec3]:
        """Return only the approach and grasp waypoints.

        Slicing the parent's list would be fragile: if the parent's sequence is
        ever reordered this task would silently grasp at the wrong height.
        """
        table_z = self.scenario.environment.table_height_m
        obj = object_pose.position
        grasp_z = table_z + self.scenario.target.size_m
        return [
            Vec3.of(obj.x, obj.y, grasp_z + float(self.scenario.task.approach_height_m)),
            Vec3.of(obj.x, obj.y, grasp_z),
        ]

    def evaluate(
        self,
        observation: Observation,
        ground_truth: GroundTruthState,
        robot_state: RobotState,
        *,
        collision: bool,
    ) -> StepOutcome:
        """Succeed once the object is grasped and held."""
        if collision:
            self.collisions += 1
        self.track_grasped(ground_truth)
        held_steps = self.progress.steps if self.ever_grasped else 0
        satisfied = {
            "grasped": self.ever_grasped,
            "held": held_steps >= self.criteria.hold_steps,
            "collision_budget": self.collisions <= self.criteria.max_collisions,
        }
        phase = Phase.GRASP if not self.ever_grasped else Phase.LIFT
        self.progress.record(phase)

        if all(satisfied.values()):
            return StepOutcome(
                status=TaskStatus.SUCCESS,
                phase=Phase.DONE,
                progress=1.0,
                satisfied=satisfied,
                message="object grasped and held",
            )
        if self.budget_exhausted():
            return StepOutcome(
                status=TaskStatus.FAILURE,
                phase=phase,
                satisfied=satisfied,
                message="budget exhausted before a stable grasp",
            )
        return StepOutcome(status=TaskStatus.RUNNING, phase=phase, satisfied=satisfied)


class PushObjectTask(Task):
    """Push the object to the target zone without grasping it.

    Included to demonstrate that the Task interface is not pick-and-place
    specific: the policy, planner and metrics are unchanged, only the success
    criterion and the waypoint sequence differ.
    """

    name = TaskName.PUSH_OBJECT

    def waypoints(self, object_pose: Pose) -> list[Vec3]:
        """Return a straight-line push from behind the object to the target."""
        table_z = self.scenario.environment.table_height_m
        obj = object_pose.position
        target = self.target_position()
        delta = np.array([target.x - obj.x, target.y - obj.y])
        norm = float(np.linalg.norm(delta))
        unit = delta / norm if norm > 1e-6 else np.array([1.0, 0.0])
        # Start behind the object and push through it to the target, so the
        # object is driven rather than merely nudged.
        height = table_z + self.scenario.target.size_m
        behind = np.array([obj.x, obj.y]) - unit * 0.12
        beyond = np.array([obj.x, obj.y]) + unit * 0.12
        return [
            Vec3.of(float(behind[0]), float(behind[1]), height),
            Vec3.of(float(beyond[0]), float(beyond[1]), height),
        ]

    def evaluate(
        self,
        observation: Observation,
        ground_truth: GroundTruthState,
        robot_state: RobotState,
        *,
        collision: bool,
    ) -> StepOutcome:
        """Succeed when the object reaches the target zone without grasping."""
        if collision:
            self.collisions += 1
        position_error = self.position_error(ground_truth)
        target = ground_truth.object_by_id(self.target_object_id())
        grasped = bool(target and target.grasped)
        satisfied = {
            "position": position_error <= self.criteria.position_tolerance_m,
            "not_grasped": not grasped,
            "collision_budget": self.collisions <= self.criteria.max_collisions,
        }
        phase = Phase.TRANSIT
        self.progress.record(phase)
        if all(satisfied.values()):
            return StepOutcome(
                status=TaskStatus.SUCCESS,
                phase=Phase.DONE,
                progress=1.0,
                satisfied=satisfied,
                message=f"pushed within {position_error * 1000:.1f} mm of the target",
            )
        if self.budget_exhausted():
            return StepOutcome(
                status=TaskStatus.FAILURE,
                phase=phase,
                satisfied=satisfied,
                message=f"budget exhausted, {position_error * 1000:.0f} mm from target",
            )
        return StepOutcome(
            status=TaskStatus.RUNNING,
            phase=phase,
            progress=float(np.clip(1.0 - position_error / 0.5, 0.0, 0.95)),
            satisfied=satisfied,
        )
