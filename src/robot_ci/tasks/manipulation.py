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
        clearance = max(self.criteria.min_lift_height_m, 0.12)

        # Grasp height, derived rather than calibrated.
        #
        # The jaw centre height `z` is bounded by two measured facts:
        #
        #   z >= table_top + pad_half_height   the pad's lower edge must clear
        #                                        the support surface
        #   z <= object_top                     the pad must still overlap the
        #                                        object's side faces
        #
        # Aiming at the object centre satisfies the second and violates the
        # first: with a 9.2 mm pad half-height and an object centre 8 mm above
        # the table, the pad's lower edge lands 0.16 mm *below* the surface.
        # The pads then rest on the tabletop, the jaws stall open, and the
        # gripper never touches the object -- a silent failure in which every
        # waypoint is still reached on time.
        #
        # The target is the midpoint of the feasible band, so the grasp sits as
        # far as possible from both failure modes. Both bounds come from the
        # scenario, so this stays simulator- and robot-agnostic: the pad height
        # is an embodiment property on RobotConfig, the object size is declared
        # on the scenario, and nothing here knows about MuJoCo.
        pad_half = float(self.scenario.robot.pad_half_height_m)
        object_half = float(self.scenario.target.size_m)
        window_lo = table_z + pad_half
        window_hi = obj.z + object_half
        grasp_z = 0.5 * (window_lo + window_hi) if window_hi > window_lo else window_lo
        self.last_grasp_window = (window_lo, window_hi, grasp_z)
        return [
            # 1. Above the object, gripper open.
            Vec3.of(obj.x, obj.y, grasp_z + approach),
            # 2. At the derived grasp height over the object.
            Vec3.of(obj.x, obj.y, grasp_z),
            # 3. Lifted clear of the support surface.
            Vec3.of(obj.x, obj.y, obj.z + clearance),
            # 4. Above the target zone, ready to lower.
            Vec3.of(target.x, target.y, table_z + self.scenario.target.size_m + clearance),
            # 5. At the target zone.
            Vec3.of(target.x, target.y, table_z + self.scenario.target.size_m),
        ]

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
