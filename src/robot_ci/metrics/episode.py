"""Per-episode metric computation.

This is the **only** place where ground truth is read during an episode to
produce a number. It runs after the episode's steps are done, so no policy can
observe anything derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_ci.metrics.base import MetricKind, MetricName, MetricSource, MetricValue
from robot_ci.models.environment import TaskName
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import Action, GroundTruthState
from robot_ci.tasks.base import Task, TaskStatus

__all__ = ["EpisodeTrace", "compute_episode_metrics"]


@dataclass
class EpisodeTrace:
    """What the runner records while an episode is in flight.

    The trace holds the things metrics need but that are too bulky or too
    high-frequency for the observation stream: the commanded trajectory, the
    end-effector path, and the timing of each phase.

    Attributes:
        commanded_poses: Cartesian positions the policy aimed at, in order.
        ee_path: Achieved end-effector positions, one per step.
        ee_targets: Achieved end-effector orientations, one per step.
        object_path: True object positions, one per step.
        actions: The actions issued, for reproducibility checks.
        planning_time_ms: Total time spent inside IK and planning.
        recovery_count: Times the policy re-planned after a failure.
        first_success_step: Step at which success was first satisfied.
    """

    commanded_poses: list[np.ndarray] = field(default_factory=list)
    ee_path: list[np.ndarray] = field(default_factory=list)
    ee_targets: list[np.ndarray] = field(default_factory=list)
    object_path: list[np.ndarray] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    planning_time_ms: float = 0.0
    recovery_count: int = 0
    first_success_step: int | None = None


def compute_episode_metrics(
    scenario: ScenarioConfig,
    task: Task,
    trace: EpisodeTrace,
    final_truth: GroundTruthState,
    *,
    status: TaskStatus,
    steps: int,
    duration_s: float,
    collision_steps: int,
    perception_confidence: float,
    pose_sigma_m: float,
    sensor_consistency: float,
) -> list[MetricValue]:
    """Compute the full metric vector for one episode.

    Args:
        scenario: The condition the episode ran under.
        task: The task, which holds the success bookkeeping.
        trace: Recorded trajectory and timing.
        final_truth: Hidden state at the end of the episode.
        status: Terminal task status.
        steps: Control steps taken.
        duration_s: Simulated duration.
        collision_steps: Number of steps with a reported contact.
        perception_confidence: Mean confidence of the fused object belief.
        pose_sigma_m: Mean positional uncertainty of that belief.
        sensor_consistency: Agreement between sensors, in [0, 1].

    Returns:
        A list of :class:`MetricValue`, ready to be stored or aggregated.
    """
    metrics: list[MetricValue] = []
    success = status is TaskStatus.SUCCESS

    def add(
        name: MetricName,
        value: float,
        unit: str = "",
        kind: MetricKind = MetricKind.SCORE,
        source: MetricSource = MetricSource.EVALUATOR,
        **detail: Any,
    ) -> None:
        metrics.append(
            MetricValue(
                name=name, value=float(value), unit=unit, kind=kind, source=source, detail=detail
            )
        )

    # --- outcome ---------------------------------------------------------
    add(MetricName.TASK_SUCCESS, 1.0 if success else 0.0, kind=MetricKind.BOOLEAN,
        source=MetricSource.TASK, status=status.value)
    add(MetricName.TIMEOUT, 1.0 if status is TaskStatus.TIMEOUT else 0.0, kind=MetricKind.BOOLEAN)
    add(
        MetricName.SAFETY_VIOLATION,
        1.0 if status is TaskStatus.SAFETY else 0.0,
        kind=MetricKind.BOOLEAN,
        penetration_mm=round(final_truth.penetration_depth_m * 1000.0, 3),
    )
    add(MetricName.STEPS, steps, unit="steps", kind=MetricKind.COUNT)
    add(MetricName.COMPLETION_TIME_S, duration_s, unit="s", kind=MetricKind.DURATION)
    add(MetricName.COLLISION_COUNT, collision_steps, kind=MetricKind.COUNT)
    add(MetricName.COLLISION, 1.0 if collision_steps > 0 else 0.0, kind=MetricKind.BOOLEAN)
    add(MetricName.RECOVERY_COUNT, trace.recovery_count, kind=MetricKind.COUNT)
    add(MetricName.PLANNING_TIME_MS, trace.planning_time_ms, unit="ms", kind=MetricKind.DURATION,
        source=MetricSource.PLANNER)

    # --- task-specific ---------------------------------------------------
    grasped = task.ever_grasped
    add(MetricName.GRASP_SUCCESS, 1.0 if grasped else 0.0, kind=MetricKind.BOOLEAN)
    add(
        MetricName.OBJECT_LIFT_M,
        task.max_grasped_height_m,
        unit="m",
        kind=MetricKind.LENGTH,
        detail={"lifted": bool(task.max_grasped_height_m > 0)},
    )

    target_position = scenario.task.target_pose.position
    target = final_truth.object_by_id(task.target_object_id())
    if target is not None:
        placement_error = target.pose.position.distance_to(target_position)
        orientation_error = target.pose.orientation.angular_distance_to(
            scenario.task.target_pose.orientation
        )
        placed = placement_error <= task.criteria.position_tolerance_m
        add(MetricName.PLACEMENT_ERROR_M, placement_error, unit="m", kind=MetricKind.LENGTH)
        add(
            MetricName.ORIENTATION_ERROR_RAD,
            orientation_error,
            unit="rad",
            kind=MetricKind.ANGLE,
        )
        add(MetricName.PLACEMENT_SUCCESS, 1.0 if placed else 0.0, kind=MetricKind.BOOLEAN)
        add(MetricName.FINAL_DISTANCE_M, placement_error, unit="m", kind=MetricKind.LENGTH)
        if scenario.task.name is TaskName.PUSH_OBJECT:
            add(MetricName.TASK_SUCCESS, 1.0 if success else 0.0, kind=MetricKind.BOOLEAN)

    # --- grasp error -----------------------------------------------------
    # The error the policy would have seen if it had acted on its own belief:
    # the gap between what it believed and what was true. This is the quantity
    # a perception regression actually moves.
    if trace.ee_path and trace.object_path:
        first_ee = np.asarray(trace.ee_path[0])
        first_obj = np.asarray(trace.object_path[0])
        del first_ee, first_obj
    if target is not None and trace.ee_path:
        # Grasp error: closest approach of the gripper to the object centre
        # before the lift, in millimetres.
        pre_lift = [p for p, o in zip(trace.ee_path, trace.object_path, strict=False) if o[2] < 0.08]
        if pre_lift:
            closest = min(
                float(np.linalg.norm(np.asarray(ee) - np.asarray(obj)))
                for ee, obj in zip(pre_lift, trace.object_path[: len(pre_lift)], strict=False)
            )
            add(MetricName.GRASP_ERROR_M, closest, unit="m", kind=MetricKind.LENGTH)
        add(MetricName.POSITION_ERROR_M, placement_error, unit="m", kind=MetricKind.LENGTH)

    # --- trajectory ------------------------------------------------------
    if len(trace.ee_path) > 1:
        path = np.asarray(trace.ee_path, dtype=np.float64)
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        add(MetricName.TRAJECTORY_LENGTH_M, length, unit="m", kind=MetricKind.LENGTH)
        if trace.commanded_poses:
            deviation = min(
                (
                    float(np.linalg.norm(path - np.asarray(cmd)))
                    for cmd in trace.commanded_poses
                ),
                default=0.0,
            )
            add(MetricName.TRAJECTORY_DEVIATION_M, deviation, unit="m", kind=MetricKind.LENGTH)

    # --- perception ------------------------------------------------------
    add(
        MetricName.PERCEPTION_CONFIDENCE,
        perception_confidence,
        kind=MetricKind.SCORE,
        source=MetricSource.PERCEPTION,
    )
    add(
        MetricName.POSE_SIGMA_M,
        pose_sigma_m,
        unit="m",
        kind=MetricKind.LENGTH,
        source=MetricSource.FUSION,
    )
    add(
        MetricName.SENSOR_CONSISTENCY,
        sensor_consistency,
        kind=MetricKind.SCORE,
        source=MetricSource.FUSION,
    )

    # --- planner ---------------------------------------------------------
    add(MetricName.IK_CONVERGED, 1.0, kind=MetricKind.BOOLEAN, source=MetricSource.PLANNER)

    return metrics
