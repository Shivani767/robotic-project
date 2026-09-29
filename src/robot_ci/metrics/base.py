"""Metric schemas.

Every measurement this framework records carries four things: a value, a unit,
where it came from, and (where meaningful) an uncertainty. A bare float is not
a metric, because a number without a unit or a provenance cannot be compared
across backends or reproduced across runs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from robot_ci.models.base import VersionedModel

__all__ = ["MetricKind", "MetricName", "MetricSource", "MetricValue"]


class MetricName(str, Enum):
    """Canonical metric identifiers.

    These strings are the contract between the evaluation layer, the database
    and the dashboard. Renaming one is a schema change.
    """

    TASK_SUCCESS = "task_success"
    GRASP_SUCCESS = "grasp_success"
    PLACEMENT_SUCCESS = "placement_success"
    COLLISION = "collision"
    COLLISION_COUNT = "collision_count"
    TIMEOUT = "timeout"
    SAFETY_VIOLATION = "safety_violation"
    GRASP_ERROR_M = "grasp_error_m"
    PLACEMENT_ERROR_M = "placement_error_m"
    POSITION_ERROR_M = "position_error_m"
    ORIENTATION_ERROR_RAD = "orientation_error_rad"
    TRAJECTORY_LENGTH_M = "trajectory_length_m"
    TRAJECTORY_DEVIATION_M = "trajectory_deviation_m"
    COMPLETION_TIME_S = "completion_time_s"
    PLANNING_TIME_MS = "planning_time_ms"
    RECOVERY_COUNT = "recovery_count"
    STEPS = "steps"
    PERCEPTION_CONFIDENCE = "perception_confidence"
    POSE_SIGMA_M = "pose_sigma_m"
    SENSOR_CONSISTENCY = "sensor_consistency"
    IK_CONVERGED = "ik_converged"
    IK_ERROR_M = "ik_error_m"
    FINAL_DISTANCE_M = "final_distance_m"
    OBJECT_LIFT_M = "object_lift_m"


class MetricKind(str, Enum):
    """How a metric should be summarised across repeated runs."""

    BOOLEAN = "boolean"
    """A 0/1 outcome; summarised as a rate with an interval."""
    COUNT = "count"
    """A non-negative count; summarised as mean and median."""
    DURATION = "duration"
    """Seconds; summarised as mean with a bootstrap interval."""
    LENGTH = "length"
    """Metres."""
    ANGLE = "angle"
    """Radians."""
    SCORE = "score"
    """A bounded score in [0, 1]."""


class MetricSource(str, Enum):
    """Which part of the system produced a measurement.

    Recording this is what lets a later report distinguish "the policy was
    worse" from "the estimator was noisier" without re-running anything.
    """

    SIMULATOR = "simulator"
    TASK = "task"
    POLICY = "policy"
    PERCEPTION = "perception"
    FUSION = "fusion"
    PLANNER = "planner"
    EVALUATOR = "evaluator"


class MetricValue(VersionedModel):
    """One measured quantity.

    Attributes:
        name: Canonical metric identifier.
        value: The measured value, in :attr:`unit`.
        unit: Physical unit, e.g. ``"m"``, ``"s"``, ``"rad"``, ``""``.
        kind: How the metric should be aggregated.
        source: Which subsystem produced it.
        step: Control step the metric refers to, when applicable.
        uncertainty: Standard error, when the value is an estimate.
        n: Sample count behind an aggregated value.
        detail: Free-form structured notes.
    """

    name: MetricName
    value: float
    unit: str = Field(default="", description="Physical unit of ``value``.")
    kind: MetricKind = Field(default=MetricKind.SCORE, description="Aggregation semantics.")
    source: MetricSource = Field(default=MetricSource.EVALUATOR, description="Producing subsystem.")
    step: int | None = Field(default=None, ge=0, description="Control step, when applicable.")
    uncertainty: float | None = Field(default=None, ge=0.0, description="Standard error, when estimated.")
    n: int | None = Field(default=None, ge=0, description="Sample count behind the value.")
    detail: dict[str, Any] = Field(default_factory=dict, description="Extra structured context.")

    def to_csv_row(self) -> dict[str, Any]:
        """Flatten for ``metrics.csv``."""
        return {
            "metric": self.name.value,
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind.value,
            "source": self.source.value,
            "step": self.step,
            "uncertainty": self.uncertainty,
            "n": self.n,
        }
