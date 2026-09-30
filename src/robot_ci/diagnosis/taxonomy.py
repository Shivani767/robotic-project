"""Structured failure categories and the evidence used to assign them.

A failure is reported as one or more *hypotheses* with weights, never as a
single confident label. The distinction is not pedantry: most failures in this
domain have more than one mechanically consistent explanation, and a classifier
that returns one label presents a guess as a finding. The weights here are
rule-based and inspectable rather than learned, so a reader can disagree with a
specific rule instead of with an opaque model -- and the rules cite the metric
that fired them, so every label can be traced to a number.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = ["FailureCategory", "FailureEvent", "classify_failure"]


class FailureCategory(StrEnum):
    """Why an episode did not succeed.

    The categories are ordered roughly from sensing to acting, which is also the
    order a causal investigation works in: a placement error is downstream of a
    pose error, which is downstream of a depth error. Ordering them this way
    lets a report present the earliest implicated subsystem first without having
    to rank them at read time.
    """

    PERCEPTION_FAILURE = "perception_failure"
    POSE_ESTIMATION_FAILURE = "pose_estimation_failure"
    DEPTH_FAILURE = "depth_failure"
    TRACKING_FAILURE = "tracking_failure"
    PLANNING_FAILURE = "planning_failure"
    CONTROL_FAILURE = "control_failure"
    GRASP_FAILURE = "grasp_failure"
    PLACEMENT_FAILURE = "placement_failure"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    SENSOR_LATENCY = "sensor_latency"
    SENSOR_CALIBRATION = "sensor_calibration"
    WORLD_MODEL_ERROR = "world_model_error"
    POLICY_REGRESSION = "policy_regression"
    UNKNOWN = "unknown"


#: Categories a classifier may emit, in investigation order. ``UNKNOWN`` is last
#: because it is a statement about the classifier, not about the robot.
_ORDER: tuple[FailureCategory, ...] = (
    FailureCategory.SENSOR_CALIBRATION,
    FailureCategory.SENSOR_LATENCY,
    FailureCategory.DEPTH_FAILURE,
    FailureCategory.PERCEPTION_FAILURE,
    FailureCategory.POSE_ESTIMATION_FAILURE,
    FailureCategory.TRACKING_FAILURE,
    FailureCategory.WORLD_MODEL_ERROR,
    FailureCategory.PLANNING_FAILURE,
    FailureCategory.GRASP_FAILURE,
    FailureCategory.PLACEMENT_FAILURE,
    FailureCategory.CONTROL_FAILURE,
    FailureCategory.COLLISION,
    FailureCategory.TIMEOUT,
    FailureCategory.POLICY_REGRESSION,
    FailureCategory.UNKNOWN,
)


class FailureEvent(BaseModel):
    """One hypothesis about why an episode failed.

    Attributes:
        category: The hypothesised cause.
        weight: Relative support in ``[0, 1]``. Weights across the events of one
            episode are not required to sum to 1: they are confidences that this
            mechanism was involved, and several mechanisms commonly are.
        evidence: The metric values that fired this rule, rendered as short
            phrases. Kept on the event so a label is never separated from the
            numbers that produced it.
        step: Control step at which the episode ended.
        message: The episode's own outcome message, verbatim.
    """

    schema_version: str = Field(default="1.0.0")
    category: FailureCategory
    weight: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    step: int = 0
    message: str = ""

    @property
    def rank(self) -> int:
        """Position in investigation order; lower is more upstream."""
        return _ORDER.index(self.category)

#: Metric name suffixes the classifier reads. Names are the enum *values*, which
#: is what the episode metric records carry.
_M_GRASP = "grasp_success"
_M_LIFT = "object_lift_m"
_M_PLACE = "placement_error_m"
_M_ORIENT = "orientation_error_rad"
_M_FINAL_DIST = "final_distance_m"
_M_TRAJ_DEV = "trajectory_deviation_m"
_M_COLLISION = "collision"
_M_TIMEOUT = "timeout"
_M_SIGMA = "pose_sigma_m"
_M_CONFIDENCE = "perception_confidence"
_M_CONSISTENCY = "sensor_consistency"
_M_IK_CONVERGED = "ik_converged"
_M_IK_ERROR_MM = "ik_error_mm"
_M_STEPS = "steps"


def classify_failure(
    metrics: dict[str, float],
    *,
    step: int = 0,
    message: str = "",
) -> list[FailureEvent]:
    """Propose causes for a failed episode, most upstream first.

    Every rule is a threshold on a measured metric and every threshold is a
    module constant, so a classification can be checked by reading the rule
    rather than by trusting it. Where two rules fire, both are reported: a
    grasp that closed on nothing because the object pose was wrong is
    simultaneously a ``POSE_ESTIMATION_FAILURE`` and a ``GRASP_FAILURE``, and
    collapsing that to one label loses the information that makes the
    distinction actionable.

    This is not a learned classifier and does not claim causal attribution. It
    reports which mechanisms are *consistent with* the numbers, which is the
    most that a rule over these metrics can support.

    Args:
        metrics: The episode's metric values, keyed by
            :class:`~robot_ci.metrics.base.MetricName` value.
        step: Control step at which the episode ended.
        message: The episode's outcome message, for provenance.

    Returns:
        Candidate causes in investigation order, or a single ``UNKNOWN`` event
        when no rule fires.
    """
    hypotheses: list[FailureEvent] = []

    def add(category: FailureCategory, weight: float, *evidence: str) -> None:
        """Record a hypothesis with its supporting evidence."""
        hypotheses.append(
            FailureEvent(
                category=category,
                weight=min(max(weight, 0.0), 1.0),
                evidence=evidence,
                step=step,
                message=message,
            )
        )

    sigma = float(metrics.get(_M_SIGMA, 0.0))
    # A metric that was never recorded must not be read as a zero. ``grasp_success``
    # defaulting to 0.0 would make every episode that simply did not report it
    # look like a grasp failure, and a classifier that invents evidence for a
    # mechanism it never observed is worse than one that stays silent. Absent
    # values are therefore carried as ``None`` and every rule that depends on one
    # is skipped.
    raw_grasp = metrics.get(_M_GRASP)
    grasp_ok = None if raw_grasp is None else float(raw_grasp) >= 0.5
    lifted = metrics.get(_M_LIFT)
    place_error = metrics.get(_M_PLACE)
    orient_error = metrics.get(_M_ORIENT)
    collided = float(metrics.get(_M_COLLISION, 0.0)) >= 0.5
    timed_out = float(metrics.get(_M_TIMEOUT, 0.0)) >= 0.5
    ik_converged = float(metrics.get(_M_IK_CONVERGED, 1.0)) >= 0.5

    # -- sensing, most upstream first -------------------------------------
    # Depth noise is the only corruption that raises the fused pose sigma, so
    # a large sigma with the perception source intact implicates depth rather
    # than the estimator.
    if sigma > 0.004:
        add(
            FailureCategory.DEPTH_FAILURE,
            min(0.5 + sigma, 0.95),
            f"pose sigma {sigma * 1000.0:.1f} mm exceeds the 4 mm grasp budget",
        )
        add(
            FailureCategory.POSE_ESTIMATION_FAILURE,
            min(0.4 + 20.0 * sigma, 0.9),
            f"object pose uncertainty {sigma * 1000.0:.1f} mm",
        )
    consistency = float(metrics.get(_M_CONSISTENCY, 1.0))
    if consistency < 0.5:
        add(
            FailureCategory.SENSOR_CALIBRATION,
            min(0.3 + 2.0 * (0.5 - consistency), 0.9),
            f"sensor consistency {consistency:.2f} below 0.50",
        )
    confidence = float(metrics.get(_M_CONFIDENCE, 1.0))
    if confidence < 0.2:
        add(
            FailureCategory.PERCEPTION_FAILURE,
            0.7,
            f"perception confidence {confidence:.2f} below 0.20",
        )
    elif confidence < 0.5 and sigma <= 0.004:
        add(
            FailureCategory.PERCEPTION_FAILURE,
            0.4,
            f"perception confidence {confidence:.2f} below 0.50 with low pose sigma",
        )

    # -- acting -----------------------------------------------------------
    if grasp_ok is False:
        # A grasp that closes on nothing while the pose is uncertain is a
        # perception failure that *manifested* as a grasp failure; the earlier
        # hypothesis is already recorded, and both are kept.
        weight = 0.85 if sigma > 0.004 else 0.6
        detail = (
            f"grasp not achieved with {sigma * 1000.0:.1f} mm pose sigma"
            if sigma > 0.004
            else "grasp not achieved with a confident pose estimate"
        )
        add(FailureCategory.GRASP_FAILURE, weight, detail)
    if grasp_ok and lifted is not None and lifted <= 1e-3:
        add(
            FailureCategory.GRASP_FAILURE,
            0.75,
            f"object never lifted (max {lifted * 1000.0:.1f} mm) despite a closed grasp",
        )
    if grasp_ok and lifted is not None and lifted > 1e-3 and not timed_out:
        if place_error is not None and place_error > 0.05:
            add(
                FailureCategory.PLACEMENT_FAILURE,
                0.8,
                f"placement error {place_error * 1000.0:.1f} mm above 50 mm",
            )
        if orient_error is not None and orient_error > 0.5:
            add(
                FailureCategory.PLACEMENT_FAILURE,
                0.6,
                f"final orientation error {orient_error:.2f} rad above 0.50",
            )
    if not ik_converged:
        add(
            FailureCategory.CONTROL_FAILURE,
            0.7,
            "IK reported non-convergence during the episode",
        )
    if metrics.get(_M_IK_ERROR_MM) is not None and float(metrics[_M_IK_ERROR_MM]) > 20.0:
        add(
            FailureCategory.PLANNING_FAILURE,
            min(0.4 + float(metrics[_M_IK_ERROR_MM]) / 200.0, 0.9),
            f"residual IK position error {float(metrics[_M_IK_ERROR_MM]):.1f} mm above 20 mm",
        )
    if collided:
        add(FailureCategory.COLLISION, 0.9, "collision reported during the episode")
    if timed_out:
        add(FailureCategory.TIMEOUT, 0.8, "episode exhausted its step budget")
    deviation = float(metrics.get(_M_TRAJ_DEV, 0.0))
    if deviation > 1.0 and not collided:
        add(
            FailureCategory.WORLD_MODEL_ERROR,
            min(0.3 + deviation / 10.0, 0.7),
            f"trajectory deviation {deviation:.2f} m from the commanded path",
        )

    if not hypotheses:
        add(
            FailureCategory.UNKNOWN,
            0.0,
            "no classification rule fired; the metrics do not explain this failure",
        )
    return sorted(hypotheses, key=lambda event: (event.rank, -event.weight))
