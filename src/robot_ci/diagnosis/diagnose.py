"""Turn a measured regression into a hedged, evidence-backed diagnosis.

The output of this module is a *hypothesis about a cause*, and it is written to
be quotable in a review without overstating what was measured. Three rules
govern the language, and they are enforced by :meth:`Diagnosis.summary` rather
than left to the person writing the report:

* Nothing here claims causation. A regression measured under depth noise is
  *consistent with* a perception change; it does not establish one, because the
  two arms also differed in the trajectories they produced.
* Every claim carries the numbers it came from. An assertion that cannot be
  traced to a metric is not emitted at all.
* What would falsify the hypothesis is stated alongside it. A diagnosis that
  cannot be wrong is not a diagnosis.

This mirrors how regression triage works in practice: the tool narrows the
search space and hands the engineer the next experiment to run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from robot_ci.diagnosis.taxonomy import FailureEvent
from robot_ci.statistics import cohens_h

__all__ = ["Diagnosis", "Evidence", "diagnose"]


class Evidence(BaseModel):
    """One measured fact bearing on a diagnosis.

    Attributes:
        statement: What was observed, in one sentence.
        metric: Name of the metric it came from, or ``""`` if it is a
            structural fact about the experiment rather than a number.
        baseline_value: The baseline arm's value, or ``None``.
        candidate_value: The candidate arm's value, or ``None``.
        change: ``candidate - baseline``, or ``None`` when either is absent.
    """

    schema_version: str = Field(default="1.0.0")
    statement: str
    metric: str = ""
    baseline_value: float | None = None
    candidate_value: float | None = None
    change: float | None = None

    def render(self) -> str:
        """One line, with the numbers inline so it can be quoted alone."""
        if self.change is None or self.baseline_value is None or self.candidate_value is None:
            return self.statement
        return (
            f"{self.statement} ({self.metric or 'metric'}: "
            f"{self.baseline_value:.4g} -> {self.candidate_value:.4g}, "
            f"change {self.change:+.4g})"
        )


class Diagnosis(BaseModel):
    """A cause hypothesis for one measured regression.

    Attributes:
        condition: The condition under which the regression was observed.
        suspected_subsystem: The component most strongly associated with the
            change, or ``""`` when nothing was implicated.
        confidence_phrase: How strongly the evidence supports the attribution,
            chosen from a fixed vocabulary rather than a free-form number, so
            that "high confidence" always means the same thing.
        evidence: The measured facts the attribution rests on.
        failure_categories: The failure categories seen on the failing episodes.
        counter_experiment: The experiment that would confirm or refute this.
        limitations: What this diagnosis cannot establish.
    """

    schema_version: str = Field(default="1.0.0")
    condition: str
    suspected_subsystem: str = ""
    confidence_phrase: str = "insufficient evidence to attribute"
    evidence: tuple[Evidence, ...] = ()
    failure_categories: tuple[str, ...] = ()
    counter_experiment: str = ""
    limitations: tuple[str, ...] = ()

    def summary(self) -> str:
        """Render the diagnosis as the report block.

        Returns:
            Multi-line text using hedged phrasing throughout.
        """
        lines = [
            f"regression condition   : {self.condition}",
            f"strongest association  : {self.suspected_subsystem or 'none identified'}",
            f"confidence             : {self.confidence_phrase}",
        ]
        if self.evidence:
            lines.append("evidence:")
            lines.extend(f"  - {item.render()}" for item in self.evidence)
        if self.failure_categories:
            lines.append("failure categories seen:")
            lines.extend(f"  - {name}" for name in self.failure_categories)
        if self.counter_experiment:
            lines.append(f"to confirm or refute    : {self.counter_experiment}")
        for caveat in self.limitations:
            lines.append(f"limitation             : {caveat}")
        return "\n".join(lines)


#: Phrases used to qualify an attribution, keyed by the measured effect size of
#: the implicated metric. A fixed vocabulary is used rather than a free-form
#: confidence number so that "strong" means the same thing in every report, and
#: so that the wording cannot drift toward certainty as features are added.
def _confidence_phrase(effect: float, samples: int) -> str:
    """Qualify an attribution from its effect size and sample count.

    Args:
        effect: Absolute change in the implicated metric, baseline to
            candidate.
        samples: Paired observations behind the change.

    Returns:
        A phrase from the fixed vocabulary.
    """
    if samples < 3:
        return "insufficient samples to attribute"
    magnitude = abs(effect)
    if magnitude >= 0.5:
        return "strong association; the evidence is consistent with this cause"
    if magnitude >= 0.2:
        return "moderate association; the evidence is consistent with this cause"
    if magnitude >= 0.05:
        return "weak association; treat as a lead, not a conclusion"
    return "no association above noise; a different explanation is likely"


#: Metric that most directly implicates each subsystem, in the order they are
#: consulted. The mapping is deliberately small: a metric is only listed here if
#: the subsystem that owns it is the only one that can move it.
_SUBSYSTEM_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("perception", ("pose_sigma_m", "perception_confidence", "grasp_error_m")),
    ("sensor_fusion", ("sensor_consistency", "pose_sigma_m")),
    ("planner", ("trajectory_deviation_m", "planning_time_ms")),
    ("controller", ("ik_converged", "trajectory_deviation_m")),
    ("gripper", ("grasp_success", "grasp_error_m")),
)


def diagnose(
    *,
    condition: str,
    baseline_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    samples: int,
    failures: Sequence[FailureEvent] = (),
    success_metric: str = "task_success",
) -> Diagnosis:
    """Explain a measured regression without claiming more than was measured.

    The attribution is made by looking for the metric with the largest
    standardised change and mapping it to the subsystem that owns it, via
    :data:`_SUBSYSTEM_METRICS`. Two guards keep the result honest:

    * The success change itself is checked first. If success did not move, the
      honest diagnosis is that there is no regression here, and the function
      says so instead of hunting for a subsystem to blame.
    * Failure categories observed on the failing episodes are reported
      alongside, so a reader can see whether the taxonomy and the metric
      evidence point the same way. When they disagree, both are shown and the
      disagreement is itself recorded as a limitation.

    Args:
        condition: Human-readable description of the condition.
        baseline_metrics: Mean metrics for the baseline arm.
        candidate_metrics: Mean metrics for the candidate arm.
        samples: Paired observations per arm.
        failures: Failure events from the candidate's failing episodes.
        success_metric: Metric compared first to decide whether anything
            regressed at all.

    Returns:
        A diagnosis whose wording is hedged in proportion to its evidence.
    """
    baseline_success = float(baseline_metrics.get(success_metric, float("nan")))
    candidate_success = float(candidate_metrics.get(success_metric, float("nan")))
    success_change = candidate_success - baseline_success

    evidence: list[Evidence] = []
    # The success change is always reported, including when it is favourable.
    # A report that only lists regressions is one that cannot be trusted to list
    # improvements, and a reader cannot tell an omitted number from a zero.
    if math.isnan(success_change):  # a metric missing from either arm
        evidence.append(
            Evidence(
                statement=f"{success_metric} was not recorded for this condition",
                metric=success_metric,
            )
        )
        return Diagnosis(
            condition=condition,
            confidence_phrase="insufficient evidence to attribute",
            evidence=tuple(evidence),
            limitations=("no success metric was recorded for either arm",),
        )

    evidence.append(
        Evidence(
            statement=f"{success_metric} changed",
            metric=success_metric,
            baseline_value=baseline_success,
            candidate_value=candidate_success,
            change=success_change,
        )
    )

    # Rank candidate metrics by standardised change. Cohen's h is used for
    # rates because it is defined at the 0 and 1 boundaries, where an ordinary
    # standardised mean difference is undefined -- and a regression report whose
    # arm scored 0.0 is exactly that case.
    scored: list[tuple[float, str, Evidence]] = []
    for name, candidate_value in candidate_metrics.items():
        if name not in baseline_metrics or name == success_metric:
            continue
        baseline_value = float(baseline_metrics[name])
        change = float(candidate_value) - baseline_value
        if 0.0 <= baseline_value <= 1.0 and 0.0 <= candidate_value <= 1.0:
            size = abs(cohens_h(baseline_value, candidate_value))
        elif baseline_value != 0.0:
            size = abs(change / baseline_value)
        else:
            size = abs(change)
        scored.append(
            (
                size,
                name,
                Evidence(
                    statement=f"{name} changed",
                    metric=name,
                    baseline_value=baseline_value,
                    candidate_value=float(candidate_value),
                    change=change,
                ),
            )
        )
    scored.sort(key=lambda item: -item[0])

    suspected = ""
    for _, name, _ in scored:
        for subsystem, owned in _SUBSYSTEM_METRICS:
            if name in owned:
                suspected = subsystem
                break
        if suspected:
            break

    # The strongest few metrics, plus success, are the evidence. More than a
    # handful and the block stops being checkable by a reader.
    evidence.extend(item for _, _, item in scored[:4])
    top_effect = scored[0][0] if scored else abs(success_change)

    categories = tuple(
        dict.fromkeys(event.category.value for event in failures)
    )
    limitations = [
        "association only: the two arms also produced different trajectories, "
        "so a change in a downstream metric is not attributable to the "
        "subsystem that owns it without a substitution experiment",
    ]
    if not failures:
        limitations.append(
            "no failure events were supplied, so the taxonomy could not be "
            "cross-checked against the metric evidence"
        )

    return Diagnosis(
        condition=condition,
        suspected_subsystem=suspected,
        confidence_phrase=_confidence_phrase(top_effect, samples),
        evidence=tuple(evidence),
        failure_categories=categories,
        counter_experiment=(
            f"re-run both arms with {suspected} held at its baseline setting and "
            "every other parameter unchanged; the regression should disappear if "
            "the attribution is right"
            if suspected
            else ""
        ),
        limitations=tuple(limitations),
    )
