"""Tests for the failure taxonomy, ablation engine and diagnosis writer.

These exist because the failure path was previously unclassified. Every failed
episode was reported as a timeout, which is a statement about the budget rather
than about the robot, and it made the taxonomy a single useless entry. The
tests here pin the three properties that make the classification worth having:
several hypotheses when several apply, evidence attached to every claim, and
hedged wording that cannot drift toward certainty.
"""

from __future__ import annotations

import math

import pytest

from robot_ci.diagnosis import (
    AblationFactor,
    FailureCategory,
    build_ofat,
    build_pairwise,
    classify_failure,
    diagnose,
    run_ablation,
)
from robot_ci.models.scenario import PolicyComponent, PolicyConfig

# --- taxonomy -------------------------------------------------------------


def test_a_deep_failure_produces_several_hypotheses_not_one() -> None:
    """A grasp that failed *because* the pose was uncertain is two findings.

    Collapsing this to a single label loses the distinction that makes it
    actionable: "the grasp failed" says the gripper is suspect, "the pose was
    uncertain" says perception is, and they call for completely different fixes.
    """
    events = classify_failure(
        {"grasp_success": 0.0, "pose_sigma_m": 0.02, "object_lift_m": 0.0},
        step=40,
        message="grasp did not close",
    )
    categories = {event.category for event in events}
    assert FailureCategory.DEPTH_FAILURE in categories
    assert FailureCategory.POSE_ESTIMATION_FAILURE in categories
    assert FailureCategory.GRASP_FAILURE in categories
    assert len(events) >= 3, "a single label would hide the upstream cause"


def test_hypotheses_are_ordered_most_upstream_first() -> None:
    """A report should name the earliest implicated subsystem first."""
    events = classify_failure(
        {"grasp_success": 0.0, "pose_sigma_m": 0.03, "collision": 1.0, "timeout": 1.0}
    )
    categories = [event.category for event in events]
    assert categories.index(FailureCategory.DEPTH_FAILURE) < categories.index(
        FailureCategory.COLLISION
    ), "sensing must be reported before actuation"


def test_every_hypothesis_carries_its_evidence() -> None:
    """A label separated from its numbers cannot be checked by a reader."""
    events = classify_failure({"grasp_success": 0.0, "pose_sigma_m": 0.02})
    assert events
    for event in events:
        assert event.evidence, f"{event.category} was asserted with no supporting metric"


def test_an_unexplained_failure_is_reported_as_unknown_not_guessed() -> None:
    """A classifier with nothing to go on must say so."""
    events = classify_failure({"task_success": 0.0, "steps": 5})
    assert [event.category for event in events] == [FailureCategory.UNKNOWN]
    assert events[0].weight == 0.0


def test_placement_failure_is_not_reported_when_nothing_was_grasped() -> None:
    """Placement accuracy is meaningless when the object was never picked up."""
    events = classify_failure({"grasp_success": 0.0, "object_lift_m": 0.0, "placement_error_m": 0.3})
    assert FailureCategory.PLACEMENT_FAILURE not in {e.category for e in events}


# --- ablation -------------------------------------------------------------


def _factors() -> list[AblationFactor]:
    """Three real policy parameters, one per subsystem that reads them."""
    return [
        AblationFactor("max_position_sigma_m", (0.004, 0.05), PolicyComponent.PERCEPTION),
        AblationFactor("clearance_bias_m", (0.0, 0.05), PolicyComponent.PLANNER),
        AblationFactor("ik_damping", (0.05, 0.2), PolicyComponent.CONTROLLER),
    ]


def _reference() -> PolicyConfig:
    return PolicyConfig(
        name="policy_v2",
        parameters={"max_position_sigma_m": 0.004, "clearance_bias_m": 0.0, "ik_damping": 0.05},
    )


def test_a_factor_needs_two_values_or_it_is_not_an_experiment() -> None:
    """A factor with one value produces no comparison, so it must be rejected."""
    with pytest.raises(ValueError, match="at least two values"):
        AblationFactor("ik_damping", (0.05,), PolicyComponent.CONTROLLER)


def test_ofat_changes_exactly_one_parameter_per_arm() -> None:
    """Each OFAT row has to be independently interpretable."""
    arms = build_ofat(_factors(), _reference())
    assert len(arms) == 3, "three factors at one non-reference value each"
    for config, _description, active in arms:
        assert len(active) == 1
        for parameter, value in config.parameters.items():
            baseline = _reference().parameters[parameter]
            if parameter in active:
                assert value != baseline
            else:
                assert value == baseline, f"{parameter} moved in an OFAT arm"


def test_pairwise_arms_cover_every_pair() -> None:
    """Interactions are invisible to OFAT, so the pairwise design must exist."""
    arms = build_pairwise(_factors(), _reference())
    assert len(arms) == 3, "three factors give three pairs"
    assert all(len(active) == 2 for _, _, active in arms)


def test_ablation_attributes_the_largest_effect_to_the_right_subsystem() -> None:
    """A substitution experiment must name the component it implicates."""

    def evaluate(config: PolicyConfig, _arm_id: str) -> tuple[float, int, dict[str, float]]:
        params = config.parameters
        rate = 0.8
        if params.get("max_position_sigma_m", 0.004) > 0.01:
            rate -= 0.5
        if params.get("ik_damping", 0.05) > 0.1:
            rate += 0.1
        return max(0.0, rate), 8, {"task_success": max(0.0, rate)}

    result = run_ablation(_reference(), _factors(), evaluate, designs=("ofat",))
    ranked = dict(result.attribution)
    assert ranked[PolicyComponent.PERCEPTION.value] == pytest.approx(-0.5)
    assert ranked[PolicyComponent.PLANNER.value] == pytest.approx(0.0), (
        "a factor that changed nothing must still be listed, with a change of "
        "zero -- otherwise 'inert' and 'forgotten' look identical"
    )
    assert abs(ranked[PolicyComponent.PERCEPTION.value]) > abs(
        ranked[PolicyComponent.CONTROLLER.value]
    )


def test_ablation_always_measures_its_own_reference() -> None:
    """Attribution is a difference; a difference needs a measured baseline."""
    seen: list[str] = []

    def evaluate(config: PolicyConfig, arm_id: str) -> tuple[float, int, dict[str, float]]:
        seen.append(arm_id)
        return 0.5, 4, {"task_success": 0.5}

    result = run_ablation(_reference(), _factors(), evaluate, designs=("ofat",))
    assert seen[0] == result.reference_arm_id, "the reference must be measured first"
    assert len(result.arms) == 1 + 3


def test_unknown_design_is_rejected_rather_than_silently_ignored() -> None:
    """A typo in a design name must not quietly run nothing."""
    with pytest.raises(ValueError, match="unknown ablation design"):
        run_ablation(_reference(), _factors(), lambda _c, _i: (0.5, 2, {}), designs=("ofat", "nope"))


def test_attribution_carries_its_own_limitation() -> None:
    """The caveat travels with the numbers so it cannot be dropped in a report."""
    result = run_ablation(_reference(), _factors(), lambda _c, _i: (0.5, 4, {}), designs=("ofat",))
    assert "not a causal claim" in result.note
    assert "association" in result.note


# --- diagnosis ------------------------------------------------------------


def test_diagnosis_never_claims_causation() -> None:
    """The wording is the deliverable, so it is asserted directly."""
    result = diagnose(
        condition="depth_noise_mm=10",
        baseline_metrics={"task_success": 0.5, "grasp_error_m": 0.003, "perception_confidence": 0.9},
        candidate_metrics={"task_success": 0.0, "grasp_error_m": 0.031, "perception_confidence": 0.4},
        samples=8,
    )
    text = result.summary().lower()
    assert "consistent with" in text
    assert "caused by" not in text
    assert "proves" not in text
    assert result.suspected_subsystem == "perception"
    assert "limitation" in text


def test_a_missing_success_metric_yields_no_attribution() -> None:
    """With nothing to compare, the honest answer is that there is no finding."""
    result = diagnose(
        condition="nominal",
        baseline_metrics={"grasp_error_m": 0.003},
        candidate_metrics={"grasp_error_m": 0.004},
        samples=8,
    )
    assert "insufficient evidence" in result.confidence_phrase
    assert result.suspected_subsystem == ""


def test_confidence_wording_scales_with_the_measured_effect() -> None:
    """A large effect and a negligible one must not read the same way."""
    strong = diagnose(
        condition="c",
        baseline_metrics={"task_success": 1.0, "perception_confidence": 0.9},
        candidate_metrics={"task_success": 0.0, "perception_confidence": 0.1},
        samples=8,
    )
    weak = diagnose(
        condition="c",
        baseline_metrics={"task_success": 0.5, "perception_confidence": 0.5},
        candidate_metrics={"task_success": 0.49, "perception_confidence": 0.498},
        samples=8,
    )
    assert "strong" in strong.confidence_phrase
    assert "weak" in weak.confidence_phrase or "no association" in weak.confidence_phrase


def test_too_few_samples_refuses_to_attribute() -> None:
    """Two episodes cannot support a subsystem claim however large the effect."""
    result = diagnose(
        condition="c",
        baseline_metrics={"task_success": 1.0, "perception_confidence": 0.9},
        candidate_metrics={"task_success": 0.0, "perception_confidence": 0.0},
        samples=2,
    )
    assert "insufficient samples" in result.confidence_phrase


def test_a_rate_change_at_the_zero_boundary_is_still_measurable() -> None:
    """Cohen's h is used precisely because the ordinary effect size is undefined here.

    A regression report whose candidate arm scored 0.0 is the common case, and a
    statistic that returns NaN there would silently drop the strongest evidence
    in the set.
    """
    result = diagnose(
        condition="c",
        baseline_metrics={"task_success": 0.5, "grasp_success": 0.5},
        candidate_metrics={"task_success": 0.0, "grasp_success": 0.0},
        samples=8,
    )
    assert not math.isnan(result.evidence[0].change or float("nan"))
    assert "grasp_success" in result.summary()


def test_wall_clock_time_cannot_implicate_a_subsystem() -> None:
    """Planning time is jitter, not behaviour.

    Measured: a diagnosis of two arms with identical success rates to sixteen
    digits attributed a regression to the planner on a 64 ms difference in
    ``planning_time_ms``. Planning time moves with machine load between runs, so
    attributing a behavioural cause to it manufactures a finding out of noise.
    """
    result = diagnose(
        condition="depth_noise_mm=0",
        baseline_metrics={
            "task_success": 1.0,
            "planning_time_ms": 1028.0,
            "placement_error_m": 0.007306,
        },
        candidate_metrics={
            "task_success": 1.0,
            "planning_time_ms": 963.4,
            "placement_error_m": 0.007306,
        },
        samples=4,
    )
    assert result.suspected_subsystem == "", (
        "no subsystem may be named when only a cost metric differs"
    )
    assert "no regression detected" in result.confidence_phrase
    # The difference is still reported, marked as excluded, rather than hidden.
    assert "excluded from attribution" in result.summary()


def test_an_unchanged_success_rate_yields_no_attribution() -> None:
    """With nothing to explain, the honest output is a null.

    Offering a subsystem anyway answers a question nobody asked, and the metrics
    that will be found to support it are the ones that move between identical
    runs.
    """
    result = diagnose(
        condition="nominal",
        baseline_metrics={"task_success": 0.75, "grasp_error_m": 0.003},
        candidate_metrics={"task_success": 0.75, "grasp_error_m": 0.003},
        samples=6,
    )
    assert result.suspected_subsystem == ""
    assert result.counter_experiment == ""
    assert "nothing to attribute" in result.confidence_phrase
    assert "no regression detected" in result.summary()


def test_a_real_regression_is_still_attributed_after_the_guards() -> None:
    """The two guards must not swallow a genuine finding."""
    result = diagnose(
        condition="depth_noise_mm=10",
        baseline_metrics={
            "task_success": 0.75,
            "planning_time_ms": 1000.0,
            "grasp_error_m": 0.003,
            "perception_confidence": 0.9,
        },
        candidate_metrics={
            "task_success": 0.0,
            "planning_time_ms": 1010.0,
            "grasp_error_m": 0.031,
            "perception_confidence": 0.3,
        },
        samples=8,
    )
    assert result.suspected_subsystem == "perception"
    assert "consistent with" in result.confidence_phrase
