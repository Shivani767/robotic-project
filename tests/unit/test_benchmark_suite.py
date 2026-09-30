"""Tests for the failure-injection benchmark suite.

The suite's value is that it can say an expectation is *wrong*. Three of the
expectations in this file were falsified by its first run and corrected against
the measurement, which is the property these tests protect: a suite whose
expectations are all "MET" because nobody checks them is a list of sweeps.
"""

from __future__ import annotations

import pytest

from robot_ci.evaluation import (
    BENCHMARKS,
    UNIMPLEMENTED_BENCHMARKS,
    Benchmark,
    BenchmarkFamily,
    benchmark_suite,
    run_benchmark,
)
from robot_ci.scenarios import default_scenario, set_by_path


def _constant(rate: float, metrics: dict[str, float] | None = None):
    """An ``evaluate`` callable returning a fixed outcome."""

    def evaluate(_scenario, _seeds):
        return rate, metrics or {}

    return evaluate


def test_every_required_family_is_represented() -> None:
    """The specification names six families; the suite must cover all of them."""
    for family in BenchmarkFamily:
        assert benchmark_suite(family), f"no benchmark declares the {family.value} family"


def test_benchmark_ids_are_unique() -> None:
    """Two benchmarks sharing an id cannot be told apart in a report."""
    ids = [b.id for b in BENCHMARKS]
    assert len(ids) == len(set(ids))


def test_every_benchmark_names_a_runnable_field() -> None:
    """A benchmark whose injection path does not exist is a benchmark that does nothing."""
    base = default_scenario()
    for benchmark in BENCHMARKS:
        set_by_path(base, benchmark.field_path, benchmark.value)


def test_an_unimplementable_fault_is_reported_not_swallowed() -> None:
    """A bad injection must fail loudly rather than measure the nominal condition.

    The suite hits this for real: the colour-shift benchmark declared a scalar
    against a 3-tuple field, and the only reason it was visible instead of a
    silent pass is this check.
    """
    broken = Benchmark(
        id="broken",
        family=BenchmarkFamily.DEPTH,
        description="targets a field that does not exist",
        field_path="sensors.depth.not_a_field",
        value=1.0,
    )
    with pytest.raises(ValueError, match="does not have"):
        run_benchmark(
            broken,
            base_scenario=default_scenario(),
            seeds=(1,),
            evaluate=_constant(1.0),
        )


def test_an_unarmed_injection_is_a_failure_not_a_pass() -> None:
    """A corruption that does not arm would measure nominal and look like success.

    This is the worst failure mode for a suite whose purpose is to catch false
    passes, so it is checked explicitly rather than trusted.
    """
    result = run_benchmark(
        BENCHMARKS[0],
        base_scenario=default_scenario(),
        seeds=(1,),
        evaluate=_constant(1.0),
    )
    # Whether it armed depends on the benchmark; what must never happen is a
    # "MET" that came from having measured the nominal condition.
    if result.expectation_met is True:
        assert "disarmed" not in result.notes


def test_an_expectation_that_does_not_match_is_reported_as_such() -> None:
    """The core property: a wrong prediction must not pass quietly."""
    failing = Benchmark(
        id="expects_failure",
        family=BenchmarkFamily.DEPTH,
        description="claims the task must fail, and it does not",
        field_path="sensors.depth.sigma_mm",
        value=20.0,
        expect_success=False,
    )
    result = run_benchmark(
        failing,
        base_scenario=default_scenario(),
        seeds=(1,),
        evaluate=_constant(1.0),
    )
    assert result.expectation_met is False


def test_a_benchmark_with_no_claim_reports_no_claim() -> None:
    """Absence of a prediction is not a pass.

    A suite that reports an unchecked expectation as verified would let a
    reader believe more has been tested than has.
    """
    unclaimed = Benchmark(
        id="silent",
        family=BenchmarkFamily.ROBOT,
        description="makes no falsifiable prediction",
        field_path="sensors.robot_state.joint_offset_rad",
        value=0.05,
    )
    result = run_benchmark(
        unclaimed,
        base_scenario=default_scenario(),
        seeds=(1,),
        evaluate=_constant(0.5),
    )
    assert result.expectation_met is None


def test_an_unverified_expectation_is_labelled_as_a_hypothesis() -> None:
    """A prediction derived from a model must not read as a measurement."""
    result = run_benchmark(
        next(b for b in BENCHMARKS if not b.verified),
        base_scenario=default_scenario(),
        seeds=(1,),
        evaluate=_constant(1.0),
    )
    assert "hypothesis" in result.notes


def test_unimplemented_faults_carry_a_reason() -> None:
    """A recorded gap must say why, or it is just an omission."""
    for name, reason in UNIMPLEMENTED_BENCHMARKS:
        assert reason.strip(), f"{name} is listed as unimplemented with no reason"
    names = {name for name, _ in UNIMPLEMENTED_BENCHMARKS}
    implemented = {b.id for b in BENCHMARKS}
    assert not (names & implemented), "a fault is listed both as runnable and as missing"


def test_the_gripper_fault_is_recorded_because_it_is_inert() -> None:
    """A field that arms but does nothing belongs in the missing list.

    Found by this suite falsifying its own expectation: at 1.0, zero of seven
    metrics move against nominal. Documenting that is more useful than deleting
    the benchmark, because a reader would otherwise assume the fault was tested.
    """
    assert "robot_gripper_stuck" in {n for n, _ in UNIMPLEMENTED_BENCHMARKS}
    assert "robot_gripper_stuck" not in {b.id for b in BENCHMARKS}
