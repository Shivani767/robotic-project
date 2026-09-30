"""The bridge from a perturbation point to two real episodes.

This exercises the only part of counterexample search that touches a simulator:
constructing the perturbed scenario, running both arms under a shared seed, and
reporting the outcome. It runs on the mock backend, which is fast and
deterministic, so the plumbing is verified without a physics engine.

The mock backend deliberately **cannot** produce a policy difference: it exposes
no Jacobian, so the inverse-kinematics policy holds its configuration and both
arms behave identically. The final test asserts that the search therefore finds
nothing there. Without that assertion a future "fix" could make the oracle
appear to discover a regression on a backend where none can exist, and the
result would look like a finding.
"""

from __future__ import annotations

import pytest

from robot_ci.counterexamples import (
    BinaryBoundarySearcher,
    CachingOracle,
    EpisodeOracle,
    PerturbationAxis,
    SearchSpace,
    perturbed_scenario,
)
from robot_ci.models.scenario import PolicyConfig, ScenarioConfig

pytestmark = pytest.mark.integration

#: Kept tiny: this is a plumbing test, not a science test.
MAX_STEPS = 6


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(scenario_id="oracle_probe", seed=11)


def _space() -> SearchSpace:
    return SearchSpace(
        [PerturbationAxis("depth_noise_mm", "sensors.depth.sigma_mm", 0.0, 60.0, "mm")]
    )


def _oracle(**overrides: object) -> EpisodeOracle:
    return EpisodeOracle(
        base_scenario=_scenario(),
        space=_space(),
        baseline=PolicyConfig(name="policy_v1"),
        candidate=PolicyConfig(name="policy_v2"),
        backend_name="mock",
        max_steps=MAX_STEPS,
        **overrides,  # type: ignore[arg-type]
    )


def test_oracle_runs_both_arms_and_prefixes_their_metrics() -> None:
    """One evaluation must produce one episode per arm, with both metric sets."""
    oracle = _oracle()
    try:
        trial = oracle.evaluate({"depth_noise_mm": 30.0})
    finally:
        oracle.close()

    assert trial.point == {"depth_noise_mm": 30.0}
    assert len(oracle.history) == 1
    assert oracle.history[0]["point"] == {"depth_noise_mm": 30.0}
    assert any(key.startswith("baseline.") for key in trial.metrics)
    assert any(key.startswith("candidate.") for key in trial.metrics)
    # The step budget override must actually reach the episode.
    assert trial.metrics["baseline.steps"] <= MAX_STEPS
    assert trial.metrics["candidate.steps"] <= MAX_STEPS


def test_oracle_is_deterministic_for_the_same_point() -> None:
    """A counterexample that is not reproducible is not a counterexample.

    The comparison deliberately excludes ``planning_time_ms``. It is measured
    with ``time.perf_counter()`` inside the IK loop, so it reflects how busy the
    machine was rather than what the policy did; measured here it differs
    between two otherwise identical runs while the other 42 metrics agree
    exactly. Asserting on the whole vector would therefore always fail, and a
    wall-clock metric compared across arms would let machine load masquerade as
    a policy difference. Everything that carries outcome -- success, status,
    grasp and placement error, steps, belief sigma -- must match exactly.
    """
    clock_dependent = {"baseline.planning_time_ms", "candidate.planning_time_ms"}
    first = _oracle()
    second = _oracle()
    try:
        left = first.evaluate({"depth_noise_mm": 45.0})
        right = second.evaluate({"depth_noise_mm": 45.0})
    finally:
        first.close()
        second.close()

    assert left.baseline_success == right.baseline_success
    assert left.candidate_success == right.candidate_success
    deterministic_left = {k: v for k, v in left.metrics.items() if k not in clock_dependent}
    deterministic_right = {k: v for k, v in right.metrics.items() if k not in clock_dependent}
    assert deterministic_left == deterministic_right
    for key in clock_dependent:
        assert key in left.metrics and key in right.metrics, (
            f"{key} disappeared from the metric vector; the exclusion above "
            "would silently stop matching anything"
        )


def test_perturbation_reaches_the_scenario_and_arms_the_corruption() -> None:
    """The point must change the conditions, and a non-zero value must enable them.

    Grid expansion once wrote a corruption magnitude without setting its
    ``enabled`` flag, so the value was recorded in the artifact and measured by
    nothing. The oracle goes through the same ``set_by_path`` path, so it
    inherits the same guard, and this test pins that it does.
    """
    space = _space()
    base = _scenario()
    assert base.sensors.is_nominal, "the base scenario must start uncorrupted"
    perturbed = perturbed_scenario(base, space, {"depth_noise_mm": 25.0})
    assert perturbed.sensors.depth.sigma_mm == 25.0
    assert perturbed.sensors.depth.enabled is True
    assert perturbed.sensors.is_nominal is False
    assert perturbed.config_hash() != base.config_hash()


def test_search_finds_nothing_where_no_regression_can_exist() -> None:
    """An honest negative.

    The mock backend has no Jacobian, so the policy cannot move the arm and the
    two arms cannot differ. A search that reported a counterexample here would be
    reporting noise, so the expected result is 'not found' with no artifact.
    """
    oracle = _oracle()
    cached = CachingOracle(oracle)
    try:
        outcome = BinaryBoundarySearcher(_space(), budget=8, tolerance=1.0).search(cached)
    finally:
        oracle.close()

    assert outcome.found is False
    assert outcome.counterexample is None
    assert "both arms fail" in outcome.notes
    # Exactly one evaluation: the search must stop the moment its premise fails.
    # A counterexample needs a baseline that *succeeds* somewhere, and here it
    # does not, so spending budget probing outward would be searching for a
    # regression that cannot exist. Aborting after the origin is the correct
    # answer, not a bug -- which is why the assertion is equality, not a floor.
    assert outcome.evaluations == 1
