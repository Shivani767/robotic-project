"""Counterexample search: validated against oracles with a known answer.

A search algorithm cannot be validated by pointing it at a system whose answer
is unknown. These tests therefore use :class:`AnalyticRegressionOracle`, an
*explicitly synthetic* harness whose regression boundary is set by construction,
so the tests can assert that each strategy recovers a number that was put there
on purpose.

That is a statement about the search code, not about any robot. The end-to-end
path from a perturbation point to two real episodes is exercised separately, in
``tests/integration/test_episode_oracle.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ci.counterexamples import (
    AdaptiveBoundarySearcher,
    BinaryBoundarySearcher,
    CachingOracle,
    Counterexample,
    DiscoveryRequest,
    GridSearcher,
    OutcomeOracle,
    PerturbationAxis,
    RandomSearcher,
    SearchSpace,
    Trial,
    discover,
    load_counterexample,
    ramp_point,
    reproduce,
    save_counterexample,
)
from robot_ci.models.scenario import ScenarioConfig

#: The candidate stops succeeding exactly here, in axis units.
DEPTH_BOUNDARY_MM = 10.0
DEPTH_MAX_MM = 60.0


def _space(*, axes: int = 1) -> SearchSpace:
    """A one- or two-axis space whose nominal value is each lower bound."""
    if axes == 1:
        return SearchSpace(
            [PerturbationAxis("depth_noise_mm", "sensors.depth.sigma_mm", 0.0, DEPTH_MAX_MM, "mm")]
        )
    return SearchSpace(
        [
            PerturbationAxis("depth_noise_mm", "sensors.depth.sigma_mm", 0.0, DEPTH_MAX_MM, "mm"),
            PerturbationAxis("occlusion_percent", "sensors.rgb.occlusion_percent", 0.0, 60.0, "%"),
        ]
    )


class AnalyticRegressionOracle(OutcomeOracle):
    """A synthetic two-arm oracle with a boundary chosen by the test.

    The baseline always succeeds. The candidate succeeds while the *first axis,
    normalized into ``[0, 1]``,* is below ``boundary``. Expressing the boundary
    in normalized units is what makes one oracle work for both the single-axis
    and the ramp case: when the searcher evaluates a ramp point, the first axis
    is exactly ``lower + t * span``, so its normalized value is ``t``.
    """

    def __init__(
        self,
        space: SearchSpace,
        *,
        boundary: float = 0.5,
        noise_seed: int = 0,
        noise: float = 0.0,
    ) -> None:
        """Configure the normalized boundary and optional stochasticity."""
        self.space = space
        self.boundary = float(boundary)
        self._rng = np.random.default_rng(noise_seed)
        self.noise = float(noise)
        self.calls = 0

    def _scale(self, point: dict[str, float]) -> float:
        """Normalized value of the first axis."""
        axis = self.space.axes[0]
        return axis.normalized(point[axis.name])

    def evaluate(self, point: dict[str, float]) -> Trial:
        """Baseline always succeeds; the candidate fails at or above the boundary."""
        self.calls += 1
        scale = self._scale(point)
        candidate_ok = scale < self.boundary
        if self.noise > 0.0 and abs(scale - self.boundary) < self.noise:
            candidate_ok = bool(self._rng.uniform() < 0.5)
        return Trial(
            point=dict(point),
            baseline_success=True,
            candidate_success=candidate_ok,
            metrics={"candidate.scale": scale},
        )

#: The candidate's boundary for the single-axis case, in axis units.
SINGLE_BOUNDARY_UNITS = DEPTH_BOUNDARY_MM


def test_binary_search_recovers_a_known_boundary() -> None:
    """The specification's worked example, with the answer known in advance.

    Exponential probe outward, then bisect. The recovered boundary must sit
    within the bracket the search reports, and the bracket must be narrower
    than the tolerance the searcher was configured with.
    """
    space = _space()
    oracle = AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)
    searcher = BinaryBoundarySearcher(space, budget=64, tolerance=0.1)
    outcome = searcher.search(oracle)

    assert outcome.found
    assert outcome.boundary_axis == "depth_noise_mm"
    assert outcome.boundary == pytest.approx(SINGLE_BOUNDARY_UNITS, abs=0.5)
    low, high = outcome.bracket
    assert low < SINGLE_BOUNDARY_UNITS <= high
    assert high - low <= 0.1 + 1e-9, "the bracket must respect the tolerance"
    # The reported counterexample is the smallest failing value evaluated, so
    # it may not be smaller than the true boundary.
    assert outcome.counterexample.point["depth_noise_mm"] >= SINGLE_BOUNDARY_UNITS - 0.1


def test_binary_search_is_logarithmic_not_linear() -> None:
    """Bisection must beat a linear scan by a wide margin, or it is pointless."""
    space = _space()
    oracle = AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)
    outcome = BinaryBoundarySearcher(space, budget=256, tolerance=1e-3).search(oracle)
    assert outcome.found
    # A 16-step linear scan is 10 probes; bisection to 1e-3 over a 60 mm span
    # is about 16. The assertion is deliberately loose: it checks the order of
    # growth, not an exact count that would break on an unrelated change.
    assert outcome.evaluations < 40, outcome.evaluations


def test_binary_search_reports_nominal_failure_distinctly() -> None:
    """A candidate that fails at nominal needs no perturbation at all.

    This is the most severe possible finding and it must not be reported as a
    boundary of zero without saying so.
    """
    space = _space()
    oracle = AnalyticRegressionOracle(space, boundary=0.0)
    outcome = BinaryBoundarySearcher(space, budget=32).search(oracle)
    assert outcome.found
    assert outcome.counterexample.point["depth_noise_mm"] == 0.0
    assert "nominal" in outcome.notes


def test_binary_search_reports_when_both_arms_fail_at_nominal() -> None:
    """If the baseline fails at nominal there is no baseline success to regress."""

    class BothFailOracle(AnalyticRegressionOracle):
        def evaluate(self, point: dict[str, float]) -> Trial:
            self.calls += 1
            return Trial(point=dict(point), baseline_success=False, candidate_success=False)

    space = _space()
    outcome = BinaryBoundarySearcher(space, budget=32).search(BothFailOracle(space))
    assert not outcome.found
    assert outcome.counterexample is None
    assert "both arms fail" in outcome.notes


def test_binary_beats_the_grid_at_matched_accuracy() -> None:
    """RQ1 in miniature, at matched accuracy rather than matched budget.

    Cost is only comparable at equal accuracy. A coarse grid can look cheap
    because a severity-ordered lattice hits the boundary quickly when the
    boundary is near nominal -- measured here, a 4 mm lattice finds it in four
    evaluations. That is not a fair comparison against bisection, which is being
    asked for a far tighter answer. Demanding the *same* tolerance from both is
    what exposes the difference: to place the boundary within 0.1 mm the lattice
    must carry 601 points spaced 0.1 mm apart, while bisection needs only
    logarithmic narrowing.
    """
    space = _space()
    boundary = DEPTH_BOUNDARY_MM / DEPTH_MAX_MM
    tolerance_mm = 0.1

    grid = GridSearcher(
        space, steps_per_axis=int(DEPTH_MAX_MM / tolerance_mm) + 1, budget=2000
    ).search(AnalyticRegressionOracle(space, boundary=boundary))
    binary = BinaryBoundarySearcher(space, budget=64, tolerance=tolerance_mm).search(
        AnalyticRegressionOracle(space, boundary=boundary)
    )

    assert grid.found and binary.found, "both strategies must find the regression"
    # Matched accuracy: each answer is within the tolerance of the true boundary.
    for name, outcome in (("grid", grid), ("binary", binary)):
        error = abs(outcome.counterexample.point["depth_noise_mm"] - DEPTH_BOUNDARY_MM)
        assert error <= tolerance_mm, f"{name} missed the boundary by {error:.3f} mm"
    # And at that accuracy, bisection is an order of magnitude cheaper.
    assert binary.evaluations < grid.evaluations / 5, (
        f"at {tolerance_mm} mm accuracy, bisection used {binary.evaluations} "
        f"evaluations and the grid used {grid.evaluations}"
    )


def test_grid_hit_is_the_smallest_counterexample_on_its_lattice() -> None:
    """A coarse lattice must not report a larger counterexample than it must."""
    space = _space()
    outcome = GridSearcher(space, steps_per_axis=8, budget=64).search(
        AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)
    )
    assert outcome.found
    step = DEPTH_MAX_MM / 7
    assert outcome.counterexample.point["depth_noise_mm"] == pytest.approx(
        DEPTH_BOUNDARY_MM, abs=step
    )


def test_random_search_is_seeded_and_reproducible() -> None:
    """An unseeded baseline can be made to look good or bad by rerunning it."""
    space = _space()
    oracle = AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)
    first = RandomSearcher(space, budget=32, seed=7).search(oracle)
    second = RandomSearcher(space, budget=32, seed=7).search(oracle)
    assert first.found == second.found
    if first.found:
        assert first.counterexample.point == second.counterexample.point
        assert first.evaluations == second.evaluations


def test_adaptive_shrinks_an_irrelevant_axis_out() -> None:
    """The coordinate shrink must separate the axis that matters from the rest.

    Only depth noise causes the failure; occlusion is carried along by the ramp.
    The shrink returns occlusion to nominal and the counterexample survives, so
    occlusion must not be listed as necessary. That is a diagnosis assembled
    from experiments rather than from correlation.
    """
    space = _space(axes=2)
    oracle = AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)
    outcome = AdaptiveBoundarySearcher(space, budget=128, coarse_steps=8, tolerance=1e-3).search(
        oracle
    )
    assert outcome.found
    assert outcome.necessary_axes == ("depth_noise_mm",)
    assert outcome.counterexample.point["occlusion_percent"] == 0.0
    assert outcome.counterexample.point["depth_noise_mm"] > 0.0


def test_adaptive_shrinks_an_irrelevant_axis_out_low_is_below_high() -> None:
    """The severity of the shrunk point must be below the ramp point it came from.

    Removing an axis must never make the reported counterexample *harder* than
    the one the bisection found; if it could, the shrink would be a bug that
    only ever shows up as a worse answer.
    """
    space = _space(axes=2)
    oracle = AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)
    outcome = AdaptiveBoundarySearcher(space, budget=128, coarse_steps=8).search(oracle)
    assert outcome.found
    assert outcome.severity < space.severity(
        ramp_point(space, outcome.bracket[1])
    ) + 1e-9


def test_adaptive_flags_a_non_monotone_response() -> None:
    """A failure that goes away again must be reported, not silently bracketed."""

    class WindowOracle(AnalyticRegressionOracle):
        """The candidate fails only inside a window of severity."""

        def evaluate(self, point: dict[str, float]) -> Trial:
            self.calls += 1
            scale = self._scale(point)
            fails = 0.25 <= scale < 0.75
            return Trial(
                point=dict(point), baseline_success=True, candidate_success=not fails
            )

    space = _space()
    outcome = AdaptiveBoundarySearcher(space, budget=64, coarse_steps=8).search(WindowOracle(space))
    assert outcome.found
    assert "non-monotone" in outcome.notes


DISCOVERY_SEED = 42


def _request(space: SearchSpace | None = None) -> DiscoveryRequest:
    return DiscoveryRequest(
        experiment_id="ce_test",
        base_scenario=ScenarioConfig(scenario_id="ce_base", seed=DISCOVERY_SEED),
        space=space or _space(),
        baseline_policy="policy_v1",
        candidate_policy="policy_v2",
        seed=DISCOVERY_SEED,
        backend="mock",
        git_commit="deadbeef",
    )


def test_discover_builds_a_portable_artifact() -> None:
    """The artifact must record the condition, the provenance and the boundary."""
    request = _request()
    oracle = AnalyticRegressionOracle(
        request.space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM
    )
    result = discover(request, oracle, BinaryBoundarySearcher(request.space, budget=64, tolerance=0.1))
    assert result.found and result.counterexample is not None
    artifact = result.counterexample
    assert artifact.baseline_policy == "policy_v1"
    assert artifact.candidate_policy == "policy_v2"
    assert artifact.seed == DISCOVERY_SEED
    assert artifact.task == request.base_scenario.task.name.value
    assert artifact.boundary == pytest.approx(DEPTH_BOUNDARY_MM, abs=0.5)
    assert artifact.boundary_axis == "depth_noise_mm"
    assert artifact.perturbation_paths["depth_noise_mm"] == "sensors.depth.sigma_mm"
    # The recorded hash must describe the *perturbed* scenario, not the base, or
    # a replay could be validated against conditions it never ran under.
    from robot_ci.counterexamples import perturbed_scenario

    perturbed = perturbed_scenario(request.base_scenario, request.space, artifact.perturbation)
    assert artifact.scenario_config_hash == perturbed.config_hash()
    assert artifact.scenario_config_hash != request.base_scenario.config_hash()


def test_artifact_round_trips_through_yaml(tmp_path) -> None:
    """A counterexample that cannot survive serialization is not portable."""
    request = _request()
    result = discover(
        request,
        AnalyticRegressionOracle(request.space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM),
        BinaryBoundarySearcher(request.space, budget=64, tolerance=0.1),
    )
    artifact = result.counterexample
    assert artifact is not None
    path = save_counterexample(artifact, tmp_path / "counterexample.yaml")
    assert path.exists()
    restored = load_counterexample(path)
    assert restored.to_dict() == artifact.to_dict()


def test_artifact_rejects_a_non_counterexample() -> None:
    """The type must not represent 'both succeeded' as a counterexample."""
    with pytest.raises(ValueError, match="baseline succeeding"):
        Counterexample(
            counterexample_id="x",
            experiment_id="e",
            base_scenario_id="s",
            scenario_config_hash="sha256:0",
            task="pick_and_place",
            seed=1,
            baseline_policy="a",
            candidate_policy="b",
            baseline_success=True,
            candidate_success=True,
        )


def test_artifact_rejects_an_unknown_layout() -> None:
    """A future artifact must be refused rather than silently misread."""
    with pytest.raises(ValueError, match="incompatible"):
        Counterexample.from_dict({"artifact_version": "9.0.0", "counterexample_id": "x"})


def test_reproduction_rate_reflects_seed_dependence() -> None:
    """RQ4: does the failure belong to the condition, or to one seed?

    The oracle here is built so only the discovery seed still shows the
    regression. A reproduction rate of 1/3 is the honest answer, and reporting
    it is the difference between a counterexample and a coincidence.
    """
    request = _request()
    space = request.space
    result = discover(
        request,
        AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM),
        BinaryBoundarySearcher(space, budget=64, tolerance=0.1),
    )
    artifact = result.counterexample
    assert artifact is not None

    def make_oracle(seed: int) -> OutcomeOracle:
        boundary = DEPTH_BOUNDARY_MM / DEPTH_MAX_MM if seed == DISCOVERY_SEED else 2.0
        return AnalyticRegressionOracle(space, boundary=boundary)

    report = reproduce(artifact, make_oracle, seeds=(DISCOVERY_SEED, 43, 44))
    assert report.attempts == 3
    assert report.reproduced == 1
    assert report.rate == pytest.approx(1 / 3)
    assert report.consistent is False, "the seeds disagreed, so the flag must say so"
    assert [t["seed"] for t in report.trials] == [DISCOVERY_SEED, 43, 44]


def test_replay_at_the_original_seed_reproduces_exactly() -> None:
    """A deterministic oracle must reproduce its own artifact every time."""
    request = _request()
    space = request.space
    result = discover(
        request,
        AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM),
        AdaptiveBoundarySearcher(space, budget=128, coarse_steps=8),
    )
    artifact = result.counterexample
    assert artifact is not None

    def make_oracle(_seed: int) -> OutcomeOracle:
        """Pure replay: the caller's factory ignores the seed on purpose."""
        return AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)

    report = reproduce(artifact, make_oracle)
    assert report.attempts == 1
    assert report.rate == 1.0
    assert report.consistent is True


def test_caching_oracle_does_not_charge_twice_for_one_point() -> None:
    """A repeat evaluation must be free, or budget accounting is a lie."""
    space = _space()
    inner = AnalyticRegressionOracle(space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM)
    oracle = CachingOracle(inner)
    point = {"depth_noise_mm": 20.0}
    first = oracle.evaluate(point)
    second = oracle.evaluate(point)
    assert first == second
    assert inner.calls == 1, "the inner oracle was asked twice for the same point"
    assert oracle.unique_evaluations == 1
    assert oracle.hits == 1
    assert oracle.evaluations == 2


def test_artifact_carries_everything_a_replay_needs(tmp_path) -> None:
    """A counterexample must travel with its scenario *and* its axis bounds.

    Without the base scenario a replaying machine has no conditions to apply
    the perturbation to; without the bounds it cannot compute severity, so the
    artifact could not be compared with a smaller one found elsewhere. Both are
    therefore embedded and asserted to survive a YAML round trip.
    """
    request = _request(_space(axes=2))
    result = discover(
        request,
        AnalyticRegressionOracle(request.space, boundary=DEPTH_BOUNDARY_MM / DEPTH_MAX_MM),
        AdaptiveBoundarySearcher(request.space, budget=128, coarse_steps=8),
    )
    artifact = result.counterexample
    assert artifact is not None

    restored = load_counterexample(save_counterexample(artifact, tmp_path / "ce.yaml"))

    # The scenario survives, unchanged.
    assert restored.base_scenario().config_hash() == request.base_scenario.config_hash()
    assert restored.base_scenario().scenario_id == "ce_base"
    # The axis bounds survive, so severity is computable on any machine.
    space = restored.space()
    assert len(space) == 2
    assert space.axis("depth_noise_mm").upper == DEPTH_MAX_MM
    assert space.axis("occlusion_percent").unit == "%"
    assert space.severity(restored.perturbation) == pytest.approx(artifact.severity)
    # And the stored severity must be the one the axes actually imply.
    assert artifact.severity < 1.0, "the shrink should have lowered the severity"
