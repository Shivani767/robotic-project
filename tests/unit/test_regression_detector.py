"""Statistics and regression detection share no simulator dependency."""

from __future__ import annotations

from robot_ci.experiments.runner import EpisodeResult
from robot_ci.regression import RegressionConfig, detect_regressions
from robot_ci.statistics import proportion_ci


def _episode(
    policy: str,
    scenario: str,
    seed: int,
    success: bool,
    extra: dict[str, object] | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        run_id=f"{scenario}__{policy}__s{seed}",
        experiment_id="test",
        scenario_id=scenario,
        policy=policy,
        seed=seed,
        config_hash="sha256:test",
        status="success" if success else "failure",
        success=success,
        steps=10,
        conditions={"scenario_id": scenario, **(extra or {})},
    )


def test_wilson_interval_covers_at_boundaries() -> None:
    empty = proportion_ci(0, 0)
    assert empty.lower == 0.0 and empty.upper == 1.0
    perfect = proportion_ci(10, 10)
    assert perfect.rate == 1.0 and perfect.lower < 1.0
    zero = proportion_ci(0, 10)
    assert zero.rate == 0.0 and zero.upper > 0.0


def test_conditional_finding_when_candidate_collapses_in_one_cell() -> None:
    results: list[EpisodeResult] = []
    for seed in range(5):
        results.append(_episode("v1", "nominal", seed, True))
        results.append(_episode("v2", "nominal", seed, True))
    for seed in range(5, 10):
        results.append(_episode("v1", "noisy", seed, True))
        results.append(_episode("v2", "noisy", seed, False))
    report = detect_regressions(
        results, baseline_name="v1", candidate_name="v2", config=RegressionConfig(min_samples=3)
    )
    assert report.global_delta < 0
    assert len(report.findings) == 1
    assert report.findings[0].condition == "noisy"
    assert report.findings[0].severity == "blocking"
    assert report.findings[0].delta == -1.0


def test_global_improvement_does_not_hide_conditional_regression() -> None:
    results: list[EpisodeResult] = []
    for seed in range(40):
        results.append(_episode("v1", "easy", seed, seed < 32))
        results.append(_episode("v2", "easy", seed, True))
    for seed in range(40, 45):
        results.append(_episode("v1", "hard", seed, True))
        results.append(_episode("v2", "hard", seed, False))
    report = detect_regressions(
        results, baseline_name="v1", candidate_name="v2", config=RegressionConfig(min_samples=3)
    )
    assert report.global_delta > 0
    assert any(f.condition == "hard" for f in report.findings)


def test_small_samples_do_not_produce_findings() -> None:
    results = [
        _episode("v1", "rare", 1, True),
        _episode("v2", "rare", 1, False),
    ]
    report = detect_regressions(
        results, baseline_name="v1", candidate_name="v2", config=RegressionConfig(min_samples=3)
    )
    assert report.findings == []
    assert len(report.cells) == 1


def test_regression_module_imports_no_simulator() -> None:
    import subprocess
    import sys

    code = "import robot_ci.regression, robot_ci.statistics, robot_ci.sensors; print('ok')"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_a_sweep_that_keeps_one_scenario_id_still_splits_into_cells() -> None:
    """Conditional analysis must condition on values, not on the scenario name.

    Grouping cells by ``scenario_id`` is right only for a grid expanded into
    distinct scenario files, where each cell gets its own id. Every in-place
    sweep -- the counterexample search, ``robot-ci diagnose``, the RQ2 harness --
    perturbs a scenario without renaming it, and then all of its cells collapse
    into one.

    Measured on a five-cell depth-noise grid, that produced a single cell
    holding all 40 episodes, one "regression" of -0.125 reported at *nominal*,
    and no mention that the worst cell was -0.25. A conditional analysis that
    conditions on nothing returns an aggregate result under a name promising
    otherwise, which is the most dangerous failure mode this project has.
    """
    results: list[EpisodeResult] = []
    for sigma, base_ok, cand_ok in ((0.0, True, True), (2.0, True, False), (4.0, False, False)):
        for seed in range(6):
            results.append(
                _episode("v1", "nominal", seed, base_ok, extra={"depth_noise_mm": sigma})
            )
            results.append(
                _episode("v2", "nominal", seed, cand_ok, extra={"depth_noise_mm": sigma})
            )
    report = detect_regressions(
        results, baseline_name="v1", candidate_name="v2", config=RegressionConfig(min_samples=3)
    )
    assert len(report.cells) == 3, f"expected 3 cells, got {len(report.cells)}"
    # sigma 0: both succeed. sigma 2: baseline all succeed, candidate all fail,
    # so -1.0. sigma 4: both fail. The aggregate over the three is -1/3, which is
    # the dilution this test exists to catch.
    deltas = sorted(cell.delta for cell in report.cells)
    assert deltas == [-1.0, 0.0, 0.0], f"per-cell deltas were {deltas}"
    # The nominal cell must not be the one carrying the regression.
    nominal = next(c for c in report.cells if c.conditions.get("depth_noise_mm") == 0.0)
    assert nominal.delta == 0.0
    assert nominal.n_baseline == 6
    # And the worst cell must be reported at full size, not averaged away.
    assert any(f.delta == -1.0 for f in report.findings)
    assert report.global_delta < 0
    assert abs(report.global_delta) < abs(min(deltas)), (
        "the point of the test: the aggregate must be milder than the cell"
    )
