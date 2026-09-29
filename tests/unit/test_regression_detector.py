"""Statistics and regression detection share no simulator dependency."""

from __future__ import annotations

from robot_ci.experiments.runner import EpisodeResult
from robot_ci.regression import RegressionConfig, detect_regressions
from robot_ci.statistics import proportion_ci


def _episode(policy: str, scenario: str, seed: int, success: bool) -> EpisodeResult:
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
        conditions={"scenario_id": scenario},
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
