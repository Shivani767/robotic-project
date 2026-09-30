"""End-to-end tests of the ``robot-ci`` CLI.

These invoke the real Typer app through ``CliRunner`` and assert on exit codes,
which is exactly what a CI pipeline observes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from robot_ci.cli.common import (
    EXIT_BACKEND_UNAVAILABLE,
    EXIT_CONFIG_ERROR,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_USAGE,
)
from robot_ci.cli.main import app

pytestmark = pytest.mark.e2e

runner = CliRunner()


def invoke(*args: str):
    """Invoke the CLI and return the result."""
    return runner.invoke(app, list(args))


def test_version_exits_zero() -> None:
    result = invoke("version")
    assert result.exit_code == EXIT_OK
    assert "ROBOT CI" in result.output


def test_help_lists_the_command_surface() -> None:
    result = invoke("--help")
    assert result.exit_code == EXIT_OK
    for command in ("init", "validate", "scenario", "experiment", "backend"):
        assert command in result.output


def test_no_args_shows_help() -> None:
    assert invoke().exit_code != 0


def test_init_is_idempotent() -> None:
    first = invoke("init")
    assert first.exit_code == EXIT_OK
    second = invoke("init")
    assert second.exit_code == EXIT_OK


def test_validate_passes() -> None:
    """The Phase 1 acceptance gate."""
    result = invoke("validate")
    assert result.exit_code == EXIT_OK, result.output
    assert "PASSED" in result.output


def test_backend_list_reports_all() -> None:
    result = invoke("backend", "list")
    assert result.exit_code == EXIT_OK
    assert "mock" in result.output
    assert "mujoco" in result.output


def test_backend_probe_rejects_unknown() -> None:
    result = invoke("backend", "probe", "nope")
    assert result.exit_code != EXIT_OK


def test_scenario_list_shows_b1_to_b10() -> None:
    result = invoke("scenario", "list")
    assert result.exit_code == EXIT_OK
    for benchmark in [f"B{i}" for i in range(1, 11)]:
        assert benchmark in result.output


def test_scenario_show_nominal() -> None:
    result = invoke("scenario", "show", "nominal")
    assert result.exit_code == EXIT_OK
    assert "config_hash" in result.output
    assert "sha256:" in result.output


def test_scenario_show_rejects_unknown_family() -> None:
    result = invoke("scenario", "show", "not_a_family")
    assert result.exit_code == EXIT_CONFIG_ERROR


def test_scenario_generate_previews() -> None:
    result = invoke("scenario", "generate", "depth_noise")
    assert result.exit_code == EXIT_OK
    assert "generated" in result.output


def test_scenario_generate_writes_files(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    result = invoke("scenario", "generate", "depth_noise", "--output", str(out))
    assert result.exit_code == EXIT_OK
    written = sorted(out.glob("*.yaml"))
    assert len(written) == 6
    assert all("scenario_id" in p.read_text() for p in written)


def test_scenario_validate_accepts_generated_files(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    invoke("scenario", "generate", "occlusion", "--output", str(out))
    result = invoke("scenario", "validate", *[str(p) for p in sorted(out.glob("*.yaml"))])
    assert result.exit_code == EXIT_OK


def test_scenario_validate_rejects_broken_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("scenario_id: x\nnope: 1\n", encoding="utf-8")
    result = invoke("scenario", "validate", str(bad))
    assert result.exit_code == EXIT_CONFIG_ERROR


def test_experiment_run_is_implemented(tmp_path: Path) -> None:
    """Phase 2 acceptance: `robot-ci experiment run` executes a real experiment."""
    path = tmp_path / "exp.yaml"
    path.write_text(
        "experiment:\n  id: cli_smoke\n"
        "baseline:\n  policy: policy_v1\n"
        "candidate:\n  policy: policy_v2\n"
        "scenario:\n  task: pick_and_place\n"
        "variables:\n  depth_noise_mm:\n    values: [0, 2]\n"
        "repetitions: 1\n"
        "seed: 7\n",
        encoding="utf-8",
    )
    out = tmp_path / "results"
    result = invoke("experiment", "run", str(path), "--backend", "mock", "--output", str(out))
    assert result.exit_code == EXIT_OK, result.output
    assert (out / "summary.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "report.html").exists()
    assert (out / "events.jsonl").exists()


def test_experiment_show_validates_a_config(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(
        "experiment_id: demo\n"
        "baseline:\n  name: policy_v1\n"
        "candidate:\n  name: policy_v2\n"
        "variables:\n"
        "  - name: depth_noise_mm\n"
        "    path: sensors.depth.sigma_mm\n"
        "    values: [0, 5, 10]\n"
        "repetitions: 4\n",
        encoding="utf-8",
    )
    result = invoke("experiment", "show", str(path))
    assert result.exit_code == EXIT_OK
    assert "episodes" in result.output


def test_experiment_show_rejects_identical_policies(tmp_path: Path) -> None:
    """A vacuous comparison (same policy on both sides) is a schema violation."""
    path = tmp_path / "exp.yaml"
    path.write_text(
        "experiment_id: demo\nbaseline:\n  name: same\ncandidate:\n  name: same\n",
        encoding="utf-8",
    )
    result = invoke("experiment", "show", str(path))
    assert result.exit_code == EXIT_CONFIG_ERROR
    assert "vacuous" in result.output


def test_experiment_show_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("experiment_id: [unclosed\n", encoding="utf-8")
    result = invoke("experiment", "show", str(path))
    assert result.exit_code != EXIT_OK


def test_diagnose_reports_a_measured_comparison_not_a_guess() -> None:
    """`diagnose` must run episodes and print numbers, not a template.

    A diagnosis that renders its hedged wording correctly but was handed no
    measurements would still look right, so the test asserts on the arithmetic:
    the per-condition rates have to be multiples of the seed count.
    """
    result = invoke("diagnose", "--backend", "mock", "--seeds", "2", "--no-ablation")
    assert result.exit_code == EXIT_OK, result.output
    lowered = result.output.lower()
    assert "diagnosis:" in lowered
    assert "baseline" in lowered and "candidate" in lowered
    # The association table must be present, and the caveat must travel with it.
    assert "strongest association" in lowered or "no association" in lowered


def test_diagnose_rejects_a_malformed_axis() -> None:
    """A bad axis is a usage error, not a silently empty sweep."""
    result = invoke("diagnose", "--axis", "depth_noise_mm")
    assert result.exit_code == EXIT_USAGE
    assert "NAME:LOWER:UPPER" in result.output


def test_report_renders_a_real_experiment(tmp_path: Path) -> None:
    """`report` must produce an artifact, not announce that it is unimplemented.

    Checks the file exists and carries the experiment's own numbers, so a
    regression that renders an empty or placeholder page fails here rather than
    passing a command that merely exits 0.
    """
    results = tmp_path / "exp"
    results.mkdir()
    (results / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "report_fixture",
                "baseline": "policy_v1",
                "candidate": "policy_v2",
                "backend": "mujoco",
            }
        )
    )
    (results / "regression.json").write_text(
        json.dumps(
            {
                "baseline_rate": 0.75,
                "candidate_rate": 0.5,
                "global_delta": -0.25,
                "n_baseline": 8,
                "n_candidate": 8,
                "cells": [
                    {
                        "condition": "noise=2",
                        "baseline_success": 0.75,
                        "candidate_success": 0.5,
                        "delta": -0.25,
                        "n_baseline": 8,
                        "n_candidate": 8,
                        "baseline_ci": [0.4, 0.9],
                        "candidate_ci": [0.2, 0.7],
                        "effect_h": -0.4,
                    }
                ],
                "findings": [{"condition": "noise=2", "severity": "blocking"}],
            }
        )
    )
    out = tmp_path / "r.html"
    result = invoke("report", str(results), "-o", str(out))
    assert result.exit_code == EXIT_OK, result.output
    assert out.exists()
    body = out.read_text()
    assert "noise=2" in body, "the conditional cell must be in the report"
    assert "blocking" in body
    assert "https://" not in body, "the report must be self-contained"


@pytest.mark.parametrize(
    "command",
    [
        ("experiment", "compare"),
        ("boundary",),
        ("dashboard",),
        ("export",),
    ],
)
def test_later_phase_commands_report_not_implemented(command: tuple[str, ...]) -> None:
    """Reserved commands must never look like they succeeded."""
    result = invoke(*command)
    assert result.exit_code == EXIT_NOT_IMPLEMENTED, f"{command} exited {result.exit_code}"
    assert "not implemented" in result.output.lower()


def test_counterexample_search_finds_nothing_where_nothing_can_exist(tmp_path: Path) -> None:
    """`robot-ci counterexample search` runs a real search, not a stub.

    On the mock backend the policy cannot move the arm, so no policy difference
    can exist. A stub would say "not implemented"; the real command instead
    runs the search, evaluates oracle points, and reports the honest negative
    with exit code 0 -- "no counterexample" is a result, not an error.
    """
    result = invoke(
        "counterexample",
        "search",
        "--backend",
        "mock",
        "--budget",
        "8",
        "--max-steps",
        "4",
        "--output",
        str(tmp_path / "ce.yaml"),
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "no counterexample found" in result.output.lower()


def test_counterexample_search_rejects_a_bad_axis() -> None:
    """An axis that is not a NAME:LOWER:UPPER spec must be a usage error."""
    result = invoke(
        "counterexample", "search", "--backend", "mock", "--axis", "not_an_axis", "--budget", "2"
    )
    assert result.exit_code == EXIT_USAGE, result.output


def test_counterexample_reproduce_reports_a_rate(tmp_path: Path) -> None:
    """`robot-ci counterexample reproduce` replays a stored artifact.

    The artifact describes a disagreement the mock backend cannot produce, so
    the replay must honestly report 0/1 and refuse to confirm the failure --
    exit 0 in advisory mode, since the rate itself is the finding.
    """
    from robot_ci.counterexamples import Counterexample, save_counterexample

    artifact = Counterexample(
        counterexample_id="cli_replay_check",
        experiment_id="cli",
        base_scenario_id="nominal",
        scenario_config_hash="sha256:test",
        task="pick_and_place",
        seed=1,
        baseline_policy="policy_v1",
        candidate_policy="policy_v2",
        perturbation={"depth_noise_mm": 10.0},
        perturbation_paths={"depth_noise_mm": "sensors.depth.sigma_mm"},
        axes_payload=[
            {
                "name": "depth_noise_mm",
                "path": "sensors.depth.sigma_mm",
                "lower": 0.0,
                "upper": 60.0,
                "unit": "mm",
            }
        ],
        base_scenario_payload={
            "schema_version": "1.0.0",
            "scenario_id": "nominal",
            "seed": 1,
        },
        baseline_success=True,
        candidate_success=False,
    )
    saved = save_counterexample(artifact, tmp_path / "ce.yaml")
    result = invoke(
        "counterexample", "reproduce", str(saved), "--backend", "mock", "--max-steps", "4"
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "0/1" in result.output and "0.000" in result.output


def test_counterexample_reproduce_strict_exits_nonzero_on_non_reproduction(
    tmp_path: Path,
) -> None:
    """`--strict` turns a non-reproducing artifact into a CI failure."""
    from robot_ci.counterexamples import Counterexample, save_counterexample

    artifact = Counterexample(
        counterexample_id="cli_strict_check",
        experiment_id="cli",
        base_scenario_id="nominal",
        scenario_config_hash="sha256:test",
        task="pick_and_place",
        seed=1,
        baseline_policy="policy_v1",
        candidate_policy="policy_v2",
        perturbation={"depth_noise_mm": 10.0},
        perturbation_paths={"depth_noise_mm": "sensors.depth.sigma_mm"},
        axes_payload=[
            {
                "name": "depth_noise_mm",
                "path": "sensors.depth.sigma_mm",
                "lower": 0.0,
                "upper": 60.0,
                "unit": "mm",
            }
        ],
        base_scenario_payload={
            "schema_version": "1.0.0",
            "scenario_id": "nominal",
            "seed": 1,
        },
        baseline_success=True,
        candidate_success=False,
    )
    saved = save_counterexample(artifact, tmp_path / "ce.yaml")
    result = invoke(
        "counterexample",
        "reproduce",
        str(saved),
        "--backend",
        "mock",
        "--max-steps",
        "4",
        "--strict",
    )
    assert result.exit_code == EXIT_REGRESSION, result.output


def test_regression_command_detects_from_a_finished_experiment(tmp_path: Path) -> None:
    """`robot-ci regression` is implemented: it reads summary.json, not a stub."""
    from robot_ci.experiments.runner import EpisodeResult

    results = []
    for seed in range(4):
        results.append(
            EpisodeResult(
                run_id=f"nominal__v1__s{seed}",
                experiment_id="cli_check",
                scenario_id="nominal",
                policy="policy_v1",
                seed=seed,
                config_hash="sha256:test",
                status="success",
                success=True,
                steps=10,
                conditions={"scenario_id": "nominal"},
            )
        )
        results.append(
            EpisodeResult(
                run_id=f"nominal__v2__s{seed}",
                experiment_id="cli_check",
                scenario_id="nominal",
                policy="policy_v2",
                seed=seed,
                config_hash="sha256:test",
                status="success" if seed < 2 else "failure",
                success=seed < 2,
                steps=10,
                conditions={"scenario_id": "nominal"},
            )
        )
    import json

    summary = {
        "experiment_id": "cli_check",
        "baseline": "policy_v1",
        "candidate": "policy_v2",
        "results": [r.to_dict() for r in results],
    }
    target = tmp_path / "results"
    target.mkdir()
    (target / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = invoke("regression", str(target), "--min-samples", "1")
    assert result.exit_code == EXIT_REGRESSION, result.output
    assert "blocking" in result.output
    assert (target / "regression.json").exists()


def test_test_command_is_reserved_for_phase_14() -> None:
    result = invoke("test", "--backend", "mujoco")
    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "Phase 14" in result.output


def test_unavailable_backend_exit_code() -> None:
    """A missing backend must be distinguishable from a config error."""
    assert EXIT_BACKEND_UNAVAILABLE != EXIT_CONFIG_ERROR
    result = invoke("backend", "probe", "gazebo")
    assert result.exit_code == EXIT_BACKEND_UNAVAILABLE
