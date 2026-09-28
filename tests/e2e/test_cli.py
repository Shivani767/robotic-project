"""End-to-end tests of the ``robot-ci`` CLI.

These invoke the real Typer app through ``CliRunner`` and assert on exit codes,
which is exactly what a CI pipeline observes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from robot_ci.cli.common import (
    EXIT_BACKEND_UNAVAILABLE,
    EXIT_CONFIG_ERROR,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
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


@pytest.mark.parametrize(
    "command",
    [
        ("experiment", "run"),
        ("experiment", "compare"),
        ("regression",),
        ("counterexample", "search"),
        ("counterexample", "reproduce"),
        ("diagnosis",),
        ("boundary",),
        ("report",),
        ("dashboard",),
        ("export",),
    ],
)
def test_later_phase_commands_report_not_implemented(command: tuple[str, ...]) -> None:
    """Reserved commands must never look like they succeeded."""
    result = invoke(*command)
    assert result.exit_code == EXIT_NOT_IMPLEMENTED, f"{command} exited {result.exit_code}"
    assert "not implemented" in result.output.lower()


def test_test_command_is_reserved_for_phase_14() -> None:
    result = invoke("test", "--backend", "mujoco")
    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "Phase 14" in result.output


def test_unavailable_backend_exit_code() -> None:
    """A missing backend must be distinguishable from a config error."""
    assert EXIT_BACKEND_UNAVAILABLE != EXIT_CONFIG_ERROR
    result = invoke("backend", "probe", "gazebo")
    assert result.exit_code == EXIT_BACKEND_UNAVAILABLE
