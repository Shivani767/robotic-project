"""Tests for HTML reporting.

The report's job is to make the conditional structure visible, so the tests
check that it does -- and specifically that it does not fall back to the
aggregate, which is the failure this module exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.reporting import ReportInputs, render_from_directory, render_report, write_report


def _regression(*, global_delta: float, cells: list[dict], findings: list[dict]) -> dict:
    return {
        "baseline_rate": 0.7,
        "candidate_rate": 0.7 + global_delta,
        "global_delta": global_delta,
        "n_baseline": 40,
        "n_candidate": 40,
        "cells": cells,
        "findings": findings,
    }


def _cell(name: str, delta: float) -> dict:
    return {
        "condition": name,
        "baseline_success": 0.75,
        "candidate_success": 0.75 + delta,
        "delta": delta,
        "n_baseline": 8,
        "n_candidate": 8,
        "baseline_ci": [0.4, 0.9],
        "candidate_ci": [0.2, 0.7],
        "effect_h": -0.4,
    }


def test_the_report_is_self_contained() -> None:
    """A CI artifact that needs a network cannot be read from a build cache."""
    html = render_report(ReportInputs("e", "a", "b"))
    assert "<!doctype html>" in html
    for pattern in ("http://", "https://", "<script", "cdn."):
        assert pattern not in html, f"report references {pattern!r} and would need a network"


def test_every_condition_cell_is_rendered() -> None:
    """The matrix is the reason the report exists; a flat summary is not a report."""
    regression = _regression(
        global_delta=-0.25,
        cells=[
            _cell("noise=0", 0.0),
            _cell("noise=1", -0.38),
            _cell("noise=2", -0.38),
        ],
        findings=[{"condition": "noise=1", "severity": "blocking"}],
    )
    html = render_report(ReportInputs("e", "policy_v1", "policy_v2", regression=regression))
    for name in ("noise=0", "noise=1", "noise=2"):
        assert name in html, f"{name} missing from the conditional table"
    assert "-0.380" in html


def test_rates_carry_intervals_and_sample_counts() -> None:
    """0.50 from 8 samples and 0.50 from 800 are different claims."""
    regression = _regression(global_delta=0.0, cells=[_cell("c", 0.0)], findings=[])
    html = render_report(ReportInputs("e", "a", "b", regression=regression))
    assert "[0.40, 0.90]" in html, "the interval must be shown next to the rate"
    assert ">8<" in html, "the sample count must be shown"


def test_severity_is_labelled_not_only_coloured() -> None:
    """Colour alone is not readable, and a reader cannot check it."""
    regression = _regression(
        global_delta=-0.2,
        cells=[_cell("bad", -0.2)],
        findings=[{"condition": "bad", "severity": "blocking"}],
    )
    html = render_report(ReportInputs("e", "a", "b", regression=regression))
    assert "blocking" in html
    assert 'class="pill blocking"' in html


def test_a_worse_cell_than_the_average_says_is_called_out() -> None:
    """The note must fire when the average understates the worst cell.

    The first version of this test keyed on the global delta being near zero,
    which missed the case that actually matters: -0.25 globally against -0.38
    in the worst cell reads as tolerable and is not.
    """
    regression = _regression(
        global_delta=-0.25,
        cells=[_cell("mild", 0.0), _cell("worst", -0.38)],
        findings=[{"condition": "worst", "severity": "blocking"}],
    )
    html = render_report(ReportInputs("e", "a", "b", regression=regression))
    assert "Read the cells" in html
    assert "0.130" in html, "the note must quantify the understatement"


def test_no_note_when_the_average_is_the_worst_case() -> None:
    """A warning that always fires is noise."""
    regression = _regression(
        global_delta=-0.38, cells=[_cell("only", -0.38)], findings=[]
    )
    html = render_report(ReportInputs("e", "a", "b", regression=regression))
    assert "Read the cells" not in html


def test_a_missing_metric_reads_na_and_not_zero() -> None:
    """An unmeasured collision count and a measured zero are different facts."""
    html = render_report(ReportInputs("e", "a", "b"))
    assert "n/a" in html
    assert "No regression analysis" in html


def test_a_missing_artifact_still_renders() -> None:
    """A failed run must still produce something to look at."""
    html = render_report(ReportInputs("e", "a", "b", regression=None, counterexamples=None))
    assert "<!doctype html>" in html
    assert "No regression analysis" in html


def test_no_counterexamples_is_stated_with_what_to_trust_instead() -> None:
    """"None found" without a search cost is a claim with nothing behind it."""
    html = render_report(
        ReportInputs("e", "a", "b", counterexamples={"items": [], "tests_to_failure": 15})
    )
    assert "None found" in html
    assert "15" in html, "the evaluations the search spent must be shown"


def test_the_arm_names_come_from_the_summary() -> None:
    """A header reading 'policy_v3 -> candidate' misnames the experiment.

    A lookup into a non-existent ``arms`` key ran before the literal strings and
    lost, which is exactly the sort of quiet default that makes a report
    untrustworthy.
    """
    results = Path("/tmp/report_fixture")
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "x",
                "baseline": "policy_v3",
                "candidate": "policy_v1",
                "backend": "mujoco",
            }
        )
    )
    html = render_from_directory(results)
    assert "policy_v3 &rarr; policy_v1" in html
    assert "-> candidate" not in html


def test_write_report_creates_the_file(tmp_path: Path) -> None:
    """Round trip through disk, since that is how the CLI uses it."""
    (tmp_path / "summary.json").write_text(json.dumps({"experiment_id": "y", "baseline": "a", "candidate": "b"}))
    written = write_report(tmp_path)
    assert written.exists()
    assert "<!doctype html>" in written.read_text()
