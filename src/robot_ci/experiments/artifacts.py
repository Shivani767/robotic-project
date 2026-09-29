"""Artifact writing.

Every experiment produces a self-describing directory:

.. code-block:: text

    results/<experiment_id>/
        config.yaml          the exact configuration, as executed
        summary.json          aggregate results and per-run records
        metrics.csv           one row per (run, metric)
        events.jsonl          the research log, if one was passed
        counterexamples.json  empty until Phase 7
        report.html           human-readable summary

The point of writing *all* of these is that a result must be auditable without
re-running the experiment: the config that produced it, the conditions, the
individual measurements, and the commit it ran at.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from robot_ci.core.io import write_json, write_yaml
from robot_ci.core.logging import JsonlLogger, get_logger
from robot_ci.core.version import ENVIRONMENT_VERSION, PACKAGE_NAME, SCHEMA_VERSION, VERSION
from robot_ci.experiments.orchestrator import ExperimentOutcome

__all__ = ["metrics_rows", "render_html_report", "write_artifacts"]

_LOG = get_logger(__name__)

#: Metrics whose name does not match anything special; all are written.
_CSV_FIELDS = (
    "run_id", "experiment_id", "scenario_id", "policy", "seed", "config_hash",
    "metric", "value", "unit", "kind", "source", "uncertainty", "n",
)


def metrics_rows(outcome: ExperimentOutcome) -> list[dict[str, Any]]:
    """Flatten every measurement into CSV rows.

    One row per (run, metric) so the file is a tidy long table that a notebook
    or a spreadsheet can pivot without parsing JSON.
    """
    rows: list[dict[str, Any]] = []
    for result in outcome.results:
        for metric in result.metrics:
            rows.append(
                {
                    "run_id": result.run_id,
                    "experiment_id": result.experiment_id,
                    "scenario_id": result.scenario_id,
                    "policy": result.policy,
                    "seed": result.seed,
                    "config_hash": result.config_hash,
                    **metric.to_csv_row(),
                }
            )
    return rows


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format a value for the HTML report, handling NaN and missing honestly.

    A metric that was not measured renders as ``n/a``. Rendering it as ``0`` would
    be a fabricated result, so the distinction is preserved all the way to the
    HTML.
    """
    if value is None or value != value:  # None or NaN
        return "n/a"
    return f"{value:.{digits}f}"


def _delta(left: float | None, right: float | None) -> float | None:
    """Return ``right - left``, or ``None`` when either side is missing."""
    if left is None or right is None:
        return None
    return right - left


def render_html_report(outcome: ExperimentOutcome) -> str:
    """Render a standalone HTML summary.

    This is a *report of the run that happened*. It contains no placeholder
    text and no illustrative numbers: if a metric was not measured, the table
    says ``n/a`` rather than showing a zero.
    """
    baseline = outcome.summaries.get(outcome.config.baseline.name)
    candidate = outcome.summaries.get(outcome.config.candidate.name)

    metric_names = sorted(
        set(baseline.per_metric if baseline else {}) | set(candidate.per_metric if candidate else {})
    )
    def _row(name: str) -> str:
        left = baseline.per_metric.get(name) if baseline else None
        right = candidate.per_metric.get(name) if candidate else None
        return (
            f"<tr><td>{name}</td><td>{_fmt(left)}</td><td>{_fmt(right)}</td>"
            f"<td>{_fmt(_delta(left, right))}</td><td>{len(outcome.results)}</td></tr>"
        )

    rows = "\n".join(_row(name) for name in metric_names)

    statuses = sorted({r.status for r in outcome.results})
    status_rows = "\n".join(
        f"<tr><td>{s}</td>"
        f"<td>{sum(1 for r in outcome.results if r.policy == outcome.config.baseline.name and r.status == s)}</td>"
        f"<td>{sum(1 for r in outcome.results if r.policy == outcome.config.candidate.name and r.status == s)}</td></tr>"
        for s in statuses
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{outcome.config.experiment_id} - ROBOT CI report</title>
<style>
 body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
 .meta {{ color: #555; font-size: 0.85rem; margin-bottom: 1.5rem; }}
 table {{ border-collapse: collapse; margin-bottom: 2rem; min-width: 640px; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 th {{ background: #f5f5f5; }}
 .note {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: 0.8rem 1rem; margin: 1rem 0; }}
 code {{ background: #f5f5f5; padding: 1px 4px; }}
</style></head><body>
<h1>ROBOT CI experiment report</h1>
<div class="meta">
 <div><b>experiment</b> <code>{outcome.config.experiment_id}</code></div>
 <div><b>baseline</b> {outcome.config.baseline.name} &nbsp;&rarr; <b>candidate</b> {outcome.config.candidate.name}</div>
 <div><b>backend</b> {outcome.backend} &nbsp; <b>git</b> <code>{outcome.git_commit}</code></div>
 <div><b>config_hash</b> <code>{outcome.config.config_hash()}</code></div>
 <div><b>episodes</b> {outcome.n_episodes} of {outcome.config.total_episodes_both_policies} expected
      &nbsp; <b>grid</b> {outcome.config.grid_size} cells &times; {outcome.config.repetitions} repetitions</div>
 <div><b>wall time</b> {outcome.wall_time_s:.1f} s &nbsp; <b>{PACKAGE_NAME}</b> {VERSION} (schema {SCHEMA_VERSION})</div>
</div>
{"<div class='note'><b>Truncated run.</b> This report covers fewer episodes than the configuration specifies, so its rates are not comparable to a full run.</div>" if outcome.n_episodes < outcome.config.total_episodes_both_policies else ""}
<h2>Success</h2>
<table><tr><th>policy</th><th>runs</th><th>success rate</th></tr>
<tr><td>{outcome.config.baseline.name}</td><td>{baseline.n_runs if baseline else 0}</td>
    <td>{_fmt(baseline.success_rate, 4) if baseline else 'n/a'}</td></tr>
<tr><td>{outcome.config.candidate.name}</td><td>{candidate.n_runs if candidate else 0}</td>
    <td>{_fmt(candidate.success_rate, 4) if candidate else 'n/a'}</td></tr>
</table>
<h2>Terminal statuses</h2>
<table><tr><th>status</th><th>{outcome.config.baseline.name}</th><th>{outcome.config.candidate.name}</th></tr>
{status_rows}
</table>
<h2>Metrics (mean over runs)</h2>
<table><tr><th>metric</th><th>{outcome.config.baseline.name}</th><th>{outcome.config.candidate.name}</th>
<th>delta</th><th>runs</th></tr>
{rows}
</table>
<p><small>Environment {ENVIRONMENT_VERSION}. Metrics without an interval are descriptive
means, not statistical evidence; interval estimation and conditional analysis arrive in
Phase 6.</small></p>
</body></html>
"""


def write_artifacts(
    outcome: ExperimentOutcome,
    output_dir: Path,
    *,
    log: JsonlLogger | None = None,
) -> dict[str, Path]:
    """Write the full artifact set for one experiment.

    Args:
        outcome: The experiment result.
        output_dir: Directory to write into; created if absent.
        log: Optional JSONL logger, bound to the experiment and closed here so
            its ``events.jsonl`` is flushed and complete.

    Returns:
        Mapping of artifact name to the path written.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    config_path = write_yaml(directory / "config.yaml", outcome.config.model_dump(mode="json"))
    written["config"] = config_path

    summary = write_json(directory / "summary.json", outcome.to_dict())
    written["summary"] = summary

    rows = metrics_rows(outcome)
    csv_path = directory / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _CSV_FIELDS})
    written["metrics"] = csv_path

    # Counterexamples arrive in Phase 7; written now as an empty, typed file so
    # downstream tooling can rely on its presence rather than special-casing.
    written["counterexamples"] = write_json(
        directory / "counterexamples.json", {"experiment_id": outcome.config.experiment_id, "items": []}
    )

    report_path = directory / "report.html"
    report_path.write_text(render_html_report(outcome), encoding="utf-8")
    written["report"] = report_path

    if log is not None:
        log.close()
        written["events"] = directory / "events.jsonl"

    _LOG.info("wrote %d artifacts to %s", len(written), directory)
    return written
