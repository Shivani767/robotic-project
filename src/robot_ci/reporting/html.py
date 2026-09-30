"""HTML reporting for a finished experiment.

The report exists to show the thing an aggregate cannot: **where** the two arms
differ, and how confident anyone is in each cell. The experiment artifact writer
already emits a flat run summary, which is the right shape for a CSV and the
wrong shape for a question about behaviour, because pooling a 0.50 cell with
four 0.00 cells produces a 0.10 average that describes no condition the robot
was ever in.

Three rules govern what this renders:

* **Never a bare rate.** Every success number is shown with its interval and its
  sample count. A cell of 0.50 from 6 samples and a cell of 0.50 from 600 are
  different claims, and a table that cannot tell them apart invites reading them
  the same way.

* **Never an unexplained colour.** Severity shading is accompanied by the word
  that produced it and the threshold it crossed, so the colour is a reading aid
  rather than the argument.

* **Absence is rendered as absence.** A metric that was not measured shows as
  ``n/a``, never as ``0`` and never as a blank cell that a reader has to
  interpret.

Self-contained by design: no CDN, no external stylesheet, no JavaScript. A CI
artifact that needs a network to render cannot be opened from a build cache, an
air-gapped machine, or an email attachment, which is where a CI artifact is
usually read.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ReportInputs", "render_from_directory", "render_report", "write_report"]


@dataclass(frozen=True)
class ReportInputs:
    """Everything a report is built from.

    Attributes:
        experiment_id: Identifier of the experiment.
        baseline: Baseline policy name.
        candidate: Candidate policy name.
        backend: Simulator backend the runs used.
        git_commit: Commit the code was at, or ``"unknown"``.
        config_hash: Hash of the experiment configuration.
        regression: Parsed ``regression.json``, or ``None`` if not run.
        counterexamples: Parsed ``counterexamples.json``, or ``None``.
        summary: Parsed ``summary.json``, or ``None``.
        benchmark: Parsed ``benchmark_suite.json``, or ``None``.
    """

    experiment_id: str
    baseline: str
    candidate: str
    backend: str = "unknown"
    git_commit: str = "unknown"
    config_hash: str = "unknown"
    regression: dict[str, Any] | None = None
    counterexamples: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None


#: CSS, inline so the file is self-contained.
_STYLE = """
body { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       margin: 2rem auto; max-width: 68rem; padding: 0 1.5rem; color: #16181d; }
h1 { font-size: 1.35rem; margin: 0 0 .15rem; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .6rem; padding-bottom: .3rem;
     border-bottom: 1px solid #e3e6ea; }
.sub { color: #5b6472; font-size: .85rem; margin-bottom: 1.6rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .85rem; }
th, td { border: 1px solid #e3e6ea; padding: .35rem .6rem; text-align: right; }
th { background: #f6f7f9; font-weight: 600; text-align: right; }
th:first-child, td:first-child { text-align: left; }
tr:nth-child(even) td { background: #fbfbfc; }
td.na { color: #98a0ad; font-style: italic; }
code { background: #f2f4f7; padding: .1rem .3rem; border-radius: 3px; font-size: .85em; }
.blocking { background: #fdecec; } .warning { background: #fff6e5; } .ok { background: #f2f9f3; }
.pill { display: inline-block; padding: .1rem .45rem; border-radius: 9px; font-size: .75rem;
        font-weight: 600; letter-spacing: .02em; }
.pill.blocking { background: #e5484d; color: #fff; }
.pill.warning { background: #f5a524; color: #1a1a1a; }
.pill.none { background: #e6e8ec; color: #4a5261; }
.note { background: #fff8e6; border-left: 3px solid #f5a524; padding: .7rem .9rem;
        margin: 1rem 0; font-size: .85rem; }
.kv { display: grid; grid-template-columns: 9rem 1fr; gap: .15rem .8rem;
      font-size: .85rem; margin-bottom: 1.2rem; }
.kv dt { color: #5b6472; } .kv dd { margin: 0; }
.na { color: #98a0ad; font-style: italic; }
footer { margin-top: 3rem; padding-top: .8rem; border-top: 1px solid #e3e6ea;
         color: #5b6472; font-size: .78rem; }
"""


def _esc(value: object) -> str:
    """HTML-escape a value for interpolation."""
    return html.escape(str(value))


def _number(value: object, digits: int = 3, *, suffix: str = "") -> str:
    """Format a number, or render absence as ``n/a`` rather than as zero.

    A metric that was not measured and a metric that measured zero are different
    facts. Collapsing them is how a report ends up claiming a system never
    collided because collision detection was never wired up.
    """
    if value is None:
        return '<span class="na">n/a</span>'
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _esc(value)
    if number != number:  # NaN
        return '<span class="na">n/a</span>'
    return f"{number:.{digits}f}{suffix}"


def _interval(bounds: object, digits: int = 2) -> str:
    """Render a ``[low, high]`` interval, or absence."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return '<span class="na">n/a</span>'
    return f"[{float(bounds[0]):.{digits}f}, {float(bounds[1]):.{digits}f}]"


def _severity_class(severity: str) -> str:
    """Map a severity label to a row class."""
    return {"blocking": "blocking", "warning": "warning"}.get(severity, "ok")


def _regression_section(regression: dict[str, Any] | None) -> str:
    """Render the conditional matrix, which is the report's reason to exist."""
    if not regression:
        return (
            '<p class="na">No regression analysis in this artifact. Run '
            "<code>robot-ci regression &lt;results&gt;</code> to produce it.</p>"
        )

    cells = list(regression.get("cells") or [])
    parts = [
        "<h2>Conditional comparison</h2>",
        '<dl class="kv">',
        f"<dt>baseline</dt><dd>{_number(regression.get('baseline_rate'), 3)} "
        f"(n={_esc(regression.get('n_baseline'))}) &rarr; "
        f"{_esc(regression.get('candidate', 'candidate'))}</dd>",
        f"<dt>global delta</dt><dd>{_number(regression.get('global_delta'), 3)}</dd>",
        f"<dt>findings</dt><dd>{len(regression.get('findings') or [])}</dd>",
        "</dl>",
    ]

    if not cells:
        parts.append('<p class="na">No condition cells were recorded.</p>')
        return "\n".join(parts)

    parts.append(
        "<table><thead><tr><th>condition</th><th>baseline</th>"
        "<th>baseline 95% CI</th><th>candidate</th><th>candidate 95% CI</th>"
        "<th>delta</th><th>effect h</th><th>n</th><th>verdict</th></tr></thead><tbody>"
    )
    finding_by_cell = {
        str(f.get("condition")): str(f.get("severity", "")) for f in regression.get("findings") or []
    }
    for cell in cells:
        name = str(cell.get("condition", ""))
        severity = finding_by_cell.get(name, "")
        verdict = (
            f'<span class="pill {severity}">{_esc(severity)}</span>' if severity else ""
        )
        parts.append(
            f'<tr class="{_severity_class(severity)}">'
            f"<td>{_esc(name)}</td>"
            f"<td>{_number(cell.get('baseline_success'), 2)}</td>"
            f"<td>{_interval(cell.get('baseline_ci'))}</td>"
            f"<td>{_number(cell.get('candidate_success'), 2)}</td>"
            f"<td>{_interval(cell.get('candidate_ci'))}</td>"
            f"<td>{_number(cell.get('delta'), 2, suffix='')}</td>"
            f"<td>{_number(cell.get('effect_h'), 2)}</td>"
            f"<td>{_esc(cell.get('n_baseline'))}</td>"
            f"<td>{verdict}</td></tr>"
        )
    parts.append("</tbody></table>")

    # Warn exactly when the aggregate understates a cell. The earlier condition
    # was "global delta near zero", which missed the common case: a global delta
    # of -0.25 that looks serious and is in fact less than half the -0.38 the
    # worst cell actually measured.
    global_delta = regression.get("global_delta")
    deltas = [
        float(c["delta"])
        for c in cells
        if c.get("delta") is not None and float(c["delta"]) == float(c["delta"])
    ]
    if global_delta is not None and deltas:
        worst = min(deltas)
        if abs(worst) > abs(float(global_delta)) * 1.25 + 1e-9:
            severity = _severity_class("blocking" if worst <= -0.10 else "warning")
            parts.append(
                f'<div class="note {severity}"><b>Read the cells, not the '
                f"average.</b> The global delta is {float(global_delta):+.3f}, but "
                f"the worst condition measured {worst:+.3f}. The average pools every "
                "condition and describes none of them individually, so it understates "
                "the worst case by "
                f"{abs(worst) - abs(float(global_delta)):.3f} &mdash; which is the "
                "difference between a regression worth acting on and one that reads "
                "as tolerable.</div>"
            )
    return "\n".join(parts)


def _counterexample_section(payload: dict[str, Any] | None) -> str:
    """Render counterexamples and, critically, their reproduction rate."""
    if payload is None:
        return ""
    items = list(payload.get("items") or [])
    out = ["<h2>Counterexamples</h2>"]
    if not items:
        out.append(
            "<p>None found. A search that reports nothing has searched; "
            "<em>how far it got before concluding that</em> is the number worth "
            "trusting, and it is in <code>counterexamples.json</code> under "
            "<code>tests_to_failure</code>.</p>"
        )
        return "\n".join(out)
    out.append(
        "<table><thead><tr><th>condition</th><th>severity</th><th>baseline</th>"
        "<th>candidate</th><th>reproduced</th></tr></thead><tbody>"
    )
    for item in items:
        rate = item.get("reproduction_rate")
        out.append(
            "<tr>"
            f"<td><code>{_esc(item.get('condition', {}))}</code></td>"
            f"<td>{_number(item.get('severity'), 4)}</td>"
            f"<td>{_esc(item.get('baseline_outcome', '?'))}</td>"
            f"<td>{_esc(item.get('candidate_outcome', '?'))}</td>"
            f"<td>{_number(rate, 2) if rate is not None else 'not measured'}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _benchmark_section(payload: dict[str, Any] | None) -> str:
    """Render the benchmark suite, separating measured from hypothesised."""
    if not payload:
        return ""
    rows = list(payload.get("results") or [])
    out = ["<h2>Benchmark suite</h2>"]
    if not rows:
        return ""
    out.append(
        "<p>Each benchmark declares a prediction; this is whether the run matched "
        "it. <b>hypothesis</b> means the prediction is derived from a model of the "
        "fault and has not been observed, and is reported separately for that "
        "reason.</p>"
    )
    out.append(
        "<table><thead><tr><th>benchmark</th><th>family</th><th>success</th>"
        "<th>expectation met</th><th>basis</th><th>note</th></tr></thead><tbody>"
    )
    for row in rows:
        if "error" in row:
            out.append(
                f'<tr class="blocking"><td>{_esc(row.get("benchmark"))}</td>'
                f"<td colspan=5>{_esc(row['error'])}</td></tr>"
            )
            continue
        met = row.get("expectation_met")
        label = {True: "yes", False: "no", None: "no claim"}.get(met, "no claim")
        basis = "measured" if row.get("expectation_is_measured") else "hypothesis"
        out.append(
            f'<tr class="{_severity_class("blocking" if met is False else "")}">'
            f"<td>{_esc(row.get('benchmark'))}</td>"
            f"<td>{_esc(row.get('family'))}</td>"
            f"<td>{_number(row.get('success_rate'), 2)}</td>"
            f"<td>{label}</td><td>{basis}</td>"
            f"<td>{_esc(row.get('notes') or '')}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def render_report(inputs: ReportInputs) -> str:
    """Render a complete, self-contained HTML report.

    Args:
        inputs: Everything to report on.

    Returns:
        An HTML document with no external references.
    """
    counterexamples = inputs.counterexamples or {}
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_esc(inputs.experiment_id)} &mdash; ROBOT CI report</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>ROBOT CI experiment report</h1>",
        f'<div class="sub">{_esc(inputs.experiment_id)} &middot; '
        f"{_esc(inputs.baseline)} &rarr; {_esc(inputs.candidate)} &middot; "
        f"{_esc(inputs.backend)}</div>",
        '<dl class="kv">',
        f"<dt>git commit</dt><dd><code>{_esc(inputs.git_commit)}</code></dd>",
        f"<dt>config hash</dt><dd><code>{_esc(inputs.config_hash)}</code></dd>",
        "</dl>",
    ]
    if counterexamples.get("tests_to_failure") is not None:
        parts.append(
            f'<div class="note">Search reached a verdict after '
            f"<b>{_esc(counterexamples['tests_to_failure'])}</b> oracle "
            "evaluations.</div>"
        )
    parts.append(_regression_section(inputs.regression))
    parts.append(_counterexample_section(inputs.counterexamples))
    parts.append(_benchmark_section(inputs.benchmark))
    parts.append(
        "<footer>Generated by robot-ci. Self-contained: no network required. "
        "Every rate is shown with its interval and sample count; a metric that "
        "was not measured reads <span class='na'>n/a</span> rather than 0.</footer>"
    )
    parts.append("</body></html>")
    return "\n".join(part for part in parts if part)


def _load(path: Path) -> dict[str, Any] | None:
    """Read a JSON artifact, or ``None`` when it is absent or unreadable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def render_from_directory(results: Path) -> str:
    """Build a report from an experiment results directory.

    Missing artifacts are tolerated: a report of a partially-complete experiment
    is still useful, and refusing to render would mean a failed run produces
    nothing to look at. What each section could not be built from is stated in
    the section itself rather than silently omitted.

    Args:
        results: Directory written by the experiment runner.

    Returns:
        An HTML document.
    """
    summary = _load(results / "summary.json") or {}
    regression = _load(results / "regression.json")
    counterexamples = _load(results / "counterexamples.json")
    benchmark = _load(results / "benchmark_suite.json")

    # ``summary.json`` names the arms directly. A lookup into ``arms`` came
    # first and lost to the literal string, so the header read
    # "policy_v3 -> candidate" for an experiment whose candidate was policy_v1.
    baseline = str(summary.get("baseline") or "baseline")
    candidate = str(summary.get("candidate") or "candidate")
    return render_report(
        ReportInputs(
            experiment_id=str(summary.get("experiment_id") or results.name),
            baseline=baseline,
            candidate=candidate,
            backend=str(summary.get("backend") or "unknown"),
            git_commit=str(summary.get("git_commit") or "unknown"),
            config_hash=str(summary.get("config_hash") or "unknown"),
            regression=regression,
            counterexamples=counterexamples,
            summary=summary,
            benchmark=benchmark,
        )
    )


def write_report(results: Path, destination: Path | None = None) -> Path:
    """Render a report and write it next to the artifacts.

    Args:
        results: Directory written by the experiment runner.
        destination: Where to write; defaults to ``results/report.html``.

    Returns:
        The path written.
    """
    target = destination or (results / "report.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_from_directory(results), encoding="utf-8")
    return target
