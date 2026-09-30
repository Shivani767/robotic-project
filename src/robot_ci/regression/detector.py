"""Behavioural regression detection: paired, conditional comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_ci.experiments.runner import EpisodeResult
from robot_ci.statistics import cohens_h, proportion_ci

__all__ = [
    "ConditionCell",
    "RegressionConfig",
    "RegressionFinding",
    "RegressionReport",
    "compare_conditions",
    "detect_regressions",
]


@dataclass(frozen=True)
class RegressionConfig:
    """Thresholds that turn a measured delta into a severity label."""

    blocking_delta: float = -0.10
    warning_delta: float = -0.05
    min_samples: int = 3


@dataclass
class ConditionCell:
    """Paired comparison at one condition (one scenario id)."""

    condition: str
    baseline_success: float
    candidate_success: float
    delta: float
    n_baseline: int
    n_candidate: int
    baseline_ci: tuple[float, float]
    candidate_ci: tuple[float, float]
    effect_h: float
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionFinding:
    """One condition where the candidate is worse by at least the threshold."""

    condition: str
    delta: float
    baseline_success: float
    candidate_success: float
    n: int
    severity: str
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionReport:
    """Full comparison: global rates plus every conditional cell."""

    baseline_rate: float
    candidate_rate: float
    global_delta: float
    n_baseline: int
    n_candidate: int
    cells: list[ConditionCell] = field(default_factory=list)
    findings: list[RegressionFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for artifacts and the dashboard."""
        return {
            "baseline_rate": self.baseline_rate,
            "candidate_rate": self.candidate_rate,
            "global_delta": self.global_delta,
            "n_baseline": self.n_baseline,
            "n_candidate": self.n_candidate,
            "cells": [
                {
                    "condition": c.condition,
                    "baseline_success": c.baseline_success,
                    "candidate_success": c.candidate_success,
                    "delta": c.delta,
                    "n_baseline": c.n_baseline,
                    "n_candidate": c.n_candidate,
                    "baseline_ci": list(c.baseline_ci),
                    "candidate_ci": list(c.candidate_ci),
                    "effect_h": c.effect_h,
                    "conditions": dict(c.conditions),
                }
                for c in self.cells
            ],
            "findings": [
                {
                    "condition": f.condition,
                    "delta": f.delta,
                    "baseline_success": f.baseline_success,
                    "candidate_success": f.candidate_success,
                    "n": f.n,
                    "severity": f.severity,
                    "conditions": dict(f.conditions),
                }
                for f in self.findings
            ],
        }


def _rate(results: list[EpisodeResult]) -> tuple[float, tuple[float, float], int]:
    n = len(results)
    if n == 0:
        return 0.0, (0.0, 1.0), 0
    successes = sum(1 for r in results if r.success)
    interval = proportion_ci(successes, n)
    return interval.rate, (interval.lower, interval.upper), n


#: Condition fields that vary *within* a cell and so must not define one.
#:
#: ``seed`` is excluded because every repetition in a cell differs by seed, and
#: including it would make each episode its own cell of size one.
_INTRA_CELL_FIELDS: frozenset[str] = frozenset({"seed"})


def _cell_key(result: EpisodeResult) -> str:
    """Return the condition cell an episode belongs to.

    The key is built from the episode's *condition values*, not from its
    scenario id alone. Grouping by scenario id is correct only for a grid
    expanded into distinct scenario files, where every cell gets its own id. It
    is silently wrong for every sweep that varies a field in place, which is
    most of the tools here: the counterexample search, ``robot-ci diagnose`` and
    the RQ2 harness all perturb a scenario without renaming it.

    The consequence of getting this wrong is severe and quiet. Measured on a
    five-cell depth-noise grid, a detector keyed on scenario id returned a
    single cell containing all 40 episodes, reported one "regression" of
    -0.125 at "nominal", and never mentioned that the worst cell was -0.25 --
    a conditional analysis that did not condition on anything, returning an
    aggregate result under a name that promised otherwise.

    Args:
        result: One episode.

    Returns:
        A stable string identifying the cell.
    """
    values = {
        f"{key}={value!r}"
        for key, value in sorted(result.conditions.items())
        if key not in _INTRA_CELL_FIELDS
    }
    return result.scenario_id + "|" + "|".join(values)


def _disambiguate(
    labels: dict[str, str],
    by_condition: dict[str, dict[str, list[EpisodeResult]]],
) -> dict[str, str]:
    """Give every cell a label a reader can tell apart.

    Cells are grouped on condition *values*, so several cells legitimately share
    one scenario id -- that is the case this whole change exists for. Printing
    the bare id then produces a report listing four findings all called
    "nominal" at different deltas, which is not a report.

    A label is left alone when it is unique. When it is not, the condition
    fields that actually differ between the colliding cells are appended, so the
    label says which level of which variable it refers to and no more.
    """
    shared: dict[str, list[str]] = {}
    for key, label in labels.items():
        shared.setdefault(label, []).append(key)

    resolved = dict(labels)
    for label, keys in shared.items():
        if len(keys) < 2:
            continue
        # Fields that vary between the colliding cells are the useful ones; ones
        # shared by all of them (task, family, scenario id) add only noise.
        variants = [
            {k: v for k, v in results[0].conditions.items() if k not in _INTRA_CELL_FIELDS}
            for results in (by_condition[key]["baseline"] or by_condition[key]["candidate"] for key in keys)
            if results
        ]
        if not variants:
            continue
        varying = sorted(
            key
            for key in variants[0]
            if any(other.get(key) != variants[0][key] for other in variants[1:])
        )
        for key, variant in zip(keys, variants, strict=True):
            detail = ", ".join(f"{name}={variant[name]!r}" for name in varying)
            resolved[key] = f"{label} [{detail}]" if detail else label
    return resolved


def compare_conditions(
    baseline: list[EpisodeResult],
    candidate: list[EpisodeResult],
    *,
    config: RegressionConfig = RegressionConfig(),
) -> RegressionReport:
    """Compare two episode sets, globally and per condition."""
    base_rate, _, n_b = _rate(baseline)
    cand_rate, _, n_c = _rate(candidate)
    by_condition: dict[str, dict[str, list[EpisodeResult]]] = {}
    #: Readable label per cell, so a report says "depth_bias" rather than the
    #: internal grouping string. Two cells can share a scenario id -- that is
    #: the whole point of the finer key -- so the label is only a label.
    labels: dict[str, str] = {}
    for arm, results in (("baseline", baseline), ("candidate", candidate)):
        for result in results:
            key = _cell_key(result)
            labels.setdefault(key, result.scenario_id)
            by_condition.setdefault(key, {"baseline": [], "candidate": []})[arm].append(result)
    cells: list[ConditionCell] = []
    findings: list[RegressionFinding] = []
    cell_labels = _disambiguate(labels, by_condition)
    for key in sorted(by_condition):
        condition = cell_labels[key]
        group = by_condition[key]
        b_rate, b_ci, b_n = _rate(group["baseline"])
        c_rate, c_ci, c_n = _rate(group["candidate"])
        delta = c_rate - b_rate
        if group["baseline"]:
            conditions = dict(group["baseline"][0].conditions)
        else:
            conditions = dict(group["candidate"][0].conditions)
        cells.append(
            ConditionCell(
                condition=condition,
                baseline_success=b_rate,
                candidate_success=c_rate,
                delta=delta,
                n_baseline=b_n,
                n_candidate=c_n,
                baseline_ci=b_ci,
                candidate_ci=c_ci,
                effect_h=cohens_h(b_rate, c_rate),
                conditions=conditions,
            )
        )
        n = min(b_n, c_n)
        if n >= config.min_samples and delta <= config.warning_delta:
            severity = "blocking" if delta <= config.blocking_delta else "warning"
            findings.append(
                RegressionFinding(
                    condition=condition,
                    delta=delta,
                    baseline_success=b_rate,
                    candidate_success=c_rate,
                    n=n,
                    severity=severity,
                    conditions=conditions,
                )
            )
    findings.sort(key=lambda f: f.delta)
    return RegressionReport(
        baseline_rate=base_rate,
        candidate_rate=cand_rate,
        global_delta=cand_rate - base_rate,
        n_baseline=n_b,
        n_candidate=n_c,
        cells=cells,
        findings=findings,
    )


def detect_regressions(
    results: list[EpisodeResult],
    *,
    baseline_name: str,
    candidate_name: str,
    config: RegressionConfig = RegressionConfig(),
) -> RegressionReport:
    """Split episodes by policy and run the conditional comparison."""
    baseline = [r for r in results if r.policy == baseline_name]
    candidate = [r for r in results if r.policy == candidate_name]
    return compare_conditions(baseline, candidate, config=config)
