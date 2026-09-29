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
    for result in baseline:
        key = by_condition.setdefault(result.scenario_id, {"baseline": [], "candidate": []})
        key["baseline"].append(result)
    for result in candidate:
        key = by_condition.setdefault(result.scenario_id, {"baseline": [], "candidate": []})
        key["candidate"].append(result)
    cells: list[ConditionCell] = []
    findings: list[RegressionFinding] = []
    for condition in sorted(by_condition):
        group = by_condition[condition]
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
