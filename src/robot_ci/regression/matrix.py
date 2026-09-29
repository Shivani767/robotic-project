"""Behavioural regression matrix: conditions x policy deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_ci.regression.detector import ConditionCell

__all__ = ["RegressionMatrix", "build_matrix"]


@dataclass
class RegressionMatrix:
    """A filterable matrix of per-condition comparisons."""

    rows: list[dict[str, Any]]

    def filter(
        self,
        *,
        severity: str | None = None,
        condition_contains: str | None = None,
        min_samples: int = 0,
    ) -> RegressionMatrix:
        """Return the sub-matrix matching all given filters."""
        rows = self.rows
        if severity is not None:
            rows = [r for r in rows if r.get("severity") == severity]
        if condition_contains is not None:
            rows = [r for r in rows if condition_contains in str(r.get("condition", ""))]
        if min_samples > 0:
            rows = [r for r in rows if min(int(r.get("n_baseline", 0)), int(r.get("n_candidate", 0))) >= min_samples]
        return RegressionMatrix(rows=list(rows))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {"rows": [dict(r) for r in self.rows]}


def _severity(delta: float, *, blocking: float = -0.10, warning: float = -0.05) -> str:
    if delta <= blocking:
        return "blocking"
    if delta <= warning:
        return "warning"
    if delta < 0:
        return "minor"
    if delta == 0:
        return "none"
    return "improvement"


def build_matrix(cells: list[ConditionCell]) -> RegressionMatrix:
    """Build the matrix from detector cells, labelling every row by severity."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.append(
            {
                "condition": cell.condition,
                "baseline_success": cell.baseline_success,
                "candidate_success": cell.candidate_success,
                "delta": cell.delta,
                "n_baseline": cell.n_baseline,
                "n_candidate": cell.n_candidate,
                "baseline_ci": list(cell.baseline_ci),
                "candidate_ci": list(cell.candidate_ci),
                "effect_h": cell.effect_h,
                "severity": _severity(cell.delta),
                "conditions": dict(cell.conditions),
            }
        )
    rows.sort(key=lambda r: float(r["delta"]))
    return RegressionMatrix(rows=rows)
