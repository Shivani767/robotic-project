"""Counterexample search strategies.

Two methods are deterministic and mirror the project specification directly
(ascending grid, scalar bisection); two are the baselines the proposed method
must beat to be worth anything (random sampling, and an adaptive ramp+bisect).
All are counted in oracle evaluations so they are comparable under a shared
budget.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_ci.counterexamples.problem import OutcomeOracle, SearchSpace, Trial

__all__ = [
    "AdaptiveBoundarySearcher",
    "BinaryBoundarySearcher",
    "GridSearcher",
    "RandomSearcher",
    "SearchOutcome",
    "Searcher",
    "ramp_point",
]


@dataclass
class SearchOutcome:
    """What a search spent and what it found.

    ``evaluations`` counts distinct oracle calls, so two strategies that find
    the same counterexample are still distinguishable by cost, and a strategy
    that finds it in fewer calls is genuinely better rather than luckier.
    """

    strategy: str
    found: bool
    counterexample: Trial | None
    evaluations: int
    severity: float
    boundary: float | None = None
    bracket: tuple[float, float] | None = None
    boundary_axis: str | None = None
    trajectory: list[Trial] = field(default_factory=list)
    necessary_axes: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary, excluding the full trajectory."""
        return {
            "strategy": self.strategy,
            "found": self.found,
            "evaluations": self.evaluations,
            "severity": round(self.severity, 6) if np.isfinite(self.severity) else None,
            "boundary": self.boundary,
            "bracket": list(self.bracket) if self.bracket else None,
            "boundary_axis": self.boundary_axis,
            "necessary_axes": list(self.necessary_axes),
            "counterexample": (
                {
                    "point": dict(self.counterexample.point),
                    "baseline_success": self.counterexample.baseline_success,
                    "candidate_success": self.counterexample.candidate_success,
                    "metrics": dict(self.counterexample.metrics),
                }
                if self.counterexample is not None
                else None
            ),
            "notes": self.notes,
        }


def ramp_point(space: SearchSpace, t: float) -> dict[str, float]:
    """Map a scalar severity ``t`` in ``[0, 1]`` onto every axis at once.

    ``t = 0`` is the nominal origin (every axis at its lower bound) and
    ``t = 1`` is the far corner. Reducing a multi-axis search to one scalar is
    what makes bisection usable in more than one dimension, and it is only valid
    while the response is monotone along the ramp -- an assumption the coarse
    scan is designed to expose rather than hide.
    """
    value = float(min(max(t, 0.0), 1.0))
    return {axis.name: axis.lower + value * axis.span for axis in space.axes}


def _improves(best: Trial | None, candidate: Trial, space: SearchSpace) -> bool:
    """Whether ``candidate`` is a strictly smaller counterexample than ``best``."""
    if not candidate.is_counterexample:
        return False
    if best is None:
        return True
    return space.severity(candidate.point) < space.severity(best.point)


class Searcher(ABC):
    """Base class for counterexample search strategies."""

    #: Registry key.
    name: str = "abstract"

    def __init__(self, space: SearchSpace, *, budget: int = 256) -> None:
        """Bind the search space and the evaluation budget.

        Args:
            space: Axes being perturbed.
            budget: Maximum number of oracle evaluations.

        Raises:
            ValueError: If the budget is not positive.
        """
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self.space = space
        self.budget = int(budget)
        self._trajectory: list[Trial] = []
        self._best: Trial | None = None

    @property
    def spent(self) -> int:
        """Evaluations used so far."""
        return len(self._trajectory)

    @property
    def exhausted(self) -> bool:
        """Whether the budget is spent."""
        return self.spent >= self.budget

    def _evaluate(self, oracle: OutcomeOracle, point: dict[str, float]) -> Trial:
        """Evaluate one clamped point, record it, and track the best so far."""
        if self.exhausted:
            raise RuntimeError(f"{self.name}: budget of {self.budget} exhausted")
        trial = oracle.evaluate(self.space.clamp_point(point))
        self._trajectory.append(trial)
        if _improves(self._best, trial, self.space):
            self._best = trial
        return trial

    @abstractmethod
    def search(self, oracle: OutcomeOracle) -> SearchOutcome:
        """Search for a counterexample and report the outcome."""

    def _outcome(self, *, found: bool, notes: str = "") -> SearchOutcome:
        """Assemble the outcome from the tracked best and the trajectory."""
        severity = self.space.severity(self._best.point) if self._best else float("inf")
        return SearchOutcome(
            strategy=self.name,
            found=bool(found and self._best is not None),
            counterexample=self._best,
            evaluations=self.spent,
            severity=severity,
            trajectory=list(self._trajectory),
            notes=notes,
        )


class GridSearcher(Searcher):
    """Ascending Cartesian lattice: the deterministic baseline.

    Visits points from the nominal origin outward by severity, so the first
    counterexample it meets is already the smallest one *on the lattice*. It is
    exhaustive and explains nothing: the budget is spread over every axis
    whether or not that axis matters, and the resolution is fixed in advance.
    """

    name = "grid"

    def __init__(self, space: SearchSpace, *, steps_per_axis: int = 4, budget: int = 256) -> None:
        """Bind the lattice resolution.

        Args:
            space: Axes being perturbed.
            steps_per_axis: Number of values per axis, including the nominal one.
            budget: Maximum number of oracle evaluations.

        Raises:
            ValueError: If ``steps_per_axis`` is below 2.
        """
        super().__init__(space, budget=budget)
        if steps_per_axis < 2:
            raise ValueError(f"steps_per_axis must be at least 2, got {steps_per_axis}")
        self.steps_per_axis = int(steps_per_axis)

    def lattice(self) -> list[dict[str, float]]:
        """Return the lattice points, ordered by ascending severity."""
        per_axis = [
            np.linspace(axis.lower, axis.upper, self.steps_per_axis) for axis in self.space.axes
        ]
        grid = np.meshgrid(*per_axis, indexing="ij")
        points = []
        for flat in zip(*[component.ravel() for component in grid], strict=True):
            points.append(
                {
                    axis.name: float(value)
                    for axis, value in zip(self.space.axes, flat, strict=True)
                }
            )
        points.sort(key=self.space.severity)
        return points

    def search(self, oracle: OutcomeOracle) -> SearchOutcome:
        """Evaluate the lattice in severity order, stopping at the first hit."""
        for point in self.lattice():
            if self.exhausted:
                return self._outcome(found=False, notes="budget exhausted before a hit")
            trial = self._evaluate(oracle, point)
            if trial.is_counterexample:
                return self._outcome(found=True, notes="first lattice hit")
        return self._outcome(found=False, notes="no lattice point is a counterexample")


class RandomSearcher(Searcher):
    """Uniform sampling over the box: the baseline to beat.

    Seeded so a comparison against another strategy is reproducible, which
    matters because an unseeded random baseline can be made to look arbitrarily
    good or bad by rerunning it.
    """

    name = "random"

    def __init__(self, space: SearchSpace, *, budget: int = 256, seed: int = 0) -> None:
        """Bind the sampling seed."""
        super().__init__(space, budget=budget)
        self.seed = int(seed)

    def search(self, oracle: OutcomeOracle) -> SearchOutcome:
        """Sample uniformly until a counterexample is found or budget runs out."""
        rng = np.random.default_rng(self.seed)
        while not self.exhausted:
            point = {
                axis.name: float(rng.uniform(axis.lower, axis.upper)) for axis in self.space.axes
            }
            trial = self._evaluate(oracle, point)
            if trial.is_counterexample:
                return self._outcome(found=True, notes="uniform sample hit")
        return self._outcome(found=False, notes="budget exhausted before a hit")


class BinaryBoundarySearcher(Searcher):
    """Bisection along one axis, between a passing and a failing point.

    This is the specification's worked example: probe outward from nominal,
    bracket the first failure, then bisect until the bracket is narrower than
    ``tolerance``. The reported counterexample is the smallest *evaluated*
    failing value, and the boundary is the midpoint of the final bracket, which
    is why the boundary is reported as an interval rather than a number.
    """

    name = "binary"

    def __init__(self, space: SearchSpace, *, budget: int = 256, tolerance: float | None = None) -> None:
        """Bind the bisection tolerance.

        Args:
            space: Axes being perturbed. Must have exactly one axis.
            budget: Maximum number of oracle evaluations.
            tolerance: Stop when the bracket is narrower than this, in axis
                units. Defaults to ``span / 256``.

        Raises:
            ValueError: If the space does not have exactly one axis.
        """
        super().__init__(space, budget=budget)
        if len(space) != 1:
            raise ValueError(
                f"binary search needs exactly one axis, got {len(space)}; "
                "use AdaptiveBoundarySearcher for a multi-axis space"
            )
        axis = space.axes[0]
        self.tolerance = float(tolerance) if tolerance is not None else max(axis.span / 256.0, 1e-9)

    def search(self, oracle: OutcomeOracle) -> SearchOutcome:
        """Bracket the first failure from nominal, then bisect it."""
        axis = self.space.axes[0]
        origin = {axis.name: axis.lower}
        first = self._evaluate(oracle, origin)
        if first.is_counterexample:
            return self._outcome(
                found=True, notes="the candidate fails at the nominal value: no perturbation is needed"
            )
        if not first.both_succeed:
            return self._outcome(
                found=False,
                notes="both arms fail at nominal, so no baseline success exists to regress from",
            )

        # Exponential probe outward until the response flips.
        last_pass = axis.lower
        first_fail: float | None = None
        step = axis.span / 4.0
        value = axis.lower + step
        while value <= axis.upper and not self.exhausted:
            trial = self._evaluate(oracle, {axis.name: value})
            if trial.is_counterexample:
                first_fail = value
                break
            if trial.both_succeed:
                last_pass = value
            step *= 2.0
            value = axis.lower + step
        if first_fail is None:
            return self._outcome(
                found=False, notes="no counterexample found anywhere in the axis range"
            )

        while first_fail - last_pass > self.tolerance and not self.exhausted:
            mid = 0.5 * (last_pass + first_fail)
            trial = self._evaluate(oracle, {axis.name: mid})
            if trial.is_counterexample:
                first_fail = mid
            else:
                last_pass = mid

        outcome = self._outcome(found=True, notes="bracketed by exponential probe then bisected")
        outcome.boundary_axis = axis.name
        outcome.bracket = (last_pass, first_fail)
        outcome.boundary = 0.5 * (last_pass + first_fail)
        return outcome


class AdaptiveBoundarySearcher(Searcher):
    """Ramp the whole space, bisect it, then shrink each axis back to nominal.

    Three phases, each doing something the previous cannot:

    1. A coarse scan along the severity ramp ``t``. This is what detects a
       *non-monotone* response -- a failure that appears and then disappears as
       perturbation grows. Bisection alone would silently return a boundary
       from a confusing region, so the scan's job is to refuse to guess.
    2. Bisection on ``t`` to place the boundary within ``tolerance``.
    3. Coordinate shrinking: return each axis to its nominal value in turn and
       keep the reduction when the counterexample survives. The axes that
       cannot be removed are the ones the failure actually needs, which is a
       diagnosis built out of experiments rather than correlation.
    """

    name = "adaptive"

    def __init__(
        self,
        space: SearchSpace,
        *,
        budget: int = 256,
        coarse_steps: int = 8,
        tolerance: float = 1.0 / 256.0,
    ) -> None:
        """Bind the coarse scan resolution and the bisection tolerance.

        Args:
            space: Axes being perturbed.
            budget: Maximum number of oracle evaluations.
            coarse_steps: Number of ramp values probed before bisecting.
            tolerance: Stop when the ramp bracket is narrower than this.

        Raises:
            ValueError: If ``coarse_steps`` is below 2.
        """
        super().__init__(space, budget=budget)
        if coarse_steps < 2:
            raise ValueError(f"coarse_steps must be at least 2, got {coarse_steps}")
        self.coarse_steps = int(coarse_steps)
        self.tolerance = float(tolerance)

    def search(self, oracle: OutcomeOracle) -> SearchOutcome:
        """Run the coarse scan, bisect the ramp, then shrink each axis."""
        origin = self._evaluate(oracle, ramp_point(self.space, 0.0))
        if origin.is_counterexample:
            return self._outcome(
                found=True, notes="the candidate fails at nominal: no perturbation is needed"
            )
        if not origin.both_succeed:
            return self._outcome(
                found=False,
                notes="both arms fail at nominal, so no baseline success exists to regress from",
            )

        # Phase 1: coarse scan. Every coarse point is evaluated, not just the
        # points up to the first hit, because the whole point of this phase is
        # to notice a response that stops failing again as severity grows. A
        # scan that stopped at the first hit could not tell the difference.
        t_pass = 0.0
        t_fail: float | None = None
        non_monotone = False
        for index in range(1, self.coarse_steps + 1):
            t = index / self.coarse_steps
            trial = self._evaluate(oracle, ramp_point(self.space, t))
            if trial.is_counterexample and t_fail is None:
                t_fail = t
            elif t_fail is not None and trial.both_succeed:
                # A pass after a failure means the ramp is not monotone, so the
                # bracket below is local and the boundary must be read as such.
                non_monotone = True
            if trial.both_succeed and t_fail is None:
                t_pass = t
        if t_fail is None:
            notes = "no counterexample found along the severity ramp"
            if non_monotone:  # pragma: no cover - requires a hit to be reachable
                notes += " (the response is non-monotone)"
            return self._outcome(found=False, notes=notes)

        # Phase 2: bisect the ramp bracket.
        while (t_fail - t_pass) > self.tolerance and not self.exhausted:
            mid = 0.5 * (t_pass + t_fail)
            trial = self._evaluate(oracle, ramp_point(self.space, mid))
            if trial.is_counterexample:
                t_fail = mid
            else:
                t_pass = mid

        # Phase 3: return each axis to its nominal value and keep the reduction
        # when the counterexample survives. The axes that survive removal are
        # the ones the failure needs; the rest were along for the ride.
        necessary: list[str] = []
        for axis in self.space.axes:
            if self.exhausted:
                necessary.append(axis.name)
                continue
            best = self._best
            assert best is not None  # a counterexample exists, so best is tracked
            reduced = dict(best.point)
            reduced[axis.name] = axis.lower
            trial = self._evaluate(oracle, reduced)
            if not trial.is_counterexample:
                necessary.append(axis.name)

        notes = "coarse scan, bisection, then coordinate shrink"
        if non_monotone:
            notes += "; the ramp response is non-monotone, so the bracket is local"
        outcome = self._outcome(found=True, notes=notes)
        outcome.bracket = (t_pass, t_fail)
        outcome.necessary_axes = tuple(necessary)
        if len(self.space) == 1:
            axis = self.space.axes[0]
            outcome.boundary_axis = axis.name
            outcome.boundary = axis.lower + 0.5 * (t_pass + t_fail) * axis.span
        else:
            outcome.boundary_axis = "severity_ramp"
            outcome.boundary = 0.5 * (t_pass + t_fail)
        return outcome
