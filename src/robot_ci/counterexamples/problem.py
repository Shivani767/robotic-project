"""The counterexample-search problem definition.

A counterexample is a point in perturbation space where the baseline succeeds
and the candidate fails. Everything in this package is expressed against two
small abstractions so that the searchers never touch a simulator:

* :class:`SearchSpace` -- the axes being perturbed and how "far" a point is.
* :class:`OutcomeOracle` -- a black box mapping a point to both arms' outcomes.

That seam is what lets the same searchers be validated in milliseconds against
an analytic oracle and then pointed at real episodes, and it is what makes the
budget accounting honest: a strategy counts evaluations of the oracle, not
seconds of wall clock.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CachingOracle",
    "OutcomeOracle",
    "PerturbationAxis",
    "SearchSpace",
    "Trial",
]


@dataclass(frozen=True)
class PerturbationAxis:
    """One swept dimension of the search space.

    Attributes:
        name: Human-readable axis name, e.g. ``"depth_noise_mm"``.
        path: Dotted path into a :class:`~robot_ci.models.scenario.ScenarioConfig`.
        lower: Smallest value considered; normally the nominal value.
        upper: Largest value considered.
        unit: Unit label, for reports.
    """

    name: str
    path: str
    lower: float
    upper: float
    unit: str = ""

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError(
                f"axis {self.name!r}: upper ({self.upper}) is below lower ({self.lower})"
            )

    @property
    def span(self) -> float:
        """Width of the axis."""
        return float(self.upper - self.lower)

    def clamp(self, value: float) -> float:
        """Clamp ``value`` into the axis bounds."""
        return float(min(max(float(value), self.lower), self.upper))

    def normalized(self, value: float) -> float:
        """Position of ``value`` along the axis, in ``[0, 1]``.

        A zero-width axis normalizes to ``0``, so a degenerate axis contributes
        nothing to severity instead of dividing by zero.
        """
        if self.span <= 0.0:
            return 0.0
        return float((self.clamp(value) - self.lower) / self.span)


@dataclass(frozen=True)
class Trial:
    """The outcome of evaluating one perturbation point for both arms."""

    point: dict[str, float]
    baseline_success: bool
    candidate_success: bool
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def is_counterexample(self) -> bool:
        """Whether the baseline succeeded and the candidate did not.

        The asymmetry matters: a point where *both* fail is not evidence about
        the candidate, and a point where the candidate succeeds is not a
        regression. Only the disagreeing direction counts.
        """
        return bool(self.baseline_success) and not bool(self.candidate_success)

    @property
    def both_succeed(self) -> bool:
        """Whether neither arm failed at this point."""
        return bool(self.baseline_success) and bool(self.candidate_success)

    @property
    def both_fail(self) -> bool:
        """Whether both arms failed, which is a task property, not a regression."""
        return not self.baseline_success and not self.candidate_success


class SearchSpace:
    """The perturbation axes, plus the severity measure used to minimize.

    Severity is the Euclidean norm of the per-axis normalized offsets. It is
    stated explicitly rather than implied because "smallest perturbation" is
    meaningless without it: 10 mm of depth noise and 20 % of occlusion are not
    comparable until each is expressed as a fraction of its own range.
    """

    def __init__(self, axes: tuple[PerturbationAxis, ...] | list[PerturbationAxis]) -> None:
        """Bind the axes and build the name index."""
        if not axes:
            raise ValueError("a search space needs at least one axis")
        self.axes: tuple[PerturbationAxis, ...] = tuple(axes)
        self._by_name = {axis.name: axis for axis in self.axes}
        if len(self._by_name) != len(self.axes):
            raise ValueError("axis names must be unique")

    def __len__(self) -> int:
        """Number of axes."""
        return len(self.axes)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is one of the axes."""
        return name in self._by_name

    def axis(self, name: str) -> PerturbationAxis:
        """Return one axis by name.

        Raises:
            KeyError: If the name is unknown, with the known names attached.
        """
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown axis {name!r}; known: {sorted(self._by_name)}") from exc

    def clamp_point(self, point: dict[str, float]) -> dict[str, float]:
        """Clamp every axis present in ``point`` into its bounds."""
        return {
            name: self._by_name[name].clamp(value)
            for name, value in point.items()
            if name in self._by_name
        }

    def severity(self, point: dict[str, float]) -> float:
        """Normalized L2 magnitude of ``point``: ``0`` at the origin.

        The origin is each axis's ``lower`` bound, so a caller should set
        ``lower`` to the nominal (unperturbed) value of that axis. The result is
        ``0`` at every axis's lower bound and ``sqrt(n)`` at every upper bound.
        """
        squares = [
            self._by_name[name].normalized(value) ** 2
            for name, value in point.items()
            if name in self._by_name
        ]
        return float(np.sqrt(sum(squares))) if squares else 0.0


class OutcomeOracle(ABC):
    """Evaluates one perturbation point for the baseline and the candidate."""

    @abstractmethod
    def evaluate(self, point: dict[str, float]) -> Trial:
        """Run both arms at ``point`` and report their outcomes."""


class CachingOracle(OutcomeOracle):
    """Memoizes an oracle so a repeated point cannot consume budget twice.

    A search that re-evaluates a point spends real compute to learn nothing and
    inflates the reported ``tests_to_failure`` for one strategy but not another,
    which would make the strategy comparison unfair.
    """

    def __init__(self, inner: OutcomeOracle) -> None:
        """Wrap ``inner``."""
        self._inner = inner
        self._cache: dict[tuple[tuple[str, float], ...], Trial] = {}
        self.hits = 0

    @staticmethod
    def _key(point: dict[str, float]) -> tuple[tuple[str, float], ...]:
        """Return a hashable, order-independent key for a point."""
        return tuple(sorted((str(k), float(v)) for k, v in point.items()))

    def evaluate(self, point: dict[str, float]) -> Trial:
        """Return the cached trial, or evaluate and store it."""
        key = self._key(point)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        trial = self._inner.evaluate(point)
        self._cache[key] = trial
        return trial

    @property
    def unique_evaluations(self) -> int:
        """Number of distinct points actually evaluated."""
        return len(self._cache)

    @property
    def evaluations(self) -> int:
        """Total evaluation calls, including cache hits."""
        return len(self._cache) + self.hits

    def reset(self) -> None:
        """Forget everything. Used when strategies share one oracle."""
        self._cache.clear()
        self.hits = 0
