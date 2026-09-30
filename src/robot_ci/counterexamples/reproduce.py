"""Replaying a counterexample artifact.

The specification requires that a stored counterexample be executable again.
This module is that guarantee, and it is deliberately strict about what it
reports: the *rate* at which the stored condition reproduces, measured over
however many seeds the caller asks for, never an assumption that it reproduces
because it once did.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from robot_ci.counterexamples.artifact import Counterexample
from robot_ci.counterexamples.problem import OutcomeOracle

__all__ = ["ReproductionReport", "reproduce"]


@dataclass
class ReproductionReport:
    """How reliably an artifact reproduced.

    Attributes:
        counterexample_id: The artifact replayed.
        seeds: The seeds used.
        attempts: Number of replays.
        reproduced: Replays that reproduced the stored disagreement.
        rate: ``reproduced / attempts``.
        baseline_successes: Replays where the baseline succeeded.
        candidate_successes: Replays where the candidate succeeded.
        trials: One record per replay, for auditing a surprising rate.
        consistent: Whether every replay agreed, whatever the agreement was.
    """

    counterexample_id: str
    seeds: tuple[int, ...]
    attempts: int
    reproduced: int
    rate: float
    baseline_successes: int
    candidate_successes: int
    trials: list[dict[str, Any]] = field(default_factory=list)
    consistent: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "counterexample_id": self.counterexample_id,
            "seeds": list(self.seeds),
            "attempts": self.attempts,
            "reproduced": self.reproduced,
            "rate": round(self.rate, 6),
            "baseline_successes": self.baseline_successes,
            "candidate_successes": self.candidate_successes,
            "consistent": self.consistent,
            "trials": list(self.trials),
        }


def reproduce(
    artifact: Counterexample,
    make_oracle: Callable[[int], OutcomeOracle],
    *,
    seeds: Sequence[int] | None = None,
) -> ReproductionReport:
    """Replay an artifact's condition and report whether it still disagrees.

    Args:
        artifact: The stored counterexample.
        make_oracle: Builds a seeded oracle, so one artifact can be replayed
            under several seeds. A caller that wants a pure replay passes a
            factory that ignores the seed.
        seeds: Seeds to replay under. Defaults to the artifact's own seed, which
            is the "same run again" case; passing more seeds measures whether
            the failure belongs to the *condition* or to one lucky draw.

    Returns:
        The :class:`ReproductionReport`.
    """
    chosen: tuple[int, ...] = tuple(seeds) if seeds is not None else (int(artifact.seed),)
    if not chosen:
        raise ValueError("at least one seed is required to reproduce an artifact")

    trials: list[dict[str, Any]] = []
    reproduced = 0
    baseline_successes = 0
    candidate_successes = 0
    for seed in chosen:
        oracle = make_oracle(int(seed))
        try:
            trial = oracle.evaluate(dict(artifact.perturbation))
        finally:
            # An oracle that owns a backend must be released, or one simulator
            # handle accumulates per replay attempt. Oracles without a close()
            # method are pure and need nothing.
            close = getattr(oracle, "close", None)
            if callable(close):
                close()
        if trial.baseline_success:
            baseline_successes += 1
        if trial.candidate_success:
            candidate_successes += 1
        if trial.is_counterexample:
            reproduced += 1
        trials.append(
            {
                "seed": int(seed),
                "baseline_success": bool(trial.baseline_success),
                "candidate_success": bool(trial.candidate_success),
                "reproduced": bool(trial.is_counterexample),
                "metrics": {k: round(float(v), 9) for k, v in sorted(trial.metrics.items())},
            }
        )

    attempts = len(chosen)
    distinct = {(t["baseline_success"], t["candidate_success"]) for t in trials}
    return ReproductionReport(
        counterexample_id=artifact.counterexample_id,
        seeds=chosen,
        attempts=attempts,
        reproduced=reproduced,
        rate=reproduced / attempts,
        baseline_successes=baseline_successes,
        candidate_successes=candidate_successes,
        trials=trials,
        consistent=len(distinct) == 1,
    )
