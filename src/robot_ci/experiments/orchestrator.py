"""Experiment orchestration: expand a grid, run both policies, aggregate.

This is the layer between a configuration and a set of results. Its job is
fairness: every cell and every seed must be presented to both policies under
byte-identical conditions, so that any difference in outcome is attributable
to the policy and not to the run order, the seed or a mutated scenario.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_ci.core.errors import EvaluationError
from robot_ci.core.logging import JsonlLogger, get_logger
from robot_ci.core.seeding import derive_seed
from robot_ci.experiments.runner import EpisodeResult, RunSpec, run_episode
from robot_ci.models.scenario import ExperimentConfig, PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import expand_grid
from robot_ci.simulation import SimulatorBackend, create_backend

__all__ = ["ExperimentOutcome", "ExperimentRunner", "PolicySummary"]

_LOG = get_logger(__name__)


@dataclass
class PolicySummary:
    """Aggregate result for one policy across every cell and seed.

    Attributes:
        policy: Policy name.
        n_runs: Number of episodes.
        success_rate: Fraction of episodes that met the success criteria.
        per_metric: Mean of every measured metric.
        statuses: Count of each terminal status.
    """

    policy: str
    n_runs: int
    success_rate: float
    per_metric: dict[str, float] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ``summary.json``."""
        return {
            "policy": self.policy,
            "n_runs": self.n_runs,
            "success_rate": round(self.success_rate, 4),
            "per_metric": {k: round(v, 6) for k, v in sorted(self.per_metric.items())},
            "statuses": dict(sorted(self.statuses.items())),
        }


@dataclass
class ExperimentOutcome:
    """Everything one experiment produced.

    Attributes:
        config: The experiment definition.
        results: One entry per episode.
        summaries: Aggregate per policy.
        git_commit: Commit the run executed at.
        backend: Backend used.
        wall_time_s: Total wall-clock duration.
    """

    config: ExperimentConfig
    results: list[EpisodeResult]
    summaries: dict[str, PolicySummary]
    git_commit: str
    backend: str
    wall_time_s: float

    @property
    def n_episodes(self) -> int:
        """Total episodes executed."""
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ``summary.json``."""
        return {
            "experiment_id": self.config.experiment_id,
            "description": self.config.description,
            "config_hash": self.config.config_hash(),
            "git_commit": self.git_commit,
            "backend": self.backend,
            "repetitions": self.config.repetitions,
            "grid_cells": self.config.grid_size,
            "episodes": self.n_episodes,
            "episodes_expected": self.config.total_episodes_both_policies,
            "wall_time_s": round(self.wall_time_s, 3),
            "baseline": self.config.baseline.name,
            "candidate": self.config.candidate.name,
            "summaries": {name: s.to_dict() for name, s in sorted(self.summaries.items())},
            "results": [r.to_dict() for r in self.results],
        }


def build_run_specs(config: ExperimentConfig) -> list[RunSpec]:
    """Expand an experiment into the full ordered list of runs.

    The order is *policy-major*: every cell is run for the baseline, then for
    the candidate. This keeps the two policies adjacent in the log, which makes
    a mismatch obvious, and it means a crash in one policy does not leave the
    other unmeasured.

    Args:
        config: The experiment definition.

    Returns:
        One :class:`RunSpec` per (cell, seed, policy).

    Raises:
        EvaluationError: If a policy name is empty or the grid is empty.
    """
    cells = expand_grid(config.scenario, config.variables)
    if not cells:
        raise EvaluationError(
            f"experiment {config.experiment_id!r} expanded to zero scenarios",
            code="experiment.empty_grid",
            experiment_id=config.experiment_id,
        )

    policies: tuple[PolicyConfig, ...] = (config.baseline, config.candidate)
    specs: list[RunSpec] = []
    for cell in cells:
        for replicate in range(config.repetitions):
            # The seed is derived from the experiment seed, the scenario id and
            # the replicate index -- and deliberately *not* from the policy
            # name. Including the policy name gives the baseline and the
            # candidate different random draws at the same cell, which
            # confounds the policy effect with the seed: a difference between
            # the two arms can no longer be attributed to the policy, because
            # the two arms were never run under the same conditions. It is the
            # exact thing an A/B harness exists to prevent, and it is invisible
            # in the output, since both arms still produce plausible numbers.
            #
            # Sharing the seed makes the comparison paired: at a given cell and
            # replicate the two policies face an identical environment draw, so
            # the paired difference is attributable to the policy.
            seed = derive_seed(config.seed, cell.scenario_id, replicate)
            for policy in policies:
                specs.append(
                    RunSpec(
                        scenario=cell,
                        policy=policy,
                        seed=seed,
                        index=len(specs),
                    )
                )
    return specs


def summarize(results: list[EpisodeResult]) -> dict[str, PolicySummary]:
    """Aggregate episode results per policy.

    Mean is used for the aggregate; success is reported as a rate with its
    sample count so a later report can attach an interval rather than presenting
    a bare percentage.
    """
    by_policy: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_policy.setdefault(result.policy, []).append(result)

    summaries: dict[str, PolicySummary] = {}
    for name, group in by_policy.items():
        successes = sum(1 for r in group if r.success)
        statuses: dict[str, int] = {}
        for result in group:
            statuses[result.status] = statuses.get(result.status, 0) + 1
        metrics: dict[str, list[float]] = {}
        for result in group:
            for metric in result.metrics:
                metrics.setdefault(metric.name.value, []).append(metric.value)
        per_metric = {
            key: float(np.mean(values)) for key, values in metrics.items() if values
        }
        summaries[name] = PolicySummary(
            policy=name,
            n_runs=len(group),
            success_rate=successes / len(group) if group else 0.0,
            per_metric=per_metric,
            statuses=statuses,
        )
    return summaries


class ExperimentRunner:
    """Runs an experiment end to end and records every artifact.

    Example:
        >>> runner = ExperimentRunner()                      # doctest: +SKIP
        >>> outcome = runner.run(config)                     # doctest: +SKIP
        >>> outcome.summaries["policy_v1"].success_rate      # doctest: +SKIP
    """

    def __init__(self, backend_name: str = "mujoco", **backend_options: Any) -> None:
        """Bind the runner to a simulator backend.

        Args:
            backend_name: Registered backend name.
            **backend_options: Passed to the backend constructor.
        """
        self.backend_name = backend_name
        self.backend_options = backend_options
        self._backend: SimulatorBackend | None = None

    @property
    def backend(self) -> SimulatorBackend:
        """The backend, constructed on first use."""
        if self._backend is None:
            self._backend = create_backend(self.backend_name, **self.backend_options)
        return self._backend

    def run(
        self,
        config: ExperimentConfig,
        *,
        output_dir: Path | None = None,
        log: JsonlLogger | None = None,
        limit: int | None = None,
    ) -> ExperimentOutcome:
        """Execute every run in the experiment.

        Args:
            config: The experiment definition.
            output_dir: When given, artifacts are written there.
            log: Optional JSONL research log.
            limit: Optional cap on the number of runs, for smoke tests. The
                number actually run is recorded in the outcome, so a truncated
                run can never be mistaken for a complete one.

        Returns:
            The :class:`ExperimentOutcome`.
        """
        started = time.perf_counter()
        specs = build_run_specs(config)
        if limit is not None:
            specs = specs[:limit]

        _LOG.info(
            "experiment %s: %d runs (backend=%s, truncated=%s)",
            config.experiment_id, len(specs), self.backend_name, limit is not None,
        )

        results: list[EpisodeResult] = []
        backend = self.backend
        for spec in specs:
            # A fresh policy per run: policies carry internal state, and
            # sharing one instance across cells would leak history between
            # conditions.
            policy = create_policy(spec.policy, spec.scenario)
            policy.reset(spec.policy, spec.seed, spec.scenario)
            result = run_episode(
                backend,
                policy,
                spec.scenario,
                seed=spec.seed,
                experiment_id=config.experiment_id,
                log=log,
            )
            results.append(result)
            if log is not None:
                log.event(
                    "run.finish",
                    run_id=result.run_id,
                    policy=result.policy,
                    scenario_id=result.scenario_id,
                    seed=result.seed,
                    success=result.success,
                    status=result.status,
                    steps=result.steps,
                )

        outcome = ExperimentOutcome(
            config=config,
            results=results,
            summaries=summarize(results),
            git_commit=results[0].git_commit if results else "unknown",
            backend=self.backend_name,
            wall_time_s=time.perf_counter() - started,
        )
        if output_dir is not None:
            from robot_ci.experiments.artifacts import write_artifacts

            write_artifacts(outcome, output_dir, log=log)
        return outcome

    def close(self) -> None:
        """Release backend resources. Idempotent."""
        if self._backend is not None:
            self._backend.close_all()
            self._backend = None

    def __enter__(self) -> ExperimentRunner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def preview(self, config: ExperimentConfig, limit: int | None = None) -> Iterator[RunSpec]:
        """Yield the runs that would execute, without running them.

        Useful for a dry run in CI: it shows the plan and the expected episode
        count without spending any simulation time.
        """
        specs = build_run_specs(config)
        return iter(specs if limit is None else specs[:limit])
