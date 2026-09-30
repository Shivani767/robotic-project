"""The bridge from a perturbation point to real episodes.

Everything else in :mod:`robot_ci.counterexamples` is simulator-agnostic. This
module is the one place that knows how to turn a point into two episodes, and it
does so through the same :func:`robot_ci.experiments.runner.run_episode` path the
experiment harness uses, so a counterexample is found under exactly the
conditions an experiment would have run.

Fairness is inherited rather than re-implemented: both arms get the *same*
scenario object and the *same* seed, which is the property Phase 3 established
and tested. A search that gave the arms different seeds would confound the
policy effect with the draw and could "find" a regression that is really noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_ci.counterexamples.discover import perturbed_scenario
from robot_ci.counterexamples.problem import OutcomeOracle, SearchSpace, Trial
from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
from robot_ci.policies import create_policy
from robot_ci.simulation import SimulatorBackend, create_backend

__all__ = ["EpisodeOracle"]


@dataclass
class EpisodeOracle(OutcomeOracle):
    """Evaluates a perturbation point by running both arms for one episode.

    Attributes:
        base_scenario: The unperturbed condition.
        space: Axes being perturbed, whose paths address the scenario.
        baseline: Baseline policy configuration.
        candidate: Candidate policy configuration.
        backend_name: Registered simulator backend.
        max_steps: Optional override of the task's step budget, so a search does
            not spend hundreds of steps per probe.
    """

    base_scenario: ScenarioConfig
    space: SearchSpace
    baseline: PolicyConfig
    candidate: PolicyConfig
    backend_name: str = "mock"
    max_steps: int | None = None
    experiment_id: str = "counterexample_search"
    history: list[dict[str, Any]] = field(default_factory=list)

    _backend: SimulatorBackend | None = field(default=None, init=False, repr=False)

    @property
    def backend(self) -> SimulatorBackend:
        """The backend, constructed on first use."""
        if self._backend is None:
            self._backend = create_backend(self.backend_name)
        return self._backend

    def close(self) -> None:
        """Release the backend. Idempotent."""
        if self._backend is not None:
            self._backend.close_all()
            self._backend = None

    def __enter__(self) -> EpisodeOracle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _run(self, policy_config: PolicyConfig, scenario: ScenarioConfig, seed: int) -> Any:
        """Run one episode for one arm under the shared scenario and seed."""
        episode = scenario
        if self.max_steps is not None:
            episode = scenario.model_copy(
                update={"task": scenario.task.model_copy(update={"max_steps": self.max_steps})}
            )
        policy = create_policy(policy_config, episode)
        policy.reset(policy_config, seed, episode)
        return run_episode(
            self.backend,
            policy,
            episode,
            seed=seed,
            experiment_id=self.experiment_id,
        )

    def evaluate(self, point: dict[str, float]) -> Trial:
        """Run both arms at ``point`` and compare them.

        Args:
            point: Axis name -> value.

        Returns:
            A :class:`Trial` whose ``metrics`` carry both arms, prefixed so a
            counterexample can be read without rerunning anything.
        """
        scenario = perturbed_scenario(self.base_scenario, self.space, point)
        seed = int(scenario.seed)
        baseline = self._run(self.baseline, scenario, seed)
        candidate = self._run(self.candidate, scenario, seed)
        metrics: dict[str, float] = {}
        for result, prefix in ((baseline, "baseline"), (candidate, "candidate")):
            for metric in result.metrics:
                metrics[f"{prefix}.{metric.name.value}"] = float(metric.value)
            metrics[f"{prefix}.steps"] = float(result.steps)
        self.history.append(
            {
                "point": dict(point),
                "baseline_success": bool(baseline.success),
                "candidate_success": bool(candidate.success),
                "baseline_status": baseline.status,
                "candidate_status": candidate.status,
            }
        )
        return Trial(
            point=dict(point),
            baseline_success=bool(baseline.success),
            candidate_success=bool(candidate.success),
            metrics=metrics,
        )
