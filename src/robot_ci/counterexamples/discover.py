"""Discovery: run a strategy and turn its result into a portable artifact.

The bridge between the abstract search and the concrete project: it rebuilds the
perturbed scenario from the axes, hashes it, and records the provenance that a
later reader needs in order to decide whether to believe the counterexample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from robot_ci.counterexamples.artifact import Counterexample
from robot_ci.counterexamples.problem import OutcomeOracle, SearchSpace
from robot_ci.counterexamples.strategies import Searcher, SearchOutcome
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.scenarios import set_by_path

__all__ = [
    "DiscoveryRequest",
    "DiscoveryResult",
    "build_artifact",
    "discover",
    "perturbed_scenario",
]


def perturbed_scenario(
    base: ScenarioConfig, space: SearchSpace, point: dict[str, float]
) -> ScenarioConfig:
    """Apply a perturbation point to a base scenario via the axes' paths."""
    scenario = base
    for name, value in sorted(point.items()):
        if name not in space:
            raise KeyError(f"point names {name!r}, which is not an axis of this space")
        scenario = set_by_path(scenario, space.axis(name).path, float(value))
    return scenario


@dataclass(frozen=True)
class DiscoveryRequest:
    """Everything the artifact needs that a search cannot know by itself."""

    experiment_id: str
    base_scenario: ScenarioConfig
    space: SearchSpace
    baseline_policy: str
    candidate_policy: str
    seed: int
    backend: str = "unknown"
    git_commit: str = "unknown"


@dataclass
class DiscoveryResult:
    """A search outcome plus the artifact it produced, if it found anything."""

    outcome: SearchOutcome
    counterexample: Counterexample | None

    @property
    def found(self) -> bool:
        """Whether a counterexample was discovered."""
        return self.counterexample is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "found": self.found,
            "search": self.outcome.to_dict(),
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
        }


def _identifier(request: DiscoveryRequest, outcome: SearchOutcome) -> str:
    """Build a stable identifier from the condition, not from the clock.

    Two runs that find the same condition under the same strategy produce the
    same identifier, which is what lets a later run be compared with an earlier
    one instead of merely added to it.
    """
    axes = "_".join(f"{name}={value:.4g}" for name, value in sorted(outcome.counterexample.point.items()))  # type: ignore[union-attr]
    return f"ce_{request.experiment_id}_{outcome.strategy}_{axes}"


def build_artifact(request: DiscoveryRequest, outcome: SearchOutcome) -> Counterexample | None:
    """Turn a successful search outcome into a counterexample artifact.

    Args:
        request: Context the search could not know.
        outcome: The finished search.

    Returns:
        The artifact, or ``None`` when the search found nothing.
    """
    trial = outcome.counterexample
    if trial is None or not outcome.found:
        return None
    scenario = perturbed_scenario(request.base_scenario, request.space, trial.point)
    metrics: dict[str, float] = {}
    for name, value in sorted(trial.metrics.items()):
        metrics[name] = float(value)
    return Counterexample(
        counterexample_id=_identifier(request, outcome),
        experiment_id=request.experiment_id,
        base_scenario_id=request.base_scenario.scenario_id,
        scenario_config_hash=scenario.config_hash(),
        task=request.base_scenario.task.name.value,
        seed=request.seed,
        baseline_policy=request.baseline_policy,
        candidate_policy=request.candidate_policy,
        perturbation=dict(trial.point),
        perturbation_paths={axis.name: axis.path for axis in request.space.axes},
        axes_payload=tuple(asdict(axis) for axis in request.space.axes),
        base_scenario_payload=request.base_scenario.model_dump(mode="json"),
        baseline_success=trial.baseline_success,
        candidate_success=trial.candidate_success,
        metrics=metrics,
        boundary=outcome.boundary,
        boundary_axis=outcome.boundary_axis,
        bracket=outcome.bracket,
        strategy=outcome.strategy,
        evaluations=outcome.evaluations,
        severity=outcome.severity,
        necessary_axes=outcome.necessary_axes,
        git_commit=request.git_commit,
        backend=request.backend,
        notes=outcome.notes,
    )


def discover(
    request: DiscoveryRequest, oracle: OutcomeOracle, strategy: Searcher
) -> DiscoveryResult:
    """Search for a counterexample and build the artifact for it.

    Args:
        request: Context the search could not know.
        oracle: The outcome oracle to evaluate against.
        strategy: The search strategy to run.

    Returns:
        The result, whose ``counterexample`` is ``None`` when nothing was found.
    """
    if strategy.space is not request.space:
        raise ValueError("the strategy was built for a different search space")
    outcome = strategy.search(oracle)
    return DiscoveryResult(outcome=outcome, counterexample=build_artifact(request, outcome))
