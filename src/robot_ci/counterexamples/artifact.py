"""The portable counterexample artifact.

A discovered regression is only useful if it survives leaving the process that
found it. This model is the serialized form: every field needed to rerun the
failure, plus the provenance needed to distrust it if the provenance is wrong.

The reproduction rate is reported rather than assumed. On a deterministic
backend replaying the same seed must reproduce every time, and a rate below 1.0
is a bug in the artifact or the backend, not a finding. The interesting number
is the rate across *different* seeds, which is what says whether the failure is
a property of the condition or of one lucky draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_ci.core.io import load_yaml, write_yaml
from robot_ci.core.version import SCHEMA_VERSION
from robot_ci.counterexamples.problem import PerturbationAxis, SearchSpace
from robot_ci.models.scenario import ScenarioConfig

__all__ = [
    "ARTIFACT_VERSION",
    "Counterexample",
    "load_counterexample",
    "save_counterexample",
]

#: Version of the artifact layout, independent of the package version.
ARTIFACT_VERSION = "1.0.0"


@dataclass
class Counterexample:
    """A reproducible failing condition.

    Attributes:
        counterexample_id: Stable identifier, safe for a filename.
        experiment_id: The experiment that produced it.
        base_scenario_id: The unperturbed scenario it derives from.
        scenario_config_hash: Hash of the *perturbed* scenario's conditions, so
            a replay can be checked against the conditions it claims.
        task: Task name.
        seed: Episode seed used at discovery.
        baseline_policy: Baseline policy name.
        candidate_policy: Candidate policy name.
        perturbation: Axis name -> value, the minimal condition found.
        perturbation_paths: Axis name -> dotted scenario path, for replay.
        base_scenario_payload: Full serialization of the unperturbed scenario.
            Embedded so the artifact is self-contained: a counterexample whose
            scenario only exists on the machine that found it is a note, not a
            reproducible experiment.
        baseline_success: Baseline outcome at the point.
        candidate_success: Candidate outcome at the point.
        metrics: Metric name -> value for both arms, prefixed by arm.
        boundary: Estimated scalar regression boundary, when one was measured.
        boundary_axis: Axis the boundary refers to.
        bracket: The final (pass, fail) bracket, when one was measured.
        strategy: Strategy that found it.
        evaluations: Oracle evaluations the strategy spent.
        severity: Normalized perturbation magnitude.
        necessary_axes: Axes the failure survived removing.
        git_commit: Commit the discovery ran at.
        backend: Simulator backend used.
        notes: Free-form honesty notes from the strategy.
        reproduction: Filled in by :mod:`robot_ci.counterexamples.reproduce`.
    """

    counterexample_id: str
    experiment_id: str
    base_scenario_id: str
    scenario_config_hash: str
    task: str
    seed: int
    baseline_policy: str
    candidate_policy: str
    perturbation: dict[str, float] = field(default_factory=dict)
    perturbation_paths: dict[str, str] = field(default_factory=dict)
    axes_payload: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    base_scenario_payload: dict[str, Any] = field(default_factory=dict)
    baseline_success: bool = False
    candidate_success: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    boundary: float | None = None
    boundary_axis: str | None = None
    bracket: tuple[float, float] | None = None
    strategy: str = ""
    evaluations: int = 0
    severity: float = 0.0
    necessary_axes: tuple[str, ...] = ()
    git_commit: str = "unknown"
    backend: str = "unknown"
    notes: str = ""
    reproduction: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.counterexample_id:
            raise ValueError("counterexample_id must be non-empty")
        if not (self.baseline_success and not self.candidate_success):
            raise ValueError(
                "a counterexample must have the baseline succeeding and the candidate "
                f"failing; got baseline={self.baseline_success}, "
                f"candidate={self.candidate_success}"
            )

    def base_scenario(self) -> ScenarioConfig:
        """Rebuild the unperturbed scenario embedded in this artifact.

        Returns:
            The scenario the perturbation applies to.

        Raises:
            ValueError: If the artifact does not carry one. A counterexample
                without its scenario cannot be replayed elsewhere, so the
                failure is explicit rather than silently defaulting to a
                nominal scenario that has nothing to do with the discovery.
        """
        if not self.base_scenario_payload:
            raise ValueError(
                f"counterexample {self.counterexample_id!r} carries no base scenario, "
                "so it cannot be replayed outside the machine that produced it"
            )
        return ScenarioConfig.model_validate(self.base_scenario_payload)

    def axes(self) -> tuple[PerturbationAxis, ...]:
        """Rebuild the search axes the perturbation was measured on.

        The bounds travel with the artifact because severity is meaningless
        without them: a value of ``10`` is only comparable to another value once
        both are scaled by their own ranges, and a replaying machine has no way
        to recover those ranges from the value alone.
        """
        return tuple(PerturbationAxis(**dict(spec)) for spec in self.axes_payload)

    def space(self) -> SearchSpace:
        """Rebuild the :class:`~robot_ci.counterexamples.problem.SearchSpace`.

        Raises:
            ValueError: If the artifact carries no axes, so severity and clamping
                cannot be computed.
        """
        if not self.axes_payload:
            raise ValueError(
                f"counterexample {self.counterexample_id!r} carries no axis bounds"
            )
        return SearchSpace(self.axes())
    def to_dict(self) -> dict[str, Any]:
        """Serialize to the nested, human-readable YAML form."""
        return {
            "artifact_version": ARTIFACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "counterexample_id": self.counterexample_id,
            "experiment_id": self.experiment_id,
            "scenario": {
                "base_scenario_id": self.base_scenario_id,
                "task": self.task,
                "config_hash": self.scenario_config_hash,
                "base": dict(self.base_scenario_payload),
            },
            "seed": self.seed,
            "perturbation": {
                name: {"value": value, "path": self.perturbation_paths.get(name, "")}
                for name, value in sorted(self.perturbation.items())
            },
            "axes": [dict(spec) for spec in self.axes_payload],
            "baseline": {"policy": self.baseline_policy, "result": "success"},
            "candidate": {"policy": self.candidate_policy, "result": "failure"},
            "metrics": {k: round(v, 9) for k, v in sorted(self.metrics.items())},
            "boundary": {
                "axis": self.boundary_axis,
                "estimate": self.boundary,
                "bracket": list(self.bracket) if self.bracket else None,
            },
            "discovery": {
                "strategy": self.strategy,
                "evaluations": self.evaluations,
                "severity": round(self.severity, 9),
                "necessary_axes": list(self.necessary_axes),
                "git_commit": self.git_commit,
                "backend": self.backend,
                "notes": self.notes,
            },
            "reproduction": dict(self.reproduction),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Counterexample:
        """Rebuild from :meth:`to_dict` output.

        Raises:
            ValueError: If the payload is not a counterexample artifact, or its
                artifact version is a newer major layout than this code knows.
        """
        version = str(payload.get("artifact_version", ""))
        if not version:
            raise ValueError("payload is not a counterexample artifact (no artifact_version)")
        if version.split(".")[0] != ARTIFACT_VERSION.split(".")[0]:
            raise ValueError(
                f"artifact layout {version} is incompatible with this code "
                f"(expects major {ARTIFACT_VERSION.split('.')[0]})"
            )
        scenario = dict(payload.get("scenario", {}))
        perturbation = dict(payload.get("perturbation", {}))
        boundary = dict(payload.get("boundary", {}))
        discovery = dict(payload.get("discovery", {}))
        bracket = boundary.get("bracket")
        return cls(
            counterexample_id=str(payload["counterexample_id"]),
            experiment_id=str(payload.get("experiment_id", "")),
            base_scenario_id=str(scenario.get("base_scenario_id", "")),
            scenario_config_hash=str(scenario.get("config_hash", "")),
            task=str(scenario.get("task", "")),
            base_scenario_payload=dict(scenario.get("base", {}) or {}),
            seed=int(payload.get("seed", 0)),
            baseline_policy=str(dict(payload.get("baseline", {})).get("policy", "")),
            candidate_policy=str(dict(payload.get("candidate", {})).get("policy", "")),
            perturbation={
                name: float(dict(spec).get("value", 0.0)) for name, spec in perturbation.items()
            },
            perturbation_paths={
                name: str(dict(spec).get("path", "")) for name, spec in perturbation.items()
            },
            axes_payload=tuple(dict(spec) for spec in payload.get("axes", ())),
            baseline_success=True,
            candidate_success=False,
            metrics={k: float(v) for k, v in dict(payload.get("metrics", {})).items()},
            boundary=None if boundary.get("estimate") is None else float(boundary["estimate"]),
            boundary_axis=boundary.get("axis"),
            bracket=(float(bracket[0]), float(bracket[1])) if bracket else None,
            strategy=str(discovery.get("strategy", "")),
            evaluations=int(discovery.get("evaluations", 0)),
            severity=float(discovery.get("severity", 0.0)),
            necessary_axes=tuple(discovery.get("necessary_axes", ())),
            git_commit=str(discovery.get("git_commit", "unknown")),
            backend=str(discovery.get("backend", "unknown")),
            notes=str(discovery.get("notes", "")),
            reproduction=dict(payload.get("reproduction", {})),
        )


def save_counterexample(artifact: Counterexample, path: str | Path) -> Path:
    """Write an artifact to YAML, returning the path written."""
    return write_yaml(path, artifact.to_dict())


def load_counterexample(path: str | Path) -> Counterexample:
    """Load an artifact from YAML or JSON."""
    return Counterexample.from_dict(load_yaml(path))
