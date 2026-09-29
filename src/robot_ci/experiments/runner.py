"""The experiment runner.

An experiment runs one policy over a grid of conditions, several times each,
and records everything needed to reproduce and audit the result.

Fairness guarantees
-------------------
* Both policies see **byte-identical scenarios**: the same condition object is
  re-derived per policy from the same seed, never mutated in place.
* The backend, task definition, metric code and evaluation criteria are shared.
  Only the policy differs.
* Every episode records the scenario's ``config_hash``, so a result can always
  be traced back to the exact conditions that produced it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_ci.core.errors import EvaluationError
from robot_ci.core.logging import JsonlLogger, get_logger
from robot_ci.metrics import EpisodeTrace, MetricValue, compute_episode_metrics
from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
from robot_ci.policies import RobotPolicy
from robot_ci.simulation import SimulatorBackend
from robot_ci.tasks import Task, TaskStatus, create_task

__all__ = ["EpisodeResult", "RunSpec", "run_episode"]

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class RunSpec:
    """One cell of the experiment grid.

    Attributes:
        scenario: The condition to run.
        policy: The policy to run.
        seed: The episode seed.
        index: Position in the full run order, for stable logging.
    """

    scenario: ScenarioConfig
    policy: PolicyConfig
    seed: int
    index: int = 0

    @property
    def run_id(self) -> str:
        """A unique, sortable identifier for this run."""
        return f"{self.scenario.scenario_id}__{self.policy.name}__s{self.seed}"


@dataclass
class EpisodeResult:
    """Everything one episode produced.

    Attributes:
        run_id: Unique run identifier.
        experiment_id: Owning experiment.
        scenario_id: Condition identifier.
        policy: Policy name.
        seed: Episode seed.
        config_hash: Hash of the scenario's experimental conditions.
        status: Terminal task status.
        success: Whether the task succeeded.
        steps: Control steps taken.
        metrics: The measured metric vector.
        conditions: Flat condition summary, for conditional analysis.
        git_commit: Commit the run was executed at, when available.
        backend: Simulator backend name.
        duration_wall_s: Wall-clock duration, for throughput reporting.
    """

    run_id: str
    experiment_id: str
    scenario_id: str
    policy: str
    seed: int
    config_hash: str
    status: str
    success: bool
    steps: int
    metrics: list[MetricValue] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)
    git_commit: str = "unknown"
    backend: str = "unknown"
    duration_wall_s: float = 0.0

    def metric(self, name: Any) -> float:
        """Return one metric by name, or NaN when it was not measured.

        Args:
            name: A :class:`~robot_ci.metrics.MetricName`.

        Returns:
            The value, or ``float("nan")``.
        """
        for entry in self.metrics:
            if entry.name is name:
                return entry.value
        return float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ``summary.json`` and the database."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "policy": self.policy,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "status": self.status,
            "success": self.success,
            "steps": self.steps,
            "backend": self.backend,
            "git_commit": self.git_commit,
            "duration_wall_s": round(self.duration_wall_s, 4),
            "conditions": dict(self.conditions),
            "metrics": {m.name.value: m.value for m in self.metrics},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeResult:
        """Rebuild from :meth:`to_dict` output (regression CLI, notebooks)."""
        from robot_ci.metrics import MetricName, MetricValue

        metrics = []
        for name, value in dict(payload.get("metrics", {})).items():
            try:
                metric_name = MetricName(name)
            except ValueError:
                continue
            metrics.append(MetricValue(name=metric_name, value=float(value)))
        return cls(
            run_id=str(payload.get("run_id", "")),
            experiment_id=str(payload.get("experiment_id", "")),
            scenario_id=str(payload.get("scenario_id", "")),
            policy=str(payload.get("policy", "")),
            seed=int(payload.get("seed", 0)),
            config_hash=str(payload.get("config_hash", "")),
            status=str(payload.get("status", "")),
            success=bool(payload.get("success", False)),
            steps=int(payload.get("steps", 0)),
            metrics=metrics,
            conditions=dict(payload.get("conditions", {})),
            git_commit=str(payload.get("git_commit", "unknown")),
            backend=str(payload.get("backend", "unknown")),
            duration_wall_s=float(payload.get("duration_wall_s", 0.0)),
        )


def _git_commit() -> str:
    """Return the current git commit, or ``"unknown"`` outside a repository."""
    import subprocess
    from pathlib import Path

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git optional
        return "unknown"


def run_episode(
    backend: SimulatorBackend,
    policy: RobotPolicy,
    scenario: ScenarioConfig,
    *,
    seed: int,
    experiment_id: str,
    log: JsonlLogger | None = None,
) -> EpisodeResult:
    """Run one policy through one scenario once.

    Args:
        backend: A simulator backend. The runner owns its lifetime.
        policy: The policy to evaluate; it is reset before use.
        scenario: The experimental condition.
        seed: Episode seed.
        experiment_id: Owning experiment, for logging.
        log: Optional JSONL research log.

    Returns:
        The episode's :class:`EpisodeResult`.

    Raises:
        EvaluationError: If the backend cannot run the scenario.
    """
    started = time.perf_counter()
    task: Task = create_task(scenario)
    trace = EpisodeTrace()

    handle = backend.reset(scenario, seed=seed)
    policy.reset(policy.config, seed, scenario)
    provider = backend.jacobian_provider(handle)
    if hasattr(policy, "attach_planner"):
        policy.attach_planner(provider)

    status = TaskStatus.RUNNING
    outcome_message = ""
    collision_steps = 0
    confidences: list[float] = []
    sigmas: list[float] = []
    last_observation = None
    last_truth = None

    try:
        for _ in range(scenario.task.max_steps):
            observation = backend.observe(handle)
            if not scenario.sensors.is_nominal:
                from robot_ci.sensors import apply_corruption

                observation, _ = apply_corruption(
                    observation,
                    scenario.sensors,
                    seed=seed,
                    step=observation.step_index,
                )
            if log is not None:
                log.event(
                    "observe",
                    step=observation.step_index,
                    ee=[round(float(v), 4) for v in observation.robot.end_effector_pose.position.array],
                )
            last_observation = observation

            planner_start = time.perf_counter()
            output = policy.act(observation)
            trace.planning_time_ms += (time.perf_counter() - planner_start) * 1000.0

            transition = backend.step(handle, output.action)
            last_truth = transition.ground_truth
            trace.actions.append(output.action)
            trace.ee_path.append(
                np.asarray(transition.robot_state.end_effector_pose.position.array, dtype=np.float64)
            )
            target_object = transition.ground_truth.object_by_id(task.target_object_id())
            if target_object is not None:
                trace.object_path.append(
                    np.asarray(target_object.pose.position.array, dtype=np.float64)
                )
            best = observation.best_object
            if best is not None:
                confidences.append(best.confidence)
                sigmas.append(best.position_sigma_m)
            commanded = output.diagnostics.get("waypoint_target")
            if commanded is not None:
                trace.commanded_poses.append(np.asarray(commanded, dtype=np.float64))

            if transition.collision:
                collision_steps += 1

            result = task.evaluate(
                observation,
                transition.ground_truth,
                transition.robot_state,
                collision=transition.collision,
            )
            status = result.status
            outcome_message = result.message
            if log is not None:
                log.event(
                    "step",
                    step=observation.step_index,
                    action=output.action.type.value,
                    phase=result.phase.value,
                    status=status.value,
                    collision=transition.collision,
                    satisfied={k: bool(v) for k, v in result.satisfied.items()},
                )
            if status is not TaskStatus.RUNNING or transition.terminated:
                break
    finally:
        backend.close(handle)

    if last_truth is None:  # pragma: no cover - only if the loop never ran
        raise EvaluationError(
            f"episode {scenario.scenario_id} produced no transition",
            code="evaluation.empty_episode",
        )
    del last_observation

    duration = handle.sim_time_s
    perception_confidence = float(np.mean(confidences)) if confidences else 0.0
    pose_sigma = float(np.mean(sigmas)) if sigmas else 0.0
    sensor_consistency = _sensor_consistency(sigmas)

    metrics = compute_episode_metrics(
        scenario,
        task,
        trace,
        last_truth,
        status=status,
        steps=handle.step_index,
        duration_s=duration,
        collision_steps=collision_steps,
        perception_confidence=perception_confidence,
        pose_sigma_m=pose_sigma,
        sensor_consistency=sensor_consistency,
    )

    if log is not None:
        log.event(
            "episode.finish",
            status=status.value,
            success=status is TaskStatus.SUCCESS,
            steps=handle.step_index,
            message=outcome_message,
        )

    return EpisodeResult(
        run_id=f"{scenario.scenario_id}__{policy.config.name}__s{seed}",
        experiment_id=experiment_id,
        scenario_id=scenario.scenario_id,
        policy=policy.config.name,
        seed=seed,
        config_hash=scenario.config_hash(),
        status=status.value,
        success=status is TaskStatus.SUCCESS,
        steps=handle.step_index,
        metrics=metrics,
        conditions=scenario.describe_conditions(),
        git_commit=_git_commit(),
        backend=backend.name,
        duration_wall_s=time.perf_counter() - started,
    )


def _sensor_consistency(sigmas: list[float]) -> float:
    """Turn a belief-uncertainty trace into a [0, 1] consistency score.

    A stable, confident belief scores near 1. The mapping is
    ``1 / (1 + sigma / reference)`` with a 10 mm reference, which is roughly
    the grasp tolerance: at 10 mm of uncertainty a grasp is marginal, and the
    score should be 0.5 there.
    """
    if not sigmas:
        return 0.0
    mean_sigma = float(np.mean(sigmas))
    return float(1.0 / (1.0 + mean_sigma / 0.010))
