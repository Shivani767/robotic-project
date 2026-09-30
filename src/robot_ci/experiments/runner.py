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

import math
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

#: Progress a stalled arm must make, in metres, to have its stall counter reset.
#:
#: Chosen an order of magnitude above the numerical noise of the end-effector
#: position and well below any real motion. A rate-limited arm closing on a
#: waypoint improves the distance every tick by far more than this, so a
#: counter that resets on this threshold never fires on a moving arm.
STALL_PROGRESS_M = 1e-3

#: Control ticks without progress after which an episode is declared stalled.
#:
#: This exists because the alternative was to let a stuck arm run to the step
#: cap. Measured on the nominal condition: a successful episode takes 76 ticks,
#: and a failing one takes the full 300 -- the arm sits at a waypoint it cannot
#: reach, the cursor never advances, and nothing in the loop notices. Two costs,
#: both of them distorting a benchmark rather than merely slowing it down:
#:
#: * Every failure is labelled TIMEOUT, so the failure taxonomy has one entry
#:   that means "something went wrong and we did not look" and nothing that
#:   names the actual cause.
#: * A run's wall-clock is set by its failures, not its successes, which is the
#:   wrong way round for a tool whose purpose is to be run many times.
#:
#: The value is a multiple of the longest approach observed (a nominal
#: approach needs a few tens of ticks), so a slow but converging arm is never
#: cut off -- only one that has stopped moving.
STALL_LIMIT_STEPS = 60


@dataclass
class _StallTracker:
    """Detects an arm that has stopped making progress toward its waypoint.

    The policy reports the waypoint it is driving to and whether it considers
    that waypoint reached, so progress is measured against the arm's own goal
    rather than a fixed point. The counter resets whenever the target changes,
    because a new waypoint is a new problem and inheriting the previous one's
    count would abort an arm that had only just been told to do something else.

    Attributes:
        target: The commanded waypoint the current count refers to.
        best: Smallest distance to ``target`` seen so far.
        ticks: Ticks spent without improving on ``best``.
        stage: The policy's waypoint index the current count refers to.
        stage_ticks: Ticks spent at ``stage`` without advancing.
        triggered: Whether the stall limit has been reached.
    """

    target: np.ndarray | None = None
    best: float = math.inf
    ticks: int = 0
    stage: int | None = None
    stage_ticks: int = 0
    triggered: bool = False

    def update(
        self,
        target: Any,
        distance: float,
        reached: bool,
        stage: Any = None,
    ) -> None:
        """Record one tick of progress toward the commanded waypoint.

        Two independent signals are tracked, because a task can stall while the
        arm is moving and an arm can be motionless while the task is not.

        *Distance to the commanded waypoint* catches an arm that cannot reach
        what it was told to reach. It cannot catch the case that actually
        accounted for every 300-tick failure measured on the nominal condition:
        the arm arrived within tolerance and then sat there, because the grasp
        never registered and the waypoint cursor therefore never advanced. The
        policy reports that never-advancing cursor as ``stage``, so a stage that
        does not move for the limit is a stall regardless of where the arm is.

        Args:
            target: The commanded waypoint, or ``None`` if the policy reported
                none. A tick with no target is not evidence of a motion stall.
            distance: Measured distance from the end-effector to ``target``.
            reached: Whether the policy considers the waypoint reached. Reaching
                it ends any motion stall: the arm is where it was told to be.
            stage: The policy's current waypoint index, or ``None`` if unknown.
        """
        if stage is not None and stage != self.stage:
            self.stage = stage
            self.stage_ticks = 0
        elif stage is not None:
            self.stage_ticks += 1

        if target is None:
            return
        point = np.asarray(target, dtype=np.float64)
        if self.target is None or not np.allclose(point, self.target, atol=1e-6):
            self.target = point.copy()
            self.best = distance
            self.ticks = 0
        elif reached:
            self.best = 0.0
            self.ticks = 0
        elif self.best - distance > STALL_PROGRESS_M:
            self.best = distance
            self.ticks = 0
        else:
            self.ticks += 1

        if self.ticks >= STALL_LIMIT_STEPS or self.stage_ticks >= STALL_LIMIT_STEPS:
            self.triggered = True


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
    stall = _StallTracker()

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
                target_point = np.asarray(commanded, dtype=np.float64)
                trace.commanded_poses.append(target_point)
                end_effector = np.asarray(
                    observation.robot.end_effector_pose.position.array, dtype=np.float64
                )
                stall.update(
                    target_point,
                    float(np.linalg.norm(end_effector - target_point)),
                    bool(output.diagnostics.get("reached_waypoint", False)),
                    stage=output.diagnostics.get("waypoint"),
                )

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
            if stall.triggered:
                # A stall is a failure with a known cause, not a timeout. The
                # distinction is the point: TIMEOUT means the budget ran out
                # without the loop learning why, and reporting a stuck arm as a
                # timeout is how a failure taxonomy ends up with a single entry
                # standing for every cause at once.
                status = TaskStatus.FAILURE
                outcome_message = (
                    f"stalled {stall.ticks} ticks from the commanded waypoint "
                    f"(best {stall.best * 1000.0:.1f} mm)"
                )
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
