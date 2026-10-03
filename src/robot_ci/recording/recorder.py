"""Capture a real experiment as it runs, without inventing anything.

What this module is for
-----------------------
Every other artifact in this project is a number. A number cannot show that a
simulator actually ran, that an arm actually moved, or that the frames in a
video actually came from a camera. This module produces the raw material for
artifacts that can: it drives the *existing* episode loop
(:func:`robot_ci.experiments.runner.run_episode`) with a recording proxy around
the backend and a per-tick hook, and writes down what the simulation emitted.

Two capture points, and why there are two
-----------------------------------------
* **A backend proxy** (:class:`RecordingBackend`) captures the *sensor* stream:
  the RGB and depth frames the simulator rendered, and the robot state it
  reported. It intercepts :meth:`observe` and :meth:`step` and delegates
  everything else, so the runner cannot tell the difference and no behaviour is
  re-implemented. The frame is taken from the *pre-corruption* observation, so
  ``camera_view.png`` is what Gazebo produced rather than what a corruption
  stage later drew on top of it.
* **A step hook** (``_capture_step``) captures the *task* stream: the phase,
  the verdict, the action sent and the ground truth used to score it. Those
  exist only inside the runner's loop, so the runner offers each tick to an
  observer. The hook in ``runner.py`` is the one seam, and it is inert when no
  recorder is attached.

What is deliberately not here
-----------------------------
* **No simulated camera.** If the backend produces no frames, the recording
  says ``camera_frames: 0`` and the composer draws an explicit
  "no camera frame" card. It never draws a plausible-looking placeholder.
* **No trajectory synthesis.** The 3-D view is projected from the *recorded*
  joint states through the *same* URDF the browser viewer uses. If the joints
  did not move, the rendered arm does not move.
* **No status upgrading.** The recorded status is the runner's status. A failed
  episode is a ``FAIL``, and the manifest says so.
"""

from __future__ import annotations

import contextlib
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from robot_ci.core.logging import get_logger
from robot_ci.core.version import SCHEMA_VERSION, SIMULATOR_VERSION, VERSION
from robot_ci.experiments.runner import EpisodeResult, StepRecord, run_episode
from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
from robot_ci.models.state import RobotState, SensorFrame
from robot_ci.policies import create_policy
from robot_ci.simulation import EpisodeHandle
from robot_ci.simulation.base import SimulatorBackend
from robot_ci.tasks import TaskStatus

__all__ = [
    "ExperimentStatus",
    "FrameSample",
    "JointSample",
    "RecordedExperiment",
    "RecordingBackend",
    "RecordingResult",
    "run_recorded_experiment",
]

_LOG = get_logger(__name__)

#: Hard ceiling on stored camera frames, for memory rather than correctness.
#:
#: A frame is kept as a raw ``uint8`` array because the video encoder needs
#: pixels, not a re-decoded JPEG. At the scenario's 160x120 that is 57 kB per
#: frame, so a 300-tick episode is ~17 MB. The ceiling exists for backends that
#: render larger, and when it is reached the recording *subsamples* rather than
#: truncates: a video that stops halfway through an episode is a broken claim,
#: whereas a uniformly subsampled one is still the whole episode at a lower
#: rate. Both counts are recorded so the difference is visible.
MAX_STORED_FRAMES = 900


class ExperimentStatus(str, Enum):
    """The terminal state of one experiment, as reported by the package.

    Deliberately not a boolean. ``NOT_RUN`` and ``NOT_SUPPORTED`` are outcomes
    the reader must be able to tell apart from ``FAIL``: the first means nobody
    tried, the second means the backend cannot, and only ``FAIL`` means the
    robot tried and did not succeed. Collapsing them is how a suite report ends
    up claiming coverage it does not have.
    """

    PASS = "PASS"
    """The task succeeded, by the task's own criteria, on the real backend."""

    FAIL = "FAIL"
    """The task did not succeed. Measured, and reported as such."""

    ERROR = "ERROR"
    """The episode could not be completed: a crash, a timeout or a bad config."""

    NOT_RUN = "NOT_RUN"
    """Configured but never executed; most often interrupted or skipped."""

    NOT_SUPPORTED = "NOT_SUPPORTED"
    """The selected backend genuinely cannot execute this condition."""

    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    """ROBOT CI has no code path for this experiment yet."""

    def is_success(self) -> bool:
        """Whether this state may be counted as a completed success."""
        return self is ExperimentStatus.PASS

    def is_attempt(self) -> bool:
        """Whether the robot actually ran and produced a measurable outcome."""
        return self in (ExperimentStatus.PASS, ExperimentStatus.FAIL)


@dataclass(frozen=True)
class JointSample:
    """One timestamped reading of the robot's own joint state.

    Recorded every control tick from the backend's proprioception, which for
    Gazebo is the simulator's ``joint_state`` topic. Nothing is interpolated.

    Attributes:
        step_index: Control tick this sample belongs to.
        sim_time_s: Simulation clock, in seconds.
        wall_time_s: Wall-clock time, for sync-offset reporting.
        joint_names: Arm joint names, in array order.
        positions: Arm joint angles, in radians.
        end_effector: Measured end-effector position, in metres.
        gripper_width_m: Measured aperture between the fingers.
    """

    step_index: int
    sim_time_s: float
    wall_time_s: float
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    end_effector: tuple[float, float, float]
    gripper_width_m: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ``joint_state.jsonl`` and the metrics figure."""
        return {
            "step": self.step_index,
            "sim_time_s": round(self.sim_time_s, 6),
            "wall_time_s": round(self.wall_time_s, 6),
            "joints": [round(float(v), 6) for v in self.positions],
            "ee": [round(float(v), 6) for v in self.end_effector],
            "gripper_width_m": round(float(self.gripper_width_m), 6),
        }


@dataclass(frozen=True)
class FrameSample:
    """One camera frame the simulator produced, with its own timestamps.

    ``rgb`` is the raw frame from the simulator's sensor topic. It is stored as
    an array rather than a re-encoded image so the encoder and the PNG writer
    both see the same pixels that were captured.

    Attributes:
        step_index: Control tick this frame was observed at.
        sim_time_s: Simulation clock when the frame was read.
        wall_time_s: Wall-clock time when the frame was read.
        rgb: ``uint8`` ``(H, W, 3)`` image, or ``None`` when none arrived.
        depth: ``float32`` ``(H, W)`` depth in metres, or ``None``.
        width_px: Frame width in pixels.
        height_px: Frame height in pixels.
        source: Backend that produced it, e.g. ``"gazebo"``.
        valid: Whether the simulator reported a usable frame.
    """

    step_index: int
    sim_time_s: float
    wall_time_s: float
    rgb: np.ndarray | None
    depth: np.ndarray | None
    width_px: int
    height_px: int
    source: str
    valid: bool

    @property
    def has_image(self) -> bool:
        """Whether a real image is present and non-degenerate."""
        return self.valid and self.rgb is not None and self.rgb.size > 0


@dataclass
class RecordingResult:
    """Bookkeeping from one recording attempt, including partial failure.

    Returned even when the episode raised, because an interrupted recording is
    still something the caller must be able to report as ``ERROR`` rather than
    lose. ``experiment`` holds whatever was captured before the fault, so a
    crash halfway through still leaves the frames that were genuinely recorded.

    Attributes:
        experiment: The recording, or ``None`` when nothing could be captured.
        error: ``""`` on success, else the failure's message.
        error_type: Exception class name, for the manifest.
        partial: Whether samples were captured before the failure.
    """

    experiment: RecordedExperiment | None
    error: str = ""
    error_type: str = ""
    partial: bool = False

    @property
    def ok(self) -> bool:
        """Whether the episode completed and produced a recording."""
        return self.experiment is not None and not self.error


def _run_versions() -> dict[str, str]:
    """Return the runtime versions stamped into every manifest."""
    from robot_ci.core.version import ENVIRONMENT_VERSION

    versions = {
        "robot_ci": VERSION,
        "schema": SCHEMA_VERSION,
        "simulator_contract": SIMULATOR_VERSION,
        "environment": ENVIRONMENT_VERSION,
        "python": platform.python_version(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
    }
    try:
        import subprocess

        out = subprocess.run(
            ["gz", "sim", "--versions"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        lines = out.stdout.strip().splitlines()
        if lines:
            versions["gazebo"] = lines[0].strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - optional
        pass
    return versions


class RecordingBackend(SimulatorBackend):
    """A backend proxy that keeps the frames and states the simulator produced.

    Every attribute other than :meth:`observe` and :meth:`step` is delegated, so
    the runner cannot tell it is wrapped and no backend behaviour is
    re-implemented. This is the same shape as
    :class:`robot_ci.ros2.nodes.PublishingBackend` and the proxy in
    ``experiments/render_episode.py``; the difference is that this one records
    both streams and tags each sample with both clocks.

    The frame is captured from :meth:`observe`, *before* the runner applies
    sensor corruption. That is the point: ``camera_view.png`` is the
    simulator's own output, and a corruption stage cannot change it afterwards.
    """

    def __init__(self, inner: SimulatorBackend) -> None:
        super().__init__()
        self._inner = inner
        self.frames: list[FrameSample] = []
        self.joints: list[JointSample] = []
        self.frames_observed = 0
        self._sample_every = 1

    def __getattr__(self, name: str) -> Any:
        """Delegate every attribute the runner asks for to the real backend."""
        return getattr(self._inner, name)

    @property
    def inner(self) -> SimulatorBackend:
        """The wrapped backend, for the few callers that need it."""
        return self._inner

    def capabilities(self) -> Any:
        """Report the wrapped backend's capabilities."""
        return self._inner.capabilities()

    def reset(self, scenario: ScenarioConfig, *, seed: int | None = None) -> EpisodeHandle:
        """Start a new episode on the wrapped backend."""
        return self._inner.reset(scenario, seed=seed)

    def render(self, handle: EpisodeHandle) -> SensorFrame:
        """Render the current sensor frame from the wrapped backend."""
        return self._inner.render(handle)

    def _joint_sample(
        self, step_index: int, sim_time_s: float, robot: RobotState, wall: float
    ) -> JointSample:
        """Build a joint sample from a backend's proprioception."""
        ee = tuple(float(v) for v in robot.end_effector_pose.position.array)
        return JointSample(
            step_index=step_index,
            sim_time_s=float(sim_time_s),
            wall_time_s=wall,
            joint_names=tuple(robot.joint_names),
            positions=tuple(float(v) for v in robot.joint_positions),
            end_effector=(float(ee[0]), float(ee[1]), float(ee[2])),
            gripper_width_m=float(robot.gripper.width_m),
        )

    def _keep_frame(
        self, frame: SensorFrame | None, step_index: int, sim_time_s: float, wall: float
    ) -> None:
        """Store one frame, honouring the memory ceiling by subsampling."""
        self.frames_observed += 1
        if self.frames_observed % self._sample_every != 0:
            return
        if frame is None:
            rgb = None
            depth = None
            width_px, height_px, camera_id, valid = 0, 0, "", False
        else:
            rgb = frame.rgb
            depth = frame.depth
            width_px, height_px, camera_id, valid = (
                int(frame.width_px),
                int(frame.height_px),
                str(frame.camera_id),
                bool(frame.valid),
            )
        self.frames.append(
            FrameSample(
                step_index=step_index,
                sim_time_s=float(sim_time_s),
                wall_time_s=wall,
                rgb=None if rgb is None else np.asarray(rgb, dtype=np.uint8).copy(),
                depth=None if depth is None else np.asarray(depth, dtype=np.float32).copy(),
                width_px=width_px,
                height_px=height_px,
                source=camera_id,
                valid=valid,
            )
        )

    def live_frame(self, frame: SensorFrame, handle: EpisodeHandle, wall: float) -> None:
        """Record a frame the caller already has, avoiding a second read.

        The 3-D renderer needs the joints from the same instant as the frame, so
        the recorder reads both once per tick and passes the frame here rather
        than letting this proxy fetch it again. Two reads would be two instants
        presented as one.
        """
        self._keep_frame(frame, handle.step_index, float(handle.sim_time_s), wall)

    def observe(self, handle: EpisodeHandle) -> Any:
        """Observe without advancing time; the frame is kept by the recorder."""
        observation = self._inner.observe(handle)
        wall = time.time()
        sim_time = float(handle.sim_time_s)
        self.joints.append(
            self._joint_sample(observation.step_index, sim_time, observation.robot, wall)
        )
        if observation.sensor_frame is not None:
            self._keep_frame(observation.sensor_frame, observation.step_index, sim_time, wall)
        if len(self.frames) > MAX_STORED_FRAMES:
            self._thin()
        return observation

    def step(self, handle: EpisodeHandle, action: Any) -> Any:
        """Advance one control tick, recording joints and frames from the result."""
        transition = self._inner.step(handle, action)
        wall = time.time()
        sim_time = float(handle.sim_time_s)
        self.joints.append(
            self._joint_sample(handle.step_index, sim_time, transition.robot_state, wall)
        )
        # Backends that render only in ``observe`` hand back a transition whose
        # observation carries no frame. Storing that as an empty FrameSample
        # would double the frame list with placeholders half of which claim
        # "no camera"; only a real frame object is a recorded frame.
        frame = getattr(transition.observation, "sensor_frame", None)
        if frame is not None:
            self._keep_frame(frame, handle.step_index, sim_time, wall)
        if len(self.frames) > MAX_STORED_FRAMES:
            self._thin()
        return transition

    def _thin(self) -> None:
        """Halve the stored frames, doubling the sampling interval.

        Subsampling rather than truncating keeps the recording covering the
        whole episode. ``frames_observed`` is untouched, so the manifest can
        report the true capture count against what was kept.
        """
        self.frames = self.frames[::2]
        self._sample_every *= 2


@dataclass
class RecordedExperiment:
    """Everything one real experiment produced, ready to be packaged.

    This is the unit the ZIP is built from. It holds the raw samples, the
    episode's measured metrics, the terminal status and the provenance needed to
    reproduce it. It is assembled only from data that was actually captured: a
    field that could not be measured is empty or absent, never filled in.

    Attributes:
        experiment_id: Stable identifier, also the package directory name.
        category: Suite category key, e.g. ``"nominal_baseline"``.
        title: Human-readable name for the slate and the README.
        condition: Short condition label, e.g. ``"sensor_conditions=nominal"``.
        task: Task name the episode ran.
        policy: Policy name.
        seed: Episode seed.
        status: Terminal status, from :class:`ExperimentStatus`.
        status_detail: One line saying why that status was reached.
        backend: Simulator backend name that ran it.
        scenario_id: The scenario's own identifier.
        config_hash: Hash of the scenario's experimental conditions.
        git_commit: Commit the run happened at.
        command: The exact command that reproduces this package.
        joints: Timestamped joint samples, one per control tick.
        frames: Timestamped camera frames.
        events: Timestamped task events (phase changes, terminal verdict).
        metrics: Measured metrics, by name.
        steps: Control ticks executed.
        duration_sim_s: Elapsed simulation time.
        duration_wall_s: Elapsed wall-clock time.
        started_wall / finished_wall: Unix times bracketing the run.
        frames_observed: Frames the simulator produced, before any subsampling.
        conditions: Flat condition summary from the scenario.
        limitations: Honest statements about what this package does not show.
        notes: Anything a reviewer needs to read the artifacts correctly.
        versions: Runtime versions, for the manifest.
        error: Error text when the episode failed; ``""`` otherwise.
        extra: Category-specific measured payload, e.g. a boundary sweep.
    """

    experiment_id: str
    category: str
    title: str
    condition: str
    task: str
    policy: str
    seed: int
    status: ExperimentStatus
    status_detail: str
    backend: str
    scenario_id: str
    config_hash: str
    git_commit: str
    command: str
    joints: list[JointSample] = field(default_factory=list)
    frames: list[FrameSample] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    duration_sim_s: float = 0.0
    duration_wall_s: float = 0.0
    started_wall: float = 0.0
    finished_wall: float = 0.0
    frames_observed: int = 0
    conditions: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_is_real(self) -> bool:
        """Whether the camera frames came from a real simulator, not a stub.

        True only for a backend that renders physics. A frame from an analytic
        renderer is not a lie, but it is not footage of the robot either, and a
        package that did not distinguish the two would be.
        """
        return self.backend in ("gazebo", "mujoco", "isaac_sim")

    @property
    def frames_with_image(self) -> int:
        """How many stored frames actually carry an image."""
        return sum(1 for frame in self.frames if frame.has_image)

    def joint_motion_rad(self) -> float:
        """Range of joint travel across the episode, the largest single joint.

        This is the measurement that answers "did the robot actually move?".
        Computed from recorded samples, so a still robot reports ~0 and a video
        that claims motion can be checked against it.
        """
        if len(self.joints) < 2:
            return 0.0
        matrix = np.asarray([sample.positions for sample in self.joints], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 0.0
        return float(np.max(np.ptp(matrix, axis=0)))

    def clock_offset_s(self) -> float:
        """Simulation time minus wall-clock time, at the end of the episode.

        Two independent clocks run during a recording: the simulator's and the
        machine's. They cannot be equal -- a simulation steps faster or slower
        than real time -- and the honest thing is to publish the offset rather
        than present one clock as the other. ``None`` becomes ``0.0`` because a
        caller doing arithmetic needs a number, and the manifest carries the
        raw pair as well.
        """
        if not self.joints:
            return 0.0
        real_wall = self.joints[-1].wall_time_s - self.joints[0].wall_time_s
        return float(self.duration_sim_s - real_wall)


def _event(
    record: StepRecord, previous_phase: str | None
) -> tuple[dict[str, Any], str]:
    """Turn one tick into a timeline event, and return the new phase.

    Only *changes* are emitted, plus the terminal verdict. A 300-tick episode
    produces a handful of events rather than 300 rows, which is what makes the
    timeline readable and keeps it a description of transitions rather than a
    second copy of the joint log.
    """
    phase = record.outcome.phase.value
    terminal = record.outcome.status is not TaskStatus.RUNNING
    if phase == previous_phase and not terminal:
        return {}, phase
    # The measured object position is carried on the first event so the 3-D
    # view and the figures can draw the object where the simulation actually
    # put it, rather than at a nominal pose. It is ground truth, recorded as
    # ground truth; it is never fed back into the belief.
    target = record.ground_truth.object_by_id("target_cube")
    position = None if target is None else [
        float(target.pose.position.x),
        float(target.pose.position.y),
        float(target.pose.position.z),
    ]
    return (
        {
            "step": record.step_index,
            "sim_time_s": round(record.sim_time_s, 6),
            "wall_time_s": round(record.wall_time_s, 6),
            "phase": phase,
            "status": record.outcome.status.value,
            "message": record.outcome.message,
            "terminal": bool(terminal),
            "object_position": position,
        },
        phase,
    )


def _status_for(
    result: EpisodeResult | None, error: str, error_type: str
) -> tuple[ExperimentStatus, str]:
    """Map the runner's outcome onto a package status and a one-line reason.

    The mapping is deliberately blunt and total: every path through this
    function either names a measured outcome or names a fault. There is no
    branch that returns ``PASS`` because nothing went wrong.
    """
    if error:
        return ExperimentStatus.ERROR, f"{error_type}: {error}"
    if result is None:  # pragma: no cover - defensive; error is set above
        return ExperimentStatus.NOT_RUN, "no result and no error was recorded"
    if result.success:
        return ExperimentStatus.PASS, f"task succeeded in {result.steps} steps"
    return (
        ExperimentStatus.FAIL,
        f"task reported {result.status!r} after {result.steps} steps",
    )


def run_recorded_experiment(
    *,
    backend: SimulatorBackend,
    scenario: ScenarioConfig,
    policy_config: PolicyConfig,
    seed: int,
    experiment_id: str,
    category: str,
    title: str,
    condition: str,
    command: str,
    limitations: Sequence[str] = (),
    notes: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> RecordingResult:
    """Run one real episode on ``backend`` and return everything it produced.

    The episode goes through :func:`robot_ci.experiments.runner.run_episode`, the
    same loop every measured number in this project comes from, with a recording
    proxy around the backend and a hook on each tick. Nothing about the episode
    is re-implemented, so a video cannot show a behaviour the task did not
    score.

    Args:
        backend: The real simulator backend. Its lifetime is owned here.
        scenario: The experimental condition.
        policy_config: Which policy to run.
        seed: Episode seed.
        experiment_id: Identifier for the package.
        category: Suite category key.
        title: Human-readable title for the slate and README.
        condition: Short condition label.
        command: Exact command that reproduces this package.
        limitations: Honest statements about what this package does not show.
        notes: Reading instructions for the artifacts.
        extra: Category-specific measured payload to carry into ``results.json``.

    Returns:
        A :class:`RecordingResult`. The episode's own failure is reported as
        ``ERROR`` rather than raised, so a suite run can record one experiment
        crashing and continue to the next.
    """
    proxy = RecordingBackend(backend)
    policy = create_policy(policy_config, scenario)
    events: list[dict[str, Any]] = []
    phase: str | None = None
    started = time.time()
    result: EpisodeResult | None = None
    error = ""
    error_type = ""

    def _hook(record: StepRecord) -> None:
        nonlocal phase
        event, phase = _event(record, phase)
        if event:
            events.append(event)

    try:
        result = run_episode(
            proxy,
            policy,
            scenario,
            seed=seed,
            experiment_id=experiment_id,
            step_hook=_hook,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        error_type = type(exc).__name__
        _LOG.warning("recording failed for %s: %s", experiment_id, error)
    finally:
        with contextlib.suppress(Exception):
            backend.close_all()

    finished = time.time()
    status, detail = _status_for(result, error, error_type)
    experiment = RecordedExperiment(
        experiment_id=experiment_id,
        category=category,
        title=title,
        condition=condition,
        task=scenario.task.name.value,
        policy=policy_config.name,
        seed=int(seed),
        status=status,
        status_detail=detail,
        backend=str(getattr(backend, "name", "unknown")),
        scenario_id=scenario.scenario_id,
        config_hash=scenario.config_hash(),
        git_commit="" if result is None else result.git_commit,
        command=command,
        joints=proxy.joints,
        frames=proxy.frames,
        events=events,
        metrics={} if result is None else {m.name.value: float(m.value) for m in result.metrics},
        steps=0 if result is None else int(result.steps),
        duration_wall_s=finished - started,
        started_wall=started,
        finished_wall=finished,
        frames_observed=proxy.frames_observed,
        conditions=dict(scenario.describe_conditions()),
        limitations=list(limitations),
        notes=list(notes),
        versions=_run_versions(),
        error=error,
        extra=dict(extra or {}),
    )
    if result is not None:
        from robot_ci.metrics import MetricName

        experiment.duration_sim_s = float(result.metric(MetricName.COMPLETION_TIME_S))
    return RecordingResult(
        experiment=experiment,
        error=error,
        error_type=error_type,
        partial=bool(error and (proxy.joints or proxy.frames)),
    )
