"""Structured world state with snapshots.

The point of a world model is not to hold a dictionary. It is to make an episode
*reconstructible*: given what was stored, a reader can rebuild the initial state,
replay what was seen, and arrive at the final state. That requires three things
this module provides and a bare dict does not:

* **Every belief carries its own uncertainty and provenance.** An object seen
  once and an object tracked for two hundred frames are both "a position", and
  only the covariance distinguishes them.
* **Snapshots are immutable and timestamped.** Reconstructing an episode means
  asking what the model believed at step *k*, and a model whose beliefs are
  mutated in place cannot answer that.
* **Staleness is explicit.** An object last seen 200 ticks ago is still in the
  model, and a consumer that cannot tell that from a fresh observation will act
  on a belief the world no longer supports. Everything here carries ``last_seen``
  and :meth:`WorldState.visible_objects` filters on it.

Deliberately not a database. Snapshots live in memory for the length of an
episode and are serialised through the experiment's own artifact writer; a store
with its own persistence would be a second source of truth about results that
are already written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from robot_ci.models.geometry import Covariance, Pose
from robot_ci.models.state import GroundTruthObject, ObjectState, RobotState, Source

__all__ = [
    "EnvironmentState",
    "SensorState",
    "Snapshot",
    "TaskState",
    "WorldModel",
    "WorldState",
]


class EnvironmentState(BaseModel):
    """Static properties of the scene, as the model understands them.

    Attributes:
        lighting_lux: Scene illuminance.
        table_height_m: Height of the support surface.
        clutter_count: Number of distractor objects believed present.
        obstacle_count: Number of obstacles believed present.
        object_friction: Believed friction coefficient.
    """

    schema_version: str = Field(default="1.0.0")
    lighting_lux: float = 500.0
    table_height_m: float = 0.0
    clutter_count: int = 0
    obstacle_count: int = 0
    object_friction: float = 0.6


class SensorState(BaseModel):
    """What the robot believes about its own sensors.

    Attributes:
        position_sigma_m: Reported positional uncertainty of the fused estimate.
        confidence: Reported fusion confidence in [0, 1].
        consistency: Sensor-agreement score in [0, 1], where 1 is perfect.
        degraded: Whether any sensor is believed to be unhealthy. Set from
            confidence dropping below a threshold rather than asserted, so a
            degraded state always has a number behind it.
    """

    schema_version: str = Field(default="1.0.0")
    position_sigma_m: float = 0.0
    confidence: float = 0.0
    consistency: float = 1.0
    degraded: bool = False


class TaskState(BaseModel):
    """Progress through the task.

    Attributes:
        name: Task identifier.
        phase: Current phase, as the task reports it.
        waypoint_index: Waypoint the policy is driving to.
        waypoint_count: Total waypoints in the plan.
        satisfied: Criteria currently met, by name.
        complete: Whether the task has been reported complete.
    """

    schema_version: str = Field(default="1.0.0")
    name: str = ""
    phase: str = ""
    waypoint_index: int = 0
    waypoint_count: int = 0
    satisfied: tuple[str, ...] = ()
    complete: bool = False


@dataclass(frozen=True)
class Snapshot:
    """An immutable record of what the model believed at one step.

    Attributes:
        step: Control step the snapshot belongs to.
        timestamp_s: Simulation time.
        objects: Every object belief, keyed by id, including stale ones. A
            snapshot that dropped them would not be reconstructible.
        robot: Robot state at this step.
        sensors: Sensor self-assessment at this step.
        task: Task progress at this step.
        environment: Scene properties, carried so a snapshot stands alone.
    """

    step: int
    timestamp_s: float
    objects: dict[str, ObjectState]
    robot: RobotState | None
    sensors: SensorState
    task: TaskState
    environment: EnvironmentState

    def object(self, object_id: str) -> ObjectState | None:
        """Belief about one object, or ``None`` if it was never seen."""
        return self.objects.get(object_id)

    def visible_objects(self, max_age_s: float) -> dict[str, ObjectState]:
        """Only the objects seen within ``max_age_s`` of this snapshot."""
        return {
            key: value
            for key, value in self.objects.items()
            if self.timestamp_s - value.last_seen_s <= max_age_s
        }


@dataclass
class WorldState:
    """Current beliefs, plus the snapshot history that produced them.

    Attributes:
        objects: Latest belief per object id.
        robot: Latest robot state.
        sensors: Latest sensor self-assessment.
        task: Latest task progress.
        environment: Scene properties.
        step: Step of the most recent update.
        timestamp_s: Simulation time of the most recent update.
        snapshots: Immutable history, in step order.
        max_snapshots: Retention limit. A long episode would otherwise hold every
            step's full object set in memory; the limit keeps the tail, which is
            the part a diagnosis reads.
    """

    objects: dict[str, ObjectState] = field(default_factory=dict)
    robot: RobotState | None = None
    sensors: SensorState = field(default_factory=SensorState)
    task: TaskState = field(default_factory=TaskState)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    step: int = 0
    timestamp_s: float = 0.0
    snapshots: list[Snapshot] = field(default_factory=list)
    max_snapshots: int = 512

    def visible_objects(self, max_age_s: float = 0.5) -> dict[str, ObjectState]:
        """Objects believed current, by staleness rather than by presence."""
        return {
            key: value
            for key, value in self.objects.items()
            if self.timestamp_s - value.last_seen_s <= max_age_s
        }

    def snapshot_at(self, step: int) -> Snapshot | None:
        """The snapshot for ``step``, if it is still retained."""
        for snapshot in self.snapshots:
            if snapshot.step == step:
                return snapshot
        return None

    def initial(self) -> Snapshot | None:
        """The earliest retained snapshot, which is the episode's opening state."""
        return self.snapshots[0] if self.snapshots else None

    def final(self) -> Snapshot | None:
        """The most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def to_dict(self) -> dict[str, object]:
        """Serialisable summary, for an experiment artifact.

        Snapshots are summarised rather than serialised in full: an artifact
        that embedded every frame's belief arrays would be unreadable and, for
        most questions, unnecessary. The per-step detail belongs in the JSONL
        research log, which exists for that.
        """
        return {
            "step": self.step,
            "timestamp_s": self.timestamp_s,
            "environment": self.environment.model_dump(mode="json"),
            "sensors": self.sensors.model_dump(mode="json"),
            "task": self.task.model_dump(mode="json"),
            "objects": {
                key: {
                    "pose": value.pose.position.array,
                    "max_sigma_m": value.covariance.max_sigma,
                    "confidence": value.confidence,
                    "source": value.source.value,
                    "last_seen_s": value.last_seen_s,
                }
                for key, value in self.objects.items()
            },
            "snapshot_count": len(self.snapshots),
            "snapshots_retained_from": self.snapshots[0].step if self.snapshots else None,
        }


class WorldModel:
    """Accumulates beliefs over an episode and records what it believed.

    Usage is one :meth:`observe` per control step, then :meth:`finalize`. The
    model never reads the simulator's ground truth -- it is fed the same
    observations the policy sees, so anything it believes can in principle be
    believed by a real robot. :meth:`note_ground_truth` records ground truth
    *separately* and only for scoring, which is what keeps the distinction
    between what was believed and what was true.
    """

    def __init__(self, *, max_snapshots: int = 512) -> None:
        """Create an empty model.

        Args:
            max_snapshots: Snapshot retention limit.
        """
        self.state = WorldState(max_snapshots=max_snapshots)
        self._ground_truth: list[dict[str, GroundTruthObject]] = []

    def observe(
        self,
        *,
        step: int,
        timestamp_s: float,
        objects: dict[str, ObjectState],
        robot: RobotState | None = None,
        task: TaskState | None = None,
        sensors: SensorState | None = None,
    ) -> Snapshot:
        """Fold one observation into the model and snapshot the result.

        Args:
            step: Control step.
            timestamp_s: Simulation time.
            objects: Object beliefs from the sensors or the fusion layer.
            robot: Robot state, if available.
            task: Task progress, if available.
            sensors: Sensor self-assessment, if available.

        Returns:
            The snapshot taken after folding the observation in.
        """
        self.state.step = step
        self.state.timestamp_s = timestamp_s
        if robot is not None:
            self.state.robot = robot
        if task is not None:
            self.state.task = task
        if sensors is not None:
            self.state.sensors = sensors
        else:
            # Derive the sensor self-assessment from the objects rather than
            # leaving it at its initial zeros, so `degraded` reflects something
            # a reader can check.
            sigmas = [value.position_sigma_m for value in objects.values()]
            self.state.sensors = SensorState(
                position_sigma_m=float(np.mean(sigmas)) if sigmas else 0.0,
                confidence=float(
                    np.mean([value.confidence for value in objects.values()])
                )
                if objects
                else 0.0,
                degraded=bool(sigmas) and float(np.mean(sigmas)) > 0.01,
            )
        for key, value in objects.items():
            # A belief that has not been seen this step keeps its old timestamp,
            # which is how staleness becomes visible rather than being reset by
            # being copied forward.
            self.state.objects[key] = value
        snapshot = Snapshot(
            step=step,
            timestamp_s=timestamp_s,
            objects=dict(self.state.objects),
            robot=self.state.robot,
            sensors=self.state.sensors,
            task=self.state.task,
            environment=self.state.environment,
        )
        self.state.snapshots.append(snapshot)
        if len(self.state.snapshots) > self.state.max_snapshots:
            del self.state.snapshots[: -self.state.max_snapshots]
        return snapshot

    # -- episode reconstruction -------------------------------------------
    #
    # Delegated so that a WorldModel is the single entry point. A caller that
    # has to know the accessors live on ``.state`` will eventually get it wrong,
    # and "reconstruct the initial state" is exactly the call that must not be
    # subtly mistyped.

    def initial(self) -> Snapshot | None:
        """The earliest retained snapshot: the episode's opening state."""
        return self.state.initial()

    def final(self) -> Snapshot | None:
        """The most recent snapshot."""
        return self.state.final()

    def snapshot_at(self, step: int) -> Snapshot | None:
        """The snapshot for ``step``, if still retained."""
        return self.state.snapshot_at(step)

    def visible_objects(self, max_age_s: float = 0.5) -> dict[str, ObjectState]:
        """Objects believed current, by staleness rather than by presence."""
        return self.state.visible_objects(max_age_s)

    def note_ground_truth(self, truth: dict[str, GroundTruthObject]) -> None:
        """Record what was actually true, for scoring only.

        Kept out of :attr:`state` deliberately. A world model that folded ground
        truth into its beliefs would be indistinguishable from one that works,
        and the difference is exactly what a policy consumes.
        """
        self._ground_truth.append(dict(truth))

    def belief_error(self, object_id: str) -> list[float]:
        """Per-step distance between the belief about an object and the truth.

        Args:
            object_id: Which object to score.

        Returns:
            One error in metres per recorded step. A per-step series rather than
            a single number, because a filter that is accurate on average and
            briefly wild is a different failure from one that is uniformly
            mediocre, and only the series distinguishes them.
        """
        errors: list[float] = []
        for snapshot, truth in zip(self.state.snapshots, self._ground_truth, strict=False):
            belief = snapshot.object(object_id)
            actual = truth.get(object_id)
            if belief is None or actual is None:
                errors.append(float("nan"))
                continue
            errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(belief.pose.position.array, dtype=np.float64)
                        - np.asarray(actual.pose.position.array, dtype=np.float64)
                    )
                )
            )
        return errors

    def track_object(self, object_id: str, pose: Pose, *, covariance: Covariance) -> None:
        """Insert or replace a tracked belief without advancing a step.

        Args:
            object_id: Object being tracked.
            pose: Believed pose.
            covariance: Positional uncertainty.
        """
        self.state.objects[object_id] = ObjectState(
            id=object_id,
            pose=pose,
            covariance=covariance,
            confidence=self.state.sensors.confidence,
            source=Source.FUSION,
            last_seen_s=self.state.timestamp_s,
        )

    def with_environment(self, **changes: object) -> WorldState:
        """Replace environment fields and return the state.

        Uses ``model_copy`` rather than ``dataclasses.replace``: EnvironmentState
        is a pydantic model, and ``replace`` on one is the wrong tool that also
        type-checks badly. Returns the same state object so callers can chain,
        but through a copy so an already-taken snapshot is not retroactively
        rewritten -- that immutability is the whole reason snapshots exist.
        """
        self.state.environment = self.state.environment.model_copy(update=dict(changes))
        return self.state
