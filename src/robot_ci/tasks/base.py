"""The Task interface.

A task converts *what the robot did* into *whether it succeeded*, using
ground truth. That is the only place in the framework where ground truth is
read during an episode, and it is deliberately separated from the policy: the
policy never sees the :class:`~robot_ci.models.state.GroundTruthState` this
module consumes.

Design rules
------------
* A task is a pure evaluator plus a waypoint generator. It never acts.
* Success criteria are expressed once, in :class:`SuccessCriteria`, and shared
  across tasks so that a "success" means the same thing everywhere.
* Termination is explicit. A task ends when its criteria are met, when a
  budget is exhausted, or when a safety limit is violated -- and it always
  reports *which*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from robot_ci.models.environment import TaskConfig, TaskName
from robot_ci.models.geometry import Pose, Vec3
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import GroundTruthState, Observation, RobotState

__all__ = [
    "Phase",
    "StepOutcome",
    "SuccessCriteria",
    "Task",
    "TaskProgress",
    "TaskStatus",
    "create_task",
]


class Phase(str, Enum):
    """Coarse phase of a manipulation task.

    Used to describe *where* an episode got to, which is what makes a failed
    attempt diagnosable rather than just "failed".
    """

    APPROACH = "approach"
    """Moving the gripper toward the object."""
    DESCEND = "descend"
    """Lowering to grasp height."""
    GRASP = "grasp"
    """Closing the gripper on the object."""
    LIFT = "lift"
    """Raising the object clear of the support surface."""
    TRANSIT = "transit"
    """Moving the grasped object to the target."""
    PLACE = "place"
    """Lowering and releasing at the target."""
    RETREAT = "retreat"
    """Withdrawing the empty gripper."""
    DONE = "done"
    """Terminal."""


class TaskStatus(str, Enum):
    """Why an episode ended."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SAFETY = "safety_violation"
    BUDGET = "step_budget_exhausted"


@dataclass
class StepOutcome:
    """What the task concluded after one step.

    Attributes:
        status: Whether the episode should continue.
        phase: The task's current phase.
        progress: Fraction of the task completed in [0, 1].
        satisfied: Individual success criteria currently met.
        message: Human-readable note for logs and reports.
    """

    status: TaskStatus
    phase: Phase
    progress: float = 0.0
    satisfied: dict[str, bool] = field(default_factory=dict)
    message: str = ""


@dataclass
class TaskProgress:
    """Aggregate state of a running task.

    Attributes:
        steps: Control steps taken so far.
        phase_counts: How many steps were spent in each phase.
        phase: Current phase.
        status: Current status.
    """

    steps: int = 0
    phase: Phase = Phase.APPROACH
    status: TaskStatus = TaskStatus.RUNNING
    phase_counts: dict[Phase, int] = field(default_factory=dict)

    def record(self, phase: Phase) -> None:
        """Count one step spent in ``phase``."""
        self.steps += 1
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        self.phase = phase


@dataclass(frozen=True)
class SuccessCriteria:
    """Quantitative definition of success, shared by every task.

    Attributes:
        position_tolerance_m: Max final object-centre error.
        orientation_tolerance_rad: Max final orientation error.
        require_grasped: Whether the object must be held at some point.
        require_release: Whether the gripper must open to succeed.
        min_lift_height_m: Required clear height above the support surface.
        hold_steps: Steps the object must be held after release.
        max_collisions: Collision count above which the episode is a failure.
    """

    position_tolerance_m: float = 0.05
    orientation_tolerance_rad: float = 0.6
    require_grasped: bool = True
    require_release: bool = True
    min_lift_height_m: float = 0.05
    hold_steps: int = 5
    max_collisions: int = 3

    @classmethod
    def from_config(cls, task: TaskConfig) -> SuccessCriteria:
        """Build criteria from a task configuration."""
        if task.name is TaskName.PICK_AND_PLACE:
            return cls(
                position_tolerance_m=task.pick_and_place.place_position_tolerance_m,
                orientation_tolerance_rad=task.pick_and_place.place_orientation_tolerance_rad,
                require_grasped=True,
                require_release=task.pick_and_place.require_release,
                min_lift_height_m=task.pick_and_place.min_lift_height_m,
                hold_steps=task.pick_and_place.hold_steps,
            )
        return cls(
            position_tolerance_m=task.grasp.success_position_tolerance_m,
            orientation_tolerance_rad=task.grasp.success_orientation_tolerance_rad,
            require_grasped=True,
            require_release=False,
            min_lift_height_m=0.0,
            hold_steps=task.grasp.hold_steps,
        )

    def to_dict(self) -> dict[str, float | int | bool]:
        """Serialize for artifacts."""
        return {
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_rad": self.orientation_tolerance_rad,
            "require_grasped": self.require_grasped,
            "require_release": self.require_release,
            "min_lift_height_m": self.min_lift_height_m,
            "hold_steps": self.hold_steps,
            "max_collisions": self.max_collisions,
        }


class Task(ABC):
    """Base class for task definitions.

    A task is instantiated per episode. It observes (what the policy believes)
    and is told (ground truth), and returns an :class:`StepOutcome`. It never
    issues actions.
    """

    #: Task name this class implements.
    name: TaskName = TaskName.PICK_AND_PLACE

    def __init__(self, scenario: ScenarioConfig) -> None:
        """Bind the task to one scenario.

        Args:
            scenario: The experimental condition. Its target object and target
                zone are what the task reasons about.
        """
        self.scenario = scenario
        self.criteria = SuccessCriteria.from_config(scenario.task)
        self.progress = TaskProgress()
        self.collisions = 0
        self.ever_grasped = False
        self.released = False
        self.hold_counter = 0
        self.max_grasped_height_m = 0.0

    # -- required API ------------------------------------------------------
    @abstractmethod
    def evaluate(
        self,
        observation: Observation,
        ground_truth: GroundTruthState,
        robot_state: RobotState,
        *,
        collision: bool,
    ) -> StepOutcome:
        """Assess the episode after one step.

        Args:
            observation: What the policy believes (no ground truth).
            ground_truth: The hidden state. Used only here.
            robot_state: Proprioception.
            collision: Whether the backend reported a contact this step.

        Returns:
            The outcome, including whether the episode should end.
        """

    @abstractmethod
    def waypoints(self, object_pose: Pose) -> list[Vec3]:
        """Return the ordered Cartesian waypoints for a pick-and-place.

        Args:
            object_pose: The object's pose, as *believed* by the policy.

        Returns:
            Waypoints in execution order, ending at the target zone.
        """

    # -- shared helpers ----------------------------------------------------
    def budget_exhausted(self) -> bool:
        """Whether the step or time budget has been spent."""
        return self.progress.steps >= self.scenario.task.max_steps

    def target_object_id(self) -> str:
        """Identifier of the object this task operates on."""
        return self.scenario.target.id

    def target_position(self) -> Vec3:
        """Where the object must end up."""
        return self.scenario.task.target_pose.position

    def track_grasped(self, ground_truth: GroundTruthState) -> None:
        """Update grasp/release bookkeeping from the hidden state."""
        target = ground_truth.object_by_id(self.target_object_id())
        if target is None:
            return
        if target.grasped:
            self.ever_grasped = True
            self.max_grasped_height_m = max(
                self.max_grasped_height_m,
                target.pose.position.z - self.scenario.environment.table_height_m,
            )
            self.released = False
            self.hold_counter = 0
        elif self.ever_grasped and not self.hold_counter:
            self.released = True

    def gripper_is_open(self, robot_state: RobotState) -> bool:
        """Whether the gripper has released."""
        return robot_state.gripper.is_closed(threshold=0.35) is False

    def position_error(self, ground_truth: GroundTruthState) -> float:
        """Distance in metres from the object to the target zone."""
        target = ground_truth.object_by_id(self.target_object_id())
        if target is None:
            return float("inf")
        return target.pose.translation_distance_to(
            Pose.from_position(self.target_position())
        )

    def orientation_error(self, ground_truth: GroundTruthState) -> float:
        """Orientation error in radians between the object and the target."""
        target = ground_truth.object_by_id(self.target_object_id())
        if target is None:
            return float("inf")
        return target.pose.orientation.angular_distance_to(
            self.scenario.task.target_pose.orientation
        )

    def check_success(self, ground_truth: GroundTruthState, robot_state: RobotState) -> dict[str, bool]:
        """Evaluate every success criterion against the hidden state.

        Returns:
            A mapping from criterion name to whether it currently holds.
        """
        position_ok = self.position_error(ground_truth) <= self.criteria.position_tolerance_m
        orientation_ok = (
            self.orientation_error(ground_truth) <= self.criteria.orientation_tolerance_rad
        )
        return {
            "position": position_ok,
            "orientation": orientation_ok,
            "grasped_at_some_point": self.ever_grasped or not self.criteria.require_grasped,
            "released": self.released or not self.criteria.require_release,
            "lifted": (
                self.max_grasped_height_m >= self.criteria.min_lift_height_m
                or not self.criteria.require_grasped
            ),
            "collision_budget": self.collisions <= self.criteria.max_collisions,
        }

    def safety_violation(self, ground_truth: GroundTruthState) -> bool:
        """Whether a hard safety limit has been crossed.

        The current definition is a deep collision penetration, which indicates
        the arm drove into geometry rather than merely touching it.
        """
        return ground_truth.penetration_depth_m > 0.02

    def _finish(
        self,
        ground_truth: GroundTruthState,
        robot_state: RobotState,
    ) -> StepOutcome:
        """Build the terminal outcome once a budget is spent."""
        satisfied = self.check_success(ground_truth, robot_state)
        if all(satisfied.values()):
            return StepOutcome(
                status=TaskStatus.SUCCESS,
                phase=Phase.DONE,
                progress=1.0,
                satisfied=satisfied,
                message="all success criteria satisfied",
            )
        failed = [name for name, ok in satisfied.items() if not ok]
        return StepOutcome(
            status=TaskStatus.FAILURE,
            phase=self.progress.phase,
            progress=0.0,
            satisfied=satisfied,
            message=f"unsatisfied criteria: {', '.join(failed)}",
        )


def create_task(scenario: ScenarioConfig) -> Task:
    """Instantiate the task a scenario asks for.

    Args:
        scenario: The experimental condition.

    Returns:
        A task instance.

    Raises:
        NotImplementedError: If the scenario names a task that has no
            implementation yet. The error is explicit rather than a silent
            fallback, so an unimplemented task can never be mistaken for a
            solved one.
    """
    from robot_ci.tasks.manipulation import PickAndPlaceTask, PushObjectTask

    registry: dict[TaskName, type[Task]] = {
        TaskName.PICK_AND_PLACE: PickAndPlaceTask,
        TaskName.GRASP_OBJECT: PickAndPlaceTask,
        TaskName.PUSH_OBJECT: PushObjectTask,
    }
    task_class = registry.get(scenario.task.name)
    if task_class is None:
        raise NotImplementedError(
            f"task {scenario.task.name.value!r} is not implemented. "
            f"Implemented: {', '.join(sorted(t.value for t in registry))}. "
            "The Task interface is task-agnostic; adding a task means adding a "
            "Task subclass, not changing the framework."
        )
    return task_class(scenario)
