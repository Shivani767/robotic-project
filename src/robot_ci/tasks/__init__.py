"""Task definitions and the shared success criteria."""

from robot_ci.tasks.base import (
    Phase,
    StepOutcome,
    SuccessCriteria,
    Task,
    TaskProgress,
    TaskStatus,
    create_task,
)
from robot_ci.tasks.manipulation import (
    TOP_DOWN,
    GraspObjectTask,
    PickAndPlaceTask,
    PushObjectTask,
)

__all__ = [
    "TOP_DOWN",
    "GraspObjectTask",
    "Phase",
    "PickAndPlaceTask",
    "PushObjectTask",
    "StepOutcome",
    "SuccessCriteria",
    "Task",
    "TaskProgress",
    "TaskStatus",
    "create_task",
]
