"""Environment and task configuration.

These models, together with :mod:`robot_ci.models.sensors` and
:mod:`robot_ci.models.robot`, fully determine the experimental conditions of an
episode. Nothing about a scenario is implicit: if a value can change behaviour
it lives in one of these objects, and therefore lands in the ``config_hash``.
"""

from __future__ import annotations

from enum import Enum
from typing import Self

import numpy as np
from pydantic import Field, model_validator

from robot_ci.models.base import VersionedModel
from robot_ci.models.geometry import Pose, Vec3

__all__ = [
    "BackgroundKind",
    "EnvironmentConfig",
    "GraspObjectParams",
    "ObjectCategory",
    "PickAndPlaceParams",
    "TaskConfig",
    "TaskName",
]


class TaskName(str, Enum):
    """Task families supported by the architecture.

    Only ``GRASP_OBJECT`` and ``PICK_AND_PLACE`` are executed by the local
    simulator (Phase 2). The remaining names exist so the interface is
    demonstrably task-agnostic, and so a scenario written for one of them
    fails with a clear "not implemented" error rather than a missing-key crash.
    """

    GRASP_OBJECT = "grasp_object"
    """Pick a single object and hold it."""
    PLACE_OBJECT = "place_object"
    """Place a held object at a target pose."""
    PICK_AND_PLACE = "pick_and_place"
    """Grasp an object and place it at a target pose. (Primary task.)"""
    PUSH_OBJECT = "push_object"
    """Push an object to a target region without grasping it."""
    INSPECT_OBJECT = "inspect_object"
    """Move the camera to observe an object from a set of viewpoints."""
    NAVIGATE_AROUND_OBSTACLE = "navigate_around_obstacle"
    """Move the end effector to a goal without colliding with an obstacle."""


class BackgroundKind(str, Enum):
    """Background appearance of the synthetic scene."""

    GRAY = "gray"
    WHITE = "white"
    WOOD = "wood"
    GRID = "grid"
    DARK = "dark"


class ObjectCategory(str, Enum):
    """Object categories, used for conditional analysis by object type."""

    CUBE = "cube"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CAN = "can"
    LIFTER = "lifter"
    """Irregular thin object that is genuinely hard to grasp (Phase 4+)."""


#: Dimensionless shape profile per object category, as multiples of
#: :attr:`ObjectSpec.size_m`. The declared size and the rendered geometry are
#: derived from this one table, so a scenario cannot claim a size the simulator
#: does not draw.
#:
#: Keeping the profile dimensionless is what lets ``size_m`` be the single
#: source of truth: changing a scenario's object size scales the rendered
#: object instead of silently leaving it at a hard-coded size.
CATEGORY_PROFILES: dict[ObjectCategory, tuple[float, float, float]] = {
    ObjectCategory.CUBE: (1.0, 1.0, 1.0),
    ObjectCategory.CYLINDER: (1.0, 1.0, 2.5),
    ObjectCategory.SPHERE: (1.0, 1.0, 1.0),
    ObjectCategory.CAN: (1.0, 1.0, 2.2),
    ObjectCategory.LIFTER: (0.5, 2.5, 3.75),
}


def object_half_extents(category: ObjectCategory, size_m: float) -> tuple[float, float, float]:
    """Return the rendered half-extents for a category at a declared size.

    Args:
        category: The object class.
        size_m: The declared characteristic half-size, in metres.

    Returns:
        Half-extents in metres, as the simulator must draw them.
    """
    return tuple(float(size_m) * k for k in CATEGORY_PROFILES[category])  # type: ignore[return-value]


def object_geom_kind(category: ObjectCategory) -> str:
    """Return the primitive shape name for a category, independent of any engine."""
    if category is ObjectCategory.CYLINDER:
        return "cylinder"
    if category is ObjectCategory.SPHERE:
        return "sphere"
    return "box"



class EnvironmentConfig(VersionedModel):
    """Physical scene description.

    Every environment dimension named in the project brief appears here:
    lighting, table geometry, friction, clutter, obstacles, background and
    gravity. Object placement lives in ObjectSpec.
    """

    lighting_lux: float = Field(default=500.0, ge=0.0, le=100000.0, description="Scene illuminance in lux; 500 is nominal.")
    lighting_uniformity: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0 is perfectly uniform lighting.")
    background: BackgroundKind = Field(default=BackgroundKind.GRAY, description="Background appearance.")
    table_height_m: float = Field(default=0.0, ge=0.0, le=2.0, description="Height of the support surface in metres.")
    table_extent_m: tuple[float, float] = Field(default=(0.50, 0.35), description="Support surface half-extent in (x, y), metres.")
    workspace_center: tuple[float, float] = Field(
        default=(0.0, 0.10),
        description=(
            "Centre of the reachable workspace in (x, y). Objects, obstacles, the "
            "camera aim and the target zone are all placed relative to this "
            "point so the layout stays coherent when the table is resized.\n\n"
            "The default sits 0.60 m in front of the robot base at y=-0.30. That "
            "separation is not cosmetic: objects spawn within 0.20 m of this "
            "point, and an arm base closer than about 0.30 m spawns them inside "
            "its own collision geometry, which then launches them at reset."
        ),
    )
    object_friction: float = Field(default=0.6, ge=0.0, le=2.0, description="Object/table friction coefficient.")
    clutter_count: int = Field(default=0, ge=0, le=64, description="Number of non-task distractor objects.")
    clutter_separation_m: float = Field(default=0.12, gt=0.0, le=2.0, description="Minimum separation between clutter objects.")
    obstacle_count: int = Field(default=0, ge=0, le=32, description="Number of static obstacles in the workspace.")
    obstacle_height_m: float = Field(default=0.12, ge=0.0, le=1.0, description="Obstacle height in metres.")
    gravity_mps2: float = Field(default=9.81, gt=0.0, le=30.0, description="Gravitational acceleration.")

    @property
    def lighting_percent(self) -> float:
        """Lighting relative to the 500 lux nominal, as a percentage."""
        return self.lighting_lux / 5.0

    @property
    def is_low_light(self) -> bool:
        """Heuristic flag for conditional slicing (not a hard physical threshold)."""
        return self.lighting_lux < 150.0

    def with_lighting(self, lux: float) -> EnvironmentConfig:
        """Return a copy at a different illuminance."""
        return self.model_copy(update={"lighting_lux": float(lux)})
class GraspObjectParams(VersionedModel):
    """Success parameters of the ``grasp_object`` task."""

    success_position_tolerance_m: float = Field(default=0.02, gt=0.0, le=0.5, description="Max final object-centre error.")
    success_orientation_tolerance_rad: float = Field(default=0.5, gt=0.0, le=np.pi, description="Max final orientation error.")
    hold_steps: int = Field(default=5, ge=1, le=1000, description="Steps the object must be held without slipping.")


class PickAndPlaceParams(VersionedModel):
    """Success parameters of the ``pick_and_place`` task (the primary task)."""

    place_position_tolerance_m: float = Field(default=0.05, gt=0.0, le=0.5, description="Max placement error.")
    place_orientation_tolerance_rad: float = Field(default=0.6, gt=0.0, le=np.pi, description="Max placement orientation error.")
    min_lift_height_m: float = Field(default=0.05, ge=0.0, le=1.0, description="Required clear height above the support surface.")
    hold_steps: int = Field(default=5, ge=1, le=1000, description="Steps the object must be held after release.")
    require_release: bool = Field(default=True, description="Whether the gripper must open to succeed.")


#: Default height a placed object's centre must reach, in metres.
#:
#: An 8 mm cube resting on a table of height 0 has its centre at 8 mm, so this
#: is the object's half-size. It is a constant rather than a lookup because
#: ``target_pose`` is a model default with no access to the object list; a
#: scenario placing a different object should set ``target_pose`` itself.
_DEFAULT_PLACE_Z = 0.008

#: Default placement target, in metres.
#:
#: Chosen to be on the table and inside the arm's comfortable top-down
#: workspace, not to be maximally far from the object. The previous default of
#: (0.30, 0.30) sat *off* the 0.50 x 0.35 m table and 0.67 m from the base
#: across a 57-degree azimuth change, which forced the wrist to swing ~7 rad of
#: total joint travel to face the target top-down. The arm could only get there
#: by rolling 43 degrees, and since the grasp is rigid the held cube rolled with
#: it: every episode placed correctly within 13 mm and then failed on
#: orientation. A real cell places the goal in the same sector as the pick.
_DEFAULT_PLACE_XY = (0.20, 0.15)


class TaskConfig(VersionedModel):
    """Which task to run, with what success criteria and budget.

    All task parameters live in one model and the active task selects the subset
    that applies. This keeps scenario files readable and enables cross-task
    comparison against a shared definition of success.
    """

    name: TaskName = Field(default=TaskName.PICK_AND_PLACE, description="Task to execute.")
    timeout_s: float = Field(default=30.0, gt=0.0, le=600.0, description="Wall-clock budget; exceeding it is a TIMEOUT failure.")
    max_steps: int = Field(default=300, ge=1, le=100000, description="Control-step budget; exceeding it is a TIMEOUT failure.")
    target_pose: Pose = Field(
        default_factory=lambda: Pose.from_position(
            Vec3.of(_DEFAULT_PLACE_XY[0], _DEFAULT_PLACE_XY[1], _DEFAULT_PLACE_Z)
        ),
        description=(
            "Goal pose for place/inspect/navigate tasks.\n\n"
            "The z component is the height the *object* must end at, not the "
            "height the gripper holds it at. The pick-and-place place waypoint "
            "sets the object down on the support surface, so the default z is "
            "the object's half-size rather than an arbitrary height: with a "
            "mid-air default of 0.1 the criterion could never be satisfied by "
            "a table placement, and every episode failed on position while "
            "visibly placing the object correctly 7 mm from the target in x "
            "and y."
        ),
    )
    grasp: GraspObjectParams = Field(default_factory=GraspObjectParams, description="Grasp task parameters.")
    pick_and_place: PickAndPlaceParams = Field(default_factory=PickAndPlaceParams, description="Pick-and-place parameters.")
    approach_height_m: float = Field(default=0.10, ge=0.0, le=1.0, description="Pre-grasp standoff height above the table.")
    grasp_height_m: float = Field(
        default=0.012,
        ge=0.0,
        le=0.5,
        description=(
            "Clearance of the tool centre point above the object centre when the "
            "gripper closes.\n\n"
            "The Panda cannot grasp top-down: joint4 is limited to one sign, so "
            "the wrist cannot flip (0 of 40 random seeds reach the pose, best "
            "error 31.5 mm). The grasp is therefore a side grasp, and the jaw "
            "centre goes to the object centre rather than above it. The small "
            "positive offset is the measured half-height of the finger pads."
        ),
    )

    @property
    def is_manipulation(self) -> bool:
        """Whether the task involves grasping an object."""
        return self.name in (
            TaskName.GRASP_OBJECT,
            TaskName.PLACE_OBJECT,
            TaskName.PICK_AND_PLACE,
            TaskName.PUSH_OBJECT,
        )

    @property
    def requires_grasping(self) -> bool:
        """Whether success requires the object to be held by the gripper."""
        return self.name in (TaskName.GRASP_OBJECT, TaskName.PICK_AND_PLACE)

    @model_validator(mode="after")
    def _validate_pick_and_place(self) -> Self:
        if self.name is TaskName.PICK_AND_PLACE:
            tol = self.pick_and_place.place_position_tolerance_m
            if tol < self.grasp.success_position_tolerance_m:
                raise ValueError(
                    "place_position_tolerance_m must be at least as loose as "
                    "grasp.success_position_tolerance_m, otherwise no single "
                    "grasp can satisfy both constraints"
                )
        return self
