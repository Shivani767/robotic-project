"""Gazebo Harmonic as a second simulator backend.

The backend lives in :mod:`robot_ci.simulation.gazebo` and is registered like any
other. This package holds the parts that need no Gazebo runtime -- SDF
generation and condition translation -- so they can be tested on a machine that
has no Gazebo, which is most machines.
"""

from __future__ import annotations

from robot_ci.gazebo.world import (
    GAZEBO_ROBOT_NAME,
    GAZEBO_WORLD_NAME,
    GazeboWorld,
    camera_intrinsics_from_scenario,
    object_pose_from_scenario,
    unsupported_conditions,
    world_sdf,
)

__all__ = [
    "GAZEBO_ROBOT_NAME",
    "GAZEBO_WORLD_NAME",
    "GazeboWorld",
    "camera_intrinsics_from_scenario",
    "object_pose_from_scenario",
    "unsupported_conditions",
    "world_sdf",
]
