"""ROS 2 integration for ROBOT CI.

This package is the boundary where the validated MuJoCo experiment is exposed
as a ROS 2 process graph. It does **not** replace anything: the scenario, the
policy, the task, the metrics and the configuration hash are the ones already
in this repository, and the episode is still driven by
:func:`robot_ci.experiments.runner.run_episode`.

What is here
------------
* :mod:`robot_ci.ros2.dependencies` -- honest detection of a usable ROS 2.
  There is no fallback transport; a missing ``rclpy`` is an error, not a stub.
* :mod:`robot_ci.ros2.encoding` -- model/message conversion, importable and
  testable with no ROS installed.
* :mod:`robot_ci.ros2.topics` -- the topic graph, defined once.
* :mod:`robot_ci.ros2.nodes` -- the real ``rclpy`` nodes. Imported lazily, so
  ``import robot_ci`` never requires ROS.
* :mod:`robot_ci.ros2.execution` -- building an episode over the graph and
  writing the execution artifact.

Nothing in this package is imported by the rest of ROBOT CI at module import
time. A checkout with no ROS 2 behaves exactly as it did before this package
existed, which a test asserts.
"""

from __future__ import annotations

from robot_ci.ros2.dependencies import (
    SUPPORTED_DISTROS,
    Ros2Environment,
    detect,
    install_hint,
    require_rclpy,
)
from robot_ci.ros2.topics import CLOCK_TOPIC, FRAMES, TOPICS, Topics

__all__ = [
    "CLOCK_TOPIC",
    "FRAMES",
    "SUPPORTED_DISTROS",
    "TOPICS",
    "Ros2Environment",
    "Topics",
    "detect",
    "install_hint",
    "require_rclpy",
]
