"""ROS 2 node entry points.

This module holds no behaviour. Each console script calls the matching function
in :mod:`robot_ci.ros2.nodes`, which is where the nodes, the adapters and the
experiment driver live.
"""

from robot_ci.ros2.nodes import main_policy, main_simulation

__all__ = ["main_policy", "main_simulation"]
