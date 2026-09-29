"""Low-level control primitives.

Kept separate from `robot_ci.policies` because these are *embodiment* concerns:
they encode what this gripper physically does with a command, which is a
property of the mechanism rather than of any decision-making layer.
"""

from robot_ci.control.gripper import GripperController, GripperControllerState

__all__ = ["GripperController", "GripperControllerState"]
