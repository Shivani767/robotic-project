"""Drive the Panda through a smooth sweep so the live demo has real motion.

This exists so that opening the browser shows a moving arm without waiting for a
policy. It is a *driver*, not a rendering trick: it publishes position commands
on Gazebo's ``cmd_pos`` topics and the arm moves because Gazebo's physics moves
it. If this process stops, the browser stops moving, because the browser only
ever draws what Gazebo reports.

Amplitudes are chosen to stay inside each joint's URDF limits. That matters:
``joint4``'s range is ``[-3.0718, -0.0698]``, so the neutral pose is roughly
-1.57 rad and a target of 0.0 is not reachable. An earlier verification commanded
0.6 rad at joint4, observed it pinned at -0.0698 and read that as instability; it
was the joint sitting on its upper limit.
"""

from __future__ import annotations

import math
import signal
import time
from dataclasses import dataclass
from xml.etree import ElementTree

from robot_ci.gazebo.live import JOINT_NAMES

__all__ = ["JOINT_TARGETS", "MotionDemo", "joint_limits", "main"]


@dataclass(frozen=True)
class JointTarget:
    """A joint's neutral angle and sweep amplitude, in radians or metres."""

    neutral: float
    amplitude: float
    period: float
    phase: float = 0.0


#: Neutral pose and sweep for each joint. ``joint4`` is centred on its own range
#: (-1.57) rather than on zero, because its limits do not include zero.
JOINT_TARGETS: dict[str, JointTarget] = {
    "joint1": JointTarget(0.0, 0.60, 12.0),
    "joint2": JointTarget(-0.40, 0.30, 9.0, phase=math.pi / 3),
    "joint3": JointTarget(0.0, 0.45, 10.0, phase=math.pi / 2),
    "joint4": JointTarget(-1.57, 0.40, 11.0, phase=math.pi),
    "joint5": JointTarget(0.0, 0.55, 8.0, phase=math.pi / 4),
    "joint6": JointTarget(0.80, 0.50, 13.0, phase=math.pi / 6),
    "joint7": JointTarget(0.0, 0.60, 7.0, phase=math.pi / 5),
    "left_finger": JointTarget(0.02, 0.02, 4.0),
    "right_finger": JointTarget(0.02, 0.02, 4.0),
}


def joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    """Read each joint's travel from the URDF.

    Args:
        urdf_path: Path to the robot URDF.

    Returns:
        A mapping of joint name to ``(lower, upper)``.
    """
    root = ElementTree.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        name = joint.get("name")
        if name:
            limits[name] = (float(limit.get("lower", -3.0)), float(limit.get("upper", 3.0)))
    return limits


class MotionDemo:
    """Publish a bounded sinusoidal sweep to every Panda joint.

    Targets are clamped to each joint's limits before being sent, so the driver
    can never ask for a pose the model cannot reach. A clamped sweep is honest
    about the model; an unclamped one produces a pinned joint that looks like a
    control failure.
    """

    def __init__(
        self,
        *,
        joints: tuple[str, ...] = JOINT_NAMES,
        rate_hz: float = 50.0,
        limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._joints = joints
        self._period = 1.0 / rate_hz
        self._limits = limits or {}
        self._running = True

    def target(self, joint: str, t: float) -> float:
        """Return the clamped command for one joint at time ``t``."""
        spec = JOINT_TARGETS[joint]
        raw = spec.neutral + spec.amplitude * math.sin(2 * math.pi * t / spec.period + spec.phase)
        lower, upper = self._limits.get(joint, (-3.0, 3.0))
        return max(lower, min(upper, raw))

    def run(self, urdf_path: str) -> None:
        """Publish commands until interrupted.

        Args:
            urdf_path: Path to the robot URDF, used for the joint limits.
        """
        self._limits = joint_limits(urdf_path)
        self._publish_forever()

    def _publish_forever(self) -> None:
        """Publish commands on ``gz.transport13`` until stopped."""
        from gz.msgs10.double_pb2 import Double
        from gz.transport13 import Node

        node = Node()
        signal.signal(signal.SIGTERM, lambda *_: self._stop())
        signal.signal(signal.SIGINT, lambda *_: self._stop())

        start = time.time()
        publishers = {
            joint: node.advertise(
                f"/model/panda/joint/{joint}/0/cmd_pos",
                Double,
            )
            for joint in self._joints
        }
        while self._running:
            t = time.time() - start
            for joint in self._joints:
                message = Double()
                message.data = self.target(joint, t)
                if not publishers[joint].publish(message):
                    self._running = False
            time.sleep(self._period)

    def _stop(self) -> None:
        self._running = False


def main() -> int:
    """Run the motion driver: ``python -m robot_ci.gazebo.motion_demo``.

    Returns:
        A process exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Drive the Panda with real Gazebo commands.")
    parser.add_argument("--urdf", default="gazebo/models/panda/panda.urdf")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    args = parser.parse_args()
    MotionDemo(rate_hz=args.rate_hz).run(args.urdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
