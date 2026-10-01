"""Topic and frame names, in one place.

Every name here appears in the documentation, in the launch file and in the
tests, so it is defined once rather than spelled out at each use. Changing a
topic name is then a one-line change plus a test failure, not a silent break of
the documented graph.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CLOCK_TOPIC", "FRAMES", "TOPICS", "Topics"]


@dataclass(frozen=True)
class Topics:
    """The topic graph.

    Standard message types are used throughout so the graph can be inspected
    with ordinary ROS 2 tooling (``ros2 topic echo``, ``rqt_image_view``,
    ``rviz2``) and recorded with ``rosbag2`` without a custom message package.
    The JSON-typed topics carry the ROBOT CI models that have no standard ROS
    equivalent; they are ``std_msgs/String`` holding a canonical JSON encoding,
    not a bespoke message type.
    """

    #: Simulation clock. ``rosgraph_msgs/msg/Clock``. Published from MuJoCo
    #: time, so subscribers using ``use_sim_time`` follow the physics.
    clock: str = "/clock"
    #: MuJoCo joint angles, velocities and efforts.
    #: ``sensor_msgs/msg/JointState``.
    joint_states: str = "/robot_ci/joint_states"
    #: Simulated end-effector pose in the base frame.
    #: ``geometry_msgs/msg/PoseStamped``.
    ee_pose: str = "/robot_ci/ee_pose"
    #: base_link -> end_effector transform. ``geometry_msgs/msg/TransformStamped``.
    ee_tf: str = "/robot_ci/tf"
    #: ``sensor_msgs/msg/Image``, ``rgb8``.
    camera_rgb: str = "/robot_ci/camera/rgb"
    #: ``sensor_msgs/msg/Image``, ``32FC1``, metres.
    camera_depth: str = "/robot_ci/camera/depth"
    #: ``sensor_msgs/msg/Image``, ``mono8`` class ids; only when the scenario
    #: requests segmentation.
    camera_segmentation: str = "/robot_ci/camera/segmentation"
    #: ``std_msgs/msg/Bool``-carrying contact flag published as ``Bool``.
    contact_state: str = "/robot_ci/contact_state"
    #: The full ROBOT CI ``Observation``, canonical JSON in ``std_msgs/String``.
    observation: str = "/robot_ci/observation"
    #: The ROBOT CI ``Action``, canonical JSON in ``std_msgs/String``.
    command: str = "/robot_ci/command"
    #: Per-episode result, canonical JSON in ``std_msgs/String``.
    episode_result: str = "/robot_ci/episode_result"

    def all(self) -> tuple[str, ...]:
        """Return every topic name, for documentation and graph assertions."""
        return (
            self.clock,
            self.joint_states,
            self.ee_pose,
            self.ee_tf,
            self.camera_rgb,
            self.camera_depth,
            self.camera_segmentation,
            self.contact_state,
            self.observation,
            self.command,
            self.episode_result,
        )


#: The single instance used by the nodes and by the tests.
TOPICS = Topics()

#: The simulation clock topic, also accessible without constructing ``Topics``.
CLOCK_TOPIC = TOPICS.clock

#: Frame ids. ``base_link`` is the MuJoCo world/arm mount frame; the scenario
#: generator places the arm there, so the end-effector pose is expressed in it.
FRAMES: dict[str, str] = {
    "base": "base_link",
    "end_effector": "end_effector",
    "camera": "camera_optical_frame",
}
