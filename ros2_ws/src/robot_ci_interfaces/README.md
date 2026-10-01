# robot_ci_interfaces -- intentionally empty

This package would hold custom ROS 2 message and service definitions. It is
deliberately not built, because nothing on the graph needs one.

Every message type used by the ROS 2 integration is a standard ROS 2 message:

| topic | type |
|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` |
| `/robot_ci/joint_states` | `sensor_msgs/msg/JointState` |
| `/robot_ci/ee_pose` | `geometry_msgs/msg/PoseStamped` |
| `/robot_ci/tf` | `geometry_msgs/msg/TransformStamped` |
| `/robot_ci/camera/rgb` | `sensor_msgs/msg/Image` (`rgb8`) |
| `/robot_ci/camera/depth` | `sensor_msgs/msg/Image` (`32FC1`) |
| `/robot_ci/camera/segmentation` | `sensor_msgs/msg/Image` (`mono8`) |
| `/robot_ci/contact_state` | `std_msgs/msg/Bool` |
| `/robot_ci/observation` | `std_msgs/msg/String` (canonical JSON `Observation`) |
| `/robot_ci/command` | `std_msgs/msg/String` (canonical JSON `Action`) |
| `/robot_ci/episode_result` | `std_msgs/msg/String` (canonical JSON result) |

The two ROBOT CI models with no standard ROS equivalent -- `Observation` and
`Action` -- travel as canonical JSON in `std_msgs/String`. The JSON is produced
and consumed by `robot_ci.ros2.encoding`, and round-trips through the same
Pydantic models, so it is lossless for the fields that matter and does not
require a second schema to keep in sync.

This is a real trade-off and not a claim that custom messages are wrong. A
custom `robot_ci_msgs/msg/Observation` would give a typed schema and compile-time
checking, at the cost of a package that must be versioned alongside the Python
models and rebuilt on both sides of every upgrade. For an integration whose
purpose is to prove that the experiment still means the same thing across a
transport boundary, reusing the models that already define the experiment is the
safer default: a typed schema that drifts from the Pydantic model would hide
exactly the divergence this work is trying to detect.

Two properties the JSON encoding does keep: the encoding is canonical
(`sort_keys`, fixed separators), so identical observations produce byte-identical
messages and a recorded bag is diffable; and the conversion is unit-tested
without ROS installed, in `tests/unit/test_ros2_encoding.py`.
