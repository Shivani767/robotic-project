"""Launch the ROBOT CI Panda pick-and-place episode over ROS 2.

Start everything with one command::

    ros2 launch robot_ci_bringup panda_pick_place.launch.py \
        policy:=policy_v1 seed:=3 depth_noise_mm:=0.0

Two processes are started:

* ``robot_ci_policy_node`` -- runs the real ROBOT CI policy and publishes
  commands on ``/robot_ci/command``;
* ``robot_ci_simulation_node`` -- owns MuJoCo, publishes state, camera frames
  and the simulation clock, and applies the commands that arrive on the topic.

The scenario is not a file path here. It is built by calling
``robot_ci.ros2.execution.build_scenario`` and handed to both nodes as a
parameter, so the launch cannot drift from the configuration the project's own
experiments use, and both processes provably share one validated object.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

#: Names of the two processes, used as ROS node names too.
POLICY_NODE = "robot_ci_policy"
SIMULATION_NODE = "robot_ci_simulation"


def _build_scenario_json(depth_noise_mm: float) -> str:
    """Build the scenario with the project's own helper and serialize it."""
    from robot_ci.ros2.execution import build_scenario

    noise = None if depth_noise_mm < 0 else depth_noise_mm
    return build_scenario(noise).model_dump_json()


def _nodes(context: object) -> list[Node]:
    """Create both nodes once the launch arguments are resolved.

    An :class:`OpaqueFunction` is required: a ``LaunchConfiguration`` can only
    be resolved against a live launch context, so the scenario cannot be built
    while the launch description is being assembled.

    Arguments are converted to real Python types here rather than being passed
    through as strings. ROS parameters are typed, and a node that declares
    ``depth_noise_mm`` as a DOUBLE rejects the string ``"0.0"`` -- which is what
    the first version of this file did, and the launch failed at startup with
    ``InvalidParameterTypeException``. Resolving here also means the scenario is
    built from the same value that is handed to the node.
    """
    noise = float(LaunchConfiguration("depth_noise_mm").perform(context))
    seed = int(LaunchConfiguration("seed").perform(context))
    policy = LaunchConfiguration("policy").perform(context)
    scenario_json = _build_scenario_json(noise)

    return [
        Node(
            package="robot_ci_adapters",
            executable="robot_ci_policy_node",
            name=POLICY_NODE,
            output="screen",
            parameters=[
                {
                    "policy": policy,
                    "seed": seed,
                    "scenario_json": scenario_json,
                    "use_sim_time": True,
                }
            ],
        ),
        Node(
            package="robot_ci_adapters",
            executable="robot_ci_simulation_node",
            name=SIMULATION_NODE,
            output="screen",
            parameters=[
                {
                    "policy": policy,
                    "seed": seed,
                    "depth_noise_mm": noise,
                    "publish_images": ParameterValue(
                        LaunchConfiguration("publish_images"), value_type=bool
                    ),
                    "image_stride": ParameterValue(
                        LaunchConfiguration("image_stride"), value_type=int
                    ),
                    "command_timeout_s": ParameterValue(
                        LaunchConfiguration("command_timeout_s"), value_type=float
                    ),
                    "use_sim_time": True,
                }
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Build the launch description for the pick-and-place episode."""
    arguments = [
        DeclareLaunchArgument(
            "policy", default_value="policy_v1", description="Policy arm to evaluate."
        ),
        DeclareLaunchArgument("seed", default_value="3", description="Episode seed."),
        DeclareLaunchArgument(
            "depth_noise_mm",
            default_value="0.0",
            description="Depth camera sigma in mm; 8.0 is the known counterexample.",
        ),
        DeclareLaunchArgument(
            "publish_images",
            default_value="true",
            description="Publish RGB and depth frames on the camera topics.",
        ),
        DeclareLaunchArgument(
            "image_stride",
            default_value="1",
            description="Publish a camera frame every Nth control tick.",
        ),
        DeclareLaunchArgument(
            "command_timeout_s",
            default_value="30.0",
            description="Seconds to wait for a command before failing the run.",
        ),
    ]

    # The scenario is resolved here, at launch time, from the same code path the
    # research experiments use, and handed to both nodes as a parameter. It is
    # not a file path each node re-reads, so the two processes are provably
    # running one validated object -- and `scenario_id` is never rewritten to
    # describe the launch, which would move the target object because
    # `scenario_id` seeds scene placement.
    return LaunchDescription(arguments + [OpaqueFunction(function=_nodes)])
