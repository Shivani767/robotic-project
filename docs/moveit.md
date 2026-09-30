# MoveIt 2 and ros2_control

**Evidence: Unavailable on current macOS — and not yet built.**

Verified on this machine: no ROS 2, no MoveIt, no `ros2_control`, and
`ros2_ws/src/` holds three empty directories. **0 files.** Nothing described
here has been written, let alone executed.

This is deliberately a short document. The Isaac page is long because there is
real code to describe; here there is not, and padding it would imply progress
that does not exist.

## The intended split

| Concern | Owner | Why |
|---|---|---|
| Collision checking, path planning | MoveIt 2 | A solved problem with a solver worth reusing |
| Joint-level control, state feedback | ros2_control | Standard hardware abstraction |
| Object pose from sensors | `robot_ci_perception` (ours) | Uncertainty propagation is the research contribution |
| Confidence gating, waypoint sequencing, regression accounting | `robot_ci` (Python) | This is the thing being studied |

The rule that keeps the project honest: **MoveIt 2 plans, robot-ci decides
whether the plan was any good.** MoveIt returning a trajectory is not evidence
that a behaviour is acceptable; only comparing it against the baseline across
conditions is.

## Interface sketch

```cpp
// robot_ci_planning — C++17, MoveIt 2 PlanningScene
moveit_msgs::MoveItErrorCode planToPose(
    const geometry_msgs::PoseStamped& goal,
    robot_ci_interfaces::TaskState& state);
```

`MoveItErrorCode` is returned verbatim into the Python `FailureEvent` taxonomy.
Mapping it to `PLANNING_FAILURE` and to anything else must be a tested
translation, not an `else` branch — an unmapped code is how a planning failure
gets reported as a control failure.

## Prerequisites before this can start

1. ROS 2 Jazzy on Ubuntu 24.04.
2. A URDF/SRDF for the target arm, with a `ros2_control` description.
3. A simulator that publishes TF2 and accepts `FollowJointTrajectory` — Isaac
   with the ROS 2 bridge, or Gazebo.
4. A decision on the Isaac/MuJoCo split, which has not been made. See
   [`isaac.md`](isaac.md): the Isaac adapter is **implemented but unverified**,
   so a MoveIt stack on top of it would be two unverified layers. MuJoCo with
   `ros2_control` is the lower-risk path and should probably go first.

## The honest summary

Phases 7 and 8 are **not started**. The `ros2_ws/src/` skeleton exists and is
empty. Nothing here should be cited as a capability of this project.
