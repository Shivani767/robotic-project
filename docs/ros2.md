# ROS 2 integration

**Evidence: Unavailable on current macOS — and not yet built.**

Two separate facts, both verified on this machine:

1. **No ROS 2 here.** `ros2` and `colcon` are not on PATH and `/opt/ros` does
   not exist. ROS 2 Jazzy targets Ubuntu 24.04; it has no macOS build, so
   nothing in this document could be executed here even if it were written.
2. **Nothing has been written.** `ros2_ws/src/` contains three **empty**
   directories — `robot_ci_interfaces`, `robot_ci_adapters`, `robot_ci_bringup`.
   Verified: 0 files in each. There are no `package.xml`, no `CMakeLists.txt`, no
   `.msg` definitions, no C++ sources.

So this is a design document, not a description of working code. It records what
the integration should be so the next person does not have to re-derive it, and
it is explicit that none of it exists.

## Why the Python side needs no ROS 2

The research engine depends on `robot_ci.simulation.base` and nothing else. A
ROS 2 integration therefore does not require ROS 2 to run experiments: the
MuJoCo, mock and synthetic backends cover the whole framework, and that is how
every result in this repository was produced.

ROS 2 is needed only for the *real robot* path: TF2 frames, ros2_control joint
interfaces, and a MoveIt 2 planning pipeline. That is what `robot_ci_adapters`
would provide.

## Intended package layout

```
ros2_ws/src/
  robot_ci_interfaces/     msg, srv, action definitions
  robot_ci_perception/     C++17: depth -> object pose
  robot_ci_sensor_fusion/  C++17: probabilistic fusion
  robot_ci_planning/       C++17: IK and planning against MoveIt 2
  robot_ci_execution/      C++17: ros2_control trajectory execution
  robot_ci_simulation/     launch files for a simulated arm
  robot_ci_adapters/       bridge: ROS 2 <-> robot_ci.simulation backends
  robot_ci_bringup/        top-level launch and parameter files
```

Only the three empty directories exist. The rest are proposals.

### Messages that would be needed

```
# robot_ci_interfaces/msg
ObjectEstimate      id, position, orientation, covariance[36], confidence,
                    source, stamp          # mirrors models.state.ObjectState
RobotState          joint positions/velocities/efforts, ee pose, gripper
TaskState           name, phase, waypoint index/count, satisfied[], complete
FailureEvent        category, weight, evidence[], step
RegressionFinding   condition, baseline, candidate, delta, ci, severity
```

`srv`: `RunEpisode` (scenario + seed -> summary), `SetPolicy` (name, params).
`action`: `RunSweep` (grid -> per-cell progress, so a sweep reports as it runs
rather than at the end).

The alignment point matters: `ObjectEstimate` and `RobotState` are *translations*
of `robot_ci.models.state`, not redefinitions. The Python side remains the
authority on the schema, and a round-trip test should be the first thing written
in the ROS package.

### Where the boundary goes

```
 robot_ci (Python)                  ROS 2 (C++17)
 -----------------                  ----------------
 Observation  ──translate──►   sensor_msgs/Image, JointState
 Action      ◄─translate──   trajectory_msgs/JointTrajectory
 EpisodeResult ──serialize─►  robot_ci_interfaces/TaskState
```

The adapter translates at the boundary and calls the existing
`SimulatorBackend` interface. It does not reimplement anything, and it does not
get a second code path through the research engine.

## What would have to be true before any of this is claimed

1. A ROS 2 Jazzy workspace on Ubuntu 24.04 with an NVIDIA GPU.
2. A C++17 build (`colcon build`) that passes with no warnings under
   `-Wall -Wextra`.
3. A round-trip test proving a Python `Observation` survives translation to
   `sensor_msgs` and back unchanged.
4. `clang-format` and `clang-tidy` clean, and GoogleTest coverage of every node.

Until all four hold, this integration is **unavailable on current hardware**, and
no document in `docs/` should say otherwise.

## The honest summary

Phases 7 and 8 of the plan are **not started**. What exists is a directory
skeleton. The design above is a reasonable starting point, not an achievement.
