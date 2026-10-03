# Gazebo Harmonic as a second simulator backend

This document describes what has actually been built and run, and -- just as
importantly -- what has not. Every claim carries one of four labels:

| Label | Meaning |
|---|---|
| **VERIFIED LIVE** | Executed on a real runtime; the output is committed. |
| **IMPLEMENTED BUT NOT VERIFIED** | Code exists, not run. |
| **UNAVAILABLE** | Cannot run here; the blocker is named. |
| **NOT APPLICABLE** | Out of scope for this backend. |

## 1. Environment (measured, not assumed)

| Property | Value | Established by |
|---|---|---|
| OS (host) | macOS 26.5, `arm64` | `sw_vers`, `uname -m` |
| ROS 2 (host) | **UNAVAILABLE** | `which ros2` -> not found |
| Gazebo (host) | **UNAVAILABLE** | `which gz` -> not found; no gazebo brew formulae |
| ROS 2 (container) | Jazzy, Ubuntu 24.04 | `docker/ros2_gazebo.Dockerfile` |
| Gazebo (container) | Harmonic **8.15.0** | `gz sim --versions` |
| Python / MuJoCo | 3.12.3 / 3.14.0 | container |

**Why Jazzy and not Harmonic's own image.** `pyproject.toml` pins
`requires-python = ">=3.11,<3.13"`. `osrf/ros:harmonic` is Ubuntu 22.04 /
CPython 3.10 and cannot import this package. Jazzy is Ubuntu 24.04 / CPython
3.12, and `ros-jazzy-ros-gz` installs Gazebo Harmonic (gz-sim8) as the default.

## 2. Architecture

```mermaid
flowchart TD
    RC["ROBOT CI<br/>run_episode / policy / metrics"] --> AD["GazeboBackend<br/>(SimulatorBackend)"]
    AD --> SDF["robot_ci.gazebo.world<br/>deterministic SDF"]
    AD --> GZT["gz.transport13<br/>Python bindings"]
    GZT -->|joint commands| GZ["Gazebo Harmonic<br/>gz sim -s (headless)"]
    GZ -->|clock / pose / image / depth| GZT
    GZT --> RC
    GZ --> P["Panda + table + object + camera"]
    GZ -.->|ros_gz_bridge| ROS["ROS 2 graph"]
```

Observation: Gazebo -> gz-transport -> `Observation` -> policy.
Command: policy -> `Action` -> gz-transport -> Gazebo joints.

## 3. What exists

| Component | File | Status |
|---|---|---|
| Deterministic SDF world generator | `src/robot_ci/gazebo/world.py` | **VERIFIED LIVE** (loads, 0 errors) |
| Placement from `(seed, scenario_id)` | `object_pose_from_scenario` | **VERIFIED LIVE** (unit tested) |
| Camera intrinsics -> Gazebo FOV (radians) | `_camera_sensor_xml` | **VERIFIED LIVE** (fixed a real load failure) |
| Depth noise / clipping translation | `_camera_sensor_xml` | **VERIFIED LIVE** (unit tested) |
| Unreproducible-condition reporting | `unsupported_conditions` | **VERIFIED LIVE** (unit tested) |
| Unit tests | `tests/unit/test_gazebo_world.py` | **VERIFIED LIVE** (14 passed) |
| ROS 2 Jazzy + Gazebo Harmonic image | `docker/ros2_gazebo.Dockerfile` | **VERIFIED LIVE** (builds; gz 8.15.0) |
| **Panda URDF** | `gazebo/models/panda/panda.urdf` | **VERIFIED LIVE** -- loads into Gazebo with 0 errors |
| Joint command topic for `joint1` | via `JointPositionController` | **VERIFIED LIVE** -- `/model/panda/joint/joint1/0/cmd_pos` exists |
| Simulation clock | Gazebo `/clock` | **VERIFIED LIVE** -- advances |
| Command topics for the other 8 joints | 9x `JointPositionController` | **VERIFIED LIVE** -- all nine exist and move the arm |
| Joint state feedback topic | `JointStatePublisher` | **VERIFIED LIVE** -- `/world/.../model/panda/joint_state` |
| Control -> motion measurement | `scripts/gazebo_motion_test.sh` | **VERIFIED LIVE** -- see `evidence_joint_motion.txt` |
| Live browser viewer | `docker compose up` | **VERIFIED LIVE** -- `evidence_live_viewer.txt` |
| Robot drawn by the Gazebo camera | camera pose change | **VERIFIED LIVE** -- see section 6 |
| `GazeboBackend(SimulatorBackend)` | `src/robot_ci/simulation/gazebo.py` | **VERIFIED LIVE** -- resets, ticks, observes |
| Gazebo PickAndPlace episodes | `scripts/gazebo_smoke_episode.py` | **VERIFIED LIVE** -- 300 real ticks, measured FAIL |
| Depth -> object belief | `GazeboBackend._detect_objects` | **VERIFIED LIVE** -- 15 mm from ground truth |
| Recorded video / images / ZIPs | `robot_ci.recording` | **VERIFIED LIVE** -- see `docs/recording.md` |
| Gazebo launch file | -- | **NOT YET WRITTEN** |

## 3a. The Panda model (VERIFIED LIVE)

`gazebo/models/panda/panda.urdf` is a Franka Emika Panda whose kinematics are
taken from the MuJoCo model this project already validates
(`mujoco/models/franka_emika_panda/panda.xml`), so Gazebo's robot has the same
morphology as the reference rather than an approximation. Joint names are
ROBOT CI's own (`joint1`..`joint7`, `left_finger`, `right_finger`), so the
control path needs no name translation.

Geometry is primitive boxes and cylinders, not the MJCF's STL/OBJ meshes. That
is a recorded trade: the meshes are large binary assets, Gazebo and MuJoCo
collision meshes differ subtly anyway, and a per-link bounding volume is
reproducible and reviewable. It is also a genuine difference between the two
robots and is listed under limitations rather than presented as equivalent.

Three real bugs were found by running Gazebo rather than by reading about it:

1. **Inertia tensors were invalid.** MuJoCo's `fullinertia` ordering is
   `(ixx, iyy, izz, ixy, ixz, iyz)` and the first draft swapped `ixz` with
   `iyz`. Gazebo refused four links with "has invalid inertia". The values are
   now derived from the MJCF programmatically rather than transcribed.
2. **`<include><uri>` is not a filesystem path.** Gazebo resolves it through
   its resource path and answers `Unable to find uri[...]`; it needs a `file://`
   URI, valid *inside the runtime* (which is why a macOS host path fails inside
   the container).
3. **The joint controller must attach to a model, and takes repeated
   `<joint_name>` elements.** In the world it refuses to initialise; inside the
   URDF, `<scoped_joints>` and a `<joint_names>` wrapper both fail with
   "Failed to get any `<joint_name>`".

## 3b. Evidence

`experiments/results/gazebo/evidence_panda_load.txt` and
`evidence_world_load.txt` record the real runs.

## 4. Determinism

* Object placement draws from
  `make_numpy_rng(scenario.seed, "gazebo-scene", scenario.scenario_id)` -- the
  same convention `simulation/mujoco_env.py` uses.
* **`scenario_id` is read, never written.** It participates in placement, so
  renaming a condition to describe a generated artifact moves the target object.
  `test_object_placement_follows_the_scenario_id` pins this.
* Nothing derives randomness from node names, PIDs, entity IDs or wall-clock.
* Physics runs at a fixed 1 ms `max_step_size`, matching the control rate.

**Not claimed:** bit-identical trajectories across MuJoCo and Gazebo. The engines
integrate differently, so the same action sequence yields different joint
trajectories. Cross-simulator *scenario-level* determinism -- same seed to same
placement, same config hash -- is what is claimed and tested.

## 5. Reproducing what is verified

```bash
docker build -f docker/ros2_gazebo.Dockerfile -t robot-ci-ros2-gazebo:jazzy .

# Generate a world from the project's own scenario
python -c "from robot_ci.gazebo import world_sdf; from robot_ci.scenarios import default_scenario; \
           open('/tmp/world.sdf','w').write(world_sdf(default_scenario()))"

# Load it in headless Gazebo and list what it publishes
docker run --rm -v /tmp:/gz robot-ci-ros2-gazebo:jazzy bash -lc \
  'export PATH=/opt/ros/$ROS_DISTRO/bin:$PATH; source /opt/ros/$ROS_DISTRO/setup.bash;
   (timeout 15 gz sim -s -r /gz/world.sdf &); sleep 12; gz topic -l'
```

## 6. The robot IS drawn by the current camera (VERIFIED LIVE)

**Correction to an earlier finding.** The Panda appears in Gazebo's own camera
image with the nine joint controllers active. This supersedes the previous
version of this section, which recorded the robot as missing: that measurement
was taken while the camera sat at `0 {cy-0.35} {table_z+0.45} 0 0.5 0`, and it
was real *for that pose*. Re-aiming the camera at the table with an explicit
look-at (`_camera_pose`) moved the arm back into frame.

The current frame at 160x120, measured from a saved depth array:

| content | evidence |
|---|---|
| table, target zone, red cube | visible in `camera_view.png` of every package |
| a Panda link and the gripper | a 1958-point cluster, 0.23 m across, 0.04-0.11 m above the table |
| the whole robot | **no** -- only the part the look-at allows; the wrist is past the top of frame |

Both the robot's own links and the target object are therefore in the same
point cloud, which is what made the perception problem below interesting: a
detector that simply took the largest cluster returned the arm.

**The control plugins and the rendering coexist.** This is the configuration the
brief asked to preserve: nine `JointPositionController` instances inside the
robot model, the arm commanded and moving, and the camera still rendering.

### What the earlier, incorrect measurement still tells us

The original characterisation was not a bug, it was a camera pose. Changing one
variable at a time and counting finite depth pixels (the table, object and zone
contribute 2144) showed, for *that* pose:

| joints | plugins | finite depth px | robot drawn? |
|---|---|---|---|
| yes | no | 3149 | yes |
| no | yes | 2144 | cube still drawn |
| yes | yes | 2144 | no |

The five sub-cases with joints *and* plugins all returned 2144. The conclusion
drawn then -- "this build's Sensors path skips a model that has both joints and
plugins" -- was **wrong**, and the evidence for that is simply that the current
pose renders the robot with joints and plugins together. The pipeline, and the
discipline of recording the number rather than the story, are what caught it.

Full transcript: `experiments/results/gazebo/evidence_robot_not_rendered.txt`
(historical; see the correction above).

## 7. Limitations and blockers

* **The task does not succeed yet.** Every recorded Gazebo episode ends
  `FAIL`. This is a measured outcome, not a missing feature: perception is
  accurate (15 mm), the arm reaches inside the 50 mm grasp radius, and the
  episode still does not place the object. It is reported as `FAIL` in every
  package and in the suite summary, and no `PASS` is claimed for any condition.
* **Grasp is a kinematic latch, not resolved finger contact.** The object
  attaches when the commanded aperture closes within 50 mm of it. This is an
  adapter behaviour, declared as such, and exists so the shared
  PickAndPlace task and policy run unchanged. It is *not* a physics claim.
* **Collision geometry differs from MuJoCo.** The URDF uses per-link primitive
  bounding volumes where the MuJoCo model uses meshes. Contact response is
  therefore *expected* to differ between the two simulators, which is a real
  source of cross-simulator disagreement rather than a defect.
* **Headless only.** Gazebo runs with `-s` (server, no GUI). The GUI is never
  required by an experiment, so this limits interactive inspection only.
* **Sensor parity is partial by construction.** Gazebo has no per-frame
  dropout, occlusion injector or blur stage; `unsupported_conditions` names each
  gap rather than dropping it silently.
* **Self-occlusion is real.** The arm moves into the camera's view as it
  reaches, and detection drops to no-belief on some ticks. That is the honest
  behaviour of a fixed camera, not a bug to suppress.

## 8. Perception: what was wrong, and how it was measured

The detector back-projects every finite depth pixel, fits the support plane,
groups the points above it, and reports the group that could be *held*. Three
findings, each measured against a saved real frame:

1. **Depth is axial, not radial.** Back-projecting under the axial convention
   gives a tabletop whose median height is `-0.0000` m; under the radial
   convention it is `+0.0255` m with a residual spread. The pinhole model in
   the code was right. (`scripts/gazebo_depth_convention.py`.)
2. **Least-squares tilt was the real defect.** Fitting a plane over *all* the
   points returned `|slope| = 0.151` on a genuinely flat table, so "above the
   plane" selected a wedge of the tabletop itself and the detector reported the
   **table** as the object, 492 mm from the cube. Restricting the fit to the
   lower envelope gives `slope = 1.0e-05`, `z0 = -0.0004` m.
3. **"Largest cluster" is wrong once the robot is in frame.** The cloud splits
   into a 56-point cluster on the tabletop (the cube, extent 0.038 m) and a
   1958-point cluster 0.23 m across (a Panda link). Taking the biggest returned
   the arm. Candidates are now filtered by two properties read from the
   scenario and measured frame rather than tuned: resting on the support plane
   (cube 0.0137 m, Panda link 0.0402 m above it) and no wider than the gripper
   can close (`max_gripper_width_m = 0.08`).

Result: belief error fell from **492 mm to 15 mm**, and the arm now reaches
within 0.025 m of the object instead of missing by 0.110 m.

## 9. What remains open

1. **Why the grasp does not latch and complete.** The end effector reaches
   0.025-0.083 m from the object depending on seed, inside the 0.05 m radius,
   and the task still reports failure. Attribution between the latch, the
   gripper command and the task's success criteria is unfinished.
2. **A cross-simulator comparison table.** Gazebo and MuJoCo now both run
   episodes, so the comparison is possible, but it has not been produced.
3. **A Gazebo launch file**, which would remove the ad-hoc process management
   from the backend's startup path.
