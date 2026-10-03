# Part 1 — implementation plan (written before any code change)

Baseline: commit `84bfef9` for the Gazebo work; working tree at `19d3fa3` plus
uncommitted recording changes. Every statement below is a *measurement*, taken
on this machine, not an inference.

## Phase 0 measurements that drive the plan

1. `pytest -q` on the host: **497 passed, 8 skipped** (8 skips are the ROS 2 and
   Gazebo-runtime integration tests, which need the container).
2. A live Gazebo episode (`scripts/gazebo_episode_trace.py`, seed 3, 0 mm) on
   `robot-ci-gazebo-demo:latest`:
   `episode status=failure success=False steps=83`;
   `belief error median 0.0146 m`;
   `closest approach ever reached: 0.1341 m`.
   The arm never reaches the object and the episode stalls.
3. The gripper in Gazebo is a **kinematic latch**: `_update_grasp` sets
   `grasp["held"]=True` whenever the commanded aperture is below 0.03 m and the
   wrist is within 0.05 m of the cube, and `ground_truth()` then *reports the
   object at the end effector plus a stored offset*. No contact is resolved and
   the object is not moved by physics. This is the behaviour Part 1 forbids.
4. The Gazebo scene does not match the scenario it claims to represent:
   * `gazebo/models/.../panda.urdf` draws a **0.04 m (40 mm) cube** and places it
     at `table_z + 0.02`; `scenario.target.size_m` is **0.008 m** (16 mm), and
     MuJoCo draws `object_half_extents(CUBE, 0.008)` = 8 mm half-extent at
     `table_z + 0.008`.
   * the visual `target_zone` is written at `(cx + 0.18, cy - 0.12)`; the task's
     placement goal is `scenario.task.target_pose.position` = `(0.2, 0.15, 0.008)`.
   * the table is a hard-coded `0.6 0.6` box; `environment.table_extent_m` is
     `(0.50, 0.35)`.
   * objects are drawn in a hard-coded `0.14 .. 0.22 m` annulus;
     `environment.object_spawn_radius_m` is `(0.05, 0.13)`, and that upper bound
     is documented as measured-necessary for solvability. The trace object sits
     at 0.218 m from the workspace centre — outside the validated reach.
5. The IK target frame is wrong. MuJoCo injects a `tcp` site at
   `pos="0 0 0.1034"` **inside the `hand` body** and solves IK for it. The Gazebo
   URDF provider (`robot_ci/gazebo/kinematics.py`) stops the chain at
   `panda_link7` — the *wrist*, 0.2104 m behind the pad centre. So the policy
   commands the wrist to where the pads should be, the commanded pose drives the
   hand through the table, and the arm stalls ~0.13 m short. The same mismatch
   makes `RobotState.end_effector_pose` (read from `panda_link7`, with a
   hard-coded identity orientation) disagree with the frame the IK solved for, so
   the policy's `reached` test can never become true and the waypoint cursor
   never advances.
6. `gz.msgs10.contact_pb2`, `gz.msgs10.pose_v_pb2` and `gz.msgs10.image_pb2` all
   import in the image, so a real contact sensor is *implementable*.

## What will change, and why

| # | File | Change | Justified by |
|---|---|---|---|
| 1 | `gazebo/models/panda/panda.urdf` | Symmetric finger origins; pad boxes re-centred so the pad centre sits at hand-frame `z = 0.1029` (the MJCF measurement) with half-height 0.0085 m and closed separation 0.011 m | measurements 4 and 5 |
| 2 | `src/robot_ci/gazebo/world.py` | Object size/mass/z from `scenario`, target zone from `task.target_pose`, table from `table_extent_m`, spawn annulus from `object_spawn_radius_m`, friction from `object_friction` on every surface | measurement 4 |
| 3 | `src/robot_ci/gazebo/kinematics.py` | Chain extended through the fixed `hand_joint` to the TCP site at hand-frame `z = 0.1034`, matching MuJoCo | measurement 5 |
| 4 | `src/robot_ci/simulation/gazebo.py` | `RobotState.end_effector_pose` is the TCP from measured joint angles (proprioceptive FK); the latch is deleted; `ground_truth` reports Gazebo's own object pose; `grasped` is a *measured* criterion (aperture closed, object inside the jaw envelope, object clear of the support surface) | measurements 3 and 5 |
| 5 | tests, `docs/gazebo.md`, `experiments/results/gazebo/` | contract, readiness, parsing, isolation, timeout, freshness, success/failure, no-mock-fallback tests; evidence logs | Phase 3–5 |

## Synchronous stepping vs asynchronous Gazebo

Not a blocker, and no shared abstraction changes. `step()` already publishes the
tick's command repeatedly and blocks until Gazebo's own `/clock` (delivered by
the joint-state stream) has advanced by exactly one control period, raising
`simulation.gazebo_clock_stalled` if it does not. That is a real synchronisation
adapter over an asynchronous simulator, and it is kept.

## Acceptance tests

* unit: TCP frame equals the MJCF `tcp` site offset; scene SDF matches the
  scenario's object size, target zone, table extent and friction.
* unit: no code path moves the object; `ground_truth` equals Gazebo's pose.
* live: 7 arm + 2 finger joints commanded and measured back; joint settling
  within tolerance.
* live: the cube is lifted by contact (its measured height rises with the jaw)
  and placed inside the target tolerance.
* experiments: seed 3 at 0 mm and 8 mm, honest PASS/FAIL/NOT_RUN.
