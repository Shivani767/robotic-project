# MuJoCo backend

**Evidence: Verified locally.** Every number here came from a run on this
machine. Commands are given so each can be re-run.

## What it is

`MuJoCoBackend` runs a 7-DOF Panda-class arm with a parallel gripper on a table,
executing a pick-and-place task. It is the primary simulator for all local work
and the source of every research result in `experiments/results/`.

```bash
robot-ci backend list          # capability table
robot-ci validate              # environment and catalogue check
```

## Embodiment

| Property | Value |
|---|---|
| Arm DOF | 7 |
| Gripper | parallel jaw, 2 DOF, 11 mm closed separation |
| Control rate | 50 Hz (0.02 s period) |
| Object | 16 mm cube (`size_m` 0.008 half-extent) |
| Place goal | (0.20, 0.15, 0.008) m |
| Workspace centre | (0.0, 0.10) |
| Table | 0.50 x 0.35 m half-extent |

The place goal was moved during development. It used to be (0.30, 0.30), which
is **off the table** and 0.67 m from the base across a 57-degree azimuth change;
reaching it top-down needed about 7 rad of joint travel, so the only nearby
solutions left the wrist 0.85 rad from the commanded orientation and the arm
dragged the held cube through 43 degrees of roll. Every episode placed correctly
within 13 mm and then failed on orientation.

## Measured behaviour

| Quantity | Value | How measured |
|---|---|---|
| Nominal solvability | 50.0% before `dc61715`, **66.7%** after | 12 seeds, `policy_v1` |
| Successful episode length | 76-77 ticks | 30 runs |
| Failed episode length | 128 ticks (stall-detected) | was 300, the cap |
| Placement error on success | 2-12 mm | 8 seeds |
| Depth-noise boundary | 1 mm | RQ5 |
| Success at 1 mm noise | 0.50 | 8 seeds |

### Why the IK tolerance is 0.5 mm

`ik_pos_tol` was 2 mm, which is no more accurate than the grasp band. Measured
across seeds, the end-effector arrives within 2.6-3.0 mm of the object on seeds
that succeed and 3.9-4.6 mm on the ones that fail, so the usable band is about
+/-1 mm wide.

| `ik_pos_tol` | solvability (8 seeds) |
|---|---|
| 2e-3 | 0.500 |
| 1e-3 | 0.625 |
| **5e-4** | **0.750** |

This is also why the tolerance is part of the reproduction record: changing it
changed task behaviour and invalidated every recorded result until they were
re-run.

## Known limits

These are measured, not suspected.

**The remaining ~33% of nominal episodes fail with the grasp not holding.** Three
hypotheses were tested and falsified:

* *Aim accuracy* — the end-effector distance to the object does **not** predict
  success: failures at 1.4 mm and 2.2 mm, successes at 2.2 mm and 2.4 mm. There
  is no threshold to tighten toward.
* *Shear during the lift* — halving the joint velocity limit changes nothing
  (0.750 either way). Quartering it fails every episode because the arm no longer
  finishes inside the step budget.
* *Grip width* — 0.6 is worse (0.375); 0.75, 0.8 and 0.9 are identical to the 0.7
  default, consistent with the close command having bottomed out against the pad
  envelope.

So the grasp is marginally stable and no lever available in the policy moves it.
This was recorded rather than tuned past, because sweeping parameters against an
outcome with no identified mechanism is how a benchmark starts reporting noise
as signal.

**Latency and occlusion are not behavioural factors here.** The policy latches
its plan on the first control tick and never consults RGB, so a delayed or
occluded frame is never acted on. Sweeping them costs episodes to measure
nothing.

**Object friction is non-inertial to tune against.** At friction 1.0 both arms
score 0.50 where 0.6 and 0.3 score 1.00: the gripper closing command was tuned
against the default, and at the high end the grasp pushes the object rather than
gripping it.

## Reproducing a number

```bash
.venv/bin/python experiments/rq5_perturbation_boundaries.py   # boundary curves
.venv/bin/python experiments/run_benchmark_suite.py           # fault injection
robot-ci test --backend mujoco --seeds 4                      # the CI gate
```

Each writes a JSON artifact under `experiments/results/` with a `status` field.
A result is only real if that field says `MEASURED`.
