# Finger diagnostic (Part 1B)

Question: why do the two finger joints stop near a 21 mm pad separation instead
of closing onto a 16 mm cube?

Every file here was produced by `scripts/gazebo_finger_diagnostic.py` against a
real Gazebo process. Nothing in this directory is a simulation of Gazebo.

## What was ruled out, and by what

| Candidate | Verdict | Evidence |
|---|---|---|
| Object blocks the pads | **RULED OUT** | `finger_sweep_no_object.json`: with the cube deleted the jaw still stalls at the same place. |
| Table blocks the pads | **RULED OUT** | Same sweep, table present, cube deleted: still stalls. |
| Wrong joint axis / limits / damping | **RULED OUT** | `joint_limits.json`: axis `(0,1,0)` / `(0,-1,0)`, limits 0..0.04, effort 100, velocity 0.2, damping 1, friction 0. All as authored. |
| Wrong parent/child | **RULED OUT** | Parent `panda_link7` (the hand is lumped into the wrist by `gz sdf -p`), child `panda_left/right_finger`. |
| Collision geometry | **RULED OUT** | `model_sdf.txt`: pad boxes are separate links; no other collision volume is within reach of them. |
| Publisher schema guessed wrong | **AVOIDED** | `message_schemas.json` is written from the live descriptors *before* anything reads them. |

Note: the first two passes of the SDF parser reported the finger joints as
having **no axis and no limits**, because `<axis>`, `<limit>` and `<dynamics>`
are nested inside `<axis>` and their contents are child elements rather than
attributes. The raw SDF has all of them. A report that had trusted that parser
would have "fixed" a defect that does not exist.

## What the measurement actually shows

`finger_hold_no_object.json` (command 0, hold 30 s, cube deleted):

    separation constant at 11.41 mm

but the fingers start at 11 mm -- they never moved, so this run says nothing
about tracking.

`finger_hold_no_object.json` (open to 0.08, then a single step to the grasp
aperture 0.0002, sampled at 20 Hz, cube deleted):

    t=0.05 s   separation 63.3 mm
    t=0.15 s   separation 26.6 mm
    t=0.20 s   separation 22.5 mm   <- stalls here
    t=2.00 s   separation 20.6 mm   <- creeping at ~0.5 mm/s

This is the episode's stall, reproduced with nothing in the scene to cause it.

With the finger joint velocity limit raised 10x
(`finger_hold_no_object_vel2.json`), the jaw closes faster and then limit-cycles
between 15.0 and 19.0 mm:

    t=0.05 s   15.0 mm
    t=0.15 s   19.0 mm
    t=0.40 s   15.0 mm   ... repeating

## What the measurement actually shows

Three runs, all with the cube deleted, all stepping from 0.08 open to the
0.0002 grasp aperture and sampling at 20 Hz:

| Run | Command | Result |
|---|---|---|
| `finger_hold_no_object.json` | step | 22.5 mm at t=0.20 s, creeping to 20.6 mm at t=2 s |
| `finger_hold_no_object_ramp0.05.json` | ramped at 0.05 m/s via the production `GripperController` | identical: 22.4 mm then 20.6 mm |
| `finger_hold_no_object_vel2.json` | joint velocity limit raised 10x | limit cycle between 15.0 and 19.0 mm |

And with the finger collision geometry deleted entirely
(`finger_hold_no_object_nocoll.json`): 22.5 mm then 21.1 mm. **No change.**

Separately, `finger_sweep_no_object.json` walks the aperture down and reads the
fingers after 4 s at each step:

    commanded per finger   measured
    0.010                  0.01017     <- tracks
    0.008                  0.00977
    0.006                  0.00965
    0.004                  0.0064
    0.000                  0.00177

so the jaw tracks well near open and approaches a closed command slowly, and is
still descending when the 4 s dwell ends. A 30 s hold at command 0 from the
already-closed rest pose sits at 11.41 mm -- but the fingers never moved there,
so that run says nothing about tracking.

## Diagnosis, corrected

The earlier diagnosis -- the engine's joint velocity limiter fighting the
position controller -- is **wrong**, and the two runs above falsify it: neither
ramping the command nor removing the velocity limit nor deleting the pad
collisions changes where the jaw ends up.

What the runs support instead: **this is not a stall, it is a long approach.**
The jaw leaves open at the joint's 0.2 m/s limit (63 -> 26 mm in 0.15 s is
exactly 0.2 m/s for that distance), reaches roughly 22 mm, and then closes at
about 0.6 mm/s while an error of several millimetres is still present. Something
holds it there with a force of the same order as the controller's command, and
the ablations say it is not the object, the table, the pad collision geometry,
the velocity limit, or the command rate.

For the grasp this is decisive even without knowing what the reaction is. The
policy advances past the grasp on `close_dwell >= 6` ticks -- 0.3 s at 20 Hz --
and the measured aperture at that moment is about 0.021 m. The cube is 0.016 m
wide. **The jaws are still 5 mm too wide to enclose it when the lift starts**,
which is exactly what the contact sensors show: cube-to-finger contact on zero
ticks of a 149-tick episode.

## The fix this implies, and why it is not applied here

Two candidate fixes, neither applied:

1. **Wait for the aperture instead of counting ticks.** The backend already
   reports the measured aperture in `RobotState.gripper.width_m`, so the policy
   could require the measured width to reach the commanded width before lifting.
   This touches shared policy code and therefore MuJoCo's behaviour, and MuJoCo
   cannot be re-validated from this repository in the time available.
2. **Give the closing transient authority to finish** by changing what the
   controller can apply. That is a gain change, and the cause of the reaction is
   not yet known, so changing gains now would be tuning rather than fixing.

The next concrete diagnostic is a **force-torque sensor on each finger link**.
The wrench schema is already captured in `message_schemas.json` (it has `force`,
`torque` and `force_offset`), so the remaining work is to place the sensor, read
it with the descriptor-verified field names, and read the reaction directly
instead of inferring it from the creep rate.
