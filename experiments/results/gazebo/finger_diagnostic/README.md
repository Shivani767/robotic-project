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

## The reaction, measured rather than inferred

`finger_hold_no_object_ft.json` puts a force-torque sensor on each finger **joint**
and reads it while the jaw closes. (The sensor must hang off the joint: Gazebo
refuses a link-parented one with "Parent entity of sensor [...] must be a joint."
Its topic is `/world/<w>/model/panda/joint/<joint>/sensor/finger_ft/forcetorque`,
read from `gz topic -l` rather than guessed.)

    t=0.15 s   sep 26.4 mm   left force.y  2.46 N   right force.y  2.66 N
    t=0.20 s   sep 22.5 mm   left force.y  4.93 N   right force.y  4.96 N
    t=0.55 s   sep 22.1 mm   left force.y  4.96 N   right force.y  4.98 N
    t=1.21 s   sep 21.4 mm   left force.y  5.01 N   right force.y  4.98 N

Commanded position error at the stall is 0.0053 - 0.0001 = 5.2 mm, and the
controller's proportional term is `p_gain * error` = 1000 * 0.0052 = **5.2 N**.

**The reading is 4.98 N.** The sensor is confirming that the controller is
applying exactly the force its position error implies -- and the joint still
does not move.

That is the sharpest statement available so far, and it eliminates several
possibilities at once:

* **Not a tracking or command error.** The applied force is exactly what the
  command asks for; the command is not being lost or mis-scaled.
* **Not an uncommanded reaction from the world.** 5 N is the controller's own
  output. Whatever balances it is not pushing with more than 5 N.
* **Not the object's inertia.** 5 N on a 15 g finger is an acceleration of
  ~330 m/s^2; reaching 5 mm would take about 0.2 ms, not the observed seconds.
* **Not the cube, the table, or the pads' collision geometry** -- all three are
  excluded by ablation, including deleting the pad collisions outright.

What is left is the engine or the joint model refusing to convert that force
into motion: a joint constraint. The one documented mechanism is the joint
velocity limit, which DART enforces by scaling the applied force. Raising it
tenfold moved the stall into a limit cycle rather than removing it, which is
consistent with that mechanism but does not confirm it.

## Removing the velocity limit altogether

`finger_hold_no_object_ft_novel.json` deletes `<velocity>` from both finger
joints rather than raising it. This does not confirm the velocity-limiter theory;
it produces a different and sharper result:

    t=0.05 s  L 0.03713  R 0.00006   force.y = 96.7 N on BOTH
    t=1.20 s  L 0.04000  R 0.00001   force.y = 96.6 N on BOTH

Two things follow.

**The controller saturates.** With no velocity limit it applies its full
`cmd_max` of 100 N and the reading sits at 96.7 N. So the force really was being
capped at 5 N before, by the limiter -- the limiter is doing something.

**The fingers do not move symmetrically.** The left finger is pinned at 0.0400,
which is its upper joint limit, while the right finger goes to the commanded
0.0001. Two mirror-image joints under two identical commands, one of which
cannot be driven off its stop at all. That asymmetry is not explained by
anything recorded so far, and it is new information: it was invisible in every
velocity-limited run because there the limiter dominated.

So: the limiter is real and it is what capped the force at 5 N, but removing it
exposes a second, larger problem underneath. Part 1B is **not complete** -- the
root cause is still not identified, and two distinct defects are now on the
record rather than one.

## One finger at a time, and the retraction of the last commit

`--only` now reaches `hold`, not just `sweep`. Commanding one finger at a time,
each from 0.04 down to a commanded 0.0001:

    only=left_finger    t=0.55  L 0.00545  R 0.03702
                        t=1.00  L 0.00521  R 0.03702
    only=right_finger   t=0.55  L 0.03711  R 0.00544
                        t=1.00  L 0.03691  R 0.00540

**This retracts the "the left finger is pinned" reading from the previous commit.**
Neither joint is stuck. Each finger, driven alone, closes from 0.04 to about
0.0052 and stops there; the uncommanded finger simply stays where it was, which
is what produced the apparent 0.0400 asymmetry. That asymmetry was an artifact of
commanding only one side, not a second defect. There is one defect, not two.

And the stall is now shown to be per-joint: it reproduces with the other finger
inert, with no cube, with no table, and with the pads' collision geometry
deleted. A single prismatic joint, commanded over a 40 mm stroke, halts 5.2 mm
in, every time.

**A caveat that weakens part of this phase's instrumentation.** At t=0.05 s of
the left-only run, the *commanded* finger's sensor reads -0.18 N while the
*uncommanded* one reads 5.49 N. That is backwards, and it says the per-finger
force attribution is not trustworthy at short timescales. The 4.98 N and 96.7 N
steady-state figures are both-sided and consistent, so the force readings stand
as evidence that a large force is applied and nothing moves -- but which sensor
reports which finger's force should not be relied on.

**A real inconsistency in the converted model, worth checking next.** In
`model_sdf.txt` each finger *joint* pose is expressed relative to its parent --
`<pose relative_to="panda_link7">`, with the -0.7854 yaw -- but each finger
*link* pose is absolute in world coordinates and carries a 180 deg roll about x
with the yaw sign flipped:

    joint left_finger   pose relative_to="panda_link7"  0.003889 0.003889 0.1515  0 0 -0.7854
    link  panda_left_finger  pose                       0.091889 -0.003889 0.8815  3.14159 0 0.7854

The y components have opposite signs and the roll is inverted, so each pad sits
mirrored across its own joint axis and inverted in z relative to the joint frame.
Whether Gazebo reconciles this at load is exactly the kind of thing to confirm
rather than assume, and it is the next thing to check.

## The finger link poses are honoured, and correcting them is not enough

`--drop-finger-link-pose` deletes the absolute `<pose>` from each finger *link*
so Gazebo derives the child link from its joint instead, changing one variable:

    baseline (absolute link poses)  t=1.00  L 0.00521   force.y  5.00 N
    --drop-finger-link-pose         t=1.00  L 0.00811   force.y  0.74 N

Two conclusions, and the second is the unwelcome one.

**The poses are honoured.** Removing them moves the stall from 5.2 mm to 8.1 mm
and drops the reaction force from 5.0 N to 0.74 N. The 180 deg roll and flipped y
in the converted SDF are not cosmetic metadata; they are what Gazebo places the
pads by, and they change the outcome.

**Correcting them does not fix the grasp.** The finger still halts short, now at
8.1 mm instead of 5.2 mm, still with a residual reaction. So the link-pose
inconsistency is real and load-bearing but is not the whole cause. Applying the
URDF poses to the finger links is a necessary correction on its own merits -- the
converted geometry genuinely disagrees with the URDF it came from -- and it is
not a fix for this stall.

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
