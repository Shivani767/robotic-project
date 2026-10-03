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

## Diagnosis

Fast approach, then a stall that holds a large position error against a ~5 N
command, then slow creep -- with and without the velocity limit. That shape is
the engine's joint velocity limiter acting against a position controller: the
limiter scales the applied force down to enforce `velocity`, so the proportional
term loses authority exactly when the error is largest. Raising the limit tenfold
moves the stall rather than removing it, which is what that mechanism predicts
and what a hard geometric block would not do.

**This is a diagnosis, not a confirmed fix.** It has not been verified by
changing one thing and re-measuring.

## The fix this implies, and why it has not been applied

The command asks the fingers to cross 40 mm in a single control tick. A real
gripper is commanded with a closure *rate*. The project already owns that
abstraction: `control/gripper.GripperController` has `closure_ramp_enabled` and
`max_closure_speed_m_s` for exactly this, and the Gazebo backend now constructs
one. Enabling the ramp on the Gazebo path, at a rate at or below the fingers'
declared 0.2 m/s, is the change the evidence points at.

It is **not applied**, because the policy advances past the grasp on
`close_dwell >= 6` ticks (0.3 s at 20 Hz), and a ramp that respects the
velocity limit needs ~0.2 s for the travel alone. Applying the ramp without
also checking that the policy waits would convert a slow grip into no grip at
all, and that must be measured rather than assumed.
