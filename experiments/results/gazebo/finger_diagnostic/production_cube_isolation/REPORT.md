# Part 1G: closing the mechanism -- friction plateau, 0.3 s dwell, blind sampling

## What was fixed and run

Three harness defects from the previous attempt, each found by reading output:

1. **FT topics**: the sensor's `<topic>` overrode the derived per-joint topic to
   one shared name, so both fingers published on one topic and my exact-string
   subscriptions missed it. Now `/finger_ft_left_finger` and
   `/finger_ft_right_finger` -- **forces read live on both fingers**.
2. **Approach knocked the cube** (17.6 mm in one step swing). The arm now
   follows the episode's own recorded path in three stages (ticks 40, 45, 56)
   with a live cube-stability check after each stage; `CUBE_KNOCKED` aborts the
   run instead of sweeping around a displaced object. Two of three runs pass at
   4 um knock; one failed loudly at stage 2 (36.7 mm) -- recorded, not hidden.
3. **Episode-timing mode**: the episode issues ONE step close and lifts after
   0.3 s; my stepped sweep gave every point 2.5 s. `--episode-timing` replays
   the episode clock (50 ms samples for 12 s).

## The mechanism, measured

### A. The close arrests at ~19-21 mm pad separation with 4.3 N

Stepped sweep (`telemetry.json`), every mid-range row:

    |Fy| = 4.27 4.30 4.33 4.34 4.37 4.44 4.45 N ... -- CONSTANT to 0.2 N

while min_gap to the CUBE is still 2.3-11.5 mm (no cube contact) and the pads
sit **0.46-0.89 mm below the tabletop** (`pad_bottom_z`), with `finger <-> table`
pairs in the accumulated contact list. Constant force opposing motion while
position error shrinks is Coulomb friction, not a spring: p x error would fall
with the error; it does not. Equilibrium q = F/p = 4.3/1000 -> pad sep
0.011 + 2(0.0043+0.0001) = **19.8 mm**, bracketing the observed plateau.

### B. The episode's clock matches the plateau

Episode-timing replay (`telemetry_episode_timing.json`), single step to q=0.0001:

    t=0.125 s  pad_sep 21.76 mm   |F| ~4.3 N   (episode at 0.15 s: 23.7 mm)
    t=0.251 s  pad_sep 19.02 mm   |F| ~4.3 N   (episode at 0.30 s: 21.3 mm)
    t=1.96 s   pad_sep 17.69 mm
    t=7.21 s   pad_sep 14.87 mm
    t=10.2 s   pad_sep 13.98 mm

Both runs collapse from ~42 mm to ~20 mm within 0.13-0.25 s and then creep at
millimetres per second under 4.3 N. The policy's `close_dwell >= 6` lifts at
0.3 s while the jaw is still at **21.3 mm** -- measured face-contact onset in
the stepped sweep is pad sep **19.2 mm** (min_gap crosses 0 exactly where the
formula predicted: q = projected_half - offset - 0.0055 = 4.08 mm). The jaw is
lifted 2.1 mm *before touching the cube*, so zero finger<->cube contact at lift
is not an instrumentation mystery -- it is physics.

### C. The blind sensor is real, demonstrated live

Stepped sweep, same rows, same ticks:

    cmd 0.0040: accumulated = [target_cube <-> panda_right_finger]  latest says (False, False)
    cmd 0.0030: accumulated = [target_cube <-> panda_right_finger]  latest says (False, False)
    cmd 0.0010: accumulated = [left pair]                           latest says (False, True)

A contact sustained for **seconds** is intermittently invisible to latest-only
sampling. The episode samples once per 50 ms tick with the same replace-latest
semantics, so the transient shove (cube +18 mm in one tick, step 58) was never
in the snapshot -- the historical "0 finger<->cube contacts" needed no
sensor fault, only unlucky sampling of a short-lived contact.

### D. The cube shove

In the replay the cube leaves spawn by t=0.125-0.25 s (offset along the jaw
axis grows to -7..-11 mm), i.e. the closing pads hit the cube's near face
first (cube sits ~1 mm off the TCP centreline) and shove it out of the jaw --
the same -18 mm escape the episode records at step 58. `contacts_accumulated`
catches the finger<->cube pair at t=0.125 s.

## Why the grasp fails, end to end

1. The grasp aim puts the pad bottoms at/below the tabletop (z_bottom =
   -0.5..-0.9 mm) -- pads physically on the table, `finger <-> table` pairs
   every tick, friction ~4.3 N per finger measured.
2. That friction cancels the finger controller's closing force at pad sep
   ~19.8 mm, right where cube face contact would begin (19.2 mm): the jaw is
   arrested at zero preload.
3. The policy lifts after a fixed 0.3 s at 21.3 mm -- 2.1 mm short of touching,
   5 mm short of gripping -- so lift happens with no cube contact: `grasped`
   only flickers True on the one tick the shove misreads as lift.
4. Whatever contact does occur is transient and the replace-latest contact
   reader misses it, which made the record say "never touched".
5. Given seconds (not 0.3 s) the jaw does creep through (14.0 mm at 10 s,
   below 16 mm) -- but by then it has already pushed the cube ~10 mm sideways
   instead of squeezing it, so time alone does not save the grasp.

## Harness defects found in my own replay (recorded, not glossed)

* At close onset the fingers were at q=0.0141, not the commanded 0.022 -- the
  single open command (published once, before the controllers' subscriptions
  were confirmed) was not verifiably held. It must be re-published and verified
  before any timing claim that depends on the starting aperture.
* Stage-2 approach knocked the cube once in three runs (transit wedge, joint6
  held 3.4 rad for 12 s in one attempt). Convergence polling per stage is the
  fix and was not implemented before this report.
* tcp_z measured 6.13 mm in the replay vs the episode's 7.02 mm, so my pad
  bite into the table is deeper than the episode's; the plateau matched anyway
  (both ~4.3 N), but the reproduction is ~0.9 mm low in z.

## Verdict

Production cube isolation: **FAIL** (grasp/lift/placement remain FAIL)
Cube necessary for stall: NO -- the arrest reproduces with the plateau at
  19-21 mm governed by pad-table friction (cube gap still 2-11 mm when the
  force locks), and contact onset sits exactly where geometry predicts.
Finger<->cube contact: **transient yes, steady no** -- caught by accumulation
  at t=0.125 s in replay and at cmd<=0.004 in the sweep; missed by latest-only
  sampling; never needed a broken sensor.
Geometric contact/overlap: face contact begins at pad sep 19.2 mm (validated);
  the episode lifts at 21.3 mm, 2.1 mm before touching.
Maximum finger stroke (cube in jaw): sweep reached 12.3 mm pad sep (q=0.00134);
  episode reached 13.98 mm only after 10 s of creep.
Reaction at stall: **4.3 N per finger measured live** (both FT sensors),
  constant while position error shrinks -- Coulomb friction against the table.
Root cause: **IDENTIFIED** -- pad-on-table friction (~4.3 N, pads 0.5-0.9 mm
  below the tabletop at the grasp aim) arrests the close at ~20 mm; the fixed
  0.3 s close_dwell lifts the jaw 2.1 mm before cube contact; the transient
  escape contact is then lost to replace-latest contact sampling.

Fix directions (NOT applied; each is a cross-backend or scene change needing
the owner's call): raise the grasp aim so pads clear the table; lengthen
close_dwell against measured aperture rather than ticks; make contact reads
accumulate over the tick instead of replace-latest.

Part 1 remains OPEN, honestly FAIL.
