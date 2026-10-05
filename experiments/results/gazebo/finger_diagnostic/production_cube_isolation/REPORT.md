# Part 1F consolidated: contact reconciliation, exact state, reproduction

## 1. Contact detection reconciled

Canonical detector: each sensor topic parsed separately, pairs matched on exact
collision names in both orientations (`scripts/production_cube_isolation.py:
contact_hits`). Raw text read from `gz topic -e`; no protobuf class guessed.

Two instrumentation bugs, opposite directions, both now closed:

* **Sweep false POSITIVE (mine).** The previous sweep concatenated the cube's and
  the table's contact messages before substring-matching, so
  `panda_right_finger` (from the TABLE sensor) combined with `target_cube` (from
  the CUBE sensor) manufactured a pair existing in neither message. The raw cube
  message at those ticks contains only `target_cube <-> table`. Archived as
  `*_concat_bug.json`. The episode's zero finger-cube contact was NOT refuted by
  that run.
* **Episode false NEGATIVE (Gazebo's sensor).** The episode reports
  `object_contacts` = cube<->table only on all 149 ticks, `contact_health.errors`
  empty and 4392+ messages -- a *live* stream. Yet at step 57->58 the cube moves
  18 mm in -y and +3 mm in z **within one tick** while the TCP moves 0.06 mm and
  the jaw commands stay closed. Nothing else can displace it: the pads were
  touching it. The contact sensor never reported a single finger<->cube pair in
  the whole episode. So the historical "0 ticks finger<->cube contact" is an
  instrumentation failure, confirmed against physics.

Corroboration: the episode's TABLE sensor does report
  `table::link::collision <-> panda::panda_right_finger::panda_right_finger_collision`
  on ticks 49-50, so finger collisions *can* appear in contacts -- the blindness
  is specific to finger<->cube pairs on the cube's sensor.

## 2. Exact failing state (failing_state.json)

tick 56, sim_t 10.649, phase DESCEND, commanded q = measured q within 0.5 mrad;
aperture 0.01056 m -> **pad separation 21.4 mm** (0.011 + aperture, pad geometry
verified in the URDF: inner faces +/-5.5 mm at q=0);
TCP [0.02193, 0.22464, 0.00702], cube [0.02259, 0.22558, 0.008] (1.6 mm apart);
qdot and cube velocity were never recorded by the episode -- recorded as null,
not inferred. Collapse: 43.9 -> 22.2 -> 12.7 -> 11.0 -> 10.6 mm over steps 52-56,
then 6 ticks of hold, then ejection.

## 3. The stall is geometric contact with the cube, predicted to 0.01 mm

A 16 mm cube whose face normal is at angle t to the jaw axis has projected width
`16*(|cos t|+|sin t|)` mm along that axis. The measured stall separation 21.4 mm
solves for **t = 26 deg or 64 deg**: a yawed side grasp, pads on the cube's
faces (or an edge), not on the diagonal (that would be 22.6 mm at 45 deg).

The jaw stops at 21.4 mm because the cube is between the pads; the policy's
close_dwell expires at 6 ticks holding it there; the squeeze then ejects the cube
(step 58, grasped flips True for one tick, cube lands 26 mm away, grasp lost).
This is the same "squeeze ejection" world.py already documents -- now with the
missing link: **the contact was real and the sensor was blind to it.**

## 4. Reproduction result (this run, closing sweep, canonical detector)

Arm at the episode's step-56 joints (settled 13 mrad joint5 / 26 mrad joint7 off
-- not exact), jaw closing 22 mm -> 0.8 mm per finger:
* no finger<->cube contact reported (correct detector, raw kept per row)
* pads reach pad separation ~13 mm, **below the cube's 16 mm face width**
* => the pads closed PAST the cube's position: in this run the cube was never
  between the jaws. The episode geometry was not reproduced; my joint-level pose
  was several mm off at the TCP, and TCP/pad poses were not captured to prove it.

So: reproduction inconclusive on state match (joint errors recorded), but the
stall itself is fully explained by section 3.

## Verdict

Production cube isolation: **FAIL** (grasp/lift/placement still FAIL)
Cube necessary for stall: **YES** -- contact with the cube at ~26 deg yaw is the
blocking object, predicted 21.36 mm vs measured 21.4 mm.
Finger<->cube contact: **physically YES, sensor-reported NO** (instrumentation
failure; displacement proves contact at step 58).
Geometric contact: **YES**, jaws span the cube's projected width at stall.
Maximum finger stroke with cube straddled: **21.4 mm pad separation** (episode);
without a cube (or when missed): full stroke.
Reaction at stall: not measured (production model has no FT sensor; 5 N figure
from the earlier diagnostic's force-torque run).
Root cause: **IDENTIFIED (blocker) = jaw-cube contact at yawed grasp + contact
sensor blind to finger pairs + squeeze ejection on close_dwell expiry.**
Grasp success: NOT achieved; two candidate fixes exist in world.py comments and
are deliberately not applied blind. Part 1 remains OPEN, honestly FAIL.
