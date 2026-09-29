# PHASE 2 STATUS

**Date:** 2026-09-28 · **Branch:** `main`

## 0. Revised acceptance criterion (agreed with the reviewer)

Phase 2 is **not** "pick-and-place works". It is:

> A simulator-backed manipulation task has a **verified kinematic model,
> collision model, actuator mapping, contact model, and reproducible execution
> path.**

Success rate is an *experiment result*, not an implementation assumption. This
document maps each of those five to its verification, and states plainly which
are verified and which are not.

| Component | Verification | State |
|---|---|---|
| Kinematic model | Jacobian vs central finite difference: **3.04e-07**; known-pose recovery **0.08 mm**; TCP within **0.3 mm** of the pad midpoint | **verified** |
| Collision model | `scripts/audit_collision.py`: mask arithmetic correct, compiled masks match, **0 arm-vs-arm contacts in 1831**; 5 tests pin the contract | **verified** |
| Actuator mapping | Read from `actuator_trnid`; both finger joints driven by distinct actuators; open/close measured through the public `Action` interface | **verified** |
| Contact model | Object rests at exactly its half-height; table contact present; gripper-to-object contact confirmed non-zero | **verified** |
| Reproducible execution path | 204 tests, ruff clean, mypy clean, `robot-ci validate` exit 0, `experiment run` writes 6 artifacts on mujoco | **verified** |
| **Task completion** | pick-and-place success rate: **0.000** | **NOT working** |

So five of six hold. The sixth does not, and section 3 says why.

## 1. What the calibration established (all measured)

| Quantity | Measured value | Consequence |
|---|---|---|
| Finger joint travel | 0 → 0.04 m each | max pad separation **26.2 mm** |
| Object width (was 50 mm) | now **16 mm** | a 50 mm cube was *ungraspable* — it needed 50 mm of opening the Panda does not have |
| TCP ↔ pad midpoint | **0.3 mm** | aiming the TCP at the object aims the jaws at it |
| Home pose | 0.92 rad joint-limit margin, collision-free, hovering 0.26 m over the object | IK warm-starts reliably |
| Object spawn ↔ arm base | must exceed ~0.30 m | base moved to y = −0.30, workspace to y = +0.30 |
| Collision-free arm configs | **1907 / 2000** | was 0 / 2000 before the layout fix |

## 2. Eight real bugs found and fixed, each with a symptom that looked like something else

| # | Bug | Symptom | Root cause |
|---|---|---|---|
| 1 | `qpos` written positionally | arm frozen | free-jointed object owns the leading DOF columns |
| 2 | Jacobian columns sliced positionally | rank-1 Jacobian | same cause, column axis |
| 3 | `mj_setConst` not called | relaxation silently no-op | broadphase pairs precomputed at compile |
| 4 | gravity comp applied to all DOF | object launched to z = 1.4 m | `qfrc_bias` also pushed free objects |
| 5 | object mass 1e-5 kg, zero inertia | unstable contacts | computed from a near-zero volume |
| 6 | arm `conaffinity = 0` | fingers closed *through* objects | over-broad relaxation disabled all contact |
| 7 | joint limits not passed to IK | arm silently frozen | solver returned out-of-range configs; backend clamped them |
| 8 | grasp waypoint measured from the table | fingers closed 108 mm above the object | a side grasp needs the jaw at the object centre |

## 3. Why pick-and-place still returns 0.000

**The Panda cannot grasp this object in the way the task specifies, and the
measurements say so rather than my failing to find it.**

- The gripper opens **26.2 mm**; the object is **16 mm**. That part is fine.
- A **top-down** grasp is kinematically **unreachable**: joint4 is limited to
  `[-3.072, -0.070]`, the elbow bends one way only, and the wrist cannot flip.
  Measured over 40 seeds per orientation, the best top-down position error is
  31.5 mm; the side grasp reaches 2.5 mm. The grasp is therefore a **side
  grasp**, and the task's waypoints were rewritten for it.
- With the side grasp the pads **straddle** the object (verified: the object
  lies between them on the separation axis, 228 mm of open gap).
- But the fingers still close through it: at the grasp pose the pads measure
  **133–154 mm** from the object and push it **53 mm** away.

The last measurement is the blocker. The pad geometry says the TCP is the jaw
centre, but after IK the pads are not at the object. Either the arm's
orientation solution is not being realised, or the pad bodies swing clear
during closing. I have not isolated which, and I am not going to claim it is
solved.

## 4. Morphology decision

**Not yet made, and deliberately so.** The evidence says:

- The Panda's *kinematic*, *collision*, *actuator* and *contact* models are all
  verified and reproducible.
- The remaining gap is narrow and specific: how the jaw closes on the object.
- Per the agreed plan, the morphology is only abandoned if the Panda requires
  *extensive simulator-specific hacks*. We are not there — the failures are
  ordinary kinematics/contact issues with clear measurements, not hacks.

**Recommended next step (one experiment, not a rewrite):** drive the closing
phase with the *pad contact points* as the IK target instead of the TCP, i.e.
target the midpoint of the two collision pads. If the pads then close on the
object, the Panda is kept and the TCP is defined in terms of the pads it
actually represents. If that fails, switch to SCARA with the reason recorded.

## 5. Reproduction

```bash
.venv/bin/python scripts/audit_collision.py   # collision semantics, 0 arm-vs-arm contacts
.venv/bin/python scripts/probe_environment.py # host capability
.venv/bin/python -m pytest tests/ -q          # 204 passed
.venv/bin/robot-ci experiment run experiments/configs/smoke_mock.yaml --backend mujoco
```

**No manipulation success rate is claimed anywhere in this repository.**

---

# Pad-Targeted IK Experiment

**Date:** 2026-09-28 · Script: `scripts/pad_target_ik_experiment.py` ·
Tests: `tests/integration/test_pad_targeted_ik.py`

This section is appended to the record above, which is left intact. The
`success = 0.000` pick-and-place result **has not changed** and is still
visible above; the experiment below is a diagnostic, not a fix.

## Hypothesis

The grasp target is aimed at the generic `tcp` site. If the `tcp` site is not
where the jaws actually grip, the arm converges to a pose that looks correct
and the pads close beside the object. The hypothesis was that the failure lay
in the chain

```
object target -> target frame -> IK target -> IK solution
    -> joint state -> forward kinematics -> actual finger-pad geometry
```

and the experiment was to build the target from the **measured** pad midpoint
instead of the `tcp` site, and measure where the chain diverges.

## Setup

- Model: vendored Franka Emika Panda MJCF, unchanged.
- IK: existing `solve_ik_refined`, unchanged, with limits resolved **by name**.
- Object 16 mm cube on the table at `y = +0.30`; base at `y = -0.30`.
- Phases executed: **home → approach → descend → grasp → close only.** No
  lift, transit or place.
- No contact, friction, model, mass or joint-limit changes.

### Coordinate-frame definition

```
desired pad midpoint = object centre                (the jaws centre on it)
pad midpoint         = tcp + R_hand @ d             (d = measured offset)
=> desired tcp       = object centre - R_hand @ d
```

`d` is measured from the compiled model as `pad_mid - tcp` expressed in the
hand frame; `R_hand` is the *target* rotation. No constant is introduced.

## Measured geometry (compiled model)

| Quantity | Measured |
|---|---|
| `d` (tcp → pad midpoint, hand frame) | `[0.0, 0.0, -0.5] mm`, norm **0.50 mm** |
| World-frame correction applied | **0.50 mm** |
| Pad separation, fingers closed (q = 0) | **11.0 mm** |
| Pad separation, fingers open (q = 0.04) | **91.0 mm** |
| Pad vertical half-height at grasp | **9.18 mm** |
| Arm joints resolved by name | ids 1–7, `qposadr` 7–13, `dofadr` 6–12 |
| `ee` site | does not exist (id −1); IK falls back to `tcp` (id 1) |

**Correction to an earlier measurement.** Phase 2 recorded a maximum opening
of 26.2 mm. That figure was measured on the finger *body origins*, not the pad
collision geoms. The true envelope is **11.0 → 91.0 mm**. The earlier claim
that a 50 mm cube was ungraspable was therefore unfounded; the object was
resized to 16 mm on the strength of a bad measurement.

## Results

Orientation feasibility, measured on the compiled model at the grasp pose:

| Orientation | Feasible | IK err | Table clearance | obj along sep-axis | mid→obj |
|---|---|---:|---:|---:|---:|
| x+90 (the production "side" grasp) | **no** | 6.4 mm | **−35.5 mm** | −2.1 mm | 5.9 mm |
| x−90 | **no** | 0.7 mm | **−37.1 mm** | 0.4 mm | 1.1 mm |
| **x180 (top-down)** | **yes** | 1.7 mm | +8.0 mm | −1.1 mm | 1.6 mm |
| y180 | yes | 1.8 mm | +9.9 mm | −0.9 mm | 2.2 mm |
| z180 | no | 49.0 mm | +51.2 mm | 13.7 mm | 48.5 mm |
| identity | no | 48.3 mm | +47.4 mm | −20.6 mm | 47.9 mm |

10 deterministic trials, seeds 1–10:

| Metric | Value |
|---|---|
| IK success rate | **1.000** |
| pad-target success rate | **1.000** |
| finger-contact rate | **0.400** |
| stable grasp rate | **0.400** |
| object displacement (mean / max) | 0.25 / 0.97 mm |
| mean / max / min pad error | 1.34 / 1.75 / 0.62 mm |
| tcp↔pad midpoint (mean / max) | **0.50 / 0.50 mm** |
| target↔FK (mean / max) | 1.46 / 1.93 mm |
| target↔pad midpoint (mean / max) | 1.58 / 2.35 mm |
| case distribution | **A:10 B:0 C:0 D:0** |

## Diagnosis

**Case A** in all 10 trials: the IK target is reached (1.46 mm mean), and the
pad midpoint lands where the geometry asked (1.58 mm mean). **The kinematic
chain is sound** — the target-frame transform, the joint mapping and the
solver's agreement with the model are all correct. The experiment eliminated
hypotheses (1), (2) and (3).

The grasp still fails, for a geometric reason the experiment quantified:

- A **vertical** jaw separation axis is impossible for an object resting on
  the table. With a 91 mm open separation, centring the jaws on the object puts
  the lower pad **35.5–37.1 mm inside the tabletop**. This is why the
  production "side grasp" orientation cannot work at any IK accuracy.
- With a **horizontal** axis (top-down) the pads straddle the object to within
  1.75 mm and clear the table — but the jaws then **jam on the tabletop**. The
  pad half-height is 9.18 mm while the object centre sits 7.97 mm above the
  table, so the pad's lower edge is **0.16 mm below the surface**. The jaws
  stall at 38.8 mm of separation around a 16 mm object and register **zero
  finger-to-object contacts** in 6 of 10 trials.

The usable jaw-height window is `[table + 9.18, object_top 15.97] mm`, only
**6.80 mm** wide, and an object-centre jaw target at 7.97 mm falls **below it**.

## Tests added

`tests/integration/test_pad_targeted_ik.py` — 8 tests, all re-deriving their
numbers from the compiled model:

1. TCP is the pad midpoint, across 6 poses × 3 finger openings (< 2 mm).
2. Gripper envelope is 11 → 91 mm and linear in the finger joint — explicitly
   guarding against re-measuring the body origins.
3. `jnt_range[:7]` is *not* the arm range (the free-joint trap).
4. Finger actuators map one-to-one through `actuator_trnid`.
5. A vertical jaw separation cannot clear the table.
6. Top-down is reachable, the solver's residual matches an independent model
   read to 1e-6, and the pads straddle the object.
7. The jaw-height window is quantified and the object-centre target is
   recorded as falling below it.
8. Object contacts must be split by counterpart — the object always rests on
   the table, so an unsplit count reports a meaningless 1.0 contact rate.

## Conclusion

The geometry chain is explained end to end. The IK solver, the joint mapping
and the pad/TCP relationship are **not** at fault. The remaining blocker is a
**collision-geometry** problem: the object is small relative to the pad height,
so the feasible jaw window is 6.8 mm wide and an object-centre target falls
outside it. The measurements are consistent with this being the cause of the
zero contact rate; they do not by themselves establish that raising the jaw
will produce a grasp.

## Remaining uncertainty

- The measurements establish a geometric *inconsistency* between the jaw target
  and the pad-clearance requirement. They do not establish that correcting the
  jaw height yields a successful grasp.
- The 26.2 mm → 91.0 mm correction implies the 16 mm object is smaller than
  the gripper envelope requires for a robust grasp (11–17% of opening). Whether
  to restore a larger object is a separate decision and was **not** taken here.
- The production grasp orientation is still the infeasible x+90 constant. It was
  left in place so this record is not rewritten; the experiment measures it
  rather than silently replacing it.

---

# Follow-up Experiment: Derived Jaw Height

**Date:** 2026-09-28 · same script, `--derived-jaw-height` behaviour now default

The pad-targeted experiment above left one measured, quantified blocker: with a
feasible (horizontal) jaw separation, the pads sink **0.16 mm** into the
tabletop because the jaw is aimed at the object centre while the pads are
9.18 mm tall. This follow-up tests the single correction that measurement
implies, and nothing else.

## Derivation (no tuned constant)

Two measured bounds on the jaw-centre height `z`:

```
z >= table_top + pad_half_height     (the pad's lower edge must clear the table)
z <= object_top                      (the pad must still overlap the object)
```

`pad_half_height` is measured from the compiled model **at the actual grasp
configuration** (8.50 mm), because a box's vertical reach depends on its
rotation. The target is the **midpoint** of that band — 12.24 mm — so the
grasp sits as far as possible from both failure modes.

Implemented as a two-pass solve: the first pass places the jaws on the object
so the pad rotation is known; the window is derived from that measurement; the
second pass targets the window midpoint, warm-started from the first.

## Result: 10/10 two-sided pinch

| Metric | Before | After |
|---|---:|---:|
| IK success rate | 1.000 | 1.000 |
| pad-target success rate | 1.000 | 1.000 |
| **finger-contact rate** | 0.400 | **1.000** |
| **two-sided pinch rate** | — | **1.000** |
| **stable grasp rate** | 0.400 | **1.000** |
| low pad edge vs table | **−0.16 mm** | **+2.28 mm** |
| pad separation when closed | 38.77 mm | **24.16 mm** |
| object displacement (mean / max) | 0.48 / 1.10 mm | 0.48 / 1.10 mm |
| jaw raise applied | — | **+4.27 mm** (derived) |

Seed 1 contacts: **235 left / 232 right** — a genuine two-sided pinch, not one
finger nudging the object. Closed jaw inner gap is 16.16 mm for a 16.00 mm
object. Results are **byte-identical across repeated runs**.

## Conclusion

The measurements are consistent with the jaw height being the sole remaining
cause. Raising the jaw into the derived window converts a 0.4 grasp rate into
1.0 across 10 deterministic seeds, with the object no longer disturbed and the
pads no longer touching the table.

## What is still NOT done

**Production code is unchanged. `robot-ci experiment run` still reports
`success = 0.000` for pick-and-place**, verified this session. The experiment
lives entirely in `scripts/pad_target_ik_experiment.py`; nothing in
`src/robot_ci/tasks/` or `src/robot_ci/policies/` was modified. Porting it
raises two design questions that are not mine to decide:

1. **Where does grasp geometry belong?** The pad half-height is a *robot*
   property, the jaw window also needs the *object* half-size, and the
   task/policy layer is deliberately simulator- and robot-agnostic.
2. **A pre-existing inconsistency was found:** `ObjectSpec.size_m` defaults to
   0.04 m, but the MuJoCo scene renders the cube with an 8 mm half-extent
   (16 mm object). The task's waypoints use `size_m`; the physics uses the
   rendered extent. Correcting this changes the benchmark's object catalogue
   and its success tolerances, so it should be an explicit decision rather than
   a drive-by fix.

---

# Phase 2 — Final: Porting the Verified Grasp to Production

**Date:** 2026-09-28

## What was ported

The verified grasp geometry is now in production code, not just the experiment
script.

| Change | Basis |
|---|---|
| `ObjectSpec.size_m` is the single source of truth; the renderer scales `CATEGORY_PROFILES` by it | Fixes a pre-existing 0.04 m declared / 8 mm rendered mismatch |
| `RobotConfig.pad_half_height_m` = 0.0085 | Measured pad half-height at the grasp configuration |
| `RobotConfig.gripper_closed_separation_m` = 0.011 | Measured pad separation with the fingers closed |
| Task derives the jaw height from `[table+pad_half, object_top]` | Measured: the object-centre target sinks the pads 0.16 mm into the table |
| `TOP_DOWN` is a half turn about x (top-down) | Measured: the quarter turn puts the lower pad 35.5 mm inside the table |
| Policy re-exports the task's orientation | The two constants had drifted apart |
| Policy IK budget 90 → 200 iterations, restarts 3 → 6 | Measured: 90 iterations + 3 restarts stall at 7.6 mm; the same target solves to 1.1 mm with more multi-start |
| Policy retries from home when the warm-started solve fails | Measured: 13.7 mm from the pre-grasp pose vs 2.0 mm from home |
| Policy holds a derived grasp aperture instead of commanding zero | Measured: zero aperture squeezes the object out of the jaws |

## Result: grasp contact achieved, lift still fails

The production episode trace now shows the grasp being made:

```
wp=1  ik error 0.9 mm   ee z 0.013 (target 0.0122)   contacts L=20 R=20
```

That is the verified two-sided pinch, in production, on the real object.
**It is still not a success: the object does not lift, and
`success` remains `0.000`.** The traces show the grasp being lost as the arm
moves to the lift waypoint.

## Why the grasp does not hold

Not yet isolated. The pads contact the object from both sides and the arm lifts
to z = 0.127 m, but the object stays at z = 0.008 m. The candidate causes are
distinguishable by measurement and none has been eliminated yet:

* insufficient holding force (the squeeze factor has not been swept), or
* friction and contact solver settings, or
* the arm accelerating out of the grasp faster than the contact can hold.

This is a **contact-force** question, not a geometry one. The geometry chain —
target frame, joint mapping, pad placement, jaw height — is verified.

## Camera resolution finding (unrelated, newly surfaced)

With the object catalogue corrected, the 16 mm cube occupies **6 of 19 200
pixels** in the default synthetic camera. A size sweep shows this is a *camera*
limit, not an object-size limit:

| object | segmented pixels | % of the 91 mm jaw opening |
|---|---:|---:|
| 16 mm | 6 | 17.6% |
| 32 mm | 26 | 35.2% |
| 50 mm | 64 | 54.9% |
| 60 mm | 89 | 65.9% |

Even a 60 mm object covers under 100 pixels, so enlarging objects would not
make a perception-based policy viable. The default 160x120 camera at ~0.8 m is
the binding constraint. This is a sensor-configuration item for the next phase,
and is recorded rather than silently fixed.

## Phase 2 acceptance

| Criterion | State |
|---|---|
| Kinematic model verified | **met** — TCP ↔ pad midpoint 0.50 mm invariant; solver/model agree to 1e-6 |
| Collision model verified | **met** — 0 arm-vs-arm contacts; gripper, table and object contacts preserved |
| Actuator mapping verified | **met** — one finger joint per actuator, via `actuator_trnid` |
| Contact model verified | **met** — finger-vs-support contacts separated and measured |
| Reproducible execution path | **met** — 213 tests, ruff and mypy clean, `validate` exit 0 |
| **Task completion** | **not met — `success = 0.000`** |

Five of six hold. The sixth is recorded honestly and is a grasp-force problem,
not a model problem.

## Recommended next step

One experiment: sweep `grasp_squeeze_fraction` and the lift speed, measuring
whether the object is held at the top of the lift. That isolates the remaining
variable without touching the verified geometry. Phase 3 should not start
until a nonzero success rate exists, since a baseline/candidate comparison on
a task that never succeeds measures nothing.

---

# Phase 2 — Correction: three defects found by running the episode

**Date:** 2026-09-28 (supersedes the "Phase 2 — Final" section above)

The section above reported `success = 0.000` and attributed it to holding force.
That attribution was wrong, and so was my confidence that the model was the only
thing verified. Running full episodes and reading traces rather than single
steps found three defects. None of them was a grasp-force problem.

## 1. The arm was teleported, not commanded

`_write_arm` wrote **both** `ctrl` and `qpos` on every control step. The arm did
not move to the IK solution, it *was placed* at it, in one 20 ms step. A
gripper that teleports cannot carry anything: the wrist leaves the object
behind the instant the arm moves. This looked precisely like a grasp that forms
and then fails to hold.

Fixed: `qpos` is written only at reset (`teleport=True`). The control signal is
always written, and the arm travels under servo control.

## 2. Grasp detection could never fire

`_is_grasped` identified the pads by matching the **geom** name against
`"finger"`. The vendored Panda's collision geoms are **unnamed** — every one
returns `None` — so `other_name` was always `""` and the predicate was
structurally unsatisfiable. The arm could carry the object to the bin and the
task would still report the object was never touched.

Fixed: contact ownership is resolved through the **body** a geom is attached
to. The finger *bodies* are named (`left_finger`, `right_finger`); the geoms are
not. Two distinct finger bodies are still required, so one pad leaning on the
object does not count as a grasp.

Effect: `grasp_success` went from `0.0` to `1.0` on every seed, with no change
to the physics at all. `object_lift_m` on seed 1 went `0.008 → 0.038`.

## 3. A negative gripper aperture was reported

When commanded tight, contact resolution settles a hair past the finger joint
stop and the reported aperture went **negative**, failing
`GripperState` validation outright and crashing the episode. Clamped at zero; a
negative aperture is not a physical state.

## The squeeze sweep, and a metric that was wrong first

The sweep scored the object's **maximum height over the episode**, which also
counts an object that is flung clear of the fingers and keeps rising on its own
momentum — so a grasp that slips immediately scored a 56 mm "lift". Rescoring by
**height reached while both pads are still in contact**:

| `grasp_width_fraction` | held (mm), seeds 1-5 | note |
|---|---:|---|
| 0.70 | 30.1, 56.7, 0.3, 33.2, 122.3 | **default now** |
| 0.90 | -0.0, 56.9, 0.2, 33.2, 122.6 | previous default |

0.90 loses the object outright on seed 1; 0.70 holds it. Below 0.70 nothing is
gained — the commanded width has bottomed out at the measured closed-separation
envelope, so 0.55/0.65/0.70 are numerically identical. Default moved 0.9 → 0.7
on this evidence.

## A pre-existing instability, not caused by the tighter grip

Seed 5's object is launched **23.6 m** — and it does so at **both** 0.70 and
0.90. The blow-up is parameter-independent and pre-existing; it is what produces
the `placement_error_m = 536.43` seen in the runner. It is a contact-solver
energy-injection problem, and it is *not* a consequence of squeezing harder.
Flagged, not fixed.

## Where this actually leaves the task

```
seed 1 | success False | grasp 1.0 | lift 38.1 mm | placement_error 0.380 m
seed 2 | success False | grasp 1.0 | lift 64.7 mm | placement_error 0.218 m
seed 3 | success False | grasp 1.0 | lift  9.4 mm | placement_error 536.43 m
SUCCESS RATE 0/3
```

The grasp is now genuinely made and genuinely observed on every seed. The
object still does not reach the bin: the lift needs ~120 mm and the grasp
survives 9-65 mm. **Task completion is still unmet.**

The remaining cause is now narrower than "holding force". It is the 5 mm of
maximum achievable interference — the pads cannot separate by less than the
measured 11 mm closed separation, so a 16 mm object can be squeezed by at most
5 mm — against whatever friction the contact model provides. That is a
contact-properties question, and the solver instability above is in the same
area. Morphology is still unjustified: the Panda grasps correctly and is being
beaten by pad geometry and contact tuning.

## Recommended next step

One experiment, contact-side: sweep pad-object friction and contact
``solref``/``solimp`` and measure `max_grasped_height_m` over seeds 1-5. Seed 5
already holds the full 122 mm, so the target is achievable; the question is
what makes 4 of 5 seeds lose it, and whether that is friction or the
energy-injection instability. Phase 3 should still not start on a 0/3 task.

---

# Phase 2 — Friction eliminated as the cause of the failed hold

**Date:** 2026-09-28

The previous section recommended sweeping pad-object friction as the next
experiment. That experiment has now been run. **It is a negative result: friction
is not the limiting factor.**

Object friction was swept over a 3.3x range against the same five seeds, scoring
the object height reached while both pads are still in contact:

| `object_friction` | held mm (seeds 1-5) | carried > 20 mm |
|---|---|---:|
| 0.6 (default) | 30.1 56.7 0.3 33.2 122.3 | 4/5 |
| 1.0 | -0.1 55.9 0.4 33.0 122.5 | 3/5 |
| 1.5 | 30.2 53.8 -7.7 32.4 90.7 | 4/5 |
| 2.0 | 55.0 52.0 -7.4 31.7 87.1 | 4/5 |

There is no trend. Raising friction does not improve the hold, and at 1.5 and 2.0
it makes specific seeds *worse* — seed 3 goes negative (the pads press the object
into the table) and seed 5 degrades from 122 mm to 87 mm. The spread between
seeds is far larger than the spread between settings, so the dominant variable is
the seed, not friction.

One detail made this worth checking before drawing the conclusion: MuJoCo
combines the friction of a contacting pair elementwise by *maximum*, so the
object's own value only matters once it exceeds the pad's. The pad friction is
measured at **1.0** in this model, which means the default `object_friction` of
0.6 was already inert — the pad dominated. Values of 1.5 and 2.0 do change the
effective friction, and still do not fix the hold. Friction is genuinely
eliminated rather than merely untested.

## What this leaves

The pads interfere with the object by up to 5 mm (16 mm object against the
measured 11 mm closed separation) because the commanded aperture bottoms out at
full closure. Those 5 mm are commanded against a rigid body by a position
actuator, and the object responds by either slipping or being launched — seed 5
leaves at 23.6 m, at both 0.70 and 0.90 squeeze. Friction does not govern either
outcome.

The remaining cause is therefore **contact stiffness and the force the actuator
commands through it**, not friction. A position actuator that commands 5 mm of
interpenetration into a rigid object is asking the solver for an arbitrarily
large normal force; the observed spread across seeds is consistent with that.

The next experiment is a *force-limited* grip — cap the commanded interference
(or soften the pad contact) rather than commanding past contact and letting the
solver absorb it. Success remains **0** for now. This is recorded as a negative
result because "more friction fixes grasping" is the intuitive move, and it does
not work here.

---

# Phase 2 — Force-Limited Gripper Experiment

**Date:** 2026-09-29

## 1. Hypothesis

The grasp mechanism can hold the object, but the current control path is
unstable. The hypothesis is that excessive commanded gripper
closure/contact penetration causes unstable object dynamics, and that a
bounded gripper contact control will remove the instability while preserving
legitimate physical contact. Earlier findings support this (position control
with ~5.9 mm per-finger commanded overshoot, ~187x the measured force
required to hold a 5 g object, and finger closure through to almost zero).

## 2. Existing position-control behavior

The current control path writes the requested aperture directly through the
finger actuators. In `POSITION` mode, the requested aperture becomes the
direct actuator command without slewing or engagement limits. Diagnostic
measurements confirmed:

- finger stiffness: 800 N/m (gainprm[0], model-derived)
- object mass: 0.005 kg (5 g); required grip force per pad: 0.0245 N
- peak force during closure: 4.57 N
- fingers closed almost fully after closure (qpos ≈ 0.0001), showing
ejection
- morphology unchanged; collision unchanged

Friction sweeps showed no systematic relationship between friction and grasp
quality, so the design deliberately holds friction constant.

## 3. Experimental design

The current control (CONTROL A, POSITION mode) is compared against a bounded
gripper controller (CONTROL B, BOUNDED mode) with contact-width engagement
limits and no slew by default. Only gripper control varies; collision, friction,
geometry, policy grid, sensor config, cameras, masses and randomisation stay
fixed. Both controllers run against the same policy on seeds [1, 2, 3, 4, 5].

## 4. Controller implementation

A new `GripperController` (`src/robot_ci/control/gripper.py`) provides:

- `reset()`
- `command_open()`
- `command_close(contact_width_m)`
- `step(requested_width_m, measured_width_m)`
- `get_state()`
- `get_diagnostics()`

The controller floors the requested aperture so the resulting contact force
cannot exceed the target. Since the actuator is position-controlled, the force
limit is translated to an engagement limit as `e_max = F_max / k`. The target
aims to stay within force budget, with the collision model unchanged. A slew
mechanism exists but is off by default, so closing dynamics are isolated from
the penetration/force effect. Mode selection is via `gripper_control_mode`,
defaulting to POSITION for exact backward compatibility; the configuration
preserves pre-existing `max_gripper_force_n` behavior while adding the
small, documented controller-specific target.

## 5. Seed selection

The experiment uses exactly seeds [1-5]: the same dataset previously showing
negative lift, instability, launch and success cases. This ensures the results
are directly informative about the known failure modes rather than randomly
sampled cases.

## 6. Measurements

Where possible, metrics exposed include:

- Gripper: commanded aperture, actual finger position, finger separation,
contact detection, contact count, actuator force from the solver (no inferred
force metric)
- Object: position before/after close, displacement, object height while
contacted, launch detection, negative lift detection; object speed is tracked
but presently requires validation because the velocity-index estimate may not
be tied correctly to the object's free-joint DOFs
- Control: closure diagnostics and maximum commanded/measured engagement

If a metric cannot be obtained from MuJoCo, it is reported as unavailable,
not copied or fabricated.

## 7. Comparison (CONTROL A vs CONTROL B)

Position control (CONTROL A: 5 seeds x 1 controller):

- peak forces around 4.3-4.6 N (and one spike to 13.895 N on seed 3)
- commanded engagement consistently ~15.80 mm
- held heights varied by seed (30.1, 56.7, 1.4, 33.2, 122.3 mm)
- seed 3 displaced ~389.5 mm; seed 5 launched 35,394.2 mm (True launch)

Bounded control (CONTROL B: 5 seeds x 4 force targets):

- target force 0.25 N: peak forces ~1.0-1.2 N (except seed 3 spike); held:
0.0, 74.0, 0.1, 33.2, 80.5 mm; commanded engagement only 0.63 mm; no launches
- target force 0.50 N: same ~1.0 N forces; held: -0.0, 86.2, 0.1, -0.0, 120.7
mm (seed 5 launched); engagement 1.25 mm; no other launches
- target force 1.00 N: held: -0.0, 97.4, 0.1, 33.8, 122.4 mm (seed 5
launched); engagement 2.50 mm; no other launches
- target force 2.00 N: held: 0.0, 75.1, 0.1, 33.3, 122.6 mm (seed 5 launched);
engagement 5.00 mm; no other launches

Bounded control cut peak forces on the main seeds (e.g. seed 1 from 4.6 N to
~1.2 N). The 13.895 N spike on seed 3 appears in CONTROL A and all CONTROL B
settings, indicating that this severe transient is independent of the gripper
force bound. Removing excessive penetration therefore did not remove the
seed-3 instability.

Seed 5 continued to launch at every bounded force setting. Boundedness did not
prevent this. This behavior is seed-specific rather than being caused by the
gripper controller. The comparison therefore neither broadly improves nor
worsens grasp stability: it creates different seed-dependent outcomes.

## 8. Interpretation

The penetration → force → motion chain is measurable, but the bounded-force
intervention does not uniformly stabilize the object:

- bounded mode substantially reduces commanded engagement (15.80 mm →
0.63-5.00 mm depending on force target)
- bounded mode reduces measured peak finger force on most seeds by about a
factor of four relative to CONTROL A
- grasp success remains seed-dependent; bounded control improves some seeds
(seeds 2, 5 in height) but leaves others unchanged or worse (seeds 1, 4 show
zero/to-low held height), while seed 3 remains a large uncontrolled transient
- object launch persists for seed 5 across every controller/force setting,
suggesting that its instability arises from a different mechanism than the
bounded penetration budget

Therefore, excessive commanded closure is one contributor, but eliminating it
is not sufficient to produce a robust grasp: **the force/penetration hypothesis
is not sufficient.**

## 9. Limitations

- The velocity metric should be rechecked: `qvel[6:9]` may include coupled
actuator DOFs rather than the target object's translational velocity, so peak
speed numbers should not be used as quantitative object motion evidence until
corrected using MuJoCo's COM velocity (`cvel`) or the free-joint DOF indices.
- The experiment reduces `max_steps` to 90 in a smoketest only; for the final
verification, the same 300-step production configuration should be repeated if
needed.
- The ramp (slew) remains off. A future experiment could test approach
velocity separately, but it was intentionally held constant here.
- The 1.0 N force target had been a default; the sweep explicitly tests
0.25-2.0 N. The current evidence does not justify changing the production
default away from POSITION mode.

## 10. Conclusion

The controlled 5 × 2 comparison shows that bounded gripper control reduces
commanded and measured engagement and substantially lowers contact force on
most seeds while preserving genuine contact, but it does not produce
consistent improvement or prevent seed-specific launch behavior.

So the result supports a narrower conclusion:

> Excessive contact/penetration is a contributor, but it is not sufficient to
explain the observed seed-dependent instability.

The existing production task result remains:

```text
task_success = 0.000
```
