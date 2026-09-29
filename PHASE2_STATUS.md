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
