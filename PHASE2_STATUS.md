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
