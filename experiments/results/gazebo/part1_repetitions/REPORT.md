# Part 1 — Five measured Gazebo repetitions at seed 3, 0 mm (policy fix 7a9014e)

**Verdict: PART 1: INCOMPLETE — success rate 0 / 5.**

## Setup

* Code: committed policy fix `7a9014e` ("grasp aim clearance floor + sustained-lift dwell").
* Environment: live Docker Gazebo Harmonic (`rc-gz`, gz 8.15.0), bind-mounted repo, container verified to contain the fix (`_GRASP_AIM_CLEARANCE_TOLERANCE_M` present, `grasp_settled ... >= 12`).
* Entry point: `scripts/gz_run.sh start 'source /opt/ros/jazzy/setup.bash && cd /workspace/robot-ci && python3 scripts/gazebo_pick_place_evidence.py --seed 3 --noise-mm 0'`, run five times back to back.
* Evidence per run: `run{N}.json` (full per-tick telemetry) and `run{N}.log` (per-stage verdicts) in this directory; consolidated `summary.json`.

## Per-run measured results

| Run | Status | Steps | Settled TCP z (m) | pad_bottom (m) | Grasp aperture (mm) | finger↔cube contact ticks | Cube flick Δy (mm) | Max lift (mm) | Grasped ticks | Placement error (mm) |
|----|--------|-------|-------------------|----------------|---------------------|---------------------------|--------------------|---------------|----|----|
| 1 | failure | 173 | 0.0080 | −0.0005 | 9.8 | none | −22.1 | 12.0 | 1 | 184.0 |
| 2 | failure | 173 | 0.0080 | −0.0005 | 9.4 | [70] | −24.5 | 14.3 | 1 | 164.5 |
| 3 | failure | 173 | 0.0080 | −0.0005 | 8.6 | [70] | −27.9 | 12.5 | 1 | 176.8 |
| 4 | failure | 173 | 0.0080 | −0.0005 | 9.2 | none | −25.2 | 12.8 | 1 | 180.4 |
| 5 | failure | 173 | 0.0080 | −0.0005 | 9.3 | none | −22.5 | 10.7 | 1 | 178.2 |

Stage verdicts are identical in all five runs: `1_world_and_control PASS`, `2_gripper_enclosed_the_object PASS` (closest TCP→cube 1.18–1.21 mm), `3_object_lifted_by_contact FAIL`, `4_object_released_into_the_target FAIL`, `5_task_outcome FAIL`.

## Exact failure mechanism (measured, not inferred)

1. **The aim fix is applied but never reached.** The policy's grasp waypoint target is `0.0095 m` (the new clearance floor: `table + pad_half + 0.001`). Telemetry shows `waypoint_target z = 0.0095` on every run, yet the settled end-effector height is `0.0080 m` in all five runs — 1.5 mm below the commanded aim. The IK residual is reported as 0.77 mm, so most of the shortfall is execution/tracking, not solve quality.
2. **Pads still bite the table.** With `pad_half_height_m = 0.0085`, `pad_bottom = 0.0080 − 0.0085 = −0.0005 m`: the pads sit 0.5 mm *below* the tabletop at the moment of close, in every run. `table ↔ right_finger` contact is recorded on 3 ticks. The pad-on-table friction the fix was designed to eliminate is therefore still present — the floor raised the *command*, not the *achievable pose*.
3. **The cube is never captured.** The cube rests at its spawn pose `[0.0226, 0.2256, 0.0080]` through the entire close (steps 55–69) with no finger↔cube contact. The measured jaw aperture at end-of-close is 8.6–9.8 mm — narrower than the 16 mm cube — so the jaws closed *past* the object without straddling it (contact `2 PASS` only certifies TCP proximity ≤ 30 mm, not enclosure).
4. **The lift transition ejects the cube.** At step 70 (waypoint 1 → 2, the first tick after the new 12-tick dwell expires) the cube is flicked 22–28 mm in −y and rises transiently to 10.7–14.3 mm — this transient is what the single `grasped=True` tick and the finger↔cube contact on that one tick (runs 2, 3) correspond to. It is a collision, not a lift; the cube is back on the table by step 72 and stays there.
5. **No placement.** The arm carries an empty gripper to the target; final cube pose remains within mm of spawn (error 164–184 mm, tolerance 50 mm).

## Interpretation

The committed fix raised the grasp aim command above the table but the arm does not *reach* the raised aim: it consistently settles at the old height (0.0080 m), so the pads still contact the table and the close still cannot enclose the cube. The sustained-lift dwell change is exercised (dwell counts to 12+ before advancing) but is irrelevant while no object is held. This is a direct, reproducible, evidence-backed failure of the fix path — **not** tuned away, per instructions. The next diagnostic step would be to determine why execution settles 1.5 mm below the commanded waypoint (IK tolerance `5e-4` vs 0.77 mm reported residual; possible table contact pushing the hand, or controller settling) before any further policy change.

## Verification on host (unchanged by the runs)

* `pytest tests -q`: **530 passed, 8 skipped** (exit 0).
* `ruff check src tests`: **All checks passed** (exit 0).
* Working tree before commit: only Part 2 recording/dashboard changes remain uncommitted, as required.
