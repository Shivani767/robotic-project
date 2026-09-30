# Measured results

Every number in this file came from a command recorded in the repository. Nothing
here is projected, interpolated, or illustrative. Where an experiment has not
been run, the table says so rather than carrying a placeholder that could later
be mistaken for a measurement.

## How to reproduce

```bash
# Phase-4 status: does the arm complete pick-and-place at all?
.venv/bin/python -m robot_ci.cli.main validate

# Task success across seeds, both policies
# (script: the seed scan reported in "Task success" below)

# Depth-noise conditional grid, 8 seeds per cell
# depth_noise_mm  v1 succ  v2 succ  delta
```

## Task success on the MuJoCo arm

`policy_v1`, `policy_v2`, bounded gripper, 200-step budget, pick-and-place.

| seeds | successes | rate |
|---|---|---|
| 1..24 scan (24 seeds) | 12 | 50.0% |
| named sweep (42, 7, 1234) x 2 arms | 4 / 6 | 66.7% |

This was **0.000** before commit `3309762`. The four defects fixed there are
documented in that commit message; each was found by measurement, not inspection.

The two arms agree on every seed tried. That is expected and is verified
separately in `tests/unit/test_policy_resolution.py`: they differ by exactly one
parameter, the confidence-gate threshold, and the only failure mode in the
nominal band is one the gate does not address.

## Depth-noise conditional grid (RQ2 setup)

8 seeds per cell (1, 3, 4, 6, 8, 9, 12, 13), bounded gripper, MuJoCo.

| depth noise | v1 success | v2 success | delta |
|---|---|---|---|
| 0 mm | 1.00 | 1.00 | +0.00 |
| 2 mm | 0.62 | 0.62 | +0.00 |
| 4 mm | 0.50 | 0.50 | +0.00 |
| 6 mm | 0.25 | 0.38 | +0.12 |
| 8 mm | 0.00 | 0.00 | +0.00 |

**This is a null result and is reported as one.** There is no success-rate
regression between the two arms on this axis. The +0.12 at 6 mm is a single
seed out of eight; with n=8 per cell it is not distinguishable from zero, and
the framework's own confidence intervals say so.

### What the candidate does cost

The candidate's gate is a bounded fail-open hold, so what it spends is steps
rather than successes. Same seeds:

| depth noise | arm | success | median steps | median completion |
|---|---|---|---|---|
| 0 mm | v1 | 1.00 | 74 | 3.72 s |
| 0 mm | v2 | 1.00 | 74 | 3.72 s |
| 4 mm | v1 | 0.50 | 188 | 9.40 s |
| 4 mm | v2 | 0.50 | 192 | 9.63 s |
| 6 mm | v1 | 0.25 | 300 | 15.00 s |
| 6 mm | v2 | 0.38 | 300 | 15.00 s |
| 8 mm | v1 | 0.00 | 300 | 15.00 s |
| 8 mm | v2 | 0.00 | 300 | 15.00 s |

Identical at nominal, +4 steps at 4 mm, and no difference at 6 mm or beyond.

### An observation that matters more than the null

At 6 mm and 8 mm the median is 300 steps -- the step cap. Under depth noise
this arm is not misplacing the object; it is **running out of budget**. Success
at 6 mm is 0.25 while `placement_error` on the failures stays small. The
binding constraint in the noise band is episode length, not grasp accuracy.

That is a finding about the harness and the control loop, not about the policy
comparison, and it is the first thing to fix before any of RQ1-RQ5 can give a
clean answer: a benchmark whose failures are dominated by a timeout is not
measuring what it claims to measure.

## Counterexample search

`robot-ci counterexample search --backend mujoco --strategy adaptive --budget 12
--axis depth_noise_mm:0:20:mm`, default scenario:

```
no counterexample found
strategy  : adaptive
evaluated : 9 points
note      : no counterexample found along the severity ramp
exit code : 0
```

A true negative, and a searched one: 9 points were evaluated before concluding
it. It used to report the same headline after evaluating **1** point, because
the default seed was one the task cannot be solved on -- see commit `88da704`.

## Confidence-gate window

The gate trips when the fused pose sigma exceeds its threshold. The mapping from
depth noise to that sigma is analytic (`sensors.corruption.belief_position_sigma_m`):

| depth noise | belief sigma | gate fires (threshold 0.004 m) |
|---|---|---|
| 0 mm | 0.0020 m | no |
| 5 mm | 0.0054 m | yes |
| 10 mm | 0.0102 m | yes |
| 50 mm | 0.0500 m | yes |
| 100 mm | 0.1000 m | yes |

The threshold was previously 0.05 m, i.e. 50 mm of depth noise -- past the point
where the arm had already failed for an unrelated reason, so the gate could
never engage and the two arms were the same policy under different labels. See
commit `b0eae7e`.

## Not yet run

These are specified in the plan and are **NOT RUN**. No numbers exist for them
and none are written anywhere in this repository.

- RQ1  search-strategy comparison (grid / random / binary / adaptive)
- RQ3  ablation engine: OFAT and interaction-aware subsystem attribution
- RQ4  counterexample reproduction rate across seeds
- RQ5  multi-axis perturbation boundary curves
- Diagnosis and failure-taxonomy output
- Dashboard
