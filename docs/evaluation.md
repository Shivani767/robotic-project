# Evaluation

**Evidence: Verified locally.** The benchmark suite ran on this machine.

## What a benchmark is here

A scenario family says *what to perturb*. A benchmark says what the perturbation
is **for**: which subsystem it stresses, what behaviour it predicts, and which
metrics would show the prediction wrong. Without that, a benchmark suite is a
list of sweeps, and every sweep produces numbers that are equally consistent with
"the fault is well modelled" and "the fault did nothing".

## The suite

20 runnable benchmarks across the six families the specification names, plus 4
faults recorded as **not effective** rather than approximated.

| family | cases | with a measured expectation |
|---|---|---|
| perception | 4 | 0 |
| depth | 4 | 2 |
| camera | 3 | 1 |
| robot | 3 | 2 |
| environment | 4 | 3 |
| planning | 2 | 0 |
| **total** | **20** | **8** |

Each declares a config path, a value, the metrics to record, and whether its
prediction has actually been observed. Run:

```bash
.venv/bin/python experiments/run_benchmark_suite.py
```

```
20 benchmarks, 0 expectation mismatches, 8 with a measured expectation
```

The 12 labelled *hypothesis* are reported separately because a prediction
derived from a model of a fault is not a measurement of it. An expectation that
makes no falsifiable claim reports `expectation_met=None` rather than passing,
since an unchecked expectation reported as verified would let a reader believe
more has been tested than has.

## The suite falsified three of my own expectations

This is the strongest evidence that it works.

| fault | I predicted | measured | verdict |
|---|---|---|---|
| `joint_offset` 0.05 rad | task fails | 6/6 succeed, +28 steps | expectation wrong |
| `joint_noise` 0.05 rad | task fails | 4/6 succeed | too strong |
| `gripper_stuck` 1.0 | grasp fails | **0 of 7 metrics move** | **fault is inert** |

The first two were corrected to assert the degradation they actually cause. The
third was moved to the recorded-missing list rather than deleted, because a
reader would otherwise assume the fault had been tested — it arms correctly and
**nothing reads it**.

Two safeguards exist for the failure mode this exposes, where a benchmark
measures the nominal condition and reports it under a fault's name:

* An injection whose corruption does not arm is reported as a **failure**, not a
  pass. `set_by_path` arms the group automatically, and the runner checks
  rather than assuming.
* A benchmark whose config path does not exist **raises**. The suite hit this
  for real: the colour-shift benchmark declared a scalar against a 3-tuple
  field, and the only reason it was visible instead of a silent pass is the
  check. `set_by_path` now accepts a sequence, since a colour cast is a
  per-channel quantity.

## Faults that are inert, and why that is a result

Camera latency, RGB occlusion and clutter are all 1.00 across their full swept
range. Not because the benchmark is weak but because the policy latches its plan
on the first control tick and never reads RGB. **A benchmark that expects
degradation there would report a false one.** That is a fact about this task and
policy pair, and it would transfer to any plan-once policy on a static object.

## Recorded as not effective

| fault | why |
|---|---|
| `perception_wrong_label` | no class-confusion model; category is fixed by the scenario |
| `planning_stale_world_model` | the policy re-reads its belief each tick; there is no cache to stale |
| `depth_quantization` | field exists in the model, not wired into the corruption layer |
| `robot_gripper_stuck` | arms correctly, 0/7 metrics move — nothing applies it |

Approximating a missing fault with something adjacent is worse than an
acknowledged gap, because the gap then stops being visible.

## The CI gate

```bash
robot-ci test --backend mujoco --seeds 4
```

Runs environment, regression and benchmark gates. Its design is dominated by one
property: **it must not report a pass it has not earned.**

* Every gate prints its coverage. Episode counts are a typed field, not parsed
  out of prose — a report that extracts its own totals from a sentence silently
  reports zero when the wording changes.
* A gate that cannot run is an **error**, never a skip.
* No blocking regressions exits 0 and prints the coverage that supports it, which
  is a different claim from having nothing to report.
* All gates run even after one fails, so one invocation reports every problem.

Found by running it: the condition column was the one being hidden. At 80 columns
rich truncated it to the bare scenario id, so four different noise levels all
printed as "nominal". The detector was producing `nominal [depth_noise_mm=2.0]`
correctly the whole time. Fixed by making the label short enough not to need
truncating.
