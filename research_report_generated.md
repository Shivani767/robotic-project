# Robot CI: research report (generated)

Generated 2026-09-30 19:31 UTC by `experiments/make_research_report.py` from the artifacts in `experiments/results/`. Every number below is read from those files; none is transcribed.

This file is machine-generated and therefore never stale with respect to the runs. The hand-written `research_report.md` carries the narrative, the method and the limitations, and is the document to read for those.

## Verification state

Record at commit `f68ec14`, overall **0**.

| check | exit |
|---|---|
| ruff (src, tests, experiments) | 0 |
| ruff format --check | 1 |
| mypy (src/robot_ci) | 0 |
| unit | 0 |
| integration | 0 |
| end to end | 0 |
| robot-ci validate | 0 |
| package imports without mujoco | 0 |

Exit codes, not output text. See `docs/verification.md` for the two failures of the verification process itself that this rule was adopted to fix.

## The end-to-end demonstration

One command, `experiments/run_end_to_end.py`, running the whole chain against fresh episodes rather than against the recorded artifacts. It reproduced every previously recorded finding independently, which is the check that matters: the earlier numbers were not produced by a process that only works once.

Arms `policy_v3` (baseline) and `policy_v1` (candidate), differing in one subsystem (perception).

| condition | baseline | candidate | delta | 95% CI (candidate) |
|---|---|---|---|---|
| nominal [depth_noise_mm=0.0] | 1.00 | 1.00 | +0.00 | [0.61, 1.00] |
| nominal [depth_noise_mm=1.0] | 0.83 | 0.33 | -0.50 | [0.10, 0.70] |
| nominal [depth_noise_mm=2.0] | 0.83 | 0.50 | -0.33 | [0.19, 0.81] |
| nominal [depth_noise_mm=4.0] | 0.50 | 0.33 | -0.17 | [0.10, 0.70] |
| nominal [depth_noise_mm=8.0] | 0.17 | 0.17 | +0.00 | [0.03, 0.56] |

Aggregate **-0.200** against a worst cell of **-0.500**, with **3** blocking findings: `nominal [depth_noise_mm=1.0]` (-0.50), `nominal [depth_noise_mm=2.0]` (-0.33), `nominal [depth_noise_mm=4.0]` (-0.17)

The aggregate is a quarter of the worst condition it averages over. Reporting only the aggregate would have shown a modest, plausibly tolerable -0.20 change, and missed the -0.50 condition entirely.

## Aggregate and conditional metrics

Arms `policy_v3` (baseline) and `policy_v1` (candidate), 8 seeds x 5 conditions.

| depth noise | baseline | candidate | delta |
|---|---|---|---|
| 0 mm | 1.00 | 1.00 | +0.00 |
| 1 mm | 0.88 | 0.50 | -0.38 |
| 2 mm | 0.88 | 0.50 | -0.38 |
| 4 mm | 0.62 | 0.25 | -0.38 |
| 8 mm | 0.25 | 0.12 | -0.12 |

Aggregate over 40 runs: **-0.250** (baseline 0.725, candidate 0.475).

Worst conditional cell: **-0.375**. The aggregate reports 67% of the severity of the worst condition it averages over, and 4 cells cross the blocking threshold.

## Counterexample search

Budget 24 oracle evaluations each, seed 3.

| strategy | evals | wall clock | counterexample | minimal |
|---|---|---|---|---|
| grid | 2 | 4.4 s | 4.00 mm | no |
| random | 1 | 1.4 s | 7.64 mm | no |
| binary | 8 | 17.1 s | 1.17 mm | yes |
| adaptive | 15 | 26.0 s | 1.17 mm | yes |

Random finds a counterexample in 1 evaluation against adaptive's 15, but needs 6.5x the perturbation to reproduce. **Tests-to-failure alone ranks random best, and that is the wrong answer.** One trial per strategy on one axis and one seed: this shows the trade-off, not that a strategy is better.

## Counterexample reproducibility

Counterexample `{'depth_noise_mm': 2.0}` replayed on 11 seeds disjoint from the discovery seed (3).

**Reproduction rate: 3/11 = 27.3%**

a rate well below 1.0 means the counterexample is a property of the discovery seed as much as of the condition. It must not be reported as a behavioural boundary without this number attached.

## Behavioural boundaries

Axes with a located boundary: `{'depth_noise_mm': 1.0}`.

Axes with no separation: 6 — `['depth_bias_mm', 'occlusion_percent', 'joint_noise_rad', 'latency_ms', 'clutter_count', 'object_friction']`.

![boundary curves](experiments/results/boundary_curves.png)

## Failure-injection benchmark

20 benchmarks, **0 expectation mismatches**, 8 with a measured expectation and 12 labelled hypotheses.

| benchmark | family | success | expectation | basis |
|---|---|---|---|---|
| perception_blur | perception | 1.00 | met | hypothesis |
| perception_occlusion | perception | 1.00 | met | hypothesis |
| perception_low_light | perception | 1.00 | met | hypothesis |
| perception_colour_shift | perception | 1.00 | met | hypothesis |
| depth_noise | depth | 0.00 | met | measured |
| depth_bias | depth | 0.00 | met | measured |
| depth_dropout | depth | 1.00 | met | hypothesis |
| depth_latency | depth | 1.00 | met | hypothesis |
| camera_latency | camera | 1.00 | met | measured |
| camera_extrinsic_error | camera | 1.00 | met | hypothesis |
| camera_intrinsic_error | camera | 1.00 | met | hypothesis |
| robot_joint_offset | robot | 1.00 | met | measured |
| robot_joint_noise | robot | 0.67 | met | measured |
| robot_encoder_delay | robot | 1.00 | met | hypothesis |
| environment_object_displacement | environment | 0.83 | met | measured |
| environment_clutter | environment | 1.00 | met | measured |
| environment_obstacle | environment | 0.00 | met | hypothesis |
| environment_friction | environment | 1.00 | met | measured |
| planning_wrong_target | planning | 0.00 | met | hypothesis |
| planning_blocked_path | planning | 1.00 | met | hypothesis |

Faults recorded as **not effective** rather than approximated:

- `perception_wrong_label` — No class-confusion model; object category is fixed by the scenario.
- `planning_stale_world_model` — The policy re-reads its own object belief each tick rather than a cached world model, so there is no staleness to inject.
- `depth_quantization` — Field exists in the sensor model but is not yet wired into the corruption application.
- `robot_gripper_stuck` — The field exists and the corruption arms correctly, but nothing applies it: measured at 1.0, zero of seven metrics move against nominal. The suite found this by falsifying the benchmark's own expectation, which is the clearest argument for a suite that checks its predictions.

## Attribution, scored against injected ground truth

`policy_v3` and `policy_v1` differ in exactly one subsystem: **perception**.

| method | implicated | episodes | correct |
|---|---|---|---|
| correlation_only | planner | 12 | **no** |
| ofat | perception | 216 | yes |
| interaction_aware | perception | 216 | yes |

OFAT and interaction-aware analysis both recover the injected cause; correlation-only analysis does not

## Falsified, withdrawn and unresolved

The results above are the ones that worked. These are not, and they are the more useful half of the record.

**Falsified by measurement, not by argument:**

* *Camera latency and RGB occlusion are behavioural factors.* Measured flat at 1.00 across 0-200 ms and 0-60%. The policy latches its plan on the first control tick and never reads RGB. Sweeping them costs episodes to measure nothing.
* *Depth bias is survivable.* A 2 mm systematic offset takes both arms to 0.00 while 1 mm of Gaussian noise still leaves the baseline at 0.67. Averaging removes noise; it cannot remove bias. A benchmark sweeping only Gaussian noise would conclude perception is robust to depth error, which is false.
* *The failing grasp is an aiming problem.* End-effector distance to the object does not predict success: failures at 1.4 mm and 2.2 mm, successes at 2.2 mm and 2.4 mm. Also tested and rejected: lift shear (halving joint velocity changes nothing) and grip width (0.6 worse, 0.75/0.8/0.9 identical to the 0.7 default).

**Withdrawn after re-running:**

* *Joint-encoder noise changes the sign of the effect along its axis* and *object friction is non-monotonic*. Both were artefacts of a loose 2 mm IK tolerance and both are flat after it was tightened. A measurement that changes when an unrelated constant changes was never a measurement of the policy.

**Unresolved, and load-bearing:**

* *Nominal solvability is 66.7%.* Roughly a third of episodes fail with the grasp not holding, and no lever in the policy moves it. Until that changes, the framework's sensitivity is bounded by its benchmark's: a real 10% regression can sit inside a 33% failure floor.
* *The strong form of RQ2 is not demonstrated.* It needs a grid where favourable and adverse cells cancel. The candidate is adverse everywhere it differs here.
* *RQ1 is one trial per strategy on one axis and one seed.*
* *The pairwise ablation design is untested on the only case it exists for* — a regression with a genuine two-factor interaction. None has been constructed.

## Not verified here

* Isaac Sim adapter — **implemented but unverified**; no macOS build, never executed.
* ROS 2 and MoveIt 2 — **not started**; `ros2_ws/src/` holds three empty directories.
* Notebooks and cloud execution — **not started**.
* The Linux CI workflow — **written, never run**.

See `docs/verification.md` for the full verified/unverified split.
