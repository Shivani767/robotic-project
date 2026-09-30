# Measured results

> **Staleness notice.** Commit `dc61715` changed `ik_pos_tol` from 2 mm to
> 0.5 mm, which changed task behaviour: nominal solvability on 12 seeds rose
> from 50.0% to 66.7%. **RQ2, RQ3 and RQ5 have been re-run and are current.
> RQ1 and RQ4 are still stale** -- they were measured before that commit and are
> labelled in place. Re-run them before quoting them.
>
> Two findings from the earlier RQ5 run were withdrawn on re-run: joint-encoder
> noise changing sign along its axis, and object friction being non-monotonic.
> Both were artefacts of the loose IK tolerance. See section 8.

Every number below came from a command in this repository. Nothing is projected,
interpolated, or illustrative. Negative results are recorded as findings, and
experiments that have not been run say so rather than carrying a placeholder
that could later be mistaken for a measurement.

## Reproducing

```bash
.venv/bin/python -m robot_ci.cli.main validate
.venv/bin/python experiments/rq1_search_strategies.py     # writes rq1_*.json
.venv/bin/python experiments/rq4_reproducibility.py        # writes rq4_*.json
.venv/bin/python experiments/rq2_conditional_vs_aggregate.py  # writes rq2_*.json
.venv/bin/python experiments/rq3_ablation_attribution.py     # writes rq3_*.json
.venv/bin/python experiments/rq5_perturbation_boundaries.py  # writes rq5_*.json
.venv/bin/python -m robot_ci.cli.main diagnose --backend mujoco \
    --axis depth_noise_mm:0:8:mm --seeds 4
```

## 1. Task success on the MuJoCo arm

`policy_v1`, 200+ step budget, pick-and-place, MuJoCo.

| seeds | successes | rate |
|---|---|---|
| 1..24 | 12 | 50.0% |

This was **0.000** before commit `3309762`. Four independent defects, each found
by measurement, are documented in that commit.

### Episode length, and why the step cap no longer dominates

| outcome | steps | note |
|---|---|---|
| success | 76-77 | |
| failure | 127-128 | stall-detected |

Failures previously ran the full 300-tick cap. The runner could not distinguish a
stuck arm from a slow one, and the case that accounted for every 300-tick
failure was not a motion stall at all: the arm arrived within tolerance and sat
there because the grasp never registered and the waypoint cursor never advanced.
Distance-to-waypoint cannot see that; the policy's waypoint index can.

## 2. policy_v1 vs policy_v2: no regression (null result)

8 seeds per cell, bounded gripper, MuJoCo.

| depth noise | v1 success | v2 success | delta |
|---|---|---|---|
| 0 mm | 1.00 | 1.00 | +0.00 |
| 2 mm | 0.62 | 0.62 | +0.00 |
| 4 mm | 0.50 | 0.50 | +0.00 |
| 6 mm | 0.25 | 0.38 | +0.12 |
| 8 mm | 0.00 | 0.00 | +0.00 |

**There is no success-rate regression between these two arms.** The +0.12 at
6 mm is one seed out of eight and the framework's own intervals do not separate
it from zero.

The candidate's gate is a bounded *fail-open* hold, so what it spends is control
ticks rather than successes: at 4 mm the candidate takes a median 192 steps
against the baseline's 188 (+0.23 s), with identical success.

This null was not the first result. The gate originally tripped at 50 mm of
depth noise, where both arms had already failed for unrelated reasons, so the two
arms were the same policy under different labels and the "null" was an artefact.
See commit `b0eae7e`.

## 3. policy_v3: a positive control, and a falsified half-hypothesis

`policy_v3` averages pose samples before committing to a plan. It exists so the
detection machinery can be validated against a regression known to exist.

6 seeds, MuJoCo:

| depth noise | v1 | v3 | delta |
|---|---|---|---|
| 0 mm | 1.00 | 1.00 | +0.00 |
| 4 mm | 0.00 | 0.17 | +0.17 |
| 8 mm | 0.00 | 0.17 | +0.17 |
| 12 mm | 0.00 | 0.17 | +0.17 |

The prediction had two halves. **The first is confirmed**: averaging wins where
depth noise dominates. **The second is falsified**: it was predicted to lose
under sensor latency, and camera latency turns out to be completely inert on
this task (v1 = 1.00 at 0, 50, 100 and 200 ms).

The reason is structural: the policy latches its plan on the first control tick
and the object never moves, so a delayed frame is never acted on. **Latency is
not a behavioural factor for any plan-once policy on a static-object task**, and
a benchmark that sweeps it is spending compute to measure nothing. Recorded
because a falsified prediction is a result.

## 4. RQ1: search strategies  <!-- STALE (pre-dc61715) -->

`experiments/rq1_search_strategies.py`. Arms are `baseline=policy_v3`,
`candidate=policy_v1` -- the searcher looks for conditions where the candidate
fails and the baseline succeeds, and `policy_v1` is the arm that regresses.

Budget 24 oracle evaluations, identical for every strategy, seed 3.

| strategy | evals | wall clock | counterexample | minimal? |
|---|---|---|---|---|
| random | **1** | 1.4 s | 7.6 mm | no |
| grid | 2 | 4.2 s | 4.0 mm | no |
| binary | 8 | 16.1 s | **0.75 mm** | yes |
| adaptive | 15 | 24.6 s | **0.75 mm** | yes |

**Tests-to-failure alone ranks random best, and that is the wrong answer.**
Random is 15x cheaper than adaptive and its counterexample is 10x less minimal:
it needs 7.6 mm of noise to reproduce where 0.75 mm suffices. A strategy that
reports only "found" is hiding half its result.

Limitations, which matter for how this is quoted:

* One trial per strategy. This cannot support a claim that any strategy is
  better in general; it shows the trade-off exists and that the two metrics must
  be reported together.
* One axis, one seed. A per-seed separation exists on seed 3 and on 0 of the
  other 11 seeds scanned, so the discovery itself is seed-specific (see RQ4).
* Binary and adaptive spend their budget *minimising*; grid and random spend
  theirs finding *something*. They are solving different problems, and the
  comparison is only fair because both metrics are shown.

## 5. RQ4: counterexample reproducibility  <!-- STALE (pre-dc61715) -->

`experiments/rq4_reproducibility.py`. The counterexample discovered above is
`depth_noise_mm = 1.0`, replayed on 11 seeds disjoint from the discovery seed.

**Reproduction rate: 4 / 11 = 36.4%**

Of the 7 non-reproducing seeds, 5 are ones where both arms succeed and 2 where
both fail. A 1 mm counterexample is too mild a condition to separate the arms on
most seeds.

This is the number that has to travel with any claim of a behavioural boundary.
A counterexample reproduced on a third of seeds is a property of the seed as
much as of the condition, and reporting it without this figure would be
misleading.

## 6. RQ2: conditional vs aggregate evaluation

`experiments/rq2_conditional_vs_aggregate.py`. Arms `baseline=policy_v3`,
`candidate=policy_v1`, 8 seeds x 5 noise levels = 40 runs per arm.

| depth noise | baseline | candidate | delta |
|---|---|---|---|
| 0 mm | 8/8 | 8/8 | +0.00 |
| 1 mm | 7/8 | 4/8 | **-0.38** |
| 2 mm | 7/8 | 4/8 | **-0.38** |
| 4 mm | 5/8 | 2/8 | **-0.38** |
| 8 mm | 2/8 | 1/8 | -0.12 |

| scoring | result |
|---|---|
| aggregate over 40 runs | baseline 0.725, candidate 0.475, **delta -0.250** |
| conditional | **4 blocking findings**, worst **-0.38** |

**The aggregate reports two thirds the severity of the worst condition it
averages over** (-0.25 against -0.38), and conditional evaluation names the
three conditions responsible. Three cells sit at exactly -0.38, which is a
consistent gradient rather than three independent coincidences.

**The stronger form of the claim is not demonstrated here.** RQ2 as specified
asks whether conditional evaluation finds regressions that aggregate metrics
*hide* -- aggregate near zero, conditional strongly negative. That requires a
grid where the candidate is better in some cells and worse in others, so the
effects cancel. On this grid the candidate is adverse in every adverse cell and
neutral at nominal, so nothing cancels and the aggregate is also adverse.
Demonstrating the strong form needs a pair whose trade-off runs in opposite
directions across two axes; no such pair has been constructed yet.

### This experiment found the project's worst bug

The first run of RQ2 returned one finding labelled "nominal" with delta -0.125
and never mentioned the -0.25 cell. The detector was grouping conditional cells
by `scenario_id`, so a sweep that perturbs a scenario in place -- which is what
the counterexample search, `robot-ci diagnose` and this script all do -- pooled
every cell into one. A conditional analysis that conditions on nothing returns
an aggregate result under a name promising otherwise, and the output looked
exactly like a correct one.

Fixed in `5f40a24`. Everything measured before that commit using the
conditional path -- including the RQ1 strategy comparison, which drives the same
searcher -- was produced by code that could not separate conditions.

## 7. RQ3: attributing a regression to a subsystem  <!-- RE-RUN post-dc61715 -->

`experiments/rq3_ablation_attribution.py`. Arms `baseline=policy_v3`,
`candidate=policy_v1`, 6 seeds at 2 mm depth noise.

**Ground truth is known, not inferred.** The two arms differ in exactly one
subsystem: `policy_v3` averages pose samples before committing to a plan, which
is a perception-stage change. A method that names the controller or the gripper
has identified the wrong thing.

| method | implicated | episodes | correct? |
|---|---|---|---|
| correlation only | **planner** | 12 | **no** |
| OFAT | **perception** | 216 | **yes** |
| interaction-aware | **perception** | 216 | **yes** |

**The conclusion survived the IK change that overturned two of RQ5's findings,
and the baseline got more confidently wrong:**

| correlation | metric | subsystem |
|---|---|---|
| -0.993 | trajectory_deviation_m | planner |
| -0.423 | placement_error_m | planner |
| +0.365 | pose_sigma_m | **perception** |
| -0.194 | grasp_error_m | gripper |

The metric that correlates almost perfectly with the outcome is a downstream
consequence of the regression, not its cause, and the perception metric that
encodes the cause now ranks *third*, further behind than before. Correlation
finds what co-varies with failure; ablation finds what causes it.

The cost is 18x the episodes (216 against 12) for a correct answer instead of a
plausible wrong one.

### Why this one held and RQ5's did not

Worth recording, because the contrast is the useful part. Both experiments were
re-run after `dc61715` changed the IK tolerance, and they went opposite ways:
RQ3's conclusion was unchanged while RQ5 lost two of its findings.

The difference is what each claim rests on. RQ3 compares a *ranking* over 36
arms, so it is integrating over many measurements and a global change in solver
accuracy moves every arm together. RQ5's withdrawn claims were *single-cell*
boundary readings on a near-threshold axis at n=6, which is exactly the shape of
claim that a global tuning change can invalidate. A method-comparison result
and a single-cell curve reading do not deserve the same confidence, and it is not
obvious in advance which is which.

### The interaction-aware design added nothing here

OFAT and interaction-aware return the same answer, because this regression has
no interaction: one parameter, one subsystem, one effect. The 21 pairwise arms
doubled the cost and changed nothing. Reported rather than presented as a
success, because a design that only pays off on regressions with genuine
interactions should be justified by evidence of such a regression, and none has
been constructed yet.

### Cost of the small grid

One noise level of six seeds. The price is noisy effects: at n=6 a single
episode moves a success rate by 0.167, which is why the arms are tied in
places. The perception effect is clear of them, which is enough to rank and not
enough to call precise.

## 8. RQ5: perturbation boundary curves  <!-- RE-RUN post-dc61715 -->

`experiments/rq5_perturbation_boundaries.py`. Each axis swept alone with the
others nominal, 6 seeds, arms `baseline=policy_v3, candidate=policy_v1`.

### Exactly one axis has a boundary

**Depth noise:**

| depth noise | baseline | candidate | delta |
|---|---|---|---|
| 0 mm | 1.00 | 1.00 | +0.00 |
| 1 mm | 0.83 | 0.33 | **-0.50** |
| 2 mm | 0.83 | 0.50 | -0.33 |
| 4 mm | 0.50 | 0.33 | -0.17 |
| 8 mm | 0.17 | 0.17 | +0.00 |

Identical at nominal, a 0.50 gap at the first swept level, converging as the
condition gets bad enough for both arms to fail. This is the curve shape the
framework is looking for.

### Six axes have no separation, and that is now a cleaner result

| axis | range swept | outcome |
|---|---|---|
| latency_ms | 0-200 | 1.00 throughout |
| occlusion_percent | 0-60 | 1.00 throughout |
| clutter_count | 0-9 | 1.00 throughout |
| object_friction | 1.0-0.1 | 1.00 throughout |
| joint_noise_rad | 0-0.05 | 1.00 until 0.05, where both fall to 0.67 |
| depth_bias_mm | 0-20 | bimodal, see below |

**Latency and occlusion are structurally inert**: the policy latches its plan on
the first control tick and never consults RGB, so a delayed or occluded frame is
never acted on. They are not behavioural factors for this task, and sweeping them
spends episodes to measure nothing.

### What changed from the pre-`dc61715` run, and why it matters

The earlier run found **two** boundaries and reported that joint-encoder noise
changed the *sign* of the effect along its axis, and that friction was
non-monotonic (0.50 at friction 1.0 against 1.00 below). **Both of those
findings were artefacts of the loose 2 mm IK tolerance.** With the solver
tightened to 0.5 mm:

* joint noise is now flat at 1.00 until 0.05 rad, where *both* arms fall
  together -- no sign change, no boundary
* friction is 1.00 at every level, so the apparent anomaly at friction 1.0 is
  gone

A measurement that changes when an unrelated tuning constant changes was never a
measurement of the policy. This is the clearest evidence in the project that
re-running after a change is not optional, and it is why the other stale
sections are labelled rather than quietly kept.

### A methodological caution from the depth-bias axis

| depth bias | baseline | candidate |
|---|---|---|
| 0 mm | 1.00 | 1.00 |
| 2 mm | 0.00 | 0.00 |
| 5 mm | **1.00** | **1.00** |
| 10 mm | 0.00 | 0.00 |
| 20 mm | 0.00 | 0.00 |

A 5 mm *systematic* offset succeeding where a 2 mm one fails is not a physical
result. At n=6 a near-threshold condition produces coin-flip cells, and a curve
through them is an artefact of which seeds happened to be drawn. Reported
because it is the clearest demonstration in the project that **n=6 cannot
support a curve**: individual cells here are 0.00 or 1.00 with nothing between.
Any axis near its threshold needs more repetitions before its shape means
anything.

## 9. Diagnosis output

`robot-ci diagnose` was validated by running it, and its first output was wrong:
it attributed a regression to the planner on a 64 ms difference in
`planning_time_ms` between two arms whose success rates agreed to sixteen digits.
Planning time is wall-clock. Two guards now prevent that, both pinned by tests:

* cost metrics are excluded from attribution (`planning_time_ms`; episode length
  and completion time are *not* excluded, because spending extra control ticks
  to reach the same place is a real behavioural cost)
* a diagnosis with no success change reports no regression and names no
  subsystem

## Not yet run

No numbers exist for these and none are written anywhere in this repository.

- The *strong* form of RQ2: a grid on which favourable and adverse cells cancel
  in the aggregate. See section 6.
- The *interaction* case for RQ3: no regression with a genuine two-factor
  interaction has been constructed, so the pairwise design is untested on the
  only situation it exists for. See section 7.
- Multi-axis boundary surfaces. RQ5 sweeps one axis at a time, which attributes
  a boundary to a factor but cannot see interactions between them.
- Dashboard, ROS 2 / MoveIt / Isaac backends, notebook backend.
