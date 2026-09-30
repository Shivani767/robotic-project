# Measured results

Every number below came from a command in this repository. Nothing is projected,
interpolated, or illustrative. Negative results are recorded as findings, and
experiments that have not been run say so rather than carrying a placeholder
that could later be mistaken for a measurement.

## Reproducing

```bash
.venv/bin/python -m robot_ci.cli.main validate
.venv/bin/python experiments/rq1_search_strategies.py     # writes rq1_*.json
.venv/bin/python experiments/rq4_reproducibility.py        # writes rq4_*.json
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

## 4. RQ1: search strategies

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

## 5. RQ4: counterexample reproducibility

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

## 6. Diagnosis output

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

- RQ2 (aggregate vs conditional evaluation) as a standalone experiment. The
  depth-noise grid in section 2 is conditional evaluation, and it is what
  established the v1/v2 null, but the aggregate-vs-conditional contrast has not
  been run as its own comparison.
- RQ3 (ablation: correlation only vs OFAT vs interaction-aware). The ablation
  engine is built, tested and exposed through `robot-ci diagnose`, but no
  ablation has been run against a real regression on the simulator.
- RQ5 (multi-axis perturbation boundary curves).
- Dashboard, ROS 2 / MoveIt / Isaac backends, notebook backend.
