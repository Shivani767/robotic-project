# End-to-end demo, as run

Every block below is real output, captured as the command executed. Nothing is
transcribed from a plan or a previous session.

Run 2026-10-01, MuJoCo on Apple Silicon, 9 stages, **18 min 23 s** wall clock
(10:26:05Z to 10:44:28Z). Exit codes were recorded per stage; the two non-zero
ones are *correct* behaviour and are labelled below.

```bash
# stage 1: 60 episodes over a 5-cell condition grid
.venv/bin/robot-ci experiment run experiments/configs/demo_v1_vs_v3.yaml
# stage 2: conditional detection over the same episodes  -> exit 1 (blocking)
.venv/bin/robot-ci regression experiments/results/demo_v1_vs_v3
# stage 3: minimise a counterexample over a 12 mm axis
.venv/bin/robot-ci counterexample search --backend mujoco --strategy adaptive \
    --scenario experiments/configs/demo_seed3.yaml \
    --baseline policy_v3 --candidate policy_v1 --axis depth_noise_mm:0:12:mm
# stage 4: replay the artifact stage 3 wrote, on seeds it never saw
.venv/bin/robot-ci counterexample reproduce \
    "experiments/results/counterexamples/ce_counterexample_search_adaptive_depth_noise_mm=1.172.yaml" \
    --backend mujoco --seeds 3,7,11 --strict
# stage 5: attribute the difference  (15 sweep episodes + 14 ablation arms)
.venv/bin/robot-ci diagnose --scenario experiments/configs/demo_seed3.yaml \
    --backend mujoco --baseline policy_v3 --candidate policy_v1 \
    --axis depth_noise_mm:0:2:mm --seeds 3
# stage 6: self-contained HTML report
.venv/bin/robot-ci report experiments/results/demo_v1_vs_v3
# stages 7-9: the three experiment harnesses. Stages 7 and 8 write to /tmp so a
# demo run cannot overwrite the committed 120-seed and 16-seed artifacts.
.venv/bin/python experiments/establish_nominal_solvability.py --seeds 24 \
    --json /tmp/demo/nominal_24.json
.venv/bin/python experiments/rq2_opposing_effect.py --seeds 6 \
    --json /tmp/demo/rq2_opposing_6.json
.venv/bin/python experiments/run_end_to_end.py
```

## 1. Experiment — 60 real episodes on MuJoCo (91 s)

```
experiment demo_v1_vs_v3  backend=mujoco
baseline policy_v3  vs  candidate policy_v1
grid 5 cells x 6 reps = 60 episodes

                  results
 policy    runs  success  mean steps
 policy_v1    30     0.267       105.9
 policy_v3    30     0.300       121.1
```

The arms no longer sit pinned at the floor. Under the seed the runner derives,
this condition is solvable, so the comparison has somewhere to happen.

## 2. Conditional regression — the stage that exits 1

```
 baseline  : policy_v3 0.300 (n=30)
 candidate : policy_v1 0.267 (n=30)
 global Δ  : -0.033
 findings  : 1 (1 blocking)

 condition                  baseline  candidate   delta  severity
 nominal__depth_noise_mm=8     0.167      0.000  -0.167  blocking
 nominal__depth_noise_mm=0     0.833      0.833  +0.000  none
 nominal__depth_noise_mm=1     0.500      0.500  +0.000  none
 nominal__depth_noise_mm=2     0.000      0.000  +0.000  none
 nominal__depth_noise_mm=4     0.000      0.000  +0.000  none
```

**This stage exits 1 because a blocking regression was found — the gate working,
not the script failing.** It is also the project's headline claim, present in a
single demo run: the aggregate is **-0.033**, under the -0.10 blocking threshold
and would pass review, while the worst cell is **-0.167** at 8 mm and is caught.
Reading the aggregate alone ships the regression.

The two bottom rows are still at the floor (both arms fail), and the detector
correctly reports `none` there rather than inventing a finding. A conditional
analysis that invented one would be worse than useless.

## 3. Counterexample search — where a window does exist

Run against a scenario at a seed where the task is solvable, the adaptive search
finds and minimises a counterexample over a 12 mm axis:

```
counterexample found
 strategy   : adaptive
 evaluated  : 15 points
 severity   : 0.0977
 condition  : depth_noise_mm=1.17188
 baseline   : policy_v3 -> success
 candidate  : policy_v1 -> failure
 boundary   : 1.1484 (depth_noise_mm)
 bracket    : [0.0938, 0.0977]
 necessary  : depth_noise_mm
```

1.17 mm out of a 12 mm axis, from 15 oracle evaluations, bracketed to 0.11 mm.
The artifact is written as self-contained YAML — the full scenario, the
perturbation, the config hash and the git commit — so it can be replayed without
this repository's state.

## 4. Reproduction — the honest number, and a path that had gone stale

Replaying **the artifact stage 3 just wrote**:

```
 counterexample: ce_counterexample_search_adaptive_depth_noise_mm=1.172
 condition     : depth_noise_mm=1.17188
 seeds         : 3, 7, 11
 reproduced    : 1/3 (0.333)
 all agreed    : NO

 seed  baseline  candidate  reproduced
    3  success   failure    yes
    7  success   success    no
   11  failure   failure    no
```

`--strict` exits 1. Seed 7 has no regression to reproduce (both arms succeed);
seed 11 has no baseline to regress *from* (both fail). A 1.17 mm counterexample is
too mild a condition to separate these policies on most seeds, which is why the
rate is 1/3 and not 3/3 — **the number is the finding, and it must be reported
with the counterexample.**

> **A defect this run caught in its own documentation.** The previous revision of
> this file reproduced `experiments/results/demo_counterexample.yaml` — a copy of
> an artifact from an earlier session, still at `depth_noise_mm=0.75`. Replaying
> it today gives **0/3**: the two seeds that used to separate now succeed for both
> arms (seed 3: 1.00 / 1.00). Meanwhile the search has been writing its output to
> `experiments/results/counterexamples/` for some time, so the documented command
> was reproducing a stale file while the live artifact sat unreproduced. The
> command above now points at the path the search printed. This is the same
> failure class the project exists to catch — a mechanism that looks correct and
> does less than it appears to — found in the demo script itself.

## 5. Diagnose — association leads, substitution decides

The sweep finds a separation at 0.5 mm:

```
 condition  baseline  candidate   delta
         0      1.00      1.00  +0.00
       0.5      1.00      0.67  -0.33
         1      0.67      0.67  +0.00
       1.5      0.67      0.67  +0.00
         2      0.67      0.67  +0.00

regression condition   : depth_noise_mm=0.5
strongest association  : planner
confidence             : weak association; treat as a lead, not a conclusion
```

The association-based path names the **planner** and labels itself weak. The
ablation then substitutes one factor at a time:

```
 arm                                 success   change
 reference configuration, unmodified   0.73    +0.00
 settle_steps=6                        0.80    +0.07   <- the documented change
 clearance_bias_m=0.05                 0.00    -0.73   <- destructive arm
 ik_damping=0.2                        0.00    -0.73   <- destructive arm

 perception   +0.02  strongest association
 planner      +0.00
 controller   +0.00
 gripper      +0.00

7 factor(s) varied over 14 arm(s).
```

The ranking lands on **perception**, the subsystem the two policies actually
differ in, while the association heuristic named the planner. Arms whose
perturbation collapsed the task (to 0.00) are excluded from the ranking, because
ranking by effect size would otherwise always select the most destructive knob —
that guard is what stops the clearance and IK arms from winning on sheer damage.

The output ends by stating its own limitation: an association measured by
substitution, not a causal claim, and confirming it requires re-running the
attributed arm with that factor restored and everything else held fixed.

## 6. Report

```
report written: experiments/results/demo_v1_vs_v3/report.html
                 (4347 bytes, self-contained)
```

Leads with the per-condition matrix rather than the aggregate — the ordering the
stage-2 result argues for.

## 7. Nominal solvability — 24 seeds, 75 s

```
policy_v1   24/24 = 1.000  95% CI [0.862, 1.000]  steps 76.4 (76-78)
policy_v2   24/24 = 1.000  95% CI [0.862, 1.000]  steps 76.4 (76-78)
policy_v3   24/24 = 1.000  95% CI [0.862, 1.000]  steps 81.4 (81-83)

Nominal solvability for the baseline is 100.0% with a 95% lower bound of
86.2% over 24 seeds.
The previously reported 66.7% lies below the confidence interval, so it is
falsified rather than merely imprecise.
No episode failed, so there is no failure mode to attribute.
```

An independent replication of the committed 120-seed artifact at a quarter of
the sample: identical rate, identical step ranges (76-78 against 81-83). The
**5.0-tick** candidate cost that stage 8's design rests on appears again, exactly
as measured.

## 8. RQ2 opposing effect — sign inversion at 6 seeds

```
aggregate delta            -0.222
favourable cells (> +0.05) 4
adverse cells    (< -0.05) 12
tied cells                 8
  largest favourable       +0.17
  largest adverse          -1.00
```

| depth noise | budget 78 | budget 80 | budget 82 | budget 84 | budget 120 | budget 300 |
|---|---|---|---|---|---|---|
| 0 mm | **-1.00** | **-1.00** | +0.00 | +0.00 | +0.00 | +0.00 |
| 1 mm | -0.67 | -0.50 | +0.17 | +0.17 | +0.17 | +0.17 |
| 2 mm | -0.83 | -0.50 | +0.00 | +0.00 | +0.00 | +0.00 |
| 4 mm | -0.50 | -0.33 | -0.17 | -0.17 | -0.17 | -0.17 |

**The inversion reproduces live.** In every row the candidate is refused at a
budget of 78-80 — where its 81-83 ticks do not fit — and stops losing from 82
up, which is exactly where stage 7's step measurement puts it. No number was
fitted to locate that boundary.

**And the seed count matters, which is itself the lesson.** The committed
16-seed artifact for the same grid reports **11 favourable / 10 adverse / 3
tied, aggregate -0.102**, with a flip in *every* row; at 6 seeds the counts are
**4 / 12 / 8, aggregate -0.222**, and the 4 mm row does not flip. The direction
of the effect and the location of its boundary reproduce; the *counts* do not,
and neither should be quoted without `n`. Stage 8 writes to `/tmp` so this
reduced run cannot overwrite the recorded artifact it disagrees with.

## 9. End-to-end chain — byte-identical replication

```
stage 2/7  conditional regression detection
  aggregate +0.062  worst cell -0.333  blocking findings 3
stage 3/7  counterexample discovery
stage 4/7  counterexample replay
stage 5/7  behavioural boundaries
stage 6/7  failure taxonomy and attribution
stage 7/7  summary

  reproduction      3/11 (27.3%) at {'depth_noise_mm': 2.0}
  boundaries        {'depth_noise_mm': 1.0}
  attribution correlation_only     planner      WRONG
  attribution ofat                 perception   correct
  attribution interaction_aware    perception   correct
```

All seven stages re-ran over fresh episodes in 180 s. The regenerated
`end_to_end.json` compared field-by-field against the committed copy is
**identical except `wall_clock_s`** (156.1 → 180.5): the grid, the detector's
eight cells, the found counterexample, the reproduction rate, the boundary and
the attribution all returned the same numbers from new episodes.

## What the demo shows, and what it does not

It shows the full pipeline on real episodes: a conditional regression the
aggregate hides (stage 2), a minimised counterexample with an honest reproduction
rate (stages 3-4), an attribution where association and substitution disagree and
substitution is right (stage 5), the RQ2 sign inversion on a two-axis grid
(stage 8), and a chain that reproduces itself exactly (stage 9).

Two things it does **not** show, stated because they are what the numbers say:

* **The perceptual adverse channel.** Stage 8's adverse axis is a *deadline*.
  The candidate's own docstring predicted latency would be the adverse channel,
  and that is falsified: averaging suppresses variance, so it can only cost when
  the averaged quantity changes, and `ObjectSpec` has no motion field. Nothing in
  this framework moves an object.
* **That these rates are stable at demo sample sizes.** The 24-seed nominal
  interval reaches down to 0.862, and stage 8's cell counts move between n=6 and
  n=16. The committed artifacts (120 seeds, 16 seeds) are the record; the demo's
  reduced runs exist to show the commands execute and the direction survives, not
  to restate the estimates.

The previous revision of this file closed with the claim that "the framework's
sensitivity is bounded by its benchmark's", drawn from a 41.7%/50% nominal
floor. **That floor was a pre-fix 12-seed sample and is falsified** — see stage 7
and `experiments/results/nominal_solvability.json` (120/120, 95% CI
[0.969, 1.000]). The limitation it supported does not exist and is withdrawn.
