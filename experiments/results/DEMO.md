# End-to-end demo, as run

Every block below is real output from the commands shown. Nothing is
transcribed from a plan or a previous session.

```bash
robot-ci experiment run experiments/configs/demo_v1_vs_v3.yaml
robot-ci regression experiments/results/demo_v1_vs_v3
robot-ci counterexample search --backend mujoco --strategy adaptive \
    --scenario experiments/configs/demo_seed3.yaml \
    --baseline policy_v3 --candidate policy_v1 --axis depth_noise_mm:0:12:mm
robot-ci counterexample reproduce experiments/results/demo_counterexample.yaml \
    --backend mujoco --seeds 3,7,11 --strict
robot-ci diagnose --scenario experiments/configs/demo_seed3.yaml --backend mujoco \
    --baseline policy_v3 --candidate policy_v1 --axis depth_noise_mm:0:2:mm --seeds 3
```

## 1. Experiment -- 60 real episodes on MuJoCo

```
experiment demo_v1_vs_v3  backend=mujoco
baseline policy_v3  vs  candidate policy_v1
grid 5 cells x 6 reps = 60 episodes

                  results
 policy    runs  success  mean steps
 policy_v1    30     0.100       116.9
 policy_v3    30     0.100       121.7
```

**The two arms are identical, and that is the correct result here -- not a bug.**
The conditional table explains why:

```
 condition                  baseline  candidate   delta  severity
 nominal__depth_noise_mm=0     0.500      0.500  +0.000  none
 nominal__depth_noise_mm=1     0.000      0.000  +0.000  none
 nominal__depth_noise_mm=2     0.000      0.000  +0.000  none
 nominal__depth_noise_mm=4     0.000      0.000  +0.000  none
 nominal__depth_noise_mm=8     0.000      0.000  +0.000  none
```

Half the conditions are unsolvable at nominal, and at 1 mm of depth noise
*every* condition is. With both arms pinned at the floor there is no separation
left to detect, so the detector correctly reports none. A conditional analysis
that invents a finding here would be worse than useless.

## 2. The condition behind that floor

Auditing the seeds the runner actually derives:

```
nominal solvability, before the spawn fix:  5/12 = 41.7%
nominal solvability, after  the spawn fix:  6/12 = 50.0%
```

Object spawns were placed in a hard-coded 0.08-0.20 m annulus about a workspace
centre 0.60 m in front of the base, so placements reached 0.80 m from it. The
radius is now a config field, `object_spawn_radius_m`, defaulting to
0.05-0.13 m.

**The remaining failures are grasp failures, not unreachable placements.**
Placements now measure 0.052-0.113 m from the centre, inside the bound, and
episodes that fail report `unsatisfied criteria: lifted` -- the arm reached the
waypoint and the grasp did not hold. A parallel gripper commanded to 70% of the
object width on a 16 mm cube holds on roughly half of placements. That is an
embodiment-tuning limit, stated rather than papered over.

## 3. Counterexample search -- where a window does exist

Run against a scenario at a seed where the task is solvable, the adaptive search
finds and minimises a counterexample over a 12 mm axis:

```
counterexample found
 strategy   : adaptive
 evaluated  : 15 points
 severity   : 0.0625
 condition  : depth_noise_mm=0.75
 baseline   : policy_v3 -> success
 candidate  : policy_v1 -> failure
 boundary   : 0.7266 (depth_noise_mm)
 bracket    : [0.0586, 0.0625]
 necessary  : depth_noise_mm
```

0.75 mm out of a 12 mm axis, from 15 oracle evaluations. The artifact is written
as self-contained YAML -- the full scenario, the perturbation, the config hash
and the git commit -- so it can be replayed without this repository's state.

## 4. Reproduction -- the honest number

```
 counterexample: ce_counterexample_search_adaptive_depth_noise_mm=0.75
 condition     : depth_noise_mm=0.75
 seeds         : 3, 7, 11
 reproduced    : 1/3 (0.333)
 all agreed    : NO

 seed  baseline  candidate  reproduced
    3  success   failure    yes
    7  failure   failure    no
   11  failure   failure    no

a counterexample that reproduces only under some seeds is a property of the
seed, not of the condition; do not report it as a regression without stating the
reproduction rate.
```

Exit code 1 under `--strict`. Of the two non-reproducing seeds, both fail for
*both* arms -- there is no baseline success to regress from. A 0.75 mm
counterexample is too mild a condition to separate these policies on most seeds.

## 5. Diagnose -- and the bug it found

The sweep finds a clean separation:

```
 condition  baseline  candidate   delta
         0      1.00      1.00  +0.00
       0.5      1.00      0.67  -0.33
         1      0.33      0.33  +0.00
       1.5      0.33      0.33  +0.00
         2      0.33      0.33  +0.00
```

The ablation then attributed the regression to the **wrong subsystem**:

```
 arm                              success   change
 reference configuration            0.53    +0.00
 max_position_sigma_m=0.004         0.53    +0.00
 settle_steps=6                     0.60    +0.07   <- the known cause
 clearance_bias_m=0.05              0.00    -0.53
 grasp_bias_m=0.01                  0.53    +0.00
 ik_damping=0.2                     0.00    -0.53
 ik_restarts=24                     0.53    +0.00
 grasp_width_fraction=0.85          0.67    +0.13

 planner      -0.13  strongest association
 controller   -0.13  strongest association
```

Raising the clearance bias to 50 mm drops the object; raising the IK damping to
0.2 stops the solver converging. Both take every episode to failure, and ranking
by absolute effect size selects for them -- while the arm that actually explains
the regression ranked third.

Fixed in `c9d9fb6`: arms at the success floor are excluded from the ranking and
reported in `collapsed_arms`. The same arms now attribute to gripper (+0.14) and
perception (+0.07).

## What the demo shows, and what it does not

It shows the full pipeline working on real episodes: two-arm experiment,
conditional detection, counterexample search and minimisation, replay with a
reproduction rate, and ablation-based attribution.

It also shows the platform's honest weakness, which is worth more than a clean
run would have been. **On the runner's own seed derivation the task is solvable
about half the time with no corruption at all**, so most cells sit at the floor
and a real 10% regression is invisible inside that. The experiment that produced
a headline result in section 3 had to pin a specific seed to get a window at
all. Until the grasp succeeds reliably, the framework's sensitivity is bounded
by its benchmark's, and that is the thing to fix next.
