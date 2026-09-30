# Diagnosis

**Evidence: Verified locally.** The taxonomy, ablation engine and diagnosis
writer all ran on this machine, and the ablation was scored against a known
injected cause.

## The chain

```
taxonomy.py    what went wrong       -- several weighted hypotheses
ablation.py    which component       -- substitution experiments
diagnose.py    write it down        -- hedged, with the numbers attached
```

Each is useless alone. A taxonomy with no attribution says "the grasp failed";
a correlation with no taxonomy says "trajectory deviation went up"; neither tells
anyone what to change.

## Failure taxonomy

Fifteen categories, ordered from sensing to acting, which is also the order an
investigation works in. A failure yields **several weighted hypotheses**, never
one label:

```
$ classify_failure(grasp_success=0, pose_sigma=0.02, timeout=1)
0.52  depth_failure              pose sigma 20.0 mm exceeds the 4 mm grasp budget
0.80  pose_estimation_failure    object pose uncertainty 20.0 mm
0.85  grasp_failure              grasp not achieved with 20.0 mm pose sigma
0.80  timeout                    episode exhausted its step budget
```

A grasp that closed on nothing *because* the pose was uncertain is
simultaneously a perception failure and a grasp failure, and collapsing that
loses the distinction that says which subsystem to look at. Every hypothesis
carries the metric values that fired its rule, so a label is never separated
from the numbers behind it.

**Absent metrics are treated as absent, not as zero.** `grasp_success`
defaulting to 0.0 made every episode that merely failed to *report* it look like
a grasp failure, and a classifier that invents evidence for a mechanism it never
observed is worse than one that stays silent.

## Ablation

Varies a policy **parameter**, one at a time, and attributes the change to the
subsystem whose code path reads it.

> Ablating over `PolicyConfig.components` would produce a table of identical
> rows and the confident conclusion that the subsystem does not matter: the
> components map is descriptive metadata, and naming a component
> `waypoint_v2` changes nothing because nothing reads it.

The simulator is injected as a callable, so the same experiment runs on every
backend, and the reference arm is always measured — attribution is a difference,
and a difference against an unmeasured baseline is a guess.

### Scored against injected ground truth

`policy_v1` and `policy_v3` differ in exactly one subsystem, so a method that
names the controller has identified the wrong thing:

| method | implicated | episodes | correct? |
|---|---|---|---|
| correlation only | **planner** | 12 | **no** |
| OFAT | **perception** | 216 | **yes** |
| interaction-aware | **perception** | 216 | **yes** |

The correlation baseline is wrong *and confident*: it ranks
`trajectory_deviation_m` first at r = −0.993, a downstream consequence rather
than a cause, while the perception metric that encodes the cause ranks third.
Correlation finds what co-varies with failure; ablation finds what causes it.

The pairwise design returned the same answer as OFAT and changed nothing,
because this regression has no interaction. Its 21 pairwise arms doubled the
cost for no gain. Reported rather than presented as a success.

### The destructive-knob trap

Found by running `robot-ci diagnose` on the simulator:

| arm | success | attributed to |
|---|---|---|
| `settle_steps=6` | 0.60 (+0.07) | **perception** ← the known cause |
| `clearance_bias_m=0.05` | **0.00** (−0.53) | planner |
| `ik_damping=0.2` | **0.00** (−0.53) | controller |

It named the planner and the controller. Neither has anything to do with it:
raising the clearance bias to 50 mm drops the object, and raising the IK damping
to 0.2 stops the solver converging. Both take every episode to failure, and
**the most destructive knob is always available, so ranking by effect size
systematically selects for it.**

Arms at the success floor are now excluded from the ranking and listed in
`collapsed_arms`, so the exclusion is visible rather than silent. The criterion
is a floor on the arm's own rate, not a drop from the reference: a first attempt
used a 0.30 drop, which also swept in a 0.80 → 0.30 arm — a strong gradient, not a
broken task.

## Writing the conclusion

`diagnose()` produces a block whose wording is hedged in proportion to its
evidence, from a fixed vocabulary so "strong" means the same thing every time.

Two guards, both added because running the command produced a wrong answer:

* **Cost metrics cannot implicate a subsystem.** A diagnosis of two arms with
  identical success rates to sixteen digits attributed a regression to the
  planner on a 64 ms difference in `planning_time_ms` — wall-clock, which moves
  with machine load between runs. Episode length and completion time are *not*
  excluded, because spending extra control ticks to reach the same place is a
  real behavioural cost.
* **No success change means no regression.** The null is the finding. An
  attribution offered anyway answers a question nobody asked, and the metrics
  that will be found to support it are the ones that move between identical
  runs.

The result always carries its limitation: substitution measures *association*,
not causation, because varying a parameter also varies the trajectories it
produces.

## Using it

```bash
robot-ci diagnose --scenario <yaml> --backend mujoco \
  --baseline policy_v3 --candidate policy_v1 \
  --axis depth_noise_mm:0:2:mm --seeds 3
```

Conditions are labelled with the value the metrics came from. An earlier version
labelled the worst cell with the *first* swept point's condition, so a regression
measured at 0.5 mm of depth noise was reported as `depth_noise_mm=0`.
