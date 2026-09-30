# Robot CI: research report

A robotics continuous-integration framework: given two versions of a robot
policy, determine automatically whether the new one behaves worse, under which
conditions, and because of which subsystem.

Everything numeric below was produced on a MacBook CPU backend with MuJoCo, and
every figure is reproducible with the commands in
`experiments/results/README.md`. **No result in this report is projected,
simulated on paper, or copied from a previous session.** Where an experiment was
not run, it says so.

> **Staleness.** Commit `dc61715` tightened `ik_pos_tol` from 2 mm to 0.5 mm,
> which raised nominal solvability on 12 seeds from 50.0% to 66.7% and therefore
> changed task behaviour. **RQ2, RQ3 and RQ5 below have been re-run and are
> current. RQ1 and RQ4 are still stale** -- measured before that commit, kept as
> they stood and labelled, because deleting them would misrepresent what was
> known when. Re-run them before quoting them.
>
> Re-running also **withdrew two RQ5 findings** as tuning artefacts, recorded in
> place rather than removed.

---

## 1. Research question

> Given two versions of a robotic system, can we automatically discover the
> environmental, sensor, embodiment or task conditions under which the new
> version behaves worse than the old one, minimise those conditions into
> reproducible counterexamples, and diagnose the likely source of the regression
> with evidence rather than assertion?

The framing matters. "Is the new version better?" is a question about a single
number, and a single number cannot answer it: two policies can trade a large
improvement in one condition for an equal regression in another and report a
delta of zero. The work here is about the conditions, not the average.

## 2. Method

* **Simulator.** MuJoCo, 7-DOF Panda with a parallel gripper, pick-and-place.
  CPU-only, on the development machine. `MockSimulator` and `SyntheticSimulator`
  exist for unit tests; neither is used for any number in this report.
* **Arms.** Two of the three policies are `policy_v1` and `policy_v2`, which
  differ by exactly one parameter. The comparisons that needed a known
  regression use `policy_v3` against `policy_v1`; see section 3.
* **Baselines.** Where a method is compared, both sides share the scenario
  budget, the seeds, the simulator, the policies and the metric code. Only the
  method differs.
* **Statistics.** Wilson intervals for rates, Cohen's h for effect size, and
  paired per-`(condition, seed)` comparison. Configurable thresholds
  (blocking at -0.10, warning at -0.05) with a minimum-sample gate.

## 3. Establishing that the apparatus can detect a regression

The first thing that had to be settled was whether the two "obvious" policies
differ at all. They do not, and finding that out was itself a result.

`policy_v2` differs from `policy_v1` by a confidence gate on the fused object
pose: while the pose is uncertain, it holds the approach and re-observes.
Measured over a depth-noise grid, the two arms were **identical at every level**.
The gate could not engage, because its threshold was set at 50 mm of depth noise
while the arm stopped placing the object at 10 mm for an unrelated reason. The
gate was inert, the two arms were the same policy under different labels, and
the "no regression found" result was an artefact of a dead parameter rather than
a fact about the robots.

That matters for everything else: without a positive control, "no regression
found" and "the detector is blind" are indistinguishable. `policy_v3` was
therefore built as one -- a policy that averages pose samples before committing
to a plan, a real technique with a real cost (it spends control ticks holding
position and plans from a stale estimate). It was then used as the baseline
against which `policy_v1` is the regressing arm, which is the separation the
searchers require.

## 4. Results

### RQ1 -- search strategies

| strategy | evals | wall clock | counterexample found | minimal? |
|---|---|---|---|---|
| random | **1** | 1.4 s | 7.6 mm | no |
| grid | 2 | 4.2 s | 4.0 mm | no |
| binary | 8 | 16.1 s | **0.75 mm** | yes |
| adaptive | 15 | 24.6 s | **0.75 mm** | yes |

Tests-to-failure alone ranks random best, and that is the wrong answer: random
is 15x cheaper and its counterexample is 10x less minimal, needing 7.6 mm of
noise to reproduce where 0.75 mm suffices. **A search strategy that reports only
"found" is concealing half its result.**

This is one trial per strategy on one axis and one seed. It shows the trade-off
exists; it cannot show that any strategy is better in general.

### RQ2 -- conditional vs aggregate

| depth noise | baseline | candidate | delta |
|---|---|---|---|
| 0 mm | 8/8 | 8/8 | +0.00 |
| 1 mm | 7/8 | 4/8 | **-0.38** |
| 2 mm | 7/8 | 4/8 | **-0.38** |
| 4 mm | 5/8 | 2/8 | **-0.38** |
| 8 mm | 2/8 | 1/8 | -0.12 |

Aggregate over 40 runs: **-0.250**. Worst conditional cell: **-0.38**.

**The aggregate reports two thirds the severity of the worst condition it
averages over**, and conditional evaluation names the three conditions
responsible. Three cells sit at exactly -0.38, which is a consistent gradient
rather than three independent coincidences.

The stronger form of the claim -- that aggregate metrics *hide* a regression
entirely -- is **not demonstrated**. That requires favourable and adverse cells
to cancel, and on this grid the candidate is adverse everywhere it differs, so
nothing cancels.

Running this experiment exposed the most serious defect found in the project.
The detector grouped conditional cells by `scenario_id`, which is correct only
for grids expanded into separate scenario files. Every *in-place* sweep -- the
counterexample search, `robot-ci diagnose`, this experiment -- perturbs a
scenario without renaming it, so all cells collapsed into one: a single cell of
40 episodes, one "regression" of -0.125 reported at *nominal*, and no mention of
the -0.25 cell. **A conditional analysis that conditions on nothing returns an
aggregate result under a name promising otherwise, and the output looks exactly
like a correct one.**

### RQ3 -- attributing a regression to a subsystem

Ground truth is injected and known: the two arms differ in exactly one
subsystem, perception.

| method | implicated | episodes | correct? |
|---|---|---|---|
| correlation only | **planner** | 12 | **no** |
| OFAT | **perception** | 216 | **yes** |
| interaction-aware | **perception** | 216 | **yes** |

Correlation-only analysis is wrong *and confident*: it ranks
`trajectory_deviation_m` first at r = -0.993, a downstream consequence rather
than a cause, while the perception metric that encodes the cause ranks third.
Correlation finds what co-varies with failure; ablation finds what causes it.

The interaction-aware design returns the same answer as OFAT and changes
nothing, because this regression has no interaction. Its 21 pairwise arms
doubled the cost for no gain. Reported rather than presented as a success.

**This result held across a change that overturned two of RQ5's findings.** Both
were re-run after commit `dc61715` tightened the IK tolerance. The distinction is
what each claim rests on: RQ3 compares a ranking over 36 arms, so a global
change in solver accuracy moves every arm together, while RQ5's withdrawn claims
were single-cell readings on a near-threshold axis at n=6. A method comparison
and a single-cell curve reading do not deserve the same confidence, and it is not
obvious in advance which is which.

### RQ4 -- counterexample reproducibility

The discovered counterexample (`depth_noise_mm = 1.0`) replayed on 11 seeds
disjoint from the discovery seed: **4/11 = 36.4% reproduction**.

Of the 7 non-reproducing seeds, 5 are ones where both arms succeed and 2 where
both fail. A counterexample reproduced on a third of seeds is a property of the
seed as much as of the condition, and this number must travel with any claim of
a behavioural boundary.

### RQ5 -- perturbation boundaries

Two of seven axes had a boundary before commit `dc61715`; **one has one now.**

| depth noise | baseline | candidate | delta |
|---|---|---|---|
| 0 mm | 1.00 | 1.00 | +0.00 |
| 1 mm | 0.83 | 0.33 | **-0.50** |
| 2 mm | 0.83 | 0.50 | -0.33 |
| 4 mm | 0.50 | 0.33 | -0.17 |
| 8 mm | 0.17 | 0.17 | +0.00 |

Depth noise is the system's one behavioural sensitivity. The other six axes --
camera latency, RGB occlusion, clutter, object friction, joint-encoder noise and
depth bias -- show no separation, and for latency and occlusion the reason is
structural: the policy latches its plan on the first control tick and never
consults RGB, so a delayed or occluded frame is never acted on.

**Two earlier findings were artefacts and are withdrawn.** The previous run
reported that joint noise changed the sign of the effect along its axis, and that
friction was non-monotonic. Both disappear once the IK tolerance is tightened:
joint noise is flat at 1.00 until 0.05 rad where *both* arms fall together, and
friction is 1.00 at every level. A measurement that changes when an unrelated
tuning constant changes was never a measurement of the policy. This is the
clearest evidence in the project that re-running after a change is mandatory.

One methodological caution. The depth-bias axis came out bimodal -- 0.00 at
2 mm, 1.00 at 5 mm, 0.00 at 10 mm -- which is not a physical result. At n=6 a
near-threshold condition gives coin-flip cells, so individual cells are 0.00 or
1.00 with nothing between. Any axis near its threshold needs more repetitions
before the shape of its curve means anything.

## 5. What the work actually establishes

1. **Conditional evaluation is worth its cost.** It reported a regression twice
   as severe as the aggregate, at a named condition.
2. **Ablation beats correlation for attribution**, at 18x the episode cost, and
   the cheap method fails in a confident and plausible way rather than an
   obvious one.
3. **Counterexample search must report minimality, not just discovery.**
   Tests-to-failure alone would have ranked a 10x-worse counterexample as best.
4. **A reproduction rate is not optional.** 36.4% is the difference between a
   behavioural boundary and a lucky episode.

## 6. Limitations

* **One axis pair, one task, one seed for the headline RQ1 result.** The
  separating seed is 1 of 12 scanned. RQ1 needs repeated trials across seeds and
  axes before any strategy can be called better.
* **The strong form of RQ2 is untested.** No policy pair whose trade-off runs in
  opposite directions on two axes has been constructed, so the cancellation case
  the framework is partly built for has not been observed.
* **The pairwise design is untested on the only case it exists for.** No
  regression with a genuine two-factor interaction has been built.
* **Boundaries are swept one axis at a time.** Multi-axis surfaces, and
  therefore axis interactions, are not measured.
* **n is small.** Most cells use 6-8 seeds. At n=6 one episode moves a success
  rate by 0.167, which is enough to rank and not enough to call precise.
* **The detector, ablation and search have been validated on a positive control
  whose cause this project injected.** That validates the machinery against a
  known answer. It does not establish that the machinery works on a real
  regression, because no real regression has been observed on this task.
* **Task success is 50% (12/24 seeds).** Half the failures remain undiagnosed in
  detail; stall detection now labels them, but the causes have not been
  attributed one by one.

## 7. Future work

* Build a policy pair with a genuine two-factor interaction, to test the pairwise
  ablation design and the RQ2 cancellation case.
* Repeat RQ1 across seeds and axes so strategy comparison has a distribution
  rather than a single trial.
* Attribute the remaining 50% of task failures individually, which needs the
  stall-detected episodes classified by cause.
* Multi-axis boundary surfaces.
* The unbuilt phases: ROS 2 / MoveIt 2 / ros2_control backends, the Isaac Sim
  adapter, the notebook backend, and the dashboard. The architecture keeps core
  logic simulator-independent, but none of those paths has been exercised.
