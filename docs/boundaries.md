# Behavioural boundaries

**Evidence: Verified locally.** RQ5 ran on this machine; one axis per condition,
6 seeds, arms `policy_v3` (baseline) and `policy_v1` (candidate).

## The method

Sweep one perturbation at a time with the others nominal, and report where the
baseline still succeeds and the candidate no longer does. One axis at a time is
deliberate: a multi-axis sweep is needed to see interactions but cannot attribute
a boundary to a factor, and attribution is the question here.

## What was found

**One axis has a boundary.**

| depth noise | baseline | candidate | delta |
|---|---|---|---|
| 0 mm | 1.00 | 1.00 | +0.00 |
| 1 mm | 0.83 | 0.33 | **-0.50** |
| 2 mm | 0.83 | 0.50 | -0.33 |
| 4 mm | 0.50 | 0.33 | -0.17 |
| 8 mm | 0.17 | 0.17 | +0.00 |

Identical at nominal, a 0.50 gap at the first swept level, both arms degrading
monotonically afterwards. This is the shape the framework exists to find.

**Six axes have no separation, and two of those are findings rather than
disappointments.**

| axis | swept | outcome |
|---|---|---|
| camera latency | 0–200 ms | 1.00 throughout |
| RGB occlusion | 0–60 % | 1.00 throughout |
| clutter | 0–9 | 1.00 throughout |
| object friction | 1.0–0.1 | 1.00 throughout |
| joint noise | 0–0.05 rad | 1.00 until 0.05, where both fall to 0.67 |
| depth bias | 0–20 mm | bimodal, see below |

**Latency and occlusion are structurally inert.** The policy latches its plan on
the first control tick and never consults RGB, so a delayed or occluded frame is
never acted on. A benchmark that sweeps them spends episodes to measure nothing.
That is a fact about the *task and policy pair*, not about the axes, and it would
transfer to any plan-once policy on a static object.

## Two findings that were withdrawn

An earlier run reported that joint-encoder noise **changed the sign** of the
effect along its axis, and that object friction was **non-monotonic** (0.50 at
friction 1.0 against 1.00 below). Both were artefacts of the loose 2 mm IK
tolerance. After `dc61715` tightened it to 0.5 mm, joint noise is flat until
0.05 rad where *both* arms fall together, and friction is 1.00 at every level.

> A measurement that changes when an unrelated tuning constant changes was never
> a measurement of the policy. Both are recorded above rather than deleted,
> because they are the clearest evidence in the project that re-running after a
> change is not optional.

## A methodological caution

The depth-bias axis came out **bimodal**: 0.00 at 2 mm, 1.00 at 5 mm, 0.00 at
10 mm. That is not physical. At n=6 a near-threshold condition produces coin-flip
cells, so individual cells are 0.00 or 1.00 with nothing between and the curve
through them reflects which seeds were drawn.

**Any axis near its threshold needs more repetitions before its shape means
anything.** The other axes here clear their thresholds by a wide margin and are
safe at n=6; the rule is not "n=6 is fine" but "check whether the axis is near
its threshold before reading a curve".

## Reproducing

```bash
.venv/bin/python experiments/rq5_perturbation_boundaries.py
```

Writes `experiments/results/rq5_perturbation_boundaries.json` with per-axis
curves, the located boundaries, and the list of axes with no separation.
