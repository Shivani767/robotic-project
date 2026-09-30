# Regression detection

**Evidence: Verified locally**, with one bug that this subsystem had, documented
below because it is the most important thing in this page.

## The idea

An aggregate success rate is the wrong instrument. Two policies can trade a
large improvement in one condition for an equal regression in another and report
a delta of exactly zero. Detection therefore happens **per condition**, and the
aggregate is reported only so the reader can see what it hides.

Measured, from the re-run RQ2 grid (8 seeds, 5 noise levels, 40 runs per arm):

| depth noise | baseline | candidate | delta |
|---|---|---|---|
| 0 mm | 8/8 | 8/8 | +0.00 |
| 1 mm | 7/8 | 4/8 | **-0.38** |
| 2 mm | 7/8 | 4/8 | **-0.38** |
| 4 mm | 5/8 | 2/8 | **-0.38** |
| 8 mm | 2/8 | 1/8 | -0.12 |

| scoring | result |
|---|---|
| aggregate over 40 runs | **-0.250** |
| worst conditional cell | **-0.375** |

**The aggregate reports two thirds the severity of the worst condition it
averages over.** Three cells sit at exactly -0.38, which reads as a gradient
rather than three coincidences.

## Grouping

Cells are grouped on the episode's **condition values**, with `seed` excluded —
it varies within every cell, so including it would make each episode a cell of
size one. Labels are disambiguated when several cells share a scenario id.

> This is the bug. The detector originally grouped on `scenario_id` alone, which
> is correct only for a grid expanded into separate scenario files, where each
> cell gets its own id. Every *in-place* sweep — the counterexample search,
> `robot-ci diagnose`, the RQ2 harness — perturbs a scenario without renaming
> it, so all cells collapsed into one. On the five-cell grid above it returned a
> single cell of 40 episodes, reported one "regression" of -0.125 at *nominal*,
> and never mentioned the -0.25 cell. **A conditional analysis that conditions
> on nothing returns an aggregate result under a name that promises otherwise,
> and the output looks exactly like a correct one.** Fixed in `5f40a24`; two
> pre-existing tests had been passing *because* of it.

## Statistics

* **Rates**: Wilson score interval, which stays valid at 0 and 1 — the case a
  regression report hits most often.
* **Effect size**: Cohen's *h*, for the same reason.
* **Means**: Student-t or bootstrap, chosen by the interval call.
* **Sample counts**: reported on every cell and every finding. A cell of 0.50
  from 6 samples is a different claim from 0.50 from 600.

## Thresholds

Configurable, and echoed in the output so a verdict can be checked against what
it was measured against:

| Threshold | Default | Meaning |
|---|---|---|
| `blocking_delta` | -0.10 | fails the run |
| `warning_delta` | -0.05 | reported, does not fail |
| `min_samples` | 3 | below this, no finding is emitted |

`min_samples` matters more than it looks. A cell with 2 seeds can produce a
delta of -1.0 that is indistinguishable from noise, and a gate that fires on it
will block a release for a coin flip.

## Using it

```bash
robot-ci regression experiments/results/<experiment>
```

Writes `regression.json` and exits non-zero when a blocking finding exists.

The **strong** form of the claim — that conditional evaluation finds regressions
that aggregate metrics *hide entirely* — is **not demonstrated**. It needs a grid
where favourable and adverse cells cancel. On the grid above the candidate is
adverse in every adverse cell, so nothing cancels and the aggregate is also
adverse. Demonstrating it needs a policy pair whose trade-off runs in opposite
directions on two axes, which has not been built.
