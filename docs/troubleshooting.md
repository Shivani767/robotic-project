# Troubleshooting

**Evidence: Verified locally.** Every failure mode here was actually hit while
building this project, on this machine. They are ordered by how much time they
cost.

## The recurring theme

The most expensive bugs in this project were not crashes. They were **code that
looked correct and silently did less than it appeared to**. Each one below
produced plausible output, and in several cases the output was indistinguishable
from a correct result.

If something looks wrong and the numbers look fine, suspect the numbers.

## Results

### "No regression found" but you expected one

Check the conditions, not the aggregate. On the demo experiment, both arms score
0.100 — because half the cells are unsolvable at nominal and every one is
unsolvable at 1 mm of noise, so both are pinned at the floor.

```bash
robot-ci regression experiments/results/<experiment>   # per-condition table
```

If a conditional cell is at the floor, there is nothing to detect there. A search
that reports a null under those conditions is being honest.

### A counterexample will not reproduce

Expected, and reported rather than hidden. Measured reproduction rates: **4/11
(36.4%)** on an earlier version, **3/11 (27.3%)** after the IK tolerance change
moved the minimal counterexample from 1.0 mm to 2.0 mm.

```bash
robot-ci counterexample reproduce <artifact> --seeds 3,4,6,8,9,12 --strict
```

`--strict` exits non-zero below 1.0. One of the non-reproducing seeds in a real
run was the *inverse* of a counterexample — baseline fails, candidate succeeds —
so the arms did disagree, just in the other direction. The oracle tests
direction.

### A benchmark expectation does not hold

That is the suite working. Three of the author's own expectations were falsified
on the first run; see [`evaluation.md`](evaluation.md) for what was wrong and what
the measurements said.

### A curve looks non-monotonic

Check whether the axis is near its threshold before reading anything into it. At
n=6, a near-threshold condition gives coin-flip cells: the depth-bias axis came
out 0.00 at 2 mm, 1.00 at 5 mm, 0.00 at 10 mm. That is sampling, not physics.

## Performance and control

### The arm does not move, or moves to the wrong place

Almost always the IK tolerance. `converged` means "the solver stopped
improving", which for an unreachable pose means it returned its best attempt and
called that success. It accepted 4.97 mm and 0.85 rad once, and the arm then
spent the episode rotating 43° toward an orientation it was blocked from
reaching.

The tolerance is now 0.5 mm, derived from the grasp band. Measured:

| `ik_pos_tol` | solvability (8 seeds) |
|---|---|
| 2e-3 | 0.500 |
| 1e-3 | 0.625 |
| 5e-4 | 0.750 |

### An episode takes 300 steps

Before stall detection, every failure ran to the step cap, so a run's wall-clock
was set by its failures and every failure was labelled TIMEOUT. Stall detection
now ends them at ~128 ticks. If a legitimate episode is being cut short, the
approach is genuinely longer than 60 ticks without progress.

### Changing a default invalidated my results

This happened. `dc61715` changed `ik_pos_tol` and every recorded result had to
be re-run; two RQ5 findings turned out to be artefacts of it. **After changing
anything that affects task behaviour, re-run the experiments and update the
staleness notices in `experiments/results/README.md`.**

## Environment

### A backend is constructible but will not run

`robot-ci backend list` distinguishes these. The Isaac backend constructs on any
machine so its capabilities can be read, and reports **"constructible, unusable
here"** on a Mac. `available` means it could be constructed; `usable` means it can
run an episode.

### `robot-ci test` passes suspiciously

Check the coverage lines. Every gate prints how many episodes it ran. A gate
with no episode count has measured nothing, whatever its verdict.

## Suspecting the framework itself

```bash
.venv/bin/python -m pytest tests/unit tests/integration tests/e2e -q
.venv/bin/python -m ruff check src/robot_ci tests experiments
.venv/bin/python -m mypy src/robot_ci
.venv/bin/python -m robot_ci.cli.main validate
```

Then, in order:

1. `experiments/results/*.json` — the run that produced the number
2. `experiments/results/README.md` — the curated record, with staleness notices
3. `research_report.md`
4. `docs/` — and the first thing to suspect when one of the above disagrees
