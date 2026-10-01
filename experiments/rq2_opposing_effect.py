"""RQ2, the opposing-effect experiment: does the candidate's trade-off invert?

The question
------------
RQ2 asks whether a candidate change can be *favourable in some conditions and
adverse in others*, such that the aggregate delta is small or zero while a
conditional regression is large. The weak form of that is easy: any condition
where the effect differs from zero. The strong form -- which no run in this
project had demonstrated -- needs the effect to **change sign**, so that the
aggregate is a near-tie between cells that disagree.

Why the averaging candidate is the right one
--------------------------------------------
``policy_v3`` is the project's positive control, and its own docstring states a
falsifiable two-sided prediction:

    "it should help where depth noise dominates and hurt where latency does,
     and RQ2 tests exactly that. If it does not, the hypothesis is wrong and is
     reported as wrong."

It gains by averaging ``settle_steps`` pose samples, which suppresses the
per-tick jitter of a noisy estimate; the cost is that it spends ticks settling
and builds its plan from a slightly older estimate.

What the screening found, and why the design changed
----------------------------------------------------
Six one-dimensional sweeps were run first. Only one had any behavioural
signature at all:

    depth_noise_mm   0.00, +0.33, +0.42, +0.17, +0.08
    latency_ms       0.00,  0.00,  0.00,  0.00
    max_steps        0.00,  0.00,  0.00,  0.00
    clutter_count    0.00,  0.00,  0.00,  0.00
    depth_bias_mm    0.00,  0.00,  0.00,  0.00

The latency prediction is **falsified**, and not by noise in the measurement:
its mechanism is unreachable here. Averaging suppresses *variance*, so it can
only cost you when the quantity being averaged is *changing*. Camera latency in
a static scene delays every sample by the same amount, and the mean of a set of
equally-delayed samples of a static object carries no extra error. The
framework has no axis that moves the object -- ``ObjectSpec`` has a position and
an orientation and no motion field at all -- so the adverse channel the
prediction names cannot be exercised.

A measured step budget, not a contrivance
-----------------------------------------
The one genuine cost of settling is control ticks, and it is small and exactly
measurable:

    nominal, 0 mm noise     policy_v1  76.3 steps (76-78)   16/16
                            policy_v3  81.3 steps (81-83)   16/16

Five extra ticks, essentially constant. Against the default 300-step budget that
is invisible, which is why the 1-D budget sweep found nothing. It is not
invisible against a *deadline*: a budget that admits the baseline's 76-78 steps
but not the candidate's 81-83 refuses the candidate and accepts the baseline at
nominal, while under noise the baseline needs up to 130 steps and is refused far
more often than the candidate.

That is a real deployment question -- does the extra work fit the deadline --
and it is the second axis here. A tighter deadline is not an artificial
handicap: it is the condition under which the candidate's mechanism is supposed
to be a cost rather than a benefit.

The design
----------
A two-dimensional grid, because a one-dimensional sweep is exactly what would
miss this: the candidate's cost and benefit live on different axes, and only
their intersection decides the sign. Stated before running, the prediction is:

  * high noise + generous budget  -> candidate wins (variance reduction pays)
  * low noise  + tight deadline   -> candidate loses (settle cost, no noise to fix)
  * some cells of each sign       -> the aggregate is a near-tie

The prediction is recorded here before the run so that it cannot be adjusted to
fit the result. If the grid shows an aggregate that is *not* near a tie, that is
the finding and it is reported as such.

Usage:
    python experiments/rq2_opposing_effect.py [--seeds N]
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from robot_ci.experiments.runner import EpisodeResult, run_episode
from robot_ci.models.scenario import PolicyConfig, variable_path
from robot_ci.policies import create_policy
from robot_ci.regression import RegressionConfig, compare_conditions
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend

RESULTS = Path("experiments/results")

#: Baseline trusts one fused estimate; candidate averages several before
#: committing. See ``_DEFAULT_V3`` for the documented trade-off.
BASELINE = "policy_v1"
CANDIDATE = "policy_v3"

#: Axis 1, the favourable channel: the candidate's variance reduction.
DEPTH_NOISE_MM = (0.0, 1.0, 2.0, 4.0)

#: Axis 2, the adverse channel: a deadline that can refuse the settle cost.
#: 76-78 covers the baseline's nominal execution and 81-83 the candidate's, so
#: 78 and 80 bracket the split; 120 and 300 are the unconstrained controls.
STEP_BUDGETS = (78, 80, 82, 84, 120, 300)

SEEDS = tuple(range(1, 17))


def build(depth_noise_mm: float, max_steps: int):
    """One cell of the grid: a scenario with the two axes pinned."""
    scenario = set_by_path(
        default_scenario(), variable_path("depth_noise_mm"), depth_noise_mm
    )
    return scenario.model_copy(
        update={"task": scenario.task.model_copy(update={"max_steps": max_steps})}
    )


def run_cell(engine: Any, depth_noise_mm: float, max_steps: int, seeds) -> tuple[list, list]:
    """Run both arms over every seed in one cell, paired by seed."""
    baseline_results: list[EpisodeResult] = []
    candidate_results: list[EpisodeResult] = []
    for arm, sink in ((BASELINE, baseline_results), (CANDIDATE, candidate_results)):
        for seed in seeds:
            scenario = build(depth_noise_mm, max_steps)
            sink.append(
                run_episode(
                    engine,
                    create_policy(PolicyConfig(name=arm), scenario),
                    scenario,
                    seed=seed,
                    experiment_id="rq2_opposing",
                )
            )
    return baseline_results, candidate_results


def main() -> int:
    """Run the grid, then hand the paired results to the existing detector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument(
        "--json", type=Path, default=RESULTS / "rq2_opposing_effect.json"
    )
    args = parser.parse_args()

    seeds = tuple(range(1, args.seeds + 1))
    engine = create_backend(args.backend)
    grid: dict[str, tuple[list, list]] = {}
    try:
        for depth, budget in product(DEPTH_NOISE_MM, STEP_BUDGETS):
            base, cand = run_cell(engine, depth, budget, seeds)
            grid[f"depth_noise_mm={depth:g},max_steps={budget}"] = (base, cand)
            b = sum(bool(r.success) for r in base)
            c = sum(bool(r.success) for r in cand)
            n = len(seeds)
            print(
                f"  noise {depth:>3.1f} mm  budget {budget:>3d}  "
                f"base {b:>2}/{n}  cand {c:>2}/{n}  delta {(c - b) / n:+.2f}",
                flush=True,
            )
    finally:
        engine.close_all()

    # Every arm is compared inside a cell, so the detector sees a paired grid
    # keyed by (condition, seed) rather than a flat pool. This is the existing
    # detector; nothing about it is modified for this experiment.
    pooled_base: list[EpisodeResult] = []
    pooled_cand: list[EpisodeResult] = []
    per_cell: list[dict[str, Any]] = []
    for condition, (base, cand) in grid.items():
        pooled_base.extend(base)
        pooled_cand.extend(cand)
        b = sum(bool(r.success) for r in base)
        c = sum(bool(r.success) for r in cand)
        n = len(base)
        per_cell.append(
            {
                "condition": condition,
                "n": n,
                "baseline": b / n,
                "candidate": c / n,
                "delta": (c - b) / n,
            }
        )

    report = compare_conditions(
        pooled_base,
        pooled_cand,
        config=RegressionConfig(blocking_delta=-0.10, warning_delta=-0.05),
    )
    favourable = [c for c in per_cell if c["delta"] > 0.05]
    adverse = [c for c in per_cell if c["delta"] < -0.05]
    tied = [c for c in per_cell if abs(c["delta"]) <= 0.05]

    print(f"\naggregate delta            {report.global_delta:+.3f}")
    print(f"favourable cells (> +0.05) {len(favourable)}")
    print(f"adverse cells    (< -0.05) {len(adverse)}")
    print(f"tied cells                 {len(tied)}")
    if favourable:
        print(f"  largest favourable       {max(c['delta'] for c in favourable):+.2f}")
    if adverse:
        print(f"  largest adverse          {min(c['delta'] for c in adverse):+.2f}")

    out = {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "backend": args.backend,
        "seeds": list(seeds),
        "axes": {
            "depth_noise_mm": list(DEPTH_NOISE_MM),
            "max_steps": list(STEP_BUDGETS),
        },
        "prediction": (
            "high noise + generous budget -> candidate wins; low noise + tight "
            "deadline -> candidate loses; aggregate near a tie"
        ),
        "cells": per_cell,
        "aggregate_delta": report.global_delta,
        "aggregate_baseline_rate": report.baseline_rate,
        "aggregate_candidate_rate": report.candidate_rate,
        "detector_findings": [
            {"condition": f.condition, "delta": f.delta, "severity": f.severity, "n": f.n}
            for f in report.findings
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

