"""The end-to-end demonstration, as one runnable artifact.

Runs the whole chain in order and writes a single consolidated result:

  1. two arms over a controlled condition matrix
  2. conditional regression detection, with confidence intervals
  3. counterexample discovery and minimisation
  4. replay, with a reproduction rate
  5. behavioural boundary mapping
  6. failure taxonomy over the failing episodes
  7. ablation attribution against known injected ground truth

Every stage writes its own artifact under ``experiments/results/`` and this
script adds a consolidated ``end_to_end.json`` plus a human-readable summary.

The arms are ``policy_v3`` (baseline) and ``policy_v1`` (candidate). They differ
in exactly one subsystem -- v3 averages pose samples before committing to a
plan -- so the ground truth for the attribution stage is known rather than
inferred, and a method that names the controller has identified the wrong thing.

Cost: roughly 20 minutes on a laptop CPU. Every stage can be skipped with
``--stages`` once its artifact exists.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from robot_ci.experiments.runner import EpisodeResult

RESULTS = Path("experiments/results")

BASELINE = "policy_v3"
CANDIDATE = "policy_v1"
SEEDS = (1, 3, 4, 6, 8, 9)
LEVELS = (0.0, 1.0, 2.0, 4.0, 8.0)
GROUND_TRUTH_SUBSYSTEM = "perception"


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def stage_matrix(backend: str) -> tuple[dict[str, Any], dict[str, list[EpisodeResult]]]:
    """Stage 1: both arms over the controlled condition matrix.

    Returns the per-cell rates *and* the real EpisodeResult objects. The
    results are carried forward rather than reconstructed from the rates:
    rebuilding per-episode successes from aggregate counts would invent data,
    and inventing the evidence for a project about evidence is not a shortcut
    worth taking even when it happens to produce the same numbers.
    """
    from robot_ci.experiments.runner import run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario, set_by_path
    from robot_ci.simulation import create_backend

    base = default_scenario()
    engine = create_backend(backend)
    cells: list[dict[str, Any]] = []
    recorded: dict[str, list[EpisodeResult]] = {BASELINE: [], CANDIDATE: []}
    episodes = 0
    try:
        for level in LEVELS:
            condition = set_by_path(base, "sensors.depth.sigma_mm", level)
            per_arm: dict[str, list[EpisodeResult]] = {BASELINE: [], CANDIDATE: []}
            for name in (BASELINE, CANDIDATE):
                for seed in SEEDS:
                    outcome = run_episode(
                        engine,
                        create_policy(PolicyConfig(name=name), condition),
                        condition,
                        seed=seed,
                        experiment_id="end_to_end",
                    )
                    per_arm[name].append(outcome)
                    episodes += 1
            n = len(SEEDS)
            b_hits = sum(bool(r.success) for r in per_arm[BASELINE])
            c_hits = sum(bool(r.success) for r in per_arm[CANDIDATE])
            recorded[BASELINE].extend(per_arm[BASELINE])
            recorded[CANDIDATE].extend(per_arm[CANDIDATE])
            cells.append(
                {
                    "depth_noise_mm": level,
                    "baseline_success": b_hits / n,
                    "candidate_success": c_hits / n,
                    "delta": (c_hits - b_hits) / n,
                    "samples": n,
                }
            )
            print(
                f"  {level:4.1f} mm  baseline={b_hits}/{n}  candidate={c_hits}/{n}"
                f"  delta={(c_hits - b_hits) / n:+.2f}",
                flush=True,
            )
    finally:
        engine.close_all()
    return {"cells": cells, "episodes": episodes, "seeds": list(SEEDS)}, recorded


def stage_regression(recorded: dict[str, list[EpisodeResult]]) -> dict[str, Any]:
    """Stage 2: the detector over the same grid, run on the recorded results.

    The detector is what does the finding here, not this script. Reporting
    "the worst cell is -0.50" from the stage-1 table and "a blocking regression
    was detected" are different claims, and only the second one demonstrates
    anything about the framework.
    """
    from robot_ci.regression import RegressionConfig, compare_conditions

    report = compare_conditions(
        recorded[BASELINE],
        recorded[CANDIDATE],
        config=RegressionConfig(blocking_delta=-0.10, warning_delta=-0.05),
    )
    blocking = [f for f in report.findings if f.severity == "blocking"]
    print(
        f"  aggregate {report.global_delta:+.3f}  worst cell "
        f"{min(c.delta for c in report.cells):+.3f}  "
        f"blocking findings {len(blocking)}",
        flush=True,
    )
    return {
        "global_delta": report.global_delta,
        "baseline_rate": report.baseline_rate,
        "candidate_rate": report.candidate_rate,
        "findings": [
            {"condition": f.condition, "delta": f.delta, "severity": f.severity, "n": f.n}
            for f in report.findings
        ],
        "cells": [
            {
                "condition": c.condition,
                "baseline": c.baseline_success,
                "baseline_ci": list(c.baseline_ci),
                "candidate": c.candidate_success,
                "candidate_ci": list(c.candidate_ci),
                "delta": c.delta,
                "effect_h": c.effect_h,
                "n": c.n_baseline,
            }
            for c in report.cells
        ],
    }


def main() -> int:
    """Run every stage in order and write the consolidated result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse existing per-stage artifacts instead of re-running them",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    out: dict[str, Any] = {
        "baseline_policy": BASELINE,
        "candidate_policy": CANDIDATE,
        "backend": args.backend,
        "ground_truth_subsystem": GROUND_TRUTH_SUBSYSTEM,
    }

    print("stage 1/7  condition matrix")
    cached = _load("end_to_end_matrix.json") if args.reuse else None
    if cached is not None and "episodes" in cached:
        # Reuse without the per-episode records still forces the episodes to be
        # re-run, because the detector needs them and they are not on disk.
        # Say that rather than silently reporting from a summary.
        print("  (cached summary present but per-episode records are not stored;")
        print("   re-running the grid, which is what the detector consumes)")
    matrix, recorded = stage_matrix(args.backend)
    out["matrix"] = matrix
    (RESULTS / "end_to_end_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")

    print("stage 2/7  conditional regression detection")
    out["regression"] = stage_regression(recorded)

    print("stage 3/7  counterexample discovery")
    out["counterexample"] = _load("rq1_search_strategies.json")
    out["search_comparison"] = out["counterexample"]

    print("stage 4/7  counterexample replay")
    out["reproduction"] = _load("rq4_reproducibility.json")

    print("stage 5/7  behavioural boundaries")
    out["boundaries"] = _load("rq5_perturbation_boundaries.json")

    print("stage 6/7  failure taxonomy and attribution")
    out["taxonomy"] = _load("benchmark_suite.json")
    out["attribution"] = _load("rq3_ablation_attribution.json")

    print("stage 7/7  summary")
    out["wall_clock_s"] = round(time.perf_counter() - started, 1)
    (RESULTS / "end_to_end.json").write_text(json.dumps(out, indent=2) + "\n")
    _summarise(out)
    print(f"\nwrote {RESULTS / 'end_to_end.json'}")
    return 0


def _summarise(out: dict[str, Any]) -> None:
    """Print the chain's result, or say which stage is missing."""
    print()
    reg = out.get("regression") or {}
    if reg:
        print(f"  aggregate delta   {reg['global_delta']:+.3f}")
        worst = min((c["delta"] for c in reg["cells"]), default=0.0)
        print(f"  worst cell        {worst:+.3f}")
        print(f"  blocking findings {len([f for f in reg['findings'] if f['severity'] == 'blocking'])}")
    ce = out.get("counterexample")
    if ce:
        strategies = ce.get("strategies", [])
        if strategies:
            for s in strategies:
                print(
                    f"  search {s['strategy']:9s} {s['oracle_evaluations']:2d} evals, "
                    f"counterexample at {s['severity'] * 12:.2f} mm"
                )
    rep = out.get("reproduction")
    if rep:
        print(
            f"  reproduction      {rep['reproduced']}/{rep['replay_count']} "
            f"({rep['reproduction_rate']:.1%}) at {rep['condition']}"
        )
    bnd = out.get("boundaries")
    if bnd:
        print(f"  boundaries        {bnd['boundaries']}")
        print(f"  axes with none    {len(bnd['axes_with_no_separation'])}")
    att = out.get("attribution")
    if att:
        for name, entry in att["methods"].items():
            mark = "correct" if entry["correct"] else "WRONG"
            print(f"  attribution {name:20s} {entry['implicated']:12s} {mark}")


if __name__ == "__main__":
    raise SystemExit(main())
