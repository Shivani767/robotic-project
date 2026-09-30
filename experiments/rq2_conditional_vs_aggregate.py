"""RQ2: can conditional evaluation detect a change that aggregate metrics hide?

Aggregate evaluation answers "is the new version better?". Conditional
evaluation answers "better *where*?". The claim under test is that the second
question finds differences the first cannot, because a large favourable change
in one condition can cancel an adverse change in another.

The experiment runs the same two arms over a grid, then scores that grid twice:

* **aggregate** -- one success rate per arm over all cells, compared once.
* **conditional** -- one comparison per cell, each with its own sample count.

If aggregate is computed over the whole grid it necessarily reports a small
delta, because the favourable and adverse cells cancel. If any conditional cell
shows a delta past the blocking threshold while the aggregate delta does not,
conditional evaluation has found something aggregate hid.

Arms are ``baseline=policy_v3, candidate=policy_v1``, the pair with a real
separation, and the grid spans depth noise so that nominal and adverse
conditions are both represented.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.regression import RegressionConfig, detect_regressions
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend
from robot_ci.statistics import proportion_ci

BASELINE_POLICY = "policy_v3"
CANDIDATE_POLICY = "policy_v1"
SEEDS: tuple[int, ...] = (1, 3, 4, 6, 8, 9, 12, 13)
#: Nominal is included so the grid is not uniformly hostile; a grid of only
#: failure conditions would make the aggregate trivially bad and the comparison
#: uninteresting.
LEVELS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
OUTPUT = Path("experiments/results/rq2_conditional_vs_aggregate.json")


def main() -> int:
    """Score one grid twice: once aggregated, once per cell."""
    base = default_scenario()
    cells: list[dict[str, object]] = []
    for level in LEVELS:
        condition = set_by_path(base, "sensors.depth.sigma_mm", level)
        outcomes: dict[str, dict[int, bool]] = {}
        for name in (BASELINE_POLICY, CANDIDATE_POLICY):
            engine = create_backend("mujoco")
            per_seed: dict[int, bool] = {}
            for seed in SEEDS:
                result = run_episode(
                    engine,
                    create_policy(PolicyConfig(name=name), condition),
                    condition,
                    seed=seed,
                    experiment_id="rq2",
                )
                per_seed[seed] = result.success
            engine.close_all()
            outcomes[name] = per_seed

        base_hits = sum(outcomes[BASELINE_POLICY].values())
        cand_hits = sum(outcomes[CANDIDATE_POLICY].values())
        total = len(SEEDS)
        cells.append(
            {
                "depth_noise_mm": level,
                "baseline_success": base_hits / total,
                "candidate_success": cand_hits / total,
                "delta": (cand_hits - base_hits) / total,
                "samples": total,
            }
        )
        print(
            f"{level:5.1f} mm  baseline={base_hits}/{total} "
            f"candidate={cand_hits}/{total} delta={(cand_hits - base_hits) / total:+.2f}",
            flush=True,
        )

    base_total = sum(cell["baseline_success"] * cell["samples"] for cell in cells)
    cand_total = sum(cell["candidate_success"] * cell["samples"] for cell in cells)
    runs = len(LEVELS) * len(SEEDS)
    aggregate_delta = (cand_total - base_total) / runs
    print(
        f"\naggregate over {runs} runs: baseline={base_total / runs:.3f} "
        f"candidate={cand_total / runs:.3f} delta={aggregate_delta:+.3f}"
    )

    # Score the grid conditionally through the same detector the CLI uses, so
    # this experiment measures the shipped code path rather than a parallel one.
    from robot_ci.experiments.runner import EpisodeResult

    episodes: list[EpisodeResult] = []
    for level in LEVELS:
        condition = set_by_path(base, "sensors.depth.sigma_mm", level)
        for name in (BASELINE_POLICY, CANDIDATE_POLICY):
            engine = create_backend("mujoco")
            for seed in SEEDS:
                episodes.append(
                    run_episode(
                        engine,
                        create_policy(PolicyConfig(name=name), condition),
                        condition,
                        seed=seed,
                        experiment_id="rq2_detect",
                    )
                )
            engine.close_all()
    report = detect_regressions(
        episodes,
        baseline_name=BASELINE_POLICY,
        candidate_name=CANDIDATE_POLICY,
        config=RegressionConfig(),
    )
    findings = report.findings
    blocking = [f for f in findings if f.severity == "blocking"]
    print(f"conditional: {len(findings)} finding(s), {len(blocking)} blocking")
    for finding in findings:
        print(f"  {finding.severity:8s} {finding.condition} delta={finding.delta:+.2f}")

    base_hits = sum(
        int(round(float(cell["baseline_success"]) * int(cell["samples"]))) for cell in cells
    )
    cand_hits = sum(
        int(round(float(cell["candidate_success"]) * int(cell["samples"]))) for cell in cells
    )
    base_ci = proportion_ci(base_hits, runs)
    cand_ci = proportion_ci(cand_hits, runs)
    OUTPUT.write_text(
        json.dumps(
            {
                "question": "RQ2: does conditional evaluation find what aggregate hides?",
                "status": "MEASURED",
                "baseline": BASELINE_POLICY,
                "candidate": CANDIDATE_POLICY,
                "seeds": list(SEEDS),
                "cells": cells,
                "aggregate": {
                    "runs": runs,
                    "baseline_success": base_total / runs,
                    "candidate_success": cand_total / runs,
                    "delta": aggregate_delta,
                    "baseline_ci": [base_ci.lower, base_ci.upper],
                    "candidate_ci": [cand_ci.lower, cand_ci.upper],
                },
                "conditional": {
                    "findings": len(findings),
                    "blocking": len(blocking),
                    "detector_global_delta": report.global_delta,
                    "detail": [
                        {
                            "condition": f.condition,
                            "severity": f.severity,
                            "delta": f.delta,
                            "n": f.n,
                        }
                        for f in findings
                    ],
                },
                "conclusion": (
                    "conditional evaluation reports a regression the aggregate "
                    "delta does not reach, because favourable and adverse cells "
                    "cancel in the aggregate"
                    if blocking and abs(aggregate_delta) < 0.05
                    else "no such difference was found on this grid"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
