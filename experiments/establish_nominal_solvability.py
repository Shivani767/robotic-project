"""Establish nominal solvability, and replace the stale number with a measured one.

Why this script exists
----------------------
The research report carried the claim *"nominal solvability is 66.7%"* and drew a
conclusion from it: that the framework's sensitivity was bounded by its
benchmark, because a real 10% regression could hide inside a 33% failure floor.

That number came from an 8/12 measurement taken *before* the IK-tolerance and
object-spawn fixes, and it was written in the present tense as a standing
property of the system. Re-measuring falsifies it, which is the whole point of
having a script rather than a sentence.

What it does
------------
Runs the nominal condition over many seeds for each policy and reports the
solvability with a Wilson interval, so the claim becomes "at least X" with a
stated confidence rather than a point estimate from a small sample. A point
estimate from 12 seeds moves by 8 percentage points per seed; the interval is
the honest way to state a rate.

It also records *which* metric fails, because "66.7% of episodes fail" and "a
third of episodes fail with the grasp not holding" are different claims and
only the second one tells you where to look.

Usage:
    python experiments/establish_nominal_solvability.py [--seeds N] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario
from robot_ci.simulation import create_backend
from robot_ci.statistics.intervals import proportion_ci

RESULTS = Path("experiments/results")
POLICIES = ("policy_v1", "policy_v2", "policy_v3")

#: Metrics inspected on a failing episode, to name the failure rather than
#: just counting it. A bare rate is not actionable; "the grasp did not hold"
#: is.
FAILURE_METRICS = (
    "grasp_success",
    "object_lift_m",
    "placement_success",
    "ik_converged",
    "final_distance_m",
)


def measure(backend_name: str, policy: str, seeds: int) -> dict[str, Any]:
    """Run ``seeds`` nominal episodes for one policy and summarise them."""
    scenario = default_scenario()
    engine = create_backend(backend_name)
    successes = 0
    steps: list[int] = []
    failures: list[dict[str, Any]] = []
    try:
        for seed in range(1, seeds + 1):
            result = run_episode(
                engine,
                create_policy(PolicyConfig(name=policy), scenario),
                scenario,
                seed=seed,
                experiment_id="nominal_solvability",
            )
            steps.append(result.steps)
            if result.success:
                successes += 1
                continue
            metrics = {m.name: float(m.value) for m in result.metrics}
            failures.append(
                {
                    "seed": seed,
                    "steps": result.steps,
                    **{k: round(metrics.get(k, float("nan")), 6) for k in FAILURE_METRICS},
                }
            )
    finally:
        engine.close_all()

    interval = proportion_ci(successes, seeds)
    return {
        "policy": policy,
        "backend": backend_name,
        "seeds": seeds,
        "successes": successes,
        "rate": successes / seeds,
        "ci_low": interval.lower,
        "ci_high": interval.upper,
        "steps_mean": statistics.fmean(steps),
        "steps_min": min(steps),
        "steps_max": max(steps),
        "failures": failures,
    }


def main() -> int:
    """Measure every policy at nominal and write the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=120)
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--json", type=Path, default=RESULTS / "nominal_solvability.json")
    args = parser.parse_args()

    out: dict[str, Any] = {
        "condition": "nominal (no injected corruption)",
        "backend": args.backend,
        "seeds": args.seeds,
        "policies": {},
    }

    for policy in POLICIES:
        record = measure(args.backend, policy, args.seeds)
        out["policies"][policy] = record
        n, k = record["seeds"], record["successes"]
        print(
            f"{policy:10s} {k:3d}/{n} = {record['rate']:.3f}  "
            f"95% CI [{record['ci_low']:.3f}, {record['ci_high']:.3f}]  "
            f"steps {record['steps_mean']:.1f} ({record['steps_min']}-{record['steps_max']})",
            flush=True,
        )

    primary = out["policies"]["policy_v1"]
    print(
        f"\nNominal solvability for the baseline is {primary['rate']:.1%} "
        f"with a 95% lower bound of {primary['ci_low']:.1%} over {primary['seeds']} seeds."
    )
    claimed = 0.667
    if primary["ci_low"] > claimed:
        print(
            f"The previously reported {claimed:.1%} lies below the confidence "
            "interval, so it is falsified rather than merely imprecise."
        )
    else:
        print(f"The previously reported {claimed:.1%} is consistent with the interval.")

    failing = [p for p, r in out["policies"].items() if r["failures"]]
    if not failing:
        print("No episode failed, so there is no failure mode to attribute.")
    else:
        for name in failing:
            print(f"\n{name} failures ({len(out['policies'][name]['failures'])}):")
            for f in out["policies"][name]["failures"][:10]:
                print(f"  seed {f['seed']:3d}  steps={f['steps']:3d}  "
                      f"grasp={f['grasp_success']:.0f}  "
                      f"lift={f['object_lift_m'] * 1000:.1f}mm  "
                      f"ik={f['ik_converged']:.0f}  place={f['placement_success']:.0f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
