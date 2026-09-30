"""Run the failure-injection benchmark suite and check it against its expectations.

Each benchmark declares what it predicts; this script reports whether the
prediction held. That is the difference between a suite and a list of sweeps: a
sweep produces numbers, and numbers are equally consistent with "the fault is
well modelled" and "the fault did nothing".

Expectations marked ``verified=False`` are hypotheses derived from a model of
the fault. They are reported the same way, but the output says which is which,
so a reader is never misled about how much of the suite is grounded.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.evaluation import BENCHMARKS, UNIMPLEMENTED_BENCHMARKS, run_benchmark
from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario
from robot_ci.simulation import create_backend

SEEDS = (1, 3, 4, 6, 8, 9)
POLICY = "policy_v1"
OUTPUT = Path("experiments/results/benchmark_suite.json")


def evaluate(scenario, seeds):
    """Run one benchmark's condition over the seeds and average the metrics."""
    engine = create_backend("mujoco")
    totals: dict[str, list[float]] = {}
    hits = 0
    try:
        for seed in seeds:
            result = run_episode(
                engine,
                create_policy(PolicyConfig(name=POLICY), scenario),
                scenario,
                seed=seed,
                experiment_id="benchmark",
            )
            hits += int(result.success)
            for metric in result.metrics:
                totals.setdefault(metric.name.value, []).append(float(metric.value))
    finally:
        engine.close_all()
    return hits / len(seeds), {k: sum(v) / len(v) for k, v in totals.items()}


def main() -> int:
    """Run every benchmark and report whether its declared expectation held."""
    base = default_scenario()
    rows = []
    for benchmark in BENCHMARKS:
        try:
            result = run_benchmark(
                benchmark, base_scenario=base, seeds=SEEDS, evaluate=evaluate
            )
        except ValueError as error:
            rows.append({"benchmark": benchmark.id, "error": str(error)})
            print(f"{benchmark.id:30s} ERROR {error}", flush=True)
            continue
        verdict = {True: "MET", False: "MISMATCH", None: "no claim"}[result.expectation_met]
        mark = "measured" if benchmark.verified else "hypothesis"
        print(
            f"{benchmark.id:30s} success={result.success_rate:.2f}  {verdict:9s} ({mark})",
            flush=True,
        )
        rows.append(
            {
                "benchmark": benchmark.id,
                "family": benchmark.family.value,
                "description": benchmark.description,
                "field_path": benchmark.field_path,
                "value": benchmark.value,
                "seeds": list(result.seeds),
                "success_rate": result.success_rate,
                "metrics": {k: round(v, 6) for k, v in sorted(result.metrics.items())},
                "expectation_met": result.expectation_met,
                "expectation_is_measured": benchmark.verified,
                "notes": result.notes,
            }
        )

    mismatches = [r for r in rows if r.get("expectation_met") is False]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "MEASURED",
                "policy": POLICY,
                "backend": "mujoco",
                "seeds": list(SEEDS),
                "benchmarks": len(BENCHMARKS),
                "mismatches": len(mismatches),
                "unimplemented_faults": [
                    {"benchmark": n, "reason": w} for n, w in UNIMPLEMENTED_BENCHMARKS
                ],
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"\n{len(BENCHMARKS)} benchmarks, {len(mismatches)} expectation mismatches, "
        f"{sum(1 for b in BENCHMARKS if b.verified)} with a measured expectation"
    )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
