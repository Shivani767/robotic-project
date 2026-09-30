"""RQ1: does structured search find a counterexample faster than random sampling?

Compares the four counterexample search strategies under an identical budget,
on identical seeds, against an identical oracle. The oracle is the real MuJoCo
episode, so "tests to failure" counts real simulator rollouts.

The comparison is only meaningful because the budget is the same for every
strategy and each strategy is given the same search space. Two things were
established by measurement before this comparison was run, and both are
recorded here because the first version of this experiment was invalid without
them:

* **No counterexample exists between policy_v1 and policy_v2.** The candidate's
  confidence gate is a bounded fail-open hold, so it costs control ticks rather
  than successes, and the two arms degrade together at every noise level. An
  earlier run of this script correctly refused to compare anything and wrote
  ``NOT RUN``.
* **A counterexample does exist once a known-behaviour policy is the
  baseline.** ``policy_v3`` averages pose samples before committing to a plan.
  Against it, ``policy_v1`` is the regressing arm -- it loses where the
  averaging wins -- which is the separation the searchers need.

Arm roles are therefore ``baseline=policy_v3, candidate=policy_v1``: the search
looks for conditions where the *candidate* fails and the *baseline* succeeds,
which is the framework's definition, applied to the arm that actually regresses.

The seed is pinned to 3 because it is one of twelve scanned seeds on which a
per-seed separation exists at all. That is a limitation, not a convenience: a
counterexample reproduced on one seed out of twelve is a property of the seed as
much as of the condition, and RQ4 exists to measure exactly that. The result
below must not be quoted without the reproduction rate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from robot_ci.counterexamples import (
    CachingOracle,
    DiscoveryRequest,
    EpisodeOracle,
    PerturbationAxis,
    SearchSpace,
    discover,
)
from robot_ci.counterexamples.strategies import (
    AdaptiveBoundarySearcher,
    BinaryBoundarySearcher,
    GridSearcher,
    RandomSearcher,
)
from robot_ci.models.scenario import PolicyConfig
from robot_ci.scenarios import default_scenario

#: Oracle evaluations each strategy may spend. Identical for all four, which is
#: the only thing that makes the comparison mean anything.
BUDGET = 24

#: Seed pinned because a per-seed separation exists here and on 0 of the other
#: eleven seeds scanned. See the module docstring.
SEED = 3

BASELINE_POLICY = "policy_v3"
CANDIDATE_POLICY = "policy_v1"
OUTPUT = Path("experiments/results/rq1_search_strategies.json")


def main() -> int:
    """Run the four strategies against one oracle and write the comparison."""
    base = default_scenario(seed=SEED)
    axis = PerturbationAxis("depth_noise_mm", "sensors.depth.sigma_mm", 0.0, 12.0, "mm")
    space = SearchSpace((axis,))

    # First establish that a counterexample exists on this axis at all.
    oracle = EpisodeOracle(
        base_scenario=base,
        space=space,
        baseline=PolicyConfig(name=BASELINE_POLICY),
        candidate=PolicyConfig(name=CANDIDATE_POLICY),
        backend_name="mujoco",
    )
    try:
        probe = CachingOracle(oracle)
        found = [p for p in _ramp() if probe.evaluate(p).is_counterexample]
    finally:
        oracle.close()
    if not found:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(
                {
                    "status": "NOT RUN",
                    "reason": "no counterexample exists on this axis, so comparing "
                    "strategies would measure four ways of failing to find nothing",
                },
                indent=2,
            )
            + "\n"
        )
        print("no counterexample on the axis; comparison not run")
        return 1
    print(f"counterexample confirmed on the axis at {found}")

    rows = []
    for name, searcher in (
        ("grid", GridSearcher(space, budget=BUDGET)),
        ("random", RandomSearcher(space, budget=BUDGET, seed=0)),
        ("binary", BinaryBoundarySearcher(space, budget=BUDGET)),
        ("adaptive", AdaptiveBoundarySearcher(space, budget=BUDGET)),
    ):
        oracle = EpisodeOracle(
            base_scenario=base,
            space=space,
            baseline=PolicyConfig(name=BASELINE_POLICY),
            candidate=PolicyConfig(name=CANDIDATE_POLICY),
            backend_name="mujoco",
        )
        try:
            request = DiscoveryRequest(
                experiment_id=f"rq1_{name}",
                base_scenario=base,
                space=space,
                baseline_policy=BASELINE_POLICY,
                candidate_policy=CANDIDATE_POLICY,
                seed=base.seed,
                backend="mujoco",
            )
            started = time.perf_counter()
            result = discover(request, CachingOracle(oracle), searcher)
            elapsed = time.perf_counter() - started
        finally:
            oracle.close()
        outcome = result.outcome
        rows.append(
            {
                "strategy": name,
                "found": outcome.found,
                "oracle_evaluations": outcome.evaluations,
                "wall_clock_s": round(elapsed, 2),
                "severity": outcome.severity,
                "boundary": outcome.boundary,
                "boundary_axis": outcome.boundary_axis,
                "points_visited": len(outcome.trajectory),
                "note": outcome.notes,
            }
        )
        print(
            f"{name:9s} found={outcome.found} "
            f"evals={outcome.evaluations:3d} boundary={outcome.boundary} {elapsed:6.1f}s"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "question": "RQ1: tests to failure, by search strategy",
                "baseline": BASELINE_POLICY,
                "candidate": CANDIDATE_POLICY,
                "seed_rationale": (
                    "a per-seed separation exists on seed 3 and on 0 of the other "
                    "11 seeds scanned; a single-seed counterexample is a property "
                    "of the seed as much as the condition"
                ),
                "status": "MEASURED",
                "axis": {"name": axis.name, "low": axis.lower, "high": axis.upper, "unit": axis.unit},
                "budget": BUDGET,
                "backend": "mujoco",
                "seed": base.seed,
                "oracle": "one paired episode per point; both arms share the seed",
                "limitation": (
                    "the oracle evaluates a single seed per point, so a "
                    "counterexample found here is a property of that seed as well "
                    "as of the condition. Reproducibility across seeds is RQ4, "
                    "not this result, and a single-seed finding must not be "
                    "reported as a regression without its reproduction rate."
                ),
                "strategies": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT}")
    return 0


def _ramp() -> list[dict[str, float]]:
    """Points across the axis, used only to confirm a separation exists."""
    return [{"depth_noise_mm": float(v)} for v in range(0, 13, 2)]


if __name__ == "__main__":
    raise SystemExit(main())
