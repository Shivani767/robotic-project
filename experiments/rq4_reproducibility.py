"""RQ4: how reproducible are automatically discovered counterexamples?

A counterexample found by search is a claim about a *condition*. Whether it is
also a claim about the system depends on how often it reproduces when the seed
changes, and a search that reports only "found" cannot tell the difference
between a genuine behavioural boundary and a lucky episode.

Method: discover the counterexample on one seed, then replay that exact
condition across an independent seed set and report the reproduction rate. The
reproduction rate is the headline number, not the discovery.

Arm roles are ``baseline=policy_v3, candidate=policy_v1``: the searcher looks for
conditions where the candidate fails and the baseline succeeds. See
``rq1_search_strategies.py`` for why the roles are inverted from the v1/v2 pair.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.counterexamples import (
    EpisodeOracle,
    PerturbationAxis,
    SearchSpace,
)
from robot_ci.models.scenario import PolicyConfig
from robot_ci.scenarios import default_scenario

BASELINE_POLICY = "policy_v3"
CANDIDATE_POLICY = "policy_v1"

#: Seed the counterexample is discovered on.
DISCOVERY_SEED = 3
#: Independent seeds the counterexample is replayed on. Disjoint from the
#: discovery seed so the reproduction rate is not inflated by the same episode.
REPLAY_SEEDS: tuple[int, ...] = (1, 4, 6, 8, 9, 12, 13, 19, 20, 22, 24)
#: Noise levels at which the separation is looked for, in mm.
LEVELS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
OUTPUT = Path("experiments/results/rq4_reproducibility.json")


def _oracle(seed: int) -> EpisodeOracle:
    axis = PerturbationAxis("depth_noise_mm", "sensors.depth.sigma_mm", 0.0, 12.0, "mm")
    return EpisodeOracle(
        base_scenario=default_scenario(seed=seed),
        space=SearchSpace((axis,)),
        baseline=PolicyConfig(name=BASELINE_POLICY),
        candidate=PolicyConfig(name=CANDIDATE_POLICY),
        backend_name="mujoco",
    )


def main() -> int:
    """Discover a counterexample on one seed, then replay it on others."""
    # 1. Confirm the condition exists on the discovery seed, and take the
    #    smallest noise level at which it does.
    oracle = _oracle(DISCOVERY_SEED)
    try:
        condition = None
        for level in LEVELS:
            trial = oracle.evaluate({"depth_noise_mm": level})
            if trial.is_counterexample:
                condition = level
                break
    finally:
        oracle.close()
    if condition is None:
        OUTPUT.write_text(
            json.dumps(
                {"status": "NOT RUN", "reason": "no counterexample on the discovery seed"},
                indent=2,
            )
            + "\n"
        )
        print("no counterexample found; RQ4 not run")
        return 1
    print(f"discovery seed {DISCOVERY_SEED}: counterexample at depth_noise_mm={condition:g}")

    # 2. Replay that one condition on independent seeds.
    rows = []
    reproduced = 0
    for seed in REPLAY_SEEDS:
        oracle = _oracle(seed)
        try:
            trial = oracle.evaluate({"depth_noise_mm": condition})
        finally:
            oracle.close()
        reproduced += int(trial.is_counterexample)
        rows.append(
            {
                "seed": seed,
                "baseline_success": trial.baseline_success,
                "candidate_success": trial.candidate_success,
                "reproduced": trial.is_counterexample,
            }
        )
        print(
            f"  seed {seed:3d}: baseline={trial.baseline_success} "
            f"candidate={trial.candidate_success} reproduced={trial.is_counterexample}",
            flush=True,
        )

    rate = reproduced / len(REPLAY_SEEDS)
    OUTPUT.write_text(
        json.dumps(
            {
                "question": "RQ4: reproduction rate of a discovered counterexample",
                "status": "MEASURED",
                "condition": {"depth_noise_mm": condition},
                "baseline": BASELINE_POLICY,
                "candidate": CANDIDATE_POLICY,
                "discovery_seed": DISCOVERY_SEED,
                "replay_seeds": list(REPLAY_SEEDS),
                "reproduced": reproduced,
                "replay_count": len(REPLAY_SEEDS),
                "reproduction_rate": rate,
                "trials": rows,
                "interpretation": (
                    "a rate well below 1.0 means the counterexample is a property "
                    "of the discovery seed as much as of the condition. It must not "
                    "be reported as a behavioural boundary without this number "
                    "attached."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nreproduction rate: {reproduced}/{len(REPLAY_SEEDS)} = {rate:.1%}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
