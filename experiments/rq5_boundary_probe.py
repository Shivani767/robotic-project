"""RQ5 (re-validated): depth-noise boundary with repeated trials + transition curve.

PRE-REGISTERED criterion (recorded in this file for reproducibility):
  The behavioural boundary is the depth-noise standard deviation at which the
  candidate's success probability falls below the blocking threshold while the
  baseline's still exceeds it. Because this simulator is stochastic, we do not
  treat a threshold as deterministic: we estimate P(success | depth_noise) for
  both arms at a grid of noise levels and report Wilson intervals. The boundary
  is reported as a value with its 95% CI, plus the separated region where any
  difference is detectable at all. This also separates seed variability,
  sensor randomness and simulator nondeterminism: the same seeds are used for
  both arms at every level, so seed-level differences are not conflated with
each arm's behaviour.

Arms: baseline=policy_v3, candidate=policy_v1.

The probe re-runs the existing rq5 in-place (same outcome semantics); nothing
else is changed. It is a *probe*, not the primary experiment, and results are
appended to the artifacts rather than altering them.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend
from robot_ci.statistics import proportion_ci

BASELINE_POLICY = "policy_v3"
CANDIDATE_POLICY = "policy_v1"
#: Repeated seeds for every (arm, level) cell: keeps within-cell noise and
#: between-arm comparison fair.
SEEDS: tuple[int, ...] = tuple(range(1, 17))
#: Finer grid around the 1.0 mm interference, plus the coarse levels for the
#: full curve.
LEVELS: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 8.0)
OUTPUT = Path("experiments/results/rq5_boundary_probe.json")


def _rate(name: str, path: str, value: float, seeds: tuple[int, ...]) -> dict:
    condition = set_by_path(default_scenario(), path, value)
    engine = create_backend("mujoco")
    try:
        records = []
        n = 0
        for seed in seeds:
            result = run_episode(
                engine,
                create_policy(PolicyConfig(name=name), condition),
                condition,
                seed=seed,
                experiment_id="rq5_boundary_probe",
            )
            records.append(int(result.success))
            n += 1
    finally:
        engine.close_all()
    interval = proportion_ci(sum(records), n)
    return {
        "seed_records": records,
        "rate": round(interval.rate, 4),
        "wilson_95_ci": [round(interval.lower, 4), round(interval.upper, 4)],
        "n": n,
    }


def main() -> int:
    """Sweep depth noise for both arms and record the transition curve."""
    baseline_path = "sensors.depth.sigma_mm"
    candidate_path = "sensors.depth.sigma_mm"
    curves = {}
    for level in LEVELS:
        b = _rate(BASELINE_POLICY, baseline_path, level, SEEDS)
        c = _rate(CANDIDATE_POLICY, candidate_path, level, SEEDS)
        curves[str(level)] = {
            "baseline": b,
            "candidate": c,
            "delta": round(c["rate"] - b["rate"], 4),
        }
    OUTPUT.write_text(json.dumps({
        "question": "RQ5 boundary: transition curve with repeated trials",
        "status": "MEASURED",
        "baseline": BASELINE_POLICY,
        "candidate": CANDIDATE_POLICY,
        "seeds": SEEDS,
        "curves": curves,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
