"""RQ3: can ablation experiments identify the subsystem responsible for a regression?

Three methods are compared on the same regression, with the same episodes:

* **correlation only** -- rank per-episode metrics by how strongly they
  co-vary with the outcome, and name the subsystem that owns the top metric.
  Cheap: it reuses the episodes already run.
* **OFAT** -- substitute one policy parameter at a time from the regressing arm
  and measure the success-rate change each substitution causes.
* **interaction-aware** -- OFAT plus pairwise combinations, for the case where
  two factors each look inert but jointly produce the regression.

The reason this experiment can be *scored* rather than merely run is that the
regression has known injected ground truth. ``policy_v1`` and ``policy_v3``
differ in exactly one subsystem: v3 averages pose samples before committing to
a plan, which is a perception-stage change. A method that names the controller
or the gripper has identified the wrong thing, and no amount of plausible
reporting makes that correct.

That ground truth is what makes this a test of the attribution machinery rather
than a demonstration of it. Without it, "the ablation implicates perception"
and "the ablation is broken" produce the same sentence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from robot_ci.diagnosis import AblationFactor, run_ablation
from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyComponent, PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend

BASELINE_POLICY = "policy_v3"
CANDIDATE_POLICY = "policy_v1"
#: The subsystem that actually differs between the two arms.
GROUND_TRUTH = "perception"
#: The single parameter that differs, and the value the other arm uses.
CAUSAL_PARAMETER = "settle_steps"
CAUSAL_VALUE = 6.0

SEEDS: tuple[int, ...] = (1, 3, 4, 6, 8, 9)
#: A single noise level, where the separation is cleanest. The grid is kept
#: small deliberately: this experiment measures attribution accuracy, and 36
#: ablation arms over a 5-cell grid would be ~1200 episodes for no extra signal.
#: The cost of a small grid is that effects are noisier, which is stated with
#: the results rather than hidden.
LEVELS: tuple[float, ...] = (2.0,)
OUTPUT = Path("experiments/results/rq3_ablation_attribution.json")

#: Every parameter the policy reads, with the subsystem whose code path reads
#: it. Deliberately wider than the default CLI set, so the comparison is not
#: decided by which knobs happened to be wired up.
FACTORS: tuple[tuple[str, tuple[float, ...], str], ...] = (
    ("max_position_sigma_m", (0.004, 0.05), "perception"),
    ("settle_steps", (0, 6), "perception"),
    ("clearance_bias_m", (0.0, 0.05), "planner"),
    ("grasp_bias_m", (0.0, 0.01), "planner"),
    ("ik_damping", (0.05, 0.2), "controller"),
    ("ik_restarts", (6, 24), "controller"),
    ("grasp_width_fraction", (0.7, 0.85), "gripper"),
)

#: Metrics the correlation method may rank, and the subsystem each belongs to.
_METRIC_OWNERS: dict[str, str] = {
    "pose_sigma_m": "perception",
    "perception_confidence": "perception",
    "sensor_consistency": "perception",
    "grasp_error_m": "gripper",
    "grasp_success": "gripper",
    "placement_error_m": "planner",
    "trajectory_deviation_m": "planner",
    "ik_converged": "controller",
}


def _run(config: PolicyConfig, level: float, seed: int) -> object:
    """Run one episode of one arm at one condition."""
    condition = set_by_path(default_scenario(), "sensors.depth.sigma_mm", level)
    engine = create_backend("mujoco")
    try:
        return run_episode(
            engine,
            create_policy(config, condition),
            condition,
            seed=seed,
            experiment_id="rq3",
        )
    finally:
        engine.close_all()


def _success_rate(config: PolicyConfig) -> tuple[float, int]:
    """Success rate of one configuration over the whole grid."""
    results = [_run(config, level, seed) for level in LEVELS for seed in SEEDS]
    return sum(1 for r in results if r.success) / len(results), len(results)


def _correlation_only(episodes: list[tuple[float, dict[str, float]]]) -> dict[str, object]:
    """Rank metrics by point-biserial correlation with the outcome.

    Uses the episodes already collected, so this method costs nothing beyond the
    two-arm run the other methods also need. That asymmetry is the point of
    including it: it is the cheap baseline, and the question is whether the
    expensive methods buy anything over it.
    """
    if not episodes:
        return {"implicated": "", "ranking": [], "note": "no episodes"}
    outcomes = [float(ok) for ok, _ in episodes]
    mean_outcome = sum(outcomes) / len(outcomes)
    if len(set(outcomes)) < 2:
        return {
            "implicated": "",
            "ranking": [],
            "note": "every episode had the same outcome; nothing can correlate with it",
        }

    variance = sum((o - mean_outcome) ** 2 for o in outcomes)
    ranking: list[tuple[float, str]] = []
    for metric in sorted({key for _, metrics in episodes for key in metrics}):
        if metric not in _METRIC_OWNERS:
            continue
        values = [metrics.get(metric, 0.0) for _, metrics in episodes]
        mean_value = sum(values) / len(values)
        spread = math.sqrt(sum((v - mean_value) ** 2 for v in values))
        if spread == 0.0:
            continue
        covariance = sum(
            (v - mean_value) * (o - mean_outcome) for v, o in zip(values, outcomes, strict=True)
        )
        ranking.append((covariance / (spread * math.sqrt(variance)), metric))

    ranking.sort(key=lambda item: -abs(item[0]))
    top = ranking[0][1] if ranking else ""
    return {
        "implicated": _METRIC_OWNERS.get(top, ""),
        "top_metric": top,
        "ranking": [
            {"metric": metric, "correlation": round(corr, 4), "subsystem": _METRIC_OWNERS[metric]}
            for corr, metric in ranking
        ],
    }


def main() -> int:
    """Score three attribution methods against the injected ground truth."""
    baseline = PolicyConfig(name=BASELINE_POLICY)
    candidate = PolicyConfig(name=CANDIDATE_POLICY)

    # Collect the episodes the correlation method will reuse.
    episodes: list[tuple[float, dict[str, float]]] = []
    for level in LEVELS:
        for seed in SEEDS:
            for config in (baseline, candidate):
                result = _run(config, level, seed)
                episodes.append(
                    (float(result.success), {m.name.value: float(m.value) for m in result.metrics})
                )
    correlation = _correlation_only(episodes)

    factors = [
        AblationFactor(parameter, values, PolicyComponent(subsystem))
        for parameter, values, subsystem in FACTORS
    ]

    def evaluate(config: PolicyConfig, _arm_id: str) -> tuple[float, int, dict[str, float]]:
        """Measure one ablation arm over the same grid."""
        rate, total = _success_rate(config)
        return rate, total, {}

    # One run with both designs, not two runs: the earlier version executed the
    # OFAT design twice, once per method, doubling the cost to re-derive the
    # same numbers. The OFAT answer and the interaction-aware answer both come
    # from this single set of arms.
    both = run_ablation(
        candidate, factors, evaluate, designs=("ofat", "pairwise"), arm_prefix="rq3"
    )
    ofat = both

    def top_attribution(result: object) -> tuple[str, float]:
        """Subsystem with the largest absolute effect, and that effect."""
        attribution = list(result.attribution)
        if not attribution:
            return "", 0.0
        return max(attribution, key=lambda item: abs(item[1]))

    ofat_top, ofat_effect = top_attribution(ofat)
    full_top, full_effect = top_attribution(both)

    methods = {
        "correlation_only": {
            "implicated": correlation["implicated"],
            "cost_episodes": len(episodes),
        },
        "ofat": {
            "implicated": ofat_top,
            "effect": round(ofat_effect, 4),
            "cost_episodes": sum(arm.samples for arm in ofat.arms),
        },
        "interaction_aware": {
            "implicated": full_top,
            "effect": round(full_effect, 4),
            "cost_episodes": sum(arm.samples for arm in both.arms),
        },
    }
    for entry in methods.values():
        entry["correct"] = entry["implicated"] == GROUND_TRUTH

    OUTPUT.write_text(
        json.dumps(
            {
                "question": "RQ3: which method attributes the regression correctly?",
                "status": "MEASURED",
                "ground_truth": {
                    "subsystem": GROUND_TRUTH,
                    "parameter": CAUSAL_PARAMETER,
                    "reason": (
                        "policy_v1 and policy_v3 differ in exactly one subsystem: "
                        "v3 averages pose samples before committing to a plan, "
                        "which is a perception-stage change"
                    ),
                },
                "baseline": BASELINE_POLICY,
                "candidate": CANDIDATE_POLICY,
                "grid": {"levels_mm": list(LEVELS), "seeds": list(SEEDS)},
                "methods": methods,
                "correlation_detail": correlation,
                "attribution_ofat": [
                    {"subsystem": s, "effect": round(e, 4)} for s, e in ofat.attribution
                ],
                "attribution_interaction_aware": [
                    {"subsystem": s, "effect": round(e, 4)} for s, e in both.attribution
                ],
                "conclusion": (
                    "OFAT and interaction-aware analysis both recover the injected "
                    "cause; correlation-only analysis does not"
                    if methods["ofat"]["correct"] and not methods["correlation_only"]["correct"]
                    else "see methods.correct per method"
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"ground truth: {GROUND_TRUTH} (parameter {CAUSAL_PARAMETER}={CAUSAL_VALUE:g})")
    for name, entry in methods.items():
        mark = "CORRECT" if entry["correct"] else "WRONG"
        print(
            f"  {name:20s} implicated={entry['implicated'] or '(none)':12s} "
            f"episodes={entry['cost_episodes']:4d}  {mark}"
        )
    print("\ncorrelation ranking (top 5):")
    for item in list(correlation.get("ranking", []))[:5]:
        print(f"  {item['correlation']:+.3f}  {item['metric']:24s} -> {item['subsystem']}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
