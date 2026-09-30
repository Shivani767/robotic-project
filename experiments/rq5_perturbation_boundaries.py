"""RQ5: can sensor-specific perturbations locate behavioural boundaries?

Produces perturbation -> failure boundary curves for one axis at a time, and
reports where each axis has a boundary at all.

The interesting part is the axes that turn out *not* to matter. A benchmark that
sweeps a factor and gets a flat curve has still learned something -- that factor
is not a behavioural factor for this task, and a sweep costing hundreds of
episodes has measured nothing. Camera latency is the case already found: the
policy latches its plan on the first control tick and the object never moves, so
a delayed frame is never acted on, and the curve is flat at 1.00 from 0 to
200 ms.

Each axis is swept alone, holding the others nominal. That is deliberate: a
multi-axis sweep is needed to see interactions, but it cannot attribute a
boundary to a factor, and this experiment is about attribution to factors.

Arms are ``baseline=policy_v3, candidate=policy_v1``: the pair with a real
separation. ``boundary_mm`` is the smallest swept level at which the candidate
stops succeeding while the baseline still does, and ``None`` means no such
level was found within the range -- which is a result, not a missing value.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig, variable_path
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend

BASELINE_POLICY = "policy_v3"
CANDIDATE_POLICY = "policy_v1"
SEEDS: tuple[int, ...] = (1, 3, 4, 6, 8, 9)
OUTPUT = Path("experiments/results/rq5_perturbation_boundaries.json")

#: axis name -> (path, swept levels, unit). Ranges are chosen to bracket the
#: point of interest rather than to fill the axis: a sweep that stays flat
#: across 40 levels has bought no more than one that stays flat across 6.
AXES: dict[str, tuple[str, tuple[float, ...], str]] = {
    "depth_noise_mm": ("sensors.depth.sigma_mm", (0.0, 1.0, 2.0, 4.0, 8.0), "mm"),
    "depth_bias_mm": ("sensors.depth.bias_mm", (0.0, 2.0, 5.0, 10.0, 20.0), "mm"),
    "occlusion_percent": ("sensors.rgb.occlusion_percent", (0.0, 10.0, 30.0, 60.0), "%"),
    "joint_noise_rad": ("sensors.robot_state.joint_noise_std_rad", (0.0, 0.005, 0.02, 0.05), "rad"),
    "latency_ms": ("sensors.camera.latency.delay_ms", (0.0, 50.0, 100.0, 200.0), "ms"),
    "clutter_count": ("environment.clutter_count", (0.0, 2.0, 5.0, 9.0), "objects"),
    "object_friction": ("environment.object_friction", (1.0, 0.6, 0.3, 0.1), "coefficient"),
}


def _rate(name: str, path: str, value: float) -> tuple[float, int]:
    """Success rate of one arm at one point on one axis."""
    condition = set_by_path(default_scenario(), path, value)
    engine = create_backend("mujoco")
    try:
        hits = 0
        for seed in SEEDS:
            hits += int(
                run_episode(
                    engine,
                    create_policy(PolicyConfig(name=name), condition),
                    condition,
                    seed=seed,
                    experiment_id="rq5",
                ).success
            )
        return hits / len(SEEDS), len(SEEDS)
    finally:
        engine.close_all()


def main() -> int:
    """Sweep each perturbation axis alone and locate the separation boundary."""
    curves: dict[str, object] = {}
    for axis, (path, levels, unit) in AXES.items():
        if variable_path(axis) is None and not path:
            continue
        points: list[dict[str, object]] = []
        boundary: float | None = None
        for level in levels:
            base_rate, _samples = _rate(BASELINE_POLICY, path, level)
            cand_rate, _ = _rate(CANDIDATE_POLICY, path, level)
            if boundary is None and base_rate >= 0.5 and cand_rate < base_rate:
                boundary = level
            points.append(
                {
                    "value": level,
                    "baseline": base_rate,
                    "candidate": cand_rate,
                    "delta": round(cand_rate - base_rate, 4),
                }
            )
            print(
                f"  {axis:20s} {level:6g} {unit:10s} "
                f"baseline={base_rate:.2f} candidate={cand_rate:.2f}",
                flush=True,
            )
        flat = all(point["delta"] == 0 for point in points)
        curves[axis] = {
            "path": path,
            "unit": unit,
            "boundary_value": boundary,
            "no_separation": flat,
            "points": points,
        }
        print(f"  {axis:20s} -> boundary={boundary} {'FLAT' if flat else ''}\n", flush=True)

    with_boundaries = {a: c["boundary_value"] for a, c in curves.items() if c["boundary_value"] is not None}
    flat_axes = [a for a, c in curves.items() if c["no_separation"]]

    OUTPUT.write_text(
        json.dumps(
            {
                "question": "RQ5: where are the behavioural boundaries, per perturbation?",
                "status": "MEASURED",
                "baseline": BASELINE_POLICY,
                "candidate": CANDIDATE_POLICY,
                "seeds": list(SEEDS),
                "method": (
                    "one axis at a time, others nominal; boundary is the smallest "
                    "swept level where the baseline succeeds and the candidate "
                    "does not"
                ),
                "boundaries": with_boundaries,
                "axes_with_no_separation": flat_axes,
                "curves": curves,
                "note": (
                    "an axis listed under axes_with_no_separation is a finding: "
                    "it is not a behavioural factor for this task, and a sweep "
                    "over it costs episodes to measure nothing"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"axes with a boundary : {sorted(with_boundaries)}")
    print(f"axes with no separation: {sorted(flat_axes)}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
