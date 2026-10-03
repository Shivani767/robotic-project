"""Probe whether a real conditional regression exists at the v2 gate.

Read-only measurement: runs the two policies on the mujoco backend at a low
and a high depth-noise level and prints the metrics that could carry a
regression. No result is asserted; the script reports what it measures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from robot_ci.experiments.orchestrator import ExperimentRunner
from robot_ci.metrics import MetricName
from robot_ci.models.scenario import ExperimentConfig

DEPTH_LEVELS = [0.0, 60.0]
MAX_STEPS = 90


def main() -> int:
    """Probe how injected depth noise changes belief error on MuJoCo."""
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "probe_noise_regression"},
            "baseline": {"policy": "policy_v1"},
            "candidate": {"policy": "policy_v2"},
            "scenario": {"task": {"name": "pick_and_place", "max_steps": MAX_STEPS}},
            "variables": {"depth_noise_mm": {"values": DEPTH_LEVELS}},
            "repetitions": 1,
            "seed": 42,
        }
    )
    out = Path("experiments/results/probe_noise_regression")
    started = time.perf_counter()
    with ExperimentRunner(backend_name="mujoco") as runner:
        outcome = runner.run(config, output_dir=out)
    rows = []
    for result in outcome.results:
        rows.append(
            {
                "scenario": result.scenario_id,
                "policy": result.policy,
                "task_success": result.success,
                "status": result.status,
                "grasp_success": result.metric(MetricName.GRASP_SUCCESS),
                "object_lift_m": result.metric(MetricName.OBJECT_LIFT_M),
                "pose_sigma_m": result.metric(MetricName.POSE_SIGMA_M),
                "grasp_error_m": result.metric(MetricName.GRASP_ERROR_M),
                "ik_converged": result.metric(MetricName.IK_CONVERGED),
                "steps": result.steps,
            }
        )
    print(json.dumps({"rows": rows, "wall_s": time.perf_counter() - started}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
