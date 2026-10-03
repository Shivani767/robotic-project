r"""Trace one real Gazebo episode step by step: what the policy saw, where it went.

The episode fails, but a single ``status=failure`` line says nothing about *why*.
This records, for every control tick, the end-effector position, the believed
object position and the hidden ground truth, so the failure can be attributed to
perception, to reaching, or to the task's own criteria -- which are three very
different findings and must not be conflated.

Run inside the container::

    docker run --rm -v "$PWD":/workspace/robot-ci -w /workspace/robot-ci \\
        robot-ci-gazebo-demo:latest \\
        bash -c 'source /opt/ros/jazzy/setup.bash && python3 scripts/gazebo_episode_trace.py'
"""

from __future__ import annotations

import numpy as np


def main() -> int:
    """Run one episode and print the per-tick error the policy actually acted on."""
    from robot_ci.experiments.runner import StepRecord, run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario
    from robot_ci.simulation.gazebo import GazeboBackend

    scenario = default_scenario(seed=3)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    backend = GazeboBackend(startup_timeout_s=120)
    rows: list[dict[str, object]] = []

    def hook(record: StepRecord) -> None:
        belief = record.observation.objects
        truth = record.ground_truth.objects
        ee = record.observation.robot.end_effector_pose.position
        row: dict[str, object] = {
            "step": record.step_index,
            "sim_t": round(record.sim_time_s, 3),
            "phase": str(record.outcome.phase),
            "ee": (round(ee.x, 4), round(ee.y, 4), round(ee.z, 4)),
        }
        if belief and truth:
            b, t = belief[0].pose.position, truth[0].pose.position
            row["belief"] = (round(b.x, 4), round(b.y, 4), round(b.z, 4))
            row["truth"] = (round(t.x, 4), round(t.y, 4), round(t.z, 4))
            row["belief_err_m"] = round(float(np.linalg.norm([b.x - t.x, b.y - t.y, b.z - t.z])), 4)
            row["ee_to_truth_m"] = round(
                float(np.linalg.norm([ee.x - t.x, ee.y - t.y, ee.z - t.z])), 4
            )
            row["grasped"] = bool(truth[0].grasped)
        rows.append(row)

    try:
        result = run_episode(
            backend=backend,
            scenario=scenario,
            policy=policy,
            seed=3,
            experiment_id="gazebo_trace",
            step_hook=hook,
        )
    finally:
        backend.close_all()

    print(f"episode status={result.status} success={result.success} steps={result.steps}")
    seen = [r for r in rows if "belief_err_m" in r]
    print(f"ticks carrying a belief: {len(seen)}/{len(rows)}")
    if seen:
        errs = [float(r["belief_err_m"]) for r in seen]
        reaches = [float(r["ee_to_truth_m"]) for r in seen]
        print(
            f"belief error: median {np.median(errs):.4f} m, max {max(errs):.4f} m\n"
            f"ee->truth   : min {min(reaches):.4f} m, median {np.median(reaches):.4f} m"
        )
        print(f"closest approach ever reached: {min(reaches):.4f} m (grasp latch needs <= 0.05)")
    print("\nstep  sim_t   phase        belief                 truth                  err_m    ee->truth")
    for row in rows[:: max(1, len(rows) // 25)]:
        if "belief" in row:
            print(
                f"{row['step']:>4}  {row['sim_t']:>6}  {row['phase']!s:<12} "
                f"{row['belief']!s:<22} {row['truth']!s:<22} "
                f"{row['belief_err_m']:>6.4f}  {row['ee_to_truth_m']:>8.4f}"
            )
        else:
            print(f"{row['step']:>4}  {row['sim_t']:>6}  {row['phase']!s:<12} (no belief)")
    for metric in result.metrics:
        print(f"  {metric.name.value} = {metric.value:.4g} {metric.unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
