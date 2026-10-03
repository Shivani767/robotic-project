"""Run the real PickAndPlace episode through Gazebo and record what happened.

This is the Phase 2/3/4 evidence in one place. It drives the *existing*
``PickAndPlacePolicy``, ``PickAndPlaceTask`` and ``run_episode`` loop through
``GazeboBackend`` and records, for every control tick, quantities that came off
Gazebo's own topics:

* the measured tool-centre-point position,
* the measured object pose (the simulator's, never reconstructed),
* the measured finger aperture (where the pads actually are),
* whether the task's grasp criterion is satisfied.

Nothing here asserts an outcome; it measures one and reports it, including the
per-stage verdicts the Phase 3 checklist asks for. A grasp that never happened
is reported as a grasp that never happened.

Run inside the container::

    scripts/gz_run.sh start 'source /opt/ros/jazzy/setup.bash \
        && python3 scripts/gazebo_pick_place_evidence.py --seed 3 --noise-mm 0'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path("experiments/results/gazebo")

#: Minimum lift the task itself requires, from ``pick_and_place.min_lift_height_m``.
REQUIRED_LIFT_M = 0.05

#: Distance from the tool centre point to the object below which the jaws can
#: physically enclose it. The pads are 20 mm deep and the object is 16 mm across,
#: so anything beyond this is not a candidate grasp whatever the command says.
ENCLOSURE_M = 0.03


def _dist(a: Any, b: Any) -> float:
    """Euclidean distance between two 3-vectors."""
    return float(
        np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    )


class _DiagnosticsProxy:
    """Wraps a policy to capture what it decided each tick.

    ``run_episode`` does not pass the policy's diagnostics to its step hook, and
    ``StepRecord`` deliberately carries only what the step produced. Rather than
    widen the shared record for one investigation, the policy is wrapped: the
    evidence keeps everything the policy said about its own decisions (which
    waypoint it was driving at, its IK residual, whether the solver converged)
    alongside what the simulator then did with it.
    """

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self.rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def act(self, observation: Any) -> Any:
        output = self._policy.act(observation)
        self.rows.append(dict(output.diagnostics))
        return output


def run(seed: int, noise_mm: float, policy_name: str, max_steps: int | None) -> dict[str, Any]:
    """Run one episode and return everything that was measured."""
    from robot_ci.experiments.runner import StepRecord, run_episode
    from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario, set_by_path
    from robot_ci.simulation.gazebo import GazeboBackend

    scenario: ScenarioConfig = default_scenario(seed=seed)
    if noise_mm:
        scenario = set_by_path(scenario, "sensors.depth.sigma_mm", float(noise_mm))
        scenario = set_by_path(
            scenario, "scenario_id", f"depth_noise_{noise_mm:g}mm_seed{seed}"
        )
    if max_steps is not None:
        scenario = set_by_path(scenario, "task.max_steps", int(max_steps))

    policy = _DiagnosticsProxy(create_policy(PolicyConfig(name=policy_name), scenario))
    backend = GazeboBackend(startup_timeout_s=180)
    rows: list[dict[str, Any]] = []
    started = time.time()

    def hook(record: StepRecord) -> None:
        ee = record.observation.robot.end_effector_pose.position
        row: dict[str, Any] = {
            "step": record.step_index,
            "sim_t": round(record.sim_time_s, 4),
            "phase": str(record.outcome.phase),
            "tcp": [round(v, 5) for v in (ee.x, ee.y, ee.z)],
            "aperture_m": round(record.observation.robot.gripper.width_m, 5),
            # What the simulator says is in contact, straight off the contact
            # sensors on the object and the table. This is the measurement that
            # replaces inference from the object's velocity.
            "object_contacts": record.transition.info.get("object_contacts", {}),
            "table_contacts": record.transition.info.get("table_contacts", {}),
            "contact_health": record.transition.info.get("contact_health", {}),
            # What was *commanded*, so a limit cycle in the controller can be
            # told apart from one in the plan: if the command is steady and the
            # joint state oscillates, the controller is the cause.
            "commanded": [round(float(v), 5) for v in (record.action.joint_position or ())],
            "measured_joints": [
                round(float(v), 5) for v in record.observation.robot.joint_positions
            ],
        }
        objects = record.ground_truth.objects
        if objects:
            obj = objects[0]
            row["object"] = [round(v, 5) for v in obj.pose.position.array]
            # The object's orientation is what the task's orientation criterion
            # is scored on, and for a cube it is invisible in the resting height:
            # every 90-degree rotation leaves the half-extent the same. Recording
            # the quaternion is the only way to tell a harmless yaw from a tumble.
            row["object_quat"] = [
                round(float(getattr(obj.pose.orientation, k)), 4)
                for k in ("w", "x", "y", "z")
            ]
            row["grasped"] = bool(obj.grasped)
            row["object_to_tcp_m"] = round(
                _dist((ee.x, ee.y, ee.z), obj.pose.position.array), 5
            )
        rows.append(row)

    try:
        result = run_episode(
            backend=backend,
            scenario=scenario,
            policy=policy,
            seed=seed,
            experiment_id=f"gazebo_pick_place_seed{seed}_{noise_mm:g}mm",
            step_hook=hook,
        )
    finally:
        backend.close_all()

    for index, row in enumerate(rows):
        diagnostics = policy.rows[index] if index < len(policy.rows) else {}
        row["policy"] = {
            "waypoint": diagnostics.get("waypoint"),
            "drive_waypoint": diagnostics.get("drive_waypoint"),
            "waypoint_target": diagnostics.get("waypoint_target"),
            "reached_waypoint": diagnostics.get("reached_waypoint"),
            "dwell": diagnostics.get("dwell"),
            "ik_error_mm": diagnostics.get("ik_error_mm"),
            "ik_converged": diagnostics.get("ik_converged"),
            "ik_orientation_error_rad": diagnostics.get("ik_orientation_error_rad"),
            "reason": diagnostics.get("reason"),
        }

    return {
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "depth_noise_mm": float(noise_mm),
        "policy": policy_name,
        "status": result.status,
        "success": bool(result.success),
        "steps": result.steps,
        "conditions": result.conditions,
        "metrics": {m.name.value: float(m.value) for m in result.metrics},
        "wall_clock_s": round(time.time() - started, 2),
        "rows": rows,
    }

def judge(record: dict[str, Any], *, table_height_m: float = 0.0) -> dict[str, Any]:
    """Turn a measured trace into the per-stage verdicts Phase 3 asks for.

    Each verdict is a claim about a measured quantity with that quantity
    printed, so a FAIL says what it failed by and never just "did not work".
    """
    rows = [r for r in record["rows"] if "object" in r]
    if not rows:
        return {"1_world_and_control": {"ok": False, "reason": "no ticks carried an object pose"}}
    tcp_to_object = [r["object_to_tcp_m"] for r in rows]
    apertures = [r["aperture_m"] for r in rows]
    object_z = [r["object"][2] for r in rows]
    grasped = [r for r in rows if r.get("grasped")]
    lifted = max(object_z) - table_height_m
    metrics = record["metrics"]
    return {
        "1_world_and_control": {
            "ok": record["steps"] > 0,
            "steps": record["steps"],
            "ticks_with_a_measured_object": len(rows),
        },
        "2_gripper_enclosed_the_object": {
            "ok": min(tcp_to_object) <= ENCLOSURE_M,
            "closest_tcp_to_object_m": round(min(tcp_to_object), 5),
            "bound_m": ENCLOSURE_M,
            "minimum_measured_aperture_m": round(min(apertures), 5),
        },
        "3_object_lifted_by_contact": {
            "ok": bool(grasped) and lifted >= REQUIRED_LIFT_M,
            "ticks_the_task_counted_as_grasped": len(grasped),
            "max_object_height_above_table_m": round(lifted, 5),
            "required_lift_m": REQUIRED_LIFT_M,
        },
        "4_object_released_into_the_target": {
            "ok": metrics.get("placement_success", 0.0) == 1.0,
            "final_object_pose": rows[-1]["object"],
            "measured_placement_error_m": metrics.get("placement_error_m"),
            "tolerance_m": 0.05,
        },
        "5_task_outcome": {
            "ok": record["success"],
            "status": record["status"],
            "placement_error_m": metrics.get("placement_error_m"),
            "grasp_error_m": metrics.get("grasp_error_m"),
            "measured_object_lift_m": metrics.get("object_lift_m"),
        },
    }


def main() -> int:
    """Parse arguments, run the episode, and write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--noise-mm", type=float, default=0.0)
    parser.add_argument("--policy", default="policy_v1")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    record = run(args.seed, args.noise_mm, args.policy, args.max_steps)
    verdicts = judge(record)
    payload = {**record, "stages": verdicts}

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"pick_place_seed{args.seed}_{args.noise_mm:g}mm_{args.policy}"
    (RESULTS / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    print(f"scenario={record['scenario_id']} policy={record['policy']}")
    print(f"status={record['status']} success={record['success']} steps={record['steps']}")
    for stage, verdict in verdicts.items():
        print(f"  {stage}: {'PASS' if verdict.get('ok') else 'FAIL'}")
        for key, value in verdict.items():
            if key != "ok":
                print(f"      {key}: {value}")
    print(f"wrote {RESULTS / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
