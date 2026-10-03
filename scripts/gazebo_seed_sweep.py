r"""Run Gazebo PickAndPlace over several seeds and report what each one measured.

Two questions this answers, both of which a single episode cannot:

* **Which seeds actually succeed?** The seed chooses where the object is placed,
  so reachability and self-occlusion both vary with it. Reporting one seed is
  reporting one draw from that distribution.
* **Why does a seed fail?** Belief error separates a *perception* failure from a
  *reaching* failure, and they need completely different fixes. A run is only
  honestly "successful" if the task's own status says so.

Run inside the container::

    docker run --rm -v "$PWD":/workspace/robot-ci -w /workspace/robot-ci \\
        robot-ci-gazebo-demo:latest bash -c \\
        'source /opt/ros/jazzy/setup.bash && python3 scripts/gazebo_seed_sweep.py'
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_ci.experiments.runner import StepRecord


def _hook_for(
    belief_errors: list[float], reaches: list[float], object_xy: list[str]
) -> Callable[[StepRecord], None]:
    """Build one seed's per-tick hook over the caller-owned accumulators.

    A factory rather than a closure over the loop: the hook runs on the
    runner's schedule, so the lists it appends to must be fresh per seed and
    owned by the caller, not captured from wherever the definition sits.
    """
    def hook(record: StepRecord) -> None:
        belief, truth = record.observation.objects, record.ground_truth.objects
        if not belief or not truth:
            return
        b, t = belief[0].pose.position, truth[0].pose.position
        belief_errors.append(float(np.linalg.norm([b.x - t.x, b.y - t.y, b.z - t.z])))
        ee = record.observation.robot.end_effector_pose.position
        reaches.append(float(np.linalg.norm([ee.x - t.x, ee.y - t.y, ee.z - t.z])))
        if object_xy[0] == "n/a":
            object_xy[0] = f"({t.x:+.3f},{t.y:+.3f})"

    return hook


def main() -> int:
    """Run one episode per seed and print a comparable table of measured outcomes."""
    from robot_ci.experiments.runner import run_episode
    from robot_ci.metrics import MetricName
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario
    from robot_ci.simulation.gazebo import GazeboBackend

    seeds = [int(value) for value in sys.argv[1:]] or [3, 7, 11, 19, 23, 42]
    print(
        f"{'seed':>4} {'status':>8} {'ok':>3} {'steps':>6} {'belief_ticks':>13} "
        f"{'belief_err':>11} {'reach_m':>9} {'grasp_ok':>9} {'obj_xy':>18}"
    )
    for seed in seeds:
        scenario = default_scenario(seed=seed)
        policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
        backend = GazeboBackend(startup_timeout_s=120)
        belief_errors: list[float] = []
        reaches: list[float] = []
        object_xy = ["n/a"]
        hook = _hook_for(belief_errors, reaches, object_xy)

        try:
            result = run_episode(
                backend=backend,
                scenario=scenario,
                policy=policy,
                seed=seed,
                experiment_id=f"gazebo_sweep_{seed}",
                step_hook=hook,
            )
        except Exception as exc:  # a crash is a measured outcome, not an excuse
            print(f"{seed:>4} {'ERROR':>8} {'-':>3} {'-':>6}  {type(exc).__name__}: {exc}")
            backend.close_all()
            continue
        finally:
            backend.close_all()

        ticks_with_belief = len(belief_errors)
        median_err = f"{np.median(belief_errors):.4f}" if belief_errors else "none"
        closest = f"{min(reaches):.4f}" if reaches else "none"
        print(
            f"{seed:>4} {result.status:>8} {result.success!s:>5} {result.steps:>6} "
            f"{ticks_with_belief:>13} {median_err:>11} {closest:>9} "
            f"{result.metric(MetricName.GRASP_SUCCESS):>9.0f} {object_xy[0]:>18}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
