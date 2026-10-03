r"""Run one real Gazebo episode and print exactly what happened.

An evidence tool, not part of the package: it exists so that "the Gazebo
backend runs PickAndPlace" is a claim backed by a transcript from this machine
rather than by unit tests. It probes the observation stream first (camera
frame, object belief, joint state) and then runs a full episode through the
same ``run_episode`` loop the research engine uses.

Run it inside the container::

    docker run --rm -v "$PWD":/workspace/robot-ci -w /workspace/robot-ci \\
        robot-ci-gazebo-demo:latest \\
        bash -c 'source /opt/ros/jazzy/setup.bash && python3 scripts/gazebo_smoke_episode.py'

Exit code 0 means the episode *finished* (success or failure are both valid
outcomes); nonzero means the run itself broke.
"""

from __future__ import annotations

import time


def main() -> int:
    """Probe the streams, run one episode, and print the measured outcome."""
    from robot_ci.experiments.runner import run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario
    from robot_ci.simulation import create_backend

    scenario = default_scenario(seed=42)
    backend = create_backend("gazebo")

    # -- observation probe --------------------------------------------------
    handle = backend.reset(scenario, seed=42)
    frame = None
    observation = None
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        observation = backend.observe(handle)
        frame = observation.sensor_frame
        if frame is not None and frame.valid and frame.rgb is not None:
            break
        time.sleep(0.5)
    if observation is None or frame is None:  # pragma: no cover - defensive
        print("probe: observe() never returned")
        backend.close_all()
        return 2
    rgb_shape = None if frame.rgb is None else tuple(frame.rgb.shape)
    print(
        f"probe: frame valid={frame.valid} rgb_shape={rgb_shape} w={frame.width_px} h={frame.height_px}"
    )
    print(
        "probe: believed objects:",
        [
            (
                o.id,
                round(float(o.confidence), 3),
                [round(float(v), 3) for v in o.pose.position.array],
            )
            for o in observation.objects
        ],
    )
    truth = backend.ground_truth(handle)
    print(
        "probe: ground truth:",
        [
            (o.id, [round(float(v), 3) for v in o.pose.position.array], o.grasped)
            for o in truth.objects
        ],
    )
    print("probe: joints:", [round(float(v), 4) for v in observation.robot.joint_positions])
    print(
        "probe: ee:",
        [round(float(v), 4) for v in observation.robot.end_effector_pose.position.array],
    )
    backend.close(handle)

    # -- full episode -------------------------------------------------------
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    started = time.time()
    try:
        result = run_episode(backend, policy, scenario, seed=42, experiment_id="gazebo_smoke")
    except Exception as exc:
        print(f"episode: RAISED {type(exc).__name__}: {exc}")
        backend.close_all()
        return 1
    wall = time.time() - started
    print(
        f"episode: status={result.status} success={result.success} steps={result.steps} wall={wall:.1f}s"
    )
    for metric in result.metrics:
        print(f"  metric {metric.name.value} = {metric.value:g} {metric.unit}")
    backend.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
