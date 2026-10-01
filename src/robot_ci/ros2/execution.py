"""Run one ROBOT CI episode across a live ROS 2 graph, and record what ran.

The driver is deliberately thin. It resolves the scenario through the same
code path every other experiment uses, hands the *exact* validated scenario to
the nodes, calls the existing :func:`run_episode`, and then writes an artifact
describing the run. Everything that decides behaviour -- the policy, the task,
the metrics, the configuration hash -- is unchanged code.

Provenance recorded per run
---------------------------
``config_hash``, ``scenario_id``, ``seed``, ``policy``, ``backend``, ROS
distro, ROS package versions, MuJoCo version, Python version, git commit and a
timestamp. ROS-specific facts are stored **next to** the behavioural result, not
merged into it, so enabling ROS 2 cannot change the configuration hash.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from robot_ci.ros2.dependencies import Ros2Environment, detect

__all__ = [
    "ROS2_RESULTS",
    "build_scenario",
    "execution_metadata",
    "ros2_artifact",
    "run_episode_over_ros",
]

#: Where the ROS 2 execution artifacts live. Kept out of ``experiments/results``
#: proper because these are single-run execution records, not the research
#: results the report is generated from.
ROS2_RESULTS = Path("experiments/results/ros2")


def build_scenario(depth_noise_mm: float | None = None) -> Any:
    """Build the scenario for a ROS 2 run.

    The scenario is the project's own ``default_scenario`` with only the depth
    noise axis swept, resolved through the same ``set_by_path`` helper every
    other experiment uses. ``scenario_id`` is deliberately **not** rewritten: it
    seeds the scene's object placement, so renaming it would move the target and
    make the run incomparable with the direct one.

    Args:
        depth_noise_mm: Depth-camera sigma in millimetres; ``None`` leaves the
            scenario's own value.

    Returns:
        The validated scenario.
    """
    from robot_ci.scenarios import default_scenario, set_by_path

    scenario = default_scenario()
    if depth_noise_mm is not None:
        scenario = set_by_path(scenario, "sensors.depth.sigma_mm", float(depth_noise_mm))
    return scenario


def _git_commit() -> str:
    """Return the commit this ran at, or ``unknown`` outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _mujoco_version() -> str:
    """Return the MuJoCo version, or a marker when it is absent."""
    try:
        import mujoco

        return str(getattr(mujoco, "__version__", "unknown"))
    except Exception:  # pragma: no cover - mujoco is an optional import path
        return "absent"


def _ros_packages() -> dict[str, str]:
    """Return the ROS distribution's package versions for the artifact."""
    packages: dict[str, str] = {}
    try:
        completed = subprocess.run(
            ["ros2", "pkg", "list"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return packages
    if completed.returncode != 0:
        return packages
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            packages[parts[0]] = parts[1]
    return packages


def execution_metadata(
    *,
    scenario: Any,
    policy_name: str,
    seed: int,
    backend_name: str,
    environment: Ros2Environment,
) -> dict[str, Any]:
    """Assemble the provenance block for one run.

    Everything here is execution metadata. None of it feeds back into the
    scenario, so switching ROS 2 on cannot perturb the experiment it reports.
    """
    return {
        "scenario_id": scenario.scenario_id,
        "config_hash": scenario.config_hash(),
        "seed": int(seed),
        "policy": policy_name,
        "backend": backend_name,
        "mujoco_version": _mujoco_version(),
        "python_version": environment.python_version,
        "git_commit": _git_commit(),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ros2": environment.to_dict(),
        "ros_packages": _ros_packages(),
    }


def run_episode_over_ros(
    *,
    policy_name: str = "policy_v1",
    seed: int = 3,
    depth_noise_mm: float | None = None,
    backend_name: str = "mujoco",
    publish_images: bool = True,
    image_stride: int = 1,
    command_timeout_s: float = 30.0,
    node_name: str = "robot_ci_simulation",
) -> dict[str, Any]:
    """Run one episode with the policy in a separate ROS 2 process.

    The MuJoCo backend lives in this process, wrapped in
    :class:`~robot_ci.ros2.nodes.PublishingBackend`; the policy lives in the
    peer node reached over ``/robot_ci/observation`` and
    ``/robot_ci/command``. Every action applied to MuJoCo is the object that
    arrived on that topic.

    Args:
        policy_name: Factory key for the policy run in the peer node.
        seed: Episode seed.
        depth_noise_mm: Optional depth-camera sigma in millimetres.
        backend_name: Simulator backend; MuJoCo in practice.
        publish_images: Whether to publish camera frames.
        image_stride: Publish a frame every Nth control tick.
        command_timeout_s: Wall-clock liveness budget per command.
        node_name: Name for this process's node.

    Returns:
        A record combining provenance, the episode result and the published
        message counts.

    Raises:
        SimulationError: Propagated from the graph when the policy node does
            not answer.
    """
    from robot_ci.core.errors import SimulationError
    from robot_ci.experiments.runner import run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.ros2.nodes import PublishingBackend, Ros2Policy, SimulationNode
    from robot_ci.ros2.topics import TOPICS
    from robot_ci.simulation import create_backend

    scenario = build_scenario(depth_noise_mm)
    scenario_json = scenario.model_dump_json()
    environment = detect()

    simulation = SimulationNode(
        scenario_json=scenario_json,
        seed=seed,
        node_name=node_name,
        publish_images=publish_images,
        image_stride=image_stride,
        command_timeout_s=command_timeout_s,
    )
    inner = create_backend(backend_name)
    backend = PublishingBackend(
        inner,
        simulation,
        publish_images=publish_images,
        image_stride=image_stride,
    )
    policy = Ros2Policy(
        PolicyConfig(name=policy_name),
        scenario,
        simulation,
        command_timeout_s=command_timeout_s,
    )

    started = time.perf_counter()
    # Wait for the peer node's subscription to match before the first
    # observation. Without this the first observation is published into an
    # unmatched subscription and lost; `exchange` would then republish it, but
    # waiting first makes the common case a single publish per step.
    if not simulation.wait_for_peer(timeout_s=min(command_timeout_s, 20.0)):
        raise SimulationError(
            f"no subscriber on {TOPICS.observation} within "
            f"{min(command_timeout_s, 20.0):.1f}s; the policy node did not join "
            "the graph, so no command can ever arrive",
            code="ros2.no_peer",
            topic=TOPICS.observation,
        )
    try:
        result = run_episode(
            backend,
            policy,
            scenario,
            seed=seed,
            experiment_id="ros2",
        )
        # Published before teardown so a subscriber can observe the run ending,
        # and copied out before the node is destroyed.
        payload = result.to_dict()
        simulation.publish_result(payload)
        graph = {
            "messages_published": dict(simulation.published),
            "commands_received": simulation.commands_received,
            "commands_applied": policy.round_trips,
            # Surplus commands are the failure mode that corrupted an earlier
            # run: commands queued for observations that were never applied get
            # consumed on later ticks. Zero is the only correct value, so it is
            # recorded rather than assumed.
            "surplus_commands": max(0, simulation.commands_received - policy.round_trips),
            "lock_step": simulation.commands_received == policy.round_trips,
            "node_name": node_name,
        }
    finally:
        inner.close_all()
        simulation.destroy()
    wall_s = time.perf_counter() - started

    record = {
        "execution": execution_metadata(
            scenario=scenario,
            policy_name=policy_name,
            seed=seed,
            backend_name=backend.name,
            environment=environment,
        ),
        "result": payload,
        "graph": graph,
        "wall_clock_s": round(wall_s, 3),
        "scenario": scenario.model_dump(mode="json"),
    }
    return record


def ros2_artifact(record: dict[str, Any], out_dir: Path = ROS2_RESULTS) -> Path:
    """Write a run record to ``out_dir`` and return the path written.

    The filename is derived from the run's own facts, so a second run at a
    different noise level cannot overwrite the first.
    """
    execution = record["execution"]
    result = record["result"]
    sigma = record["scenario"]["sensors"]["depth"]["sigma_mm"]
    condition = "nominal" if float(sigma) == 0.0 else f"depth_noise_{sigma:g}mm"
    name = (
        f"{condition}_seed{execution['seed']}_"
        f"{execution['policy']}_{'success' if result['success'] else 'failure'}.json"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return target
