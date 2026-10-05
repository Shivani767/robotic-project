"""Part 1D: does the articulated arm chain above the finger reproduce the stall?

Part 1C narrowed the search to one structural difference. Every minimal finger
model traverses the full 0-40 mm stroke -- with the Panda's own gains, its own
joint and child-link poses, its parent's real collision, and nine controller
plugins on one model. The full Panda stalls. What no reproduction has yet included
is that ``panda_link7`` is not fixed: the model is anchored to the world at
``panda_link0`` and hangs through seven revolute joints, each with its own
position controller.

This module builds that chain and drives the finger across its range while
recording the finger's command, position, velocity and joint reaction together
with all seven revolute joints' positions and velocities.

Design rules, from the brief:

  * diagnosis only -- nothing in the Panda is modified to make anything work;
  * one structural variable per variant, named explicitly on the CLI;
  * no guessed protobuf. Joint state is parsed from ``gz topic -e`` text, and the
    one message read through the transport API (``Wrench``) is the same import the
    already-working finger diagnostic uses. Everything optional fails soft.

Variants, per sections 3 and 11:

  A     arm present but rigid (seven revolute joints converted to fixed)
  B     dynamic arm, no arm controllers
  C     dynamic arm, seven controllers with one uniform gain set
  D     dynamic arm, the Panda's exact per-joint gains
  D2    D with every arm controller removed
  D3    D with gravity off

``--disable-arm-controller joint3`` ablates exactly one, per section 6.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

MODEL_SDF = Path("experiments/results/gazebo/finger_diagnostic/model_sdf.txt")
WORLD = "armchain_world"
MODEL = "panda"
FINGER = "left_finger"
ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")

# gazebo/world.py, verbatim, so the "exact" variants really are exact.
PANDA_GAINS: dict[str, tuple[float, ...]] = {
    "joint1": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
    "joint2": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
    "joint3": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
    "joint4": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
    "joint5": (4000.0, 100.0, 100.0, 10.0, -10.0, 12.0),
    "joint6": (4000.0, 100.0, 100.0, 10.0, -10.0, 12.0),
    "joint7": (4000.0, 100.0, 100.0, 10.0, -10.0, 12.0),
    FINGER: (1000.0, 100.0, 50.0, 4.0, -4.0, 100.0),
}
GENERIC_ARM_GAIN = (1000.0, 0.0, 50.0, 10.0, -10.0, 87.0)

COMMAND_Q = (0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040)
SETTLE_S = 1.5


def _text(parent: ET.Element, tag: str, value: str) -> ET.Element:
    """Element carrying its value as text. This build reads SDF that way."""
    node = ET.SubElement(parent, tag)
    node.text = value
    return node


def load_chain() -> ET.Element:
    """The converted Panda model, exactly as Gazebo's own URDF->SDF produced it.

    Read from model_sdf.txt rather than reconstructed, so variant D is the real
    thing and not a re-derivation that could quietly differ.
    """
    for model in ET.parse(MODEL_SDF).getroot().iter("model"):
        if model.get("name") == "panda":
            return copy.deepcopy(model)
    raise RuntimeError(f"no panda model in {MODEL_SDF}")


def _make_rigid(model: ET.Element) -> None:
    """Variant A: keep every link and the whole geometry, make the arm rigid."""
    for joint in model.findall("joint"):
        if joint.get("name") in ARM:
            joint.set("type", "fixed")
            axis = joint.find("axis")
            if axis is not None:
                joint.remove(axis)


def _strip_collision(model: ET.Element) -> None:
    """D keeps geometry but no gripper collision, per the brief."""
    for link in model.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)


def _controller(model: ET.Element, joint: str, gains: tuple[float, ...]) -> None:
    p_gain, i_gain, d_gain, i_max, i_min, cmd_max = gains
    plugin = ET.SubElement(
        model,
        "plugin",
        {
            "filename": "gz-sim-joint-position-controller-system",
            "name": "gz::sim::systems::JointPositionController",
        },
    )
    for key, value in (
        ("joint_name", joint),
        ("p_gain", f"{p_gain:g}"),
        ("i_gain", f"{i_gain:g}"),
        ("d_gain", f"{d_gain:g}"),
        ("i_max", f"{i_max:g}"),
        ("i_min", f"{i_min:g}"),
        ("cmd_max", f"{cmd_max:g}"),
        ("cmd_min", f"{-cmd_max:g}"),
    ):
        _text(plugin, key, value)


def _add_ft_sensor(model: ET.Element) -> None:
    """A reaction sensor on the finger JOINT, named as Gazebo requires."""
    for joint in model.findall("joint"):
        if joint.get("name") != FINGER:
            continue
        sensor = ET.SubElement(joint, "sensor", {"name": "finger_ft", "type": "force_torque"})
        _text(sensor, "always_on", "1")
        _text(sensor, "update_rate", "100")
        _text(sensor, "topic", "/sensor/finger_ft/forcetorque")
        _text(sensor, "frame", "child")
        _text(sensor, "measure_direction", "child_to_parent")


def build(
    *,
    arm_mode: str = "dynamic",
    arm_controllers: str = "panda",
    gravity: bool = True,
    disable_arm_controller: str | None = None,
    gripper_collision: bool = False,
    use_project_world: bool = False,
) -> str:
    """Render one variant's world. Each argument is exactly one variable.

    ``use_project_world`` swaps in the project's own ``world_sdf`` -- the real
    placement, the real table, the real plugin set -- with the cube removed. That
    is the one configuration matching the failing episode except for the object,
    so it tests the table directly instead of approximating it.
    """
    if use_project_world:
        from robot_ci.gazebo.world import world_sdf
        from robot_ci.scenarios import default_scenario

        return world_sdf(
            default_scenario(), include_object=False, include_table=True, finger_collisions=True
        )

    model = load_chain()
    # model_sdf.txt already carries the world anchor and all nine controllers.
    # Strip every model-level plugin first, then add back exactly what this
    # variant specifies -- otherwise "no arm controllers" would silently keep the
    # originals and the ablation would measure nothing.
    for plugin in model.findall("plugin"):
        model.remove(plugin)
    if arm_mode == "rigid":
        _make_rigid(model)
    if not gripper_collision:
        _strip_collision(model)
    _add_ft_sensor(model)

    # The finger always keeps its controller; only the ARM's is ablated.
    _controller(model, FINGER, PANDA_GAINS[FINGER])
    if arm_controllers != "none":
        exact = arm_controllers == "panda"
        for joint in ARM:
            if disable_arm_controller == joint:
                continue
            _controller(model, joint, PANDA_GAINS[joint] if exact else GENERIC_ARM_GAIN)

    if not any(j.get("name") == "world_to_panda_base" for j in model.findall("joint")):
        anchor = ET.Element("joint", {"name": "world_to_panda_base", "type": "fixed"})
        _text(anchor, "parent", "world")
        _text(anchor, "child", "panda_link0")
        model.insert(0, anchor)
    # State feedback goes on the MODEL, as in gazebo/world.py; putting it in the
    # world publishes nothing at all.
    ET.SubElement(
        model,
        "plugin",
        {"filename": "gz-sim-joint-state-publisher-system", "name": "gz::sim::systems::JointStatePublisher"},
    )

    sdf = ET.Element("sdf", {"version": "1.11"})
    world = ET.SubElement(sdf, "world", {"name": WORLD})
    ET.SubElement(
        world,
        "plugin",
        {"filename": "gz-sim-physics-system", "name": "gz::sim::systems::Physics"},
    )
    _text(world, "gravity", "0 0 -9.8" if gravity else "0 0 0")
    for filename, name in (
        ("gz-sim-forcetorque-system", "gz::sim::systems::ForceTorque"),
        ("gz-sim-dynamic-pose-info-system", "gz::sim::systems::DynamicPoseInfo"),
    ):
        ET.SubElement(world, "plugin", {"filename": filename, "name": name})
    world.append(model)
    ET.indent(sdf, space="  ")
    return ET.tostring(sdf, encoding="unicode") + "\n"


def _topics() -> list[str]:
    """Advertised topics, read from the CLI rather than assumed."""
    out = subprocess.run(
        ["gz", "topic", "-l"], capture_output=True, text=True, timeout=60, check=False
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _echo(topic: str, timeout: int = 6) -> str:
    """One message as text. Parsing text avoids guessing any message class."""
    try:
        out = subprocess.run(
            ["gz", "topic", "-e", "-t", topic, "-n", "1"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ""
    return out.stdout


def parse_joint_state(text: str) -> dict[str, dict[str, float]]:
    """Parse every ``name:`` block's position and velocity.

    The text form interleaves the fields, so this walks lines and attaches each
    scalar to the most recently named joint. Tolerates a missing velocity.
    """
    out: dict[str, dict[str, float]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("name:"):
            current = line.split(":", 1)[1].strip().strip('"')
            out.setdefault(current, {})
        elif current is not None and line.startswith(("position:", "velocity:")):
            key, _, value = line.partition(":")
            with contextlib.suppress(ValueError):
                out[current][key] = float(value.strip())
    return out


def run(**kwargs: Any) -> dict[str, Any]:
    """Drive the finger across its range and record the whole chain's response.

    Every reading is optional to the run: a missing topic yields nulls and a
    recorded note, never an exception that takes the experiment down.
    """
    label = kwargs.pop("label", "run")
    sdf_path = Path(f"/tmp/armchain_{label}.sdf")
    sdf_text = build(**kwargs)
    sdf_path.write_text(sdf_text)
    # Read the world name out of the SDF. It is not always WORLD: the project
    # world is robot_ci_pick_place, and assuming otherwise silently produced a
    # run with no joint_state topic and every reading null.
    world = ET.fromstring(sdf_text).find("world")
    world_name = world.get("name") if world is not None else WORLD
    sim_log = Path(f"/tmp/armchain_{label}_sim.log").open("w+")  # noqa: SIM115 - outlives try
    # The project world pulls the robot in with <include><uri>panda</uri>, which
    # resolves against GZ_SIM_RESOURCE_PATH. Without it the world fails with
    # "Unable to find uri[panda]" and advertises nothing, which is why the first
    # two project-world runs produced no joint_state at any startup delay.
    env = dict(os.environ)
    env["GZ_SIM_RESOURCE_PATH"] = str(Path("gazebo/models").resolve())
    process = subprocess.Popen(
        # -s: server only, or it opens a GUI and dies advertising nothing.
        ["gz", "sim", "-r", "-s", "-v", "4", str(sdf_path)],
        stdout=sim_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    result: dict[str, Any] = {
        "label": label,
        "config": dict(kwargs),
        "sdf_path": str(sdf_path),
        "world_name": world_name,
    }
    try:
        # Poll for the state topic instead of sleeping a guessed interval. The
        # project world loads the Sensors and Contact systems on top of the
        # minimal world's plugin set and was still not ready after a fixed 14 s,
        # which produced a run with no joint_state and every reading null.
        state_topic = f"/world/{world_name}/model/{MODEL}/joint_state"
        deadline = time.monotonic() + 60.0
        topics: list[str] = []
        while time.monotonic() < deadline:
            topics = _topics()
            if state_topic in topics:
                break
            time.sleep(5.0)
        result["startup_wait_s"] = round(60.0 - max(0.0, deadline - time.monotonic()), 1)
        ft_topic = f"/world/{world_name}/model/{MODEL}/joint/{FINGER}/sensor/finger_ft/forcetorque"
        link7_topic = f"/world/{world_name}/model/{MODEL}/link/panda_link7/pose"
        result["topics_present"] = {
            "joint_state": state_topic in topics,
            "finger_ft": ft_topic in topics,
            "link7_pose": link7_topic in topics,
        }
        result["joints_reported"] = sorted(parse_joint_state(_echo(state_topic)).keys())

        from gz.msgs10.double_pb2 import Double
        from gz.msgs10.wrench_pb2 import Wrench
        from gz.transport13 import Node

        node = Node()
        force = {"x": 0.0, "y": 0.0, "z": 0.0, "messages": 0}

        def _on_wrench(msg: Any) -> None:
            force["x"] = round(float(msg.force.x), 4)
            force["y"] = round(float(msg.force.y), 4)
            force["z"] = round(float(msg.force.z), 4)
            force["messages"] = int(force["messages"]) + 1

        if result["topics_present"]["finger_ft"]:
            node.subscribe(Wrench, ft_topic, _on_wrench)
        publisher = node.advertise(f"/model/{MODEL}/joint/{FINGER}/0/cmd_pos", Double)
        time.sleep(1.0)

        rows: list[dict[str, Any]] = []
        for q in COMMAND_Q:
            message = Double()
            message.data = float(q)
            publisher.publish(message)
            time.sleep(SETTLE_S)
            state = parse_joint_state(_echo(state_topic))
            finger = state.get(FINGER, {})
            row: dict[str, Any] = {
                "commanded_q_m": q,
                "measured_q_m": finger.get("position"),
                "qdot_m_s": finger.get("velocity"),
                "error_m": (
                    None if finger.get("position") is None else round(finger["position"] - q, 6)
                ),
                "reaction_N": {k: force[k] for k in ("x", "y", "z")},
                "reaction_messages": force["messages"],
                "arm": {
                    j: {
                        "position": state.get(j, {}).get("position"),
                        "velocity": state.get(j, {}).get("velocity"),
                        "drift": (
                            None
                            if j not in state or not rows
                            or rows[0]["arm"].get(j, {}).get("position") is None
                            else round(state[j]["position"] - rows[0]["arm"][j]["position"], 9)
                        ),
                    }
                    for j in ARM
                },
            }
            if result["topics_present"]["link7_pose"]:
                row["link7_pose_text"] = _echo(link7_topic, timeout=4)[:400]
            rows.append(row)
        result["rows"] = rows

        measured = [r["measured_q_m"] for r in rows if r["measured_q_m"] is not None]
        result["max_q_m"] = max(measured) if measured else None
        result["full_stroke"] = bool(measured and max(measured) > 0.03)
        result["final_error_m"] = rows[-1].get("error_m") if rows else None
        result["final_reaction_N"] = rows[-1].get("reaction_N") if rows else None
        worst = 0.0
        for r in rows:
            for j in ARM:
                v = r["arm"].get(j, {}).get("drift")
                if v is not None:
                    worst = max(worst, abs(v))
        result["max_arm_drift_rad"] = round(worst, 9)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        sim_log.seek(0)
        result["sim_log_tail"] = sim_log.read()[-2500:]
        sim_log.close()
    return result


VARIANTS: dict[str, dict[str, Any]] = {
    "A": {"arm_mode": "rigid", "arm_controllers": "none"},
    "B": {"arm_mode": "dynamic", "arm_controllers": "none"},
    "C": {"arm_mode": "dynamic", "arm_controllers": "generic"},
    "D": {"arm_mode": "dynamic", "arm_controllers": "panda"},
    "D2": {"arm_mode": "dynamic", "arm_controllers": "none"},
    "D3": {"arm_mode": "dynamic", "arm_controllers": "panda", "gravity": False},
}


def main() -> int:
    """Run one named variant, or render its SDF with --emit-sdf."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="D", choices=sorted(VARIANTS))
    parser.add_argument("--disable-arm-controller", default=None, choices=list(ARM))
    parser.add_argument(
        "--project-world",
        action="store_true",
        help="use the project's own world_sdf (real placement and table), cube removed",
    )
    parser.add_argument(
        "--gripper-collision",
        action="store_true",
        help="keep the gripper link collisions that variant D strips",
    )
    parser.add_argument("--emit-sdf")
    parser.add_argument("--result")
    args = parser.parse_args()

    config: dict[str, Any] = dict(VARIANTS[args.variant])
    if args.disable_arm_controller:
        config["disable_arm_controller"] = args.disable_arm_controller
    if args.gripper_collision:
        config["gripper_collision"] = True
    if args.project_world:
        config["use_project_world"] = True
    name = args.variant + (
        f"_no_{args.disable_arm_controller}" if args.disable_arm_controller else ""
    )
    if args.gripper_collision:
        name += "_coll"
    if args.project_world:
        name += "_projworld"

    if args.emit_sdf:
        Path(args.emit_sdf).write_text(build(**config))
        print(f"wrote {args.emit_sdf}")
        return 0

    payload = run(label=name, **config)
    out = Path(args.result or f"experiments/results/gazebo/arm_chain/armchain_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    shown = {k: v for k, v in payload.items() if k not in ("rows", "topics", "sim_log_tail")}
    print(json.dumps(shown, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
