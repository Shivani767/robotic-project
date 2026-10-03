"""Why do the finger joints stop around 21 mm instead of closing on the cube?

Part 1B. The grasp blocker, investigated with instruments rather than with
guesses. Nothing here changes the robot: this script reads the generated SDF,
dumps the *actual* protobuf schemas of the messages it subscribes to, drives the
fingers through a sweep under several ablations, and writes what it measured to
``experiments/results/gazebo/finger_diagnostic/``.

Why it is written this way
--------------------------
Three earlier attempts to read Gazebo's contact message guessed the field names
(``collisions[0]``, ``wrenches``, ``wrench.force.normal``), each raised inside a
subscriber callback, and each produced a stream that reported *nothing* while
looking perfectly healthy. So this script:

* dumps the real descriptor of every message type it touches and writes it next
  to the results, so a wrong field name is visible rather than silent;
* gives every subscriber a ``health()`` that distinguishes "no messages" from
  "messages that raised" from "messages with zero contacts";
* changes one variable per run and records which one.

Run inside the container, one mode at a time::

    python3 scripts/gazebo_finger_diagnostic.py schema
    python3 scripts/gazebo_finger_diagnostic.py sdf
    python3 scripts/gazebo_finger_diagnostic.py sweep --no-object
    python3 scripts/gazebo_finger_diagnostic.py sweep --no-table
    python3 scripts/gazebo_finger_diagnostic.py sweep --no-finger-collision
    python3 scripts/gazebo_finger_diagnostic.py sweep --only left_finger
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from robot_ci.gazebo.live import BridgeHealth, JointStateReader
from robot_ci.gazebo.world import GAZEBO_ROBOT_NAME, GAZEBO_WORLD_NAME, world_sdf
from robot_ci.scenarios import default_scenario

RESULTS = Path("experiments/results/gazebo/finger_diagnostic")
URDF = "gazebo/models/panda/panda.urdf"
JOINT_STATE_TOPIC = f"/world/{GAZEBO_WORLD_NAME}/model/{GAZEBO_ROBOT_NAME}/joint_state"
OBJECT_TOPIC = (
    f"/world/{GAZEBO_WORLD_NAME}/model/target_cube/link/link/sensor/object_contacts/contact"
)

#: Apertures the jaw is swept through, in metres of commanded opening. The
#: mystery is a stall near 0.021 m, so the sweep straddles that densely.
SWEEP_APERTURES_M = (0.025, 0.022, 0.021, 0.020, 0.018, 0.016, 0.014, 0.012, 0.008, 0.0)

#: How long to hold each commanded aperture before reading it back.
SETTLE_S = 4.0

#: The closed pad separation this URDF implies, in metres: two finger joints
#: 11 mm apart, with 15 mm pads whose inner faces sit on the joint axes.
CLOSED_SEPARATION_M = 0.011

#: Width of the cube this scenario places. The jaw must open to at least this
#: before it can touch the object, and it cannot close below it.
OBJECT_WIDTH_M = 2.0 * float(default_scenario(seed=3).target.size_m)


def _describe(message_class: Any) -> dict[str, Any]:
    """Return a nested field description of a protobuf message class.

    Recursive on message-typed fields, so the whole tree a subscriber might
    reach is written down *before* anything reads it.
    """
    # A nested field hands back a MessageDescriptor, not a class, so the two are
    # normalised here rather than at every call site.
    descriptor = getattr(message_class, "DESCRIPTOR", message_class)
    out: dict[str, Any] = {}
    for field in descriptor.fields:
        entry: dict[str, Any] = {
            "type": field.type,
            "repeated": bool(field.label == field.LABEL_REPEATED),
        }
        if field.message_type is not None:
            entry["message"] = field.message_type.full_name
            entry["fields"] = _describe(field.message_type)
        out[field.name] = entry
    return out


def dump_schemas() -> dict[str, Any]:
    """Write the real schema of every message type this script subscribes to."""
    schemas: dict[str, Any] = {}
    try:
        from gz.msgs10.contacts_pb2 import Contacts

        schemas["Contacts"] = _describe(Contacts)
    except ImportError as exc:
        schemas["Contacts"] = {"unavailable": str(exc)}
    for module, name in (("wrench_pb2", "Wrench"), ("pose_v_pb2", "Pose_V")):
        try:
            cls = getattr(__import__(f"gz.msgs10.{module}", fromlist=[name]), name)
            schemas[name] = _describe(cls)
        except Exception as exc:
            schemas[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "message_schemas.json").write_text(
        json.dumps(schemas, indent=2, sort_keys=True), encoding="utf-8"
    )
    return schemas


def _attr(element: Any, key: str) -> Any:
    """Attribute of an optional SDF element, or ``None``."""
    return None if element is None else element.get(key)


def dump_sdf() -> dict[str, Any]:
    """Write the SDF Gazebo actually loads, plus the parsed finger properties.

    The source URDF is not the evidence here: ``gz sdf -p`` rewrites the joint
    frames, lumps the fixed hand joint and can change limits, so what Gazebo
    loads is what gets recorded.
    """
    sdf = world_sdf(default_scenario(seed=3), robot_urdf=URDF)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_sdf.txt").write_text(sdf, encoding="utf-8")

    root = ElementTree.fromstring(sdf)
    model = next(m for m in root.iter("model") if m.get("name") == GAZEBO_ROBOT_NAME)
    joints: dict[str, Any] = {}
    for joint in model.findall("joint"):
        name = joint.get("name") or ""
        if "finger" not in name:
            continue
        # In the converted SDF, axis, limit and dynamics are NESTED INSIDE
        # <axis>, not siblings of it. A parser that looked for siblings found
        # none of them and would have reported the finger joints as having no
        # axis and no limits at all, which is a conclusion about the robot that
        # the SDF plainly contradicts.
        axis = joint.find("axis")
        limit = axis.find("limit") if axis is not None else None
        dynamics = axis.find("dynamics") if axis is not None else None
        pose = joint.find("pose")
        parent = joint.find("parent")
        child = joint.find("child")
        joints[name] = {
            "type": joint.get("type"),
            "parent": None if parent is None else parent.text,
            "child": None if child is None else child.text,
            "pose": None if pose is None else pose.text,
            "pose_relative_to": _attr(pose, "relative_to"),
            # <axis><xyz>, <limit><lower>, <limit><effort> and friends are
            # child ELEMENTS holding text, not attributes. Reading them as
            # attributes returns None for a joint that has every one of them
            # present and correct, which is how the first pass of this report
            # concluded the fingers had no axis and no limits.
            "axis": None if axis is None else axis.findtext("xyz"),
            "lower": None if limit is None else limit.findtext("lower"),
            "upper": None if limit is None else limit.findtext("upper"),
            "effort": None if limit is None else limit.findtext("effort"),
            "velocity": None if limit is None else limit.findtext("velocity"),
            "stiffness": None if limit is None else limit.findtext("stiffness"),
            "damping": None if limit is None else limit.findtext("damping"),
            "friction": None if limit is None else limit.findtext("friction"),
            "dynamics_damping": None if dynamics is None else dynamics.findtext("damping"),
            "dynamics_friction": None if dynamics is None else dynamics.findtext("friction"),
        }
    collisions: dict[str, Any] = {}
    for collision in model.iter("collision"):
        box = collision.find("geometry/box")
        cylinder = collision.find("geometry/cylinder")
        pose = collision.find("pose")
        collisions[collision.get("name", "")] = {
            "pose": None if pose is None else pose.text,
            "pose_relative_to": _attr(pose, "relative_to"),
            "box_size": None if box is None else box.get("size"),
            "cylinder": None
            if cylinder is None
            else {"length": cylinder.findtext("length"), "radius": cylinder.findtext("radius")},
            "surface_friction": collision.findtext("surface/friction/ode/mu"),
        }
    payload = {
        "object_width_m": OBJECT_WIDTH_M,
        "closed_separation_m": CLOSED_SEPARATION_M,
        "finger_joints": joints,
        "collisions": collisions,
    }
    (RESULTS / "joint_limits.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def ablate(
    sdf: str,
    *,
    no_object: bool,
    no_table: bool,
    no_finger_collision: bool,
    finger_velocity_limit: float | None = None,
) -> str:
    """Return the world with the requested parts removed, one variable at a time.

    ``finger_velocity_limit`` overrides the converted joints' ``<velocity>`` on
    the two finger joints only, and nothing else. It exists to test whether the
    engine's joint velocity limit is what stops the jaw during a fast close.
    """
    root = ElementTree.fromstring(sdf)
    if finger_velocity_limit is not None:
        for joint in root.iter("joint"):
            if "finger" not in (joint.get("name") or ""):
                continue
            limit = joint.find("axis/limit")
            if limit is not None:
                limit.find("velocity").text = repr(float(finger_velocity_limit))
    world = root.find("world")
    assert world is not None
    for model in list(world.findall("model")):
        name = model.get("name")
        if (name == "target_cube" and no_object) or (name == "table" and no_table):
            world.remove(model)
    if no_finger_collision:
        robot = next(m for m in world.findall("model") if m.get("name") == GAZEBO_ROBOT_NAME)
        for link in robot.iter("link"):
            if "finger" not in (link.get("name") or ""):
                continue
            for collision in list(link.findall("collision")):
                link.remove(collision)
    return ElementTree.tostring(root, encoding="unicode")


def _wait_for_topic(process: Any, topic: str, timeout_s: float = 180.0) -> bool:
    """Block until Gazebo publishes ``topic``, or the child dies."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            listing = subprocess.run(
                ["gz", "topic", "-l"], capture_output=True, text=True, timeout=15
            ).stdout.split()
        except (OSError, subprocess.SubprocessError):
            time.sleep(1.0)
            continue
        if topic in listing:
            return True
        time.sleep(1.0)
    return False


def sweep(
    *,
    label: str,
    no_object: bool,
    no_table: bool,
    no_finger_collision: bool,
    only: str | None,
    finger_velocity_limit: float | None = None,
) -> dict[str, Any]:
    """Command the jaw through a sweep and record what it actually did.

    Runs against the real Gazebo process, publishing straight onto the same
    ``cmd_pos`` topics the backend uses, so nothing about the control path
    differs from an episode except that no policy is driving it.
    """
    from gz.msgs10.double_pb2 import Double
    from gz.transport13 import Node

    sdf = ablate(
        world_sdf(default_scenario(seed=3), robot_urdf=URDF),
        no_object=no_object,
        no_table=no_table,
        no_finger_collision=no_finger_collision,
        finger_velocity_limit=finger_velocity_limit,
    )
    world_file = Path(f"/tmp/finger_diag_{label}.sdf")
    world_file.write_text(sdf, encoding="utf-8")

    process = subprocess.Popen(
        ["gz", "sim", "-s", "-r", str(world_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    health = BridgeHealth()
    reader = JointStateReader(JOINT_STATE_TOPIC, health)
    publishers: dict[str, Any] = {}
    node = Node()
    for joint in ("left_finger", "right_finger"):
        publishers[joint] = node.advertise(
            f"/model/{GAZEBO_ROBOT_NAME}/joint/{joint}/0/cmd_pos", Double
        )
    rows: list[dict[str, Any]] = []
    try:
        if not _wait_for_topic(process, JOINT_STATE_TOPIC):
            return {
                "label": label,
                "error": "joint_state never appeared",
                "ablation": {
                    "no_object": no_object,
                    "no_table": no_table,
                    "no_finger_collision": no_finger_collision,
                    "only": only,
                },
            }
        reader.start()
        time.sleep(2.0)

        # Open first, then walk the sweep down. Starting closed would test a
        # different question: whether the pads can *open* from a closed pose.
        for aperture in (max(SWEEP_APERTURES_M), *SWEEP_APERTURES_M):
            commanded = aperture / 2.0
            for joint in publishers:
                if only is not None and joint != only:
                    continue
                message = Double()
                message.data = float(commanded)
                publishers[joint].publish(message)
            deadline = time.monotonic() + SETTLE_S
            while time.monotonic() < deadline:
                time.sleep(0.05)
            positions, sim_time = reader.latest()
            left = positions.get("left_finger")
            right = positions.get("right_finger")
            travelled = None
            if left is not None and right is not None:
                travelled = left + right
            rows.append(
                {
                    "commanded_aperture_m": round(aperture, 5),
                    "commanded_per_finger_m": round(commanded, 5),
                    "left_m": None if left is None else round(left, 5),
                    "right_m": None if right is None else round(right, 5),
                    "measured_aperture_m": None if travelled is None else round(travelled, 5),
                    "pad_separation_m": None
                    if travelled is None
                    else round(CLOSED_SEPARATION_M + travelled, 5),
                    "error_m": None if travelled is None else round(aperture - travelled, 5),
                    "sim_time_s": round(sim_time, 3),
                }
            )
        payload = {
            "label": label,
            "ablation": {
                "no_object": no_object,
                "no_table": no_table,
                "no_finger_collision": no_finger_collision,
                "only": only,
            },
            "object_width_m": OBJECT_WIDTH_M,
            "closed_separation_m": CLOSED_SEPARATION_M,
            "settle_s": SETTLE_S,
            "rows": rows,
            "reader": {
                "messages": health.joint_messages,
                "joint_stream": health.joint_stream,
                "errors": list(health.errors),
            },
        }
    finally:
        reader.stop()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"finger_sweep_{label}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def hold_trace(
    *,
    label: str,
    aperture: float,
    seconds: float,
    no_object: bool,
    no_table: bool,
    step_from: float | None = None,
    step_settle_s: float = 2.0,
    rate_hz: float = 20.0,
    finger_velocity_limit: float | None = None,
) -> dict[str, Any]:
    """Command one aperture and watch the fingers for a long time.

    The sweep walks the command down and reports where the jaw happens to be
    after a fixed dwell, which cannot distinguish "the controller is slow" from
    "something is blocking it": both look like a stall. A single command held for
    many seconds does distinguish them. A blocked finger traces a flat line; a
    slow one decays monotonically toward the target.
    """
    from gz.msgs10.double_pb2 import Double
    from gz.transport13 import Node

    sdf = ablate(
        world_sdf(default_scenario(seed=3), robot_urdf=URDF),
        no_object=no_object,
        no_table=no_table,
        no_finger_collision=False,
        finger_velocity_limit=finger_velocity_limit,
    )
    world_file = Path(f"/tmp/finger_hold_{label}.sdf")
    world_file.write_text(sdf, encoding="utf-8")
    process = subprocess.Popen(
        ["gz", "sim", "-s", "-r", str(world_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    health = BridgeHealth()
    reader = JointStateReader(JOINT_STATE_TOPIC, health)
    node = Node()
    publishers = {
        joint: node.advertise(
            f"/model/{GAZEBO_ROBOT_NAME}/joint/{joint}/0/cmd_pos", Double
        )
        for joint in ("left_finger", "right_finger")
    }
    samples: list[dict[str, Any]] = []
    try:
        if not _wait_for_topic(process, JOINT_STATE_TOPIC):
            return {"label": label, "error": "joint_state never appeared"}
        reader.start()
        time.sleep(2.0)
        # Optional opening step first, so the closing transient can be watched
        # from a known starting pose. The policy's own sequence is exactly this:
        # wide open while descending, then a single step to the grasp aperture.
        if step_from is not None:
            for publisher in publishers.values():
                message = Double()
                message.data = float(step_from / 2.0)
                publisher.publish(message)
            time.sleep(step_settle_s)
        step_time = time.monotonic()
        for publisher in publishers.values():
            message = Double()
            message.data = float(aperture / 2.0)
            publisher.publish(message)
        deadline = time.monotonic() + seconds
        period = 1.0 / rate_hz
        while time.monotonic() < deadline:
            time.sleep(min(1.0, period)) if rate_hz <= 1.0 else time.sleep(period)
            positions, sim_time = reader.latest()
            left = positions.get("left_finger")
            right = positions.get("right_finger")
            samples.append(
                {
                    "t_s": round(time.monotonic() - step_time, 3),
                    "sim_time_s": round(sim_time, 2),
                    "left_m": None if left is None else round(left, 5),
                    "right_m": None if right is None else round(right, 5),
                    "pad_separation_m": None
                    if left is None or right is None
                    else round(CLOSED_SEPARATION_M + left + right, 5),
                }
            )
        payload = {
            "label": label,
            "commanded_aperture_m": aperture,
            "commanded_per_finger_m": round(aperture / 2.0, 5),
            "stepped_from_aperture_m": step_from,
            "rate_hz": rate_hz,
            "hold_s": seconds,
            "ablation": {
                "no_object": no_object,
                "no_table": no_table,
                "finger_velocity_limit": finger_velocity_limit,
            },
            "samples": samples,
            "reader": {"messages": health.joint_messages, "errors": list(health.errors)},
        }
    finally:
        reader.stop()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"finger_hold_{label}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> int:
    """Dispatch one diagnostic mode and print what it measured."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("schema", "sdf", "sweep", "hold"))
    parser.add_argument("--aperture", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--step-from", type=float, default=None)
    parser.add_argument("--finger-velocity-limit", type=float, default=None)
    parser.add_argument("--no-object", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--no-finger-collision", action="store_true")
    parser.add_argument("--only", choices=("left_finger", "right_finger"), default=None)
    args = parser.parse_args()

    if args.mode == "schema":
        schemas = dump_schemas()
        for name, body in schemas.items():
            fields = sorted(body) if isinstance(body, dict) else body
            print(f"{name}: {fields if not isinstance(body, dict) else sorted(body)}")
        print(f"wrote {RESULTS / 'message_schemas.json'}")
        return 0
    if args.mode == "sdf":
        payload = dump_sdf()
        for name, joint in payload["finger_joints"].items():
            print(f"{name}: {json.dumps(joint, sort_keys=True)}")
        print(f"wrote {RESULTS / 'model_sdf.txt'} and {RESULTS / 'joint_limits.json'}")
        return 0

    if args.mode == "hold":
        label = "no_object" if args.no_object else "baseline"
    if args.finger_velocity_limit is not None:
        label = f"{label}_vel{args.finger_velocity_limit:g}"
        result = hold_trace(
            label=label,
            aperture=args.aperture,
            seconds=args.seconds,
            no_object=args.no_object,
            no_table=args.no_table,
            step_from=args.step_from,
            finger_velocity_limit=args.finger_velocity_limit,
        )
        if "error" in result:
            print(f"hold: ERROR {result['error']}")
            return 1
        print(f"hold: commanded {result['commanded_per_finger_m']} per finger")
        for sample in result["samples"]:
            print(
                f"  t={sample['t_s']:>5}s L {sample['left_m']} R {sample['right_m']} "
                f"sep {sample['pad_separation_m']}"
            )
        print(f"wrote {RESULTS / ('finger_hold_' + label + '.json')}")
        return 0

    label = "baseline"
    if args.no_object:
        label = "no_object"
    elif args.no_table:
        label = "no_table"
    elif args.no_finger_collision:
        label = "no_finger_collision"
    elif args.only:
        label = f"only_{args.only}"
    if args.finger_velocity_limit is not None:
        label = f"{label}_vel{args.finger_velocity_limit:g}"
    result = sweep(
        label=label,
        no_object=args.no_object,
        no_table=args.no_table,
        no_finger_collision=args.no_finger_collision,
        only=args.only,
        finger_velocity_limit=args.finger_velocity_limit,
    )
    if "error" in result:
        print(f"{label}: ERROR {result['error']}")
        return 1
    print(f"{label}: reader={result['reader']}")
    for row in result["rows"]:
        print(
            f"  cmd {row['commanded_aperture_m']:.4f} -> sep "
            f"{row['pad_separation_m']}  (L {row['left_m']}, R {row['right_m']}, "
            f"err {row['error_m']})"
        )
    print(f"wrote {RESULTS / ('finger_sweep_' + label + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
