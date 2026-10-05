"""Minimal prismatic-joint reproduction, per Part 1C section 6.

Nothing from the Panda survives here: no arm, no table, no cube, no camera, no
contact sensor, no gripper, one parent link, one prismatic joint, one child link.
If the joint stalls here the bug is in how Gazebo configures a prismatic joint;
if it does not, the bug is in whatever the Panda conversion added around it.

The variants cover section 7:

  base    collision on, ordinary inertia, damping 1, friction 0  (the URDF values)
  visual  no collision at all
  simple  a deliberately plain inertia box
  free    friction and damping driven to zero

The finger's mass, inertia, collision box and joint limits are taken from
gazebo/models/panda/panda.urdf rather than invented, so "minimal" means minimal
in model topology and identical in the numbers that matter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree as ET

FINGER_MASS = "0.015"
FINGER_INERTIA = {
    "ixx": "1.559e-06",
    "ixy": "0",
    "ixz": "0",
    "iyy": "1.336e-06",
    "iyz": "0",
    "izz": "9.543e-07",
}
PAD_SIZE = ("0.02", "0.015", "0.013")
PAD_ORIGIN = ("0", "0.0075", "0.0584")
JOINT_ORIGIN = ("0", "0.0055", "0.0445")
LOWER, UPPER = "0.0", "0.04"
EFFORT, VELOCITY = "100", "0.2"
DAMPING, FRICTION = "1.0", "0.0"


def _element(parent: ET.Element, tag: str, attrs: dict[str, str] | None = None) -> ET.Element:
    """Build one element with attributes.

    Attributes go in a dict, not ``**kwargs``: SDF spells several of them with
    underscores (``joint_name``, ``p_gain``) and one without (``xyz``), and
    kwargs mangle both.
    """
    return ET.SubElement(parent, tag, attrs or {})


def _text(parent: ET.Element, tag: str, value: str) -> ET.Element:
    """Build one element carrying its value as *text*, not an attribute.

    This Gazebo (8.15.0, aarch64) reads pose, mass, inertia, limit and dynamics
    values as element text. Written as attributes it emits "XML Attribute[xyz]
    in element[pose] not defined in SDF" and then fails the world with "Empty
    string used when setting a required parameter". The same text form is what
    ``gz sdf -p`` writes out, which is where the convention comes from.
    """
    node = ET.SubElement(parent, tag)
    node.text = value
    return node


def _pose(parent: ET.Element, tag: str, xyz: str, rpy: str = "0 0 0") -> ET.Element:
    """A six-value pose, in the text form this build expects."""
    return _text(parent, tag, f"{xyz} {rpy}")


def _inertial(
    link: ET.Element, mass: str, inertia: dict[str, str], origin: tuple[str, ...] | None
) -> None:
    """Mass properties and the origin they are expressed about."""
    node = _element(link, "inertial")
    if origin is not None:
        _pose(node, "pose", " ".join(origin))
    _text(node, "mass", mass)
    tensor = _element(node, "inertia")
    for key, value in inertia.items():
        _text(tensor, key, value)


def _box(link: ET.Element, size: tuple[str, ...], origin: tuple[str, ...]) -> None:
    collision = _element(link, "collision", {"name": f"{link.get('name')}_collision"})
    _pose(collision, "pose", " ".join(origin))
    _text(_element(collision, "geometry"), "box", " ".join(size))


def build(variant: str, world_name: str = "minifinger_world") -> str:
    """Return the whole minimal world SDF for one section-7 variant."""
    collision = variant != "visual"
    mass = FINGER_MASS
    inertia = dict(FINGER_INERTIA)
    origin: tuple[str, ...] | None = PAD_ORIGIN
    damping, friction = DAMPING, FRICTION
    if variant == "simple":
        # A deliberately plain inertia box: equal principal moments, origin on
        # the body frame, nothing borrowed from the Panda's measured tensor.
        inertia = {
            "ixx": "1.0e-05",
            "ixy": "0",
            "ixz": "0",
            "iyy": "1.0e-05",
            "iyz": "0",
            "izz": "1.0e-05",
        }
        origin = ("0", "0", "0")
    elif variant == "free":
        damping, friction = "0.0", "0.0"

    parent_cylinder = variant == "parentcoll"

    sdf = ET.Element("sdf", {"version": "1.11"})
    world = _element(sdf, "world", {"name": world_name})
    _element(world, "plugin", {"filename": "gz-sim-physics-system", "name": "gz::sim::systems::Physics"})
    _text(world, "gravity", "0 0 -9.8")
    _element(world, "plugin", {"filename": "gz-sim-joint-position-controller-system", "name": "gz::sim::systems::JointPositionController"})

    model = _element(world, "model", {"name": "minifinger"})
    ET.SubElement(model, "static").text = "false"
    _pose(model, "pose", "0 0 0.5")

    parent = _element(model, "link", {"name": "parent"})
    _inertial(
        parent,
        "1.0",
        {"ixx": "0.01", "ixy": "0", "ixz": "0", "iyy": "0.01", "iyz": "0", "izz": "0.01"},
        ("0", "0", "0"),
    )
    _box(parent, ("0.05", "0.05", "0.05"), ("0", "0", "0"))
    if parent_cylinder:
        # panda_link7 carries a cylinder collision of radius 0.045 and length 0.1
        # centred on its own origin, and both finger pads sit 44.5 mm above the
        # finger joint origin on that same link. The minimal model starts with a
        # small box that is nowhere near them; giving the parent that real
        # cylinder is the isolated version of candidate 11/12, collision
        # self-interaction against the parent's own geometry.
        cylinder = _element(parent, "collision", {"name": "parent_cylinder"})
        _text(_element(cylinder, "geometry"), "cylinder", "0.045 0.1")

    joint = _element(model, "joint", {"name": "slide", "type": "prismatic"})
    _pose(joint, "pose", " ".join(JOINT_ORIGIN))
    _text(joint, "parent", "parent")
    _text(joint, "child", "finger")
    axis = _element(joint, "axis")
    _element(axis, "xyz").text = "0 1 0"
    limit = _element(axis, "limit")
    _text(limit, "lower", LOWER)
    _text(limit, "upper", UPPER)
    _text(limit, "effort", EFFORT)
    _text(limit, "velocity", VELOCITY)
    dynamics = _element(axis, "dynamics")
    _text(dynamics, "damping", damping)
    _text(dynamics, "friction", friction)

    finger = _element(model, "link", {"name": "finger"})
    if variant == "pandalink":
        # Reproduce the Panda's own pair of poses verbatim, read out of
        # model_sdf.txt: the joint is relative to panda_link7 with a -0.7854 yaw,
        # and the child link carries an ABSOLUTE world pose with a 180 deg roll
        # and the yaw sign flipped. The minimal model otherwise tracks perfectly,
        # and this is the one structural difference between it and the finger.
        joint.find("pose").text = "0.0038890874388765278 0.0038890871541754891 0.1515 0 0 -0.78539820000000005"
        # Already six values; _pose would append "0 0 0" and make it nine.
        _text(finger, "pose", "0.091889087 -0.003889093 0.881500000 3.141592616 0.000000038 0.785398200")
    _inertial(finger, mass, inertia, origin)
    if collision:
        _box(finger, PAD_SIZE, PAD_ORIGIN)

    # The controller is a plugin on the MODEL, not the world and not the joint:
    # the world placement is refused outright, and this build does not define a
    # <plugins> wrapper inside <joint>.
    controller = _element(
        model,
        "plugin",
        {"filename": "gz-sim-joint-position-controller-system", "name": "gz::sim::systems::JointPositionController"},
    )
    # State feedback lives on the MODEL too, alongside the controller, exactly as
    # in gazebo/world.py. Put it in the world and no joint_state is ever
    # published, which reads as "the joint does not move" rather than as a
    # missing reader.
    _element(
        model,
        "plugin",
        {"filename": "gz-sim-joint-state-publisher-system", "name": "gz::sim::systems::JointStatePublisher"},
    )
    for key, value in (
        ("joint_name", JOINT),
        *(
            # The finger gains exactly as gazebo/world.py configures them --
            # p 1000, i 100, d 50, integral clamped to +/-4, effort 100 -- while
            # every other variant uses a plain proportional controller. This is
            # replication for isolation, not tuning.
            (("p_gain", "1000"), ("i_gain", "100"), ("d_gain", "50"),
             ("i_max", "4"), ("i_min", "-4"), ("cmd_max", "100"), ("cmd_min", "-100"))
            if variant == "pandagains"
            else (("p_gain", "1000"), ("i_gain", "0"), ("d_gain", "0"),
                  ("i_max", "0.1"), ("i_min", "-0.1"), ("cmd_max", "50"), ("cmd_min", "-50"))
        ),
    ):
        _text(controller, key, value)

    ET.indent(sdf, space="  ")
    return ET.tostring(sdf, encoding="unicode") + "\n"


# --- running it -------------------------------------------------------------
#
# The sweep lives in this file rather than a shell one-liner because a shell
# chain through `docker exec -d` loses most of its output in this harness, and
# because Part 1C section 9 forbids guessing protobuf APIs. So the command goes
# out over gz.transport with Double (module confirmed to exist), and link poses
# are read by parsing the *text* of `gz topic -e` -- no message class guessed.

COMMAND_Q = (0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040)
WORLD = "minifinger_world"
MODEL = "minifinger"
JOINT = "slide"
SETTLE_S = 1.2


def _topics() -> list[str]:
    """Every advertised topic, straight from the CLI."""
    out = subprocess.run(
        ["gz", "topic", "-l"], capture_output=True, text=True, timeout=60, check=False
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _echo_once(topic: str, timeout: int = 6) -> str:
    """Read one message as text via the CLI, so no message class is guessed."""
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


def _scalar_after(text: str, marker: str) -> float | None:
    """Read the scalar that follows a ``name:`` line naming our joint."""
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("name:") and JOINT in line:
            for follow in lines[index + 1 :]:
                if follow.startswith(marker):
                    try:
                        return float(follow.split(":", 1)[1].strip())
                    except ValueError:
                        return None
    return None


def _xyz(text: str) -> list[float] | None:
    """Read the x/y/z triple from a text dump."""
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("x:"):
            try:
                return [
                    float(lines[index + offset].split(":", 1)[1].strip()) for offset in range(3)
                ]
            except (ValueError, IndexError):
                return None
    return None


def run(variant: str) -> dict[str, object]:
    """Drive the minimal joint through its whole range and record what it does."""
    sdf_path = Path(f"/tmp/min_{variant}.sdf")
    sdf_path.write_text(build(variant))
    sim_log = Path(f"/tmp/min_{variant}_sim.log").open("w+")  # noqa: SIM115 - must outlive the try block
    process = subprocess.Popen(
        # -s: server only. Without it gz sim tries to open a GUI, the container
        # has no display, and it dies before advertising a single topic -- which
        # produced a clean-looking result with every reading null.
        ["gz", "sim", "-r", "-s", "-v", "4", str(sdf_path)],
        stdout=sim_log,
        stderr=subprocess.STDOUT,
    )
    result: dict[str, object] = {"variant": variant, "sdf_path": str(sdf_path)}
    try:
        time.sleep(10.0)
        topics = _topics()
        result["topics"] = topics
        # Topic shape copied from the project's own live control path
        # (simulation/gazebo.py:510), not invented: advertise on
        # /model/<m>/joint/<j>/<index>/cmd_pos.
        cmd_topic = f"/model/{MODEL}/joint/{JOINT}/0/cmd_pos"
        state_topic = f"/world/{WORLD}/model/{MODEL}/joint_state"
        pose_topic = f"/world/{WORLD}/model/{MODEL}/link/finger/pose"
        result["topics_present"] = {
            "cmd": cmd_topic in topics,
            "joint_state": state_topic in topics,
            "finger_pose": pose_topic in topics,
        }
        result["pose_like_topics"] = [t for t in topics if "pose" in t.lower()]

        from gz.msgs10.double_pb2 import Double
        from gz.transport13 import Node

        node = Node()
        publisher = node.advertise(cmd_topic, Double)
        result["topics_present"]["cmd"] = bool(publisher)
        time.sleep(1.0)

        rows: list[dict[str, object]] = []
        first_y: float | None = None
        for q in COMMAND_Q:
            message = Double()
            message.data = float(q)
            publisher.publish(message)
            time.sleep(SETTLE_S)
            measured = _scalar_after(_echo_once(state_topic), "position:")
            pose_text = _echo_once(pose_topic) if pose_topic in topics else ""
            position = _xyz(pose_text) if pose_text else None
            if position is not None and first_y is None:
                first_y = position[1]
            rows.append(
                {
                    "commanded_q_m": q,
                    "measured_q_m": measured,
                    "error_m": None if measured is None else round(measured - q, 6),
                    "link_position": position,
                    "displacement_from_q0": (
                        None if position is None or first_y is None else round(position[1] - first_y, 6)
                    ),
                }
            )
        result["rows"] = rows
        tracked = [r for r in rows if r["measured_q_m"] is not None]
        result["max_abs_error_m"] = (
            max(abs(float(r["error_m"])) for r in tracked) if tracked else None
        )
        result["full_stroke"] = bool(
            tracked and max(float(r["measured_q_m"]) for r in tracked) > 0.03
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        # If the world refused to load, every reading below is null and the
        # reason is in here. Discarding this stream is how the first attempt
        # produced a clean-looking result with nothing in it.
        sim_log.seek(0)
        result["sim_log_tail"] = sim_log.read()[-4000:]
        sim_log.close()
    return result


def main() -> int:
    """Write an SDF with --out, or run the sweep with --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", default="base", choices=("base", "visual", "simple", "free", "parentcoll", "pandalink", "pandagains")
    )
    parser.add_argument("--out")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--result",
        default="experiments/results/gazebo/finger_diagnostic/minimal_result.json",
    )
    args = parser.parse_args()
    if args.run:
        payload = run(args.variant)
        target = Path(args.result)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({k: v for k, v in payload.items() if k != "topics"}, indent=2))
        return 0
    if not args.out:
        parser.error("--out is required unless --run is given")
    Path(args.out).write_text(build(args.variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
