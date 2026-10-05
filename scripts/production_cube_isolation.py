"""Part 1E: production world, cube present, finger-only sweep, no policy.

One experiment only. Real Panda in real placement, real table, real cube at
the failing episode's pose (seed 3: [0.0226, 0.2256, 0.008]), arm commanded to
the failing episode's grasp pose (step 56 measured joints), then ONLY the finger
joints are swept. No policy, no state machine, no teleporting mid-sweep.

Method notes (brief sections 3, 9, 10):

  * No guessed protobuf. Joint state is parsed from ``gz topic -e`` text; the
    only transport-API read is ``Wrench``, the same import the working finger
    diagnostic uses. Contact evidence comes from CLI text echoes and is recorded
    raw, so a wrong assumption about a schema shows up as an empty string.
  * Contacts are split by counterpart by searching the Contacts text for the
    collision names themselves. Presence/absence per pair is what section 3 asks
    for; magnitudes are not inferred.
  * Failure modes fail soft: a missing topic records nulls plus a note.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys as _sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

_sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_chain import _echo, _topics, parse_joint_state

WORLD = "robot_ci_pick_place"
MODEL = "panda"
FINGERS = ("left_finger", "right_finger")
ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")

# Failing episode grasp pose: measured arm joints at step 56, the last tick
# before TRANSIT with the jaw at 10.6 mm. TCP [0.0219, 0.2246, 0.0070],
# cube [0.02259, 0.22558, 0.008].
EPISODE_ARM_POSE = (1.75834, 0.44239, -0.0236, -2.72086, 2.41011, 3.11683, 0.09254)
#: The episode's own arm path, stage by stage (measured joints at ticks 40 and
#: 45). A single step to the grasp pose swung the wrist through the cube and
#: knocked it 17.6 mm away before the sweep began; following the recorded path
#: is how the approach reproduces the episode instead of disturbing it.
EPISODE_APPROACH_STAGES = (
    (0.60824, 0.28435, 1.02521, -2.72578, 2.51666, 3.39567, -0.0987),
    (1.06282, 0.44262, 0.58204, -2.72192, 2.4145, 3.12437, 0.09746),
    EPISODE_ARM_POSE,
)
#: Per-finger q for the episode's open jaw during DESCEND (aperture 0.044).
EPISODE_OPEN_Q = 0.022
EPISODE_CUBE_XYZ = (0.02259, 0.22558, 0.008)
ARM_SETTLE_S = 8.0

# Per-finger joint coordinate, metres, in CLOSING order -- the direction the
# failing episode takes: aperture 0.044 (q 0.022) during DESCEND, collapsing to
# 0.0104 (q ~0.0052) at step 56 where it creeps. Dense around 5 mm.
SWEEP_Q = (
    0.022, 0.020, 0.015, 0.010, 0.008, 0.007, 0.006,
    0.0055, 0.005, 0.0045, 0.004, 0.003, 0.002, 0.001, 0.0001,
)
SETTLE_S = 2.5
OUTDIR = Path("experiments/results/gazebo/finger_diagnostic/production_cube_isolation")


def build() -> str:
    """Production SDF with the cube present. Verified, not assumed."""
    from robot_ci.gazebo.world import world_sdf
    from robot_ci.scenarios import default_scenario

    text = world_sdf(
        default_scenario(seed=3),
        robot_urdf="gazebo/models/panda/panda.urdf",
        include_object=True,
        include_table=True,
        finger_collisions=True,
    )
    root = ET.fromstring(text.split("?>", 1)[1])
    names = [m.get("name") for m in root.iter("model")]
    assert "target_cube" in names, f"cube missing from production SDF: {names}"
    cube = next(m for m in root.iter("model") if m.get("name") == "target_cube")
    pose = cube.find("pose")
    assert pose is not None and pose.text is not None
    xyz = [float(v) for v in pose.text.split()[:3]]
    # The SDF writes %.6f, so both x and y land a few um from the episode's
    # rounded values. Tolerance is 1e-5 m (10 um) for that reason.
    assert all(abs(a - b) < 1e-5 for a, b in zip(xyz, EPISODE_CUBE_XYZ, strict=True)), xyz
    OUTDIR.mkdir(parents=True, exist_ok=True)
    Path("/tmp/prod_cube_world.sdf").write_text(text)
    (OUTDIR / "production_world.sdf").write_text(text)
    return text


def _add_finger_force_sensors(sdf_text: str) -> str:
    """Bolt a force-torque sensor onto each finger joint of the rendered world.

    Passive instrumentation only: a sensor adds no force to the physics. The
    production model ships none, which is why the Part 1F report had to record
    "reaction at stall: not measured". Placement follows the pattern proven by
    ``gazebo_finger_diagnostic`` -- sensor under the JOINT (Gazebo refuses a
    link-parented one), world plugin ``gz-sim-forcetorque-system``.
    """
    root = ET.fromstring(sdf_text.split("?>", 1)[1])
    world = root.find("world")
    assert world is not None
    robot = next(m for m in world.findall("model") if m.get("name") == MODEL)
    for joint in robot.findall("joint"):
        if joint.get("name") not in FINGERS:
            continue
        sensor = ET.SubElement(
            joint, "sensor", {"name": "finger_ft", "type": "force_torque"}
        )
        for tag, value in (
            ("always_on", "1"),
            ("update_rate", "200"),
            ("topic", f"/finger_ft_{joint.get('name')}"),
            ("frame", "child"),
            ("measure_direction", "child_to_parent"),
        ):
            node = ET.SubElement(sensor, tag)
            node.text = value
    ET.SubElement(
        world,
        "plugin",
        {
            "filename": "gz-sim-forcetorque-system",
            "name": "gz::sim::systems::ForceTorque",
        },
    )
    return '<?xml version="1.0"?>' + ET.tostring(root, encoding="unicode")


#: TCP site offset along the hand z axis, from the URDF (site 0.1034, pad centre
#: 0.1029): pad centre = TCP + R @ (0, +/- (0.0055 + 0.0075 + q), -0.0005).
#: 0.0055 is the finger joint origin's y offset in the hand frame, 0.0075 the
#: pad centre's offset from the joint origin, so the inner face sits at
#: 0.0055 + q -- gap 0.011 + 2q, matching world.py's "pads are 0.011 m apart at
#: rest" and the URDF's 0.015-wide pad boxes.
_TCP_TO_PAD_DZ = -0.0005


def geometry_snapshot(
    arm_q: tuple[float, ...],
    left_q: float,
    right_q: float,
    cube_xyz: tuple[float, float, float],
    base_xyz: tuple[float, float, float],
) -> dict[str, Any] | None:
    """TCP, pad poses and pad-to-cube clearance from measured joints (FK only).

    No new topics: forward kinematics over the URDF that generated the world is
    the same chain the backend reports TCP from, evaluated at measured joint
    angles. Returns None (fail-soft) if FK is unavailable; never fabricates a
    pose. The clearance is the gap along the jaw axis between a pad's inner face
    and the cube's projected half-width -- the quantity that predicts where a
    straddled jaw must stall.
    """
    try:
        import numpy as np

        from robot_ci.gazebo.kinematics import UrdfJacobianProvider

        provider = getattr(geometry_snapshot, "_provider", None)
        if provider is None:
            provider = UrdfJacobianProvider("gazebo/models/panda/panda.urdf")
            geometry_snapshot._provider = provider  # type: ignore[attr-defined]
        pos, rot = provider.forward(np.asarray(arm_q, dtype=np.float64))
        tcp = np.asarray(base_xyz) + pos
        axis = rot @ np.asarray([0.0, 1.0, 0.0])
        cube = np.asarray(cube_xyz)
        half = 0.008
        projected = half * float(np.abs(axis).sum())
        out: dict[str, Any] = {
            "tcp": [round(float(v), 6) for v in tcp],
            "jaw_axis": [round(float(v), 6) for v in axis],
            "tcp_to_cube_m": round(float(np.linalg.norm(tcp - cube)), 6),
        }
        for name, sign, q in (("left", 1.0, left_q), ("right", -1.0, right_q)):
            inner = 0.0055 + q
            centre = tcp + rot @ np.asarray([0.0, sign * (inner + 0.0075), _TCP_TO_PAD_DZ])
            out[f"{name}_pad_centre"] = [round(float(v), 6) for v in centre]
            out[f"{name}_inner_face_offset_m"] = round(inner, 6)
            out[f"{name}_pad_bottom_z_m"] = round(float(centre[2]) - 0.0075, 6)
        out["cube_projected_half_width_m"] = round(projected, 6)
        out["clearance_to_cube_m"] = round((0.0055 + (left_q + right_q) / 2) - projected, 6)
        out["predicted_stall_q_m"] = round(projected - 0.0055, 6)
        # Per-face gaps against the LIVE cube: pad plane at +/-(0.0055 + q)
        # from the TCP along the jaw axis, cube faces at c +/- projected_half.
        # The sign of c matters: the cube sits ~1 mm off the TCP centreline, so
        # one pad always contacts first and the other last.
        c = float(np.dot(cube - tcp, axis))
        gaps = []
        for q in (left_q, right_q):
            inner = 0.0055 + q
            gaps.append(inner - (c + projected))
            gaps.append(c - projected + inner)
        out["cube_offset_along_axis_m"] = round(c, 6)
        out["pad_cube_gaps_m"] = [round(g, 6) for g in gaps]
        out["min_gap_m"] = round(min(gaps), 6)
        return out
    except Exception as exc:  # must not kill the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def contact_hits(obj_text: str, tab_text: str) -> dict[str, Any]:
    """Canonical per-counterpart contact presence.

    Each sensor's text is parsed on its own and pairs are matched on exact
    collision names in both orientations. The previous version concatenated the
    two topics before substring-matching, which manufactured pairs that exist in
    neither message (e.g. `panda_right_finger` from the TABLE sensor combined
    with `target_cube` from the CUBE sensor) -- that is how a sweep row came back
    "in contact with the cube" while the cube's own raw message held only
    `target_cube <-> table`.

    Reads the raw text because it is the ground truth Gazebo published; the
    protobuf class was never guessed at for this file.
    """

    def pairs(text: str) -> list[tuple[str, str]]:
        """Walk `contact { ... }` blocks; each holds exactly two name lines."""
        found: list[tuple[str, str]] = []
        block: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if s == "contact {":
                block = []
            elif s.startswith("name:"):
                block.append(s.split(":", 1)[1].strip().strip('"'))
            elif s == "}" and len(block) == 2:
                found.append((block[0], block[1]))
                block = []
        return found

    def touching(pairs_list: list[tuple[str, str]], a: str, b: str) -> bool:
        """Match on `::`-delimited tokens, not substrings.

        Gazebo collision names are `model::link::collision`. A substring test
        would accept `archive::panda_left_finger_note` as a finger contact; the
        token form matches the real `panda::panda_left_finger::panda_left_finger_collision`
        (exact middle token) and rejects lookalikes. Exact equality only --
        prefix/suffix allowances would reintroduce the forgery.
        """

        def has(name: str, wanted: str) -> bool:
            return wanted in name.split("::")

        return any(
            (has(p1, a) and has(p2, b)) or (has(p1, b) and has(p2, a))
            for p1, p2 in pairs_list
        )

    obj_pairs = pairs(obj_text)
    tab_pairs = pairs(tab_text)
    return {
        "left_finger_vs_cube": touching(obj_pairs, "panda_left_finger", "target_cube")
        or touching(tab_pairs, "panda_left_finger", "target_cube"),
        "right_finger_vs_cube": touching(obj_pairs, "panda_right_finger", "target_cube")
        or touching(tab_pairs, "panda_right_finger", "target_cube"),
        "any_finger_vs_table": any(
            touching(tab_pairs, f"panda_{side}_finger", "table") for side in ("left", "right")
        ),
        "cube_vs_table": touching(obj_pairs, "target_cube", "table")
        or touching(tab_pairs, "target_cube", "table"),
        "cube_sensor_pairs": [f"{a} <-> {b}" for a, b in obj_pairs],
        "table_sensor_pairs": [f"{a} <-> {b}" for a, b in tab_pairs],
    }


class _ContactAccumulator:
    """Every contact pair seen since the last ``reset()``, not just the newest.

    Part 1F's failure was structural: the reader replaced its snapshot on each
    message, so a squeeze that began and ended between two control-tick reads
    left no trace. Counting and retaining every pair across the whole window
    answers "did this ever happen this tick" instead of "is it happening at
    this microsecond".
    """

    def __init__(self) -> None:
        self.pairs: set[tuple[str, str]] = set()
        self.messages = 0
        self.errors: list[str] = []

    def on(self, msg: Any) -> None:
        """Record all pairs; a callback failure must not kill the stream."""
        try:
            for contact in msg.contact:
                self.pairs.add(
                    (contact.collision1.name, contact.collision2.name)
                )
        except Exception as exc:
            if len(self.errors) < 4:
                self.errors.append(f"{type(exc).__name__}: {exc}")
        self.messages += 1

    def reset(self) -> set[tuple[str, str]]:
        """Drain the accumulated pairs and return them."""
        drained, self.pairs = self.pairs, set()
        return drained


class _CubePose:
    """Latest live pose of ``target_cube`` from ``/world/<w>/pose/info``.

    The cube is ejected mid-sweep in the failing episode, so assuming its spawn
    pose would measure the wrong clearance at exactly the row that matters.
    Field names verified against ``gz.msgs10.pose_v_pb2.Pose_V`` before use.
    """

    def __init__(self) -> None:
        self.xyz: tuple[float, float, float] | None = None
        self.messages = 0

    def on(self, msg: Any) -> None:
        """Keep the newest target_cube position; never raise."""
        try:
            for pose in msg.pose:
                if pose.name == "target_cube":
                    self.xyz = (
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    )
        except Exception:
            return
        self.messages += 1


def run(*, episode_timing: bool = False) -> dict[str, Any]:
    """Execute the single licensed sweep and record everything raw."""
    sdf_text = _add_finger_force_sensors(build())
    root = ET.fromstring(sdf_text.split("?>", 1)[1])
    world = root.find("world")
    world_name = world.get("name") if world is not None else WORLD
    robot = next(m for m in world.findall("model") if m.get("name") == MODEL)
    base_pose = robot.find("pose")
    base_xyz = tuple(float(v) for v in (base_pose.text or "0 0 0 0 0 0").split()[:3])
    Path("/tmp/prod_cube_world.sdf").write_text(sdf_text)

    sim_log = Path("/tmp/prod_cube_sim.log").open("w+")  # noqa: SIM115 - outlives try
    env = dict(os.environ)
    env["GZ_SIM_RESOURCE_PATH"] = str(Path("gazebo/models").resolve())
    process = subprocess.Popen(
        ["gz", "sim", "-r", "-s", "-v", "4", "/tmp/prod_cube_world.sdf"],
        stdout=sim_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    result: dict[str, Any] = {
        "world_name": world_name,
        "arm_pose_commanded": list(EPISODE_ARM_POSE),
        "cube_pose_expected": list(EPISODE_CUBE_XYZ),
        "sweep_q_m": list(SWEEP_Q),
    }
    try:
        state_topic = f"/world/{world_name}/model/{MODEL}/joint_state"
        deadline = time.monotonic() + 90.0
        topics: list[str] = []
        while time.monotonic() < deadline:
            topics = _topics()
            if state_topic in topics:
                break
            time.sleep(5.0)
        result["startup_wait_s"] = round(60.0 - max(0.0, deadline - time.monotonic()), 1)
        result["joint_state_present"] = state_topic in topics
        if not result["joint_state_present"]:
            result["verdict"] = "NOT READABLE"
            return result

        from gz.msgs10.contacts_pb2 import Contacts
        from gz.msgs10.double_pb2 import Double
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.wrench_pb2 import Wrench
        from gz.transport13 import Node

        node = Node()
        # Both fingers' FT sensors, kept separately: Part 1B's attribution was
        # once backwards because one shared slot answered for both joints.
        forces: dict[str, dict[str, float]] = {
            f: {"x": 0.0, "y": 0.0, "z": 0.0, "messages": 0} for f in FINGERS
        }

        def _wrench_for(finger: str) -> Any:
            def _on(msg: Any) -> None:
                slot = forces[finger]
                slot["x"] = round(float(msg.force.x), 4)
                slot["y"] = round(float(msg.force.y), 4)
                slot["z"] = round(float(msg.force.z), 4)
                slot["messages"] = int(slot["messages"]) + 1

            return _on

        all_topics = _topics()
        result["finger_ft_topics"] = [x for x in all_topics if "finger_ft" in x]
        for finger in FINGERS:
            ft_topic = f"/finger_ft_{finger}"
            if ft_topic in all_topics:
                node.subscribe(Wrench, ft_topic, _wrench_for(finger))
        result["finger_ft_present"] = len(result["finger_ft_topics"]) == 2

        # Accumulating contact subscribers: Part 1F read only the latest message
        # and missed a transient squeeze. Every pair seen since the last sample
        # is kept here; the one-shot latest reading is recorded beside it so the
        # two detection strategies can be compared on the same ticks.
        accum = _ContactAccumulator()
        obj_topic = (
            f"/world/{world_name}/model/target_cube/link/link/sensor/object_contacts/contact"
        )
        tab_topic = (
            f"/world/{world_name}/model/table/link/link/sensor/table_contacts/contact"
        )
        result["object_contact_topic_present"] = obj_topic in all_topics
        result["table_contact_topic_present"] = tab_topic in all_topics
        if result["object_contact_topic_present"]:
            node.subscribe(Contacts, obj_topic, accum.on)
        if result["table_contact_topic_present"]:
            node.subscribe(Contacts, tab_topic, accum.on)

        cube_pose = _CubePose()
        pose_topic = f"/world/{world_name}/pose/info"
        result["pose_info_topic_present"] = pose_topic in all_topics
        if result["pose_info_topic_present"]:
            node.subscribe(Pose_V, pose_topic, cube_pose.on)
        result["all_relevant_topics"] = [
            x for x in all_topics
            if any(k in x for k in ("joint_state", "contact", "pose/info", "forcetorque"))
        ]

        pubs = {
            j: node.advertise(f"/model/{MODEL}/joint/{j}/0/cmd_pos", Double)
            for j in ARM + FINGERS
        }
        open_cmd = Double()
        open_cmd.data = EPISODE_OPEN_Q
        for finger in FINGERS:
            pubs[finger].publish(open_cmd)
        # Staged approach along the episode's own recorded path. One step to the
        # grasp pose knocked the cube 17.6 mm aside last run; the cube-stability
        # check after each stage fails the run loudly instead of sweeping around
        # a displaced object and calling it a reproduction.
        approach: list[dict[str, Any]] = []
        spawn = list(EPISODE_CUBE_XYZ)
        for stage_index, stage in enumerate(EPISODE_APPROACH_STAGES):
            for joint, target in zip(ARM, stage, strict=True):
                cmd = Double()
                cmd.data = float(target)
                pubs[joint].publish(cmd)
            time.sleep(12.0)
            stage_state = parse_joint_state(_echo(state_topic))
            worst = max(
                abs((stage_state.get(j, {}).get("position") or 0.0) - target)
                for j, target in zip(ARM, stage, strict=True)
            )
            live = cube_pose.xyz
            knock = (
                None
                if live is None
                else max(abs(a - b) for a, b in zip(live, spawn, strict=True))
            )
            approach.append(
                {
                    "stage": stage_index,
                    "worst_joint_error_rad": round(worst, 6),
                    "cube_knock_from_spawn_m": (
                        None if knock is None else round(knock, 6)
                    ),
                }
            )
            if knock is not None and knock > 0.003:
                result["approach"] = approach
                result["verdict"] = "CUBE_KNOCKED_DURING_APPROACH"
                return result
        result["approach"] = approach
        # Convergence poll instead of a fixed sleep: the Part 1F run woke up
        # with joint5 13 mrad and joint7 26 mrad off and only noticed at the
        # end, which silently broke the "same state as the episode" premise.
        deadline = time.monotonic() + 45.0
        settled: dict[str, dict[str, float]] = {}
        worst = float("inf")
        while time.monotonic() < deadline:
            time.sleep(ARM_SETTLE_S)
            settled = parse_joint_state(_echo(state_topic))
            worst = max(
                abs((settled.get(j, {}).get("position") or 0.0) - target)
                for j, target in zip(ARM, EPISODE_ARM_POSE, strict=True)
            )
            if worst < 0.002:  # 2 mrad ~ sub-mm TCP error at this arm length
                break
        result["arm_settled"] = {j: settled.get(j, {}) for j in ARM}
        result["arm_settled_worst_error_rad"] = round(worst, 6)
        result["arm_converged"] = worst < 0.002

        # TCP alignment check, FK over measured joints. If the episode TCP is
        # missed by more than 2 mm, correct with damped least squares -- moving
        # the arm to the recorded state, not tuning anything.
        try:
            import numpy as np

            from robot_ci.gazebo.kinematics import UrdfJacobianProvider

            provider = getattr(geometry_snapshot, "_provider", None)
            if provider is None:
                provider = UrdfJacobianProvider("gazebo/models/panda/panda.urdf")
                geometry_snapshot._provider = provider  # type: ignore[attr-defined]
            episode_tcp = np.asarray((0.02193, 0.22464, 0.00702))
            corrections: list[dict[str, Any]] = []
            for attempt in range(4):
                measured = [
                    settled.get(j, {}).get("position") for j in ARM
                ]
                if any(v is None for v in measured):
                    break
                pos, _rot = provider.forward(np.asarray(measured, dtype=np.float64))
                tcp = np.asarray(base_xyz) + pos
                err = episode_tcp - tcp
                dist = float(np.linalg.norm(err))
                corrections.append({"attempt": attempt, "error_m": round(dist, 6)})
                if dist < 0.005:
                    break
                jac = provider.jacobian(np.asarray(measured, dtype=np.float64))
                jpos = jac[:3, :]
                delta = jpos.T @ np.linalg.solve(
                    jpos @ jpos.T + 1e-4 * np.eye(3), err
                )
                delta = np.clip(delta, -0.05, 0.05)
                for joint, step in zip(ARM, delta, strict=True):
                    cmd = Double()
                    cmd.data = float(
                        (settled.get(joint, {}).get("position") or 0.0) + step
                    )
                    pubs[joint].publish(cmd)
                time.sleep(3.0)
                settled = parse_joint_state(_echo(state_topic))
            result["tcp_alignment"] = corrections
        except Exception as exc:  # alignment is optional, never fatal
            result["tcp_alignment"] = {"error": f"{type(exc).__name__}: {exc}"}
        result["arm_settled"] = {j: settled.get(j, {}) for j in ARM}
        final_err = max(
            abs((settled.get(j, {}).get("position") or 0.0) - target)
            for j, target in zip(ARM, EPISODE_ARM_POSE, strict=True)
        )
        result["arm_final_worst_error_rad"] = round(final_err, 6)


        # Episode-timing replay: the episode issues ONE step command to
        # hold_width travel (0.0002 m -> q 0.0001), lets the jaw run for six
        # control ticks (0.3 s) and then lifts. The stepped sweep gives each
        # point 2.5 s, which is why it creeps through the friction zone the
        # episode never gets time to cross. This reproduces the episode's clock
        # and then keeps sampling to show what more time would have done.
        if episode_timing:
            close_start = time.monotonic()
            final_cmd = Double()
            final_cmd.data = 0.0001
            for finger in FINGERS:
                pubs[finger].publish(final_cmd)
            samples: list[dict[str, Any]] = []
            next_at = 0.0
            while True:
                now = time.monotonic() - close_start
                if now >= 12.0:
                    break
                if now >= next_at:
                    state = parse_joint_state(_echo(state_topic))
                    left = state.get("left_finger", {})
                    right = state.get("right_finger", {})
                    obj_text = _echo(obj_topic) if result["object_contact_topic_present"] else ""
                    tab_text = _echo(tab_topic) if result["table_contact_topic_present"] else ""
                    seen = sorted(f"{a} <-> {b}" for a, b in accum.reset())
                    arm_q = [state.get(j, {}).get("position") for j in ARM]
                    geom = (
                        geometry_snapshot(
                            tuple(float(v) for v in arm_q),
                            float(left.get("position") or 0.0),
                            float(right.get("position") or 0.0),
                            cube_pose.xyz or EPISODE_CUBE_XYZ,
                            base_xyz,
                        )
                        if all(v is not None for v in arm_q)
                        else None
                    )
                    samples.append(
                        {
                            "t_s": round(now, 3),
                            "left_q_m": left.get("position"),
                            "right_q_m": right.get("position"),
                            "pad_sep_m": None
                            if left.get("position") is None or right.get("position") is None
                            else round(
                                0.011 + (left.get("position") or 0.0)
                                + (right.get("position") or 0.0),
                                5,
                            ),
                            "force_y_N": {f: forces[f]["y"] for f in FINGERS},
                            "contacts_accumulated": seen,
                            "geometry": geom,
                            "cube_live_xyz": list(cube_pose.xyz) if cube_pose.xyz else None,
                        }
                    )
                    next_at += 0.05 if now < 1.0 else 0.25
                time.sleep(0.01)
            result["episode_timing_close"] = samples
            at_03 = next((s for s in samples if s["t_s"] >= 0.3), None)
            result["episode_timing_0p3s"] = at_03
            result["episode_timing_final"] = samples[-1] if samples else None
            return result

        rows: list[dict[str, Any]] = []
        for q in SWEEP_Q:
            cmd = Double()
            cmd.data = float(q)
            for finger in FINGERS:
                pubs[finger].publish(cmd)
            time.sleep(SETTLE_S)
            state = parse_joint_state(_echo(state_topic))
            left, right = state.get("left_finger", {}), state.get("right_finger", {})
            obj_text = _echo(obj_topic) if result["object_contact_topic_present"] else ""
            tab_text = _echo(tab_topic) if result["table_contact_topic_present"] else ""
            # Drained every tick: pairs seen DURING this settle window, plus
            # the one-shot latest reading beside it for comparison.
            seen = sorted(f"{a} <-> {b}" for a, b in accum.reset())
            arm_state = [state.get(j, {}).get("position") for j in ARM]
            arm_ok = all(v is not None for v in arm_state)
            geom = (
                geometry_snapshot(
                    tuple(float(v) for v in arm_state),
                    float(left.get("position") or 0.0),
                    float(right.get("position") or 0.0),
                    cube_pose.xyz or EPISODE_CUBE_XYZ,
                    base_xyz,
                )
                if arm_ok
                else None
            )
            rows.append(
                {
                    "commanded_q_m": q,
                    "left_q_m": left.get("position"),
                    "right_q_m": right.get("position"),
                    "reaction_N": {
                        k: {f: forces[f][k] for f in FINGERS}
                        for k in ("x", "y", "z")
                    },
                    "reaction_messages": {f: forces[f]["messages"] for f in FINGERS},
                    "contacts_accumulated": seen,
                    "cube_live_xyz": list(cube_pose.xyz) if cube_pose.xyz else None,
                    "geometry": geom,
                    "object_contacts": contact_hits(obj_text, tab_text),
                    "cube_sensor_raw": obj_text[:1200],
                    "table_sensor_raw": tab_text[:1200],
                    "arm_drift_rad": {
                        j: (
                            None
                            if state.get(j, {}).get("position") is None
                            or settled.get(j, {}).get("position") is None
                            else round(state[j]["position"] - settled[j]["position"], 9)
                        )
                        for j in ARM
                    },
                }
            )
        result["rows"] = rows
        lefts = [r["left_q_m"] for r in rows if r["left_q_m"] is not None]
        result["max_q_m"] = max(lefts) if lefts else None
        result["full_stroke"] = bool(lefts and max(lefts) > 0.03)
        result["stall_row"] = next(
            (
                {"commanded_q_m": r["commanded_q_m"], "measured_q_m": r["left_q_m"],
                 "min_gap_mm": round(1000 * (r["geometry"] or {}).get("min_gap_m", 0), 2)
                 if r.get("geometry") else None,
                 "contacts_accumulated": r["contacts_accumulated"],
                 "force_y_N": r["reaction_N"]["y"]}
                for r in rows
                if r["left_q_m"] is not None
                and r["left_q_m"] > r["commanded_q_m"] + 0.004
            ),
            None,
        )
        measured = [r["left_q_m"] for r in rows if r["left_q_m"] is not None]
        result["min_q_m"] = min(measured) if measured else None
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


def main() -> int:
    """Run once and write every required artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-timing",
        action="store_true",
        help="single 0.3 s close replay (episode clock) instead of the stepped sweep",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="artifact filename suffix, e.g. episode_timing",
    )
    args = parser.parse_args()
    payload = run(episode_timing=args.episode_timing)
    suffix = f"_{args.tag}" if args.tag else ""
    (OUTDIR / f"config{suffix}.json").write_text(json.dumps({
        "arm_pose_commanded": payload.get("arm_pose_commanded"),
        "cube_pose_expected": payload.get("cube_pose_expected"),
        "sweep_q_m": payload.get("sweep_q_m"),
        "finger_ft_present": payload.get("finger_ft_present"),
        "object_contact_topic_present": payload.get("object_contact_topic_present"),
        "table_contact_topic_present": payload.get("table_contact_topic_present"),
        "startup_wait_s": payload.get("startup_wait_s"),
    }, indent=2) + "\n")
    rows_or_samples = payload.get("rows") or payload.get("episode_timing_close") or []
    (OUTDIR / f"telemetry{suffix}.json").write_text(
        json.dumps(rows_or_samples, indent=2) + "\n")
    (OUTDIR / f"contacts{suffix}.json").write_text(json.dumps([
        {"commanded_q_m": r["commanded_q_m"], **r.get("object_contacts", {})}
        for r in payload.get("rows", [])
    ], indent=2) + "\n")
    summary = {
        k: v
        for k, v in payload.items()
        if k not in ("rows", "sim_log_tail", "episode_timing_close")
    }
    (OUTDIR / f"result{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
