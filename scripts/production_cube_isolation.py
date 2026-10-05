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
        return any(
            (a in p1 and b in p2) or (a in p2 and b in p1)
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


def run() -> dict[str, Any]:
    """Execute the single licensed sweep and record everything raw."""
    sdf_text = build()
    root = ET.fromstring(sdf_text.split("?>", 1)[1])
    world = root.find("world")
    world_name = world.get("name") if world is not None else WORLD

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

        ft_topic = (
            f"/world/{world_name}/model/{MODEL}/joint/left_finger"
            "/sensor/finger_ft/forcetorque"
        )
        result["finger_ft_present"] = ft_topic in _topics()
        if result["finger_ft_present"]:
            node.subscribe(Wrench, ft_topic, _on_wrench)

        pubs = {
            j: node.advertise(f"/model/{MODEL}/joint/{j}/0/cmd_pos", Double)
            for j in ARM + FINGERS
        }
        for joint, target in zip(ARM, EPISODE_ARM_POSE, strict=True):
            cmd = Double()
            cmd.data = float(target)
            pubs[joint].publish(cmd)
        open_cmd = Double()
        open_cmd.data = EPISODE_OPEN_Q
        for finger in FINGERS:
            pubs[finger].publish(open_cmd)
        time.sleep(ARM_SETTLE_S)
        settled = parse_joint_state(_echo(state_topic))
        result["arm_settled"] = {j: settled.get(j, {}) for j in ARM}

        obj_topic = (
            f"/world/{world_name}/model/target_cube/link/link/sensor/object_contacts/contact"
        )
        tab_topic = (
            f"/world/{world_name}/model/table/link/link/sensor/table_contacts/contact"
        )
        result["object_contact_topic_present"] = obj_topic in _topics()
        result["table_contact_topic_present"] = tab_topic in _topics()

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
            rows.append(
                {
                    "commanded_q_m": q,
                    "left_q_m": left.get("position"),
                    "right_q_m": right.get("position"),
                    "reaction_N": {k: force[k] for k in ("x", "y", "z")},
                    "reaction_messages": force["messages"],
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
                {"commanded_q_m": r["commanded_q_m"], "measured_q_m": r["left_q_m"]}
                for r in reversed(rows)
                if r["left_q_m"] is not None and r["left_q_m"] < (r["commanded_q_m"] - 0.004)
            ),
            None,
        )
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
    args = parser.parse_args()
    del args
    payload = run()
    (OUTDIR / "config.json").write_text(json.dumps({
        "arm_pose_commanded": payload.get("arm_pose_commanded"),
        "cube_pose_expected": payload.get("cube_pose_expected"),
        "sweep_q_m": payload.get("sweep_q_m"),
        "finger_ft_present": payload.get("finger_ft_present"),
        "object_contact_topic_present": payload.get("object_contact_topic_present"),
        "table_contact_topic_present": payload.get("table_contact_topic_present"),
        "startup_wait_s": payload.get("startup_wait_s"),
    }, indent=2) + "\n")
    (OUTDIR / "telemetry.json").write_text(
        json.dumps(payload.get("rows", []), indent=2) + "\n")
    (OUTDIR / "contacts.json").write_text(json.dumps([
        {"commanded_q_m": r["commanded_q_m"], **r["object_contacts"]}
        for r in payload.get("rows", [])
    ], indent=2) + "\n")
    (OUTDIR / "result.json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k not in ("rows", "sim_log_tail")},
        indent=2) + "\n")
    print(json.dumps(
        {k: v for k, v in payload.items() if k not in ("rows", "sim_log_tail")},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
