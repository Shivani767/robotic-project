"""Unit tests for the live Gazebo bridge and the demo motion driver.

These run without Gazebo: the readers' parsing, the health semantics and the
kinematics extraction are pure functions of their inputs, so a machine with no
simulator can still check the logic that decides what the browser shows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_ci.gazebo.live import JOINT_NAMES, BridgeHealth, JointStateReader, robot_kinematics
from robot_ci.gazebo.motion_demo import JOINT_TARGETS, MotionDemo, joint_limits

URDF = Path(__file__).resolve().parents[2] / "gazebo" / "models" / "panda" / "panda.urdf"


@pytest.fixture(scope="module")
def urdf_path() -> Path:
    assert URDF.exists(), f"Panda URDF missing at {URDF}"
    return URDF


# --------------------------------------------------------------------- health
def test_health_is_not_ready_before_any_data_arrives():
    health = BridgeHealth()
    health.panda_loaded = True
    assert not health.ready(), "a loaded model with no streams is not a working demo"


def test_health_requires_every_stream_including_the_served_page():
    health = BridgeHealth(panda_loaded=True, joint_stream=True, camera_stream=True)
    assert not health.ready(), "streams alone do not prove anyone can open the viewer"
    health.viewer_served = True
    assert health.ready()


def test_health_report_is_json_serialisable():
    report = BridgeHealth(panda_loaded=True).as_dict()
    assert json.loads(json.dumps(report))["panda_loaded"] is True


def test_joint_reader_parses_gazebo_json_and_tracks_sim_time():
    health = BridgeHealth()
    reader = JointStateReader("/unused", health)
    reader._consume(
        json.dumps(
            {
                "header": {"stamp": {"sec": 12, "nsec": 500_000_000}},
                # Real Gazebo JointState JSON nests angles under
                # joint[].axis1.position. A flat name/position guess (which the
                # code once made) reads zeros from this shape, so the fixture
                # uses the true nesting.
                "joint": [
                    {"name": name, "axis1": {"position": position}}
                    for name, position in zip(
                        list(JOINT_NAMES),
                        [0.1, 0.2, 0.3, -1.57, 0.5, 0.8, 0.7, 0.02, 0.02],
                        strict=True,
                    )
                ],
            }
        )
    )
    positions, sim_time = reader.latest()
    assert positions["joint4"] == pytest.approx(-1.57)
    assert positions["left_finger"] == pytest.approx(0.02)
    assert sim_time == pytest.approx(12.5)
    assert health.joint_stream is True
    assert health.joint_messages == 1


def test_joint_reader_rejects_malformed_documents():
    health = BridgeHealth()
    reader = JointStateReader("/unused", health)
    reader._consume("not json")
    positions, _ = reader.latest()
    assert all(value == 0.0 for value in positions.values())
    assert health.joint_stream is False


def test_joint_reader_keeps_only_known_joints():
    health = BridgeHealth()
    reader = JointStateReader("/unused", health)
    reader._consume(json.dumps({"joint": [{"name": "unexpected", "axis1": {"position": 9.9}}]}))
    positions, _ = reader.latest()
    assert "unexpected" not in positions
    assert len(positions) == len(JOINT_NAMES)


# --------------------------------------------------------------- kinematics
def test_kinematics_exposes_every_panda_joint(urdf_path: Path):
    kinematics = robot_kinematics(urdf_path)
    names = {joint["name"] for joint in kinematics["joints"]}
    # The nine actuated joints are all present; the model also carries a fixed
    # hand joint, which is kept so the finger frames resolve through it.
    assert set(JOINT_NAMES) <= names
    hand = next(j for j in kinematics["joints"] if j["name"] == "hand_joint")
    assert hand["type"] == "fixed"


def test_kinematics_carries_axes_and_limits_for_rendering(urdf_path: Path):
    kinematics = robot_kinematics(urdf_path)
    joint4 = next(j for j in kinematics["joints"] if j["name"] == "joint4")
    # These bounds are the reason the neutral pose is -1.57 rad, not 0: joint4
    # cannot reach 0.0 at all.
    assert joint4["lower"] == pytest.approx(-3.0718)
    assert joint4["upper"] == pytest.approx(-0.0698)
    assert joint4["upper"] > 0.0 or joint4["upper"] < 0.0
    assert joint4["type"] == "revolute"


# ------------------------------------------------------------------ driver
def test_motion_targets_stay_inside_the_urdf_limits(urdf_path: Path):
    limits = joint_limits(str(urdf_path))
    demo = MotionDemo(limits=limits)
    for joint, (lower, upper) in limits.items():
        targets = [demo.target(joint, t / 10.0) for t in range(400)]
        assert min(targets) >= lower, f"{joint} commanded below its lower limit"
        assert max(targets) <= upper, f"{joint} commanded above its upper limit"


def test_motion_driver_actually_sweeps_each_joint(urdf_path: Path):
    limits = joint_limits(str(urdf_path))
    demo = MotionDemo(limits=limits)
    for joint in JOINT_TARGETS:
        targets = [demo.target(joint, t / 10.0) for t in range(400)]
        lower, upper = limits[joint]
        # A finger travels 40 mm while a shoulder sweeps radians, so the check is
        # against each joint's own range rather than an absolute angle. The bar is
        # 10%, not more: the Panda's joint ranges are ~5.8 rad wide, so a
        # deliberate demo sweep is a fraction of one, and a joint that is truly
        # static still sits at 0%.
        travelled = (max(targets) - min(targets)) / (upper - lower)
        assert travelled > 0.10, f"{joint} barely moves ({travelled:.1%} of range)"


def test_joint4_neutral_is_negative_because_its_range_is(urdf_path: Path):
    # Guards the earlier mistake: commanding joint4 to 0.6 rad pinned it on its
    # upper limit and looked like a controller fault.
    demo = MotionDemo(limits=joint_limits(str(urdf_path)))
    assert demo.target("joint4", 0.0) < 0.0
    assert JOINT_TARGETS["joint4"].neutral < 0.0
