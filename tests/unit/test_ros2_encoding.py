"""Tests for the ROS 2 encoding layer.

These run **without ROS 2 installed**, which is the point: the conversion
arithmetic is where a silent unit or axis-order bug would hide, and it must be
covered by the ordinary suite rather than only by a runtime test that most
machines skip. The rclpy-dependent behaviour is covered by
``tests/integration/test_ros2_runtime.py``, which skips explicitly when ROS 2 is
absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from robot_ci.models.geometry import Pose, Quaternion, Vec3
from robot_ci.models.state import Action, GripperState, Observation, RobotState, SensorFrame
from robot_ci.ros2 import encoding
from robot_ci.ros2.topics import TOPICS


# -- time ---------------------------------------------------------------------
def test_sim_time_converts_to_nanoseconds() -> None:
    """MuJoCo seconds become ROS nanoseconds on the simulation clock."""
    assert encoding.sim_time_to_ns(0.0) == 0
    assert encoding.sim_time_to_ns(1.0) == 1_000_000_000
    assert encoding.sim_time_to_ns(0.05) == 50_000_000


def test_sim_time_is_monotonic_across_a_control_timestep() -> None:
    """Stamps never go backwards as the episode clock advances.

    This is the property that makes a subscriber's ordering check meaningful; a
    non-monotonic stamp would look like a clock bug and be diagnosed as physics.
    """
    stamps = [encoding.sim_time_to_ns(i * 0.05) for i in range(200)]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_negative_time_is_clamped_not_wrapped() -> None:
    """A tiny negative value from floating point clamps rather than going negative."""
    assert encoding.sim_time_to_ns(-1e-12) == 0


def test_stamp_round_trips() -> None:
    """Converting a stamp back to seconds returns the original value."""
    for value in (0.0, 0.05, 3.25, 76 * 0.05):
        assert encoding.stamp_to_sim_time(encoding.sim_time_to_ns(value)) == pytest.approx(
            value, abs=1e-9
        )


# -- payload encoding ---------------------------------------------------------
def test_payload_encoding_is_canonical_and_stable() -> None:
    """The same payload encodes byte-identically regardless of key order.

    A recorded bag is only diffable if identical observations produce identical
    messages, which is what `sort_keys` buys.
    """
    first = encoding.encode_payload({"b": 1, "a": 2})
    second = encoding.encode_payload({"a": 2, "b": 1})
    assert first == second
    assert encoding.decode_payload(first) == {"a": 2, "b": 1}


def sample_observation() -> Observation:
    """Build a small but complete observation."""
    return Observation(
        step_index=7,
        timestamp_s=0.35,
        robot=RobotState(
            joint_positions=(0.1, -0.2, 0.3),
            joint_velocities=(0.01, 0.02, 0.03),
            joint_names=("panda_joint1", "panda_joint2", "panda_joint3"),
            end_effector_pose=Pose(
                position=Vec3.of(0.4, 0.1, 0.5),
                orientation=Quaternion(w=0.0, x=0.0, y=1.0, z=0.0),
            ),
            gripper=GripperState(width_m=0.04, max_width_m=0.08, is_grasping=False),
            timestamp_s=0.35,
        ),
        seed=3,
        scenario_id="nominal",
        sensor_frame=SensorFrame(
            camera_id="rgb_camera",
            timestamp_s=0.35,
            width_px=4,
            height_px=3,
            intrinsics=(1.0, 1.0, 2.0, 1.5),
            valid=True,
        ),
    )


def test_observation_round_trips_through_the_payload() -> None:
    """An observation survives encode/decode with its semantics intact.

    `sensor_frame` carries metadata only -- pixels travel on the Image topics --
    so the assertion is on the fields a policy reads, and on the `valid` flag
    that distinguishes a dropped frame from a missing topic.
    """
    original = sample_observation()
    payload = encoding.observation_to_payload(original)
    restored = encoding.observation_from_payload(payload)

    assert restored.step_index == original.step_index
    assert restored.seed == original.seed
    assert restored.scenario_id == original.scenario_id
    assert restored.timestamp_s == pytest.approx(original.timestamp_s)
    assert np.allclose(restored.robot.joint_positions, original.robot.joint_positions)
    assert restored.robot.joint_names == original.robot.joint_names
    assert restored.robot.gripper.is_grasping is False
    assert restored.sensor_frame is not None
    assert restored.sensor_frame.valid is True
    assert restored.sensor_frame.width_px == 4
    assert restored.sensor_frame.intrinsics == (1.0, 1.0, 2.0, 1.5)


def test_observation_payload_is_json_serializable() -> None:
    """The payload contains no NumPy arrays, which would not survive JSON."""
    text = encoding.encode_payload(encoding.observation_to_payload(sample_observation()))
    assert isinstance(json.loads(text), dict)
    assert "sensor_frame" in text


def test_quaternion_is_reordered_from_w_first_to_ros_xyz_first() -> None:
    """ROBOT CI stores (w,x,y,z); ROS expects (x,y,z,w).

    A quarter-turn about Y is the case that makes the bug visible: swapping the
    order yields a non-identity rotation from an identity input.
    """
    rotation = Quaternion(w=0.0, x=0.0, y=1.0, z=0.0)
    x, y, z, w = encoding.quaternion_to_ros(rotation)
    assert (x, y, z, w) == (0.0, 1.0, 0.0, 0.0)
    assert encoding.quaternion_from_ros(x, y, z, w) == {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}


def test_identity_quaternion_survives_the_reorder() -> None:
    """Identity must stay identity; anything else means the order is wrong."""
    x, y, z, w = encoding.quaternion_to_ros(Quaternion.identity())
    assert (x, y, z, w) == (0.0, 0.0, 0.0, 1.0)


# -- joint states -------------------------------------------------------------
def test_joint_state_fields_use_provided_names() -> None:
    """Names, positions and velocities line up for a sensor_msgs/JointState."""
    state = RobotState(
        joint_positions=(0.1, 0.2), joint_velocities=(1.0, 2.0), joint_names=("a", "b")
    )
    fields = encoding.joint_state_fields(state)
    assert fields["name"] == ["a", "b"]
    assert fields["position"] == [0.1, 0.2]
    assert fields["velocity"] == [1.0, 2.0]


def test_joint_state_fields_synthesize_missing_names() -> None:
    """A state with no names gets Panda-consistent ones rather than blanks."""
    fields = encoding.joint_state_fields(RobotState(joint_positions=(0.0, 0.0, 0.0)))
    assert fields["name"] == ["panda_joint1", "panda_joint2", "panda_joint3"]


def test_joint_state_fields_reject_a_name_position_mismatch() -> None:
    """Mismatched lengths are an error, not something to pad or truncate.

    ``RobotState`` already refuses to be constructed with mismatched names, so
    this exercises the adapter's own guard rather than the model's: the adapter
    is handed whatever a backend returns, and a ROS consumer would reject a
    JointState whose arrays disagree. Failing here names the actual cause
    instead of producing a topic that silently drops joints.
    """
    from types import SimpleNamespace

    state = SimpleNamespace(
        joint_positions=(0.1, 0.2),
        joint_names=("only_one",),
        joint_velocities=(),
        joint_efforts=(),
    )
    with pytest.raises(ValueError, match="length mismatch"):
        encoding.joint_state_fields(state)


# -- actions ------------------------------------------------------------------
def test_action_round_trips_through_the_payload() -> None:
    """A joint-position command survives the wire encoding unchanged."""
    action = Action.joint_positions((0.11, -0.21, 0.31))
    restored = encoding.action_from_payload(encoding.action_to_payload(action))
    assert restored.type is action.type
    assert np.allclose(restored.joint_position, action.joint_position)


def test_malformed_action_is_rejected_on_receipt() -> None:
    """A command missing its payload fails validation rather than being applied."""
    with pytest.raises(ValidationError):
        encoding.action_from_payload({"type": "joint_position"})


# -- image buffers ------------------------------------------------------------
def test_rgb_buffer_is_contiguous_uint8() -> None:
    """The RGB buffer is C-contiguous uint8, as sensor_msgs/Image requires."""
    rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    buffer = encoding.rgb_to_bytes(rgb)
    assert isinstance(buffer, bytes)
    assert len(buffer) == rgb.size
    assert np.frombuffer(buffer, dtype=np.uint8).reshape(rgb.shape).tolist() == rgb.tolist()


def test_depth_buffer_stays_float32_in_metres() -> None:
    """Depth is not rescaled for display; a unit change here would corrupt it."""
    depth = np.array([[0.5, 0.25]], dtype=np.float64)
    buffer = encoding.depth_to_float32_bytes(depth)
    restored = np.frombuffer(buffer, dtype=np.float32).reshape(1, 2)
    assert np.allclose(restored, np.array([[0.5, 0.25]], dtype=np.float32))
    assert restored.dtype == np.float32


# -- topic graph --------------------------------------------------------------
def test_every_topic_is_absolute_and_namespaced() -> None:
    """Topics are absolute so a namespace cannot silently redirect them."""
    for name in TOPICS.all():
        assert name.startswith("/robot_ci/") or name == "/clock"


def test_topics_are_unique() -> None:
    """No two roles share a name, which would make the graph ambiguous."""
    names = TOPICS.all()
    assert len(names) == len(set(names))


# -- dependency behaviour -----------------------------------------------------
def test_importing_robot_ci_does_not_require_ros() -> None:
    """A checkout with no ROS 2 behaves exactly as it did before.

    This is the guarantee that makes the integration additive rather than a new
    dependency: nothing in the package imports ``rclpy`` at module scope, so a
    machine without ROS can still run every existing experiment.
    """
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import robot_ci, robot_ci.ros2, robot_ci.ros2.encoding;"
            " assert 'rclpy' not in sys.modules, 'rclpy was imported';"
            " print('clean')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "clean" in completed.stdout


def test_detect_reports_a_reason_when_ros_is_missing() -> None:
    """Detection explains itself, and never claims availability falsely."""
    from robot_ci.ros2.dependencies import detect

    environment = detect()
    assert isinstance(environment.available, bool)
    if not environment.available:
        assert environment.reason, "an unavailable environment must say why"
        assert environment.missing


def test_require_rclpy_raises_with_an_actionable_hint() -> None:
    """`require_rclpy` fails loudly, naming the distro and the install command."""
    from robot_ci.core.errors import BackendUnavailableError
    from robot_ci.ros2.dependencies import detect, require_rclpy

    if detect().available:
        assert require_rclpy().available is True
        return
    with pytest.raises(BackendUnavailableError) as excinfo:
        require_rclpy()
    assert excinfo.value.code == "ros2.unavailable"
    hint = excinfo.value.details["hint"]
    assert "ros-jazzy-ros-base" in hint
    assert "Humble is not supported" in hint


def test_build_scenario_does_not_rewrite_the_scenario_id() -> None:
    """Sweeping depth noise must not rename the condition.

    `scenario_id` seeds the scene's object placement, so rewriting it to label a
    ROS 2 run would move the target and make the run incomparable with the
    direct one. This is the test that would catch a regression of exactly that
    bug.
    """
    from robot_ci.ros2.execution import build_scenario

    nominal = build_scenario()
    noisy = build_scenario(8.0)
    assert nominal.scenario_id == noisy.scenario_id
    assert noisy.sensors.depth.sigma_mm == 8.0
    assert nominal.sensors.depth.sigma_mm == 0.0


def test_build_scenario_matches_the_direct_experiment_helper() -> None:
    """The ROS 2 path resolves the scenario exactly as the direct path does."""
    from robot_ci.ros2.execution import build_scenario
    from robot_ci.scenarios import default_scenario, set_by_path

    assert build_scenario().config_hash() == default_scenario().config_hash()
    expected = set_by_path(default_scenario(), "sensors.depth.sigma_mm", 8.0)
    assert build_scenario(8.0).config_hash() == expected.config_hash()
