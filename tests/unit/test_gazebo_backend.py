"""Tests for the Gazebo backend's logic that needs no Gazebo runtime.

The parts exercised here are the ones that would otherwise only be tested by
running an episode: action decoding, the joint-mapping contract, capability
reporting, and the depth-perception maths fed a synthetic point cloud. A
backend that only worked when someone watched a real episode would be untested
on every machine that is not the container, which is most of them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from robot_ci.core.errors import BackendUnavailableError
from robot_ci.models.state import Action, ActionType
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import SimulatorBackend, create_backend, probe_backend
from robot_ci.simulation.gazebo import ARM_JOINTS, FINGER_JOINTS, GazeboBackend

URDF = Path(__file__).resolve().parents[2] / "gazebo" / "models" / "panda" / "panda.urdf"


# ------------------------------------------------------------- registration
def test_gazebo_is_registered_and_resolves_to_the_real_class() -> None:
    """The registry's gazebo entry must now import, not raise."""
    backend = create_backend("gazebo")
    try:
        assert isinstance(backend, GazeboBackend)
        assert isinstance(backend, SimulatorBackend)
    finally:
        backend.close_all()


def test_probe_reports_constructible_even_without_a_gazebo_runtime() -> None:
    """Constructible and runnable are different facts and must not be merged.

    ``GazeboBackend`` constructs anywhere. Only running it needs Gazebo. A
    machine without Gazebo must still be able to see the backend's capabilities,
    which is what makes ``robot-ci validate`` useful before a container exists.
    """
    result = probe_backend("gazebo")
    assert result["available"] is True
    assert result["capabilities"] is not None
    if not result["usable"]:
        assert result["reason"]


def test_unknown_options_are_rejected_rather_than_ignored() -> None:
    """A silently ignored option is a configuration error that surfaces later."""
    with pytest.raises(BackendUnavailableError):
        GazeboBackend(nonexistent_option=1)


def test_capabilities_describe_real_gazebo_not_a_wish_list() -> None:
    """The capability flags are what let the engine refuse bad configurations."""
    caps = GazeboBackend().capabilities()
    assert caps.physics and caps.collision, "Gazebo runs contact dynamics"
    assert caps.rgb and caps.depth, "the generated world has an rgbd_camera"
    assert caps.segmentation is False, "no segmentation sensor is generated"
    assert caps.requires_gpu is False, "the demo runs on CPU-only containers"
    assert caps.deterministic is False, "ODE integrates against wall-clock"


# ------------------------------------------------------------ action decode
def test_joint_position_action_decodes_to_targets_and_aperture() -> None:
    joints, width = GazeboBackend._decode(
        Action.joint_positions([0.1, 0.2, 0.3, -1.5, 0.4, 0.5, 0.6])
    )
    assert joints == [pytest.approx(v) for v in (0.1, 0.2, 0.3, -1.5, 0.4, 0.5, 0.6)]
    assert width is None, "no aperture was commanded, so none is invented"


def test_arm_and_gripper_action_carries_both_channels() -> None:
    action = Action(
        type=ActionType.ARM_AND_GRIPPER,
        joint_position=(0.0,) * 7,
        gripper_width_m=0.04,
    )
    joints, width = GazeboBackend._decode(action)
    assert len(joints) == 7
    assert width == pytest.approx(0.04)


def test_gripper_only_action_leaves_arm_targets_to_the_backend() -> None:
    joints, width = GazeboBackend._decode(Action(type=ActionType.GRIPPER, gripper_width_m=0.02))
    assert joints == []
    assert width == pytest.approx(0.02)


def test_an_action_the_backend_cannot_execute_is_refused_by_name() -> None:
    """Cartesian deltas need a Jacobian this backend does not provide."""
    action = Action(type=ActionType.CARTESIAN_DELTA, cartesian_delta=(0.0,) * 6)
    with pytest.raises(Exception) as excinfo:
        GazeboBackend._decode(action)
    assert "cartesian_delta" in str(excinfo.value)


# ------------------------------------------------------- joint name contract
def test_arm_joint_names_match_the_scenario_robot() -> None:
    """The command mapping must be the robot the scenario actually describes.

    The scenario's arm order is what a policy's joint vector is indexed by, so
    a mismatch here would silently command the wrong joints rather than fail.
    """
    assert list(ARM_JOINTS) == list(default_scenario().robot.all_joint_names[:7])
    assert list(FINGER_JOINTS) == list(default_scenario().robot.all_joint_names[7:])


# --------------------------------------------------------------- perception
def test_largest_cluster_picks_the_bigger_of_two_blobs() -> None:
    """Perception must choose the object, not a stray smaller point."""
    small = np.array([[0.0, 0.0, 0.02], [0.01, 0.0, 0.02], [0.0, 0.01, 0.02]])
    big = np.array([[0.2, 0.2, 0.03]] * 40)
    cluster = GazeboBackend._largest_cluster(np.vstack([small, big]))
    assert cluster.shape[0] == 40
    assert cluster[:, 0].mean() == pytest.approx(0.2)


def test_perception_reports_nothing_when_there_is_no_depth() -> None:
    """Silence must produce no belief, not a default one.

    An object reported when nothing was seen would be a fabricated target for
    the policy to aim at, which is the exact failure the ground-truth boundary
    in :class:`Observation` exists to prevent.
    """
    backend = GazeboBackend()
    depth = type("D", (), {"latest": lambda _self: (None, None, 0, 0)})()
    backend._episodes["fake"] = {"depth": depth}
    handle = type(
        "H",
        (),
        {
            "episode_id": "fake",
            "scenario": default_scenario(),
            "step_index": 0,
            "sim_time_s": 0.0,
        },
    )()
    try:
        assert backend._detect_objects(handle) == ()
    finally:
        backend._episodes.clear()


def test_camera_geometry_is_read_from_the_same_helper_that_writes_the_sdf() -> None:
    """Perception must not drift from the pose Gazebo was actually given."""
    from robot_ci.gazebo.world import _camera_pose

    scenario = default_scenario()
    eye, forward, right, up = GazeboBackend()._camera_geometry(scenario)
    x, y, z, _roll, _pitch, _yaw = (float(v) for v in _camera_pose(scenario).split())
    assert (eye[0], eye[1], eye[2]) == pytest.approx((x, y, z))
    # The basis must be orthonormal: a broken basis shears the back-projection
    # and quietly biases every belief the policy acts on.
    basis = np.vstack([forward, right, up])
    assert basis @ basis.T == pytest.approx(np.eye(3), abs=1e-9)


# ------------------------------------------------------------------ scenario
@pytest.mark.skipif(
    shutil.which("gz") is None,
    reason=(
        "generating a world needs Gazebo's own URDF converter ('gz sdf -p'), "
        "which only exists in the container. Run in Docker: "
        "docker compose exec gazebo-demo pytest tests/unit"
    ),
)
def test_generated_world_carries_a_nine_joint_robot() -> None:
    """The backend's own world must contain a commandable robot."""
    from robot_ci.gazebo.world import world_sdf

    sdf = world_sdf(default_scenario(), robot_urdf=str(URDF))
    assert sdf.count("JointPositionController") == 9
    assert "JointStatePublisher" in sdf


@pytest.mark.skipif(
    shutil.which("gz") is None,
    reason=(
        "generating a world needs Gazebo's own URDF converter ('gz sdf -p'), "
        "which only exists in the container. Run in Docker: "
        "docker compose exec gazebo-demo pytest tests/unit"
    ),
)
def test_depth_noise_condition_changes_the_generated_world() -> None:
    """0 mm and 8 mm must be different worlds, not one relabelled."""
    from robot_ci.gazebo.world import world_sdf

    nominal = world_sdf(default_scenario(), robot_urdf=str(URDF))
    noisy = world_sdf(
        set_by_path(default_scenario(), "sensors.depth.sigma_mm", 8.0),
        robot_urdf=str(URDF),
    )
    assert nominal != noisy
