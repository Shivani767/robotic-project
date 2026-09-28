"""Integration tests for the MuJoCo backend.

These require MuJoCo to be installed and exercise the real physics + rendering
path. They are marked ``integration`` because they are slower than unit tests,
but they still run on a stock MacBook CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ci.core.errors import SimulationError
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import Action
from robot_ci.simulation import create_backend
from robot_ci.simulation.mujoco import MuJoCoBackend
from robot_ci.simulation.mujoco_env import build_scene_xml, scene_cache_key

pytestmark = pytest.mark.integration

mujoco = pytest.importorskip("mujoco", reason="MuJoCo is not installed")


@pytest.fixture
def mj_backend() -> MuJoCoBackend:
    backend = MuJoCoBackend()
    yield backend
    backend.close_all()


def test_generated_mjcf_is_valid_xml() -> None:
    import xml.etree.ElementTree as ET

    xml = build_scene_xml(ScenarioConfig())
    ET.fromstring(xml)


def test_mujoco_compiles_the_generated_scene() -> None:
    xml = build_scene_xml(ScenarioConfig())
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nu == 9, "7 arm joints + 2 finger actuators"
    assert model.nq >= 7


def test_scene_cache_key_changes_with_conditions() -> None:
    a = ScenarioConfig()
    b = ScenarioConfig(environment={"lighting_lux": 100.0})
    assert scene_cache_key(a) != scene_cache_key(b)


def test_scene_cache_key_ignores_rgb_corruption() -> None:
    """A change that does not alter the MJCF should reuse the compiled model."""
    a = ScenarioConfig()
    b = ScenarioConfig(sensors={"rgb": {"enabled": True, "occlusion_percent": 50.0}})
    assert scene_cache_key(a) == scene_cache_key(b)


def test_object_rests_on_the_table(mj_backend: MuJoCoBackend) -> None:
    handle = mj_backend.reset(ScenarioConfig(), seed=5)
    target = mj_backend.ground_truth(handle).target_object
    assert target is not None
    assert target.pose.position.z == pytest.approx(0.025, abs=0.01)


def test_render_produces_a_non_blank_image(mj_backend: MuJoCoBackend) -> None:
    scenario = ScenarioConfig()
    handle = mj_backend.reset(scenario, seed=5)
    frame = mj_backend.render(handle)
    rgb = frame.rgb_array()
    depth = frame.depth_array()
    assert rgb is not None and rgb.shape == (scenario.resolution[0], scenario.resolution[1], 3)
    assert rgb.dtype == np.uint8
    assert float(rgb.std()) > 5.0, "rendered image is blank"
    assert depth is not None
    valid = np.isfinite(depth)
    assert valid.mean() > 0.9
    assert float(np.nanmax(depth) - np.nanmin(depth)) > 0.1


def test_determinism_across_instances() -> None:
    scenario = ScenarioConfig()
    a = create_backend("mujoco")
    b = create_backend("mujoco")
    try:
        frame_a = a.render(a.reset(scenario, seed=11))
        frame_b = b.render(b.reset(scenario, seed=11))
        assert np.allclose(a_depth := frame_a.depth_array(), b_depth := frame_b.depth_array(), equal_nan=True)
        assert a_depth is not None and b_depth is not None
    finally:
        a.close_all()
        b.close_all()


def test_seed_changes_the_scene() -> None:
    scenario = ScenarioConfig()
    backend = create_backend("mujoco")
    try:
        p1 = backend.ground_truth(backend.reset(scenario, seed=1)).target_object.pose.position
        p2 = backend.ground_truth(backend.reset(scenario, seed=2)).target_object.pose.position
        assert p1 != p2
    finally:
        backend.close_all()


def test_joint_positions_are_arm_joints_not_free_joints(mj_backend: MuJoCoBackend) -> None:
    """Regression guard: qpos[:dof] once read the object's free joint."""
    scenario = ScenarioConfig()
    handle = mj_backend.reset(scenario, seed=5)
    observed = mj_backend.observe(handle).robot.joint_positions
    truth = mj_backend.ground_truth(handle).true_joint_positions
    assert np.allclose(observed, truth)
    # Home configuration, after gravity settling.
    assert abs(observed[0]) < 0.5
    assert abs(observed[1] - scenario.robot.home_position[1]) < 0.5


def test_stepping_advances_physics(mj_backend: MuJoCoBackend) -> None:
    scenario = ScenarioConfig()
    handle = mj_backend.reset(scenario, seed=5)
    before = mj_backend.ground_truth(handle).true_ee_pose.position
    for _ in range(20):
        mj_backend.step(handle, Action.joint_positions(scenario.robot.home_position))
    after = mj_backend.ground_truth(handle).true_ee_pose.position
    assert handle.step_index == 20
    assert before.distance_to(after) < 0.05, "arm should hold station under position control"


def test_gripper_command_changes_aperture(mj_backend: MuJoCoBackend) -> None:
    scenario = ScenarioConfig()
    handle = mj_backend.reset(scenario, seed=5)
    wide = mj_backend.observe(handle).robot.gripper.width_m
    for _ in range(30):
        mj_backend.step(handle, Action.gripper(0.01))
    narrow = mj_backend.observe(handle).robot.gripper.width_m
    assert narrow < wide


def test_lighting_changes_the_render(mj_backend: MuJoCoBackend) -> None:
    """A dimmer scene must render dimmer, otherwise B8 measures nothing."""
    dark = mj_backend.render(mj_backend.reset(ScenarioConfig(environment={"lighting_lux": 30.0}), seed=5))
    bright = mj_backend.render(mj_backend.reset(ScenarioConfig(environment={"lighting_lux": 2000.0}), seed=5))
    dark_mean = float(dark.rgb_array().mean())
    bright_mean = float(bright.rgb_array().mean())
    assert dark_mean < bright_mean, f"dark={dark_mean:.1f} bright={bright_mean:.1f}"


def test_camera_latency_config_does_not_break_rendering(mj_backend: MuJoCoBackend) -> None:
    scenario = ScenarioConfig(
        sensors={"camera": {"enabled": True, "latency": {"enabled": True, "delay_ms": 50.0}}}
    )
    handle = mj_backend.reset(scenario, seed=5)
    assert mj_backend.render(handle).has_pixels()


def test_unsupported_dof_is_rejected_with_a_clear_message(mj_backend: MuJoCoBackend) -> None:
    from robot_ci.models.robot import RobotConfig

    scenario = ScenarioConfig(robot=RobotConfig.ur5e())
    with pytest.raises(SimulationError, match="embodiment"):
        mj_backend.reset(scenario, seed=1)


def test_ground_truth_differs_from_belief_structure(mj_backend: MuJoCoBackend) -> None:
    """Truth carries the target flag and grasp state; belief does not."""
    handle = mj_backend.reset(ScenarioConfig(), seed=5)
    truth = mj_backend.ground_truth(handle)
    belief = mj_backend.observe(handle)
    assert truth.target_object is not None and truth.target_object.is_target
    assert all("is_target" not in obj.model_dump() for obj in belief.objects)
