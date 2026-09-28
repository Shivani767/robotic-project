"""Scenario, policy and experiment schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robot_ci.models.environment import TaskName
from robot_ci.models.geometry import Vec3
from robot_ci.models.robot import ArmModel, RobotConfig
from robot_ci.models.scenario import (
    ExperimentConfig,
    ObjectSpec,
    PolicyComponent,
    PolicyConfig,
    ScenarioConfig,
)
from robot_ci.models.sensors import CameraIntrinsics, DepthCorruptionConfig, SensorBundleConfig


def test_default_scenario_has_one_target() -> None:
    scenario = ScenarioConfig()
    assert len([o for o in scenario.objects if o.is_target]) == 1
    assert scenario.target.display_name == "red_cube"


def test_unknown_field_is_rejected() -> None:
    """A typo in YAML must be an error, not a silently ignored setting."""
    with pytest.raises(ValidationError):
        ScenarioConfig(typo_field=1)


def test_two_targets_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one target"):
        ScenarioConfig(
            objects=(
                ObjectSpec(id="a", is_target=True),
                ObjectSpec(id="b", is_target=True),
            )
        )


def test_config_hash_is_stable_and_sensitive() -> None:
    a = ScenarioConfig()
    b = ScenarioConfig()
    assert a.config_hash() == b.config_hash()
    assert a.config_hash().startswith("sha256:")


def test_config_hash_changes_with_conditions() -> None:
    a = ScenarioConfig()
    b = ScenarioConfig(sensors=SensorBundleConfig(depth=DepthCorruptionConfig(enabled=True, sigma_mm=5.0)))
    assert a.config_hash() != b.config_hash()


def test_config_hash_changes_with_embodiment() -> None:
    a = ScenarioConfig(robot=RobotConfig.panda())
    b = ScenarioConfig(robot=RobotConfig.ur5e())
    assert a.config_hash() != b.config_hash()


def test_describe_conditions_reports_corruption() -> None:
    scenario = ScenarioConfig(
        sensors=SensorBundleConfig(depth=DepthCorruptionConfig(enabled=True, sigma_mm=12.0))
    )
    assert scenario.describe_conditions()["depth_noise_mm"] == 12.0


def test_policy_differs_from() -> None:
    a = PolicyConfig(name="v1", components={PolicyComponent.PERCEPTION: "p1", PolicyComponent.PLANNER: "pl1"})
    b = PolicyConfig(name="v2", components={PolicyComponent.PERCEPTION: "p2", PolicyComponent.PLANNER: "pl1"})
    assert a.differs_from(b) == (PolicyComponent.PERCEPTION,)


def test_policy_component_swap() -> None:
    b = PolicyConfig(name="v2", components={PolicyComponent.PERCEPTION: "p2"})
    hybrid = b.with_component(PolicyComponent.PERCEPTION, "p1")
    assert hybrid.components[PolicyComponent.PERCEPTION] == "p1"
    assert b.components[PolicyComponent.PERCEPTION] == "p2"  # original untouched


def test_experiment_rejects_identical_policies() -> None:
    with pytest.raises(ValidationError, match="vacuous"):
        ExperimentConfig(baseline=PolicyConfig(name="same"), candidate=PolicyConfig(name="same"))


def test_experiment_grid_and_episode_counts() -> None:
    config = ExperimentConfig(repetitions=5)
    from robot_ci.models.scenario import VariableSpec

    with_var = config.model_copy(
        update={"variables": (VariableSpec(name="d", path="sensors.depth.sigma_mm", values=(0.0, 5.0, 10.0)),)}
    )
    assert with_var.grid_size == 3
    assert with_var.total_episodes == 15
    assert with_var.total_episodes_both_policies == 30


def test_ur5e_has_six_joints() -> None:
    robot = RobotConfig.ur5e()
    assert robot.dof == 6
    assert len(robot.arm_joint_names) == 6
    assert robot.model is ArmModel.UR5E


def test_home_position_must_respect_limits() -> None:
    with pytest.raises(ValidationError):
        RobotConfig.panda(home_position=(99.0,) * 7)


def test_robot_random_configuration_is_within_limits() -> None:
    import numpy as np

    robot = RobotConfig.panda()
    rng = np.random.default_rng(0)
    assert robot.limits.within(robot.random_configuration(rng))


def test_explicit_object_position_is_preserved() -> None:
    spec = ObjectSpec(id="c", position=Vec3.of(0.3, 0.1, 0.025))
    scenario = ScenarioConfig(objects=(spec,))
    assert scenario.target.position is not None
    assert scenario.target.position.array.tolist() == [0.3, 0.1, 0.025]


def test_resolution_follows_intrinsics() -> None:
    scenario = ScenarioConfig(sensors=SensorBundleConfig(intrinsics=CameraIntrinsics(width_px=64, height_px=48)))
    assert scenario.resolution == (48, 64)


def test_task_name_roundtrip() -> None:
    scenario = ScenarioConfig(task={"name": "grasp_object"})
    assert scenario.task.name is TaskName.GRASP_OBJECT
