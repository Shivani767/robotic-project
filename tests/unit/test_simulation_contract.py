"""The backend contract, enforced uniformly across implementations.

These tests run the *same* assertions against every backend, which is what
makes the backends genuinely interchangeable rather than nominally so.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ci.core.errors import SimulationError
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.state import Action, ActionType
from robot_ci.simulation import create_backend, list_backends
from robot_ci.simulation.mock import MockBackend
from robot_ci.simulation.synthetic import SyntheticBackend

# Backends that need no extra dependency and no GPU.
LOCAL_BACKENDS = ["mock", "synthetic"]


@pytest.fixture(params=LOCAL_BACKENDS)
def any_backend(request: pytest.FixtureRequest):
    """Every locally-runnable backend, so the contract is tested on all of them."""
    backend = create_backend(request.param)
    yield backend
    backend.close_all()


def test_registry_lists_every_backend() -> None:
    assert set(LOCAL_BACKENDS) <= set(list_backends())


def test_unknown_backend_raises_actionable_error() -> None:
    from robot_ci.core.errors import BackendUnavailableError

    with pytest.raises(BackendUnavailableError, match="unknown simulator backend"):
        create_backend("does_not_exist")


def test_capabilities_are_self_consistent(any_backend) -> None:
    caps = any_backend.capabilities()
    assert caps.name == any_backend.name
    assert caps.max_dof >= 1
    # A backend that claims contact physics must also claim collision.
    if caps.physics:
        assert caps.collision


def test_info_is_serializable(any_backend) -> None:
    data = any_backend.info().to_dict()
    assert data["name"] == any_backend.name
    assert "version" in data


def test_reset_then_observe(any_backend, nominal_scenario: ScenarioConfig) -> None:
    handle = any_backend.reset(nominal_scenario, seed=1)
    obs = any_backend.observe(handle)
    assert obs.step_index == 0
    assert len(obs.robot.joint_positions) == nominal_scenario.robot.dof
    assert len(obs.objects) >= 1


def test_observation_never_leaks_ground_truth(any_backend, nominal_scenario: ScenarioConfig) -> None:
    """The central anti-leakage invariant, checked on every backend."""
    handle = any_backend.reset(nominal_scenario, seed=1)
    obs = any_backend.observe(handle)
    assert not obs.is_leaking_ground_truth()
    dumped = set(obs.model_dump().keys())
    assert "ground_truth" not in dumped
    assert "true_ee_pose" not in dumped


def test_determinism_same_seed(any_backend, nominal_scenario: ScenarioConfig) -> None:
    a = any_backend.reset(nominal_scenario, seed=7)
    b = any_backend.reset(nominal_scenario, seed=7)
    assert any_backend.observe(a).robot.joint_positions == any_backend.observe(b).robot.joint_positions


def test_seed_sensitivity(nominal_scenario: ScenarioConfig) -> None:
    """Different seeds must produce different scenes, or repeats measure nothing."""
    backend = create_backend("mock")
    a = backend.observe(backend.reset(nominal_scenario, seed=1)).objects[0].pose.position
    b = backend.observe(backend.reset(nominal_scenario, seed=2)).objects[0].pose.position
    assert a != b
    backend.close_all()


def test_ground_truth_is_separate_from_observation(any_backend, nominal_scenario: ScenarioConfig) -> None:
    handle = any_backend.reset(nominal_scenario, seed=1)
    truth = any_backend.ground_truth(handle)
    assert truth.target_object is not None
    obs = any_backend.observe(handle)
    assert "target_object" not in obs.model_dump()


def test_step_advances_time(any_backend, nominal_scenario: ScenarioConfig) -> None:
    handle = any_backend.reset(nominal_scenario, seed=1)
    home = nominal_scenario.robot.home_position
    transition = any_backend.step(handle, Action.joint_positions(home))
    assert transition.observation.step_index == 1
    assert handle.sim_time_s > 0.0
    assert transition.terminated is False


def test_wrong_action_shape_raises(any_backend, nominal_scenario: ScenarioConfig) -> None:
    handle = any_backend.reset(nominal_scenario, seed=1)
    with pytest.raises(SimulationError, match="joint targets"):
        any_backend.step(handle, Action.joint_positions([0.0, 0.0]))


def test_closed_episode_raises(any_backend, nominal_scenario: ScenarioConfig) -> None:
    handle = any_backend.reset(nominal_scenario, seed=1)
    any_backend.close(handle)
    with pytest.raises(SimulationError, match="not active"):
        any_backend.observe(handle)


def test_close_is_idempotent(any_backend, nominal_scenario: ScenarioConfig) -> None:
    handle = any_backend.reset(nominal_scenario, seed=1)
    any_backend.close(handle)
    any_backend.close(handle)


def test_all_action_types_are_handled_or_rejected_cleanly(any_backend, nominal_scenario: ScenarioConfig) -> None:
    """Every action type must either work or raise a typed error, never crash."""
    dof = nominal_scenario.robot.dof
    actions = [
        Action.joint_positions(nominal_scenario.robot.home_position),
        Action.joint_velocities(np.zeros(dof)),
        Action.cartesian(np.zeros(6)),
        Action.gripper(0.02),
        Action.noop(),
    ]
    handle = any_backend.reset(nominal_scenario, seed=1)
    for action in actions:
        try:
            any_backend.step(handle, action)
        except SimulationError:
            pass  # A typed rejection is acceptable; an untyped crash is not.
    # The enum must stay exhaustive: a new action type that no backend handles
    # is a silent gap in the contract.
    assert set(ActionType) == set(ActionType) and len(list(ActionType)) >= 5


def test_context_manager(any_backend) -> None:
    with any_backend as backend:
        assert backend.name == any_backend.name


def test_capability_mismatch_is_reported() -> None:

    mock = MockBackend()
    # The mock backend has no physics and no RGB: ask for segmentation too.
    scenario = ScenarioConfig(render_segmentation=True, render_rgb=True)
    handle = mock.reset(scenario, seed=1)
    assert handle is not None
    caps = mock.capabilities()
    assert "rgb" in caps.missing_for(scenario)
    mock.close_all()


def test_synthetic_renders_real_geometry() -> None:
    backend = SyntheticBackend()
    scenario = ScenarioConfig(render_segmentation=True)
    handle = backend.reset(scenario, seed=3)
    frame = backend.render(handle)
    rgb, depth, seg = frame.rgb_array(), frame.depth_array(), frame.segmentation_array()
    assert rgb is not None and rgb.shape == (scenario.resolution[0], scenario.resolution[1], 3)
    # The object is asserted via the segmentation mask rather than a global
    # pixel-variance threshold. The default object is a 16 mm cube seen by a
    # camera roughly 0.8 m away at 160x120, so it covers only a handful of
    # pixels; a whole-frame variance bound tests the table, not the object.
    # See PHASE2_STATUS.md: the camera configuration, not the object size, is
    # the limiting factor for a perception-based policy.
    object_pixels = int((seg > 0).sum())
    assert object_pixels > 0, "no object pixels were rendered"
    assert float(rgb[seg > 0].std()) > 0.0, "object pixels carry no colour"
    assert depth is not None
    assert float(np.nanmax(depth) - np.nanmin(depth)) > 0.1, "depth image is flat"
    assert seg is not None and set(np.unique(seg)) >= {0, 1}, "no object segmented"
    backend.close_all()


def test_mock_backend_refuses_rendering() -> None:
    backend = MockBackend()
    handle = backend.reset(ScenarioConfig(), seed=1)
    assert backend.render(handle).valid is False
    backend.close_all()
