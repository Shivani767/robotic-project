"""Corruption is deterministic under seed (Phase 5 acceptance)."""

from __future__ import annotations

import pytest

from robot_ci.models.scenario import ScenarioConfig
from robot_ci.models.sensors import (
    DepthCorruptionConfig,
    RGBCorruptionConfig,
    SensorBundleConfig,
)
from robot_ci.sensors import (
    apply_corruption,
    belief_position_sigma_m,
    corruption_is_active,
)
from robot_ci.simulation import create_backend


def test_nominal_bundle_is_inactive() -> None:
    assert corruption_is_active(SensorBundleConfig()) is False


def test_clean_observation_passes_through_unchanged() -> None:
    backend = create_backend("mock")
    try:
        handle = backend.reset(ScenarioConfig(scenario_id="t", seed=1), seed=5)
        obs = backend.observe(handle)
        out, summary = apply_corruption(obs, SensorBundleConfig(), seed=5, step=0)
        assert out is obs
        assert summary.confidence_scale == 1.0
    finally:
        backend.close_all()


def test_depth_noise_inflates_sigma_monotonically() -> None:
    levels = [0.0, 5.0, 10.0, 20.0, 60.0]
    sigmas = [
        belief_position_sigma_m(
            SensorBundleConfig(depth=DepthCorruptionConfig(enabled=True, sigma_mm=v))
        )
        for v in levels
    ]
    assert sigmas == sorted(sigmas)
    assert sigmas[-1] > 0.05 > sigmas[0]


def test_corruption_is_deterministic_and_step_sensitive() -> None:
    backend = create_backend("mock")
    try:
        handle = backend.reset(ScenarioConfig(scenario_id="t", seed=1), seed=5)
        obs = backend.observe(handle)
        cfg = SensorBundleConfig(depth=DepthCorruptionConfig(enabled=True, sigma_mm=60.0))
        first, _ = apply_corruption(obs, cfg, seed=5, step=0)
        repeat, _ = apply_corruption(obs, cfg, seed=5, step=0)
        other_step, _ = apply_corruption(obs, cfg, seed=5, step=1)
        assert first.objects[0].pose.position.array.tolist() == repeat.objects[0].pose.position.array.tolist()
        assert first.objects[0].pose.position.array.tolist() != other_step.objects[0].pose.position.array.tolist()
        assert first.objects[0].position_sigma_m == pytest.approx(belief_position_sigma_m(cfg))
    finally:
        backend.close_all()


def test_dropout_hides_objects_probabilistically_but_reproducibly() -> None:
    backend = create_backend("mock")
    try:
        handle = backend.reset(ScenarioConfig(scenario_id="t", seed=1), seed=5)
        obs = backend.observe(handle)
        cfg = SensorBundleConfig(
            depth=DepthCorruptionConfig(enabled=True, sigma_mm=1.0, dropout_percent=100.0)
        )
        out, _ = apply_corruption(obs, cfg, seed=5, step=0)
        assert all(not obj.visible for obj in out.objects)
        assert all(obj.confidence == 0.0 for obj in out.objects)
    finally:
        backend.close_all()


def test_occlusion_reduces_confidence() -> None:
    backend = create_backend("mock")
    try:
        handle = backend.reset(ScenarioConfig(scenario_id="t", seed=1), seed=5)
        obs = backend.observe(handle)
        cfg = SensorBundleConfig(rgb=RGBCorruptionConfig(enabled=True, occlusion_percent=40.0))
        confidences = set()
        for step in range(20):
            out, _ = apply_corruption(obs, cfg, seed=5, step=step)
            confidences.add(round(out.objects[0].confidence, 6))
        # Occlusion scales confidence down (0.9 * 0.6 = 0.54) when visible,
        # and hides the object (confidence 0) on some draws.
        assert confidences <= {0.0, 0.54}
        assert max(confidences) == pytest.approx(0.54)
        assert max(confidences) < obs.objects[0].confidence
    finally:
        backend.close_all()


def test_sweeping_a_magnitude_arms_its_enabled_switch() -> None:
    """A grid sweep that writes sigma must not leave the corruption disabled."""
    from robot_ci.models.scenario import ExperimentConfig
    from robot_ci.scenarios import expand_grid

    config = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "enablement"},
            "baseline": {"policy": "policy_v1"},
            "candidate": {"policy": "policy_v2"},
            "scenario": {"task": "pick_and_place"},
            "variables": {"depth_noise_mm": {"values": [0, 60]}},
            "repetitions": 1,
            "seed": 42,
        }
    )
    cells = expand_grid(config.scenario, config.variables)
    clean = next(c for c in cells if c.sensors.depth.sigma_mm == 0.0)
    noisy = next(c for c in cells if c.sensors.depth.sigma_mm == 60.0)
    assert clean.sensors.is_nominal is True
    assert noisy.sensors.depth.enabled is True
    assert noisy.sensors.is_nominal is False


def test_zero_sweep_keeps_nominal() -> None:
    """Writing an explicit zero must not arm the corruption."""
    from robot_ci.models.scenario import ScenarioConfig
    from robot_ci.scenarios.loader import set_by_path

    scenario = set_by_path(ScenarioConfig(scenario_id="z", seed=1), "sensors.depth.sigma_mm", 0.0)
    assert scenario.sensors.is_nominal is True
