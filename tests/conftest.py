"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_ci.models.scenario import ExperimentConfig, ScenarioConfig, VariableSpec
from robot_ci.simulation.mock import MockBackend


@pytest.fixture
def nominal_scenario() -> ScenarioConfig:
    """A clean, uncorrupted scenario."""
    return ScenarioConfig(scenario_id="test_nominal", seed=42)


@pytest.fixture
def backend() -> MockBackend:
    """A mock backend, closed automatically after the test."""
    instance = MockBackend()
    yield instance
    instance.close_all()


@pytest.fixture
def grid_experiment() -> ExperimentConfig:
    """A small two-variable experiment used across several tests."""
    return ExperimentConfig(
        experiment_id="test_exp",
        variables=(
            VariableSpec(name="depth_noise_mm", path="sensors.depth.sigma_mm", values=(0.0, 5.0, 10.0)),
            VariableSpec(name="occlusion_percent", path="sensors.rgb.occlusion_percent", values=(0.0, 20.0)),
        ),
        repetitions=3,
        seed=7,
    )


@pytest.fixture
def repo_root() -> Path:
    """Repository root, discovered from this file's location."""
    return Path(__file__).resolve().parents[1]
