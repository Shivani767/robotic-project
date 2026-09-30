"""The fallback condition must be one the task can actually be completed on.

This is a simulator-level property, so it cannot be asserted in a unit test: it
is the reason the counterexample search was reporting a confident and worthless
"both arms fail at nominal". A bare ``ScenarioConfig`` is well-formed, but its
default seed 0 is a nominal condition the arm does not solve, and a search
built on it stopped after one evaluation instead of searching.

The test is slow by nature -- it runs real episodes -- so it lives here rather
than in ``tests/unit``. It is the guard on a CI gate: without it, a false
negative passes a release having evaluated nothing.
"""

from __future__ import annotations

import pytest

from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import DEFAULT_SEED, default_scenario
from robot_ci.simulation import create_backend

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("seed", [DEFAULT_SEED])
def test_the_pinned_default_seed_is_solvable(seed: int) -> None:
    """The seed a bare ``robot-ci counterexample search`` uses must succeed.

    If this fails, every search invoked without ``--scenario`` is reporting a
    false negative, and the seed in ``scenarios.DEFAULT_SEED`` has to move to
    one that works -- or the arm has regressed and the pinned number is now
    optimistic.
    """
    scenario = default_scenario(seed=seed)
    backend = create_backend("mujoco")
    try:
        policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
        result = run_episode(backend, policy, scenario, seed=seed, experiment_id="default_solvable")
    finally:
        backend.close_all()

    assert result.success, (
        f"the pinned default seed {seed} no longer completes the task "
        f"(status={result.status}); counterexample search invoked without "
        "--scenario would report a false 'both arms fail at nominal'"
    )
