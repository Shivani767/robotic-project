"""Naming a policy must produce that policy.

An experiment file names its arms -- ``candidate: {policy: policy_v2}`` -- and
only the name reaches the model. The parameters that distinguish the two
policies live in each policy's default config, so a resolver that does not
merge them produces two arms with different names and identical behaviour.

That failure is silent and it is the worst kind: the experiment runs, both arms
report complete metric vectors, the summary says the candidate made no
difference, and a reader concludes the candidate does not help. The candidate
was never run.
"""

from __future__ import annotations

import pytest

from robot_ci.models.scenario import ExperimentConfig, PolicyConfig
from robot_ci.policies import create_policy, resolve_policy_config

SCENARIO_ROW = {"scenario": {"task": "pick_and_place"}}


def _config(**overrides: object) -> ExperimentConfig:
    payload = {
        "experiment": {"id": "resolution_check"},
        "baseline": {"policy": "policy_v1"},
        "candidate": {"policy": "policy_v2"},
        **SCENARIO_ROW,
        **overrides,
    }
    return ExperimentConfig.model_validate(payload)


def test_named_policies_are_behaviourally_distinct():
    """The guard against the vacuous comparison.

    Asserts on the parameters the policy actually branches on, not on the
    config's labels: comparing names is exactly what failed to catch this.
    """
    config = _config()
    baseline = create_policy(config.baseline, config.scenario)
    candidate = create_policy(config.candidate, config.scenario)

    assert baseline.max_position_sigma_m != candidate.max_position_sigma_m, (
        "both arms resolved to the same confidence gate, so the harness is "
        "comparing a policy against itself"
    )
    assert baseline.clearance_bias_m != candidate.clearance_bias_m


def test_named_policy_gets_its_declared_components():
    """A named policy must also carry its component map, for ablation."""
    config = _config()
    candidate = create_policy(config.candidate, config.scenario)
    assert candidate.config.components, "the named policy's components were dropped"


def test_explicit_parameters_beat_the_named_defaults():
    """Naming a policy supplies defaults; it must not forbid overrides."""
    explicit = PolicyConfig(name="policy_v2", parameters={"max_position_sigma_m": 0.5})
    resolved = resolve_policy_config(explicit)
    assert resolved.parameters["max_position_sigma_m"] == 0.5
    # ...and the defaults it did not mention are still filled in.
    assert "clearance_bias_m" in resolved.parameters


def test_unknown_policy_name_is_passed_through_unchanged():
    """An unregistered name must not be silently rewritten.

    A config that matches no known policy is returned as-is, so an unknown
    implementation still fails loudly at ``create_policy`` rather than being
    quietly turned into a known policy.
    """
    unknown = PolicyConfig(name="policy_does_not_exist", parameters={"a": 1.0})
    assert resolve_policy_config(unknown) is unknown


def test_baseline_and_candidate_are_not_the_same_policy():
    """End-to-end: the two arms of a real experiment file differ."""
    config = _config()
    baseline = create_policy(config.baseline, config.scenario)
    candidate = create_policy(config.candidate, config.scenario)

    signature = lambda p: (  # noqa: E731 - a local one-off key is clearer inline
        p.max_position_sigma_m,
        p.clearance_bias_m,
        tuple(sorted((k.value, v) for k, v in p.config.components.items())),
    )
    assert signature(baseline) != signature(candidate)


@pytest.mark.parametrize("name", ["policy_v1", "policy_v2"])
def test_resolution_is_idempotent(name: str):
    """Resolving twice must change nothing, or defaults would compound."""
    once = resolve_policy_config(PolicyConfig(name=name))
    twice = resolve_policy_config(once)
    assert once.parameters == twice.parameters
    assert once.components == twice.components


def test_null_gate_keeps_identical_policies_identical() -> None:
    """Guard the documented Phase 3 null result against false divergence.

    Two policies that differ only in a confidence gate must be bitwise
    identical whenever the measured uncertainty stays below the gate.  If a
    later change silently makes v1 and v2 diverge at low noise, this fails.
    Hold the uncertainty at zero by closing the gate path: both policies share
    the same scenario and backend, so any difference is a real one.
    """
    from robot_ci.models.scenario import ScenarioConfig
    from robot_ci.policies import create_policy
    from robot_ci.simulation import create_backend

    scenario = ScenarioConfig(scenario_id="nullgate", seed=11)
    v1 = create_policy(PolicyConfig(name="policy_v1"), scenario)
    v2 = create_policy(PolicyConfig(name="policy_v2"), scenario)
    assert v1.max_position_sigma_m == pytest.approx(1e9)
    assert v2.max_position_sigma_m == pytest.approx(0.05)

    backend = create_backend("mujoco")
    observed: dict[str, list[float]] = {"policy_v1": [], "policy_v2": []}
    for name, policy in (("policy_v1", v1), ("policy_v2", v2)):
        handle = backend.reset(scenario, seed=11)
        policy.reset(policy.config, 11, scenario)
        policy.attach_planner(backend.jacobian_provider(handle))
        for _ in range(12):
            transition = backend.step(handle, policy.act(backend.observe(handle)).action)
            observed[name].append(float(transition.robot_state.end_effector_pose.position.array[2]))
        backend.close(handle)
    backend.close_all()

    assert len(observed["policy_v1"]) == len(observed["policy_v2"]) == 12
    for first, second in zip(observed["policy_v1"], observed["policy_v2"], strict=True):
        assert first == pytest.approx(second, abs=1e-9)

