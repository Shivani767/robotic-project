"""Phase 3 acceptance: the two policies must run under identical conditions.

A comparison harness exists to attribute a difference in outcome to the thing
that was changed. That only works if everything *else* is held fixed. The
guarantee is easy to state, easy to break, and -- critically -- invisible when
broken: both arms still run, both still produce plausible numbers, and the
seed confound shows up only as extra variance that a reader would reasonably
blame on the policy.

The seed was in fact derived from the policy name, so the baseline and the
candidate never saw the same random draw. These tests pin the pairing.
"""

from __future__ import annotations

from collections import defaultdict

from robot_ci.experiments.orchestrator import build_run_specs
from robot_ci.models.scenario import ExperimentConfig

#: The compact file form, translated by ``ExperimentConfig``'s own pre-validator,
#: so this exercises the same path an experiment YAML file takes.
YAML_FORM = {
    "experiment": {"id": "pairing_check"},
    "baseline": {"policy": "policy_v1"},
    "candidate": {"policy": "policy_v2"},
    "scenario": {"task": "pick_and_place"},
    "variables": {"depth_noise_mm": {"values": [0, 5]}},
    "repetitions": 3,
    "seed": 42,
}


def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(YAML_FORM)


def test_both_policies_share_a_seed_at_each_cell():
    """Each (cell, replicate) must hand both policies the same seed.

    Different seeds per policy means the two arms are not comparable: the
    measured difference mixes the policy effect with the seed effect and
    neither can be recovered from the numbers afterwards.
    """
    by_cell_and_seed: dict[tuple[str, int], set[str]] = defaultdict(set)
    for spec in build_run_specs(_config()):
        by_cell_and_seed[(spec.scenario.scenario_id, spec.seed)].add(spec.policy.name)

    assert by_cell_and_seed, "the grid expanded to nothing"
    for (scenario_id, seed), policies in by_cell_and_seed.items():
        assert policies == {"policy_v1", "policy_v2"}, (
            f"cell {scenario_id!r} seed {seed} was not run by both policies: "
            f"got {sorted(policies)}"
        )


def test_each_cell_has_one_seed_per_replicate_and_none_shared_between_cells():
    """Replicates must be independent, and cells must not reuse a seed."""
    specs = build_run_specs(_config())
    seeds_by_cell: dict[str, set[int]] = defaultdict(set)
    for spec in specs:
        seeds_by_cell[spec.scenario.scenario_id].add(spec.seed)

    for scenario_id, seeds in seeds_by_cell.items():
        assert len(seeds) == _config().repetitions, (
            f"cell {scenario_id!r} should have one seed per replicate, "
            f"got {len(seeds)}"
        )

    all_seeds = [s for seeds in seeds_by_cell.values() for s in seeds]
    assert len(all_seeds) == len(set(all_seeds)), "cells reused a seed"


def test_both_policies_see_an_identical_scenario():
    """The scenario is the other half of 'identical conditions'.

    A cell that differs in its resolved scenario between the two arms is as
    confounded as one that differs in seed.
    """
    hashes_by_cell: dict[str, set[str]] = defaultdict(set)
    for spec in build_run_specs(_config()):
        hashes_by_cell[spec.scenario.scenario_id].add(spec.scenario.config_hash())

    for scenario_id, hashes in hashes_by_cell.items():
        assert len(hashes) == 1, (
            f"cell {scenario_id!r} resolved to {len(hashes)} distinct scenarios; "
            "the two policies would not be comparable"
        )


def test_run_specs_are_stable_across_calls():
    """Rebuilding the plan must not reshuffle anything.

    A plan that changes between the baseline and candidate executions would
    silently un-pair them, so the derivation has to be a pure function of the
    configuration.
    """
    first = [(s.policy.name, s.scenario.scenario_id, s.seed) for s in build_run_specs(_config())]
    second = [(s.policy.name, s.scenario.scenario_id, s.seed) for s in build_run_specs(_config())]
    assert first == second


def test_every_policy_is_run_against_every_cell():
    """No cell may be measured for one policy but not the other."""
    specs = build_run_specs(_config())
    cells = {s.scenario.scenario_id for s in specs}
    for policy in ("policy_v1", "policy_v2"):
        seen = {s.scenario.scenario_id for s in specs if s.policy.name == policy}
        assert seen == cells, f"{policy} did not cover every cell"
