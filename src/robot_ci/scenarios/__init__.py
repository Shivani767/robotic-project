"""Scenario definition, catalogue and grid expansion."""

from typing import TYPE_CHECKING

from robot_ci.scenarios.catalogue import (
    FAMILIES,
    ScenarioFamily,
    family_names,
    get_family,
    list_families,
)
from robot_ci.scenarios.loader import (
    expand_grid,
    load_scenario,
    save_scenario,
    scenario_from_dict,
    set_by_path,
    sweep_scenario,
)

if TYPE_CHECKING:  # pragma: no cover - keeps the runtime import graph acyclic
    from robot_ci.models.scenario import ScenarioConfig

__all__ = [
    "FAMILIES",
    "ScenarioFamily",
    "default_scenario",
    "expand_grid",
    "family_names",
    "get_family",
    "list_families",
    "load_scenario",
    "save_scenario",
    "scenario_from_dict",
    "set_by_path",
    "sweep_scenario",
]

#: Seed used by :func:`default_scenario` when the caller does not name one.
#:
#: Pinned, and not merely for reproducibility. ``ScenarioConfig`` defaults to
#: seed 0, which this arm cannot solve: measured, 55 mm of placement error
#: against 5.2 mm here. A search that fell back to it stopped after one
#: evaluation and reported a false "both arms fail at nominal". The value is
#: fixed so that a bare ``robot-ci counterexample search`` searches the same
#: condition on every machine, and searches a condition that can be regressed
#: from at all.
DEFAULT_SEED = 42


def default_scenario(seed: int = DEFAULT_SEED) -> "ScenarioConfig":
    """Build the uncorrupted control condition, for when no file is given.

    Commands that take an optional ``--scenario`` need *some* condition to run
    against, and defaulting to a bare ``ScenarioConfig()`` is a trap. That object
    is well-formed -- it does contain the target and the placement goal -- but it
    carries ``seed=0``, and seed 0 is a nominal condition this arm cannot solve:
    measured, it places the object 55 mm off with a 44 mm lift, against 5.2 mm
    on seed 42.

    The consequence was not a crash but a confident wrong answer.
    ``robot-ci counterexample search`` with no ``--scenario`` evaluated exactly
    one point and reported::

        no counterexample found
        note: both arms fail at nominal, so no baseline success exists

    Every clause of that is true and the conclusion is worthless. The baseline
    did fail at nominal, so there was nothing to regress from, and the search
    stopped instead of searching. It is a false negative produced by an
    unsolvable default seed, reported with the same confidence as a measured
    one -- and since this is a CI gate, it would pass a release having evaluated
    nothing. Pinning a seed that has been measured to succeed turns the same
    invocation into a real search: 9 points evaluated along the severity ramp,
    and an honest "no counterexample found along the severity ramp".

    This is the catalogue's nominal family, which is the condition every other
    family is defined relative to, so a search with no ``--scenario`` is
    searching the same control the benchmark suite uses.

    Args:
        seed: Episode seed. Pinned by default so a bare invocation searches the
            same condition on every machine, and so that default is one the
            task is known to be solvable on.

    Returns:
        A nominal, fully populated scenario.
    """
    return get_family("nominal").build(0.0, seed=seed, scenario_id="nominal")
