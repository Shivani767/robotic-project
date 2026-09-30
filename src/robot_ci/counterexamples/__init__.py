"""Counterexample discovery: search, minimize, and replay.

The research loop this package implements:

    oracle.evaluate(point) -> Trial            a black-box two-arm outcome
    Searcher.search(oracle) -> SearchOutcome   find the smallest counterexample
    build_artifact(request, outcome)           make it portable executable YAML
    reproduce(artifact, make_oracle)           prove it still happens

No module here imports a simulator. The concrete bridge from a perturbation
point to real episodes lives in :mod:`robot_ci.counterexamples.episode_oracle`.
"""

from robot_ci.counterexamples.artifact import (
    ARTIFACT_VERSION,
    Counterexample,
    load_counterexample,
    save_counterexample,
)
from robot_ci.counterexamples.discover import (
    DiscoveryRequest,
    DiscoveryResult,
    build_artifact,
    discover,
    perturbed_scenario,
)
from robot_ci.counterexamples.episode_oracle import EpisodeOracle
from robot_ci.counterexamples.problem import (
    CachingOracle,
    OutcomeOracle,
    PerturbationAxis,
    SearchSpace,
    Trial,
)
from robot_ci.counterexamples.reproduce import ReproductionReport, reproduce
from robot_ci.counterexamples.strategies import (
    AdaptiveBoundarySearcher,
    BinaryBoundarySearcher,
    GridSearcher,
    RandomSearcher,
    Searcher,
    SearchOutcome,
    ramp_point,
)

__all__ = [
    "ARTIFACT_VERSION",
    "AdaptiveBoundarySearcher",
    "BinaryBoundarySearcher",
    "CachingOracle",
    "Counterexample",
    "DiscoveryRequest",
    "DiscoveryResult",
    "EpisodeOracle",
    "GridSearcher",
    "OutcomeOracle",
    "PerturbationAxis",
    "RandomSearcher",
    "ReproductionReport",
    "SearchOutcome",
    "SearchSpace",
    "Searcher",
    "Trial",
    "build_artifact",
    "discover",
    "load_counterexample",
    "perturbed_scenario",
    "ramp_point",
    "reproduce",
    "save_counterexample",
]
