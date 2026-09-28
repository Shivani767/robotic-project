"""Simulator backends.

The research engine depends only on :mod:`robot_ci.simulation.base`. Concrete
backends are imported lazily through the registry, so importing this package
does **not** import MuJoCo (or any other simulator). That is what keeps
``robot_ci.regression`` and the counterexample search runnable on a machine
where the primary simulator is not installed.
"""

from robot_ci.simulation.base import (
    BackendCapabilities,
    BackendInfo,
    EpisodeHandle,
    SimulatorBackend,
    TerminationReason,
    Transition,
    create_backend,
    list_backends,
    probe_backend,
    register_backend,
)

__all__ = [
    "BackendCapabilities",
    "BackendInfo",
    "EpisodeHandle",
    "SimulatorBackend",
    "TerminationReason",
    "Transition",
    "create_backend",
    "list_backends",
    "probe_backend",
    "register_backend",
]
