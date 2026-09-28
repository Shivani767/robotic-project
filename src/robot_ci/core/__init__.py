"""Core infrastructure: errors, seeding, hashing, logging, I/O and workspace.

This package deliberately has **no** dependency on NumPy, Pydantic, ROS 2 or any
simulator, so it can be imported by tooling, tests and remote nodes alike.
"""

from __future__ import annotations

from robot_ci.core.errors import (
    BackendUnavailableError,
    ConfigurationError,
    CounterexampleError,
    EvaluationError,
    NotImplementedYetError,
    PolicyError,
    RobotCIError,
    ScenarioError,
    SchemaVersionError,
    SimulationError,
    WorkspaceError,
)
from robot_ci.core.hashing import canonical_json, hash_bytes, hash_file, hash_payload
from robot_ci.core.io import dumps_yaml, load_config_file, load_yaml, write_json, write_yaml
from robot_ci.core.logging import JsonlLogger, LogRecord, configure_logging, get_logger, utc_now_iso
from robot_ci.core.seeding import derive_seed, make_numpy_rng, make_python_rng, spawn_streams
from robot_ci.core.version import (
    ENVIRONMENT_VERSION,
    PACKAGE_NAME,
    SCHEMA_VERSION,
    SIMULATOR_VERSION,
    VERSION,
)
from robot_ci.core.workspace import DirectoryLayout, Workspace, find_workspace_root, get_workspace

__all__ = [
    "ENVIRONMENT_VERSION",
    "PACKAGE_NAME",
    "SCHEMA_VERSION",
    "SIMULATOR_VERSION",
    "VERSION",
    "BackendUnavailableError",
    "ConfigurationError",
    "CounterexampleError",
    "DirectoryLayout",
    "EvaluationError",
    "JsonlLogger",
    "LogRecord",
    "NotImplementedYetError",
    "PolicyError",
    "RobotCIError",
    "ScenarioError",
    "SchemaVersionError",
    "SimulationError",
    "Workspace",
    "WorkspaceError",
    "canonical_json",
    "configure_logging",
    "derive_seed",
    "dumps_yaml",
    "find_workspace_root",
    "get_logger",
    "get_workspace",
    "hash_bytes",
    "hash_file",
    "hash_payload",
    "load_config_file",
    "load_yaml",
    "make_numpy_rng",
    "make_python_rng",
    "spawn_streams",
    "utc_now_iso",
    "write_json",
    "write_yaml",
]
