"""Version constants for ROBOT CI.

``SCHEMA_VERSION`` is bumped whenever a serialized schema changes
incompatibly. Artifacts record it so old results are never silently
misinterpreted.
"""

from __future__ import annotations

__all__ = [
    "ENVIRONMENT_VERSION",
    "PACKAGE_NAME",
    "SCHEMA_VERSION",
    "SIMULATOR_VERSION",
    "VERSION",
]

#: Distribution version. Kept in sync with ``pyproject.toml``.
VERSION = "0.1.0"

PACKAGE_NAME = "robot-ci"

#: Semantic version of every persisted Pydantic schema in :mod:`robot_ci.models`.
SCHEMA_VERSION = "1.0.0"

#: Version stamped onto simulator observation contracts. Bump this when the
#: meaning of a field changes, so that cross-backend comparisons stay honest.
SIMULATOR_VERSION = "0.1.0"

#: Version of the scenario environment generator. Recorded in every artifact.
ENVIRONMENT_VERSION = "0.1.0"
