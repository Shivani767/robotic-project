"""Typed error hierarchy for ROBOT CI.

Every error carries a stable ``code`` so that machine-readable logs and CI
integrations can branch on failures without string matching on messages.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "CounterexampleError",
    "EvaluationError",
    "NotImplementedYetError",
    "PolicyError",
    "RobotCIError",
    "ScenarioError",
    "SchemaVersionError",
    "SimulationError",
    "WorkspaceError",
]


class RobotCIError(Exception):
    """Base class for every error raised by this package.

    Args:
        message: Human readable description.
        code: Stable machine-readable identifier, e.g. ``"scenario.invalid"``.
        details: Optional structured payload for logs and reports.
    """

    default_code: str = "robot_ci.error"

    def __init__(self, message: str, *, code: str | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


class ConfigurationError(RobotCIError):
    """Raised when a configuration file is missing, malformed or inconsistent."""

    default_code = "config.invalid"


class SchemaVersionError(ConfigurationError):
    """Raised when a serialized artifact uses an unsupported schema version."""

    default_code = "config.schema_version"


class WorkspaceError(RobotCIError):
    """Raised when the project workspace layout cannot be resolved."""

    default_code = "workspace.invalid"


class ScenarioError(RobotCIError):
    """Raised when a scenario definition is invalid or cannot be built."""

    default_code = "scenario.invalid"


class SimulationError(RobotCIError):
    """Raised when a simulator backend fails during an episode."""

    default_code = "simulation.failed"


class BackendUnavailableError(SimulationError):
    """Raised when a requested backend cannot run in the current environment.

    This is the *expected* error on macOS for ``isaac_sim`` and ``ros2``. The
    message is written to be actionable rather than alarming.
    """

    default_code = "simulation.backend_unavailable"


class PolicyError(RobotCIError):
    """Raised when a policy violates the :class:`~robot_ci.policies.base.RobotPolicy` protocol."""

    default_code = "policy.invalid"


class EvaluationError(RobotCIError):
    """Raised when metrics cannot be computed or an episode result is malformed."""

    default_code = "evaluation.failed"


class CounterexampleError(RobotCIError):
    """Raised when a counterexample cannot be searched, minimized or reproduced."""

    default_code = "counterexample.failed"


class NotImplementedYetError(RobotCIError):
    """Raised by CLI commands reserved for a later development phase.

    The project deliberately surfaces unimplemented phases instead of returning
    fake success, so that a user can never mistake a stub for a real result.
    """

    default_code = "cli.not_implemented"
