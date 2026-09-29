"""The RobotPolicy interface.

A policy maps observations to actions. It never sees ground truth, never reads
the simulator's internals, and never decides success -- that is the task's job.
Keeping this boundary narrow is what allows the same policy to be evaluated
against any backend and compared against any other policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
from robot_ci.models.state import Action, Observation

__all__ = ["PolicyDiagnostics", "PolicyOutput", "RobotPolicy"]


class PolicyDiagnostics(dict[str, Any]):
    """Structured diagnostics emitted alongside every action.

    A ``dict`` subclass so it is directly JSON-serializable into the research
    log, while still being readable as attributes in a debugger.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


class PolicyOutput:
    """What a policy returns each step.

    Attributes:
        action: The command to send to the backend.
        diagnostics: Optional structured notes about internal state.
    """

    __slots__ = ("action", "diagnostics")

    def __init__(self, action: Action, diagnostics: PolicyDiagnostics | None = None) -> None:
        self.action = action
        self.diagnostics = diagnostics or PolicyDiagnostics()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<PolicyOutput {self.action.type.value}>"


class RobotPolicy(ABC):
    """Base class for every policy.

    Lifecycle per episode::

        policy.reset(config, seed)
        while not done:
            observation = backend.observe(handle)
            output = policy.act(observation)
            transition = backend.step(handle, output.action)
    """

    #: Factory key used in :class:`~robot_ci.models.scenario.PolicyConfig`.
    implementation: str = "abstract"

    def __init__(self, config: PolicyConfig, scenario: ScenarioConfig) -> None:
        """Bind the policy to its configuration and the scenario it must solve.

        Args:
            config: Declares components, hyperparameters and budgets.
            scenario: The experimental condition. A policy needs it to know which
                object is the target and what the task's waypoints are, but it
                receives only *beliefs* about the world, never ground truth.
        """
        self.config = config
        self.scenario = scenario
        self.seed = 0
        self.step_count = 0
        self._state: dict[str, Any] = {}

    # -- required API ------------------------------------------------------
    @abstractmethod
    def reset(self, config: PolicyConfig, seed: int, scenario: ScenarioConfig | None = None) -> None:
        """Begin a new episode.

        Args:
            config: The policy configuration for this run.
            seed: Episode seed, for reproducible tie-breaking.
            scenario: Optional replacement scenario, for reusing one policy
                instance across a grid of conditions.
        """

    @abstractmethod
    def act(self, observation: Observation) -> PolicyOutput:
        """Choose an action for the current observation.

        Args:
            observation: Everything the policy is allowed to see.

        Returns:
            The action plus diagnostics.
        """

    # -- introspection -----------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Return the policy's internal state, for logging and debugging."""
        return {"step_count": self.step_count, **self._state}

    def get_diagnostics(self) -> PolicyDiagnostics:
        """Return structured diagnostics about the policy's current belief."""
        return PolicyDiagnostics(
            policy=self.config.name,
            version=self.config.version,
            step_count=self.step_count,
        )

    def _next(self) -> int:
        """Increment and return the step counter."""
        self.step_count += 1
        return self.step_count

    def _rng(self, stream: str) -> np.random.Generator:
        """Return a deterministic generator for one named internal stream.

        Policies must never share a global RNG: two policies running the same
        scenario have to see the same tie-breaks, or a difference in outcome
        could come from the RNG rather than from the policy.
        """
        from robot_ci.core.seeding import make_numpy_rng

        return make_numpy_rng(self.seed + self.config.seed_offset, self.config.name, stream)
