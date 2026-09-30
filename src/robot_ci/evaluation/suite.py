"""The end-to-end regression gate: what `robot-ci test` runs.

This is the command a change is gated on, so its design is dominated by one
concern: **it must not report success it has not earned.** A gate that can pass
having measured nothing is worse than no gate, because it converts an absence of
evidence into evidence of absence and then stops anyone looking.

Four properties follow from that, and each is enforced rather than intended:

* **Coverage is reported next to the verdict.** Every gate states how many
  episodes, conditions and seeds it actually ran. A verdict without a count
  beside it is not checkable.
* **An unreachable gate is a failure, not a skip.** If a scenario cannot be
  built or an episode cannot run, that is reported as a gate that did not pass.
* **Blocking is configured, not assumed.** Thresholds are the project's own and
  are echoed in the output, so a reader can see what the verdict was measured
  against.
* **Null is a result.** Finding no regression exits 0 and says so with the
  coverage that supports it, which is different from finding nothing to report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "GateResult",
    "GateStatus",
    "SuiteConfig",
    "SuiteReport",
    "default_suite_config",
]


class GateStatus(StrEnum):
    """Outcome of one gate."""

    PASSED = "passed"
    FAILED = "failed"
    #: The gate could not run. Distinct from passing, and never counted as it.
    ERROR = "error"


class GateResult(BaseModel):
    """One gate's outcome, with the coverage that produced it.

    Attributes:
        name: What was checked.
        status: Outcome.
        detail: One line a reader can act on.
        coverage: What the gate actually exercised, in words.
        episodes: Episodes this gate ran, as a number. Explicit rather than
            parsed out of ``coverage``: a report that extracts its own totals
            from prose silently reports zero when the wording changes, and a gate
            that says it ran nothing while claiming it passed is the exact
            failure this module exists to prevent.
        blocking: Whether a failure here should fail the run.
    """

    schema_version: str = Field(default="1.0.0")
    name: str
    status: GateStatus
    detail: str
    coverage: str = ""
    episodes: int = 0
    blocking: bool = True


@dataclass(frozen=True)
class SuiteConfig:
    """What the gate runs and what it considers a failure.

    Attributes:
        backend: Simulator backend.
        baseline: Baseline policy name.
        candidate: Candidate policy name.
        seeds: Seeds per condition per arm.
        levels: Values of the swept variable.
        axis_path: Scenario path the sweep writes.
        axis_name: Name used in output.
        blocking_delta: Success-rate change that fails the run.
        warning_delta: Success-rate change that warns.
        run_benchmarks: Whether to run the failure-injection suite.
        benchmark_families: Restrict the suite to these families; empty means all.
    """

    backend: str = "mujoco"
    baseline: str = "policy_v1"
    candidate: str = "policy_v2"
    seeds: int = 4
    levels: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0)
    axis_path: str = "sensors.depth.sigma_mm"
    axis_name: str = "depth_noise_mm"
    blocking_delta: float = -0.10
    warning_delta: float = -0.05
    run_benchmarks: bool = True
    benchmark_families: tuple[str, ...] = ()


def default_suite_config(backend: str = "mujoco", *, seeds: int = 4) -> SuiteConfig:
    """The gate's default configuration for a backend.

    Args:
        backend: Backend to run against.
        seeds: Seeds per condition per arm.

    Returns:
        A configuration; the seed count is the only thing that scales with the
        time a caller is willing to spend.
    """
    return SuiteConfig(backend=backend, seeds=seeds)


class SuiteReport(BaseModel):
    """The whole run: every gate, and what the exit code follows from.

    Attributes:
        gates: Gate outcomes, in execution order.
        episodes: Total episodes the run executed.
        conditions: Distinct conditions exercised.
        seeds: Seeds per condition.
        blocking_delta: Threshold the verdict was measured against, echoed so a
            reader does not have to look it up.
        notes: Anything a reader needs to interpret the verdict.
    """

    schema_version: str = Field(default="1.0.0")
    gates: tuple[GateResult, ...] = ()
    episodes: int = 0
    conditions: int = 0
    seeds: int = 0
    blocking_delta: float = -0.10
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> tuple[GateResult, ...]:
        """Gates that failed, including those that could not run."""
        return tuple(g for g in self.gates if g.status is not GateStatus.PASSED)

    @property
    def blocking_failures(self) -> tuple[GateResult, ...]:
        """Failures that should make the command exit non-zero."""
        return tuple(g for g in self.failed if g.blocking)

    def summary_lines(self) -> list[str]:
        """Render the report as the lines the CLI prints."""
        lines = [
            f"Scenarios      : {self.conditions} conditions x {self.seeds} seeds x 2 arms",
            f"Episodes       : {self.episodes}",
            f"Blocking delta : {self.blocking_delta:+.2f} success rate",
            "",
        ]
        width = max((len(g.name) for g in self.gates), default=4)
        for gate in self.gates:
            mark = {GateStatus.PASSED: "PASS", GateStatus.FAILED: "FAIL", GateStatus.ERROR: "ERR "}[
                gate.status
            ]
            lines.append(f"  [{mark}] {gate.name.ljust(width)}  {gate.detail}")
            if gate.coverage:
                lines.append(f"         {'':<{width}}  coverage: {gate.coverage}")
        if not self.failed:
            lines.append("")
            lines.append("No blocking regressions. This is a measured null, not an absence of checks:")
            for gate in self.gates:
                lines.append(f"  - {gate.name}: {gate.coverage or gate.detail}")
        for note in self.notes:
            lines.append(f"note: {note}")
        return lines


def run_suite(
    config: SuiteConfig,
    *,
    environment: Callable[[], GateResult],
    regression: Callable[[SuiteConfig], GateResult],
    benchmarks: Callable[[SuiteConfig], GateResult] | None = None,
) -> SuiteReport:
    """Run every gate and collect the report.

    Gates are executed in order and all of them run even when one fails. A CI
    gate that stops at the first failure hides every later problem behind the
    first, and the person fixing it will then find the next one anyway.

    Args:
        config: What to run.
        environment: Checks the environment and scenario catalogue.
        regression: Runs the two-arm sweep and scores it.
        benchmarks: Optionally runs the failure-injection suite.

    Returns:
        The full report, whose ``blocking_failures`` decides the exit code.
    """
    gates: list[GateResult] = [environment(), regression(config)]
    if benchmarks is not None:
        gates.append(benchmarks(config))
    return SuiteReport(
        gates=tuple(gates),
        episodes=sum(gate.episodes for gate in gates),
        conditions=len(config.levels),
        seeds=config.seeds,
        blocking_delta=config.blocking_delta,
        notes=(),
    )
