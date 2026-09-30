"""Ablation: attribute a behavioural difference to a subsystem.

The question an ablation answers is *which component changed the behaviour*,
not *whether behaviour changed*. The regression detector answers the second; a
two-arm comparison that finds a difference has, by construction, already
established that something differs, and naming the thing is the useful next
step.

Design constraints that shaped this module:

* **Ablatable things must actually change behaviour.** The policy's
  ``components`` map is descriptive metadata -- naming a component
  ``"waypoint_v2"`` does not make the planner behave differently, because
  nothing reads it. Ablating over it would produce a table of identical rows
  and a confident conclusion that the subsystem does not matter. So the factors
  here are the policy *parameters*, each of which is read by a specific
  subsystem, and each factor declares which subsystem owns it. The attribution
  is then grounded in the code path the parameter actually drives.
* **The simulator is injected, not imported.** ``run_ablation`` takes an
  ``evaluate`` callable, so the same experiment runs against MuJoCo, the mock
  backend, or Isaac Sim without this module knowing which.
* **The reference arm is always measured.** Attribution is a difference, and a
  difference against an unmeasured reference is a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from robot_ci.models.scenario import PolicyComponent, PolicyConfig

__all__ = [
    "AblationArm",
    "AblationFactor",
    "AblationResult",
    "build_ofat",
    "build_pairwise",
    "run_ablation",
]


@dataclass(frozen=True)
class AblationFactor:
    """One policy parameter that can be varied, and who owns it.

    Attributes:
        parameter: Key in :attr:`PolicyConfig.parameters`.
        values: Candidate values. The first is not special: the reference
            config decides which value is the baseline.
        subsystem: The component whose code path reads ``parameter``. This is
            the attribution the report will make, so it must name a parameter a
            reader can grep for and confirm is read.
        label: Human-readable description for a report.
    """

    parameter: str
    values: tuple[float, ...]
    subsystem: PolicyComponent
    label: str = ""

    def __post_init__(self) -> None:
        """Reject a factor that cannot produce a comparison."""
        if len(self.values) < 2:
            raise ValueError(
                f"ablation factor {self.parameter!r} needs at least two values to "
                f"compare, got {self.values!r}"
            )
        if not self.label:
            object.__setattr__(self, "label", self.parameter)


class AblationArm(BaseModel):
    """One configuration in the ablation, and what it measured.

    Attributes:
        arm_id: Stable identifier, e.g. ``ablation:ofat:ik_damping=0.1``.
        description: What was changed relative to the reference.
        parameters: The full parameter map this arm ran with.
        success_rate: Fraction of episodes that succeeded.
        samples: Episodes run.
        metrics: Mean of each metric across the arm's episodes.
        active_factors: Parameters this arm changed relative to the reference.
    """

    schema_version: str = Field(default="1.0.0")
    arm_id: str
    description: str
    parameters: dict[str, float]
    success_rate: float
    samples: int
    metrics: dict[str, float] = Field(default_factory=dict)
    active_factors: tuple[str, ...] = ()


class AblationResult(BaseModel):
    """The outcome of an ablation, including its attribution.

    Attributes:
        arms: Every arm measured, reference first.
        attribution: Subsystem name paired with the success-rate change caused
            by varying it, largest absolute change first. A subsystem whose
            factor changed nothing is still listed, with a change of zero.
        reference_arm_id: Identifier of the reference arm.
        note: The caveat that must travel with the numbers, or ``""``.
    """

    schema_version: str = Field(default="1.0.0")
    arms: tuple[AblationArm, ...]
    attribution: tuple[tuple[str, float], ...] = ()
    reference_arm_id: str = ""
    note: str = ""


def build_ofat(
    factors: Sequence[AblationFactor],
    reference: PolicyConfig,
) -> list[tuple[PolicyConfig, str, tuple[str, ...]]]:
    """One-factor-at-a-time: change each parameter, hold the rest at reference.

    OFAT answers "which knob" and cannot answer "which combination". It is the
    right first experiment because each row is independently interpretable, and
    the wrong only experiment because two factors that interact will each look
    inert. ``build_pairwise`` covers that case; neither replaces the other.

    Args:
        factors: Factors to vary.
        reference: The configuration every arm is derived from.

    Returns:
        ``(config, description, active_factors)`` per arm. The reference itself
        is not included -- ``run_ablation`` measures it once, and duplicating it
        here would spend an equal number of episodes re-measuring a known arm.
    """
    arms: list[tuple[PolicyConfig, str, tuple[str, ...]]] = []
    for factor in factors:
        for value in factor.values:
            if value == reference.parameters.get(factor.parameter):
                continue
            arms.append(
                (
                    _with_parameter(reference, factor.parameter, value),
                    f"{factor.parameter}={value:g}",
                    (factor.parameter,),
                )
            )
    return arms


def build_pairwise(
    factors: Sequence[AblationFactor],
    reference: PolicyConfig,
) -> list[tuple[PolicyConfig, str, tuple[str, ...]]]:
    """Every combination of two factors, at their non-reference values.

    Args:
        factors: Factors to vary.
        reference: The configuration arms are derived from.

    Returns:
        ``(config, description, active_factors)`` per arm. Each factor
        contributes its first non-reference value, so the design is quadratic in
        the factor count rather than the product of every value of every factor
        -- enough to expose an interaction without spending the whole grid.
    """
    chosen: list[tuple[AblationFactor, float]] = []
    for factor in factors:
        for value in factor.values:
            if value != reference.parameters.get(factor.parameter):
                chosen.append((factor, value))
                break
    arms: list[tuple[PolicyConfig, str, tuple[str, ...]]] = []
    for index, (first, first_value) in enumerate(chosen):
        for second, second_value in chosen[index + 1 :]:
            config = _with_parameter(reference, first.parameter, first_value)
            config = _with_parameter(config, second.parameter, second_value)
            arms.append(
                (
                    config,
                    f"{first.parameter}={first_value:g}+{second.parameter}={second_value:g}",
                    (first.parameter, second.parameter),
                )
            )
    return arms


def _with_parameter(config: PolicyConfig, parameter: str, value: float) -> PolicyConfig:
    """Return a copy of ``config`` with one parameter replaced."""
    return config.model_copy(update={"parameters": {**config.parameters, parameter: value}})


def run_ablation(
    reference: PolicyConfig,
    factors: Iterable[AblationFactor],
    evaluate: Callable[[PolicyConfig, str], tuple[float, int, dict[str, float]]],
    *,
    designs: Sequence[str] = ("ofat",),
    arm_prefix: str = "ablation",
) -> AblationResult:
    """Measure the reference and every ablation arm, then attribute the change.

    Args:
        reference: The configuration all arms are derived from.
        factors: Parameters to vary, each declaring its owning subsystem.
        evaluate: Runs one arm and returns ``(success_rate, samples, mean
            metrics)``. Injected so this module never imports a simulator, which
            is what lets the same experiment run on every backend.
        designs: Any of ``"ofat"`` and ``"pairwise"``.
        arm_prefix: Prefix for generated arm identifiers.

    Returns:
        The measured arms and the attribution, reference first.

    Raises:
        ValueError: If ``designs`` names an unknown design.
    """
    factor_list = list(factors)
    builders = {"ofat": build_ofat, "pairwise": build_pairwise}
    unknown = [name for name in designs if name not in builders]
    if unknown:
        raise ValueError(f"unknown ablation design(s) {unknown}; choose from {sorted(builders)}")

    arms: list[AblationArm] = []
    reference_id = f"{arm_prefix}:reference"
    rate, samples, mean_metrics = evaluate(reference, reference_id)
    arms.append(
        AblationArm(
            arm_id=reference_id,
            description="reference configuration, unmodified",
            parameters=dict(reference.parameters),
            success_rate=rate,
            samples=samples,
            metrics=mean_metrics,
        )
    )

    for name in designs:
        for config, description, active in builders[name](factor_list, reference):
            arm_id = f"{arm_prefix}:{name}:{description}"
            arm_rate, arm_samples, arm_metrics = evaluate(config, arm_id)
            arms.append(
                AblationArm(
                    arm_id=arm_id,
                    description=description,
                    parameters=dict(config.parameters),
                    success_rate=arm_rate,
                    samples=arm_samples,
                    metrics=arm_metrics,
                    active_factors=active,
                )
            )

    return AblationResult(
        arms=tuple(arms),
        attribution=_attribute(factor_list, arms, reference_id),
        reference_arm_id=reference_id,
        note=_attribution_note(factor_list, arms),
    )


def _attribute(
    factors: Sequence[AblationFactor],
    arms: Sequence[AblationArm],
    reference_id: str,
) -> tuple[tuple[str, float], ...]:
    """Rank subsystems by the success-rate change their factor produced.

    A subsystem is credited with the *mean* change over the one-factor arms
    that vary it, so a parameter with several values is not over-counted by
    simply having more rows. Pairwise arms are excluded: their effect belongs to
    the combination, and crediting it to either factor would be arbitrary.
    """
    reference = next(arm for arm in arms if arm.arm_id == reference_id)
    owner = {factor.parameter: factor for factor in factors}
    effects: dict[str, list[float]] = {}
    for arm in arms:
        if arm.arm_id == reference_id or len(arm.active_factors) != 1:
            continue
        factor = owner.get(arm.active_factors[0])
        if factor is None:
            continue
        effects.setdefault(factor.subsystem.value, []).append(
            arm.success_rate - reference.success_rate
        )
    ranked = [
        (subsystem, sum(deltas) / len(deltas)) for subsystem, deltas in effects.items()
    ]
    return tuple(sorted(ranked, key=lambda item: -abs(item[1])))


def _attribution_note(factors: Sequence[AblationFactor], arms: Sequence[AblationArm]) -> str:
    """State the limitation that always accompanies an attribution.

    Carried on the result rather than left to the report so it cannot be
    dropped when the table is copied out of context.
    """
    return (
        f"{len(factors)} factor(s) varied over {len(arms) - 1} arm(s). "
        "Attribution here is an association measured by substitution, not a "
        "causal claim: varying a parameter also varies the trajectories it "
        "produces, so a component can be credited with an effect that belongs "
        "to the episodes it happened to run on. Confirming an attribution "
        "requires re-running the attributed arm with that factor returned to "
        "its reference value and everything else held fixed."
    )
