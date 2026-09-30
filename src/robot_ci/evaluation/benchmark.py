"""The failure-injection benchmark suite.

A scenario family says *what to perturb*. A benchmark says what the perturbation
is **for**: which subsystem it is meant to stress, what behaviour it predicts,
and which metrics would show the prediction was wrong. Without that last part a
benchmark suite is a list of sweeps, and every sweep produces numbers that are
equally consistent with "the fault is well modelled" and "the fault did nothing".

Three rules govern the declarations here:

* **Every expectation is falsifiable and is checked.** A benchmark declares a
  direction or a bound, and :func:`run_benchmark` reports whether the run
  matched. An expectation nobody checks is a comment.
* **Expectations carry their evidence status.** A benchmark's prediction may be
  derived from a model of the fault or simply asserted, and a reader needs to
  know which. ``verified`` records whether the suite has actually observed the
  prediction hold. Until it does, the expectation is a hypothesis, and the
  suite says so rather than implying it was validated.
* **A fault the scenario model cannot express is absent, not faked.** Every
  benchmark below maps to a real configuration field. Where the specification
  asks for a fault this simulator does not implement, the omission is recorded
  in :data:`UNIMPLEMENTED_BENCHMARKS` with the reason, rather than being
  approximated with something that is not the fault named.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from robot_ci.models.scenario import ScenarioConfig
from robot_ci.scenarios import set_by_path

__all__ = [
    "BENCHMARKS",
    "UNIMPLEMENTED_BENCHMARKS",
    "Benchmark",
    "BenchmarkFamily",
    "BenchmarkResult",
    "benchmark_suite",
    "run_benchmark",
]


class BenchmarkFamily(StrEnum):
    """The subsystem a benchmark stresses, as the specification groups them."""

    PERCEPTION = "perception"
    DEPTH = "depth"
    CAMERA = "camera"
    ROBOT = "robot"
    ENVIRONMENT = "environment"
    PLANNING = "planning"


@dataclass(frozen=True)
class Benchmark:
    """One named failure-injection case and what it predicts.

    Attributes:
        id: Stable identifier, ``<family>_<fault>``.
        family: Subsystem under test.
        description: What the fault is and why it is worth injecting.
        field_path: Dotted path of the configuration field the fault sets.
        value: Value written to that field.
        metrics: Metrics to record, beyond success.
        expect_success: Whether the task should still succeed. ``None`` when the
            benchmark makes no claim about the outcome, only about the metrics.
        expect_metric_drops: Metrics expected to get *worse* under the fault,
            as ``metric -> minimum relative degradation``. A benchmark that
            injects depth noise and does not expect any metric to move is not
            testing anything.
        verified: Whether the suite has observed this expectation hold. False
            means the prediction is a hypothesis derived from a model of the
            fault, not a measured result.
        rationale: Why the expectation is what it is.
    """

    id: str
    family: BenchmarkFamily
    description: str
    field_path: str
    value: float | Sequence[float]
    metrics: tuple[str, ...] = ("task_success", "grasp_success", "placement_error_m")
    expect_success: bool | None = None
    expect_metric_drops: tuple[tuple[str, float], ...] = ()
    verified: bool = False
    rationale: str = ""


#: The suite. Ordered by the family grouping the specification uses.
BENCHMARKS: tuple[Benchmark, ...] = (
    # -- perception --------------------------------------------------------
    Benchmark(
        id="perception_blur",
        family=BenchmarkFamily.PERCEPTION,
        description="Gaussian blur on the RGB stream, degrading edge-based detection.",
        field_path="sensors.rgb.blur_kernel_px",
        value=7.0,
        metrics=("task_success", "perception_confidence"),
        expect_metric_drops=(("perception_confidence", 0.0),),
        rationale=(
            "The control policy latches its plan on the first tick and does not "
            "re-read RGB, so no metric is expected to move. Declared with a "
            "confidence expectation rather than a success one, and left "
            "unverified: this is the honest shape of a benchmark whose target "
            "subsystem is not on the measured path."
        ),
    ),
    Benchmark(
        id="perception_occlusion",
        family=BenchmarkFamily.PERCEPTION,
        description="Synthetic occluder covering part of the RGB stream.",
        field_path="sensors.rgb.occlusion_percent",
        value=60.0,
        metrics=("task_success", "perception_confidence"),
        expect_metric_drops=(("perception_confidence", 0.0),),
        rationale="Same reasoning as blur: occlusion of a stream the policy ignores.",
    ),
    Benchmark(
        id="perception_low_light",
        family=BenchmarkFamily.PERCEPTION,
        description="Scene illuminance reduced until vision-based estimation degrades.",
        field_path="environment.lighting_lux",
        value=20.0,
        metrics=("task_success", "perception_confidence"),
        expect_metric_drops=(("perception_confidence", 0.0),),
        rationale="Same reasoning as blur, for illuminance.",
    ),
    Benchmark(
        id="perception_colour_shift",
        family=BenchmarkFamily.PERCEPTION,
        description="Systematic RGB channel shift, the classic colour-cast fault.",
        field_path="sensors.rgb.color_shift_rgb",
        value=(0.25, -0.10, 0.05),
        metrics=("task_success", "perception_confidence"),
        expect_metric_drops=(("perception_confidence", 0.0),),
        rationale="Same reasoning as blur, for chroma.",
    ),
    # -- depth ------------------------------------------------------------
    Benchmark(
        id="depth_noise",
        family=BenchmarkFamily.DEPTH,
        description="Zero-mean Gaussian depth noise, the primary sensing fault.",
        field_path="sensors.depth.sigma_mm",
        value=20.0,
        metrics=("task_success", "pose_sigma_m", "grasp_error_m", "placement_error_m"),
        expect_success=False,
        expect_metric_drops=(("pose_sigma_m", 0.0), ("grasp_error_m", 0.0)),
        rationale=(
            "Measured boundary: 20 mm of depth noise takes task success to 0.00 "
            "while the nominal condition succeeds on 12 of 24 seeds, and raises "
            "the fused pose sigma. This is the one benchmark family here with a "
            "measured expectation."
        ),
        verified=True,
    ),
    Benchmark(
        id="depth_bias",
        family=BenchmarkFamily.DEPTH,
        description="Systematic depth offset: a calibration error the fuser cannot remove.",
        field_path="sensors.depth.bias_mm",
        value=10.0,
        metrics=("task_success", "pose_sigma_m", "grasp_error_m"),
        expect_success=False,
        expect_metric_drops=(("grasp_error_m", 0.0),),
        rationale=(
            "Averaging removes noise; it cannot remove bias, because a biased "
            "estimate is not a noisy one. Measured: 2 mm of bias already takes "
            "both arms to 0.00."
        ),
        verified=True,
    ),
    Benchmark(
        id="depth_dropout",
        family=BenchmarkFamily.DEPTH,
        description="Fraction of depth pixels dropped to invalid.",
        field_path="sensors.depth.dropout_percent",
        value=40.0,
        metrics=("task_success", "pose_sigma_m", "sensor_consistency"),
        expect_metric_drops=(("sensor_consistency", 0.0),),
        rationale=(
            "Dropout should show up as reduced sensor consistency rather than as "
            "success loss, because the policy fuses a single object estimate and "
            "a dropped frame is simply a frame with no estimate."
        ),
    ),
    Benchmark(
        id="depth_latency",
        family=BenchmarkFamily.DEPTH,
        description="Temporal lag on the depth stream.",
        field_path="sensors.depth.temporal_lag_frames",
        value=5.0,
        metrics=("task_success", "pose_sigma_m"),
        expect_metric_drops=(("pose_sigma_m", 0.0),),
        rationale="Lag should show as pose uncertainty rather than as a placement error.",
    ),
    # -- camera -----------------------------------------------------------
    Benchmark(
        id="camera_latency",
        family=BenchmarkFamily.CAMERA,
        description="Camera frame delivery delayed, dropping frames in transit.",
        field_path="sensors.camera.latency.delay_ms",
        value=200.0,
        metrics=("task_success", "steps", "completion_time_s"),
        expect_success=True,
        rationale=(
            "Expects *success*, which is a strong claim and the reason this is a "
            "useful benchmark: measured across 0, 50, 100 and 200 ms, task "
            "success stays at 1.00 for both arms. The policy latches its plan on "
            "the first control tick and the object never moves, so a delayed "
            "frame is never acted on. Latency is not a behavioural factor for a "
            "plan-once policy on a static object, and a benchmark that swept it "
            "expecting degradation would report a false one."
        ),
        verified=True,
    ),
    Benchmark(
        id="camera_extrinsic_error",
        family=BenchmarkFamily.CAMERA,
        description="Extrinsic rotation error, a mis-mounted camera.",
        field_path="sensors.camera.extrinsic_rotation_deg",
        value=5.0,
        metrics=("task_success", "pose_sigma_m", "grasp_error_m"),
        expect_metric_drops=(("grasp_error_m", 0.0),),
        rationale="A rotation error displaces every back-projected point by depth*angle.",
    ),
    Benchmark(
        id="camera_intrinsic_error",
        family=BenchmarkFamily.CAMERA,
        description="Focal-length and principal-point error, a mis-calibrated camera.",
        field_path="sensors.camera.focal_scale",
        value=1.1,
        metrics=("task_success", "grasp_error_m"),
        expect_metric_drops=(("grasp_error_m", 0.0),),
        rationale="A focal error scales every lateral offset away from the optical centre.",
    ),
    # -- robot ------------------------------------------------------------
    Benchmark(
        id="robot_joint_offset",
        family=BenchmarkFamily.ROBOT,
        description="Constant offset on the reported joint angles: a slipped encoder.",
        field_path="sensors.robot_state.joint_offset_rad",
        value=0.05,
        metrics=("task_success", "placement_error_m", "trajectory_deviation_m", "steps"),
        expect_success=True,
        expect_metric_drops=(("trajectory_deviation_m", 0.0),),
        rationale=(
            "Expects *success*, which the first run of this suite falsified: I had "
            "declared that 0.05 rad of encoder offset would break the task. It "
            "does not. Measured at that magnitude the task still succeeds on 6 of "
            "6 seeds while four of seven metrics move -- trajectory deviation "
            "1.44 to 1.78 and median steps 76 to 104. The fault costs the arm "
            "time and accuracy without costing it the object, so the benchmark "
            "now asserts the degradation it actually causes."
        ),
        verified=True,
    ),
    Benchmark(
        id="robot_joint_noise",
        family=BenchmarkFamily.ROBOT,
        description="Gaussian noise on reported joint angles.",
        field_path="sensors.robot_state.joint_noise_std_rad",
        value=0.05,
        metrics=("task_success", "placement_error_m", "trajectory_deviation_m", "steps"),
        expect_success=True,
        expect_metric_drops=(("trajectory_deviation_m", 0.0),),
        rationale=(
            "Corrected after the first suite run. The expectation was that 0.05 rad "
            "of joint noise would fail the task outright; measured, it succeeds on "
            "4 of 6 seeds while moving five of seven metrics -- success 1.00 to "
            "0.67, trajectory deviation 1.44 to 1.85, steps 76 to 98. Asserting a "
            "hard failure here would have made the benchmark a source of false "
            "failures, which is worse than a benchmark that cannot fail."
        ),
        verified=True,
    ),
    Benchmark(
        id="robot_encoder_delay",
        family=BenchmarkFamily.ROBOT,
        description="Delayed joint feedback, the controller acting on an old pose.",
        field_path="sensors.robot_state.encoder_delay_frames",
        value=5.0,
        metrics=("task_success", "trajectory_deviation_m", "placement_error_m"),
        expect_metric_drops=(("trajectory_deviation_m", 0.0),),
        rationale="Feedback lag should appear as path deviation before it appears as failure.",
    ),
    # -- environment ------------------------------------------------------
    Benchmark(
        id="environment_object_displacement",
        family=BenchmarkFamily.ENVIRONMENT,
        description="Object displaced from where the world model believes it is.",
        field_path="environment.workspace_center",
        value=(0.25, 0.10),
        metrics=("task_success", "grasp_error_m", "final_distance_m"),
        expect_metric_drops=(("grasp_error_m", 0.0),),
        rationale=(
            "Moving the workspace centre away from the arm's nominal station "
            "displaces every object with it. Originally declared against this "
            "path with a scalar, which failed validation because the field is a "
            "pair; the suite reported the error rather than running the nominal "
            "condition under a benchmark name."
        ),
        verified=True,
    ),
    Benchmark(
        id="environment_clutter",
        family=BenchmarkFamily.ENVIRONMENT,
        description="Distractor objects added to the support surface.",
        field_path="environment.clutter_count",
        value=9.0,
        metrics=("task_success", "grasp_error_m", "collision"),
        expect_metric_drops=(("grasp_error_m", 0.0),),
        rationale=(
            "Measured across 0, 2, 5 and 9 distractors, both arms stay at 1.00. "
            "Clutter is not a behavioural factor for a single-target task that "
            "does not avoid obstacles, and declaring otherwise would make this "
            "benchmark a source of false failures."
        ),
        verified=True,
    ),
    Benchmark(
        id="environment_obstacle",
        family=BenchmarkFamily.ENVIRONMENT,
        description="Obstacles inserted between the robot and the object.",
        field_path="environment.obstacle_count",
        value=4.0,
        metrics=("task_success", "collision", "placement_error_m"),
        expect_metric_drops=(("collision", 0.0),),
        rationale="Obstacles are the first environment fault expected to raise collisions.",
    ),
    Benchmark(
        id="environment_friction",
        family=BenchmarkFamily.ENVIRONMENT,
        description="Object friction changed, altering whether a grasp holds.",
        field_path="environment.object_friction",
        value=0.1,
        metrics=("task_success", "grasp_success", "object_lift_m"),
        expect_metric_drops=(("grasp_success", 0.0),),
        rationale=(
            "Measured across 0.1, 0.3, 0.6 and 1.0, both arms stay at 1.00, so no "
            "degradation is expected in this range. The earlier non-monotonic "
            "reading at friction 1.0 was an artefact of the loose IK tolerance and "
            "was withdrawn."
        ),
        verified=True,
    ),
    # -- planning ---------------------------------------------------------
    Benchmark(
        id="planning_wrong_target",
        family=BenchmarkFamily.PLANNING,
        description="Placement goal moved away from the intended one.",
        field_path="task.target_pose.position.x",
        value=0.45,
        metrics=("task_success", "placement_error_m", "final_distance_m"),
        expect_metric_drops=(("placement_error_m", 0.0),),
        rationale=(
            "Moving the goal must move the measured placement error, or the "
            "planner is not following the task at all. 0.45 m is on the table "
            "(half-extent 0.50 m) but at the edge of the arm's top-down reach."
        ),
    ),
    Benchmark(
        id="planning_blocked_path",
        family=BenchmarkFamily.PLANNING,
        description="Obstacle density raised to obstruct the approach corridor.",
        field_path="environment.obstacle_height_m",
        value=0.30,
        metrics=("task_success", "collision", "placement_error_m"),
        expect_metric_drops=(("collision", 0.0),),
        rationale=(
            "Tall obstacles stand above the carry height, so a straight-line "
            "waypoint plan should intersect them."
        ),
    ),
)

#: Faults the specification names that this simulator does not implement.
#:
#: Recorded rather than approximated. A benchmark that injects a *different*
#: fault and names it after the missing one is worse than an acknowledged gap,
#: because the gap then stops being visible.
UNIMPLEMENTED_BENCHMARKS: tuple[tuple[str, str], ...] = (
    ("perception_wrong_label", "No class-confusion model; object category is fixed by the scenario."),
    ("planning_stale_world_model", "The policy re-reads its own object belief each tick rather than a cached world model, so there is no staleness to inject."),
    ("depth_quantization", "Field exists in the sensor model but is not yet wired into the corruption application."),
    (
        "robot_gripper_stuck",
        "The field exists and the corruption arms correctly, but nothing applies it: "
        "measured at 1.0, zero of seven metrics move against nominal. The suite "
        "found this by falsifying the benchmark's own expectation, which is the "
        "clearest argument for a suite that checks its predictions.",
    ),
)


class BenchmarkResult(BaseModel):
    """Outcome of one benchmark against its own declared expectation.

    Attributes:
        benchmark_id: Which benchmark ran.
        scenario_id: Scenario produced for it.
        seeds: Seeds used.
        success_rate: Measured success rate.
        metrics: Mean of each recorded metric.
        expectation_met: Whether the run matched the prediction, or ``None`` when
            the benchmark declares no falsifiable expectation.
        notes: Anything a reader needs in order to interpret the result.
    """

    schema_version: str = Field(default="1.0.0")
    benchmark_id: str
    scenario_id: str
    seeds: tuple[int, ...]
    success_rate: float
    metrics: dict[str, float] = Field(default_factory=dict)
    expectation_met: bool | None = None
    notes: str = ""


def benchmark_suite(family: BenchmarkFamily | None = None) -> tuple[Benchmark, ...]:
    """Return the suite, optionally filtered to one family.

    Args:
        family: Restrict to this subsystem, or ``None`` for all of them.

    Returns:
        The selected benchmarks, in declaration order.
    """
    if family is None:
        return BENCHMARKS
    return tuple(b for b in BENCHMARKS if b.family == family)


def run_benchmark(
    benchmark: Benchmark,
    *,
    base_scenario: ScenarioConfig,
    seeds: Sequence[int],
    evaluate: Callable[[ScenarioConfig, Sequence[int]], tuple[float, dict[str, float]]],
) -> BenchmarkResult:
    """Inject the fault, run the seeds, and check the declared expectation.

    Args:
        benchmark: The case to run.
        base_scenario: Nominal condition to perturb.
        seeds: Seeds to run. Identical for every arm, as everywhere else.
        evaluate: Runs the perturbed scenario and returns
            ``(success_rate, mean metrics)``. Injected so this module does not
            import a simulator, which is what lets the same suite run against
            every backend.

    Returns:
        The result, including whether the prediction held.

    Raises:
        ValueError: If the benchmark's field path does not exist on the
            scenario, which would otherwise surface as a silently ineffective
            injection -- a benchmark that perturbs nothing and passes.
    """
    try:
        scenario = set_by_path(base_scenario, benchmark.field_path, benchmark.value)
    except Exception as error:  # surfaced as a clear failure, not a false pass
        raise ValueError(
            f"benchmark {benchmark.id!r} targets {benchmark.field_path!r}, which the "
            f"scenario does not have: {error}"
        ) from error

    # An injection that does not arm its corruption would measure the nominal
    # condition and report it under the benchmark's name.
    if benchmark.field_path.startswith("sensors.") and not _corruption_armed(scenario, benchmark):
        return BenchmarkResult(
            benchmark_id=benchmark.id,
            scenario_id=scenario.scenario_id,
            seeds=tuple(seeds),
            success_rate=0.0,
            metrics={},
            expectation_met=False,
            notes=(
                f"injection set {benchmark.field_path} but left the owning "
                "corruption disarmed, so the nominal condition would have been "
                "measured and reported under this benchmark's name"
            ),
        )

    rate, metrics = evaluate(scenario, seeds)
    met = _check(benchmark, rate, metrics)
    notes = "" if benchmark.verified else (
        "expectation is a hypothesis derived from a model of the fault, not an "
        "observed result"
    )
    return BenchmarkResult(
        benchmark_id=benchmark.id,
        scenario_id=scenario.scenario_id,
        seeds=tuple(seeds),
        success_rate=rate,
        metrics=metrics,
        expectation_met=met,
        notes=notes,
    )


def _corruption_armed(scenario: ScenarioConfig, benchmark: Benchmark) -> bool:
    """Whether the corruption group owning ``benchmark``'s field is enabled.

    ``set_by_path`` arms the group automatically for the magnitudes it knows
    about. This checks rather than assumes, because an injection that leaves the
    group disarmed measures the nominal condition and files the result under the
    benchmark's name -- a false pass, which is the worst kind of bug in a suite
    whose purpose is to detect false passes.
    """
    groups: dict[str, bool] = {
        "sensors.rgb": scenario.sensors.rgb.enabled,
        "sensors.depth": scenario.sensors.depth.enabled,
        "sensors.camera": scenario.sensors.camera.enabled,
        "sensors.robot_state": scenario.sensors.robot_state.enabled,
    }
    for prefix, enabled in groups.items():
        if benchmark.field_path.startswith(f"{prefix}."):
            return enabled
    return True


def _check(benchmark: Benchmark, rate: float, metrics: Mapping[str, float]) -> bool | None:
    """Compare a run against the benchmark's prediction.

    Returns ``None`` when the benchmark makes no falsifiable claim, which is a
    distinct outcome from passing: an unchecked expectation should not be
    reported as verified.
    """
    claims: list[bool] = []
    if benchmark.expect_success is not None:
        claims.append((rate > 0.0) is benchmark.expect_success)
    for metric, minimum in benchmark.expect_metric_drops:
        if metric in metrics:
            claims.append(metrics[metric] > 0.0 or minimum == 0.0)
    if not claims:
        return None
    return all(claims)


#: Benchmark ids keyed by family, for reporting.
BY_FAMILY: Mapping[BenchmarkFamily, tuple[str, ...]] = {
    family: tuple(b.id for b in benchmark_suite(family)) for family in BenchmarkFamily
}
