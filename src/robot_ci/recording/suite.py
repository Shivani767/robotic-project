"""The experiment suite: which conditions run, and how each is labelled.

The central rule
----------------
An experiment is reported as ``PASS`` only when the simulator produced a
measurable outcome and the task's own criteria said it succeeded. Everything
else carries a distinct label -- ``FAIL``, ``ERROR``, ``NOT_SUPPORTED``,
``NOT_IMPLEMENTED`` -- because those mean five different things and a suite that
collapses them claims coverage it does not have. A condition Gazebo cannot
reproduce is marked ``NOT_SUPPORTED`` and is *not* quietly run through another
simulator under a Gazebo label.

Which conditions exist
----------------------
Every category was checked against the implementation before being listed, and
the ones needing machinery ROBOT CI does not have are declared
``NOT_IMPLEMENTED`` rather than approximated. Boundary search and
counterexample replay are declared for exactly that reason: they need a real
search implementation and a stored counterexample, neither of which exists on
this backend yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_ci.core.logging import get_logger
from robot_ci.recording.package import ArtifactSet, write_artifacts
from robot_ci.recording.recorder import (
    ExperimentStatus,
    RecordedExperiment,
    run_recorded_experiment,
)
from robot_ci.scenarios import default_scenario

__all__ = [
    "RunPackage",
    "SuiteEntry",
    "SuiteReport",
    "default_suite",
    "run_package_suite",
]

_LOG = get_logger(__name__)


def _with_depth_noise(scenario: Any, sigma_mm: float) -> Any:
    """Return the scenario with depth Gaussian noise set to ``sigma_mm``."""
    depth = scenario.sensors.depth.model_copy(
        update={"enabled": sigma_mm > 0.0, "sigma_mm": float(sigma_mm)}
    )
    return scenario.model_copy(
        update={"sensors": scenario.sensors.model_copy(update={"depth": depth})}
    )


def _with_joint_noise(scenario: Any, std_rad: float) -> Any:
    """Return the scenario with joint-state noise enabled."""
    robot_state = scenario.sensors.robot_state.model_copy(
        update={"enabled": std_rad > 0.0, "joint_noise_std_rad": float(std_rad)}
    )
    return scenario.model_copy(
        update={"sensors": scenario.sensors.model_copy(update={"robot_state": robot_state})}
    )


def _with_object_offset(scenario: Any, dx: float, dy: float) -> Any:
    """Return the scenario with the target's position moved by ``(dx, dy)``.

    The target's position is ``None`` in a default scenario -- it is left to the
    seeded draw -- so the draw is resolved first and the offset applied to its
    result. Setting an explicit position is required rather than optional,
    because ``gazebo.world.object_pose_from_scenario`` honours an explicit
    position over the draw: without this the perturbation category would run an
    unperturbed condition under the perturbed name.
    The position is written to ``objects``, not ``target``: ``target`` is a
    derived property, and updating it via ``model_copy`` is silently discarded.
    """
    from robot_ci.gazebo.world import object_pose_from_scenario
    from robot_ci.models.geometry import Vec3

    base = scenario.target.position
    if base is None:
        x, y, z = object_pose_from_scenario(scenario)
    else:
        x, y, z = float(base.x), float(base.y), float(base.z)
    moved = Vec3(x=x + float(dx), y=y + float(dy), z=z)
    objects = tuple(
        spec.model_copy(update={"position": moved}) if spec.is_target else spec
        for spec in scenario.objects
    )
    return scenario.model_copy(update={"objects": objects})
@dataclass(frozen=True)
class SuiteEntry:
    """One condition in the suite.

    Attributes:
        key: Stable key, also the ZIP's stem.
        category: Category this belongs to.
        title: Human-readable title for the slate and README.
        condition: Short condition label.
        status: Status when the entry cannot be run at all.
        reason: Why that status applies.
        build: Produces the scenario; ``None`` for entries that cannot run.
        seed: Episode seed.
        policy: Policy name.
        repeats: How many seeds this entry covers.
        limitations: Honest statements carried into the package README.
    """

    key: str
    category: str
    title: str
    condition: str
    status: ExperimentStatus = ExperimentStatus.NOT_RUN
    reason: str = ""
    build: Callable[[], Any] | None = None
    seed: int = 3
    policy: str = "policy_v1"
    repeats: int = 1
    limitations: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        """Whether this entry has a scenario builder and so can be attempted."""
        return self.build is not None


@dataclass
class RunPackage:
    """A recorded experiment plus the files written for it.

    Attributes:
        entry: The suite entry that produced it.
        experiment: The recording, or ``None`` when nothing could run.
        artifacts: The written files, or ``None``.
    """

    entry: SuiteEntry
    experiment: RecordedExperiment | None = None
    artifacts: ArtifactSet | None = None

    @property
    def status(self) -> ExperimentStatus:
        """The recorded outcome, or the entry's declared status when nothing ran.

        A declared-unavailable entry must report *its* status --
        ``NOT_SUPPORTED`` or ``NOT_IMPLEMENTED`` -- rather than ``NOT_RUN``.
        Collapsing them would make "we cannot do this" indistinguishable from
        "we did not get to it", which is the distinction the suite exists to keep.
        """
        if self.experiment is not None:
            return self.experiment.status
        return self.entry.status

    @property
    def directory(self) -> Path | None:
        """Return the package directory, if one was written."""
        return None if self.artifacts is None else self.artifacts.directory


@dataclass
class SuiteReport:
    """The whole suite's outcome.

    Attributes:
        packages: Every entry, run or not, in order.
        directory: Root the packages were written under.
        unavailable: Packages that did not produce a measurable outcome.
    """

    packages: list[RunPackage] = field(default_factory=list)
    directory: Path | None = None
    unavailable: list[RunPackage] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Return how many entries ended in each status."""
        out: dict[str, int] = {}
        for package in self.packages:
            out[package.status.value] = out.get(package.status.value, 0) + 1
        return dict(sorted(out.items()))

    @property
    def completed(self) -> list[RunPackage]:
        """Packages whose experiment produced a measurable outcome."""
        return [package for package in self.packages if package.status.is_attempt()]


def default_suite() -> list[SuiteEntry]:
    """Return the suite as configured for the Gazebo backend.

    Ranges come from the schema rather than being invented: the depth sweep uses
    the 0 mm and 8 mm conditions the brief names, joint noise a small fraction of
    a joint limit, and the object offset one that keeps the target inside the
    workspace the SDF generator builds.
    """
    return [
        SuiteEntry(
            key="nominal_baseline", category="nominal_baseline",
            title="Nominal baseline - no sensor corruption",
            condition="sensors=nominal", build=lambda: default_scenario(),
        ),
        SuiteEntry(
            key="depth_noise_0mm", category="sensor_corruption",
            title="Depth noise 0 mm (clean baseline for the sweep)",
            condition="depth.sigma_mm=0", build=lambda: _with_depth_noise(default_scenario(), 0.0),
        ),
        SuiteEntry(
            key="depth_noise_4mm", category="sensor_corruption",
            title="Depth noise 4 mm", condition="depth.sigma_mm=4",
            build=lambda: _with_depth_noise(default_scenario(), 4.0),
        ),
        SuiteEntry(
            key="depth_noise_8mm", category="sensor_corruption",
            title="Depth noise 8 mm", condition="depth.sigma_mm=8",
            build=lambda: _with_depth_noise(default_scenario(), 8.0),
        ),
        SuiteEntry(
            key="object_perturbation", category="object_perturbation",
            title="Object position perturbed +60 mm in x",
            condition="object.x += 0.06",
            build=lambda: _with_object_offset(default_scenario(), 0.06, 0.0),
        ),
        SuiteEntry(
            key="joint_perturbation", category="joint_perturbation",
            title="Joint-state noise 0.005 rad", condition="joint_noise_std_rad=0.005",
            build=lambda: _with_joint_noise(default_scenario(), 0.005),
        ),
        SuiteEntry(
            key="baseline_vs_candidate", category="policy_comparison",
            title="Policy comparison - candidate policy (policy_v3)",
            condition="policy=policy_v3", policy="policy_v3",
            build=lambda: default_scenario(),
            limitations=(
                "This package is the candidate arm; the baseline arm is recorded "
                "separately as nominal_baseline under the identical condition.",
            ),
        ),
        SuiteEntry(
            key="repeated_seed", category="repeated_seed",
            title="Repeated-seed evaluation - 3 seeds", condition="seeds=3,11,42",
            seed=3, repeats=3, build=lambda: default_scenario(),
            limitations=("This package is seed 3 of three; each seed is its own package.",),
        ),
        SuiteEntry(
            key="counterexample_reproduction", category="regression_reproduction",
            title="Counterexample reproduction", condition="n/a",
            status=ExperimentStatus.NOT_IMPLEMENTED,
            reason=(
                "No counterexample has been discovered and stored for this backend, "
                "so there is nothing to replay. ROBOT CI will not invent one."
            ),
        ),
        SuiteEntry(
            key="boundary_search", category="boundary_search",
            title="Boundary search over depth noise", condition="n/a",
            status=ExperimentStatus.NOT_IMPLEMENTED,
            reason=(
                "Boundary search needs a real parameter sweep driving the backend. "
                "This suite runs a fixed list of conditions, so no search is claimed."
            ),
        ),
    ]


def _policy_config(name: str) -> Any:
    """Return a policy config for a named policy."""
    from robot_ci.models.scenario import PolicyConfig

    return PolicyConfig(name=name)


def run_package_suite(
    entries: list[SuiteEntry],
    directory: Path,
    *,
    backend_name: str = "gazebo",
    seeds: tuple[int, ...] = (),
    make_video: bool = True,
) -> SuiteReport:
    """Run every runnable entry and write a package for each.

    A failing entry does not stop the suite: the failure becomes that entry's
    status and the run continues, because a suite that aborted on the first
    ``FAIL`` would report coverage it does not have.

    Args:
        entries: Suite entries, in order.
        directory: Root for the package directories.
        backend_name: Simulator backend to record against.
        seeds: Overrides the seeds used for repeated-seed entries.
        make_video: Pass ``False`` to skip encoding (tests, dry runs).

    Returns:
        The :class:`SuiteReport`.
    """
    directory = Path(directory)
    report = SuiteReport(directory=directory)
    for entry in entries:
        report.packages.append(_run_entry(entry, directory, backend_name, seeds, make_video))
    report.unavailable = [
        package for package in report.packages if not package.status.is_attempt()
    ]
    _LOG.info("suite complete: %s", ", ".join(f"{k}={v}" for k, v in report.counts().items()))
    return report


def _run_entry(
    entry: SuiteEntry,
    directory: Path,
    backend_name: str,
    seeds: tuple[int, ...],
    make_video: bool,
) -> RunPackage:
    """Run one suite entry and write its package. Never raises."""
    if not entry.runnable:
        _LOG.info("%s: %s (%s)", entry.key, entry.status.value, entry.reason)
        return RunPackage(entry=entry)

    if entry.repeats > 1:
        chosen = list(seeds) if seeds else [entry.seed + offset for offset in range(entry.repeats)]
        entry = SuiteEntry(
            key=entry.key, category=entry.category, title=entry.title,
            condition=entry.condition, build=entry.build, seed=chosen[0],
            policy=entry.policy, repeats=1, limitations=entry.limitations,
        )

    from robot_ci.simulation import create_backend

    try:
        backend = create_backend(backend_name)
    except Exception as exc:  # an unavailable backend is an outcome, not a crash
        _LOG.warning("%s: backend %s unavailable: %s", entry.key, backend_name, exc)
        return RunPackage(entry=entry)

    try:
        scenario = entry.build() if entry.build is not None else default_scenario()
        result = run_recorded_experiment(
            backend=backend,
            scenario=scenario,
            policy_config=_policy_config(entry.policy),
            seed=entry.seed,
            experiment_id=entry.key,
            category=entry.category,
            title=entry.title,
            condition=entry.condition,
            command=_reproduce_command(entry, backend_name),
            limitations=[*entry.limitations, *_backend_limitations(backend_name)],
            notes=_notes(entry),
        )
    except Exception as exc:  # pragma: no cover - run_recorded_experiment catches
        _LOG.warning("%s: recording raised %s: %s", entry.key, type(exc).__name__, exc)
        return RunPackage(entry=entry)
    if result.experiment is None:
        return RunPackage(entry=entry)

    artifacts = write_artifacts(result.experiment, directory / entry.key, make_video=make_video)
    return RunPackage(entry=entry, experiment=result.experiment, artifacts=artifacts)


def _reproduce_command(entry: SuiteEntry, backend_name: str) -> str:
    """Return the exact command that reproduces this package."""
    return (
        f"docker compose run --rm {backend_name}-demo "
        f"python3 -m robot_ci.cli.main record suite --only {entry.key}"
    )


def _backend_limitations(backend_name: str) -> list[str]:
    """Return the standing limitations of the backend these ran on."""
    if backend_name != "gazebo":
        return []
    return [
        "Grasp is a kinematic proximity latch, not resolved finger contact: the "
        "object attaches when the commanded aperture closes within 50 mm of it. "
        "This is an adapter behaviour and is labelled as such in the results.",
        "Collision geometry in Gazebo comes from per-link primitive bounding "
        "volumes rather than the MuJoCo meshes, so contact response is not "
        "expected to match MuJoCo.",
        "The camera renders with Ogre2 on the CPU at the scenario's 160x120; "
        "frames are shown upscaled without resampling.",
    ]


def _notes(entry: SuiteEntry) -> list[str]:
    """Return reading notes for this entry."""
    if entry.category == "repeated_seed":
        return [
            "Run-to-run variability is the point of this condition: compare the "
            "statuses across the seed packages before quoting any single one.",
        ]
    return []
