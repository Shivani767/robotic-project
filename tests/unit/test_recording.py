"""Tests for recording, rendering, packaging and the suite's labelling.

These cover the failure modes that are *silent*: a figure that quietly omits
data, a manifest whose metadata disagrees with the run, a ZIP a reviewer cannot
open, and -- most importantly -- a status promoted from ``FAIL`` or
``NOT_IMPLEMENTED`` into something that reads as success. None of those raise,
which is exactly why they need a test.

Nothing here runs a simulator. The recording is driven from constructed samples,
so these run on any machine; the evidence that the pipeline works against a
*real* simulator lives under ``experiments/results/gazebo/``.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from robot_ci.recording.bundle import write_suite_summary, zip_package, zip_suite
from robot_ci.recording.package import (
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    _compose_frames,
    encode_video,
    ffmpeg_binary,
    write_artifacts,
)
from robot_ci.recording.recorder import (
    ExperimentStatus,
    FrameSample,
    JointSample,
    RecordedExperiment,
)
from robot_ci.recording.suite import RunPackage, SuiteEntry, SuiteReport, default_suite


def _joints(count: int = 12, moving: bool = True) -> list[JointSample]:
    """Build joint samples, optionally with real travel between them."""
    samples = []
    for index in range(count):
        angle = 0.4 + (0.9 * index / max(count - 1, 1) if moving else 0.0)
        samples.append(
            JointSample(
                step_index=index,
                sim_time_s=index * 0.05,
                wall_time_s=1_700_000_000.0 + index * 0.2,
                joint_names=("j1", "j2", "j3", "j4", "j5", "j6", "j7"),
                positions=(angle, angle * 0.5, -angle, 0.2, 0.1, 0.3, -0.1),
                end_effector=(0.1 * index, 0.2, 0.3 + 0.01 * index),
                gripper_width_m=0.04,
            )
        )
    return samples


def _frames(count: int = 6, with_image: bool = True) -> list[FrameSample]:
    """Build camera frames with recognisable content."""
    frames = []
    for index in range(count):
        rgb = None
        if with_image:
            rgb = np.full((12, 16, 3), (10 + index * 20, 40, 90), dtype=np.uint8)
            rgb[2:6, 3:7] = (220, 30, 30)
        frames.append(
            FrameSample(
                step_index=index,
                sim_time_s=index * 0.05,
                wall_time_s=1_700_000_000.0 + index * 0.2,
                rgb=rgb,
                depth=np.linspace(0.4, 1.0, 192, dtype=np.float32).reshape(12, 16),
                width_px=16,
                height_px=12,
                source="camera_rig",
                valid=with_image,
            )
        )
    return frames


def _experiment(
    status: ExperimentStatus = ExperimentStatus.PASS, with_camera: bool = True
) -> RecordedExperiment:
    """Build a recorded experiment for packaging tests."""
    frames = _frames(with_image=with_camera)
    return RecordedExperiment(
        experiment_id="demo",
        category="nominal_baseline",
        title="Demo experiment",
        condition="sensors=nominal",
        task="pick_and_place",
        policy="policy_v1",
        seed=3,
        status=status,
        status_detail=(
            "the task reported success" if status is ExperimentStatus.PASS else "did not succeed"
        ),
        backend="gazebo",
        scenario_id="nominal",
        config_hash="a" * 64,
        git_commit="deadbeef",
        command="python3 scripts/record_suite.py --only demo",
        joints=_joints(),
        frames=frames,
        events=[
            {"sim_time_s": 0.0, "label": "approach", "kind": "phase"},
            {"sim_time_s": 0.4, "label": "grasp", "kind": "phase"},
            {"sim_time_s": 0.6, "label": "placed", "kind": "PASS"},
        ],
        metrics={"task_success": 1.0 if status is ExperimentStatus.PASS else 0.0, "steps": 12.0},
        steps=12,
        duration_sim_s=0.6,
        duration_wall_s=2.4,
        started_wall=1_700_000_000.0,
        finished_wall=1_700_000_002.4,
        frames_observed=len(frames),
        conditions={"depth.sigma_mm": 0.0},
        limitations=["a stated limitation"],
        notes=["a note"],
        versions={"robot_ci": "0.1.0", "python": "3.12.3"},
    )
# --------------------------------------------------------------- measurement
def test_joint_motion_reflects_recorded_travel() -> None:
    """Motion is computed from the samples, so a still robot reports zero."""
    assert _experiment().joint_motion_rad() > 0.5
    still = _experiment()
    object.__setattr__(still, "joints", _joints(moving=False))
    assert still.joint_motion_rad() == pytest.approx(0.0)


def test_camera_is_real_only_for_rendering_backends() -> None:
    """A stub backend's frames must not be labelled simulator footage."""
    assert _experiment().camera_is_real is True
    stub = _experiment()
    object.__setattr__(stub, "backend", "mock")
    assert stub.camera_is_real is False


def test_clock_offset_is_sim_minus_wall() -> None:
    """The published offset is the difference of the two real clocks."""
    assert _experiment().clock_offset_s() == pytest.approx(0.6 - 2.2, abs=1e-6)


# ------------------------------------------------------------------ packaging
def test_write_artifacts_produces_every_declared_file(tmp_path: Path) -> None:
    """All the declared artifacts are written for a recording that has data."""
    artifacts = write_artifacts(_experiment(), tmp_path / "demo", make_video=False)
    assert artifacts.manifest_path is not None and artifacts.manifest_path.exists()
    assert artifacts.results_path is not None and artifacts.results_path.exists()
    assert artifacts.readme_path is not None and artifacts.readme_path.exists()
    assert set(artifacts.images) == {
        "overview.png", "robot_view.png", "camera_view.png", "metrics.png", "timeline.png",
    }
    for path in artifacts.images.values():
        assert path.exists() and path.stat().st_size > 0
    assert artifacts.warnings == []


def test_figures_are_written_without_camera_data(tmp_path: Path) -> None:
    """A run with no camera frame still produces a package, and says so."""
    artifacts = write_artifacts(
        _experiment(with_camera=False), tmp_path / "demo", make_video=False
    )
    assert artifacts.images["camera_view.png"].exists()
    assert "none were captured" in artifacts.readme_path.read_text(encoding="utf-8")


def test_figures_are_written_without_joint_samples(tmp_path: Path) -> None:
    """A run with no joint samples still produces a package, and says so."""
    experiment = _experiment()
    object.__setattr__(experiment, "joints", [])
    artifacts = write_artifacts(experiment, tmp_path / "demo", make_video=False)
    assert artifacts.images["robot_view.png"].exists()
    assert artifacts.images["metrics.png"].exists()


def test_manifest_metadata_matches_the_run(tmp_path: Path) -> None:
    """Manifest metadata is the run's own, not a template's."""
    experiment = _experiment()
    artifacts = write_artifacts(experiment, tmp_path / "demo", make_video=False)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == experiment.seed
    assert manifest["policy"] == experiment.policy
    assert manifest["config_hash"] == experiment.config_hash
    assert manifest["command"] == experiment.command
    assert manifest["recording"]["joint_samples"] == len(experiment.joints)
    assert manifest["recording"]["camera_frames_with_image"] == experiment.frames_with_image


def test_manifest_hashes_match_the_files_on_disk(tmp_path: Path) -> None:
    """Recorded hashes must correspond to the bytes actually written."""
    artifacts = write_artifacts(_experiment(), tmp_path / "demo", make_video=False)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    hashes = manifest["artifact_sha256"]
    assert hashes, "manifest recorded no artifact hashes"
    for name, digest in hashes.items():
        actual = hashlib.sha256((artifacts.directory / name).read_bytes()).hexdigest()
        assert digest == actual, f"hash mismatch for {name}"


def test_results_do_not_promote_a_failure(tmp_path: Path) -> None:
    """A FAIL stays a FAIL in the JSON and is not marked as succeeded."""
    artifacts = write_artifacts(
        _experiment(ExperimentStatus.FAIL), tmp_path / "demo", make_video=False
    )
    results = json.loads(artifacts.results_path.read_text(encoding="utf-8"))
    assert results["status"] == "FAIL"
    assert results["succeeded"] is False
    assert results["attempted"] is True
    assert results["metrics"]["task_success"] == 0.0


def test_readme_states_status_and_reproduction(tmp_path: Path) -> None:
    """The README carries the status and the command that reproduces it."""
    artifacts = write_artifacts(
        _experiment(ExperimentStatus.FAIL), tmp_path / "demo", make_video=False
    )
    readme = artifacts.readme_path.read_text(encoding="utf-8")
    assert "**Status: `FAIL`**" in readme
    assert "python3 scripts/record_suite.py --only demo" in readme
    assert "a stated limitation" in readme


def test_missing_video_is_declared_not_implied(tmp_path: Path) -> None:
    """A package with no video says so rather than implying one exists."""
    artifacts = write_artifacts(_experiment(), tmp_path / "demo", make_video=False)
    assert artifacts.video is None
    assert "No video in this package" in artifacts.readme_path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------- zips
def test_zip_package_contains_every_file_and_reopens(tmp_path: Path) -> None:
    """The archive holds the package and passes its own integrity check."""
    package = tmp_path / "demo"
    write_artifacts(_experiment(), package, make_video=False)
    archive = zip_package(package, tmp_path / "demo.zip")
    with zipfile.ZipFile(archive) as opened:
        names = set(opened.namelist())
        assert opened.testzip() is None
    for name in ("manifest.json", "results.json", "README.md", "overview.png"):
        assert name in names
    assert archive.stat().st_size > 0


def test_zip_is_deterministic(tmp_path: Path) -> None:
    """Two archives of the same package are byte-identical."""
    package = tmp_path / "demo"
    write_artifacts(_experiment(), package, make_video=False)
    first = zip_package(package, tmp_path / "a.zip").read_bytes()
    second = zip_package(package, tmp_path / "b.zip").read_bytes()
    assert first == second


def test_zip_package_rejects_a_missing_package(tmp_path: Path) -> None:
    """Archiving a directory that does not exist fails loudly."""
    with pytest.raises(FileNotFoundError):
        zip_package(tmp_path / "absent", tmp_path / "absent.zip")


# --------------------------------------------------------------------- suite
def _declared_entry() -> SuiteEntry:
    """Build an entry that is declared unavailable."""
    return SuiteEntry(
        key="boundary_search", category="boundary_search", title="Boundary search",
        condition="n/a", status=ExperimentStatus.NOT_IMPLEMENTED,
        reason="no search implementation",
    )


def test_unavailable_entries_are_not_given_packages(tmp_path: Path) -> None:
    """Declared-unavailable entries contribute a summary row and no directory."""
    report = SuiteReport(
        packages=[RunPackage(entry=_declared_entry())],
        directory=tmp_path,
        unavailable=[RunPackage(entry=_declared_entry())],
    )
    summary_path, suite_zip = write_suite_summary(report, tmp_path, archive=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["entries"][0]["status"] == "NOT_IMPLEMENTED"
    assert summary["entries"][0]["zip"] is None
    assert summary["entries"][0]["directory"] is None
    assert "no search implementation" in summary["unavailable"][0]["reason"]
    assert suite_zip is None
    assert report.completed == []


def test_suite_markdown_lists_what_did_not_run(tmp_path: Path) -> None:
    """The Markdown index names every unavailable condition and its reason."""
    report = SuiteReport(
        packages=[RunPackage(entry=_declared_entry())],
        directory=tmp_path,
        unavailable=[RunPackage(entry=_declared_entry())],
    )
    write_suite_summary(report, tmp_path, archive=False)
    markdown = (tmp_path / "SUITE.md").read_text(encoding="utf-8")
    assert "NOT_IMPLEMENTED" in markdown
    assert "no search implementation" in markdown
    assert "Not available" in markdown


def test_suite_zip_omits_entries_that_never_ran(tmp_path: Path) -> None:
    """The suite archive cannot imply a package for an unrun condition."""
    report = SuiteReport(directory=tmp_path)
    report.unavailable = []
    archive = zip_suite(report, tmp_path / "suite.zip")
    with zipfile.ZipFile(archive) as opened:
        assert opened.testzip() is None
        assert opened.namelist() == []


def test_default_suite_declares_what_it_cannot_run() -> None:
    """Unsupported categories are declared, never silently attempted."""
    suite = {entry.key: entry for entry in default_suite()}
    replay = suite["counterexample_reproduction"]
    assert replay.status is ExperimentStatus.NOT_IMPLEMENTED
    assert replay.runnable is False
    assert replay.reason
    assert suite["boundary_search"].status is ExperimentStatus.NOT_IMPLEMENTED
    for key in ("nominal_baseline", "depth_noise_0mm", "depth_noise_8mm"):
        assert suite[key].runnable is True


def test_suite_conditions_actually_differ() -> None:
    """Perturbed conditions really move the thing they claim to perturb."""
    from robot_ci.gazebo.world import object_pose_from_scenario

    suite = {entry.key: entry for entry in default_suite()}
    base = object_pose_from_scenario(suite["nominal_baseline"].build())
    moved = object_pose_from_scenario(suite["object_perturbation"].build())
    assert moved[0] == pytest.approx(base[0] + 0.06, abs=1e-9)
    assert suite["depth_noise_8mm"].build().sensors.depth.sigma_mm == pytest.approx(8.0)
    assert suite["depth_noise_0mm"].build().sensors.depth.sigma_mm == pytest.approx(0.0)
    assert suite["joint_perturbation"].build().sensors.robot_state.joint_noise_std_rad > 0.0


def test_suite_policy_conditions_use_different_policies() -> None:
    """The policy-comparison entry names a policy the baseline does not."""
    suite = {entry.key: entry for entry in default_suite()}
    assert suite["baseline_vs_candidate"].policy != suite["nominal_baseline"].policy


def test_suite_resolves_policy_configs() -> None:
    """Every runnable entry resolves to a real policy implementation."""
    from robot_ci.recording.suite import _policy_config

    for entry in default_suite():
        if entry.runnable:
            assert _policy_config(entry.policy).implementation


def test_repeated_seed_entries_cover_three_seeds_independently() -> None:
    """Repeated-seed variability is three packages, not one package plus prose.

    Each of seeds 3, 11 and 42 has its own suite key, runnable builder and
    episode seed, so filtering by ``--only`` cannot silently drop two-thirds of
    the evidence. Each package carries the same cross-reference note and caveat
    so a single package read in isolation still names the other two.
    """
    from robot_ci.recording.suite import _notes

    suite = {entry.key: entry for entry in default_suite()}
    seeds = {"repeated_seed_3": 3, "repeated_seed_11": 11, "repeated_seed_42": 42}
    assert set(seeds) <= set(suite), "all three seeds must be separate suite keys"
    for key, seed in seeds.items():
        entry = suite[key]
        assert entry.runnable is True
        assert entry.category == "repeated_seed"
        assert entry.seed == seed
        assert entry.condition == f"seed={seed}"
        note = " ".join([*_notes(entry), *entry.limitations])
        assert "3" in note and "11" in note and "42" in note
        assert "variability" in note.lower()


# --------------------------------------------------------------------- video
def test_compose_frames_are_1080p_and_labelled() -> None:
    """Every composed frame is the declared size and labels both panes."""
    frames = list(_compose_frames(_experiment()))
    assert len(frames) > 1
    for image in (frames[0], frames[-1]):
        assert image.size == (VIDEO_WIDTH, VIDEO_HEIGHT)


def test_compose_frames_survive_no_camera_and_no_joints() -> None:
    """Composition still yields frames when the run produced almost nothing."""
    experiment = _experiment(with_camera=False)
    object.__setattr__(experiment, "joints", [])
    frames = list(_compose_frames(experiment))
    assert frames and frames[0].size == (VIDEO_WIDTH, VIDEO_HEIGHT)


def test_encode_video_writes_a_playable_mp4(tmp_path: Path) -> None:
    """The MP4 is real when an encoder exists, and honestly absent otherwise."""
    if ffmpeg_binary() is None:
        pytest.skip("no ffmpeg available in this environment")
    arrays = [
        np.asarray(frame, dtype=np.uint8)
        for frame in list(_compose_frames(_experiment()))[:12]
    ]
    destination = tmp_path / "experiment.mp4"
    assert encode_video(arrays, destination) is True
    assert destination.exists() and destination.stat().st_size > 0
    assert destination.read_bytes()[4:8] == b"ftyp"


# ----------------------------------------------------------------- launcher
def test_launcher_refuses_unknown_keys(tmp_path: Path) -> None:
    """An unavailable key is refused, not run."""
    from robot_ci.recording.launcher import ExperimentLauncher

    launcher = ExperimentLauncher(root=tmp_path)
    started, message = launcher.start("boundary_search")
    assert started is False
    assert "unavailable" in message
    assert launcher.status().running is False


def test_launcher_refuses_concurrent_runs(tmp_path: Path) -> None:
    """A second start is refused while one run holds the lock."""
    import json as _json

    from robot_ci.recording.launcher import ExperimentLauncher, _status_file

    launcher = ExperimentLauncher(root=tmp_path)
    _status_file(tmp_path).write_text(
        _json.dumps({"running": True, "key": "nominal_baseline", "started_wall": 1.0}),
        encoding="utf-8",
    )
    started, message = launcher.start("nominal_baseline")
    assert started is False
    assert "nominal_baseline" in message


def test_launcher_suite_rows_mark_run_packages(tmp_path: Path) -> None:
    """Suite rows show which keys have packages and which cannot run."""
    from robot_ci.recording.launcher import ExperimentLauncher

    (tmp_path / "nominal_baseline").mkdir()
    launcher = ExperimentLauncher(root=tmp_path)
    rows = {row["key"]: row for row in launcher.suite()}
    assert rows["nominal_baseline"]["package"] == "nominal_baseline"
    assert rows["nominal_baseline"]["runnable"] is True
    assert rows["boundary_search"]["runnable"] is False
    assert rows["boundary_search"]["zip"] is None


def test_artifact_index_lists_disk_contents(tmp_path: Path) -> None:
    """The index mirrors what is on disk, nothing invented."""
    from robot_ci.gazebo.live import _artifact_index

    assert _artifact_index(tmp_path / "missing") == []
    package = tmp_path / "nominal_baseline"
    package.mkdir()
    (package / "experiment.mp4").write_bytes(b"fake")
    (tmp_path / "nominal_baseline.zip").write_bytes(b"zip")
    (tmp_path / "complete_experiment_suite.zip").write_bytes(b"suite")
    index = _artifact_index(tmp_path)
    package_entry = next(entry for entry in index if entry.get("files") == ["experiment.mp4"])
    assert package_entry["key"] == "nominal_baseline"
    assert package_entry["package"] == "artifacts/nominal_baseline"
    assert package_entry["zip"] == "artifacts/nominal_baseline.zip"
    suite_entry = next(entry for entry in index if "files" not in entry)
    assert suite_entry["zip"] == "artifacts/complete_experiment_suite.zip"
def test_encode_video_leaves_no_file_when_it_cannot(tmp_path: Path) -> None:
    """A failed encode leaves no partial MP4 behind."""
    destination = tmp_path / "experiment.mp4"
    assert encode_video([], destination) is False
    assert not destination.exists()
