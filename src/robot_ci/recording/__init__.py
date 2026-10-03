"""Recorded runs of real experiments: capture, render, package.

The modules here exist to make one claim checkable: that a published result
came from a simulator that actually ran. They do that by recording what the
simulation produced -- joint states, camera frames, ground truth -- while the
episode is in flight, and by refusing to invent anything to fill a gap.

The pieces, in order:

* :mod:`robot_ci.recording.recorder` -- the recording proxy and the run loop.
* :mod:`robot_ci.recording.render3d` -- the 3-D view from recorded joint states.
* :mod:`robot_ci.recording.package` -- video, stills, manifest, results, README.
* :mod:`robot_ci.recording.suite` -- which conditions run, and how they are labelled.
* :mod:`robot_ci.recording.bundle` -- the per-experiment and suite ZIP files.
"""

from robot_ci.recording.bundle import write_suite_summary, zip_package, zip_suite
from robot_ci.recording.launcher import ExperimentLauncher, LaunchStatus, artifacts_root
from robot_ci.recording.package import ArtifactSet, ffmpeg_binary, write_artifacts
from robot_ci.recording.recorder import (
    ExperimentStatus,
    FrameSample,
    JointSample,
    RecordedExperiment,
    RecordingBackend,
    RecordingResult,
    run_recorded_experiment,
)
from robot_ci.recording.suite import (
    RunPackage,
    SuiteEntry,
    SuiteReport,
    default_suite,
    run_package_suite,
)

__all__ = [
    "ArtifactSet",
    "ExperimentLauncher",
    "ExperimentStatus",
    "FrameSample",
    "JointSample",
    "LaunchStatus",
    "RecordedExperiment",
    "RecordingBackend",
    "RecordingResult",
    "RunPackage",
    "SuiteEntry",
    "SuiteReport",
    "artifacts_root",
    "default_suite",
    "ffmpeg_binary",
    "run_package_suite",
    "run_recorded_experiment",
    "write_artifacts",
    "write_suite_summary",
    "zip_package",
    "zip_suite",
]
