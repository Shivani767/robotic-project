r"""Record the visual experiment suite and write one ZIP per condition.

This is the batch entry point: it runs the suite without anyone clicking through
the dashboard, writes a package per condition, zips each one, and produces the
suite summary. Run it inside the Gazebo container::

    docker run --rm -v "$PWD":/workspace/robot-ci -w /workspace/robot-ci \\
        robot-ci-gazebo-demo:latest bash -c \\
        'source /opt/ros/jazzy/setup.bash && python3 scripts/record_suite.py --only nominal_baseline'

Only conditions that actually ran get a ZIP. Anything declared unavailable
appears in the summary with its reason, and the exit code is non-zero when no
condition completed, so a CI job cannot read an empty suite as a pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_ci.recording.bundle import write_suite_summary, zip_package
from robot_ci.recording.suite import SuiteEntry, default_suite, run_package_suite

#: Where packages go unless overridden. Git-ignored: these are generated media.
DEFAULT_OUTPUT = Path("artifacts/visual_reports")


def main(argv: list[str] | None = None) -> int:
    """Run the suite, write packages and ZIPs, and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="gazebo", help="simulator backend")
    parser.add_argument(
        "--only", action="append", default=None, help="run only these suite keys (repeatable)"
    )
    parser.add_argument("--seed", type=int, default=None, help="override the episode seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument("--no-video", action="store_true", help="skip video encoding")
    args = parser.parse_args(argv)

    entries = default_suite()
    if args.only:
        wanted = set(args.only)
        entries = [entry for entry in entries if entry.key in wanted]
        missing = wanted - {entry.key for entry in entries}
        if missing:
            print(f"unknown suite key(s): {sorted(missing)}", file=sys.stderr)
            return 2
    if args.seed is not None:
        entries = [
            entry if not entry.runnable else SuiteEntry(**{**entry.__dict__, "seed": args.seed})
            for entry in entries
        ]

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    report = run_package_suite(
        entries, output, backend_name=args.backend, make_video=not args.no_video
    )

    zipped = []
    for package in report.packages:
        if package.directory is None:
            continue
        archive = package.directory.with_suffix(".zip")
        try:
            zip_package(package.directory, archive)
            zipped.append(archive)
        except Exception as exc:
            print(f"zip failed for {package.entry.key}: {exc}", file=sys.stderr)

    summary_path, suite_zip = write_suite_summary(report, output)
    print("\n=== suite summary ===")
    for package in report.packages:
        experiment = package.experiment
        if experiment is None:
            print(f"{package.entry.key:<28} {package.status.value:<16} {package.entry.reason}")
            continue
        print(
            f"{package.entry.key:<28} {package.status.value:<16} steps={experiment.steps:<4}"
            f" motion={experiment.joint_motion_rad():.4f}rad"
            f" camera_frames={experiment.frames_with_image}"
        )
    print("\ncounts:", ", ".join(f"{k}={v}" for k, v in report.counts().items()))
    print("summary:", summary_path)
    for archive in zipped:
        print("zip:", archive)
    if suite_zip is not None:
        print("suite zip:", suite_zip)
    return 0 if report.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
