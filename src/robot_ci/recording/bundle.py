"""Turn written packages into the ZIP files a reviewer downloads.

Two shapes are produced
-----------------------
* one ZIP per experiment, holding that experiment's video, images, JSON,
  measured logs and reproduction instructions; and
* one suite ZIP holding every completed package, its summary table, and an index
  of the conditions that did not run.

The honesty rule
----------------
A ZIP is only built for an experiment that ran. A condition that was declared
unavailable contributes its summary *row*, not a fabricated package, so the
archive and the index cannot disagree about what exists. Every archive is
reopened and verified after being written: building one successfully is not the
same as a reviewer being able to open it.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from robot_ci.core.logging import get_logger

__all__ = ["write_suite_summary", "zip_package", "zip_suite"]

_LOG = get_logger(__name__)

#: Fixed timestamp inside archives, so two runs of one experiment produce
#: byte-comparable files and a diff of the ZIPs is meaningful.
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _add(archive: zipfile.ZipFile, arcname: str, payload: bytes) -> None:
    """Add one entry with deterministic metadata."""
    info = zipfile.ZipInfo(arcname, date_time=_FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def zip_package(directory: Path, destination: Path) -> Path:
    """Zip one experiment package and verify it can be reopened.

    Args:
        directory: The package directory to archive.
        destination: ``.zip`` path to write.

    Returns:
        The written archive path.

    Raises:
        RuntimeError: If the archive does not reopen, or a file is missing.
    """
    directory = Path(directory)
    destination = Path(destination)
    if not directory.is_dir():
        raise FileNotFoundError(f"package directory does not exist: {directory}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            _add(archive, str(path.relative_to(directory)), path.read_bytes())

    with zipfile.ZipFile(destination) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise RuntimeError(f"zip is corrupt at {broken}")
        names = set(archive.namelist())
    missing = {str(p.relative_to(directory)) for p in files} - names
    if missing:
        raise RuntimeError(f"zip is missing {len(missing)} file(s): {sorted(missing)[:5]}")
    return destination


def _rows(report: Any) -> list[dict[str, Any]]:
    """Flatten the report into one summary row per entry."""
    rows = []
    for package in report.packages:
        experiment = package.experiment
        rows.append(
            {
                "key": package.entry.key,
                "category": package.entry.category,
                "title": package.entry.title,
                "condition": package.entry.condition,
                "status": package.status.value,
                "detail": package.entry.reason
                or (experiment.status_detail if experiment is not None else "not run"),
                "seed": experiment.seed if experiment is not None else None,
                "policy": experiment.policy if experiment is not None else package.entry.policy,
                "backend": experiment.backend if experiment is not None else None,
                "steps": experiment.steps if experiment is not None else 0,
                "task_success": (
                    experiment.metrics.get("task_success") if experiment is not None else None
                ),
                "joint_motion_rad": (
                    round(experiment.joint_motion_rad(), 6) if experiment is not None else 0.0
                ),
                "camera_frames": (
                    experiment.frames_with_image if experiment is not None else 0
                ),
                "video": bool(package.artifacts and package.artifacts.video),
                "zip": f"{package.entry.key}.zip" if package.status.is_attempt() else None,
                "directory": package.directory.name if package.directory is not None else None,
            }
        )
    return rows


def write_suite_summary(
    report: Any, directory: Path, *, archive: bool = True
) -> tuple[Path, Path | None]:
    """Write the suite summary, and optionally the whole-suite ZIP.

    Args:
        report: The :class:`~robot_ci.recording.suite.SuiteReport`.
        directory: Root the packages live under.
        archive: Whether to build the combined ZIP.

    Returns:
        ``(summary_path, zip_path_or_None)``.
    """
    directory = Path(directory)
    rows = _rows(report)
    summary = {
        "schema": "robot_ci.visual_experiment.suite/1",
        "counts": report.counts(),
        "entries": rows,
        "unavailable": [
            {"key": p.entry.key, "status": p.status.value, "reason": p.entry.reason}
            for p in report.unavailable
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    markdown_path = directory / "SUITE.md"
    markdown_path.write_text(_suite_markdown(report, rows), encoding="utf-8")
    zip_path = zip_suite(report, directory / "complete_experiment_suite.zip") if archive else None
    return summary_path, zip_path


def _suite_markdown(report: Any, rows: list[dict[str, Any]]) -> str:
    """Render the suite summary as Markdown, statuses verbatim."""
    counts = ", ".join(f"{k}={v}" for k, v in report.counts().items())
    lines = [
        "# ROBOT CI visual experiment suite",
        "",
        f"Status counts: {counts}",
        "",
        "Only conditions that actually ran have a package. Everything else is",
        "listed under **Not available** with the reason it did not run.",
        "",
        "| Experiment | Condition | Status | Seed | Steps | task_success | Joint travel (rad) | Camera frames | Video |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        success = row["task_success"]
        lines.append(
            f"| `{row['key']}` | {row['condition']} | **{row['status']}** | "
            f"{row['seed'] if row['seed'] is not None else '-'} | {row['steps']} | "
            f"{'-' if success is None else f'{success:.0f}'} | "
            f"{row['joint_motion_rad']:.4f} | {row['camera_frames']} | "
            f"{'yes' if row['video'] else 'no'} |"
        )
    unavailable = [row for row in rows if row["status"] not in ("PASS", "FAIL")]
    if unavailable:
        lines += ["", "## Not available", "", "| Experiment | Status | Reason |", "|---|---|---|"]
        lines += [f"| `{r['key']}` | {r['status']} | {r['detail']} |" for r in unavailable]
    lines.append("")
    return "\n".join(lines)


def zip_suite(report: Any, destination: Path) -> Path:
    """Build the combined suite ZIP from the individual packages.

    Each completed package is added whole, plus its own ZIP when present, the
    summary JSON and the Markdown index.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, Path]] = []
    for package in report.packages:
        directory = package.directory
        if directory is None or not directory.is_dir():
            continue
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            files.append((f"{package.entry.key}/{path.relative_to(directory)}", path))
        own_zip = directory.with_suffix(".zip")
        if own_zip.exists():
            files.append((f"packages/{own_zip.name}", own_zip))
    root = Path(report.directory) if report.directory is not None else destination.parent
    for extra in ("suite_summary.json", "SUITE.md"):
        candidate = Path(root) / extra
        if candidate.exists():
            files.append((extra, candidate))

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in files:
            _add(archive, arcname, path.read_bytes())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("suite zip failed verification")
    return destination
