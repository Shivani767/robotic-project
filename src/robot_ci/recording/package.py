"""Turn a :class:`~robot_ci.recording.recorder.RecordedExperiment` into files.

What a package has to contain
-----------------------------
``experiment.mp4``, ``overview.png``, ``robot_view.png``, ``camera_view.png``,
``metrics.png``, ``timeline.png``, ``manifest.json``, ``results.json`` and a
``README.md`` that says what actually ran. Nine artifacts, and the only honest
source for all of them is the recording: every pixel here is projected from a
recorded joint sample, resized from a recorded camera frame, or plotted from a
recorded metric.

The two views, and why they are labelled
----------------------------------------
The left pane is :mod:`robot_ci.recording.render3d`: the arm drawn through the
same URDF the browser viewer uses, from the recorded joint angles. The right
pane is the simulator's own RGB output. They come from one episode's stream but
are *not* interchangeable, so each pane carries a permanent label saying which
is which. A viewer must never have to guess which one is footage.

Where the video comes from
--------------------------
Encoding is delegated to ``ffmpeg``: a system install first, then the binary
inside the ``imageio-ffmpeg`` wheel, because the demo image ships no system
ffmpeg and adding one would pull a large dependency into an otherwise small
CPU-only image. When neither exists :func:`encode_video` returns ``False`` and
the package records that fact rather than implying a video it does not have.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_ci.core.logging import get_logger
from robot_ci.recording.recorder import RecordedExperiment

__all__ = ["ArtifactSet", "encode_video", "ffmpeg_binary", "write_artifacts"]

_LOG = get_logger(__name__)

#: Video geometry: 1920x1080 at 30 fps is the target, and the pane layout below
#: is laid out for it.
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30

#: Seconds of title slate before the first frame.
SLATE_SECONDS = 2.0

_PANE_BG = (16, 18, 22)
_TEXT = (236, 240, 246)
_MUTED = (150, 158, 170)
_STATUS_COLOURS = {
    "PASS": (74, 190, 120),
    "FAIL": (222, 96, 84),
    "ERROR": (232, 168, 68),
    "NOT_RUN": (130, 138, 150),
    "NOT_SUPPORTED": (130, 138, 150),
    "NOT_IMPLEMENTED": (130, 138, 150),
}


def _font(size: int) -> Any:
    """Return a PIL font, falling back to the bitmap default when none loads.

    A missing TrueType font must not cost the package its labels, so the
    fallback is deliberate rather than an error.
    """
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:  # pragma: no cover - unreadable font file
                continue
    return ImageFont.load_default()


def _fit(image: Any, size: tuple[int, int]) -> Any:
    """Letterbox ``image`` into ``size`` on :data:`_PANE_BG`, keeping aspect."""
    from PIL import Image

    target_w, target_h = size
    source_w, source_h = image.size
    if source_w <= 0 or source_h <= 0:
        return Image.new("RGB", size, _PANE_BG)
    scale = min(target_w / source_w, target_h / source_h)
    resized = image.resize(
        (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, _PANE_BG)
    canvas.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return canvas
@dataclass
class ArtifactSet:
    """The files one experiment package actually wrote.

    Attributes:
        directory: Where the files are.
        video: Encoded video path, or ``None`` when encoding was not possible.
        images: Written image paths by name.
        manifest_path / results_path / readme_path: The JSON and Markdown files.
        warnings: Reasons something is missing or degraded.
    """

    directory: Path
    video: Path | None = None
    images: dict[str, Path] = field(default_factory=dict)
    manifest_path: Path | None = None
    results_path: Path | None = None
    readme_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Describe the written files, relative to the package directory."""
        return {
            "video": None if self.video is None else self.video.name,
            "images": {name: path.name for name, path in sorted(self.images.items())},
            "manifest": None if self.manifest_path is None else self.manifest_path.name,
            "results": None if self.results_path is None else self.results_path.name,
            "readme": None if self.readme_path is None else self.readme_path.name,
            "warnings": list(self.warnings),
        }
def _status_colour(status: str) -> tuple[int, int, int]:
    """Return the badge colour for a status, defaulting to grey."""
    return _STATUS_COLOURS.get(status, _MUTED)


def ffmpeg_binary() -> str | None:
    """Return a usable ``ffmpeg`` path, or ``None`` if there is none.

    A system install on ``PATH`` is tried first, then the binary bundled in the
    ``imageio-ffmpeg`` wheel.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - wheel present but unusable
        return None
    return candidate if candidate and Path(candidate).exists() else None


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, or ``""`` when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:  # pragma: no cover - unreadable artifact
        return ""
    return digest.hexdigest()


def encode_video(frames: Any, path: Path, fps: int = VIDEO_FPS) -> bool:
    """Encode ``frames`` to an H.264 MP4 by piping raw RGB into ffmpeg.

    Frames are consumed lazily, so a long episode streams straight to the
    encoder instead of being materialised first: a 300-tick 1080p recording is
    ~1.9 GB as a list.

    Args:
        frames: Any iterable of ``uint8`` ``(H, W, 3)`` frames or PIL images.
        path: Destination ``.mp4``.
        fps: Frames per second.

    Returns:
        ``True`` when a non-empty file was written. On failure no file is left
        behind: a truncated MP4 is worse than an absent one, because it opens in
        some players and shows a partial episode as though it were the whole.
    """
    binary = ffmpeg_binary()
    if binary is None:
        _LOG.warning("no ffmpeg available; video not encoded")
        return False

    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return False
    height, width = np.asarray(first, dtype=np.uint8).shape[:2]

    command = [
        binary, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-movflags", "+faststart", str(path),
    ]
    errors = ""
    try:
        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        ) as pipe:
            assert pipe.stdin is not None
            pipe.stdin.write(np.ascontiguousarray(first, dtype=np.uint8).tobytes())
            for frame in iterator:
                pipe.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            pipe.stdin.close()
            errors = pipe.stderr.read().decode("utf-8", "replace") if pipe.stderr else ""
            code = pipe.wait(timeout=300)
    except (OSError, subprocess.SubprocessError, BrokenPipeError) as exc:
        _LOG.warning("ffmpeg failed: %s", exc)
        path.unlink(missing_ok=True)
        return False
    if code != 0 or not path.exists() or path.stat().st_size == 0:
        _LOG.warning("ffmpeg exited %s: %s", code, errors[-400:])
        path.unlink(missing_ok=True)
        return False
    return True


def _no_camera_card(size: tuple[int, int], reason: str) -> Any:
    """Return a pane that says plainly that no camera frame exists.

    Used when the simulator produced no image. Drawing anything else -- a
    placeholder render, a repeated still -- would let a package imply footage
    that was never captured.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (26, 24, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, size[0] - 8, size[1] - 8], outline=(92, 96, 104), width=2)
    big, small = _font(40), _font(24)
    for text, dy, font, colour in (
        ("NO CAMERA FRAME", -20, big, (226, 140, 132)),
        (reason[:78], 30, small, _MUTED),
    ):
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(
            ((size[0] - (box[2] - box[0])) // 2, size[1] // 2 + dy),
            text, font=font, fill=colour,
        )
    return image


def _slate_frame(experiment: RecordedExperiment, size: tuple[int, int]) -> Any:
    """Render the title card: what is running, on what, and with what outcome."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (12, 13, 16))
    draw = ImageDraw.Draw(image)
    big, title, mid = _font(46), _font(40), _font(26)
    draw.text((64, 96), "ROBOT CI", font=big, fill=_TEXT)
    draw.text((64, 152), experiment.title[:52], font=title, fill=_TEXT)
    draw.line([(64, 216), (size[0] - 64, 216)], fill=(58, 62, 70), width=2)

    y = 252
    for label, value in (
        ("Task", experiment.task),
        ("Policy", experiment.policy),
        ("Seed", str(experiment.seed)),
        ("Backend", experiment.backend),
        ("Condition", experiment.condition),
        ("Scenario", f"{experiment.scenario_id} ({experiment.config_hash[:12]})"),
    ):
        draw.text((64, y), label, font=mid, fill=_MUTED)
        draw.text((320, y), str(value)[:60], font=mid, fill=_TEXT)
        y += 44

    status = experiment.status.value
    colour = _status_colour(status)
    draw.rounded_rectangle([64, y + 22, 296, y + 84], radius=10, fill=colour)
    badge = draw.textbbox((0, 0), status, font=big)
    draw.text(
        (64 + (232 - (badge[2] - badge[0])) // 2, y + 28), status, font=big, fill=(10, 12, 14)
    )
    draw.text((320, y + 40), experiment.status_detail[:66], font=mid, fill=_TEXT)
    draw.text(
        (64, size[1] - 72),
        "Recorded from a live simulation. Left: 3-D view from recorded joint states. "
        "Right: simulator camera.",
        font=_font(20), fill=_MUTED,
    )
    return image
def _compose_frame(
    experiment: RecordedExperiment,
    scene3d: Any,
    camera_pane: Any,
    event_text: str,
    size: tuple[int, int],
) -> Any:
    """Compose one split-screen frame with its overlays.

    Each pane is labelled permanently. The overlays carry the experiment's own
    recorded values -- status, step, sim clock, phase -- so a frame can be read
    without the manifest beside it.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, _PANE_BG)
    draw = ImageDraw.Draw(image)
    half = size[0] // 2
    label, overlay, small = _font(26), _font(28), _font(22)
    image.paste(_fit(scene3d, (half, size[1] - 56)), (0, 56))
    image.paste(_fit(camera_pane, (half, size[1] - 56)), (half, 56))

    draw.text((18, 16), "3D VIEW (recorded joint states)", font=label, fill=_TEXT)
    draw.text((half + 18, 16), "CAMERA (simulator output)", font=label, fill=_TEXT)
    draw.rectangle([half - 2, 0, half + 2, size[1]], fill=(48, 52, 60))

    status = experiment.status.value
    badge_text = f"{status}  step {experiment.steps}  t {experiment.duration_sim_s:6.2f}s"
    badge = draw.textbbox((0, 0), badge_text, font=overlay)
    draw.rounded_rectangle(
        [size[0] - (badge[2] - badge[0]) - 36, 12, size[0] - 16, 12 + (badge[3] - badge[1]) + 18],
        radius=8, fill=_status_colour(status),
    )
    draw.text(
        (size[0] - (badge[2] - badge[0]) - 28, 16), badge_text, font=overlay, fill=(10, 12, 14)
    )
    if event_text:
        draw.text((18, size[1] - 38), event_text[:110], font=small, fill=_MUTED)
    return image


def _timeline_context(events: list[dict[str, Any]], sim_time: float) -> str:
    """Return the most recent event at or before ``sim_time``, for the overlay."""
    latest = ""
    for event in events:
        if float(event.get("sim_time_s", -1.0)) <= sim_time + 1e-9:
            latest = str(event.get("label", ""))
    return latest


def _object_position(experiment: RecordedExperiment) -> tuple[float, float, float] | None:
    """Return the object's measured position, read back from the events.

    Ground truth is recorded on the first event by the recorder, so the 3-D view
    can draw the object where the simulation actually put it rather than at a
    nominal pose. ``None`` when no event carried one, which leaves the object
    undrawn instead of guessing.
    """
    for event in reversed(experiment.events):
        position = event.get("object_position")
        if isinstance(position, (list, tuple)) and len(position) == 3:
            return (float(position[0]), float(position[1]), float(position[2]))
    return None


def _compose_frames(experiment: RecordedExperiment) -> Any:
    """Yield every video frame: a slate, then one per recorded joint sample.

    This is a generator on purpose. A 300-tick episode at 1920x1080 is ~1.9 GB
    if held in a list, which would swap on the laptop this runs on; producing
    frames one at a time and feeding them straight to the encoder keeps the peak
    allocation to a single frame.

    The slate is prepended so the artifact is self-describing when played without
    its README. Frames are written at a constant rate regardless of how fast the
    simulation ran, because a viewer needs a real-time clock; the mapping from
    frame index to simulation time is stated by the ``t`` overlay.
    """
    from PIL import Image

    from robot_ci.recording.render3d import Camera, SceneInfo, load_kinematics, render_frame

    size = (VIDEO_WIDTH, VIDEO_HEIGHT)
    slate = _slate_frame(experiment, size)
    slate_frames = max(1, int(SLATE_SECONDS * VIDEO_FPS))
    for _ in range(slate_frames):
        yield slate
    if not experiment.joints:
        return

    urdf = urdf_path()
    kin = load_kinematics(urdf) if urdf is not None else None
    camera = Camera(width=960, height=1024)
    scene = SceneInfo(object_position=_object_position(experiment))
    camera_frames = _camera_frames_by_step(experiment)
    for sample in experiment.joints:
        pane3d = (
            render_frame(kin, np.asarray(sample.positions), scene, camera)
            if kin is not None
            else Image.new("RGB", (camera.width, camera.height), (30, 30, 34))
        )
        shot = camera_frames.get(sample.step_index)
        pane_cam = (
            Image.fromarray(np.asarray(shot.rgb, dtype=np.uint8))
            if shot is not None and shot.rgb is not None
            else _no_camera_card((camera.width, camera.height), "simulator sent no image here")
        )
        yield _compose_frame(
            experiment,
            pane3d,
            pane_cam,
            _timeline_context(experiment.events, sample.sim_time_s),
            size,
        )


def _camera_frames_by_step(experiment: RecordedExperiment) -> dict[int, Any]:
    """Index the recorded camera frames by control tick."""
    return {frame.step_index: frame for frame in experiment.frames if frame.has_image}


def urdf_path() -> Path | None:
    """Return the Panda URDF used for the 3-D view, if it is present."""
    candidate = Path(__file__).resolve().parents[3] / "gazebo" / "models" / "panda" / "panda.urdf"
    return candidate if candidate.exists() else None
def _manifest(experiment: RecordedExperiment, artifacts: ArtifactSet) -> dict[str, Any]:
    """Build ``manifest.json``: provenance, configuration and artifact hashes."""
    paths: list[Path] = ([artifacts.video] if artifacts.video else []) + sorted(
        artifacts.images.values()
    )
    return {
        "schema": "robot_ci.visual_experiment.manifest/1",
        "experiment_id": experiment.experiment_id,
        "category": experiment.category,
        "title": experiment.title,
        "condition": experiment.condition,
        "status": experiment.status.value,
        "status_detail": experiment.status_detail,
        "task": experiment.task,
        "policy": experiment.policy,
        "seed": experiment.seed,
        "backend": experiment.backend,
        "scenario_id": experiment.scenario_id,
        "config_hash": experiment.config_hash,
        "git_commit": experiment.git_commit,
        "command": experiment.command,
        "conditions": experiment.conditions,
        "versions": experiment.versions,
        "recording": {
            "steps": experiment.steps,
            "joint_samples": len(experiment.joints),
            "camera_frames_stored": len(experiment.frames),
            "camera_frames_with_image": experiment.frames_with_image,
            "camera_frames_observed": experiment.frames_observed,
            "camera_is_real": experiment.camera_is_real,
            "video_fps": VIDEO_FPS,
            "video_size": [VIDEO_WIDTH, VIDEO_HEIGHT],
            "duration_sim_s": round(experiment.duration_sim_s, 4),
            "duration_wall_s": round(experiment.duration_wall_s, 4),
            "started_wall": experiment.started_wall,
            "finished_wall": experiment.finished_wall,
            "joint_motion_rad": round(experiment.joint_motion_rad(), 6),
            "events": len(experiment.events),
        },
        "synchronisation": {
            "clocks": ["simulation clock", "host wall clock"],
            "clock_offset_s": round(experiment.clock_offset_s(), 4),
            "note": (
                "Frames and joint samples carry both clocks. The simulator's "
                "clock is authoritative and does not run at 1:1 with wall time, "
                "so the two are never interchanged in these artifacts."
            ),
        },
        "artifacts": artifacts.as_dict(),
        "artifact_sha256": {path.name: _sha256(path) for path in paths},
        "limitations": experiment.limitations,
        "notes": experiment.notes,
    }


def _results(experiment: RecordedExperiment) -> dict[str, Any]:
    """Build ``results.json``: the raw measured outcome, with no promotion."""
    return {
        "schema": "robot_ci.visual_experiment.results/1",
        "experiment_id": experiment.experiment_id,
        "category": experiment.category,
        "status": experiment.status.value,
        "status_detail": experiment.status_detail,
        "succeeded": experiment.status.is_success(),
        "attempted": experiment.status.is_attempt(),
        "task": experiment.task,
        "policy": experiment.policy,
        "seed": experiment.seed,
        "backend": experiment.backend,
        "steps": experiment.steps,
        "duration_sim_s": experiment.duration_sim_s,
        "duration_wall_s": experiment.duration_wall_s,
        "metrics": dict(experiment.metrics),
        "joint_motion_rad": experiment.joint_motion_rad(),
        "events": experiment.events,
        "error": experiment.error,
        "measurements": experiment.extra,
    }


def _readme(experiment: RecordedExperiment, artifacts: ArtifactSet) -> str:
    """Build the package README: what ran, how it was captured, what it is not."""
    frames = experiment.frames_with_image
    lines = [
        f"# {experiment.title}",
        "",
        f"**Status: `{experiment.status.value}`** -- {experiment.status_detail}",
        "",
        "## What ran",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Task | `{experiment.task}` |",
        f"| Policy | `{experiment.policy}` |",
        f"| Seed | `{experiment.seed}` |",
        f"| Backend | `{experiment.backend}` (a real simulator process) |",
        f"| Condition | `{experiment.condition}` |",
        f"| Scenario | `{experiment.scenario_id}` |",
        f"| Config hash | `{experiment.config_hash}` |",
        f"| Commit | `{experiment.git_commit or 'unknown'}` |",
        f"| Control ticks | {experiment.steps} |",
        f"| Simulated time | {experiment.duration_sim_s:.2f} s |",
        f"| Wall time | {experiment.duration_wall_s:.2f} s |",
        f"| Joint samples | {len(experiment.joints)} |",
        f"| Camera frames with image | {frames} of {experiment.frames_observed} observed |",
        f"| Largest joint travel | {experiment.joint_motion_rad():.4f} rad |",
        "",
        "## Measured metrics",
        "",
    ]
    if experiment.metrics:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| `{name}` | {value:.6g} |" for name, value in sorted(experiment.metrics.items())]
    else:
        lines.append("_No metrics were measured: the episode did not complete._")
    if experiment.error:
        lines += ["", f"**Error:** `{experiment.error}`"]

    lines += [
        "",
        "## How this was captured",
        "",
        "Both panes of `experiment.mp4` come from the one episode recorded above:",
        "",
        "* **3D VIEW (recorded joint states)** -- the Panda drawn through the same",
        "  URDF the browser viewer uses, from the joint angles Gazebo reported. If",
        "  the joints had not moved, the arm in this pane would not move either.",
        "* **CAMERA (simulator output)** -- the simulator's own RGB render, "
        "captured"
        "  before any sensor-corruption stage ran, so it is the raw sensor output.",
        "",
        "`camera_view.png` is a real frame from that stream"
        + (f" ({frames} were captured)." if frames else " -- **none were captured.**"),
        "",
    ]
    if not artifacts.video:
        lines += [
            "> **No video in this package.** "
            + (artifacts.warnings[0] if artifacts.warnings else "Encoding was not attempted."),
            "",
        ]
    if experiment.events:
        lines += ["## Episode timeline", "", "| sim time (s) | event |", "|---|---|"]
        lines += [
            f"| {float(event.get('sim_time_s', 0.0)):.3f} | {event.get('label', '')} |"
            for event in experiment.events
        ]
        lines.append("")

    lines += ["## Reproduce", "", "```bash", experiment.command, "```", ""]
    lines += [
        f"The seed, scenario id and config hash above pin the condition. "
        f"`manifest.json` records the runtime versions"
        f" ({', '.join(f'{k}={v}' for k, v in sorted(experiment.versions.items()))})"
        f" needed to repeat it.",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"* {item}" for item in experiment.limitations] or [
        "* No limitations were recorded for this package."
    ]
    lines += [f"* {item}" for item in experiment.notes]
    if artifacts.warnings:
        lines += ["", "## Packaging warnings", ""]
        lines += [f"* {warning}" for warning in artifacts.warnings]
    lines.append("")
    return "\n".join(lines)
def write_artifacts(
    experiment: RecordedExperiment, directory: Path, *, make_video: bool = True
) -> ArtifactSet:
    """Write the whole artifact set for one recorded experiment.

    Args:
        experiment: The recording to package.
        directory: Destination, created if absent.
        make_video: When ``False``, only the stills and JSON are written. Used by
            tests and dry runs that want artifacts without an encode.

    Returns:
        The :class:`ArtifactSet` describing what was written.
    """
    from robot_ci.recording.figures import (
        write_camera_view,
        write_metrics_plot,
        write_overview,
        write_robot_view,
        write_timeline,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactSet(directory=directory)

    for name, writer in (
        ("overview.png", write_overview),
        ("robot_view.png", write_robot_view),
        ("camera_view.png", write_camera_view),
        ("metrics.png", write_metrics_plot),
        ("timeline.png", write_timeline),
    ):
        try:
            artifacts.images[name] = Path(writer(experiment, directory))
        except Exception as exc:  # a missing figure is reported, not fatal
            _LOG.warning("%s: figure %s failed: %s", experiment.experiment_id, name, exc)
            artifacts.warnings.append(f"{name}: {type(exc).__name__}: {exc}")

    if make_video:
        try:
            if not experiment.joints:
                artifacts.warnings.append("no joint samples recorded; no video written")
            elif encode_video(_compose_frames(experiment), directory / "experiment.mp4"):
                artifacts.video = directory / "experiment.mp4"
            else:
                artifacts.warnings.append("video encoding failed; no video written")
        except Exception as exc:
            _LOG.warning("%s: video composition failed: %s", experiment.experiment_id, exc)
            artifacts.warnings.append(f"video: {type(exc).__name__}: {exc}")

    artifacts.manifest_path = directory / "manifest.json"
    artifacts.manifest_path.write_text(
        json.dumps(_manifest(experiment, artifacts), indent=2, default=str), encoding="utf-8"
    )
    artifacts.results_path = directory / "results.json"
    artifacts.results_path.write_text(
        json.dumps(_results(experiment), indent=2, default=str), encoding="utf-8"
    )
    artifacts.readme_path = directory / "README.md"
    artifacts.readme_path.write_text(_readme(experiment, artifacts), encoding="utf-8")
    return artifacts
