"""The five PNGs in every experiment package, drawn from recorded samples.

Each figure answers one question, and each is generated only from what the
recording measured:

* ``overview.png``   -- what ran, and how it ended.
* ``robot_view.png`` -- the arm, drawn from recorded joint angles.
* ``camera_view.png``-- one real frame from the simulator's camera.
* ``metrics.png``    -- plots of the measured metrics and joint trajectory.
* ``timeline.png``   -- the episode's timestamped events.

The rule these all obey
-----------------------
A figure is never drawn from data the run did not produce. When the camera
emitted no frame, ``camera_view.png`` says so in words instead of showing a
plausible picture; when no events were timestamped, ``timeline.png`` says the
episode produced none. A reader who finds an artifact has been told which of
those two situations they are in before they read a number off it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_ci.core.logging import get_logger
from robot_ci.recording.recorder import RecordedExperiment

__all__ = [
    "write_camera_view",
    "write_metrics_plot",
    "write_overview",
    "write_robot_view",
    "write_timeline",
]

_LOG = get_logger(__name__)

_WIDTH = 1280
_TEXT = (236, 240, 246)
_MUTED = (150, 158, 170)
_ACCENT = (108, 168, 240)
_STATUS_COLOURS = {
    "PASS": (74, 190, 120),
    "FAIL": (222, 96, 84),
    "ERROR": (232, 168, 68),
}


def _font(size: int) -> Any:
    """Return a PIL font, falling back to the bitmap default when none loads."""
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:  # pragma: no cover
                continue
    return ImageFont.load_default()


def _status_colour(status: str) -> tuple[int, int, int]:
    """Return the badge colour for a status."""
    return _STATUS_COLOURS.get(status, (130, 138, 150))


def _urdf_path() -> Path | None:
    """Return the Panda URDF used for the 3-D view, if present."""
    candidate = Path(__file__).resolve().parents[3] / "gazebo" / "models" / "panda" / "panda.urdf"
    return candidate if candidate.exists() else None


def _object_position(experiment: RecordedExperiment) -> tuple[float, float, float] | None:
    """Return the object's measured position, from an event if one recorded it.

    The 3-D view draws the object where the run said it was, rather than at a
    nominal position, so the picture and the metrics agree.
    """
    for event in reversed(experiment.events):
        position = event.get("object_position")
        if isinstance(position, (list, tuple)) and len(position) == 3:
            return (float(position[0]), float(position[1]), float(position[2]))
    return None


def write_overview(experiment: RecordedExperiment, directory: Path) -> Path:
    """Write ``overview.png``: the configuration and the measured outcome."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (_WIDTH, 720), (14, 15, 19))
    draw = ImageDraw.Draw(image)
    draw.text((48, 44), "ROBOT CI experiment overview", font=_font(40), fill=_TEXT)
    draw.text(
        (48, 96), experiment.title[:70], font=_font(28), fill=_ACCENT
    )

    y: int = 156
    for label, value in (
        ("Status", experiment.status.value),
        ("Task", experiment.task),
        ("Policy", experiment.policy),
        ("Seed", str(experiment.seed)),
        ("Backend", experiment.backend),
        ("Condition", experiment.condition),
        ("Scenario", experiment.scenario_id),
        ("Config hash", experiment.config_hash[:24]),
        ("Commit", experiment.git_commit or "unknown"),
        ("Control ticks", str(experiment.steps)),
        ("Simulated time", f"{experiment.duration_sim_s:.2f} s"),
        ("Wall time", f"{experiment.duration_wall_s:.2f} s"),
        ("Camera frames", f"{experiment.frames_with_image} with image"),
        ("Joint travel (max)", f"{experiment.joint_motion_rad():.4f} rad"),
    ):
        draw.text((48, y), label, font=_font(22), fill=_MUTED)
        colour = _status_colour(experiment.status.value) if label == "Status" else _TEXT
        draw.text((330, y), str(value)[:74], font=_font(22), fill=colour)
        y += 34

    draw.text((48, y + 18), "Detail", font=_font(22), fill=_MUTED)
    draw.text((330, y + 18), experiment.status_detail[:70], font=_font(22), fill=_TEXT)

    metrics_y: int = int(y) + 84
    draw.text((48, metrics_y), "Measured metrics", font=_font(24), fill=_TEXT)
    metrics_y += 40
    if experiment.metrics:
        for metric_name, metric_value in sorted(experiment.metrics.items())[:8]:
            draw.text((48, metrics_y), metric_name, font=_font(20), fill=_MUTED)
            draw.text((480, metrics_y), f"{metric_value:.6g}", font=_font(20), fill=_TEXT)
            metrics_y += 28
    else:
        draw.text((48, metrics_y), "none measured", font=_font(20), fill=_MUTED)

    path = Path(directory) / "overview.png"
    image.save(path)
    return path
def write_robot_view(experiment: RecordedExperiment, directory: Path) -> Path:
    """Write ``robot_view.png``: the arm at its last recorded configuration.

    Drawn from the final joint sample, which is what the simulator reported at
    the end of the episode. When the run produced no joint samples the figure
    says so rather than drawing the robot's home pose.
    """
    from PIL import Image, ImageDraw

    from robot_ci.recording.render3d import Camera, SceneInfo, load_kinematics, render_frame

    image = Image.new("RGB", (_WIDTH, 720), (14, 15, 19))
    draw = ImageDraw.Draw(image)
    draw.text(
        (40, 34), "Robot view - 3D projection of recorded joint states",
        font=_font(30), fill=_TEXT,
    )

    urdf = _urdf_path()
    if urdf is None or not experiment.joints:
        reason = (
            "panda.urdf not found" if urdf is None else "the run recorded no joint samples"
        )
        draw.text((40, 360), f"No robot view: {reason}", font=_font(26), fill=(226, 140, 132))
        path = Path(directory) / "robot_view.png"
        image.save(path)
        return path

    kin = load_kinematics(urdf)
    scene = SceneInfo(object_position=_object_position(experiment))
    view = render_frame(
        kin,
        np.asarray(experiment.joints[-1].positions, dtype=np.float64),
        scene,
        Camera(width=1200, height=620),
    )
    image.paste(view, (40, 84))
    draw.text(
        (40, 672),
        f"final joint sample, step {experiment.joints[-1].step_index}, "
        f"sim t {experiment.joints[-1].sim_time_s:.3f}s",
        font=_font(20), fill=_MUTED,
    )
    path = Path(directory) / "robot_view.png"
    image.save(path)
    return path


def write_camera_view(experiment: RecordedExperiment, directory: Path) -> Path:
    """Write ``camera_view.png``: one real frame from the simulator's camera.

    The frame with the most *content* is chosen -- highest count of finite depth
    pixels -- rather than the first one, because a camera often publishes its
    first frames while the scene is still rendering. If no frame carries an
    image, the figure states that instead of drawing anything.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (_WIDTH, 720), (14, 15, 19))
    draw = ImageDraw.Draw(image)
    draw.text((40, 34), "Camera view - simulator RGB output", font=_font(30), fill=_TEXT)

    best: tuple[int, Any] | None = None
    for frame in experiment.frames:
        if not frame.has_image:
            continue
        score = int(np.isfinite(frame.depth).sum()) if frame.depth is not None else 0
        if best is None or score > best[0]:
            best = (score, frame)

    if best is None:
        draw.text(
            (40, 340), "No camera frame was produced by this run.",
            font=_font(28), fill=(226, 140, 132),
        )
        draw.text(
            (40, 392),
            f"{experiment.frames_observed} frames observed, 0 carried an image.",
            font=_font(22), fill=_MUTED,
        )
    else:
        _score, frame = best
        source = Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8))
        # NEAREST, so the reader sees the sensor's own pixels rather than an
        # invented interpolation between them.
        scale = min(1160 / source.width, 560 / source.height)
        shown = source.resize(
            (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
            Image.Resampling.NEAREST,
        )
        image.paste(shown, (40 + (1160 - shown.width) // 2, 84 + (560 - shown.height) // 2))
        draw.text(
            (40, 664),
            f"step {frame.step_index}, sim t {frame.sim_time_s:.3f}s, "
            f"{frame.width_px}x{frame.height_px} from '{frame.source}' "
            f"(shown {shown.width}x{shown.height}, nearest-neighbour)",
            font=_font(20), fill=_MUTED,
        )
    path = Path(directory) / "camera_view.png"
    image.save(path)
    return path


def _plot_series(
    draw: Any,
    box: tuple[int, int, int, int],
    xs: list[float],
    ys: list[float],
    colour: tuple[int, int, int],
    *,
    label: str = "",
) -> bool:
    """Draw one series inside ``box``, auto-scaled to its own range."""
    if len(xs) < 2 or len(ys) != len(xs):
        return False
    left, top, right, bottom = box[0] + 52, box[1] + 14, box[2] - 14, box[3] - 30
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    if y_max - y_min < 1e-12:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    points = [
        (
            left + (x - x_min) / (x_max - x_min) * (right - left),
            bottom - (y - y_min) / (y_max - y_min) * (bottom - top),
        )
        for x, y in zip(xs, ys, strict=True)
    ]
    draw.line([(int(px), int(py)) for px, py in points], fill=colour, width=2)
    draw.text(
        (left, box[3] - 24), f"{x_min:.1f} .. {x_max:.1f} s", font=_font(16), fill=_MUTED
    )
    draw.text((box[0] + 10, top), f"{y_max:.3g}", font=_font(16), fill=_MUTED)
    draw.text((box[0] + 10, bottom - 16), f"{y_min:.3g}", font=_font(16), fill=_MUTED)
    if label:
        draw.text((right - 150, top + 2), label, font=_font(16), fill=colour)
    return True


def write_metrics_plot(experiment: RecordedExperiment, directory: Path) -> Path:
    """Write ``metrics.png``: measured metrics and the recorded trajectory.

    The left panel is load-bearing: it is the plot that shows whether the robot
    physically moved, drawn straight from the joint samples Gazebo published.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (_WIDTH, 760), (14, 15, 19))
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), "Metrics - measured from this run", font=_font(30), fill=_TEXT)

    left = (40, 80, 620, 400)
    draw.text((46, 62), "End-effector position over time", font=_font(20), fill=_ACCENT)
    draw.rectangle(left, outline=(58, 62, 70), width=2)
    if experiment.joints:
        times = [s.sim_time_s for s in experiment.joints]
        for index, (name, colour) in enumerate(
            (("ee_z", (108, 168, 240)), ("ee_x", (240, 168, 108)), ("ee_y", (150, 214, 160)))
        ):
            _plot_series(
                draw, left, times,
                [s.end_effector[index] for s in experiment.joints], colour, label=name,
            )
    else:
        draw.text((60, 240), "no joint samples recorded", font=_font(20), fill=_MUTED)

    right = (660, 80, 1240, 400)
    draw.text((666, 62), "Gripper aperture (m)", font=_font(20), fill=_ACCENT)
    draw.rectangle(right, outline=(58, 62, 70), width=2)
    if experiment.joints:
        _plot_series(
            draw, right,
            [s.sim_time_s for s in experiment.joints],
            [s.gripper_width_m for s in experiment.joints],
            (214, 160, 240), label="width",
        )

    panel = (40, 430, 1240, 740)
    draw.rectangle(panel, outline=(58, 62, 70), width=2)
    draw.text((46, 412), "Episode metrics (recorded values)", font=_font(20), fill=_ACCENT)
    if experiment.metrics:
        rows = sorted(experiment.metrics.items())
        span = max(abs(v) for _, v in rows) or 1.0
        top = 472
        for name, value in rows:
            draw.text((60, top), name[:44], font=_font(18), fill=_MUTED)
            draw.text((330, top), f"{value:.4g}", font=_font(18), fill=_TEXT)
            draw.rectangle(
                [430, top + 4, 430 + max(1, int(abs(value) / span * 780)), top + 20],
                fill=(96, 132, 188),
            )
            top += 26
    else:
        draw.text((60, 560), "no metrics measured", font=_font(20), fill=_MUTED)

    path = Path(directory) / "metrics.png"
    image.save(path)
    return path


def write_timeline(experiment: RecordedExperiment, directory: Path) -> Path:
    """Write ``timeline.png``: the episode's timestamped events on a time axis.

    Plotted on the recorded simulation clock, so the axis is the one the
    simulator reported rather than the host's.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (_WIDTH, 560), (14, 15, 19))
    draw = ImageDraw.Draw(image)
    draw.text(
        (40, 34), "Episode timeline - recorded events (simulation clock)",
        font=_font(30), fill=_TEXT,
    )

    track_left, track_right, axis_y = 90, 1200, 300
    draw.line([(track_left, axis_y), (track_right, axis_y)], fill=(70, 76, 86), width=3)

    events = experiment.events
    if not events:
        draw.text(
            (40, 260), "This run produced no timestamped events.",
            font=_font(24), fill=_MUTED,
        )
    else:
        end = max(float(event.get("sim_time_s", 0.0)) for event in events)
        span = max(end, 1e-6)
        for index, event in enumerate(events):
            when = float(event.get("sim_time_s", 0.0))
            x = track_left + when / span * (track_right - track_left)
            kind = str(event.get("kind", ""))
            colour = _status_colour(kind) if kind in _STATUS_COLOURS else _ACCENT
            draw.line([(int(x), axis_y - 26), (int(x), axis_y + 26)], fill=colour, width=3)
            label = str(event.get("label", ""))[:40]
            y = axis_y - 74 if index % 2 == 0 else axis_y + 44
            draw.text((int(x) + 6, y), f"{when:.2f}s  {label}", font=_font(17), fill=colour)
        draw.text((track_left, axis_y + 14), "0 s", font=_font(17), fill=_MUTED)
        draw.text((track_right - 70, axis_y + 14), f"{span:.1f}s", font=_font(17), fill=_MUTED)

    draw.text(
        (40, 500),
        f"{len(events)} event(s); {experiment.steps} control ticks; "
        f"{experiment.duration_sim_s:.2f}s simulated, {experiment.duration_wall_s:.2f}s wall.",
        font=_font(19), fill=_MUTED,
    )
    path = Path(directory) / "timeline.png"
    image.save(path)
    return path
