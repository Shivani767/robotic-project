r"""Build the LinkedIn-ready demo package from verified Gazebo Harmonic data.

Source of truth
---------------
* Motion/measured outcomes: the Part-1 evidence run
  ``experiments/results/gazebo/pick_place_seed3_0mm_policy_v1.json``
  (115 ticks, task_success=1.0, placement_error_m=0.022, object_lift_m=0.066).
  Every 3-D frame is projected from that run's recorded joint states through
  the same Panda URDF the browser viewer uses. Nothing is re-simulated or
  hand-posed.
* Camera still: one genuine Gazebo camera frame, copied byte-for-byte from the
  separately recorded
  ``artifacts/visual_reports/nominal_baseline/camera_view.png``. The evidence
  run above captured joint-state telemetry but no camera stream, so its video
  cannot show synchronized camera footage. The package says that plainly: the
  right pane and ``images/camera_view.png`` are labelled as a reference still
  from a separate Gazebo recording, NOT as this episode's footage.

Run inside the Gazebo container (it ships PIL and the imageio-ffmpeg wheel)::

    docker compose run --rm gazebo-demo bash -lc \\
      'source /opt/ros/jazzy/setup.bash && python3 scripts/make_linkedin_demo.py'

Output: ``artifacts/linkedin/robot_ci_demo.zip``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "experiments/results/gazebo/pick_place_seed3_0mm_policy_v1.json"
CAM_STILL = REPO / "artifacts/visual_reports/nominal_baseline/camera_view.png"
STAGE = REPO / "artifacts/linkedin/stage"
ZIP_PATH = REPO / "artifacts/linkedin/robot_ci_demo.zip"

WIDTH, HEIGHT, FPS = 1920, 1080, 30
REPEAT = 10  # hold every measured tick 10 frames: 30 video-fps / 3 Hz telemetry
SLATE_S, OUTRO_S = 2, 4

TITLE = "ROBOT CI -- Panda pick-and-place on Gazebo Harmonic"
CONDITION = "nominal / seed 3 / policy_v1"
OUTCOME_LINE = "measured: task_success=1.0, placement_error 0.022 m, lift 0.066 m"


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    from PIL import ImageFont as _IF

    return _IF.load_default()


def _ffmpeg() -> str:
    found = __import__("shutil").which("ffmpeg")
    if found:
        return found
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def _fit(image, size: tuple[int, int]):
    from PIL import Image

    target_w, target_h = size
    scale = min(target_w / image.size[0], target_h / image.size[1])
    small = image.resize(
        (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))),
        Image.LANCZOS,
    )
    canvas = Image.new("RGB", size, (16, 18, 22))
    canvas.paste(small, ((target_w - small.size[0]) // 2, (target_h - small.size[1]) // 2))
    return canvas


def main() -> int:
    """Build every artifact, verify the MP4 and ZIP, and return 0 on success."""
    from PIL import Image, ImageDraw

    from robot_ci.recording.render3d import Camera, SceneInfo, load_kinematics, render_frame

    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    rows = data["rows"]
    assert data["success"] is True and metrics["task_success"] == 1.0, (
        "evidence run must be the measured success"
    )
    assert len(rows) == 115, f"expected 115 measured ticks, got {len(rows)}"

    kin = load_kinematics(REPO / "gazebo/models/panda/panda.urdf")
    camera = Camera(
        eye=(0.65, -0.60, 0.55),
        target=(0.04, 0.14, 0.14),
        fov_deg=42.0,
        width=936,
        height=954,
    )
    # Scene mirrors the generated world for this scenario: table centred on the
    # workspace with 0.35 m half-extent, the placement disc at the task's own
    # target pose (0.2, 0.15), and the 16 mm cube (size_m is the half extent).
    scene = SceneInfo(
        table_center=(0.0, 0.1),
        table_half=0.35,
        table_top_z=0.0,
        zone_center=(0.2, 0.15),
        zone_radius=0.06,
        object_half=0.008,
    )
    still = Image.open(CAM_STILL).convert("RGB")
    still_pane = _fit(still, (936, 954))  # once, not per frame

    for sub in ("video", "images", "results", "logs"):
        (STAGE / sub).mkdir(parents=True, exist_ok=True)

    label, small, tiny = _font(30), _font(24), _font(20)
    big, mid = _font(72), _font(34)
    total_frames = SLATE_S * FPS + len(rows) * REPEAT + OUTRO_S * FPS

    def pane_3d(row) -> Image.Image:
        return render_frame(
            kin,
            np.asarray(row["measured_joints"][:7], dtype=np.float64),
            replace(scene, object_position=tuple(float(v) for v in row["object"])),
            camera,
        )

    def compose(view: Image.Image, row) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), (10, 11, 14))
        d = ImageDraw.Draw(img)
        img.paste(view, (12, 84))
        img.paste(still_pane, (972, 84))
        d.text((24, 24), "3D VIEW (measured joint states)", font=label, fill=(236, 240, 246))
        d.text((984, 24), "CAMERA - reference still, separate run", font=label, fill=(232, 168, 68))
        phase = str(row["phase"]).replace("Phase.", "")
        grasp = "GRASPED" if row.get("grasped") else "open"
        bar = (
            f"pick_and_place | policy_v1 | seed 3 | nominal | "
            f"step {row['step']}/114 | {phase} | t={row['sim_t']:.2f}s | "
            f"aperture {row['aperture_m'] * 1000:.1f}mm | {grasp}"
        )
        d.text((24, HEIGHT - 40), bar, font=small, fill=(150, 158, 170))
        d.text(
            (24, 60),
            "camera pane is a labelled reference still - this episode recorded joint telemetry only",
            font=tiny,
            fill=(232, 168, 68),
        )
        return img

    def card(title: str, lines: list[tuple[str, tuple[int, int, int]]]) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), (10, 11, 14))
        d = ImageDraw.Draw(img)
        d.text((120, 260), title, font=big, fill=(236, 240, 246))
        for i, (line, colour) in enumerate(lines):
            d.text((120, 400 + i * 62), line, font=mid, fill=colour)
        return img

    MUTED = (150, 158, 170)
    GREEN = (74, 190, 120)
    title = card(
        "ROBOT CI",
        [
            ("Panda pick-and-place on Gazebo Harmonic", MUTED),
            (CONDITION, MUTED),
            (OUTCOME_LINE, GREEN),
            ("3D pane: measured joints | camera pane: labelled reference still", MUTED),
        ],
    )
    outro = card(
        "MEASURED OUTCOME",
        [
            ("task_success=1.0  grasp=1.0  placement=1.0", GREEN),
            (f"placement_error {metrics['placement_error_m']:.3f} m  (tol 0.05)", (236, 240, 246)),
            (f"object_lift {metrics['object_lift_m']:.3f} m  (min 0.05)", (236, 240, 246)),
            ("115 ticks, 9.54 s simulated, seed 3, nominal, policy_v1", MUTED),
        ],
    )

    # Still frames captured while streaming: grasp moment + robot view + overview.
    grasp_row = next(r for r in rows if r["step"] == 69)
    grasp_pane = pane_3d(grasp_row)
    title.save(STAGE / "images" / "overview.png")
    grasp_pane.save(STAGE / "images" / "robot_view.png")

    mp4 = STAGE / "video" / "robot_ci_demo.mp4"
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-movflags", "+faststart", str(mp4)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    def write(img: Image.Image) -> None:
        proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    thumbnail_done = False
    for _ in range(SLATE_S * FPS):
        write(title)
    for row in rows:
        frame = compose(pane_3d(row), row)
        if row["step"] == 69 and not thumbnail_done:
            frame.save(STAGE / "images" / "thumbnail.png")
            thumbnail_done = True
        for _ in range(REPEAT):
            write(frame)
    for _ in range(OUTRO_S * FPS):
        write(outro)
    _, errors = proc.communicate(timeout=600)
    assert proc.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0, (
        f"ffmpeg failed: {errors[-400:]}"
    )
    assert thumbnail_done, "grasp-frame thumbnail was never captured"

    # The camera still must be byte-identical to the source recording.
    shutil.copyfile(CAM_STILL, STAGE / "images" / "camera_view.png")

    # Telemetry: one row per measured tick, straight from the evidence JSON.
    with (STAGE / "results" / "telemetry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["step", "sim_t_s", "phase", "grasped", "aperture_m",
             "tcp_x", "tcp_y", "tcp_z", "obj_x", "obj_y", "obj_z",
             "j1", "j2", "j3", "j4", "j5", "j6", "j7"]
        )
        for row in rows:
            writer.writerow(
                [row["step"], row["sim_t"], str(row["phase"]).replace("Phase.", ""),
                 int(bool(row.get("grasped"))), row["aperture_m"],
                 *row["tcp"], *row["object"], *row["measured_joints"][:7]]
            )

    results = {
        "schema": "robot_ci.linkedin_demo.results/1",
        "source_evidence": str(EVIDENCE.relative_to(REPO)),
        "measured": {
            "success": data["success"],
            "status": data["status"],
            "steps": data["steps"],
            "task_success": metrics["task_success"],
            "grasp_success": metrics["grasp_success"],
            "placement_success": metrics["placement_success"],
            "placement_error_m": metrics["placement_error_m"],
            "object_lift_m": metrics["object_lift_m"],
            "completion_time_s": metrics["completion_time_s"],
            "sim_span_s": round(rows[-1]["sim_t"] - rows[0]["sim_t"], 4),
            "grasped_ticks": sum(1 for r in rows if r.get("grasped")),
        },
        "video": {
            "path": "video/robot_ci_demo.mp4",
            "width": WIDTH, "height": HEIGHT, "fps": FPS,
            "frames": total_frames,
            "seconds": round(total_frames / FPS, 3),
        },
        "camera_provenance": {
            "still_source": str(CAM_STILL.relative_to(REPO)),
            "synchronized_with_episode": False,
            "note": "The evidence run captured joint-state telemetry but no camera "
                    "stream. The camera pane shows a genuine Gazebo frame from a "
                    "separate recording, labelled on-frame as a reference still.",
        },
    }
    (STAGE / "results" / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for name in ("episode_seed3_0mm_latest.log", "stage_check_latest.log"):
        src = REPO / "experiments/results/gazebo" / name
        if src.exists():
            (STAGE / "logs" / name).write_bytes(src.read_bytes())

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip() or "unknown"
    manifest_payload = {
        "schema": "robot_ci.linkedin_demo.manifest/1",
        "title": TITLE,
        "condition": CONDITION,
        "git_commit": git_commit,
        "source_evidence_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
        "camera_still_sha256": hashlib.sha256(CAM_STILL.read_bytes()).hexdigest(),
        "video": {"frames": total_frames, "fps": FPS, "width": WIDTH, "height": HEIGHT},
    }
    files = sorted(p for p in STAGE.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest_payload["artifact_sha256"] = {
        str(p.relative_to(STAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    (STAGE / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    (STAGE / "README.md").write_text(
        f"""# {TITLE}

## What this is
A presentation package built **only** from measured Gazebo Harmonic data.
No trajectory was re-simulated, hand-posed, or fabricated.

## Measured outcome (evidence run: nominal / seed 3 / policy_v1)
- success: {data['success']} ({data['status']}), {data['steps']} control ticks
- task_success={metrics['task_success']}, grasp_success={metrics['grasp_success']},
  placement_success={metrics['placement_success']}
- placement_error_m={metrics['placement_error_m']:.4f} (tolerance 0.05 m),
  object_lift_m={metrics['object_lift_m']:.4f} (required 0.05 m)
- {sum(1 for r in rows if r.get('grasped'))} of 115 ticks with the object grasped;
  phases APPROACH 47 / DESCEND 22 / TRANSIT 22 / PLACE 23 / DONE 1

## What each file is
- `video/robot_ci_demo.mp4` ({total_frames / FPS:.0f} s, {WIDTH}x{HEIGHT} @ {FPS} fps): split-screen.
  **Left pane**: the Panda re-projected in 3-D from this run's 115 recorded joint
  states (10 video frames per measured tick, so one tick = {REPEAT / FPS:.1f} s of video).
  **Right pane**: a genuine Gazebo camera frame used as a *reference still* -
  labelled in amber on every frame.
- `images/thumbnail.png`: the grasp moment (tick 69, first GRASPED tick).
- `images/overview.png`: opening title card with the measured outcome.
- `images/robot_view.png`: 3-D view at the grasp moment.
- `images/camera_view.png`: byte-identical copy of the source camera frame.
- `results/results.json`, `results/telemetry.csv`: measured numbers, one row per tick.
- `manifest.json`: SHA-256 of every packaged file (self excluded).
- `logs/`: simulator logs shipped with the evidence run.

## Reproduce
```bash
# 1. the measured episode (writes experiments/results/gazebo/pick_place_seed3_0mm_policy_v1.json)
docker compose run --rm gazebo-demo bash -lc \\
  'source /opt/ros/jazzy/setup.bash && python3 scripts/gazebo_pick_place_evidence.py --seed 3 --noise-mm 0'
# 2. this package
docker compose run --rm gazebo-demo bash -lc \\
  'source /opt/ros/jazzy/setup.bash && python3 scripts/make_linkedin_demo.py'
```

Commit: {git_commit}
Evidence SHA-256: {manifest_payload['source_evidence_sha256']}

## Capabilities vs measurements
Measured in this run: a complete grasp-lift-carry-place with the numbers above.
Measured elsewhere (separate evidence, see `docs/recording.md`): the Part-2
suite records 8 Gazebo conditions with MP4/PNG/JSON/ZIP exports, dashboard and
CLI. The suite's nominal conditions are recorded FAILs at the 300-tick horizon
(`artifacts/visual_reports/suite_summary.json`) - that is reported as-is, not
promoted to success anywhere in this package.

## Limitations
- The evidence run recorded joint-state telemetry but no camera stream, so no
  synchronized camera motion exists for this episode. The amber right pane and
  `images/camera_view.png` come from a different Gazebo episode on the same
  simulator build; object and arm pose do not correspond tick-for-tick.
- The 3-D pane is a kinematic projection of measured joint angles, not a
  re-simulation: physics is never re-run to produce these frames.
- Simulator: Gazebo Harmonic gz-sim 8.15.0, ROS 2 Jazzy, Python 3.12,
  linux-aarch64 (Docker Desktop on macOS).
""",
        encoding="utf-8",
    )

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in STAGE.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(STAGE))
    with zipfile.ZipFile(ZIP_PATH) as archive:
        broken = archive.testzip()
        assert broken is None, f"ZIP failed integrity check at {broken}"
        names = set(archive.namelist())
    required = {
        "video/robot_ci_demo.mp4", "images/thumbnail.png", "images/overview.png",
        "images/robot_view.png", "images/camera_view.png", "results/results.json",
        "manifest.json", "README.md", "results/telemetry.csv",
    }
    missing = required - names
    assert not missing, f"ZIP missing {sorted(missing)}"
    print(f"wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
