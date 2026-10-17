"""Build the LinkedIn-ready robotics demo from verified Gazebo Harmonic data.

Source of truth
---------------
* Motion and measured outcomes come from the Part-1 evidence run
  ``experiments/results/gazebo/pick_place_seed3_0mm_policy_v1.json``
  (115 ticks, task_success=1.0, placement_error_m=0.022, object_lift_m=0.066).
* Every 3-D frame is rendered by MuJoCo from the repository's *real* Panda
  model (``mujoco/models/franka_emika_panda/panda.xml`` + all 67 mesh assets),
  driven by that run's recorded joint states, finger aperture, and -- because
  this run *does* record object pose -- the measured cube position and
  orientation. Nothing is re-simulated or hand-posed.

Run it (Pillow, numpy and mujoco are available in the project venv)::

    python scripts/make_linkedin_demo.py

Output: ``artifacts/linkedin/robot_ci_demo.zip`` plus the standalone
``video/robot_ci_demo.mp4``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "experiments/results/gazebo/pick_place_seed3_0mm_policy_v1.json"
OUT_DIR = REPO / "artifacts/linkedin"
STAGE = OUT_DIR / "stage"
ZIP_PATH = OUT_DIR / "robot_ci_demo.zip"

WIDTH, HEIGHT, FPS = 1920, 1080, 30

# --------------------------------------------------------------------------- #
# Typography helpers
# --------------------------------------------------------------------------- #
_FONT_CACHE: dict[tuple[int, bool], object] = {}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    key = (size, bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
    ]
    font = None
    for cand in candidates:
        if cand and Path(cand).exists():
            try:
                font = ImageFont.truetype(cand, size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def _text_w(draw, s: str, font) -> int:
    box = draw.textbbox((0, 0), s, font=font)
    return box[2] - box[0]


# --------------------------------------------------------------------------- #
# Phase / timing logic derived from the measured telemetry
# --------------------------------------------------------------------------- #
def phase_name(raw: str) -> str:
    """Normalise a recorded phase label to a short uppercase name."""
    return str(raw).replace("Phase.", "").upper()


def stage_label(row) -> str:
    """Derive the demo stage label for one telemetry row."""
    ph = phase_name(row["phase"])
    lifted = float(row["object"][2]) > 0.012
    if ph == "APPROACH":
        return "APPROACH"
    if ph == "DESCEND":
        return "DESCEND"
    if ph == "TRANSIT":
        return "LIFT" if lifted else "TRANSIT"
    if ph == "PLACE":
        return "PLACE"
    return "SUCCESS"


# --------------------------------------------------------------------------- #
# Overlay / composition
# --------------------------------------------------------------------------- #
INK = (26, 30, 36)
MUTED = (96, 104, 116)
PANEL_BG = (246, 247, 249)
ACCENT = (20, 88, 160)
GREEN = (26, 122, 74)


def _panel(size, radius: int = 16, fill=PANEL_BG, alpha: int = 205):
    from PIL import Image
    from PIL import ImageDraw as _ID

    w, h = size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = _ID.Draw(overlay)
    od.rounded_rectangle(
        [0, 0, w - 1, h - 1],
        radius=radius,
        fill=(*fill, alpha),
        outline=(208, 213, 220, 255),
        width=1,
    )
    return overlay

REPEAT = 7  # interpolate 3 Hz telemetry toward 30 video-fps (measured states preserved)
INTRO_S, OUTRO_S = 3, 9

TITLE = "ROBOT CI"
SUBTITLE = "Telemetry-driven reconstruction"
CONDITION = "nominal / seed 3 / policy_v1  ·  Gazebo Harmonic"


def compose_frame(
    base_img,
    row,
    *,
    show_phase: bool = True,
    show_status: bool = True,
    t_override: float | None = None,
    header: str = TITLE,
    header_sub: str = SUBTITLE,
):
    """Overlay telemetry, phase and status captions onto one episode frame."""
    from PIL import ImageDraw

    img = base_img.convert("RGB")
    draw = ImageDraw.Draw(img)

    f_title = _font(40, bold=True)
    f_sub = _font(26)
    f_phase = _font(32, bold=True)
    f_small = _font(24)
    f_tiny = _font(22)

    draw.text((48, 40), header, font=f_title, fill=INK)
    draw.text((50, 92), header_sub, font=f_sub, fill=MUTED)

    if show_phase:
        stage = stage_label(row)
        t = t_override if t_override is not None else float(row["sim_t"])
        label = f"{stage}   ·   t = {t:5.2f} s"
        tw = _text_w(draw, label, f_phase)
        pw, phh = tw + 48, 56
        px, py = WIDTH - pw - 48, 44
        img.paste(_panel((pw, phh)), (px, py), _panel((pw, phh)))
        draw = ImageDraw.Draw(img)
        draw.text((px + 24, py + 11), label, font=f_phase, fill=ACCENT)

    if show_status:
        grasped = "GRASPED" if row.get("grasped") else "gripper open"
        ap = float(row["aperture_m"]) * 1000.0
        lift = float(row["object"][2]) - 0.008
        lines = [
            ("ROBOT CI — Telemetry-driven reconstruction", f_tiny, MUTED),
            (f"gripper: {grasped}   aperture {ap:4.1f} mm", f_small, INK),
            (f"cube lift: {lift * 1000:4.1f} mm   policy_v1  seed 3", f_small, INK),
        ]
        line_h = 34
        ph = 26 + line_h * len(lines)
        panel = _panel((640, ph))
        img.paste(panel, (48, HEIGHT - ph - 36), panel)
        draw = ImageDraw.Draw(img)
        yy = HEIGHT - ph - 36 + 15
        for text, font, colour in lines:
            draw.text((68, yy), text, font=font, fill=colour)
            yy += line_h

    return img


def title_card():
    """Render the opening title card for the demo video."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (WIDTH, HEIGHT), (246, 247, 249))
    draw = ImageDraw.Draw(img)
    f_big = _font(120, bold=True)
    f_sub = _font(46)
    f_line = _font(32)
    f_tiny = _font(24)
    tw = _text_w(draw, TITLE, f_big)
    draw.text(((WIDTH - tw) / 2, 360), TITLE, font=f_big, fill=INK)
    sw = _text_w(draw, SUBTITLE, f_sub)
    draw.text(((WIDTH - sw) / 2, 520), SUBTITLE, font=f_sub, fill=ACCENT)
    for i, line in enumerate(
        [CONDITION, "Franka Emika Panda · measured joint & object telemetry"]
    ):
        w = _text_w(draw, line, f_line)
        draw.text(((WIDTH - w) / 2, 620 + i * 46), line, font=f_line, fill=MUTED)
    note = "Re-projected from a recorded episode — not live Gazebo footage."
    nw = _text_w(draw, note, f_tiny)
    draw.text(((WIDTH - nw) / 2, HEIGHT - 120), note, font=f_tiny, fill=MUTED)
    return img


def outro_card(data, metrics):
    """Render the closing summary card from the measured metrics."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (WIDTH, HEIGHT), (246, 247, 249))
    draw = ImageDraw.Draw(img)
    f_big = _font(64, bold=True)
    f_line = _font(38)
    f_key = _font(34)
    f_val = _font(40, bold=True)
    f_tiny = _font(24)

    head = "Verified result"
    hw = _text_w(draw, head, f_big)
    draw.text(((WIDTH - hw) / 2, 150), head, font=f_big, fill=INK)

    rows = [
        ("task success", "PASS" if metrics["task_success"] == 1.0 else "FAIL", GREEN),
        ("placement error", f"{metrics['placement_error_m'] * 1000:5.1f} mm", ACCENT),
        ("object lift", f"{metrics['object_lift_m'] * 1000:5.1f} mm", ACCENT),
        ("completion time", f"{metrics['completion_time_s']:.2f} s", MUTED),
        ("control ticks", f"{int(data['steps'])}", MUTED),
    ]
    y = 300
    x0 = WIDTH // 2 - 320
    for key, val, colour in rows:
        draw.text((x0, y), key, font=f_key, fill=MUTED)
        vw = _text_w(draw, val, f_val)
        draw.text((x0 + 660 - vw, y - 6), val, font=f_val, fill=colour)
        y += 78
        draw.line([(x0, y - 24), (x0 + 660, y - 24)], fill=(225, 228, 233), width=1)

    tag = "ROBOT CI — Telemetry-driven reconstruction"
    tagw = _text_w(draw, tag, f_line)
    draw.text(((WIDTH - tagw) / 2, 760), tag, font=f_line, fill=INK)
    cta = "github.com/Shivani767/robotic-project"
    cw = _text_w(draw, cta, f_key)
    draw.text(((WIDTH - cw) / 2, 830), cta, font=f_key, fill=ACCENT)
    note = "Values measured in the evidence run; rendered from recorded telemetry."
    nw = _text_w(draw, note, f_tiny)
    draw.text(((WIDTH - nw) / 2, HEIGHT - 90), note, font=f_tiny, fill=MUTED)
    return img



# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #
def _readme(data, metrics, rows, results, git_commit, total_frames, nmesh) -> str:
    grasped_ticks = sum(1 for r in rows if r.get("grasped"))
    return f"""# {TITLE} — {SUBTITLE}

A presentation package reconstructed **only** from measured Gazebo Harmonic data.
No trajectory was re-simulated, hand-posed, or fabricated.

## Reconstruction method
- **Renderer:** {results['renderer']} with the real Franka Emika Panda meshes
  ({nmesh} meshes) from `mujoco/models/franka_emika_panda/panda.xml`.
  Each of the 7 arm joints and both finger joints is driven directly from the
  recorded telemetry; frames are interpolated between the 3 Hz ticks for
  30 fps playback without altering any measured start/end state.
- **Camera:** a single fixed three-quarter studio camera keeps the arm,
  gripper, cube and target zone in frame for the whole episode. No camera
  jumps. A top-down plan view is exported separately as `images/camera_view.png`.
- **Object motion:** this evidence run records the target cube's position AND
  orientation (`object`, `object_quat`), so the cube's trajectory is measured,
  not invented.

## Telemetry source & provenance
- Evidence run: `{results['telemetry_source']}`
- Git commit: `{git_commit}`
- Condition: {CONDITION}
- Seed: {data.get('seed', 3)}  ·  Policy: {data.get('policy', 'policy_v1')}

## Measured outcome (verified)
- success: {data['success']} ({data['status']}), {data['steps']} control ticks
- task_success={metrics['task_success']}, grasp_success={metrics['grasp_success']},
  placement_success={metrics['placement_success']}
- placement_error_m={metrics['placement_error_m']:.4f} (tolerance 0.05 m)
- object_lift_m={metrics['object_lift_m']:.4f} (required 0.05 m)
- completion_time_s={metrics['completion_time_s']:.2f}
- {grasped_ticks} of {len(rows)} ticks with the object grasped

## What each file is
- `video/robot_ci_demo.mp4` ({total_frames / FPS:.1f} s, {WIDTH}x{HEIGHT} @ {FPS} fps):
  telemetry-driven reconstruction of the measured pick-and-place.
- `images/thumbnail.png`, `overview.png`, `robot_view.png`: three-quarter renders
  from recorded joint states.
- `images/camera_view.png`: top-down plan view (genuine reconstruction, not a
  camera capture from the episode — the run recorded joint/object telemetry, no
  RGB stream).
- `results/results.json`, `results/telemetry.csv`: measured values and the
  per-tick telemetry used to render.
- `logs/`: source and render logs.
- `manifest.json`: SHA-256 of every artifact.

## Limitations
- The episode recorded joint and object telemetry but no synchronized camera
  stream, so no literal "camera footage" of this run exists. All visuals are a
  physics-faithful 3-D reconstruction from that telemetry, labelled
  "Telemetry-driven reconstruction".

## Reproduce
```
python3 scripts/make_linkedin_demo.py
```
"""


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #
def main() -> int:
    """Compose the recorded episode into the LinkedIn demo video."""
    from PIL import Image

    from robot_ci.recording.mujoco_scene import PandaScene

    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    rows = data["rows"]
    assert data["success"] is True and metrics["task_success"] == 1.0
    assert len(rows) == 115, f"expected 115 measured ticks, got {len(rows)}"

    scene = PandaScene(WIDTH, HEIGHT)
    hero = scene.three_quarter_camera()
    plan = scene.plan_camera()

    for sub in ("video", "images", "results", "logs"):
        (STAGE / sub).mkdir(parents=True, exist_ok=True)

    intro_frames = INTRO_S * FPS
    main_frames = len(rows) * REPEAT
    outro_frames = OUTRO_S * FPS
    total_frames = intro_frames + main_frames + outro_frames

    intro = title_card()
    outro = outro_card(data, metrics)

    frame_dir = STAGE / "_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for _ in range(intro_frames):
        intro.save(frame_dir / f"{idx:05d}.png")
        idx += 1

    for ti, row in enumerate(rows):
        j = np.asarray(row["measured_joints"][:7], dtype=np.float64)
        ap = float(row["aperture_m"])
        obj = tuple(float(v) for v in row["object"])
        quat = tuple(float(v) for v in row.get("object_quat", (1.0, 0.0, 0.0, 0.0)))
        nxt = rows[ti + 1] if ti + 1 < len(rows) else row
        jn = np.asarray(nxt["measured_joints"][:7], dtype=np.float64)
        apn = float(nxt["aperture_m"])
        objn = tuple(float(v) for v in nxt["object"])
        quatn = tuple(float(v) for v in nxt.get("object_quat", quat))
        for k in range(REPEAT):
            s = k / REPEAT
            ji = j * (1 - s) + jn * s
            api = ap * (1 - s) + apn * s
            obji = tuple(a * (1 - s) + b * s for a, b in zip(obj, objn, strict=False))
            qi = np.array(
                tuple(a * (1 - s) + b * s for a, b in zip(quat, quatn, strict=False)),
                dtype=np.float64
            )
            qi = qi / (np.linalg.norm(qi) + 1e-9)
            frame = scene.render_studio(ji, api, obji, tuple(qi), camera=hero)
            t_i = float(row["sim_t"]) * (1 - s) + float(nxt["sim_t"]) * s
            orow = {
                "phase": nxt["phase"] if s >= 0.5 else row["phase"],
                "sim_t": t_i,
                "aperture_m": api,
                "object": list(obji),
                "grasped": nxt.get("grasped") if s >= 0.5 else row.get("grasped"),
            }
            composed = compose_frame(Image.fromarray(frame), orow)
            composed.save(frame_dir / f"{idx:05d}.png")
            idx += 1

    for _ in range(outro_frames):
        outro.save(frame_dir / f"{idx:05d}.png")
        idx += 1

    assert idx == total_frames, (idx, total_frames)

    ff = _ffmpeg()
    video_path = STAGE / "video" / "robot_ci_demo.mp4"
    cmd = [
        ff, "-y", "-framerate", str(FPS),
        "-i", str(frame_dir / "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=1920:1080", "-r", str(FPS),
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{res.stderr[-2000:]}")

    def still_at(frac: float, camera):
        ti = min(len(rows) - 1, max(0, int(frac * (len(rows) - 1))))
        row = rows[ti]
        j = np.asarray(row["measured_joints"][:7], dtype=np.float64)
        quat = tuple(float(v) for v in row.get("object_quat", (1.0, 0.0, 0.0, 0.0)))
        fr = scene.render_studio(
            j, float(row["aperture_m"]),
            tuple(float(v) for v in row["object"]), quat, camera=camera,
        )
        return Image.fromarray(fr)

    thumb = still_at(0.80, hero)
    thumb = compose_frame(thumb, rows[int(0.80 * (len(rows) - 1))])
    thumb.save(STAGE / "images" / "thumbnail.png")

    still_at(0.62, hero).save(STAGE / "images" / "overview.png")
    still_at(0.60, hero).save(STAGE / "images" / "robot_view.png")
    still_at(0.62, plan).save(STAGE / "images" / "camera_view.png")


    with (STAGE / "results" / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["step", "sim_t_s", "phase",
             "j1", "j2", "j3", "j4", "j5", "j6", "j7",
             "aperture_m", "obj_x", "obj_y", "obj_z", "grasped"]
        )
        for row in rows:
            j = row["measured_joints"][:7]
            w.writerow(
                [row["step"], f"{row['sim_t']:.4f}", phase_name(row["phase"]),
                 *[f"{float(v):.6f}" for v in j],
                 f"{float(row['aperture_m']):.6f}",
                 *[f"{float(v):.6f}" for v in row["object"]],
                 int(bool(row.get("grasped")))]
            )

    results = {
        "schema": "robot_ci.linkedin_demo.results/2",
        "title": TITLE,
        "subtitle": SUBTITLE,
        "condition": CONDITION,
        "telemetry_source": str(EVIDENCE.relative_to(REPO)),
        "renderer": "MuJoCo " + __import__("mujoco").__version__,
        "model": "mujoco/models/franka_emika_panda/panda.xml (real meshes)",
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
        "object_pose_recorded": True,
        "reconstruction_note": (
            "Cube position AND orientation are measured in this evidence run, so "
            "object motion is not fabricated. Robot motion is driven directly from "
            "recorded j1..j7 and finger aperture; frames are interpolated between "
            "3 Hz ticks for 30 fps playback without altering the measured states."
        ),
    }
    (STAGE / "results" / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for name in ("episode_seed3_0mm_latest.log", "stage_check_latest.log"):
        src = REPO / "experiments/results/gazebo" / name
        if src.exists():
            (STAGE / "logs" / name).write_bytes(src.read_bytes())
    (STAGE / "logs" / "render.log").write_text(
        f"MuJoCo render: {total_frames} frames @ {FPS} fps, {WIDTH}x{HEIGHT}\n"
        f"model meshes: {scene.model.nmesh}\n"
        f"interpolated from 115 measured ticks (REPEAT={REPEAT})\n",
        encoding="utf-8",
    )

    git_commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    (STAGE / "README.md").write_text(
        _readme(data, metrics, rows, results, git_commit, total_frames, scene.model.nmesh),
        encoding="utf-8",
    )
    manifest = {
        "schema": "robot_ci.linkedin_demo.manifest/2",
        "title": f"{TITLE} — {SUBTITLE}",
        "condition": CONDITION,
        "git_commit": git_commit,
        "source_evidence_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
        "renderer": results["renderer"],
        "video": {"frames": total_frames, "fps": FPS, "width": WIDTH, "height": HEIGHT},
    }
    files = sorted(
        p for p in STAGE.rglob("*")
        if p.is_file() and p.name != "manifest.json" and "_frames" not in p.parts
    )
    manifest["artifact_sha256"] = {
        str(p.relative_to(STAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (STAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    shutil.rmtree(STAGE / "_frames", ignore_errors=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(STAGE))

    dest_video = REPO / "video" / "robot_ci_demo.mp4"
    dest_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(video_path, dest_video)

    print(f"OK  video  {video_path}  ({video_path.stat().st_size/1e6:.1f} MB)")
    print(f"OK  zip    {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"OK  frames {total_frames}  seconds {total_frames / FPS:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
