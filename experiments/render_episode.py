r"""Render one episode to an animated GIF, so the robot can be watched working.

Every other artifact in this repository is a table of numbers: a success rate,
a delta, a confidence interval. Those are the right things to reason with, and
they are impossible to look at. This script closes that gap by driving the real
episode loop through the real runner and keeping the RGB frame the simulator
renders after each control tick.

Two properties matter for the output to be evidence rather than decoration:

* **It is the same loop.** The frames come from ``run_episode`` -- the runner
  every number in this project came out of -- with a recording proxy wrapped
  around the backend. Nothing about the episode is re-implemented here, so a
  video cannot show a behaviour the metrics did not see.
* **It is labelled.** Every GIF is written next to a sidecar JSON naming the
  policy, seed, condition, terminal status and step count, and the caption is
  burned into the frame. A video of a robot failing, presented without its
  condition, is the exact error this project exists to catch.

The camera is ``rgb_camera``: a fixed third-person view of the workspace that
the scene generator places for every scenario. It is the same camera the
robot's own perception stack consumes, so the view is the simulator's, not a
viewer's chosen angle.

What this script deliberately does *not* do
-------------------------------------------
**It does not rename ``scenario_id``.** The scene's object placement RNG is
seeded from ``(scenario.seed, "mujoco-scene", scenario.scenario_id)`` -- see
``simulation/mujoco_env.py`` -- because the orchestrator derives each episode's
seed from the cell's ``scenario_id`` (``experiments/orchestrator.py``). The
identifier *is* an input to the physics, not a caption. The first version of
this script rewrote it to describe the clip and thereby moved the target
object: it rendered a failure at a seed that succeeds. The identifier is left
exactly as the project builds it, and the condition is recorded in the
sidecar and the caption instead.

**It does not raise the resolution.** ``ScenarioConfig.resolution`` is a
derived property of ``sensors.intrinsics.shape``, and the intrinsics set the
camera's FOV and the depth sampling the policy reads. Rendering larger would
therefore change the episode being filmed. Frames are rendered at the
scenario's own resolution and upscaled for viewing; the sidecar records both,
so the picture is never mistaken for native pixels.

Usage::

    python experiments/render_episode.py --policy policy_v1 --seed 3
    python experiments/render_episode.py --policy policy_v1 --seed 3 \\
        --depth-noise-mm 8 --out experiments/results/episode_fail.gif
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path("experiments/results")

#: Render every Nth control tick. One frame per tick at 20 Hz is more frames
#: than a GIF needs and proportionally slower to render; every second tick
#: still resolves the arm's motion because the arm moves a fraction of a joint
#: limit per tick.
DEFAULT_STRIDE = 2

#: Frame rate of the written GIF. Chosen so an episode that takes 76 control
#: ticks at 20 Hz plays back at roughly real time rather than as a blur.
DEFAULT_FPS = 20

#: Integer upscale applied to each frame for viewing. The scenario renders at
#: 160x120 because that is what its perception stack consumes; the episode is
#: filmed at that size and then enlarged, never re-rendered larger, because the
#: intrinsics that set the frame size also set the FOV the policy looks through.
DEFAULT_SCALE = 3


class RecordingBackend:
    """A backend proxy that keeps the rendered frame after each step.

    Everything other than :meth:`step` is delegated to the wrapped backend, so
    the runner cannot tell the difference and no behaviour is re-implemented.
    The frame is taken *after* the step, so frame *n* is the state the runner
    itself observed at tick *n*.
    """

    def __init__(self, inner: Any, *, stride: int, max_frames: int) -> None:
        self._inner = inner
        self._stride = max(1, stride)
        self._max_frames = max(1, max_frames)
        self.frames: list[np.ndarray] = []
        self.ticks = 0
        self.render_seconds = 0.0

    def __getattr__(self, name: str) -> Any:
        """Delegate every attribute the runner asks for to the real backend."""
        return getattr(self._inner, name)

    def step(self, handle: Any, action: Any) -> Any:
        """Advance the episode, then render if this tick is sampled."""
        transition = self._inner.step(handle, action)
        self.ticks += 1
        if self.ticks % self._stride == 0 and len(self.frames) < self._max_frames:
            started = time.perf_counter()
            frame = self._inner.render(handle)
            self.render_seconds += time.perf_counter() - started
            if frame.rgb is not None:
                self.frames.append(np.asarray(frame.rgb).astype(np.uint8))
        return transition


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a caption bar into the bottom of a frame.

    Rendering the label into the image rather than only into the filename is
    deliberate: a GIF gets detached from its directory, and an unlabelled
    clip of a robot arm is a claim with no condition attached.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    bar = 26
    draw.rectangle([0, height - bar, width, height], fill=(16, 20, 26))
    draw.text((8, height - bar + 7), text, fill=(235, 238, 242))
    return np.asarray(image)


def _filmstrip(frames: list[np.ndarray]) -> np.ndarray:
    """Compose five evenly spaced frames side by side.

    A GIF cannot be placed in a document and a reviewer scrolling a diff cannot
    play one, so the same episode is also written as a single still image.
    """
    picks = np.linspace(0, len(frames) - 1, num=min(5, len(frames))).astype(int)
    return np.concatenate([frames[i] for i in picks], axis=1)


def _upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    """Enlarge a frame for viewing, without changing what was rendered.

    The scenario renders at 160x120 because that is what its perception stack
    consumes. A GIF at that size is unreadable, so frames are resampled up for
    display. The sidecar records the native resolution alongside the scale, so
    the picture is never mistaken for native pixels.
    """
    if scale <= 1:
        return frame
    from PIL import Image

    image = Image.fromarray(frame)
    resampling = Image.Resampling.LANCZOS
    return np.asarray(image.resize((image.width * scale, image.height * scale), resampling))


def render_episode(
    *,
    policy_name: str,
    seed: int,
    backend_name: str,
    depth_noise_mm: float,
    max_steps: int | None,
    stride: int,
    fps: int,
    scale: int,
    out: Path,
) -> dict[str, Any]:
    """Run one episode, record it, and write the GIF plus its sidecar.

    Returns:
        The sidecar metadata that was written next to the GIF.
    """
    from PIL import Image

    from robot_ci.experiments.runner import run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario, set_by_path
    from robot_ci.simulation import create_backend

    # The scenario is used exactly as the project builds it. Only the sensor
    # condition is swept and ``scenario_id`` is left alone: the scene's object
    # placement RNG is seeded from it, so rewriting the identifier to describe
    # the clip would move the target object (see the module docstring).
    scenario = set_by_path(default_scenario(), "sensors.depth.sigma_mm", depth_noise_mm)
    if max_steps is not None:
        scenario = scenario.model_copy(
            update={"task": scenario.task.model_copy(update={"max_steps": max_steps})}
        )

    engine = create_backend(backend_name)
    recorder = RecordingBackend(engine, stride=stride, max_frames=10_000)
    started = time.perf_counter()
    try:
        outcome = run_episode(
            recorder,
            create_policy(PolicyConfig(name=policy_name), scenario),
            scenario,
            seed=seed,
            experiment_id="render_episode",
        )
    finally:
        engine.close_all()
    wall_s = time.perf_counter() - started

    if not recorder.frames:
        raise SystemExit("no frames were recorded; the scenario may have render_rgb disabled")

    native = scenario.resolution
    verdict = "SUCCESS" if outcome.success else f"FAILED ({outcome.status})"
    label = (
        f"{policy_name}  seed={seed}  depth_noise={depth_noise_mm:g}mm  "
        f"{verdict}  steps={outcome.steps}/{scenario.task.max_steps}"
    )
    frames = [_caption(_upscale(frame, scale), label) for frame in recorder.frames]

    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
        optimize=True,
    )

    still_path = out.with_name(out.stem + "_filmstrip.png")
    Image.fromarray(_filmstrip(frames)).save(still_path)

    sidecar = {
        "gif": str(out),
        "filmstrip": str(still_path),
        "policy": policy_name,
        "seed": seed,
        "backend": backend_name,
        "scenario_id": scenario.scenario_id,
        "depth_noise_mm": depth_noise_mm,
        "max_steps": scenario.task.max_steps,
        "native_resolution": list(native),
        "display_scale": scale,
        "output_resolution": [native[0] * scale, native[1] * scale],
        "status": outcome.status,
        "success": outcome.success,
        "steps": outcome.steps,
        "config_hash": outcome.config_hash,
        "frames": len(frames),
        "stride": stride,
        "fps": fps,
        "render_seconds": round(recorder.render_seconds, 3),
        "episode_wall_s": round(wall_s, 3),
        "caption": label,
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Render one episode to an animated GIF of the robot working.",
    )
    parser.add_argument("--policy", default="policy_v1", help="policy arm to run")
    parser.add_argument("--seed", type=int, default=3, help="episode seed")
    parser.add_argument("--backend", default="mujoco", help="simulator backend")
    parser.add_argument(
        "--depth-noise-mm",
        type=float,
        default=0.0,
        help=(
            "depth camera noise for the episode; 0.0 is the nominal condition "
            "in which both arms solve the task"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the step budget; the scenario default is used otherwise",
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="render every Nth tick")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="GIF playback rate")
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help=(
            "integer upscale applied to each rendered frame for viewing; the "
            "episode itself always runs at the scenario's native resolution"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="output GIF path")
    args = parser.parse_args(argv)

    out = args.out or (
        RESULTS / f"episode_{args.policy}_seed{args.seed}_depth{args.depth_noise_mm:g}mm.gif"
    )

    sidecar = render_episode(
        policy_name=args.policy,
        seed=args.seed,
        backend_name=args.backend,
        depth_noise_mm=args.depth_noise_mm,
        max_steps=args.max_steps,
        stride=args.stride,
        fps=args.fps,
        scale=args.scale,
        out=out,
    )

    print(f"gif        : {out}")
    print(f"filmstrip  : {sidecar['filmstrip']}")
    print(f"sidecar    : {out.with_suffix('.json')}")
    print(
        f"outcome    : {sidecar['status']}  success={sidecar['success']}  steps={sidecar['steps']}"
    )
    print(
        f"frames     : {sidecar['frames']} ({sidecar['stride']}-tick stride, {sidecar['fps']} fps)"
    )
    print(
        f"cost       : {sidecar['episode_wall_s']}s total, "
        f"{sidecar['render_seconds']}s of it rendering",
        flush=True,
    )
    return 0 if sidecar["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
