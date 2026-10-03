r"""Is the phantom depth wedge the robot, the sun, or the world itself?

The depth image contains a wedge in the bottom-right whose depth is 10-25%
shorter than the table plane at the same pixel (ratio down to 0.745), which
tilts the plane fit and defeats object detection. RGB says the wedge is grey
table-coloured surface. This probe strips one model at a time from the
generated SDF and re-measures the row-100 depth profile, so the source is
identified by removal rather than by argument.

Run inside the container::

    docker run --rm -v "$PWD":/workspace/robot-ci -w /workspace/robot-ci \\
        robot-ci-gazebo-demo:latest \\
        bash -c 'source /opt/ros/jazzy/setup.bash && python3 scripts/gazebo_depth_wedge_probe.py'
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import numpy as np


def _strip_model(sdf: str, name: str) -> str:
    """Remove one ``<model name=...>`` block from an SDF document.

    Gazebo's own URDF converter emits ``name="panda"`` while the hand-written
    parts of the generator use single quotes, so both spellings are matched --
    a strip that silently no-ops would make every variant below look identical.
    """
    for quote in ('"', "'"):
        start = sdf.find(f"<model name={quote}{name}{quote}>")
        if start >= 0:
            break
    if start < 0:
        return sdf
    end = sdf.find("</model>", start)
    if end < 0:
        return sdf
    return sdf[:start] + sdf[end + len("</model>") :]


def _strip_sun(sdf: str) -> str:
    """Remove the directional light (and its shadows) from the SDF."""
    start = sdf.find("<light")
    if start < 0:
        return sdf
    end = sdf.find("</light>", start)
    if end < 0:
        # Self-closing light elements are not used by the generator, but if one
        # appears the unmodified document is returned rather than corrupted.
        return sdf
    return sdf[:start] + sdf[end + len("</light>") :]


def _capture(label: str, sdf: str) -> None:
    """Start Gazebo on ``sdf``, grab one depth frame, print the profile."""
    from robot_ci.gazebo.world import GAZEBO_WORLD_NAME
    from robot_ci.simulation.gazebo import _DepthStream

    path = Path(f"/tmp/wedge_probe_{label}.sdf")
    path.write_text(sdf, encoding="utf-8")
    process = subprocess.Popen(
        ["gz", "sim", "-s", "-r", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        stream = _DepthStream(GAZEBO_WORLD_NAME)
        depth = None
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            depth, _rgb, width, _height = stream.latest()
            if depth is not None and width > 0:
                break
            time.sleep(0.5)
        if depth is None:
            print(f"{label}: NO FRAME")
            return
        row = 100
        profile = [
            round(float(depth[row, c]), 3) if np.isfinite(depth[row, c]) else None
            for c in range(95, 160, 5)
        ]
        print(f"{label}: row100 cols95..155 step5 -> {profile}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            process.kill()
        time.sleep(2.0)


def main() -> int:
    """Run the three variants: full world, no robot, no sun."""
    from robot_ci.gazebo.world import world_sdf
    from robot_ci.scenarios import default_scenario

    base = world_sdf(default_scenario(seed=42), robot_urdf="gazebo/models/panda/panda.urdf")
    # Sanity: the panda block must be findable or the strips below are lies.
    assert "<model name=\"panda\">" in base or "<model name='panda'>" in base, (
        "panda block not found; strip would be a no-op"
    )
    assert re.search(r"<light[^>]*>", base), "no light found"

    _capture("full", base)
    _capture("no_robot", _strip_model(base, "panda"))
    _capture("no_sun", _strip_sun(base))
    _capture("no_robot_no_sun", _strip_model(_strip_sun(base), "panda"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
