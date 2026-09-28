"""Phase 0 environment probe.

Read-only. Detects OS, architecture, Python, C++ toolchain, CMake, git, Docker,
ROS 2, GPU/CUDA and — critically — whether MuJoCo can build a model, step
physics, and render offscreen.

Usage:
    .venv/bin/python scripts/probe_environment.py

Exit code 0 if MuJoCo physics + rendering both work (the project's minimum
local requirement), 1 otherwise.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    """Run a command, returning stripped stdout or the error text."""
    """Run a command, returning stripped stdout or the error text."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return (out.stdout or out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<error: {exc}>"


def _which(name: str) -> str:
    return shutil.which(name) or "NOT FOUND"


def header(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===")


def main() -> int:
    """Entry point. Returns a POSIX exit code."""
    header("OS / ARCH")
    print(f"platform     : {platform.system()} {platform.release()}")
    print(f"machine      : {platform.machine()}")
    print(f"processor    : {_run(['sysctl', '-n', 'machdep.cpu.brand_string'])}")
    print(f"logical cpus : {os.cpu_count()}")
    print(f"python       : {sys.version.split()[0]} at {sys.executable}")

    header("TOOLCHAIN")
    for tool in ("clang++", "cmake", "ninja", "git", "docker", "ros2", "colcon",
                 "nvidia-smi", "nvcc"):
        print(f"{tool:<12}: {_which(tool)}")
    try:
        print(f"{'mem (GiB)':<12}: {int(_run(['sysctl', '-n', 'hw.memsize'])) // 2**30}")
    except (ValueError, TypeError):
        print(f"{'mem (GiB)':<12}: unknown")

    header("MUJOCO")
    try:
        import mujoco
    except Exception as exc:
        print(f"IMPORT FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"mujoco version: {mujoco.__version__}")
    xml = (
        '<mujoco><worldbody><light pos="0 0 2"/>'
        '<geom type="plane" size="1 1 .1" rgba="0.8 0.8 0.8 1"/>'
        '<body pos="0 0 0.05"><freejoint/>'
        '<geom type="box" size=".05 .05 .05" rgba="1 0 0 1"/></body>'
        '<camera name="cam" pos="0.6 0 0.4" xyaxes="0 -1 0 0.4 0 0.7"/>'
        "</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(50):
        mujoco.mj_step(model, data)
    print(f"physics      : OK (nq={model.nq} nv={model.nv} z={data.qpos[2]:.4f} m)")

    render_ok = False
    for backend in ("osmesa", "egl", "glfw"):
        os.environ["MUJOCO_GL"] = backend
        try:
            renderer = mujoco.Renderer(model, 64, 64)
            renderer.update_scene(data, camera="cam")
            pixels = renderer.render()
            print(
                f"render       : OK via {backend} "
                f"shape={pixels.shape} mean={pixels.mean():.2f} std={pixels.std():.2f}"
            )
            render_ok = bool(pixels.std() > 0.0)
            renderer.close()
            break
        except Exception as exc:
            print(f"render       : {backend} failed -> {type(exc).__name__}: {str(exc)[:120]}")

    if not render_ok:
        print("\nRESULT: physics OK but offscreen rendering FAILED.")
        return 1
    print("\nRESULT: MuJoCo physics + offscreen rendering both work. Local CPU research loop is viable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
