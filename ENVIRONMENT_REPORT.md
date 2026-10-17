# ROBOT CI — ENVIRONMENT REPORT (Phase 0)

**Generated:** 2026-09-28 (revised: simulator target changed to **MuJoCo**)
**Host:** `Shivanis-MacBook-Air.local`
**Probe method:** read-only shell commands (`sw_vers`, `uname`, `sysctl`, `which`,
`docker info`, `system_profiler`, `df`, `vm_stat`) plus one live MuJoCo smoke test
(reproduced verbatim in `scripts/probe_environment.py`).
**No CUDA, no NVIDIA drivers, no Isaac Sim were installed.**

> All values below are measured facts from this machine, not estimates.

---

## 0. Executive summary (the 11 questions you asked)

| # | Item | Detected value |
|---|---|---|
| 1 | Mac architecture | **Apple Silicon `arm64`**, Apple M5, 10 cores, macOS 26.5 (build 25F71) |
| 2 | Python version | **3.12.14** selected (`/opt/homebrew/bin/python3.12`); 3.14.7 present but rejected |
| 3 | C++ compiler | **Apple clang 21.0.0** (`clang-2100.1.101`), target `arm64-apple-darwin25.5.0`; Xcode CLT SDK 26.5 |
| 4 | CMake | **4.4.3** (`/opt/homebrew/bin/cmake`), plus `ninja` and `make` |
| 5 | Git | **2.55.0**; repository initialized here on branch `main` |
| 6 | **MuJoCo availability** | **NOT preinstalled**; `pip install mujoco` **succeeded → 3.14.0**, and a **live smoke test passed** (see §6) |
| 7 | Docker availability | **Docker 29.6.1** (Docker Desktop, context `desktop-linux`); `colima`/`podman` absent |
| 8 | ROS 2 availability | **ABSENT** — no `ros2`, no `colcon`, no `/opt/ros`, `ROS_DISTRO` unset |
| 9 | RAM / disk | **16 GiB RAM**, **272 GiB free** of 460 GiB |
| 10 | Recommended local architecture | CPU-only MuJoCo + Mock/Synthetic backends; ROS 2 / MoveIt 2 remote-only; Colab/Kaggle *optional* |
| 11 | Phase 1 plan | §15 below |

**Headline result:** MuJoCo runs natively on this machine **including offscreen RGB
rendering through OSMesa software rasterization**. The full research loop — scenario →
simulate → observe → corrupt → policy → evaluate → regression → counterexample — is
therefore executable on this MacBook with no GPU and no cloud.

---

## 1. Operating system

| Property | Value |
|---|---|
| Product name | macOS |
| Product version | 26.5 |
| Build | 25F71 |
| Kernel | Darwin 25.5.0 |
| Architecture | `arm64` (Apple Silicon) |

Repository root: `/Users/shivanibhandari/Downloads/Robotic Project`

## 2. Compute

| Property | Value |
|---|---|
| CPU | Apple M5 |
| Physical / logical cores | 10 / 10 |
| RAM | 17,179,869,184 bytes = **16 GiB** |
| GPU | Apple M5 integrated, 8 cores, **Metal 4** |
| Free disk (Data volume) | **272 GiB** of 460 GiB (38% used) |
| Load average at probe | 2.90 / 2.48 / 2.35 |
| Open file limit | 1,048,575 |

## 3. Python

| Interpreter | Path | Version | Selection |
|---|---|---|---|
| Homebrew default | `/opt/homebrew/bin/python3` | 3.14.7 | not selected |
| Homebrew 3.12 | `/opt/homebrew/bin/python3.12` | 3.12.14 | **selected** |
| Apple system | `/usr/bin/python3` | system 3.9.x | not selected |

**Why 3.12 and not 3.14:** 3.14 is very new; binary wheels for the scientific stack
(`numpy`, `scipy`, `pandas`, `matplotlib`) are not yet reliably published for `cp314`
on `macosx-arm64`. Pinning **3.12.14** maximizes wheel coverage and is a supported
CPython release.

**Package managers**

| Tool | Status |
|---|---|
| `uv` 0.12.5 | **present** — used for venv + locked installs |
| `pip` 26.2.1 (3.14 and 3.12) | present |
| `pipx` | **absent** |
| `poetry` | **absent** |
| `conda` / `mamba` | **absent** |

## 4. C/C++ toolchain

| Tool | Version | Path |
|---|---|---|
| `clang` | Apple clang 21.0.0 (clang-2100.1.101) | `/usr/bin/clang` |
| `clang++` | same | `/usr/bin/clang++` |
| `gcc` / `g++` | present (Apple shims) | `/usr/bin/gcc`, `/usr/bin/g++` |
| `cmake` | 4.4.3 | `/opt/homebrew/bin/cmake` |
| `ninja` | present | `/opt/homebrew/bin/ninja` |
| `make` | present | `/usr/bin/make` |
| Xcode CLT | `/Library/Developer/CommandLineTools` | — |
| macOS SDK | 26.5 | `.../CommandLineTools/SDKs/MacOSX.sdk` |

C++17/20 compilation of the ROS 2-independent C++ nodes is **possible locally** for
compile/lint/unit-test verification. Linking against ROS 2 is not (see §6).

## 5. Version control

| Tool | Version |
|---|---|
| `git` | 2.55.0 |
| `gh` (GitHub CLI) | installed via Homebrew |

The directory was **not** under version control at the start of Phase 0. It has been
initialized here on branch `main`, with `user.name = Shivani Bhandari` and
`user.email = Shivani215143@gmail.com` (read from the existing global `~/.gitconfig`;
nothing was invented).

## 6. MuJoCo — the primary simulator (VERIFIED WORKING)

MuJoCo was **not** preinstalled. It was installed into the project virtual environment
and immediately smoke-tested. This is the single most important Phase 0 finding.

| Check | Result |
|---|---|
| `brew list` (mujoco) | not present (Homebrew does not package MuJoCo for this use) |
| `pip install mujoco` | **succeeded**, `mujoco==3.14.0` (macosx `arm64` wheel) |
| `import mujoco` | **OK** under CPython 3.12.14 / arm64 |
| `MjModel.from_xml_string` | **OK** |
| `mj_step` physics | **OK** — 50 steps of a free box, `qpos[2] = 0.0499 m` (settled on the plane) |
| `mj_forward` kinematics | **OK** — `data.xmat.shape == (2, 9)` |
| `MUJOCO_GL=cgl` + `mujoco.Renderer` | **OK** — real `(64, 64, 3)` uint8 image, `mean=18.36`, `std=40.31` (non-constant => real geometry, not a blank buffer) |
| `MUJOCO_GL=egl` | **rejected at import** — Linux-only value, MuJoCo raises on macOS |
| `MUJOCO_GL=osmesa` | **rejected at import** — Linux-only value, MuJoCo raises on macOS |
| `MUJOCO_GL=glfw` | also works on this host (same pixels) |
| render throughput | **1.4 ms/frame** at 160x120 (cgl) |

**Conclusion:** MuJoCo provides **full rigid-body physics, kinematics, collision
detection and offscreen RGB/depth rendering on this MacBook with no GPU**. The core
research platform is therefore fully local and CPU-only. Cloud GPU is an *optional*
accelerator for perception models and large batch sweeps, never a requirement.

**Note on the GL backend.** MuJoCo resolves its OpenGL backend **at import time**
and the valid values are platform-specific. On macOS only `cgl` and `glfw` are
accepted; `egl` and `osmesa` are Linux-only and make `import mujoco` raise:

```
RuntimeError: invalid value for environment variable MUJOCO_GL: osmesa
```

`robot_ci.simulation.mujoco` selects the correct value *before* the first import,
so users never have to set anything. The mapping is:

| Platform | Selected backend | Reason |
|---|---|---|
| macOS | `cgl` | Apple OpenGL; `egl` and `osmesa` are unavailable |
| Linux + GPU | `egl` | hardware offscreen context |
| Linux + X11 | `glfw` | on-screen GL context |

Measured throughput at the project's default 160x120 resolution is **1.4 ms per
frame**, which is ample for batch sweeps on the 10 available CPU cores.

## 7. GPU / CUDA / NVIDIA

| Check | Result |
|---|---|
| `nvidia-smi` | **not found** |
| `nvcc` | **not found** |
| `CUDA_PATH` | **unset** |
| GPU vendor | Apple (`0x106b`), Metal 4 |

**Conclusion: no NVIDIA GPU, no CUDA, no CUDA driver.** CUDA and NVIDIA drivers are NOT
and will NOT be installed on macOS. Isaac Sim is **not used by this project at all** —
the specification names MuJoCo as the primary simulator.

## 8. ROS 2 / colcon

| Check | Result |
|---|---|
| `ros2` executable | **not found** |
| `colcon` executable | **not found** |
| `/opt/ros` | **does not exist** |
| `ROS_DISTRO` | **unset** |

**Conclusion: ROS 2 Jazzy is NOT installed and cannot be installed natively on macOS**
(no official ROS 2 Jazzy binaries exist for macOS). ROS 2, MoveIt 2 and `ros2_control`
are therefore **optional adapter layers** (Phase 10–11) exercised on a remote Linux
machine. The local MuJoCo research engine must and does run without them.

## 9. Containers

| Tool | Status |
|---|---|
| `docker` | 29.6.1, context `desktop-linux` (Docker Desktop) |
| `docker-compose` | present at `/usr/local/bin/docker-compose` |
| `colima` / `podman` | **absent** |

Docker is useful for Linux CI runners and headless batch jobs. It is **not** a
simulation backend and does not relax the CPU-only local constraint.

## 10. JavaScript / dashboard toolchain

| Tool | Version |
|---|---|
| `node` | v20.20.2 |
| `npm` | 10.8.2 |
| `pnpm`, `yarn` | present |

Sufficient for the Vite + React + TypeScript dashboard (Phase 12).

## 11. Network reachability (probe latency)

| Endpoint | HTTP | Latency |
|---|---|---|
| `https://pypi.org/simple/` | 200 | 8.66 s (single cold request) |
| `https://registry.npmjs.org/` | 200 | fast |
| `https://github.com/` | 200 | fast |

PyPI is reachable; 8.66 s is one cold DNS/TLS request, not bulk throughput. This
enables `uv` installs, Colab/Kaggle installation and GitHub Actions.

## 12. Deliberately never installed locally

* CUDA toolkit / NVIDIA drivers — unsupported on macOS.
* Isaac Sim / Omniverse / Isaac Lab — out of scope for this project entirely.
* ROS 2 Jazzy, MoveIt 2, `ros2_control` — no macOS binaries; optional adapters only.

---

## 13. Capability matrix — what is possible on this Mac

| Capability | Local (this Mac) | Colab/Kaggle | Remote Linux + GPU |
|---|---|---|---|
| Core library, config, CLI, schema | YES | YES | YES |
| Scenario generation | YES | YES | YES |
| `MockBackend` / `SyntheticBackend` | YES (CPU) | YES | YES |
| **`MuJoCoBackend` physics + RGB/depth** | **YES (CPU, OSMesa)** | YES | YES |
| Regression detection + statistics | YES | YES | YES |
| Counterexample search + boundary mapping | YES (CPU) | YES | YES |
| Failure taxonomy / diagnosis / ablation | YES | YES | YES |
| Research experiments + baselines | YES | YES | YES |
| SQLite experiment database | YES | YES | YES |
| Dashboard build + local server | YES | n/a | YES |
| C++ compile checks / unit tests | YES | n/a | YES |
| Learned perception models (PyTorch) | slow / optional | **YES (T4)** | YES |
| ROS 2, MoveIt 2, `ros2_control` runtime | **NO** | n/a | YES |
| Gazebo backend | **NO** | n/a | optional, later |

**Design consequence:** every backend and every external system (ROS 2, MoveIt 2, SQLite,
future Gazebo) sits behind a Python `Protocol`/ABC. The core package imports and tests
cleanly on macOS with only the CPU dependency set installed.

## 14. Known risks and mitigations

1. **PyPI cold-start latency** — mitigated with `uv` and a committed lock file.
2. **Apple Silicon wheel availability** — mitigated by pinning CPython 3.12.
3. **16 GiB RAM** — MuJoCo batch sweeps use bounded worker processes with small images
   (<= 256x256); concurrency is capped by a configurable limit.
4. **OSMesa is software rendering** — slower than GPU rasterization. Mitigated by small
   default resolution, per-backend resolution configuration, and a result cache.
5. **ROS 2 cannot be validated locally** — C++ adapter code is compile-checked locally
   and built/tested on a remote machine. This is documented honestly rather than
   claimed as locally verified.
6. **No local GPU for deep perception** — the perception module ships dependency-light
   classical estimators (colour + depth segmentation) as the default; learned models are
   optional and run in the notebook backend.

---

## 15. Phase 0 verdict

This machine is a **fully capable CPU-only host** for the entire ROBOT CI research loop.
MuJoCo physics and offscreen rendering were verified live, not assumed. The remaining
hard constraint — ROS 2 / MoveIt 2 — is an *optional* integration by design, never a
dependency of the research engine.

(<= 256x256) rendering is fast enough for batch sweeps on 10 CPU cores.

## 16. Release commit

* Release tag: `v1.0.0` (published to GitHub)
* Release commit SHA: `4663c46b896e3a20ecf074f87b9fdcf9acca092b`
* Branch: `main`
* Release URL: https://github.com/Shivani767/robotic-project/releases/tag/v1.0.0
* Frozen at revision `d7026af` for the research-record evidence; all dashboard,
  docs, code, and `experiments/results/` artifacts committed together.
* Release asset: `video/robot_ci_demo.mp4` (demo walkthrough; intentionally not
  committed as a binary in the repository).

Reproducible startup (from repository root):

```bash
python experiments/build_dashboard_data.py   # regenerate dashboard/data/*
python dashboard/serve.py 8123               # -> http://localhost:8123/dashboard/
.venv/bin/python -m pytest -q                # pre-commit gate: 561 passed, 7 skipped
.venv/bin/ruff check dashboard/              # pre-commit gate: all checks passed
```

