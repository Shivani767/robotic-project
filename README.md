# ROBOT CI

**Robot manipulation experiments, measured on real simulators — recorded, packaged, and reproducible.**

ROBOT CI is a continuous-integration framework for physical AI: given two
versions of a robot policy, perception model, planner, or inference stack, it
determines automatically whether the new one behaves **worse**, under **which**
conditions, and because of **which** subsystem. Every claim — success, placement
error, collisions — is measured from simulator telemetry and hashed into a
manifest; nothing is narrated. A run that fails is reported as failed.

## What it does

1. **Controls a 7-DOF Panda arm** through pick-and-place in Gazebo Harmonic
   (and MuJoCo) via inverse kinematics, with real joint-state feedback each
   control tick.
2. **Records every experiment**: actual joint states, simulation time, task
   events, metrics, success/failure, and runtime versions — timestamped.
3. **Renders a run video**: live joint-state 3D view combined with the actual
   simulator camera frame, synchronized.
4. **Packages evidence per experiment**: MP4 + PNG figures + `results.json` +
   `manifest.json` (SHA-256 of every artifact) + logs + telemetry, zipped.
5. **Runs an experiment suite** with honest PASS / FAIL / NOT_RUN /
   NOT_SUPPORTED / NOT_IMPLEMENTED status per category.

No trajectories are fabricated or replayed. If a source is unavailable (for
example, no camera stream for a given episode), it is marked unavailable — never
manufactured.

---

## Research contributions (RQ1–RQ5)

ROBOT CI answers five research questions, each measured against artifacts in
`experiments/results/`:

| Question | What it asks | Measured finding |
|---|---|---|
| **RQ1** | Which search strategy finds a counterexample fastest? | Strategy comparison is single-seed / single-axis; shows a trade-off exists, not that one strategy is better in general. |
| **RQ2** | Conditional vs aggregate detection — can the aggregate hide a regression? | **Yes.** Conditional analysis caught a worst cell (−0.167 at `depth_noise_mm=8`, blocking) that the aggregate (−0.033) hid. |
| **RQ3** | Which subsystem is associated with a difference? | `settle_steps` is live and favourable (+0.167), **not** inert: `policy_v3` is bit-identical to `policy_v1` at `settle_steps=6` on 30/30 cells. Attribution: correlation-only → planner **WRONG**; OFAT → perception correct; interaction-aware → perception correct. |
| **RQ4** | Does a counterexample reproduce on disjoint seeds? | **Reproduction rate 4/11 = 36.4%** at `depth_noise_mm=3.0` — a rate is not optional. |
| **RQ5** | Where is the behavioral boundary? | Perturbation boundaries mapped (depth-bias axis is bimodal near threshold — a sampling shape, not a physical result). |

These are drawn from [`research_report.md`](research_report.md), regenerated
from `experiments/results/`.

### Retired hypotheses (reported as failed, not dropped)

| Hypothesis | Claim | Verdict | Why |
|---|---|---|---|
| H1 | Aggregate regression detection is sufficient. | **Rejected** | Conditional detection caught what the aggregate hid. |
| H2 | `policy_v2`'s confidence gate is a real change. | **Rejected** | v2 measured identical to v1 at every noise level (inert). |
| H3 | `settle_steps` is an inert perturbation lever. | **Rejected** | Controlled re-measurement: live and favourable (+0.167). |
| H4 | RQ2 opposing-effect claim holds on the current revision. | **Retired on evidence** | Controlled re-run (24 matched seeds × 4 depth-noise levels × 6 budgets + 150-seed follow-up) did not reproduce it. |
| H5 | Adverse channel is perceptual latency / a deadline. | **Falsified** | `ObjectSpec` has no motion field; nothing in the framework moves an object. |
| H6 | There is a ~33% failure floor (nominal solvability 66.7%). | **Falsified** | Stale pre-fix 8/12 measurement; fixed code yields 120/120 on MuJoCo, 95% CI [0.969, 1.000]. |

### Verified results

**Nominal solvability (120 seeds per policy, fixed grasp-aim code):**

| policy | success | rate | 95% Wilson CI | steps (mean) |
|---|---|---|---|---|
| `policy_v1` | 120/120 | 1.000 | [0.969, 1.000] | 103.8 |
| `policy_v2` | 120/120 | 1.000 | [0.969, 1.000] | 103.8 |
| `policy_v3` | 120/120 | 1.000 | [0.969, 1.000] | 108.8 |

The old 66.7% figure is falsified — it was a stale 8/12 pre-fix measurement,
outside the confidence interval. There is no 33% failure floor.

**Gazebo pick-and-place (measured on a real Gazebo run):**

| Metric | Measured |
|---|---|
| Task success | **true** (grasp + lift + placement all pass) |
| Object lift | 0.0657 m (≥ 0.05 required) |
| Placement error | 0.0221 m (≤ 0.05 tolerance) |
| Ticks with the object grasped | 45 / 115 |

See [`docs/gazebo.md`](docs/gazebo.md) for the engineering log and
[`experiments/results/DEMO.md`](experiments/results/DEMO.md) for the full
stage-by-stage CLI demo with real captured output.


---

## Architecture

**Recorded evidence, not a live backend.** The research dashboard
(`dashboard/`) is a static, read-only console over
`build_dashboard_data.py` + `dashboard/serve.py`. It recomputes every
aggregate from `experiments/results/` and runs 30 consistency checks; missing
metrics say "Not measured." There is no live camera, no simulator backend, and
no WebSocket in the frozen console.

**Backends:** Gazebo (`src/robot_ci/` Gazebo backend), MuJoCo
(`src/robot_ci/` MuJoCo backend), with ROS 2 / MoveIt 2 and an Isaac Sim
adapter as unbuilt backends (future work). Unit + contract tests cover
recording, hashes, metadata, partial failure, and statuses (`tests/`).

---

## Screenshots

Six 1920 × 1080 dashboard screenshots are planned for the research release:

`01-overview.png` · `02-panda-replay.png` · `03-boundary-map.png` ·
`04-counterexample-discovery.png` · `05-subsystem-attribution.png` ·
`06-evidence-provenance.png`

They are captured per [`docs/screenshots/capture.md`](docs/screenshots/capture.md)
(viewport exactly 1920 × 1080, lossless PNG, no upscaling or retouching).
**This terminal is headless** — no browser, no display server, no WebGL
renderer — so the screenshots could not be captured here and must be taken on a
machine with a display. They are intentionally not fabricated and not committed
as placeholders; once captured they belong in `docs/screenshots/`.

---

## Demo

A recorded walkthrough video is attached to the GitHub release:

- **GitHub release v1.0.0:** https://github.com/Shivani767/robotic-project/releases/tag/v1.0.0
- **Demo asset:** `robot_ci_demo.mp4` (release asset)
- **Full CLI walkthrough:** [`experiments/results/DEMO.md`](experiments/results/DEMO.md)

---

## Quick start

The platform runs on macOS through Docker Compose (ARM64-compatible):

```bash
docker compose up --build
```

Open the live experiment console at **http://localhost:8000** (WebSocket on
8001). From the browser you can select a category/condition, set seed / policy /
repeats, start or stop a run, watch live joint states, and download the video,
images, JSON results, and per-experiment ZIP.

### Research dashboard (frozen recorded evidence — no Docker needed)

```bash
python experiments/build_dashboard_data.py   # regenerate dashboard/data/*.json
python dashboard/serve.py 8123              # → http://localhost:8123/dashboard/
```

Read-only console over revision `d7026af`; recomputes aggregates from
`experiments/results/` (30 consistency checks). See
[`docs/dashboard.md`](docs/dashboard.md) and
[`docs/demo_walkthrough.md`](docs/demo_walkthrough.md).

### Run the suite from the command line

```bash
# Record the full experiment suite (video + artifacts + ZIPs)
docker compose run --rm gazebo-demo python -m robot_ci.cli.main record --suite
```

Generated artifacts are written under `artifacts/visual_reports/` and excluded
from Git (see [`.gitignore`](.gitignore)).

---

## Repository layout

| Path | Contents |
|---|---|
| `src/robot_ci/` | Backends (Gazebo, MuJoCo), policies, tasks, recorder, packaging, CLI, dashboard |
| `tests/` | Unit + contract tests (recording, hashes, metadata, partial-failure, status) |
| `scripts/` | Runner entry points and the LinkedIn demo builder |
| `docs/` | Engineering notes (`gazebo.md` — failure-and-fix log), dashboard, recording, and research docs |
| `gazebo/` | Panda URDF and world models |
| `experiments/` | Scenario/condition definitions and recorded evidence under `results/` |
| `dashboard/` | Static research dashboard (`index.html`, `viewer.js`, `serve.py`) |
| `video/` | Recorded demo walkthrough (GitHub release asset; intentionally not committed) |

---

## Status and limitations (stated honestly)

- **What is measured:** the full pick-and-place run above, in Gazebo Harmonic;
  nominal solvability 120/120 on MuJoCo; RQ1–RQ5 findings in `research_report.md`.
- **Screenshot capture:** this environment is headless (no browser / display /
  WebGL), so the six release screenshots were not captured here. See
  [`docs/screenshots/capture.md`](docs/screenshots/capture.md).
- **Camera constraint:** where no camera stream exists for an episode, the 3D
  pane is a kinematic projection of *measured* joint states (not a
  re-simulation), and the camera pane is labelled as a reference still.
  Disclosed per package, never faked.
- **Suite statuses** (PASS / FAIL / NOT_RUN / NOT_SUPPORTED /
  NOT_IMPLEMENTED) are recorded per experiment; categories the backend cannot
  perform are marked, not implied.
- **Report is a snapshot:** `research_report.md` is a narrative synopsis;
  `research_report_generated.md` is the machine-generated authoritative record
  regenerated from `experiments/results/`.

---

## Versions

ROS 2 Jazzy + Gazebo Harmonic, running in Docker (see `docker-compose.yml`).
Full environment details, dependency list, reproducible startup commands, and
the release-commit record are in
[`ENVIRONMENT_REPORT.md`](ENVIRONMENT_REPORT.md).

---

**Automated behavioural regression testing, counterexample discovery and
boundary mapping for physical AI.**

> Treat a robot policy, perception model, planner or inference stack the way you
> treat production software: build it, test it against a baseline, find the exact
> conditions where the new version behaves *worse*, minimise those conditions into
> a reproducible counterexample, and say which subsystem is associated with the
> regression.

ROBOT CI is not a robot-control demo. The primary artifact is a **reproducible
evaluation and verification framework**; the simulator is the laboratory.

