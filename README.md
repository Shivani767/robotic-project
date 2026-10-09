# ROBOT CI

**Robot manipulation experiments, measured on real simulators — recorded, packaged, and reproducible.**

ROBOT CI runs a Panda arm through pick-and-place tasks in a physics
simulator, records every tick of real joint state, and turns each run into a
synchronized video, figures, a JSON results file, and a downloadable ZIP. The
core idea is **evidence over assertion**: every claim the platform makes —
success, placement error, collisions — is measured from simulator telemetry and
hashed into a manifest, never narrated. A run that fails is reported as failed.

## What it does

1. **Control a Panda arm** in Gazebo Harmonic (and MuJoCo) through inverse
   kinematics, with real joint-state feedback each control tick.
2. **Record each experiment**: actual joint states, simulation time, task
   events, metrics, success/failure, and runtime versions — timestamped.
3. **Render a video** of the run: a live-joint-state 3D view of the Panda
   combined with the actual Gazebo camera frame in a synchronized split screen.
4. **Package the evidence**: MP4 + PNG figures + `results.json` +
   `manifest.json` (SHA-256 of every artifact) + logs + telemetry, zipped per
   experiment.
5. **Run a suite** of experiment categories (baseline, perturbations,
   repeated seeds) with honest PASS / FAIL / NOT_RUN / NOT_SUPPORTED /
   NOT_IMPLEMENTED status.

No trajectories are fabricated or replayed. If a source is unavailable (for
example, no camera stream for a given episode), it is marked unavailable — never
manufactured.

## Measured outcome (the headline result)

The verified Part 1 integration achieves a **complete physical pick-and-place
in Gazebo** — the arm approaches, grasps, lifts 65 mm, transports, and places
the object **22 mm from the target** (50 mm tolerance):

| Metric | Measured |
|---|---|
| Task success | **true** (grasp + lift + placement all pass) |
| Object lift | 0.0657 m (≥ 0.05 required) |
| Placement error | 0.0221 m (≤ 0.05 tolerance) |
| Ticks with the object grasped | 45 / 115 |

This is evidence from a real Gazebo run, not a claim. See
[`docs/gazebo.md`](docs/gazebo.md) for the engineering that got there, and
[`experiments/results/DEMO.md`](experiments/results/DEMO.md) for how to
reproduce it.

## Quick start

The platform runs on macOS through Docker Compose (ARM64-compatible).

```bash
docker compose up --build
```

Then open the dashboard at **http://localhost:8000** (WebSocket on 8001). From
the browser you can select an experiment category and condition, set seed /
policy / repeats, start or stop a run, watch live joint states, and download the
generated video, images, JSON results, and per-experiment ZIP.

### Run the suite from the command line

```bash
# Record the full experiment suite (video + artifacts + ZIPs)
docker compose run --rm gazebo-demo python -m robot_ci.cli.main record --suite
```

Generated artifacts are written under `artifacts/visual_reports/` and excluded
from Git (see [`.gitignore`](.gitignore)).

## Repository layout

| Path | Contents |
|---|---|
| `src/robot_ci/` | Backends (Gazebo, MuJoCo), policies, tasks, recorder, packaging, CLI, dashboard |
| `tests/` | Unit + contract tests (recording, hashes, metadata, partial-failure, status) |
| `scripts/` | Runner entry points and the LinkedIn demo builder |
| `docs/` | Engineering notes (`gazebo.md` — the failure-and-fix log), dashboard and recording docs |
| `gazebo/` | Panda URDF and world models |
| `experiments/` | Scenario/condition definitions and recorded evidence under `results/` |

## Status and limitations (stated honestly)

- **What is measured:** the full pick-and-place run above, in Gazebo Harmonic.
- **Known limitation:** the Pandas's absence from the Gazebo camera is a real
  constraint. Where no camera stream exists for an episode, the 3D pane is a
  kinematic projection of *measured* joint states (not a re-simulation), and the
  camera pane is labelled as a reference still. This is disclosed in each
  package's `README.md`, never faked into the live stream.
- **Suite statuses** (PASS / FAIL / NOT_RUN / NOT_SUPPORTED / NOT_IMPLEMENTED)
  are recorded per experiment in the suite index. Categories the backend cannot
  perform are marked, not implied.

## Versions

ROS 2 Jazzy + Gazebo Harmonic, running in Docker (see `docker-compose.yml`).
Full environment details are in [`ENVIRONMENT_REPORT.md`](ENVIRONMENT_REPORT.md);
the curated research write-up is
[`experiments/research_report.md`](experiments/research_report.md).


**Automated behavioural regression testing, counterexample discovery and
boundary mapping for physical AI.**

> Treat a robot policy, perception model, planner or inference stack the way you
> treat production software: build it, test it against a baseline, find the exact
> conditions where the new version behaves *worse*, minimise those conditions into
> a reproducible counterexample, and say which subsystem is associated with the
> regression.

ROBOT CI is not a robot-control demo. The primary artifact is a **reproducible
evaluation and verification framework**; the simulator is the laboratory.

