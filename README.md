# ROBOT CI

**Automated behavioural regression testing, counterexample discovery and
boundary mapping for physical AI.**

> Treat a robot policy, perception model, planner or inference stack the way you
> treat production software: build it, test it against a baseline, find the exact
> conditions where the new version behaves *worse*, minimise those conditions into
> a reproducible counterexample, and say which subsystem is associated with the
> regression.

ROBOT CI is not a robot-control demo. The primary artifact is a **reproducible
evaluation and verification framework**; the simulator is the laboratory.

---

## 1. Problem

Aggregate benchmark numbers hide localized failures. A new perception model can
raise overall success from 92% to 94% while collapsing from 82% to 51% once depth
noise exceeds ~8 mm. Engineering teams ship that change because the headline
metric improved.

ROBOT CI makes the hidden regression the *object of study*.

## 2. Research contribution

| Contribution | What it produces |
|---|---|
| Controlled regression testing | Per-condition baseline vs candidate with sample counts and intervals |
| Counterexample discovery | Search for `x` where baseline succeeds and the candidate fails |
| Minimal counterexamples | A portable, replayable YAML artifact reproducing the failure |
| Boundary mapping | Estimated scalar boundary (1-D) and behavioural maps (2-D) |
| Evidence-based diagnosis | Ablations + hedged claims traceable to recorded evidence |
| Reproducible evaluation | Git commit, config hash, seed, backend and simulator version on every result |

## 3. Architecture

```
                 robot_ci.regression / counterexamples / boundary
                 robot_ci.statistics / diagnosis        <-- NO simulator import
                                 |
                          SimulatorBackend (ABC)
                    /            |             \
             MockBackend  SyntheticBackend  MuJoCoBackend     [Gazebo: later]
                    \            |             /
                     +-----------+-------------+
                                 |
              Observation | Action | RobotState | GroundTruthState
                                 |
              sensors (corruption) -> perception -> sensor_fusion
                                 |
                          world_model            (belief, never truth)
                                 |
                      RobotPolicy -> Task -> metrics -> results
                                 |
                    SQLite + JSONL + CSV + HTML artifacts
```

**The research engine never imports MuJoCo.** That is the single most important
architectural rule, and it is what lets regression analysis, search and statistics
run anywhere.

**Ground truth never reaches a policy.** `Observation` has no ground-truth field;
truth is a separate type reachable only by the evaluator. This is enforced by
tests, not convention.

## 4. Why MuJoCo

* Runs on Apple Silicon with **no GPU and no CUDA** (verified: OSMesa/CGL offscreen rendering).
* Fast rigid-body physics with real contact dynamics, friction and grasping.
* Programmatic MJCF generation keeps the scenario YAML the single source of truth.
* Freely available, so results are reproducible by anyone.

Isaac Sim is **not** used. ROS 2 and MoveIt 2 are **optional adapters** (remote
Linux only) and never dependencies of the research engine.

## 5. Quickstart (macOS, CPU only)

```bash
git clone <repo> && cd robot-ci
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

robot-ci init          # create the directory layout
robot-ci validate      # acceptance gate: must exit 0
robot-ci scenario list
robot-ci backend list
```

## 5a. Watch the Panda run in Gazebo, live, in a browser

One command, then one URL. No ROS 2, Gazebo or Python robotics install on
macOS: everything simulation-side runs in Docker on Apple Silicon.

```bash
cd "/Users/shivanibhandari/Downloads/Robotic Project"
docker compose up
```

When the container reports ready, open:

```
http://localhost:8000
```

The startup prints what it has actually observed, and refuses to say "ready"
before the data is real:

```
[robot-ci] Gazebo 8.15.0
[robot-ci] Gazebo ready; Panda loaded with 9 joint controllers
[robot-ci] joint-state stream detected
[robot-ci] camera stream detected
[robot-ci] viewer ready -> http://localhost:8000
```

What the page shows, and where each thing comes from:

| Panel | Source | Live? |
|---|---|---|
| 3D Panda | `/world/robot_ci_pick_place/model/panda/joint_state` | yes — angles are Gazebo's |
| Joint state table (9 joints) | same topic | yes |
| RGB camera | `.../sensor/robot_ci_camera/image` | yes — real frames |
| Status | seed, depth noise, condition, sim time, health | yes |

The 3D arm is built from the joint origins, axes and limits in
`gazebo/models/panda/panda.urdf` (the same URDF Gazebo loads), then posed by
whatever angles Gazebo publishes. If the simulation stops, the arm stops.

**Known gap, stated plainly:** the RGB panel renders the real scene, but the
Panda itself does not appear in the Gazebo camera image. The 3D panel does show
it, moving, from live state. See `docs/gazebo.md` for what was ruled out.

To stop: `Ctrl-C`, then `docker compose down`.


## 6. Status

Each phase below is marked from evidence, not intent: what exists, what was run,
and what was never run. `docs/verification.md` holds the full split.

| Phase | Scope | State |
|---|---|---|
| 0 | Environment probe and report | **done** — `ENVIRONMENT_REPORT.md`, observed on this machine |
| 1 | Schemas, config, CLI, backend interface, Mock/Synthetic/MuJoCo backends, scenarios | **done** |
| 2 | Robot, table, objects, cameras, tasks | **done** — programmatic MJCF; 120/120 nominal |
| 3 | Policies | **done** — `policy_v1` (reference), `policy_v2` (confidence gate, measured inert), `policy_v3` (averaging) |
| 4 | Metrics suite + SQLite store | **done** |
| 5 | Sensor corruption engine | **done** — axes swept in RQ5; four faults recorded as *not effective* rather than approximated |
| 6 | Conditional regression detection | **done** — per-cell rates with intervals; found the project's worst bug |
| 7 | Counterexample search + minimisation | **done** — RQ1 strategies, RQ4 reproduction rate |
| 8 | Behavioural boundary mapping | **done** — RQ5: 1 boundary of 7 axes swept |
| 9 | Failure taxonomy, evidence, ablation, diagnosis | **done** — RQ3, scored against injected ground truth |
| 10 | ROS 2 interfaces (**remote only**) | **not started** — `ros2_ws/` holds no files |
| 11 | MoveIt 2 planner (**remote only**) | **not started** |
| 12 | Dashboard | **not started** — `dashboard/` holds no files; `robot-ci dashboard` exists as a stub |
| 13 | Colab/Kaggle runners, export/import | **not started** — `notebooks/` is empty |
| 14 | Benchmark suite + RQ1–RQ5 | **done** — 20 benchmarks, 0 expectation mismatches; RQ2's strong form demonstrated |
| 15 | Verification path | **`verify.sh` verified locally; the Linux workflow has never run** — `.github/workflows/verify.yml` |

## 5b. Reproducing this

```bash
bash scripts/verify.sh            # one definition of "verified", ~12 min
bash scripts/verify.sh --quick    # skip the experiments, ~1 min
echo $?                           # 0 only if every check exited 0
```

`scripts/verify.sh` is the single entry point, shared with CI. It checks **exit
codes, not output text**, and writes a machine-readable record to
`experiments/results/verification_baseline.json`.

That rule was not a style preference. For several rounds this project reported
"ruff clean" while a `B905` violation sat above the summary line, because the
checks were piped through `tail -n 1`, which prints "All checks passed!" as the
last line of a run that also printed a violation. Writing the script then found a
**second** hole on its first run: `ruff format` had never been run at all, and
reported 40 files. Two independent gaps in the checker, one invisible to the tool
that had been run every time.

This is the project's thesis turned on itself: a mechanism that looks correct and
does less than it appears to is the failure mode this project exists to detect,
and it was present in the verification process before it was present in the robot.

## 7. Documentation

* [`research_report_generated.md`](research_report_generated.md) - the research
  report, generated from the artifacts by
  `experiments/make_research_report.py`. No number in it is transcribed.
* [`docs/verification.md`](docs/verification.md) - what is verified locally and
  what is not, including the two failures of the verification process itself
* `ENVIRONMENT_REPORT.md` - measured machine capabilities and the MuJoCo finding
* `IMPLEMENTATION_PLAN.md` - phase map, acceptance criteria, risk register
* [`docs/`](docs/README.md) - architecture, backends, methodology and
  troubleshooting. Every page carries an evidence label: **Verified locally**,
  **Implemented but unverified**, or **Unavailable on current macOS**

## 8. Limitations (stated honestly)

* The Phase 1 arm is a DH-derived kinematic approximation of a Panda, not the
  official MJCF. This is documented in the builder and replaced in Phase 2.
* MuJoCo on macOS uses software OpenGL. At 160x120 it renders in ~6.5 ms, which
  is adequate for batch sweeps but not for photorealism.
* The Mock and Synthetic backends do not model contact dynamics; manipulation
  claims must come from `--backend mujoco`.
* ROS 2 / MoveIt 2 cannot be validated on macOS and are not yet built.
  `ros2_ws/src/` holds three **empty directories** — 0 files. No scaffolding was
  written to make the tree look more complete than it is.
* The Isaac Sim adapter is **implemented but unverified**. It has never been
  executed, because Isaac has no macOS build. Its contract is tested against
  mocks; the live `IsaacSession` is **not implemented**.
* `.github/workflows/verify.yml` has **never run**. It is the path toward Linux
  verification, not a verification.
* `ruff format` reports 40 files it would restyle. `ruff check` (the linter)
  passes on all of them; `ruff format` is a different tool that had not been run
  until `scripts/verify.sh` was written.

## 9. License

MIT - see `LICENSE`.
