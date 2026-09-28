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

## 6. Status

| Phase | Scope | State |
|---|---|---|
| 0 | Environment probe and report | **done** - see `ENVIRONMENT_REPORT.md` |
| 1 | Schemas, config, CLI, backend interface, Mock/Synthetic/MuJoCo backends, scenarios | **done** |
| 2 | Robot, table, objects, cameras, tasks | planned |
| 3 | Policies | planned |
| 4 | Metrics | planned |
| 5 | Sensor corruption | planned |
| 6 | Regression detection | planned |
| 7 | Counterexamples | planned |
| 8 | Boundary mapping | planned |
| 9 | Diagnosis and ablation | planned |
| 10-11 | ROS 2 / MoveIt 2 (remote only) | planned |
| 12-14 | Dashboard, cloud, benchmarks, research report | planned |

**No experimental results are reported yet.** Commands for later phases exit with
code 5 and say so; they never return fabricated numbers.

## 7. Documentation

* `ENVIRONMENT_REPORT.md` - measured machine capabilities and the MuJoCo finding
* `IMPLEMENTATION_PLAN.md` - phase map, acceptance criteria, risk register
* `docs/` - architecture, experiment design, and phase guides (in progress)

## 8. Limitations (stated honestly)

* The Phase 1 arm is a DH-derived kinematic approximation of a Panda, not the
  official MJCF. This is documented in the builder and replaced in Phase 2.
* MuJoCo on macOS uses software OpenGL. At 160x120 it renders in ~6.5 ms, which
  is adequate for batch sweeps but not for photorealism.
* The Mock and Synthetic backends do not model contact dynamics; manipulation
  claims must come from `--backend mujoco`.
* ROS 2 / MoveIt 2 cannot be validated on macOS and are not yet built.

## 9. License

MIT - see `LICENSE`.
