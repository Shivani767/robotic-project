# ROBOT CI — IMPLEMENTATION PLAN

Derived from `ENVIRONMENT_REPORT.md` (Phase 0, revised for **MuJoCo**).
Each phase ends with `implement -> test -> run -> inspect -> document -> commit`.

## Ground rules

1. **No fabricated results.** Every number in the docs is produced by a command in this
   repo. Unrun experiments are labelled `NOT RUN`.
2. **The research engine never imports MuJoCo.** It talks to `SimulatorBackend`.
   `robot_ci.regression`, `robot_ci.counterexamples`, `robot_ci.boundary`,
   `robot_ci.statistics` have zero simulator dependencies.
3. **Mac stays first-class.** `pytest` and `robot-ci test --backend mujoco` must pass with
   CPU only, no cloud, no ROS 2.
4. **No ground-truth leakage.** Policies and perception receive `Observation` only.
   `GroundTruthState` is a separate type, reachable only by the evaluator. Enforced by
   tests, not by convention.
5. **Determinism.** An episode is a pure function of
   `(scenario, policy, seed, backend_version)`.
6. **Small conventional commits.** One concern per commit.

## Architecture boundary (the whole point)

```
robot_ci.regression / counterexamples / boundary / statistics / diagnosis
        |  depends on models + metrics ONLY  (no mujoco import)
        v
SimulatorBackend (ABC)  <- robot_ci.simulation.base
   |            |              |
MockBackend  SyntheticBackend  MuJoCoBackend     [GazeboBackend: later, optional]
   |            |              |
   +------------+--------------+
                 v
      Observation | Action | RobotState | GroundTruthState
                 v
      sensors (corruption) -> perception -> sensor_fusion
                 v
          world_model  (estimated state; never ground truth)
                 v
            RobotPolicy -> Task -> metrics -> results
```

## Phase map

| Phase | Scope | Acceptance |
|---|---|---|
| **0** | environment probe, `ENVIRONMENT_REPORT.md` | `scripts/probe_environment.py` exits 0 |
| **1** | schemas, config, CLI, `SimulatorBackend`, `MockBackend`, `SyntheticBackend`, `MuJoCoBackend`, scenarios | `robot-ci validate` exits 0; pytest green; real MuJoCo frame produced |
| 2 | Panda arm, table, objects, cameras, target region in MJCF | `robot-ci experiment run --backend mujoco` executes episodes |
| 3 | `RobotPolicy` + `BaselinePolicy`/`CandidatePolicy`/`RuleBasedPolicy`/`MockPolicy`, task execution | policies run under identical conditions |
| 4 | metrics suite + SQLite store | metrics computed and persisted |
| 5 | sensor corruption engine (RGB/depth/camera/robot-state) | corruption deterministic under seed |
| 6 | conditional regression detection + regression matrix | injected regression detected with evidence |
| 7 | counterexample search (random/grid/binary) + minimization | minimal counterexample artifact + reproduce |
| 8 | behavioral boundary mapping (1-D binary, 2-D map, adaptive) | boundary estimate stable across seeds |
| 9 | failure taxonomy, evidence, ablation, diagnosis | diagnosis traceable to evidence |
| 10 | ROS 2 interfaces, TF2, robot-state bridge (**remote only**) | colcon build on remote Linux |
| 11 | `MotionPlanner` interface, `SimplePlanner`, `MoveItPlanner` (**remote only**) | remote build |
| 12 | React+TS dashboard | `npm run build` passes; renders real artifacts only |
| 13 | Colab/Kaggle runners, experiment export/import | notebook executes end to end |
| 14 | benchmark suite B1-B10 + baselines + RQ1-RQ5 | real numbers, no fabrication |
| 15 | `research_report.md` from executed artifacts only | no `NOT RUN` cells left |

## Phase 1 detail (current phase)

1. `pyproject.toml` — hatchling, `src/` layout, `ruff`/`mypy`/`pytest`, `robot-ci` script.
2. `robot_ci.models` — versioned, frozen, `extra="forbid"` Pydantic v2 schemas.
3. `robot_ci.core` — errors, BLAKE2b seed derivation, canonical-JSON hashing, JSONL,
   YAML I/O, workspace, versions.
4. `robot_ci.simulation.base` — `SimulatorBackend` ABC + registry with lazy imports.
5. `robot_ci.simulation.mock` — analytic, no physics, for unit tests.
6. `robot_ci.simulation.synthetic` — NumPy ray-cast depth + shaded RGB.
7. `robot_ci.simulation.mujoco` — programmatic MJCF, `MUJOCO_GL=osmesa`, RGB/depth/seg.
8. `robot_ci.scenarios` — YAML loader, catalogue, grid expander.
9. `robot_ci.cli` — Typer; later commands honestly report `not implemented - Phase N`.
10. Tests: unit + integration, including a real MuJoCo render assertion.

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| OSMesa software rendering is slow | Medium | small default resolution, configurable, capped batch workers |
| Ground-truth leakage invalidates results | **Critical** | separate types + dedicated leak tests |
| Statistical over-claiming | High | n, CI and test name on every claim; hedged language in diagnosis |
| Simulator realism gap | High | stated limitation; shared observation contract across backends |
| Scope explosion | High | phase gates; no phase starts before the previous acceptance criterion |
