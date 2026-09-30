# Architecture

**Evidence: Verified locally.** Every claim here is checkable with
`robot-ci validate` and the test suite on this machine.

## The one boundary that matters

Everything above `robot_ci.simulation.base` imports **only** from it. That
module is the whole contract between the research engine and any physics engine:

```
      policies        regression      counterexamples     diagnosis
          \                 |                 |                /
           \                |                 |               /
            +---- import only from SimulatorBackend ----+
                               |
     MockBackend  SyntheticBackend  MuJoCoBackend  IsaacSimBackend
          |              |               |              |
     fake states   analytic rays   contact dynamics   Omniverse prims
```

The consequence that matters in practice: `import robot_ci.regression` does not
import MuJoCo, and `IsaacSimBackend` is a registered name whose module has no
Isaac import at module scope. Both were verified by running them here, on a
machine with no Omniverse and no ROS 2.

## Layers

| Layer | Package | Responsibility | Depends on |
|---|---|---|---|
| Core | `core` | Errors, logging, seeding, config hashing, workspace | nothing |
| Models | `models` | Versioned scenario/robot/environment/geometry/state types | `core` |
| Scenarios | `scenarios` | Catalogue of families, grid expansion, path setting | `models` |
| Simulation | `simulation` | The backend interface and its implementations | `models` |
| Sensors | `sensors` | Deterministic corruption of observations | `models` |
| Tasks | `tasks` | Task interface, success criteria, waypoint generation | `models` |
| Planning | `planning` | IK, trajectory helpers | `models`, `core` |
| Control | `control` | Gripper command shaping | `models` |
| Perception | `perception` | Depth-frame object pose with propagated uncertainty | `models` |
| Sensor fusion | `sensor_fusion` | Kalman filter over repeated object estimates | `models` |
| World model | `world_model` | Structured beliefs with reconstructible snapshots | `models` |
| Policies | `policies` | The policy interface and the concrete policies | `models`, `tasks`, `planning` |
| Metrics | `metrics` | Metric definitions and aggregation | `models` |
| Experiments | `experiments` | Runner, orchestration, artifacts, pairing | most of the above |
| Statistics | `statistics` | Wilson, bootstrap, Cohen's h, Welch | `core` |
| Regression | `regression` | Conditional comparison, thresholds, matrix | `statistics`, `experiments` |
| Counterexamples | `counterexamples` | Search strategies, oracle, artifacts, replay | `regression`, `experiments` |
| Diagnosis | `diagnosis` | Failure taxonomy, ablation, hedged attribution | `statistics` |
| Evaluation | `evaluation` | Benchmark suite, CI gate | `experiments`, `regression` |
| Reporting | `reporting` | Self-contained HTML | none (reads artifacts) |
| CLI | `cli` | Every command | all of the above |

Dependencies point strictly downward. `reporting` is at the bottom of the
dependency graph despite being near the top of the user-facing one, because it
reads artifacts from disk rather than importing the engine.

## Determinism, and where it stops

`reset(scenario, seed)` fully determines an episode. Everything random derives
from `(scenario.seed, scenario_id, purpose)` through `core.seeding`, never from a
global RNG.

**Verified locally** for the mock, synthetic and MuJoCo backends.

**It does not hold for Isaac Sim**, which is non-deterministic by default; the
backend reports `deterministic=False` and its `capabilities().notes` says so. A
seed fixes the scene, not the trajectory, so cross-run variance has to be carried
rather than assumed away.

## Ground-truth separation

`observe()` returns an `Observation` with no ground truth. Ground truth is
retrieved through a separate `ground_truth()` method that the policy layer never
calls. This split is what makes perception evaluation honest: a policy that
appears to localise an object well cannot be reading the answer.

Verified by the observation/ground-truth tests and by the fusion work, where the
world model records ground truth separately and exposes it only through
`belief_error()`.

## Data flow of one episode

```
scenario.yaml ─┐
               ├─► orchestrator ──► run_episode ──► backend.reset()
seeds          ┘        │                              │
                        │                     per control tick:
                        │                       observe()   ──► corruption ──► policy.act()
                        │                       step()      ──► transition
                        │                       task.evaluate()
                        ▼
                   artifacts: summary.json, metrics.csv, events.jsonl,
                               counterexamples.json, regression.json, report.html
```

Corruption is applied to the **observation**, inside `run_episode`, not inside
the backend. The backend therefore never knows it is being perturbed, which is
what lets a benchmark sweep corruption against a backend that has no concept of
it.

## Artifact and authority order

When two files disagree, this is the order of authority:

1. `experiments/results/*.json` — written by the run that produced them
2. `experiments/results/README.md` — the curated record, with staleness notices
3. `research_report.md` — the narrative
4. `docs/` — the explanation, and the first thing to suspect when one of the
   above disagrees with it
