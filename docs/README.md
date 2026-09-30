# Documentation

Every document here carries an evidence label. The labels are not decoration:
this project makes claims about what it measured, and a reader needs to know
which parts of a page are measurement and which are intent.

| Label | Meaning |
|---|---|
| **Verified locally** | Run on this machine (macOS, CPU) and the output is reproducible with the command shown. |
| **Implemented but unverified** | Code exists and its contract is tested, but it has not been executed against the real system it targets. |
| **Unavailable on current macOS** | The target system cannot run here at all, so nothing about it has been executed. |

A claim without one of these labels should be treated as unproven.

## Documents

| File | Covers | Status |
|---|---|---|
| [architecture.md](architecture.md) | Layers, the backend boundary, data flow, why the seams are where they are | Verified locally |
| [mujoco.md](mujoco.md) | The MuJoCo backend, embodiment, task, measured behaviour and limits | Verified locally |
| [isaac.md](isaac.md) | The Isaac Sim adapter, its contract, and what is *not* proven | Implemented but unverified |
| [ros2.md](ros2.md) | Intended ROS 2 architecture and the current state of the workspace | Unavailable on current macOS |
| [moveit.md](moveit.md) | Intended MoveIt 2 / ros2_control integration | Unavailable on current macOS |
| [experiments.md](experiments.md) | Experiment config, runner, artifacts, reproducibility | Verified locally |
| [regression.md](regression.md) | Conditional detection, statistics, thresholds, severity | Verified locally |
| [counterexamples.md](counterexamples.md) | Search strategies, minimisation, reproduction | Verified locally |
| [boundaries.md](boundaries.md) | Perturbation boundary mapping and what it found | Verified locally |
| [diagnosis.md](diagnosis.md) | Failure taxonomy, ablation, hedged attribution | Verified locally |
| [evaluation.md](evaluation.md) | The benchmark suite, its expectations, and the false-positive problem | Verified locally |
| [cloud.md](cloud.md) | Running on Colab/Kaggle/Azure, and what moves to the GPU host | Partly verified, partly not |
| [troubleshooting.md](troubleshooting.md) | Failure modes actually hit while building this, and their fixes | Verified locally |

## Where the numbers live

* [`../experiments/results/README.md`](../experiments/results/README.md) — every measured result
* [`../experiments/results/DEMO.md`](../experiments/results/DEMO.md) — an end-to-end run, including one that found nothing
* [`../research_report.md`](../research_report.md) — the research questions, results and limitations
* [`../ENVIRONMENT_REPORT.md`](../ENVIRONMENT_REPORT.md) — the machine this was built on

## A note on what "verified" covers

The most recent change to the measurement stack was commit `dc61715`, which
tightened the IK tolerance and altered task behaviour. Every research result was
re-run afterwards. If you find a result in a document that contradicts
`experiments/results/README.md`, the results file is the authority and the
document is stale.
