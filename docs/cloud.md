# Cloud and remote execution

**Evidence: Partly verified, partly not.** The local half of this page is
verified on this machine. Nothing about Colab, Kaggle, Azure or the GPU host has
been run — no notebook exists (`notebooks/` is empty) and no remote host has been
used.

The one thing that matters most here is a fact about the *architecture*, and it
is verified: **the whole research engine runs without ROS 2, without Isaac, and
without a GPU.** Every result in this repository was produced on a laptop CPU.
Cloud is for the things that genuinely need it, not for the research loop.

## What actually needs a GPU

| Work | Needs a GPU? | Why |
|---|---|---|
| Regression detection, statistics, reports | **no** | numpy and stdlib |
| Counterexample search | **no** | it calls the MuJoCo backend |
| Boundary sweeps, benchmark suite | **no** | same |
| Ablation, diagnosis, taxonomy | **no** | same |
| MuJoCo episodes | **no** | CPU physics |
| Isaac Sim rendering | **yes** | photorealistic rasterisation |
| MoveIt 2 with a physics backend | usually | not required by this project |

So: move the *experiment* to the cloud only if the experiment involves Isaac, or
if the episode count has grown past what a laptop will finish in a working day.
A 60-episode run takes 72.8 s here.

## What has not been done

* `notebooks/` is **empty** — 0 files. The specification asks for five
  (data analysis, regression analysis, counterexample analysis, sensor ablation,
  policy comparison). None exists.
* `robot-ci export` is still a reserved stub, so there is no packaging path to
  Colab or Kaggle yet.
* No remote GPU host has been used, and no SSH workflow has been tested.

## The intended split, and why the Mac half comes first

```
MacBook (verified)                     Linux GPU host (unverified)
──────────────────                     ──────────────────────
experiment design                      Isaac Sim
scenario authoring                     ROS 2 / MoveIt 2
regression analysis                    RGB/depth generation
report generation                      large-scale sweeps
```

Results cross between the two as **artifacts**, not as objects. Every run writes
`config.yaml`, `summary.json`, `metrics.csv`, `counterexamples.json` and a
git commit. A remote result is analysed on the Mac by reading the artifact, which
is why `reporting` depends on nothing but the filesystem.

## Before using the cloud

1. **Decide the backend.** MuJoCo on a cloud CPU is usually enough and costs
   nothing. Isaac needs a GPU with a large VRAM budget and a Linux host, and the
   adapter for it is [implemented but unverified](isaac.md) — so the first remote
   run is likely to be spent on the adapter, not on results.
2. **Make the config portable.** Absolute paths are not supported anywhere in the
   project; scenarios are resolved relative to the working directory or given
   explicitly.
3. **Keep the seed derivation.** Seeds derive from
   `(experiment_seed, scenario_id, replicate)`, so a result computed on one
   machine is the same result on another, given the same config and code.

## Colab and Kaggle specifically

Both are Linux CPU environments with generous RAM and no GPU guaranteed. They fit
the "analysis and statistics" half of this page, not the physics half:

```bash
pip install -e .          # the package has no compiled extensions
```

The package is pure Python with numpy, pydantic, typer and rich, so there is no
build step to fail. Note that `mujoco` must be installed separately; the
`mock` and `synthetic` backends work without it, which is how the unit suite runs
anywhere.

**None of this has been executed.** It is the reasoning, not a report.
