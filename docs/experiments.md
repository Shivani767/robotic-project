# Experiments

**Evidence: Verified locally.** Every command here has been run on this machine.

## What an experiment is

One two-arm comparison over a grid of conditions, with both arms seeing
byte-identical scenarios at every cell and seed. That pairing guarantee is the
whole point: a difference between the arms is then attributable to the policy
and not to an environment draw.

## Config

```yaml
experiment:
  id: demo_v1_vs_v3
  max_steps: 200          # a successful episode needs 76-77 ticks
backend: mujoco
baseline:  { policy: policy_v3 }
candidate: { policy: policy_v1 }
scenario:
  task: { name: pick_and_place, max_steps: 200 }
variables:
  depth_noise_mm: { values: [0, 1, 2, 4, 8] }
repetitions: 6
seed: 3
```

```bash
robot-ci experiment run experiments/configs/demo_v1_vs_v3.yaml
```

`max_steps` is worth attention. The default catalogue value of 300 is safe; a
shorter one truncates the episodes under comparison and scores the resulting
timeouts as failures. 90 was used in an early experiment and did exactly that.

## Seeds

The runner **derives** each seed rather than counting up:

```python
derive_seed(experiment_seed, cell.scenario_id, replicate)
```

Consequences worth knowing:

* The same experiment always produces the same seeds — reproducibility is not a
  separate mechanism.
* Seeds are large hashed integers, and they are **not** the same as a sweep of
  small integers. The demo experiment derives seeds under which the v1/v3
  separation does not appear at all, because those object placements are
  unsolvable. Every measurement in `experiments/results/` states its seed list.
* If you need a specific seed, write a scenario file for it and pass
  `--scenario`, as the counterexample harness does.

## Artifacts

Every run writes to `experiments/results/<experiment_id>/`:

| File | Contents |
|---|---|
| `config.yaml` | The exact configuration, as run |
| `summary.json` | Per-arm rates, statuses, wall time, git commit, config hash |
| `metrics.csv` | Tidy long table: one row per (run, metric) |
| `events.jsonl` | Per-tick event stream |
| `counterexamples.json` | Any found, with reproduction data |
| `regression.json` | Per-condition comparison, written by `robot-ci regression` |
| `report.html` | Self-contained report, written by `robot-ci report` |

## Reproducibility record

`summary.json` carries the git commit and a `config_hash` (`sha256:` over the
scenario's experimental conditions, excluding `schema_version` so a compatible
schema bump does not invalidate history). A result without those two cannot be
traced back and should not be quoted.

## Worked example, including a null

```bash
robot-ci experiment run experiments/configs/demo_v1_vs_v3.yaml
robot-ci regression experiments/results/demo_v1_vs_v3
```

produces:

```
 condition                  baseline  candidate   delta  severity
 nominal__depth_noise_mm=0     0.500      0.500  +0.000  none
 nominal__depth_noise_mm=1     0.000      0.000  +0.000  none
 nominal__depth_noise_mm=2     0.000      0.000  +0.000  none
```

**No regression, and that is the correct answer.** Half the cells are
unsolvable at nominal and every one is unsolvable at 1 mm of depth noise, so both
arms are pinned at the floor and there is no separation left to detect. A
conditional analysis that reported something here would be worse than useless.

The full transcript is in `experiments/results/DEMO.md`.

## Cost

60 episodes took 72.8 s on this machine with `policy_v1`/`policy_v3`. The grid
scales as `cells x repetitions x 2`; the benchmark suite and the ablation are
the expensive ones, and both accept fewer seeds when the question does not need
them.
