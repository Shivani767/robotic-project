# Phase 3 — Policy family and execution under identical conditions

**Date:** 2026-09-28

## Scope

Phase 3 is the `RobotPolicy` family — baseline, candidate, rule-based, mock — and
the guarantee that they can be compared. The acceptance criterion is narrow and
is about the *harness*, not about finding a winner:

> policies run under identical conditions

Two defects made that guarantee false. Both were silent.

## Defect 1: the two arms never saw the same seed

`build_run_specs` derived its seed as

```python
derive_seed(config.seed, cell.scenario_id, policy.name, replicate)
```

The policy name is an input to the derivation, so the baseline and the candidate
received **different random draws at the same cell**. The docstring on the same
function claimed the seed was "identical for both policies at the same cell", and
`ExperimentConfig`'s own docstring claimed the arms run under "byte-identical
conditions". The code contradicted both.

This is not a cosmetic mismatch. It confounds the policy effect with the seed
effect, so a difference between the two arms can no longer be attributed to the
policy: the arms were never run under the same conditions. Nothing in the output
reveals it — both arms complete, both produce full metric vectors, and the
inflation shows up only as extra variance that a reader would reasonably blame on
the candidate.

Measured before the fix, for a 2-cell grid at 3 replicates: every
`(cell, seed)` group contained exactly one policy, and no cell had a seed shared
across arms. After the fix each `(cell, replicate)` has one seed, and its group
contains both policies.

## Defect 2: naming a policy did not select that policy

An experiment file names its arms:

```yaml
baseline:  {policy: policy_v1}
candidate: {policy: policy_v2}
```

Only the *name* survives the file parse. `PolicyConfig.implementation` defaults
to `"rule_based"`, the registry maps `"rule_based"` to the `policy_v1` factory,
and the factories apply their distinguishing parameters only when handed `None`.
So the candidate arm instantiated a policy that was **behaviourally policy_v1,
under a policy_v2 label**.

Measured, before the fix:

```
baseline  sigma=1000000000.0 clearance=0.0
candidate sigma=1000000000.0 clearance=0.0
```

Identical. The experiment would have run eight episodes, produced a complete
table, reported "no difference between policy_v1 and policy_v2", and been read as
evidence that the candidate does not help. The candidate was never run. Of
everything in this phase this is the most dangerous, because the failure mode is
an authoritative-looking null result.

Fixed with `resolve_policy_config`, which takes the named policy's defaults as a
base and lets anything set explicitly win. Naming a policy now means getting that
policy:

```
baseline  sigma=1000000000.0 clearance=0.00  components=fusion_v1
candidate sigma=0.05         clearance=0.01  components=fusion_v1_gated
```

## The run

```
robot-ci experiment run experiments/configs/phase3_pairing.yaml
  backend=mujoco, 2 cells x 2 reps = 8 episodes
```

Artifacts in `experiments/results/phase3_pairing/`: `config.yaml`,
`events.jsonl`, `metrics.csv`, `summary.json`, `report.html`,
`counterexamples.json`.

The pairing is verified **in the artifacts**, not just in the unit tests: each
cell carries exactly two seeds (one per replicate), each seed is used by both
policies, and each cell resolves to a single `config_hash`.

```
task_success   v1 0.0000   v2 0.0000
grasp_success  v1 1.0000   v2 1.0000
collision      v1 0.0000   v2 0.0000
```

The strongest evidence that the conditions really are identical is that the two
arms agree on continuous metrics to floating-point precision. For one cell:

```
grasp_error_m   v1 0.0057779037576823015
                v2 0.005777903757682151
```

The arms differ in the last four digits. They followed the same trajectory.

## Why the comparison is still uninformative — and it is not a bug

`task_success` is `0.000` for both arms. That is the honest result, and there
are two independent reasons, both of which have to be stated rather than hidden
behind a table:

1. **The task does not succeed for either policy.** As recorded in
   `PHASE2_STATUS.md`, the grasp is made and observed but the pick-and-place does
   not complete on most seeds. A comparison on a task that never succeeds
   measures nothing, which is why this experiment must not be read as a result
   about the policies even though it produced a difference table.

2. **The grid does not exercise the candidate's change.** `policy_v2`'s only
   behavioural difference is a confidence gate at `position_sigma > 0.05`. The
   measured `pose_sigma_m` is `0.002` in every run of this grid, so the gate
   never fires and the two policies are *expected* to behave identically. A null
   result here is a statement about the grid, not about the gate.

Exercising the gate needs conditions that actually raise pose uncertainty. The
depth-noise values in this grid (0 and 5 mm) are too small. That is Phase 5
territory — the corruption engine — and it is the natural place to look for a
conditional regression.

## What Phase 3 delivers

| Criterion | State |
|---|---|
| Policy family present (baseline, candidate, rule-based, mock) | **met** |
| Policies run under identical conditions | **met, and now enforced by test** |
| Identical conditions verified in produced artifacts | **met** |
| A measurable policy difference | **not met — and not reachable on this grid** |

Twelve tests were added: five pinning the seed/scenario pairing, seven pinning
policy resolution. The pairing test is checked to discriminate — under the old
derivation it produces six unpaired groups, under the fix three paired ones.

## Recommended next step

Do not run a wider Phase 3 grid. Raising the grid multiplies episodes without
changing either reason above. The order that actually unblocks progress:

1. Make the task succeed (`PHASE2_STATUS.md`: contact stiffness, not friction —
   friction has now been eliminated).
2. Then re-run this same two-cell experiment, where a difference would mean
   something.
3. Only then go to Phase 5 corruption, where the candidate's gate has a reason
   to fire.
