# ROBOT CI visual experiment suite

Status counts: FAIL=8, NOT_IMPLEMENTED=2

Only conditions that actually ran have a package. Everything else is
listed under **Not available** with the reason it did not run.

| Experiment | Condition | Status | Seed | Steps | task_success | Joint travel (rad) | Camera frames | Video |
|---|---|---|---|---|---|---|---|---|
| `nominal_baseline` | sensors=nominal | **FAIL** | 3 | 300 | 0 | 0.9230 | 300 | yes |
| `depth_noise_0mm` | depth.sigma_mm=0 | **FAIL** | 3 | 300 | 0 | 1.1747 | 300 | yes |
| `depth_noise_4mm` | depth.sigma_mm=4 | **FAIL** | 3 | 300 | 0 | 0.9039 | 300 | yes |
| `depth_noise_8mm` | depth.sigma_mm=8 | **FAIL** | 3 | 300 | 0 | 2.0758 | 300 | yes |
| `object_perturbation` | object.x += 0.06 | **FAIL** | 3 | 300 | 0 | 0.7962 | 300 | yes |
| `joint_perturbation` | joint_noise_std_rad=0.005 | **FAIL** | 3 | 300 | 0 | 1.0884 | 300 | yes |
| `baseline_vs_candidate` | policy=policy_v3 | **FAIL** | 3 | 300 | 0 | 2.0756 | 300 | yes |
| `repeated_seed` | seeds=3,11,42 | **FAIL** | 3 | 300 | 0 | 1.0502 | 300 | yes |
| `counterexample_reproduction` | n/a | **NOT_IMPLEMENTED** | - | 0 | - | 0.0000 | 0 | no |
| `boundary_search` | n/a | **NOT_IMPLEMENTED** | - | 0 | - | 0.0000 | 0 | no |

## Not available

| Experiment | Status | Reason |
|---|---|---|
| `counterexample_reproduction` | NOT_IMPLEMENTED | No counterexample has been discovered and stored for this backend, so there is nothing to replay. ROBOT CI will not invent one. |
| `boundary_search` | NOT_IMPLEMENTED | Boundary search needs a real parameter sweep driving the backend. This suite runs a fixed list of conditions, so no search is claimed. |
