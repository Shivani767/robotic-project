# PHASE 2 STATUS

**Date:** 2026-09-28 · **Branch:** `main`

This file records what Phase 2 delivers, what is verified, and what is **not**
working. Read section 4 before trusting any manipulation result.

---

## 1. Built and verified

| Component | State | Evidence |
|---|---|---|
| `SimulatorBackend` ABC + lazy registry | working | 194 tests; contract tests run against every backend |
| `MockBackend` | working | deterministic, seed-sensitive, no ground-truth leakage |
| `SyntheticBackend` | working | NumPy ray-cast depth + shaded RGB |
| `MuJoCoBackend` | working | physics, RGB/depth/seg, `cgl` offscreen rendering |
| `Task` interface | working | `pick_and_place`, `grasp_object`, `push_object`; other names raise `NotImplementedError` |
| `RobotPolicy` interface | working | `policy_v1`, `policy_v2`, `MockPolicy` |
| Metric suite | working | 20 canonical metrics, each with unit, kind and source |
| `ExperimentRunner` | working | `robot-ci experiment run` on mock / synthetic / mujoco |
| Artifact writing | working | `config.yaml`, `summary.json`, `metrics.csv`, `events.jsonl`, `report.html`, `counterexamples.json` |
| Damped-least-squares IK | **solver verified** | Jacobian matches finite differences to `3e-07` |
| Official Franka Panda MJCF | vendored | Apache-2.0, from `mujoco_menagerie` |

`robot-ci experiment run experiments/configs/smoke_mock.yaml --backend mujoco`
executes 8 real episodes and writes all six artifacts.

## 2. The arm-model problem, and what resolved it

Phase 2 initially used a hand-written MJCF arm derived from the Panda DH table.
It **could not reach a tabletop grasp point**: the point sat **39.5–61 mm outside
the measured reachable set**, and no sign convention or base offset fixed it
(seven approaches tried, all documented in the git history). The IK solver was
blameless — verified against finite differences to `3e-07`, and it recovers a
known pose to **0.08 mm**.

**Resolution:** the official `franka_emika_panda/panda.xml` (Apache-2.0) is now
vendored under `mujoco/models/`, and `build_scene_xml` composes the *scene*
around it instead of synthesizing the arm. The grasp point is now **2.4–4.2 cm**
from the nearest reachable pose, versus 39.5–61 mm before.

Five further real bugs surfaced during that migration, all fixed and all recorded
here because each produced a *plausible-looking* failure rather than a crash:

| Bug | Symptom | Root cause |
|---|---|---|
| Positional `qpos` writes | arm frozen | free-jointed objects occupy the leading DOF columns; joints must be addressed by name |
| Positional Jacobian columns | rank-1 Jacobian | same cause, on the column axis |
| `mj_setConst` not called | self-collision relaxation silently a no-op | broadphase pairs are precomputed at compile time |
| Gravity compensation on every DOF | object launched into the air | `qfrc_bias` was applied to free-floating objects too |
| Degenerate object mass (1e-5 kg, zero inertia) | unstable contacts | mass was computed from a near-zero volume |

## 3. What still does not work

**A pick-and-place does not currently succeed.** `success` is `0.000` on every
episode. The task is blocked on two things, both open:

1. **Arm self-collision.** The official collision meshes self-collide at
   essentially every randomly sampled configuration (`link2↔link5`, `link5↔hand`,
   and others). The relaxation that was supposed to remove intra-arm contacts
   does not take effect, so the episode cannot currently find a collision-free
   start pose. The `mj_setConst` fix is in place but the relaxation is still
   not measurably working.
2. **Grasp geometry is unverified.** The TCP site, the finger actuator mapping
   and the grasp waypoint height have each been set by inspection. The arm now
   executes the full waypoint sequence (approach → descend → lift → transit →
   place) but the grasp does not close on the object, so the object is never
   lifted.

**No manipulation success rates, no regression results, and no benchmark numbers
are claimed anywhere in this repository.**

## 4. Options for Phase 3 (decision needed)

| Option | Effort | Risk | Notes |
|---|---|---|---|
| **A. Finish the self-collision relaxation and tune grasp height empirically** | small–medium | low | The remaining work is bounded: make intra-arm contacts actually disabled, then sweep the grasp TCP height against the measured contact. |
| **B. Switch to a 4-DOF SCARA-style arm sized to the table** | medium | low | Nothing in the framework assumes 7 DOF. A simpler arm has a trivially verifiable workspace, which suits a *regression-testing* platform better than a realistic one. |
| **C. Freeze the embodiment and accept a reach-and-touch benchmark** | small | medium | The task becomes "move the TCP to within tolerance of a target" rather than a manipulation success. Honest, but weaker. |

**Recommendation: A, with C as the fallback.** The remaining gap is two specific,
measurable behaviours, not an architectural problem. If the self-collision
relaxation proves stubborn, B is the more honest choice for a framework whose
subject is regression methodology rather than manipulation quality.

## 5. Verification state at the time of writing

```
194 passed          pytest, 44 source files
All checks passed   ruff
Success             mypy, 44 source files
exit 0              robot-ci validate
```
