# PHASE 2 STATUS

**Date:** 2026-09-28
**Branch:** `main` · **Commit:** `33e70d5`

This file exists to record, precisely and without spin, what Phase 2 delivers,
what is verified, and what is blocked.

---

## 1. What is built and verified

| Component | State | Evidence |
|---|---|---|
| `SimulatorBackend` ABC + registry | working | 194 tests, incl. contract tests run against every backend |
| `MockBackend` | working | deterministic, seed-sensitive, no ground-truth leakage |
| `SyntheticBackend` | working | NumPy ray-cast depth + shaded RGB, real geometry |
| `MuJoCoBackend` | working | physics, RGB/depth/seg, `cgl` offscreen rendering at 1.4 ms/frame |
| `Task` interface | working | `pick_and_place`, `grasp_object`, `push_object`; other names raise `NotImplementedError` |
| `RobotPolicy` interface | working | `policy_v1`, `policy_v2`, `MockPolicy` |
| Metric suite | working | 20 canonical metrics, each with unit, kind and source |
| `ExperimentRunner` | working | `robot-ci experiment run` on mock / synthetic / mujoco |
| Artifact writing | working | `config.yaml`, `summary.json`, `metrics.csv`, `events.jsonl`, `report.html`, `counterexamples.json` |
| Damped-least-squares IK | **solver verified** | Jacobian matches finite differences to **3e-07**; solves to **0.08 mm** when it converges |
| Collision classes | working | contact metrics measure manipulation, not capsule fidelity |

`robot-ci experiment run experiments/configs/smoke_mock.yaml --backend mujoco`
executes 8 real episodes and writes all six artifacts.

---

## 2. The blocker: the arm model cannot reach a tabletop grasp point

**Symptom.** Every episode runs 300 steps and reports `success: 0.000`.

**Diagnosis (measured, not assumed).**

The IK solver is correct. Verified two ways:

* the analytic Jacobian matches a central finite difference to `3.04e-07`;
* seeded with a pose the arm itself produced, the solver returns to it within
  `0.08 mm`.

The solver therefore fails because the **targets are outside the reachable set**:

```
closest approach of the end effector to the grasp point (0, 0.05, 0.06):
  61 mm      <- for the hand-derived chain
  39.5 mm    <- best over all DH sign conventions and base offsets tried
```

At grasp height (z in [0.04, 0.10]) the reachable set sits **behind and below** the
base, not on the table in front of it. Random seeds are also mostly unhelpful:
only ~5% of uniformly sampled configurations put the end effector in the grasp
band at all.

**What was tried, and what each attempt showed.**

| Attempt | Result |
|---|---|
| DH-derived chain (Craig convention) | same 61 mm gap — so the chain structure is not the cause |
| Negating `d` in the DH table | in-front reach 0 → 93/100 samples, but still 38 cm short |
| Negating `alpha` | 60/100 in front, still 47 cm short |
| Null-space posture bias | made convergence *worse* (20/100 → 1/100), reverted |
| Position-first then orientation solve | 0/100 — confirms the targets are unreachable, not the coupling |
| Multi-start home-pose search | best margin 0.62 rad, but the waypoint chain still stalled |
| Robot base moved to y = -0.10 … -0.62 | workspace never entered the grasp-height reachable set |

**Conclusion.** The blocker is the hand-written MJCF arm, not the framework. A
synthesised 7-DOF chain from a DH table does not reproduce the Franka Panda's
workspace, and no amount of re-signing or re-seeding fixes that.

**Why not simply fetch the official model?** MuJoCo 3.14.0 ships no model library
(`site-packages/mujoco/` contains only `include/`), and the `model/panda/panda.xml`
paths on GitHub return 404 for the tags tried. The official MJCF is not reachable
from this machine at present.

---

## 3. Why the rest of the platform is still valid

The failure is localised, and the tests demonstrate it:

* the MuJoCo backend still simulates, renders and steps correctly;
* every schema, metric, artifact and registry path is exercised by a test;
* the ground-truth separation, determinism and seed-sensitivity invariants hold;
* experiments run end to end and produce auditable artifacts.

A successful pick-and-place is not a tuning parameter away; it requires a correct
arm model.

---

## 4. Options for Phase 3 (decision needed)

| Option | Effort | Risk | Notes |
|---|---|---|---|
| **A. Vendor the official Franka MJCF** into `mujoco/models/` (Apache-2.0, from the mujoco_menagerie or the franka_description repo) | small | low | The intended design: the MJCF is *data*, and the framework already loads it as such. Needs a reachable source. |
| **B. Build a simpler arm whose workspace is verified by construction** (e.g. a 4-DOF SCARA-style arm sized to the table) | medium | low | Nothing in the framework assumes 7 DOF except `MuJoCoBackend`'s `dof != 7` guard, which would be relaxed. |
| **C. Keep hand-derived geometry, add a workspace-fitting search** that places the arm base *and* the table from the measured reachable set | medium | medium | Proven to work numerically, but the layout becomes derived from the arm rather than designed. |

**Recommendation: Option A.** It is the smallest change, it removes the
hand-written kinematics entirely, and it matches how the project is meant to work.
Option B is the fallback if no reachable Panda MJCF source exists.

---

## 5. Explicitly not claimed

* **No manipulation success rates.** The `0.000` above is a real measurement of
  the current state, not a benchmark result.
* No regression has been detected, because no task can succeed yet.
* The IK solver is verified; the *arm* is not. These are separate claims and
  only the first is currently true.
