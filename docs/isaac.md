# Isaac Sim adapter

**Evidence: Implemented but unverified.**

Isaac Sim is not installed on this machine, has **no macOS build at all**, and
requires an NVIDIA GPU. **No line of the Isaac-facing path has been executed
against a real Isaac installation.** Nothing in this document should be read as
a claim that the adapter works with Isaac Sim.

What *is* verified is the code around it: the module imports with no Omniverse
present, availability reporting is truthful, the translation layer round-trips
and fails loudly, and the backend drives a full episode through the project's
own simulator interface against a mock session. 24 tests, none requiring Isaac.

## What exists

| File | Role | Isaac imports |
|---|---|---|
| `simulation/isaac_sim.py` | Backend, availability probe, session protocol | none at module scope |
| `simulation/isaac_translate.py` | Scenario/state/sensor translation | **none at all** |

```bash
robot-ci backend list     # isaac_sim: constructible, unusable here
robot-ci validate         # lists isaac_sim with the reason
```

## The design, and why

**Isaac is never imported at module scope.** Not for annotations, not for
constants, not behind a `TYPE_CHECKING`. Every import goes through
`_require_isaac()`. A test asserts that no `isaacsim` symbol leaks into the
module namespace, because an `import isaacsim` at the top of a file would make
`robot-ci validate` and the backend registry unimportable on a laptop.

**Translation is a separate module with zero Isaac imports.** Conversions are
expressed against a small documented wire format — dicts, lists and floats in SI
units — rather than against Omniverse objects. Two consequences: the logic is
testable anywhere, and the Omniverse surface is a short greppable list
(`ISAAC_ARTICULATION_API`) rather than code spread through conversions. A
missing prim method is reported by name:

```
the Isaac prim does not provide get_dof_positions, ... ; this adapter targets
the articulation API listed in isaac_translate.ISAAC_ARTICULATION_API and may
need updating for this Isaac version
```

**A session is injected as a `Protocol`.** `IsaacSession` describes what the
adapter needs from a live app. A real implementation wraps an Omniverse
`SimulationApp`; the tests supply a mock of the same shape and the backend cannot
tell the difference. This is what pins the adapter to the project's simulator
interface rather than to my assumptions about Omniverse.

**Constructible is not the same as usable.** The backend constructs without Isaac
on purpose, so a Mac user can still read what the backend would provide. That
made `probe_backend` report Isaac as *available*, which would tell someone their
plan works on a machine that cannot run it. The probe now carries `usable`
alongside `available`, and `SimulatorBackend.is_operational()` is the general
hook. `robot-ci backend list` prints **"constructible, unusable here"** for Isaac
and is unchanged for every backend that works.

## What the translation layer fixes

Decisions that are made once, in one place, rather than re-derived per call:

| Convention | Choice | Why it matters |
|---|---|---|
| Quaternions | Isaac `(x,y,z,w)` ↔ project `(w,x,y,z)` | Wrong order gives a *plausible mirrored pose*, not an error |
| Action targets | position target gets the commanded pose; velocity target zeroed | Sending a position to the velocity channel is a silent control bug |
| Action arrays | all sized from `dof + gripper_dof` | A flat DOF array; disagreeing lengths describe two different robots |
| Render size | `resolution` is `(height, width)` | Transposing it gives a correctly-shaped, wrongly-oriented image |
| Depth | metres | Isaac is SI, so no scaling |
| Object placement | `ObjectSpec.position` honoured when set, else Isaac solves it | Matches MuJoCo, so a cross-backend comparison is meaningful |
| Object uncertainty | isotropic, from a caller-supplied sigma | Isaac annotators report no covariance; inventing one per object would be fiction |

A non-unit wire quaternion is rejected outright rather than normalised silently,
because an unnormalised value means the upstream conversion lost a component and
the orientation would then be wrong in a way no downstream check would catch.

## Capabilities as declared

```python
BackendCapabilities(
    physics=True, collision=True, rgb=True, depth=True, segmentation=True,
    force_torque=True, joint_efforts=True, grasp_latch=False,
    deterministic=False, max_dof=32, requires_gpu=True,
)
```

`deterministic=False` is a real constraint, not a disclaimer: Isaac is
non-deterministic by default, so a seed fixes the scene and not the trajectory.
Cross-run variance must be carried in any comparison, which is different from
every other backend in the project.

`jacobian_provider()` returns `None` by default. Isaac has analytic articulation
Jacobians, but only through a prim API this adapter does not depend on, and
returning zeros would let a policy fail silently instead of degrading.

## To actually run it

1. On a Linux NVIDIA host, install Isaac Sim and a USD asset for the arm.
2. Implement `IsaacSession` against a live `SimulationApp` — this is the one piece
   that does not exist yet, and it is where all the unverified work is.
3. Check the API surface matches:

```python
from robot_ci.simulation.isaac_translate import ISAAC_ARTICULATION_API
```

4. Verify availability before planning compute:

```bash
robot-ci backend list      # isaac_sim must read "yes", not "unusable here"
```

5. Then, and only then, run an experiment and compare against the MuJoCo numbers
   in [`mujoco.md`](mujoco.md) on the same scenarios.

## What must not be claimed

* That the adapter runs with Isaac Sim. It has never been executed there.
* That Isaac and MuJoCo produce comparable results. No cross-backend comparison
  has been run, and the `deterministic=False` capability means it is not a
  like-for-like comparison to begin with.
* That photorealistic rendering changes any measured result. Nothing has been
  measured.

A test in the suite states the premise outright: if Isaac ever becomes importable
here, the mock-based tests stop proving anything and say so. That test failing is
the signal to replace them with real ones.
