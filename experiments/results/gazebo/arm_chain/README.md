# Part 1D: does the articulated arm chain reproduce the finger stall?

`scripts/arm_chain.py` builds the exact Panda chain taken straight out of
`model_sdf.txt` -- world anchor, seven revolute joints, `panda_link7`, the finger
joint -- with no table, no cube, no camera, no contact sensor and no gripper
collision, then commands `left_finger` across 0-40 mm while recording every joint
and the finger's joint reaction.

## The answer is no

| Variant | Arm chain | Arm controllers | Gravity | Full stroke? | Max q | Final err | Result |
|---|---|---|---|---|---:|---:|---|
| D | exact Panda | exact Panda gains | ON | **yes** | 38.604 mm | -1.396 mm | **FINGER TRACKS** |

Not run yet: A (rigid), B (dynamic/no controllers), C (dynamic/generic), D2
(exact/no controllers), D3 (gravity off), and the one-controller-at-a-time
ablation. Section 12's question is nonetheless answerable from D alone, because D
is the configuration closest to the one that fails.

**The articulated arm chain does not reproduce the stall.** The brief warned
against calling the arm chain the cause merely because the full Panda contains it;
here the measurement says the opposite.

## What the trace actually shows

    cmd mm   measured mm   error mm   qdot m/s
      0         0.34         +0.34      +0.200
      5         0.82         -4.18      -0.199
     10         6.62         -3.38      +0.200
     15        12.14         -2.86      +0.200
     20        17.39         -2.61      -0.199
     25        22.77         -2.23      -0.199
     30        28.09         -1.91      -0.200
     35        33.37         -1.63      -0.199
     40        38.60         -1.40      -0.200

The error is negative and **shrinks monotonically** as the command rises, and
`qdot` alternates at +/-0.1997 m/s against a declared velocity limit of 0.2 m/s.
That is a **velocity-limit limit cycle**, and it is the same shape as the limit
cycle Part 1B recorded when the velocity limit was raised tenfold. It is not a
mechanical stall at a fixed position.

So the finger is rate-limited and still converging when each sample is taken.
Whatever produces the full Panda's flat ~21 mm stall is therefore neither the arm
chain nor anything present in variant D.

## What D still lacks against the failing full Panda

  * gripper link collisions -- stripped here, as the brief required
  * the table and the target cube
  * the Contact and Sensors systems
  * `right_finger`, present but never commanded in this sweep

## Next single experiment

Re-run D with gripper collisions restored, changing nothing else. If the stall
appears, collision self-interaction inside the gripper is the cause and can be
isolated link by link. If it still tracks, the remaining difference is the scene,
not the mechanism, and the search must move there.

## D+projworld: the real world minus the cube tracks

The final unblock was trivial once licensed: `world_sdf` takes `robot_urdf` and
embeds the converted robot as a `<model>` -- the production path all along --
instead of emitting an `<include>` that needs a `model.config` nobody has. With
that, the project world loads clean (no sim errors), `joint_state` appears after
~9 s, and the sweep reads back:

    D+projworld   full stroke, max q 38.444 mm, final err -1.556 mm

Real placement (bolted to the table), real table, real plugin set, cube removed --
**the finger tracks.** The table is falsified.

That reframes everything before it. The ~21 mm stall with its ~5 N reaction needs
the cube present, or the specific episode arm pose; it is reproduced by nothing in
the mechanism, the collisions, the gains, the poses, the plugin count, the arm
chain, or the table.

Root cause: NOT IDENTIFIED. Part 1 remains open.
