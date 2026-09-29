"""Audit the compiled model's collision semantics.

Reports every geom's body, contype and conaffinity, then measures which
pairs actually generate contacts. Written to answer three questions with
evidence rather than assumption:

1. What masks does the compiled model actually carry?
2. Is a relaxation applied before or after the broadphase is built?
3. Does a deliberately relaxed pair really stop contacting?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from robot_ci.models.scenario import ScenarioConfig
from robot_ci.simulation import create_backend
from robot_ci.simulation.mujoco_env import _CONAFFINITY, _CONTYPE

ARM_GEOM_BODIES = ("link", "hand", "finger")


def body_of(model, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or "world"


def is_arm(body: str) -> bool:
    return body.startswith("link") or body in {"hand", "left_finger", "right_finger"}


def dump_masks(model) -> list[tuple[int, str, str, int, int]]:
    rows = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<mesh {gid}>"
        body = body_of(model, gid)
        rows.append((gid, name, body, int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])))
    return rows


def pair_contacts(model, data, q, g1: int, g2: int) -> tuple[int, float]:
    """Place the arm at ``q`` and measure contacts between two geoms.

    Returns:
        ``(count, min_distance)`` where ``count`` is the number of active
        contacts between the pair over 20 physics substeps and ``min_distance``
        is the deepest penetration seen (``-inf`` when there is none).
    """
    for index in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index + 1}")
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = float(q[index])
    data.qvel[:] = 0.0
    count, deepest = 0, float("inf")
    for _ in range(20):
        mujoco.mj_step(model, data)
        for c in range(data.ncon):
            pair = {int(data.contact[c].geom1), int(data.contact[c].geom2)}
            if pair == {g1, g2}:
                count += 1
                deepest = min(deepest, float(data.contact[c].dist))
    return count, deepest


def main() -> int:
    print("=" * 78)
    print("1. DECLARED SCENE CLASSES (from mujoco_env)")
    print("=" * 78)
    for name, ct in _CONTYPE.items():
        print(f"   {name:<8} contype=0b{ct:05b} ({ct:>3})  conaffinity=0b{_CONAFFINITY[name]:05b} ({_CONAFFINITY[name]:>3})")

    backend = create_backend("mujoco")
    handle = backend.reset(ScenarioConfig(), seed=5)
    model, data = handle._opaque["model"], handle._opaque["data"]

    print()
    print("=" * 78)
    print("2. COMPILED MODEL: arm geom masks (after any post-compile edit)")
    print("=" * 78)
    rows = dump_masks(model)
    arm_rows = [r for r in rows if is_arm(r[2])]
    world_rows = [r for r in rows if r[2] in {"world", "target_cube"}]
    print(f"   arm geoms: {len(arm_rows)}   world/object geoms: {len(world_rows)}")
    for gid, name, body, ct, ca in arm_rows[:6]:
        print(f"   {name:<22} body={body:<14} ct=0b{ct:05b} ca=0b{ca:05b}")
    print("   ...")
    for gid, name, body, ct, ca in arm_rows[-3:]:
        print(f"   {name:<22} body={body:<14} ct=0b{ct:05b} ca=0b{ca:05b}")
    print()
    for gid, name, body, ct, ca in world_rows:
        print(f"   {name:<22} body={body:<14} ct=0b{ct:05b} ca=0b{ca:05b}")

    # Distinct mask signatures.
    sig = {}
    for gid, name, body, ct, ca in arm_rows:
        sig.setdefault((ct, ca), []).append(body)
    print()
    print("   distinct arm mask signatures:")
    for (ct, ca), bodies in sorted(sig.items()):
        print(f"      ct=0b{ct:05b} ca=0b{ca:05b}  ->  {len(bodies)} geoms, bodies {sorted(set(bodies))}")

    print()
    print("=" * 78)
    print("3. COLLISION RULE CHECK  (contact iff (ctA & caB) or (ctB & caA))")
    print("=" * 78)

    def hit(a, b):
        return bool((a[0] & b[1]) or (b[0] & a[1]))

    def sample(kind):
        for gid, name, body, ct, ca in rows:
            if kind == "arm" and is_arm(body):
                return (ct, ca), name, body
            if kind == "object" and body == "target_cube":
                return (ct, ca), name, body
            if kind == "world" and body == "world":
                return (ct, ca), name, body
        raise KeyError(kind)

    arm_m, arm_n, arm_b = sample("arm")
    obj_m, obj_n, _ = sample("object")
    wld_m, wld_n, _ = sample("world")
    print(f"   arm    {arm_n:<20} ({arm_b})  ct=0b{arm_m[0]:05b} ca=0b{arm_m[1]:05b}")
    print(f"   object {obj_n:<20}          ct=0b{obj_m[0]:05b} ca=0b{obj_m[1]:05b}")
    print(f"   world  {wld_n:<20}          ct=0b{wld_m[0]:05b} ca=0b{wld_m[1]:05b}")
    print()
    print(f"   arm    vs arm    -> masks allow contact: {hit(arm_m, arm_m)}   (want False)")
    print(f"   arm    vs object -> masks allow contact: {hit(arm_m, obj_m)}   (want True)")
    print(f"   arm    vs world  -> masks allow contact: {hit(arm_m, wld_m)}   (want True)")
    print(f"   object vs world  -> masks allow contact: {hit(obj_m, wld_m)}   (want True)")

    print()
    print("=" * 78)
    print("4. EMPIRICAL: find a configuration where link2 and link5 actually contact")
    print("=" * 78)
    g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link2_geom")
    g5 = None
    for gid, name, body, ct, ca in rows:
        if body == "link5" and name.endswith(("_collision", "hand_c")) or (body == "link5" and "collision" in name):
            g5 = gid
            break
    if g5 is None:
        g5 = next(r[0] for r in rows if r[2] == "link5")
    print(f"   link2_geom id={g2}  link5 geom id={g5} ({rows[g5][1]})")

    lo = model.jnt_range[:7, 0].copy()
    hi = model.jnt_range[:7, 1].copy()
    found, tested = None, 0
    for seed in range(4000):
        q = np.random.default_rng(seed).uniform(lo, hi)
        c, _d = pair_contacts(model, data, q, g2, g5)
        tested += 1
        if c:
            found = (seed, q.copy(), c)
            break
    if found is None:
        print(f"   no link2<->link5 contact found in {tested} random configurations")
    else:
        seed, q, c = found
        print(f"   FOUND at seed {seed}: {c} contact steps between the pair")
        print(f"   q = {np.round(q, 4).tolist()}")
        print("   -> a seed is available; re-run the audit after relaxing to prove")
        print("      the same seed produces zero contacts.")

    backend.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
