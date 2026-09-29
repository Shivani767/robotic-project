"""Audit the compiled model's collision semantics.

Reports every geom's body, contype and conaffinity, then measures which pairs
actually generate contacts. Written to answer three questions with evidence
rather than assumption:

1. What masks does the compiled model actually carry?
2. Is a relaxation applied before or after the broadphase is built?
3. Does a deliberately relaxed pair really stop contacting?

Run with:
    .venv/bin/python scripts/audit_collision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from robot_ci.models.scenario import ScenarioConfig
from robot_ci.simulation import create_backend
from robot_ci.simulation.mujoco_env import _CONAFFINITY, _CONTYPE, _is_arm_body


def _body(model, geom_id: int) -> str:
    """Name of the body a geom belongs to."""
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        or "world"
    )


def _classify(model, geom_id: int) -> str:
    """Classify a geom as arm, world, or by its body name."""
    body = _body(model, geom_id)
    return "arm" if _is_arm_body(body) else body


def _masks(model) -> list[tuple[str, str, int, int]]:
    """Return (class, geom name, contype, conaffinity) for every geom."""
    rows = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<mesh {gid}>"
        rows.append(
            (
                _classify(model, gid),
                name,
                int(model.geom_contype[gid]),
                int(model.geom_conaffinity[gid]),
            )
        )
    return rows


def _arm_qpos(model, data, q: np.ndarray) -> None:
    """Place the arm at q, addressing joints by name."""
    for index in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index + 1}")
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = float(q[index])
    data.qvel[:] = 0.0


def main() -> int:
    """Print the collision audit."""
    print("=" * 78)
    print("1. DECLARED SCENE CLASSES")
    print("=" * 78)
    for name, ct in _CONTYPE.items():
        print(
            f"   {name:<8} contype={ct:>3}  conaffinity={_CONAFFINITY[name]:>3}"
        )

    backend = create_backend("mujoco")
    handle = backend.reset(ScenarioConfig(), seed=5)
    model, data = handle._opaque["model"], handle._opaque["data"]
    rows = _masks(model)

    print()
    print("=" * 78)
    print("2. COMPILED MODEL: distinct mask signatures")
    print("=" * 78)
    signatures: dict[tuple[str, int, int], int] = {}
    for cls, _name, ct, ca in rows:
        signatures[(cls, ct, ca)] = signatures.get((cls, ct, ca), 0) + 1
    for (cls, ct, ca), count in sorted(signatures.items()):
        print(f"   {cls:<12} contype={ct:<3} conaffinity={ca:<3}  x{count}")

    print()
    print("=" * 78)
    print("3. MASK ARITHMETIC  (contact iff (ctA & caB) or (ctB & caA))")
    print("=" * 78)

    def allows(a: str, b: str) -> bool:
        ct_a, ca_a = _CONTYPE[a], _CONAFFINITY[a]
        ct_b, ca_b = _CONTYPE[b], _CONAFFINITY[b]
        return bool((ct_a & ca_b) or (ct_b & ca_a))

    for a, b, want in (
        ("arm", "arm", False),
        ("arm", "object", True),
        ("arm", "world", True),
        ("object", "world", True),
    ):
        got = allows(a, b)
        print(f"   {a:<7} vs {b:<7} -> {got!s:<5} (want {want!s:<5}) {'OK' if got == want else 'FAIL'}")

    print()
    print("=" * 78)
    print("4. EMPIRICAL: arm-vs-arm contacts over 400 random configurations")
    print("=" * 78)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(7)]
    lo = np.array([model.jnt_range[j, 0] for j in jids])
    hi = np.array([model.jnt_range[j, 1] for j in jids])
    arm_pairs = 0
    total = 0
    for seed in range(400):
        _arm_qpos(model, data, np.random.default_rng(seed).uniform(lo, hi))
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            total += 1
            if _classify(model, int(data.contact[c].geom1)) == "arm" and _classify(
                model, int(data.contact[c].geom2)
            ) == "arm":
                arm_pairs += 1
    print(f"   total contacts {total}, arm-vs-arm {arm_pairs}")
    print(f"   relaxation effective: {arm_pairs == 0}")

    backend.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
