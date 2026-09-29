"""Collision-model semantics.

These tests pin the *contract* rather than the tuning: which geom pairs are
allowed to contact, and that a deliberate relaxation actually takes effect in
the compiled model. They exist because a wrong mask produces a plausible-looking
failure rather than an error -- an over-broad relaxation lets the fingers close
through an object, and an under-broad one freezes the arm in self-collision.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robot_ci.models.scenario import ScenarioConfig
from robot_ci.simulation import create_backend
from robot_ci.simulation.mujoco_env import _CONAFFINITY, _CONTYPE, _is_arm_body

pytestmark = pytest.mark.integration
mujoco = pytest.importorskip("mujoco")

ARM_BODIES_PREFIX = ("link",)


@pytest.fixture
def compiled():
    """A freshly reset scene.

    Function-scoped on purpose: several tests below step the physics hundreds of
    times, and a shared model would let one test's state leak into the next
    (the object falls through the table, then everything downstream fails).
    """
    backend = create_backend("mujoco")
    handle = backend.reset(ScenarioConfig(), seed=5)
    yield backend, handle._opaque["model"], handle._opaque["data"]
    backend.close_all()


def _body(model, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or "world"


def _classify(model, geom_id: int) -> str:
    """Classify a geom as ``arm``, ``world`` or the object body's own name."""
    body = _body(model, geom_id)
    if _is_arm_body(body):
        return "arm"
    if body == "world":
        return "world"
    return body


def _arm_qpos(model, data, q: np.ndarray) -> None:
    """Place the arm, addressing joints by name."""
    for index in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index + 1}")
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = float(q[index])
    data.qvel[:] = 0.0


def test_mask_assignment_satisfies_the_collision_rule() -> None:
    """The declared bit assignment must encode exactly the intended policy."""
    def allows(a: str, b: str) -> bool:
        ct_a, ca_a = _CONTYPE[a], _CONAFFINITY[a]
        ct_b, ca_b = _CONTYPE[b], _CONAFFINITY[b]
        return bool((ct_a & ca_b) or (ct_b & ca_a))

    assert not allows("arm", "arm"), "the arm must not collide with itself"
    assert allows("arm", "object"), "the gripper must be able to touch an object"
    assert allows("arm", "world"), "the arm must be able to touch the table"
    assert allows("object", "world"), "the object must rest on the table"


def test_compiled_model_carries_the_declared_masks(compiled) -> None:
    """Post-compile editing must have actually reached the model."""
    _backend, model, _data = compiled
    seen: dict[str, tuple[int, int]] = {}
    for gid in range(model.ngeom):
        key = _classify(model, gid)
        seen.setdefault(key, (int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])))

    # The object geoms are classified by body name ("target_cube") but the
    # declared class is called "object".
    expected = {
        "arm": "arm",
        "world": "world",
        "target_cube": "object",
    }
    for key, cls in expected.items():
        assert key in seen, f"no geom classified as {key!r}; saw {sorted(seen)}"
        want = (_CONTYPE[cls], _CONAFFINITY[cls])
        assert seen[key] == want, f"{key} carries masks {seen[key]}, expected {want}"


def test_relaxed_arm_never_generates_a_self_contact(compiled) -> None:
    """The decisive test: no arm-vs-arm contact, at any sampled configuration.

    This is the empirical counterpart to the mask arithmetic. If broadphase
    pairs were stale, the arithmetic would still pass while contacts persisted.
    """
    _backend, model, data = compiled
    lo, hi = model.jnt_range[:7, 0].copy(), model.jnt_range[:7, 1].copy()

    arm_pairs = 0
    total = 0
    for seed in range(400):
        _arm_qpos(model, data, np.random.default_rng(seed).uniform(lo, hi))
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
            total += 1
            if _classify(model, g1) == "arm" and _classify(model, g2) == "arm":
                arm_pairs += 1

    assert total > 0, "no contacts at all: the sampling is not exercising contact"
    assert arm_pairs == 0, f"{arm_pairs} arm-vs-arm contacts survived the relaxation"


def test_gripper_still_reaches_the_object(compiled) -> None:
    """Relaxation must not disable the contacts the metrics depend on."""
    _backend, model, data = compiled
    lo, hi = model.jnt_range[:7, 0].copy(), model.jnt_range[:7, 1].copy()
    arm_object = 0
    for seed in range(400):
        _arm_qpos(model, data, np.random.default_rng(seed).uniform(lo, hi))
        for _ in range(20):
            mujoco.mj_step(model, data)
            for c in range(data.ncon):
                g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
                k1, k2 = _classify(model, g1), _classify(model, g2)
                if {k1, k2} == {"arm", "target_cube"}:
                    arm_object += 1
    assert arm_object > 0, "the arm can never touch the object: over-relaxed"


def test_object_rests_on_the_table(compiled) -> None:
    """The object-to-table contact is what makes the scene stable."""
    _backend, model, data = compiled
    for _ in range(300):
        mujoco.mj_step(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    assert abs(float(data.xpos[bid][2]) - 0.025) < 0.005
    table_contacts = any(
        {model.geom_bodyid[int(data.contact[c].geom1)], model.geom_bodyid[int(data.contact[c].geom2)]}
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(data.contact[c].geom1)) == "table"
        for c in range(data.ncon)
    )
    assert table_contacts, "the object is not resting on the table"
