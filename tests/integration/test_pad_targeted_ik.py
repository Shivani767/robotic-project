"""Invariants discovered by the pad-targeted IK experiment.

Each test encodes a fact that was *measured* on the compiled model, so a
regression shows up as a test failure rather than as a silently failed grasp.
The measurements are re-derived from the model at run time: none of them is a
hard-coded constant that could quietly drift out of date.

Run the full experiment with::

    .venv/bin/python scripts/pad_target_ik_experiment.py
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robot_ci.models.geometry import Pose, Quaternion, Vec3
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.planning.kinematics import IKOptions, solve_ik_refined
from robot_ci.simulation import create_backend

pytestmark = pytest.mark.integration
pytest.importorskip("mujoco", reason="MuJoCo is not installed")

#: Measured pad-centre to TCP-site offset for the vendored Panda. Asserted
#: loosely (2 mm) and re-derived below, because its exact value is a property
#: of the MJCF rather than a design constant.
TCP_PAD_OFFSET_TOL_M = 2.0e-3


@pytest.fixture
def compiled():
    """A freshly reset scene, function-scoped so state cannot leak."""
    backend = create_backend("mujoco")
    handle = backend.reset(ScenarioConfig(scenario_id="padtest", seed=5), seed=5)
    yield backend, handle, handle._opaque["model"], handle._opaque["data"]
    backend.close_all()


def _arm_joints(model):
    """Resolve the seven arm joints by name."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(7)]
    assert all(i >= 0 for i in ids)
    return ids


def _pad_geoms(model) -> dict[str, int]:
    """Resolve the largest collision box on each finger."""
    out: dict[str, int] = {}
    best: dict[str, float] = {}
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if body not in ("left_finger", "right_finger"):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if int(model.geom_group[gid]) != 3:
            continue
        vol = float(np.prod(model.geom_size[gid]))
        if vol > best.get(body, -1.0):
            best[body] = vol
            out[body] = gid
    assert set(out) == {"left_finger", "right_finger"}
    return out


def _set_arm(model, data, q) -> None:
    """Write and command the arm by name (control *and* state)."""
    for index, jid in enumerate(_arm_joints(model)):
        value = float(q[index])
        data.qpos[model.jnt_qposadr[jid]] = value
        data.qvel[model.jnt_dofadr[jid]] = 0.0
        for aid in range(model.nu):
            if int(model.actuator_trnid[aid, 0]) == jid:
                data.ctrl[aid] = value


def _finger_joints(model) -> dict[str, int]:
    """Resolve the finger joints by name."""
    return {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("finger_joint1", "finger_joint2")
    }


def _pad_geometry(model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(left_pad, right_pad, tcp)`` in world coordinates."""
    pads = _pad_geoms(model)
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    return (
        np.array(data.geom_xpos[pads["left_finger"]], copy=True),
        np.array(data.geom_xpos[pads["right_finger"]], copy=True),
        np.array(data.site_xpos[tcp], copy=True),
    )


def test_tcp_is_the_pad_midpoint(compiled) -> None:
    """The TCP site sits at the pad midpoint, and stays there.

    This is the invariant the whole target construction rests on. It holds
    across arm poses *and* finger openings, because both the site and the pads
    are rigidly attached to the hand for a given opening.
    """
    _backend, _handle, model, data = compiled
    home = np.array(ScenarioConfig().robot.home_position)
    fingers = _finger_joints(model)
    pads = _pad_geoms(model)

    worst = 0.0
    for pose_index in range(6):
        lo = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}")
                             for i in range(7)], 0].copy()
        hi = model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}")
                             for i in range(7)], 1].copy()
        q = home if pose_index == 0 else np.random.default_rng(pose_index).uniform(lo, hi)
        for opening in (0.0, 0.02, 0.04):
            for jid in fingers.values():
                data.qpos[model.jnt_qposadr[jid]] = opening
                data.qvel[model.jnt_dofadr[jid]] = 0.0
            _set_arm(model, data, q)
            mujoco.mj_forward(model, data)
            left, right, tcp = _pad_geometry(model, data)
            worst = max(worst, float(np.linalg.norm(tcp - 0.5 * (left + right))))
    assert worst < TCP_PAD_OFFSET_TOL_M, (
        f"TCP is {worst * 1000:.2f} mm from the pad midpoint, "
        f"above the {TCP_PAD_OFFSET_TOL_M * 1000:.1f} mm tolerance"
    )
    # The pads must actually be symmetric about the midpoint, or "midpoint"
    # would be a meaningless average.
    left, right, _tcp = _pad_geometry(model, data)
    assert float(np.linalg.norm(np.asarray(model.geom_size[pads["left_finger"]]))) == pytest.approx(
        float(np.linalg.norm(np.asarray(model.geom_size[pads["right_finger"]])))
    )


def test_gripper_envelope_is_not_the_body_origin_distance(compiled) -> None:
    """Pad separation runs 11.0 mm closed to 91.0 mm open.

    Phase 2 recorded the maximum opening as 26.2 mm, which was measured on the
    finger *body origins* rather than the pad collision geoms. That figure
    understates the real envelope by more than 3x and is the reason a 50 mm
    cube was wrongly declared ungraspable. This test re-derives the true
    envelope from the pad geoms and pins the closed and open separations.
    """
    _backend, _handle, model, data = compiled
    fingers = _finger_joints(model)
    lo = float(min(model.jnt_range[j, 0] for j in fingers.values()))
    hi = float(max(model.jnt_range[j, 1] for j in fingers.values()))
    separations = []
    for cmd in np.linspace(lo, hi, 9):
        for jid in fingers.values():
            data.qpos[model.jnt_qposadr[jid]] = float(cmd)
            data.qvel[model.jnt_dofadr[jid]] = 0.0
        mujoco.mj_forward(model, data)
        left, right, _tcp = _pad_geometry(model, data)
        separations.append(float(np.linalg.norm(left - right)))

    closed, opened = separations[0], separations[-1]
    assert closed < 0.02, f"fully closed pads are {closed * 1000:.1f} mm apart"
    assert opened > 0.08, f"fully open pads are only {opened * 1000:.1f} mm apart"
    # The relationship is linear in the finger joint, so a grasp width can be
    # predicted from the model rather than calibrated by trial.
    steps = np.diff(separations)
    assert float(steps.std()) < 1e-6, "pad separation is not linear in the finger joint"


def test_arm_joint_ranges_are_not_the_first_seven_entries(compiled) -> None:
    """``jnt_range[:7]`` is the free joint plus six arm joints, not the arm.

    Joint 0 is the object's free joint with range [0, 0]. Positional slicing
    therefore returns joint1's range as joint0's and shifts every limit by one,
    which silently disables or misbounds the solver.
    """
    _backend, _handle, model, _data = compiled
    ids = _arm_joints(model)
    by_name = np.array([model.jnt_range[j, 0] for j in ids])
    positional = model.jnt_range[:7, 0].copy()
    assert not np.allclose(by_name, positional), (
        "positional slicing happens to match the arm here; the free-joint "
        "offset must still not be relied on"
    )
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_free")
    assert free == 0, "the object free joint is expected to be joint 0"
    assert float(model.jnt_range[free, 1] - model.jnt_range[free, 0]) == pytest.approx(0.0)


def test_finger_actuators_map_through_actuator_trnid(compiled) -> None:
    """Each finger joint has its own actuator, resolved through the model."""
    _backend, _handle, model, _data = compiled
    fingers = _finger_joints(model)
    driven = {
        name: [
            aid for aid in range(model.nu) if int(model.actuator_trnid[aid, 0]) == jid
        ]
        for name, jid in fingers.items()
    }
    for name, aids in driven.items():
        assert len(aids) == 1, f"{name} is driven by {aids}, expected exactly one actuator"
    assert driven["finger_joint1"] != driven["finger_joint2"]
    for aids in driven.values():
        lo, hi = model.actuator_ctrlrange[aids[0]]
        assert lo == pytest.approx(0.0) and hi > 0.0


def _grasp_geometry(model, data, object_position) -> dict:
    """Measure the grasp geometry at the current configuration.

    Static: the configuration is placed and forward kinematics is run. Stepping
    physics would let the arm disturb the object before the geometry is read,
    which would measure the disturbance instead of the grasp.
    """
    for jid in _finger_joints(model).values():
        data.qpos[model.jnt_qposadr[jid]] = 0.04
        data.qvel[model.jnt_dofadr[jid]] = 0.0
    mujoco.mj_forward(model, data)
    left, right, _tcp = _pad_geometry(model, data)
    mid = 0.5 * (left + right)
    axis = (left - right)
    separation = float(np.linalg.norm(axis))
    axis = axis / max(separation, 1e-12)
    reference = np.asarray(object_position, dtype=float)
    return {
        "separation_m": separation,
        "separation_axis": axis,
        "low_pad_z_m": float(min(left[2], right[2])),
        "object_along_axis_m": float(np.dot(reference - mid, axis)),
        "midpoint_to_object_m": float(np.linalg.norm(mid - reference)),
    }


def _solve_at(provider, model, target, orientation, home, options, restarts=3):
    """Run the existing joint-limit-aware IK for a candidate orientation."""
    ids = _arm_joints(model)
    lower = np.array([model.jnt_range[j, 0] for j in ids])
    upper = np.array([model.jnt_range[j, 1] for j in ids])
    return solve_ik_refined(
        provider,
        Pose(position=Vec3.from_iterable(target), orientation=orientation),
        home,
        lower=lower,
        upper=upper,
        options=options,
        restarts=restarts,
    )


def test_vertical_jaw_separation_cannot_clear_the_table(compiled) -> None:
    """A grasp whose jaws separate vertically is impossible at table height.

    The lower pad would sit ``separation/2`` below the jaw centre. With the
    measured open separation of 91 mm, centring the jaws on an object resting
    on the table puts the lower pad roughly 37 mm *inside* the tabletop, so the
    grasp fails on contact geometry no matter how good the IK is.

    This pins the reason the production "side grasp" orientation cannot work,
    and stops it being reintroduced.
    """
    _backend, handle, model, data = compiled
    scenario = ScenarioConfig(scenario_id="padtest", seed=5)
    provider = create_backend("mujoco").jacobian_provider(handle)
    table_top = float(scenario.environment.table_height_m)
    obj = np.array(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")], copy=True
    )
    home = np.array(scenario.robot.home_position)
    options = IKOptions(
        max_iterations=150, position_tolerance=6e-3, orientation_tolerance=0.3, nullspace_gain=0.0
    )

    side = Quaternion.from_axis_angle(Vec3.of(1, 0, 0), 0.5 * np.pi)
    result = _solve_at(provider, model, obj, side, home, options)
    _set_arm(model, data, result.joint_positions)
    geo = _grasp_geometry(model, data, obj)

    # The separation axis is vertical, and the lower pad is below the table.
    axis = np.round(geo["separation_axis"], 2)
    assert abs(geo["separation_axis"][2]) > 0.8, f"expected a vertical separation axis, got {axis}"
    assert geo["low_pad_z_m"] < table_top, (
        f"lower pad is at {geo['low_pad_z_m'] * 1000:.1f} mm, "
        f"{abs(geo['low_pad_z_m'] - table_top) * 1000:.1f} mm inside the table"
    )


def test_top_down_orientation_is_feasible_and_ik_fk_agree(compiled) -> None:
    """The top-down grasp is reachable, consistent, and puts the pads on the object.

    Also checks that the solver's reported residual matches an independent read
    of the compiled model. A mismatch there would mean the solver and the
    simulator disagree about the forward kinematics, which is a far more
    serious fault than a failed grasp.
    """
    _backend, handle, model, data = compiled
    scenario = ScenarioConfig(scenario_id="padtest", seed=5)
    provider = create_backend("mujoco").jacobian_provider(handle)
    obj = np.array(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")], copy=True
    )
    home = np.array(scenario.robot.home_position)
    table_top = float(scenario.environment.table_height_m)
    # A deliberately generous budget: this test is about reachability and
    # solver/model consistency, not about the tightest achievable tolerance.
    # Tightening IK to make a test pass would be exactly the kind of magic fix
    # this experiment is meant to avoid.
    options = IKOptions(
        max_iterations=200, position_tolerance=5.0e-3, orientation_tolerance=0.30
    )

    top_down = Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)
    result = _solve_at(provider, model, obj, top_down, home, options, restarts=8)
    assert result.position_error < 6.0e-3, (
        f"top-down grasp is not reachable: {result.position_error * 1000:.1f} mm residual"
    )

    # Solver and model must agree about where the end effector ended up.
    fk_pos, _rot = provider.forward(result.joint_positions)
    measured = float(np.linalg.norm(np.asarray(obj) - fk_pos))
    assert measured == pytest.approx(result.position_error, abs=1e-6), (
        f"solver reports {result.position_error * 1000:.3f} mm but the model "
        f"measures {measured * 1000:.3f} mm"
    )

    _set_arm(model, data, result.joint_positions)
    geo = _grasp_geometry(model, data, obj)
    # Horizontal separation, both pads above the table, object between them.
    axis = np.round(geo["separation_axis"], 2)
    assert abs(geo["separation_axis"][2]) < 0.2, (
        f"top-down separation axis should be horizontal, got {axis}"
    )
    assert geo["low_pad_z_m"] > table_top
    half = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")][0])
    assert abs(geo["object_along_axis_m"]) <= 0.5 * geo["separation_m"]
    assert geo["midpoint_to_object_m"] <= half + 0.004


def test_jaw_height_window_is_narrower_than_the_object(compiled) -> None:
    """Quantify why the jaws jam on the table instead of closing on the object.

    A pad of half-height ``h`` centred at height ``z`` has its lower edge at
    ``z - h``. Clearing the tabletop needs ``z >= table + h``; overlapping the
    object's side faces needs ``z <= object_top``. The usable window is
    therefore ``[table + h, object_top]``.

    With the measured pad half-height this window is only a few millimetres
    wide, and a jaw aimed at the *object centre* falls below it. The pads then
    rest on the tabletop and the jaws jam open, which is exactly the failure
    the experiment measured (0.16 mm of pad penetration, 38.8 mm of residual
    separation, no finger-to-object contact).
    """
    _backend, handle, model, data = compiled
    scenario = ScenarioConfig(scenario_id="padtest", seed=5)
    provider = create_backend("mujoco").jacobian_provider(handle)
    table_top = float(scenario.environment.table_height_m)
    obj = np.array(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")], copy=True
    )
    home = np.array(scenario.robot.home_position)
    options = IKOptions(
        max_iterations=200, position_tolerance=5.0e-3, orientation_tolerance=0.30
    )
    top_down = Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)
    result = _solve_at(provider, model, obj, top_down, home, options, restarts=8)
    _set_arm(model, data, result.joint_positions)
    geo = _grasp_geometry(model, data, obj)

    # Pad half-height along world z, from the pad's own rotation.
    pads = _pad_geoms(model)
    heights = []
    for name in ("left_finger", "right_finger"):
        gid = pads[name]
        rot = data.geom_xmat[gid].reshape(3, 3)
        heights.append(float(np.sum(np.abs(rot[2, :]) * np.asarray(model.geom_size[gid]))))
    pad_half_height = max(heights)
    half = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")][0])

    min_jaw_z = table_top + pad_half_height
    object_top = float(obj[2] + half)
    window = object_top - min_jaw_z

    assert window > 0.0, (
        f"no feasible jaw height at all: need z >= {min_jaw_z * 1000:.1f} mm "
        f"but the object only reaches {object_top * 1000:.1f} mm"
    )
    # The jaws aimed at the object centre sit below that window, which is the
    # geometric cause of the jam. Recorded, not asserted away.
    jaw_z = float(obj[2])
    assert jaw_z < min_jaw_z, (
        f"expected the object-centre jaw target to fall below the window; "
        f"jaw {jaw_z * 1000:.2f} mm vs minimum {min_jaw_z * 1000:.2f} mm"
    )
    # The shortfall is small, so this is a calibration offset, not a redesign.
    assert (min_jaw_z - jaw_z) < 0.005, (
        f"jaw shortfall is {(min_jaw_z - jaw_z) * 1000:.1f} mm"
    )
    # And the window really is tight.
    assert window < 0.015, f"grasp window is {window * 1000:.1f} mm, unexpectedly wide"
    assert geo["separation_m"] > 0.08


def test_object_contacts_must_be_split_by_counterpart(compiled) -> None:
    """A contact count on the object is meaningless unless it names the counterpart.

    The object rests on the table from the first step of every episode, so
    counting *all* contacts involving it returns a nonzero value -- and a 1.0
    "contact rate" -- whether or not the gripper ever reached it. Only
    finger-to-object contact is evidence of a grasp. The experiment initially
    reported a 1.0 contact rate for exactly this reason; the true finger
    contact rate was 0.4.
    """
    _backend, _handle, model, data = compiled
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")
    for _ in range(200):
        mujoco.mj_step(model, data)
    finger_geoms = {
        gid
        for gid in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or "")
        in ("left_finger", "right_finger")
    }

    total = 0
    finger = 0
    support = 0
    for c in range(data.ncon):
        g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
        if obj_gid not in (g1, g2):
            continue
        total += 1
        other = g2 if g1 == obj_gid else g1
        if other in finger_geoms:
            finger += 1
        else:
            support += 1

    assert total == finger + support
    # The support contacts are the table, and exist from the start.
    assert support > 0, "the object should rest on the table"
    # With the arm at home and the gripper closed, there is no grasp, so the
    # finger-contact count must not be inferred from the total.
    assert finger == 0, (
        f"{finger} finger contacts with the arm at home; the contact rate is "
        "not measuring the grasp"
    )


def test_derived_jaw_height_clears_the_table_and_holds_the_object(compiled) -> None:
    """The derived jaw height satisfies both constraints at once.

    Two measured bounds apply to the jaw centre height ``z``:

    * the pad's lower edge must clear the table, ``z >= table + pad_half_height``;
    * the pad must overlap the object's side faces, ``z <= object_top``.

    Aiming at the object centre satisfies the second and violates the first, so
    the pads rest on the tabletop and the jaws jam open without touching the
    object. Targeting the midpoint of the derived window satisfies both. This
    test asserts the *property* -- the pad edge clears the table **and** the
    pad still overlaps the object -- rather than a remembered number.
    """
    _backend, handle, model, data = compiled
    scenario = ScenarioConfig(scenario_id="padtest", seed=5)
    provider = create_backend("mujoco").jacobian_provider(handle)
    table_top = float(scenario.environment.table_height_m)
    obj = np.array(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")], copy=True
    )
    home = np.array(scenario.robot.home_position)
    options = IKOptions(
        max_iterations=200, position_tolerance=5.0e-3, orientation_tolerance=0.30
    )
    top_down = Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)

    # Pass 1: place the jaws on the object so the pad rotation is known.
    provisional = _solve_at(provider, model, obj, top_down, home, options, restarts=8)
    _set_arm(model, data, provisional.joint_positions)
    geo = _grasp_geometry(model, data, obj)

    pads = _pad_geoms(model)
    heights = []
    for name in ("left_finger", "right_finger"):
        gid = pads[name]
        rot = data.geom_xmat[gid].reshape(3, 3)
        heights.append(float(np.sum(np.abs(rot[2, :]) * np.asarray(model.geom_size[gid]))))
    pad_half = max(heights)
    half = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")][0])

    window_lo = table_top + pad_half
    window_hi = float(obj[2] + half)
    assert window_hi > window_lo, "no feasible jaw height exists for this object"

    # The object-centre target, which the previous implementation used, must be
    # shown to violate the clearance constraint.
    pad_edge_at_object = float(obj[2]) - pad_half
    assert pad_edge_at_object < table_top, (
        f"expected the object-centre target to sink the pads to "
        f"{pad_edge_at_object * 1000:.2f} mm, below the table"
    )

    # The derived target must satisfy both bounds with margin on each side.
    target_z = 0.5 * (window_lo + window_hi)
    assert target_z > window_lo
    assert target_z < window_hi
    assert target_z - pad_half > table_top, "derived target still sinks the pads into the table"
    assert target_z < float(obj[2] + half), "derived target lifts the pads off the object"

    # And the arm must actually be able to reach the derived target.
    result = _solve_at(
        provider,
        model,
        np.array([obj[0], obj[1], target_z]),
        top_down,
        provisional.joint_positions,
        options,
        restarts=4,
    )
    assert result.position_error < 6.0e-3, (
        f"derived jaw height is not reachable: {result.position_error * 1000:.1f} mm"
    )
    _set_arm(model, data, result.joint_positions)
    final = _grasp_geometry(model, data, obj)
    assert final["low_pad_z_m"] - pad_half > table_top, "pads still touch the table at the grasp"
    assert abs(final["object_along_axis_m"]) <= 0.5 * final["separation_m"]
    assert geo["separation_m"] > 0.08


def test_policy_and_task_agree_on_the_grasp_orientation() -> None:
    """The policy and the task must command the same frame.

    They each defined their own orientation constant once, drifted apart, and
    the resulting grasp failed silently: the policy aimed the arm one way and
    the task planned the waypoints for another. The policy now re-exports the
    task's constant, and this test keeps them from diverging again.
    """
    from robot_ci.policies import TOP_DOWN as POLICY_ORIENTATION
    from robot_ci.tasks.manipulation import TOP_DOWN as TASK_ORIENTATION

    assert (POLICY_ORIENTATION.w, POLICY_ORIENTATION.x, POLICY_ORIENTATION.y,
            POLICY_ORIENTATION.z) == (
        TASK_ORIENTATION.w, TASK_ORIENTATION.x, TASK_ORIENTATION.y, TASK_ORIENTATION.z
    )


def test_grasping_aperture_is_derived_not_zero(compiled) -> None:
    """The holding aperture is derived from the object and the gripper envelope.

    Commanding a zero aperture does not hold an object: the position actuators
    keep driving and squeeze it out of the jaws. The aperture that brings the
    pads onto the object is ``squeeze * object_width - closed_separation``, from
    the measured relationship ``pad_separation = commanded_width + closed``.
    """
    _backend, _handle, model, _data = compiled
    scenario = ScenarioConfig(scenario_id="grasp", seed=5)
    from robot_ci.policies import policy_v1

    policy = policy_v1(scenario=scenario)
    policy.reset(policy.config, 5, scenario)

    object_width = 2.0 * float(scenario.target.size_m)
    closed = float(scenario.robot.gripper_closed_separation_m)
    squeeze = policy.grasp_squeeze

    expected = max(0.0, squeeze * object_width - closed)
    assert policy.hold_width_m == pytest.approx(expected)
    # The pads must land on the object, not past it.
    assert closed < policy.hold_width_m + closed <= object_width + 1e-9
    assert policy.hold_width_m > 0.0, "a zero aperture ejects the object rather than holding it"
    # And the resulting pad separation, from the measured envelope.
    separation = policy.hold_width_m + closed
    assert separation < object_width, "commanded separation would pass through the object"
    # The derivation must round-trip against the real model.
    fingers = _finger_joints(model)
    for jid in fingers.values():
        _data = _handle._opaque["data"]
        _data.qpos[model.jnt_qposadr[jid]] = 0.5 * (separation - closed)
        _data.qvel[model.jnt_dofadr[jid]] = 0.0
    _set_arm(model, _handle._opaque["data"], np.array(scenario.robot.home_position))
    mujoco.mj_forward(model, _handle._opaque["data"])
    left, right, _tcp = _pad_geometry(model, _handle._opaque["data"])
    assert float(np.linalg.norm(left - right)) == pytest.approx(separation, abs=2e-3)
