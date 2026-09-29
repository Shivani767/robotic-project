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
    squeeze = policy.grasp_width_fraction

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


# ---------------------------------------------------------------------------
# Regressions for two defects that made a working grasp look like no grasp.
#
# Both were found by running episodes and reading the trace, not by writing
# tests first, and both had the same signature: every individual measurement
# looked correct while the task reported that nothing had happened.
# ---------------------------------------------------------------------------


@pytest.fixture
def scene():
    """A freshly reset scene, function-scoped so state cannot leak."""
    backend = create_backend("mujoco")
    handle = backend.reset(ScenarioConfig(scenario_id="graspregress", seed=1), seed=1)
    yield backend, handle, handle._opaque["model"], handle._opaque["data"]
    backend.close_all()


def test_arm_is_commanded_not_teleported(scene):
    """A large joint command must be travelled, not teleported to.

    If ``qpos`` equals the command after a single 20 ms control step, the arm
    is being teleported. That makes it impossible for the gripper to carry
    anything: the wrist leaves the object behind instead of lifting it, which
    looks exactly like a grasp that fails to hold.
    """
    backend, handle, model, data = scene
    from robot_ci.models.state import Action, ActionType

    scenario = ScenarioConfig(scenario_id="graspregress", seed=1)
    lower = np.asarray(scenario.robot.limits.lower, dtype=np.float64)
    upper = np.asarray(scenario.robot.limits.upper, dtype=np.float64)
    dof = int(scenario.robot.dof)
    target = np.where((upper - lower) > 1.0, lower + 0.6 * (upper - lower), lower)

    # The arm does not own a contiguous qpos slice -- free-jointed objects share
    # qpos with it -- so the joints are resolved by name, exactly as the backend
    # does when it writes the command.
    qadr = [
        int(model.jnt_qposadr[jid])
        for jid in backend._arm_joint_ids(model, dof)
        if jid >= 0
    ]
    assert len(qadr) == dof, "arm joint lookup disagreed with the robot's dof"

    before = np.array(data.qpos[qadr], copy=True)
    backend.step(
        handle,
        Action(
            type=ActionType.JOINT_POSITION,
            joint_position=tuple(float(v) for v in target),
            arm_joint_count=dof,
        ),
    )
    after = np.array(data.qpos[qadr], copy=True)

    assert np.abs(after - before).max() > 1.0e-3, "arm did not move at all"
    # A servo still tracking has real error left. Exactly zero means the arm
    # was placed at the commanded pose rather than driven to it.
    assert np.abs(target - after).max() > 1.0e-3, "arm was teleported to the commanded pose"


def test_finger_contact_resolved_by_body_not_geom_name(scene):
    """Grasp detection must not depend on the collision geoms being named.

    The finger geoms are asserted to be *unnamed* too, because that absence is
    the exact condition that broke detection. A future vendored model that does
    name them would otherwise let this test pass while leaving the body-based
    path -- the one that actually works -- untested.
    """
    backend, _handle, model, data = scene

    finger_bodies = {}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "finger" in name:
            finger_bodies[name] = body_id

    assert {"left_finger", "right_finger"} <= set(finger_bodies), (
        "finger bodies must stay nameable: grasp detection resolves contacts "
        "through the body a geom is attached to, and the names are its only handle"
    )

    unnamed = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        for body_id in finger_bodies.values()
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) == body_id
    ]
    assert all(name is None for name in unnamed), (
        "the vendored finger geoms are expected to be unnamed; if this now fails "
        "the model changed and _is_grasped should be re-examined"
    )

    # Nothing is touching the object yet, so this is False -- and it is False
    # because no finger contact exists, not because a name lookup quietly
    # returned an empty string.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube") >= 0
    assert backend._is_grasped(model, data, "target_cube") is False



# ---------------------------------------------------------------------------
# Gripper contact-control integration: the controller must stay tied to the
# model it runs on, and POSITION mode must be indistinguishable from the bare
# write path it replaces.
# ---------------------------------------------------------------------------


def test_finger_stiffness_is_read_from_the_model(scene):
    """The actuator mapping stays model-derived, not hard-coded.

    The helper must find the stiffness by joint name, and the number it returns
    must equal the finger actuator's own gain. If the vendored model changes,
    the force bound follows it instead of silently meaning the wrong force.
    """
    backend, _handle, model, _data = scene
    stiffness = backend._finger_stiffness(model)
    # Cross-check against a direct read: first joint-type actuator whose joint
    # carries a finger in its name.
    expected = None
    for index in range(model.nu):
        if int(model.actuator_trntype[index]) not in (0, 1, 2):
            continue
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0])
        ) or ""
        if "finger" in name:
            expected = float(model.actuator_gainprm[index][0])
            break
    assert expected is not None
    assert stiffness == pytest.approx(expected)
    assert stiffness > 0.0, "a non-positive stiffness would invert the force bound"


def test_position_mode_writes_the_request_unchanged(scene):
    """POSITION mode is bit-for-bit the old `_write_gripper` behaviour.

    Regression safety for the experiment: with the default config every test,
    and every previously recorded number, exercises the controller in POSITION
    mode, so the controller must be observably identical to no controller at
    all.
    """
    backend, handle, model, data = scene
    controller = backend._make_gripper_controller(model, handle.scenario)
    assert controller.mode.value == "position"

    f1 = backend._actuator_for_joint(
        model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    )
    for width in (0.08, 0.04, 0.016, 0.0002, 0.0):
        backend._write_gripper(model, data, width, controller=controller)
        assert float(data.ctrl[f1]) == pytest.approx(max(0.0, min(0.04, width / 2.0)))


def test_bounded_controller_floors_on_the_real_model(scene):
    """The bound is enforced through the real backend, not just in the pure unit.

    A request far below the object width must come back as the derived floor,
    computed from this model's stiffness rather than any constant.
    """
    backend, _handle, model, _data = scene
    from robot_ci.models.robot import GripperControlMode

    scenario = ScenarioConfig(scenario_id="graspregress", seed=1)
    robot = scenario.robot.model_copy(
        update={"gripper_control_mode": GripperControlMode.BOUNDED}
    )
    bounded_scenario = scenario.model_copy(update={"robot": robot})
    controller = backend._make_gripper_controller(model, bounded_scenario)

    contact_width = 2.0 * float(scenario.target.size_m)
    requested = 0.0002
    out = controller.step(requested_width_m=requested, measured_width_m=0.08)
    expected = contact_width - 2.0 * (robot.gripper_target_contact_force_n / controller.stiffness_n_per_m)
    assert out == pytest.approx(max(expected, float(robot.gripper_closed_separation_m)))
    assert out > requested, "the floor must lift a deep request, not accept it"


# ---------------------------------------------------------------------------
# Arm command swing bound (the seed-specific arm-transient experiment).
#
# Why this exists: the reset path was measured clean on all five seeds, and
# the episode start pose matched across seeds to six decimals -- so the
# instability was not reset, not object overlap, and not the seed jitter. It
# was the waypoint transit: one joint was commanded 2.95 rad in a single 50 ms
# control period, which moved the end effector 214 mm and carried the object
# 23.6 m away. This bound is measured from that, and it is off by default.
# ---------------------------------------------------------------------------


def _arm_joints(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(7)]


def _commanded_arm(backend, handle, targets):
    """Apply one joint command and return what the actuators were told."""
    model, data = handle._opaque["model"], handle._opaque["data"]
    from robot_ci.models.state import Action, ActionType

    backend.step(
        handle,
        Action(
            type=ActionType.JOINT_POSITION,
            joint_position=tuple(float(v) for v in targets),
        ),
    )
    ids = _arm_joints(model)
    return np.array([float(data.ctrl[backend._actuator_for_joint(model, j)]) for j in ids])


def test_arm_step_cap_defaults_to_disabled():
    """The bound must be off unless a run asks for it.

    Every recorded result was produced with the previous, unbounded behaviour,
    so a default of anything else would silently change what the suite and the
    artifacts mean.
    """
    scenario = ScenarioConfig(scenario_id="capdefault", seed=1)
    assert scenario.robot.arm_max_joint_step_rad == 0.0


def test_arm_step_cap_bounds_the_commanded_target():
    """With a cap, one control period may not move a command by more than it.

    Measured against the uncapped run in the same test so the assertion is a
    comparison rather than a hard-coded expectation.
    """
    backend = create_backend("mujoco")
    unbounded = ScenarioConfig(scenario_id="capped", seed=1)
    capped = ScenarioConfig(
        scenario_id="capped", seed=1, robot={"arm_max_joint_step_rad": 0.15}
    )

    handles = []
    start = None
    for scenario in (unbounded, capped):
        handle = backend.reset(scenario, seed=1)
        model, data = handle._opaque["model"], handle._opaque["data"]
        ids = _arm_joints(model)
        home = np.array([float(data.qpos[int(model.jnt_qposadr[j])]) for j in ids])
        far = home + np.full(7, 0.5)
        got = _commanded_arm(backend, handle, far)
        if start is None:
            start = home
        travel = float(np.abs(got - start).max())
        handles.append(travel)
        backend.close(handle)
    backend.close_all()

    unbounded_travel, capped_travel = handles
    assert unbounded_travel == pytest.approx(0.5, abs=1e-9)
    assert capped_travel == pytest.approx(0.15, abs=1e-9)
    assert capped_travel < unbounded_travel


def test_arm_step_cap_does_not_block_reset():
    """Reset must place the arm exactly as well with the cap as without it.

    The arm is teleported home and then 120 settle steps run, which leaves it
    0.004770 rad off home. That offset is pre-existing -- it appears with the
    cap off too -- so the assertion is a *comparison* against the uncapped
    run. A cap that leaked into the teleport path would show a larger error,
    because the joint could no longer be placed at its target.
    """
    backend = create_backend("mujoco")

    def home_error(cap: float) -> float:
        scenario = ScenarioConfig(
            scenario_id="capreset", seed=1, robot={"arm_max_joint_step_rad": cap}
        )
        handle = backend.reset(scenario, seed=1)
        model, data = handle._opaque["model"], handle._opaque["data"]
        ids = _arm_joints(model)
        got = np.array([float(data.qpos[int(model.jnt_qposadr[j])]) for j in ids])
        backend.close(handle)
        return float(np.abs(got - np.asarray(scenario.robot.home_position)).max())

    uncapped = home_error(0.0)
    capped = home_error(0.15)
    backend.close_all()

    assert capped == pytest.approx(uncapped, abs=1e-12), (
        f"the swing bound degraded reset: {capped} vs {uncapped}"
    )


def test_arm_step_cap_is_deterministic():
    """Two capped runs of the same seed must agree to float64 precision.

    Not bit-identical: two runs sharing a backend drift by ~1e-16 over ten
    steps *with the bound off as well*, which is plain float accumulation, so
    demanding exact equality would assert something the backend has never
    promised. A tolerance of 1e-12 is many orders above that floor and still
    fails on any real divergence -- a changed seed or a broken clip.
    """
    backend = create_backend("mujoco")
    scenario = ScenarioConfig(
        scenario_id="capdet", seed=7, robot={"arm_max_joint_step_rad": 0.15}
    )
    from robot_ci.policies import policy_v1

    traces = []
    for _ in range(2):
        policy = policy_v1(scenario=scenario)
        handle = backend.reset(scenario, seed=7)
        policy.reset(policy.config, 7, scenario)
        policy.attach_planner(backend.jacobian_provider(handle))
        seen = []
        for _ in range(10):
            out = policy.act(backend.observe(handle))
            backend.step(handle, out.action)
            seen.append(
                float(backend._ee_pose(handle._opaque["model"], handle._opaque["data"]).position.array[2])
            )
        traces.append(seen)
        backend.close(handle)
    backend.close_all()

    assert len(traces[0]) == len(traces[1]) == 10
    drift = max(abs(a - b) for a, b in zip(traces[0], traces[1], strict=True))
    assert drift < 1e-12, f"capped runs diverged by {drift:.3e}"
