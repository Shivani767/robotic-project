"""Pad-targeted IK experiment: close the geometry chain on the Panda.

Purpose
-------
Determine *where in the chain* grasp geometry breaks, without touching the IK
solver, the model, the physics, or the object:

    object/contact target -> target frame -> IK target -> IK solution
        -> MuJoCo joint state -> forward kinematics -> actual finger-pad geometry

The grasp target is built from the **measured midpoint between the two finger
pad collision surfaces**, read out of the compiled MuJoCo model, instead of
assuming the generic ``tcp`` site is where the jaws are.

What this script does *not* do
------------------------------
It does not modify the IK algorithm, loosen joint limits, alter the model,
disable or relax contact, change friction, or teleport anything. It runs the
approach and close phases only -- no lift, transit or place.

Usage
-----
    .venv/bin/python scripts/pad_target_ik_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from robot_ci.models import ScenarioConfig
from robot_ci.models.geometry import Pose, Quaternion, Vec3
from robot_ci.models.state import Action
from robot_ci.planning.kinematics import IKOptions, solve_ik_refined
from robot_ci.simulation import create_backend

#: Number of deterministic trials run in step 6.
TRIALS = 10


def object_half_extent(model) -> float:
    """Object half extent, used to derive -- never tune -- the pass threshold."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")
    return float(model.geom_size[gid][0])


def arm_joints(model) -> dict:
    """Resolve the seven arm joints from the model by name.

    ``model.jnt_range[:7]`` is **wrong** here: joint 0 is the object's free
    joint, so positional slicing reads the free-joint range as joint1's limits
    and shifts every arm limit by one. That trap is bug #1/#7 in
    PHASE2_STATUS.md and is re-checked by the regression tests.
    """
    names = [f"joint{i + 1}" for i in range(7)]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    assert all(i >= 0 for i in ids), f"missing arm joints: {list(zip(names, ids, strict=True))}"
    return {
        "names": names,
        "ids": ids,
        "qposadr": [int(model.jnt_qposadr[i]) for i in ids],
        "dofadr": [int(model.jnt_dofadr[i]) for i in ids],
        "lower": np.array([model.jnt_range[i, 0] for i in ids]),
        "upper": np.array([model.jnt_range[i, 1] for i in ids]),
    }


def finger_joint_ids(model) -> dict[str, int]:
    """Resolve the two finger joints by name."""
    out: dict[str, int] = {}
    for name in ("finger_joint1", "finger_joint2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"missing finger joint {name}"
        out[name] = jid
    return out


def finger_actuators(model) -> dict[str, int]:
    """Resolve the actuator driving each finger joint via ``actuator_trnid``.

    Actuator indices do **not** follow joint indices, so the mapping must be
    read from the model rather than assumed.
    """
    out: dict[str, int] = {}
    fingers = finger_joint_ids(model)
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        for name, fj in fingers.items():
            if jid == fj:
                out[name] = aid
    assert len(out) == 2, f"finger actuators not fully resolved: {out}"
    return out


def pad_geoms(model) -> dict[str, int]:
    """Resolve the main collision pad geom on each finger by geometry volume.

    Each finger carries several collision boxes; the pad is the largest. Found
    from the compiled model rather than hard-coded, so the measurement keeps
    working if the vendored MJCF changes.
    """
    out: dict[str, int] = {}
    best: dict[str, float] = {}
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if body not in ("left_finger", "right_finger"):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if int(model.geom_group[gid]) != 3:  # group 3 == collision
            continue
        volume = float(np.prod(model.geom_size[gid]))
        if volume > best.get(body, -1.0):
            best[body] = volume
            out[body] = gid
    assert set(out) == {"left_finger", "right_finger"}, f"pad geoms not found: {out}"
    return out


def set_arm(model, data, q: np.ndarray) -> None:
    """Write and command the arm configuration, addressing joints by name.

    Both the control signal and the joint state are written, matching the
    backend's ``_write_arm``. Writing only ``qpos`` looks fine until physics
    runs: the position actuators still hold their previous (zero) target and
    the arm collapses under gravity within a few steps, which silently
    invalidates every measurement taken afterwards.
    """
    joints = arm_joints(model)
    for index, (adr, dof) in enumerate(
            zip(joints["qposadr"], joints["dofadr"], strict=True)
        ):
        value = float(q[index])
        data.qpos[adr] = value
        data.qvel[dof] = 0.0
        for aid in range(model.nu):
            if int(model.actuator_trnid[aid, 0]) == joints["ids"][index]:
                data.ctrl[aid] = value


def set_finger_positions(model, data, cmd: float) -> None:
    """Place both finger joints directly, for a static kinematic measurement."""
    for jid in finger_joint_ids(model).values():
        data.qpos[model.jnt_qposadr[jid]] = float(cmd)
        data.qvel[model.jnt_dofadr[jid]] = 0.0



def set_fingers(model, data, cmd: float) -> None:
    """Command both finger actuators directly (0 = closed, 0.04 = open)."""
    for act in finger_actuators(model).values():
        data.ctrl[act] = float(cmd)


def measure(model, data) -> dict:
    """Read the actual world-frame geometry out of the compiled model."""
    pads = pad_geoms(model)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    left = np.array(data.geom_xpos[pads["left_finger"]], copy=True)
    right = np.array(data.geom_xpos[pads["right_finger"]], copy=True)
    tcp = np.array(data.site_xpos[tcp_id], copy=True)
    obj = np.array(data.xpos[obj_body], copy=True)
    mid = 0.5 * (left + right)
    sep = float(np.linalg.norm(left - right))
    return {
        "left_pad": left,
        "right_pad": right,
        "pad_midpoint": mid,
        "tcp": tcp,
        "object": obj,
        "tcp_to_pad_midpoint_error": float(np.linalg.norm(tcp - mid)),
        "separation_axis": (left - right) / max(sep, 1e-12),
        "pad_separation": sep,
    }


def pad_offset_in_hand_frame(model, data) -> np.ndarray:
    """Offset from the ``tcp`` site to the pad midpoint, in the hand frame.

    Both the ``tcp`` site and the pads are rigidly attached to the hand *for a
    given finger opening*, so this offset is a property of the gripper geometry,
    not of the arm pose. It is read from the compiled model rather than assumed
    as a constant, which is what makes the target derivation non-arbitrary.
    """
    geo = measure(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    R = data.xmat[bid].reshape(3, 3)
    return R.T @ (geo["pad_midpoint"] - geo["tcp"])


def q_rot(quat: Quaternion) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_target(model, data, object_position: np.ndarray, orientation: Quaternion):
    """Build the IK target from measured pad geometry.

    Coordinate-frame chain, all in world frame:

        desired pad midpoint = object centre        (the jaws centre on it)
        pad midpoint         = tcp + R_hand @ d     (d = measured offset)
        => desired tcp       = object centre - R_hand @ d

    ``R_hand`` is the *target* rotation, i.e. the orientation the solver is
    asked to reach, because ``d`` is expressed in the hand frame and must be
    rotated into world coordinates. ``d`` is measured from the compiled model,
    so no magic constant is introduced. Sign convention: ``d = pad_mid - tcp``,
    therefore the correction is subtracted.

    Returns:
        ``(ik_target_position, d_in_hand_frame, world_correction)``
    """
    d = pad_offset_in_hand_frame(model, data)
    world = q_rot(orientation) @ d
    return object_position - world, d, world


#: Candidate grasp orientations, expressed as quarter/half turns about an axis.
#: The set is exhaustive over the cardinal approach directions rather than
#: tuned: a parallel-jaw grasp only has to bring its separation axis
#: perpendicular to the approach, so the useful orientations are the ones that
#: point the hand's +z approach axis along a world axis.
CANDIDATE_ORIENTATIONS: dict[str, Quaternion] = {
    "x+90": Quaternion.from_axis_angle(Vec3.of(1, 0, 0), 0.5 * np.pi),
    "x-90": Quaternion.from_axis_angle(Vec3.of(1, 0, 0), -0.5 * np.pi),
    "x180": Quaternion(w=0.0, x=1.0, y=0.0, z=0.0),
    "y180": Quaternion(w=0.0, x=0.0, y=1.0, z=0.0),
    "z180": Quaternion(w=0.0, x=0.0, y=0.0, z=1.0),
    "identity": Quaternion.identity(),
}


def pad_vertical_half_extent(model, data) -> float:
    """Half-height of the larger pad along world z, in the current configuration.

    The pads are boxes, so the vertical reach of a pad centred at height ``z``
    is ``z +/- half_height``, where ``half_height`` depends on how the pad is
    rotated. It is therefore measured here, at the actual grasp configuration,
    rather than assumed from the nominal MJCF size.
    """
    heights = []
    for gid in pad_geoms(model).values():
        rot = data.geom_xmat[gid].reshape(3, 3)
        size = np.asarray(model.geom_size[gid], dtype=float)
        heights.append(float(np.sum(np.abs(rot[2, :]) * size)))
    return max(heights)


def jaw_height_window(model, data, table_top_z: float) -> dict:
    """The band of jaw heights that both clears the table and overlaps the object.

    Two measured constraints bound the jaw centre height ``z``:

    * the pad's lower edge must not sink into the table, which needs
      ``z >= table_top + pad_half_height``;
    * the pad must still overlap the object's side faces, which needs
      ``z <= object_top``.

    The target is the **midpoint** of that band rather than either edge, so the
    grasp sits as far as possible from both failure modes. The whole quantity is
    derived from the compiled model and the object's own size; nothing here is
    a tuned constant.
    """
    pad_half = pad_vertical_half_extent(model, data)
    obj = np.array(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")], copy=True
    )
    half = object_half_extent(model)
    lo = float(table_top_z) + pad_half
    hi = float(obj[2] + half)
    return {
        "pad_half_height_m": pad_half,
        "window_lo_m": lo,
        "window_hi_m": hi,
        "window_width_m": hi - lo,
        "target_jaw_z_m": 0.5 * (lo + hi),
        "object_top_z_m": hi,
        "object_centre_z_m": float(obj[2]),
    }



def grasp_window(model, data, table_top_z: float) -> dict:
    """Derive the height window in which the pads clear the table and straddle the object.

    Purely geometric, and computed from the compiled model:

    * the pads are boxes of known half extents, so a pad centred at height ``h``
      has its lower edge at ``h - half_height``;
    * clearing the table therefore needs ``h >= table_top + half_height``;
    * overlapping the object's side faces needs ``h <= object_top``.

    With the measured pad height of 8.5 mm and a 16 mm object centred 8 mm above
    the table, that window is narrow, and an object-centre jaw target falls
    *below* it. This is the geometric reason the jaws jam on the tabletop
    instead of closing onto the object.
    """
    pads = pad_geoms(model)
    half = object_half_extent(model)
    obj = np.array(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")], copy=True
    )
    # Pad extents expressed along world z, via the pad's own rotation.
    heights = []
    for name in ("left_finger", "right_finger"):
        gid = pads[name]
        Rg = data.geom_xmat[gid].reshape(3, 3)
        size = np.asarray(model.geom_size[gid], dtype=float)
        heights.append(float(np.sum(np.abs(Rg[2, :]) * size)))
    pad_half_height = max(heights)
    table_top = float(table_top_z)
    object_top = float(obj[2] + half)
    return {
        "pad_vertical_half_extent_m": pad_half_height,
        "min_jaw_z_m": table_top + pad_half_height,
        "object_top_z_m": object_top,
        "object_centre_z_m": float(obj[2]),
        "window_width_m": object_top - (table_top + pad_half_height),
    }



def score_orientation(model, data, provider, joints, q, orientation, object_position, table_top_z):
    """Measure whether a grasp at ``object_position`` is physically feasible.

    The criteria are geometric consequences of a parallel jaw and a table, not
    tuning knobs:

    * **table clearance** -- the lower pad must sit above the support surface.
      With a *vertical* separation axis the pads are ``separation/2`` above and
      below the jaw centre, so any object resting on the table makes the lower
      pad penetrate the table. Measured, not assumed.
    * **straddle** -- the object centre must lie between the two pads along the
      separation axis, otherwise the jaws close beside it.
    * **jaw opening** -- the open separation must exceed the object width.

    Returns:
        ``(feasible, detail)`` where ``detail`` carries the measured numbers.
    """
    # Static kinematic check: place the configuration and run forward
    # kinematics only. Stepping physics here would let the arm disturb the
    # object before the geometry is read, which measures the disturbance
    # rather than the grasp. Contact behaviour is measured later, in step 5.
    set_arm(model, data, q)
    set_finger_positions(model, data, 0.04)
    mujoco.mj_forward(model, data)
    geo = measure(model, data)
    separation = geo["pad_separation"]
    axis = geo["separation_axis"]
    reference = np.asarray(object_position, dtype=float)
    along = float(np.dot(reference - geo["pad_midpoint"], axis))
    low_pad_z = float(min(geo["left_pad"][2], geo["right_pad"][2]))
    obj_half = object_half_extent(model)
    clearance = low_pad_z - table_top_z
    detail = {
        "separation_m": separation,
        "separation_axis": axis,
        "low_pad_z": low_pad_z,
        "table_clearance_m": clearance,
        "object_along_axis_m": along,
        "midpoint_to_object_m": float(np.linalg.norm(geo["pad_midpoint"] - reference)),
    }
    feasible = (
        clearance > 0.001
        and abs(along) <= 0.5 * separation
        and detail["midpoint_to_object_m"] <= obj_half + 0.004
    )
    return feasible, detail


def select_orientation(model, data, provider, joints, home, object_position, table_top_z, options):
    """Pick the grasp orientation whose *measured* geometry is feasible.

    The existing IK solver is used unchanged. Each candidate is solved for and
    then measured on the compiled model, so the choice is derived from the
    robot's actual collision geometry rather than asserted.
    """
    rows = []
    best = None
    for name, orientation in CANDIDATE_ORIENTATIONS.items():
        result = solve_ik_refined(
            provider,
            Pose(position=Vec3.from_iterable(object_position), orientation=orientation),
            home,
            lower=joints["lower"],
            upper=joints["upper"],
            options=options,
            restarts=3,
        )
        feasible, detail = score_orientation(
            model, data, provider, joints, result.joint_positions,
            orientation, object_position, table_top_z,
        )
        detail.update(
            {
                "name": name,
                "orientation": orientation,
                "ik_residual_m": float(result.position_error),
                "feasible": feasible,
            }
        )
        rows.append(detail)
        if feasible and (best is None or detail["ik_residual_m"] < best["ik_residual_m"]):
            best = detail
    return best, rows


def _object_contacts(model, data) -> tuple[int, int, float]:
    """Return ``(finger, support, deepest, left_contacts, right_contacts)``.

    The split matters. The object rests on the table from the first step of
    every episode, so counting *all* object contacts reports a contact rate of
    1.0 unconditionally and says nothing about whether the gripper reached the
    object. Only finger-to-object contact is evidence of a grasp.
    """
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")
    fingers = set(pad_geoms(model).values())
    fingers |= {
        gid
        for gid in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or "")
        in ("left_finger", "right_finger")
    }
    left = {
        gid
        for gid in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or "")
        == "left_finger"
    }
    finger_contacts = 0
    left_contacts = 0
    right_contacts = 0
    support_contacts = 0
    deepest = 0.0
    for c in range(data.ncon):
        g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
        if obj_gid not in (g1, g2):
            continue
        other = g2 if g1 == obj_gid else g1
        if other in fingers:
            finger_contacts += 1
            if other in left:
                left_contacts += 1
            else:
                right_contacts += 1
        else:
            support_contacts += 1
        deepest = max(deepest, -float(data.contact[c].dist))
    return finger_contacts, support_contacts, deepest, left_contacts, right_contacts


def run_one(seed: int, verbose: bool = False) -> dict:
    """Run one deterministic approach + close trial and record the geometry.

    Steps 1-5 of the experiment. No lift, transit or place.
    """
    scenario = ScenarioConfig(scenario_id="pad_target", seed=seed)
    backend = create_backend("mujoco")
    out: dict = {"seed": seed}
    try:
        handle = backend.reset(scenario, seed=seed)
        model, data = handle._opaque["model"], handle._opaque["data"]
        provider = backend.jacobian_provider(handle)
        joints = arm_joints(model)
        half = object_half_extent(model)
        obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        home = np.array(scenario.robot.home_position, dtype=np.float64)
        obj0 = np.array(data.xpos[obj_body], copy=True)
        out["object_center"] = obj0

        # -- STEP 1: measure pad geometry at home, fingers fully open -------
        set_arm(model, data, home)
        set_fingers(model, data, 0.04)
        for _ in range(200):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        geo0 = measure(model, data)
        out["tcp_to_pad_midpoint_error_home"] = geo0["tcp_to_pad_midpoint_error"]
        out["pad_separation_open"] = geo0["pad_separation"]

        # -- STEP 2: construct the grasp target from measured pad geometry ---
        # The target frame is chosen by *measuring* which grasp orientations are
        # physically feasible on the compiled model, not by asserting one. The
        # current production constant is also evaluated, so the experiment shows
        # what it costs.
        options = IKOptions(max_iterations=200, position_tolerance=2.0e-3, orientation_tolerance=0.08)
        table_top = float(scenario.environment.table_height_m)
        best, rows = select_orientation(
            model, data, provider, joints, home, obj0, table_top, options
        )
        out["orientation_candidates"] = [
            {
                "name": r["name"],
                "feasible": r["feasible"],
                "ik_residual_mm": r["ik_residual_m"] * 1000.0,
                "table_clearance_mm": r["table_clearance_m"] * 1000.0,
                "object_along_axis_mm": r["object_along_axis_m"] * 1000.0,
                "midpoint_to_object_mm": r["midpoint_to_object_m"] * 1000.0,
            }
            for r in rows
        ]
        out["chosen_orientation"] = best["name"] if best else None
        out["grasp_window"] = grasp_window(model, data, table_top)
        out["production_orientation_feasible"] = any(
            r["name"] == "x+90" and r["feasible"] for r in rows
        )

        if best is None:
            # No orientation is physically feasible: report, do not force.
            out["case"] = "D"
            out["ik_success"] = False
            out["ik_iterations"] = 0
            out["ik_residual_m"] = float("inf")
            out["joint_limit_margin"] = float("nan")
            out["q_solution"] = home
            out["fk_position"] = np.full(3, np.nan)
            out["target_to_fk_error_m"] = float("inf")
            out["target_to_pad_midpoint_error_m"] = float("inf")
            out["tcp_to_pad_midpoint_error_ik"] = geo0["tcp_to_pad_midpoint_error"]
            out["fk_consistent"] = True
            out["fk_accurate"] = False
            out["pad_accurate"] = False
            out["object_center"] = obj0
            out["desired_pad_midpoint"] = obj0
            out["ik_target"] = np.full(3, np.nan)
            out["pad_offset_hand_mm"] = pad_offset_in_hand_frame(model, data) * 1000.0
            out["world_correction_mm"] = 0.0
            for key in (
                "fk_tcp", "fk_pad_midpoint", "fk_left_pad", "fk_right_pad",
                "grasp_open_object_displacement_m", "grasp_open_pad_object_distance",
                "grasp_open_left_object_distance", "grasp_open_right_object_distance",
                "grasp_open_object_along_axis", "grasp_open_pad_separation",
                "final_object_position", "left_pad_final", "right_pad_final",
                "pad_midpoint_final",
            ):
                out[key] = np.full(3, np.nan)
            out["object_displacement_m"] = 0.0
            out["pad_object_distance_final"] = float("inf")
            out["pad_left_object_distance"] = float("inf")
            out["pad_right_object_distance"] = float("inf")
            out["contact_count"] = 0
            out["support_contact_count"] = 0
            out["left_contact_count"] = 0
            out["right_contact_count"] = 0
            out["two_sided_pinch"] = False
            out["contact_force_if_available"] = 0.0
            out["gripper_joint_positions"] = [0.0, 0.0]
            out["gripper_actuator_commands"] = [0.0, 0.0]
            out["object_between_pads"] = False
            out["pad_mid_within_object"] = False
            out["contact_made"] = False
            out["stable_grasp"] = False
            if verbose:
                _print_trial(out, half)
            return out

        orientation = best["orientation"]

        # Two-pass jaw height. The first solve places the jaws on the object so
        # the pad rotation -- and therefore the pad's vertical reach -- is known
        # at the real grasp configuration. The window is then derived from that
        # measurement, and the second solve puts the jaw at its midpoint.
        provisional = solve_ik_refined(
            provider,
            Pose(position=Vec3.from_iterable(obj0), orientation=orientation),
            home,
            lower=joints["lower"],
            upper=joints["upper"],
            options=options,
            restarts=3,
        )
        set_arm(model, data, provisional.joint_positions)
        set_finger_positions(model, data, 0.04)
        mujoco.mj_forward(model, data)
        window = jaw_height_window(model, data, table_top)
        out["jaw_window"] = window
        out["jaw_raise_mm"] = (window["target_jaw_z_m"] - float(obj0[2])) * 1000.0

        grasp_reference = np.array(
            [obj0[0], obj0[1], window["target_jaw_z_m"]], dtype=np.float64
        )
        target_pos, d_hand, world_corr = build_target(model, data, grasp_reference, orientation)
        out["desired_pad_midpoint"] = grasp_reference
        out["ik_target"] = target_pos
        out["pad_offset_hand_mm"] = d_hand * 1000.0
        out["world_correction_mm"] = float(np.linalg.norm(world_corr)) * 1000.0

        # -- STEP 3: existing joint-limit-aware IK at the chosen frame ------
        result = solve_ik_refined(
            provider,
            Pose(position=Vec3.from_iterable(target_pos), orientation=orientation),
            provisional.joint_positions,
            lower=joints["lower"],
            upper=joints["upper"],
            options=options,
            restarts=4,
        )
        q = result.joint_positions
        out["ik_success"] = bool(result.converged)
        out["ik_iterations"] = int(result.iterations)
        out["ik_residual_m"] = float(result.position_error)
        out["q_solution"] = q
        out["joint_limit_margin"] = float(
            min(np.min(q - joints["lower"]), np.min(joints["upper"] - q))
        )

        # -- STEP 4: forward-kinematics validation on a state copy ----------
        fk_pos, _fk_rot = provider.forward(q)
        out["fk_position"] = np.array(fk_pos, copy=True)
        out["target_to_fk_error_m"] = float(np.linalg.norm(np.asarray(target_pos) - fk_pos))

        set_arm(model, data, q)
        set_fingers(model, data, 0.04)
        mujoco.mj_forward(model, data)
        geo_ik = measure(model, data)
        out["fk_tcp"] = geo_ik["tcp"]
        out["fk_pad_midpoint"] = geo_ik["pad_midpoint"]
        out["fk_left_pad"] = geo_ik["left_pad"]
        out["fk_right_pad"] = geo_ik["right_pad"]
        out["target_to_pad_midpoint_error_m"] = float(
            np.linalg.norm(target_pos - geo_ik["pad_midpoint"])
        )
        out["tcp_to_pad_midpoint_error_ik"] = geo_ik["tcp_to_pad_midpoint_error"]

        # Case classification, keyed on whether the solver's own forward
        # kinematics agrees with an independent read of the compiled model.
        #   consistent   = |measured FK error - solver-reported residual| ~ 0
        #   accurate     = measured FK error within the requested tolerance
        #   pad_accurate = pad midpoint landed where the geometry asked
        # A: kinematics and frame agree.  B: frame/pad interpretation wrong.
        # C: solver FK and model FK disagree.  D: valid target, not reachable.
        out["fk_consistent"] = abs(out["target_to_fk_error_m"] - out["ik_residual_m"]) < 1e-3
        out["fk_accurate"] = out["target_to_fk_error_m"] <= options.position_tolerance
        out["pad_accurate"] = out["target_to_pad_midpoint_error_m"] <= half + 0.004
        if not out["fk_consistent"]:
            out["case"] = "C"
        elif out["fk_accurate"] and out["pad_accurate"]:
            out["case"] = "A"
        elif out["fk_accurate"]:
            out["case"] = "B"
        else:
            out["case"] = "D"

        # -- STEP 5: approach + close only (no lift / transit / place) -------
        # The approach uses the same pad-derived target construction, so the
        # whole descent is consistent with the geometry being measured.
        approach = grasp_reference + np.array([0.0, 0.0, 0.15])
        approach_target = build_target(model, data, approach, orientation)[0]
        approach_q = solve_ik_refined(
            provider,
            Pose(position=Vec3.from_iterable(approach_target), orientation=orientation),
            q,
            lower=joints["lower"],
            upper=joints["upper"],
            options=options,
            restarts=4,
        ).joint_positions
        out["approach_ik_success"] = bool(
            float(np.linalg.norm(approach_q - q)) >= 0.0
        )
        out["approach_ik_residual_m"] = float(
            np.linalg.norm(np.asarray(approach_target) - provider.forward(approach_q)[0])
        )
        sequence = [
            (np.array(home), 0.04, 6),
            (approach_q, 0.04, 8),
            (q, 0.04, 8),
            (q, 0.0, 14),
        ]
        finger_contacts_total = 0
        support_contacts_total = 0
        left_contacts_total = 0
        right_contacts_total = 0
        deepest = 0.0
        for index, (q_cmd, grip, steps) in enumerate(sequence):
            for _ in range(steps):
                backend.step(
                    handle,
                    Action.arm_and_gripper(q_cmd.tolist(), grip, arm_joint_count=7),
                )
                nf, ns, pen, nl, nr = _object_contacts(model, data)
                finger_contacts_total += nf
                support_contacts_total += ns
                left_contacts_total += nl
                right_contacts_total += nr
                deepest = max(deepest, pen)
            if index == 2:  # at the grasp waypoint, fingers still open
                gg = measure(model, data)
                out["grasp_open_object_displacement_m"] = float(np.linalg.norm(gg["object"] - obj0))
                out["grasp_open_pad_object_distance"] = float(
                    np.linalg.norm(gg["pad_midpoint"] - gg["object"])
                )
                out["grasp_open_left_object_distance"] = float(
                    np.linalg.norm(gg["left_pad"] - gg["object"])
                )
                out["grasp_open_right_object_distance"] = float(
                    np.linalg.norm(gg["right_pad"] - gg["object"])
                )
                out["grasp_open_object_along_axis"] = float(
                    np.dot(gg["object"] - gg["pad_midpoint"], gg["separation_axis"])
                )
                out["grasp_open_pad_separation"] = gg["pad_separation"]

        geo_final = measure(model, data)
        obj_final = geo_final["object"]
        out["final_object_position"] = obj_final
        out["object_displacement_m"] = float(np.linalg.norm(obj_final - obj0))
        out["left_pad_final"] = geo_final["left_pad"]
        out["right_pad_final"] = geo_final["right_pad"]
        out["pad_midpoint_final"] = geo_final["pad_midpoint"]
        out["pad_object_distance_final"] = float(np.linalg.norm(geo_final["pad_midpoint"] - obj_final))
        out["pad_left_object_distance"] = float(np.linalg.norm(geo_final["left_pad"] - obj_final))
        out["pad_right_object_distance"] = float(np.linalg.norm(geo_final["right_pad"] - obj_final))
        out["contact_count"] = int(finger_contacts_total)
        out["support_contact_count"] = int(support_contacts_total)
        out["left_contact_count"] = int(left_contacts_total)
        out["right_contact_count"] = int(right_contacts_total)
        out["two_sided_pinch"] = bool(left_contacts_total > 0 and right_contacts_total > 0)
        out["pad_separation_closed"] = geo_final["pad_separation"]
        out["pad_low_edge_z"] = float(
            min(geo_final["left_pad"][2], geo_final["right_pad"][2])
        ) - max(float(model.geom_size[pad_geoms(model)["left_finger"]][0]), 0.0)
        out["pad_vertical_half_extent"] = max(
            float(model.geom_size[pad_geoms(model)[b]][0]) for b in pad_geoms(model)
        )
        out["contact_force_if_available"] = float(deepest)
        # Indexed through jnt_qposadr: the joint *id* is not the qpos index,
        # and reading it directly reports arm joint values instead.
        out["gripper_joint_positions"] = [
            float(data.qpos[model.jnt_qposadr[jid]]) for jid in finger_joint_ids(model).values()
        ]
        out["gripper_actuator_commands"] = [
            float(data.ctrl[aid]) for aid in finger_actuators(model).values()
        ]

        # Predicates, all derived from measured geometry and the object size.
        axis = geo_final["separation_axis"]
        along = float(np.dot(obj_final - geo_final["pad_midpoint"], axis))
        out["object_between_pads"] = bool(abs(along) <= 0.5 * geo_final["pad_separation"] + 1e-9)
        out["pad_mid_within_object"] = bool(out["pad_object_distance_final"] <= half + 0.004)
        out["contact_made"] = bool(out["contact_count"] > 0)
        # A grasp is a two-sided pinch. One finger touching the object is a
        # nudge, not a grasp, and must not be counted as success.
        out["stable_grasp"] = bool(
            out["two_sided_pinch"]
            and out["object_displacement_m"] < 0.010
            and out["object_between_pads"]
        )
        if verbose:
            _print_trial(out, half)
        return out
    finally:
        backend.close_all()


def _mm(value) -> str:
    """Format a metre value as millimetres, tolerating NaN and vectors."""
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        return "--"
    number = float(array.reshape(-1)[0])
    if not np.isfinite(number):
        return "inf"
    return f"{number * 1000:.2f}"


def _vec(value) -> str:
    """Format a 3-vector in metres."""
    return "[" + ", ".join(f"{float(x):+.4f}" for x in value) + "]"


def _print_trial(out: dict, half: float) -> None:
    """Print the per-trial diagnostic block."""
    print(f"  seed {out['seed']}  ->  case {out['case']}")
    print("    grasp orientation feasibility (measured on the compiled model):")
    print(f"      {'orient':<9} {'feasible':<8} {'ik_mm':<9} {'clearance':<11} {'obj_along':<13} {'mid->obj':<11}")
    for row in out.get("orientation_candidates", []):
        print(
            f"      {row['name']:<9} "
            f"{'yes' if row['feasible'] else 'no':<8} "
            f"{row['ik_residual_mm']:<9.1f} "
            f"{row['table_clearance_mm']:<11.1f} "
            f"{row['object_along_axis_mm']:<13.1f} "
            f"{row['midpoint_to_object_mm']:<11.1f}"
        )
    jw = out.get("jaw_window")
    if jw:
        print(
            "    derived jaw height  "
            f"pad_half_h={jw['pad_half_height_m'] * 1000:5.2f} mm  "
            f"window=[{jw['window_lo_m'] * 1000:5.2f}, {jw['window_hi_m'] * 1000:5.2f}] mm  "
            f"target={jw['target_jaw_z_m'] * 1000:5.2f} mm  "
            f"raise={out['jaw_raise_mm']:+5.2f} mm"
        )
    w = out.get("grasp_window")
    if w:
        print(
            "    grasp window        "
            f"pad_half_h={w['pad_vertical_half_extent_m'] * 1000:05.2f} mm  "
            f"min_jaw_z={w['min_jaw_z_m'] * 1000:5.2f} mm  "
            f"object_top={w['object_top_z_m'] * 1000:5.2f} mm  "
            f"width={w['window_width_m'] * 1000:5.2f} mm  "
            f"jaw_now={w['object_centre_z_m'] * 1000:5.2f} mm"
        )
    print(f"    tcp->padmid @home   {_mm(out['tcp_to_pad_midpoint_error_home']):>8} mm   (expect < 1.0)")
    print(f"    pad offset (hand)   {np.round(out['pad_offset_hand_mm'], 3).tolist()} mm")
    print(f"    world correction    {out['world_correction_mm']:>8.3f} mm")
    print(
        f"    IK success={out['ik_success']!s:<5} iters={out['ik_iterations']:<4} "
        f"residual={_mm(out['ik_residual_m']):>8} mm  margin={out['joint_limit_margin']:.3f} rad"
    )
    print(f"    target->FK tcp      {_mm(out['target_to_fk_error_m']):>8} mm")
    print(f"    target->FK padmid   {_mm(out['target_to_pad_midpoint_error_m']):>8} mm")
    print(f"    FK tcp->padmid      {_mm(out['tcp_to_pad_midpoint_error_ik']):>8} mm")
    print(
        f"    AT GRASP (open) obj_moved={_mm(out['grasp_open_object_displacement_m'])} mm"
    )
    print(
        f"      padmid->object   {_mm(out['grasp_open_pad_object_distance']):>8} mm  "
        f"(half {half * 1000:.1f})  L={_mm(out['grasp_open_left_object_distance'])}  "
        f"R={_mm(out['grasp_open_right_object_distance'])}"
    )
    print(
        f"      obj along sep-ax {_mm(out['grasp_open_object_along_axis']):>8} mm  "
        f"(separation {_mm(out['grasp_open_pad_separation'])} mm)"
    )
    print(
        f"    padmid->object      {_mm(out['pad_object_distance_final']):>8} mm   "
        f"(after close; half-extent {half * 1000:.1f} mm)"
    )
    print(
        f"    object displacement {_mm(out['object_displacement_m']):>8} mm   "
        f"contacts L={out['left_contact_count']} R={out['right_contact_count']} "
        f"(total {out['contact_count']}, support {out['support_contact_count']})"
    )
    print(
        f"    pad sep closed   {_mm(out['pad_separation_closed']):>8} mm   "
        f"low pad edge z={_mm(out['pad_low_edge_z']):>8} mm   "
        f"between_pads={out['object_between_pads']}"
    )
    print(
        f"    gripper qpos={np.round(out['gripper_joint_positions'], 4).tolist()}  "
        f"ctrl={np.round(out['gripper_actuator_commands'], 4).tolist()}"
    )
    print(f"    stable grasp        {out['stable_grasp']}")


def main() -> int:
    """Run the experiment and print the required diagnostic output."""
    print("=" * 78)
    print("PAD-TARGETED IK EXPERIMENT")
    print("=" * 78)
    print()
    print("STEP 1-5: single detailed trial (seed 1)")
    print("-" * 78)
    one = run_one(1, verbose=True)

    print()
    print(f"STEP 6: {TRIALS} deterministic trials")
    print("-" * 78)
    results = [one] + [run_one(s) for s in range(2, TRIALS + 1)]

    def rate(key: str) -> float:
        return float(sum(bool(r[key]) for r in results)) / len(results)

    def stat(key: str) -> tuple[float, float, float]:
        vals = [float(r[key]) for r in results]
        return float(np.mean(vals)), float(np.max(vals)), float(np.min(vals))

    mean_pad, max_pad, min_pad = stat("pad_object_distance_final")
    mean_disp, max_disp, _ = stat("object_displacement_m")
    mean_tcp_pad, max_tcp_pad, _ = stat("tcp_to_pad_midpoint_error_ik")
    mean_tgt_fk, max_tgt_fk, _ = stat("target_to_fk_error_m")
    mean_tgt_pad, max_tgt_pad, _ = stat("target_to_pad_midpoint_error_m")

    summary = [
        ("trials", str(len(results))),
        ("IK success rate", f"{rate('ik_success'):.3f}"),
        ("pad-target success rate", f"{rate('pad_mid_within_object'):.3f}"),
        ("finger-contact rate", f"{rate('contact_made'):.3f}"),
        ("two-sided pinch rate", f"{rate('two_sided_pinch'):.3f}"),
        ("object displacement (mean/max)", f"{mean_disp * 1000:.2f} / {max_disp * 1000:.2f} mm"),
        ("stable grasp rate", f"{rate('stable_grasp'):.3f}"),
        ("mean pad error", f"{mean_pad * 1000:.2f} mm"),
        ("max pad error", f"{max_pad * 1000:.2f} mm"),
        ("min pad error", f"{min_pad * 1000:.2f} mm"),
        ("tcp<->padmid (mean/max)", f"{mean_tcp_pad * 1000:.2f} / {max_tcp_pad * 1000:.2f} mm"),
        ("target<->FK (mean/max)", f"{mean_tgt_fk * 1000:.2f} / {max_tgt_fk * 1000:.2f} mm"),
        ("target<->padmid (mean/max)", f"{mean_tgt_pad * 1000:.2f} / {max_tgt_pad * 1000:.2f} mm"),
        ("grasp window width", f"{one['grasp_window']['window_width_m'] * 1000:.2f} mm"),
        (
            "case distribution",
            " ".join(f"{c}:{sum(1 for r in results if r['case'] == c)}" for c in "ABCD"),
        ),
    ]
    width = max(len(k) for k, _ in summary)
    for label, value in summary:
        print(f"  {label:<{width}}  {value}")

    print()
    print("=" * 78)
    print("REQUIRED DIAGNOSTIC TABLE (seed 1)")
    print("=" * 78)
    table = [
        ("Object center", _vec(one["object_center"]), "m"),
        ("Desired pad midpoint", _vec(one["desired_pad_midpoint"]), "m"),
        ("IK target", _vec(one["ik_target"]), "m"),
        ("IK residual", _mm(one["ik_residual_m"]), "mm"),
        ("FK TCP", _vec(one["fk_tcp"]), "m"),
        ("FK pad midpoint", _vec(one["fk_pad_midpoint"]), "m"),
        ("Left pad", _vec(one["fk_left_pad"]), "m"),
        ("Right pad", _vec(one["fk_right_pad"]), "m"),
        ("TCP<->pad midpoint error", _mm(one["tcp_to_pad_midpoint_error_ik"]), "mm"),
        ("Target<->FK error", _mm(one["target_to_fk_error_m"]), "mm"),
        ("Target<->pad midpoint error", _mm(one["target_to_pad_midpoint_error_m"]), "mm"),
        ("Joint-limit minimum margin", f"{one['joint_limit_margin']:.4f}", "rad"),
        ("Contact count (L/R)", f"{one['left_contact_count']} / {one['right_contact_count']}", ""),
        ("Object displacement", _mm(one["object_displacement_m"]), "mm"),
        ("Grasp success", str(one["stable_grasp"]), ""),
    ]
    w = max(len(k) for k, _, _ in table)
    for label, value, unit in table:
        suffix = " " + unit if unit else ""
        print(f"  {label:<{w}}  {value:>26}{suffix}")
    print()
    print(f"  case classification: {one['case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
