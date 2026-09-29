"""MuJoCo environment construction.

The MJCF is generated **programmatically from a ScenarioConfig** rather than
loaded from a static file. That is a deliberate architectural choice:

* every experimental condition (lighting, object placement, friction, clutter,
  obstacles, camera pose, resolution) becomes a visible part of the scene
  description, so it lands in the scenario's ``config_hash``;
* there is no second source of truth (an MJCF file could disagree with the
  scenario YAML);
* adding an object category does not require editing XML.

The generated model is written to a cache directory keyed by its hash so that
repeated runs of the same scenario reuse a compiled model, which is a large win
for batch sweeps.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from robot_ci.core.errors import SimulationError
from robot_ci.core.hashing import hash_payload
from robot_ci.models.environment import ObjectCategory
from robot_ci.models.scenario import ScenarioConfig

__all__ = [
    "build_scene_xml",
    "object_geoms",
    "scene_cache_key",
]

#: Half-extents (x, y, z) in metres per object category. The spec's objects are
#: a red cube, a blue cube, a green cylinder and a yellow can; the sizes below
#: are chosen so all four are graspable by a Panda parallel gripper at the
#: nominal 80 mm aperture.
_CATEGORY_HALF_EXTENTS: dict[ObjectCategory, tuple[float, float, float]] = {
    ObjectCategory.CUBE: (0.025, 0.025, 0.025),
    ObjectCategory.CYLINDER: (0.022, 0.022, 0.045),
    ObjectCategory.SPHERE: (0.024, 0.024, 0.024),
    ObjectCategory.CAN: (0.033, 0.033, 0.050),
    ObjectCategory.LIFTER: (0.006, 0.050, 0.075),
}


def object_geoms(category: ObjectCategory) -> tuple[tuple[float, float, float], str]:
    """Return ``(half_extents, mujoco_geom_type)`` for a category.

    Args:
        category: The object class.

    Returns:
        Half-extents in metres and the MJCF geom type string.
    """
    extents = _CATEGORY_HALF_EXTENTS[category]
    geom_type = "cylinder" if category is ObjectCategory.CYLINDER else (
        "sphere" if category is ObjectCategory.SPHERE else "box"
    )
    return extents, geom_type


def _f(value: float) -> str:
    """Format a float compactly for XML output."""
    return f"{float(value):.6g}"


def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    """Return the ``xyaxes`` triple pair orienting a MuJoCo camera.

    MuJoCo cameras look down their local **-z** axis with **+y** up. The
    ``xyaxes`` attribute is ``(x-axis, y-axis)`` of the camera frame expressed in
    the parent frame, so we construct an orthonormal basis from the view
    direction.

    Args:
        eye: Camera position in the world frame.
        target: Point the camera should look at.

    Returns:
        ``(x0, y0, z0, x1, y1, z1)`` for ``xyaxes="..."``.
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    forward = forward / max(norm, 1e-9)
    # Camera -z is the view direction, so camera +z points backwards.
    z_axis = -forward
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    return tuple(  # type: ignore[return-value]
        float(v) for v in (*x_axis, *y_axis)
    )


#: (x, y, z) offset of each joint frame from its parent, in metres.
#: Per-joint body pose relative to its parent, following the Franka Emika Panda
#: kinematic structure: ``(pos_xyz, axisangle_xyz_angle, capsule_length,
#: capsule_radius)``. The alternating quarter-turns about x and y are what turn
#: the DH serial chain into a human-like arm; a chain of parallel z-axes would
#: be kinematically correct but physically absurd (it would be a flagpole).
#: Per-joint body frame relative to its parent, following the Franka Emika Panda
#: kinematic structure. Each row is ``(pos_xyz, euler_xyz, axisangle_xyz_angle,
#: capsule_length, capsule_radius)`` and at most one of ``euler``/``axisangle`` is
#: non-zero. The frames alternate quarter-turns about x and y, which is what
#: turns a DH serial chain into an arm that can actually reach a table.
_CHAIN: tuple[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float], float, float], ...] = (
    # joint 1 - base yaw, 333 mm above the mounting plate
    ((0.000, 0.000, 0.333), (1.5707963267948966, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.150, 0.045),
    # joint 2 - shoulder pitch
    ((0.000, 0.000, 0.000), (1.5707963267948966, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.150, 0.042),
    # joint 3 - elbow, 316 mm along the upper arm
    ((0.000, 0.000, 0.316), (1.5707963267948966, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.130, 0.040),
    # joint 4 - forearm roll, laterally offset by a/4
    ((0.0825, 0.0825, 0.000), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.785398163397), 0.120, 0.038),
    # joint 5 - wrist pitch, 384 mm along the forearm
    ((-0.0825, 0.384, 0.000), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -0.785398163397), 0.100, 0.035),
    # joint 6 - wrist roll
    ((0.0880, 0.000, 0.000), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.785398163397), 0.080, 0.032),
    # joint 7 - flange roll
    ((0.107, 0.000, 0.000), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -0.785398163397), 0.050, 0.028),
)

#: Every arm joint rotates about its own local z axis; the body frames above
#: supply all the structure.
_JOINT_AXIS = "0 0 1"

#: Capsule length (m) and radius (m) of each arm link, used for collision notes.
_LINK_LENGTHS = tuple(row[3] for row in _CHAIN)
_LINK_RADII = tuple(row[4] for row in _CHAIN)


def _frame_attr(
    pos: tuple[float, float, float],
    euler: tuple[float, float, float],
    axisangle: tuple[float, float, float, float],
) -> str:
    """Render a body's ``pos`` plus whichever orientation attribute is set.

    MuJoCo's ``axisangle`` is ``axis_x axis_y axis_z angle`` -- an *axis*, not a
    rotation vector. Encoding it as a vector silently produces a zero rotation,
    which collapses the kinematic structure, so the two are kept distinct here.
    """
    x, y, z = pos
    attributes = f'pos="{_f(x)} {_f(y)} {_f(z)}"'
    if any(abs(v) > 1e-12 for v in euler):
        attributes += f' euler="{" ".join(_f(v) for v in euler)}"'
    elif abs(axisangle[3]) > 1e-12:
        attributes += f' axisangle="{" ".join(_f(v) for v in axisangle)}"'
    return attributes


# --- Collision classes -------------------------------------------------------
#
# MuJoCo collides two geoms when (contype_a & conaffinity_b) or
# (contype_b & conaffinity_a) is non-zero. The classes below are chosen so that
# the contacts this project actually studies are enabled, and the ones that are
# not, are not.
#
#   class      contype  conaffinity   collides with
#   ---------  -------  -----------   -----------------------------------------
#   world      0b0100   0b1000        objects, gripper      (NOT arm links)
#   object     0b0010   0b1100        world, gripper, links
#   gripper    0b1000   0b0100        world, objects        (NOT arm links)
#   arm link   0b0001   0b0010        objects               (NOT world/gripper)
#
# Why arm links are excluded from table and gripper contact
# --------------------------------------------------------
# The links here are capsule *approximations* of a Panda, not the official
# collision meshes. At ordinary manipulation poses the approximated forearm
# grazes the table edge, and the resulting constraint forces (hundreds of
# newton-metres) are indistinguishable from a control failure -- they would
# dominate the contact metrics and mask the behaviour under study.
#
# Real deployments check arm/table clearance in the *motion planner*
# (MoveIt 2, Phase 11), not in contact dynamics. Excluding the approximation
# from that check here is a deliberate, recorded simplification, and it makes
# the remaining contact counts a measure of the policy's behaviour rather than
# of the fidelity of a hand-written collision model.
_CONTYPE = {
    "world": 0b0100,
    "object": 0b0010,
    "gripper": 0b1000,
    "link": 0b0001,
}
_CONAFFINITY = {
    "world": 0b1000,
    "object": 0b1100,
    "gripper": 0b0100,
    "link": 0b0010,
}


def _mask(name: str) -> str:
    """Return the ``contype``/``conaffinity`` attribute pair for a class."""
    return f'contype="{_CONTYPE[name]}" conaffinity="{_CONAFFINITY[name]}"'


def _link_geom_xml(index: int, radius: float) -> str:
    """Return the collision geom for arm link ``index``.

    The capsule is oriented along the offset from this body to the next one,
    which is the true direction of the link segment. When that offset is zero
    (a co-located joint pair) a sphere at the joint is used instead, which is
    frame-independent and therefore always correct.
    """
    nxt = _CHAIN[index + 1][0] if index + 1 < len(_CHAIN) else (0.0, 0.0, _LINK_LENGTHS[index])
    norm = math.sqrt(sum(component * component for component in nxt))
    if norm < 1e-6:
        return (
            f'<geom name="link{index + 1}_geom" type="sphere" size="{_f(radius)}" '
            f'rgba="0.72 0.74 0.78 1" ' + _mask("link") + '/>'
        )
    axis = " ".join(_f(c / norm) for c in nxt)
    return (
        f'<geom name="link{index + 1}_geom" type="capsule" '
        f'fromto="0 0 0 0 0 {_f(norm)}" size="{_f(radius)}" zaxis="{axis}" '
        f'rgba="0.72 0.74 0.78 1" ' + _mask("link") + '/>'
    )


def _build_robot_xml(scenario: ScenarioConfig) -> tuple[str, str]:
    """Emit a 7-DOF serial arm with a parallel gripper as MJCF.

    Joints and actuators are named ``joint1..joint7`` / ``act1..act7``, matching
    :attr:`RobotConfig.arm_joint_names`, so the mapping from a policy's joint
    index to a MuJoCo actuator is the identity.

    Returns:
        ``(worldbody_xml, actuator_xml)``. The blocks are returned separately
        because MJCF requires ``<actuator>`` to be a sibling of ``<worldbody>``.
    """
    dof = scenario.robot.dof
    if dof != 7:
        raise SimulationError(
            f"MuJoCoBackend currently models a 7-DOF arm; scenario requests dof={dof}. "
            "The UR5e embodiment is not yet available in the MuJoCo backend.",
            code="simulation.embodiment_unsupported",
            requested_dof=dof,
        )

    lower = scenario.robot.limits.lower
    upper = scenario.robot.limits.upper
    kv = scenario.robot.limits.max_velocity
    # Position-servo gains. These must be stiff enough that the arm tracks
    # an IK solution within a few control steps; the Panda's real low-level
    # controller runs at ~1 kHz with far higher gains plus full gravity
    # compensation, so these are a deliberate simplification, noted as such.
    kp = (900.0,) * 7
    damping = (12.0,) * 7

    # Collision groups: the arm is group 2, the world (table, objects,
    # obstacles) is group 1. contype/conaffinity are chosen so the arm
    # collides with the world but not with itself. Self-collision is a
    # motion-planning concern; modelling it in contact dynamics would let a
    # merely-folded arm generate constraint forces that look like a control
    # failure. It is listed as a simplification, not silently omitted.

    # The whole arm is mounted on a pedestal at the configured base position.
    base = scenario.robot.base_position
    body: list[str] = [f'    <body name="arm_base" pos="{_f(base.x)} {_f(base.y)} {_f(base.z)}">',
                       '    <body name="link0" pos="0 0 0">']

    # Link geometry.
    #
    # A capsule's ``fromto`` is expressed in its body's frame, and the chain's
    # frames carry quarter-turn rotations, so a capsule along local z would run
    # sideways and could reach through the table. Each capsule is therefore
    # oriented with ``zaxis`` along the offset that actually leads to the *next*
    # body, which is the true direction of the link segment.
    #
    # Self-collision between arm links is disabled via contype/conaffinity
    # groups: a planned trajectory is assumed kinematically feasible, and
    # modelling self-collision in contact dynamics would let a merely folded arm
    # produce constraint forces that look like a control failure.
    indent = "      "
    body.append(
        '      <geom name="link0_geom" type="sphere" size="0.048" '
        'rgba="0.25 0.25 0.28 1" ' + _mask("link") + '/>'
    )
    for index in range(7):
        pos, euler, axisangle, _length, radius = _CHAIN[index]
        body.append(f'{indent}<body name="link{index + 1}" {_frame_attr(pos, euler, axisangle)}>')
        indent += "  "
        body.append(
            f'{indent}<joint name="joint{index + 1}" type="hinge" axis="{_JOINT_AXIS}" '
            f'range="{_f(lower[index])} {_f(upper[index])}" damping="{_f(damping[index])}" '
            f'armature="0.1"/>'
        )
        body.append(
            f'{indent}{_link_geom_xml(index, radius)}'
        )

    # Flange and parallel gripper, nested inside the last arm link.
    body.append(f'{indent}<body name="hand" pos="0 0 0.048">')
    indent += "  "
    body.append(
        f'{indent}<geom name="hand_geom" type="box" size="0.030 0.030 0.026" '
        f'rgba="0.20 0.20 0.22 1" ' + _mask("gripper") + '/>'
    )
    body.append(
        f'{indent}<site name="ee" pos="0 0 0.060" size="0.006" rgba="0 0 0 0" type="sphere"/>'
    )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        body.append(f'{indent}<body name="{side}_finger" pos="0 {_f(0.035 * sign)} 0.028">')
        indent += "  "
        finger_range = "0 0.04" if sign > 0 else "-0.04 0"
        body.append(
            f'{indent}<joint name="{side}_finger_joint" type="slide" axis="0 1 0" '
            f'range="{finger_range}" damping="5"/>'
        )
        body.append(
            f'{indent}<geom name="{side}_finger_geom" type="box" size="0.009 0.009 0.026" '
            f'rgba="0.85 0.30 0.10 1" ' + _mask("gripper") + '/>'
        )
        body.append(f"{indent[:-2]}</body>")
        indent = indent[:-2]
    body.append(f"{indent[:-2]}</body>")   # close hand
    indent = indent[:-2]
    for _ in range(7):                   # close link7 .. link1
        body.append(f"{indent}</body>")
        indent = indent[:-2]
    body.append(f"{indent}</body>")       # close link0
    body.append("    </body>")            # close arm_base

    actuator: list[str] = ["  <actuator>"]
    for index in range(7):
        kv_i = kv[index] if index < len(kv) else 2.0
        actuator.append(
            f'    <position name="act{index + 1}" joint="joint{index + 1}" kp="{_f(kp[index])}" '
            f'kv="{_f(kv_i)}" ctrlrange="{_f(lower[index])} {_f(upper[index])}" forcerange="-150 150"/>'
        )
    actuator.append('    <position name="left_finger_act" joint="left_finger_joint" kp="600" kv="12" ctrlrange="0 0.04" forcerange="-40 40"/>')
    actuator.append('    <position name="right_finger_act" joint="right_finger_joint" kp="600" kv="12" ctrlrange="-0.04 0" forcerange="-40 40"/>')
    actuator.append("  </actuator>")

    return "\n".join(body), "\n".join(actuator)


def build_scene_xml(scenario: ScenarioConfig) -> str:
    """Build the complete MJCF document for a scenario.

    The document is assembled from explicit sections because MJCF requires
    ``<actuator>`` to be a *sibling* of ``<worldbody>``, not a child of it.

    Args:
        scenario: The experimental condition.

    Returns:
        A self-contained MJCF document as a string.

    Raises:
        SimulationError: If the scenario requests an embodiment this builder
            does not model yet.
    """
    env = scenario.environment
    intrinsics = scenario.sensors.effective_intrinsics()
    height, width = scenario.resolution
    fov_deg = float(
        np.degrees(2.0 * np.arctan((max(width, height) / 2.0) / max(intrinsics.fx, 1e-6)))
    )
    eye = " ".join(_f(v) for v in scenario.camera_position.array)
    cam_axes = " ".join(_f(v) for v in _look_at_quat(scenario.camera_position.array, scenario.camera_look_at.array))
    extent_x, extent_y = env.table_extent_m

    # Lighting: MuJoCo `diffuse` is an intensity in [0,1]. A true zero would make
    # the image black, which is a rendering failure rather than a lighting
    # condition, so illuminance is mapped through a saturating curve.
    diffuse = float(np.clip(env.lighting_lux / 500.0, 0.02, 1.0)) ** 0.6
    ambient = float(np.clip(0.25 + 0.35 * (env.lighting_lux / 500.0), 0.05, 0.8))
    table_rgba = {
        "gray": "0.55 0.55 0.58 1",
        "white": "0.92 0.92 0.92 1",
        "wood": "0.62 0.45 0.28 1",
        "grid": "0.50 0.50 0.50 1",
        "dark": "0.12 0.12 0.14 1",
    }[env.background.value]

    world: list[str] = [
        # MuJoCo `diffuse`/`ambient` take either one scalar or exactly three
        # components (RGBA is rejected), so a grey ramp is written as RGB only.
        f'    <light name="key" pos="0.4 -0.4 1.4" dir="-0.3 0.3 -1" '
        f'diffuse="{_f(diffuse)} {_f(diffuse)} {_f(diffuse)}" '
        f'ambient="{_f(ambient)} {_f(ambient)} {_f(ambient)}" directional="true"/>',
        f'    <light name="fill" pos="-0.5 0.3 1.0" dir="0.4 -0.2 -1" '
        f'diffuse="{_f(0.5 * diffuse)} {_f(0.5 * diffuse)} {_f(0.5 * diffuse)}"/>',
        # A finite box rather than an infinite plane: the workspace must be
        # bounded so the robot base can sit beside it without ever "standing"
        # on the surface.
        f'    <geom name="table" type="box" '
        f'pos="{_f(env.workspace_center[0])} {_f(env.workspace_center[1])} '
        f'{_f(env.table_height_m - 0.02)}" '
        f'size="{_f(extent_x)} {_f(extent_y)} 0.02" '
        f'material="table_mat" friction="{_f(env.object_friction)} 0.005 0.0001" rgba="{table_rgba}" '
        f'{_mask("world")}/>',
    ]
    for cam in ("rgb_camera", "depth_camera", "seg_camera"):
        world.append(f'    <camera name="{cam}" pos="{eye}" xyaxes="{cam_axes}" fovy="{_f(fov_deg)}" mode="fixed"/>')
    world.append(
        f'    <site name="target_zone" pos="{" ".join(_f(v) for v in scenario.task.target_pose.position.array)}" '
        f'size="0.06 0.002 0.06" type="cylinder" rgba="0.1 0.9 0.3 0.30"/>'
    )

    # Obstacles.
    for index in range(env.obstacle_count):
        angle = 2.0 * np.pi * index / max(env.obstacle_count, 1)
        radius = 0.26
        half_h = env.obstacle_height_m / 2.0
        world.append(
            f'    <geom name="obstacle{index}" type="box" size="0.03 0.03 {_f(half_h)}" '
            f'pos="{_f(env.workspace_center[0] + radius * np.cos(angle))} "'
            f'{_f(env.workspace_center[1] + radius * np.sin(angle))} {_f(env.table_height_m + half_h)}" '
            f'rgba="0.30 0.30 0.35 1" {_mask("world")}/>'
        )

    # Objects: explicit position wins, otherwise sample deterministically.
    from robot_ci.core.seeding import make_numpy_rng

    rng = make_numpy_rng(scenario.seed, "mujoco-scene", scenario.scenario_id)
    palette = {
        ObjectCategory.CUBE: "0.85 0.15 0.15 1",
        ObjectCategory.CYLINDER: "0.15 0.45 0.85 1",
        ObjectCategory.SPHERE: "0.20 0.70 0.30 1",
        ObjectCategory.CAN: "0.90 0.80 0.20 1",
        ObjectCategory.LIFTER: "0.70 0.70 0.75 1",
    }
    occupied: list[np.ndarray] = []
    for index, spec in enumerate(scenario.objects):
        extents, geom_type = object_geoms(spec.category)
        xy: np.ndarray | None
        if spec.position is not None:
            xy = np.array([spec.position.array[0], spec.position.array[1]])
        else:
            xy = None
            for _ in range(80):
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                radius = float(rng.uniform(0.08, 0.20))
                candidate = np.array(env.workspace_center) + radius * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                min_gap = env.clutter_separation_m + 0.07
                if all(float(np.linalg.norm(candidate - other)) > min_gap for other in occupied):
                    xy = candidate
                    break
            if xy is None:
                xy = np.array(env.workspace_center) + np.array([0.24 + 0.03 * index, 0.0])
        assert xy is not None  # narrowed: both branches above assign it
        occupied.append(xy)
        explicit_z = spec.position.array[2] if spec.position is not None else 0.0
        z = float(explicit_z) if explicit_z > 0.0 else env.table_height_m + extents[2]
        quat = spec.orientation
        mass = 0.08 * 8.0 * extents[0] * extents[1] * extents[2]
        world.append(
            f'    <body name="{spec.id}" pos="{_f(xy[0])} {_f(xy[1])} {_f(z)}" '
            f'quat="{_f(quat.w)} {_f(quat.x)} {_f(quat.y)} {_f(quat.z)}">'
            f'<freejoint name="{spec.id}_free"/>'
            f'<geom name="{spec.id}" type="{geom_type}" size="{" ".join(_f(v) for v in extents)}" '
            f'rgba="{palette[spec.category]}" mass="{_f(mass)}" '
            f'friction="{_f(env.object_friction)} 0.01 0.0002" {_mask("object")}/>'
            f'</body>'
        )

    robot_worldbody, robot_actuator = _build_robot_xml(scenario)

    return "\n".join(
        [
            '<mujoco model="robot_ci_tabletop">',
            '  <compiler angle="radian" autolimits="true"/>',
            '  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>',
            '  <visual>',
            '    <global offwidth="1024" offheight="1024"/>',
            '  </visual>',
            '  <asset>',
            '    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.20 0.20 0.22" rgb2="0.30 0.30 0.33" width="128" height="128"/>',
            '    <material name="table_mat" texture="grid_tex" texrepeat="4 4" reflectance="0.05"/>',
            '  </asset>',
            '  <worldbody>',
            # The robot comes first so that the arm's joints occupy the leading
            # qpos slots. Object free-joints follow. Consumers still address
            # joints by name, but a conventional order keeps exported models
            # readable and stable across scenarios.
            robot_worldbody,
            *world,
            '  </worldbody>',
            robot_actuator,
            '</mujoco>',
        ]
    )


def scene_cache_key(scenario: ScenarioConfig) -> str:
    """Return a stable cache key for a generated scene.

    Only the fields that change the MJCF are included, so a scenario that differs
    only in, say, RGB corruption reuses the same compiled model.
    """
    payload: dict[str, Any] = {
        "environment": scenario.environment.model_dump(mode="json"),
        "robot": scenario.robot.model_dump(mode="json"),
        "task_target": scenario.task.target_pose.model_dump(mode="json"),
        "objects": [o.model_dump(mode="json") for o in scenario.objects],
        "seed": scenario.seed,
        "scenario_id": scenario.scenario_id,
        "camera": {
            "position": scenario.camera_position.model_dump(mode="json"),
            "look_at": scenario.camera_look_at.model_dump(mode="json"),
            "resolution": list(scenario.resolution),
        },
    }
    digest = hashlib.blake2b(
        hash_payload(payload).encode("utf-8"), digest_size=16
    ).hexdigest()
    return digest
