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
_CHAIN = (
    (0.0, 0.0, 0.150),
    (0.0, 0.0, 0.000),
    (0.0, -0.020, 0.150),
    (0.0, 0.0, 0.100),
    (0.0, 0.0, 0.100),
    (0.0, 0.0, 0.080),
    (0.0, 0.0, 0.060),
)
#: Hinge axis of each arm joint in its parent frame.
_AXES = ("1 0 0", "0 0 1", "0 0 1", "1 0 0", "0 0 1", "1 0 0", "0 0 1")
#: Capsule length (m) and radius (m) of each of the seven arm links.
_LINK_LENGTHS = (0.13, 0.13, 0.11, 0.10, 0.09, 0.06, 0.055)
_LINK_RADII = (0.060, 0.060, 0.055, 0.050, 0.045, 0.040, 0.035)


def _build_robot_xml(scenario: ScenarioConfig) -> tuple[str, str]:
    """Emit a 7-DOF serial arm with a parallel gripper as MJCF.

    The chain is a DH-derived kinematic tree with one position actuator per arm
    joint plus one per finger. Joint and actuator names are
    ``joint1..joint7`` / ``act1..act7``, matching
    :attr:`RobotConfig.arm_joint_names`, so the mapping from policy joint index
    to MuJoCo actuator index is the identity.

    Returns:
        ``(worldbody_xml, actuator_xml)``. The two blocks are returned
        separately because MJCF requires the ``<actuator>`` element to be a
        sibling of ``<worldbody>``, not a child of it.
    """
    dof = scenario.robot.dof
    if dof != 7:
        raise SimulationError(
            f"MuJoCoBackend currently models a 7-DOF arm; scenario requests dof={dof}. "
            "The UR5e embodiment is scheduled for Phase 2.",
            code="simulation.embodiment_unsupported",
            requested_dof=dof,
        )

    lower = scenario.robot.limits.lower
    upper = scenario.robot.limits.upper
    kv = scenario.robot.limits.max_velocity
    kp = (80.0,) * 7
    damping = (4.0,) * 7

    # The nesting depth is grown and shrunk explicitly: a serial chain of
    # link0..link7 plus a hand has nine open <body> elements, and hand-writing
    # the closing tags is exactly the kind of off-by-one that produces an
    # invalid MJCF.
    body: list[str] = ['    <body name="link0" pos="0 0 0">']
    body.append(
        '      <geom name="link0_geom" type="capsule" fromto="0 0 0 0 0 0.14" '
        'size="0.08" rgba="0.25 0.25 0.28 1" contype="1" conaffinity="1"/>'
    )
    indent = "      "
    for index in range(7):
        dx, dy, dz = _CHAIN[index]
        body.append(f'{indent}<body name="link{index + 1}" pos="{_f(dx)} {_f(dy)} {_f(dz)}">')
        indent += "  "
        body.append(
            f'{indent}<joint name="joint{index + 1}" type="hinge" axis="{_AXES[index]}" '
            f'range="{_f(lower[index])} {_f(upper[index])}" damping="{_f(damping[index])}" armature="0.1"/>'
        )
        body.append(
            f'{indent}<geom name="link{index + 1}_geom" type="capsule" '
            f'fromto="0 0 0 0 0 {_f(_LINK_LENGTHS[index])}" size="{_f(_LINK_RADII[index])}" '
            f'rgba="0.72 0.74 0.78 1" contype="1" conaffinity="1"/>'
        )

    # Flange and parallel gripper, nested inside the last arm link.
    body.append(f'{indent}<body name="hand" pos="0 0 0.055">')
    indent += "  "
    body.append(f'{indent}<geom name="hand_geom" type="box" size="0.035 0.035 0.03" rgba="0.20 0.20 0.22 1" contype="1" conaffinity="1"/>')
    body.append(f'{indent}<site name="ee" pos="0 0 0.055" size="0.006" rgba="0 0 0 0" type="sphere"/>')
    for side, sign in (("left", 1.0), ("right", -1.0)):
        body.append(f'{indent}<body name="{side}_finger" pos="0 {_f(0.035 * sign)} 0.03">')
        indent += "  "
        rng_axis = "0 1 0"
        rng_range = "0 0.04" if sign > 0 else "-0.04 0"
        body.append(f'{indent}<joint name="{side}_finger_joint" type="slide" axis="{rng_axis}" range="{rng_range}" damping="5"/>')
        body.append(f'{indent}<geom name="{side}_finger_geom" type="box" size="0.008 0.008 0.022" rgba="0.85 0.30 0.10 1" contype="1" conaffinity="1"/>')
        body.append(f"{indent[:-2]}</body>")
        indent = indent[:-2]
    body.append(f"{indent[:-2]}</body>")   # close hand
    indent = indent[:-2]
    for _ in range(7):                   # close link7 .. link1
        body.append(f"{indent}</body>")
        indent = indent[:-2]
    body.append(f"{indent}</body>")       # close link0

    actuator: list[str] = ["  <actuator>"]
    for index in range(7):
        kv_i = kv[index] if index < len(kv) else 2.0
        actuator.append(
            f'    <position name="act{index + 1}" joint="joint{index + 1}" kp="{_f(kp[index])}" '
            f'kv="{_f(kv_i)}" ctrlrange="{_f(lower[index])} {_f(upper[index])}" forcerange="-87 87"/>'
        )
    actuator.append('    <position name="left_finger_act" joint="left_finger_joint" kp="500" kv="10" ctrlrange="0 0.04" forcerange="-20 20"/>')
    actuator.append('    <position name="right_finger_act" joint="right_finger_joint" kp="500" kv="10" ctrlrange="-0.04 0" forcerange="-20 20"/>')
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
        f'    <geom name="table" type="plane" pos="0 0 {_f(env.table_height_m)}" size="{_f(extent_x)} {_f(extent_y)} 0.01" '
        f'material="table_mat" friction="{_f(env.object_friction)} 0.005 0.0001" rgba="{table_rgba}"/>',
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
        radius = 0.22
        half_h = env.obstacle_height_m / 2.0
        world.append(
            f'    <geom name="obstacle{index}" type="box" size="0.03 0.03 {_f(half_h)}" '
            f'pos="{_f(radius * np.cos(angle))} {_f(radius * np.sin(angle))} {_f(env.table_height_m + half_h)}" '
            f'rgba="0.30 0.30 0.35 1" contype="1" conaffinity="1"/>'
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
                radius = float(rng.uniform(0.14, 0.30))
                candidate = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                min_gap = env.clutter_separation_m + 0.07
                if all(float(np.linalg.norm(candidate - other)) > min_gap for other in occupied):
                    xy = candidate
                    break
            if xy is None:
                xy = np.array([0.32 + 0.03 * index, 0.0])
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
            f'friction="{_f(env.object_friction)} 0.01 0.0002" contype="1" conaffinity="1"/>'
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
