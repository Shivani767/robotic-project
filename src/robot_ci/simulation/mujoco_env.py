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
import os
import re
from pathlib import Path
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
# MuJoCo generates a contact between geoms A and B iff
#
#     (contype_A & conaffinity_B) != 0  OR  (contype_B & conaffinity_A) != 0
#
# Three disjoint bits let the arm avoid colliding with *itself* while still
# colliding with the table and with objects:
#
#   class      contype  conaffinity   may collide with
#   ---------  -------  -----------   -----------------------------------------
#   arm        ARM      WORLD|OBJECT  world, object      (never another arm)
#   object     OBJECT   WORLD|OBJ|ARM world, object, arm
#   world      WORLD    WORLD|OBJ|ARM world, object, arm
#
# The arm's conaffinity deliberately omits its own bit, which is the *only*
# thing that disables intra-arm contact; every other bit is preserved so the
# gripper still meets the object and the arm still meets the table.
#
# An earlier version set the arm's conaffinity to 0, which silently disabled
# *all* arm contact and let the fingers close through the object. The audit in
# ``scripts/audit_collision.py`` exists to catch exactly that class of mistake.
_CONTYPE: dict[str, int] = {"arm": 0b100, "object": 0b010, "world": 0b001}
_CONAFFINITY: dict[str, int] = {
    "arm": 0b001 | 0b010,
    "object": 0b001 | 0b010 | 0b100,
    "world": 0b001 | 0b010 | 0b100,
}


def _mask(name: str) -> str:
    """Return the ``contype``/``conaffinity`` attribute pair for a class."""
    return f'contype="{_CONTYPE[name]}" conaffinity="{_CONAFFINITY[name]}"'


# --- Official arm model -----------------------------------------------------

#: Directory holding the vendored arm models. The MJCF is *data*, not code: the
#: hand-written kinematic chain that previously lived here produced an arm whose
#: workspace excluded the tabletop entirely (the grasp point sat 40-60 mm outside
#: the reachable set), which no amount of re-signing the DH table could fix.
#: The official Franka Emika Panda MJCF is used instead, and this module only
#: composes the *scene* around it.
_ARM_MODEL_DIR: Path | None = None

#: Cached compiled models, keyed by resolved path.
_ARM_MODEL_CACHE: dict[str, Any] = {}


def arm_model_dir() -> Path:
    """Return the directory containing the vendored MJCF arm models.

    Resolution order:

    1. ``$ROBOT_CI_MJCF_DIR`` -- lets a user point at their own models.
    2. ``<repo>/mujoco/models`` -- the vendored copy.

    Raises:
        SimulationError: If no model directory can be found.
    """
    global _ARM_MODEL_DIR
    if _ARM_MODEL_DIR is not None:
        return _ARM_MODEL_DIR

    candidates = []
    env_dir = os.environ.get("ROBOT_CI_MJCF_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path(__file__).resolve().parents[3] / "mujoco" / "models")

    for candidate in candidates:
        if (candidate / "franka_emika_panda" / "panda.xml").exists():
            _ARM_MODEL_DIR = candidate
            return candidate

    raise SimulationError(
        "no MJCF arm model found. Expected "
        f"{candidates[0] / 'franka_emika_panda' / 'panda.xml'} or one of "
        f"{[str(c) for c in candidates]}. Set $ROBOT_CI_MJCF_DIR to point at a "
        "directory containing the vendored models.",
        code="simulation.arm_model_missing",
        searched=[str(c) for c in candidates],
    )


def load_arm_model(mujoco: Any) -> Any:
    """Compile and cache the official arm model.

    Compiling an MJCF with 81 collision geoms is not cheap, and a batch
    experiment resets the same model thousands of times, so the compiled model
    is cached by path.

    Args:
        mujoco: The imported ``mujoco`` module.

    Returns:
        The compiled :class:`mujoco.MjModel`.
    """
    key = str(arm_model_dir().resolve())
    cached = _ARM_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = mujoco.MjModel.from_xml_path(str(arm_model_dir() / "franka_emika_panda" / "panda.xml"))
    _ARM_MODEL_CACHE[key] = model
    return model


#: Name of the arm's root body, shifted by ``RobotConfig.base_position`` after
#: compilation. MuJoCo requires ``<include>`` to be a direct child of
#: ``<mujoco>``, so the pedestal offset cannot be expressed in the generated
#: scene and is applied to the compiled model instead.
_ARM_ROOT_BODY = "link0"


def offset_arm_base(model: Any, base: Any) -> None:
    """Translate the arm's root body by ``base``.

    The arm is mounted on a pedestal beside the table rather than standing on
    the tabletop: with the base at the table origin the forearm sweeps through
    the table surface at ordinary poses, and the resulting contact forces are
    indistinguishable from a control failure.
    """
    import mujoco

    mj: Any = mujoco  # the C extension exposes no usable type stubs

    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _ARM_ROOT_BODY)
    if body_id < 0:  # pragma: no cover - depends on the vendored model
        return
    model.body_pos[body_id] += np.array([base.x, base.y, base.z], dtype=np.float64)


def _inject_tcp_site(xml: str) -> str:
    """Insert the tool-centre-point site inside the ``hand`` body.

    Returns the patched XML, or the input unchanged if the hand body cannot be
    located (in which case the caller reports a missing TCP rather than
    silently falling back to a frame with a zero Jacobian).
    """
    opening = '<body name="hand"'
    start = xml.find(opening)
    if start < 0:
        return xml
    depth = 0
    index = start
    while index < len(xml):
        if xml.startswith("<body", index):
            depth += 1
        elif xml.startswith("</body>", index):
            depth -= 1
            if depth == 0:
                return (
                    xml[:index]
                    + '  <site name="tcp" pos="0 0 0.1034" size="0.005" '
                    'rgba="0 0 1 0" type="sphere"/>\n  '
                    + xml[index:]
                )
        index += 1
    return xml


def relax_arm_self_collision(model: Any) -> int:
    """Put the arm into the ``arm`` collision class so it stops hitting itself.

    The vendored Panda's collision meshes are accurate, and that accuracy means
    the arm self-collides at ordinary configurations (link2 against link5, link5
    against the hand, and so on). Preventing those is a *motion-planning*
    concern in a real deployment; here the planner's joint-limit clamping and a
    seeded home pose already keep the arm in a valid region, so intra-arm
    contacts carry no information and would only corrupt the contact metrics.

    Contacts that do carry information -- gripper to object, object to table,
    arm to table -- are preserved by construction: the arm keeps its WORLD and
    OBJECT bits and loses only its own.

    Returns:
        The number of geoms whose masks were rewritten.
    """
    import mujoco  # noqa: PLC0415 - imported lazily; MuJoCo is optional

    mj: Any = mujoco  # the C extension exposes no usable type stubs

    arm_ct, arm_ca = _CONTYPE["arm"], _CONAFFINITY["arm"]
    changed = 0
    for index in range(model.ngeom):
        body_id = int(model.geom_bodyid[index])
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or ""
        if not _is_arm_body(body):
            continue
        if int(model.geom_contype[index]) == arm_ct and int(model.geom_conaffinity[index]) == arm_ca:
            continue
        model.geom_contype[index] = arm_ct
        model.geom_conaffinity[index] = arm_ca
        changed += 1

    if changed:
        # MuJoCo precomputes broadphase pairs at compile time, so editing the
        # masks afterwards is a no-op until the derived constants are rebuilt.
        mj.mj_setConst(model, mj.MjData(model))
    return changed


def _is_arm_body(name: str) -> bool:
    """Whether a body belongs to the robot arm rather than the scene."""
    return name.startswith("link") or name in {"hand", "left_finger", "right_finger"}


def patched_arm_path() -> Path:
    """Return a cache-local copy of the arm MJCF with a resolvable meshdir.

    The vendored ``panda.xml`` declares ``<compiler meshdir="assets">``.
    MuJoCo resolves that relative to the *top-level* document, so when the arm
    is ``<include>``d from a generated scene the assets are looked up in the
    wrong place. Rewriting the directive to an absolute path into a cache
    directory fixes it without touching the vendored source, which keeps the
    upstream file byte-identical to what was downloaded.
    """
    model_dir = arm_model_dir()
    source = model_dir / "franka_emika_panda" / "panda.xml"
    assets = (model_dir / "franka_emika_panda" / "assets").as_posix()
    cache = model_dir / "franka_emika_panda" / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "panda_patched.xml"

    text = source.read_text(encoding="utf-8")
    patched = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets}"', text)
    # Add a tool-centre-point site 100 mm beyond the flange, between the
    # fingers. The flange frame is roughly 50 mm behind where the fingers meet,
    # so commanding it to a grasp pose places the object outside the gripper.
    if 'name="tcp"' not in patched:
        # The site must be declared INSIDE the <body name="hand"> element: a
        # site placed directly under <worldbody> lands at the world origin and
        # its Jacobian is identically zero.
        patched = _inject_tcp_site(patched)
    if not target.exists() or target.read_text(encoding="utf-8") != patched:
        target.write_text(patched, encoding="utf-8")
    return target


def build_scene_xml(scenario: ScenarioConfig) -> str:
    """Build the scene MJCF that wraps the official arm model.

    The arm itself comes from the vendored Franka MJCF via ``<include>``. This
    function owns only the *experiment*: table, objects, obstacles, lights,
    cameras and the target zone. Keeping the split is what makes the scenario
    the single source of truth for experimental conditions while the arm
    geometry stays verified third-party data.

    Args:
        scenario: The experimental condition.

    Returns:
        A self-contained MJCF document as a string.

    Raises:
        SimulationError: If the arm model directory cannot be found.
    """
    env = scenario.environment

    intrinsics = scenario.sensors.effective_intrinsics()
    height, width = scenario.resolution
    fov_deg = float(
        np.degrees(2.0 * np.arctan((max(width, height) / 2.0) / max(intrinsics.fx, 1e-6)))
    )
    eye = " ".join(_f(v) for v in scenario.camera_position.array)
    cam_axes = " ".join(
        _f(v) for v in _look_at_quat(scenario.camera_position.array, scenario.camera_look_at.array)
    )
    extent_x, extent_y = env.table_extent_m

    # Lighting: MuJoCo `diffuse` is an intensity in [0,1]. A true zero would
    # make the image black, which is a rendering failure rather than a
    # lighting condition, so illuminance maps through a saturating curve.
    diffuse = float(np.clip(env.lighting_lux / 500.0, 0.02, 1.0)) ** 0.6
    ambient = float(np.clip(0.25 + 0.35 * (env.lighting_lux / 500.0), 0.05, 0.8))
    table_rgba = {
        "gray": "0.55 0.55 0.58 1",
        "white": "0.92 0.92 0.92 1",
        "wood": "0.62 0.45 0.28 1",
        "grid": "0.50 0.50 0.50 1",
        "dark": "0.12 0.12 0.14 1",
    }[env.background.value]

    # --- collision classes -------------------------------------------------
    #
    # MuJoCo collides two geoms when (contype_a & conaffinity_b) or
    # (contype_b & conaffinity_a) is non-zero. The official arm's own geoms
    # keep the menagerie defaults; the classes below are assigned only to the
    # scene geoms this module creates, so the arm is not modified.
    world_mask = _mask("world")
    object_mask = _mask("object")

    world: list[str] = [
        f'    <light name="key" pos="0.4 -0.4 1.4" dir="-0.3 0.3 -1" '
        f'diffuse="{_f(diffuse)} {_f(diffuse)} {_f(diffuse)}" '
        f'ambient="{_f(ambient)} {_f(ambient)} {_f(ambient)}" directional="true"/>',
        f'    <light name="fill" pos="-0.5 0.3 1.0" dir="0.4 -0.2 -1" '
        f'diffuse="{_f(0.5 * diffuse)} {_f(0.5 * diffuse)} {_f(0.5 * diffuse)}"/>',
        f'    <geom name="table" type="box" '
        f'pos="{_f(env.workspace_center[0])} {_f(env.workspace_center[1])} '
        f'{_f(env.table_height_m - 0.02)}" size="{_f(extent_x)} {_f(extent_y)} 0.02" '
        f'friction="{_f(env.object_friction)} 0.005 0.0001" rgba="{table_rgba}" {world_mask}/>',
    ]
    for cam in ("rgb_camera", "depth_camera", "seg_camera"):
        world.append(
            f'    <camera name="{cam}" pos="{eye}" xyaxes="{cam_axes}" '
            f'fovy="{_f(fov_deg)}" mode="fixed"/>'
        )
    world.append(
        f'    <site name="target_zone" '
        f'pos="{" ".join(_f(v) for v in scenario.task.target_pose.position.array)}" '
        f'size="0.06 0.002 0.06" type="cylinder" rgba="0.1 0.9 0.3 0.30"/>'
    )

    for index in range(env.obstacle_count):
        angle = 2.0 * np.pi * index / max(env.obstacle_count, 1)
        radius = 0.26
        half_h = env.obstacle_height_m / 2.0
        world.append(
            f'    <geom name="obstacle{index}" type="box" size="0.03 0.03 {_f(half_h)}" '
            f'pos="{_f(env.workspace_center[0] + radius * np.cos(angle))} '
            f'{_f(env.workspace_center[1] + radius * np.sin(angle))} '
            f'{_f(env.table_height_m + half_h)}" rgba="0.30 0.30 0.35 1" {world_mask}/>'
        )

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
        assert xy is not None
        occupied.append(xy)
        explicit_z = spec.position.array[2] if spec.position is not None else 0.0
        z = float(explicit_z) if explicit_z > 0.0 else env.table_height_m + extents[2]
        quat = spec.orientation
        # Mass from the object's volume at a plausible density. A degenerate
        # mass (or a zero inertia, which is what happens when a geom has no
        # `<default>` class to supply one) makes the contact solver behave
        # erratically -- the object visibly launches into the air on the first
        # contact -- so this is computed explicitly rather than left to a
        # default that may not exist.
        volume = 8.0 * extents[0] * extents[1] * extents[2]
        mass = float(np.clip(volume * 250.0, 5.0e-3, 2.0))
        world.append(
            f'    <body name="{spec.id}" pos="{_f(xy[0])} {_f(xy[1])} {_f(z)}" '
            f'quat="{_f(quat.w)} {_f(quat.x)} {_f(quat.y)} {_f(quat.z)}">'
            f'<freejoint name="{spec.id}_free"/>'
            f'<geom name="{spec.id}" type="{geom_type}" '
            f'size="{" ".join(_f(v) for v in extents)}" '
            f'rgba="{palette[spec.category]}" mass="{_f(mass)}" '
            f'friction="{_f(env.object_friction)} 0.01 0.0002" {object_mask}/>'
            f'</body>'
        )

    # The vendored model already defines tuned position servos for the seven arm
    # joints, so none are re-declared here: a second actuator on the same joint
    # makes the two fight and the arm does not move. It does *not* actuate the
    # fingers, so symmetric gripper servos are added below.
    gripper_actuators = [
        f'    <position name="{side}_act" joint="{side}" '
        f'kp="800" kv="25" ctrlrange="0 0.04" forcerange="-40 40"/>'
        for side in ("finger_joint1", "finger_joint2")
    ]

    return "\n".join(
        [
            '<mujoco model="robot_ci_tabletop">',
            # The included arm re-declares <compiler meshdir="assets">, and
            # MuJoCo resolves that relative to the top-level document, so the
            # two declarations fight over the same setting. A patched copy of
            # the arm -- with the meshdir attribute removed and an absolute
            # asset path substituted -- is generated into the cache directory.
            # The vendored file itself is never modified.
            '  <compiler angle="radian" autolimits="true"/>',
            '  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>',
            '  <visual>',
            '    <global offwidth="1024" offheight="1024"/>',
            '  </visual>',
            '  <statistic center="0 0 0.3" extent="0.8"/>',
            "  <worldbody>",
            # Scene geoms sit at world top level: a free joint is only legal on
            # a top-level body, and the objects must not inherit the arm's base
            # transform.
            *world,
            "  </worldbody>",
            # Joint-space position servos for the seven arm joints, matching the
            # menagerie actuator indices 0..6.
            f'  <include file="{patched_arm_path()}"/>',
            "  <actuator>",
            *gripper_actuators,
            "  </actuator>",
            "</mujoco>",
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
