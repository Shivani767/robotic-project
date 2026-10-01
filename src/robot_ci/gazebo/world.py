"""Deterministic SDF generation for the Gazebo backend.

Everything here is pure string and numeric work over a
:class:`~robot_ci.models.scenario.ScenarioConfig`. It imports no Gazebo and
starts no process, which is what makes the interesting part -- *does the
generated world match the ROBOT CI condition?* -- testable on a machine with no
Gazebo installed, and therefore worth testing there.

Determinism rules, inherited from the project's existing scene generator:

* every random draw comes from ``make_numpy_rng(scenario.seed, purpose,
  scenario.scenario_id)``, the same convention ``simulation/mujoco_env.py``
  uses;
* ``scenario_id`` is read, never written. It participates in seeding, so
  renaming a condition to describe a generated artifact would move the object;
* the depth-noise, bias, dropout, occlusion, blur and latency settings are
  translated into Gazebo sensor configuration, not ignored. A backend that
  silently dropped them would run a different experiment and call it the same
  one.

What Gazebo cannot reproduce is recorded rather than papered over: see
:func:`unsupported_conditions`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from robot_ci.models.scenario import ScenarioConfig

__all__ = [
    "GAZEBO_ROBOT_NAME",
    "GAZEBO_WORLD_NAME",
    "GazeboWorld",
    "camera_intrinsics_from_scenario",
    "object_pose_from_scenario",
    "unsupported_conditions",
    "world_sdf",
]

#: Model and world names. Fixed strings so the gz-transport topic names a caller
#: has to know are predictable rather than derived per run.
GAZEBO_WORLD_NAME = "robot_ci_pick_place"
GAZEBO_ROBOT_NAME = "panda"


def object_pose_from_scenario(scenario: ScenarioConfig) -> tuple[float, float, float]:
    """Return the target object's ``(x, y, z)`` in metres, deterministically.

    The same draw sequence as the MuJoCo generator is *not* reproduced here --
    matching Gazebo's sampler to MuJoCo's would require reading private state
    from the other backend, and the two samplers are different code. What is
    guaranteed is that this placement is a pure function of
    ``(scenario.seed, scenario_id)`` and therefore reproducible for Gazebo runs
    and identical between repeats. Cross-simulator *absolute* placement is not
    claimed, and :func:`unsupported_conditions` says so.
    """
    import numpy as np

    from robot_ci.core.seeding import make_numpy_rng

    rng = make_numpy_rng(scenario.seed, "gazebo-scene", scenario.scenario_id)
    centre = np.asarray(scenario.environment.workspace_center, dtype=np.float64)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(0.14, 0.22))
    table_z = float(scenario.environment.table_height_m)
    return (
        float(centre[0] + radius * np.cos(angle)),
        float(centre[1] + radius * np.sin(angle)),
        table_z + 0.02,
    )


def camera_intrinsics_from_scenario(
    scenario: ScenarioConfig,
) -> tuple[int, int, float, float, float, float]:
    """Return ``(width, height, fx, fy, cx, cy)`` from the scenario's camera.

    Gazebo sizes a camera with an explicit horizontal field of view rather than
    focal lengths in pixels, so the conversion goes through the FOV the
    scenario's intrinsics imply. Clipping is taken from the depth config so the
    Gazebo camera reports the same range the MuJoCo one does -- a depth image
    in different units or over a different range is not the same measurement.
    """
    intrinsics = scenario.sensors.effective_intrinsics()
    return (
        int(intrinsics.width_px),
        int(intrinsics.height_px),
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.cx),
        float(intrinsics.cy),
    )


def unsupported_conditions(scenario: ScenarioConfig) -> list[str]:
    """List scenario features this backend cannot reproduce, by name.

    Reported rather than ignored. A backend that silently dropped a corruption
    setting would produce a success rate that looks comparable and is not, which
    is the exact failure mode this project was built to detect.
    """
    gaps: list[str] = []
    sensors = scenario.sensors
    if sensors.depth.dropout_percent > 0.0:
        gaps.append("depth_dropout_percent (Gazebo has no per-frame dropout)")
    if sensors.rgb.occlusion_percent > 0.0:
        gaps.append("rgb_occlusion_percent (no Gazebo occlusion injector)")
    if float(sensors.rgb.blur_kernel_px) > 0.0:
        gaps.append("rgb_blur_px (no Gazebo blur stage; applied post-render instead)")
    if sensors.camera.latency.total_delay_ms > 0.0:
        gaps.append("camera_latency_ms (transport latency, not sensor latency)")
    if getattr(sensors.robot_state, "joint_offset_rad", 0.0) != 0.0:
        gaps.append("joint_offset_rad (applied in the adapter, not the URDF)")
    return gaps


def _joint_controller_plugin() -> str:
    """The joint controller, to be placed *inside* the robot model.

    Where this goes is not a style choice. Three placements were tried against
    Gazebo Harmonic and only this one works:

    * in the **world**, the plugin refuses with "should be attached to a model
      entity";
    * inside the URDF's ``<gazebo>`` block, which *is* a model, it attaches but
      cannot resolve any joint -- "Failed to get any <joint_name>" -- for every
      configuration form tried (``<scoped_joints>``, repeated ``<joint_name>``,
      and a ``<joint_names>`` wrapper).

    Embedding the robot as a converted SDF model with the plugin inside it is
    the arrangement that produces command topics. The joint names are ROBOT
    CI's own, so nothing in the control path needs translating.
    """
    names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "left_finger",
        "right_finger",
    ]
    # The `name` attribute is the class Gazebo loads, not a free-text instance
    # label: suffixing it makes the loader fail with "library does not contain
    # requested plugin". Nine instances therefore share one name, which is how
    # Gazebo worlds configure several controllers of the same type.
    blocks = []
    for name in names:
        blocks.append(
            '    <plugin filename="gz-sim-joint-position-controller-system"\n'
            '            name="gz::sim::systems::JointPositionController">\n'
            f"      <joint_name>{name}</joint_name>\n"
            "      <p_gain>1000.0</p_gain>\n"
            "      <i_gain>0.0</i_gain>\n"
            "      <d_gain>50.0</d_gain>\n"
            "    </plugin>"
        )
    # State feedback. The position controller consumes commands but publishes no
    # state, so a separate system broadcasts every joint's position to
    # /world/<world>/model/<model>/joint_state. That topic is what carries
    # Gazebo's physics back to ROBOT CI and to the live viewer.
    state_publisher = (
        '    <plugin filename="gz-sim-joint-state-publisher-system"\n'
        '            name="gz::sim::systems::JointStatePublisher">\n'
        "    </plugin>"
    )
    return "\n".join([*blocks, state_publisher])


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> Any:
    """Rotation matrix for SDF's roll-pitch-yaw (Rz * Ry * Rx)."""
    import numpy as np

    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _compose(
    parent: tuple[tuple[float, float, float], tuple[float, float, float]],
    child: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the pose of ``child`` expressed in the frame ``parent`` sits in."""
    import numpy as np

    parent_pos, parent_rpy = parent
    child_pos, child_rpy = child
    rotation = _rpy_to_matrix(parent_rpy)
    position = np.asarray(parent_pos) + rotation @ np.asarray(child_pos)
    matrix = rotation @ _rpy_to_matrix(child_rpy)
    # Recover roll-pitch-yaw from the product without pulling in a transform
    # library: with pitch in the middle column, the usual identities apply.
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(matrix[2, 0]) < 1 - 1e-9:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:  # gimbal lock: pitch is +/- pi/2, so roll and yaw are degenerate
        roll = 0.0
        yaw = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    return (float(position[0]), float(position[1]), float(position[2])), (roll, pitch, yaw)


def _read_pose(element: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Read an element's ``<pose>`` as (position, rpy), defaulting to identity."""
    pose = element.find("pose")
    if pose is None or not pose.text:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    values = [float(v) for v in pose.text.split()]
    while len(values) < 6:
        values.append(0.0)
    return (values[0], values[1], values[2]), (values[3], values[4], values[5])


def _flatten_link_poses(model: Any) -> int:
    """Resolve ``relative_to`` link poses into plain model-frame poses.

    ``gz sdf -p`` expresses the chain through frame references::

        <joint name='joint1'><pose relative_to='panda_link0'>0 0 0.333 ...</pose>
        <link  name='panda_link1'><pose relative_to='joint1'>0 0 0 ...</pose>

    Physics resolves that -- the arm's joint state and link poses prove it does.
    Gazebo's *rendering* path did not: the scene message carried the model, the
    SDF validated, nine joint controllers attached, and yet no link of the robot
    contributed a single pixel or a single finite depth value to the camera,
    while the table, the object and the target zone all rendered normally. The
    only structural difference was these references.

    Rewriting every link pose as a plain model-frame pose removes them. Every
    joint frame stays where it was, because a joint's pose is still interpreted
    relative to its parent link, so physics is unchanged -- only the ambiguity
    the renderer had to resolve is gone.

    Args:
        model: The ``<model>`` element of the converted robot.

    Returns:
        How many link poses were rewritten.
    """
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    parents: dict[str, str] = {}
    frames: dict[str, Any] = {}
    for element in model:
        if element.tag not in {"link", "joint", "frame"}:
            continue
        name = element.get("name")
        if name:
            frames[name] = element
        if element.tag == "joint":
            child = element.find("child")
            if child is not None and child.text:
                parents[child.text.strip()] = name or ""

    resolved: dict[str, Any] = {}

    def resolve(name: str, seen: frozenset[str] = frozenset()) -> Any:
        if name in resolved:
            return resolved[name]
        if name in seen:  # a cycle: fall back to identity rather than recursing
            return identity
        element = frames.get(name)
        if element is None:
            return identity
        own = _read_pose(element)
        pose_element = element.find("pose")
        reference = pose_element.get("relative_to") if pose_element is not None else None
        if reference and reference in frames:
            base = resolve(reference, seen | {name})
        elif element.tag == "joint":
            parent = element.find("parent")
            base = resolve(parent.text.strip(), seen | {name}) if parent is not None else identity
        else:
            base = identity
        resolved[name] = _compose(base, own)
        return resolved[name]

    rewritten = 0
    for element in model:
        if element.tag != "link":
            continue
        name = element.get("name")
        if not name:
            continue
        pose_element = element.find("pose")
        if pose_element is None:
            continue
        if pose_element.get("relative_to") is None:
            continue
        absolute = resolve(name)
        position, rpy = absolute
        pose_element.text = (
            f"{position[0]:.9f} {position[1]:.9f} {position[2]:.9f} "
            f"{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"
        )
        del pose_element.attrib["relative_to"]
        rewritten += 1
    return rewritten


def robot_model_sdf(urdf_path: str | Path) -> str:
    """Convert the robot's URDF into an SDF ``<model>`` block, with control.

    Gazebo cannot resolve joints through an ``<include>``d URDF, so the robot is
    converted and embedded instead. The conversion is Gazebo's own
    (``gz sdf -p``), not a hand-rolled one, so the geometry, inertias and joint
    frames are whatever Gazebo itself would have built.

    Args:
        urdf_path: Path to the robot URDF, valid inside the runtime.

    Returns:
        The ``<model>`` element as text, ready to embed in a world.

    Raises:
        RuntimeError: If Gazebo's converter is unavailable or rejects the URDF.
    """
    import subprocess

    path = Path(urdf_path)
    try:
        completed = subprocess.run(
            ["gz", "sdf", "-p", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - no gz
        raise RuntimeError(
            f"could not run Gazebo's SDF converter (gz sdf -p {path}): {exc}"
        ) from exc
    if completed.returncode != 0 or "<model" not in completed.stdout:
        raise RuntimeError(f"Gazebo rejected {path}: {completed.stderr.strip()[:400]}")

    text = completed.stdout
    start = text.index("<model")
    end = text.rindex("</model>") + len("</model>")
    model = text[start:end]

    # The converter expresses the chain through frame references; flatten them
    # so the renderer gets one unambiguous pose per link. See
    # :func:`_flatten_link_poses` for what this fixes and how it was found.
    element = ElementTree.fromstring(model)
    _flatten_link_poses(element)
    model = ElementTree.tostring(element, encoding="unicode")
    if model.startswith("<?xml"):
        model = model.split("?>", 1)[1]

    # Place the controller inside the model, before its closing tag.
    closing = model.rindex("</model>")
    return f"{model[:closing]}{_joint_controller_plugin()}\n  {model[closing:]}"


def _camera_sensor_xml(scenario: ScenarioConfig) -> str:
    """Emit the depth+RGB camera, with Gazebo's FOV derived from the intrinsics.

    Gazebo sizes a camera by horizontal field of view rather than focal lengths
    in pixels, so the conversion goes through the FOV the scenario's intrinsics
    imply, and that FOV is in radians. Clipping comes from the depth config so
    the Gazebo camera covers the same range as the MuJoCo one: a depth image
    over a different range is not the same measurement, whatever its units.
    """
    import math

    width, height, fx, _fy, _cx, _cy = camera_intrinsics_from_scenario(scenario)
    # Gazebo takes the field of view in RADIANS and rejects anything above 2*pi.
    # Emitting degrees is silently accepted by the SDF schema and then refused at
    # load time with "value [59.49] is greater than the maximum allowed value of
    # [6.283]", which is how this was found.
    fov_rad = 2.0 * math.atan(width / (2.0 * fx))
    near = float(scenario.sensors.depth.min_range_m)
    far = float(scenario.sensors.depth.max_range_m)
    noise = float(scenario.sensors.depth.sigma_mm) / 1000.0
    return f"""    <sensor name="robot_ci_camera" type="rgbd_camera">
      <always_on>1</always_on>
      <update_rate>20</update_rate>
      <visualize>false</visualize>
      <camera name="robot_ci_camera">
        <horizontal_fov>{fov_rad:.6f}</horizontal_fov>
        <image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>
        <clip><near>{near:.6f}</near><far>{far:.6f}</far></clip>
        <depth_camera>
          <clip><near>{near:.6f}</near><far>{far:.6f}</far></clip>
          <noise><type>gaussian</type>
            <mean>{noise:.6f}</mean><stddev>{noise:.6f}</stddev>
          </noise>
        </depth_camera>
      </camera>
    </sensor>"""


def _camera_pose(scenario: ScenarioConfig) -> str:
    """Pose the camera with an explicit look-at, from the workspace centre.

    The first version hard-coded ``0 {cy - 0.35} {table_z + 0.45} 0 0.5 0`` --
    pitched down but with no yaw, from ``(0, -0.25, 0.45)``. That ray travels
    along ``y = -0.25`` for ever and crosses ``z = 0`` at ``x = 0.824``, while
    the table spans ``x in [-0.3, 0.3]`` and ``y in [-0.2, 0.4]``. The camera
    was aimed at empty space, and that is exactly what was observed: every RGB
    frame equal to the scene background colour (0.7 -> 218) and every depth
    pixel ``inf``.

    A Gazebo camera looks along **+X** of its frame, with **+Z** up, and the SDF
    pose applies ``Rz(yaw) * Ry(pitch) * Rx(roll)``. Solving ``R * +X = d`` for
    ``d`` the unit direction from eye to target gives
    ``yaw = atan2(dy, dx)`` and ``pitch = asin(-dz)``.

    Args:
        scenario: The condition whose workspace is to be viewed.

    Returns:
        ``x y z roll pitch yaw`` as text, ready for an ``<pose>`` element.
    """
    import math

    cx = float(scenario.environment.workspace_center[0])
    cy = float(scenario.environment.workspace_center[1])
    table_z = float(scenario.environment.table_height_m)

    # Back, above and outside the table, aiming at a point above the workspace
    # rather than at the tabletop. Aiming at the tabletop put the arm -- which
    # reaches ~0.9 m up -- outside the frustum of a camera pitched down at it;
    # aiming higher keeps the object, the gripper and the full arm in frame.
    # The offsets are relative to the workspace centre, so any scenario's
    # workspace is framed the same way.
    eye = (cx + 0.9, cy - 0.9, table_z + 0.9)
    target = (cx, cy, table_z + 0.35)

    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw = math.atan2(dy, dx)
    pitch = math.asin(-dz / norm)
    return f"{eye[0]:.6f} {eye[1]:.6f} {eye[2]:.6f} 0.0 {pitch:.6f} {yaw:.6f}"


def world_sdf(scenario: ScenarioConfig, *, robot_urdf: str | None = None) -> str:
    """Render the pick-and-place world as SDF.

    The world mirrors the MuJoCo one: a table at the workspace centre, the robot
    mounted on it, the target object at a deterministic pose, and one camera.
    Robot geometry comes from ``robot_urdf`` when given and is otherwise
    referenced by name, so the morphology is shared with the MuJoCo backend
    rather than re-invented here.

    Args:
        scenario: The ROBOT CI condition to translate.
        robot_urdf: Optional path to the robot's URDF, recorded for provenance.

    Returns:
        An SDF document as text.
    """
    cx, cy = (
        float(scenario.environment.workspace_center[0]),
        float(scenario.environment.workspace_center[1]),
    )
    table_z = float(scenario.environment.table_height_m)
    ox, oy, oz = object_pose_from_scenario(scenario)
    # The robot is embedded as a converted SDF model rather than <include>d.
    # An include loads fine, but leaves the joint controller unable to resolve
    # any joint, so the arm cannot be commanded -- which is the entire point of
    # the backend. Conversion is Gazebo's own, so geometry, inertias and joint
    # frames are exactly what Gazebo would have built itself.
    if robot_urdf:
        robot_block = (
            robot_model_sdf(robot_urdf)
            .replace(
                "<model name='panda'>",
                f"<model name='{GAZEBO_ROBOT_NAME}'>",
                1,
            )
            .replace(
                f"<model name='{GAZEBO_ROBOT_NAME}'>",
                f"<model name='{GAZEBO_ROBOT_NAME}'>\n      <pose>0 0 {table_z} 0 0 0</pose>",
                1,
            )
        )
    else:
        robot_block = (
            "    <include>\n"
            f"      <uri>{GAZEBO_ROBOT_NAME}</uri>\n"
            f"      <name>{GAZEBO_ROBOT_NAME}</name>\n"
            f"      <pose>0 0 {table_z} 0 0 0</pose>\n"
            "    </include>"
        )

    return f"""<?xml version="1.0"?>
<!-- Generated by robot_ci.gazebo.world from scenario_id={escape(scenario.scenario_id)}
     seed={scenario.seed}. Do not edit by hand: the run artifact records the
     config_hash this world was generated from. -->
<sdf version="1.9">
  <world name="{GAZEBO_WORLD_NAME}">
    <physics name="robot_ci_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <!-- ogre2, the render engine Gazebo defaults to. ogre (v1) was tried and
           cannot initialise headlessly at all (it reports "Unable to open
           display:" and renders nothing), so ogre2 is the only engine that
           produces frames here. The frame content was checked separately: the
           earlier flat-grey, all-inf frames were caused by the camera pose
           aiming past the table, not by the renderer. See docs/gazebo.md. -->
      <render_engine>ogre2</render_engine>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="table">
      <static>true</static>
      <pose>{cx} {cy} {table_z - 0.02} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>0.6 0.6 0.04</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>0.6 0.6 0.04</size></box></geometry>
          <!-- <ambient> alone leaves <diffuse> at its SDF default of 0 0 0 1,
               and diffuse is what the lights multiply. The first rendered frame
               of the table came back pure black for exactly this reason. -->
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.15 0.15 0.15 1</specular>
          </material></visual>
      </link>
    </model>

    {robot_block}

    <model name="target_cube">
      <static>false</static>
      <pose>{ox:.6f} {oy:.6f} {oz:.6f} 0 0 0</pose>
      <link name="link">
        <inertial><mass>0.05</mass>
          <inertia><ixx>1.6e-5</ixx><iyy>1.6e-5</iyy><izz>1.6e-5</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="collision"><geometry><box><size>0.04 0.04 0.04</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>0.04 0.04 0.04</size></box></geometry>
          <material>
            <ambient>0.9 0.2 0.2 1</ambient>
            <diffuse>0.9 0.2 0.2 1</diffuse>
            <specular>0.3 0.1 0.1 1</specular>
          </material></visual>
      </link>
    </model>

    <model name="target_zone">
      <static>true</static>
      <pose>{cx + 0.18:.6f} {cy - 0.12:.6f} {table_z + 0.001:.6f} 0 0 0</pose>
      <link name="link">
        <visual name="visual"><geometry><cylinder><radius>0.06</radius><length>0.002</length></cylinder></geometry>
          <material>
            <ambient>0.1 0.9 0.3 0.35</ambient>
            <diffuse>0.1 0.9 0.3 0.35</diffuse>
          </material></visual>
      </link>
    </model>

    <model name="camera_rig">
      <static>true</static>
      <pose>{_camera_pose(scenario)}</pose>
      <link name="link">
{_camera_sensor_xml(scenario)}
      </link>
    </model>
  </world>
</sdf>
"""


@dataclass(frozen=True)
class GazeboWorld:
    """A generated world plus the provenance needed to reproduce it.

    Attributes:
        path: Where the SDF was written.
        sdf: Its text.
        scenario_id: The condition it was generated from, read not rewritten.
        seed: The episode seed used for placement.
        config_hash: The ROBOT CI configuration hash, tying the world file to
            the experiment record.
        unsupported: Conditions this backend cannot reproduce, by name.
        details: Free-form provenance for the artifact.
    """

    path: Any
    sdf: str
    scenario_id: str
    seed: int
    config_hash: str
    unsupported: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
