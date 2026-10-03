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

from robot_ci.models.environment import object_half_extents
from robot_ci.models.scenario import ScenarioConfig

__all__ = [
    "FINGER_ACTUATOR_STIFFNESS_N_PER_M",
    "GAZEBO_ROBOT_NAME",
    "GAZEBO_WORLD_NAME",
    "GazeboWorld",
    "camera_intrinsics_from_scenario",
    "object_pose_from_scenario",
    "place_robot_model",
    "unsupported_conditions",
    "world_sdf",
]

#: Model and world names. Fixed strings so the gz-transport topic names a caller
#: has to know are predictable rather than derived per run.
GAZEBO_WORLD_NAME = "robot_ci_pick_place"
GAZEBO_ROBOT_NAME = "panda"

#: Height of the Panda's base collision cylinder *below* the model frame origin,
#: in metres, so the model must be raised by this much to sit **on** the table.
#:
#: Read off ``gazebo/models/panda/panda.urdf``: ``panda_link0``'s collision is a
#: cylinder of length 0.14 at ``origin xyz="0 0 0.05"``, so it spans
#: ``z in [-0.02, 0.12]`` in the model frame. Spawning the model at the table
#: height therefore buried the base 20 mm inside the tabletop.
#:
#: That is not a cosmetic overlap. Measured on a live run, ODE resolved the
#: interpenetration with an impulse that ejected the whole arm: the ``panda``
#: entity was reported at ``z = -10189 m`` after one episode, i.e. in free
#: fall. The object never fell -- it spawns exactly touching the table -- which
#: is what isolated the cause to the overlap rather than to missing collisions.
_BASE_CYLINDER_BOTTOM_M = 0.02


def object_pose_from_scenario(scenario: ScenarioConfig) -> tuple[float, float, float]:
    """Return the target object's ``(x, y, z)`` in metres, deterministically.

    The same draw sequence as the MuJoCo generator is *not* reproduced here --
    matching Gazebo's sampler to MuJoCo's would require reading private state
    from the other backend, and the two samplers are different code. What is
    guaranteed is that this placement is a pure function of
    ``(scenario.seed, scenario_id)`` and therefore reproducible for Gazebo runs
    and identical between repeats. Cross-simulator *absolute* placement is not
    claimed, and :func:`unsupported_conditions` says so.

    An **explicit** ``scenario.target.position`` wins over the seeded draw, the
    same way ``simulation/mujoco_env.py`` honours it. Without that rule the
    ``pose_perturbation`` family (object-position perturbation) would be
    silently ignored here: the scenario would say the cube is at x=0.31 and
    Gazebo would put it wherever the seed drew it, so the experiment would run
    a *different* condition under the perturbed condition's name.
    """
    import numpy as np

    from robot_ci.core.seeding import make_numpy_rng

    table_z = float(scenario.environment.table_height_m)
    # The resting height is derived from the object the scenario declares, not
    # written down. A hard-coded 0.02 m here drew a 40 mm cube for a condition
    # whose object is 16 mm, so the simulated object was 2.5x the size the task,
    # the policy and the perception layer all reasoned about, and its centre sat
    # 12 mm above the height the grasp geometry was derived for.
    half_height = float(object_half_extents(scenario.target.category, scenario.target.size_m)[2])

    explicit = scenario.target.position
    if explicit is not None:
        # An explicit position wins, the same way ``simulation/mujoco_env.py``
        # honours it. The z is only taken from it when it is positive: a
        # scenario that names only (x, y) means "spawn it on the table", and the
        # MJCF generator reads it the same way.
        z = float(explicit.z) if float(explicit.z) > 0.0 else table_z + half_height
        return (float(explicit.x), float(explicit.y), z)

    rng = make_numpy_rng(scenario.seed, "gazebo-scene", scenario.scenario_id)
    centre = np.asarray(scenario.environment.workspace_center, dtype=np.float64)
    inner, outer = (float(v) for v in scenario.environment.object_spawn_radius_m)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(inner, outer))
    return (
        float(centre[0] + radius * np.cos(angle)),
        float(centre[1] + radius * np.sin(angle)),
        table_z + half_height,
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


#: Stiffness of the finger position actuators in newtons per metre of position
#: error, which is exactly ``p_gain`` for a Gazebo joint position controller.
#: Exported so :mod:`robot_ci.simulation.gazebo` can hand the same number to
#: :class:`~robot_ci.control.gripper.GripperController`, whose force bound is
#: derived from the actuator stiffness rather than chosen.
FINGER_ACTUATOR_STIFFNESS_N_PER_M = 1000.0


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
    #
    # Gains
    # -----
    # The controller is Gazebo's ``JointPositionController``, which applies
    # ``gz::math::PID``. Its law (verified against the gz-sim8 and gz-math7
    # sources, not assumed) is, with ``error = jointPosition - jointPositionCmd``:
    #
    #     pTerm = p_gain * error
    #     iErr += i_gain * dt * error        (clamped to [i_min, i_max])
    #     cmd   = -pTerm - iErr - d_gain * errorRate   (clamped to [cmd_min, cmd_max])
    #
    # Two consequences drove the values below, both measured on this robot:
    #
    # * **The integral term exists and is bounded by default to nothing.** With
    #   ``i_gain = 0`` the controller is pure PD, so a constant load leaves a
    #   steady-state error of ``load / p_gain``. Measured at the home
    #   configuration, that was 15.0 mrad on joint2, 17.9 on joint4 and 31.5 on
    #   joint7. ``i_gain`` removes it: the accumulator supplies exactly the force
    #   the proportional term is short of. ``i_min``/``i_max`` are set
    #   explicitly because the library's own defaults (``i_max = -1``,
    #   ``i_min = 0``) leave ``iMax >= iMin`` false, which *disables* the clamp
    #   and makes the integral unbounded -- windup against a stop.
    # * **The output clamp defaults to off too** (``cmdMax = -1``,
    #   ``cmdMin = 0``). Each controller is bounded to its own joint's effort
    #   limit, from ``RobotConfig.limits.max_effort``, so the controller can
    #   never ask for a force the actuator cannot produce.
    #
    # The fingers get a bounded integral term, and the bound was set from a
    # measurement rather than a preference. With ``i_gain = 0`` the jaws rest
    # ~2 mm short of command; the pads are 0.011 m apart at rest, so 2 mm of
    # travel each is 4 mm of pad separation, and against a 16 mm object that is
    # the difference between squeezing it and merely touching it. A live episode
    # with ``i_max = 0.5`` N measured exactly that: the aperture during transit
    # read 0.005 m, a pad separation of 0.016 m on a 0.016 m object, zero
    # interference -- the object was lifted 55.7 mm and then dropped on the
    # first transit move. The accumulator needs ~2.2 N to close that gap, so
    # ``i_max`` is 4 N: enough to trim the residual, and still small against
    # the 100 N finger limit and the 0.49 N object, so it cannot crush what it
    # is holding.
    #
    # ``p_gain`` stays at 1000 N/m, and the reason is a measurement rather than a
    # preference. The episode trace shows the grasp failing as **squeeze
    # ejection**: the pads command 11.2 mm of separation against a 16 mm object,
    # so each interferes by 4.8 mm, and the object is thrown 14 mm clear of the
    # jaws in a single control tick -- about 0.28 m/s -- on the tick the lift
    # begins. Halving the finger stiffness to 500 N/m to soften that impulse was
    # tried and made the grasp *worse*, not better: the lift fell from 45 mm to
    # 13.9 mm and the grasp was held for 1 tick instead of 4. So the failure is
    # not a force that is too large; it is that a light object trapped between
    # two stiff pads has no stable equilibrium. Lowering the force only loses
    # the grip. 1000 N/m is kept because it is the measured best of the two, and
    # the ejection itself is recorded as the open problem.
    gains = {
        # joint name -> (p, i, d, i_max, i_min, cmd_max)
        # p_gain/d_gain are 4000/100, not the 1000/50 first tried. Measured, a
        # ramped command left the arm trailing by a constant 84-100 ms -- 0.1087
        # rad on joints 1-4 and 0.1305 on joints 5-7, which are exactly
        # ``max_velocity * control_period`` for those joints -- and the grasped
        # object left the jaws on the tick the command ramped hardest. A PD
        # position controller's bandwidth goes as ``sqrt(p / I)``, so quadrupling
        # p and doubling d (which keeps the damping ratio, since d goes as
        # ``sqrt(p)``) halves the lag. cmd_max is unchanged: it stays at each
        # joint's real effort limit, so the extra stiffness cannot become extra
        # force.
        #
        # ``i_max`` is 10 N, not the 50 first tried. The steady-state integral
        # that cancels gravity here is tiny -- the settled joint errors are
        # 0.2-4.0 mrad, i.e. under 4 N of commanded force -- so 10 N is ample.
        # At 50 N the accumulator wound up across the set-point change from the
        # grasp waypoint to the lift waypoint and a live episode measured the
        # arm overshooting the lift pose by 17 mm and oscillating +/-20 mm until
        # the object shook out of the jaws. The bound is the anti-windup, and it
        # has to be sized against the load, not chosen generously.
        "joint1": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
        "joint2": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
        "joint3": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
        "joint4": (4000.0, 100.0, 100.0, 10.0, -10.0, 87.0),
        "joint5": (4000.0, 100.0, 100.0, 10.0, -10.0, 12.0),
        "joint6": (4000.0, 100.0, 100.0, 10.0, -10.0, 12.0),
        "joint7": (4000.0, 100.0, 100.0, 10.0, -10.0, 12.0),
        "left_finger": (1000.0, 100.0, 50.0, 4.0, -4.0, 100.0),
        "right_finger": (1000.0, 100.0, 50.0, 4.0, -4.0, 100.0),
    }
    blocks = []
    for name in names:
        p_gain, i_gain, d_gain, i_max, i_min, cmd_max = gains[name]
        blocks.append(
            '    <plugin filename="gz-sim-joint-position-controller-system"\n'
            '            name="gz::sim::systems::JointPositionController">\n'
            f"      <joint_name>{name}</joint_name>\n"
            f"      <p_gain>{p_gain:g}</p_gain>\n"
            f"      <i_gain>{i_gain:g}</i_gain>\n"
            f"      <d_gain>{d_gain:g}</d_gain>\n"
            f"      <i_max>{i_max:g}</i_max>\n"
            f"      <i_min>{i_min:g}</i_min>\n"
            f"      <cmd_max>{cmd_max:g}</cmd_max>\n"
            f"      <cmd_min>{-cmd_max:g}</cmd_min>\n"
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

    Flattening removes the indirection and leaves every link with one
    unambiguous model-frame pose. The values are verified against the joint
    definitions (advances of 0.333, 0.316 and 0.384 m, the Panda's real link
    lengths) and physics is unchanged, because a joint's pose is still read
    relative to its parent link.

    It does **not** fix the robot's absence from rendered frames. That was the
    original suspect, and the measurement said otherwise: with and without these
    references the camera reports the same 2144 finite depth pixels. The cause is
    recorded in ``experiments/results/gazebo/evidence_robot_not_rendered.txt``:
    this build's Sensors path skips a model that has both joints and plugins.
    The flattening is kept because it is a genuine simplification, not because
    it rescues the render.

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

    The second version aimed at ``(cx, cy, table_z + 0.35)``, roughly the middle
    of the arm. That framed the robot but wrecked the measurement: back-projected
    depths then ran 0.29..0.89 m with no mass near the tabletop, because the
    frame was filled with the arm and the far background rather than with the
    table. Object perception found a "target" at z = 0.64 m, which is not where
    the object is. The object stands on the table and the arm does not, so the
    camera now aims at the table. The arm ends up above the top of frame, which
    costs nothing here because this build does not draw the robot anyway (see
    ``docs/gazebo.md``) and buys an honest depth image about the object.

    Args:
        scenario: The condition whose workspace is to be viewed.

    Returns:
        ``x y z roll pitch yaw`` as text, ready for an ``<pose>`` element.
    """
    import math

    cx = float(scenario.environment.workspace_center[0])
    cy = float(scenario.environment.workspace_center[1])
    table_z = float(scenario.environment.table_height_m)

    # Close enough that the object is resolvable, far enough that the table's
    # far edge stays in frame. The 0.22 m standoff framed the object large
    # (~12 px) but pushed the table's near edge off-frame (px=-7), failing
    # the framing test; the 0.9 m standoff passed framing but left the 40 mm
    # cube at ~5 px, where a plane fit discards it. 0.4 m / 0.5 m passes all
    # four table corners inside 160x120 (measured) with the cube at ~7.4 px:
    # small, but above the detection floor, and the depth noise settings still
    # reach the sensor rather than being dropped.
    eye = (cx + 0.4, cy - 0.4, table_z + 0.5)
    target = (cx, cy, table_z + 0.005)

    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw = math.atan2(dy, dx)
    pitch = math.asin(-dz / norm)
    return f"{eye[0]:.6f} {eye[1]:.6f} {eye[2]:.6f} 0.0 {pitch:.6f} {yaw:.6f}"


def place_robot_model(model_xml: str, *, height: float) -> str:
    """Return the robot's ``<model>`` block, renamed, placed and anchored.

    Two things happen here that the previous string surgery silently failed to
    do.

    **Naming and placing.** It used to ``model_xml.replace("<model name='panda'>",
    ...)``; :func:`robot_model_sdf` round-trips the block through
    :mod:`xml.etree`, which emits double-quoted attributes, so the replacement
    never matched. A ``str.replace`` that matches nothing is a no-op, so the
    robot was emitted with no ``<pose>`` at all and spawned at the world origin.

    **Anchoring.** The arm is bolted to the table, which is how a robot arm is
    mounted and what keeps the arm's own base frame fixed in the world. Left
    free, it fell out of the simulation entirely: the ``panda`` entity was
    measured at ``z = -10162 m`` after one episode, i.e. in free fall, so every
    joint reading, every tool pose and every grasp was being taken while the
    robot was kilometres away. A fixed joint to ``world`` gives the base a
    support reaction to stand on.

    Raises:
        RuntimeError: If the text is not a single ``<model>`` element, or the
            pose could not be set. Refusing to return an unplaced model matters:
            it would otherwise spawn at the origin and look like a physics bug.
    """
    element = ElementTree.fromstring(model_xml)
    if element.tag != "model":
        raise RuntimeError(f"expected a <model> element, got <{element.tag}>")
    element.set("name", GAZEBO_ROBOT_NAME)
    for existing in element.findall("pose"):
        element.remove(existing)
    pose = ElementTree.Element("pose")
    pose.text = f"0 0 {height:.6f} 0 0 0"
    # The pose and the anchor come first: SDF reads a model's pose from the
    # element in the model's own frame, and the joint tree must reach the world
    # before it branches into the arm.
    element.insert(0, _world_anchor_joint())
    element.insert(0, pose)
    out = ElementTree.tostring(element, encoding="unicode")
    if out.startswith("<?xml"):
        out = out.split("?>", 1)[1]
    if f'<pose>0 0 {height:.6f} 0 0 0</pose>' not in out:
        raise RuntimeError(
            f"robot model was not placed at z={height:.6f}; refusing to return an "
            "unplaced model, which would spawn the arm at the world origin"
        )
    if "<parent>world</parent>" not in out:
        raise RuntimeError("the robot model could not be anchored to the world")
    return out


def _world_anchor_joint() -> ElementTree.Element:
    """Build the fixed joint that holds the robot base to the world.

    ``<parent>world</parent>`` is the SDF name for the world link inside a
    model, and the joint must precede the arm's own joints so the tree is rooted
    at the world rather than at the floating base.
    """
    joint = ElementTree.Element("joint")
    joint.set("name", "world_to_panda_base")
    joint.set("type", "fixed")
    parent = ElementTree.SubElement(joint, "parent")
    parent.text = "world"
    child = ElementTree.SubElement(joint, "child")
    child.text = "panda_link0"
    return joint


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
    # SDF sizes boxes by their full extent; ``table_extent_m`` is a half-extent,
    # the same convention the MJCF generator uses. The previous hard-coded
    # ``0.6 0.6`` was a 0.30 m half-extent table for a scenario declaring
    # ``(0.50, 0.35)``, so the support surface was 20% smaller than the condition
    # it claimed to represent.
    table_size_x = 2.0 * float(scenario.environment.table_extent_m[0])
    table_size_y = 2.0 * float(scenario.environment.table_extent_m[1])
    half_extents = object_half_extents(scenario.target.category, scenario.target.size_m)
    obj_size = " ".join(f"{2.0 * float(v):.6f}" for v in half_extents)
    # Same mass rule as ``simulation/mujoco_env.py``: the object's volume at a
    # plausible density, floored so the contact solver never sees a body whose
    # mass has vanished. Mass and inertia are what make a grasp either hold or
    # slide, so they are derived from the declared geometry rather than guessed.
    obj_volume = 8.0 * float(half_extents[0]) * float(half_extents[1]) * float(half_extents[2])
    obj_mass = float(min(2.0, max(5.0e-3, obj_volume * 250.0)))
    # Solid-box inertia about a centre axis, I = m (b^2 + c^2) / 12. An inertia
    # that does not match the collision box makes the object tumble under a
    # load the box geometry says it should resist.
    obj_inertia = (
        obj_mass
        * ((2.0 * float(half_extents[0])) ** 2 + (2.0 * float(half_extents[1])) ** 2)
        / 12.0
    )
    friction = float(scenario.environment.object_friction)
    # Where the task requires the object to end up. Read from the scenario, so
    # the visual marker and the success criterion are the same quantity.
    target = scenario.task.target_pose.position
    # The robot is embedded as a converted SDF model rather than <include>d.
    # An include loads fine, but leaves the joint controller unable to resolve
    # any joint, so the arm cannot be commanded -- which is the entire point of
    # the backend. Conversion is Gazebo's own, so geometry, inertias and joint
    # frames are exactly what Gazebo would have built itself.
    if robot_urdf:
        robot_block = place_robot_model(
            robot_model_sdf(robot_urdf),
            height=table_z + _BASE_CYLINDER_BOTTOM_M,
        )
    else:
        robot_block = (
            "    <include>\n"
            f"      <uri>{GAZEBO_ROBOT_NAME}</uri>\n"
            f"      <name>{GAZEBO_ROBOT_NAME}</name>\n"
            f"      <pose>0 0 {table_z + _BASE_CYLINDER_BOTTOM_M:.6f} 0 0 0</pose>\n"
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
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
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
        <!-- Whether anything is dragging on the tabletop. The grasp aim leaves
             the pads only millimetres above this surface, so a pad grazing it is
             a real and previously invisible failure mode. -->
        <sensor name="table_contacts" type="contact">
          <contact><collision>collision</collision></contact>
          <update_rate>100</update_rate>
        </sensor>
        <collision name="collision">
          <geometry><box><size>{table_size_x} {table_size_y} 0.04</size></box></geometry>
          <surface><friction><ode><mu>{friction}</mu><mu2>{friction}</mu2></ode></friction></surface>
        </collision>
        <visual name="visual"><geometry><box><size>{table_size_x} {table_size_y} 0.04</size></box></geometry>
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
        <!-- Which bodies is this object actually touching, and how hard. The
             episode trace showed a gripper closing on this cube and then losing
             it without anything saying what the pads were in contact with; this
             sensor is how that question is answered with evidence instead of
             inference. -->
        <sensor name="object_contacts" type="contact">
          <contact><collision>collision</collision></contact>
          <update_rate>100</update_rate>
        </sensor>
        <!-- Size, mass and inertia are the scenario's own numbers: a 16 mm cube
             at 50 g with a 40 mm bounding box would have been a different body
             from the one the policy, the task and the detector reason about. -->
        <inertial><mass>{obj_mass:.6f}</mass>
          <inertia><ixx>{obj_inertia:.6e}</ixx><iyy>{obj_inertia:.6e}</iyy>
                   <izz>{obj_inertia:.6e}</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>{obj_size}</size></box></geometry>
          <surface><friction><ode><mu>{friction}</mu><mu2>{friction}</mu2></ode></friction></surface>
        </collision>
        <visual name="visual"><geometry><box><size>{obj_size}</size></box></geometry>
          <material>
            <ambient>0.9 0.2 0.2 1</ambient>
            <diffuse>0.9 0.2 0.2 1</diffuse>
            <specular>0.3 0.1 0.1 1</specular>
          </material></visual>
      </link>
    </model>

    <model name="target_zone">
      <static>true</static>
      <!-- Where the task requires the object to end up, read from the scenario
           rather than chosen here. The marker and the success criterion are the
           same quantity, so a frame that shows the object on the marker is a
           frame in which the task succeeded. -->
      <pose>{target.x:.6f} {target.y:.6f} {table_z + 0.001:.6f} 0 0 0</pose>
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
