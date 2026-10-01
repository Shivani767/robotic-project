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
from typing import Any
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
    from pathlib import Path

    cx, cy = (
        float(scenario.environment.workspace_center[0]),
        float(scenario.environment.workspace_center[1]),
    )
    table_z = float(scenario.environment.table_height_m)
    ox, oy, oz = object_pose_from_scenario(scenario)
    # Gazebo resolves <uri> through its resource path, so a plain filesystem
    # path is refused with "Unable to find uri[...]". file:// makes it an
    # absolute URI, which is what the include actually needs.
    robot_uri = (
        f"file://{escape(str(Path(robot_urdf).resolve()))}" if robot_urdf else GAZEBO_ROBOT_NAME
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
          <material><ambient>0.7 0.7 0.7 1</ambient></material></visual>
      </link>
    </model>

    <include>
      <uri>{robot_uri}</uri>
      <name>{GAZEBO_ROBOT_NAME}</name>
      <pose>0 0 {table_z} 0 0 0</pose>
    </include>

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
          <material><ambient>0.9 0.2 0.2 1</ambient></material></visual>
      </link>
    </model>

    <model name="target_zone">
      <static>true</static>
      <pose>{cx + 0.18:.6f} {cy - 0.12:.6f} {table_z + 0.001:.6f} 0 0 0</pose>
      <link name="link">
        <visual name="visual"><geometry><cylinder><radius>0.06</radius><length>0.002</length></cylinder></geometry>
          <material><ambient>0.1 0.9 0.3 0.35</ambient></material></visual>
      </link>
    </model>

    <model name="camera_rig">
      <static>true</static>
      <pose>{cx:.6f} {cy - 0.35:.6f} {table_z + 0.45:.6f} 0 0.5 0</pose>
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
