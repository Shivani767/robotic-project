r"""Telemetry-driven 3-D reconstruction of the Panda pick-and-place episode.

This module is the renderer behind the LinkedIn demo video.  It is deliberately
*reconstruction*, not simulation:

* The articulated robot is the real Franka Emika Panda loaded from the project's
  ``mujoco/models/franka_emika_panda/panda.xml`` together with all 67 mesh files
  in ``.../assets`` (the same CAD-derived visuals the sim uses).  No lines,
  capsules or stick figures are drawn -- the actual link meshes are rendered and
  the joint transforms come from MuJoCo's forward kinematics.
* Each frame is posed purely from the recorded joint telemetry of the evidence
  run (j1..j7) plus the recorded gripper aperture.  The red cube is placed at
  the recorded object pose (obj_x, obj_y, obj_z).  Nothing is re-simulated and
  no grasp physics is invented: the cube follows the measured object pose.
* The table and the evidence-supported target disc mirror the generated world
  (table half-extent 0.5 x 0.35 m centred on (0, 0.1), top at z = 0; placement
  target at (0.2, 0.15) with a 0.06 m radius).

The finger actuators are driven from the recorded aperture exactly like the
simulation does: ``qpos_finger = clamp(aperture_m / 2, 0, 0.04)`` (see
``src/robot_ci/simulation/mujoco.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PANDA_XML = REPO / "mujoco/models/franka_emika_panda/panda.xml"

# Scene geometry mirrors the generated world for the demo scenario (see
# experiments/configs/demo_seed3.yaml and the sim world builder).
TABLE_CENTER = (0.0, 0.1)
TABLE_HALF = (0.5, 0.35)
TABLE_TOP_Z = 0.0
TABLE_THICK = 0.02
ZONE_CENTER = (0.2, 0.15)
ZONE_RADIUS = 0.06
OBJECT_HALF = 0.008  # size_m is the half extent -> 16 mm cube

_ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6", "j7")
_FINGER = 0.04  # max finger travel per side (m)


@dataclass(frozen=True)
class Camera:
    """A fixed three-quarter camera (MuJoCo free-camera parameters)."""

    azimuth: float
    elevation: float
    distance: float
    lookat: tuple[float, float, float]
    width: int = 1920
    height: int = 1080


def _inject_scene(xml: str) -> str:
    """Resolve mesh paths and append the table, target disc and cube to the model."""
    assets = (PANDA_XML.parent / "assets").resolve()
    xml = xml.replace('meshdir="assets"', f'meshdir="{assets}"')

    # Larger offscreen framebuffer + a neutral studio backdrop and soft shadows.
    visual = (
        '<visual>'
        '<global offwidth="3840" offheight="2160"/>'
        '<quality shadowsize="2048" offsamples="2"/>'
        '<headlight ambient="0.42 0.42 0.42" diffuse="0.5 0.5 0.5" specular="0.12 0.12 0.12"/>'
        '<rgba haze="0.11 0.12 0.15 1"/>'
        '</visual>'
    )
    xml = xml.replace("</mujoco>", visual + "</mujoco>")

    tx, ty = TABLE_CENTER
    zx, zy = ZONE_CENTER
    extra = f"""
<light name="key" pos="0.5 -0.55 1.35" dir="0 -0.08 -1" diffuse="0.72 0.72 0.72" specular="0.3 0.3 0.3" castshadow="true"/>
<light name="fill" pos="-0.75 0.5 1.1" dir="0 0 -1" diffuse="0.34 0.34 0.4" castshadow="false"/>
<geom name="table" type="box" size="{TABLE_HALF[0]} {TABLE_HALF[1]} {TABLE_THICK}"
       pos="{tx} {ty} {TABLE_TOP_Z - TABLE_THICK}" rgba="0.30 0.31 0.33 1" friction="0.6 0.005 0.0001"/>
<geom name="zone" type="cylinder" size="{ZONE_RADIUS} 0.0012"
       pos="{zx} {zy} {TABLE_TOP_Z + 0.0012}" rgba="0.20 0.68 0.42 0.55"/>
<body name="cube" pos="0 0 0">
  <joint name="cube_free" type="free"/>
  <geom name="cube_g" type="box" size="{OBJECT_HALF} {OBJECT_HALF} {OBJECT_HALF}" rgba="0.86 0.16 0.16 1"/>
</body>
"""
    return xml.replace("</worldbody>", extra + "</worldbody>")



class PandaRenderer:
    """Render fixed-camera frames of the Panda posed from measured telemetry."""

    def __init__(self, camera: Camera, width: int | None = None, height: int | None = None):
        import mujoco

        self._mj = mujoco
        xml = _inject_scene(PANDA_XML.read_text(encoding="utf-8"))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.width = width or camera.width
        self.height = height or camera.height
        self.renderer = mujoco.Renderer(self.model, self.width, self.height)
        self.camera = camera
        self._mjcam = mujoco.MjvCamera()
        self._mjcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._mjcam.azimuth = camera.azimuth
        self._mjcam.elevation = camera.elevation
        self._mjcam.distance = camera.distance
        self._mjcam.lookat[:] = camera.lookat

        self._arm_adr = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINTS
        ]
        self._cube_adr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")

    def pose(self, joints: np.ndarray, aperture_m: float, object_xyz: np.ndarray) -> None:
        """Apply a measured joint state, aperture and object pose, then run FK."""
        d = self.data
        for adr, val in zip(self._arm_adr, joints, strict=True):
            d.qpos[adr] = float(val)
        finger = min(max(aperture_m / 2.0, 0.0), _FINGER)
        d.qpos[7] = finger  # left finger
        d.qpos[8] = finger  # right finger
        d.qpos[self._cube_adr + 0 : self._cube_adr + 3] = np.asarray(object_xyz, dtype=float)
        d.qpos[self._cube_adr + 3 : self._cube_adr + 7] = (1.0, 0.0, 0.0, 0.0)  # identity quat
        self._mj.mj_forward(self.model, d)

    def render(self) -> np.ndarray:
        """Return an RGB uint8 frame at the configured resolution."""
        self.renderer.update_scene(self.data, camera=self._mjcam)
        return self.renderer.render()


def interpolate(rows: list[dict], pos: float) -> dict:
    """Linearly interpolate a telemetry record at float index ``pos``.

    ``pos`` ranges over the measured samples; endpoints are preserved exactly so
    the measured start/end states are never altered and no motion is invented --
    only the samples *between* the recorded ticks are filled in for smoothness.
    """
    n = len(rows)
    if pos <= 0:
        return rows[0]
    if pos >= n - 1:
        return rows[-1]
    i = int(math.floor(pos))
    frac = pos - i
    a, b = rows[i], rows[i + 1]
    out: dict = {}
    for key in (
        "sim_t_s", "aperture_m", "obj_x", "obj_y", "obj_z",
        "j1", "j2", "j3", "j4", "j5", "j6", "j7",
    ):
        out[key] = a[key] + (b[key] - a[key]) * frac
    out["step"] = b["step"]
    out["phase"] = b["phase"]
    out["grasped"] = 1 if b["grasped"] else 0
    return out

