"""Telemetry-driven MuJoCo renderer for the real Panda robot.

This module loads the repository's genuine Franka Emika Panda model
(``mujoco/models/franka_emika_panda/panda.xml`` together with all 67 mesh
assets) and augments it with the scenario's table, target zone and cube so the
recorded episode can be re-projected faithfully.

Every frame is driven from measured joint telemetry -- arm joints j1..j7, the
finger aperture, and (because the evidence run *does* record object pose) the
cube's free-joint position. Nothing is hand-posed or re-simulated.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
PANDA_XML = REPO / "mujoco/models/franka_emika_panda/panda.xml"

# Offscreen framebuffer large enough for 1080p renders, soft shadows, 4x MSAA.
_VISUAL = (
    '<visual>'
    '<global offwidth="1920" offheight="1080"/>'
    '<quality shadowsize="4096" offsamples="4"/>'
    '<headlight ambient="0.35 0.35 0.38" diffuse="0.5 0.5 0.5" '
    'specular="0.15 0.15 0.15"/>'
    '</visual>'
)

# Scenario geometry mirrored from experiments/configs/demo_seed3.yaml:
# table top at z=0 (half-extent 0.5 x 0.35 m centred at (0, 0.1)), placement
# disc at the task target pose (0.2, 0.15) radius 0.06 m, 16 mm cube
# (size_m 0.008 is the half-extent).
_SCENE = (
    '<light name="key" pos="0.45 -0.6 1.5" dir="0 -0.12 -1" '
    'diffuse="0.72 0.72 0.75" specular="0.35 0.35 0.35" castshadow="true"/>'
    '<light name="fill" pos="-0.85 0.55 1.2" dir="0 0 -1" '
    'diffuse="0.32 0.32 0.38" castshadow="false"/>'
    '<light name="rim" pos="0.1 0.85 1.1" dir="0 0 -1" '
    'diffuse="0.22 0.22 0.28" castshadow="false"/>'
    '<geom name="table" type="box" size="0.5 0.35 0.02" pos="0 0.1 -0.02" '
    'rgba="0.30 0.31 0.33 1"/>'
    '<geom name="zone" type="cylinder" size="0.06 0.001" '
    'pos="0.2 0.15 0.0015" rgba="0.20 0.65 0.40 0.55"/>'
    '<body name="cube" pos="0.02 0.226 0.008">'
    '<joint name="cube_free" type="free"/>'
    '<geom name="cube_g" type="box" size="0.008 0.008 0.008" '
    'rgba="0.85 0.16 0.16 1"/>'
    '</body>'
)

# Neutral studio background.
_BG = (0.87, 0.88, 0.90)


class PandaScene:
    """A loaded, augmented Panda scene rendered from measured telemetry."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        xml = PANDA_XML.read_text(encoding="utf-8")
        assets = (PANDA_XML.parent / "assets").resolve()
        xml = xml.replace('meshdir="assets"', f'meshdir="{assets}"')
        xml = xml.replace("</worldbody>", _SCENE + "</worldbody>")
        xml = xml.replace("</mujoco>", _VISUAL + "</mujoco>")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Scene geom/body ids.
        self._name = mujoco.mj_name2id
        self._zone = self._name(self.model, mujoco.mjtObj.mjOBJ_GEOM, "zone")
        self._table = self._name(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")

        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.enable_shadows = True
        # Neutral studio background colour.
        self.model.vis.headlight.ambient[:] = [0.35, 0.35, 0.38]

        self.width, self.height = width, height
        # Fixed three-quarter camera: keeps arm, gripper, cube and zone in frame.
        self._cam = self._make_camera()

    def _make_camera(
        self,
        azimuth: float = 130.0,
        elevation: float = -22.0,
        distance: float = 1.65,
        lookat: tuple[float, float, float] = (0.08, 0.11, 0.10),
    ) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = distance
        cam.lookat = list(lookat)
        return cam

    @staticmethod
    def studio_backdrop(height: int, width: int) -> np.ndarray:
        """A soft neutral studio gradient to composite the scene onto."""
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        # Vertical gradient, brighter at top; subtle horizontal falloff.
        base = 0.965 - 0.06 * y
        base = base - 0.015 * (x - 0.5)
        rgb = np.clip(base, 0.0, 1.0)
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
        return (rgb * 255.0).astype(np.uint8)

    def render_studio(
        self,
        joints: np.ndarray,
        aperture_m: float,
        cube_xyz: tuple[float, float, float],
        cube_quat: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
        camera: mujoco.MjvCamera | None = None,
    ) -> np.ndarray:
        """Render one frame composited onto a neutral studio backdrop."""
        rgb, mask = self.render_masked(joints, aperture_m, cube_xyz, cube_quat, camera)
        backdrop = self.studio_backdrop(self.height, self.width)
        out = np.where(mask[:, :, None], rgb, backdrop)
        return out.astype(np.uint8)

    def three_quarter_camera(self) -> mujoco.MjvCamera:
        """The fixed three-quarter hero camera."""
        return self._cam

    def plan_camera(self) -> mujoco.MjvCamera:
        """A top-down plan view: whole workspace, cube start -> zone target."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = 90.0
        cam.elevation = -88.0
        cam.distance = 1.05
        cam.lookat = [0.10, 0.16, 0.0]
        return cam

    def render(
        self,
        joints: np.ndarray,
        aperture_m: float,
        cube_xyz: tuple[float, float, float],
        cube_quat: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
        camera: mujoco.MjvCamera | None = None,
    ) -> np.ndarray:
        """Project one frame from measured joint/aperture/object states."""
        d = self.data
        d.qpos[:7] = np.asarray(joints, dtype=np.float64)
        # Aperture is the sum of both finger offsets (see simulation/mujoco.py).
        finger = max(0.0, min(0.04, float(aperture_m) / 2.0))
        d.qpos[7] = finger
        d.qpos[8] = finger
        d.qpos[9:12] = cube_xyz
        d.qpos[12:16] = cube_quat
        mujoco.mj_forward(self.model, d)

        self.renderer.update_scene(d, camera=camera or self._cam)
        return self.renderer.render()

    def render_masked(
        self,
        joints: np.ndarray,
        aperture_m: float,
        cube_xyz: tuple[float, float, float],
        cube_quat: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
        camera: mujoco.MjvCamera | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb, mask) where mask marks foreground pixels (bool)."""
        d = self.data
        d.qpos[:7] = np.asarray(joints, dtype=np.float64)
        finger = max(0.0, min(0.04, float(aperture_m) / 2.0))
        d.qpos[7] = finger
        d.qpos[8] = finger
        d.qpos[9:12] = cube_xyz
        d.qpos[12:16] = cube_quat
        mujoco.mj_forward(self.model, d)

        self.renderer.update_scene(d, camera=camera or self._cam)
        rgb = self.renderer.render()
        # Second pass with segmentation to obtain an object mask.
        self.renderer.enable_segmentation_rendering()
        try:
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        geom_ids = seg[:, :, 0].astype(np.int32)
        mask = geom_ids >= 0
        return rgb, mask

