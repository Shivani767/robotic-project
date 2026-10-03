"""A software 3-D view of the Panda, driven by recorded joint states.

Why this exists
---------------
The live browser dashboard already shows the arm in 3-D, and it does that from
the *same* URDF and the *same* live joint stream. A recorded experiment needs
that view to survive the episode, so this module reproduces it offline: the
recorded joint angles are pushed through the URDF's own kinematic tree and
projected with a plain pinhole camera, and the result is a frame per tick.

What it is not
--------------
It is **not** camera footage and it is never presented as such. There is no
lighting model, no materials and no occlusion beyond a painter's sort, so it
cannot be mistaken for the simulator's own renderer -- and the composed video
labels it ``3D VIEW (URDF joint states)`` precisely so the distinction is on the
frame. The Gazebo camera is a separate, real sensor stream.

The scene is drawn from *measured* positions: the object is placed where the
recorded ground truth says it was, and the table and target zone come from the
scenario the episode ran.

Coordinates
-----------
Everything is in the robot base frame, which is also the world frame the Panda
is spawned at. The camera is placed the way the browser viewer places it: a
three-quarter view from above the front-right, looking at the workspace.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Camera",
    "Kinematics",
    "SceneInfo",
    "forward_kinematics",
    "load_kinematics",
    "render_frame",
]

#: Segment thickness in pixels at the nominal render size, before scaling.
_BASE_THICKNESS = 3

#: Link colours, brightest at the tip so the approach axis reads even when the
#: arm is folded on itself in a 2-D projection.
_LINK_COLOURS = [
    (150, 158, 168),
    (168, 176, 186),
    (186, 194, 204),
    (204, 212, 222),
    (222, 230, 240),
    (238, 244, 250),
    (248, 251, 255),
]
_FINGER_COLOUR = (120, 200, 255)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the rotation for URDF fixed-axis roll-pitch-yaw.

    URDF defines ``rpy`` as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. Using any other
    order silently misplaces the links, and the error only becomes obvious at
    the wrist, where three axes meet.
    """
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the rotation of ``angle`` radians about a unit ``axis``."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = (np.asarray(axis, dtype=np.float64) / norm).tolist()
    c, s = np.cos(angle), np.sin(angle)
    k = 1.0 - c
    return np.array(
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ],
        dtype=np.float64,
    )


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a rotation and a translation."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


@dataclass(frozen=True)
class Kinematics:
    """The robot's kinematic tree, read from its URDF.

    Attributes:
        joints: One entry per joint, in file order.
        link_names: Every link declared in the URDF.
    """

    joints: tuple[dict[str, Any], ...]
    link_names: tuple[str, ...]


def load_kinematics(urdf_path: str | Path) -> Kinematics:
    """Read the kinematic tree out of a URDF, exactly as the viewer does.

    Reusing the browser viewer's own extraction keeps the recorded 3-D view and
    the live one describing the same robot. A second, hand-written description
    is how a visualization ends up disagreeing with the thing it depicts.

    Args:
        urdf_path: Path to the robot URDF.

    Returns:
        A :class:`Kinematics` built from the file.
    """
    from robot_ci.gazebo.live import robot_kinematics

    data = robot_kinematics(urdf_path)
    return Kinematics(
        joints=tuple(dict(joint) for joint in data["joints"]),
        link_names=tuple(data["links"]),
    )


def forward_kinematics(kin: Kinematics, q: np.ndarray) -> dict[str, np.ndarray]:
    """Return the world position of every joint child frame for ``q``.

    The chain is evaluated in URDF file order starting from the identity at the
    robot base, applying each joint's fixed origin and then its motion (rotation
    about ``axis`` for revolute joints, translation along ``axis`` for
    prismatic ones). The mapping mirrors ``viewer.js`` ``computeFrames``: the
    same origins, axes and ordering, so the recorded frames and the live browser
    view agree on where each link is for the same joint angles.

    Args:
        kin: Kinematic tree from :func:`load_kinematics`.
        q: Joint values in the order of ``kin.joints``.

    Returns:
        Mapping of link/child name to ``(3,)`` world position. The base
        entry is the robot base origin.
    """
    values = np.asarray(q, dtype=np.float64).reshape(-1)
    transform = np.eye(4)
    positions: dict[str, np.ndarray] = {"base": np.zeros(3)}
    for index, joint in enumerate(kin.joints):
        rotation = _rpy_matrix(*[float(v) for v in joint["rpy"]])
        translation = np.asarray(joint["xyz"], dtype=np.float64)
        transform = transform @ _homogeneous(rotation, translation)
        value = float(values[index]) if index < values.size else 0.0
        joint_type = str(joint.get("type", "fixed"))
        axis = np.asarray(joint.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
        if joint_type == "revolute":
            transform = transform @ _homogeneous(_axis_matrix(axis, value), np.zeros(3))
        elif joint_type == "prismatic":
            norm = float(np.linalg.norm(axis))
            direction = axis / norm if norm > 1e-12 else axis
            transform = transform @ _homogeneous(np.eye(3), direction * value)
        positions[str(joint["child"])] = transform[:3, 3].copy()
    return positions


class Camera:
    """A plain pinhole camera looking at the workspace.

    Deliberately the simplest camera that produces a correct projection. There
    is no distortion model because the view is a diagram, not a sensor.
    """

    def __init__(
        self,
        *,
        eye: tuple[float, float, float] = (0.85, -0.85, 1.05),
        target: tuple[float, float, float] = (0.0, 0.10, 0.35),
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        fov_deg: float = 42.0,
        width: int = 480,
        height: int = 420,
    ) -> None:
        """Place the camera.

        Args:
            eye: Camera position in the base frame.
            target: Point the camera looks at.
            up: World up vector.
            fov_deg: Vertical field of view.
            width: Render width in pixels.
            height: Render height in pixels.
        """
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        forward = self.target - self.eye
        forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
        right = np.cross(forward, np.asarray(up, dtype=np.float64))
        right = right / max(float(np.linalg.norm(right)), 1e-9)
        true_up = np.cross(right, forward)
        # Rows are the camera axes, so world -> camera is a single matmul.
        self._basis = np.stack([right, true_up, forward])
        self._focal = 0.5 * self.height / np.tan(np.radians(fov_deg) / 2.0)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixels.

        Args:
            points: ``(N, 3)`` world positions.

        Returns:
            ``(screen, depth)`` where ``screen`` is ``(N, 2)`` in pixels with the
            origin at the top-left, and ``depth`` is ``(N,)`` in metres along the
            view axis. Points at or behind the camera get ``NaN`` screen
            coordinates so the caller can drop them rather than draw them
            mirrored through the projection plane.
        """
        local = (np.asarray(points, dtype=np.float64) - self.eye) @ self._basis.T
        depth = local[:, 2]
        safe = np.where(depth > 1e-6, depth, 1e-6)
        u = self.width * 0.5 + self._focal * (local[:, 0] / safe)
        v = self.height * 0.5 - self._focal * (local[:, 1] / safe)
        screen = np.stack([u, v], axis=1)
        screen[depth <= 1e-6] = np.nan
        return screen, depth


@dataclass(frozen=True)
class SceneInfo:
    """Static scene geometry for the 3-D view, from the generated world.

    The values mirror ``robot_ci.gazebo.world``'s SDF: the tabletop is a
    0.6 x 0.6 x 0.04 box centred on the workspace with its top at
    ``table_top_z``, the target zone is a 0.06 m disc at workspace centre +
    (0.18, -0.12) in world xy, and the object is the drawn 0.04 m cube.
    Passing them in rather than importing the scenario keeps the renderer a
    pure function of what was measured.
    """

    table_center: tuple[float, float] = (0.0, 0.1)
    table_half: float = 0.3
    table_thickness: float = 0.04
    table_top_z: float = 0.0
    zone_center: tuple[float, float] = (0.18, -0.02)
    zone_radius: float = 0.06
    object_position: tuple[float, float, float] | None = None
    object_half: float = 0.02


def _fill_polygon(
    draw: Any, camera: Camera, points: np.ndarray, colour: tuple[int, int, int]
) -> None:
    """Project a 3-D polygon and fill it, dropping it if any vertex is behind."""
    screen, _depth = camera.project(np.asarray(points, dtype=np.float64))
    if np.any(np.isnan(screen)):
        return
    draw.polygon([tuple(float(v) for v in p) for p in screen], fill=colour)


def _draw_segment(
    draw: Any, camera: Camera, start: np.ndarray, end: np.ndarray, colour: tuple[int, int, int],
    width: int
) -> None:
    """Project a 3-D line segment and stroke it, skipping it if behind."""
    screen, _depth = camera.project(np.asarray([start, end], dtype=np.float64))
    if np.any(np.isnan(screen)):
        return
    draw.line([tuple(float(v) for v in p) for p in screen], fill=colour, width=width)


def render_frame(
    kin: Kinematics,
    joints: np.ndarray,
    scene: SceneInfo,
    camera: Camera | None = None,
) -> Any:
    """Render one frame of the Panda from recorded joint angles.

    The arm is drawn as thick projected segments rather than solid meshes: a
    skeleton makes the *joint angles* legible, which is the thing a recorded
    experiment is trying to show, and it avoids implying a fidelity this
    renderer does not have. Links are painted back-to-front by projected depth
    so the near ones overlap correctly.

    Args:
        kin: Kinematic tree from :func:`load_kinematics`.
        joints: Arm joint angles in radians.
        scene: Static scene geometry, including the object's measured position.
        camera: Viewpoint; a three-quarter view is used when omitted.

    Returns:
        A ``PIL.Image`` in RGB mode.
    """
    from PIL import Image, ImageDraw

    view = camera or Camera(
        width=max(scene_extent_px(scene) * 2, 320),
        height=max(scene_extent_px(scene) * 2, 300),
    )
    image = Image.new("RGB", (view.width, view.height), (18, 20, 24))
    draw = ImageDraw.Draw(image)

    _fill_polygon(
        draw,
        view,
        np.array(
            [
                [scene.table_center[0] - scene.table_half, scene.table_center[1] - scene.table_half,
                 scene.table_top_z],
                [scene.table_center[0] + scene.table_half, scene.table_center[1] - scene.table_half,
                 scene.table_top_z],
                [scene.table_center[0] + scene.table_half, scene.table_center[1] + scene.table_half,
                 scene.table_top_z],
                [scene.table_center[0] - scene.table_half, scene.table_center[1] + scene.table_half,
                 scene.table_top_z],
            ]
        ),
        (72, 78, 88),
    )
    # The target zone, as the disc the task places into.
    _fill_polygon(
        draw,
        view,
        np.array(
            [
                [scene.zone_center[0] + scene.zone_radius * c, scene.zone_center[1]
                 + scene.zone_radius * s, scene.table_top_z + 0.002]
                for c, s in (
                    (1, 0), (0.7, 0.7), (0, 1), (-0.7, 0.7), (-1, 0),
                    (-0.7, -0.7), (0, -1), (0.7, -0.7),
                )
            ]
        ),
        (46, 150, 82),
    )
    if scene.object_position is not None:
        _fill_box(
            draw,
            view,
            np.asarray(scene.object_position, dtype=np.float64),
            scene.object_half,
            (214, 74, 66),
        )

    frames = forward_kinematics(kin, np.asarray(joints, dtype=np.float64))
    thickness = max(2, view.width // 160)
    # The Panda is a serial chain, so the returned frames are already in chain
    # order from the base outwards. Consecutive positions are therefore the
    # links, and drawing them in depth order makes near links overlap far ones.
    chain = list(frames.items())
    segments = [
        (name_a, pos_a, pos_b)
        for (name_a, pos_a), (_name_b, pos_b) in itertools.pairwise(chain)
    ]
    # Painter's algorithm: far links first so near ones overlap them.
    segments.sort(key=lambda item: -float(np.linalg.norm((item[1] + item[2]) * 0.5 - view.eye)))
    for index, (name, start, end) in enumerate(segments):
        colour = _FINGER_COLOUR if "finger" in name else _LINK_COLOURS[min(6, index // 2)]
        _draw_segment(draw, view, start, end, colour, thickness)
    return image


def _fill_box(
    draw: Any, camera: Camera, centre: np.ndarray, half: float, colour: tuple[int, int, int]
) -> None:
    """Draw an axis-aligned cube of edge ``2 * half`` at ``centre``."""
    signs = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    corners = [centre + half * np.array([float(a), float(b), float(c)]) for a, b, c in signs]
    faces = [
        (0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
        (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5),
    ]
    for face in faces:
        _fill_polygon(draw, camera, np.array([corners[i] for i in face]), colour)


def scene_extent_px(scene: SceneInfo) -> int:
    """Return a sensible default render size for a scene, in pixels."""
    return 480
