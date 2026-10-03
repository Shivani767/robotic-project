r"""Decide, from one real saved Gazebo depth frame, what the depth numbers mean.

The detector fits a support plane to the back-projected cloud and calls anything
above it the object. On the real frame the least-squares plane came out tilted
(slope ~0.149 in x) instead of flat at the table top (z = 0), which selects half
the table as "the object". A flat table can only produce a tilted fit if the
back-projection is systematically wrong, so this measures which of the two
possible depth conventions Gazebo is publishing:

* **axial** -- ``z`` is the distance along the camera's optical axis, which is
  what the current code assumes; and
* **radial** -- ``z`` is the distance from the eye to the surface along the
  viewing ray, which is larger than the axial depth by ``1/cos(angle)`` and
  varies across the frame.

Whichever convention makes the tabletop flat at z = 0 is the correct one. Run::

    .venv/bin/python scripts/gazebo_depth_convention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_ci.gazebo.world import (
    _camera_pose,
    camera_intrinsics_from_scenario,
    object_pose_from_scenario,
)
from robot_ci.scenarios import default_scenario

#: The tabletop's true height, from the SDF's table box (top face at z = 0).
TABLE_TOP_Z = 0.0


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the camera's ``(eye, forward, right, up)`` from the SDF helper."""
    scenario = default_scenario(seed=42)
    x, y, z, _roll, pitch, yaw = (float(v) for v in _camera_pose(scenario).split())
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return (
        np.array([x, y, z]),
        np.array([cy * cp, sy * cp, -sp]),
        np.array([-sy, cy, 0.0]),
        np.array([cy * sp, sy * sp, cp]),
    )


def _backproject(depth: np.ndarray, radial: bool) -> np.ndarray:
    """Back-project every finite pixel under the chosen depth convention."""
    eye, forward, right, up = _geometry()
    _w, _h, fx, fy, cx, cy = camera_intrinsics_from_scenario(default_scenario(seed=42))
    rows, cols = np.nonzero(np.isfinite(depth))
    z = depth[rows, cols].astype(np.float64)
    if radial:
        # A ray makes angle theta with the optical axis and z = d*cos(theta), so
        # the axial depth is smaller by exactly that factor. Scaling by it here
        # turns a radial reading into the axial one the pinhole model wants.
        nx = (cols + 0.5 - cx) / fx
        ny = (rows + 0.5 - cy) / fy
        z = z / np.sqrt(nx * nx + ny * ny + 1.0)
    return (
        eye[None, :]
        + z[:, None] * forward[None, :]
        - ((cols + 0.5 - cx) / fx * z)[:, None] * right[None, :]
        - ((rows + 0.5 - cy) / fy * z)[:, None] * up[None, :]
    )


def _plane_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the least-squares plane coefficients and per-point residuals."""
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(points.shape[0])])
    solution, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    return solution, points[:, 2] - design @ solution


def _robust_plane(points: np.ndarray, iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Fit the support plane while ignoring what stands on top of it.

    The table is the *lower* surface, so the fit is one-sided: points above the
    current estimate -- the object, and anything further back in the scene --
    are downweighted instead of being allowed to rotate the plane. A plain
    least-squares fit over every point does the opposite, which is why the real
    frame produced a plane tilted by 0.149 and a "detection" on the tabletop.

    Returns ``(coefficients, inlier_mask)``.
    """
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(points.shape[0])])
    keep = np.ones(points.shape[0], dtype=bool)
    solution = np.zeros(3)
    for _ in range(iterations):
        solution, *_ = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)
        residual = points[:, 2] - design @ solution
        # One-sided: only upward departures are treated as outliers, so the
        # estimate can only ever climb, never be pulled under the tabletop.
        spread = float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
        keep = residual < max(3.0 * spread, 0.004)
        if int(keep.sum()) < 32:
            break
    return solution, keep


def main() -> int:
    """Compare the two conventions against the known flat tabletop."""
    path = Path(".scratch/camera_probe/depth.npy")
    if not path.exists():
        print(f"missing {path}; run scripts/gazebo_camera_probe.py first")
        return 1
    depth = np.load(path)
    scenario = default_scenario(seed=42)
    cube = np.array(object_pose_from_scenario(scenario))

    print(f"frame {depth.shape}, finite {int(np.isfinite(depth).sum())}")
    print(f"true cube {np.round(cube, 4).tolist()}, tabletop z = {TABLE_TOP_Z}\n")
    for label, radial in (("axial (current code)", False), ("radial", True)):
        points = _backproject(depth, radial)
        solution, residual = _plane_fit(points)
        # Points near the cube are the object; everything else should be table.
        dist = np.linalg.norm(points - cube[None, :], axis=1)
        cube_mask = dist < 0.06
        table_z = points[~cube_mask, 2]
        print(f"--- {label} ---")
        print(
            f"  plane slope dx={solution[0]:+.5f} dy={solution[1]:+.5f} z0={solution[2]:+.5f}"
        )
        print(f"  |slope| = {float(np.hypot(solution[0], solution[1])):.5f} (0 means flat)")
        print(f"  residual rms = {float(np.sqrt((residual**2).mean())):.5f} m")
        print(
            f"  non-cube z: median={np.median(table_z):+.4f} "
            f"p05={np.percentile(table_z, 5):+.4f} p95={np.percentile(table_z, 95):+.4f}"
        )
        print(
            f"  cube-cloud centroid = {np.round(points[cube_mask].mean(axis=0), 4).tolist()}"
            f"  (n={int(cube_mask.sum())})"
        )
        print(f"  points with residual>0.015: {int((residual > 0.015).sum())}")
        print(f"  robust one-sided IRLS plane: {_robust_plane(points)!r}")
        print()

    # The cube's own pixel, as an independent check on the convention.
    eye, forward, right, up = _geometry()
    _w, _h, fx, fy, cx, cy = camera_intrinsics_from_scenario(scenario)
    local = cube - eye
    axial = float(local @ forward)
    col = cx + (-(local @ right) / axial) * fx
    row = cy - ((local @ up) / axial) * fy
    ri, ci = int(round(row)), int(round(col))
    print(f"cube pixel (col={ci}, row={ri}), true axial depth {axial:.4f} m")
    patch = depth[max(ri - 2, 0) : ri + 3, max(ci - 2, 0) : ci + 3]
    print(f"  depth there = {np.round(patch.flatten(), 4).tolist()}")
    radial_expected = axial * float(
        np.sqrt(((ci + 0.5 - cx) / fx) ** 2 + ((ri + 0.5 - cy) / fy) ** 2 + 1.0)
    )
    print(f"  radial equivalent of {axial:.4f} m would be {radial_expected:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
