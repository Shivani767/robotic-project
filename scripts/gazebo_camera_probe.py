r"""Capture one real Gazebo frame plus the perception pipeline's diagnosis.

Saves the RGB frame, the depth array and a JSON report into
``.scratch/camera_probe/`` so the back-projection maths can be checked against
the pixels the simulator actually produced -- rather than against a
self-consistent pair of formulas that may both be wrong.

Run inside the container::

    docker run --rm -v "$PWD":/workspace/robot-ci -w /workspace/robot-ci \\
        robot-ci-gazebo-demo:latest \\
        bash -c 'source /opt/ros/jazzy/setup.bash && python3 scripts/gazebo_camera_probe.py'
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    """Grab a frame, run perception, and write the evidence bundle."""
    from robot_ci.gazebo.world import (
        _camera_pose,
        camera_intrinsics_from_scenario,
        object_pose_from_scenario,
    )
    from robot_ci.scenarios import default_scenario
    from robot_ci.simulation import create_backend
    from robot_ci.simulation.gazebo import GazeboBackend

    out = Path(".scratch/camera_probe")
    out.mkdir(parents=True, exist_ok=True)

    scenario = default_scenario(seed=42)
    backend = create_backend("gazebo")
    handle = backend.reset(scenario, seed=42)
    frame = None
    observation = None
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        observation = backend.observe(handle)
        frame = observation.sensor_frame
        if frame is not None and frame.valid and frame.rgb is not None:
            break
        time.sleep(0.5)
    if observation is None or frame is None or frame.rgb is None:
        print("probe: no frame arrived")
        backend.close_all()
        return 1

    depth, _rgb_raw, width, height = backend._state(handle)["depth"].latest()
    np.save(out / "depth.npy", depth if depth is not None else np.zeros((1, 1)))
    try:
        from PIL import Image

        Image.fromarray(frame.rgb).resize((width * 4, height * 4), Image.NEAREST).save(
            out / "rgb_x4.png"
        )
    except Exception as exc:
        print(f"probe: png save failed: {exc}")

    # -- expected pixel of the true cube, under both row conventions ---------
    x, y, z, _r, pitch, yaw = (float(v) for v in _camera_pose(scenario).split())
    cp, sp = np.cos(pitch), np.sin(pitch)
    cyy, syy = np.cos(yaw), np.sin(yaw)
    eye = np.array([x, y, z])
    fwd = np.array([cyy * cp, syy * cp, -sp])
    right = np.array([-syy, cyy, 0.0])
    up = np.array([cyy * sp, syy * sp, cp])
    _w, _h, fx, fy, cx, cy = camera_intrinsics_from_scenario(scenario)
    cube = np.array(object_pose_from_scenario(scenario))
    local = cube - eye
    d = float(local @ fwd)
    col = cx + (-(local @ right) / d) * fx
    row_as_stored = cy + ((local @ up) / d) * fy  # the formula's own convention
    row_mirrored = cy - ((local @ up) / d) * fy  # standard pinhole convention
    report: dict[str, object] = {
        "cube_truth": [round(float(v), 4) for v in cube],
        "cube_pixel_stored_convention": [round(col, 1), round(row_as_stored, 1)],
        "cube_pixel_mirrored_convention": [round(col, 1), round(row_mirrored, 1)],
        "cube_depth_m": round(d, 4),
        "camera_pose": _camera_pose(scenario),
    }
    if depth is not None and depth.shape == (height, width):
        for label, row in (("stored", row_as_stored), ("mirrored", row_mirrored)):
            r0, c0 = int(round(row)), int(round(col))
            if 0 <= r0 < height and 0 <= c0 < width:
                report[f"depth_at_{label}"] = [
                    round(float(depth[r0, c0]), 4),
                    round(float(depth[max(0, r0 - 1) : r0 + 2, max(0, c0 - 1) : c0 + 2].min()), 4),
                    round(float(depth[max(0, r0 - 1) : r0 + 2, max(0, c0 - 1) : c0 + 2].max()), 4),
                ]

    # -- where perception actually looked -----------------------------------
    beliefs = backend._detect_objects(handle)
    report["belief"] = [
        {
            "id": o.id,
            "confidence": round(float(o.confidence), 3),
            "position": [round(float(v), 4) for v in o.pose.position.array],
        }
        for o in beliefs
    ]

    # -- detection diagnostics: what did the residual actually select? -------
    if depth is not None and depth.shape == (height, width):
        eye2, fwd2, right2, up2 = backend._camera_geometry(scenario)
        finite = np.isfinite(depth)
        rr, cc = np.nonzero(finite)
        zz = depth[rr, cc]
        points = (
            eye2[None, :]
            + zz[:, None] * fwd2[None, :]
            - ((cc + 0.5 - cx) / fx * zz)[:, None] * right2[None, :]
            - ((rr + 0.5 - cy) / fy * zz)[:, None] * up2[None, :]
        )
        design = np.column_stack([points[:, 0], points[:, 1], np.ones(points.shape[0])])
        solution, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
        residual = points[:, 2] - design @ solution
        above = residual > 0.015
        diag: dict[str, object] = {
            "plane_fit_z_at_xy": [round(float(solution[0]), 5), round(float(solution[1]), 5), round(float(solution[2]), 5)],
            "finite_points": int(above.size),
            "above_threshold": int(above.sum()),
        }
        if above.sum():
            sel_rows, sel_cols = rr[above], cc[above]
            sel_pts = points[above]
            diag["selected_pixel_bbox"] = [
                int(sel_cols.min()), int(sel_cols.max()), int(sel_rows.min()), int(sel_rows.max())
            ]
            diag["selected_world_centroid"] = [round(float(v), 4) for v in sel_pts.mean(axis=0)]
            # The largest voxel cluster, as the detector sees it.
            cluster = GazeboBackend._largest_cluster(sel_pts)
            diag["cluster_centroid"] = [round(float(v), 4) for v in cluster.mean(axis=0)] if len(cluster) else None
            diag["cluster_size"] = int(len(cluster))
        # Residual at the cube's own pixel, to see whether the real cube even
        # clears the threshold.
        cube_row, cube_col = int(round(row_mirrored)), int(round(col))
        near = (np.abs(rr - cube_row) <= 3) & (np.abs(cc - cube_col) <= 3)
        if near.any():
            diag["cube_neighbourhood"] = {
                "pixels": int(near.sum()),
                "residual_max": round(float(residual[near].max()), 4),
                "residual_min": round(float(residual[near].min()), 4),
                "depth_mean": round(float(zz[near].mean()), 4),
                "world_centroid": [round(float(v), 4) for v in points[near].mean(axis=0)],
            }
        # Where is the cube's true world position among the residuals?
        dist = np.linalg.norm(points - cube[None, :], axis=1)
        diag["nearest_point_to_cube"] = {
            "distance_m": round(float(dist.min()), 4),
            "residual": round(float(residual[int(dist.argmin())]), 4),
            "pixel": [int(cc[dist.argmin()]), int(rr[dist.argmin()])],
        }
        report["detection_diag"] = diag

    truth = backend.ground_truth(handle)
    report["ground_truth"] = [
        {"id": o.id, "position": [round(float(v), 4) for v in o.pose.position.array]}
        for o in truth.objects
    ]
    report["joints"] = [round(float(v), 4) for v in observation.robot.joint_positions]
    report["ee"] = [round(float(v), 4) for v in observation.robot.end_effector_pose.position.array]
    if depth is not None and depth.shape == (height, width):
        finite = np.isfinite(depth)
        report["depth_stats"] = {
            "finite_px": int(finite.sum()),
            "min_m": round(float(depth[finite].min()), 4) if finite.any() else None,
            "max_m": round(float(depth[finite].max()), 4) if finite.any() else None,
        }

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    backend.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
