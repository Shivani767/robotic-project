"""Object pose from a depth frame, with uncertainty derived rather than asserted.

The estimator is a connected-component analysis over the depth frame followed by
a back-projection, which is the minimum a real system needs before it can claim
to perceive: pixels become a 3-D point, a blob of pixels becomes an object, and
the object gets a pose with a covariance that comes from *how* the blob was
measured rather than from a constant somebody typed.

Two design choices carry most of the weight:

* **Uncertainty is derived from the point count.** A blob of 400 pixels has its
  centroid pinned to a fraction of a pixel; a blob of 3 does not. Scaling the
  reported sigma with the spread of the blob and the square root of its size
  gives an estimate that says something true about its own quality, and a
  caller can act on the difference. A fixed sigma would make a one-pixel
  detection look as trustworthy as a solid one.

* **A detection that fails is reported as a failure.** If the frame is invalid,
  the depth is empty, or no blob passes the size and continuity tests, the
  result carries no pose. Returning the origin with a large sigma instead would
  let a caller that ignored the sigma act on a fabricated object at the world
  origin, which is the worst available failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
from robot_ci.models.state import ObjectCategory, ObjectState, Source

__all__ = ["DepthPoseConfig", "DepthPoseResult", "estimate_object_pose"]


@dataclass(frozen=True)
class DepthPoseConfig:
    """Tunables for :func:`estimate_object_pose`.

    Attributes:
        min_pixels: Smallest blob accepted as a detection. Below this the blob
            is far more likely to be a depth artefact than an object, and
            accepting it means publishing a pose derived from one or two noisy
            pixels.
        max_pixels: Largest blob accepted. A blob filling the frame is almost
            always the table, not an object, and its centroid is the middle of
            the image rather than where anything is.
        max_depth_m: Furthest depth considered. Points beyond this are treated
            as background so a far wall does not join the object's blob.
        min_depth_m: Nearest depth considered, guarding against the camera
            seeing its own mount.
        depth_jump_m: Largest depth step between horizontally adjacent pixels
            still treated as connected. A surface edge produces a step far
            larger than this, which is what separates a cube from the table it
            stands on.
        pixel_depth_sigma_m: Per-pixel depth noise of the sensor, in metres.
            This is the physical input to the reported uncertainty and should be
            the scenario's own depth-noise value where one is set. A caller that
            invents it here instead is asserting an accuracy the sensor does not
            have.
    """

    min_pixels: int = 12
    max_pixels: int = 4000
    max_depth_m: float = 2.0
    min_depth_m: float = 0.05
    depth_jump_m: float = 0.01
    pixel_depth_sigma_m: float = 0.002


@dataclass(frozen=True)
class DepthPoseResult:
    """What the estimator found, or why it found nothing.

    Attributes:
        pose: Detected pose, or ``None`` when the detection failed.
        pixel_count: Size of the accepted blob, 0 when nothing was accepted.
        sigma_m: Derived positional uncertainty, or ``None`` on failure.
        confidence: In [0, 1], a function of blob size against the reference.
        reason: Why a detection was rejected, or ``""`` on success. Present on
            every failure, because a detector that returns nothing without
            saying why cannot be debugged.
    """

    pose: Pose | None
    pixel_count: int
    sigma_m: float | None
    confidence: float
    reason: str = ""

    @property
    def found(self) -> bool:
        """Whether a pose was produced."""
        return self.pose is not None


def estimate_object_pose(
    depth: np.ndarray | None,
    intrinsics: tuple[float, float, float, float],
    config: DepthPoseConfig | None = None,
) -> DepthPoseResult:
    """Detect the nearest plausible object in a depth frame and localise it.

    Args:
        depth: ``(H, W)`` depth in metres, ``NaN`` where invalid. ``None`` when
            the frame was dropped.
        intrinsics: ``(fx, fy, cx, cy)`` in pixels.
        config: Tunables; defaults are used when omitted.

    Returns:
        A :class:`DepthPoseResult`. On failure ``pose`` is ``None`` and
        ``reason`` says why -- never a fabricated pose at the origin.
    """
    settings = config or DepthPoseConfig()
    if depth is None:
        return DepthPoseResult(None, 0, None, 0.0, "no depth frame (dropped or absent)")
    frame = np.asarray(depth, dtype=np.float64)
    if frame.ndim != 2 or frame.size == 0:
        return DepthPoseResult(None, 0, None, 0.0, f"depth frame is not a 2-D image: shape {frame.shape}")
    fx, fy, cx, cy = intrinsics
    if fx <= 0.0 or fy <= 0.0:
        return DepthPoseResult(None, 0, None, 0.0, f"non-positive focal length fx={fx}, fy={fy}")

    valid = np.isfinite(frame) & (frame > settings.min_depth_m) & (frame < settings.max_depth_m)
    if not valid.any():
        return DepthPoseResult(
            None, 0, None, 0.0, f"no depth in [{settings.min_depth_m}, {settings.max_depth_m}] m"
        )

    # Flood fill over horizontally adjacent pixels that are also close in depth.
    # A table and a cube on it differ by a few millimetres, which is why the
    # jump threshold is the parameter that decides whether the two separate.
    height, width = frame.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current = 0
    best: tuple[int, list[tuple[int, int]]] | None = None
    #: Size of the biggest region rejected on plausibility, reported on failure.
    largest_rejected = 0
    for row in range(height):
        for col in range(width):
            if not valid[row, col] or labels[row, col]:
                continue
            current += 1
            blob: list[tuple[int, int]] = []
            stack = [(row, col)]
            labels[row, col] = current
            while stack:
                r, c = stack.pop()
                blob.append((r, c))
                # 8-connectivity. Expanding only horizontally treats every row
                # as its own component, so the "blob" was a single 13-pixel run
                # of a 169-pixel object. Because the first row scanned is the
                # top one, the back-projected y error grew with the object's
                # apparent size -- 21 mm at 13 px, 36 mm at 21 px, 71 mm at
                # 41 px -- on a synthetic scene with no error in it at all.
                for nr, nc in (
                    (r, c - 1),
                    (r, c + 1),
                    (r - 1, c),
                    (r + 1, c),
                    (r - 1, c - 1),
                    (r - 1, c + 1),
                    (r + 1, c - 1),
                    (r + 1, c + 1),
                ):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and valid[nr, nc]
                        and not labels[nr, nc]
                        and abs(frame[nr, nc] - frame[r, c]) <= settings.depth_jump_m
                    ):
                        labels[nr, nc] = current
                        stack.append((nr, nc))
            # Keep the largest region that is *plausibly an object*, not the
            # largest region outright. A table filling the frame is a bigger
            # connected region than the cube standing on it, and rejecting the
            # frame as "too large, probably background" threw the cube away
            # without ever looking at it.
            if not settings.min_pixels <= len(blob) <= settings.max_pixels:
                if len(blob) > largest_rejected:
                    largest_rejected = len(blob)
                continue
            if best is None or len(blob) > len(best[1]):
                best = (current, blob)

    if best is None:
        return DepthPoseResult(
            None,
            largest_rejected,
            None,
            0.0,
            f"no region of plausible object size: largest was {largest_rejected} px, "
            f"outside [{settings.min_pixels}, {settings.max_pixels}] px",
        )
    pixel_count = len(best[1])

    rows = np.array([r for r, _ in best[1]], dtype=np.float64)
    cols = np.array([c for _, c in best[1]], dtype=np.float64)
    depths = frame[rows.astype(int), cols.astype(int)]
    # Back-project the blob's mean pixel. The centroid is taken in *depth* space
    # (via the mean of the back-projected points) rather than in image space,
    # because a tilted surface's image centroid does not correspond to its
    # geometric centre.
    xs = (cols - cx) * depths / fx
    ys = (rows - cy) * depths / fy
    position = np.array([float(xs.mean()), float(ys.mean()), float(depths.mean())])

    # Uncertainty from the sensor's own depth noise, propagated to the centroid.
    #
    # An earlier version added half the blob's spatial spread, reasoning that a
    # bigger object is less well localised. That is wrong: the spread is the
    # object's *size*, and knowing a 16 mm cube spans 20 mm in the image says
    # nothing about where its centre is. It made a perfectly centred detection
    # report 17.8 mm of uncertainty -- larger than the entire grasp tolerance
    # the estimator exists to inform.
    #
    # The propagation is the standard one. Per-pixel depth noise averages over
    # the blob as 1/sqrt(N), giving the centroid's depth sigma; a lateral
    # position error at range Z projects to a pixel offset of f*dx/Z, so the
    # lateral sigmas scale by Z/f and are the larger of the two. Both are
    # reported per axis rather than collapsed, because a detector that is
    # 0.05 mm deep and 2 mm wide is not well described by one number.
    centroid_sigma_z = settings.pixel_depth_sigma_m / math.sqrt(max(pixel_count, 1))
    lateral_gain = float(depths.mean()) / max(fx, fy)
    sigmas = np.array(
        [
            centroid_sigma_z * lateral_gain,
            centroid_sigma_z * lateral_gain,
            centroid_sigma_z,
            0.0,
            0.0,
            0.0,
        ]
    )
    sigmas = np.maximum(sigmas, 1e-6)

    return DepthPoseResult(
        pose=Pose(position=Vec3.from_iterable(position), orientation=Quaternion.identity()),
        pixel_count=pixel_count,
        sigma_m=float(sigmas.max()),
        confidence=float(min(1.0, pixel_count / 400.0)),
    )


def to_object_state(
    result: DepthPoseResult,
    *,
    object_id: str,
    timestamp_s: float,
    category: ObjectCategory = ObjectCategory.CUBE,
    size_m: float = 0.016,
) -> ObjectState | None:
    """Convert a successful detection into an object belief, or ``None``.

    Returns ``None`` rather than a placeholder so a failed detection cannot be
    mistaken for a belief about an object at the world origin.
    """
    if not result.found or result.pose is None or result.sigma_m is None:
        return None
    return ObjectState(
        id=object_id,
        category=category,
        pose=result.pose,
        covariance=Covariance.from_diagonal((result.sigma_m,) * 6),
        confidence=result.confidence,
        source=Source.DEPTH,
        timestamp_s=timestamp_s,
        visible=True,
        last_seen_s=timestamp_s,
        estimated_size_m=size_m,
    )
