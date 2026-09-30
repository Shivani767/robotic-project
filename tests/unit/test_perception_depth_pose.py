"""Tests for depth-based object pose estimation.

Calibration is the property under test, not detection rate. An estimator that
finds the cube and reports 17 mm of uncertainty while actually being accurate
to 0.1 mm is worse than useless: a confidence gate upstream will trust it, and
the grasp budget downstream will be spent defending against an error that is not
there.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ci.perception import (
    DepthPoseConfig,
    estimate_object_pose,
    to_object_state,
)

HEIGHT, WIDTH = 120, 160
FX = FY = 140.0
CX, CY = 80.0, 60.0
#: Object position in metres, chosen to project inside the frame.
TRUTH = np.array([0.15, 0.05, 0.50])
INTRINSICS = (FX, FY, CX, CY)


def _frame(half: int, *, noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """A depth frame containing one flat square at ``TRUTH``."""
    frame = np.full((HEIGHT, WIDTH), np.nan)
    u = int(round(CX + FX * TRUTH[0] / TRUTH[2]))
    v = int(round(CY + FY * TRUTH[1] / TRUTH[2]))
    patch = np.full((2 * half + 1, 2 * half + 1), TRUTH[2])
    if noise and rng is not None:
        patch = patch + rng.normal(0.0, noise, patch.shape)
    frame[v - half : v + half + 1, u - half : u + half + 1] = patch
    return frame


def test_a_square_object_localises_exactly_on_a_clean_frame() -> None:
    """A symmetric blob at a known depth has a known centre.

    This is the test that caught the flood fill expanding horizontally only,
    which returned a single 13-pixel row of a 169-pixel object. The error grew
    with the object's apparent size -- 21 mm, 36 mm, 71 mm -- on a scene with no
    error in it, because the first row scanned was the top one.
    """
    result = estimate_object_pose(_frame(6), INTRINSICS)
    assert result.found
    assert result.pixel_count == 13 * 13, "the whole blob must be one component, not one row"
    assert np.allclose(result.pose.position.array, TRUTH, atol=1e-6)


@pytest.mark.parametrize("noise", [0.001, 0.002, 0.004, 0.008])
def test_reported_uncertainty_tracks_actual_error(noise: float) -> None:
    """The reported sigma must be the same order as the error actually made.

    The config is told the sensor's own per-pixel noise, so the reported figure
    is a propagation of something real rather than a constant somebody typed.
    """
    config = DepthPoseConfig(pixel_depth_sigma_m=noise)
    errors, sigmas = [], []
    for seed in range(16):
        result = estimate_object_pose(
            _frame(8, noise=noise, rng=np.random.default_rng(seed)), INTRINSICS, config
        )
        assert result.found
        errors.append(float(np.linalg.norm(result.pose.position.array - TRUTH)))
        sigmas.append(result.sigma_m)
    error, sigma = float(np.mean(errors)), float(np.mean(sigmas))
    ratio = sigma / max(error, 1e-12)
    assert 0.4 < ratio < 2.0, (
        f"sensor noise {noise * 1000:.0f} mm: reported {sigma * 1000:.3f} mm against "
        f"actual {error * 1000:.3f} mm (ratio {ratio:.2f})"
    )


def test_a_bigger_blob_is_localised_better() -> None:
    """More pixels must mean more certainty, following a square-root law."""
    sigmas = []
    for half in (2, 6, 10):
        result = estimate_object_pose(_frame(half), INTRINSICS, DepthPoseConfig(min_pixels=1))
        assert result.found
        sigmas.append(result.sigma_m)
    assert sigmas[0] > sigmas[1] > sigmas[2], f"sigmas did not fall with blob size: {sigmas}"


def test_a_failed_detection_reports_no_pose() -> None:
    """Absence must be absence, not a pose at the origin.

    Returning the origin with a large sigma would let a caller that ignores the
    sigma act on a fabricated object in the middle of the table.
    """
    for frame, intrinsics, expected in (
        (None, INTRINSICS, "dropped"),
        (np.full((HEIGHT, WIDTH), np.nan), INTRINSICS, "no depth"),
        (_frame(6), (0.0, 0.0, CX, CY), "focal"),
        (np.zeros((4, 4)), INTRINSICS, "no depth"),
    ):
        result = estimate_object_pose(frame, intrinsics)
        assert result.pose is None
        assert result.sigma_m is None
        assert expected in result.reason, f"unhelpful reason {result.reason!r} for {expected}"
        assert to_object_state(result, object_id="cube", timestamp_s=0.0) is None


def test_a_too_small_blob_is_rejected_rather_than_guessed() -> None:
    """One or two pixels are a depth artefact far more often than an object."""
    result = estimate_object_pose(_frame(1), INTRINSICS, DepthPoseConfig(min_pixels=12))
    assert not result.found
    assert "9 px" in result.reason and "12" in result.reason, (
        f"the reason must name the size and the bound, got {result.reason!r}"
    )
    assert result.pixel_count == 9, "the rejected size is still worth reporting"


def test_a_frame_filling_blob_is_rejected_as_background() -> None:
    """A frame-sized blob is the table, not an object, and nothing else is there."""
    frame = np.full((HEIGHT, WIDTH), 0.5)
    result = estimate_object_pose(frame, INTRINSICS, DepthPoseConfig(max_pixels=1000))
    assert not result.found
    assert "outside" in result.reason
    assert result.pixel_count == HEIGHT * WIDTH, "the rejected size is still worth reporting"


def test_depth_jump_separates_an_object_from_the_table_it_stands_on() -> None:
    """The continuity threshold is what splits a cube from its support."""
    frame = np.full((HEIGHT, WIDTH), 0.55)
    u = int(round(CX + FX * TRUTH[0] / 0.50))
    v = int(round(CY + FY * TRUTH[1] / 0.50))
    frame[v - 4 : v + 5, u - 4 : u + 5] = 0.50
    result = estimate_object_pose(frame, INTRINSICS, DepthPoseConfig(depth_jump_m=0.01))
    assert result.found
    assert result.pixel_count == 81, "the table must not have been absorbed into the object"
    assert result.pose.position.z == pytest.approx(0.50, abs=1e-6)
