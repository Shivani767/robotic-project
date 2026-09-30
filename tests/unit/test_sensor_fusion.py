"""Tests for the sensor-fusion filter.

The property that matters is not that the filter reduces error -- any average
does that -- but that the uncertainty it *reports* is honest. A filter that is
confidently wrong is worse than one that reports no covariance, because a
downstream confidence gate will act on it. Both bugs found while writing this
were overconfidence bugs, so calibration is what the tests check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_ci.models.geometry import Covariance, Pose, Vec3
from robot_ci.models.state import ObjectState, Source
from robot_ci.sensor_fusion import FusionConfig, ObjectPoseFilter

TRUTH = np.array([0.30, 0.10])
#: Per-axis measurement noise of the synthetic sensor, in metres.
SENSOR_SIGMA = 0.006
DT = 0.05


def _measurement(x: float, y: float, t: float, sigma: float = SENSOR_SIGMA) -> ObjectState:
    """One noisy object estimate."""
    return ObjectState(
        id="cube",
        pose=Pose(position=Vec3.of(x, y, 0.008)),
        covariance=Covariance.isotropic(sigma, 6),
        confidence=0.8,
        source=Source.DEPTH,
        timestamp_s=t,
        visible=True,
    )


def _fuse(
    n: int, trial: int, *, sigma: float = SENSOR_SIGMA
) -> tuple[np.ndarray, float]:
    """Fuse ``n`` samples and return the estimate and its reported sigma."""
    return _fuse_samples(
        [TRUTH + np.random.default_rng(trial + step).normal(0.0, sigma, 2) for step in range(n)],
        sigma,
    )


def _fuse_samples(samples: list[np.ndarray], sigma: float = SENSOR_SIGMA) -> tuple[np.ndarray, float]:
    """Fuse a given list of samples, so a test can compare against the same data."""
    fuse = ObjectPoseFilter(config=FusionConfig(process_acceleration=0.02))
    for step, noisy in enumerate(samples):
        fuse.update(_measurement(noisy[0], noisy[1], step * DT, sigma), dt=DT)
    n = len(samples)
    estimate = fuse.estimate("cube", now_s=n * DT)
    assert estimate is not None
    return np.asarray(estimate.pose.position.array), math.sqrt(estimate.covariance.matrix[0, 0])


@pytest.mark.parametrize("n", [1, 4, 12, 24])
def test_reported_uncertainty_tracks_actual_error(n: int) -> None:
    """Calibration, not accuracy, is the property under test.

    The reported sigma must be the same order as the error it actually makes.
    A ratio near 1 is calibrated; a ratio near 0.01 is a filter that tells a
    downstream gate the estimate is 300x better than it is.
    """
    errors, sigmas = [], []
    for trial in range(40):
        position, reported = _fuse(n, trial)
        errors.append(float(np.linalg.norm(position[:2] - TRUTH)))
        sigmas.append(reported)
    error, sigma = float(np.mean(errors)), float(np.mean(sigmas))
    ratio = sigma / error
    assert 0.5 < ratio < 2.0, (
        f"n={n}: reported sigma {sigma * 1000:.2f} mm against actual error "
        f"{error * 1000:.2f} mm (ratio {ratio:.2f}); a filter outside this band "
        "is either overconfident or refusing to learn"
    )


def test_more_samples_reduce_error_and_uncertainty_together() -> None:
    """Both the estimate and its reported sigma must improve with evidence.

    A filter whose error falls while its reported sigma stays flat, or rises,
    is not describing its own behaviour.
    """
    previous_error, previous_sigma = math.inf, math.inf
    for n in (1, 4, 12, 24):
        errors, sigmas = [], []
        for trial in range(30):
            position, reported = _fuse(n, trial)
            errors.append(float(np.linalg.norm(position[:2] - TRUTH)))
            sigmas.append(reported)
        error, sigma = float(np.mean(errors)), float(np.mean(sigmas))
        assert error < previous_error, f"error rose at n={n}"
        assert sigma < previous_sigma, f"reported sigma rose at n={n}"
        previous_error, previous_sigma = error, sigma


def test_the_filter_beats_a_single_sample_and_tracks_a_batch_mean() -> None:
    """Fusing must be worth far more than one measurement, and close to a mean.

    The lower bound is the one that matters. The covariance used to be
    initialised to a fixed 1e-3, which was 28x the measurement variance here and
    gave a first-step Kalman gain near 0.96, so the filter tracked the most
    recent sample rather than averaging: 3.2 mm mean error where a plain average
    of the same samples gives 2.3 mm. Worse than doing nothing clever, while
    looking like a sophisticated estimator.

    The upper bound is looser than one might expect. A recursive filter weights
    samples exponentially while a batch mean weights them equally, so at n=12 the
    filter has a slightly smaller *effective* sample size and lands around 11%
    behind the mean. That is the price of streaming against a static target and
    is the correct trade; the test bounds it rather than demanding parity.
    """
    fused_errors, averaged_errors, single_errors = [], [], []
    for trial in range(40):
        rng = np.random.default_rng(trial)
        samples = [TRUTH + rng.normal(0.0, SENSOR_SIGMA, 2) for _ in range(12)]
        # The *same* samples on both sides, or the comparison is between two
        # different noise draws rather than between two estimators.
        position, _ = _fuse_samples(samples)
        fused_errors.append(float(np.linalg.norm(position[:2] - TRUTH)))
        averaged_errors.append(float(np.linalg.norm(np.asarray(samples).mean(axis=0) - TRUTH)))
        single_errors.append(float(np.linalg.norm(samples[0] - TRUTH)))
    single = float(np.mean(single_errors))
    assert np.mean(fused_errors) < single * 0.5, (
        f"fusing 12 samples gave {np.mean(fused_errors) * 1000:.2f} mm against "
        f"{single * 1000:.2f} mm for one; the filter is not accumulating evidence"
    )
    assert np.mean(fused_errors) <= np.mean(averaged_errors) * 1.25, (
        f"fused {np.mean(fused_errors) * 1000:.2f} mm against a plain average of "
        f"{np.mean(averaged_errors) * 1000:.2f} mm on the same samples"
    )


def test_a_louder_sensor_is_trusted_less() -> None:
    """A measurement that declares more noise must move the estimate less."""
    quiet = ObjectPoseFilter(config=FusionConfig())
    loud = ObjectPoseFilter(config=FusionConfig())
    rng = np.random.default_rng(7)
    for step in range(8):
        noisy = TRUTH + rng.normal(0.0, 0.01, 2)
        quiet.update(_measurement(noisy[0], noisy[1], step * DT, sigma=0.002), dt=DT)
        loud.update(_measurement(noisy[0], noisy[1], step * DT, sigma=0.05), dt=DT)
    quiet_sigma = math.sqrt(quiet.estimate("cube", now_s=1.0).covariance.matrix[0, 0])
    loud_sigma = math.sqrt(loud.estimate("cube", now_s=1.0).covariance.matrix[0, 0])
    assert loud_sigma > quiet_sigma, "a noisier sensor must leave more uncertainty"


def test_provenance_and_age_are_reported() -> None:
    """A fused belief must say what went into it and how old it is."""
    fuse = ObjectPoseFilter(config=FusionConfig())
    for step in range(4):
        fuse.update(_measurement(TRUTH[0], TRUTH[1], step * DT), dt=DT)
    estimate = fuse.estimate("cube", now_s=1.0)
    assert estimate.provenance[0] is Source.DEPTH
    assert estimate.updates == 4
    assert estimate.source is Source.FUSION
    assert estimate.measurements_s == pytest.approx(1.0 - 3 * DT, abs=1e-9)


def test_no_estimate_before_any_measurement() -> None:
    """An empty filter reports nothing rather than the origin."""
    assert ObjectPoseFilter().estimate("cube", now_s=0.0) is None


def test_a_negative_timestep_leaves_the_covariance_alone() -> None:
    """Out-of-order timestamps must not propagate a step backwards.

    The constant-velocity propagation multiplies the covariance by terms in
    ``dt**2``; a negative ``dt`` would make the process-noise block *negative*,
    producing a covariance that is no longer positive semi-definite. The filter
    reports a stale estimate in that case, which is safe; reporting an impossible
    one is not.
    """
    fuse = ObjectPoseFilter(config=FusionConfig())
    for step in range(3):
        fuse.update(_measurement(TRUTH[0], TRUTH[1], step * DT), dt=DT)
    before = fuse._P.copy()
    fuse.predict(-1.0)
    assert np.allclose(fuse._P, before), "a negative dt must not touch the covariance"
    assert np.all(np.linalg.eigvalsh(fuse._P) > -1e-12), "covariance must stay positive semi-definite"
