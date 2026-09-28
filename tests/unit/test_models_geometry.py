"""Geometry primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3


def test_vec3_arithmetic() -> None:
    a, b = Vec3.of(1, 2, 3), Vec3.of(4, 5, 6)
    assert (a + b).array.tolist() == [5, 7, 9]
    assert (b - a).array.tolist() == [3, 3, 3]
    assert (a * 2).array.tolist() == [2, 4, 6]
    assert (2 * a).array.tolist() == [2, 4, 6]


def test_vec3_norm_and_distance() -> None:
    assert Vec3.of(3, 4, 0).norm == pytest.approx(5.0)
    assert Vec3.of(0, 0, 0).distance_to(Vec3.of(1, 2, 2)) == pytest.approx(3.0)


def test_vec3_from_iterable_accepts_arrays() -> None:
    assert Vec3.from_iterable(np.array([1.0, 2.0, 3.0])).array.tolist() == [1.0, 2.0, 3.0]


def test_vec3_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="3-element"):
        Vec3.from_iterable([1.0, 2.0])


def test_quaternion_is_normalized() -> None:
    q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
    assert q.w == pytest.approx(1.0)
    assert float(np.linalg.norm(q.array)) == pytest.approx(1.0)


def test_quaternion_rejects_zero_norm() -> None:
    with pytest.raises(ValidationError, match="too small"):
        Quaternion(w=0.0, x=0.0, y=0.0, z=0.0)


def test_quaternion_rotation_about_z() -> None:
    q = Quaternion.from_axis_angle(Vec3.of(0, 0, 1), math.pi / 2)
    rotated = q.rotate(Vec3.of(1, 0, 0))
    assert rotated.x == pytest.approx(0.0, abs=1e-9)
    assert rotated.y == pytest.approx(1.0, abs=1e-9)


def test_quaternion_angle_roundtrip() -> None:
    q = Quaternion.from_axis_angle(Vec3.of(0, 1, 0), 1.0)
    assert q.angle == pytest.approx(1.0, abs=1e-9)


def test_quaternion_sign_ambiguity() -> None:
    a = Quaternion.from_axis_angle(Vec3.of(0, 0, 1), 0.7)
    b = Quaternion.from_axis_angle(Vec3.of(0, 0, 1), 0.7)
    assert a.angular_distance_to(b) == pytest.approx(0.0, abs=1e-9)


def test_pose_combined_distance() -> None:
    a = Pose.from_position(Vec3.of(0, 0, 0))
    b = Pose.from_position(Vec3.of(0.3, 0.4, 0))
    pos, rot = a.combined_distance_to(b)
    assert pos == pytest.approx(0.5)
    assert rot == pytest.approx(0.0, abs=1e-9)


def test_covariance_shapes() -> None:
    assert Covariance.zeros(3).matrix.shape == (3, 3)
    assert Covariance.zeros(6).matrix.shape == (6, 6)


def test_covariance_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError, match="requires"):
        Covariance(values=(1.0, 2.0), dimension=3)


def test_covariance_isotropic_sigmas() -> None:
    cov = Covariance.isotropic(0.02, dimension=6)
    assert np.allclose(cov.sigmas, 0.02)
    assert cov.is_symmetric()
    assert cov.is_invertible


def test_covariance_from_diagonal() -> None:
    cov = Covariance.from_diagonal((0.1, 0.2, 0.3))
    assert np.allclose(cov.sigmas, [0.1, 0.2, 0.3])
    assert cov.trace == pytest.approx(0.01 + 0.04 + 0.09)


def test_covariance_sum_is_elementwise() -> None:
    a = Covariance.isotropic(0.1)
    b = Covariance.isotropic(0.2)
    assert np.allclose(a.plus(b).sigmas, np.full(3, np.hypot(0.1, 0.2)))


def test_covariance_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="cannot add"):
        Covariance.zeros(3).plus(Covariance.zeros(6))


def test_covariance_singular_is_flagged() -> None:
    cov = Covariance(values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dimension=3)
    assert not cov.is_invertible
