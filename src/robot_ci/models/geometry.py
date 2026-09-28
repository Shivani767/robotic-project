"""Geometry primitives.

These are deliberately tiny and dependency-free. Every geometric quantity that
crosses a module or process boundary (simulator, policy, evaluation, report)
is expressed with these types, so units and conventions are stated once:

* **lengths** are metres,
* **angles** are radians,
* **quaternions** are ``(w, x, y, z)`` with unit norm,
* **timestamps** are float seconds on a monotonic episode clock unless a model
  documents otherwise.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Self

import numpy as np
from pydantic import Field, field_validator, model_validator

from robot_ci.models.base import VersionedModel

__all__ = [
    "Covariance",
    "Degrees",
    "Pose",
    "Quaternion",
    "Radians",
    "Vec3",
    "zero_covariance",
]

Radians = Annotated[float, Field(description="Angle in radians.")]
Degrees = Annotated[float, Field(description="Angle in degrees.")]


def _as_float_triple(value: Any) -> tuple[float, float, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"expected a 3-element sequence, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


class Vec3(VersionedModel):
    """A 3-vector in metres (position) or metres/second (velocity)."""

    x: float = Field(description="X component in metres.")
    y: float = Field(description="Y component in metres.")
    z: float = Field(description="Z component in metres.")

    model_config = {**VersionedModel.model_config, "frozen": True}

    @classmethod
    def of(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Self:
        """Construct from scalar components."""
        return cls(x=float(x), y=float(y), z=float(z))

    @classmethod
    def from_iterable(cls, values: Any) -> Self:
        """Construct from any 3-element sequence or NumPy array."""
        x, y, z = _as_float_triple(values)
        return cls(x=x, y=y, z=z)

    @property
    def array(self) -> np.ndarray:
        """Return the vector as a ``float64`` NumPy array of shape ``(3,)``."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def norm(self) -> float:
        """Euclidean magnitude in metres."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def distance_to(self, other: Vec3) -> float:
        """Euclidean distance in metres to ``other``."""
        return float(np.linalg.norm(self.array - other.array))

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3.of(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3.of(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3.of(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__


class Quaternion(VersionedModel):
    """A rotation as ``(w, x, y, z)`` with non-zero norm.

    The norm is normalized on construction. Identity is ``(1, 0, 0, 0)``.
    """

    w: float = Field(default=1.0, description="Real part.")
    x: float = Field(default=0.0, description="i component.")
    y: float = Field(default=0.0, description="j component.")
    z: float = Field(default=0.0, description="k component.")

    model_config = {**VersionedModel.model_config, "frozen": True}

    @field_validator("w", "x", "y", "z")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("quaternion components must be finite")
        return value

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        """Normalize the quaternion before the fields are constructed.

        A ``mode="after"`` validator that returns a *new* model is not supported
        by Pydantic when validating through ``__init__``, so normalization has to
        happen on the incoming mapping instead.
        """
        if not isinstance(data, dict):
            return data
        try:
            w, x, y, z = (float(data.get(k, 0.0)) for k in ("w", "x", "y", "z"))
        except (TypeError, ValueError):
            return data
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-9:
            raise ValueError("quaternion norm is too small to normalize")
        if abs(norm - 1.0) > 1e-12:
            return {**data, "w": w / norm, "x": x / norm, "y": y / norm, "z": z / norm}
        return data

    @classmethod
    def identity(cls) -> Quaternion:
        """Return the identity rotation."""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle: float) -> Quaternion:
        """Build a rotation of ``angle`` radians about ``axis`` (need not be unit)."""
        unit = axis.array
        magnitude = float(np.linalg.norm(unit))
        if magnitude < 1e-9:
            return cls.identity()
        unit = unit / magnitude
        half = 0.5 * float(angle)
        sin_half = math.sin(half)
        return cls(
            w=math.cos(half),
            x=float(unit[0]) * sin_half,
            y=float(unit[1]) * sin_half,
            z=float(unit[2]) * sin_half,
        )

    @property
    def array(self) -> np.ndarray:
        """Return the quaternion as a ``float64`` array of shape ``(4,)``."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    @property
    def angle(self) -> float:
        """Rotation magnitude in radians, in ``[0, pi]``."""
        return float(2.0 * math.acos(max(-1.0, min(1.0, abs(self.w)))))

    def angular_distance_to(self, other: Quaternion) -> float:
        """Geodesic distance in radians to ``other`` (sign-ambiguity aware)."""
        dot = float(abs(np.dot(self.array, other.array)))
        return float(2.0 * math.acos(max(-1.0, min(1.0, dot))))

    def rotate(self, vector: Vec3) -> Vec3:
        """Rotate a 3-vector by this quaternion (Hamilton convention)."""
        v = vector.array
        q = self.array
        u, s = q[1:], q[0]
        rotated = 2.0 * float(np.dot(u, v)) * u + (s**2 - float(np.dot(u, u))) * v
        rotated = rotated + 2.0 * s * np.cross(u, v)
        return Vec3.from_iterable(rotated)




class Covariance(VersionedModel):
    """A symmetric 3x3 or 6x6 covariance matrix in row-major order.

    Attributes:
        values: Flattened, row-major entries.
        dimension: ``3`` for a position covariance, ``6`` for a pose covariance
            in ``[x, y, z, roll, pitch, yaw]`` order.
    """

    values: tuple[float, ...] = Field(description="Row-major flattened covariance entries.")
    dimension: int = Field(default=3, ge=3, le=6, description="3 for position, 6 for pose.")

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected = self.dimension * self.dimension
        if len(self.values) != expected:
            raise ValueError(
                f"dimension {self.dimension} requires {expected} values, got {len(self.values)}"
            )
        return self

    @classmethod
    def zeros(cls, dimension: int = 3) -> Covariance:
        """Return an all-zero covariance (perfect knowledge; for tests only)."""
        return cls(values=tuple([0.0] * (dimension * dimension)), dimension=dimension)

    @classmethod
    def isotropic(cls, sigma: float, dimension: int = 3) -> Covariance:
        """Return ``sigma**2 * I`` — equal, independent variance on every axis."""
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        values = tuple(
            sigma**2 if index % (dimension + 1) == 0 else 0.0
            for index in range(dimension * dimension)
        )
        return cls(values=values, dimension=dimension)

    @classmethod
    def from_diagonal(cls, sigmas: tuple[float, ...]) -> Covariance:
        """Build a diagonal covariance from per-axis standard deviations."""
        dimension = len(sigmas)
        if dimension not in (3, 6):
            raise ValueError(f"diagonal covariance needs 3 or 6 sigmas, got {dimension}")
        values: list[float] = []
        for row in range(dimension):
            for col in range(dimension):
                values.append(sigmas[row] ** 2 if row == col else 0.0)
        return cls(values=tuple(values), dimension=dimension)

    @classmethod
    def from_matrix(cls, matrix: Any) -> Covariance:
        """Build from a nested sequence or square NumPy array."""
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"covariance must be a square 2-D array, got shape {array.shape}")
        dimension = int(array.shape[0])
        if dimension not in (3, 6):
            raise ValueError(f"covariance dimension must be 3 or 6, got {dimension}")
        return cls(values=tuple(float(v) for v in array.reshape(-1)), dimension=dimension)

    @property
    def matrix(self) -> np.ndarray:
        """Return the covariance as a ``(dimension, dimension)`` array."""
        return np.array(self.values, dtype=np.float64).reshape(self.dimension, self.dimension)

    @property
    def sigmas(self) -> np.ndarray:
        """Per-axis standard deviations (square root of the diagonal)."""
        return np.sqrt(np.clip(np.diag(self.matrix), 0.0, None))

    @property
    def trace(self) -> float:
        """Sum of per-axis variances."""
        return float(np.trace(self.matrix))

    @property
    def max_sigma(self) -> float:
        """Largest per-axis standard deviation."""
        return float(self.sigmas.max())

    @property
    def is_invertible(self) -> bool:
        """Whether the covariance is non-singular (needed for Kalman updates)."""
        return bool(np.all(np.linalg.eigvalsh(self.matrix) > 1e-12))

    def is_symmetric(self, atol: float = 1e-9) -> bool:
        """Whether the stored matrix is symmetric within ``atol``."""
        return bool(np.allclose(self.matrix, self.matrix.T, atol=atol))

    def plus(self, other: Covariance) -> Covariance:
        """Return the sum of two covariances (independent measurement sources)."""
        if self.dimension != other.dimension:
            raise ValueError(f"cannot add {self.dimension}D and {other.dimension}D covariances")
        return Covariance.from_matrix(self.matrix + other.matrix)


def zero_covariance(dimension: int = 3) -> Covariance:
    """Convenience wrapper around :meth:`Covariance.zeros`."""
    return Covariance.zeros(dimension)

class Pose(VersionedModel):
    """A rigid-body pose: position in metres plus orientation."""

    position: Vec3 = Field(default_factory=lambda: Vec3.of(0.0, 0.0, 0.0))
    orientation: Quaternion = Field(default_factory=Quaternion.identity)

    @classmethod
    def from_position(cls, position: Vec3) -> Pose:
        """Build a pose with identity orientation at ``position``."""
        return cls(position=position, orientation=Quaternion.identity())

    @property
    def array(self) -> np.ndarray:
        """Return ``[x, y, z, qw, qx, qy, qz]`` as a ``float64`` array."""
        return np.concatenate([self.position.array, self.orientation.array])

    def translation_distance_to(self, other: Pose) -> float:
        """Positional distance in metres (orientation ignored)."""
        return self.position.distance_to(other.position)

    def combined_distance_to(self, other: Pose) -> tuple[float, float]:
        """Return ``(position_error_m, orientation_error_rad)``."""
        return (
            self.translation_distance_to(other),
            self.orientation.angular_distance_to(other.orientation),
        )

