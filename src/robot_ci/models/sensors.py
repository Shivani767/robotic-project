"""Sensor and corruption configuration schema.

This module defines the *configuration surface* for the corruption framework.
The numeric effects are applied in Phase 5 by
:mod:`robot_ci.sensors.corruption`; defining the schema in Phase 1 keeps the
scenario format stable so experiments written now remain loadable later.

Every corruption is deterministic given ``seed``. The seed is stored **inside**
the configuration so a scenario file is self-describing: replaying it does not
require knowing where its randomness came from.
"""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import Field, model_validator

from robot_ci.models.base import VersionedModel

__all__ = [
    "CameraCorruptionConfig",
    "CameraIntrinsics",
    "DepthCorruptionConfig",
    "ExtrinsicPerturbationMode",
    "ForceTorqueCorruptionConfig",
    "LatencyConfig",
    "RGBCorruptionConfig",
    "RobotStateCorruptionConfig",
    "SensorBundleConfig",
]


class ExtrinsicPerturbationMode(str, Enum):
    """How camera extrinsics are perturbed."""

    NONE = "none"
    """Leave extrinsics untouched."""
    TRANSLATION = "translation"
    """Additive offset in metres, sampled once per episode."""
    ROTATION = "rotation"
    """Additive rotation in radians, sampled once per episode."""
    BOTH = "both"
    """Additive translation and rotation, sampled once per episode."""


class LatencyConfig(VersionedModel):
    """Temporal delay applied to a sensor stream.

    Attributes:
        enabled: Master switch for this latency source.
        delay_ms: Constant delay in milliseconds.
        jitter_ms: Uniform jitter of +/- this many milliseconds.
        dropped_frame_probability: Probability that a frame is lost entirely.
    """

    enabled: bool = Field(default=False, description="Master switch for latency injection.")
    delay_ms: float = Field(default=0.0, ge=0.0, le=5000.0, description="Constant delay in ms.")
    jitter_ms: float = Field(default=0.0, ge=0.0, le=5000.0, description="Uniform +/- jitter in ms.")
    dropped_frame_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability a frame is dropped.")

    @property
    def total_delay_ms(self) -> float:
        """Nominal (jitter-free) delay; the counterexample-visible magnitude."""
        return self.delay_ms if self.enabled else 0.0

    @property
    def is_active(self) -> bool:
        """Whether this latency source changes the observation stream."""
        return self.enabled and (
            self.delay_ms > 0.0 or self.jitter_ms > 0.0 or self.dropped_frame_probability > 0.0
        )


class RGBCorruptionConfig(VersionedModel):
    """Appearance corruptions applied to the colour image.

    Parameters use physical/intuitive units so a scenario reads like a
    description of the world (a brightness percentage, not a gain of 0.4).
    """

    enabled: bool = Field(default=False, description="Master switch for RGB corruption.")
    gaussian_noise_std: float = Field(default=0.0, ge=0.0, le=1.0, description="Additive Gaussian sigma in [0,1] intensity units.")
    gaussian_noise_mean: float = Field(default=0.0, ge=-1.0, le=1.0, description="Additive Gaussian mean offset in [0,1] units.")
    blur_kernel_px: int = Field(default=0, ge=0, le=31, description="Odd blur kernel size in pixels; 0 disables.")
    brightness_scale: float = Field(default=1.0, ge=0.0, le=4.0, description="Multiplicative gain; 1.0 is nominal.")
    contrast_scale: float = Field(default=1.0, ge=0.0, le=4.0, description="Contrast about mid-grey; 1.0 is nominal.")
    occlusion_percent: float = Field(default=0.0, ge=0.0, le=95.0, description="Fraction of image area blocked by occluders.")
    motion_blur_px: float = Field(default=0.0, ge=0.0, le=31.0, description="Linear motion blur length in pixels.")
    color_shift_rgb: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="Additive per-channel shift in [-1,1] units.")

    @model_validator(mode="after")
    def _odd_kernel(self) -> Self:
        if self.blur_kernel_px % 2 == 0 and self.blur_kernel_px > 0:
            raise ValueError(
                f"blur_kernel_px must be odd for a true Gaussian kernel, got {self.blur_kernel_px}"
            )
        return self

    @property
    def is_active(self) -> bool:
        """Whether any RGB corruption would change the image."""
        if not self.enabled:
            return False
        return bool(
            self.gaussian_noise_std > 0.0
            or self.gaussian_noise_mean != 0.0
            or self.blur_kernel_px > 0
            or self.brightness_scale != 1.0
            or self.contrast_scale != 1.0
            or self.occlusion_percent > 0.0
            or self.motion_blur_px > 0.0
            or any(c != 0.0 for c in self.color_shift_rgb)
        )

    @property
    def nominal_brightness_percent(self) -> float:
        """Brightness expressed as a percentage of the nominal level."""
        return self.brightness_scale * 100.0


class DepthCorruptionConfig(VersionedModel):
    """Corruptions applied to the depth image.

    ``sigma_mm``/``bias_mm`` are millimetres because that is how depth-sensor
    error is specified in practice; the framework converts to metres internally.
    """

    enabled: bool = Field(default=False, description="Master switch for depth corruption.")
    sigma_mm: float = Field(default=0.0, ge=0.0, le=500.0, description="Gaussian depth noise sigma in mm.")
    bias_mm: float = Field(default=0.0, ge=-500.0, le=500.0, description="Systematic depth offset in mm.")
    dropout_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Fraction of depth pixels set invalid.")
    quantization_mm: float = Field(default=0.0, ge=0.0, le=100.0, description="Depth quantisation step in mm; 0 disables.")
    max_range_m: float = Field(default=6.0, gt=0.0, description="Pixels beyond this range are invalid.")
    min_range_m: float = Field(default=0.05, ge=0.0, description="Pixels closer than this range are invalid.")
    temporal_lag_frames: int = Field(default=0, ge=0, le=60, description="Depth image delayed by N frames.")

    @model_validator(mode="after")
    def _ranges(self) -> Self:
        if self.min_range_m >= self.max_range_m:
            raise ValueError(
                f"min_range_m ({self.min_range_m}) must be below max_range_m ({self.max_range_m})"
            )
        return self

    @property
    def is_active(self) -> bool:
        """Whether any depth corruption would change the depth image."""
        if not self.enabled:
            return False
        return bool(
            self.sigma_mm > 0.0
            or self.bias_mm != 0.0
            or self.dropout_percent > 0.0
            or self.quantization_mm > 0.0
            or self.temporal_lag_frames > 0
        )

    @property
    def sigma_m(self) -> float:
        """Gaussian noise sigma converted to metres."""
        return self.sigma_mm / 1000.0

    @property
    def bias_m(self) -> float:
        """Systematic bias converted to metres."""
        return self.bias_mm / 1000.0


class CameraCorruptionConfig(VersionedModel):
    """Whole-camera corruptions: extrinsics, intrinsics and latency."""

    enabled: bool = Field(default=False, description="Master switch for camera corruption.")
    extrinsic_mode: ExtrinsicPerturbationMode = Field(default=ExtrinsicPerturbationMode.NONE, description="How extrinsics are perturbed.")
    extrinsic_translation_m: float = Field(default=0.0, ge=0.0, le=1.0, description="Extrinsic translation offset magnitude in m.")
    extrinsic_rotation_deg: float = Field(default=0.0, ge=0.0, le=45.0, description="Extrinsic rotation offset magnitude in deg.")
    focal_scale: float = Field(default=1.0, ge=0.5, le=2.0, description="Multiplicative focal-length scale; 1.0 nominal.")
    principal_point_shift_px: float = Field(default=0.0, ge=0.0, le=200.0, description="Principal point shift magnitude in px.")
    latency: LatencyConfig = Field(default_factory=LatencyConfig, description="Frame delivery delay.")

    @property
    def is_active(self) -> bool:
        """Whether any camera corruption would change observations."""
        if not self.enabled:
            return False
        return bool(
            self.extrinsic_mode is not ExtrinsicPerturbationMode.NONE
            or self.focal_scale != 1.0
            or self.principal_point_shift_px > 0.0
            or self.latency.is_active
        )


class RobotStateCorruptionConfig(VersionedModel):
    """Corruptions applied to joint-state feedback.

    ``joint_offset_rad`` models calibration or mechanical bias;
    ``joint_noise_std_rad`` models stochastic encoder noise;
    ``encoder_delay_frames`` models a controller acting on stale state.
    """

    enabled: bool = Field(default=False, description="Master switch for robot-state corruption.")
    joint_noise_std_rad: float = Field(default=0.0, ge=0.0, le=0.5, description="Per-joint encoder noise sigma in rad.")
    joint_offset_rad: float = Field(default=0.0, ge=0.0, le=0.5, description="Per-joint constant offset magnitude in rad.")
    offset_joint_indices: tuple[int, ...] = Field(default=(), description="Joint indices receiving the offset; empty means all.")
    encoder_delay_frames: int = Field(default=0, ge=0, le=30, description="Frames of state delay seen by the controller.")
    gripper_force_scale: float = Field(default=1.0, ge=0.0, le=2.0, description="Multiplicative gripper force scale; 0 means failure.")
    gripper_stuck: bool = Field(default=False, description="Gripper fails to open or close at all.")

    @property
    def is_active(self) -> bool:
        """Whether any robot-state corruption would change the episode."""
        if not self.enabled:
            return False
        return bool(
            self.joint_noise_std_rad > 0.0
            or self.joint_offset_rad > 0.0
            or self.encoder_delay_frames > 0
            or self.gripper_force_scale != 1.0
            or self.gripper_stuck
        )


class ForceTorqueCorruptionConfig(VersionedModel):
    """Corruptions applied to the wrist force/torque stream."""

    enabled: bool = Field(default=False, description="Master switch for force/torque corruption.")
    bias_n: float = Field(default=0.0, ge=0.0, le=100.0, description="Constant force bias magnitude in N.")
    torque_bias_nm: float = Field(default=0.0, ge=0.0, le=50.0, description="Constant torque bias magnitude in Nm.")
    noise_std_n: float = Field(default=0.0, ge=0.0, le=100.0, description="Force noise sigma in N.")
    dropout_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Fraction of samples lost.")

    @property
    def is_active(self) -> bool:
        """Whether any force/torque corruption would change the episode."""
        return self.enabled and bool(
            self.bias_n > 0.0
            or self.torque_bias_nm > 0.0
            or self.noise_std_n > 0.0
            or self.dropout_percent > 0.0
        )


class CameraIntrinsics(VersionedModel):
    """Pinhole camera intrinsics.

    Units follow the OpenCV convention so a corrupted image can be handed to an
    OpenCV detector unchanged.
    """

    width_px: int = Field(default=160, ge=8, le=4096, description="Image width in pixels.")
    height_px: int = Field(default=120, ge=8, le=4096, description="Image height in pixels.")
    fx: float = Field(default=140.0, gt=0.0, description="Focal length in px along x.")
    fy: float = Field(default=140.0, gt=0.0, description="Focal length in px along y.")
    cx: float = Field(default=80.0, description="Principal point x in px.")
    cy: float = Field(default=60.0, description="Principal point y in px.")

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` in the order NumPy expects."""
        return (self.height_px, self.width_px)

    @property
    def matrix(self) -> list[list[float]]:
        """The 3x3 intrinsic matrix ``K`` as nested lists."""
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]

    def scaled(self, factor: float) -> CameraIntrinsics:
        """Return intrinsics with focal lengths and principal point scaled."""
        return CameraIntrinsics(
            width_px=self.width_px, height_px=self.height_px,
            fx=self.fx * factor, fy=self.fy * factor,
            cx=self.cx * factor, cy=self.cy * factor,
        )

    def shifted(self, dx: float, dy: float) -> CameraIntrinsics:
        """Return intrinsics with the principal point shifted by ``(dx, dy)`` pixels."""
        return CameraIntrinsics(
            width_px=self.width_px, height_px=self.height_px,
            fx=self.fx, fy=self.fy, cx=self.cx + dx, cy=self.cy + dy,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(fx, fy, cx, cy)``, the form attached to a sensor frame."""
        return (self.fx, self.fy, self.cx, self.cy)


class SensorBundleConfig(VersionedModel):
    """The complete sensor configuration of a scenario.

    Attributes:
        seed: Seed for all corruption randomness. Stored here so a scenario file
            is self-describing and replayable on its own.
        intrinsics: Nominal (uncorrupted) camera intrinsics.
        rgb: RGB appearance corruption settings.
        depth: Depth corruption settings.
        camera: Whole-camera corruption settings.
        robot_state: Joint-state corruption settings.
        force_torque: Wrist force/torque corruption settings.
        control_hz: Rate at which sensor frames are produced.
    """

    seed: int = Field(default=0, ge=0, description="Seed governing all sensor corruption randomness.")
    intrinsics: CameraIntrinsics = Field(default_factory=CameraIntrinsics, description="Nominal camera intrinsics.")
    rgb: RGBCorruptionConfig = Field(default_factory=RGBCorruptionConfig, description="RGB appearance corruption.")
    depth: DepthCorruptionConfig = Field(default_factory=DepthCorruptionConfig, description="Depth corruption.")
    camera: CameraCorruptionConfig = Field(default_factory=CameraCorruptionConfig, description="Camera-level corruption.")
    robot_state: RobotStateCorruptionConfig = Field(default_factory=RobotStateCorruptionConfig, description="Joint-state corruption.")
    force_torque: ForceTorqueCorruptionConfig = Field(default_factory=ForceTorqueCorruptionConfig, description="Force/torque corruption.")
    control_hz: float = Field(default=20.0, gt=0.0, le=1000.0, description="Sensor frame rate in Hz.")

    @property
    def is_nominal(self) -> bool:
        """Whether the bundle produces clean, uncorrupted observations."""
        return not any(
            (
                self.rgb.is_active,
                self.depth.is_active,
                self.camera.is_active,
                self.robot_state.is_active,
                self.force_torque.is_active,
            )
        )

    def active_sources(self) -> tuple[str, ...]:
        """Names of every active corruption source, for reports and CSV columns."""
        sources: list[str] = []
        if self.rgb.is_active:
            sources.append("rgb")
        if self.depth.is_active:
            sources.append("depth")
        if self.camera.is_active:
            sources.append("camera")
        if self.robot_state.is_active:
            sources.append("robot_state")
        if self.force_torque.is_active:
            sources.append("force_torque")
        return tuple(sources)

    def effective_intrinsics(self) -> CameraIntrinsics:
        """Intrinsics after applying the configured intrinsic perturbations."""
        intrinsics = self.intrinsics
        if self.camera.focal_scale != 1.0:
            intrinsics = intrinsics.scaled(self.camera.focal_scale)
        if self.camera.principal_point_shift_px > 0.0:
            intrinsics = intrinsics.shifted(self.camera.principal_point_shift_px, 0.0)
        return intrinsics
