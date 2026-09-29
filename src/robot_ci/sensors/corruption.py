"""Sensor corruption engine: deterministic observation degradation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_ci.core.seeding import make_numpy_rng
from robot_ci.models.geometry import Covariance, Vec3
from robot_ci.models.sensors import SensorBundleConfig
from robot_ci.models.state import Observation

__all__ = [
    "CorruptionSummary",
    "apply_corruption",
    "belief_position_sigma_m",
    "corruption_is_active",
]


def corruption_is_active(config: SensorBundleConfig) -> bool:
    """Whether any corruption in the bundle would change an observation."""
    return not config.is_nominal


def belief_position_sigma_m(config: SensorBundleConfig, *, base_sigma_m: float = 0.002) -> float:
    """Predict the belief uncertainty a config produces, without sampling."""
    sigma_m = float(base_sigma_m)
    depth = config.depth
    if depth.enabled:
        sigma_m = float(np.sqrt(sigma_m**2 + (depth.sigma_mm / 1000.0) ** 2))
    rgb = config.rgb
    if rgb.enabled:
        sigma_m += 0.0005 * float(rgb.occlusion_percent)
        sigma_m += 0.0002 * float(rgb.blur_kernel_px)
        sigma_m += 0.0100 * float(rgb.gaussian_noise_std)
    return sigma_m


@dataclass(frozen=True)
class CorruptionSummary:
    """What one corruption application changed, for logs and reports."""

    depth_sigma_m: float
    depth_bias_m: float
    position_noise_m: float
    confidence_scale: float
    joint_noise_rad: float
    frame_age_ms: float
    frame_dropped: bool


def apply_corruption(
    observation: Observation,
    config: SensorBundleConfig,
    *,
    seed: int,
    step: int = 0,
) -> tuple[Observation, CorruptionSummary]:
    """Apply deterministic sensor corruption to an observation."""
    summary = CorruptionSummary(
        depth_sigma_m=0.0,
        depth_bias_m=0.0,
        position_noise_m=0.0,
        confidence_scale=1.0,
        joint_noise_rad=0.0,
        frame_age_ms=0.0,
        frame_dropped=False,
    )
    if config.is_nominal:
        return observation, summary
    master = int(seed) + int(config.seed) * 1_000_003
    depth_sigma_m = float(config.depth.sigma_mm / 1000.0) if config.depth.enabled else 0.0
    depth_bias_m = float(config.depth.bias_mm / 1000.0) if config.depth.enabled else 0.0
    confidence_scale = 1.0
    if config.rgb.enabled:
        occ = float(config.rgb.occlusion_percent)
        confidence_scale = float(np.clip(1.0 - occ / 100.0, 0.0, 1.0))
        if config.rgb.blur_kernel_px > 0:
            confidence_scale *= float(np.clip(1.0 - 0.02 * config.rgb.blur_kernel_px, 0.3, 1.0))
        if config.rgb.gaussian_noise_std > 0:
            confidence_scale *= float(np.clip(1.0 - config.rgb.gaussian_noise_std, 0.3, 1.0))
    dropout_p = 0.0
    if config.depth.enabled:
        dropout_p = float(np.clip(config.depth.dropout_percent / 100.0, 0.0, 1.0))
    expected_sigma = belief_position_sigma_m(config)
    corrupted_objects = []
    for index, obj in enumerate(observation.objects):
        rng = make_numpy_rng(master, "belief", index, step)
        noise = rng.normal(0.0, depth_sigma_m, size=3) if depth_sigma_m > 0 else np.zeros(3)
        bias_vec = np.zeros(3)
        if depth_bias_m != 0.0:
            bias_vec = np.array([0.0, 0.0, depth_bias_m])
        shifted = obj.pose.position.array + noise + bias_vec
        visible = bool(obj.visible)
        if visible and dropout_p > 0 and float(rng.uniform()) < dropout_p:
            visible = False
        if config.rgb.enabled and visible:
            occ = float(config.rgb.occlusion_percent)
            hide_p = 1.0 if occ >= 95.0 else (occ / 100.0 * 0.5 if occ > 0 else 0.0)
            if hide_p > 0 and float(rng.uniform()) < hide_p:
                visible = False
        new_confidence = float(np.clip(obj.confidence * confidence_scale, 0.0, 1.0))
        if not visible:
            new_confidence = 0.0
        corrupted_objects.append(
            obj.model_copy(
                update={
                    "pose": obj.pose.model_copy(update={"position": Vec3.from_iterable(shifted)}),
                    "covariance": Covariance.isotropic(expected_sigma, dimension=6),
                    "confidence": new_confidence,
                    "visible": visible,
                }
            )
        )
    joint_positions = np.asarray(observation.robot.joint_positions, dtype=np.float64)
    joint_noise = 0.0
    if config.robot_state.enabled:
        std = float(config.robot_state.joint_noise_std_rad)
        offset = float(config.robot_state.joint_offset_rad)
        if std > 0:
            rng = make_numpy_rng(master, "joints", step)
            joint_positions = joint_positions + rng.normal(0.0, std, size=joint_positions.shape)
            joint_noise = std
        if offset != 0.0:
            joint_positions = joint_positions + offset
            joint_noise = float(np.sqrt(joint_noise**2 + offset**2))
    robot = observation.robot.model_copy(
        update={"joint_positions": tuple(float(v) for v in joint_positions)}
    )
    frame_age_ms = 0.0
    frame_dropped = False
    if config.camera.enabled and config.camera.latency.is_active:
        latency = config.camera.latency
        frame_age_ms = float(latency.delay_ms)
        if latency.jitter_ms > 0:
            rng = make_numpy_rng(master, "latency", step)
            frame_age_ms += float(rng.uniform(-latency.jitter_ms, latency.jitter_ms))
            frame_age_ms = max(0.0, frame_age_ms)
        if latency.dropped_frame_probability > 0:
            rng = make_numpy_rng(master, "drop", step)
            frame_dropped = bool(float(rng.uniform()) < latency.dropped_frame_probability)
    corrupted = observation.model_copy(
        update={
            "objects": tuple(corrupted_objects),
            "robot": robot,
            "frame_age_ms": frame_age_ms,
            "frame_dropped": frame_dropped,
        }
    )
    position_noise = float(np.sqrt(max(0.0, expected_sigma**2 - 0.002**2)))
    summary = CorruptionSummary(
        depth_sigma_m=depth_sigma_m,
        depth_bias_m=depth_bias_m,
        position_noise_m=position_noise,
        confidence_scale=confidence_scale,
        joint_noise_rad=joint_noise,
        frame_age_ms=frame_age_ms,
        frame_dropped=frame_dropped,
    )
    return corrupted, summary
