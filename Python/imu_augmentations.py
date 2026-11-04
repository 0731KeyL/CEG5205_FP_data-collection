"""Label-preserving augmentations for fixed-rate multi-IMU sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentationInfo:
    crop_fraction: float
    shift_samples: int
    packet_loss_bursts: int
    sensor_outage_sensor: int | None
    imu1_rotation_deg: float
    imu2_rotation_deg: float
    imu1_accel_gain: float
    imu2_accel_gain: float
    imu1_gyro_gain: float
    imu2_gyro_gain: float
    max_gyro_bias_rad_s: float


def resample_sequence(values: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a (time, channel) sequence to target_length."""
    if values.ndim != 2:
        raise ValueError("values must have shape (time, channels)")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(values) == 0:
        raise ValueError("cannot resample an empty sequence")
    if len(values) == target_length:
        return values.astype(np.float32, copy=True)

    old_positions = np.linspace(0.0, 1.0, len(values))
    new_positions = np.linspace(0.0, 1.0, target_length)
    result = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        result[:, channel] = np.interp(
            new_positions,
            old_positions,
            values[:, channel],
        )
    return result


def crop_and_resize(
    values: np.ndarray,
    target_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Select a random contiguous part of a trial and resize it."""
    crop_fraction = float(rng.uniform(0.70, 1.0))
    crop_length = max(3, int(round(len(values) * crop_fraction)))
    crop_length = min(crop_length, len(values))
    max_start = len(values) - crop_length
    start = int(rng.integers(0, max_start + 1))
    cropped = values[start:start + crop_length]
    return resample_sequence(cropped, target_length), crop_fraction


def shift_with_padding(
    values: np.ndarray,
    visible_mask: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the event in time and mark inserted samples as invisible."""
    shifted_values = np.zeros_like(values)
    shifted_mask = np.zeros_like(visible_mask, dtype=bool)

    if shift >= 0:
        available = len(values) - shift
        if available > 0:
            shifted_values[shift:] = values[:available]
            shifted_mask[shift:] = visible_mask[:available]
    else:
        amount = -shift
        available = len(values) - amount
        if available > 0:
            shifted_values[:available] = values[amount:]
            shifted_mask[:available] = visible_mask[amount:]
    return shifted_values, shifted_mask


def add_packet_loss(
    values: np.ndarray,
    visible_mask: np.ndarray,
    rng: np.random.Generator,
    bursts: int = 2,
    max_burst: int = 5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Simulate short packet bursts using sample-and-hold plus a mask."""
    result = values.copy()
    mask = visible_mask.copy()
    added = 0
    for _ in range(bursts):
        if rng.random() > 0.55:
            continue
        length = int(rng.integers(1, max_burst + 1))
        if length >= len(result):
            continue
        start = int(rng.integers(0, len(result) - length + 1))
        hold_from = max(0, start - 1)
        result[start:start + length] = result[hold_from]
        mask[start:start + length] = False
        added += 1
    return result, mask, added


def add_sensor_outage(
    values: np.ndarray,
    visible_mask: np.ndarray,
    rng: np.random.Generator,
    min_fraction: float = 0.10,
    max_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Drop one six-channel IMU for a contiguous part of the sequence."""
    if values.shape[1] != 12:
        raise ValueError("sensor outage expects exactly 12 channels")
    if rng.random() > 0.45:
        return values.copy(), visible_mask.copy(), None

    sensor = int(rng.integers(0, 2))
    channels = slice(sensor * 6, sensor * 6 + 6)
    fraction = float(rng.uniform(min_fraction, max_fraction))
    length = max(1, int(round(len(values) * fraction)))
    start = int(rng.integers(0, len(values) - length + 1))

    result = values.copy()
    mask = visible_mask.copy()
    hold_from = max(0, start - 1)
    result[start:start + length, channels] = result[hold_from, channels]
    mask[start:start + length, channels] = False
    return result, mask, sensor


def apply_physical_perturbation(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Apply small rigid rotations, gains, and gyro bias per IMU."""
    if values.shape[1] != 12:
        raise ValueError("physical perturbation expects 12 channels")
    result = values.copy()
    info = {}

    for sensor in range(2):
        start = sensor * 6
        accel = result[:, start:start + 3]
        gyro = result[:, start + 3:start + 6]
        angles = np.deg2rad(rng.uniform(-4.0, 4.0, size=3))
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        rotation = np.array([
            [cy * cz, -cy * sz, sy],
            [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
            [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
        ])
        accel_gain = float(rng.uniform(0.95, 1.05))
        gyro_gain = float(rng.uniform(0.95, 1.05))
        gyro_bias = rng.uniform(-0.01, 0.01, size=3)

        result[:, start:start + 3] = (accel @ rotation.T) * accel_gain
        result[:, start + 3:start + 6] = (
            (gyro @ rotation.T) * gyro_gain + gyro_bias
        )
        info[f"imu{sensor + 1}_rotation_deg"] = float(
            np.linalg.norm(np.rad2deg(angles))
        )
        info[f"imu{sensor + 1}_accel_gain"] = accel_gain
        info[f"imu{sensor + 1}_gyro_gain"] = gyro_gain
        info[f"imu{sensor + 1}_gyro_bias_rad_s"] = float(
            np.max(np.abs(gyro_bias))
        )
    return result, info


def augment_window(
    values: np.ndarray,
    *,
    target_length: int = 50,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, AugmentationInfo]:
    """Create a label-preserving window with loss masks.

    Args:
        values: Raw or scaled array with shape (time, channels).
        target_length: Output length used by the model.
        seed: Optional deterministic seed.

    Returns:
        values, visible_mask, augmentation metadata
    """
    if values.ndim != 2:
        raise ValueError("values must have shape (time, channels)")
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float32)

    cropped, crop_fraction = crop_and_resize(values, target_length, rng)
    visible = np.ones_like(cropped, dtype=bool)

    max_shift = max(1, int(round(target_length * 0.20)))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    shifted, visible = shift_with_padding(cropped, visible, shift)
    shifted, physical = apply_physical_perturbation(shifted, rng)

    shifted, visible, packet_bursts = add_packet_loss(
        shifted,
        visible,
        rng,
    )
    shifted, visible, sensor = add_sensor_outage(
        shifted,
        visible,
        rng,
    )

    return shifted, visible, AugmentationInfo(
        crop_fraction=crop_fraction,
        shift_samples=shift,
        packet_loss_bursts=packet_bursts,
        sensor_outage_sensor=sensor,
        imu1_rotation_deg=physical["imu1_rotation_deg"],
        imu2_rotation_deg=physical["imu2_rotation_deg"],
        imu1_accel_gain=physical["imu1_accel_gain"],
        imu2_accel_gain=physical["imu2_accel_gain"],
        imu1_gyro_gain=physical["imu1_gyro_gain"],
        imu2_gyro_gain=physical["imu2_gyro_gain"],
        max_gyro_bias_rad_s=max(
            physical["imu1_gyro_bias_rad_s"],
            physical["imu2_gyro_bias_rad_s"],
        ),
    )
