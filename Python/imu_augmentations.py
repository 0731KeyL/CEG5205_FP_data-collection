"""Label-preserving augmentations for fixed-rate multi-IMU sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "augmentation_config.json"


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
    time_warp_fraction: float
    imu_desync_sensor: int | None
    imu_desync_samples: int
    accel_noise_std: float
    gyro_noise_std: float
    lowpass_alpha: float
    channel_dropout_channel: int | None
    applied_operations: str


def load_augmentation_config(path=None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config.get("pipeline"), list):
        raise ValueError("augmentation config requires a pipeline list")
    return config


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
    min_fraction: float = 0.70,
    max_fraction: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Select a random contiguous part of a trial and resize it."""
    crop_fraction = float(rng.uniform(min_fraction, max_fraction))
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
    probability: float = 0.45,
    min_fraction: float = 0.10,
    max_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Drop one six-channel IMU for a contiguous part of the sequence."""
    if values.shape[1] != 12:
        raise ValueError("sensor outage expects exactly 12 channels")
    if rng.random() > probability:
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


def time_warp(
    values: np.ndarray,
    rng: np.random.Generator,
    max_fraction: float = 0.08,
) -> tuple[np.ndarray, float]:
    """Apply a mild monotonic time warp while preserving sequence length."""
    length = len(values)
    if length < 3:
        return values.copy(), 0.0
    amount = float(rng.uniform(-max_fraction, max_fraction))
    positions = np.linspace(0.0, 1.0, length)
    warped = positions + amount * np.sin(np.pi * positions)
    warped = np.clip(warped, 0.0, 1.0)
    result = np.empty_like(values)
    for channel in range(values.shape[1]):
        result[:, channel] = np.interp(
            warped,
            positions,
            values[:, channel],
        )
    return result, amount


def add_imu_desync(
    values: np.ndarray,
    visible_mask: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = 2,
) -> tuple[np.ndarray, np.ndarray, int | None, int]:
    """Delay one IMU by one or two samples."""
    sensor = int(rng.integers(0, 2))
    shift = int(rng.integers(1, max_shift + 1))
    channels = slice(sensor * 6, sensor * 6 + 6)
    result = values.copy()
    mask = visible_mask.copy()
    result[shift:, channels] = values[:-shift, channels]
    mask[shift:, channels] = visible_mask[:-shift, channels]
    result[:shift, channels] = 0
    mask[:shift, channels] = False
    return result, mask, sensor, shift


def add_gaussian_noise(
    values: np.ndarray,
    visible_mask: np.ndarray,
    rng: np.random.Generator,
    accel_std: float = 0.03,
    gyro_std: float = 0.005,
) -> tuple[np.ndarray, float, float]:
    """Add small SI-unit sensor noise to visible values."""
    result = values.copy()
    for start in (0, 6):
        accel = result[:, start:start + 3]
        gyro = result[:, start + 3:start + 6]
        accel_mask = visible_mask[:, start:start + 3]
        gyro_mask = visible_mask[:, start + 3:start + 6]
        accel += rng.normal(0.0, accel_std, size=accel.shape) * accel_mask
        gyro += rng.normal(0.0, gyro_std, size=gyro.shape) * gyro_mask
    return result, accel_std, gyro_std


def apply_lowpass_variation(
    values: np.ndarray,
    rng: np.random.Generator,
    alpha_min: float = 0.55,
    alpha_max: float = 0.85,
) -> tuple[np.ndarray, float]:
    """Apply a mild causal exponential smoothing."""
    alpha = float(rng.uniform(alpha_min, alpha_max))
    result = values.copy()
    for sample in range(1, len(result)):
        result[sample] = alpha * values[sample] + (1.0 - alpha) * result[sample - 1]
    return result, alpha


def add_channel_dropout(
    values: np.ndarray,
    visible_mask: np.ndarray,
    rng: np.random.Generator,
    min_length: int = 2,
    max_length: int = 8,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Drop one axis for a short interval."""
    channel = int(rng.integers(0, values.shape[1]))
    length = min(
        len(values),
        int(rng.integers(min_length, max_length + 1)),
    )
    start = int(rng.integers(0, len(values) - length + 1))
    result = values.copy()
    mask = visible_mask.copy()
    hold_from = max(0, start - 1)
    result[start:start + length, channel] = result[hold_from, channel]
    mask[start:start + length, channel] = False
    return result, mask, channel


def apply_physical_perturbation(
    values: np.ndarray,
    rng: np.random.Generator,
    rotation_deg: float = 4.0,
    gain_min: float = 0.95,
    gain_max: float = 1.05,
    gyro_bias_rad_s: float = 0.01,
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
        angles = np.deg2rad(
            rng.uniform(-rotation_deg, rotation_deg, size=3)
        )
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        rotation = np.array([
            [cy * cz, -cy * sz, sy],
            [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
            [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
        ])
        accel_gain = float(rng.uniform(gain_min, gain_max))
        gyro_gain = float(rng.uniform(gain_min, gain_max))
        gyro_bias = rng.uniform(
            -gyro_bias_rad_s,
            gyro_bias_rad_s,
            size=3,
        )

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


def _op_time_warp(values, mask, rng, params):
    result, amount = time_warp(
        values,
        rng,
        params.get("max_fraction", 0.08),
    )
    return result, mask, {"time_warp_fraction": amount}


def _op_time_shift(values, mask, rng, params):
    max_shift = max(
        1,
        int(round(len(values) * params.get("max_fraction", 0.20))),
    )
    shift = int(rng.integers(-max_shift, max_shift + 1))
    result, visible = shift_with_padding(values, mask, shift)
    return result, visible, {"shift_samples": shift}


def _op_physical_perturbation(values, mask, rng, params):
    result, info = apply_physical_perturbation(
        values,
        rng,
        rotation_deg=params.get("rotation_deg", 4.0),
        gain_min=params.get("gain_min", 0.95),
        gain_max=params.get("gain_max", 1.05),
        gyro_bias_rad_s=params.get("gyro_bias_rad_s", 0.01),
    )
    return result, mask, {
        "imu1_rotation_deg": info["imu1_rotation_deg"],
        "imu2_rotation_deg": info["imu2_rotation_deg"],
        "imu1_accel_gain": info["imu1_accel_gain"],
        "imu2_accel_gain": info["imu2_accel_gain"],
        "imu1_gyro_gain": info["imu1_gyro_gain"],
        "imu2_gyro_gain": info["imu2_gyro_gain"],
        "max_gyro_bias_rad_s": max(
            info["imu1_gyro_bias_rad_s"],
            info["imu2_gyro_bias_rad_s"],
        ),
    }


def _op_imu_desync(values, mask, rng, params):
    result, visible, sensor, shift = add_imu_desync(
        values,
        mask,
        rng,
        params.get("max_shift", 2),
    )
    return result, visible, {
        "imu_desync_sensor": sensor,
        "imu_desync_samples": shift,
    }


def _op_gaussian_noise(values, mask, rng, params):
    result, accel_std, gyro_std = add_gaussian_noise(
        values,
        mask,
        rng,
        params.get("accel_std", 0.03),
        params.get("gyro_std", 0.005),
    )
    return result, mask, {
        "accel_noise_std": accel_std,
        "gyro_noise_std": gyro_std,
    }


def _op_lowpass_variation(values, mask, rng, params):
    result, alpha = apply_lowpass_variation(
        values,
        rng,
        params.get("alpha_min", 0.55),
        params.get("alpha_max", 0.85),
    )
    return result, mask, {"lowpass_alpha": alpha}


def _op_packet_loss(values, mask, rng, params):
    result, visible, bursts = add_packet_loss(
        values,
        mask,
        rng,
        params.get("bursts", 2),
        params.get("max_burst", 5),
    )
    return result, visible, {"packet_loss_bursts": bursts}


def _op_channel_dropout(values, mask, rng, params):
    result, visible, channel = add_channel_dropout(
        values,
        mask,
        rng,
        params.get("min_length", 2),
        params.get("max_length", 8),
    )
    return result, visible, {"channel_dropout_channel": channel}


def _op_sensor_outage(values, mask, rng, params):
    result, visible, sensor = add_sensor_outage(
        values,
        mask,
        rng,
        probability=1.0,
        min_fraction=params.get("min_fraction", 0.10),
        max_fraction=params.get("max_fraction", 0.30),
    )
    return result, visible, {"sensor_outage_sensor": sensor}


AUGMENTATION_REGISTRY = {
    "time_warp": _op_time_warp,
    "time_shift": _op_time_shift,
    "physical_perturbation": _op_physical_perturbation,
    "imu_desync": _op_imu_desync,
    "gaussian_noise": _op_gaussian_noise,
    "lowpass_variation": _op_lowpass_variation,
    "packet_loss": _op_packet_loss,
    "channel_dropout": _op_channel_dropout,
    "sensor_outage": _op_sensor_outage,
}


def augment_window(
    values: np.ndarray,
    *,
    target_length: int = 50,
    seed: int | None = None,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, AugmentationInfo]:
    """Create a configured label-preserving window with loss masks."""
    if values.ndim != 2:
        raise ValueError("values must have shape (time, channels)")
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float32)
    config = config or load_augmentation_config()
    crop = config.get("crop", {})
    result, crop_fraction = crop_and_resize(
        values,
        target_length,
        rng,
        crop.get("min_fraction", 0.70),
        crop.get("max_fraction", 1.0),
    )
    visible = np.ones_like(result, dtype=bool)
    info = {
        "shift_samples": 0,
        "packet_loss_bursts": 0,
        "sensor_outage_sensor": None,
        "imu1_rotation_deg": 0.0,
        "imu2_rotation_deg": 0.0,
        "imu1_accel_gain": 1.0,
        "imu2_accel_gain": 1.0,
        "imu1_gyro_gain": 1.0,
        "imu2_gyro_gain": 1.0,
        "max_gyro_bias_rad_s": 0.0,
        "time_warp_fraction": 0.0,
        "imu_desync_sensor": None,
        "imu_desync_samples": 0,
        "accel_noise_std": 0.0,
        "gyro_noise_std": 0.0,
        "lowpass_alpha": 1.0,
        "channel_dropout_channel": None,
    }
    applied = []
    for operation in config["pipeline"]:
        name = operation["name"]
        if name not in AUGMENTATION_REGISTRY:
            raise ValueError(f"unknown augmentation: {name}")
        if rng.random() > float(operation.get("probability", 1.0)):
            continue
        result, visible, details = AUGMENTATION_REGISTRY[name](
            result,
            visible,
            rng,
            operation,
        )
        info.update(details)
        applied.append(name)

    return result, visible, AugmentationInfo(
        crop_fraction=crop_fraction,
        **info,
        applied_operations="|".join(applied),
    )
