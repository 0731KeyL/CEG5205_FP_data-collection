"""Timestamp, windowing, feature, duplicate, and split utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


ACC_CHANNELS = [0, 1, 2, 6, 7, 8]
GYRO_CHANNELS = [3, 4, 5, 9, 10, 11]


def resample_to_uniform_time(
    values: np.ndarray,
    timestamps_ms: np.ndarray,
    sampling_rate_hz: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a supplied device-time axis onto a uniform sampling grid."""
    values = np.asarray(values, dtype=np.float64)
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(timestamps_ms):
        raise ValueError("values and timestamps must have matching 2-D lengths")
    if len(values) < 2:
        raise ValueError("at least two samples are required")

    order = np.argsort(timestamps_ms, kind="stable")
    values = values[order]
    timestamps_ms = timestamps_ms[order]
    elapsed = (timestamps_ms - timestamps_ms[0]) / 1000.0
    keep = np.concatenate([[True], np.diff(elapsed) > 0])
    values = values[keep]
    elapsed = elapsed[keep]
    if len(values) < 2:
        raise ValueError("timestamps contain fewer than two unique samples")

    duration_s = float(elapsed[-1])
    sample_count = int(np.floor(duration_s * sampling_rate_hz)) + 1
    uniform_time = np.arange(sample_count, dtype=np.float64) / sampling_rate_hz
    uniform_values = np.empty((sample_count, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        uniform_values[:, channel] = np.interp(
            uniform_time,
            elapsed,
            values[:, channel],
        )
    return uniform_values, uniform_time, duration_s


def event_aware_window(
    values: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Center a fixed window near the main motion event."""
    if values.ndim != 2:
        raise ValueError("values must have shape (time, channels)")
    if values.shape[1] != 12:
        raise ValueError("event-aware windowing expects 12 channels")

    values = np.asarray(values, dtype=np.float32)
    original_length = len(values)
    visible = np.ones_like(values, dtype=bool)

    if original_length <= target_length:
        result = np.zeros((target_length, values.shape[1]), dtype=np.float32)
        mask = np.zeros_like(result, dtype=bool)
        offset = (target_length - original_length) // 2
        result[offset:offset + original_length] = values
        mask[offset:offset + original_length] = True
        return result, mask, {
            "window_start_s": -offset / 50.0,
            "event_center_s": 0.0,
            "motion_level": 0.0,
            "padding_samples": target_length - original_length,
        }

    acceleration = values[:, ACC_CHANNELS]
    gyroscope = values[:, GYRO_CHANNELS]
    acc_energy = np.zeros(original_length, dtype=np.float64)
    gyro_energy = np.zeros(original_length, dtype=np.float64)
    acc_energy[1:] = np.linalg.norm(
        np.diff(acceleration, axis=0),
        axis=1,
    )
    gyro_energy[1:] = np.linalg.norm(gyroscope[1:], axis=1)
    energy = acc_energy + 0.8 * gyro_energy
    if len(energy) >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        energy = np.convolve(energy, kernel, mode="same")

    motion_level = float(np.max(energy))
    if motion_level < 0.15:
        center = original_length / 2.0
    else:
        active = energy >= max(0.20 * motion_level, 0.05)
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            center = original_length / 2.0
        else:
            weights = energy[indices]
            center = float(np.average(indices, weights=weights))

    start = int(round(center - target_length / 2.0))
    start = min(max(0, start), original_length - target_length)
    result = values[start:start + target_length]
    mask = visible[start:start + target_length]
    return result, mask, {
        "window_start_s": start / 50.0,
        "event_center_s": center / 50.0,
        "motion_level": motion_level,
        "padding_samples": 0,
    }


def feature_names(channel_names: list[str]) -> list[str]:
    names = []
    for channel in channel_names:
        for statistic in (
            "mean", "std", "min", "max", "rms", "mean_abs_diff",
        ):
            names.append(f"{channel}_{statistic}")
    for sensor in (1, 2):
        for quantity in ("acc_mag", "gyro_mag"):
            for statistic in ("mean", "std", "max"):
                names.append(f"imu{sensor}_{quantity}_{statistic}")
    return names


def extract_window_features(
    values: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    """Extract mask-aware summary features for feature-based models."""
    values = np.asarray(values, dtype=np.float64)
    visible_mask = np.asarray(visible_mask, dtype=bool)
    features = []
    for channel in range(values.shape[1]):
        selected = values[:, channel][visible_mask[:, channel]]
        if len(selected) == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        diff = np.diff(selected)
        features.extend([
            float(selected.mean()),
            float(selected.std()),
            float(selected.min()),
            float(selected.max()),
            float(np.sqrt(np.mean(selected ** 2))),
            float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        ])

    for start in (0, 6):
        accel = values[:, start:start + 3]
        gyro = values[:, start + 3:start + 6]
        accel_mask = visible_mask[:, start:start + 3].all(axis=1)
        gyro_mask = visible_mask[:, start + 3:start + 6].all(axis=1)
        for vector, valid in ((accel, accel_mask), (gyro, gyro_mask)):
            if valid.any():
                magnitude = np.linalg.norm(vector[valid], axis=1)
                features.extend([
                    float(magnitude.mean()),
                    float(magnitude.std()),
                    float(magnitude.max()),
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
    return np.asarray(features, dtype=np.float32)


def near_duplicate_report(
    windows: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    """Find highly similar recordings within the same gesture class."""
    reports = [{
        "duplicate_of": "",
        "duplicate_similarity": 0.0,
        "duplicate_rms_ratio": float("inf"),
        "near_duplicate": False,
    } for _ in range(len(windows))]

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        for offset, index in enumerate(indices):
            vector = windows[index].astype(np.float64)
            vector = vector - vector.mean(axis=0, keepdims=True)
            vector_norm = np.linalg.norm(vector)
            for other in indices[offset + 1:]:
                other_vector = windows[other].astype(np.float64)
                other_vector = other_vector - other_vector.mean(
                    axis=0,
                    keepdims=True,
                )
                denominator = vector_norm * np.linalg.norm(other_vector)
                similarity = (
                    float(np.sum(vector * other_vector) / denominator)
                    if denominator > 0
                    else 0.0
                )
                rms = float(
                    np.sqrt(np.mean((windows[index] - windows[other]) ** 2))
                )
                scale = float(np.sqrt(np.mean(windows[index] ** 2))) or 1.0
                rms_ratio = rms / scale
                if similarity > reports[index]["duplicate_similarity"]:
                    reports[index] = {
                        "duplicate_of": str(other),
                        "duplicate_similarity": similarity,
                        "duplicate_rms_ratio": rms_ratio,
                        "near_duplicate": False,
                    }
                if similarity > reports[other]["duplicate_similarity"]:
                    reports[other] = {
                        "duplicate_of": str(index),
                        "duplicate_similarity": similarity,
                        "duplicate_rms_ratio": rms_ratio,
                        "near_duplicate": False,
                    }
    return reports


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def assign_splits(records: list[dict], seed: int) -> str:
    """Split by subject when possible, otherwise stratify source trials."""
    subjects = sorted({str(record["subject_id"]) for record in records})
    if len(subjects) >= 3:
        ordered = sorted(subjects, key=lambda value: _stable_hash(value, seed))
        test_count = max(1, round(0.20 * len(ordered)))
        val_count = max(1, round(0.20 * len(ordered)))
        if len(ordered) - test_count - val_count < 1:
            val_count = max(0, len(ordered) - test_count - 1)
        test_subjects = set(ordered[:test_count])
        val_subjects = set(ordered[test_count:test_count + val_count])
        for record in records:
            subject = str(record["subject_id"])
            record["split"] = (
                "test" if subject in test_subjects
                else "validation" if subject in val_subjects
                else "train"
            )
        return "subject-disjoint"

    by_class = defaultdict(list)
    for index, record in enumerate(records):
        by_class[str(record["gesture"])].append(index)

    for indices in by_class.values():
        grouped = defaultdict(list)
        for index in indices:
            group = str(records[index].get(
                "leakage_group",
                records[index]["file"],
            ))
            grouped[group].append(index)
        ordered_groups = sorted(
            grouped,
            key=lambda group: _stable_hash(group, seed),
        )
        count = len(ordered_groups)
        if count == 1:
            test_count = 0
            val_count = 0
        elif count == 2:
            test_count = 1
            val_count = 0
        else:
            test_count = max(1, round(0.20 * count))
            val_count = max(1, round(0.15 * count))
            while count - test_count - val_count < 1:
                if val_count > 1:
                    val_count -= 1
                else:
                    test_count -= 1

        for position, group in enumerate(ordered_groups):
            split = "train"
            if position < test_count:
                split = "test"
            elif position < test_count + val_count:
                split = "validation"
            for index in grouped[group]:
                records[index]["split"] = split
    return "within-subject-stratified-source-group"
