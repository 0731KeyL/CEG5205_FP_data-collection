"""Convert the raw dual-MPU6050 CSV files into a model-ready dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from imu_augmentations import augment_window, resample_sequence
from project_config import load_project_config


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "real_imu_data"
DEFAULT_OUTPUT = BASE_DIR / "processed_imu_data"

PROJECT_CONFIG = load_project_config()
CLASS_NAMES = PROJECT_CONFIG["gestures"]
CHANNELS = [
    "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2", "gx2", "gy2", "gz2",
]
ACC_CHANNELS = [0, 1, 2, 6, 7, 8]
GYRO_CHANNELS = [3, 4, 5, 9, 10, 11]

ACCEL_LSB_PER_G = 8192.0
GYRO_LSB_PER_DEG_S = 65.5
GRAVITY_M_S2 = 9.80665
DEG_TO_RAD = np.pi / 180.0

LEGACY_SUBJECT_METADATA = {
    "subject01": {
        "hand_side": "left",
        "handedness": "unknown",
        "sensor_positions": "index|thumb",
        "address_mapping": "0x68/0x69 to finger mapping unresolved",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare real dual-IMU recordings for training."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=3,
        help="Label-preserving augmented copies per real recording (default: 3).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def first_text(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame:
        return default
    values = frame[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    return "|".join(sorted(values.unique())) if len(values) else default


def convert_to_si(raw: np.ndarray) -> np.ndarray:
    values = raw.astype(np.float64, copy=True)
    values[:, ACC_CHANNELS] *= GRAVITY_M_S2 / ACCEL_LSB_PER_G
    values[:, GYRO_CHANNELS] *= DEG_TO_RAD / GYRO_LSB_PER_DEG_S
    return values.astype(np.float32)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    windows = []
    masks = []
    labels = []
    records = []
    source_root = args.source.resolve()

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = args.source / class_name
        for path in sorted(class_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            missing = [column for column in CHANNELS if column not in frame]
            if missing:
                raise ValueError(f"{path.name}: missing channels {missing}")

            raw = frame[CHANNELS].to_numpy(dtype=np.float64)
            physical = convert_to_si(raw)
            window = resample_sequence(physical, args.window_length)

            timestamp_ms = (
                frame["timestamp_ms"].to_numpy(dtype=np.float64)
                if "timestamp_ms" in frame
                else np.arange(len(frame), dtype=np.float64) * 20.0
            )
            duration_s = (
                float((timestamp_ms[-1] - timestamp_ms[0]) / 1000.0)
                if len(timestamp_ms) > 1
                else 0.0
            )
            rate_hz = (
                float((len(frame) - 1) / duration_s)
                if duration_s > 0
                else float("nan")
            )
            subject_id = first_text(frame, "subject_id", "unknown")
            legacy = LEGACY_SUBJECT_METADATA.get(subject_id, {})

            windows.append(window)
            masks.append(np.ones_like(window, dtype=bool))
            labels.append(label_index)
            records.append({
                "file": path.name,
                "relative_path": str(path.resolve().relative_to(source_root)),
                "gesture": class_name,
                "label": label_index,
                "subject_id": subject_id,
                "hand_side": first_text(
                    frame,
                    "hand_side",
                    legacy.get("hand_side", "unknown"),
                ),
                "handedness": first_text(
                    frame,
                    "handedness",
                    legacy.get("handedness", "unknown"),
                ),
                "imu1_position": first_text(
                    frame,
                    "imu1_position",
                    "unknown",
                ),
                "imu2_position": first_text(
                    frame,
                    "imu2_position",
                    "unknown",
                ),
                "sensor_positions": first_text(
                    frame,
                    "sensor_positions",
                    legacy.get("sensor_positions", "unknown"),
                ),
                "address_mapping": first_text(
                    frame,
                    "address_mapping",
                    legacy.get("address_mapping", "unknown"),
                ),
                "original_samples": len(frame),
                "duration_s": duration_s,
                "rate_hz": rate_hz,
                "output_samples": args.window_length,
            })

    if not windows:
        raise RuntimeError(f"No real CSV recordings found below {args.source}")

    x = np.stack(windows).astype(np.float32)
    visible_mask = np.stack(masks)
    y = np.asarray(labels, dtype=np.int64)
    manifest = pd.DataFrame(records)

    np.savez_compressed(
        args.output / "real_imu_windows.npz",
        x=x,
        y=y,
        visible_mask=visible_mask,
        class_names=np.asarray(CLASS_NAMES),
        channel_names=np.asarray(CHANNELS),
        source_files=manifest["file"].to_numpy(dtype=str),
        subjects=manifest["subject_id"].to_numpy(dtype=str),
        hand_sides=manifest["hand_side"].to_numpy(dtype=str),
    )
    manifest.to_csv(args.output / "manifest.csv", index=False)

    if args.augment_copies < 0:
        raise ValueError("--augment-copies cannot be negative")

    augmented_windows = []
    augmented_masks = []
    augmented_labels = []
    augmented_source_indices = []
    augmented_is_synthetic = []
    augmentation_rows = []
    for source_index, (window, label) in enumerate(zip(x, y)):
        augmented_windows.append(window)
        augmented_masks.append(np.ones_like(window, dtype=bool))
        augmented_labels.append(int(label))
        augmented_source_indices.append(source_index)
        augmented_is_synthetic.append(False)
        augmentation_rows.append({
            "source_index": source_index,
            "source_file": manifest.iloc[source_index]["file"],
            "gesture": CLASS_NAMES[int(label)],
            "is_synthetic": False,
            "crop_fraction": 1.0,
            "shift_samples": 0,
            "packet_loss_bursts": 0,
            "sensor_outage_sensor": "",
        })

        for copy_index in range(args.augment_copies):
            augmented, visible_mask, info = augment_window(
                window,
                target_length=args.window_length,
                seed=args.seed + source_index * 1009 + copy_index,
            )
            augmented_windows.append(augmented)
            augmented_masks.append(visible_mask)
            augmented_labels.append(int(label))
            augmented_source_indices.append(source_index)
            augmented_is_synthetic.append(True)
            augmentation_rows.append({
                "source_index": source_index,
                "source_file": manifest.iloc[source_index]["file"],
                "gesture": CLASS_NAMES[int(label)],
                "is_synthetic": True,
                "crop_fraction": info.crop_fraction,
                "shift_samples": info.shift_samples,
                "packet_loss_bursts": info.packet_loss_bursts,
                "sensor_outage_sensor": (
                    "" if info.sensor_outage_sensor is None
                    else info.sensor_outage_sensor
                ),
            })

    augmented_x = np.stack(augmented_windows).astype(np.float32)
    augmented_visible_mask = np.stack(augmented_masks)
    augmented_y = np.asarray(augmented_labels, dtype=np.int64)
    source_indices = np.asarray(augmented_source_indices, dtype=np.int64)
    is_synthetic = np.asarray(augmented_is_synthetic, dtype=bool)

    np.savez_compressed(
        args.output / "training_windows_with_augmentation.npz",
        x=augmented_x,
        y=augmented_y,
        visible_mask=augmented_visible_mask,
        source_index=source_indices,
        is_synthetic=is_synthetic,
        class_names=np.asarray(CLASS_NAMES),
        channel_names=np.asarray(CHANNELS),
    )
    pd.DataFrame(augmentation_rows).to_csv(
        args.output / "augmented_manifest.csv",
        index=False,
    )

    class_counts = Counter(manifest["gesture"])
    subject_counts = Counter(manifest["subject_id"])
    summary = {
        "recordings": len(manifest),
        "shape": list(x.shape),
        "channels": CHANNELS,
        "sampling_rate_hz": "per-file; see manifest.csv",
        "acceleration_units": "m/s^2",
        "angular_velocity_units": "rad/s",
        "class_counts": dict(class_counts),
        "subject_counts": dict(subject_counts),
        "hand_side_counts": dict(Counter(manifest["hand_side"])),
        "sensor_positions": sorted(
            set(manifest["sensor_positions"].astype(str))
        ),
        "address_mapping": sorted(
            set(manifest["address_mapping"].astype(str))
        ),
        "augmentation_copies_per_recording": args.augment_copies,
        "augmented_dataset_shape": list(augmented_x.shape),
        "augmented_rows": int(is_synthetic.sum()),
        "cross_subject_evaluation_available": len(subject_counts) >= 2,
        "warning": (
            "Existing recordings contain only one subject. This dataset supports "
            "pipeline development and within-subject diagnostics, not claims of "
            "cross-user gesture-recognition accuracy."
            if len(subject_counts) < 2
            else ""
        ),
        "augmentation_policy": (
            "Augmented rows retain source_index. Split by source_index or subject "
            "before training so copies of one recording never cross train/test."
        ),
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
