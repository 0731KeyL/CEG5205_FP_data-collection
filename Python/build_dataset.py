"""Build a leakage-safe, versioned, model-ready IMU dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from imu_augmentations import (
    augment_window,
    load_augmentation_config,
)
from imu_processing import (
    assign_splits,
    event_aware_window,
    extract_window_features,
    feature_names,
    near_duplicate_report,
    resample_to_uniform_time,
)
from imu_quality import assess_trial
from project_config import load_project_config


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "real_imu_data"
DEFAULT_OUTPUT = BASE_DIR / "dataset_v002"

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
        description="Build a split, quality-checked, model-agnostic IMU dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-rate", type=float, default=50.0)
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--augment-copies", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-config",
        type=Path,
        default=BASE_DIR / "augmentation_config.json",
    )
    parser.add_argument(
        "--split-registry",
        type=Path,
        default=BASE_DIR / "split_registry.json",
        help="Persistent map from stable trial IDs to frozen data splits.",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Include review-grade real trials in train/validation/test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing dataset directory. Use only for local rebuilds.",
    )
    return parser.parse_args()


def first_text(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame:
        return default
    values = frame[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    return "|".join(sorted(values.unique())) if len(values) else default


def first_float(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else None


def convert_to_si(raw: np.ndarray) -> np.ndarray:
    values = raw.astype(np.float64, copy=True)
    values[:, ACC_CHANNELS] *= GRAVITY_M_S2 / ACCEL_LSB_PER_G
    values[:, GYRO_CHANNELS] *= DEG_TO_RAD / GYRO_LSB_PER_DEG_S
    return values.astype(np.float32)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_quality_rows(frame: pd.DataFrame, raw: np.ndarray) -> list[list]:
    timestamp = frame.get(
        "timestamp_ms",
        pd.Series(np.arange(len(frame), dtype=np.float64) * 20.0),
    ).to_numpy(dtype=np.float64)
    recording_time = frame.get(
        "recording_time_s",
        pd.Series(np.arange(len(frame), dtype=np.float64) * 0.02),
    ).to_numpy(dtype=np.float64)
    sequence = frame.get(
        "seq",
        pd.Series(np.arange(len(frame), dtype=np.int64)),
    ).to_numpy(dtype=np.int64)
    return [
        [int(sequence[i]), float(timestamp[i]), float(recording_time[i]), *raw[i]]
        for i in range(len(frame))
    ]


def union_find_group(
    indices: list[int],
    parent: dict[int, int],
) -> None:
    if not indices:
        return
    root = indices[0]
    for index in indices[1:]:
        parent[index] = root


def find(parent: dict[int, int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def apply_split_registry(
    records: list[dict],
    eligible_indices: list[int],
    registry_path: Path,
    split_policy: str,
) -> None:
    registry = {"schema_version": 1, "split_policy": split_policy, "trials": {}}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        old_policy = registry.get("split_policy")
        if old_policy and old_policy != split_policy:
            raise ValueError(
                f"Split policy changed from {old_policy!r} to {split_policy!r}. "
                "Use a new split registry/dataset version."
            )
    frozen = registry.setdefault("trials", {})

    eligible = set(eligible_indices)
    for index, record in enumerate(records):
        if index not in eligible:
            continue
        key = record["source_id"]
        existing = frozen.get(key)
        if existing is None:
            frozen[key] = {
                "split": record["split"],
                "sha256": record["sha256"],
                "subject_id": record["subject_id"],
                "gesture": record["gesture"],
                "relative_path": record["relative_path"],
            }
        elif existing["sha256"] != record["sha256"]:
            raise ValueError(
                f"Raw trial changed without a new source ID: {key}. "
                "Create a new version instead of mutating existing raw data."
            )
        else:
            record["split"] = existing["split"]

    # A newly discovered near-duplicate must stay with the source it duplicates.
    for index in eligible_indices:
        record = records[index]
        if not record["near_duplicate"] or not record["duplicate_of"]:
            continue
        target_index = int(record["duplicate_of"])
        target = records[target_index]
        record["split"] = target["split"]
        frozen[record["source_id"]]["split"] = target["split"]

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_leakage_groups(
    records: list[dict],
    windows: np.ndarray,
    labels: np.ndarray,
    similarity_threshold: float,
    rms_threshold: float,
) -> pd.DataFrame:
    duplicate_rows = near_duplicate_report(windows, labels)
    parent = {index: index for index in range(len(records))}

    hashes = defaultdict(list)
    for index, record in enumerate(records):
        hashes[record["sha256"]].append(index)
    for group in hashes.values():
        union_find_group(group, parent)

    for index, report in enumerate(duplicate_rows):
        if not report["duplicate_of"]:
            continue
        if (
            report["duplicate_similarity"] >= similarity_threshold
            and report["duplicate_rms_ratio"] <= rms_threshold
        ):
            other = int(report["duplicate_of"])
            parent[find(parent, index)] = find(parent, other)
            report["near_duplicate"] = True
        else:
            report["near_duplicate"] = False

    roots = {index: find(parent, index) for index in range(len(records))}
    for index, record in enumerate(records):
        record["leakage_group"] = f"group_{roots[index]:04d}"
        report = duplicate_rows[index]
        record["duplicate_of"] = report["duplicate_of"]
        record["duplicate_similarity"] = report["duplicate_similarity"]
        record["duplicate_rms_ratio"] = report["duplicate_rms_ratio"]
        record["near_duplicate"] = report["near_duplicate"]

    return pd.DataFrame(duplicate_rows)


def normalization_stats(
    windows: np.ndarray,
    masks: np.ndarray,
    features: np.ndarray,
) -> dict:
    all_values = np.concatenate(windows, axis=0)
    all_masks = np.concatenate(masks, axis=0)
    channel_mean = np.zeros(all_values.shape[1], dtype=np.float32)
    channel_std = np.ones(all_values.shape[1], dtype=np.float32)
    for channel in range(all_values.shape[1]):
        selected = all_values[:, channel][all_masks[:, channel]]
        if len(selected):
            channel_mean[channel] = float(selected.mean())
            channel_std[channel] = float(selected.std())
    channel_std[channel_std < 1e-8] = 1.0
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    return {
        "channel_mean": channel_mean.astype(np.float32),
        "channel_std": channel_std.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
    }


def save_split(
    output: Path,
    split: str,
    windows: list[np.ndarray],
    masks: list[np.ndarray],
    labels: list[int],
    sources: list[int],
    source_ids: list[str],
    synthetic: list[bool],
    features: list[np.ndarray],
    feature_name_list: list[str],
    dataset_id: str,
) -> None:
    x = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    mask = np.stack(masks).astype(bool)
    source = np.asarray(sources, dtype=np.int64)
    source_id = np.asarray(source_ids, dtype=str)
    synthetic_array = np.asarray(synthetic, dtype=bool)
    feature_array = np.stack(features).astype(np.float32)
    length = mask.any(axis=2).sum(axis=1).astype(np.int16)

    np.save(output / f"X_{split}.npy", x)
    np.save(output / f"y_{split}.npy", y)
    np.save(output / f"mask_{split}.npy", mask)
    np.save(output / f"source_{split}.npy", source)
    np.save(output / f"source_id_{split}.npy", source_id)
    np.save(output / f"is_synthetic_{split}.npy", synthetic_array)
    np.save(output / f"features_{split}.npy", feature_array)
    np.save(output / f"length_{split}.npy", length)
    np.savez_compressed(
        output / f"{split}.npz",
        x=x,
        y=y,
        mask=mask,
        length=length,
        source_index=source,
        source_id=source_id,
        is_synthetic=synthetic_array,
        features=feature_array,
        class_names=np.asarray(CLASS_NAMES),
        channel_names=np.asarray(CHANNELS),
        feature_names=np.asarray(feature_name_list),
        dataset_id=np.asarray(dataset_id),
    )


def main():
    args = parse_args()
    if args.augment_copies < 0:
        raise ValueError("--augment-copies cannot be negative")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise FileExistsError(
            f"Dataset directory is not empty: {args.output}. "
            "Create a new version directory or pass --force for a local rebuild."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_version = args.output.name
    dataset_id = (
        f"{dataset_version}_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    augmentation_config = load_augmentation_config(args.augmentation_config)

    records = []
    windows = []
    window_masks = []
    channel_names = CHANNELS
    features_name_list = feature_names(CHANNELS)

    for label_index, class_name in enumerate(CLASS_NAMES):
        for path in sorted((args.source / class_name).glob("*.csv")):
            frame = pd.read_csv(path)
            missing = [name for name in CHANNELS if name not in frame]
            if missing:
                raise ValueError(f"{path.name}: missing channels {missing}")

            raw = frame[CHANNELS].to_numpy(dtype=np.float64)
            timestamps = frame.get(
                "timestamp_ms",
                pd.Series(np.arange(len(frame), dtype=np.float64) * 20.0),
            ).to_numpy(dtype=np.float64)
            physical = convert_to_si(raw)
            uniform, uniform_time, resampled_duration = resample_to_uniform_time(
                physical,
                timestamps,
                args.sampling_rate,
            )
            window, visible_mask, event_info = event_aware_window(
                uniform,
                args.window_length,
            )

            subject_id = first_text(frame, "subject_id", "unknown")
            legacy = LEGACY_SUBJECT_METADATA.get(subject_id, {})
            target_duration = first_float(frame, "target_duration_s")
            quality = assess_trial(
                build_quality_rows(frame, raw),
                expected_duration_s=target_duration,
                target_rate_hz=args.sampling_rate,
            )

            records.append({
                "source_index": len(records),
                "source_id": f"{subject_id}|{path.resolve().relative_to(args.source.resolve()).as_posix()}",
                "file": path.name,
                "relative_path": str(path.resolve().relative_to(args.source.resolve())),
                "sha256": file_sha256(path),
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
                "target_duration_s": (
                    "" if target_duration is None else target_duration
                ),
                "duration_s": resampled_duration,
                "quality_grade": quality["grade"],
                "quality_flags": quality["flags"],
                "sampling_rate_hz": quality["rate_hz"],
                "missing_frames": quality["missing"],
                "saturated_values": quality["saturated"],
                "timestamp_jitter_ms": quality["jitter_ms"],
                "resampled_samples": len(uniform),
                **event_info,
            })
            windows.append(window)
            window_masks.append(visible_mask)

    if not windows:
        raise RuntimeError(f"No recordings found below {args.source}")

    x = np.stack(windows).astype(np.float32)
    x_masks = np.stack(window_masks)
    y = np.asarray([record["label"] for record in records], dtype=np.int64)
    duplicate_report = make_leakage_groups(
        records,
        x,
        y,
        similarity_threshold=0.995,
        rms_threshold=0.05,
    )
    duplicate_report.insert(0, "source_index", np.arange(len(records)))
    duplicate_report.to_csv(args.output / "duplicate_report.csv", index=False)

    for record in records:
        if record["near_duplicate"]:
            if record["quality_grade"] == "accepted":
                record["quality_grade"] = "review"
            flags = set(filter(None, str(record["quality_flags"]).split("|")))
            flags.add("near_duplicate")
            record["quality_flags"] = "|".join(sorted(flags))

    eligible_indices = [
        index for index, record in enumerate(records)
        if record["quality_grade"] == "accepted"
        or (args.include_review and record["quality_grade"] == "review")
    ]
    if not eligible_indices:
        raise RuntimeError(
            "No accepted recordings. Use --include-review to include review rows."
        )

    # Record quality before assigning splits so excluded rows remain auditable.
    split_policy = assign_splits(records, args.seed)
    apply_split_registry(
        records,
        eligible_indices,
        args.split_registry,
        split_policy,
    )
    for index in range(len(records)):
        if index not in eligible_indices:
            records[index]["split"] = "excluded"

    split_arrays = {
        "train": {"x": [], "mask": [], "y": [], "source": [], "source_id": [], "synthetic": [], "features": []},
        "validation": {"x": [], "mask": [], "y": [], "source": [], "source_id": [], "synthetic": [], "features": []},
        "test": {"x": [], "mask": [], "y": [], "source": [], "source_id": [], "synthetic": [], "features": []},
    }
    augmentation_rows = []

    for index in eligible_indices:
        record = records[index]
        split = record["split"]
        window = x[index]
        mask = x_masks[index]
        split_arrays[split]["x"].append(window)
        split_arrays[split]["mask"].append(mask)
        split_arrays[split]["y"].append(int(record["label"]))
        split_arrays[split]["source"].append(index)
        split_arrays[split]["source_id"].append(record["source_id"])
        split_arrays[split]["synthetic"].append(False)
        split_arrays[split]["features"].append(
            extract_window_features(window, mask)
        )
        augmentation_rows.append({
            "source_index": index,
            "source_file": record["file"],
            "gesture": record["gesture"],
            "split": split,
            "is_synthetic": False,
            "augmentation": "none",
        })

        if split != "train":
            continue
        for copy_index in range(args.augment_copies):
            augmented, augmented_mask, info = augment_window(
                window,
                target_length=args.window_length,
                seed=args.seed + index * 1009 + copy_index,
                config=augmentation_config,
            )
            split_arrays[split]["x"].append(augmented)
            split_arrays[split]["mask"].append(augmented_mask)
            split_arrays[split]["y"].append(int(record["label"]))
            split_arrays[split]["source"].append(index)
            split_arrays[split]["source_id"].append(record["source_id"])
            split_arrays[split]["synthetic"].append(True)
            split_arrays[split]["features"].append(
                extract_window_features(augmented, augmented_mask)
            )
            augmentation_rows.append({
                "source_index": index,
                "source_file": record["file"],
                "gesture": record["gesture"],
                "split": split,
                "is_synthetic": True,
                "augmentation": json.dumps(info.__dict__),
            })

    for split in ("train", "validation", "test"):
        part = split_arrays[split]
        if not part["x"]:
            raise RuntimeError(f"Split {split!r} is empty")
        if split != "train" and any(part["synthetic"]):
            raise RuntimeError(
                f"Synthetic rows are not allowed in split {split!r}"
            )
        save_split(
            args.output,
            split,
            part["x"],
            part["mask"],
            part["y"],
            part["source"],
            part["source_id"],
            part["synthetic"],
            part["features"],
            features_name_list,
            dataset_id,
        )

    source_sets = {
        split: set(split_arrays[split]["source"])
        for split in ("train", "validation", "test")
    }
    if (
        source_sets["train"] & source_sets["validation"]
        or source_sets["train"] & source_sets["test"]
        or source_sets["validation"] & source_sets["test"]
    ):
        raise RuntimeError("Source leakage detected between data splits")

    train_part = split_arrays["train"]
    normalization = normalization_stats(
        np.stack(train_part["x"]),
        np.stack(train_part["mask"]),
        np.stack(train_part["features"]),
    )
    np.savez(
        args.output / "normalization.npz",
        **normalization,
    )
    pd.DataFrame(augmentation_rows).to_csv(
        args.output / "augmented_manifest.csv",
        index=False,
    )
    manifest = pd.DataFrame(records)
    manifest.to_csv(args.output / "manifest.csv", index=False)
    manifest[[
        "source_index",
        "file",
        "gesture",
        "subject_id",
        "hand_side",
        "quality_grade",
        "quality_flags",
        "sampling_rate_hz",
        "duration_s",
        "missing_frames",
        "saturated_values",
        "timestamp_jitter_ms",
        "duplicate_of",
        "duplicate_similarity",
        "duplicate_rms_ratio",
        "near_duplicate",
        "split",
    ]].to_csv(args.output / "quality_report.csv", index=False)
    split_snapshot = {
        record["source_id"]: {
            "split": record["split"],
            "sha256": record["sha256"],
            "subject_id": record["subject_id"],
            "gesture": record["gesture"],
            "relative_path": record["relative_path"],
        }
        for record in records
        if record["split"] != "excluded"
    }
    with (args.output / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(split_snapshot, file, ensure_ascii=False, indent=2)

    metainfo = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "raw_recording_source": "upstream CEG5205 final-project data collector",
        "class_names": CLASS_NAMES,
        "channel_names": channel_names,
        "feature_names": features_name_list,
        "acceleration_unit": "m/s^2",
        "angular_velocity_unit": "rad/s",
        "sampling_rate_hz": args.sampling_rate,
        "window_length": args.window_length,
        "split_policy": split_policy,
        "split_registry_sha256": (
            file_sha256(args.split_registry)
            if args.split_registry.exists()
            else ""
        ),
        "augmentation_policy": "training-only",
        "augmentation_copies_per_train_recording": args.augment_copies,
        "augmentation_config_sha256": file_sha256(args.augmentation_config),
        "normalization_source": "visible training values only",
        "quality_policy": {
            "grades": ["accepted", "review", "rejected"],
            "include_review": args.include_review,
            "rate_hz_range": [45.0, 55.0],
            "max_missing_rate": 0.01,
            "saturation_limit": 32000,
            "near_duplicate_threshold": {
                "similarity": 0.995,
                "relative_rms": 0.05,
            },
        },
        "class_counts": dict(Counter(manifest["gesture"])),
        "split_counts": {
            split: len(split_arrays[split]["x"])
            for split in ("train", "validation", "test")
        },
        "real_counts": {
            split: int((~np.asarray(split_arrays[split]["synthetic"])).sum())
            for split in ("train", "validation", "test")
        },
        "synthetic_counts": {
            split: int(np.asarray(split_arrays[split]["synthetic"]).sum())
            for split in ("train", "validation", "test")
        },
        "quality_counts": dict(Counter(manifest["quality_grade"])),
        "subject_counts": dict(Counter(manifest["subject_id"])),
        "hand_side_counts": dict(Counter(manifest["hand_side"])),
        "synthetic_train_rows": int(
            np.asarray(train_part["synthetic"], dtype=bool).sum()
        ),
        "sensor_positions": sorted(
            set(manifest["sensor_positions"].astype(str))
        ),
        "address_mapping": sorted(
            set(manifest["address_mapping"].astype(str))
        ),
        "cross_subject_evaluation_available": (
            manifest["subject_id"].nunique() >= 2
        ),
    }
    with (args.output / "metainfo.json").open("w", encoding="utf-8") as file:
        json.dump(metainfo, file, ensure_ascii=False, indent=2)
    with (args.output / "feature_names.json").open("w", encoding="utf-8") as file:
        json.dump(features_name_list, file, ensure_ascii=False, indent=2)
    (args.output / "augmentation_config.json").write_text(
        json.dumps(augmentation_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output / "label_map.json").open("w", encoding="utf-8") as file:
        json.dump(
            {name: index for index, name in enumerate(CLASS_NAMES)},
            file,
            ensure_ascii=False,
            indent=2,
        )

    split_class_counts = {}
    for split in ("train", "validation", "test"):
        split_class_counts[split] = np.bincount(
            np.asarray(split_arrays[split]["y"], dtype=np.int64),
            minlength=len(CLASS_NAMES),
        ).tolist()
    readme = f"""# IMU Classification Dataset `{dataset_version}`

This directory is the frozen data interface for a downstream classifier.
Do not edit these files in place and do not regenerate augmentation inside a
model-training repository.

When new real recordings are added, create the next numbered output directory;
do not overwrite this version. Retrain and version the
downstream model for every dataset version.

## Canonical files

Load one of:

```python
import numpy as np
train = np.load("train.npz")
x = train["x"]              # float32, shape {np.stack(split_arrays['train']['x']).shape}
features = train["features"]  # float32, shape {np.stack(split_arrays['train']['features']).shape}
y = train["y"]              # int64 class index
mask = train["mask"]        # bool, same shape as x
length = train["length"]    # valid time steps
source_id = train["source_id"]  # stable raw-trial identity
```

Use `features_<split>.npy` and `normalization.npz` for feature-based models, or
`X_<split>.npy` and `mask_<split>.npy` for a sequence model.

## Split policy

`{split_policy}`

Validation and test contain real recordings only. Synthetic rows occur only
in `train`.

## Class counts

```text
train:      {split_class_counts['train']}
validation: {split_class_counts['validation']}
test:       {split_class_counts['test']}
```

## File integrity

- Full definitions are in `metainfo.json`, `manifest.csv`, and
  `quality_report.csv`.
- Verify the package with `checksums.sha256`.

## Important limitation

The current package contains one subject only. Results obtained with this
split are within-subject diagnostics and do not demonstrate cross-user
generalization.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")

    checksum_files = sorted(
        path for path in args.output.glob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    with (args.output / "checksums.sha256").open("w", encoding="utf-8") as file:
        for path in checksum_files:
            file.write(f"{file_sha256(path)}  {path.name}\n")

    print(json.dumps(metainfo, ensure_ascii=False, indent=2))
    for split in ("train", "validation", "test"):
        print(
            split,
            "real", int((~np.asarray(split_arrays[split]["synthetic"])).sum()),
            "synthetic", int(np.asarray(split_arrays[split]["synthetic"]).sum()),
        )
    print("Output:", args.output)


if __name__ == "__main__":
    main()
