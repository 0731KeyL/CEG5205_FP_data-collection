
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

REAL_DIR = BASE_DIR / "real_imu_data"
OUTPUT_DIR = BASE_DIR / "augmented_imu_data"

TARGET_COUNT = 100
RANDOM_SEED = 42

GESTURES = [
    "idle",
    "forward",
    "backward",
    "left",
    "right",
    "grasp",
    "release",
    "other"
]

CHANNEL_NAMES = [
    "ax1", "ay1", "az1",
    "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2",
    "gx2", "gy2", "gz2"
]

ACC_CHANNELS = [0, 1, 2, 6, 7, 8]
GYRO_CHANNELS = [3, 4, 5, 9, 10, 11]

rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def augment_signal(values):

    result = values.astype(float).copy()
    length = len(result)

    if length < 3:
        return result

    # Apply a small common time warp to all channels

    warp_strength = rng.uniform(-0.06, 0.06)

    positions = np.linspace(0, 1, length)

    warped_positions = (
        positions
        + warp_strength
        * np.sin(np.pi * positions)
    )

    warped_positions = np.clip(
        warped_positions, 0, 1
    )

    for channel in range(12):
        result[:, channel] = np.interp(
            warped_positions,
            positions,
            result[:, channel]
        )

    # Apply consistent scaling to each IMU

    for start in [0, 6]:

        acc_factor = rng.uniform(0.95, 1.05)
        gyro_factor = rng.uniform(0.95, 1.05)

        result[:, start:start + 3] *= acc_factor
        result[:, start + 3:start + 6] *= gyro_factor

    # Add low-amplitude sensor noise

    for channel in ACC_CHANNELS:
        noise = rng.normal(
            0, 15, length
        )
        result[:, channel] += noise

    for channel in GYRO_CHANNELS:
        noise = rng.normal(
            0, 3, length
        )
        result[:, channel] += noise

    return np.clip(
        np.rint(result),
        -32768,
        32767
    ).astype(np.int16)


summary = []

for gesture in GESTURES:

    source_dir = REAL_DIR / gesture
    target_dir = OUTPUT_DIR / gesture

    target_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    source_files = sorted(
        source_dir.glob("*.csv")
    ) if source_dir.exists() else []

    valid_sources = []

    for filepath in source_files:

        try:
            df = pd.read_csv(filepath)

            if len(df) < 3:
                continue

            if not all(
                name in df.columns
                for name in CHANNEL_NAMES
            ):
                continue

            valid_sources.append((filepath, df))

        except Exception as error:
            print("Skipping:", filepath.name, error)

    real_count = len(valid_sources)

    if real_count == 0:
        print(
            f"{gesture}: No valid real data. "
            "Cannot generate synthetic samples."
        )
        continue

    # Copy real recordings

    for filepath, df in valid_sources:

        output_file = target_dir / filepath.name

        real_df = df.copy()

        real_df["is_synthetic"] = False
        real_df["source_file"] = filepath.name

        real_df.to_csv(
            output_file,
            index=False
        )

    synthetic_needed = max(
        0,
        TARGET_COUNT - real_count
    )

    # Generate synthetic recordings

    for index in range(synthetic_needed):

        source_index = index % real_count

        filepath, source_df = valid_sources[source_index]

        synthetic_df = source_df.copy()

        original_values = synthetic_df[
            CHANNEL_NAMES
        ].to_numpy(dtype=float)

        synthetic_values = augment_signal(
            original_values
        )

        synthetic_df.loc[
            :, CHANNEL_NAMES
        ] = synthetic_values

        synthetic_df["gesture"] = gesture
        synthetic_df["is_synthetic"] = True
        synthetic_df["source_file"] = filepath.name

        # Create new synthetic sequence numbers

        if "seq" in synthetic_df.columns:
            synthetic_df["seq"] = np.arange(
                len(synthetic_df)
            )

        # Preserve the original sampling interval pattern

        if "timestamp_ms" in synthetic_df.columns:

            original_timestamps = source_df[
                "timestamp_ms"
            ].to_numpy(dtype=float)

            relative_timestamps = (
                original_timestamps
                - original_timestamps[0]
            )

            synthetic_df["timestamp_ms"] = (
                np.rint(relative_timestamps)
                .astype(np.int64)
            )

        if "recording_time_s" in synthetic_df.columns:

            synthetic_df["recording_time_s"] = (
                synthetic_df["timestamp_ms"] / 1000.0
            )

        output_name = (
            f"{gesture}_synthetic_"
            f"{index + 1:03d}.csv"
        )

        synthetic_df.to_csv(
            target_dir / output_name,
            index=False
        )

    total_count = real_count + synthetic_needed

    summary.append({
        "gesture": gesture,
        "real_count": real_count,
        "synthetic_count": synthetic_needed,
        "total_count": total_count
    })

    print(
        f"{gesture}: "
        f"Real={real_count}, "
        f"Synthetic={synthetic_needed}, "
        f"Total={total_count}"
    )

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_DIR / "augmentation_summary.csv",
    index=False
)

print()
print("Data augmentation completed.")
print("Output directory:", OUTPUT_DIR)
