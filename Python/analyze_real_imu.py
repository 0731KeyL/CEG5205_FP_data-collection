
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "real_imu_data"
RESULT_DIR = BASE_DIR / "real_imu_analysis"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_RATE = 50
EXPECTED_INTERVAL_MS = 1000 / EXPECTED_RATE

# Find recorded CSV files

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}"
    )

print("Found", len(csv_files), "CSV files")

results = []

# Analyze each recording

for filepath in csv_files:

    df = pd.read_csv(filepath)

    required_columns = [
        "seq",
        "timestamp_ms",
        "recording_time_s"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):
        print("Skipping:", filepath.name)
        continue

    if len(df) < 2:
        print("Insufficient samples:", filepath.name)
        continue

    seq = df["seq"].to_numpy(dtype=np.int64)

    timestamps = df[
        "timestamp_ms"
    ].to_numpy(dtype=float)

    recording_time = df[
        "recording_time_s"
    ].to_numpy(dtype=float)

    seq_diff = np.diff(seq)
    time_diff = np.diff(timestamps)

    arrival_diff = (
        np.diff(recording_time) * 1000
    )

    missing_frames = int(
        np.maximum(seq_diff - 1, 0).sum()
    )

    duplicate_frames = int(
        np.sum(seq_diff == 0)
    )

    out_of_order = int(
        np.sum(seq_diff < 0)
    )

    duration = (
        timestamps[-1] - timestamps[0]
    ) / 1000

    if duration > 0:
        actual_rate = (
            len(df) - 1
        ) / duration
    else:
        actual_rate = np.nan

    mean_interval = np.mean(time_diff)
    jitter = np.std(time_diff)

    mean_arrival = np.mean(arrival_diff)
    arrival_jitter = np.std(arrival_diff)

    expected_frames = (
        len(df) + missing_frames
    )

    loss_rate = (
        missing_frames / expected_frames * 100
        if expected_frames > 0
        else 0
    )

    results.append({
        "filename": filepath.name,
        "samples": len(df),
        "sampling_rate_hz": actual_rate,
        "mean_interval_ms": mean_interval,
        "sampling_jitter_ms": jitter,
        "arrival_interval_ms": mean_arrival,
        "arrival_jitter_ms": arrival_jitter,
        "missing_frames": missing_frames,
        "duplicate_frames": duplicate_frames,
        "out_of_order_frames": out_of_order,
        "missing_rate_percent": loss_rate
    })

    print()
    print("File:", filepath.name)
    print("Samples:", len(df))
    print("Sampling rate:", round(actual_rate, 2), "Hz")
    print("Mean interval:", round(mean_interval, 3), "ms")
    print("Sampling jitter:", round(jitter, 3), "ms")
    print("Arrival jitter:", round(arrival_jitter, 3), "ms")
    print("Missing frames:", missing_frames)
    print("Missing rate:", round(loss_rate, 3), "%")

    # Plot sampling intervals

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        time_diff,
        linewidth=1.2,
        label="Actual sampling interval"
    )

    ax.axhline(
        EXPECTED_INTERVAL_MS,
        linestyle="--",
        label="Target: 20 ms (50 Hz)"
    )

    ax.axhline(
        mean_interval,
        linestyle=":",
        label=f"Measured mean: {mean_interval:.2f} ms"
    )

    ax.set_title(
        f"Real IMU Sampling Interval - {filepath.stem}"
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sampling Interval (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / f"{filepath.stem}_interval.png",
        dpi=200
    )

    plt.close(fig)

# Save analysis summary

if not results:
    raise RuntimeError("No valid recordings found.")

summary = pd.DataFrame(results)

summary.to_csv(
    RESULT_DIR / "real_imu_summary.csv",
    index=False
)

print()
print("Analysis completed.")
print("Results saved to:", RESULT_DIR)
