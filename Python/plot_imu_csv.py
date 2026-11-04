
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

DATA_DIRS = [
    BASE_DIR / "real_imu_data",
    BASE_DIR / "augmented_imu_data"
]

OUTPUT_DIR = BASE_DIR / "imu_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_NAMES = [
    "ax1", "ay1", "az1",
    "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2",
    "gx2", "gy2", "gz2"
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4c78a8",
    "#f58518"
]

ACC_SCALE = 8192.0
GYRO_SCALE = 65.5

ACC_DISPLAY_SCALE = 10.0
GYRO_DISPLAY_SCALE = 0.08


def find_csv(filename):

    path = Path(filename)

    if path.is_file():
        return path

    matches = []

    for directory in DATA_DIRS:
        if directory.exists():
            matches.extend(
                directory.rglob(path.name)
            )

    if not matches:
        raise FileNotFoundError(
            f"Cannot find CSV file: {filename}"
        )

    if len(matches) > 1:
        print("Multiple matching files found:")
        for match in matches:
            print(match)

        raise ValueError(
            "Please provide the full CSV path."
        )

    return matches[0]


def plot_imu_csv(filename, show=True):

    filepath = find_csv(filename)

    df = pd.read_csv(filepath)

    missing = [
        name for name in CHANNEL_NAMES
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing channels: {missing}"
        )

    if "timestamp_ms" in df.columns:

        timestamps = df[
            "timestamp_ms"
        ].to_numpy(dtype=float)

        time_axis = (
            timestamps - timestamps[0]
        ) / 1000.0

    elif "recording_time_s" in df.columns:

        timestamps = df[
            "recording_time_s"
        ].to_numpy(dtype=float)

        time_axis = (
            timestamps - timestamps[0]
        )

    else:

        time_axis = (
            np.arange(len(df)) / 50.0
        )

    values = df[
        CHANNEL_NAMES
    ].to_numpy(dtype=float)

    scaled = values.copy()

    for index in [0, 1, 2, 6, 7, 8]:

        scaled[:, index] = (
            scaled[:, index] / ACC_SCALE
        ) * ACC_DISPLAY_SCALE

    for index in [3, 4, 5, 9, 10, 11]:

        scaled[:, index] = (
            scaled[:, index] / GYRO_SCALE
        ) * GYRO_DISPLAY_SCALE

    if "gesture" in df.columns:
        gesture = str(df["gesture"].iloc[0])
    else:
        gesture = filepath.parent.name

    fig, ax = plt.subplots(
        figsize=(13, 6),
        dpi=150
    )

    for i, name in enumerate(CHANNEL_NAMES):

        ax.plot(
            time_axis,
            scaled[:, i],
            color=COLORS[i],
            linewidth=1.3,
            label=name
        )

    ax.set_title(
        f"IMU data (12 channels), gesture = {gesture}",
        fontsize=14
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scaled Signal Value")

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="upper right",
        ncol=4,
        fontsize=8,
        frameon=True
    )

    ax.set_xlim(
        time_axis[0],
        time_axis[-1]
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / f"{filepath.stem}.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Input:", filepath)
    print("Samples:", len(df))
    print("Gesture:", gesture)
    print("Figure saved:", output_path)

    if show:
        plt.show()

    plt.close(fig)

    return output_path


if __name__ == "__main__":

    FILE_NAME = "grasp_synthetic_015.csv"

    plot_imu_csv(FILE_NAME)
