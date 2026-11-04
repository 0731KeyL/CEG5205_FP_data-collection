"""Honest baseline for the existing real two-IMU gesture recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from imu_augmentations import resample_sequence
from project_config import load_project_config


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "real_imu_data"
DEFAULT_OUT_DIR = BASE_DIR / "real_imu_analysis" / "baseline"
CHANNELS = [
    "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2", "gx2", "gy2", "gz2",
]
GESTURES = load_project_config()["gestures"]
ACC_CHANNELS = [0, 1, 2, 6, 7, 8]
GYRO_CHANNELS = [3, 4, 5, 9, 10, 11]
ACC_SCALE = 8192.0
GYRO_SCALE = 65.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a real-data-only IMU gesture baseline."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_real_dataset(data_dir: Path, window_length: int):
    rows = []
    feature_rows = []
    labels = []
    subjects = []

    for gesture_index, gesture in enumerate(GESTURES):
        gesture_dir = data_dir / gesture
        for path in sorted(gesture_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            missing = [name for name in CHANNELS if name not in frame.columns]
            if missing:
                raise ValueError(f"{path.name}: missing channels {missing}")

            values = frame[CHANNELS].to_numpy(dtype=np.float32)
            values[:, ACC_CHANNELS] /= ACC_SCALE
            values[:, GYRO_CHANNELS] /= GYRO_SCALE
            values = resample_sequence(values, window_length)

            subject_id = (
                str(frame["subject_id"].iloc[0])
                if "subject_id" in frame and frame["subject_id"].notna().any()
                else "unknown"
            )
            labels.append(gesture_index)
            subjects.append(subject_id)
            feature_rows.append(extract_features(values))
            rows.append({
                "file": path.name,
                "path": str(path.relative_to(data_dir)),
                "gesture": gesture,
                "subject_id": subject_id,
                "samples": len(frame),
            })

    if not feature_rows:
        raise RuntimeError(f"No CSV recordings found below {data_dir}")
    return (
        rows,
        np.stack(feature_rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects),
    )


def extract_features(values: np.ndarray) -> np.ndarray:
    """Extract scale-aware statistics from a (time, channel) window."""
    features = [
        values.mean(axis=0),
        values.std(axis=0),
        values.min(axis=0),
        values.max(axis=0),
        np.sqrt(np.mean(values ** 2, axis=0)),
        np.mean(np.abs(np.diff(values, axis=0)), axis=0),
    ]

    # Vector magnitudes retain useful information without depending on mounting axes.
    for sensor_start in (0, 6):
        accel = values[:, sensor_start:sensor_start + 3]
        gyro = values[:, sensor_start + 3:sensor_start + 6]
        accel_magnitude = np.linalg.norm(accel, axis=1)
        gyro_magnitude = np.linalg.norm(gyro, axis=1)
        for magnitude in (accel_magnitude, gyro_magnitude):
            features.extend([
                magnitude.mean(keepdims=True),
                magnitude.std(keepdims=True),
                magnitude.max(keepdims=True),
            ])
    return np.concatenate(features).astype(np.float32)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, features, labels, subjects = load_real_dataset(
        args.data_dir,
        args.window_length,
    )

    unique_subjects = np.unique(subjects)
    if len(unique_subjects) >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=min(5, len(unique_subjects)),
            shuffle=True,
            random_state=args.seed,
        )
        split = list(splitter.split(features, labels, groups=subjects))
        evaluation_name = "grouped cross-validation by subject"
    else:
        split = list(LeaveOneOut().split(features))
        evaluation_name = (
            "leave-one-file-out within one subject; "
            "diagnostic only, not cross-subject accuracy"
        )

    base_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10.0,
            class_weight="balanced",
            decision_function_shape="ovr",
        ),
    )

    predictions = cross_val_predict(
        base_model,
        features,
        labels,
        cv=split,
        n_jobs=-1,
    )

    target_names = GESTURES
    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(target_names)),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "evaluation": evaluation_name,
        "subjects": sorted(map(str, unique_subjects)),
        "recordings": len(records),
        "window_length": args.window_length,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "limitations": (
            "Only real recordings are evaluated. Synthetic files are excluded. "
            "With one subject, this cannot establish cross-user generalization."
        ),
    }

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"summary": metrics, "per_class": report},
            file,
            ensure_ascii=False,
            indent=2,
        )

    prediction_frame = pd.DataFrame(records)
    prediction_frame["true_label"] = [GESTURES[i] for i in labels]
    prediction_frame["predicted_label"] = [
        GESTURES[i] for i in predictions
    ]
    prediction_frame["correct"] = labels == predictions
    prediction_frame.to_csv(
        args.out_dir / "predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(len(target_names)),
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set(
        xticks=np.arange(len(target_names)),
        yticks=np.arange(len(target_names)),
        xticklabels=target_names,
        yticklabels=target_names,
        xlabel="Predicted",
        ylabel="True",
        title="Real-data baseline confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Output:", args.out_dir)


if __name__ == "__main__":
    main()
