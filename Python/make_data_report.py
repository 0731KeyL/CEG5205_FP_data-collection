"""Generate presentation-ready evidence for the IMU data design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset_v002"
DEFAULT_OUTPUT = BASE_DIR / "data_review"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def split_composition_figure(dataset: Path, output: Path):
    rows = []
    for split in ("train", "validation", "test"):
        with np.load(dataset / f"{split}.npz", allow_pickle=False) as data:
            synthetic = data["is_synthetic"]
        rows.append({
            "split": split,
            "real": int((~synthetic).sum()),
            "synthetic": int(synthetic.sum()),
        })
    frame = pd.DataFrame(rows).set_index("split")
    axis = frame.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 4.5),
        color=["#2563eb", "#f59e0b"],
    )
    axis.set_title("Dataset split composition")
    axis.set_ylabel("Windows")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(["real", "synthetic"])
    for container in axis.containers:
        axis.bar_label(container, label_type="center", fontsize=9)
    axis.figure.tight_layout()
    path = output / "split_composition.png"
    axis.figure.savefig(path, dpi=200)
    plt.close(axis.figure)
    return path


def class_balance_figure(manifest: Path, output: Path):
    frame = pd.read_csv(manifest)
    counts = (
        frame[frame["split"] != "excluded"]
        .groupby(["gesture", "split"])
        .size()
        .unstack(fill_value=0)
    )
    axis = counts.plot(kind="bar", figsize=(10, 5))
    axis.set_title("Real trial balance by class and split")
    axis.set_ylabel("Real trials")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=35)
    axis.figure.tight_layout()
    path = output / "class_balance.png"
    axis.figure.savefig(path, dpi=200)
    plt.close(axis.figure)
    return path


def quality_figure(quality_report: Path, output: Path):
    frame = pd.read_csv(quality_report)
    colors = {
        "accepted": "#0f766e",
        "review": "#d97706",
        "rejected": "#b91c1c",
    }
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for grade, group in frame.groupby("quality_grade"):
        axis.scatter(
            group["sampling_rate_hz"],
            group["duration_s"],
            s=36,
            alpha=0.8,
            label=grade,
            color=colors.get(grade),
        )
    axis.axvspan(45.0, 55.0, color="#94a3b8", alpha=0.12)
    axis.set_xlabel("Measured sampling rate (Hz)")
    axis.set_ylabel("Trial duration (s)")
    axis.set_title("Real trial quality")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = output / "quality_overview.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def augmentation_figure(dataset: Path, output: Path):
    with np.load(dataset / "train.npz", allow_pickle=False) as data:
        windows = data["x"]
        masks = data["mask"]
        source_ids = data["source_id"].astype(str)
        synthetic = data["is_synthetic"]
        channel_names = data["channel_names"].astype(str)
        class_names = data["class_names"].astype(str)
        y = data["y"]

    groups = {}
    for index, source_id in enumerate(source_ids):
        groups.setdefault(source_id, []).append(index)
    candidates = [
        indexes for indexes in groups.values()
        if len(indexes) >= 4 and np.asarray(synthetic)[indexes].any()
    ]
    preferred_order = {"grasp": 0, "forward": 1, "backward": 2, "left": 3, "right": 4}
    candidates.sort(
        key=lambda indexes: preferred_order.get(
            str(class_names[int(y[indexes[0]])]),
            99,
        )
    )
    if not candidates:
        return None
    indexes = candidates[0]
    indexes = sorted(indexes, key=lambda index: bool(synthetic[index]))
    indexes = indexes[:4]

    channel = 0
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for axis, index in zip(axes, indexes):
        values = windows[index, :, channel]
        visible = masks[index, :, channel]
        axis.plot(values, color="#2563eb", linewidth=1.5)
        for step, is_visible in enumerate(visible):
            if not is_visible:
                axis.axvspan(step - 0.5, step + 0.5, color="#f97316", alpha=0.18)
        kind = "augmented" if synthetic[index] else "real"
        axis.set_ylabel("m/s²")
        axis.set_title(
            f"{kind} · class={class_names[int(y[index])]} · {channel_names[channel]}"
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Sample index at 50 Hz")
    fig.suptitle("Same-source augmentation and visibility mask")
    fig.tight_layout()
    path = output / "augmentation_example.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metainfo = json.loads(
        (args.dataset / "metainfo.json").read_text(encoding="utf-8")
    )
    manifest = args.dataset / "manifest.csv"
    quality = args.dataset / "quality_report.csv"

    split_composition_figure(args.dataset, args.output)
    class_balance_figure(manifest, args.output)
    quality_figure(quality, args.output)
    augmentation_figure(args.dataset, args.output)

    frame = pd.read_csv(quality)
    real_count = len(frame)
    accepted = int((frame["quality_grade"] == "accepted").sum())
    review = int((frame["quality_grade"] == "review").sum())
    rejected = int((frame["quality_grade"] == "rejected").sum())
    report = f"""# Data Design Review

## One-line result

The current data package contains **{real_count} unique real trials**, with
**{accepted} accepted**, **{review} review**, and **{rejected} rejected**.
Only training receives augmentation; validation and test remain real-only.

## Collection interface

![Collection UI](collection_ui_demo.png)

## Split composition

![Split composition](split_composition.png)

```text
train:      {metainfo['split_counts']['train']} total
validation: {metainfo['split_counts']['validation']} total
test:       {metainfo['split_counts']['test']} total
```

Synthetic counts:

```text
train:      {metainfo['synthetic_counts']['train']}
validation: {metainfo['synthetic_counts']['validation']}
test:       {metainfo['synthetic_counts']['test']}
```

## Class coverage

![Class balance](class_balance.png)

`grasp` and `release` remain limited by real-data availability. The pipeline
reports that limitation instead of using synthetic copies to hide it.

## Quality evidence

![Quality overview](quality_overview.png)

Hard checks cover sampling rate, duration, missing frames, sequence order, and
sensor saturation. Timestamp jitter and near-duplicate similarity are also
recorded.

## Augmentation and missing data

![Augmentation example](augmentation_example.png)

Augmented rows retain their `source_id`, and the model input includes a
visible mask. Validation and test never receive synthetic rows.

## Leakage prevention

- Split is frozen in `split_registry.json`.
- Every split stores stable `source_id`.
- Source sets are checked for overlap during dataset construction.
- Existing trial fingerprints cannot change silently.
- A new data release creates `dataset_vNNN`; old packages remain immutable.

## Task mapping

| Work area | Evidence |
|---|---|
| Task 1 | Collection UI, metadata, real-time 12-channel visualization, Demo mode |
| Task 3 | Quality grading, source-group split, augmentation, normalization, feature export |
| Task 4 | Model-agnostic NPZ contract, checksum loader, causal buffer and latency boundary |
| Robustness | Packet loss, sample-and-hold, one-IMU outage, visible masks, rotation, gain, gyro bias |

## Demo flow

1. Launch `imu_real_batch_recorder.py --demo`.
2. Start one batch and show real-time signal and quality status.
3. Run `build_dataset.py --output dataset_vNNN` for the next data release.
4. Load a split with `load_dataset.py`.
5. Compare the old and new model only after retraining on the new dataset version.

## Explicit limitations

- Only one subject is present.
- `grasp` and `release` have only two real trials each.
- Validation has no real `grasp/release` examples.
- The `0x68/0x69` to index/thumb mapping remains unresolved.
- Real hardware integration still needs a final serial test.
"""
    (args.output / "DATA_DESIGN_REVIEW.md").write_text(
        report,
        encoding="utf-8",
    )
    print(args.output / "DATA_DESIGN_REVIEW.md")


if __name__ == "__main__":
    main()
