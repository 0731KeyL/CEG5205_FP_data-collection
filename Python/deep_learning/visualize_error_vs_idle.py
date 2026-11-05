"""Compare the misclassified release trial with an idle trial from the test split."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from gesture_mlp import GestureMLP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from Python.load_dataset import file_sha256, load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "runs/gesture_mlp_20261105_192725_061193.pt")
    parser.add_argument("--dataset", type=Path, default=HERE.parent / "dataset_v003")
    parser.add_argument("--idle-index", type=int, help="Zero-based test row; default is the first true idle row")
    parser.add_argument("--output-dir", type=Path, default=HERE / "visualizations/error_vs_idle")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if file_sha256(args.dataset / "checksums.sha256") != checkpoint["dataset_checksum"]:
        raise ValueError("Dataset checksum differs from the checkpoint.")
    test = load_split(args.dataset, "test")
    names = checkpoint["class_names"]
    if names != test["class_names"].tolist():
        raise ValueError("Class mapping differs from the checkpoint.")
    model = GestureMLP(args.dataset / "normalization.npz", num_classes=len(names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(test["features"]).float()).softmax(1).numpy()
    predictions = probabilities.argmax(1)
    filename = "subject01_release_20261104_095157_trial001.csv"
    indices = [i for i, source in enumerate(test["source_id"]) if str(source).endswith("/" + filename)]
    if len(indices) != 1:
        raise ValueError("Expected the identified release trial in the test split.")
    error = indices[0]
    if predictions[error] == test["y"][error]:
        raise ValueError("This checkpoint does not misclassify the selected trial.")
    idle_rows = np.flatnonzero(test["y"] == names.index("idle"))
    idle = int(idle_rows[0]) if args.idle_index is None else args.idle_index
    if idle not in idle_rows:
        raise ValueError("--idle-index must identify a true idle test row.")
    selected = [error, idle]
    colors = ["tab:red", "tab:blue"]
    captions = [f"true={names[int(test['y'][i])]}, predicted={names[int(predictions[i])]}" for i in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    time = np.arange(test["x"].shape[1]) / float(json.loads((args.dataset / "metainfo.json").read_text())["sampling_rate_hz"])
    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    for c, ax in enumerate(axes.flat):
        for i, color, caption in zip(selected, colors, captions):
            signal = np.where(test["mask"][i, :, c], test["x"][i, :, c], np.nan)
            ax.plot(time, signal, color=color, label=caption)
        ax.set_title(str(test["channel_names"][c]))
        ax.set_ylabel("m/s²" if c % 6 < 3 else "rad/s")
        ax.set_xlabel("Window time (s)")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
    fig.suptitle("Model input windows: release error versus idle (masked values omitted)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "signals.png", dpi=160)

    mean = model.feature_mean.numpy()
    std = model.feature_std.numpy()
    z = (test["features"][selected] - mean) / std
    feature_names = test["feature_names"]
    top = np.argsort(np.abs(z[0] - z[1]))[-15:]
    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 13))
    for row, color, caption in zip(z, colors, captions):
        axes2[0].plot(np.arange(84), row, color=color, label=caption)
    axes2[0].set(xlabel="Feature index", ylabel="Standardized feature", title="All 84 model input features")
    axes2[0].legend()
    positions = np.arange(len(top))
    axes2[1].barh(positions - 0.2, z[0, top], height=0.4, color=colors[0])
    axes2[1].barh(positions + 0.2, z[1, top], height=0.4, color=colors[1])
    axes2[1].set_yticks(positions, labels=feature_names[top])
    axes2[1].set(xlabel="Standardized feature", title="15 largest absolute differences between standardized features")
    for offset, i, color, caption in zip((-0.2, 0.2), selected, colors, captions):
        axes2[2].bar(np.arange(len(names)) + offset, probabilities[i], width=0.4, color=color, label=caption)
    axes2[2].set_xticks(np.arange(len(names)), labels=names)
    axes2[2].set(ylabel="Softmax probability", ylim=(0, 1), title="Model predictions")
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "features_and_predictions.png", dpi=160)
    for i in selected:
        print(f"test row {i}: {test['source_id'][i]} | true={names[int(test['y'][i])]}, predicted={names[int(predictions[i])]}")
    print("Plots saved to", args.output_dir)
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()

