"""Load and verify the frozen model-agnostic dataset contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(dataset_dir: Path) -> None:
    checksum_file = dataset_dir / "checksums.sha256"
    if not checksum_file.exists():
        raise FileNotFoundError(f"Missing {checksum_file}")
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        path = dataset_dir / name.strip()
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")


def load_split(dataset_dir: str | Path, split: str) -> dict:
    dataset_dir = Path(dataset_dir)
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    verify_checksums(dataset_dir)
    with np.load(dataset_dir / f"{split}.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_v002",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
    )
    args = parser.parse_args()
    data = load_split(args.dataset, args.split)
    print(args.split, {name: value.shape for name, value in data.items()})


if __name__ == "__main__":
    main()
