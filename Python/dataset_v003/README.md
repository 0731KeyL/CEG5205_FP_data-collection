# IMU Classification Dataset `dataset_v003`

This directory is the frozen data interface for a downstream classifier.
Do not edit these files in place and do not regenerate augmentation inside a
model-training repository.

When new real recordings are added, create a new output directory such as
`dataset_v003`; do not overwrite this version. Retrain and version the
downstream model for every dataset version.

## Canonical files

Load one of:

```python
import numpy as np
train = np.load("train.npz")
x = train["x"]              # float32, shape (180, 50, 12)
features = train["features"]  # float32, shape (180, 84)
y = train["y"]              # int64 class index
mask = train["mask"]        # bool, same shape as x
length = train["length"]    # valid time steps
source_id = train["source_id"]  # stable raw-trial identity
```

Use `features_<split>.npy` and `normalization.npz` for feature-based models, or
`X_<split>.npy` and `mask_<split>.npy` for a sequence model.

## Split policy

`within-subject-stratified-source-group`

Validation and test contain real recordings only. Synthetic rows occur only
in `train`.

## Class counts

```text
train:      [24, 24, 24, 24, 24, 4, 4, 52]
validation: [2, 2, 2, 2, 2, 0, 0, 3]
test:       [2, 2, 2, 2, 2, 1, 1, 4]
```

## File integrity

- Full definitions are in `metainfo.json`, `manifest.csv`, and
  `quality_report.csv`.
- Verify the package with `checksums.sha256`.

## Important limitation

The current package contains one subject only. Results obtained with this
split are within-subject diagnostics and do not demonstrate cross-user
generalization.

## Six-class label revision (2026-11-05)

At the user's request, grasp and release are merged into other in this directory. Labels are idle=0, forward=1, backward=2, left=3, right=4, other=5. The dataset_id now ends in _sixclass. Signal arrays, statistical features, source identifiers, split membership and normalization are unchanged. Original source paths still identify the original recording. Previous eight-class model checkpoints require retraining for this label map.
