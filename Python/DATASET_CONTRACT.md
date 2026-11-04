# Downstream Dataset Contract

`build_dataset.py` produces a frozen dataset directory such as
`dataset_v002/`. A downstream model must consume this directory and must
not re-read `real_imu_data/`, regenerate splits, or apply new augmentation.

## Required files

```text
train.npz
validation.npz
test.npz
normalization.npz
metainfo.json
label_map.json
checksums.sha256
splits.json
```

Each split NPZ contains:

| Key | Shape | Meaning |
|---|---|---|
| `x` | `N × 50 × 12` | SI-unit IMU windows |
| `y` | `N` | integer class label |
| `mask` | `N × 50 × 12` | visible-value mask |
| `length` | `N` | valid time steps |
| `features` | `N × 84` | optional summary features for feature-based models |
| `source_index` | `N` | original real-trial index |
| `is_synthetic` | `N` | augmentation indicator |

`class_names`, `channel_names`, and `feature_names` are embedded in every
split NPZ and are also stored in the JSON metadata.

## Loader

```python
from load_dataset import load_split

train = load_split("dataset_v002", "train")
validation = load_split("dataset_v002", "validation")
test = load_split("dataset_v002", "test")

# Feature-based model
X_train = train["features"]
y_train = train["y"]

# Sequence model
X_train_sequence = train["x"]
mask_train = train["mask"]
length_train = train["length"]
```

The loader verifies every file against `checksums.sha256` before returning data.

## Rules for downstream work

1. Use the provided class-to-index mapping.
2. Use `normalization.npz`; do not compute new normalization on validation or
   test.
3. Use class weighting if using the full 8-class objective; `grasp` and
   `release` are underrepresented.
4. Report validation and test metrics separately.
5. Do not merge `source_index` values across splits.
6. Do not add new augmentation inside the model repository. A data change
   requires a new dataset version.

## Versioning

Any change to split membership, normalization, channel order, label mapping,
or augmentation policy requires a new dataset version and a new changelog
entry. Never overwrite a dataset version that another group may already use.

## Continuous collection and retraining

When new real recordings are available:

1. keep the old dataset directory unchanged;
2. rebuild into `dataset_v003`, `dataset_v004`, and so on;
3. train a new model version using the new `dataset_id`;
4. record dataset checksum, model version, seed, and metrics in the model run
   metadata;
5. compare the new model with the old model under the same evaluation rules.

A model result is incomplete unless it names the dataset version and verifies
that dataset's checksum.
