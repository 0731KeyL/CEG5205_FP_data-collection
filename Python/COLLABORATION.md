# 数据组与模型组协作约定

## 分工边界

```text
数据组：raw CSV → 质量检查 → 预处理 → 增强 → split → dataset bundle
模型组：dataset bundle → classifier → metrics
```

模型组只读取冻结数据包，不直接读取 `real_imu_data/`，不在模型仓库中重新增强或划分数据。

## 目录与版本

```text
Python code/
├── imu_real_batch_recorder.py
├── imu_quality.py
├── imu_processing.py
├── imu_augmentations.py
├── build_dataset.py
├── load_dataset.py
└── split_registry.json

Frozen artifacts/
└── dataset_v001/
    ├── train.npz
    ├── validation.npz
    ├── test.npz
    ├── normalization.npz
    ├── metainfo.json
    └── checksums.sha256
```

任何 split、增强、normalization 或 label 变化都必须生成新的 `vNNN`，不能覆盖旧数据包。
`split_registry.json` 负责冻结已有真实 trial 的 split；持续采集时必须保留。

## 持续采集后的重训闭环

```text
新增真实 trial
→ 质量门禁通过
→ build_dataset.py --output dataset_vNNN
→ 模型组重新训练
→ 新模型记录 dataset_id 和 checksum
→ 与旧模型在同一评估口径下对比
→ 发布新的 model version
```

旧数据包和旧模型结果必须保留。新数据只会产生新的 dataset version，
不能直接追加到模型组已经使用的旧数据包。

比较新旧模型时分两栏报告：

- 固定旧测试集：按 `source_id` 取旧版与新版测试集的交集，用于同口径比较；
- 扩展测试集：使用新版全部 test，用于观察新数据加入后的整体表现。

## Git 协作

- 数据组继续在 `origin` 分支开发采集和数据管线。
- `upstream` 保留原始采集基线。
- 模型组使用单独分支，例如 `model/v1`。
- 模型组修改模型代码时提交到自己的分支/PR，不改数据包。
- 数据组修改数据管线时创建新提交和新数据版本。
- 硬件固件有更新时，通过 `upstream` 同步后再合并。

## 给其他组的复用方式

复制仓库后只改 `collection_config.json`：

- 手势类别；
- 传感器位置文本；
- 地址映射说明；
- 默认采集参数。

采集、质量门禁、文件保存、session manifest 和数据集构建逻辑保持不变。

## 交付检查

```cmd
python load_dataset.py --split train
python load_dataset.py --split validation
python load_dataset.py --split test
```

loader 会验证 `checksums.sha256`。校验失败时不得开始模型实验。

## 版本记录

模型结果应注明所用 `dataset_id`。数据管线发生修改时，保留旧数据包并在需要时生成新版本；不要覆盖其他组正在使用的数据。
