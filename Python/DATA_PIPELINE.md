# IMU 数据处理与增强流程

## 当前数据事实

- 真实数据：74 条，按 8 类动作分目录保存。
- 被试：仅 `subject01`，不能进行跨被试评价。
- 真实类别数量：`idle/forward/backward/left/right` 各 10 条，`grasp/release` 各 2 条，`other` 20 条。
- 采样率：约 49.04–50.05 Hz。
- 每条原始记录：49–100 行，约 0.96–1.98 秒。
- 现有真实文件未发现重复帧、乱序帧、传感器饱和或恒定通道。
- 原始文件没有左右手、惯用手和 IMU 佩戴位置信息。
- 2026-11-04 经采集者确认：现有 `subject01` 数据使用左手，两个 IMU 分别位于食指和拇指。
- `0x68/mpu1` 与 `0x69/mpu2` 具体对应食指还是拇指仍未确认，不能根据地址猜测。

## 原始字段

```text
seq, timestamp_ms, recording_time_s,
ax1, ay1, az1, gx1, gy1, gz1,
ax2, ay2, az2, gx2, gy2, gz2,
gesture, subject_id, trial
```

新版采集脚本会额外写入：

```text
hand_side, handedness, sensor_positions, address_mapping
```

## 旧协议兼容的默认参数与质量门禁

- 采样率目标为 50 Hz；
- 每个 trial 默认 1 秒，可按组调整；
- 每组默认 10 个 trial，可按组调整；
- 开始前默认倒数 3 秒，trial 间隔默认 0 秒，均可按组调整；
- 被试、左右手和惯用手未填完整时不能开始；
- 传感器布局固定为 `index|thumb`，地址映射记录为待确认；
- 采集程序自动检查 45–55 个样本、0.9–1.1 秒、45–55 Hz、
  连续帧号、丢帧率和传感器饱和；
- 合格 trial 写入 `real_imu_data/<gesture>/`，不合格 trial 写入
  `rejected_imu_data/<gesture>/`；
- 每组 trial 自动汇总到 `real_imu_data/_sessions/<session_id>.csv`；
- Demo 模式写入 `demo_imu_data/`，不会进入正式训练数据。

## 单位换算

保存值是 int16 原始计数，处理脚本将其转换为 SI：

```text
acceleration_m_s2 = raw_acc / 8192 * 9.80665
angular_velocity_rad_s = raw_gyro / 65.5 * pi / 180
```

变换后数据形状为：

```text
N × 50 × 12
```

含义依次为样本数、固定 1 秒窗口、两个 IMU 的加速度与角速度。

## 一键生成训练数据

```bash
python prepare_imu_dataset.py
```

默认生成：

```text
processed_imu_data/real_imu_windows.npz
processed_imu_data/training_windows_with_augmentation.npz
processed_imu_data/manifest.csv
processed_imu_data/augmented_manifest.csv
processed_imu_data/summary.json
```

真实窗口与增强窗口分开保存。增强数据默认每条真实记录增加 3 个版本，且保留 `source_index`：

```text
真实数据 74 条
增强版本 222 条
合计 296 条
```

这些增强版本不是独立被试或独立 trial。训练/验证/测试划分必须以 `source_index` 或后续的 `subject_id` 为组，不能让同一条记录的副本跨集合。

## 正式分类数据集

```bash
python build_dataset.py --output dataset_v002
```

持续采集时使用递增版本，例如 `dataset_v003`；已有数据包不得原地覆盖。

该流程会依次完成：

1. 读取 device timestamp，并重采样到统一 50 Hz；
2. 按运动能量选择事件中心窗口；
3. 检查采样率、缺帧、饱和、时间戳 jitter 和重复样本；
4. 先按 subject 或 source group 固定 split；
5. 只对训练集应用增强；
6. 输出 split 级 `X/y/mask/source/features`；
7. 只使用训练集可见值计算 normalization；
8. 生成 `metainfo.json`、manifest、quality/duplicate report 和 checksums。

当前正式输出位于：

```text
dataset_v002/
├── train.npz / validation.npz / test.npz
├── X_train.npy / X_validation.npy / X_test.npy
├── y_train.npy / y_validation.npy / y_test.npy
├── mask_*.npy
├── source_*.npy
├── is_synthetic_*.npy
├── features_*.npy
├── feature_names.json
├── label_map.json
├── normalization.npz
├── manifest.csv
├── quality_report.csv
├── duplicate_report.csv
├── augmented_manifest.csv
├── metainfo.json
└── checksums.sha256
```

`train.npz`、`validation.npz`、`test.npz` 是下游的正式接口，每个文件包含
`x / y / mask / length / source_index / is_synthetic / features / class_names /
channel_names / feature_names`。具体读取方式见 `DATASET_CONTRACT.md`。

当前单被试数据的实际输出：

```text
train       45 条真实 + 135 条增强
validation  13 条真实
test        16 条真实
```

三个 split 的 `source_index` 无交集；validation/test 中没有增强数据。
由于 `grasp/release` 各只有 2 条真实数据，validation 中这两个类别为空，
test 中各有 1 条。该限制会在 metainfo 和报告中保留。

数据版本更新后的闭环：

```text
新增真实 trial
→ 重新运行质量与构建
→ 生成新 dataset version
→ 下游 retrain
→ 生成与 dataset_id 绑定的新 model version
→ 在同一评估口径下比较旧/新模型
```

## 当前增强

`imu_augmentations.py` 实现：

- 70%–100% 随机窗口裁剪后重采样到 50 点；
- 单调时间弯曲；
- 动作窗口在时间轴上随机平移，最多约 ±20%；
- 1–5 个样本的随机连续丢帧；
- 丢帧位置使用 sample-and-hold，并返回 visible mask；
- 随机单通道短时掉线；
- 随机选择一个 IMU 连续中断 10%–30% 的窗口；
- 两个 IMU 之间 1–2 个采样点的时间错位；
- 小幅度 SI 单位高斯噪声；
- 轻微因果低通滤波差异；
- 输出 `(sequence, visible_mask, augmentation_info)`。

此外还实现了物理上较合理的旋转、每个 IMU 的加速度/角速度增益和 gyro bias 扰动。
增强顺序、概率和强度由 `augmentation_config.json` 控制；新增强只需注册一个
NumPy 函数并加入配置，不需要修改数据集构建主流程。

当前没有声称这些增强能替代真实多人数据。

## 真实数据基线

```bash
python train_imu_baseline.py
```

该脚本只使用 `real_imu_data/`，不读取合成数据。由于只有一个被试，当前采用 leave-one-file-out 作为开发诊断，而不是跨用户准确率。

当前运行结果：

```text
recording: 74
accuracy: 0.9324
balanced accuracy: 0.7375
macro F1: 0.7261
```

五个方向类和 `other` 分类较好，但 `grasp` 与 `release` 全部被预测为 `other`。这说明现有两条真实记录不足以训练可靠抓取/释放分类器，不能通过 98 条同源合成文件掩盖。

输出位于：

```text
real_imu_analysis/baseline/metrics.json
real_imu_analysis/baseline/predictions.csv
real_imu_analysis/baseline/confusion_matrix.png
```

## 已知限制

1. 只有一个被试，不能报告跨用户泛化。
2. 现有数据确认是左手，但两个 I²C 地址与食指/拇指的对应关系仍未知。
3. `grasp/release` 真实数据各只有 2 条，最多用于流程验证。
4. 旧版 `augmented_imu_data/` 虽然每类 100 个文件，但绝大部分来自同一真实记录的小幅扰动。
5. 两张传感器轴向与空间安装关系没有文档化，不能安全地做任意轴翻转或左右手镜像。
