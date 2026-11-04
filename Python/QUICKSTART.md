# 新同学数据采集 Quickstart

## 1. 安装

在仓库根目录执行：

```cmd
cd Python
python -m pip install -r requirements.txt
```

## 2. 无硬件检查界面

```cmd
python imu_real_batch_recorder.py --demo
```

填写被试编号、左右手和惯用手，点击“开始批量采集”。
Demo 数据只写入 `demo_imu_data/`，不会进入真实数据集。

## 3. 使用真实硬件采集

1. 连接 Arduino Mega 和两个 MPU6050。
2. 使用现有固件，串口波特率为 `38400`。
3. 在设备管理器确认 COM 口。
4. 运行：

```cmd
python imu_real_batch_recorder.py --port COM6
```

## 4. 固定布局与采集参数

默认传感器布局：

```text
IMU 位置：食指 + 拇指
```

默认协议：

```text
准备倒数：3 秒
单次时长：1 秒
每组 trial：10 次
trial 间隔：0 秒
```

这些默认值可在 `collection_config.json` 中修改。

## 5. 采集后检查

合格 trial 自动进入：

```text
real_imu_data/<gesture>/
```

不合格 trial 进入：

```text
rejected_imu_data/<gesture>/
```

每组汇总：

```text
real_imu_data/_sessions/<session_id>.csv
```

查看 `quality_status` 和 `quality_flags`。不要手工把 rejected 文件移入真实数据目录。

## 6. 构建下游数据集

```cmd
python build_dataset.py --output dataset_v003
```

新数据到达后使用下一个版本目录，例如：

```cmd
python build_dataset.py --output dataset_v004
```

不要覆盖模型组正在使用的旧版本；每个新数据集版本都需要重新训练并生成新的模型版本。
`split_registry.json` 会固定旧 trial 的 split，持续采集时必须保留并提交，
不要删除或随意修改。

输出：

```text
dataset_v002/train.npz
dataset_v002/validation.npz
dataset_v002/test.npz
dataset_v002/normalization.npz
dataset_v002/metainfo.json
```

## 7. 下游加载

```python
from load_dataset import load_split

train = load_split("dataset_v002", "train")
X = train["features"]   # optional summary features: N × 84
y = train["y"]          # labels: N
```

## 8. 替换手势类别或传感器布局

只修改 `collection_config.json`：

```json
{
  "gestures": ["idle", "forward", "backward", "left", "right"],
  "sensor_layout": {
    "positions": "wrist|forearm",
    "display": "手腕 + 前臂",
    "address_mapping": "confirm before collection"
  }
}
```

不要为不同机械臂项目直接改写核心采集脚本。

## 常见问题

- **没有串口**：确认 Arduino 驱动、USB 和设备管理器中的 COM 口。
- **采集界面能打开但没有数据**：检查 `--port` 和两端 `38400` 波特率。
- **trial 被拒绝**：查看 `quality_flags`，常见原因是丢帧、饱和、采样率异常。
- **下游结果异常**：检查是否使用了正确的 `dataset_vNNN` 和 checksum，不要直接读取 raw CSV。
