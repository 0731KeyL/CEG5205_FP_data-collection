# CEG5205 Final Project — IMU Data Acquisition

## English Version

### 1. Project Structure

```text
Python/
├── imu_real_batch_recorder.py
├── analyze_real_imu.py
├── augment_imu_dataset.py
├── plot_imu_csv.py
│
├── real_imu_data/
├── augmented_imu_data/
├── real_imu_analysis/
└── imu_figures/
```

### 2. File Descriptions

| File / Folder | Description |
|---|---|
| `imu_real_batch_recorder.py` | Collects real-time 12-channel IMU data, displays live plots, labels gestures, and saves CSV recordings. |
| `analyze_real_imu.py` | Analyzes sampling rate, sampling jitter, and missing frames. |
| `augment_imu_dataset.py` | Generates synthetic IMU recordings using time warping, magnitude scaling, and Gaussian noise. Expands each gesture category to 100 CSV files. |
| `plot_imu_csv.py` | Plots 12-channel IMU signals from a selected CSV file and saves a PNG figure. |
| `real_imu_data/` | Original IMU recordings, organized into eight gesture folders. |
| `augmented_imu_data/` | Real and synthetic recordings, organized by gesture. |
| `real_imu_analysis/` | Sampling analysis results and figures. |
| `imu_figures/` | Generated 12-channel IMU signal plots. |

### 3. Installation

Python 3.10 is recommended.

```bash
python -m pip install numpy pandas matplotlib pyserial pyqtgraph PySide6
```

### 4. How to Run

**Step 1 — Record real IMU data**

Connect the Arduino Mega 2560 and close the Arduino Serial Monitor.

```bash
python imu_real_batch_recorder.py
```

**Step 2 — Analyze recorded data**

```bash
python analyze_real_imu.py
```

**Step 3 — Generate synthetic data**

```bash
python augment_imu_dataset.py
```

**Step 4 — Plot IMU signals**

Set `FILE_NAME` in `plot_imu_csv.py`, then run:

```bash
python plot_imu_csv.py
```

### 5. Notes

- Arduino serial port: `COM5`
- Baud rate: `38400`
- Target sampling rate: `50 Hz`
- Sensor channels: `12` (two MPU6050 sensors)
- Gesture labels: `idle`, `forward`, `backward`, `left`, `right`, `grasp`, `release`, `other`
- Synthetic data should be used for training augmentation only. Keep the original test data separate.

---

## 中文版本

### 1. 项目文件结构

```text
Python/
├── imu_real_batch_recorder.py
├── analyze_real_imu.py
├── augment_imu_dataset.py
├── plot_imu_csv.py
│
├── real_imu_data/
├── augmented_imu_data/
├── real_imu_analysis/
└── imu_figures/
```

### 2. 文件说明

| 文件 / 文件夹 | 功能 |
|---|---|
| `imu_real_batch_recorder.py` | 采集真实 12 通道 IMU 数据，实时绘图、动作标注、批量保存 CSV。 |
| `analyze_real_imu.py` | 分析采样频率、采样抖动和丢帧情况。 |
| `augment_imu_dataset.py` | 使用时间扭曲、幅值缩放和高斯噪声生成合成数据，将每类动作扩充到 100 个 CSV。 |
| `plot_imu_csv.py` | 读取指定 CSV，绘制 12 通道 IMU 曲线并保存 PNG 图片。 |
| `real_imu_data/` | 按八类动作分类保存的原始真实数据。 |
| `augmented_imu_data/` | 按动作分类保存的真实数据副本及合成数据。 |
| `real_imu_analysis/` | 采样性能分析结果及图片。 |
| `imu_figures/` | 生成的 12 通道 IMU 信号图。 |

### 3. 环境安装

推荐使用 Python 3.10。

```bash
python -m pip install numpy pandas matplotlib pyserial pyqtgraph PySide6
```

### 4. 运行方法

**第一步：采集真实 IMU 数据**

连接 Arduino Mega 2560，关闭 Arduino Serial Monitor。

```bash
python imu_real_batch_recorder.py
```

**第二步：分析采样质量**

```bash
python analyze_real_imu.py
```

**第三步：生成合成数据**

```bash
python augment_imu_dataset.py
```

**第四步：绘制 IMU 曲线**

在 `plot_imu_csv.py` 中修改 `FILE_NAME`，然后运行：

```bash
python plot_imu_csv.py
```

### 5. 注意事项

- Arduino 串口：`COM5`
- 波特率：`38400`
- 目标采样频率：`50 Hz`
- 数据通道：`12`（两个 MPU6050）
- 动作标签：`idle`、`forward`、`backward`、`left`、`right`、`grasp`、`release`、`other`
- 合成数据仅用于辅助训练，测试集应保留独立的真实数据。