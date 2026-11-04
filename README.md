# CEG5205 Final Project — IMU Data Collection

Dual MPU6050 IMU data acquisition, visualization, gesture dataset generation, and analysis.

**Repository:** https://github.com/lxy336/CEG5205_FP_data-collection

## English

### 1. Project Overview

This project uses two MPU6050 sensors connected to an Arduino Mega 2560 to collect 12-channel IMU data at approximately 50 Hz. Python programs are provided for real-time recording, data analysis, augmentation, and visualization.

### 2. Project Structure

```text
CEG5205_FP_Materials/
├── MPU6050_raw/
│   ├── MPU6050_raw.ino
│   ├── MPU6050.cpp
│   ├── MPU6050.h
│   ├── I2Cdev.cpp
│   └── I2Cdev.h
│
├── Python/
│   ├── imu_real_batch_recorder.py
│   ├── analyze_real_imu.py
│   ├── augment_imu_dataset.py
│   ├── plot_imu_csv.py
│   ├── real_imu_data/
│   ├── augmented_imu_data/
│   ├── real_imu_analysis/
│   └── imu_figures/
│
└── README.md
```

### 3. File Descriptions

| File / Folder | Description |
|---|---|
| `MPU6050_raw/` | Arduino firmware and required sensor libraries. |
| `MPU6050_raw.ino` | Reads two MPU6050 sensors and sends 12-channel data via USB serial. |
| `imu_real_batch_recorder.py` | Real-time visualization, gesture labeling, and batch CSV recording. |
| `analyze_real_imu.py` | Analyzes sampling rate, jitter, and missing frames. |
| `augment_imu_dataset.py` | Generates synthetic IMU data to expand each gesture category to 100 CSV recordings. |
| `plot_imu_csv.py` | Plots 12-channel IMU signals from a selected CSV file. |
| `real_imu_data/` | Original recordings organized by gesture. |
| `augmented_imu_data/` | Original data copies and synthetic recordings. |
| `real_imu_analysis/` | Sampling analysis results and plots. |
| `imu_figures/` | Generated IMU signal figures. |

### 4. Installation and Usage

Clone the repository:

```bash
git clone https://github.com/lxy336/CEG5205_FP_data-collection.git
cd CEG5205_FP_data-collection/Python
```

Install Python dependencies:

```bash
python -m pip install numpy pandas matplotlib pyserial pyqtgraph PySide6
```

Upload `MPU6050_raw.ino` to the Arduino Mega 2560 using Arduino IDE.

Run the Python programs:

| Task | Command |
|---|---|
| Record IMU data | `python imu_real_batch_recorder.py` |
| Analyze recordings | `python analyze_real_imu.py` |
| Generate synthetic data | `python augment_imu_dataset.py` |
| Plot IMU signals | `python plot_imu_csv.py` |

For plotting, modify `FILE_NAME` in `plot_imu_csv.py` before running.

### 5. Configuration

- Board: Arduino Mega 2560
- Sensors: 2 × MPU6050
- I²C addresses: `0x68`, `0x69`
- Serial port: `COM5` (change according to your computer)
- Baud rate: `38400`
- Target sampling rate: `50 Hz`
- Gesture classes: `idle`, `forward`, `backward`, `left`, `right`, `grasp`, `release`, `other`


**Note:** Synthetic recordings are intended for training augmentation. Model evaluation should use independent real recordings.

---

## 中文

### 1. 项目简介

本项目使用 Arduino Mega 2560 连接两个 MPU6050 传感器，以约 50 Hz 的频率采集 12 通道 IMU 数据。

Python 程序用于实时数据采集、手势标注、数据分析、数据增强和信号可视化。

### 2. 文件说明

| 文件 / 文件夹 | 功能 |
|---|---|
| `MPU6050_raw/` | Arduino 程序及所需传感器库。 |
| `MPU6050_raw.ino` | 读取两个 MPU6050，并通过 USB 串口发送 12 通道数据。 |
| `imu_real_batch_recorder.py` | 实时绘图、动作标注、批量采集并保存 CSV。 |
| `analyze_real_imu.py` | 分析采样频率、采样抖动和丢帧情况。 |
| `augment_imu_dataset.py` | 生成合成 IMU 数据，将每类动作扩充至 100 个 CSV。 |
| `plot_imu_csv.py` | 根据指定 CSV 绘制 12 通道 IMU 曲线。 |
| `real_imu_data/` | 按动作分类保存的真实采集数据。 |
| `augmented_imu_data/` | 真实数据副本及生成的合成数据。 |
| `real_imu_analysis/` | 采样质量分析结果和图片。 |
| `imu_figures/` | 生成的 IMU 信号曲线图。 |

### 3. 环境安装与运行

克隆仓库：

```bash
git clone https://github.com/lxy336/CEG5205_FP_data-collection.git
cd CEG5205_FP_data-collection/Python
```

安装依赖：

```bash
python -m pip install numpy pandas matplotlib pyserial pyqtgraph PySide6
```

使用 Arduino IDE 将 `MPU6050_raw.ino` 上传至 Arduino Mega 2560。

运行 Python 程序：

| 功能 | 运行命令 |
|---|---|
| 真实数据采集 | `python imu_real_batch_recorder.py` |
| 数据质量分析 | `python analyze_real_imu.py` |
| 合成数据生成 | `python augment_imu_dataset.py` |
| IMU 曲线绘制 | `python plot_imu_csv.py` |

绘图前，需要在 `plot_imu_csv.py` 中修改 `FILE_NAME`。

### 4. 硬件与数据配置

- 开发板：Arduino Mega 2560
- 传感器：2 个 MPU6050
- I²C 地址：`0x68`、`0x69`
- 串口：`COM5`（根据实际电脑修改）
- 波特率：`38400`
- 目标采样频率：`50 Hz`
- 动作类别：`idle`、`forward`、`backward`、`left`、`right`、`grasp`、`release`、`other`


**注意：** 合成数据用于辅助模型训练，模型测试应使用独立的真实采集数据。