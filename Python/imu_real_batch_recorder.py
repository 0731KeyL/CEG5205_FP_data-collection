
import argparse
import csv
import os
import re
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import serial
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtWidgets
from serial.tools import list_ports

from imu_quality import (
    TARGET_RATE_HZ,
    assess_trial,
)
from project_config import load_project_config


PROJECT_CONFIG = load_project_config()
DEFAULT_PROTOCOL = PROJECT_CONFIG["defaults"]
DEFAULT_DURATION_S = int(DEFAULT_PROTOCOL["duration_s"])
DEFAULT_TRIALS = int(DEFAULT_PROTOCOL["trials"])
DEFAULT_COUNTDOWN_S = int(DEFAULT_PROTOCOL["countdown_s"])
DEFAULT_REST_S = int(DEFAULT_PROTOCOL["rest_s"])
DEFAULT_BAUD_RATE = int(PROJECT_CONFIG.get("serial", {}).get("baud_rate", 38400))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a labelled two-IMU gesture dataset."
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("CEG5205_PORT", ""),
        help="Serial port, for example COM5. Defaults to CEG5205_PORT.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=int(os.environ.get("CEG5205_BAUD", str(DEFAULT_BAUD_RATE))),
        help="Serial baud rate; defaults to collection_config.json.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a virtual 50 Hz dual-IMU source without Arduino hardware.",
    )
    return parser.parse_args()


ARGS = parse_args()
PORT = ARGS.port.strip()
BAUD_RATE = ARGS.baud
DEMO_MODE = ARGS.demo

WINDOW_SECONDS = 3.0
PLOT_FPS = 25

ACC_SCALE = 8192.0
GYRO_SCALE = 65.5

ACC_DISPLAY_SCALE = 10.0
GYRO_DISPLAY_SCALE = 0.08

CHANNEL_NAMES = [
    "ax1", "ay1", "az1",
    "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2",
    "gx2", "gy2", "gz2"
]

GESTURES = PROJECT_CONFIG["gestures"]

HAND_SIDES = ["Select", "left", "right"]
HANDEDNESS = ["Select", "right-handed", "left-handed", "ambidextrous"]
FIXED_SENSOR_POSITIONS = PROJECT_CONFIG["sensor_layout"]["positions"]
FIXED_SENSOR_DISPLAY = PROJECT_CONFIG["sensor_layout"]["display"]
FIXED_ADDRESS_MAPPING = PROJECT_CONFIG["sensor_layout"]["address_mapping"]

DATA_DIR_NAME = "demo_imu_data" if DEMO_MODE else "real_imu_data"
REJECTED_DIR_NAME = (
    "rejected_demo_imu_data" if DEMO_MODE else "rejected_imu_data"
)
SAVE_DIR = Path(__file__).resolve().parent / DATA_DIR_NAME
REJECTED_DIR = Path(__file__).resolve().parent / REJECTED_DIR_NAME
SESSION_DIR = SAVE_DIR / "_sessions"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)


class SerialAcquisition:

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.buffer = deque(maxlen=500)
        self.recorded_rows = []

        self.received = 0
        self.invalid = 0
        self.missing = 0
        self.previous_seq = None

        self.recording = False
        self.record_start = 0.0

        self.connected = False
        self.error = ""

        self.thread = None

    def start(self):
        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        self.thread.start()

    def run(self):
        if not PORT:
            available = ", ".join(
                port.device for port in list_ports.comports()
            )
            suffix = f" Available ports: {available}." if available else ""
            with self.lock:
                self.error = (
                    "No serial port selected. Restart with "
                    "--port COMx or set CEG5205_PORT."
                    + suffix
                )
            return

        try:
            with serial.Serial(
                PORT,
                BAUD_RATE,
                timeout=0.05
            ) as ser:

                time.sleep(2)
                ser.reset_input_buffer()

                with self.lock:
                    self.connected = True

                while not self.stop_event.is_set():

                    raw = ser.readline()

                    if not raw:
                        continue

                    try:
                        line = raw.decode(
                            "ascii",
                            errors="strict"
                        ).strip()

                        parts = line.split(",")

                        if len(parts) != 14:
                            continue

                        sample = [int(v) for v in parts]

                    except (UnicodeDecodeError, ValueError):
                        with self.lock:
                            self.invalid += 1
                        continue

                    seq = sample[0]
                    timestamp_ms = sample[1]
                    values = sample[2:]

                    arrival_time = time.perf_counter()

                    with self.lock:

                        if self.previous_seq is not None:
                            if seq > self.previous_seq + 1:
                                self.missing += (
                                    seq - self.previous_seq - 1
                                )

                        self.previous_seq = seq

                        self.buffer.append(
                            (timestamp_ms / 1000.0, values)
                        )

                        self.received += 1

                        if self.recording:
                            elapsed = (
                                arrival_time - self.record_start
                            )

                            self.recorded_rows.append([
                                seq,
                                timestamp_ms,
                                round(elapsed, 6),
                                *values
                            ])

        except (serial.SerialException, OSError) as error:
            with self.lock:
                self.error = str(error)

        finally:
            with self.lock:
                self.connected = False

    def begin_recording(self):
        with self.lock:
            self.recorded_rows = []
            self.record_start = time.perf_counter()
            self.recording = True

    def set_gesture(self, gesture):
        # The serial source receives labels only from the recording UI.
        return None

    def end_recording(self):
        with self.lock:
            self.recording = False
            rows = self.recorded_rows
            self.recorded_rows = []

        return rows

    def snapshot(self):
        with self.lock:
            return (
                list(self.buffer),
                self.received,
                self.missing,
                self.connected,
                self.error
            )

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2)


class DemoAcquisition:
    """Virtual dual-IMU source using the same interface as the serial worker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.buffer = deque(maxlen=500)
        self.recorded_rows = []
        self.received = 0
        self.missing = 0
        self.recording = False
        self.record_start = 0.0
        self.connected = False
        self.error = ""
        self.thread = None
        self.gesture = "idle"
        self.rng = np.random.default_rng(20261104)

    def set_gesture(self, gesture):
        with self.lock:
            self.gesture = gesture

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        period = 0.02
        next_tick = time.perf_counter()
        started = next_tick
        seq = 0

        with self.lock:
            self.connected = True

        while not self.stop_event.is_set():
            wait = next_tick - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            next_tick += period

            now = time.perf_counter()
            elapsed = now - started
            with self.lock:
                gesture = self.gesture
            values = self.generate_sample(elapsed, gesture)
            timestamp_ms = int(round(elapsed * 1000.0))
            arrival_time = time.perf_counter()

            with self.lock:
                self.buffer.append((timestamp_ms / 1000.0, values))
                self.received += 1
                if self.recording:
                    self.recorded_rows.append([
                        seq,
                        timestamp_ms,
                        round(arrival_time - self.record_start, 6),
                        *values,
                    ])
            seq += 1

            if time.perf_counter() - next_tick > period * 5:
                next_tick = time.perf_counter() + period

        with self.lock:
            self.connected = False

    def generate_sample(self, elapsed, gesture):
        values = np.zeros(12, dtype=np.float64)

        # Two sensors are mounted differently, so gravity occupies different axes.
        values[2] = 8192.0
        values[6] = 8192.0

        phase = 2.0 * np.pi * 0.8 * elapsed
        wave = np.sin(phase)
        pulse = 0.5 + 0.5 * np.sin(phase)

        if gesture == "forward":
            values[1] -= 3600.0 * wave
            values[7] -= 3300.0 * wave
            values[3] += 210.0 * wave
            values[9] += 210.0 * wave
        elif gesture == "backward":
            values[1] += 3600.0 * wave
            values[7] += 3300.0 * wave
            values[3] -= 210.0 * wave
            values[9] -= 210.0 * wave
        elif gesture == "left":
            values[0] -= 3800.0 * wave
            values[6] -= 3400.0 * wave
            values[4] -= 230.0 * wave
            values[10] -= 230.0 * wave
        elif gesture == "right":
            values[0] += 3800.0 * wave
            values[6] += 3400.0 * wave
            values[4] += 230.0 * wave
            values[10] += 230.0 * wave
        elif gesture == "grasp":
            values[7] += 2600.0 * pulse
            values[8] += 900.0 * pulse
            values[10] += 180.0 * pulse
        elif gesture == "release":
            values[7] -= 2600.0 * pulse
            values[8] -= 900.0 * pulse
            values[10] -= 180.0 * pulse
        elif gesture == "other":
            values[0] += 1900.0 * np.sin(2.1 * phase)
            values[4] += 700.0 * np.cos(1.3 * phase)
            values[7] += 1400.0 * np.sin(0.7 * phase)
            values[10] += 850.0 * np.cos(1.9 * phase)

        values += self.rng.normal(0.0, 18.0, size=12)
        return np.clip(np.rint(values), -32768, 32767).astype(int).tolist()

    def begin_recording(self):
        with self.lock:
            self.recorded_rows = []
            self.record_start = time.perf_counter()
            self.recording = True

    def end_recording(self):
        with self.lock:
            self.recording = False
            rows = self.recorded_rows
            self.recorded_rows = []
        return rows

    def snapshot(self):
        with self.lock:
            return (
                list(self.buffer),
                self.received,
                self.missing,
                self.connected,
                self.error,
            )

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


worker = DemoAcquisition() if DEMO_MODE else SerialAcquisition()
worker.start()

app = pg.mkQApp("IMU Collection Studio")
app.setStyle("Fusion")
app.setFont(QtWidgets.QApplication.font())

pg.setConfigOptions(
    background="#0b1120",
    foreground="#94a3b8",
    antialias=False
)

APP_STYLE = """
QWidget {
    background: #0b1120;
    color: #e2e8f0;
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
}
QFrame#Card, QFrame#Header, QFrame#MetricCard {
    background: #121b2d;
    border: 1px solid #24344f;
    border-radius: 16px;
}
QFrame#Header {
    background: #111a2b;
    border-color: #24344f;
}
QLabel#Title {
    font-size: 25px;
    font-weight: 700;
    color: #f8fafc;
}
QLabel#Subtitle, QLabel#Muted, QLabel#Caption {
    color: #8fa2bd;
}
QLabel#SectionTitle {
    color: #f8fafc;
    font-size: 16px;
    font-weight: 650;
}
QLabel#MetricValue {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 700;
}
QLabel#FixedValue {
    background: #0d1728;
    border: 1px solid #2b3c59;
    border-radius: 9px;
    padding: 8px 10px;
    color: #7dd3fc;
    font-weight: 600;
}
QLabel#Status {
    background: #0d1728;
    border: 1px solid #263752;
    border-radius: 9px;
    padding: 9px 12px;
    color: #a9bad1;
}
QLabel#ModeBadge {
    background: #3b2d12;
    border: 1px solid #7c5b1d;
    border-radius: 11px;
    padding: 5px 10px;
    color: #fbbf24;
    font-weight: 650;
}
QLineEdit, QComboBox, QSpinBox {
    min-height: 22px;
    padding: 8px 10px;
    background: #0d1728;
    border: 1px solid #2b3c59;
    border-radius: 9px;
    color: #e2e8f0;
    selection-background-color: #0ea5e9;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #38bdf8;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox QAbstractItemView {
    background: #121b2d;
    border: 1px solid #2b3c59;
    selection-background-color: #0f5f86;
    outline: 0;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 18px;
    background: #18243a;
    border: none;
}
QPushButton {
    min-height: 24px;
    padding: 9px 14px;
    border-radius: 10px;
    border: 1px solid #2b3c59;
    background: #172337;
    color: #dbe7f5;
    font-weight: 650;
}
QPushButton:hover {
    border-color: #38bdf8;
}
QPushButton#StartButton {
    background: #0f766e;
    border-color: #14b8a6;
    color: #f0fdfa;
    font-size: 14px;
}
QPushButton#StartButton:hover {
    background: #0d9488;
}
QPushButton#StopButton {
    background: #3a1d25;
    border-color: #7f3445;
    color: #fda4af;
}
QPushButton:disabled {
    background: #141d2d;
    border-color: #202d43;
    color: #586980;
}
QProgressBar {
    min-height: 16px;
    max-height: 16px;
    border: none;
    border-radius: 8px;
    background: #1a263b;
    text-align: center;
    color: #dbe7f5;
}
QProgressBar::chunk {
    border-radius: 8px;
    background: #38bdf8;
}
QToolTip {
    background: #172337;
    color: #e2e8f0;
    border: 1px solid #2b3c59;
    padding: 6px;
}
"""

window = QtWidgets.QWidget()
window.setObjectName("MainWindow")
window.setWindowTitle(
    "CEG5205 · IMU 数据采集工作台"
    + (" · 虚拟演示" if DEMO_MODE else " · 串口模式")
)
window.resize(1480, 900)
window.setMinimumSize(1180, 760)
window.setStyleSheet(APP_STYLE)

root_layout = QtWidgets.QVBoxLayout(window)
root_layout.setContentsMargins(22, 20, 22, 20)
root_layout.setSpacing(16)

header = QtWidgets.QFrame()
header.setObjectName("Header")
header_layout = QtWidgets.QHBoxLayout(header)
header_layout.setContentsMargins(22, 16, 22, 16)
header_layout.setSpacing(14)

title_block = QtWidgets.QVBoxLayout()
title_block.setSpacing(2)
title_label = QtWidgets.QLabel("IMU Collection Studio")
title_label.setObjectName("Title")
subtitle_label = QtWidgets.QLabel(
    "双 MPU6050 · 50 Hz · 批量手势数据采集"
)
subtitle_label.setObjectName("Subtitle")
title_block.addWidget(title_label)
title_block.addWidget(subtitle_label)

mode_badge = QtWidgets.QLabel(
    "●  虚拟数据源" if DEMO_MODE else f"●  串口 {PORT or '未选择'}"
)
mode_badge.setObjectName("ModeBadge")

header_layout.addLayout(title_block)
header_layout.addStretch(1)
header_layout.addWidget(mode_badge)
root_layout.addWidget(header)

content_layout = QtWidgets.QHBoxLayout()
content_layout.setSpacing(16)
root_layout.addLayout(content_layout, 1)

control_card = QtWidgets.QFrame()
control_card.setObjectName("Card")
control_card.setFixedWidth(350)
control_layout = QtWidgets.QVBoxLayout(control_card)
control_layout.setContentsMargins(18, 18, 18, 18)
control_layout.setSpacing(12)

control_title = QtWidgets.QLabel("采集设置")
control_title.setObjectName("SectionTitle")
control_subtitle = QtWidgets.QLabel("填写元数据后开始一组 trial")
control_subtitle.setObjectName("Muted")
control_layout.addWidget(control_title)
control_layout.addWidget(control_subtitle)
protocol_label = QtWidgets.QLabel(
    f"旧协议兼容默认值 · {TARGET_RATE_HZ:.0f} Hz · "
    f"{DEFAULT_DURATION_S} s/trial · "
    f"{DEFAULT_TRIALS} trials · "
    f"{DEFAULT_COUNTDOWN_S} s 预备 · "
    f"{DEFAULT_REST_S} s 间隔"
)
protocol_label.setObjectName("Status")
protocol_label.setWordWrap(True)
control_layout.addWidget(protocol_label)

subject_input = QtWidgets.QLineEdit("subject01")
hand_combo = QtWidgets.QComboBox()
hand_combo.addItems(HAND_SIDES)
handedness_combo = QtWidgets.QComboBox()
handedness_combo.addItems(HANDEDNESS)
sensor_layout_value = QtWidgets.QLabel(FIXED_SENSOR_DISPLAY)
sensor_layout_value.setObjectName("FixedValue")
sensor_layout_value.setToolTip(FIXED_ADDRESS_MAPPING)
gesture_combo = QtWidgets.QComboBox()
gesture_combo.addItems(GESTURES)
duration_spin = QtWidgets.QSpinBox()
duration_spin.setRange(1, 60)
duration_spin.setValue(DEFAULT_DURATION_S)
trials_spin = QtWidgets.QSpinBox()
trials_spin.setRange(1, 100)
trials_spin.setValue(DEFAULT_TRIALS)
countdown_spin = QtWidgets.QSpinBox()
countdown_spin.setRange(0, 30)
countdown_spin.setValue(DEFAULT_COUNTDOWN_S)
rest_spin = QtWidgets.QSpinBox()
rest_spin.setRange(0, 30)
rest_spin.setValue(DEFAULT_REST_S)

fields = QtWidgets.QGridLayout()
fields.setHorizontalSpacing(10)
fields.setVerticalSpacing(7)


def add_field(widget, row, column, label_text, span=1):
    label = QtWidgets.QLabel(label_text)
    label.setObjectName("Muted")
    fields.addWidget(label, row * 2, column, 1, span)
    fields.addWidget(widget, row * 2 + 1, column, 1, span)


add_field(subject_input, 0, 0, "被试编号", 2)
add_field(hand_combo, 1, 0, "采集手")
add_field(handedness_combo, 1, 1, "惯用手")
add_field(sensor_layout_value, 2, 0, "固定传感器布局", 2)
add_field(gesture_combo, 3, 0, "手势类别", 2)
add_field(duration_spin, 4, 0, "单次时长 / 秒")
add_field(trials_spin, 4, 1, "采集次数")
add_field(countdown_spin, 5, 0, "准备倒数 / 秒")
add_field(rest_spin, 5, 1, "trial 间隔 / 秒")
fields.setColumnStretch(0, 1)
fields.setColumnStretch(1, 1)
control_layout.addLayout(fields)
control_layout.addStretch(1)

start_button = QtWidgets.QPushButton("开始批量采集")
stop_button = QtWidgets.QPushButton("停止当前采集")
start_button.setObjectName("StartButton")
stop_button.setObjectName("StopButton")
start_button.setEnabled(False)
stop_button.setEnabled(False)
control_layout.addWidget(start_button)
control_layout.addWidget(stop_button)

content_layout.addWidget(control_card)

right_layout = QtWidgets.QVBoxLayout()
right_layout.setSpacing(14)
content_layout.addLayout(right_layout, 1)

metric_layout = QtWidgets.QHBoxLayout()
metric_layout.setSpacing(12)
right_layout.addLayout(metric_layout)


def make_metric(title):
    card = QtWidgets.QFrame()
    card.setObjectName("MetricCard")
    card_layout = QtWidgets.QVBoxLayout(card)
    card_layout.setContentsMargins(16, 13, 16, 13)
    card_layout.setSpacing(3)
    title_widget = QtWidgets.QLabel(title)
    title_widget.setObjectName("Caption")
    value_widget = QtWidgets.QLabel("--")
    value_widget.setObjectName("MetricValue")
    card_layout.addWidget(title_widget)
    card_layout.addWidget(value_widget)
    metric_layout.addWidget(card, 1)
    return value_widget


rate_label = make_metric("实时采样率")
received_label = make_metric("已接收样本")
missing_label = make_metric("缺失帧")
progress_value = make_metric("采集进度")
rate_label.setText("-- Hz")
received_label.setText("0")
missing_label.setText("0")
progress_value.setText("0 / 0")

CONNECTING_MESSAGE = (
    "正在启动虚拟 50 Hz 双 IMU 数据源…"
    if DEMO_MODE
    else (
        f"正在连接 {PORT}，波特率 {BAUD_RATE}…"
        if PORT
        else "未选择串口，请用 --port COMx 或 --demo 启动。"
    )
)
status_label = QtWidgets.QLabel(CONNECTING_MESSAGE)
status_label.setObjectName("Status")
status_label.setWordWrap(True)
right_layout.addWidget(status_label)

progress_bar = QtWidgets.QProgressBar()
progress_bar.setValue(0)
progress_bar.setTextVisible(False)
right_layout.addWidget(progress_bar)

plot_card = QtWidgets.QFrame()
plot_card.setObjectName("Card")
plot_layout = QtWidgets.QVBoxLayout(plot_card)
plot_layout.setContentsMargins(16, 14, 16, 14)
plot_layout.setSpacing(8)

plot_header = QtWidgets.QHBoxLayout()
plot_title = QtWidgets.QLabel("实时 IMU 波形")
plot_title.setObjectName("SectionTitle")
plot_hint = QtWidgets.QLabel("2 × MPU6050 · 12 通道 · 3 秒滚动窗口")
plot_hint.setObjectName("Muted")
plot_header.addWidget(plot_title)
plot_header.addStretch(1)
plot_header.addWidget(plot_hint)
plot_layout.addLayout(plot_header)

plot_widget = pg.PlotWidget()
plot_widget.setMinimumHeight(430)
plot_layout.addWidget(plot_widget, 1)
right_layout.addWidget(plot_card, 1)

plot_widget.setBackground("#0b1120")
plot_widget.setLabel("bottom", "时间", units="s")
plot_widget.setLabel("left", "归一化幅值")
plot_widget.getPlotItem().hideAxis("top")
plot_widget.getPlotItem().hideAxis("right")
for axis_name in ("left", "bottom"):
    axis = plot_widget.getAxis(axis_name)
    axis.setPen(pg.mkPen("#33445f", width=1))
    axis.setTextPen(pg.mkPen("#8fa2bd"))
    axis.setStyle(tickLength=-4)

plot_widget.showGrid(
    x=True,
    y=True,
    alpha=0.2
)

legend = plot_widget.addLegend(offset=(-10, 10))
legend.setBrush(pg.mkBrush(18, 27, 45, 220))
legend.setPen(pg.mkPen("#2b3c59"))

plot_widget.setXRange(-WINDOW_SECONDS, 0, padding=0)
plot_widget.setYRange(-45, 45, padding=0)

plot_widget.setMouseEnabled(x=False, y=False)
plot_widget.enableAutoRange(x=False, y=False)

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#4c78a8", "#f58518"
]

curves = []

for i in range(12):
    curve = plot_widget.plot(
        pen=pg.mkPen(colors[i], width=1.5),
        name=CHANNEL_NAMES[i]
    )
    curve.setClipToView(True)
    curves.append(curve)

state = "idle"
trial_number = 0
total_trials = 0
batch_valid_trials = 0
batch_invalid_trials = 0
session_manifest_path = None
session_manifest_rows = []

record_seconds = DEFAULT_DURATION_S
countdown_seconds = DEFAULT_COUNTDOWN_S
rest_seconds = DEFAULT_REST_S

phase_start = 0.0

current_subject = ""
current_hand_side = ""
current_handedness = ""
current_sensor_positions = FIXED_SENSOR_POSITIONS
current_address_mapping = FIXED_ADDRESS_MAPPING
current_gesture = ""
session_id = ""

last_rate_time = time.perf_counter()
last_rate_count = 0


def safe_name(value):
    return re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        value.strip()
    )


def set_controls_enabled(enabled):
    for widget in [
        subject_input,
        hand_combo,
        handedness_combo,
        gesture_combo,
        duration_spin,
        trials_spin,
        countdown_spin,
        rest_spin
    ]:
        widget.setEnabled(enabled)


def metadata_ready():
    return bool(
        safe_name(subject_input.text())
        and hand_combo.currentText() in {"left", "right"}
        and handedness_combo.currentText() in {
            "right-handed",
            "left-handed",
            "ambidextrous",
        }
    )


def refresh_start_button():
    _, _, _, connected, error = worker.snapshot()
    start_button.setEnabled(
        state == "idle"
        and connected
        and not error
        and metadata_ready()
    )

def append_session_manifest(filepath, quality):
    if session_manifest_path is None:
        return
    session_manifest_rows.append({
        "session_id": session_id,
        "subject_id": current_subject,
        "hand_side": current_hand_side,
        "handedness": current_handedness,
        "sensor_positions": current_sensor_positions,
        "address_mapping": current_address_mapping,
        "gesture": current_gesture,
        "trial": trial_number,
        "file": "" if filepath is None else str(filepath),
        "valid": quality["valid"],
        "quality_flags": quality["flags"],
        "samples": quality["samples"],
        "target_duration_s": record_seconds,
        "duration_s": round(quality["duration_s"], 6),
        "rate_hz": round(quality["rate_hz"], 3),
        "missing_frames": quality["missing"],
        "saturated_values": quality["saturated"],
    })
    pd.DataFrame(session_manifest_rows).to_csv(
        session_manifest_path,
        index=False,
    )



def save_trial(rows):
    global batch_valid_trials, batch_invalid_trials

    quality = assess_trial(rows, record_seconds)
    if not rows:
        batch_invalid_trials += 1
        append_session_manifest(None, quality)
        status_label.setText(
            f"Trial {trial_number} 不合格：没有采到样本"
        )
        return quality

    destination = SAVE_DIR if quality["valid"] else REJECTED_DIR
    gesture_dir = destination / current_gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{current_subject}_"
        f"{current_hand_side}_"
        f"{current_gesture}_"
        f"{session_id}_"
        f"trial{trial_number:03d}.csv"
    )

    filepath = gesture_dir / filename
    suffix = 2
    while filepath.exists():
        filename = (
            f"{current_subject}_"
            f"{current_hand_side}_"
            f"{current_gesture}_"
            f"{session_id}_"
            f"trial{trial_number:03d}_v{suffix:03d}.csv"
        )
        filepath = gesture_dir / filename
        suffix += 1

    header = [
        "seq",
        "timestamp_ms",
        "recording_time_s",
        *CHANNEL_NAMES,
        "gesture",
        "subject_id",
        "trial",
        "hand_side",
        "handedness",
        "sensor_positions",
        "address_mapping",
        "target_duration_s",
        "quality_status",
        "quality_flags",
    ]

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)

        for row in rows:
            writer.writerow([
                *row,
                current_gesture,
                current_subject,
                trial_number,
                current_hand_side,
                current_handedness,
                current_sensor_positions,
                current_address_mapping,
                record_seconds,
                quality["grade"],
                quality["flags"],
            ])

    if quality["valid"]:
        batch_valid_trials += 1
        status_label.setText(
            f"Trial {trial_number} {quality['grade']} · "
            f"{quality['rate_hz']:.1f} Hz · "
            f"{quality['samples']} samples"
        )
    else:
        batch_invalid_trials += 1
        status_label.setText(
            f"Trial {trial_number} 不合格 · {quality['flags']}"
        )
    progress_value.setText(
        f"{trial_number} / {total_trials} · "
        f"{batch_valid_trials} valid"
    )
    append_session_manifest(filepath, quality)
    print(
        f"Saved trial {trial_number}: {quality['valid']=} "
        f"{quality['samples']} samples -> {filepath}"
    )
    return quality



def set_idle(message):
    global state

    state = "idle"
    set_controls_enabled(True)

    _, _, _, connected, error = worker.snapshot()

    start_button.setEnabled(False)
    refresh_start_button()
    stop_button.setEnabled(False)

    status_label.setText(message)


def start_batch():
    global state, trial_number, total_trials
    global record_seconds, countdown_seconds
    global rest_seconds, phase_start
    global current_subject, current_hand_side, current_handedness
    global current_gesture, session_id
    global session_manifest_path, session_manifest_rows
    global batch_valid_trials, batch_invalid_trials

    _, _, _, connected, error = worker.snapshot()

    if not connected or error:
        status_label.setText("数据源尚未连接。")
        return

    subject = safe_name(subject_input.text())

    if not subject:
        QtWidgets.QMessageBox.warning(
            window,
            "Invalid Subject",
            "Please enter a Subject ID."
        )
        return

    hand_side = hand_combo.currentText()
    handedness = handedness_combo.currentText()

    if (
        hand_side == "Select"
        or handedness == "Select"
    ):
        QtWidgets.QMessageBox.warning(
            window,
            "Missing Metadata",
            "Please specify hand side and handedness."
        )
        return

    current_subject = subject
    current_hand_side = hand_side
    current_handedness = handedness
    current_gesture = gesture_combo.currentText()
    worker.set_gesture(current_gesture)

    total_trials = trials_spin.value()
    record_seconds = duration_spin.value()
    countdown_seconds = countdown_spin.value()
    rest_seconds = rest_spin.value()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_manifest_path = SESSION_DIR / f"{session_id}.csv"
    session_manifest_rows = []
    batch_valid_trials = 0
    batch_invalid_trials = 0

    trial_number = 1
    state = "countdown"
    phase_start = time.perf_counter()

    progress_bar.setMaximum(total_trials)
    progress_bar.setValue(0)
    progress_value.setText(f"0 / {total_trials} · 0 valid")

    set_controls_enabled(False)
    start_button.setEnabled(False)
    stop_button.setEnabled(True)

    print("Batch started:", current_gesture)


def finish_batch():
    worker.set_gesture("idle")
    set_idle("本组采集完成，可以继续下一组。")


def stop_batch():
    if state == "recording":
        rows = worker.end_recording()
        save_trial(rows)

    worker.set_gesture("idle")
    set_idle("当前采集已停止。")


def update_batch(now):
    global state, phase_start, trial_number

    if state == "idle":
        return

    elapsed = now - phase_start

    if state == "countdown":
        remaining = max(0, countdown_seconds - elapsed)

        status_label.setText(
            f"Trial {trial_number}/{total_trials} "
            f"- Prepare: {remaining:.1f}s"
        )

        if elapsed >= countdown_seconds:
            worker.begin_recording()
            phase_start = time.perf_counter()
            state = "recording"

    elif state == "recording":
        status_label.setText(
            f"Trial {trial_number}/{total_trials} "
            f"- Recording: {elapsed:.1f}s"
        )

        if elapsed >= record_seconds:
            rows = worker.end_recording()
            save_trial(rows)

            progress_bar.setValue(trial_number)

            if trial_number >= total_trials:
                finish_batch()
            else:
                state = "rest"
                phase_start = time.perf_counter()

    elif state == "rest":
        remaining = max(0, rest_seconds - elapsed)

        status_label.setText(
            f"Rest: {remaining:.1f}s"
        )

        if elapsed >= rest_seconds:
            trial_number += 1
            state = "countdown"
            phase_start = time.perf_counter()


def update_plot():
    global last_rate_time, last_rate_count

    snapshot, total, lost, connected, error = worker.snapshot()

    if len(snapshot) >= 2:
        timestamps = np.array(
            [item[0] for item in snapshot],
            dtype=float
        )

        values = np.array(
            [item[1] for item in snapshot],
            dtype=float
        ).T

        latest_time = timestamps[-1]

        mask = timestamps >= latest_time - WINDOW_SECONDS

        timestamps = timestamps[mask]
        values = values[:, mask]

        scaled = values.copy()

        for index in [0, 1, 2, 6, 7, 8]:
            scaled[index] = (
                scaled[index] / ACC_SCALE
            ) * ACC_DISPLAY_SCALE

        for index in [3, 4, 5, 9, 10, 11]:
            scaled[index] = (
                scaled[index] / GYRO_SCALE
            ) * GYRO_DISPLAY_SCALE

        x = timestamps - latest_time

        for i in range(12):
            curves[i].setData(x, scaled[i])

    now = time.perf_counter()
    elapsed = now - last_rate_time

    if elapsed >= 1:
        rate = (total - last_rate_count) / elapsed

        rate_label.setText(f"{rate:.1f} Hz")
        received_label.setText(f"{total:,}")
        missing_label.setText(f"{lost:,}")

        last_rate_time = now
        last_rate_count = total

    if error:
        if state == "recording":
            rows = worker.end_recording()
            save_trial(rows)

        if state != "idle":
            set_idle("Recording stopped: serial error")

        status_label.setText(f"数据源错误：{error}")
        start_button.setEnabled(False)

    elif not connected:
        start_button.setEnabled(False)

        if state == "idle":
            status_label.setText(CONNECTING_MESSAGE)

    elif state == "idle":
        refresh_start_button()

        if status_label.text() == CONNECTING_MESSAGE:
            status_label.setText(
                "虚拟数据源已就绪 · 可开始批量采集"
                if DEMO_MODE
                else f"串口数据源已就绪 · {PORT} · {BAUD_RATE} baud"
            )

    if connected and not error:
        update_batch(now)


def close_event(event):
    if state == "recording":
        rows = worker.end_recording()
        save_trial(rows)

    timer.stop()
    worker.stop()
    event.accept()


start_button.clicked.connect(start_batch)
stop_button.clicked.connect(stop_batch)
for widget in [subject_input]:
    widget.textChanged.connect(lambda *_: refresh_start_button())
for widget in [hand_combo, handedness_combo]:
    widget.currentIndexChanged.connect(lambda *_: refresh_start_button())

timer = QtCore.QTimer()
timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
timer.timeout.connect(update_plot)
timer.start(int(1000 / PLOT_FPS))

window.closeEvent = close_event

window.show()
app.exec()

worker.stop()

print("Program closed.")
