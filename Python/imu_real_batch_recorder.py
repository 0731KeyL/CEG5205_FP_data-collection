
import csv
import re
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import serial
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtWidgets

PORT = "COM5"
BAUD_RATE = 38400

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

GESTURES = [
    "idle",
    "forward",
    "backward",
    "left",
    "right",
    "grasp",
    "release",
    "other"
]

SAVE_DIR = Path(__file__).resolve().parent / "real_imu_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


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


worker = SerialAcquisition()
worker.start()

app = pg.mkQApp("Real IMU Batch Recorder")

pg.setConfigOptions(
    background="w",
    foreground="k",
    antialias=False
)

window = QtWidgets.QWidget()
window.setWindowTitle(
    "CEG5205 - Real IMU Batch Recorder"
)
window.resize(1600, 900)

layout = QtWidgets.QVBoxLayout(window)

controls = QtWidgets.QGridLayout()

subject_input = QtWidgets.QLineEdit("subject01")

gesture_combo = QtWidgets.QComboBox()
gesture_combo.addItems(GESTURES)

duration_spin = QtWidgets.QSpinBox()
duration_spin.setRange(1, 60)
duration_spin.setValue(1)

trials_spin = QtWidgets.QSpinBox()
trials_spin.setRange(1, 100)
trials_spin.setValue(10)

countdown_spin = QtWidgets.QSpinBox()
countdown_spin.setRange(0, 30)
countdown_spin.setValue(3)

rest_spin = QtWidgets.QSpinBox()
rest_spin.setRange(0, 30)
rest_spin.setValue(0)

controls.addWidget(QtWidgets.QLabel("Subject ID"), 0, 0)
controls.addWidget(subject_input, 0, 1)

controls.addWidget(QtWidgets.QLabel("Gesture"), 0, 2)
controls.addWidget(gesture_combo, 0, 3)

controls.addWidget(QtWidgets.QLabel("Duration (s)"), 1, 0)
controls.addWidget(duration_spin, 1, 1)

controls.addWidget(QtWidgets.QLabel("Trials"), 1, 2)
controls.addWidget(trials_spin, 1, 3)

controls.addWidget(QtWidgets.QLabel("Countdown (s)"), 2, 0)
controls.addWidget(countdown_spin, 2, 1)

controls.addWidget(QtWidgets.QLabel("Rest (s)"), 2, 2)
controls.addWidget(rest_spin, 2, 3)

layout.addLayout(controls)

buttons = QtWidgets.QHBoxLayout()

start_button = QtWidgets.QPushButton("Start Batch")
stop_button = QtWidgets.QPushButton("Stop Batch")

start_button.setEnabled(False)
stop_button.setEnabled(False)

buttons.addWidget(start_button)
buttons.addWidget(stop_button)

layout.addLayout(buttons)

status_label = QtWidgets.QLabel("Connecting to Arduino...")
rate_label = QtWidgets.QLabel("Receive rate: -- Hz")
frame_label = QtWidgets.QLabel("Received: 0 | Missing: 0")

layout.addWidget(status_label)
layout.addWidget(rate_label)
layout.addWidget(frame_label)

progress_bar = QtWidgets.QProgressBar()
progress_bar.setValue(0)
layout.addWidget(progress_bar)

plot_widget = pg.PlotWidget()
layout.addWidget(plot_widget)

plot_widget.setTitle(
    "Real-Time IMU Data (12 Channels)"
)

plot_widget.setLabel("bottom", "Time", units="s")
plot_widget.setLabel("left", "Signal Value (Scaled)")

plot_widget.showGrid(
    x=True,
    y=True,
    alpha=0.2
)

plot_widget.addLegend(offset=(-10, 10))

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

record_seconds = 3
countdown_seconds = 3
rest_seconds = 2

phase_start = 0.0

current_subject = ""
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
        gesture_combo,
        duration_spin,
        trials_spin,
        countdown_spin,
        rest_spin
    ]:
        widget.setEnabled(enabled)



def save_trial(rows):
    if not rows:
        print("No samples recorded in trial", trial_number)
        return

    gesture_dir = SAVE_DIR / current_gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{current_subject}_"
        f"{current_gesture}_"
        f"{session_id}_"
        f"trial{trial_number:03d}.csv"
    )

    filepath = gesture_dir / filename

    header = [
        "seq",
        "timestamp_ms",
        "recording_time_s",
        *CHANNEL_NAMES,
        "gesture",
        "subject_id",
        "trial"
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
                trial_number
            ])

    print(
        f"Saved trial {trial_number}: "
        f"{len(rows)} samples -> {filepath}"
    )



def set_idle(message):
    global state

    state = "idle"
    set_controls_enabled(True)

    _, _, _, connected, error = worker.snapshot()

    start_button.setEnabled(connected and not error)
    stop_button.setEnabled(False)

    status_label.setText(message)


def start_batch():
    global state, trial_number, total_trials
    global record_seconds, countdown_seconds
    global rest_seconds, phase_start
    global current_subject, current_gesture, session_id

    _, _, _, connected, error = worker.snapshot()

    if not connected or error:
        status_label.setText("Arduino is not connected")
        return

    subject = safe_name(subject_input.text())

    if not subject:
        QtWidgets.QMessageBox.warning(
            window,
            "Invalid Subject",
            "Please enter a Subject ID."
        )
        return

    current_subject = subject
    current_gesture = gesture_combo.currentText()

    total_trials = trials_spin.value()
    record_seconds = duration_spin.value()
    countdown_seconds = countdown_spin.value()
    rest_seconds = rest_spin.value()

    session_id = time.strftime("%Y%m%d_%H%M%S")

    trial_number = 1
    state = "countdown"
    phase_start = time.perf_counter()

    progress_bar.setMaximum(total_trials)
    progress_bar.setValue(0)

    set_controls_enabled(False)
    start_button.setEnabled(False)
    stop_button.setEnabled(True)

    print("Batch started:", current_gesture)


def finish_batch():
    set_idle("Batch finished")


def stop_batch():
    if state == "recording":
        rows = worker.end_recording()
        save_trial(rows)

    set_idle("Batch stopped")


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

        rate_label.setText(
            f"Receive rate: {rate:.2f} Hz"
        )

        frame_label.setText(
            f"Received: {total} | Missing: {lost}"
        )

        last_rate_time = now
        last_rate_count = total

    if error:
        if state == "recording":
            rows = worker.end_recording()
            save_trial(rows)

        if state != "idle":
            set_idle("Recording stopped: serial error")

        status_label.setText(f"Serial error: {error}")
        start_button.setEnabled(False)

    elif not connected:
        start_button.setEnabled(False)

        if state == "idle":
            status_label.setText("Connecting to Arduino...")

    elif state == "idle":
        start_button.setEnabled(True)

        if status_label.text() == "Connecting to Arduino...":
            status_label.setText("Status: Ready")

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

timer = QtCore.QTimer()
timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
timer.timeout.connect(update_plot)
timer.start(int(1000 / PLOT_FPS))

window.closeEvent = close_event

window.show()
app.exec()

worker.stop()

print("Program closed.")
