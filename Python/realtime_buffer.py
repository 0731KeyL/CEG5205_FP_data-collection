"""Causal streaming buffer for low-latency inference."""

from __future__ import annotations

from collections import deque

import numpy as np


class RealtimeWindowBuffer:
    """Keep only past samples and expose a fixed-size causal window."""

    def __init__(self, window_samples: int = 20, channels: int = 12):
        if window_samples <= 0:
            raise ValueError("window_samples must be positive")
        self.window_samples = int(window_samples)
        self.channels = int(channels)
        self._values = deque(maxlen=self.window_samples)
        self._visible = deque(maxlen=self.window_samples)
        self._timestamps = deque(maxlen=self.window_samples)

    def append(self, values, timestamp_ms, visible=None):
        sample = np.asarray(values, dtype=np.float32).reshape(-1)
        if sample.shape[0] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {sample.shape[0]}"
            )
        if visible is None:
            visible_sample = np.ones(self.channels, dtype=bool)
        else:
            visible_sample = np.asarray(visible, dtype=bool).reshape(-1)
            if visible_sample.shape[0] != self.channels:
                raise ValueError("visible mask shape does not match channels")
        self._values.append(sample)
        self._visible.append(visible_sample)
        self._timestamps.append(float(timestamp_ms))

    @property
    def ready(self) -> bool:
        return len(self._values) == self.window_samples

    @property
    def fill_ratio(self) -> float:
        return len(self._values) / self.window_samples

    def get_window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return causal (values, visible_mask, timestamps) to the model."""
        if not self.ready:
            raise RuntimeError(
                f"buffer needs {self.window_samples} samples; "
                f"have {len(self._values)}"
            )
        return (
            np.stack(self._values),
            np.stack(self._visible),
            np.asarray(self._timestamps, dtype=np.float64),
        )

    def clear(self):
        self._values.clear()
        self._visible.clear()
        self._timestamps.clear()
