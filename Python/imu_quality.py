"""Fixed acquisition protocol and per-trial quality checks."""

from __future__ import annotations

import numpy as np


STANDARD_DURATION_S = 1
STANDARD_TRIALS = 10
STANDARD_COUNTDOWN_S = 3
STANDARD_REST_S = 0
TARGET_RATE_HZ = 50.0
MIN_RATE_HZ = 45.0
MAX_RATE_HZ = 55.0
MIN_DURATION_FRACTION = 0.90
MAX_DURATION_FRACTION = 1.10
MAX_MISSING_RATE = 0.01
SATURATION_LIMIT = 32000


def assess_trial(
    rows,
    expected_duration_s=STANDARD_DURATION_S,
    target_rate_hz=TARGET_RATE_HZ,
):
    """Check one recorded trial and return a serialisable quality summary."""
    if not rows:
        return {
            "valid": False,
            "grade": "rejected",
            "flags": "no_samples",
            "samples": 0,
            "duration_s": 0.0,
            "rate_hz": 0.0,
            "missing": 0,
            "saturated": 0,
            "jitter_ms": 0.0,
            "missing_rate": 1.0,
        }

    sequence = np.asarray([row[0] for row in rows], dtype=np.int64)
    timestamps = np.asarray([row[1] for row in rows], dtype=np.float64)
    values = np.asarray([row[3:] for row in rows], dtype=np.float64)
    sequence_delta = np.diff(sequence)
    time_delta = np.diff(timestamps)
    duration_s = (
        float((timestamps[-1] - timestamps[0]) / 1000.0)
        if len(timestamps) > 1
        else 0.0
    )
    rate_hz = (
        float((len(timestamps) - 1) / duration_s)
        if duration_s > 0
        else 0.0
    )
    missing = int(np.maximum(sequence_delta - 1, 0).sum())
    duplicate = int((sequence_delta == 0).sum())
    out_of_order = int((sequence_delta < 0).sum())
    missing_rate = missing / (len(rows) + missing)
    jitter_ms = float(np.std(time_delta)) if len(time_delta) else 0.0
    saturated = int((np.abs(values) >= SATURATION_LIMIT).sum())
    flags = []

    if expected_duration_s is not None:
        expected_samples = expected_duration_s * target_rate_hz
        if len(rows) < MIN_DURATION_FRACTION * expected_samples:
            flags.append("too_few_samples")
        if len(rows) > MAX_DURATION_FRACTION * expected_samples + 2:
            flags.append("too_many_samples")
        if not (
            MIN_DURATION_FRACTION * expected_duration_s
            <= duration_s
            <= MAX_DURATION_FRACTION * expected_duration_s
        ):
            flags.append("duration_out_of_range")
    if not MIN_RATE_HZ <= rate_hz <= MAX_RATE_HZ:
        flags.append("sample_rate_out_of_range")
    if missing_rate > MAX_MISSING_RATE:
        flags.append("missing_frames")
    if duplicate:
        flags.append("duplicate_sequence")
    if out_of_order:
        flags.append("out_of_order_sequence")
    if saturated:
        flags.append("sensor_saturation")
    warning_flags = []
    if jitter_ms > 3.0:
        warning_flags.append("high_timestamp_jitter")

    grade = "rejected" if flags else ("review" if warning_flags else "accepted")

    return {
        "valid": grade != "rejected",
        "grade": grade,
        "flags": "|".join(flags + warning_flags),
        "samples": len(rows),
        "duration_s": duration_s,
        "rate_hz": rate_hz,
        "missing": missing,
        "saturated": saturated,
        "jitter_ms": jitter_ms,
        "missing_rate": missing_rate,
    }
