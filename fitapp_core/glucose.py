"""Glucose curve analysis.

Inputs are timestamped readings (mg/dL). Outputs cover the metrics that
matter for both consumer (FitApp) and clinical (Vitalstack) surfaces:
    - time_in_range (TIR) per the international consensus 70–180 mg/dL
    - excursion analysis (peak, time-to-peak, area-under-curve, glucose
      management indicator)
    - hypo/hyper event detection

No ML — deterministic, defensible math. Caller decides how to display.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict


class GlucoseReading(TypedDict):
    timestamp: str  # ISO 8601
    value: int      # mg/dL
    context: str    # 'fasting' | 'pre_meal' | 'post_meal' | 'random'


class TIRResult(TypedDict):
    in_range_pct: float       # % readings in 70–180
    below_70_pct: float
    above_180_pct: float
    above_250_pct: float      # severe hyperglycemia
    below_54_pct: float       # severe hypoglycemia
    n_readings: int
    mean_glucose: float
    std_dev: float
    cv_pct: float             # coefficient of variation — diabetic stability proxy
    gmi: float                # Glucose Management Indicator (estimated A1C)


# International consensus targets (Battelino 2019, Diabetes Care)
RANGE_LOW = 70
RANGE_HIGH = 180
SEVERE_LOW = 54
SEVERE_HIGH = 250


def time_in_range(readings: list[GlucoseReading]) -> TIRResult:
    """Time-in-range metrics + GMI estimated A1C from a list of readings.

    GMI formula: 3.31 + 0.02392 × mean_glucose (Bergenstal 2018).
    """
    n = len(readings)
    if n == 0:
        return {
            "in_range_pct": 0.0, "below_70_pct": 0.0, "above_180_pct": 0.0,
            "above_250_pct": 0.0, "below_54_pct": 0.0,
            "n_readings": 0, "mean_glucose": 0.0, "std_dev": 0.0,
            "cv_pct": 0.0, "gmi": 0.0,
        }

    in_range = sum(1 for r in readings if RANGE_LOW <= r["value"] <= RANGE_HIGH)
    below = sum(1 for r in readings if r["value"] < RANGE_LOW)
    severe_below = sum(1 for r in readings if r["value"] < SEVERE_LOW)
    above = sum(1 for r in readings if r["value"] > RANGE_HIGH)
    severe_above = sum(1 for r in readings if r["value"] > SEVERE_HIGH)

    mean = sum(r["value"] for r in readings) / n
    variance = sum((r["value"] - mean) ** 2 for r in readings) / n
    std = variance ** 0.5
    cv = (std / mean * 100) if mean else 0.0
    gmi = round(3.31 + 0.02392 * mean, 2)

    return {
        "in_range_pct": round(in_range / n * 100, 1),
        "below_70_pct": round(below / n * 100, 1),
        "below_54_pct": round(severe_below / n * 100, 1),
        "above_180_pct": round(above / n * 100, 1),
        "above_250_pct": round(severe_above / n * 100, 1),
        "n_readings": n,
        "mean_glucose": round(mean, 1),
        "std_dev": round(std, 1),
        "cv_pct": round(cv, 1),
        "gmi": gmi,
    }


def glucose_excursion(readings: list[GlucoseReading]) -> dict[str, float | int]:
    """Peak + time-to-peak + delta + AUC over baseline for a meal-window
    series (typically 6–10 readings spanning 0–120 minutes post-meal).

    Caller is responsible for selecting only the post-meal window.
    """
    if len(readings) < 2:
        return {"peak": 0, "delta": 0, "time_to_peak_min": 0, "auc_above_baseline": 0}

    sorted_r = sorted(readings, key=lambda r: r["timestamp"])
    t0 = datetime.fromisoformat(sorted_r[0]["timestamp"].replace("Z", "+00:00"))
    baseline = sorted_r[0]["value"]
    peak_value = baseline
    peak_t = t0

    for r in sorted_r:
        if r["value"] > peak_value:
            peak_value = r["value"]
            peak_t = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))

    # Trapezoidal AUC above baseline
    auc = 0.0
    for i in range(1, len(sorted_r)):
        a, b = sorted_r[i - 1], sorted_r[i]
        ta = datetime.fromisoformat(a["timestamp"].replace("Z", "+00:00"))
        tb = datetime.fromisoformat(b["timestamp"].replace("Z", "+00:00"))
        dt_min = (tb - ta).total_seconds() / 60
        # only count area above baseline
        va = max(0, a["value"] - baseline)
        vb = max(0, b["value"] - baseline)
        auc += (va + vb) / 2 * dt_min

    return {
        "peak": peak_value,
        "delta": peak_value - baseline,
        "time_to_peak_min": int((peak_t - t0).total_seconds() / 60),
        "auc_above_baseline": round(auc, 1),
    }
