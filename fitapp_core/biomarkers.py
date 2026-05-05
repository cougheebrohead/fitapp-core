"""Biomarker anomaly detection + recovery / readiness score.

Two pure-stdlib helpers that the member apps + trainer dashboards
key off of:

  biomarker_anomalies(readings, baseline=None)
      Flag readings that deviate >2σ from a rolling baseline. Used to
      surface "your resting HR jumped 12% — possible illness" or
      "weight dropped 4 lb in 5 days — possible aggressive deficit"
      notices. No ML — defensible, regulator-friendly statistics.

  recovery_score(hrv_ms, sleep_hours, resting_hr, baseline_hr,
                 training_load_7d=None, days_since_rest=None)
      Returns a 0–100 readiness score plus a tier
      (peak / ready / moderate / depleted / drained).

Why deterministic math instead of ML: clinical surfaces (Vitalstack)
need to defend recommendations to auditors; consumer surfaces don't
need a black box for what's a 6-variable problem.
"""

from __future__ import annotations

from typing import Literal, TypedDict


# ─── biomarker anomaly detection ──────────────────────────────────────


class Reading(TypedDict):
    timestamp: str       # ISO 8601
    metric: str          # 'resting_hr' | 'hrv_ms' | 'weight_kg' | 'sleep_h' | 'systolic_bp' | 'diastolic_bp'
    value: float


class BaselineStats(TypedDict):
    mean: float
    std: float
    n: int


class Anomaly(TypedDict):
    timestamp: str
    metric: str
    value: float
    baseline_mean: float
    z_score: float                                # how many σ from baseline
    severity: Literal["mild", "moderate", "severe"]
    direction: Literal["high", "low"]
    note: str                                     # short member-facing copy


# Member-facing copy — calibrated to be informative without being
# alarmist. The trainer console gets the same data with clinical
# framing on its side of the wall.
_NOTES: dict[tuple[str, str], str] = {
    ("resting_hr", "high"):     "Resting heart rate is up vs your baseline — could be illness, alcohol, or undertraining recovery.",
    ("resting_hr", "low"):      "Resting heart rate dropped sharply — usually fine if you're trained, but flag if you feel off.",
    ("hrv_ms", "low"):          "HRV is suppressed — your nervous system is under load. Easier session today.",
    ("hrv_ms", "high"):         "HRV is unusually elevated — green light to push if you wanted a hard session.",
    ("weight_kg", "low"):       "Weight dropped fast — make sure protein + hydration are dialed in.",
    ("weight_kg", "high"):      "Weight spiked — most likely sodium / glycogen, not fat. Flag if it persists 5+ days.",
    ("sleep_h", "low"):         "Sleep is short vs your usual — recovery + appetite signals will be off today.",
    ("systolic_bp", "high"):    "Systolic BP is elevated — single reading isn't diagnostic. Re-check tonight.",
    ("diastolic_bp", "high"):   "Diastolic BP is elevated — re-check at rest, flag your trainer if it stays up.",
}


def _baseline(readings: list[float]) -> BaselineStats:
    """Mean + std for a rolling baseline. Caller passes the prior
    14 days. With <3 readings we return zeros so the anomaly check
    silently no-ops (don't alarm before we have data)."""
    n = len(readings)
    if n < 3:
        return {"mean": 0.0, "std": 0.0, "n": n}
    mean = sum(readings) / n
    variance = sum((r - mean) ** 2 for r in readings) / n
    return {"mean": round(mean, 2), "std": round(variance ** 0.5, 2), "n": n}


def _severity(z: float) -> Literal["mild", "moderate", "severe"]:
    az = abs(z)
    if az >= 3.5:
        return "severe"
    if az >= 2.5:
        return "moderate"
    return "mild"


def biomarker_anomalies(
    readings: list[Reading],
    baseline_readings: dict[str, list[float]] | None = None,
    z_threshold: float = 2.0,
) -> list[Anomaly]:
    """Return a list of anomalies for the provided readings.

    `baseline_readings` is keyed by metric and holds the 14-day prior
    history per metric. If omitted, we auto-build a baseline from the
    first 80% of `readings` and check the last 20% — this is convenient
    for retrospective batch analysis.
    """
    if not readings:
        return []

    # If no explicit baseline, split the input
    if baseline_readings is None:
        sorted_readings = sorted(readings, key=lambda r: r["timestamp"])
        cutoff = int(len(sorted_readings) * 0.8)
        baseline_readings = {}
        check_readings = sorted_readings[cutoff:]
        for r in sorted_readings[:cutoff]:
            baseline_readings.setdefault(r["metric"], []).append(float(r["value"]))
    else:
        check_readings = readings

    out: list[Anomaly] = []
    for r in check_readings:
        history = baseline_readings.get(r["metric"], [])
        bl = _baseline(history)
        if bl["std"] == 0.0:
            continue  # not enough baseline yet
        z = (float(r["value"]) - bl["mean"]) / bl["std"]
        if abs(z) < z_threshold:
            continue
        direction: Literal["high", "low"] = "high" if z > 0 else "low"
        out.append({
            "timestamp": r["timestamp"],
            "metric": r["metric"],
            "value": float(r["value"]),
            "baseline_mean": bl["mean"],
            "z_score": round(z, 2),
            "severity": _severity(z),
            "direction": direction,
            "note": _NOTES.get((r["metric"], direction),
                               f"{r['metric']} is unusually {direction} — flag your trainer if it persists."),
        })
    return out


# ─── recovery / readiness score ───────────────────────────────────────


class RecoveryResult(TypedDict):
    score: int                    # 0–100
    tier: Literal["peak", "ready", "moderate", "depleted", "drained"]
    factors: dict[str, float]     # individual sub-scores 0–100
    advice: str                   # short single-line member copy


def _tier(score: int) -> Literal["peak", "ready", "moderate", "depleted", "drained"]:
    if score >= 85:
        return "peak"
    if score >= 70:
        return "ready"
    if score >= 50:
        return "moderate"
    if score >= 30:
        return "depleted"
    return "drained"


_ADVICE: dict[str, str] = {
    "peak":      "Green light. Today's the day to push the hard session.",
    "ready":     "Solid recovery. Train as planned.",
    "moderate":  "Train, but pull back intensity 10–15%. Save the big effort for tomorrow.",
    "depleted":  "Active recovery only. Walk, mobility, light technique work.",
    "drained":   "Rest day. Sleep early. Hydrate. Real food.",
}


def recovery_score(
    *,
    hrv_ms: float | None = None,            # last night, RMSSD ms
    sleep_hours: float | None = None,        # last night
    resting_hr: float | None = None,         # last night, bpm
    baseline_resting_hr: float | None = None,
    training_load_7d: float | None = None,   # arbitrary 0–100; caller defines
    days_since_rest: int | None = None,
) -> RecoveryResult:
    """Compose a 0–100 readiness score from up to 5 inputs.

    Each input contributes a sub-score (0–100). The final score is the
    weighted average of whichever inputs were provided. Missing inputs
    are skipped (we don't penalize people who don't have a Whoop).

    Weights tuned for the typical client mix:
        HRV:      30  (most predictive of CNS state)
        Sleep:    25
        RHR Δ:    20
        Load:     15
        Days off: 10
    """
    factors: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    if hrv_ms is not None:
        # Normalize: <30 = 0, >80 = 100, linear in between (population-level
        # heuristic; trainer can override with personalized baseline later).
        hrv_score = max(0.0, min(100.0, (hrv_ms - 30) / 50 * 100))
        factors["hrv"] = round(hrv_score, 1)
        weighted_sum += hrv_score * 30
        weight_total += 30

    if sleep_hours is not None:
        # 7.5h = perfect; 4h = floor; 9h = perfect.
        if 7.0 <= sleep_hours <= 9.5:
            sleep_score = 100.0
        elif sleep_hours < 7.0:
            sleep_score = max(0.0, (sleep_hours - 4) / 3 * 100)
        else:
            sleep_score = max(60.0, 100 - (sleep_hours - 9.5) * 8)  # over-sleeping mild penalty
        factors["sleep"] = round(sleep_score, 1)
        weighted_sum += sleep_score * 25
        weight_total += 25

    if resting_hr is not None and baseline_resting_hr is not None and baseline_resting_hr > 0:
        # Δ above baseline = bad. -5 bpm = peak (100), 0 = good (90),
        # +5 = moderate (60), +10+ = drained (0).
        delta = resting_hr - baseline_resting_hr
        if delta <= -3:
            rhr_score = 100.0
        elif delta <= 0:
            rhr_score = 90.0 - delta * 3  # -3..0 maps 100..90
        elif delta <= 10:
            rhr_score = max(0.0, 90 - delta * 9)
        else:
            rhr_score = 0.0
        factors["resting_hr"] = round(rhr_score, 1)
        weighted_sum += rhr_score * 20
        weight_total += 20

    if training_load_7d is not None:
        # 0 = under-trained (75), 50 = sweet spot (95), 100 = overreached (40)
        if training_load_7d <= 50:
            load_score = 75 + (training_load_7d / 50) * 20
        else:
            load_score = max(0.0, 95 - (training_load_7d - 50) * 1.1)
        factors["load_balance"] = round(load_score, 1)
        weighted_sum += load_score * 15
        weight_total += 15

    if days_since_rest is not None:
        # 0–1 = fresh, 4 = ok, 7+ = needs a day off
        if days_since_rest <= 1:
            days_score = 100.0
        elif days_since_rest <= 4:
            days_score = 90.0 - (days_since_rest - 1) * 5
        else:
            days_score = max(0.0, 75 - (days_since_rest - 4) * 12)
        factors["days_since_rest"] = round(days_score, 1)
        weighted_sum += days_score * 10
        weight_total += 10

    if weight_total == 0:
        # No inputs at all
        return {
            "score": 0,
            "tier": "moderate",
            "factors": {},
            "advice": "Connect a wearable or log sleep + HRV to start seeing your readiness score.",
        }

    score = int(round(weighted_sum / weight_total))
    score = max(0, min(100, score))
    tier = _tier(score)
    return {
        "score": score,
        "tier": tier,
        "factors": factors,
        "advice": _ADVICE[tier],
    }
