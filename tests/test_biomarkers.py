"""Tests for fitapp_core.biomarkers — anomaly detection + recovery score."""
from __future__ import annotations

from fitapp_core import biomarker_anomalies, recovery_score


def test_no_anomalies_when_within_normal_range():
    readings = [
        {"timestamp": "2026-05-01T07:00:00Z", "metric": "resting_hr", "value": 60},
        {"timestamp": "2026-05-02T07:00:00Z", "metric": "resting_hr", "value": 61},
        {"timestamp": "2026-05-03T07:00:00Z", "metric": "resting_hr", "value": 59},
        {"timestamp": "2026-05-04T07:00:00Z", "metric": "resting_hr", "value": 60},
    ]
    out = biomarker_anomalies(readings)
    assert out == []


def test_high_resting_hr_flagged():
    # 8 days of baseline at ~60, then a day at 80 (massive spike)
    readings = [
        {"timestamp": f"2026-05-{i:02d}T07:00:00Z", "metric": "resting_hr", "value": 60 + (i % 2)}
        for i in range(1, 9)
    ]
    readings.append({"timestamp": "2026-05-09T07:00:00Z", "metric": "resting_hr", "value": 80})
    out = biomarker_anomalies(readings)
    assert len(out) == 1
    assert out[0]["metric"] == "resting_hr"
    assert out[0]["direction"] == "high"
    assert out[0]["z_score"] > 2
    assert "heart rate is up" in out[0]["note"].lower()


def test_explicit_baseline():
    bl = {"hrv_ms": [50, 52, 48, 51, 49, 50, 51, 50, 49, 52]}
    today = [{"timestamp": "2026-05-10T07:00:00Z", "metric": "hrv_ms", "value": 25}]
    out = biomarker_anomalies(today, baseline_readings=bl)
    assert len(out) == 1
    assert out[0]["direction"] == "low"
    assert out[0]["severity"] in ("moderate", "severe")


def test_recovery_score_full_inputs():
    r = recovery_score(
        hrv_ms=68, sleep_hours=8.0, resting_hr=58,
        baseline_resting_hr=60, training_load_7d=45, days_since_rest=2,
    )
    assert 70 <= r["score"] <= 100
    assert r["tier"] in ("peak", "ready")
    assert all(k in r["factors"] for k in ("hrv","sleep","resting_hr","load_balance","days_since_rest"))


def test_recovery_score_drained():
    r = recovery_score(
        hrv_ms=28, sleep_hours=4.5, resting_hr=72,
        baseline_resting_hr=60, training_load_7d=92, days_since_rest=8,
    )
    assert r["score"] <= 50
    assert r["tier"] in ("drained", "depleted", "moderate")


def test_recovery_score_partial_inputs():
    # Only sleep + HRV (typical free-tier user)
    r = recovery_score(hrv_ms=60, sleep_hours=7.5)
    assert r["score"] > 0
    assert "hrv" in r["factors"]
    assert "sleep" in r["factors"]
    assert "resting_hr" not in r["factors"]


def test_recovery_score_no_inputs():
    r = recovery_score()
    assert r["score"] == 0
    assert "wearable" in r["advice"].lower() or "sleep" in r["advice"].lower()


def test_recovery_score_advice_per_tier():
    peak = recovery_score(hrv_ms=85, sleep_hours=8.5, resting_hr=55, baseline_resting_hr=58)
    assert peak["tier"] in ("peak", "ready")
    assert len(peak["advice"]) > 0


def test_short_baseline_no_false_positives():
    # Only 2 readings — not enough baseline to compute std
    readings = [
        {"timestamp": "2026-05-01T07:00:00Z", "metric": "resting_hr", "value": 60},
        {"timestamp": "2026-05-02T07:00:00Z", "metric": "resting_hr", "value": 100},
    ]
    out = biomarker_anomalies(readings)
    assert out == []
