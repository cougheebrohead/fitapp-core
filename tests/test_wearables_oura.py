"""Unit tests for fitapp_core.wearables.oura.

We don't hit the real Oura API — these tests exercise the pure-Python
helpers (PKCE, authorize URL shape, webhook verifier) and a stubbed
fetch_window via http stubbing. Network-touching helpers (exchange_code,
refresh_token, fetch_window) get stubbed at the urllib boundary.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import urllib.parse
from unittest.mock import patch

from fitapp_core.wearables import oura


# ─── PKCE ─────────────────────────────────────────────────────────────


def test_pkce_pair_lengths():
    v, c = oura.pkce_pair()
    assert len(v) == 86      # 64 bytes -> 86 url-safe-b64 chars (no padding)
    assert len(c) == 43      # SHA-256 digest -> 43 url-safe-b64 chars


def test_pkce_challenge_matches_verifier():
    v, c = oura.pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
    assert c == expected


def test_pkce_verifier_unique():
    seen = {oura.pkce_pair()[0] for _ in range(20)}
    assert len(seen) == 20    # 64 bytes of randomness => collision-free in practice


# ─── authorize_url ────────────────────────────────────────────────────


def test_authorize_url_shape():
    v, _ = oura.pkce_pair()
    url = oura.authorize_url("CID", "https://app/cb", "state-abc", v)
    assert url.startswith("https://cloud.ouraring.com/oauth/authorize?")
    qs = dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))
    assert qs["client_id"] == "CID"
    assert qs["redirect_uri"] == "https://app/cb"
    assert qs["state"] == "state-abc"
    assert qs["response_type"] == "code"
    assert qs["code_challenge_method"] == "S256"
    # challenge must be deterministic from verifier
    expected = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
    assert qs["code_challenge"] == expected


def test_authorize_url_default_scopes():
    v, _ = oura.pkce_pair()
    url = oura.authorize_url("CID", "https://x/cb", "s", v)
    qs = dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))
    for s in ("email", "personal", "daily", "heartrate"):
        assert s in qs["scope"]


def test_authorize_url_custom_scopes():
    v, _ = oura.pkce_pair()
    url = oura.authorize_url("CID", "https://x/cb", "s", v, scopes=["email", "daily"])
    qs = dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))
    assert qs["scope"] == "email daily"


# ─── Webhook verifier ─────────────────────────────────────────────────


def test_webhook_empty_token_rejects():
    assert oura.verify_webhook("", {"X-Oura-Webhook-Token": "x"}, b"") is False


def test_webhook_match():
    assert oura.verify_webhook("secret", {"X-Oura-Webhook-Token": "secret"}, b"{}") is True


def test_webhook_match_lowercase_header():
    assert oura.verify_webhook("secret", {"x-oura-webhook-token": "secret"}, b"{}") is True


def test_webhook_mismatch():
    assert oura.verify_webhook("secret", {"X-Oura-Webhook-Token": "WRONG"}, b"{}") is False


def test_webhook_oversized_body_rejects():
    assert oura.verify_webhook("s", {"X-Oura-Webhook-Token": "s"}, b"X" * 3_000_000) is False


def test_webhook_missing_header():
    assert oura.verify_webhook("s", {}, b"{}") is False


# ─── fetch_window normalizer ─────────────────────────────────────────
# We stub _get_json so the test doesn't touch the network. The test
# proves that fetch_window correctly merges multi-endpoint responses
# into one row per day with the right field mapping.


def _stub_responses(mapping):
    """Return a function that mimics _get_json by path."""
    def _stub(path, access_token, start_date, end_date):
        return mapping.get(path, {"data": []})
    return _stub


def test_fetch_window_merges_endpoints():
    sleep_resp = {
        "data": [{
            "day": "2026-05-05",
            "average_hrv": 48.7,
            "lowest_heart_rate": 51,
            "contributors": {"efficiency": 91},
        }],
    }
    sleep_full_resp = {
        "data": [{
            "day": "2026-05-05",
            "total_sleep_duration": 27000,   # seconds = 7.5 h
            "rem_sleep_duration": 5400,      # 90 min
            "deep_sleep_duration": 4200,     # 70 min
            "average_heart_rate": 56,
            "average_hrv": 47.0,
            "readiness": {"temperature_deviation": -0.1},
        }],
    }
    mapping = {
        "/daily_sleep":     sleep_resp,
        "/sleep":           sleep_full_resp,
        "/daily_readiness": {"data": []},
        "/daily_activity":  {"data": []},
        "/daily_spo2":      {"data": []},
    }
    with patch.object(oura, "_get_json", side_effect=_stub_responses(mapping)):
        rows = oura.fetch_window("tok", "2026-05-04", "2026-05-05")

    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "oura"
    assert r["external_id"] == "oura:2026-05-05"
    assert r["recorded_at"] == "2026-05-05T23:59:00Z"
    f = r["fields"]
    # daily_sleep wins for HRV + RHR (covered before /sleep fallback)
    assert f["hrv_rmssd_ms"] == 48.7
    assert f["resting_hr"] == 51
    # /sleep contributes durations + efficiency
    assert f["sleep_hours"] == 7.5
    assert f["sleep_rem_min"] == 90
    assert f["sleep_deep_min"] == 70
    assert f["sleep_efficiency"] == 91


def test_fetch_window_falls_back_to_sleep_endpoint_for_hr():
    """When /daily_sleep doesn't surface lowest_heart_rate, the average
    from /sleep is the resting proxy."""
    mapping = {
        "/daily_sleep":     {"data": [{"day": "2026-05-05"}]},
        "/sleep":           {"data": [{
            "day": "2026-05-05",
            "average_heart_rate": 58,
            "average_hrv": 45.0,
            "total_sleep_duration": 25200,
        }]},
        "/daily_readiness": {"data": []},
        "/daily_activity":  {"data": []},
        "/daily_spo2":      {"data": []},
    }
    with patch.object(oura, "_get_json", side_effect=_stub_responses(mapping)):
        rows = oura.fetch_window("tok", "2026-05-05", "2026-05-05")
    f = rows[0]["fields"]
    assert f["resting_hr"] == 58
    assert f["hrv_rmssd_ms"] == 45.0
    assert f["sleep_hours"] == 7.0


def test_fetch_window_drops_empty_days():
    """A day with no extractable fields produces no row."""
    mapping = {
        "/daily_sleep":     {"data": [{"day": "2026-05-05"}]},   # no fields
        "/sleep":           {"data": []},
        "/daily_readiness": {"data": []},
        "/daily_activity":  {"data": []},
        "/daily_spo2":      {"data": []},
    }
    with patch.object(oura, "_get_json", side_effect=_stub_responses(mapping)):
        rows = oura.fetch_window("tok", "2026-05-05", "2026-05-05")
    assert rows == []


def test_fetch_window_sorted_by_day():
    mapping = {
        "/daily_sleep": {"data": [
            {"day": "2026-05-07", "average_hrv": 50, "lowest_heart_rate": 50},
            {"day": "2026-05-05", "average_hrv": 48, "lowest_heart_rate": 52},
            {"day": "2026-05-06", "average_hrv": 49, "lowest_heart_rate": 51},
        ]},
        "/sleep":           {"data": []},
        "/daily_readiness": {"data": []},
        "/daily_activity":  {"data": []},
        "/daily_spo2":      {"data": []},
    }
    with patch.object(oura, "_get_json", side_effect=_stub_responses(mapping)):
        rows = oura.fetch_window("tok", "2026-05-05", "2026-05-07")
    days = [r["external_id"] for r in rows]
    assert days == ["oura:2026-05-05", "oura:2026-05-06", "oura:2026-05-07"]


# ─── Surface re-export ────────────────────────────────────────────────


def test_module_reachable_from_package_root():
    import fitapp_core
    assert hasattr(fitapp_core, "wearables")
    assert fitapp_core.wearables.oura is oura


# ─── runner ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = []
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            fails.append((fn.__name__, e))
            print(f"  ✗ {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(fails)}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
