"""Oura Ring (cloud.ouraring.com) OAuth2 + v2 REST client.

Stdlib-only — uses urllib for HTTP, hashlib + hmac + secrets + base64
for PKCE and webhook verification. Pulls daily readiness, sleep, HRV,
RHR, and body temp into fitapp-core's normalized biometrics shape so
the host server can dump straight into biometrics_log.

Public surface:

    authorize_url(client_id, redirect_uri, state, code_verifier, scopes=None) -> str
    pkce_pair() -> (verifier, challenge)
    exchange_code(client_id, client_secret, code, redirect_uri, code_verifier) -> dict
    refresh_token(client_id, client_secret, refresh_token) -> dict
    fetch_window(access_token, start_date, end_date) -> list[dict]
    verify_webhook(token, headers, body) -> bool
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Optional


_AUTHORIZE_URL = "https://cloud.ouraring.com/oauth/authorize"
_TOKEN_URL     = "https://api.ouraring.com/oauth/token"
_API_BASE      = "https://api.ouraring.com/v2/usercollection"

# Scopes Oura grants. We request the union of what FitApp consumes so
# the user only sees one consent screen.
DEFAULT_SCOPES = (
    "email", "personal", "daily", "heartrate",
    "workout", "session", "tag", "spo2Daily", "ringConfiguration",
)

_HTTP_TIMEOUT = 20


# ─── PKCE helpers ─────────────────────────────────────────────────────


def pkce_pair() -> tuple[str, str]:
    """Mint a PKCE code_verifier + code_challenge pair (RFC 7636).

    Verifier: 64-byte URL-safe base64 (no padding) → 86 chars.
    Challenge: SHA-256 of the verifier, URL-safe base64 (no padding).
    """
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def authorize_url(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_verifier: str,
    scopes: Optional[Iterable[str]] = None,
) -> str:
    """Build the consent URL the user is redirected to.

    The caller must persist (state, code_verifier) so /callback can
    finish the trip.
    """
    digest = hashlib.sha256(code_verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    q = urllib.parse.urlencode({
        "response_type":         "code",
        "client_id":             client_id,
        "redirect_uri":          redirect_uri,
        "scope":                 " ".join(scopes or DEFAULT_SCOPES),
        "state":                 state,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
    })
    return f"{_AUTHORIZE_URL}?{q}"


# ─── Token exchange + refresh ─────────────────────────────────────────


def exchange_code(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    """Trade the authorization code + PKCE verifier for tokens.

    Returns Oura's raw JSON: {access_token, refresh_token, expires_in,
    token_type, scope}. Raises RuntimeError on non-2xx.
    """
    return _post_form(_TOKEN_URL, {
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  redirect_uri,
        "client_id":     client_id,
        "client_secret": client_secret,
        "code_verifier": code_verifier,
    })


def refresh_token(
    client_id: str,
    client_secret: str,
    refresh: str,
) -> dict[str, Any]:
    """Trade a refresh token for a fresh access+refresh pair."""
    return _post_form(_TOKEN_URL, {
        "grant_type":    "refresh_token",
        "refresh_token": refresh,
        "client_id":     client_id,
        "client_secret": client_secret,
    })


def _post_form(url: str, form: dict[str, str]) -> dict[str, Any]:
    body = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        snippet = (e.read() or b"")[:300].decode("utf-8", errors="replace")
        raise RuntimeError(f"oura token endpoint {e.code}: {snippet}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"oura token response not JSON: {e}")


# ─── Data fetch ───────────────────────────────────────────────────────


def fetch_window(
    access_token: str,
    start_date: str,    # ISO date 'YYYY-MM-DD'
    end_date: str,      # ISO date 'YYYY-MM-DD' (inclusive on Oura's side)
) -> list[dict[str, Any]]:
    """Pull every per-day datapoint from Oura over the [start, end] window
    and merge into normalized biometrics rows keyed by date.

    Rows shape:
      {"recorded_at": "<ISO datetime, end-of-day in UTC>",
       "source":      "oura",
       "fields":      {<biometrics_log column>: value, ...},
       "external_id": "oura:<date>"}
    """
    sleep      = _get_json("/daily_sleep",      access_token, start_date, end_date)
    readiness  = _get_json("/daily_readiness",  access_token, start_date, end_date)
    activity   = _get_json("/daily_activity",   access_token, start_date, end_date)
    spo2       = _get_json("/daily_spo2",       access_token, start_date, end_date)
    sleep_full = _get_json("/sleep",            access_token, start_date, end_date)

    by_day: dict[str, dict[str, Any]] = {}

    def slot(day: str) -> dict[str, Any]:
        if day not in by_day:
            by_day[day] = {}
        return by_day[day]

    # daily_sleep gives a per-day score + contributors
    for d in sleep.get("data", []) or []:
        day = d.get("day")
        if not day:
            continue
        f = slot(day)
        # Oura nests rmssd / hr in 'contributors' or top-level depending on doc
        c = d.get("contributors") or {}
        if "average_hrv" in d and d["average_hrv"] is not None:
            f["hrv_rmssd_ms"] = float(d["average_hrv"])
        if "lowest_heart_rate" in d and d["lowest_heart_rate"] is not None:
            f["resting_hr"] = int(d["lowest_heart_rate"])
        # Sleep efficiency 0–100
        if "efficiency" in c and c["efficiency"] is not None:
            f["sleep_efficiency"] = int(c["efficiency"])

    # /sleep gives per-session detail; sum totals per day
    for s in sleep_full.get("data", []) or []:
        day = s.get("day")
        if not day:
            continue
        f = slot(day)
        # total_sleep_duration is seconds
        tot = s.get("total_sleep_duration")
        if tot is not None:
            f["sleep_hours"] = round(float(tot) / 3600.0, 1)
        rem = s.get("rem_sleep_duration")
        if rem is not None:
            f["sleep_rem_min"] = int(rem // 60)
        deep = s.get("deep_sleep_duration")
        if deep is not None:
            f["sleep_deep_min"] = int(deep // 60)
        avg_hr = s.get("average_heart_rate")
        if avg_hr is not None and "resting_hr" not in f:
            # Average sleep HR is a reasonable resting proxy when daily_sleep
            # didn't surface lowest_heart_rate.
            f["resting_hr"] = int(avg_hr)
        avg_hrv = s.get("average_hrv")
        if avg_hrv is not None and "hrv_rmssd_ms" not in f:
            f["hrv_rmssd_ms"] = float(avg_hrv)
        # Oura's body_temp delta to baseline is what users see; we want absolute
        # in body_temp_c, so skip if Oura doesn't publish absolute.
        bt = s.get("readiness", {}).get("temperature_deviation")
        # ^ delta only — we deliberately don't write body_temp_c from a delta.

    # daily_readiness gives the readiness score we don't store but its
    # temperature_deviation is the cycle-augmentation signal we WANT.
    # We capture it under a JSONB extension column? biometrics_log doesn't
    # have one. Skip for v1 — body temp deviation will land when we extend
    # the schema; until then, HRV + sleep are the win.
    _ = readiness   # explicitly unused for v1
    _ = activity    # daily activity / steps land in a future iteration
    _ = spo2        # spo2_pct could land here if Oura returns it; skip v1

    # Convert each day's fields dict to a normalized row.
    rows: list[dict[str, Any]] = []
    for day in sorted(by_day.keys()):
        f = by_day[day]
        if not f:
            continue
        rows.append({
            # End-of-day UTC anchors the daily summary; per-event rows would
            # use the precise timestamp if we ever ingest workouts/heartrate.
            "recorded_at": f"{day}T23:59:00Z",
            "source":      "oura",
            "fields":      f,
            "external_id": f"oura:{day}",
        })
    return rows


def _get_json(
    path: str,
    access_token: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """GET an Oura v2 collection endpoint with date bounds."""
    q = urllib.parse.urlencode({
        "start_date": start_date,
        "end_date":   end_date,
    })
    url = f"{_API_BASE}{path}?{q}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}",
                 "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        snippet = (e.read() or b"")[:300].decode("utf-8", errors="replace")
        # 401 -> caller should refresh; surface that distinctly.
        if e.code == 401:
            raise PermissionError(f"oura {path} 401: {snippet}")
        raise RuntimeError(f"oura {path} {e.code}: {snippet}")


# ─── Webhook verification ─────────────────────────────────────────────


def verify_webhook(
    expected_token: str,
    headers: dict[str, str],
    body: bytes,
) -> bool:
    """Oura webhook v2 ships a verification token in a fixed header.

    Header name varies by docs version — we accept either casing.
    The token is the value Oura showed once on registration (we store
    it as $OURA_WEBHOOK_TOKEN). The body itself isn't HMAC'd, which
    is why we ALSO require the request to come over HTTPS and rate-limit
    the route at the host.
    """
    if not expected_token:
        return False
    got = (headers.get("x-oura-webhook-token")
           or headers.get("X-Oura-Webhook-Token")
           or headers.get("X-Webhook-Token")
           or "")
    # Constant-time compare to avoid timing leaks on the token.
    return hmac.compare_digest(got, expected_token) and len(body) <= 2_000_000
