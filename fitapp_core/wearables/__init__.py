"""Wearable provider integrations.

Each submodule exposes the same shape so a single dispatcher can wire
them up uniformly:

    authorize_url(client_id, redirect_uri, state, code_verifier, scopes) -> str
    exchange_code(client_id, client_secret, code, redirect_uri, code_verifier) -> dict
    refresh_token(client_id, client_secret, refresh_token) -> dict
    fetch_window(access_token, since_iso, until_iso) -> list[dict]   # normalized rows
    verify_webhook(token, headers, body_bytes) -> bool

`fetch_window` returns rows in fitapp-core's normalized biometrics shape:
    {
      "recorded_at": "2026-05-06T07:42:00Z",
      "source":      "oura",
      "fields":      {"hrv_rmssd_ms": 48.2, "resting_hr": 54, ...},
      "external_id": "<provider-side row id, used for idempotent upsert>"
    }
"""
from . import oura

__all__ = ["oura"]
