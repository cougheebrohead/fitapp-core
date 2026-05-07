"""CSRF protection — double-submit cookie pattern.

The double-submit pattern works because:
  1. Server mints a random token, stores it in an HttpOnly cookie AND
     returns it in the response body (or in a `X-CSRF-Token` header).
  2. Client JS reads the body/header value, includes it on every
     state-changing request as a `X-CSRF-Token` header.
  3. Server compares the cookie value to the header value with
     constant-time equality.

A CSRF attacker can send a forged POST but cannot read the cookie
(SameSite=Strict + HttpOnly). Without the matching header, the
request is rejected.

Usage:

    # Login response: mint + set cookie + return in body
    token = csrf_token()
    headers["Set-Cookie"] = cookie_writer("csrf", token, http_only=False)
    return {"ok": True, "csrf": token}

    # Every state-changing request:
    cookie_value = parse cookie header for "csrf"
    header_value = self.headers.get("X-CSRF-Token", "")
    if not verify_csrf(cookie_value, header_value):
        return self._j({"error": "csrf_invalid"}, 403)
"""
from __future__ import annotations

import hmac
import secrets


def csrf_token() -> str:
    """Mint a fresh CSRF token. URL-safe, 32 bytes of entropy."""
    return secrets.token_urlsafe(32)


def verify_csrf(cookie_value: str | None, header_value: str | None) -> bool:
    """Compare cookie token to header token in constant time.

    Returns True only on exact match. Returns False if either is
    missing, empty, or non-matching. Never raises.

    Length-mismatch returns False but is still constant-time within
    the matched-length prefix to avoid leaking length via timing.
    """
    if not cookie_value or not header_value:
        return False
    a = cookie_value.encode("utf-8")
    b = header_value.encode("utf-8")
    if len(a) != len(b):
        # hmac.compare_digest tolerates different lengths but returns
        # False — and it's constant-time within the bytes compared.
        return False
    return hmac.compare_digest(a, b)
