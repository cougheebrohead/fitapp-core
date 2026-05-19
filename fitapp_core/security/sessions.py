"""Session-management primitives — token mint, cookie I/O, bearer parse.

DB-agnostic and stdlib-only, in the same shape as `passwords.py` and
`ratelimit.py`: pure functions, no global state, no I/O beyond what
the caller wires in.

Consolidated from the three sibling apps (FitApp / elh-coach /
elh-health) which all converged on:

  * `secrets.token_urlsafe(32)` for the opaque session token
  * SHA-256 of the raw token as the at-rest fingerprint
  * 30-day default TTL
  * `Authorization: Bearer <token>` for transport
  * `Secure; HttpOnly; SameSite=Lax` cookie defaults when the app
    bothers to set a cookie (most ride pure bearer)

Anything tied to an app's schema (tenant_id / org_id / sid /
refresh-token rotation / user_sessions row shape) stays in that
app's `auth.py`. This module is the mechanical chassis only.

Public API:

    new_session_token()                 -> str
    hash_token(token)                   -> str  (sha256 hex; never log raw)
    default_ttl_seconds()               -> int  (env SESSION_TTL_SECONDS or 30d)
    bearer_from_authorization(header)   -> str | None
    SessionCookie(name=..., ...)
        .to_set_cookie_header(token, ttl_seconds, ...) -> str
    parse_cookie_header(header, name)   -> str | None
"""
from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from http import cookies as _http_cookies
from typing import Literal

# 30 days — what FitApp's TOKEN_TTL and elh-{coach,health}'s
# SESSION_TTL_DAYS both settled on. Keep this here so the value lives
# in exactly one place.
_DEFAULT_TTL_SECONDS = 30 * 24 * 3600

# Token length is 32 raw bytes → 43 chars url-safe-base64. Matches
# what every sibling app already mints; we keep the call site shape
# identical so existing token-length checks and DB columns don't
# need to widen.
_TOKEN_NBYTES = 32

SameSite = Literal["Strict", "Lax", "None"]


def new_session_token() -> str:
    """Mint a cryptographically random, URL-safe opaque session token.

    Returns a 43-character base64url string (no padding). Identical
    construction to `secrets.token_urlsafe(32)` so the resulting
    tokens are byte-for-byte equivalent to what the sibling apps
    were minting before consolidation.
    """
    return secrets.token_urlsafe(_TOKEN_NBYTES)


def hash_token(token: str) -> str:
    """SHA-256 the raw token, return hex.

    Use this everywhere you'd otherwise store the raw token. The raw
    token only ever lives in transit (response body or cookie);
    the DB row holds the hash so a stolen DB dump can't impersonate.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def default_ttl_seconds() -> int:
    """Session TTL in seconds.

    Reads env `SESSION_TTL_SECONDS` if set and parseable as a
    positive int; otherwise returns 30 days (the value all three
    sibling apps converged on before consolidation).
    """
    raw = os.environ.get("SESSION_TTL_SECONDS")
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _DEFAULT_TTL_SECONDS


def bearer_from_authorization(header: str | None) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header.

    Case-insensitive on the scheme keyword (RFC 7235 §2.1 says scheme
    is case-insensitive). Returns None on missing header, wrong
    scheme, or empty token. Strips surrounding whitespace from the
    token value.
    """
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, value = parts[0], parts[1].strip()
    if scheme.lower() != "bearer":
        return None
    if not value:
        return None
    return value


@dataclass(frozen=True)
class SessionCookie:
    """Configuration for a session-shaped cookie.

    Encapsulates the cookie name plus the default attribute set so
    callers don't have to remember Secure/HttpOnly/SameSite on every
    write. Defaults match the safest sensible profile for a session:
    HttpOnly (no JS access), Secure (TLS only), SameSite=Lax (spec
    default; survives top-level navigations needed for OAuth bounces
    while still blocking cross-site POST).
    """

    name: str
    path: str = "/"
    secure: bool = True
    http_only: bool = True
    same_site: SameSite = "Lax"
    domain: str | None = None

    def to_set_cookie_header(
        self,
        token: str,
        ttl_seconds: int | None = None,
        *,
        secure: bool | None = None,
        same_site: SameSite | None = None,
        domain: str | None = None,
        path: str | None = None,
    ) -> str:
        """Build the Set-Cookie line for `token`.

        `ttl_seconds=None` mints a session cookie (no Max-Age — the
        browser drops it on close). `ttl_seconds=0` deletes the
        cookie (browsers treat Max-Age=0 as immediate expiry).

        Any keyword override beats the dataclass default for that
        single call — useful for, e.g., temporarily relaxing
        SameSite for an OAuth-callback round-trip.
        """
        if same_site is None:
            same_site = self.same_site
        if same_site not in ("Strict", "Lax", "None"):
            raise ValueError(f"invalid same_site: {same_site!r}")
        eff_secure = self.secure if secure is None else secure
        if same_site == "None" and not eff_secure:
            # Browsers reject SameSite=None without Secure. Refuse
            # to emit a header the UA will silently drop.
            raise ValueError("SameSite=None requires Secure=True")
        eff_domain = self.domain if domain is None else domain
        eff_path = self.path if path is None else path

        parts = [f"{self.name}={token}", f"Path={eff_path}"]
        if ttl_seconds is not None:
            parts.append(f"Max-Age={int(ttl_seconds)}")
        if eff_domain:
            parts.append(f"Domain={eff_domain}")
        if eff_secure:
            parts.append("Secure")
        if self.http_only:
            parts.append("HttpOnly")
        parts.append(f"SameSite={same_site}")
        return "; ".join(parts)

    def to_delete_header(self) -> str:
        """Build the Set-Cookie line that clears this cookie.

        Browsers honor any cookie with Max-Age=0 + matching Path.
        Equivalent to `to_set_cookie_header("", 0)` but reads more
        clearly at the call site.
        """
        return self.to_set_cookie_header("", 0)


def parse_cookie_header(header: str | None, name: str) -> str | None:
    """Pull a single cookie value out of a raw `Cookie:` header.

    Uses stdlib `http.cookies.SimpleCookie` so we get RFC 6265
    handling for quoted values, escape sequences, and malformed
    pairs (which it silently skips rather than raising). Returns
    None when the header is missing, the cookie isn't present, or
    the value is empty.
    """
    if not header or not name:
        return None
    try:
        jar: _http_cookies.SimpleCookie = _http_cookies.SimpleCookie()
        jar.load(header)
    except Exception:
        return None
    morsel = jar.get(name)
    if morsel is None:
        return None
    val = morsel.value
    return val if val else None


__all__ = [
    "new_session_token",
    "hash_token",
    "default_ttl_seconds",
    "bearer_from_authorization",
    "SessionCookie",
    "parse_cookie_header",
]
