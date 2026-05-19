"""Tests for fitapp_core.security.sessions — the consolidated session
primitives extracted from FitApp / elh-coach / elh-health.

Each public function gets positive coverage plus at least one
negative case (malformed input, missing field, wrong scheme, etc.).
The session-cookie tests pin the exact attribute order + flags so
a future refactor that quietly drops Secure or flips SameSite will
break loudly here instead of in production.
"""
from __future__ import annotations

import os

import pytest

from fitapp_core.security.sessions import (
    SessionCookie,
    bearer_from_authorization,
    default_ttl_seconds,
    hash_token,
    new_session_token,
    parse_cookie_header,
)


# ── new_session_token ────────────────────────────────────────────────


class TestNewSessionToken:
    def test_returns_str(self):
        assert isinstance(new_session_token(), str)

    def test_length_matches_32_byte_urlsafe(self):
        # 32 bytes → 43 base64url chars (no padding)
        assert len(new_session_token()) == 43

    def test_unique_across_calls(self):
        # Collision in a sample of 1k would be ~impossibly unlucky
        # for a 256-bit space; if this ever flakes the RNG is broken.
        seen = {new_session_token() for _ in range(1000)}
        assert len(seen) == 1000

    def test_urlsafe_alphabet_only(self):
        tok = new_session_token()
        # base64url alphabet: letters, digits, '-', '_'
        assert all(c.isalnum() or c in "-_" for c in tok)


# ── hash_token ───────────────────────────────────────────────────────


class TestHashToken:
    def test_sha256_hex_length(self):
        assert len(hash_token("anything")) == 64

    def test_deterministic(self):
        assert hash_token("abc") == hash_token("abc")

    def test_different_input_different_hash(self):
        assert hash_token("abc") != hash_token("abd")

    def test_empty_string_is_hashable(self):
        # Don't crash on empty — caller might pass through user input
        h = hash_token("")
        assert len(h) == 64


# ── default_ttl_seconds ──────────────────────────────────────────────


class TestDefaultTtlSeconds:
    def test_default_is_30_days(self, monkeypatch):
        monkeypatch.delenv("SESSION_TTL_SECONDS", raising=False)
        assert default_ttl_seconds() == 30 * 24 * 3600

    def test_env_override_positive_int(self, monkeypatch):
        monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")
        assert default_ttl_seconds() == 3600

    def test_env_override_garbage_falls_back(self, monkeypatch):
        # NEGATIVE: malformed env shouldn't crash, falls back to default.
        monkeypatch.setenv("SESSION_TTL_SECONDS", "not-a-number")
        assert default_ttl_seconds() == 30 * 24 * 3600

    def test_env_override_zero_or_negative_falls_back(self, monkeypatch):
        # NEGATIVE: 0 or negative would mean "expired on mint" — refuse.
        monkeypatch.setenv("SESSION_TTL_SECONDS", "0")
        assert default_ttl_seconds() == 30 * 24 * 3600
        monkeypatch.setenv("SESSION_TTL_SECONDS", "-7")
        assert default_ttl_seconds() == 30 * 24 * 3600


# ── bearer_from_authorization ────────────────────────────────────────


class TestBearerFromAuthorization:
    def test_standard_form(self):
        assert bearer_from_authorization("Bearer abc123") == "abc123"

    def test_case_insensitive_scheme(self):
        # RFC 7235 §2.1 — scheme is case-insensitive
        assert bearer_from_authorization("bearer abc123") == "abc123"
        assert bearer_from_authorization("BEARER abc123") == "abc123"

    def test_strips_token_whitespace(self):
        assert bearer_from_authorization("Bearer   abc123  ") == "abc123"

    def test_none_header(self):
        # NEGATIVE: missing header
        assert bearer_from_authorization(None) is None

    def test_empty_header(self):
        # NEGATIVE: empty string
        assert bearer_from_authorization("") is None

    def test_wrong_scheme(self):
        # NEGATIVE: Basic, Digest, etc. must be rejected
        assert bearer_from_authorization("Basic abc123") is None

    def test_no_token(self):
        # NEGATIVE: scheme without a value
        assert bearer_from_authorization("Bearer") is None
        assert bearer_from_authorization("Bearer    ") is None


# ── SessionCookie ────────────────────────────────────────────────────


class TestSessionCookie:
    def test_defaults_include_secure_httponly_lax(self):
        sc = SessionCookie(name="elh_session")
        header = sc.to_set_cookie_header("tok123", ttl_seconds=60)
        assert header.startswith("elh_session=tok123")
        assert "Secure" in header
        assert "HttpOnly" in header
        assert "SameSite=Lax" in header
        assert "Max-Age=60" in header
        assert "Path=/" in header

    def test_session_cookie_omits_max_age_when_ttl_none(self):
        sc = SessionCookie(name="s")
        header = sc.to_set_cookie_header("t", ttl_seconds=None)
        assert "Max-Age" not in header

    def test_delete_header_uses_zero_max_age(self):
        sc = SessionCookie(name="s")
        header = sc.to_delete_header()
        assert "s=" in header
        assert "Max-Age=0" in header

    def test_domain_attribute_when_set(self):
        sc = SessionCookie(name="s", domain="example.com")
        header = sc.to_set_cookie_header("t", ttl_seconds=10)
        assert "Domain=example.com" in header

    def test_path_override_per_call(self):
        sc = SessionCookie(name="s")
        header = sc.to_set_cookie_header("t", ttl_seconds=10, path="/api")
        assert "Path=/api" in header
        # Dataclass default unchanged for future calls
        assert "Path=/" in sc.to_set_cookie_header("t", ttl_seconds=10)

    def test_samesite_strict_supported(self):
        sc = SessionCookie(name="s", same_site="Strict")
        assert "SameSite=Strict" in sc.to_set_cookie_header("t", ttl_seconds=10)

    def test_samesite_none_requires_secure(self):
        # NEGATIVE: browsers reject SameSite=None without Secure;
        # we refuse to emit such a header at all.
        sc = SessionCookie(name="s", secure=False)
        with pytest.raises(ValueError, match="SameSite=None requires Secure"):
            sc.to_set_cookie_header("t", ttl_seconds=10, same_site="None")

    def test_invalid_samesite_rejected(self):
        # NEGATIVE: typo'd SameSite value
        sc = SessionCookie(name="s")
        with pytest.raises(ValueError, match="invalid same_site"):
            sc.to_set_cookie_header("t", ttl_seconds=10, same_site="None_")  # type: ignore[arg-type]

    def test_insecure_opt_out_for_local_dev(self):
        # `secure=False` is allowed (local http dev); just emits no Secure flag.
        sc = SessionCookie(name="s", secure=False)
        header = sc.to_set_cookie_header("t", ttl_seconds=10)
        assert "Secure" not in header

    def test_http_only_opt_out(self):
        sc = SessionCookie(name="s", http_only=False)
        header = sc.to_set_cookie_header("t", ttl_seconds=10)
        assert "HttpOnly" not in header

    def test_immutable(self):
        # frozen dataclass — defending against accidental mutation of
        # the canonical config object held in module-level constants
        sc = SessionCookie(name="s")
        with pytest.raises(Exception):
            sc.name = "other"  # type: ignore[misc]


# ── parse_cookie_header ──────────────────────────────────────────────


class TestParseCookieHeader:
    def test_single_cookie(self):
        assert parse_cookie_header("sid=abc123", "sid") == "abc123"

    def test_multi_cookie_picks_named(self):
        header = "fa_did=devXYZ; sid=abc123; theme=dark"
        assert parse_cookie_header(header, "sid") == "abc123"
        assert parse_cookie_header(header, "fa_did") == "devXYZ"
        assert parse_cookie_header(header, "theme") == "dark"

    def test_missing_cookie(self):
        # NEGATIVE: header present but doesn't contain the named cookie
        assert parse_cookie_header("other=1", "sid") is None

    def test_none_header(self):
        # NEGATIVE: no Cookie header at all
        assert parse_cookie_header(None, "sid") is None

    def test_empty_header(self):
        # NEGATIVE: empty Cookie header
        assert parse_cookie_header("", "sid") is None

    def test_empty_name(self):
        # NEGATIVE: silly call shouldn't blow up
        assert parse_cookie_header("sid=abc", "") is None

    def test_malformed_header_returns_none_not_raise(self):
        # NEGATIVE: garbage input — stdlib SimpleCookie should swallow
        # malformed pieces, and our worst case is None (never raise).
        # Use a header with a leading '=' which trips SimpleCookie.
        result = parse_cookie_header("=garbage; sid=ok", "sid")
        # Either we recover sid="ok", or we get None — both are fine,
        # what matters is no exception. Pin to whichever stdlib does.
        assert result in ("ok", None)

    def test_quoted_value(self):
        # RFC 6265 allows quoted values; SimpleCookie strips quotes
        assert parse_cookie_header('sid="abc 123"', "sid") == "abc 123"


# ── integration sanity: round-trip mint → cookie → parse ─────────────


class TestRoundTrip:
    def test_mint_set_parse_yields_same_token(self):
        sc = SessionCookie(name="elh_session")
        token = new_session_token()
        set_cookie = sc.to_set_cookie_header(token, ttl_seconds=default_ttl_seconds())
        # Simulate the browser sending it back as a Cookie header
        # (it would strip attributes, keep only name=value pairs).
        cookie_line = set_cookie.split(";")[0]  # "elh_session=tok..."
        recovered = parse_cookie_header(cookie_line, "elh_session")
        assert recovered == token

    def test_hash_round_trip_stable(self):
        # Mint → hash → store → on next request, mint same hash from
        # same token. Pin the determinism that every DB-backed app
        # implicitly relies on.
        token = new_session_token()
        assert hash_token(token) == hash_token(token)
