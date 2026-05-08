"""Tests for fitapp_core.security primitives."""
import time

import pytest

from fitapp_core.security import (
    secure_headers,
    cookie_writer,
    safe_html,
    safe_attr,
    upload_validate,
    UploadError,
    hash_password,
    verify_password,
    needs_rehash,
    csrf_token,
    verify_csrf,
    rate_limit_check,
    burst_detect,
)
from fitapp_core.security.headers import delete_cookie
from fitapp_core.security.escape import safe_url, safe_js_string
from fitapp_core.security.ratelimit import lockout_status


# ── headers ──────────────────────────────────────────────────────────

class TestSecureHeaders:
    def test_baseline_keys_present(self):
        h = secure_headers()
        assert "Strict-Transport-Security" in h
        assert "X-Content-Type-Options" in h
        assert "X-Frame-Options" in h
        assert "Referrer-Policy" in h
        assert "Permissions-Policy" in h
        assert "Content-Security-Policy" in h

    def test_hsts_has_preload(self):
        assert "preload" in secure_headers()["Strict-Transport-Security"]

    def test_x_frame_deny(self):
        assert secure_headers()["X-Frame-Options"] == "DENY"

    def test_csp_default_self(self):
        assert "default-src 'self'" in secure_headers()["Content-Security-Policy"]

    def test_csp_frame_ancestors_none(self):
        assert "frame-ancestors 'none'" in secure_headers()["Content-Security-Policy"]

    def test_csp_blocks_inline_scripts_when_disabled(self):
        h = secure_headers(allow_inline_scripts=False)
        assert "'unsafe-inline'" not in h["Content-Security-Policy"].split("style-src")[0]

    def test_canonical_origin_in_connect_src(self):
        h = secure_headers(canonical_origin="https://example.com")
        assert "https://example.com" in h["Content-Security-Policy"]


class TestCookieWriter:
    def test_default_flags(self):
        c = cookie_writer("sess", "abc")
        assert "HttpOnly" in c
        assert "Secure" in c
        assert "SameSite=Strict" in c
        assert "Path=/" in c

    def test_max_age(self):
        assert "Max-Age=3600" in cookie_writer("a", "b", max_age=3600)

    def test_invalid_same_site(self):
        with pytest.raises(ValueError):
            cookie_writer("a", "b", same_site="garbage")

    def test_same_site_none_requires_secure(self):
        with pytest.raises(ValueError):
            cookie_writer("a", "b", same_site="None", secure=False)

    def test_delete_cookie(self):
        d = delete_cookie("sess")
        assert "Max-Age=0" in d
        assert "HttpOnly" in d


# ── escape ───────────────────────────────────────────────────────────

class TestSafeHtml:
    def test_escapes_lt_gt(self):
        assert safe_html("<script>") == "&lt;script&gt;"

    def test_escapes_quotes(self):
        assert safe_html('a"b') == "a&quot;b"
        assert safe_html("a'b") == "a&#x27;b"

    def test_escapes_amp(self):
        assert safe_html("a&b") == "a&amp;b"

    def test_none_to_empty(self):
        assert safe_html(None) == ""

    def test_int_to_str(self):
        assert safe_html(42) == "42"

    def test_safe_attr_alias(self):
        assert safe_attr("<x>") == safe_html("<x>")


class TestSafeUrl:
    def test_encodes_special(self):
        assert safe_url("a b") == "a%20b"
        assert safe_url("a/b?c=d") == "a%2Fb%3Fc%3Dd"

    def test_none_to_empty(self):
        assert safe_url(None) == ""


class TestSafeJsString:
    def test_escapes_lt_gt(self):
        out = safe_js_string("<>")
        assert "<" not in out and ">" not in out

    def test_escapes_quote(self):
        assert "'" not in safe_js_string("'")
        assert '"' not in safe_js_string('"')


# ── uploads ──────────────────────────────────────────────────────────

class TestUploadValidate:
    def test_jpeg_ok(self):
        r = upload_validate(b"\xff\xd8\xff" + b"\x00" * 100, declared_filename="x.jpg")
        assert r.extension == "jpeg"
        assert r.media_type == "image/jpeg"

    def test_png_ok(self):
        r = upload_validate(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, declared_filename="x.png")
        assert r.extension == "png"

    def test_pdf_ok_when_allowed(self):
        r = upload_validate(b"%PDF-1.4\n" + b"\x00" * 100, allowed_extensions=("pdf",))
        assert r.extension == "pdf"

    def test_too_large(self):
        with pytest.raises(UploadError) as e:
            upload_validate(b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024))
        assert e.value.http_status == 413
        assert e.value.code == "too_large"

    def test_wrong_magic(self):
        # GIF magic bytes: not in default allowlist
        with pytest.raises(UploadError) as e:
            upload_validate(b"GIF89a" + b"\x00" * 100)
        assert e.value.http_status == 415

    def test_path_separator_filename(self):
        with pytest.raises(UploadError) as e:
            upload_validate(b"\xff\xd8\xff" + b"\x00", declared_filename="../etc/passwd")
        assert e.value.http_status == 400

    def test_backslash_filename(self):
        with pytest.raises(UploadError):
            upload_validate(b"\xff\xd8\xff" + b"\x00", declared_filename="a\\b.jpg")

    def test_dotfile_filename(self):
        with pytest.raises(UploadError):
            upload_validate(b"\xff\xd8\xff" + b"\x00", declared_filename=".env")

    def test_empty_payload(self):
        with pytest.raises(UploadError) as e:
            upload_validate(b"")
        assert e.value.http_status == 400


# ── passwords ────────────────────────────────────────────────────────

class TestPasswords:
    def test_argon2_roundtrip(self):
        h = hash_password("hunter2")
        assert h.startswith("$argon2")
        assert verify_password(h, "hunter2") is True
        assert verify_password(h, "wrong") is False

    def test_argon2_no_rehash_needed(self):
        assert needs_rehash(hash_password("x")) is False

    def test_pbkdf2_legacy_format_verifies(self):
        # Build a legacy "pbkdf2_sha256$..." string by hand
        import hashlib
        salt = bytes(range(16))
        dk = hashlib.pbkdf2_hmac("sha256", b"hunter2", salt, 100_000)
        legacy = f"pbkdf2_sha256$100000${salt.hex()}${dk.hex()}"
        assert verify_password(legacy, "hunter2") is True
        assert verify_password(legacy, "wrong") is False
        # legacy should signal rehash
        assert needs_rehash(legacy) is True

    def test_fitapp_raw_blob_legacy_verifies(self):
        # 32-byte salt + 32-byte dk @ 200K SHA256 (matches FitApp auth.hash_password)
        import hashlib
        salt = bytes(range(32))
        dk = hashlib.pbkdf2_hmac("sha256", b"hunter2", salt, 200_000)
        raw = (salt + dk).hex()
        assert verify_password(raw, "hunter2") is True
        assert verify_password(raw, "wrong") is False
        assert needs_rehash(raw) is True

    def test_verify_none_safe(self):
        assert verify_password(None, "x") is False

    def test_verify_garbage_safe(self):
        assert verify_password("garbage", "x") is False


# ── csrf ─────────────────────────────────────────────────────────────

class TestCsrf:
    def test_token_unique(self):
        assert csrf_token() != csrf_token()

    def test_match(self):
        t = csrf_token()
        assert verify_csrf(t, t) is True

    def test_mismatch(self):
        assert verify_csrf(csrf_token(), csrf_token()) is False

    def test_empty(self):
        assert verify_csrf(None, "x") is False
        assert verify_csrf("x", None) is False
        assert verify_csrf("", "x") is False

    def test_length_mismatch(self):
        assert verify_csrf("a", "ab") is False


# ── ratelimit ────────────────────────────────────────────────────────

class TestRateLimit:
    def test_under_threshold_allowed(self):
        r = rate_limit_check(now=100.0, history=[90.0, 95.0], max_count=5, window_s=60)
        assert r.allowed is True
        assert r.current_count == 2

    def test_at_threshold_blocked(self):
        r = rate_limit_check(now=100.0, history=[90, 91, 92, 93, 94],
                             max_count=5, window_s=60)
        assert r.allowed is False
        assert r.current_count == 5

    def test_aged_out_allowed(self):
        # All events > window_s ago
        r = rate_limit_check(now=200.0, history=[10, 20, 30],
                             max_count=2, window_s=60)
        assert r.allowed is True
        assert r.current_count == 0

    def test_retry_after_positive(self):
        r = rate_limit_check(now=100.0, history=[90, 91, 92, 93, 94],
                             max_count=5, window_s=60)
        # oldest is 90; window ends at 90+60=150; retry_after = 150-100 = 50
        assert r.retry_after_s == 50


class TestBurstDetect:
    def test_below_threshold(self):
        assert burst_detect(now=100, history=[95, 96], threshold=5, window_s=60) is False

    def test_at_threshold(self):
        assert burst_detect(now=100, history=[95, 96, 97, 98, 99],
                            threshold=5, window_s=60) is True

    def test_aged_out(self):
        assert burst_detect(now=200, history=[10, 20, 30, 40, 50],
                            threshold=5, window_s=60) is False


class TestLockoutStatus:
    def test_below_threshold(self):
        locked, remaining = lockout_status(
            now=100.0, failure_count=5, last_failure=99.0,
            threshold=10, base_lock_s=900,
        )
        assert locked is False
        assert remaining == 0

    def test_at_threshold_locks(self):
        locked, remaining = lockout_status(
            now=100.0, failure_count=10, last_failure=100.0,
            threshold=10, base_lock_s=900,
        )
        assert locked is True
        assert remaining == 900

    def test_lockout_doubles_per_burst(self):
        # 11 fails, base 900 → 1800
        locked, remaining = lockout_status(
            now=100.0, failure_count=11, last_failure=100.0,
            threshold=10, base_lock_s=900, cap_lock_s=10_000,
        )
        assert locked is True
        assert remaining == 1800

    def test_lockout_cap(self):
        # 100 fails would be huge — cap to cap_lock_s
        locked, remaining = lockout_status(
            now=100.0, failure_count=100, last_failure=100.0,
            threshold=10, base_lock_s=900, cap_lock_s=7200,
        )
        assert locked is True
        assert remaining == 7200

    def test_aged_out_unlocks(self):
        locked, remaining = lockout_status(
            now=2000.0, failure_count=10, last_failure=100.0,
            threshold=10, base_lock_s=900,
        )
        assert locked is False
        assert remaining == 0


# ── regression: FitApp consumer's 16-byte salt + 32-byte dk base64 format ──

def test_fitapp_consumer_base64_legacy_verifies():
    """FitApp's auth.hash_password uses 16-byte salt + 32-byte dk
    base64-encoded (NOT 32+32 hex). Regression for that layout."""
    import base64, hashlib
    salt = bytes(range(16))                                # 16 bytes
    dk = hashlib.pbkdf2_hmac("sha256", b"hunter2", salt, 200_000)  # 32 bytes
    blob_b64 = base64.b64encode(salt + dk).decode("ascii")
    assert verify_password(blob_b64, "hunter2") is True
    assert verify_password(blob_b64, "wrong") is False
    assert needs_rehash(blob_b64) is True


def test_needs_rehash_empty_falsey_returns_false():
    """Iron Dome regression: needs_rehash on falsey input must be
    False, not True. An empty stored_hash isn't a legacy format —
    it's just empty (e.g. OAuth-only user with no password set).
    Returning True would trigger an infinite re-hash loop on every
    successful login."""
    assert needs_rehash('') is False
    assert needs_rehash(None) is False
    assert needs_rehash(b'') is False
