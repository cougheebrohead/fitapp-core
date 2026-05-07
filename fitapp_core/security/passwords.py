"""Argon2id password hashing with PBKDF2 lazy migration.

Stored hash format:
  - new (preferred):  starts with `$argon2id$...` (the standard PHC string)
  - legacy (FitApp):  raw bytes (salt[:16] + dk[16:]) per `auth.hash_password`
  - legacy (coach + health):
      `pbkdf2_sha256$<iter>$<salt_hex>$<hash_hex>` per their auth.py

`verify_password(stored, plain)` sniffs the prefix and dispatches.
On a successful PBKDF2 verify, callers should re-hash with
`hash_password(plain)` and write back so the next login uses Argon2id.

Use `needs_rehash(stored)` to check whether the stored hash should
be upgraded after a successful login.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, InvalidHashError
    _hasher = PasswordHasher(
        time_cost=3,           # OWASP 2024 recommendation
        memory_cost=65536,     # 64 MiB
        parallelism=4,
        hash_len=32,
        salt_len=16,
    )
    _ARGON2_AVAILABLE = True
except ImportError:
    _hasher = None
    _ARGON2_AVAILABLE = False


# Legacy PBKDF2 params for the raw-blob FitApp consumer format
# (base64 of salt + dk, 200K iters, SHA-256). FitApp uses 16-byte salt +
# 32-byte derived key (= 48 bytes total). An older variant used 32+32 (= 64).
# verify_password tries the longer layout first, then the shorter.
_FITAPP_PBKDF2_ITERS = 200_000
_FITAPP_LAYOUTS = (
    # (salt_len, dk_len) — try 16+32 first (current FitApp), then 32+32 (older)
    (16, 32),
    (32, 32),
)

# coach + health use a self-describing format already; we parse it.


def hash_password(plain: str) -> str:
    """Hash a plaintext password with Argon2id.

    Returns the standard PHC encoded string (begins with `$argon2id$`).
    Use `verify_password` to check. Raises if argon2-cffi is missing.
    """
    if not _ARGON2_AVAILABLE:
        raise RuntimeError(
            "argon2-cffi not installed. Add to requirements.txt: argon2-cffi>=23,<26"
        )
    if not isinstance(plain, str) or not plain:
        raise ValueError("password must be a non-empty string")
    return _hasher.hash(plain)


def verify_password(stored: str | bytes, plain: str) -> bool:
    """Check `plain` against `stored`, sniffing the format.

    Supports:
      - Argon2id PHC strings ($argon2id$...)
      - Legacy "pbkdf2_sha256$<iter>$<salt_hex>$<hash_hex>" (coach + health)
      - Legacy 64-byte raw FitApp blob (32-byte salt + 32-byte dk @ 200K SHA256)

    Returns True only on exact constant-time match. Returns False on
    any mismatch, malformed input, or unrecognized format. Never raises
    on user-content failures.
    """
    if stored is None:
        return False

    # Argon2id PHC
    if isinstance(stored, str) and stored.startswith("$argon2"):
        if not _ARGON2_AVAILABLE:
            return False
        try:
            return _hasher.verify(stored, plain)
        except (VerifyMismatchError, InvalidHashError, Exception):
            return False

    # Self-describing PBKDF2 string (coach + health)
    if isinstance(stored, str) and stored.startswith("pbkdf2_sha256$"):
        try:
            _, iters_s, salt_hex, hash_hex = stored.split("$")
            iters = int(iters_s)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(hash_hex)
            actual = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iters)
            return hmac.compare_digest(expected, actual)
        except Exception:
            return False

    # Legacy FitApp raw blob — try base64 (current) then hex (older variants)
    candidates: list[bytes] = []
    if isinstance(stored, (bytes, bytearray, memoryview)):
        candidates.append(bytes(stored))
    elif isinstance(stored, str):
        s = stored.strip()
        # Try base64 first — FitApp consumer's auth.hash_password() uses this
        try:
            import base64
            decoded = base64.b64decode(s, validate=True)
            candidates.append(decoded)
        except Exception:
            pass
        # Try hex — older format variants
        try:
            candidates.append(bytes.fromhex(s))
        except ValueError:
            pass
    else:
        return False

    for b in candidates:
        for salt_len, dk_len in _FITAPP_LAYOUTS:
            if len(b) != salt_len + dk_len:
                continue
            salt = b[:salt_len]
            expected_dk = b[salt_len:]
            actual_dk = hashlib.pbkdf2_hmac(
                "sha256", plain.encode("utf-8"), salt, _FITAPP_PBKDF2_ITERS
            )
            if hmac.compare_digest(expected_dk, actual_dk):
                return True
    return False


def needs_rehash(stored: str | bytes) -> bool:
    """Return True if `stored` is a legacy hash that should be upgraded
    to Argon2id after a successful verify.

    Use this in your login handler:

        if verify_password(row.password_hash, submitted):
            if needs_rehash(row.password_hash):
                row.password_hash = hash_password(submitted)
                # write back…

    Argon2id PHC strings return False (no rehash needed), unless argon2's
    internal `check_needs_rehash` says params are stale (e.g. we
    later raised time_cost).
    """
    if stored is None:
        return False
    if isinstance(stored, str) and stored.startswith("$argon2"):
        if not _ARGON2_AVAILABLE:
            return False
        try:
            return _hasher.check_needs_rehash(stored)
        except Exception:
            return True
    # Anything not Argon2 should be migrated.
    return True


def random_token(n_bytes: int = 32) -> str:
    """Generate a URL-safe random token (e.g. for password reset links).

    Returns a hex string. Use for short-lived single-use tokens.
    """
    return secrets.token_hex(n_bytes)
