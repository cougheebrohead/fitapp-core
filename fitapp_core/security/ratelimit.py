"""Rate limiting + burst detection — abstract primitives.

DB-agnostic, like `fitapp_core.audit`: the caller plugs in its own
ledger backend (Postgres table, Redis, in-memory). This module gives
you the windowing math + decision logic.

Two flavors:

1. `rate_limit_check(now, history, max, window_s)` — sliding-window
   count. Returns (allowed, retry_after, current_count).

2. `burst_detect(now, history, threshold, window_s)` — boolean
   spike detector. Returns True when the burst threshold is met
   inside the window. Use for failed-auth bursts, signature
   mismatch bursts, etc., that should fire incidents.

History format: a list/iterable of unix timestamps (int seconds).
The caller is responsible for persistence + pruning. Trim history
to `now - window_s` periodically.

Example (per-IP login):

    history = pg.fetchall("SELECT ts FROM login_failures
                           WHERE ip_hash=$1 AND ts > now()-interval '1 hour'", ip)
    allowed, retry_after, count = rate_limit_check(time.time(), history, 30, 3600)
    if not allowed:
        return self._j({"error": "too_many_attempts", "retry_after": retry_after}, 429)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after_s: int
    current_count: int


def rate_limit_check(
    now: float,
    history: Iterable[float],
    max_count: int,
    window_s: int,
) -> RateLimitResult:
    """Sliding-window check. Allowed if `< max_count` events in the past
    `window_s` seconds. retry_after is the seconds until the oldest
    event ages out (0 if allowed).
    """
    if max_count < 1 or window_s < 1:
        raise ValueError("max_count and window_s must be positive")
    cutoff = now - window_s
    in_window = [t for t in history if t >= cutoff]
    count = len(in_window)
    if count < max_count:
        return RateLimitResult(allowed=True, retry_after_s=0, current_count=count)
    # Find the oldest event still in the window. Once it ages out, the
    # caller can retry.
    oldest = min(in_window)
    retry_after = max(1, int((oldest + window_s) - now))
    return RateLimitResult(
        allowed=False, retry_after_s=retry_after, current_count=count
    )


def burst_detect(
    now: float,
    history: Iterable[float],
    threshold: int,
    window_s: int,
) -> bool:
    """Return True if `>= threshold` events occurred in the past
    `window_s` seconds. Used for incident-firing detectors that should
    page on-call when something abnormal is happening.
    """
    cutoff = now - window_s
    n = 0
    for t in history:
        if t >= cutoff:
            n += 1
            if n >= threshold:
                return True
    return False


def lockout_status(
    now: float,
    failure_count: int,
    last_failure: float | None,
    *,
    threshold: int = 10,
    base_lock_s: int = 15 * 60,
    cap_lock_s: int = 120 * 60,
) -> tuple[bool, int]:
    """Return (locked, seconds_remaining).

    Lock policy: at `threshold` consecutive failures the account is
    locked for `base_lock_s`. Subsequent burst-on-burst failures
    double the duration up to `cap_lock_s`.

    `failure_count` is the persistent counter the caller maintains;
    `last_failure` is the timestamp of the most recent failure.

    Successful login resets failure_count to 0 — caller's job, not ours.
    """
    if failure_count < threshold or last_failure is None:
        return False, 0
    # Lock duration = base * 2^(burst_index), capped.
    bursts_over = max(0, failure_count - threshold)
    duration = min(cap_lock_s, base_lock_s * (2 ** bursts_over))
    unlocks_at = last_failure + duration
    if now >= unlocks_at:
        return False, 0
    return True, int(unlocks_at - now)
