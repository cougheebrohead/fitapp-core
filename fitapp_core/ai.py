"""fitapp-core.ai — portable Barbara client + fallback chain.

Extracted from FitApp.engine on 2026-05-19 so sibling apps (elh-coach,
elh-health, future projects) can reuse the same primary-LLM client and
quality-graded fallback chain without copy-pasting.

Public surface:
    BarbaraConfig         — URL / auth / model / CF-Access knobs (dataclass)
    BarbaraClient         — `.text()`, `.vision()`, `.chat()`, `.chat_stream()`
    output_quality(text, expect) -> float in [0.0, 1.0]
    output_ok(text, expect, threshold=None) -> bool
    REFUSAL_RE            — compiled refusal-phrase regex
    CallCounters          — calls / failures / tool-loop telemetry
    FallbackChain         — try N providers in order; record + log failures

Design rules:
    - stdlib only, same as the rest of fitapp_core
    - no env-var reads inside the module; config in via dataclass/params
    - host app injects the learning-log writer (we never touch their DB)
    - all behavior preserved exactly from the engine.py original
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Union

# SSL context with a certifi-CA-bundle fallback for hosts whose system
# trust store is empty (common on macOS framework Pythons + freshly-
# built venvs). Production Linux Pythons fall through to /etc/ssl/certs
# and never hit the fallback. Mirrors engine.py exactly.
_ctx = ssl.create_default_context()
if _ctx.cert_store_stats().get("x509_ca", 0) == 0:
    try:
        import certifi  # type: ignore[import-not-found]
        _ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 — best-effort, fall back to system
        pass


# ─── Refusal regex ────────────────────────────────────────────────────
#
# Refusal phrases we never want to ship as a usable response. Used by
# output_ok() to detect prose-mode failures.
#
# Note on the `as an ai` clause: the chat system prompt persona is
# "You are FitApp's AI nutritionist", and qwen2.5 frequently echoes the
# persona as "As an AI nutritionist, …" which is NOT a refusal. So we
# only treat `as an ai…` as a refusal when it's followed (within ~80
# chars, same sentence) by an explicit refusal verb ("I cannot",
# "I can't", "I am/I'm unable", "I do not have", or "language model").
REFUSAL_RE = re.compile(
    r"\b(i\s*(?:'?m|am)\s*(?:sorry|unable|not\s+able)|i\s+cannot\s+(?:help|assist|provide)|"
    r"as\s+an\s+ai(?:[^.\n]{0,80}?(?:i\s+(?:cannot|can'?t|am\s+unable|'?m\s+unable|'?m\s+not\s+able|do\s+not\s+have)|language\s+model))|"
    r"i\s+do\s+not\s+have\s+(?:access|the\s+ability)|"
    r"i\s+can(?:not|'t)\s+(?:help|assist|provide))",
    re.IGNORECASE,
)


# ─── Output quality scoring ──────────────────────────────────────────

DEFAULT_QUALITY_THRESHOLD = 0.5


def output_quality(text: Any, expect: str = "json") -> float:
    """Score a model's output on [0.0, 1.0]. Used to decide when to trigger
    fallback rescue and learning-log capture.

    Returns 0.0 for null/error/empty, 0.5 for borderline/malformed, 1.0 for
    strong output. Mirrors engine._barbara_output_quality exactly.
    """
    if text is None:
        return 0.0
    if expect == "json":
        if isinstance(text, dict):
            if text.get("error"):
                return 0.0
            return 1.0 if text else 0.3
        if isinstance(text, list):
            return 1.0 if len(text) > 0 else 0.0
        if isinstance(text, str):
            s = text.strip()
            if not s:
                return 0.0
            cleaned = re.sub(r"^```json\s*", "", s)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and parsed.get("error"):
                    return 0.0
                return 1.0 if parsed else 0.3
            except Exception:  # noqa: BLE001
                m = re.search(r"\{[\s\S]*\}", cleaned)
                if not m:
                    return 0.2  # unparseable JSON
                try:
                    parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group()))
                    return 0.7 if parsed else 0.3  # partial extraction
                except Exception:  # noqa: BLE001
                    return 0.2  # still malformed
        return 0.0
    # prose
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:  # noqa: BLE001
            return 0.0
    s = text.strip()
    if not s:
        return 0.0
    if len(s) < 4:
        return 0.2  # too short
    if REFUSAL_RE.search(s[:240]):
        return 0.1  # refusal detected
    return 1.0


def output_ok(text: Any, expect: str = "json",
              threshold: Optional[float] = None) -> bool:
    """Returns True when output_quality >= threshold. Default 0.5."""
    t = DEFAULT_QUALITY_THRESHOLD if threshold is None else float(threshold)
    return output_quality(text, expect) >= t


# ─── Call counters (replaces module-level globals in engine.py) ──────

@dataclass
class CallCounters:
    """Process-lifetime telemetry for primary-LLM (Barbara) call sites.

    Replaces the module-level BARBARA_* globals in engine.py. Engine
    creates a singleton at import time and shares it across call sites;
    sibling apps do the same.
    """

    calls: int = 0
    failures: int = 0
    tool_loops_total: int = 0
    tool_loops_capped: int = 0
    tool_loops_hop_sum: int = 0

    def record(self, ok: bool) -> None:
        """Record a single primary-call attempt and whether it produced
        usable output (post quality-gate)."""
        self.calls += 1
        if not ok:
            self.failures += 1

    def record_tool_loop(self, hops_taken: int, capped: bool = False) -> None:
        """Record tool-loop statistics: hop count and whether loop hit cap."""
        self.tool_loops_total += 1
        self.tool_loops_hop_sum += int(hops_taken)
        if capped:
            self.tool_loops_capped += 1

    def snapshot(self) -> dict:
        """Return engine.barbara_stats()-shaped dict for /admin telemetry."""
        return {
            "calls":     self.calls,
            "failures":  self.failures,
            "fail_rate": (self.failures / self.calls) if self.calls else 0.0,
            "tool_loops_total":    self.tool_loops_total,
            "tool_loops_capped":   self.tool_loops_capped,
            "tool_loops_avg_hops": (self.tool_loops_hop_sum / self.tool_loops_total)
                                   if self.tool_loops_total else 0.0,
        }


# ─── BarbaraClient ────────────────────────────────────────────────────

@dataclass
class BarbaraConfig:
    """Per-host configuration for a BarbaraClient instance.

    The `url` default points at the named tunnel; the `auth` default is
    empty so a misconfigured deploy fails fast at the proxy instead of
    silently shipping a baked-in token. Host apps MUST inject the bearer
    via `BarbaraConfig(auth=os.environ["BARBARA_AUTH"])` (or equivalent).
    An empty `auth` means no `Authorization` header is sent — Barbara's
    proxy will then 401 and the FallbackChain rolls forward to Gemini/Claude.
    """

    # Tunnel + bearer. `auth` deliberately defaults to empty: see docstring.
    url: str = "https://barbara.barbara-ai.org"
    auth: str = ""

    # Model names. Same Ollama defaults as engine.py.
    text_model:   str = "qwen2.5:32b"
    vision_model: str = "qwen2.5vl:7b"
    haiku_model:  str = "qwen2.5:14b"

    # Timeouts (seconds). Match engine.py call sites exactly.
    text_timeout:        int = 60
    vision_timeout:      int = 90
    chat_timeout:        int = 120
    chat_stream_timeout: int = 180

    # Cloudflare Access service-token (off by default; CF Access app was
    # deleted 2026-05-19 after policy save lockout). Set cf_access_on=True
    # AND provide both halves to re-enable.
    cf_access_on:     bool = False
    cf_access_id:     str  = ""
    cf_access_secret: str  = ""

    # User-Agent banner. Lets target services log/trace requests.
    user_agent: str = "FitApp-Engine/1.0 (Barbara)"

    def url_clean(self) -> str:
        """Trimmed URL with trailing slash removed."""
        return (self.url or "").strip().rstrip("/")

    def cf_access_headers(self) -> dict:
        """Return CF-Access headers when configured; else {}.

        Sending these headers when CF Access is not active is harmless —
        Cloudflare just ignores them. (Engine.py shipped this same way.)
        """
        if not self.cf_access_on:
            return {}
        if not (self.cf_access_id and self.cf_access_secret):
            return {}
        return {
            "CF-Access-Client-Id":     self.cf_access_id,
            "CF-Access-Client-Secret": self.cf_access_secret,
        }


def _strip_json_codefence(text: str) -> str:
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return re.sub(r",\s*([}\]])", r"\1", text)


def _coerce_json_or_error(text: str, raw: bool) -> Any:
    """Mirror of engine.py JSON-coercion path: strip codefence, fix
    trailing commas, then `{...}` regex rescue. Returns:
        raw=True  → the cleaned text string
        raw=False → parsed dict / {'error': ...} on parse failure
    """
    if raw:
        return text
    cleaned = _strip_json_codefence(text)
    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group()))
            except Exception:  # noqa: BLE001
                pass
    return {"error": "AI response parse failed"}


def _unavailable(raw: bool) -> Any:
    return {"error": "AI service temporarily unavailable"} if not raw else ""


class BarbaraClient:
    """Thin client for Barbara's Ollama-compatible /api/chat endpoint.

    Caller injects config (URL, auth, models, CF-Access). Methods mirror
    engine.py call sites exactly so the refactor preserves behavior:
        text(prompt, raw=False)
        vision(image_bytes, media_type, system_prompt, user_prompt, raw=False)
        chat(system, history, user_message, max_tokens=800, model=None, fmt=None)
        chat_stream(system, history, user_message, max_tokens=800)

    Optional image_resizer callable lets callers plug in Pillow downsize
    without forcing a Pillow dep here. Same signature as engine.py's
    _maybe_downsize. None = pass-through.
    """

    def __init__(self, config: Optional[BarbaraConfig] = None,
                 image_resizer: Optional[Callable[..., bytes]] = None,
                 log_prefix: str = "barbara"):
        self.config = config or BarbaraConfig()
        self._resize = image_resizer
        self._log_prefix = log_prefix

    # ----- internal helpers -----

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json",
             "User-Agent": self.config.user_agent}
        if self.config.auth:
            h["Authorization"] = f"Bearer {self.config.auth}"
        h.update(self.config.cf_access_headers())
        return h

    def _err(self, where: str, exc: BaseException) -> None:
        print(f"[{self._log_prefix}.{where}] {type(exc).__name__}: "
              f"{str(exc)[:300]}", file=sys.stderr)

    # ----- public API -----

    def text(self, prompt: str, raw: bool = False) -> Any:
        """Call Barbara's text model for typed-meal parsing / similar."""
        base = self.config.url_clean()
        if not base:
            return _unavailable(raw)
        try:
            body = json.dumps({
                "model":    self.config.text_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "format":   "json",
                "options":  {"temperature": 0.1, "num_predict": 1500},
            }).encode()
            req = urllib.request.Request(f"{base}/api/chat", data=body,
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=self.config.text_timeout,
                                        context=_ctx) as r:
                result = json.loads(r.read())
            text = result.get("message", {}).get("content", "")
            return _coerce_json_or_error(text, raw)
        except Exception as e:  # noqa: BLE001
            self._err("text", e)
            return _unavailable(raw)

    def vision(self, image_bytes: bytes, media_type: str,
               system_prompt: str, user_prompt: str,
               raw: bool = False) -> Any:
        """Call Barbara's vision model with an image payload."""
        base = self.config.url_clean()
        if not base:
            return _unavailable(raw)
        try:
            if self._resize is not None:
                try:
                    image_bytes = self._resize(image_bytes, media_type, max_dim=1024)
                except Exception:  # noqa: BLE001 — best-effort downsize
                    pass
            b64 = base64.standard_b64encode(image_bytes).decode()
            body = json.dumps({
                "model": self.config.vision_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt, "images": [b64]},
                ],
                "stream":  False,
                "format":  "json",
                # 900 caps the worst-case generation time. Typical food photo
                # responses are 500-800 tokens; 900 leaves headroom for complex
                # 10-item plates.
                "options": {"temperature": 0.1, "num_predict": 900},
            }).encode()
            req = urllib.request.Request(f"{base}/api/chat", data=body,
                                         headers=self._headers())
            with urllib.request.urlopen(
                req, timeout=self.config.vision_timeout, context=_ctx
            ) as r:
                result = json.loads(r.read())
            text = result.get("message", {}).get("content", "")
            return _coerce_json_or_error(text, raw)
        except Exception as e:  # noqa: BLE001
            self._err("vision", e)
            return _unavailable(raw)

    def chat(self, system: str, history: list, user_message: str,
             max_tokens: int = 800, model: Optional[str] = None,
             fmt: Optional[str] = None) -> dict:
        """Multi-turn chat. Returns:
            {"text": str, "model": str, "tokens_in": int|None, "tokens_out": int|None}
        or {"error": str, "text": ""} on failure / unset URL.
        """
        base = self.config.url_clean()
        if not base:
            return {"error": "Barbara not configured", "text": ""}
        chosen_model = model or self.config.text_model
        msgs = self._build_messages(system, history, user_message)
        try:
            payload: dict = {
                "model":    chosen_model,
                "messages": msgs,
                "stream":   False,
                "options":  {"temperature": 0.4, "num_predict": int(max_tokens)},
            }
            if fmt:
                payload["format"] = fmt
            body = json.dumps(payload).encode()
            req = urllib.request.Request(f"{base}/api/chat", data=body,
                                         headers=self._headers())
            with urllib.request.urlopen(
                req, timeout=self.config.chat_timeout, context=_ctx
            ) as r:
                result = json.loads(r.read())
            text = (result.get("message") or {}).get("content", "") or ""
            return {
                "text":       text.strip(),
                "model":      result.get("model") or chosen_model,
                "tokens_in":  result.get("prompt_eval_count"),
                "tokens_out": result.get("eval_count"),
            }
        except Exception as e:  # noqa: BLE001
            self._err("chat", e)
            return {"error": f"barbara chat failed: {e}", "text": ""}

    def chat_stream(self, system: str, history: list, user_message: str,
                    max_tokens: int = 800) -> Iterator[dict]:
        """Generator yielding the same event shape as engine.chat_stream():
            {"type": "text", "delta": "<chunk>"}, ..., {"type": "done", ...}
        or {"type": "error", "error": "..."} on failure.
        """
        base = self.config.url_clean()
        if not base:
            yield {"type": "error", "error": "Barbara not configured"}
            return
        msgs = self._build_messages(system, history, user_message)
        try:
            body = json.dumps({
                "model":    self.config.text_model,
                "messages": msgs,
                "stream":   True,
                "options":  {"temperature": 0.4, "num_predict": int(max_tokens)},
            }).encode()
            req = urllib.request.Request(f"{base}/api/chat", data=body,
                                         headers=self._headers())
            full_text: list[str] = []
            model_name = self.config.text_model
            tokens_in = tokens_out = None
            with urllib.request.urlopen(
                req, timeout=self.config.chat_stream_timeout, context=_ctx
            ) as r:
                buf = b""
                while True:
                    chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line.decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            continue
                        if ev.get("model"):
                            model_name = ev["model"]
                        msg = ev.get("message") or {}
                        delta = msg.get("content") or ""
                        if delta:
                            full_text.append(delta)
                            yield {"type": "text", "delta": delta}
                        if ev.get("done"):
                            if ev.get("prompt_eval_count") is not None:
                                tokens_in = ev["prompt_eval_count"]
                            if ev.get("eval_count") is not None:
                                tokens_out = ev["eval_count"]
            yield {
                "type":       "done",
                "model":      model_name,
                "tokens_in":  tokens_in,
                "tokens_out": tokens_out,
                "text":       "".join(full_text).strip(),
            }
        except urllib.error.HTTPError as e:
            snippet = (e.read() or b"")[:300].decode("utf-8", errors="replace")
            yield {"type": "error",
                   "error": f"barbara stream upstream {e.code}: {snippet}"}
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "error": f"barbara stream failed: {e}"}

    @staticmethod
    def _build_messages(system: str, history: list, user_message: str) -> list:
        """Build the Ollama-shaped messages list. Same trim/role rules as
        engine.py: caps each content at 6000 chars; only user/assistant
        roles from history; system optional."""
        msgs: list = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in (history or []):
            r = (m or {}).get("role")
            c = ((m or {}).get("content") or "").strip()
            if r in ("user", "assistant") and c:
                msgs.append({"role": r, "content": c[:6000]})
        msgs.append({"role": "user",
                     "content": (user_message or "").strip()[:6000]})
        return msgs


# ─── Fallback chain ──────────────────────────────────────────────────


@dataclass
class FallbackChain:
    """Try a list of provider callables in order until one returns
    "good" output as judged by `is_good`.

    Each provider is a zero-arg callable (caller closes over its args)
    returning the provider's native shape. After each call we:
        1. Run `is_good(result)` — pure check, no side effects.
        2. If good: optionally fire `on_success(name, result)` and return.
        3. If bad : optionally fire `on_failure(name, result)` and try next.

    `learning_log_writer`, when provided, is called once on first
    fallback success with the shape:
        learning_log_writer(call_site, system_prompt, user_input,
                            primary_output, fallback_output)
    matching FitApp.db.log_barbara_failure. Best-effort; exceptions are
    swallowed so the production hot path never breaks on a missing row.

    `counters`, when provided, sees `.record(ok)` for the FIRST provider
    only (engine.py only records the primary, never the fallbacks).

    Returns the first good result, or the LAST result tried (good-or-not)
    if every provider fails. Callers decide what to do with all-failed
    output (often they yield an error event or return a degraded shape).
    """

    providers: list[tuple[str, Callable[[], Any]]] = field(default_factory=list)
    is_good:   Callable[[Any], bool] = field(default=lambda r: bool(r))
    counters:  Optional[CallCounters] = None
    learning_log_writer: Optional[Callable[..., None]] = None
    call_site: str = ""
    system_prompt:  Any = ""
    user_input:     Any = ""
    on_success: Optional[Callable[[str, Any], None]] = None
    on_failure: Optional[Callable[[str, Any], None]] = None

    def run(self) -> tuple[str, Any]:
        """Execute the chain. Returns (provider_name, result) for the
        first good result, or (last_name, last_result) when all fail."""
        last_name = ""
        last_result: Any = None
        primary_output: Any = None
        for i, (name, fn) in enumerate(self.providers):
            try:
                result = fn()
            except Exception as e:  # noqa: BLE001 — record + advance
                result = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            last_name, last_result = name, result
            ok = False
            try:
                ok = bool(self.is_good(result))
            except Exception:  # noqa: BLE001 — defensive
                ok = False
            if i == 0:
                primary_output = result
                if self.counters is not None:
                    self.counters.record(ok)
            if ok:
                # Only log when a NON-PRIMARY provider rescued us.
                if i > 0 and self.learning_log_writer is not None:
                    try:
                        self.learning_log_writer(
                            self.call_site,
                            self.system_prompt,
                            self.user_input,
                            primary_output,
                            result,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                if self.on_success is not None:
                    try:
                        self.on_success(name, result)
                    except Exception:  # noqa: BLE001
                        pass
                return name, result
            if self.on_failure is not None:
                try:
                    self.on_failure(name, result)
                except Exception:  # noqa: BLE001
                    pass
        return last_name, last_result


# ─── Claude client + result cache + batch API ────────────────────────
#
# ClaudeClient is a stdlib-only Anthropic wrapper that structures every
# call for MAXIMUM prompt-cache hit rate. The big static instruction
# blob goes into `system` with `cache_control: ephemeral`, so repeat
# calls with the same instructions pay ~10% of the input-token price.
#
# SqliteResultCache is a SHA-keyed answer cache (default 24h TTL) that
# short-circuits repeated identical requests entirely — zero API cost
# on any duplicate.
#
# ClaudeBatch wraps /v1/messages/batches for 50%-off async workloads
# (weekly reports, daily briefings, ingest pipelines).
#
# All three are optional and composable: caller creates whichever they
# need, wires the cache into the client, and calls .messages().


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"


@dataclass
class ClaudeConfig:
    """Per-host configuration for a ClaudeClient instance.

    `api_key` MUST be injected by the host (env var, secret manager,
    etc.). An empty key returns {"error": "no api key", "text": ""}
    from every call — the FallbackChain then rolls to the next provider.
    """

    api_key: str = ""
    base_url: str = "https://api.anthropic.com"
    api_version: str = "2023-06-01"
    default_model: str = DEFAULT_CLAUDE_MODEL
    timeout: int = 60
    user_agent: str = "fitapp-core-claude/1.0"

    # When True (default), the `system` prompt on every .messages() call
    # is wrapped with cache_control: ephemeral. Cached input tokens are
    # billed at ~10% of standard input rate. Anthropic requires at
    # least 1024 tokens (Sonnet) / 2048 tokens (Haiku) in a cache block
    # or the cache_control is silently ignored — small prompts don't
    # break, they just don't cache.
    cache_system_by_default: bool = True


class SqliteResultCache:
    """SHA-keyed answer cache backed by SQLite.

    Wraps `(model, system, user, opts) -> response`. Default 24h TTL.
    Thread-safe via a module-level RLock (SQLite in WAL mode is fine
    for concurrent reads, but we serialize writes to avoid rare
    "database is locked" storms in high-fanout batch jobs).

    Cache misses are cheap (one indexed lookup); hits skip the API
    call entirely. `gc()` reaps expired rows and returns the count.

    Design notes:
    - stdlib only (sqlite3, hashlib, json, time)
    - `.db` file lives wherever the caller decides (default:
      `~/.fitapp-core/result-cache.db`). Same file across processes
      = shared cache. Not appropriate for multi-host deploys (use
      Redis at that scale) but perfect for a single Render service.
    - JSON is the storage format; anything json-serializable can be
      cached. The client stores {'text', 'model', 'tokens_in', ...}
      shaped dicts.
    """

    _lock = threading.RLock()

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base = os.path.expanduser("~/.fitapp-core")
            os.makedirs(base, exist_ok=True)
            db_path = os.path.join(base, "result-cache.db")
        self.db_path = db_path
        self._init()

    def _init(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, "
                "value TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL, "
                "created_at INTEGER NOT NULL)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_expires "
                "ON cache(expires_at)"
            )
            con.commit()

    @staticmethod
    def make_key(model: str, system: Any, user: Any,
                 extra: Optional[dict] = None) -> str:
        """Deterministic SHA-256 key. `extra` is any additional
        cache-affecting knob (temperature, max_tokens, image bytes'
        digest, etc.) the caller wants folded into the key."""
        h = hashlib.sha256()
        h.update(b"m|"); h.update(str(model).encode("utf-8"))
        h.update(b"|s|")
        h.update(_stable_dumps(system).encode("utf-8"))
        h.update(b"|u|")
        h.update(_stable_dumps(user).encode("utf-8"))
        if extra:
            h.update(b"|x|")
            h.update(_stable_dumps(extra).encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[dict]:
        with self._lock, sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT value, expires_at FROM cache WHERE key=?", (key,)
            ).fetchone()
        if not row:
            return None
        value_json, expires_at = row
        if int(time.time()) > int(expires_at):
            return None  # expired; GC sweeps async
        try:
            return json.loads(value_json)
        except Exception:  # noqa: BLE001
            return None

    def set(self, key: str, value: dict, ttl_s: int = 86400) -> None:
        now = int(time.time())
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock, sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO cache(key, value, expires_at, created_at) "
                "VALUES (?,?,?,?)",
                (key, payload, now + int(ttl_s), now),
            )
            con.commit()

    def gc(self) -> int:
        """Delete expired rows. Returns count deleted. Safe to call
        anytime; workers can hook a nightly cron to this."""
        with self._lock, sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "DELETE FROM cache WHERE expires_at < ?", (int(time.time()),)
            )
            con.commit()
            return cur.rowcount


def _stable_dumps(obj: Any) -> str:
    """json.dumps with sort_keys + ensure_ascii=False for stable hashing.
    Non-JSON objects fall back to repr() which is stable enough for
    cache keys of already-simple call payloads."""
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(obj)


def _system_block(system: Union[str, list, None],
                  cache: bool) -> Optional[list]:
    """Build the `system` field for Anthropic messages, applying
    cache_control: ephemeral when `cache` is True.

    Accepts:
    - str          → single text block
    - list of dict → passed through, but cache_control added to the
                     final block if `cache` is True (Anthropic supports
                     up to 4 cache breakpoints; we default to the last).
    - None / empty → returns None (caller omits the field)
    """
    if not system:
        return None
    if isinstance(system, str):
        block: dict = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]
    # list case — passthrough with tail cache
    blocks = list(system)
    if cache and blocks:
        tail = dict(blocks[-1])
        tail.setdefault("cache_control", {"type": "ephemeral"})
        blocks[-1] = tail
    return blocks


class ClaudeClient:
    """Stdlib Anthropic /v1/messages client with cache_control by default
    and optional result-cache short-circuit.

    Public surface:
        .messages(system, user, ...) -> dict
        .batch_create(requests)      -> str batch_id
        .batch_status(batch_id)      -> dict
        .batch_results(batch_id)     -> Iterator[dict]

    Response shape (from .messages()):
        {"text": str,
         "model": str,
         "tokens_in": int|None,
         "tokens_out": int|None,
         "cache_read_input_tokens": int,
         "cache_creation_input_tokens": int,
         "stop_reason": str|None,
         "cached": bool}
    On failure: {"error": "...", "text": ""}
    """

    def __init__(self, config: Optional[ClaudeConfig] = None,
                 cache: Optional[SqliteResultCache] = None,
                 counters: Optional[CallCounters] = None,
                 log_prefix: str = "claude"):
        self.config = config or ClaudeConfig()
        self._cache = cache
        self._counters = counters
        self._log_prefix = log_prefix

    # ----- internal -----

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": self.config.api_version,
            "User-Agent": self.config.user_agent,
        }
        if extra:
            h.update(extra)
        return h

    def _err(self, where: str, exc: BaseException) -> None:
        print(f"[{self._log_prefix}.{where}] {type(exc).__name__}: "
              f"{str(exc)[:300]}", file=sys.stderr)

    # ----- public -----

    def messages(self,
                 system: Union[str, list, None],
                 user: Union[str, list],
                 model: Optional[str] = None,
                 max_tokens: int = 1024,
                 temperature: float = 0.1,
                 cache_system: Optional[bool] = None,
                 cache_ttl_s: int = 86400,
                 use_result_cache: bool = True,
                 extra_headers: Optional[dict] = None,
                 stop_sequences: Optional[list] = None) -> dict:
        """Single /v1/messages call.

        `system` — the big static instruction blob. Auto-cached unless
                    `cache_system=False`. For long prompts (≥1024 tok
                    on Sonnet) this cuts input-token cost by ~90% on
                    repeat calls.
        `user`   — str (converted to a single text block) or a list of
                    content blocks (for vision + text combined).
        `use_result_cache` — when True and a SqliteResultCache is wired
                    in, look up (model, system, user, opts) and skip
                    the API call on hit.
        """
        model = model or self.config.default_model
        if not self.config.api_key:
            if self._counters is not None:
                self._counters.record(False)
            return {"error": "no api key", "text": ""}

        do_cache = (self.config.cache_system_by_default
                    if cache_system is None else bool(cache_system))

        # Result-cache lookup — before any network I/O.
        cache_key: Optional[str] = None
        if use_result_cache and self._cache is not None:
            cache_key = SqliteResultCache.make_key(
                model, system, user,
                extra={"max_tokens": max_tokens, "temperature": temperature,
                       "stop_sequences": stop_sequences or []},
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                if self._counters is not None:
                    self._counters.record(True)
                return {**hit, "cached": True}

        # Build user content list.
        if isinstance(user, str):
            content: list = [{"type": "text", "text": user}]
        else:
            content = list(user)

        body: dict = {
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "messages": [{"role": "user", "content": content}],
        }
        sysb = _system_block(system, do_cache)
        if sysb is not None:
            body["system"] = sysb
        if stop_sequences:
            body["stop_sequences"] = list(stop_sequences)

        req = urllib.request.Request(
            f"{self.config.base_url}/v1/messages",
            data=json.dumps(body).encode(),
            headers=self._headers(extra_headers),
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout, context=_ctx
            ) as r:
                result = json.loads(r.read())
        except urllib.error.HTTPError as e:
            snippet = (e.read() or b"")[:300].decode("utf-8", errors="replace")
            self._err("messages", e)
            if self._counters is not None:
                self._counters.record(False)
            return {"error": f"claude http {e.code}: {snippet}", "text": ""}
        except Exception as e:  # noqa: BLE001
            self._err("messages", e)
            if self._counters is not None:
                self._counters.record(False)
            return {"error": f"claude call failed: {e}", "text": ""}

        # Extract text; join every text block (Claude may emit multiple).
        text_parts: list = []
        for block in (result.get("content") or []):
            if (block or {}).get("type") == "text":
                text_parts.append(block.get("text", ""))
        text = "".join(text_parts)

        usage = result.get("usage") or {}
        out = {
            "text": text,
            "model": result.get("model") or model,
            "tokens_in":  usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"),
            "cache_read_input_tokens":
                int(usage.get("cache_read_input_tokens") or 0),
            "cache_creation_input_tokens":
                int(usage.get("cache_creation_input_tokens") or 0),
            "stop_reason": result.get("stop_reason"),
            "cached": False,
        }

        if self._counters is not None:
            self._counters.record(bool(text))

        # Store in result cache — even error-shaped short responses are
        # skipped by the `text` guard so we never memoize failures.
        if cache_key is not None and text:
            try:
                self._cache.set(cache_key, out, ttl_s=cache_ttl_s)
            except Exception:  # noqa: BLE001
                pass  # cache write failures never block the hot path

        return out

    # ----- batch API — 50% off async workloads -----

    def batch_create(self, requests: list[dict]) -> dict:
        """Submit a batch. Each request must be:
            {"custom_id": "<caller-supplied id>",
             "params":    {system, user, model?, max_tokens?, ...}}

        `params` uses the same shape as .messages() and gets the same
        cache_control treatment.

        Returns the raw API response (contains `id`, `processing_status`,
        etc.) or {"error": ...}.
        """
        if not self.config.api_key:
            return {"error": "no api key"}
        if not requests:
            return {"error": "empty batch"}

        api_requests: list = []
        for r in requests:
            params = dict(r.get("params") or {})
            system = params.pop("system", None)
            user = params.pop("user", "")
            model = params.pop("model", self.config.default_model)
            max_tokens = int(params.pop("max_tokens", 1024))
            temperature = float(params.pop("temperature", 0.1))
            cache_system = params.pop("cache_system",
                                      self.config.cache_system_by_default)

            content = ([{"type": "text", "text": user}]
                       if isinstance(user, str) else list(user))
            api_body: dict = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content}],
            }
            sysb = _system_block(system, bool(cache_system))
            if sysb is not None:
                api_body["system"] = sysb
            for k, v in params.items():
                api_body[k] = v

            api_requests.append({
                "custom_id": str(r.get("custom_id") or ""),
                "params": api_body,
            })

        body = json.dumps({"requests": api_requests}).encode()
        req = urllib.request.Request(
            f"{self.config.base_url}/v1/messages/batches",
            data=body,
            headers=self._headers({
                "anthropic-beta": "message-batches-2024-09-24",
            }),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout, context=_ctx
            ) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            snippet = (e.read() or b"")[:300].decode("utf-8", errors="replace")
            self._err("batch_create", e)
            return {"error": f"batch_create http {e.code}: {snippet}"}
        except Exception as e:  # noqa: BLE001
            self._err("batch_create", e)
            return {"error": f"batch_create failed: {e}"}

    def batch_status(self, batch_id: str) -> dict:
        req = urllib.request.Request(
            f"{self.config.base_url}/v1/messages/batches/"
            f"{urllib.parse.quote(batch_id, safe='')}",
            headers=self._headers({
                "anthropic-beta": "message-batches-2024-09-24",
            }),
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout, context=_ctx
            ) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            self._err("batch_status", e)
            return {"error": f"batch_status failed: {e}"}

    def batch_results(self, batch_id: str) -> Iterator[dict]:
        """Stream JSONL results for a completed batch. Each line is
        `{"custom_id": ..., "result": {...}}`. Caller decodes."""
        status = self.batch_status(batch_id)
        results_url = (status or {}).get("results_url")
        if not results_url:
            yield {"error": "no results_url",
                   "status": (status or {}).get("processing_status")}
            return
        req = urllib.request.Request(
            results_url,
            headers=self._headers({
                "anthropic-beta": "message-batches-2024-09-24",
            }),
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout, context=_ctx
            ) as r:
                buf = b""
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:  # noqa: BLE001
            self._err("batch_results", e)
            yield {"error": f"batch_results failed: {e}"}





__all__ = [
    "REFUSAL_RE",
    "DEFAULT_QUALITY_THRESHOLD",
    "output_quality",
    "output_ok",
    "CallCounters",
    "BarbaraConfig",
    "BarbaraClient",
    "FallbackChain",
    # Claude client + result cache + batch API
    "DEFAULT_CLAUDE_MODEL",
    "ClaudeConfig",
    "ClaudeClient",
    "SqliteResultCache",
]
