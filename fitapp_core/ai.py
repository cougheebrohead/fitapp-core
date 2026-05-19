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
import json
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

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


__all__ = [
    "REFUSAL_RE",
    "DEFAULT_QUALITY_THRESHOLD",
    "output_quality",
    "output_ok",
    "CallCounters",
    "BarbaraConfig",
    "BarbaraClient",
    "FallbackChain",
]
