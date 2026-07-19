"""Tests for fitapp_core.ai — the Barbara client + fallback chain.

Goals:
  - output_quality / output_ok cover json + prose shapes, including
    refusal detection and JSON-mode rescue from trailing-comma garbage.
  - REFUSAL_RE matches real qwen2.5 refusals without false-positiving
    on persona-echo ("As an AI nutritionist, …").
  - CallCounters increment + snapshot math is sound, especially the
    zero-call edge case in fail_rate.
  - FallbackChain ordering, learning-log injection on rescue, counters
    record only the primary, exceptions short-circuit to next provider.
"""

import io
import json
import os
import time
from unittest.mock import patch

from fitapp_core.ai import (
    BarbaraClient,
    BarbaraConfig,
    CallCounters,
    ClaudeClient,
    ClaudeConfig,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_QUALITY_THRESHOLD,
    FallbackChain,
    REFUSAL_RE,
    SqliteResultCache,
    output_ok,
    output_quality,
)


# ─── output_quality / output_ok ──────────────────────────────────────

def test_quality_none_is_zero():
    assert output_quality(None) == 0.0
    assert output_quality(None, expect="prose") == 0.0


def test_quality_json_dict_with_data_is_one():
    assert output_quality({"name": "apple", "calories": 95}) == 1.0


def test_quality_json_dict_with_error_is_zero():
    assert output_quality({"error": "AI unavailable"}) == 0.0


def test_quality_json_empty_dict_is_partial():
    # Empty payload is "shape OK but useless" — partial credit, below threshold.
    assert output_quality({}) == 0.3


def test_quality_json_list_with_items():
    assert output_quality([{"a": 1}]) == 1.0
    assert output_quality([]) == 0.0


def test_quality_json_string_parseable_full():
    assert output_quality('{"k": 1}') == 1.0


def test_quality_json_string_trailing_comma_rescued():
    # The cleanup regex (",\\s*([}\\]])" → "\\1") rescues trailing commas
    # before json.loads. Mirrors the engine.py behavior.
    assert output_quality('{"k": 1,}') == 1.0


def test_quality_json_string_partial_extraction_via_regex():
    # Surrounded text — regex rescue path. Engine awards 0.7.
    text = 'sure! here you go: {"k": 1} hope that helps'
    assert output_quality(text) == 0.7


def test_quality_json_string_unparseable_is_low():
    assert output_quality("definitely not json") == 0.2


def test_quality_json_string_empty_is_zero():
    assert output_quality("") == 0.0
    assert output_quality("   ") == 0.0


def test_quality_prose_short_is_low():
    assert output_quality("ok", expect="prose") == 0.2


def test_quality_prose_long_is_full():
    assert output_quality("Here is a careful answer with substance.",
                          expect="prose") == 1.0


def test_quality_prose_refusal_is_near_zero():
    assert output_quality("I'm sorry, I cannot help with that.",
                          expect="prose") == 0.1


def test_quality_prose_persona_echo_is_not_a_refusal():
    # "As an AI nutritionist, here are your macros…" must NOT trip the
    # refusal regex. This was the qwen2.5 false-positive that drove the
    # ~80-char same-sentence guard in the regex.
    text = ("As an AI nutritionist, I can help you balance your macros. "
            "Try 40/30/30 carbs/protein/fat to start.")
    assert output_quality(text, expect="prose") == 1.0


def test_output_ok_uses_default_threshold():
    # Empty dict (0.3) is BELOW the 0.5 default — output_ok returns False.
    assert output_ok({}) is False
    assert output_ok({"x": 1}) is True


def test_output_ok_custom_threshold():
    # 0.7 partial-extraction should pass a 0.6 threshold but fail at 0.8.
    text = 'sure! here you go: {"k": 1} hope that helps'
    assert output_ok(text, threshold=0.6) is True
    assert output_ok(text, threshold=0.8) is False


def test_default_threshold_is_05():
    assert DEFAULT_QUALITY_THRESHOLD == 0.5


# ─── REFUSAL_RE ──────────────────────────────────────────────────────

def test_refusal_re_catches_apology():
    assert REFUSAL_RE.search("I'm sorry, I won't do that.")


def test_refusal_re_catches_cannot_help():
    assert REFUSAL_RE.search("I cannot help with this request.")


def test_refusal_re_catches_language_model_disclaimer():
    assert REFUSAL_RE.search(
        "As an AI language model, I do not have personal opinions.")


def test_refusal_re_misses_persona_echo():
    # Same string as in the prose quality test above. Belt-and-braces:
    # both quality() and the regex agree this is not a refusal.
    text = ("As an AI nutritionist, I can help you balance your macros. "
            "Try 40/30/30 carbs/protein/fat to start.")
    assert REFUSAL_RE.search(text) is None


# ─── CallCounters ────────────────────────────────────────────────────

def test_counters_start_zero():
    c = CallCounters()
    snap = c.snapshot()
    assert snap["calls"] == 0
    assert snap["failures"] == 0
    assert snap["fail_rate"] == 0.0  # no divide-by-zero
    assert snap["tool_loops_total"] == 0
    assert snap["tool_loops_avg_hops"] == 0.0


def test_counters_record_success_and_failure():
    c = CallCounters()
    c.record(True)
    c.record(True)
    c.record(False)
    snap = c.snapshot()
    assert snap["calls"] == 3
    assert snap["failures"] == 1
    assert snap["fail_rate"] == 1 / 3


def test_counters_tool_loop_avg_and_cap():
    c = CallCounters()
    c.record_tool_loop(3, capped=False)
    c.record_tool_loop(5, capped=True)
    snap = c.snapshot()
    assert snap["tool_loops_total"] == 2
    assert snap["tool_loops_capped"] == 1
    assert snap["tool_loops_avg_hops"] == 4.0


# ─── FallbackChain ───────────────────────────────────────────────────

def test_chain_primary_succeeds_short_circuits():
    calls: list[str] = []

    def primary():
        calls.append("p")
        return {"text": "yes"}

    def secondary():
        calls.append("s")
        return {"text": "should not run"}

    chain = FallbackChain(
        providers=[("primary", primary), ("secondary", secondary)],
        is_good=lambda r: bool(r and r.get("text")),
    )
    name, result = chain.run()
    assert name == "primary"
    assert result == {"text": "yes"}
    assert calls == ["p"]  # secondary never called


def test_chain_falls_through_on_bad_primary():
    def primary():
        return {"error": "down"}

    def secondary():
        return {"text": "rescued"}

    chain = FallbackChain(
        providers=[("primary", primary), ("secondary", secondary)],
        is_good=lambda r: bool(r and not r.get("error")),
    )
    name, result = chain.run()
    assert name == "secondary"
    assert result == {"text": "rescued"}


def test_chain_records_primary_only_in_counters():
    counters = CallCounters()

    chain = FallbackChain(
        providers=[
            ("primary",   lambda: {"error": "fail"}),
            ("secondary", lambda: {"text": "ok"}),
        ],
        is_good=lambda r: bool(r and not r.get("error")),
        counters=counters,
    )
    chain.run()
    # One call recorded (primary), one failure. Secondary doesn't touch counters.
    assert counters.calls == 1
    assert counters.failures == 1


def test_chain_logs_rescue_to_writer():
    seen: dict = {}

    def writer(call_site, system, user_input, primary_out, fallback_out):
        seen["call_site"]    = call_site
        seen["system"]       = system
        seen["user_input"]   = user_input
        seen["primary"]      = primary_out
        seen["fallback"]     = fallback_out

    chain = FallbackChain(
        providers=[
            ("primary",   lambda: {"error": "fail"}),
            ("secondary", lambda: {"text": "rescued"}),
        ],
        is_good=lambda r: bool(r and not r.get("error")),
        learning_log_writer=writer,
        call_site="unit_test",
        system_prompt="SYS",
        user_input="USR",
    )
    chain.run()
    assert seen["call_site"] == "unit_test"
    assert seen["primary"] == {"error": "fail"}
    assert seen["fallback"] == {"text": "rescued"}


def test_chain_does_not_log_when_primary_succeeds():
    fired: list = []
    chain = FallbackChain(
        providers=[
            ("primary",   lambda: {"text": "great"}),
            ("secondary", lambda: {"text": "unused"}),
        ],
        is_good=lambda r: bool(r and r.get("text")),
        learning_log_writer=lambda *a, **kw: fired.append(a),
        call_site="ok_path",
    )
    chain.run()
    assert fired == []  # no rescue, no log


def test_chain_exception_treated_as_failure():
    # If a provider raises, the chain should record an {"error": ...}
    # result and advance to the next. Mirrors engine.py: a thrown
    # urllib error must not crash the request handler.
    def boom():
        raise RuntimeError("network down")

    def rescuer():
        return {"text": "saved"}

    chain = FallbackChain(
        providers=[("primary", boom), ("secondary", rescuer)],
        is_good=lambda r: bool(r and r.get("text")),
    )
    name, result = chain.run()
    assert name == "secondary"
    assert result == {"text": "saved"}


def test_chain_all_fail_returns_last_result():
    chain = FallbackChain(
        providers=[
            ("a", lambda: {"error": "1"}),
            ("b", lambda: {"error": "2"}),
            ("c", lambda: {"error": "3"}),
        ],
        is_good=lambda r: bool(r and not r.get("error")),
    )
    name, result = chain.run()
    assert name == "c"
    assert result == {"error": "3"}


def test_chain_callbacks_fire():
    successes: list = []
    failures:  list = []
    chain = FallbackChain(
        providers=[
            ("primary",   lambda: {"error": "x"}),
            ("secondary", lambda: {"text": "y"}),
        ],
        is_good=lambda r: bool(r and not r.get("error")),
        on_success=lambda n, r: successes.append(n),
        on_failure=lambda n, r: failures.append(n),
    )
    chain.run()
    assert failures == ["primary"]
    assert successes == ["secondary"]


# ─── BarbaraConfig / BarbaraClient (no-network paths) ────────────────

def test_config_url_clean_strips_trailing_slash():
    c = BarbaraConfig(url="https://x.example.org/  ")
    assert c.url_clean() == "https://x.example.org"


def test_config_cf_access_off_returns_empty_headers():
    c = BarbaraConfig(cf_access_on=False,
                      cf_access_id="abc", cf_access_secret="def")
    assert c.cf_access_headers() == {}


def test_config_cf_access_on_but_empty_creds_returns_empty():
    c = BarbaraConfig(cf_access_on=True,
                      cf_access_id="", cf_access_secret="")
    assert c.cf_access_headers() == {}


def test_config_cf_access_full_returns_both_headers():
    c = BarbaraConfig(cf_access_on=True,
                      cf_access_id="ID", cf_access_secret="SEC")
    h = c.cf_access_headers()
    assert h == {"CF-Access-Client-Id": "ID",
                 "CF-Access-Client-Secret": "SEC"}


def test_client_text_returns_unavailable_when_url_blank():
    # Empty URL → no network call, deterministic shape.
    client = BarbaraClient(BarbaraConfig(url=""))
    assert client.text("hi") == {"error": "AI service temporarily unavailable"}
    assert client.text("hi", raw=True) == ""


def test_client_vision_returns_unavailable_when_url_blank():
    client = BarbaraClient(BarbaraConfig(url=""))
    out = client.vision(b"\x00", "image/jpeg", "sys", "usr")
    assert out == {"error": "AI service temporarily unavailable"}


def test_client_chat_returns_error_when_url_blank():
    client = BarbaraClient(BarbaraConfig(url=""))
    out = client.chat("sys", [], "hello")
    assert out == {"error": "Barbara not configured", "text": ""}


def test_client_chat_stream_emits_error_when_url_blank():
    client = BarbaraClient(BarbaraConfig(url=""))
    events = list(client.chat_stream("sys", [], "hi"))
    assert events == [{"type": "error", "error": "Barbara not configured"}]


def test_client_build_messages_trims_and_filters():
    msgs = BarbaraClient._build_messages(
        "SYS",
        [
            {"role": "user",      "content": " hi "},
            {"role": "assistant", "content": "hello"},
            {"role": "system",    "content": "should be dropped"},
            {"role": "user",      "content": ""},  # empty → dropped
            None,                                  # None → dropped
        ],
        "what's up?",
    )
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2] == {"role": "assistant", "content": "hello"}
    assert msgs[-1] == {"role": "user", "content": "what's up?"}
    assert len(msgs) == 4  # system + 2 valid history + final user


def test_client_build_messages_caps_content_at_6000_chars():
    huge = "x" * 9000
    msgs = BarbaraClient._build_messages("", [{"role": "user", "content": huge}],
                                         huge)
    # Both history user content and final user content trimmed to 6000.
    assert len(msgs[0]["content"]) == 6000
    assert len(msgs[1]["content"]) == 6000


# ─── SqliteResultCache ───────────────────────────────────────────────

def _tmp_cache(tmp_path):
    return SqliteResultCache(db_path=str(tmp_path / "cache.db"))


def test_cache_get_miss_returns_none(tmp_path):
    cache = _tmp_cache(tmp_path)
    assert cache.get("no-such-key") is None


def test_cache_set_then_get_roundtrip(tmp_path):
    cache = _tmp_cache(tmp_path)
    key = SqliteResultCache.make_key("claude-sonnet-4-6", "SYS", "USR")
    cache.set(key, {"text": "hello", "tokens_in": 42}, ttl_s=60)
    got = cache.get(key)
    assert got == {"text": "hello", "tokens_in": 42}


def test_cache_ttl_expiry(tmp_path):
    cache = _tmp_cache(tmp_path)
    key = SqliteResultCache.make_key("m", "s", "u")
    cache.set(key, {"text": "hi"}, ttl_s=-1)  # already expired
    assert cache.get(key) is None


def test_cache_gc_removes_expired(tmp_path):
    cache = _tmp_cache(tmp_path)
    live = SqliteResultCache.make_key("m", "s", "live")
    dead = SqliteResultCache.make_key("m", "s", "dead")
    cache.set(live, {"text": "l"}, ttl_s=3600)
    cache.set(dead, {"text": "d"}, ttl_s=-10)
    removed = cache.gc()
    assert removed == 1
    assert cache.get(live) == {"text": "l"}
    assert cache.get(dead) is None


def test_cache_key_deterministic_and_order_stable(tmp_path):
    # Same inputs → same key. Even with dict user content in different
    # insertion orders, the sort_keys=True stable dumps guarantees a
    # stable hash.
    k1 = SqliteResultCache.make_key(
        "m", "SYS", {"b": 2, "a": 1}, extra={"t": 0.1, "m": 100})
    k2 = SqliteResultCache.make_key(
        "m", "SYS", {"a": 1, "b": 2}, extra={"m": 100, "t": 0.1})
    assert k1 == k2


def test_cache_key_different_when_inputs_differ(tmp_path):
    k1 = SqliteResultCache.make_key("m", "SYS", "user A")
    k2 = SqliteResultCache.make_key("m", "SYS", "user B")
    assert k1 != k2


# ─── ClaudeClient — no-network path ──────────────────────────────────

def test_claude_no_api_key_returns_error_shape():
    client = ClaudeClient(ClaudeConfig(api_key=""))
    out = client.messages("sys", "user")
    assert out == {"error": "no api key", "text": ""}


def test_claude_no_api_key_records_failure_when_counters_wired():
    counters = CallCounters()
    client = ClaudeClient(ClaudeConfig(api_key=""), counters=counters)
    client.messages("sys", "user")
    assert counters.calls == 1
    assert counters.failures == 1


# ─── ClaudeClient — mocked network ────────────────────────────────────

class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _ok_response_bytes(text: str = "hi", model: str = "claude-sonnet-4-6",
                       tokens_in: int = 100, tokens_out: int = 20,
                       cache_read: int = 0, cache_creation: int = 0) -> bytes:
    return json.dumps({
        "id": "msg_abc",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens":  tokens_in,
            "output_tokens": tokens_out,
            "cache_read_input_tokens":     cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    }).encode()


def test_claude_messages_returns_normalized_shape():
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"))
    fake = _FakeResp(_ok_response_bytes(text="answer"))
    with patch("urllib.request.urlopen", return_value=fake):
        out = client.messages("SYS", "hello")
    assert out["text"] == "answer"
    assert out["model"] == "claude-sonnet-4-6"
    assert out["tokens_in"] == 100
    assert out["tokens_out"] == 20
    assert out["cached"] is False


def test_claude_messages_wraps_system_with_cache_control_by_default():
    # Intercept the Request object built to assert body shape.
    captured: dict = {}

    def fake_urlopen(req, timeout=None, context=None):
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(_ok_response_bytes())

    client = ClaudeClient(ClaudeConfig(api_key="sk-test"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.messages("big static prompt", "small dynamic user")

    body = captured["body"]
    assert body["system"] == [{
        "type": "text", "text": "big static prompt",
        "cache_control": {"type": "ephemeral"},
    }]
    assert body["messages"] == [{"role": "user",
                                  "content": [{"type": "text",
                                                "text": "small dynamic user"}]}]
    # Anthropic version + key headers present. (urllib.Request lowercases
    # header keys internally.)
    assert captured["headers"]["X-api-key"] == "sk-test"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"


def test_claude_messages_can_disable_cache():
    captured: dict = {}
    def fake_urlopen(req, timeout=None, context=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(_ok_response_bytes())
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.messages("SYS", "USR", cache_system=False)
    sys_blocks = captured["body"]["system"]
    assert sys_blocks == [{"type": "text", "text": "SYS"}]
    assert "cache_control" not in sys_blocks[0]


def test_claude_messages_accepts_content_blocks_for_vision():
    # User is a list of content blocks (image + text) instead of a str.
    captured: dict = {}
    def fake_urlopen(req, timeout=None, context=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(_ok_response_bytes())
    user_blocks = [
        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/jpeg",
                                      "data": "AAAA"}},
        {"type": "text", "text": "what's in this image?"},
    ]
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.messages("SYS", user_blocks)
    assert captured["body"]["messages"][0]["content"] == user_blocks


def test_claude_messages_reads_cache_hit_and_skips_network(tmp_path):
    cache = _tmp_cache(tmp_path)
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"), cache=cache)

    key = SqliteResultCache.make_key(
        DEFAULT_CLAUDE_MODEL, "SYS", "USR",
        extra={"max_tokens": 1024, "temperature": 0.1, "stop_sequences": []},
    )
    cache.set(key, {"text": "pre-baked", "model": "x", "tokens_in": 1,
                    "tokens_out": 1, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0, "stop_reason": "end_turn",
                    "cached": False},
              ttl_s=60)

    with patch("urllib.request.urlopen",
               side_effect=AssertionError("network should not be called")):
        out = client.messages("SYS", "USR")

    assert out["text"] == "pre-baked"
    assert out["cached"] is True


def test_claude_messages_writes_to_cache_on_success(tmp_path):
    cache = _tmp_cache(tmp_path)
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"), cache=cache)
    with patch("urllib.request.urlopen",
               return_value=_FakeResp(_ok_response_bytes(text="fresh"))):
        client.messages("SYS", "USR")
    # Second call must hit the cache — assert by making urlopen raise.
    with patch("urllib.request.urlopen",
               side_effect=AssertionError("cached, should not call")):
        out = client.messages("SYS", "USR")
    assert out["text"] == "fresh"
    assert out["cached"] is True


def test_claude_messages_http_error_returns_error_shape():
    import urllib.error as ue
    def raise_http(*_a, **_kw):
        raise ue.HTTPError(url="", code=529, msg="overloaded",
                            hdrs=None, fp=io.BytesIO(b"overloaded"))
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"))
    with patch("urllib.request.urlopen", side_effect=raise_http):
        out = client.messages("SYS", "USR")
    assert "error" in out and "529" in out["error"]
    assert out["text"] == ""


def test_claude_batch_create_requires_api_key():
    client = ClaudeClient(ClaudeConfig(api_key=""))
    out = client.batch_create([{"custom_id": "x", "params": {"user": "u"}}])
    assert out == {"error": "no api key"}


def test_claude_batch_create_empty_request_returns_error():
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"))
    out = client.batch_create([])
    assert out == {"error": "empty batch"}


def test_claude_batch_create_wraps_system_with_cache_control():
    captured: dict = {}
    def fake_urlopen(req, timeout=None, context=None):
        captured["body"] = json.loads(req.data.decode())
        captured["headers"] = dict(req.headers)
        return _FakeResp(json.dumps({"id": "batch_abc",
                                     "processing_status": "in_progress"}).encode())
    client = ClaudeClient(ClaudeConfig(api_key="sk-test"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.batch_create([{
            "custom_id": "row-1",
            "params": {"system": "SYS", "user": "USR", "max_tokens": 500},
        }])
    body = captured["body"]
    assert body["requests"][0]["custom_id"] == "row-1"
    inner = body["requests"][0]["params"]
    assert inner["system"] == [{"type": "text", "text": "SYS",
                                 "cache_control": {"type": "ephemeral"}}]
    assert inner["max_tokens"] == 500
    # Batch API is a beta — header must be present.
    assert captured["headers"].get("Anthropic-beta") == \
        "message-batches-2024-09-24"
