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

from fitapp_core.ai import (
    BarbaraClient,
    BarbaraConfig,
    CallCounters,
    DEFAULT_QUALITY_THRESHOLD,
    FallbackChain,
    REFUSAL_RE,
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
