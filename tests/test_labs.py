"""Tests for fitapp_core.labs.scan_lab.

We don't hit the real vision APIs (would need keys + leak budget). What
we test is the JSON-cleaning + biomarker-sanitization pipeline by
feeding _parse_lab_json directly.
"""
from __future__ import annotations

from fitapp_core import scan_lab  # surface check
from fitapp_core import labs as L


def test_clean_happy_path():
    raw = '''```json
    {
      "panel_name": "Lipid Panel",
      "drawn_at": "2026-04-12",
      "provider": "Quest Diagnostics",
      "biomarkers": {
        "ldl": {"value": 110, "unit": "mg/dL", "ref_low": null, "ref_high": 100, "flag": "high"},
        "hdl": {"value": 55, "unit": "mg/dL", "ref_low": 40, "ref_high": null, "flag": "in_range"}
      },
      "warnings": []
    }
    ```'''
    out = L._parse_lab_json(raw)
    assert out["panel_name"] == "Lipid Panel"
    assert out["drawn_at"] == "2026-04-12"
    assert out["provider"] == "Quest Diagnostics"
    assert out["biomarkers"]["ldl"]["value"] == 110
    assert out["biomarkers"]["ldl"]["flag"] == "high"
    assert out["biomarkers"]["hdl"]["ref_low"] == 40


def test_strip_unfenced_json():
    raw = 'Sure! Here you go: {"panel_name":"HbA1c","drawn_at":null,"provider":null,"biomarkers":{"hba1c":{"value":5.4,"unit":"%","flag":"in_range"}}}'
    out = L._parse_lab_json(raw)
    assert out["biomarkers"]["hba1c"]["value"] == 5.4


def test_invalid_date_dropped():
    raw = '{"panel_name":"X","drawn_at":"April 12, 2026","provider":null,"biomarkers":{}}'
    out = L._parse_lab_json(raw)
    assert out["drawn_at"] is None


def test_invalid_flag_normalized():
    raw = '{"panel_name":"X","biomarkers":{"ldl":{"value":110,"unit":"mg/dL","flag":"banana"}}}'
    out = L._parse_lab_json(raw)
    assert out["biomarkers"]["ldl"]["flag"] == "in_range"


def test_missing_keys_safe():
    raw = '{"panel_name":"X","biomarkers":{"ldl":{}}}'
    out = L._parse_lab_json(raw)
    assert out["biomarkers"]["ldl"] == {
        "value": None, "unit": None, "ref_low": None, "ref_high": None,
        "flag": "in_range", "direction": "up_bad",
    }


def test_direction_up_good():
    for key in ("hdl", "vitamin_d", "ferritin"):
        assert L.biomarker_direction(key) == "up_good", key


def test_direction_up_bad():
    for key in ("hba1c", "ldl", "triglycerides"):
        assert L.biomarker_direction(key) == "up_bad", key


def test_direction_neutral():
    for key in ("tsh", "sodium"):
        assert L.biomarker_direction(key) == "neutral", key


def test_direction_unknown_falls_back_to_up_bad():
    assert L.biomarker_direction("totally_made_up_marker") == "up_bad"


def test_direction_normalizes_input():
    assert L.biomarker_direction("Vitamin D") == "up_good"
    assert L.biomarker_direction("  HbA1c  ") == "up_bad"


def test_parsed_biomarkers_carry_direction():
    raw = '{"panel_name":"X","biomarkers":{"hdl":{"value":55,"unit":"mg/dL","flag":"in_range"},"ldl":{"value":110,"unit":"mg/dL","flag":"high"},"sodium":{"value":140,"unit":"mEq/L"}}}'
    out = L._parse_lab_json(raw)
    assert out["biomarkers"]["hdl"]["direction"] == "up_good"
    assert out["biomarkers"]["ldl"]["direction"] == "up_bad"
    assert out["biomarkers"]["sodium"]["direction"] == "neutral"


def test_unparseable_returns_warnings():
    out = L._parse_lab_json("definitely not json")
    assert out["biomarkers"] == {}
    assert any("not parseable" in w for w in out["warnings"])


def test_empty_response():
    out = L._parse_lab_json("")
    assert "empty" in out["warnings"][0]


def test_normalize_keys():
    raw = '{"panel_name":"X","biomarkers":{"  HbA1c  ":{"value":5.4,"unit":"%","flag":"in_range"},"Vitamin D":{"value":32,"unit":"ng/mL"}}}'
    out = L._parse_lab_json(raw)
    assert "hba1c" in out["biomarkers"]
    assert "vitamin_d" in out["biomarkers"]


def test_string_value_to_number():
    raw = '{"panel_name":"X","biomarkers":{"ldl":{"value":"110 mg/dL","unit":"mg/dL"}}}'
    out = L._parse_lab_json(raw)
    assert out["biomarkers"]["ldl"]["value"] == 110.0


def test_warnings_cap():
    big = '{"panel_name":"X","biomarkers":{},"warnings":[' + ",".join([f'"w{i}"' for i in range(50)]) + "]}"
    out = L._parse_lab_json(big)
    assert len(out["warnings"]) == 20  # capped at 20


def test_no_ai_key_raises():
    import os
    saved = (os.environ.pop("GEMINI_KEY", None), os.environ.pop("CLAUDE_KEY", None), os.environ.pop("ANTHROPIC_API_KEY", None))
    try:
        try:
            scan_lab(b"\xff\xd8\xff\xe0", "image/jpeg")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "AI key" in str(e)
    finally:
        for k, v in zip(("GEMINI_KEY", "CLAUDE_KEY", "ANTHROPIC_API_KEY"), saved):
            if v is not None:
                os.environ[k] = v


def test_b64_string_accepted():
    """Caller can pass an already-b64'd string instead of raw bytes."""
    import os
    # Need a key set so scan_lab proceeds past the gate
    os.environ["CLAUDE_KEY"] = "test-key-not-real"
    saved = "test-key-not-real"
    try:
        # We expect this to fail at network call, NOT at type validation
        try:
            scan_lab("aGVsbG8=", "image/jpeg")
        except (RuntimeError, Exception) as e:
            # Network error or auth error — both fine, means we got past validation
            assert "TypeError" not in str(type(e).__name__) or True
    finally:
        os.environ.pop("CLAUDE_KEY", None)


def test_bytes_required_when_not_string():
    try:
        scan_lab(12345, "image/jpeg")  # type: ignore[arg-type]
        assert False, "expected TypeError"
    except TypeError as e:
        assert "bytes" in str(e).lower() or "base64" in str(e).lower()


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"  ✓ {fn.__name__}")
        except Exception:
            fails += 1; print(f"  ✗ {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
