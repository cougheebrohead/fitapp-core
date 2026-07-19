"""Tests for fitapp_core.menu.pick_menu_items.

Network paths (Gemini, Claude) are stubbed at the urllib boundary.
Pure-function tests: parsing + sanitization + ranking + bounds.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

from fitapp_core import pick_menu_items
from fitapp_core import menu as M


# ─── _parse_picks ─────────────────────────────────────────────────────


def test_parses_clean_json():
    text = json.dumps({
        "venue": "Chipotle",
        "picks": [
            {"name": "Chicken bowl, brown rice, fajita veggies, salsa",
             "rank": 1, "calories": 580, "protein": 48, "carbs": 60, "fat": 18,
             "fits_macros": True, "allergen_flags": [],
             "why": "Hits remaining 480 cal / 38g protein with room to spare.",
             "modifications": "no cheese"},
            {"name": "Steak salad", "rank": 2, "calories": 520, "protein": 42,
             "carbs": 22, "fat": 30, "fits_macros": True, "allergen_flags": ["dairy"],
             "why": "Lean cut, lower carb.", "modifications": ""},
        ],
        "avoid": [{"name": "Burrito with rice + beans + cheese + sour cream",
                   "why": "Pushes you 200+ cal over and contains dairy."}],
        "warnings": [],
    })
    out = M._parse_picks(text)
    assert out["venue"] == "Chipotle"
    assert len(out["picks"]) == 2
    assert out["picks"][0]["rank"] == 1
    assert out["picks"][0]["calories"] == 580
    assert out["picks"][0]["fits_macros"] is True
    assert "dairy" in out["picks"][1]["allergen_flags"]
    assert len(out["avoid"]) == 1


def test_strips_markdown_fences():
    raw = '```json\n{"venue":"Sweetgreen","picks":[],"avoid":[],"warnings":[]}\n```'
    out = M._parse_picks(raw)
    assert out["venue"] == "Sweetgreen"


def test_handles_unfenced_json_with_preamble():
    raw = 'Sure! Here you go:\n{"venue":"Blaze","picks":[{"name":"Pizza","rank":1}],"avoid":[],"warnings":[]}'
    out = M._parse_picks(raw)
    assert out["venue"] == "Blaze"
    assert out["picks"][0]["name"] == "Pizza"


def test_unparseable_returns_warnings():
    out = M._parse_picks("not json at all")
    assert out["picks"] == []
    assert any("not parseable" in w for w in out["warnings"])


def test_empty_response():
    out = M._parse_picks("")
    assert out["picks"] == []
    assert any("empty" in w for w in out["warnings"])


def test_picks_capped_at_eight():
    text = json.dumps({
        "picks": [{"name": f"Item {i}", "rank": i, "calories": 500, "protein": 20}
                  for i in range(1, 20)],
        "avoid": [], "warnings": [],
    })
    out = M._parse_picks(text)
    assert len(out["picks"]) == 8


def test_picks_sorted_by_rank():
    text = json.dumps({
        "picks": [
            {"name": "C", "rank": 3, "calories": 500},
            {"name": "A", "rank": 1, "calories": 500},
            {"name": "B", "rank": 2, "calories": 500},
        ],
        "avoid": [], "warnings": [],
    })
    out = M._parse_picks(text)
    assert [p["name"] for p in out["picks"]] == ["A", "B", "C"]


def test_avoid_capped_at_six():
    text = json.dumps({
        "picks": [],
        "avoid": [{"name": f"X{i}", "why": "."} for i in range(20)],
        "warnings": [],
    })
    out = M._parse_picks(text)
    assert len(out["avoid"]) == 6


def test_string_lengths_truncated():
    text = json.dumps({
        "picks": [{"name": "x" * 500, "rank": 1, "why": "y" * 1000,
                   "modifications": "z" * 500}],
        "avoid": [], "warnings": [],
    })
    out = M._parse_picks(text)
    p = out["picks"][0]
    assert len(p["name"]) <= 240
    assert len(p["why"]) <= 400
    assert len(p["modifications"]) <= 200


def test_allergen_flags_normalized():
    text = json.dumps({
        "picks": [{"name": "Pad Thai", "rank": 1, "allergen_flags": ["PEANUT", "Soy ", "egg"]}],
        "avoid": [], "warnings": [],
    })
    out = M._parse_picks(text)
    flags = out["picks"][0]["allergen_flags"]
    assert "peanut" in flags and "soy" in flags and "egg" in flags


def test_int_coercion_from_strings():
    text = json.dumps({
        "picks": [{"name": "X", "rank": 1, "calories": "580 kcal", "protein": "48g"}],
        "avoid": [], "warnings": [],
    })
    out = M._parse_picks(text)
    p = out["picks"][0]
    assert p["calories"] == 580
    assert p["protein"] == 48


# ─── pick_menu_items end-to-end (provider stubbed) ───────────────────


def test_pick_menu_items_uses_gemini_first(monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "fake-gemini")
    monkeypatch.delenv("CLAUDE_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    captured = {}

    def fake_gemini(b64, media_type, api_key, prompt):
        captured["b64"] = b64
        captured["prompt"] = prompt
        captured["api_key"] = api_key
        return {"venue": "Stub", "picks": [], "avoid": [], "raw_text": "", "warnings": []}

    with patch.object(M, "_gemini_pick", side_effect=fake_gemini):
        out = pick_menu_items(b"\xff\xd8\xff\xe0", "image/jpeg",
                              context={"remaining_calories": 500,
                                       "allergies": "dairy",
                                       "venue_hint": "Sweetgreen"})

    assert out["venue"] == "Stub"
    assert "USER CONTEXT" in captured["prompt"]
    assert "Remaining today: 500 cal" in captured["prompt"]
    assert "Allergies (NEVER recommend foods containing these): dairy" in captured["prompt"]
    assert "Venue (hint): Sweetgreen" in captured["prompt"]
    assert captured["api_key"] == "fake-gemini"


def test_pick_menu_items_falls_back_to_claude(monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "g")
    monkeypatch.setenv("CLAUDE_KEY", "c")

    def fail_gemini(*a, **k): raise RuntimeError("gemini down")

    captured = {}
    def fake_claude(b64, media_type, api_key, system_prompt, ctx_block):
        captured["called"] = True
        captured["system_prompt"] = system_prompt
        captured["ctx_block"] = ctx_block
        return {"venue": "Claude-Stub", "picks": [], "avoid": [], "raw_text": "", "warnings": []}

    with patch.object(M, "_gemini_pick", side_effect=fail_gemini), \
         patch.object(M, "_claude_pick", side_effect=fake_claude):
        out = pick_menu_items(b"\xff\xd8\xff\xe0", "image/jpeg")

    assert captured.get("called") is True
    assert out["venue"] == "Claude-Stub"
    # Refactor invariant: static instructions live in system_prompt,
    # per-request user context lives separately (cache-friendly split).
    assert "menu nutritionist" in captured["system_prompt"].lower()


def test_pick_menu_items_no_keys_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        pick_menu_items(b"\xff\xd8\xff\xe0", "image/jpeg")
    except RuntimeError as e:
        assert "no ai key" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError when no AI key configured")


def test_pick_menu_items_accepts_b64_string(monkeypatch):
    monkeypatch.setenv("CLAUDE_KEY", "c")
    monkeypatch.delenv("GEMINI_KEY", raising=False)
    captured = {}
    def fake_claude(b64, media_type, api_key, system_prompt, ctx_block):
        captured["b64"] = b64
        return {"venue": None, "picks": [], "avoid": [], "raw_text": "", "warnings": []}
    with patch.object(M, "_claude_pick", side_effect=fake_claude):
        pick_menu_items("aGVsbG8=", "image/jpeg")
    assert captured["b64"] == "aGVsbG8="


def test_pick_menu_items_rejects_non_bytes_non_str(monkeypatch):
    monkeypatch.setenv("CLAUDE_KEY", "c")
    try:
        pick_menu_items(12345, "image/jpeg")  # type: ignore[arg-type]
    except TypeError as e:
        assert "bytes" in str(e).lower() or "base64" in str(e).lower()
        return
    raise AssertionError("expected TypeError for int input")


# ─── runner ──────────────────────────────────────────────────────────


class _M:
    """Tiny monkeypatch shim so the script runner mimics pytest's monkeypatch."""
    def __init__(self):
        self._undo = []
    def setenv(self, k, v):
        prev = os.environ.get(k)
        os.environ[k] = v
        self._undo.append((k, prev))
    def delenv(self, k, raising=True):
        if k in os.environ:
            prev = os.environ.get(k)
            del os.environ[k]
            self._undo.append((k, prev))
        elif raising:
            raise KeyError(k)
    def undo(self):
        for k, v in self._undo:
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


import os  # for the shim
import inspect


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = []
    for fn in fns:
        sig = inspect.signature(fn)
        mp = _M()
        try:
            if "monkeypatch" in sig.parameters:
                fn(mp)
            else:
                fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            fails.append((fn.__name__, e))
            print(f"  ✗ {fn.__name__}: {e}")
            traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{len(fns) - len(fails)}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
