"""Unit tests for fitapp_core.recipes.find_recipes.

Network paths stubbed at the urllib boundary. Exercises the parser,
sanitization, ranking, and the prompt-builder context block.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

from fitapp_core import find_recipes
from fitapp_core import recipes as R


def test_parses_clean_json():
    text = json.dumps({
        "recipes": [
            {"name": "Sheet-pan chicken thighs",
             "rank": 1, "calories": 540, "protein": 42, "carbs": 38, "fat": 22,
             "prep_time_min": 35, "fits_macros": True, "allergen_safe": True,
             "ingredients": [
                {"item": "Chicken thighs", "qty": "1 lb"},
                {"item": "Broccoli", "qty": "2 cups"}],
             "steps": ["Preheat oven to 425F.", "Toss broccoli with oil."],
             "why": "Hits remaining 480 cal / 38g protein.",
             "modifications": "swap broccoli for green beans"}
        ],
        "warnings": [],
    })
    out = R._parse_recipes(text)
    assert len(out["recipes"]) == 1
    r = out["recipes"][0]
    assert r["name"].startswith("Sheet-pan")
    assert r["calories"] == 540
    assert r["fits_macros"] is True
    assert r["allergen_safe"] is True
    assert len(r["ingredients"]) == 2
    assert r["ingredients"][0]["qty"] == "1 lb"
    assert len(r["steps"]) == 2


def test_strips_markdown_fences():
    raw = '```json\n{"recipes":[],"warnings":[]}\n```'
    out = R._parse_recipes(raw)
    assert out["recipes"] == []


def test_unparseable_returns_warnings():
    out = R._parse_recipes("not json at all")
    assert out["recipes"] == []
    assert any("not parseable" in w for w in out["warnings"])


def test_recipes_capped_at_five():
    text = json.dumps({
        "recipes": [{"name": f"R{i}", "rank": i, "calories": 500} for i in range(1, 12)],
        "warnings": [],
    })
    out = R._parse_recipes(text)
    assert len(out["recipes"]) == 5


def test_recipes_sorted_by_rank():
    text = json.dumps({
        "recipes": [
            {"name": "C", "rank": 3},
            {"name": "A", "rank": 1},
            {"name": "B", "rank": 2},
        ],
        "warnings": [],
    })
    out = R._parse_recipes(text)
    assert [r["name"] for r in out["recipes"]] == ["A", "B", "C"]


def test_string_ingredient_falls_through():
    text = json.dumps({
        "recipes": [{"name": "X", "rank": 1, "ingredients": ["1 lb chicken thighs", "2 cups broccoli"]}],
        "warnings": [],
    })
    out = R._parse_recipes(text)
    ings = out["recipes"][0]["ingredients"]
    assert len(ings) == 2
    assert ings[0]["item"] == "1 lb chicken thighs"
    assert ings[0]["qty"] == ""


def test_int_coercion_from_strings():
    text = json.dumps({
        "recipes": [{"name": "X", "rank": 1, "calories": "540 kcal", "prep_time_min": "35 min"}],
        "warnings": [],
    })
    out = R._parse_recipes(text)
    r = out["recipes"][0]
    assert r["calories"] == 540
    assert r["prep_time_min"] == 35


def test_lengths_truncated():
    text = json.dumps({
        "recipes": [{"name": "x" * 500, "rank": 1, "why": "y" * 1000,
                     "modifications": "z" * 500}],
        "warnings": [],
    })
    out = R._parse_recipes(text)
    r = out["recipes"][0]
    assert len(r["name"]) <= 200
    assert len(r["why"]) <= 400
    assert len(r["modifications"]) <= 300


def test_find_recipes_uses_gemini_first(monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "fake-gemini")
    monkeypatch.delenv("CLAUDE_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured = {}
    def fake_gem(prompt, key):
        captured["prompt"] = prompt; captured["key"] = key
        return {"recipes": [], "warnings": []}
    with patch.object(R, "_gemini_recipes", side_effect=fake_gem):
        find_recipes(context={
            "remaining_calories": 500, "remaining_protein": 38,
            "allergies": "dairy", "meal_type": "dinner",
            "specialty": ["GLP-1: protein floor 1.5g/kg"],
        })
    p = captured["prompt"]
    assert "USER CONTEXT" in p
    assert "Remaining today: 500 cal" in p
    assert "Allergies (NEVER include): dairy" in p
    assert "Meal: dinner" in p
    assert "GLP-1: protein floor" in p
    assert captured["key"] == "fake-gemini"


def test_find_recipes_falls_back_to_claude(monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "g")
    monkeypatch.setenv("CLAUDE_KEY", "c")
    def boom(*a, **k): raise RuntimeError("gemini down")
    captured = {"called": False}
    def claude(prompt, key):
        captured["called"] = True
        return {"recipes": [], "warnings": []}
    with patch.object(R, "_gemini_recipes", side_effect=boom), \
         patch.object(R, "_claude_recipes", side_effect=claude):
        find_recipes(context={"remaining_calories": 500})
    assert captured["called"] is True


def test_find_recipes_no_keys_raises(monkeypatch):
    for k in ("GEMINI_KEY", "CLAUDE_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    try:
        find_recipes(context={"remaining_calories": 500})
    except RuntimeError as e:
        assert "no ai key" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError")


# ─── runner with monkeypatch shim (consistent with other test files) ──


class _M:
    def __init__(self): self._undo = []
    def setenv(self, k, v):
        prev = os.environ.get(k); os.environ[k] = v
        self._undo.append((k, prev))
    def delenv(self, k, raising=True):
        if k in os.environ:
            prev = os.environ.get(k); del os.environ[k]
            self._undo.append((k, prev))
        elif raising:
            raise KeyError(k)
    def undo(self):
        for k, v in self._undo:
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


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
            if "monkeypatch" in sig.parameters: fn(mp)
            else: fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            fails.append((fn.__name__, e))
            print(f"  ✗ {fn.__name__}: {e}")
            traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{len(fns) - len(fails)}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
