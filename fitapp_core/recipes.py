"""Recipe finder — AI-suggested recipes ranked against the user's
remaining macros, allergens, conditions, dietary preferences, and any
active specialty modules (GLP-1 / perimenopause / pregnancy).

Single public entrypoint:

    find_recipes(context, lang=None) -> dict

`context` shape:
    {
      "remaining_calories": int, "remaining_protein": int,
      "remaining_carbs": int, "remaining_fat": int,
      "allergies":     "comma string",
      "conditions":    "comma string",
      "dietary":       "comma string",
      "specialty":     ["GLP-1...", "PREGNANCY..."],   # life-stage blocks
      "meal_type":     "breakfast|lunch|dinner|snack" | None,
      "ingredients_on_hand": "free text"               # optional, "what's in fridge"
    }

Returns:
    {
      "recipes": [
        {
          "name": "Sheet-pan chicken thighs with broccoli + sweet potato",
          "rank": 1,
          "calories": 540, "protein": 42, "carbs": 38, "fat": 22,
          "prep_time_min": 35,
          "fits_macros": true,
          "allergen_safe": true,
          "ingredients": [
            {"item": "Boneless skinless chicken thighs", "qty": "1 lb"},
            ...
          ],
          "steps": ["Preheat oven to 425F.", "Toss broccoli with olive oil...", ...],
          "why":  "Hits remaining 480 cal / 38g protein with room. No dairy.",
          "modifications": "swap broccoli for green beans if you don't have it"
        }
      ],
      "warnings": [...]
    }

Stdlib-only (urllib + json + re). Same Gemini-first / Claude-fallback
pattern as menu.py.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional


_MAX_RESPONSE_BYTES = 200_000
_MODEL_GEMINI = "gemini-2.0-flash"
_MODEL_CLAUDE = "claude-sonnet-4-6"


_PROMPT_HEAD = """You are a personal-nutrition recipe coach. Suggest 3-5
recipes ranked for THIS user RIGHT NOW based on their remaining macros
today, allergens, conditions, dietary preferences, and life-stage modules.

Return STRICT JSON only — no markdown fences.

Output shape:
{
  "recipes": [
    {
      "name":           "<recipe title, full sentence>",
      "rank":           <1..N, 1 is best fit>,
      "calories":       <integer kcal estimate per serving>,
      "protein":        <grams>,
      "carbs":          <grams>,
      "fat":            <grams>,
      "prep_time_min":  <integer total time including cook>,
      "fits_macros":    <true if calories <= remaining and protein >= 0.4*remaining_protein>,
      "allergen_safe":  <true if no listed allergen>,
      "ingredients": [
        {"item": "<ingredient name>", "qty": "<amount + unit>"}
      ],
      "steps": ["<step 1>", "<step 2>", ...],
      "why":            "<1-2 sentences citing user numbers and constraints>",
      "modifications":  "<short string of substitutions if relevant, else empty>"
    }
  ],
  "warnings": ["<short notes about ambiguity or missing info>"]
}

Hard rules:
- NEVER include an ingredient containing one of the listed allergens.
- For users with diabetes / GLP-1 / cardio: protein-first, lean cuts,
  low-glycemic sides. For users with kidney conditions: watch sodium
  + protein. For PREGNANCY: NEVER include raw fish, raw shellfish,
  high-mercury fish (shark/swordfish/king mackerel/tilefish/marlin/
  bigeye tuna), deli meats unheated, soft unpasteurized cheeses, raw
  eggs, alcohol, liver, raw sprouts.
- Calories per serving must fit within remaining_calories (or close);
  recipe yields 1 serving unless explicitly stated "Yields N".
- Round macros to integers. Don't fabricate exact precision.
- Steps should be cookable: time + temperature + sensory cues.
- Total prep_time_min covers prep + cook.
- If meal_type is given, suggest only recipes appropriate for that meal.
- If ingredients_on_hand is given, prefer recipes that use those
  ingredients prominently.
"""


def find_recipes(
    context: Optional[dict[str, Any]] = None,
    lang: Optional[str] = None,
) -> dict[str, Any]:
    """Build context block, call vision-free LLM, parse + sanitize."""
    ctx = context or {}
    ctx_lines = ["USER CONTEXT:"]
    if (ctx.get("remaining_calories") is not None
            or ctx.get("remaining_protein") is not None):
        ctx_lines.append(
            f"  Remaining today: {ctx.get('remaining_calories', '?')} cal, "
            f"{ctx.get('remaining_protein', '?')}g protein, "
            f"{ctx.get('remaining_carbs', '?')}g carbs, "
            f"{ctx.get('remaining_fat', '?')}g fat."
        )
    if ctx.get("allergies"):
        ctx_lines.append(f"  Allergies (NEVER include): {ctx['allergies']}")
    if ctx.get("conditions"):
        ctx_lines.append(f"  Conditions: {ctx['conditions']}")
    if ctx.get("dietary"):
        ctx_lines.append(f"  Dietary preferences: {ctx['dietary']}")
    if ctx.get("meal_type"):
        ctx_lines.append(f"  Meal: {ctx['meal_type']}")
    if ctx.get("ingredients_on_hand"):
        ctx_lines.append(f"  In the kitchen: {ctx['ingredients_on_hand']}")
    for block in (ctx.get("specialty") or []):
        ctx_lines.append("  " + str(block).replace("\n", " "))
    ctx_block = "\n".join(ctx_lines)

    text_prompt = _PROMPT_HEAD + "\n\n" + ctx_block
    if lang and lang != "en":
        text_prompt = f"Respond entirely in {lang}.\n\n" + text_prompt

    gem_key = (os.environ.get("GEMINI_KEY") or "").strip().strip('"').strip("'")
    claude_key = (
        (os.environ.get("CLAUDE_KEY") or "").strip().strip('"').strip("'")
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    )
    if not gem_key and not claude_key:
        raise RuntimeError("find_recipes: no AI key configured")

    last_err: Optional[Exception] = None
    if gem_key:
        try:
            return _gemini_recipes(text_prompt, gem_key)
        except Exception as e:
            last_err = e
    if claude_key:
        try:
            return _claude_recipes(text_prompt, claude_key)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"find_recipes failed: {last_err}")


# ─── providers ────────────────────────────────────────────────────────


def _gemini_recipes(prompt: str, api_key: str) -> dict[str, Any]:
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 3000},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{_MODEL_GEMINI}:generateContent?key=" + urllib.parse.quote(api_key))
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read(_MAX_RESPONSE_BYTES + 1)
    result = json.loads(raw)
    text = (
        result.get("candidates", [{}])[0]
              .get("content", {}).get("parts", [{}])[0].get("text", "")
    )
    return _parse_recipes(text)


def _claude_recipes(prompt: str, api_key: str) -> dict[str, Any]:
    body = {
        "model": _MODEL_CLAUDE,
        "max_tokens": 3000,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read(_MAX_RESPONSE_BYTES + 1)
    result = json.loads(raw)
    text = (result.get("content", [{}])[0] or {}).get("text", "")
    return _parse_recipes(text)


# ─── parsing ──────────────────────────────────────────────────────────


def _parse_recipes(text: str) -> dict[str, Any]:
    if not text:
        return _empty(["empty model response"])
    cleaned = re.sub(r"^```json\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return _empty(["model output not parseable"])
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            return _empty(["model output not parseable"])

    recipes_in = parsed.get("recipes") or []
    recipes_out: list[dict[str, Any]] = []
    for i, rcp in enumerate(recipes_in[:5]):
        if not isinstance(rcp, dict):
            continue
        ingredients = []
        for ing in (rcp.get("ingredients") or [])[:30]:
            if isinstance(ing, dict):
                item = _str(ing.get("item"))[:120]
                qty  = _str(ing.get("qty") or ing.get("amount"))[:60]
                if item:
                    ingredients.append({"item": item, "qty": qty})
            elif isinstance(ing, str):
                s = _str(ing)
                if s:
                    ingredients.append({"item": s[:120], "qty": ""})
        steps = [_str(s)[:400] for s in (rcp.get("steps") or [])[:20] if _str(s)]
        recipes_out.append({
            "name":          _str(rcp.get("name"))[:200],
            "rank":          _int(rcp.get("rank")) or (i + 1),
            "calories":      _int(rcp.get("calories")),
            "protein":       _int(rcp.get("protein")),
            "carbs":         _int(rcp.get("carbs")),
            "fat":           _int(rcp.get("fat")),
            "prep_time_min": _int(rcp.get("prep_time_min") or rcp.get("time_min")),
            "fits_macros":   bool(rcp.get("fits_macros")),
            "allergen_safe": rcp.get("allergen_safe", True) is not False,
            "ingredients":   ingredients,
            "steps":         steps,
            "why":           _str(rcp.get("why"))[:400],
            "modifications": _str(rcp.get("modifications"))[:300],
        })
    recipes_out.sort(key=lambda x: x.get("rank") or 99)
    warnings = [_str(w)[:300] for w in (parsed.get("warnings") or []) if _str(w)][:10]
    return {"recipes": recipes_out, "warnings": warnings}


def _empty(warnings: list[str]) -> dict[str, Any]:
    return {"recipes": [], "warnings": warnings}


def _str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+", v)
        if m:
            try:
                return int(m.group())
            except ValueError:
                return None
    return None
