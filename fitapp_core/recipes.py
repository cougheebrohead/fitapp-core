"""Recipe finder + meal swap — AI-suggested recipes ranked against the
user's remaining macros, allergens, conditions, dietary preferences,
and any active specialty modules (GLP-1 / perimenopause / pregnancy).

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

from fitapp_core.ai import ClaudeClient, ClaudeConfig, SqliteResultCache


_MAX_RESPONSE_BYTES = 200_000
_MODEL_GEMINI = "gemini-2.0-flash"
_MODEL_CLAUDE = "claude-sonnet-4-6"


def _claude_client(api_key: str) -> ClaudeClient:
    """Build a ClaudeClient with the shared result cache. Reads
    FITAPP_CORE_CACHE_DB from env if set, else uses the default
    (~/.fitapp-core/result-cache.db)."""
    return ClaudeClient(
        ClaudeConfig(api_key=api_key, default_model=_MODEL_CLAUDE, timeout=45),
        cache=SqliteResultCache(
            db_path=os.environ.get("FITAPP_CORE_CACHE_DB") or None
        ),
    )


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

    # Static instructions in `system` (cache-friendly), dynamic user
    # context stays outside. Language directive prepended to system so
    # the per-language cache entry stays warm on repeat calls.
    system_prompt = _PROMPT_HEAD
    if lang and lang != "en":
        system_prompt = f"Respond entirely in {lang}.\n\n" + system_prompt
    # Gemini has no cache_control; keep its combined-string interface.
    gemini_prompt = system_prompt + "\n\n" + ctx_block

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
            return _gemini_recipes(gemini_prompt, gem_key)
        except Exception as e:
            last_err = e
    if claude_key:
        try:
            return _claude_recipes(system_prompt, ctx_block, claude_key)
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


def _claude_recipes(system_prompt: str, user_context: str,
                     api_key: str) -> dict[str, Any]:
    out = _claude_client(api_key).messages(
        system=system_prompt,
        user=user_context,
        max_tokens=3000,
        temperature=0.4,
    )
    if out.get("error"):
        raise RuntimeError(out["error"])
    return _parse_recipes(out.get("text", ""))


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


_SWAP_PROMPT = """You are a meal-swap nutritionist. The user logged a meal
that doesn't fit their remaining macros / allergens / life-stage perfectly.
Suggest 3 alternative meals that:
  1. fit their remaining macros today better than the logged meal,
  2. respect their allergens and life-stage rules absolutely,
  3. are realistic substitutes (same meal type, similar prep effort).

Return STRICT JSON:
{
  "swaps": [
    {
      "name":           "<replacement meal name>",
      "rank":           <1..3>,
      "calories":       <int>,
      "protein":        <int>,
      "carbs":          <int>,
      "fat":            <int>,
      "delta_calories": <int>,    // negative means saves calories vs original
      "delta_protein":  <int>,    // positive means adds protein
      "why":            "<1-2 sentence reason citing user numbers>",
      "modifications":  "<short string of substitutions if relevant, else empty>"
    }
  ],
  "warnings": ["<short note>"]
}

Hard rules:
- NEVER suggest a meal containing one of the listed allergens.
- For pregnancy: NEVER include raw fish / sushi / soft unpasteurized
  cheese / deli meats / alcohol / high-mercury fish.
- For GLP-1: protein-floor (>= 25g per meal); high fiber; no further
  restriction.
- Don't suggest the same meal back. Each swap must be different from
  the original.
- delta values are computed against the logged meal's macros.
- Keep names natural ('Greek yogurt + berries + walnuts' not 'high-protein dairy bowl').
"""


def suggest_meal_swaps(
    original_meal: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
    lang: Optional[str] = None,
) -> dict[str, Any]:
    """Given a meal the user logged + their context, return ranked swaps.

    `original_meal` shape:
        {"name": "...", "calories": int, "protein": int, "carbs": int, "fat": int}

    `context` shape: same as find_recipes (remaining macros + allergies +
    conditions + dietary + specialty + meal_type).

    Returns: {"swaps": [...], "warnings": [...]} with same field shape
    as the recipes parser plus delta_calories / delta_protein.
    """
    om = original_meal or {}
    ctx = context or {}
    ctx_lines = ["LOGGED MEAL (user wants alternatives):"]
    ctx_lines.append(
        f"  {om.get('name', 'Meal')} — "
        f"{om.get('calories', '?')} cal, {om.get('protein', '?')}p, "
        f"{om.get('carbs', '?')}c, {om.get('fat', '?')}f"
    )
    ctx_lines.append("\nUSER CONTEXT:")
    if (ctx.get("remaining_calories") is not None
            or ctx.get("remaining_protein") is not None):
        ctx_lines.append(
            f"  Remaining today (BEFORE this meal): "
            f"{ctx.get('remaining_calories', '?')} cal, "
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
        ctx_lines.append(f"  Meal type: {ctx['meal_type']}")
    for block in (ctx.get("specialty") or []):
        ctx_lines.append("  " + str(block).replace("\n", " "))
    ctx_block = "\n".join(ctx_lines)

    # Static swap-rules in `system` (cache-friendly), dynamic per-user
    # context stays outside.
    system_prompt = _SWAP_PROMPT
    if lang and lang != "en":
        system_prompt = f"Respond entirely in {lang}.\n\n" + system_prompt
    gemini_prompt = system_prompt + "\n\n" + ctx_block

    gem_key = (os.environ.get("GEMINI_KEY") or "").strip().strip('"').strip("'")
    claude_key = (
        (os.environ.get("CLAUDE_KEY") or "").strip().strip('"').strip("'")
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    )
    if not gem_key and not claude_key:
        raise RuntimeError("suggest_meal_swaps: no AI key configured")

    last_err: Optional[Exception] = None
    if gem_key:
        try:
            return _swaps_from_gemini(gemini_prompt, gem_key)
        except Exception as e:
            last_err = e
    if claude_key:
        try:
            return _swaps_from_claude(system_prompt, ctx_block, claude_key)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"suggest_meal_swaps failed: {last_err}")


def _swaps_from_gemini(prompt: str, api_key: str) -> dict[str, Any]:
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1500},
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
    return _parse_swaps(text)


def _swaps_from_claude(system_prompt: str, user_context: str,
                        api_key: str) -> dict[str, Any]:
    out = _claude_client(api_key).messages(
        system=system_prompt,
        user=user_context,
        max_tokens=1500,
        temperature=0.4,
    )
    if out.get("error"):
        raise RuntimeError(out["error"])
    return _parse_swaps(out.get("text", ""))


def _parse_swaps(text: str) -> dict[str, Any]:
    if not text:
        return {"swaps": [], "warnings": ["empty model response"]}
    cleaned = re.sub(r"^```json\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return {"swaps": [], "warnings": ["model output not parseable"]}
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            return {"swaps": [], "warnings": ["model output not parseable"]}

    swaps_in = parsed.get("swaps") or []
    swaps_out: list[dict[str, Any]] = []
    for i, sw in enumerate(swaps_in[:3]):
        if not isinstance(sw, dict):
            continue
        swaps_out.append({
            "name":           _str(sw.get("name"))[:160],
            "rank":           _int(sw.get("rank")) or (i + 1),
            "calories":       _int(sw.get("calories")),
            "protein":        _int(sw.get("protein")),
            "carbs":          _int(sw.get("carbs")),
            "fat":            _int(sw.get("fat")),
            "delta_calories": _int(sw.get("delta_calories")),
            "delta_protein":  _int(sw.get("delta_protein")),
            "why":            _str(sw.get("why"))[:300],
            "modifications":  _str(sw.get("modifications"))[:200],
        })
    swaps_out.sort(key=lambda x: x.get("rank") or 99)
    warnings = [_str(w)[:300] for w in (parsed.get("warnings") or []) if _str(w)][:5]
    return {"swaps": swaps_out, "warnings": warnings}


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
