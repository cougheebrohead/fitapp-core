"""Restaurant menu picker — vision-model "what should I order?".

Single public entrypoint:

    pick_menu_items(image_bytes, media_type, context, lang=None) -> dict

Sends a photo of a restaurant menu to a vision LLM together with the
user's nutritional context (today's remaining macros + allergens +
conditions + dietary preferences) and gets back a ranked list of
suggested items with per-item macros and reasoning.

Result shape:

    {
      "venue":          "Chipotle" | None,
      "picks": [
        {
          "name":       "Chicken bowl, brown rice, fajita veggies, salsa, guac, lettuce",
          "rank":       1,                    # 1..N, lower = better fit
          "calories":   720,
          "protein":    52,
          "carbs":      62,
          "fat":        28,
          "fits_macros":true,
          "allergen_flags": [],               # ["dairy", "gluten", ...]
          "why":        "Hits remaining 480 cal / 38g protein with room to spare.
                         No dairy or wheat (your allergens). Skip cheese to keep it.",
          "modifications": "no cheese, light rice"
        },
        ...
      ],
      "avoid": [
        {"name": "Burrito with rice + beans + cheese + sour cream",
         "why":  "Pushes you 200+ cal over and contains dairy."}
      ],
      "raw_text": "...optional model commentary...",
      "warnings": ["unit ambiguity..."]
    }

Stdlib-only: urllib + json + re + base64 (same shape as scan_lab).
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Optional


_MAX_RESPONSE_BYTES = 200_000


_PROMPT_HEAD = """You are a menu nutritionist. The image is a photo of a
restaurant menu. The user's nutritional context appears below.

Task: rank the 3-5 BEST items on this menu for THIS user RIGHT NOW based
on the macros they still have left today and their allergen / condition /
dietary constraints. Return STRICT JSON only — no markdown fences.

Output shape:
{
  "venue": "<chain name if recognizable, else null>",
  "picks": [
    {
      "name":           "<item name + key modifications, full sentence>",
      "rank":           <1..N, 1 is best fit>,
      "calories":       <integer kcal estimate>,
      "protein":        <grams>,
      "carbs":          <grams>,
      "fat":            <grams>,
      "fits_macros":    <true if fits user's remaining macros today, else false>,
      "allergen_flags": [<"dairy"|"gluten"|"egg"|"soy"|"peanut"|"tree_nut"|"shellfish"|"fish"|"sesame"|"other"|...>],
      "why":            "<1-2 sentences why this fits, citing the user's numbers>",
      "modifications":  "<short string of mods if needed, e.g. 'no cheese, dressing on side', else empty>"
    }
  ],
  "avoid": [
    {"name": "<item to avoid>", "why": "<1 sentence reason>"}
  ],
  "warnings": ["<short notes about ambiguous portions or missing menu info>"]
}

Hard rules:
- Use kcal estimates grounded in the visible menu (portion sizes if shown,
  standard restaurant servings otherwise). Don't invent precise numbers
  if the menu is vague — round and add a warning.
- Hide allergen items the user listed. NEVER recommend something
  containing one of their allergens.
- For users with diabetes / GLP-1 / cardio conditions: prioritize protein
  density, lean cuts, low-glycemic sides. For users with kidney conditions:
  watch sodium + protein.
- "fits_macros": true if the calorie estimate is <= remaining and protein
  is >= 0.4 * remaining_protein. False otherwise (recommend anyway if
  it's the best lower-cal option, but flag).
- If the image isn't a menu, return:
  {"venue": null, "picks": [], "avoid": [], "warnings": ["not a menu"]}.
"""


def pick_menu_items(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    context: Optional[dict[str, Any]] = None,
    lang: Optional[str] = None,
) -> dict[str, Any]:
    """Photo + context -> ranked picks dict.

    Args:
      image_bytes: raw bytes (or already-base64'd str)
      media_type:  IANA media type for the image
      context:     {
        "remaining_calories": int, "remaining_protein": int,
        "remaining_carbs": int, "remaining_fat": int,
        "allergies":   "comma string",
        "conditions":  "comma string",
        "dietary":     "comma string",
        "venue_hint":  "Chipotle"   (optional)
      }
      lang:        ISO code; influences `why` text language

    Returns: dict per module docstring.
    """
    if isinstance(image_bytes, str):
        b64 = image_bytes
    elif isinstance(image_bytes, (bytes, bytearray)):
        b64 = base64.b64encode(bytes(image_bytes)).decode()
    else:
        raise TypeError(
            f"image_bytes must be bytes or base64 str, got {type(image_bytes).__name__}"
        )

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
        ctx_lines.append(f"  Allergies (NEVER recommend foods containing these): {ctx['allergies']}")
    if ctx.get("conditions"):
        ctx_lines.append(f"  Conditions: {ctx['conditions']}")
    if ctx.get("dietary"):
        ctx_lines.append(f"  Dietary preferences: {ctx['dietary']}")
    if ctx.get("venue_hint"):
        ctx_lines.append(f"  Venue (hint): {ctx['venue_hint']}")
    ctx_block = "\n".join(ctx_lines)

    # Static instructions go to `system` (cache-friendly). Dynamic user
    # context stays in `ctx_block` and is sent alongside the image.
    system_prompt = _PROMPT_HEAD
    if lang and lang != "en":
        system_prompt = f"Respond entirely in {lang}.\n\n" + system_prompt
    gemini_prompt = system_prompt + "\n\n" + ctx_block  # Gemini has no cache_control

    gem_key = (os.environ.get("GEMINI_KEY") or "").strip().strip('"').strip("'")
    claude_key = (
        (os.environ.get("CLAUDE_KEY") or "").strip().strip('"').strip("'")
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    )
    if not gem_key and not claude_key:
        raise RuntimeError("pick_menu_items: no AI key configured")

    last_err: Optional[Exception] = None
    if gem_key:
        try:
            return _gemini_pick(b64, media_type, gem_key, gemini_prompt)
        except Exception as e:
            last_err = e
    if claude_key:
        try:
            return _claude_pick(b64, media_type, claude_key,
                                 system_prompt, ctx_block)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"pick_menu_items failed: {last_err}")


# ─── providers ────────────────────────────────────────────────────────


def _gemini_pick(b64: str, media_type: str, api_key: str, prompt: str) -> dict[str, Any]:
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": media_type, "data": b64}},
            ],
        }],
        "generationConfig": {"temperature": 0.15, "maxOutputTokens": 2400},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent?key=" + urllib.parse.quote(api_key)
    )
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
    return _parse_picks(text)


def _claude_pick(b64: str, media_type: str, api_key: str,
                  system_prompt: str, ctx_block: str) -> dict[str, Any]:
    # Static menu-picker instructions in `system` (cache-friendly).
    # Image + per-request user context stay in the user message.
    # use_result_cache=False — every menu image is unique, so cache
    # hit rate is ~0 and hashing big base64 blobs is wasted CPU.
    from fitapp_core.ai import ClaudeClient, ClaudeConfig
    client = ClaudeClient(ClaudeConfig(
        api_key=api_key, default_model="claude-sonnet-4-6", timeout=45,
    ))
    out = client.messages(
        system=system_prompt,
        user=[
            {"type": "image", "source": {"type": "base64",
                                          "media_type": media_type,
                                          "data": b64}},
            {"type": "text", "text": ctx_block},
        ],
        max_tokens=2400,
        temperature=0.1,
        use_result_cache=False,
    )
    if out.get("error"):
        raise RuntimeError(out["error"])
    return _parse_picks(out.get("text", ""))


# ─── parsing ──────────────────────────────────────────────────────────


def _parse_picks(text: str) -> dict[str, Any]:
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

    picks_in = parsed.get("picks") or []
    picks_out: list[dict[str, Any]] = []
    for i, p in enumerate(picks_in[:8]):
        if not isinstance(p, dict):
            continue
        picks_out.append({
            "name":           _str(p.get("name"))[:240],
            "rank":           _int(p.get("rank")) or (i + 1),
            "calories":       _int(p.get("calories")),
            "protein":        _int(p.get("protein")),
            "carbs":          _int(p.get("carbs")),
            "fat":            _int(p.get("fat")),
            "fits_macros":    bool(p.get("fits_macros")),
            "allergen_flags": [_str(x).lower()[:40] for x in (p.get("allergen_flags") or []) if _str(x)],
            "why":            _str(p.get("why"))[:400],
            "modifications":  _str(p.get("modifications"))[:200],
        })
    picks_out.sort(key=lambda x: x.get("rank") or 99)

    avoid_in = parsed.get("avoid") or []
    avoid_out: list[dict[str, Any]] = []
    for a in avoid_in[:6]:
        if not isinstance(a, dict):
            continue
        avoid_out.append({
            "name": _str(a.get("name"))[:240],
            "why":  _str(a.get("why"))[:300],
        })

    return {
        "venue":    _str(parsed.get("venue")) or None,
        "picks":    picks_out,
        "avoid":    avoid_out,
        "raw_text": "",
        "warnings": [_str(w)[:300] for w in (parsed.get("warnings") or []) if _str(w)][:10],
    }


def _empty(warnings: list[str]) -> dict[str, Any]:
    return {"venue": None, "picks": [], "avoid": [], "raw_text": "", "warnings": warnings}


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
