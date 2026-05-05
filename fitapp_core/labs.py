"""Lab report OCR + biomarker extraction.

Single public entrypoint:

    scan_lab(image_bytes, media_type, lang=None) -> dict

Sends the image to a vision LLM with a tightly-scoped prompt. Returns:

    {
        "panel_name":   "Lipid Panel" | "Comprehensive Metabolic Panel" | "HbA1c" | …,
        "drawn_at":     "2026-04-12" | None,            # ISO date if visible
        "provider":     "Quest Diagnostics" | None,     # lab company if visible
        "biomarkers": {
            "hba1c":            {"value": 5.4, "unit": "%",     "ref_low": None, "ref_high": 5.7,  "flag": "in_range"},
            "fasting_glucose":  {"value": 92,  "unit": "mg/dL", "ref_low": 70,   "ref_high": 100,  "flag": "in_range"},
            "total_cholesterol":{"value": 178, "unit": "mg/dL", "ref_low": None, "ref_high": 200,  "flag": "in_range"},
            "ldl":              {"value": 110, "unit": "mg/dL", "ref_low": None, "ref_high": 100,  "flag": "high"},
            "hdl":              {"value": 55,  "unit": "mg/dL", "ref_low": 40,   "ref_high": None, "flag": "in_range"},
            "triglycerides":    {"value": 88,  "unit": "mg/dL", "ref_low": None, "ref_high": 150,  "flag": "in_range"},
            "tsh":              {"value": 1.8, "unit": "mIU/L", "ref_low": 0.4,  "ref_high": 4.5,  "flag": "in_range"},
            "vitamin_d":        {"value": 32,  "unit": "ng/mL", "ref_low": 30,   "ref_high": 100,  "flag": "in_range"},
            "ferritin":         {"value": 65,  "unit": "ng/mL", "ref_low": 13,   "ref_high": 150,  "flag": "in_range"},
            "crp":              {"value": 0.8, "unit": "mg/L",  "ref_low": None, "ref_high": 3.0,  "flag": "in_range"},
            …
        },
        "raw_text":     "…optional OCR'd text the model surfaced…",
        "warnings":     ["unit ambiguity on hba1c"],
    }

Design choices:
- Tries Gemini Flash (free tier) first, falls back to Claude Vision.
- Caller passes API keys via env (GEMINI_KEY / CLAUDE_KEY / ANTHROPIC_API_KEY).
- All numeric values returned as JSON numbers; unit/flag are strings.
- Biomarker keys are normalized snake_case English (hba1c, ldl, vitamin_d).
- We do NOT auto-write to a DB — the caller decides whether to save.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Optional


_PROMPT = """You are a medical lab report parser. Extract structured biomarker
values from this image of a lab report. Return STRICT JSON only — no markdown
fences, no commentary.

Output shape (any field may be null if unreadable):
{
  "panel_name": "Lipid Panel" or whatever the report calls itself,
  "drawn_at": "YYYY-MM-DD" if a collection or specimen date is visible,
  "provider": "Quest", "LabCorp", "Cleveland HeartLab", etc. — null if unclear,
  "biomarkers": {
    "<snake_case_key>": {
      "value": <number>,
      "unit": "<unit string e.g. mg/dL>",
      "ref_low": <number or null>,
      "ref_high": <number or null>,
      "flag": "low" | "in_range" | "high"
    },
    ...
  },
  "warnings": ["<short note about anything ambiguous>"]
}

Recognized biomarker keys (use these exact keys; add more snake_case ones if
you see additional values):
  hba1c, fasting_glucose, glucose, insulin, homa_ir,
  total_cholesterol, ldl, hdl, vldl, triglycerides, lipoprotein_a,
  apo_b, apo_a1, non_hdl_cholesterol,
  tsh, free_t3, free_t4, total_t3, total_t4, reverse_t3, tpo_antibodies,
  vitamin_d, vitamin_b12, folate, iron, ferritin, transferrin, tibc,
  hemoglobin, hematocrit, rbc, wbc, platelets, mcv, mch, mchc, rdw,
  alt, ast, ggt, alkaline_phosphatase, total_bilirubin, albumin,
  bun, creatinine, egfr, sodium, potassium, chloride, co2, calcium,
  crp, hs_crp, esr, homocysteine, uric_acid, cortisol,
  testosterone_total, testosterone_free, shbg, estradiol, progesterone,
  dhea_s, igf_1, prolactin, fsh, lh, psa, hcg

Rules:
- If you see a value but no reference range printed, return ref_low=null
  and ref_high=null. Do NOT invent ranges.
- "flag" must come from the report's own H/L/N markings or from the
  printed range; if no range is visible AND no marking, return "in_range".
- Use the unit printed on the report. Common: mg/dL, mmol/L, %, mIU/L,
  ng/mL, ug/dL, IU/L, U/L, K/uL, M/uL, fL, pg, g/dL, mg/L.
- For HbA1c, the unit is usually "%" (NGSP) or "mmol/mol" (IFCC). Keep
  whichever is shown. Don't convert.
- If the same biomarker appears twice with different units, pick the
  primary value and add a warning describing the duplication.
- If the image is not a lab report, return:
  {"panel_name": null, "drawn_at": null, "provider": null,
   "biomarkers": {}, "warnings": ["not a lab report"]}
"""


# Hard cap on response size (bytes). Lab reports rarely exceed 50 KB of
# extracted text; we cap to keep parsing predictable.
_MAX_RESPONSE_BYTES = 200_000


def scan_lab(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    lang: Optional[str] = None,
) -> dict[str, Any]:
    """Send image to a vision LLM, return parsed biomarker dict.

    Args:
      image_bytes: raw image bytes (PNG, JPEG, HEIC, WebP, GIF supported).
      media_type:  IANA media type ("image/jpeg" / "image/png" / etc).
      lang:        optional ISO language hint passed to the model.

    Returns: dict shape documented in module docstring.

    Raises:
      RuntimeError: if no AI key configured or both providers failed.
    """
    if isinstance(image_bytes, str):
        # Allow callers to pass an already-base64'd string.
        b64 = image_bytes
    elif isinstance(image_bytes, (bytes, bytearray)):
        b64 = base64.b64encode(bytes(image_bytes)).decode()
    else:
        raise TypeError(
            f"image_bytes must be bytes or base64 str, got {type(image_bytes).__name__}"
        )

    gem_key = os.environ.get("GEMINI_KEY", "").strip()
    claude_key = (
        os.environ.get("CLAUDE_KEY", "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )
    if not gem_key and not claude_key:
        raise RuntimeError(
            "scan_lab: no AI key configured (set GEMINI_KEY or CLAUDE_KEY)"
        )

    last_err: Optional[Exception] = None
    if gem_key:
        try:
            return _gemini_scan(b64, media_type, gem_key, lang=lang)
        except Exception as e:
            last_err = e

    if claude_key:
        try:
            return _claude_scan(b64, media_type, claude_key, lang=lang)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"scan_lab failed: {last_err}")


# ─── providers ────────────────────────────────────────────────────────


def _gemini_scan(b64: str, media_type: str, api_key: str,
                 lang: Optional[str] = None) -> dict[str, Any]:
    text_prompt = _PROMPT
    if lang:
        text_prompt = f"Output language hint: {lang}.\n\n" + text_prompt
    body = {
        "contents": [{
            "parts": [
                {"text": text_prompt},
                {"inline_data": {"mime_type": media_type, "data": b64}},
            ],
        }],
        "generationConfig": {
            "temperature": 0.05,
            "maxOutputTokens": 4096,
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent?key=" + urllib.parse.quote(api_key)
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read(_MAX_RESPONSE_BYTES + 1)
    result = json.loads(raw)
    text = (
        result.get("candidates", [{}])[0]
              .get("content", {}).get("parts", [{}])[0].get("text", "")
    )
    return _parse_lab_json(text)


def _claude_scan(b64: str, media_type: str, api_key: str,
                 lang: Optional[str] = None) -> dict[str, Any]:
    text_prompt = _PROMPT
    if lang:
        text_prompt = f"Output language hint: {lang}.\n\n" + text_prompt
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": media_type,
                                             "data": b64}},
                {"type": "text", "text": text_prompt},
            ],
        }],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read(_MAX_RESPONSE_BYTES + 1)
    result = json.loads(raw)
    text = (result.get("content", [{}])[0] or {}).get("text", "")
    return _parse_lab_json(text)


# ─── parsing helpers ──────────────────────────────────────────────────


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ─── biomarker direction map ─────────────────────────────────────────
#
# Per-biomarker clinical direction. "up_good" = rising values are
# health-positive, "up_bad" = rising values are concerning, "neutral"
# = context-dependent / no strong default. Used by trend-delta UIs to
# color ↑/↓ correctly (e.g. HDL up = green, LDL up = red, sodium up =
# dim/neutral). Keys are in normalized form (see _normalize_key).

_UP_GOOD = {
    "hdl_cholesterol", "hdl",
    "vitamin_d", "vit_d",
    "ferritin", "iron", "iron_saturation", "transferrin_saturation",
    "t3", "free_t3", "total_t3",
    "t4", "free_t4", "total_t4",
    "total_testosterone", "testosterone_total",
    "free_testosterone", "testosterone_free",
    "b12", "vitamin_b12",
    "magnesium",
    "albumin",
    "gfr", "egfr",
}

_UP_BAD = {
    "hba1c", "a1c",
    "fasting_glucose", "glucose",
    "ldl_cholesterol", "ldl",
    "total_cholesterol",
    "triglycerides",
    "crp", "hs_crp",
    "alt", "ast", "ggt", "alkaline_phosphatase",
    "creatinine", "bun", "uric_acid", "homocysteine",
    "insulin", "fasting_insulin",
    "fibrinogen", "lp_a", "lipoprotein_a", "apob", "apo_b",
    "vldl", "esr",
    "blood_pressure_systolic", "blood_pressure_diastolic", "sbp", "dbp",
}

_NEUTRAL = {
    "tsh",
    "sodium", "potassium", "chloride", "co2",
    "calcium", "phosphorus",
    "hemoglobin", "hematocrit",
    "mcv", "mch", "mchc", "rdw",
    "platelets", "wbc", "rbc",
    "neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils",
}

BIOMARKER_DIRECTION: dict[str, str] = {
    **{k: "up_good" for k in _UP_GOOD},
    **{k: "up_bad"  for k in _UP_BAD},
    **{k: "neutral" for k in _NEUTRAL},
}


def biomarker_direction(key: str) -> str:
    """Return 'up_good' | 'up_bad' | 'neutral' for a biomarker key.

    Unknown keys fall back to 'up_bad' — that's the safe default for
    trend-color UIs because the most common biomarkers people watch
    (glucose, lipids, inflammation, liver/kidney) are all up_bad. A
    false up_bad on an unknown marker over-flags rather than under-flags,
    which is the correct failure mode for clinical trend cues.
    """
    norm = _normalize_key(key) or ""
    return BIOMARKER_DIRECTION.get(norm, "up_bad")


def _parse_lab_json(text: str) -> dict[str, Any]:
    """Strip markdown fences + parse JSON. Sanitize biomarker shape."""
    if not text:
        return _empty_result(["empty model response"])
    cleaned = re.sub(r"^```json\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return _empty_result(["model output not parseable"])
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            return _empty_result(["model output not parseable"])

    biomarkers_in = parsed.get("biomarkers") or {}
    biomarkers_out: dict[str, dict[str, Any]] = {}
    for key, entry in list(biomarkers_in.items())[:120]:  # sanity cap
        if not isinstance(entry, dict):
            continue
        norm_key = _normalize_key(key)
        if not norm_key:
            continue
        biomarkers_out[norm_key] = _sanitize_biomarker(entry, norm_key)

    return {
        "panel_name": _str_or_none(parsed.get("panel_name")),
        "drawn_at":   _date_or_none(parsed.get("drawn_at")),
        "provider":   _str_or_none(parsed.get("provider")),
        "biomarkers": biomarkers_out,
        "warnings":   _strs_or_empty(parsed.get("warnings")),
    }


def _empty_result(warnings: list[str]) -> dict[str, Any]:
    return {
        "panel_name": None,
        "drawn_at":   None,
        "provider":   None,
        "biomarkers": {},
        "warnings":   warnings,
    }


def _str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _date_or_none(v: Any) -> Optional[str]:
    """Accept YYYY-MM-DD only; reject anything else (don't guess)."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    return None


def _strs_or_empty(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        s = _str_or_none(x)
        if s:
            out.append(s[:300])
    return out[:20]


def _normalize_key(k: str) -> Optional[str]:
    if not isinstance(k, str):
        return None
    norm = re.sub(r"[^a-z0-9_]+", "_", k.strip().lower()).strip("_")
    return norm[:60] if norm else None


def _to_number(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUMERIC_RE.search(v)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _sanitize_biomarker(entry: dict[str, Any], key: str = "") -> dict[str, Any]:
    flag = entry.get("flag")
    if flag not in ("low", "in_range", "high"):
        flag = "in_range"
    return {
        "value":     _to_number(entry.get("value")),
        "unit":      _str_or_none(entry.get("unit")),
        "ref_low":   _to_number(entry.get("ref_low")),
        "ref_high":  _to_number(entry.get("ref_high")),
        "flag":      flag,
        "direction": biomarker_direction(key),
    }
