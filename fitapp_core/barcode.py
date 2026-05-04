"""Barcode lookup chain.

Open Food Facts (global, crowdsourced) → USDA Branded Foods (US, free,
manufacturer-direct) → not found. Includes:
    - GTIN-8/12/13/14 checksum validation
    - Junk-OFF-entry filter (rejects entries with date-as-name or
      impossible per-100g nutrients so the chain falls through to USDA)
    - Per-serving math reconciling OFF's serving_quantity with per-100g
      values, with USDA's separate per-100g + per-serving fields

The chain reproduces FitApp's production logic and was extracted on
2026-05-04. FitApp itself was not modified.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

OFF_HOST = "world.openfoodfacts.org"
USDA_HOST = "api.nal.usda.gov"
USER_AGENT = "fitapp-core/0.1 (contact@elh.example)"
TIMEOUT_SEC = 10


# ────────────────────────────────────────────────────────────────────
#  GTIN checksum
# ────────────────────────────────────────────────────────────────────

def valid_gtin_checksum(digits: str) -> bool:
    """EAN-13 / UPC-A / EAN-8 / GTIN-14 checksum verification.

    Catches OCR misreads before we hit Open Food Facts with garbage digits.
    Algorithm: rightmost data digit gets weight 3, then alternating 1, 3,
    1, 3, ... going left. Sum mod 10, then 10 minus that mod 10 must equal
    the check digit.
    """
    if not isinstance(digits, str) or not digits.isdigit():
        return False
    if len(digits) not in (8, 12, 13, 14):
        return False
    body = digits[:-1]
    check = int(digits[-1])
    total = 0
    for i, ch in enumerate(reversed(body)):
        weight = 3 if i % 2 == 0 else 1
        total += int(ch) * weight
    return (10 - (total % 10)) % 10 == check


# ────────────────────────────────────────────────────────────────────
#  OFF entry quality filter
# ────────────────────────────────────────────────────────────────────

def off_entry_is_usable(p: dict[str, Any], n: dict[str, Any]) -> bool:
    """Filter out malformed crowdsourced OFF entries.

    Real example caught: barcode 049000050103 (Coca-Cola 2L US) had
    product_name = '11/30/25' (a date) and serving_quantity = 2000g (the
    entire bottle), producing 2800-cal-780g-sugar-per-serving output.
    Rejecting unusable OFF entries lets the chain fall through to USDA
    where the data is manufacturer-correct (138 cal, 39g sugar / 355ml).
    """
    name = (p.get("product_name") or p.get("product_name_en") or "").strip()
    if len(name) < 2:
        return False
    if re.match(r"^[\d/.\-: ]+$", name):
        return False
    sugar_100 = n.get("sugars_100g") or 0
    cal_100 = n.get("energy-kcal_100g") or 0
    try:
        if float(sugar_100) > 100:  # impossible (more sugar than mass)
            return False
        if float(cal_100) > 900:  # above pure-fat cap
            return False
    except (TypeError, ValueError):
        pass
    return True


# ────────────────────────────────────────────────────────────────────
#  Open Food Facts
# ────────────────────────────────────────────────────────────────────

_INGREDIENT_LANG_FALLBACK = (
    "ingredients_text",
    "ingredients_text_en",
    "ingredients_text_fr",
    "ingredients_text_es",
    "ingredients_text_de",
    "ingredients_text_pt",
    "ingredients_text_it",
    "ingredients_text_ja",
    "ingredients_text_zh",
    "ingredients_text_ar",
    "ingredients_text_ko",
    "ingredients_text_hi",
)


def _http_json(url: str, timeout: int = TIMEOUT_SEC) -> dict[str, Any] | None:
    """GET JSON with a polite user agent. Returns None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _coerce_nova(raw: Any) -> int | None:
    """OFF returns nova_group as int, string, or empty string. Normalize."""
    if raw in (None, ""):
        return None
    try:
        n = int(raw)
        return n if n in (1, 2, 3, 4) else None
    except (TypeError, ValueError):
        return None


def barcode(code: str) -> dict[str, Any]:
    """Open Food Facts lookup. Returns {found: True, ...} or {found: False}.

    Tries v2 (better international data) then v0. Filters out unusable
    crowdsourced entries so callers can chain through to USDA.
    """
    data = None
    for ver in ("v2", "v0"):
        data = _http_json(f"https://{OFF_HOST}/api/{ver}/product/{code}.json")
        if data and data.get("status") in ("success", 1, "success_with_warnings"):
            break
        data = None

    if not data or data.get("status") not in ("success", 1, "success_with_warnings"):
        return {"found": False}

    p = data.get("product", data)
    n = p.get("nutriments", {}) or {}
    if not off_entry_is_usable(p, n):
        return {"found": False}

    # Multi-language ingredient text
    ingredients_text = ""
    for k in _INGREDIENT_LANG_FALLBACK:
        v = p.get(k)
        if v:
            ingredients_text = v
            break

    name = (
        p.get("product_name")
        or p.get("product_name_en")
        or p.get("product_name_fr")
        or p.get("product_name_es")
        or p.get("product_name_de")
        or p.get("abbreviated_product_name")
        or ""
    )
    nutriscore = p.get("nutriscore_grade", "") or p.get("nutrition_grades", "")
    nova = _coerce_nova(p.get("nova_group") or p.get("nova_groups"))

    serving_qty: float | None = None
    sq = p.get("serving_quantity") or n.get("serving_quantity")
    if sq is not None:
        try:
            serving_qty = float(sq)
        except (TypeError, ValueError):
            serving_qty = None

    def per_serving(key_100g: str, key_serving: str) -> float:
        v_serv = n.get(key_serving)
        if v_serv not in (None, ""):
            try:
                return float(v_serv)
            except (TypeError, ValueError):
                pass
        v_100 = n.get(key_100g)
        if v_100 not in (None, "") and serving_qty:
            try:
                return float(v_100) * (serving_qty / 100.0)
            except (TypeError, ValueError):
                pass
        return 0.0

    cal_100 = n.get("energy-kcal_100g") or n.get("energy-kcal") or 0
    return {
        "found": True,
        "source": "open_food_facts",
        "code": code,
        "name": name,
        "brand": p.get("brands", "") or "",
        "serving": p.get("serving_size", "100g"),
        "serving_quantity_g": serving_qty,
        "image_url": p.get("image_front_small_url") or p.get("image_url") or "",
        "nutriscore": nutriscore,
        "nova_group": nova,
        "ecoscore": p.get("ecoscore_grade", "") or "",
        "categories": p.get("categories", "") or "",
        "origin": p.get("origins") or p.get("manufacturing_places") or "",
        "ingredients_text": ingredients_text,
        "n": {
            "calories": round(per_serving("energy-kcal_100g", "energy-kcal_serving") or float(cal_100)),
            "protein":  round(per_serving("proteins_100g", "proteins_serving") or float(n.get("proteins_100g", 0) or 0), 1),
            "carbs":    round(per_serving("carbohydrates_100g", "carbohydrates_serving") or float(n.get("carbohydrates_100g", 0) or 0), 1),
            "fat":      round(per_serving("fat_100g", "fat_serving") or float(n.get("fat_100g", 0) or 0), 1),
            "sugar":    round(per_serving("sugars_100g", "sugars_serving") or float(n.get("sugars_100g", 0) or 0), 1),
            "sodium":   round((per_serving("sodium_100g", "sodium_serving") * 1000) or (float(n.get("sodium_100g", 0) or 0) * 1000), 0),
            "fiber":    round(per_serving("fiber_100g", "fiber_serving") or float(n.get("fiber_100g", 0) or 0), 1),
            "saturated_fat": round(per_serving("saturated-fat_100g", "saturated-fat_serving") or float(n.get("saturated-fat_100g", 0) or 0), 1),
            "salt":     round(float(n.get("salt_100g", 0) or 0), 1),
        },
        "n_100g": {
            "calories": round(float(cal_100)),
            "protein":  round(float(n.get("proteins_100g", 0) or 0), 1),
            "carbs":    round(float(n.get("carbohydrates_100g", 0) or 0), 1),
            "fat":      round(float(n.get("fat_100g", 0) or 0), 1),
            "sugar":    round(float(n.get("sugars_100g", 0) or 0), 1),
            "sodium":   round(float(n.get("sodium_100g", 0) or 0) * 1000, 0),
            "fiber":    round(float(n.get("fiber_100g", 0) or 0), 1),
            "saturated_fat": round(float(n.get("saturated-fat_100g", 0) or 0), 1),
        },
    }


# ────────────────────────────────────────────────────────────────────
#  USDA FoodData Central — Branded Foods
# ────────────────────────────────────────────────────────────────────

_USDA_UNIT_LABEL = {
    "grm": "g", "g": "g", "gram": "g",
    "mlt": "ml", "ml": "ml", "milliliter": "ml",
    "flz": "fl oz", "fl oz": "fl oz", "floz": "fl oz",
    "oz": "oz", "ounce": "oz",
}


def usda_barcode(code: str, api_key: str | None = None) -> dict[str, Any]:
    """USDA Branded Foods lookup by GTIN. Free, US-focused.

    USDA stores GTINs as zero-padded 14-digit strings (UPC-A 016000275287
    becomes "00016000275287"). Their search engine does literal string
    match on gtinUpc, so we pad the query the same way.
    """
    api_key = api_key or os.environ.get("USDA_API_KEY") or "DEMO_KEY"
    target = code.lstrip("0")
    gtin14 = code.zfill(14)
    match = None
    for query in (gtin14, code):
        url = (
            f"https://{USDA_HOST}/fdc/v1/foods/search"
            f"?api_key={api_key}&query={urllib.parse.quote(query)}"
            f"&dataType=Branded&pageSize=10"
        )
        data = _http_json(url)
        if not data:
            continue
        for f in data.get("foods", []):
            gtin = (f.get("gtinUpc") or "").strip().lstrip("0")
            if gtin and gtin == target:
                match = f
                break
        if match:
            break

    if not match:
        return {"found": False}

    n100 = {
        "calories": 0, "protein": 0, "carbs": 0, "fat": 0,
        "sugar": 0, "sodium": 0, "fiber": 0, "saturated_fat": 0,
    }
    seen: set[str] = set()

    def _set(key: str, val: float | int) -> None:
        if key not in seen:
            n100[key] = val
            seen.add(key)

    # USDA returns each nutrient multiple times in one response (per-100g
    # plus per-label-serving plus duplicates). First occurrence is per-100g.
    for nu in match.get("foodNutrients", []) or []:
        nm = (nu.get("nutrientName", "") or "").lower()
        val = nu.get("value", 0) or 0
        unit = (nu.get("unitName", "") or "").lower()
        if "energy" in nm and "kcal" in unit:
            _set("calories", round(val))
        elif "protein" in nm:
            _set("protein", round(val, 1))
        elif "carbohydrate" in nm and "by difference" in nm:
            _set("carbs", round(val, 1))
        elif "total lipid" in nm or nm == "total fat":
            _set("fat", round(val, 1))
        elif "sugars" in nm and "total" in nm:
            _set("sugar", round(val, 1))
        elif nm in ("sodium, na", "sodium"):
            _set("sodium", round(val, 0))
        elif "fiber" in nm and "total" in nm:
            _set("fiber", round(val, 1))
        elif "saturated" in nm and "fatty" in nm:
            _set("saturated_fat", round(val, 1))

    serving_qty = match.get("servingSize") or 0
    serving_unit = (match.get("servingSizeUnit") or "g").lower()
    try:
        qty_g = float(serving_qty)
        if serving_unit in ("ml", "milliliter"):
            pass  # treat ml ≈ g for water-density beverages
        elif serving_unit in ("fl oz", "fluid oz", "floz"):
            qty_g = qty_g * 29.5735
        elif serving_unit in ("oz", "ounce"):
            qty_g = qty_g * 28.3495
    except (TypeError, ValueError):
        qty_g = 0

    scale = (qty_g / 100.0) if qty_g else 1.0
    n_serving = {
        k: (round(v * scale, 1) if isinstance(v, float) else round(v * scale))
        for k, v in n100.items()
    } if qty_g else dict(n100)

    unit_label = _USDA_UNIT_LABEL.get(serving_unit, serving_unit)

    return {
        "found": True,
        "source": "usda_branded",
        "code": code,
        "name": match.get("description", "") or "Unknown product",
        "brand": match.get("brandName") or match.get("brandOwner") or "",
        "serving": f"{int(serving_qty)}{unit_label}" if serving_qty else "per 100g",
        "serving_quantity_g": qty_g if qty_g else None,
        "image_url": "",
        "nutriscore": "",
        "nova_group": None,
        "ecoscore": "",
        "categories": match.get("foodCategory", "") or "",
        "origin": match.get("marketCountry", "") or "",
        "ingredients_text": match.get("ingredients", "") or "",
        "n": n_serving,
        "n_100g": n100,
    }


# ────────────────────────────────────────────────────────────────────
#  Chain
# ────────────────────────────────────────────────────────────────────

def barcode_with_fallback(code: str, usda_api_key: str | None = None) -> dict[str, Any]:
    """OFF → USDA chain. First hit wins, source-tagged. Both DBs are free.

    OFF is global; USDA fills the US gap on store-brand and regional CPG
    that haven't been crowdsourced into OFF yet.
    """
    if not isinstance(code, str) or not code:
        return {"found": False, "code": code, "reason": "empty code"}
    off = barcode(code)
    if off.get("found"):
        return off
    usda = usda_barcode(code, api_key=usda_api_key)
    if usda.get("found"):
        return usda
    return {"found": False, "code": code, "reason": "Not in Open Food Facts or USDA Branded."}
