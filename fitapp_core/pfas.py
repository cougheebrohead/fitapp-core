"""PFAS / forever-chemicals scanner — shared engine.

Stdlib-only. No AI dependency in this module. Two surfaces:

    pfas_category_lookup(*texts) -> dict | None
        Substring-match free text (product name, subcategory, materials)
        against PFAS_CATEGORY_KNOWLEDGE. Returns the matching entry +
        matched key, or None. Catches everyday products that don't get
        flagged by keyword chemical scans because their PFAS risk lives
        in the category (toothpicks, paper plates, dental floss, etc.).

    analyze_pfas_in_text(text) -> dict
        Keyword scan for explicit PFAS-family chemicals in an ingredient
        / materials string. Independent of the broader toxics analyzer
        in consumer apps. Returns flags + severity counts.

    barcode_universal(code) -> dict
        OFF (food) → OBF (cosmetics) → OPF (general products) chain.
        Each consumer can compose this with their own AI scan + the
        category lookup + analyzer above to deliver the full scanner UX.

Constants surfaced for direct use by consumers building responses:
    PFAS_CATEGORY_KNOWLEDGE      — 27 product categories with risk + alt
    PFAS_GENERIC_GUIDANCE        — generic label-check field guide
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

USER_AGENT = "fitapp-core/0.1"
TIMEOUT_SEC = 10


# ────────────────────────────────────────────────────────────────
# PFAS category knowledge — fallback when chemical names aren't on
# the label but the product *category* has well-known PFAS history.
# Keys are lowercase substrings; first matching key wins.
# ────────────────────────────────────────────────────────────────
PFAS_CATEGORY_KNOWLEDGE: dict[str, dict[str, str]] = {
    # Food-contact paper / molded fiber
    "toothpick": {
        "risk": "medium",
        "note": "Many wooden toothpicks are surface-treated with grease- or moisture-resistant coatings that historically included PFAS. Look for explicit 'PFAS-free' or '100% uncoated wood' on the box. Mint-flavored picks are higher risk; plain birch is usually lower.",
        "alt": "Plain birch wood toothpicks from a brand that states 'no coatings' or 'PFAS-free' on the packaging.",
    },
    "paper plate": {
        "risk": "high",
        "note": "Most molded-fiber and bleached paper plates use grease-proofing chemicals; PFAS were standard in this category until ~2024 and many brands still use replacement fluorochemicals. Look for 'PFAS-free' on the box.",
        "alt": "Bare bamboo, palm-leaf, or uncoated paper plates labeled 'PFAS-free' / 'no grease-proofing chemicals'.",
    },
    "paper bowl": {
        "risk": "high",
        "note": "Molded paper bowls almost universally have grease-resistance coatings. Look for explicit 'PFAS-free' labeling.",
        "alt": "Glass or ceramic bowls, or paper bowls explicitly labeled 'PFAS-free'.",
    },
    "takeout container": {
        "risk": "high",
        "note": "Molded-fiber and 'compostable' takeout containers were a major historical PFAS source. Many vendors switched to PFAS-free fiber post-2023 but you can't tell without the label. If unmarked, assume PFAS-treated.",
        "alt": "Glass, stainless, or fiber containers explicitly labeled 'PFAS-free' / 'no fluorochemicals'.",
    },
    "molded fiber": {
        "risk": "high",
        "note": "Molded fiber food packaging is the highest-risk paper category for PFAS. Unless labeled PFAS-free, assume treated.",
        "alt": "Glass, stainless, or bare-fiber containers with PFAS-free certification.",
    },
    "popcorn bag": {
        "risk": "high",
        "note": "Microwave popcorn bags were the textbook PFAS food-contact case. Major brands reformulated 2020-2023 but heat + grease still pull residual fluorochemicals into the popcorn. Stovetop kernels avoid the entire risk.",
        "alt": "Loose popcorn kernels popped on the stovetop or in an air popper.",
    },
    "fast food wrapper": {
        "risk": "high",
        "note": "Greaseproof fast-food and bakery wrappers historically used PFAS as the cheapest grease barrier. Many chains reformulated, but residual fluorochemicals are still detectable in much of the supply.",
        "alt": "Order food without wrappers, or transfer to your own container before eating.",
    },
    "pizza box": {
        "risk": "medium",
        "note": "Many pizza boxes — especially the bottom liner — have been PFAS-treated to resist grease soak-through. Risk drops sharply once the pizza is removed; reheat on a plate, not in the box.",
        "alt": "Transfer leftover slices to a glass or ceramic plate before reheating.",
    },
    "parchment paper": {
        "risk": "medium",
        "note": "Most parchment paper is silicone-coated (low risk), but some 'non-stick' parchment uses fluorochemicals. Look for 'unbleached', 'silicone-coated', or 'PFAS-free' on the box.",
        "alt": "Silicone baking mats or PFAS-free unbleached parchment.",
    },
    "non-stick foil": {
        "risk": "high",
        "note": "'Non-stick' aluminum foil is typically PTFE-coated — that's a fluoropolymer and a direct PFAS source under heat.",
        "alt": "Plain aluminum foil (no non-stick coating), or parchment paper.",
    },
    # Cookware
    "non-stick pan": {
        "risk": "high",
        "note": "PTFE / Teflon non-stick coatings are fluoropolymers — they shed PFAS when overheated, scratched, or aging. 'PFOA-free' does NOT mean PFAS-free; PTFE itself is PFAS.",
        "alt": "Cast iron, carbon steel, or stainless steel skillets.",
    },
    "non-stick frying": {
        "risk": "high",
        "note": "Non-stick cookware coatings are fluoropolymers (PFAS family). Heat and wear release breakdown products.",
        "alt": "Cast iron, carbon steel, or stainless steel.",
    },
    "air fryer basket": {
        "risk": "medium",
        "note": "Most air fryer baskets are PTFE-coated. The coating breaks down faster at air-fryer temps than at stovetop temps. Replace if scratched.",
        "alt": "Stainless-steel-basket air fryers (Breville, Cuisinart non-stick-free models) or oven roasting.",
    },
    # Personal care
    "dental floss": {
        "risk": "high",
        "note": "Many 'glide'-style flosses (notably Glide and similar PTFE flosses) are made from fluoropolymer fibers — a direct PFAS source held against the gum line. Studies have measured higher PFAS levels in regular users.",
        "alt": "Nylon, silk, or biodegradable flosses labeled 'PFAS-free' or 'no PTFE'.",
    },
    "floss pick": {
        "risk": "medium",
        "note": "Plastic floss picks vary — some use PTFE thread (PFAS), others nylon. Hard to tell without the label.",
        "alt": "Floss picks explicitly labeled PFAS-free, or traditional nylon/silk floss.",
    },
    "mascara": {
        "risk": "medium",
        "note": "Long-wear, waterproof, and 'smudge-proof' mascaras commonly include fluorinated compounds for water resistance. Check ingredients for anything starting with 'perfluoro-', 'polyfluoro-', or 'PTFE'.",
        "alt": "Mascaras certified PFAS-free or marketed as 'clean beauty'.",
    },
    "foundation": {
        "risk": "medium",
        "note": "Long-wear and waterproof foundations commonly contain fluorinated compounds. Scan ingredients for 'perfluoro-', 'polyfluoro-', or 'PTFE'.",
        "alt": "Foundations from brands certified PFAS-free (Sephora Clean+, Credo, etc.).",
    },
    "sunscreen": {
        "risk": "low",
        "note": "Most US sunscreens don't contain PFAS, but a small subset of waterproof / sport formulas use fluorinated UV filters or film-formers. Scan ingredients for any 'perfluoro-' / 'polyfluoro-' terms.",
        "alt": "Mineral (zinc/titanium) sunscreens from brands with PFAS-free certification.",
    },
    # Textiles
    "waterproof": {
        "risk": "high",
        "note": "Waterproof / water-repellent (DWR) coatings on outdoor gear were almost universally PFAS-based until the recent industry shift to C0 (non-fluorinated) DWRs. Brands like Patagonia, REI, and Vaude have phased out PFAS; most others haven't.",
        "alt": "Gear with 'PFC-free DWR' or 'C0 DWR' on the tag (Patagonia, REI Co-op, Páramo, etc.).",
    },
    "water repellent": {
        "risk": "high",
        "note": "DWR (durable water repellent) finishes are PFAS-based unless explicitly labeled 'PFC-free' or 'C0 DWR'.",
        "alt": "Look for 'PFC-free DWR' / 'C0 DWR' on the tag.",
    },
    "rain jacket": {
        "risk": "high",
        "note": "Rain jackets use DWR coatings that were almost universally PFAS-based until recently. Look for 'PFC-free DWR' or 'C0 DWR' on the tag.",
        "alt": "PFC-free DWR jackets (Patagonia, REI Co-op, Páramo, etc.) or rubber rain gear.",
    },
    "ski jacket": {
        "risk": "high",
        "note": "Ski / snow jackets use heavy DWR treatments that were PFAS-based by default. Phase-outs are recent and brand-specific.",
        "alt": "Ski outerwear from brands with confirmed PFC-free DWR programs.",
    },
    "raincoat": {
        "risk": "high",
        "note": "Standard rain gear DWRs are PFAS-based unless explicitly labeled 'PFC-free' or 'C0 DWR'.",
        "alt": "Rain gear labeled 'PFC-free DWR' or made of inherently waterproof rubber/PVC.",
    },
    "stain resistant": {
        "risk": "high",
        "note": "Stain-resistance treatments on carpets, furniture, and clothing are the original PFAS application (Scotchgard family). Untreated or 'no stain treatment' versions avoid the risk.",
        "alt": "Untreated natural-fiber textiles (wool, cotton, linen) without aftermarket stain treatment.",
    },
    "carpet": {
        "risk": "medium",
        "note": "Wall-to-wall carpets historically used heavy PFAS stain treatments. Many manufacturers reformulated post-2020 but older carpet is a persistent indoor PFAS source.",
        "alt": "Hard flooring or wool rugs without aftermarket stain treatment.",
    },
    # Household
    "scotchgard": {
        "risk": "high",
        "note": "Scotchgard and similar stain-repellent sprays are PFAS-based. Original 3M Scotchgard reformulated in 2003 but later versions still rely on shorter-chain fluorochemicals.",
        "alt": "Skip the spray treatment; spot-clean instead.",
    },
    "stain spray": {
        "risk": "high",
        "note": "Aftermarket stain-resistance sprays are a primary PFAS application. Most still use fluorochemicals despite reformulations.",
        "alt": "Skip aftermarket treatment, or choose products labeled 'fluorine-free' / 'PFC-free'.",
    },
}


PFAS_GENERIC_GUIDANCE: dict[str, Any] = {
    "title": "How to check the label yourself",
    "tips": [
        "Look for explicit \"PFAS-free\" / \"PFC-free\" / \"fluorine-free\" on the packaging — that's the only definitive answer.",
        "Scan ingredient and material lists for anything starting with \"perfluoro-\", \"polyfluoro-\", or containing \"fluoro\", \"PTFE\", or \"Teflon\".",
        "Be suspicious of: grease-resistant paper, non-stick coatings, waterproof/stain-resistant textiles, long-wear cosmetics, and \"compostable\" molded-fiber food containers.",
        "Plain, uncoated, single-material products (bare wood, glass, stainless steel, cast iron, untreated cotton/linen) are almost always PFAS-free.",
    ],
}


# ────────────────────────────────────────────────────────────────
# PFAS-family keyword analyzer.
# Severity follows EWG / European Chemicals Agency assessment:
#   high   — direct fluoropolymer fibers / coatings (PTFE, PFOA, PFOS)
#   medium — fluorinated treatments / specific PFAS-precursors
#   low    — broad fluoro- terms that may or may not be PFAS
# ────────────────────────────────────────────────────────────────
PFAS_KEYWORDS: dict[str, dict[str, str]] = {
    "ptfe": {"severity": "high", "reason": "Polytetrafluoroethylene — a fluoropolymer in the PFAS family. Used in non-stick cookware, dental floss, and water-repellent textiles."},
    "polytetrafluoroethylene": {"severity": "high", "reason": "PTFE — fluoropolymer in the PFAS family."},
    "teflon": {"severity": "high", "reason": "Brand name for PTFE, a fluoropolymer in the PFAS family."},
    "pfoa": {"severity": "high", "reason": "Perfluorooctanoic acid — a legacy PFAS chemical phased out by major US manufacturers but persistent in the environment."},
    "pfos": {"severity": "high", "reason": "Perfluorooctanesulfonic acid — restricted PFAS chemical with documented health effects."},
    "perfluoro": {"severity": "high", "reason": "Perfluorinated compound — part of the PFAS family by definition."},
    "polyfluoro": {"severity": "high", "reason": "Polyfluorinated compound — part of the PFAS family by definition."},
    "fluoropolymer": {"severity": "high", "reason": "Fluoropolymer — a class of PFAS used in coatings, fibers, and films."},
    "fluorotelomer": {"severity": "high", "reason": "Fluorotelomer — PFAS precursor that breaks down to PFOA and related compounds."},
    "dwr": {"severity": "medium", "reason": "Durable water repellent treatment — typically PFAS-based unless labeled 'C0 DWR' or 'PFC-free'."},
    "scotchgard": {"severity": "medium", "reason": "Scotchgard stain-resistance treatment — PFAS-based."},
    "gore-tex": {"severity": "medium", "reason": "Gore-Tex membrane (older versions) — PTFE-based. Newer 'ePE' versions are PFAS-free."},
    "stainmaster": {"severity": "medium", "reason": "Stainmaster carpet treatment — historically PFAS-based."},
    "fluorocarbon": {"severity": "medium", "reason": "Fluorocarbon — broad term that includes PFAS compounds."},
}

_PFAS_SEV_WEIGHT = {"high": 30, "medium": 15, "low": 5}


def analyze_pfas_in_text(text: str | None) -> dict[str, Any]:
    """Keyword scan for PFAS-family chemicals. Returns:
        {
          'score': 0-100 (higher = cleaner; None when text missing),
          'verdict': 'clean'|'caution'|'avoid'|'toxic'|'unknown',
          'flags': [{'name', 'reason', 'severity'}, ...],
          'high_count', 'medium_count', 'low_count'
        }
    """
    if not text:
        return {"score": None, "verdict": "unknown", "flags": [],
                "high_count": 0, "medium_count": 0, "low_count": 0}

    lower = text.lower()
    flags: list[dict[str, str]] = []
    seen: set[str] = set()
    for needle, meta in PFAS_KEYWORDS.items():
        if needle in seen:
            continue
        if needle in lower:
            flags.append({"name": needle, "reason": meta["reason"],
                          "severity": meta["severity"]})
            seen.add(needle)

    penalty = sum(_PFAS_SEV_WEIGHT.get(f["severity"], 0) for f in flags)
    score = max(0, 100 - penalty)
    if score >= 85: verdict = "clean"
    elif score >= 60: verdict = "caution"
    elif score >= 30: verdict = "avoid"
    else: verdict = "toxic"

    sev_order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda f: sev_order.get(f["severity"], 3))

    return {
        "score": score,
        "verdict": verdict,
        "flags": flags,
        "high_count": sum(1 for f in flags if f["severity"] == "high"),
        "medium_count": sum(1 for f in flags if f["severity"] == "medium"),
        "low_count": sum(1 for f in flags if f["severity"] == "low"),
    }


def pfas_category_lookup(*texts: str | None) -> dict[str, Any] | None:
    """Substring-match free text against PFAS_CATEGORY_KNOWLEDGE.

    Returns the first matching entry plus the matched key, or None.
    Pass any combination of product_name, subcategory, category,
    materials — strings are joined and lowercased.
    """
    blob = " ".join(t for t in texts if t).lower()
    if not blob:
        return None
    for key, entry in PFAS_CATEGORY_KNOWLEDGE.items():
        if key in blob:
            return {"key": key, **entry}
    return None


# ────────────────────────────────────────────────────────────────
# Universal barcode chain — Open Beauty Facts + Open Products Facts
# Food barcodes should hit fitapp_core.barcode first; this is the
# cosmetics + general-products fallback for the PFAS scanner UX.
# ────────────────────────────────────────────────────────────────

def _off_family_lookup(host: str, category: str, code: str) -> dict[str, Any] | None:
    """Fetch a product from an Open*Facts host. Returns the normalized
    result dict on success, None on miss or transport error."""
    try:
        req = urllib.request.Request(
            f"https://{host}/api/v2/product/{code}.json",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as r:
            data = json.loads(r.read())
    except Exception:
        return None
    if data.get("status") not in ("success", 1, "success_with_warnings"):
        return None
    p = data.get("product") or data
    materials = (p.get("ingredients_text", "")
                 or p.get("ingredients_text_en", "")
                 or p.get("materials", "")
                 or p.get("packaging", ""))
    return {
        "found": True,
        "category": category,
        "product_name": p.get("product_name", "") or p.get("product_name_en", ""),
        "name": p.get("product_name", "") or p.get("product_name_en", ""),
        "brand": p.get("brands", ""),
        "image_url": p.get("image_front_small_url", "") or p.get("image_url", ""),
        "materials": materials,
        "origin": p.get("origins", "") or p.get("manufacturing_places", ""),
    }


def barcode_universal(code: str) -> dict[str, Any]:
    """OFF (food) → OBF (cosmetics) → OPF (general products) chain.

    Returns {found: True, category, product_name, ...} on hit, or
    {found: False, code} on miss. Stdlib-only, no API keys.
    """
    # Food first (largest catalog)
    food = _off_family_lookup("world.openfoodfacts.org", "food", code)
    if food:
        return food
    # Cosmetics / personal care
    beauty = _off_family_lookup("world.openbeautyfacts.org", "cosmetics", code)
    if beauty:
        return beauty
    # General products / household
    products = _off_family_lookup("world.openproductsfacts.org", "household", code)
    if products:
        return products
    return {"found": False, "code": code}


def compose_scan_response(barcode_result: dict[str, Any]) -> dict[str, Any]:
    """One-call helper: take a barcode_universal() result and add the
    PFAS toxics analysis + category-knowledge layer + (on miss) the
    generic field guide. Consumers can just return this dict as JSON.
    """
    if not barcode_result.get("found"):
        return {**barcode_result, "generic_guidance": PFAS_GENERIC_GUIDANCE}

    materials = barcode_result.get("materials") or barcode_result.get("ingredients_text") or ""
    toxics = analyze_pfas_in_text(materials)
    category_pfas = pfas_category_lookup(
        barcode_result.get("product_name", ""),
        barcode_result.get("name", ""),
        barcode_result.get("subcategory", ""),
        barcode_result.get("category", ""),
        materials,
    )

    out = dict(barcode_result)
    out["toxics"] = toxics
    if category_pfas:
        out["category_pfas"] = category_pfas
    return out
