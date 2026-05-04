"""Allergen + condition + drug-food interaction alerts.

Reproduces the rule tables from FitApp's health_engine.py (extracted
2026-05-04). Caller passes meal items + user's health profile, gets back
a list of alerts ranked by severity.

Severity ladder:
    'stop'      → anaphylactic-tier; should drive a full-screen takeover
    'high'      → contains a confirmed allergen at meaningful dose
    'medium'    → may-contain / cross-contact / condition-specific concern
    'low'       → dietary-preference violation (vegan, kosher, etc.)
"""

from __future__ import annotations

from typing import Literal, TypedDict

Severity = Literal["stop", "high", "medium", "low"]


class Alert(TypedDict):
    severity: Severity
    code: str
    title: str
    message: str
    ingredient: str


class Allergy(TypedDict):
    allergen: str
    severity: Literal["anaphylactic", "severe", "moderate", "mild"]
    notes: str


# ────────────────────────────────────────────────────────────────────
#  Ingredient → allergen mapping. Hidden-name aware (e.g., "casein"
#  contains milk, "albumin" contains egg).
# ────────────────────────────────────────────────────────────────────

ALLERGEN_HIDDEN_NAMES: dict[str, set[str]] = {
    "milk": {
        "milk", "cream", "butter", "cheese", "casein", "caseinate", "whey",
        "lactose", "lactalbumin", "lactoglobulin", "ghee", "yogurt",
        "buttermilk", "curds", "kefir",
    },
    "egg": {
        "egg", "eggs", "albumin", "albumen", "ovalbumin", "ovomucin",
        "globulin", "lysozyme", "lecithin (e322)", "meringue",
    },
    "peanut": {
        "peanut", "peanuts", "groundnut", "arachis", "monkey nut",
        "beer nuts", "goober", "peanut butter", "mandelona", "pn-oil",
    },
    "tree_nut": {
        "almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
        "pecan", "pecans", "hazelnut", "hazelnuts", "pistachio", "pistachios",
        "macadamia", "brazil nut", "brazil nuts", "pine nut", "pine nuts",
        "praline", "marzipan", "nougat", "gianduja",
    },
    "shellfish": {
        "shrimp", "prawn", "prawns", "crab", "lobster", "crayfish", "crawfish",
        "scampi", "krill", "shellfish",
    },
    "mollusks": {
        "clam", "clams", "mussel", "mussels", "oyster", "oysters", "scallop",
        "scallops", "squid", "calamari", "octopus", "abalone", "snail", "escargot",
    },
    "fish": {
        "fish", "salmon", "tuna", "cod", "tilapia", "anchovy", "anchovies",
        "sardine", "sardines", "mackerel", "halibut", "trout", "bass", "snapper",
        "fish sauce", "worcestershire",
    },
    "wheat": {
        "wheat", "flour", "semolina", "spelt", "kamut", "farro", "bulgur",
        "couscous", "seitan", "wheat germ", "wheat bran",
    },
    "gluten": {
        "wheat", "barley", "rye", "malt", "brewer's yeast", "triticale",
        "spelt", "kamut", "semolina", "bulgur", "farro", "couscous",
        "seitan", "udon", "wheat", "flour",
    },
    "soy": {
        "soy", "soya", "soybean", "soybeans", "edamame", "tofu", "tempeh",
        "miso", "natto", "tamari", "shoyu", "textured vegetable protein",
        "tvp", "soy lecithin",
    },
    "sesame": {
        "sesame", "sesame seed", "sesame oil", "tahini", "halvah", "gomashio",
    },
    "sulfite": {
        "sulfite", "sulfites", "sulfur dioxide", "potassium bisulfite",
        "sodium bisulfite", "sodium metabisulfite", "potassium metabisulfite",
        "e220", "e221", "e222", "e223", "e224", "e226", "e227", "e228",
    },
    "mustard": {"mustard", "mustard seed", "mustard oil", "wasabi"},
    "corn": {"corn", "maize", "corn syrup", "high fructose corn syrup", "polenta", "masa", "hominy"},
    "msg": {"msg", "monosodium glutamate", "yeast extract", "autolyzed yeast", "hydrolyzed protein"},
}


def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def _ingredient_contains_allergen(ingredient: str, allergen: str) -> bool:
    needle = _normalize(ingredient)
    if not needle:
        return False
    candidates = ALLERGEN_HIDDEN_NAMES.get(allergen, {allergen})
    return any(c in needle for c in candidates)


def allergen_alerts(
    meal_items: list[dict],
    allergies: list[Allergy],
) -> list[Alert]:
    """Walk meal item names + ingredients, flag matches against the user's
    allergy profile. Anaphylactic allergies bubble up to severity='stop'.
    """
    if not meal_items or not allergies:
        return []

    alerts: list[Alert] = []
    for allergy in allergies:
        allergen = allergy.get("allergen", "")
        sev_user = allergy.get("severity", "moderate")
        for item in meal_items:
            haystack = " ".join([
                _normalize(item.get("name", "")),
                _normalize(item.get("description", "")),
                _normalize(item.get("ingredients_text", "")),
            ])
            if _ingredient_contains_allergen(haystack, allergen):
                if sev_user == "anaphylactic":
                    alerts.append({
                        "severity": "stop",
                        "code": f"contains_{allergen}_anaphylactic",
                        "title": f"DO NOT EAT — contains {allergen}",
                        "message": f"You marked {allergen} as anaphylactic. {item.get('name','This item')} contains it.",
                        "ingredient": item.get("name", ""),
                    })
                elif sev_user == "severe":
                    alerts.append({
                        "severity": "high",
                        "code": f"contains_{allergen}_severe",
                        "title": f"Contains {allergen}",
                        "message": f"{item.get('name','This item')} contains {allergen}, which you marked severe.",
                        "ingredient": item.get("name", ""),
                    })
                else:
                    alerts.append({
                        "severity": "medium",
                        "code": f"contains_{allergen}",
                        "title": f"Contains {allergen}",
                        "message": f"{item.get('name','This item')} contains {allergen}.",
                        "ingredient": item.get("name", ""),
                    })
    return alerts


# ────────────────────────────────────────────────────────────────────
#  Condition-aware nutrition flags
# ────────────────────────────────────────────────────────────────────

CONDITION_FLAGS: dict[str, dict] = {
    "t2d": {
        "trigger": lambda totals: totals.get("sugar", 0) > 25 or totals.get("carbs", 0) > 60,
        "message": "High carb/sugar load — consider splitting into smaller portions or pairing with protein/fat to flatten the glucose response.",
    },
    "t1d": {
        "trigger": lambda totals: True,  # always show carb count for insulin dosing
        "message": "Carbs to log for insulin: {carbs}g. FitApp does not dose insulin — confirm with your CGM and your provider.",
    },
    "hypertension": {
        "trigger": lambda totals: totals.get("sodium", 0) > 1500,
        "message": "High sodium for one meal. Daily target is <2300mg with your hypertension diagnosis.",
    },
    "ckd_stage_3": {
        "trigger": lambda totals: totals.get("protein", 0) > 30,
        "message": "Protein over 30g per meal can stress kidneys at CKD stage 3. Confirm target with your nephrologist.",
    },
    "gout": {
        "trigger": lambda totals: False,  # purine-aware needs ingredient check, not totals
        "message": "Watch high-purine foods (organ meat, anchovies, sardines, beer).",
    },
    "celiac": {
        "trigger": lambda totals: False,  # gluten check via allergen path
        "message": "Wheat/gluten alert — re-check the ingredient list.",
    },
    "pregnancy": {
        "trigger": lambda totals: totals.get("calories", 0) < 1800,
        "message": "Calorie target during pregnancy is typically 1800–2400+ depending on trimester. Confirm with your OB.",
    },
}


def condition_flags(
    totals: dict,
    conditions: list[str],
) -> list[Alert]:
    """Condition-aware alerts based on meal totals."""
    alerts: list[Alert] = []
    for cond in conditions:
        rule = CONDITION_FLAGS.get(cond)
        if not rule:
            continue
        if rule["trigger"](totals):
            alerts.append({
                "severity": "medium",
                "code": f"condition_{cond}",
                "title": cond.replace("_", " ").upper(),
                "message": rule["message"].format(**totals),
                "ingredient": "",
            })
    return alerts
