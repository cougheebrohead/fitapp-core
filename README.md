# fitapp-core

Internal Python package. Shared health/fitness intelligence consumed by FitApp, CoachHQ, and Vitalstack. Stdlib-only by design.

## What's in here

| Module | Purpose |
|---|---|
| `fitapp_core.barcode` | OFF → USDA barcode lookup chain, GTIN checksum, junk-entry filter |
| `fitapp_core.macros` | Mifflin-St Jeor BMR + TDEE + goal-aware macro targets + water target |
| `fitapp_core.cycle` | Menstrual cycle phase resolver (28-day default, scales for non-28) |
| `fitapp_core.glucose` | Time-in-range, GMI A1C, post-meal excursion + AUC |
| `fitapp_core.alerts` | Allergen rule table (hidden-name aware) + condition-aware nutrition flags |
| `fitapp_core.audit` | Audit event constructor with SHA-256 chain digest for tamper-evidence |

## Install (private — git URL)

```bash
pip install git+https://github.com/cougheebrohead/fitapp-core.git@main
```

For a pinned version use a tag: `git+https://github.com/cougheebrohead/fitapp-core.git@v0.1.0`

## Usage

```python
from fitapp_core import (
    barcode_with_fallback,
    bmr_mifflin, tdee, macro_targets,
    cycle_phase,
    time_in_range,
    allergen_alerts, condition_flags,
    audit_event, AuditAction,
)

# Barcode chain (OFF then USDA)
r = barcode_with_fallback("7894900010015")          # → Coca-Cola 350ml via OFF
r = barcode_with_fallback("049000050103", usda_api_key=USDA_KEY)  # → Coke 2L via USDA

# Macros
bmr = bmr_mifflin(30, "female", 65, 165)            # 1389 kcal
goals = macro_targets(tdee(bmr, "moderate"), "lose", 65, "female")

# Cycle
phase = cycle_phase("2026-04-25")                   # → {"phase": "luteal", "day": 22, ...}

# Glucose
tir = time_in_range(readings)                       # → TIR + GMI

# Alerts
alerts = allergen_alerts(meal_items, user.allergies)
flags  = condition_flags(meal.totals, user.conditions)

# Audit
ev = audit_event(actor_id="u123", action=AuditAction.READ_LAB,
                 resource_type="lab", resource_id="lab456")
db.audit_log.insert(ev)  # caller persists
```

## Design rules

1. **No I/O outside of HTTP-to-OFF/USDA in `barcode.py`.** Everything else is pure functions.
2. **No env reads.** API keys come in as arguments. The host app owns secrets.
3. **No DB.** This package builds dicts; callers persist.
4. **Stdlib only.** Adding a dependency requires a written justification.
5. **Versioned strict.** Public surface in `fitapp_core/__init__.py`. Bumps follow semver.

## Run tests

```bash
pip install -e ".[test]"
pytest                          # all
pytest -m "not network"         # offline only
```

The barcode tests intentionally hit live OFF + USDA so we know within seconds when an upstream contract drifts. Mark `network` to skip on offline CI runs.

## Source

Extracted from FitApp on 2026-05-04 (commit `f4752a4` of `cougheebrohead/fitapp`). FitApp itself was not modified — this package is a parallel-extraction so the new products (CoachHQ, Vitalstack) can import the same logic without touching FitApp's deployed surface.
