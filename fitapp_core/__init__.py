"""fitapp-core — shared health/fitness intelligence.

Stdlib-only by design. Consumers (FitApp, CoachHQ, Vitalstack) stay free
of dependency drift. AI calls take the API key as an argument so the
package never reads env directly — secrets flow from the host app.

Public surface (semver):
    barcode_with_fallback(code) -> dict       # OFF -> USDA chain
    valid_gtin_checksum(digits) -> bool       # GTIN-8/12/13/14 verify
    bmr_mifflin(age, sex, weight_kg, height_cm) -> float
    tdee(bmr, activity) -> float
    macro_targets(tdee, goal, weight_kg) -> dict
    cycle_phase(last_period_iso, today_iso=None) -> dict
    glucose_excursion(readings) -> dict
    allergen_alerts(meal_items, allergies) -> list
    audit_event(actor, action, resource, details) -> dict
"""

from .version import __version__
from .barcode import (
    barcode as off_barcode,
    barcode_with_fallback,
    usda_barcode,
    valid_gtin_checksum,
    off_entry_is_usable,
)
from .macros import bmr_mifflin, tdee, macro_targets, water_target
from .cycle import cycle_phase
from .glucose import glucose_excursion, time_in_range
from .alerts import allergen_alerts, condition_flags
from .audit import audit_event, AuditAction
from .exceptions import FitAppCoreError, ConfigError, ProviderError
from .scraper import scrape_brand

__all__ = [
    "__version__",
    "off_barcode",
    "barcode_with_fallback",
    "usda_barcode",
    "valid_gtin_checksum",
    "off_entry_is_usable",
    "bmr_mifflin",
    "tdee",
    "macro_targets",
    "water_target",
    "cycle_phase",
    "glucose_excursion",
    "time_in_range",
    "allergen_alerts",
    "condition_flags",
    "audit_event",
    "AuditAction",
    "FitAppCoreError",
    "ConfigError",
    "ProviderError",
    "scrape_brand",
]
