"""FHIR R4 lab-result parser.

Why this exists: Quest Diagnostics, LabCorp, and most major US labs
expose patient data through provider portals that let the patient
download their results as FHIR JSON Bundles. They do NOT (as of 2026)
expose a direct OAuth API to third-party consumer apps — those routes
are enterprise/HIPAA-gated. A user-uploaded FHIR file is the realistic
lab connector that's available today.

Single public entrypoint:

    parse_fhir_lab(content: str | bytes) -> dict

Returns the same shape `scan_lab` returns so the host server can hand
the result to existing render/save paths unchanged. Each biomarker
includes `direction` from BIOMARKER_DIRECTION so trend coloring is
correct on first save.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from .labs import biomarker_direction, _normalize_key


# ─── LOINC -> biomarker key map ──────────────────────────────────────
#
# Curated for the panels FitApp's UI displays as first-class. Anything
# not in this map still rides through with a normalized snake_case key
# derived from the Observation's `code.text` or `display`.

_LOINC_TO_KEY: dict[str, str] = {
    # Glycemic
    "4548-4":   "hba1c",
    "17856-6":  "hba1c",                  # IFCC mmol/mol variant
    "1558-6":   "fasting_glucose",
    "2345-7":   "glucose",
    "1554-5":   "glucose",                # plasma glucose
    "14771-0":  "insulin",
    "20448-7":  "insulin",
    # Lipid panel
    "2093-3":   "total_cholesterol",
    "13457-7":  "ldl_cholesterol",
    "2089-1":   "ldl_cholesterol",        # LDL direct
    "18262-6":  "ldl_cholesterol",
    "2085-9":   "hdl_cholesterol",
    "2571-8":   "triglycerides",
    "13457-7":  "ldl_cholesterol",
    "13458-5":  "vldl",
    "43396-1":  "non_hdl_cholesterol",
    "33914-3":  "egfr",
    # Apo / Lp(a)
    "1869-7":   "apo_a1",
    "1884-6":   "apo_b",
    "10835-7":  "lp_a",
    "74115-4":  "lp_a",
    # Thyroid
    "3016-3":   "tsh",
    "11580-8":  "tsh",
    "3024-7":   "free_t4",
    "3026-2":   "free_t3",
    "3050-2":   "total_t4",
    "3053-6":   "total_t3",
    # Vitamins / minerals
    "49541-6":  "vitamin_d",              # 25-OH D total
    "1989-3":   "vitamin_d",
    "2284-8":   "vitamin_b12",
    "2132-9":   "vitamin_b12",
    "2284-8":   "vitamin_b12",
    "2284-8":   "vitamin_b12",
    "2276-4":   "ferritin",
    "2498-4":   "iron",
    "2502-3":   "iron",
    "2500-7":   "iron",
    "11572-5":  "iron_saturation",
    "2501-5":   "transferrin_saturation",
    "2532-0":   "magnesium",
    # Inflammation / coag
    "1988-5":   "crp",
    "30522-7":  "hs_crp",
    "32209-5":  "esr",
    "13970-9":  "fibrinogen",
    "13458-5":  "homocysteine",
    "13965-9":  "homocysteine",
    # Liver
    "1742-6":   "alt",
    "1920-8":   "ast",
    "2324-2":   "ggt",
    "6768-6":   "alkaline_phosphatase",
    "1751-7":   "albumin",
    "1968-7":   "total_bilirubin",
    # Kidney / electrolytes
    "2160-0":   "creatinine",
    "3094-0":   "bun",
    "33914-3":  "egfr",
    "62238-1":  "egfr",
    "2951-2":   "sodium",
    "2823-3":   "potassium",
    "2075-0":   "chloride",
    "2028-9":   "co2",
    "17861-6":  "calcium",
    "2885-2":   "phosphorus",
    "3084-1":   "uric_acid",
    # CBC
    "718-7":    "hemoglobin",
    "4544-3":   "hematocrit",
    "789-8":    "rbc",
    "6690-2":   "wbc",
    "777-3":    "platelets",
    "787-2":    "mcv",
    "785-6":    "mch",
    "786-4":    "mchc",
    "788-0":    "rdw",
    "770-8":    "neutrophils",
    "736-9":    "lymphocytes",
    "5905-5":   "monocytes",
    "713-8":    "eosinophils",
    "706-2":    "basophils",
    # Sex hormones
    "2986-8":   "total_testosterone",
    "2991-8":   "free_testosterone",
    "13967-5":  "shbg",
    "2243-4":   "estradiol",
    "2839-9":   "progesterone",
    "2842-3":   "prolactin",
    "15067-2":  "fsh",
    "10501-5":  "lh",
    "34962-1":  "dhea_s",
    "2484-4":   "dhea_s",
    "2986-8":   "total_testosterone",
    # Other
    "9343-7":   "psa",
    "2106-3":   "fibrinogen",
    "2532-0":   "magnesium",
    "2823-3":   "potassium",
}


_FLAG_FROM_INTERP = {
    "H": "high", "HH": "high", "HU": "high", "A": "high",
    "L": "low",  "LL": "low",  "LU": "low",
    "N": "in_range", "NEG": "in_range",
}


def parse_fhir_lab(content: Any) -> dict[str, Any]:
    """Parse a FHIR R4 Bundle, DiagnosticReport, or Observation list.

    Accepts a JSON string, bytes, or already-parsed dict.

    Returns:
      {"panel_name": "...", "drawn_at": "YYYY-MM-DD" | None,
       "provider":   "...", "biomarkers": {...}, "warnings": [...],
       "raw_text":   ""}
    """
    warnings: list[str] = []

    # Normalize input to a dict
    if isinstance(content, (bytes, bytearray)):
        try:
            content = content.decode("utf-8")
        except UnicodeDecodeError:
            return _empty(["FHIR file is not valid UTF-8"])
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return _empty([f"FHIR JSON parse failed: {e}"])
    elif isinstance(content, dict):
        data = content
    else:
        return _empty([f"unsupported input type: {type(content).__name__}"])

    rt = (data.get("resourceType") or "").strip()
    if rt == "Bundle":
        return _parse_bundle(data, warnings)
    if rt == "DiagnosticReport":
        return _parse_diagnostic_report(data, [], warnings)
    if rt == "Observation":
        return _parse_observations([data], warnings,
                                   panel_name="Single observation",
                                   drawn_at=_observation_date(data),
                                   provider=None)

    warnings.append(f"unsupported resourceType: {rt!r}")
    return _empty(warnings)


# ─── Bundle parser ──────────────────────────────────────────────────


def _parse_bundle(bundle: dict, warnings: list[str]) -> dict[str, Any]:
    entries = bundle.get("entry") or []
    # Pull every Observation by id so DiagnosticReport.result references resolve.
    obs_by_id: dict[str, dict] = {}
    obs_list: list[dict] = []
    reports: list[dict] = []
    for e in entries:
        r = (e or {}).get("resource") or {}
        rt = r.get("resourceType")
        if rt == "Observation":
            obs_list.append(r)
            rid = r.get("id")
            if rid:
                obs_by_id[rid] = r
                obs_by_id[f"Observation/{rid}"] = r
        elif rt == "DiagnosticReport":
            reports.append(r)

    # Prefer a DiagnosticReport (gives panel_name, drawn date, provider).
    if reports:
        # If multiple reports, parse the first; warn about the rest.
        if len(reports) > 1:
            warnings.append(f"{len(reports)} DiagnosticReports in bundle; using the first")
        return _parse_diagnostic_report(reports[0], obs_list, warnings, obs_by_id=obs_by_id)

    # Naked observations only — no panel context.
    return _parse_observations(obs_list, warnings,
                               panel_name="Lab Observations",
                               drawn_at=_first_obs_date(obs_list),
                               provider=None)


def _parse_diagnostic_report(
    report: dict,
    bundle_obs: list[dict],
    warnings: list[str],
    obs_by_id: Optional[dict[str, dict]] = None,
) -> dict[str, Any]:
    panel = (
        (report.get("code") or {}).get("text")
        or _coding_display(((report.get("code") or {}).get("coding") or []))
        or "Lab Panel"
    )
    drawn = report.get("effectiveDateTime") or report.get("issued") or ""
    drawn_iso = _iso_date(drawn)
    provider = _provider_from_report(report)

    # Resolve observations: prefer report.result references, fall back to
    # `contained` Observations, fall back to bundle-wide observations.
    obs_list: list[dict] = []
    obs_by_id = obs_by_id or {}
    for ref in (report.get("result") or []):
        ref_str = ref.get("reference") if isinstance(ref, dict) else None
        if ref_str and ref_str in obs_by_id:
            obs_list.append(obs_by_id[ref_str])
        # also handle bare ids
        if ref_str and ref_str.startswith("urn:uuid:"):
            uid = ref_str.split("urn:uuid:", 1)[1]
            if uid in obs_by_id:
                obs_list.append(obs_by_id[uid])
    if not obs_list:
        for c in (report.get("contained") or []):
            if (c or {}).get("resourceType") == "Observation":
                obs_list.append(c)
    if not obs_list:
        obs_list = list(bundle_obs)

    return _parse_observations(obs_list, warnings,
                               panel_name=panel[:120],
                               drawn_at=drawn_iso,
                               provider=provider)


def _parse_observations(
    observations: Iterable[dict],
    warnings: list[str],
    *,
    panel_name: str,
    drawn_at: Optional[str],
    provider: Optional[str],
) -> dict[str, Any]:
    biomarkers: dict[str, dict[str, Any]] = {}
    for obs in observations:
        try:
            entry = _observation_to_entry(obs)
        except Exception as e:
            warnings.append(f"observation parse failed: {e}")
            continue
        if not entry:
            continue
        key, payload = entry
        if not key:
            continue
        # Add direction from BIOMARKER_DIRECTION
        payload["direction"] = biomarker_direction(key)
        biomarkers[key] = payload

    return {
        "panel_name": panel_name,
        "drawn_at":   drawn_at,
        "provider":   provider,
        "biomarkers": biomarkers,
        "warnings":   warnings,
        "raw_text":   "",
    }


# ─── Observation → biomarker entry ──────────────────────────────────


def _observation_to_entry(obs: dict) -> Optional[tuple[str, dict[str, Any]]]:
    code = obs.get("code") or {}
    codings = code.get("coding") or []
    key = _key_from_codings(codings) or _normalize_key(code.get("text") or "")
    if not key:
        return None

    # Value: prefer valueQuantity → valueString → component (first numeric).
    value = None
    unit = None
    vq = obs.get("valueQuantity")
    if isinstance(vq, dict):
        value = _to_float(vq.get("value"))
        unit = (vq.get("unit") or vq.get("code") or "").strip() or None

    if value is None:
        vs = obs.get("valueString")
        if isinstance(vs, str):
            m = re.search(r"-?\d+(?:\.\d+)?", vs)
            if m:
                try:
                    value = float(m.group())
                except ValueError:
                    pass

    # Reference range
    ref_low = ref_high = None
    rr_list = obs.get("referenceRange") or []
    if rr_list:
        rr = rr_list[0] or {}
        low = (rr.get("low") or {}).get("value")
        high = (rr.get("high") or {}).get("value")
        ref_low  = _to_float(low)
        ref_high = _to_float(high)
        if not unit:
            unit = (rr.get("low") or {}).get("unit") or (rr.get("high") or {}).get("unit")

    # Flag from interpretation OR derived from ranges.
    flag = "in_range"
    interp_codes: list[str] = []
    for ic in (obs.get("interpretation") or []):
        for c in (ic.get("coding") or []):
            code_val = (c.get("code") or "").upper()
            if code_val:
                interp_codes.append(code_val)
    if interp_codes:
        for ic in interp_codes:
            if ic in _FLAG_FROM_INTERP:
                flag = _FLAG_FROM_INTERP[ic]
                break
    elif value is not None:
        if ref_high is not None and value > ref_high:
            flag = "high"
        elif ref_low is not None and value < ref_low:
            flag = "low"

    return key, {
        "value":    value,
        "unit":     unit,
        "ref_low":  ref_low,
        "ref_high": ref_high,
        "flag":     flag,
    }


def _key_from_codings(codings: list[dict]) -> Optional[str]:
    """Walk a `code.coding` list, return our biomarker key for the first
    LOINC entry that maps. Falls through if no LOINC."""
    for c in codings:
        system = (c.get("system") or "").lower()
        code   = (c.get("code") or "").strip()
        if not code:
            continue
        if "loinc" in system and code in _LOINC_TO_KEY:
            return _LOINC_TO_KEY[code]
    # If no LOINC mapping, try the display text
    for c in codings:
        disp = c.get("display") or ""
        norm = _normalize_key(disp)
        if norm:
            return norm
    return None


# ─── helpers ────────────────────────────────────────────────────────


def _empty(warnings: list[str]) -> dict[str, Any]:
    return {
        "panel_name": None,
        "drawn_at":   None,
        "provider":   None,
        "biomarkers": {},
        "warnings":   warnings,
        "raw_text":   "",
    }


def _provider_from_report(report: dict) -> Optional[str]:
    perf = report.get("performer") or []
    if perf and isinstance(perf, list):
        first = perf[0] or {}
        disp = first.get("display") or (first.get("reference") or "")
        if disp:
            return str(disp)[:120]
    issuer = (report.get("performer") or [{}])[0] if report.get("performer") else {}
    if isinstance(issuer, dict) and issuer.get("display"):
        return str(issuer["display"])[:120]
    return None


def _coding_display(codings: list[dict]) -> Optional[str]:
    for c in codings or []:
        if c.get("display"):
            return c["display"]
    return None


def _observation_date(obs: dict) -> Optional[str]:
    return _iso_date(obs.get("effectiveDateTime") or obs.get("issued") or "")


def _first_obs_date(obs_list: list[dict]) -> Optional[str]:
    for o in obs_list:
        d = _observation_date(o)
        if d:
            return d
    return None


def _iso_date(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    return m.group(0) if m else None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None
