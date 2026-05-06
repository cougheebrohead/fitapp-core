"""Tests for fitapp_core.fhir.parse_fhir_lab."""
from __future__ import annotations

import json
import sys

from fitapp_core import parse_fhir_lab
from fitapp_core import fhir as F


# ─── single-observation parse ─────────────────────────────────────────


def test_observation_with_loinc_hba1c():
    obs = {
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]},
        "valueQuantity": {"value": 5.4, "unit": "%"},
        "referenceRange": [{"low": {"value": 4.0}, "high": {"value": 5.6}}],
        "effectiveDateTime": "2026-04-22",
    }
    out = parse_fhir_lab(obs)
    bm = out["biomarkers"]
    assert "hba1c" in bm
    assert bm["hba1c"]["value"] == 5.4
    assert bm["hba1c"]["unit"]  == "%"
    assert bm["hba1c"]["ref_low"]  == 4.0
    assert bm["hba1c"]["ref_high"] == 5.6
    assert bm["hba1c"]["flag"] == "in_range"
    assert bm["hba1c"]["direction"] == "up_bad"
    assert out["drawn_at"] == "2026-04-22"


def test_observation_high_flag_from_interpretation():
    obs = {
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org", "code": "13457-7"}]},
        "valueQuantity": {"value": 140, "unit": "mg/dL"},
        "interpretation": [{"coding": [{"code": "H"}]}],
        "referenceRange": [{"high": {"value": 100}}],
    }
    out = parse_fhir_lab(obs)
    assert out["biomarkers"]["ldl_cholesterol"]["flag"] == "high"
    assert out["biomarkers"]["ldl_cholesterol"]["direction"] == "up_bad"


def test_observation_high_flag_from_range_when_no_interp():
    obs = {
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org", "code": "13457-7"}]},
        "valueQuantity": {"value": 140, "unit": "mg/dL"},
        "referenceRange": [{"high": {"value": 100}}],
    }
    out = parse_fhir_lab(obs)
    assert out["biomarkers"]["ldl_cholesterol"]["flag"] == "high"


def test_observation_low_flag_derived():
    obs = {
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org", "code": "49541-6"}]},
        "valueQuantity": {"value": 22, "unit": "ng/mL"},
        "referenceRange": [{"low": {"value": 30}, "high": {"value": 100}}],
    }
    out = parse_fhir_lab(obs)
    assert out["biomarkers"]["vitamin_d"]["flag"] == "low"
    assert out["biomarkers"]["vitamin_d"]["direction"] == "up_good"


def test_observation_falls_back_to_text_when_no_loinc():
    obs = {
        "resourceType": "Observation",
        "code": {"text": "HDL Cholesterol"},
        "valueQuantity": {"value": 55, "unit": "mg/dL"},
    }
    out = parse_fhir_lab(obs)
    bm = out["biomarkers"]
    assert "hdl_cholesterol" in bm
    assert bm["hdl_cholesterol"]["value"] == 55


# ─── DiagnosticReport parse ───────────────────────────────────────────


def test_diagnostic_report_with_contained_observations():
    report = {
        "resourceType": "DiagnosticReport",
        "code": {"text": "Lipid Panel"},
        "effectiveDateTime": "2026-04-22T08:00:00Z",
        "performer": [{"display": "Quest Diagnostics"}],
        "contained": [
            {
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "13457-7"}]},
                "valueQuantity": {"value": 118, "unit": "mg/dL"},
                "interpretation": [{"coding": [{"code": "H"}]}],
                "referenceRange": [{"high": {"value": 100}}],
            },
            {
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "2085-9"}]},
                "valueQuantity": {"value": 58, "unit": "mg/dL"},
                "referenceRange": [{"low": {"value": 40}}],
            },
        ],
    }
    out = parse_fhir_lab(report)
    assert out["panel_name"] == "Lipid Panel"
    assert out["drawn_at"] == "2026-04-22"
    assert out["provider"] == "Quest Diagnostics"
    assert out["biomarkers"]["ldl_cholesterol"]["flag"] == "high"
    assert out["biomarkers"]["hdl_cholesterol"]["value"] == 58


# ─── Bundle parse ────────────────────────────────────────────────────


def test_bundle_with_referenced_observations():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {
                "resourceType": "DiagnosticReport",
                "code": {"text": "Comprehensive Metabolic + Lipid + A1C"},
                "effectiveDateTime": "2026-04-22",
                "performer": [{"display": "LabCorp"}],
                "result": [
                    {"reference": "Observation/o-hba1c"},
                    {"reference": "Observation/o-ldl"},
                    {"reference": "Observation/o-tsh"},
                ],
            }},
            {"resource": {
                "resourceType": "Observation",
                "id": "o-hba1c",
                "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
                "valueQuantity": {"value": 5.4, "unit": "%"},
                "referenceRange": [{"high": {"value": 5.6}}],
            }},
            {"resource": {
                "resourceType": "Observation",
                "id": "o-ldl",
                "code": {"coding": [{"system": "http://loinc.org", "code": "13457-7"}]},
                "valueQuantity": {"value": 118, "unit": "mg/dL"},
                "interpretation": [{"coding": [{"code": "H"}]}],
                "referenceRange": [{"high": {"value": 100}}],
            }},
            {"resource": {
                "resourceType": "Observation",
                "id": "o-tsh",
                "code": {"coding": [{"system": "http://loinc.org", "code": "3016-3"}]},
                "valueQuantity": {"value": 1.8, "unit": "mIU/L"},
                "referenceRange": [{"low": {"value": 0.4}, "high": {"value": 4.5}}],
            }},
        ],
    }
    out = parse_fhir_lab(bundle)
    assert out["panel_name"].startswith("Comprehensive")
    assert out["provider"] == "LabCorp"
    assert out["drawn_at"] == "2026-04-22"
    bm = out["biomarkers"]
    assert bm["hba1c"]["value"] == 5.4
    assert bm["hba1c"]["direction"] == "up_bad"
    assert bm["ldl_cholesterol"]["flag"] == "high"
    assert bm["tsh"]["direction"] == "neutral"


def test_bundle_observations_only_no_report():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "2085-9"}]},
                "valueQuantity": {"value": 60, "unit": "mg/dL"},
                "effectiveDateTime": "2026-04-15",
            }},
        ],
    }
    out = parse_fhir_lab(bundle)
    assert "hdl_cholesterol" in out["biomarkers"]
    assert out["biomarkers"]["hdl_cholesterol"]["direction"] == "up_good"
    assert out["drawn_at"] == "2026-04-15"


# ─── input variants ──────────────────────────────────────────────────


def test_accepts_json_string():
    obs = {"resourceType": "Observation",
           "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
           "valueQuantity": {"value": 5.5}}
    out = parse_fhir_lab(json.dumps(obs))
    assert out["biomarkers"]["hba1c"]["value"] == 5.5


def test_accepts_bytes():
    obs = {"resourceType": "Observation",
           "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
           "valueQuantity": {"value": 5.5}}
    out = parse_fhir_lab(json.dumps(obs).encode())
    assert out["biomarkers"]["hba1c"]["value"] == 5.5


def test_invalid_json_returns_warning():
    out = parse_fhir_lab("not json at all")
    assert out["biomarkers"] == {}
    assert any("parse failed" in w for w in out["warnings"])


def test_unknown_resource_type():
    out = parse_fhir_lab({"resourceType": "Patient"})
    assert out["biomarkers"] == {}
    assert any("unsupported" in w for w in out["warnings"])


def test_empty_dict():
    out = parse_fhir_lab({})
    assert out["biomarkers"] == {}


# ─── runner ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = []
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            fails.append((fn.__name__, e))
            print(f"  ✗ {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(fails)}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
