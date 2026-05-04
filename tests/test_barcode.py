"""Tests for the barcode chain. These hit live OFF + USDA APIs by design —
we want to know immediately when an upstream contract changes.
Network-marked so CI can skip on offline runs.
"""

import pytest

from fitapp_core import (
    barcode_with_fallback,
    valid_gtin_checksum,
    off_entry_is_usable,
)


# ───────────────────────── checksum unit tests (offline) ─────────────────────────

@pytest.mark.parametrize(
    "code,expected",
    [
        ("7894900010015", True),    # Brazilian Coke EAN-13
        ("016000275287",  True),    # Cheerios UPC-A
        ("00016000275287", True),   # Same as 14-digit
        ("049000050103",  True),    # Coke 2L US
        ("7894900010014", False),   # bad check digit
        ("123",           False),   # too short
        ("abc12345",      False),   # non-numeric
        ("00012345670",   False),   # wrong length
        ("",              False),
    ],
)
def test_gtin_checksum(code, expected):
    assert valid_gtin_checksum(code) is expected


def test_off_entry_filter_rejects_date_name():
    """Real production case — Coke 2L US had product_name='11/30/25'."""
    p = {"product_name": "11/30/25"}
    n = {}
    assert off_entry_is_usable(p, n) is False


def test_off_entry_filter_accepts_real_product():
    p = {"product_name": "Coca-Cola 350ml"}
    n = {"sugars_100g": 10.5, "energy-kcal_100g": 39}
    assert off_entry_is_usable(p, n) is True


def test_off_entry_filter_rejects_impossible_sugar():
    p = {"product_name": "Real Name"}
    n = {"sugars_100g": 200, "energy-kcal_100g": 100}
    assert off_entry_is_usable(p, n) is False


# ───────────────────────── live chain (network) ─────────────────────────

@pytest.mark.network
def test_brazilian_coke_via_off():
    r = barcode_with_fallback("7894900010015")
    assert r.get("found") is True
    assert r.get("source") == "open_food_facts"
    assert "Coca" in (r.get("name") or "")
    assert (r.get("n") or {}).get("sugar", 0) > 30  # Coke is 37g sugar / 350ml


@pytest.mark.network
def test_us_coke_2l_falls_through_to_usda():
    """Real bug regression: this code's OFF entry has corrupted name +
    serving size. Filter must reject it so chain falls to USDA."""
    r = barcode_with_fallback("049000050103")
    assert r.get("found") is True
    assert r.get("source") == "usda_branded"
    sugar = (r.get("n") or {}).get("sugar", 0)
    cal = (r.get("n") or {}).get("calories", 0)
    # Sanity bounds — anything wildly outside is a regression
    assert 30 <= sugar <= 60
    assert 100 <= cal <= 200


@pytest.mark.network
def test_junk_barcode_returns_not_found():
    r = barcode_with_fallback("0000000000000")
    assert r.get("found") is False
