"""Tests for fitapp_core.scraper.

Network-mocked: no real HTTP calls. Each test feeds a synthetic HTML
through the parser to confirm we extract the right fields.
"""
from __future__ import annotations

from unittest.mock import patch

from fitapp_core import scrape_brand
from fitapp_core import scraper as S


def _fake_fetch(html: str, final_url: str = "https://example.com/"):
    def f(_url):
        return final_url, html
    return f


def test_extract_basic_meta():
    html = """<!doctype html><html><head>
        <title>Equinox - It's Not Fitness, It's Life</title>
        <meta property="og:site_name" content="Equinox">
        <meta property="og:description" content="Premium fitness clubs.">
        <meta property="og:image" content="https://cdn.example.com/og.jpg">
        <meta name="theme-color" content="#000000">
        <link rel="icon" href="/favicon.png">
    </head><body>x</body></html>"""
    with patch.object(S, "_fetch", _fake_fetch(html, "https://www.equinox.com/")):
        r = scrape_brand("https://equinox.com")
    assert r["ok"] is True
    assert r["name"] == "Equinox"  # og:site_name beats title
    assert r["tagline"] == "Premium fitness clubs."
    assert r["primary_color"] == "#000000"
    assert r["hero_image"] == "https://cdn.example.com/og.jpg"
    # Logo candidates: og:image first, then favicon
    assert r["logo_candidates"][0].endswith("/og.jpg")
    assert any("favicon.png" in u for u in r["logo_candidates"])


def test_title_cleanup_picks_brand_segment():
    html = """<!doctype html><html><head>
        <title>Best Gyms - Crunch Fitness</title>
    </head><body></body></html>"""
    with patch.object(S, "_fetch", _fake_fetch(html, "https://www.crunch.com/")):
        r = scrape_brand("https://crunch.com")
    # Hostname is "crunch", "Crunch Fitness" segment matches → that wins
    assert r["name"] == "Crunch Fitness"


def test_relative_logo_absolutized():
    html = """<!doctype html><html><head>
        <title>Foo</title>
        <link rel="icon" href="/static/icon.png">
    </head><body></body></html>"""
    with patch.object(S, "_fetch", _fake_fetch(html, "https://foo.example/x")):
        r = scrape_brand("https://foo.example/x")
    assert "https://foo.example/static/icon.png" in r["logo_candidates"]


def test_color_extraction_from_css_filters_grey():
    html = """<!doctype html><html><head>
        <title>X</title>
        <style>
            :root { --brand:#a41e35; }
            body { color:#222222; background:#ffffff; }
            .a { color:#A41E35; }
            .b { color:#a41e36; }
            .c { color:#888; } /* grey, should be filtered */
        </style>
    </head><body></body></html>"""
    with patch.object(S, "_fetch", _fake_fetch(html, "https://x.example/")):
        r = scrape_brand("x.example")
    cands = r["color_candidates"]
    assert "#a41e35" in cands
    # The grey #888 should NOT make it in
    assert "#888888" not in cands


def test_failure_returns_error_field_no_raise():
    def boom(_):
        raise OSError("nope")
    with patch.object(S, "_fetch", boom):
        r = scrape_brand("https://broken.example/")
    assert r["ok"] is False
    assert "Couldn't reach" in (r["error"] or "")
    # All fields safely None / empty
    assert r["name"] is None
    assert r["logo_candidates"] == []


def test_hex_normalize_short_form():
    assert S._normalize_hex("#fff") == "#ffffff"
    assert S._normalize_hex("#A41E35") == "#a41e35"
    assert S._normalize_hex("rgb(164, 30, 53)") == "#a41e35"
    assert S._normalize_hex("not a color") is None


def test_font_hint_skips_system():
    html = """<style>body { font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; }</style>"""
    assert S._font_hint(html) == "Inter"
    html2 = """<style>body { font-family: 'Newsreader', Georgia, serif; }</style>"""
    assert S._font_hint(html2) == "Newsreader"


def test_bare_hostname_input_coerced():
    html = "<title>Foo</title>"
    with patch.object(S, "_fetch", _fake_fetch(html, "https://foo.example/")):
        r = scrape_brand("foo.example")
    assert r["ok"] is True


def test_unsupported_scheme_returns_error():
    r = scrape_brand("ftp://nope.example")
    assert r["ok"] is False
    assert "http" in (r["error"] or "").lower()


def test_hostname_to_name_strips_www():
    assert S._hostname_to_name("https://www.equinox.com/") == "Equinox"
    assert S._hostname_to_name("https://body-by-chosen.com/") == "Body-by-chosen"


def test_max_html_bytes_does_not_explode():
    big = "<title>Big</title>" + ("<div></div>" * 200_000)
    with patch.object(S, "_fetch", _fake_fetch(big, "https://big.example/")):
        r = scrape_brand("big.example")
    assert r["ok"] is True
    assert r["name"] == "Big"
