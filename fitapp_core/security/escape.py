"""Universal output escape — single entry point for all HTML interpolation.

Stored XSS is the single highest-impact code-level vector across the
estate. Every place a server interpolates user-controlled text into
HTML must route through `safe_html`. The CSP allows
`script-src 'unsafe-inline'` (because consumer apps embed UI as
single-file HTML) so an unescaped reflection executes.

Usage:

    body = f"<h1>Welcome, {safe_html(user.name)}</h1>"

For attribute contexts, use `safe_attr` (encodes additional chars
that matter inside attribute values):

    body = f'<a href="{safe_attr(url)}">click</a>'

Both helpers handle None gracefully (returns empty string) so callers
don't need a None check.
"""
from __future__ import annotations

import html
from urllib.parse import quote


def safe_html(value: object) -> str:
    """Escape `value` for inclusion in HTML body text or attribute values.

    Calls `html.escape(s, quote=True)` so &, <, >, ", ' are all encoded.
    None becomes "". Non-string values are str()'d first.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return html.escape(value, quote=True)


def safe_attr(value: object) -> str:
    """Same as safe_html — kept as a separate name for grep-ability and
    in case attribute-specific encoding diverges in future."""
    return safe_html(value)


def safe_url(value: object) -> str:
    """URL-component encode `value`. Use inside ?query= or path segments.

    Does NOT validate scheme — callers checking external URLs should
    additionally verify scheme is https before embedding.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return quote(value, safe="")


def safe_js_string(value: object) -> str:
    """Escape `value` for inclusion as a JS string literal inside a
    <script> block.

    Avoid using this if at all possible — prefer JSON-encoded data
    islands (e.g. data-* attributes + JSON.parse). This helper exists
    for legacy templates that interpolate strings directly into JS.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    out = []
    for ch in value:
        cp = ord(ch)
        if ch in ("\\", "'", '"', "\n", "\r", "\t", "<", ">", "&", "/"):
            out.append("\\u%04x" % cp)
        elif cp < 0x20 or cp == 0x7f:
            out.append("\\u%04x" % cp)
        else:
            out.append(ch)
    return "".join(out)
