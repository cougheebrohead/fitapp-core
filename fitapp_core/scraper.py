"""Brand scraper — given a public website URL, returns a brand kit.

Stdlib-only. Used by the ELH Coach + ELH Health onboarding wizards to
auto-fill a prospect's brand assets when standing up a sales-demo
tenant.

Design principles:
  - Best-effort, never raise. Missing fields come back as None.
  - One HTTP fetch for the homepage, one optional fetch for the favicon
    if og:image / icon link found nothing. Hard cap at 3 requests.
  - 8s timeout per request. Total wall-clock under 25s in the worst case.
  - No image color extraction (would need Pillow). We extract colors
    from <meta name="theme-color">, CSS custom properties, and inline
    style attributes. If none found, primary_color is None and the
    wizard prompts for manual entry.
  - Identifies itself in User-Agent — no spoofing. Sites that block us
    get a clean failure that the wizard can fall back from.
"""

from __future__ import annotations

import gzip
import io
import re
import socket
import ssl
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional

USER_AGENT = "ELHBrandScraper/1.0 (+sales-engineering; contact head@deanslist.net)"
FETCH_TIMEOUT = 8.0
MAX_HTML_BYTES = 2_000_000  # 2 MB cap so a misconfigured server can't OOM us

# Hex-color matcher — 3 or 6 digits. We reject 8-digit (#RRGGBBAA) here
# because most brand systems express their primary as RRGGBB and we want
# clean values for the UI.
_HEX_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
# rgb(r,g,b) — extract the three channels for normalization to hex.
_RGB_RE = re.compile(
    r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})",
    re.IGNORECASE,
)


class _Stop(Exception):
    """Raised internally to bail out of the parser early once we have
    enough — keeps us from walking the whole DOM of e.g. equinox.com."""


class _BrandParser(HTMLParser):
    """Single-pass HTML parser that captures the fields we care about.

    We deliberately avoid BeautifulSoup so this stays stdlib-only and
    portable to fitapp-core (which is the shared engine, no third-party
    deps allowed).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: Optional[str] = None
        self.in_title = False
        self.og_title: Optional[str] = None
        self.og_site_name: Optional[str] = None
        self.og_description: Optional[str] = None
        self.og_image: Optional[str] = None
        self.meta_description: Optional[str] = None
        self.theme_color: Optional[str] = None
        self.icon_href: Optional[str] = None
        self.apple_icon_href: Optional[str] = None
        # Body-end heuristic: stop once we've passed </head> AND have
        # at least a logo candidate, to avoid parsing 5MB of marketing
        # markup we don't need.
        self._head_closed = False

    # --- start tags ---
    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "title":
            self.in_title = True
            return
        if tag == "meta":
            name = (a.get("name") or "").lower()
            prop = (a.get("property") or "").lower()
            content = a.get("content") or ""
            if not content:
                return
            if name == "description" and not self.meta_description:
                self.meta_description = content.strip()
            elif name == "theme-color" and not self.theme_color:
                self.theme_color = content.strip()
            elif prop == "og:title" and not self.og_title:
                self.og_title = content.strip()
            elif prop == "og:site_name" and not self.og_site_name:
                self.og_site_name = content.strip()
            elif prop == "og:description" and not self.og_description:
                self.og_description = content.strip()
            elif prop == "og:image" and not self.og_image:
                self.og_image = content.strip()
            return
        if tag == "link":
            rel = (a.get("rel") or "").lower()
            href = a.get("href") or ""
            if not href:
                return
            if "apple-touch-icon" in rel and not self.apple_icon_href:
                self.apple_icon_href = href
            elif ("icon" in rel) and not self.icon_href:
                # Skip mask-icons (monochrome SVGs); we want full-color logos.
                if "mask-icon" not in rel:
                    self.icon_href = href
            return

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "head":
            self._head_closed = True
            # If we already have what we need, stop parsing the body.
            if self.og_image or self.apple_icon_href:
                raise _Stop()

    def handle_data(self, data):
        if self.in_title and not self.title:
            d = data.strip()
            if d:
                self.title = d


def _fetch(url: str) -> tuple[str, str]:
    """Fetch a URL; return (final_url, decoded_text). Raises on failure."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "gzip",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT, context=ctx) as resp:
        final = resp.geturl()
        raw = resp.read(MAX_HTML_BYTES + 1)
        if len(raw) > MAX_HTML_BYTES:
            raw = raw[:MAX_HTML_BYTES]
        if resp.headers.get("Content-Encoding", "").lower() == "gzip":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError):
                # Truncated gzip from our cap — try partial decode.
                try:
                    raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
                except Exception:
                    pass
        # Detect charset from Content-Type, fall back to utf-8 with replace.
        ctype = resp.headers.get("Content-Type", "")
        m = re.search(r"charset=([\w\-]+)", ctype, re.IGNORECASE)
        enc = m.group(1) if m else "utf-8"
        try:
            text = raw.decode(enc, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        return final, text


def _absolutize(base: str, href: str) -> str:
    """Resolve a possibly-relative href against the base URL."""
    if not href:
        return href
    return urllib.parse.urljoin(base, href.strip())


def _normalize_hex(s: str) -> Optional[str]:
    """Return a #rrggbb hex, or None if the input doesn't parse."""
    s = (s or "").strip()
    if not s:
        return None
    m = _HEX_RE.search(s)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return "#" + h.lower()
    m2 = _RGB_RE.search(s)
    if m2:
        try:
            r, g, b = (max(0, min(255, int(x))) for x in m2.groups())
            return f"#{r:02x}{g:02x}{b:02x}"
        except ValueError:
            return None
    return None


def _extract_css_colors(html: str, max_scan: int = 200_000) -> list[str]:
    """Pull plausible brand colors out of inline <style> blocks and
    style="" attributes. Returns a frequency-ordered list of #rrggbb.

    We cap scanned bytes because a 5MB marketing page can have thousands
    of colors and we only want the prominent ones."""
    chunk = html[:max_scan]
    counts: dict[str, int] = {}
    for m in _HEX_RE.finditer(chunk):
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        h = "#" + h.lower()
        # Drop near-white / near-black — they're chrome, not brand.
        r = int(h[1:3], 16)
        g = int(h[3:5], 16)
        b = int(h[5:7], 16)
        # Skip colors with low chroma close to black/white — body text,
        # backgrounds, borders. Keep anything saturated or mid-tone.
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        chroma = max_c - min_c
        if max_c < 32 or min_c > 224:
            continue
        if chroma < 24 and 64 < max_c < 200:
            # Mid grey — probably text. Skip.
            continue
        counts[h] = counts.get(h, 0) + 1
    return sorted(counts, key=lambda k: counts[k], reverse=True)


def _font_hint(html: str) -> Optional[str]:
    """First non-system font-family seen in inline CSS, or None.

    We want a hint we can match against Google Fonts later — not the
    full stack. So `font-family: 'Inter', sans-serif` returns 'Inter'."""
    m = re.search(
        r"font-family\s*:\s*([^;\}]+)",
        html[:200_000],
        re.IGNORECASE,
    )
    if not m:
        return None
    raw = m.group(1).strip()
    # Take the first entry, strip quotes.
    first = raw.split(",")[0].strip().strip('"\'')
    if not first or first.lower() in (
        "inherit", "initial", "unset", "revert",
        "system-ui", "-apple-system", "blinkmacsystemfont",
        "sans-serif", "serif", "monospace",
    ):
        # Real font is probably the next one.
        parts = [p.strip().strip('"\'') for p in raw.split(",")]
        for p in parts[1:]:
            if p and p.lower() not in (
                "inherit", "initial", "unset", "revert",
                "system-ui", "-apple-system", "blinkmacsystemfont",
                "sans-serif", "serif", "monospace",
            ):
                return p
        return None
    return first


def scrape_brand(url: str) -> dict:
    """Best-effort brand scrape. Always returns a dict — never raises.

    Output keys (any may be None):
      ok            — True if the homepage fetch succeeded
      error         — error string when ok=False
      source_url    — final URL after redirects
      name          — best-guess company name
      tagline       — short description
      logo_url      — absolute URL to a logo image candidate
      logo_candidates — up to 4 logo URLs ranked best-first
      primary_color — #rrggbb or None
      color_candidates — up to 6 plausible brand colors ranked
      hero_image    — og:image absolute URL or None
      font_hint     — bare font name like "Inter" or None
    """
    out: dict = {
        "ok": False,
        "error": None,
        "source_url": None,
        "name": None,
        "tagline": None,
        "logo_url": None,
        "logo_candidates": [],
        "primary_color": None,
        "color_candidates": [],
        "hero_image": None,
        "font_hint": None,
    }

    # Coerce bare hostnames into https://
    parsed = urllib.parse.urlparse(url.strip())
    if not parsed.scheme:
        url = "https://" + url.strip().lstrip("/")
        parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        out["error"] = "Only http(s) URLs are supported."
        return out

    try:
        final_url, html = _fetch(url)
    except (urllib.error.URLError, socket.timeout, ssl.SSLError, OSError, ValueError) as e:
        out["error"] = f"Couldn't reach the site: {e.__class__.__name__}: {e}"
        return out
    except Exception as e:
        out["error"] = f"Fetch failed: {e}"
        return out

    parser = _BrandParser()
    try:
        parser.feed(html)
    except _Stop:
        pass
    except Exception:
        # Malformed HTML — keep whatever we got.
        pass

    base = final_url
    out["ok"] = True
    out["source_url"] = final_url
    host_name = _hostname_to_name(final_url)
    out["name"] = (
        parser.og_site_name
        or _clean_title(parser.og_title, host_name)
        or _clean_title(parser.title, host_name)
        or host_name
    )
    out["tagline"] = parser.og_description or parser.meta_description

    # Logo candidates ranked: og:image, apple-touch-icon, link-icon, /favicon.ico
    candidates: list[str] = []
    for href in (parser.og_image, parser.apple_icon_href, parser.icon_href):
        if href:
            abs_href = _absolutize(base, href)
            if abs_href and abs_href not in candidates:
                candidates.append(abs_href)
    fav = _absolutize(base, "/favicon.ico")
    if fav not in candidates:
        candidates.append(fav)
    out["logo_candidates"] = candidates[:4]
    out["logo_url"] = candidates[0] if candidates else None
    out["hero_image"] = (
        _absolutize(base, parser.og_image) if parser.og_image else None
    )

    # Colors: theme-color first, then CSS color frequency.
    theme = _normalize_hex(parser.theme_color or "")
    css_colors = _extract_css_colors(html)
    ranked: list[str] = []
    if theme and theme not in ranked:
        ranked.append(theme)
    for c in css_colors:
        if c not in ranked:
            ranked.append(c)
        if len(ranked) >= 6:
            break
    out["color_candidates"] = ranked
    out["primary_color"] = ranked[0] if ranked else None

    out["font_hint"] = _font_hint(html)
    return out


_TITLE_SEPS = re.compile(r"\s*[\|—–\-·•:]{1,2}\s*")


def _clean_title(raw: Optional[str], host_name: str) -> Optional[str]:
    """Best-effort brand name from a <title> or og:title.

    Strategy:
      1. Split on common SEO separators (|, -, —, ·, •, :)
      2. If any segment contains the hostname name, prefer that segment
      3. Otherwise take the shortest segment >= 3 chars
      4. Trim common SEO boilerplate ('Official Site', 'Home', etc.)
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    parts = [p.strip() for p in _TITLE_SEPS.split(raw) if p.strip()]
    if not parts:
        return raw

    host_lower = host_name.lower()
    # Prefer a segment that mentions the hostname-derived name and is short
    for p in parts:
        if host_lower in p.lower() and len(p) <= 40:
            return _strip_boilerplate(p)

    # Otherwise: shortest segment with at least 3 chars
    candidates = [p for p in parts if len(p) >= 3]
    if not candidates:
        return _strip_boilerplate(parts[0])
    candidates.sort(key=len)
    return _strip_boilerplate(candidates[0])


_BOILERPLATE = re.compile(
    r"\b(official\s+site|official\s+website|home\s*page|welcome\s+to)\b",
    re.IGNORECASE,
)


def _strip_boilerplate(s: str) -> str:
    s = _BOILERPLATE.sub("", s).strip()
    # Trim trailing punctuation left over from removal
    return s.strip(" -|·•:").strip() or s


def _hostname_to_name(url: str) -> str:
    """Fallback name from the registrable hostname.
    'https://www.equinox.com/' -> 'Equinox'."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        host = host.lower().lstrip("www.")
        # Drop the TLD and capitalize the registrable.
        parts = host.split(".")
        if len(parts) >= 2:
            return parts[0].capitalize()
        return host.capitalize() if host else "Prospect"
    except Exception:
        return "Prospect"
