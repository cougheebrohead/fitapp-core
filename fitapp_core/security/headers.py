"""HTTP response security headers.

`secure_headers()` returns the canonical baseline every ELH server
sends on every response. `cookie_writer()` builds the Set-Cookie line
for session-shaped cookies.

The CSP is intentionally permissive on `script-src 'unsafe-inline'`
because the consumer apps embed their UI as single-file HTML. Stored
XSS is mitigated at the OUTPUT-ESCAPE layer (`safe_html`), not by CSP.
For services that don't need inline scripts (admin UIs), pass
`allow_inline_scripts=False` to tighten the CSP.

Usage from a stdlib `http.server`:

    for k, v in secure_headers(canonical_origin="https://example.com").items():
        self.send_header(k, v)
"""
from __future__ import annotations

from typing import Mapping


def secure_headers(
    *,
    canonical_origin: str | None = None,
    extra_connect: tuple[str, ...] = (),
    extra_script: tuple[str, ...] = (),
    extra_style: tuple[str, ...] = (),
    extra_frame: tuple[str, ...] = (),
    allow_inline_scripts: bool = True,
    allow_inline_styles: bool = True,
    allow_data_images: bool = True,
    allow_blob_media: bool = True,
    permit_camera: bool = False,
    permit_geolocation: bool = False,
    permit_microphone: bool = False,
) -> Mapping[str, str]:
    """Build the baseline secure-headers dict.

    Returned headers (always set):
      - Strict-Transport-Security (HSTS preload)
      - X-Content-Type-Options
      - X-Frame-Options
      - Referrer-Policy
      - Permissions-Policy
      - Content-Security-Policy

    `canonical_origin` is added to `connect-src` / `frame-ancestors`
    where appropriate. Pass each app's actual origin so XHR + EventSource
    can reach itself.
    """
    perms = []
    perms.append(f"camera={'(self)' if permit_camera else '()'}")
    perms.append(f"microphone={'(self)' if permit_microphone else '()'}")
    perms.append(f"geolocation={'(self)' if permit_geolocation else '()'}")
    perms.append("interest-cohort=()")
    perms.append("usb=()")
    perms.append("payment=(self)")

    script_sources = ["'self'"]
    if allow_inline_scripts:
        script_sources.append("'unsafe-inline'")
    script_sources.extend(extra_script)

    style_sources = ["'self'"]
    if allow_inline_styles:
        style_sources.append("'unsafe-inline'")
    style_sources.extend(extra_style)

    img_sources = ["'self'", "https:"]
    if allow_data_images:
        img_sources.append("data:")
    if allow_blob_media:
        img_sources.append("blob:")

    media_sources = ["'self'", "https:"]
    if allow_blob_media:
        media_sources.append("blob:")

    connect_sources = ["'self'"]
    if canonical_origin:
        connect_sources.append(canonical_origin)
    connect_sources.extend(extra_connect)

    frame_sources = list(extra_frame)

    csp_parts = [
        "default-src 'self'",
        f"script-src {' '.join(script_sources)}",
        f"style-src {' '.join(style_sources)}",
        f"img-src {' '.join(img_sources)}",
        f"media-src {' '.join(media_sources)}",
        f"connect-src {' '.join(connect_sources)}",
        f"frame-src {' '.join(frame_sources)}" if frame_sources else None,
        "font-src 'self' data:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]

    csp = "; ".join(p for p in csp_parts if p)

    return {
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": ", ".join(perms),
        "Content-Security-Policy": csp,
    }


def cookie_writer(
    name: str,
    value: str,
    *,
    max_age: int | None = None,
    path: str = "/",
    secure: bool = True,
    http_only: bool = True,
    same_site: str = "Strict",
    domain: str | None = None,
) -> str:
    """Build a Set-Cookie directive for a session-shaped cookie.

    Defaults are the safest possible: HttpOnly, Secure, SameSite=Strict.
    Override only when you have a clear reason — e.g. SameSite=Lax for
    an OAuth-callback cookie that needs to survive a cross-site GET.

    Reject deletion cookies via `delete_cookie()` instead of overloading
    this with a sentinel max_age=0; the intent is clearer.
    """
    if same_site not in {"Strict", "Lax", "None"}:
        raise ValueError(f"invalid same_site: {same_site!r}")
    if same_site == "None" and not secure:
        # Browsers reject SameSite=None without Secure on modern versions.
        raise ValueError("SameSite=None requires Secure=True")
    parts = [f"{name}={value}"]
    parts.append(f"Path={path}")
    if max_age is not None:
        parts.append(f"Max-Age={int(max_age)}")
    if domain:
        parts.append(f"Domain={domain}")
    if secure:
        parts.append("Secure")
    if http_only:
        parts.append("HttpOnly")
    parts.append(f"SameSite={same_site}")
    return "; ".join(parts)


def delete_cookie(name: str, *, path: str = "/", domain: str | None = None) -> str:
    """Build a Set-Cookie line that deletes a previously-set cookie.

    Browsers honor any cookie with Max-Age=0 + Path matching the original.
    """
    parts = [f"{name}=", f"Path={path}", "Max-Age=0", "HttpOnly", "Secure", "SameSite=Strict"]
    if domain:
        parts.insert(2, f"Domain={domain}")
    return "; ".join(parts)
