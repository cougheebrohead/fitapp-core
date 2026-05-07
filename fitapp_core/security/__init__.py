"""ELH Estate security middleware — shared across all 4 servers.

DB-agnostic primitives (same convention as `fitapp_core.audit`): each
function returns dicts/strings the caller persists or emits. No DB
calls, no I/O. The ELH Health enterprise audit table, the CoachHQ
SaaS table, and the FitApp consumer table all have different shapes
— this module gives them a common security surface to share.

Public API:

    from fitapp_core.security import (
        secure_headers,        # HSTS+nosniff+X-Frame+Referrer+Permissions+CSP
        cookie_writer,         # HttpOnly+Secure+SameSite=Strict cookies
        safe_html,             # html.escape with quote=True (XSS guard)
        safe_attr,             # context-aware attribute escape
        upload_validate,       # size cap + magic-byte sniff + traversal guard
        UploadError,
        hash_password,         # Argon2id
        verify_password,       # sniffs argon2/pbkdf2 prefixes, lazy-migrate signal
        needs_rehash,
        csrf_token,            # mint a double-submit token
        verify_csrf,           # constant-time compare cookie vs header
        rate_limit_check,      # sliding-window check (caller passes ledger fn)
        burst_detect,          # detector for failed-auth / signature mismatch / etc.
    )

Every helper is sync, dependency-light (stdlib + argon2-cffi), and
free of side-effects beyond the inputs given. No global state.
"""

from .headers import secure_headers, cookie_writer
from .escape import safe_html, safe_attr
from .uploads import upload_validate, UploadError
from .passwords import hash_password, verify_password, needs_rehash
from .csrf import csrf_token, verify_csrf
from .ratelimit import rate_limit_check, burst_detect

__all__ = [
    "secure_headers",
    "cookie_writer",
    "safe_html",
    "safe_attr",
    "upload_validate",
    "UploadError",
    "hash_password",
    "verify_password",
    "needs_rehash",
    "csrf_token",
    "verify_csrf",
    "rate_limit_check",
    "burst_detect",
]
