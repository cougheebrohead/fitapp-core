"""Multipart upload validation — size cap, magic-byte sniff, traversal guard.

Default policy:
  - 10 MB hard cap on the whole part
  - Whitelist content types: JPEG, PNG, HEIC/HEIF, PDF
  - Reject filenames containing / or \\ or \\..\\
  - Server caller must rename to a UUID before storage; client filename
    is for display only

Usage:

    from fitapp_core.security.uploads import upload_validate, UploadError

    try:
        result = upload_validate(image_bytes, declared_filename="meal.jpg")
    except UploadError as e:
        return self._j({"error": str(e), "code": e.code}, e.http_status)
    safe_filename = f"{uuid.uuid4()}.{result.extension}"
    # ... persist with `safe_filename`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

# Magic byte signatures, keyed by canonical extension.
MAGIC = {
    "jpeg": [b"\xff\xd8\xff"],
    "png":  [b"\x89PNG\r\n\x1a\n"],
    # HEIC/HEIF: ftyp box at offset 4. Match common brands.
    "heic": [b"ftypheic", b"ftypmif1", b"ftypheix", b"ftyphevc", b"ftyphevx"],
    "pdf":  [b"%PDF-"],
}

EXT_TO_MEDIA = {
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "heic": "image/heic",
    "pdf":  "application/pdf",
}


class UploadError(ValueError):
    """Raised when validation fails. Carries an HTTP status + machine code.

    Attributes:
      http_status:  413 (too large), 415 (wrong type), 400 (bad filename)
      code:         short machine-readable identifier
    """

    def __init__(self, message: str, *, http_status: int, code: str) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.code = code


@dataclass(frozen=True)
class UploadResult:
    extension: str        # one of {jpeg, png, heic, pdf}
    media_type: str       # canonical content-type derived from magic
    size_bytes: int


def _check_magic(payload: bytes, allowed: Iterable[str]) -> str | None:
    """Return the matched extension, or None if no allowed magic matched."""
    for ext in allowed:
        for sig in MAGIC.get(ext, ()):
            # HEIC magic is at offset 4, others at 0.
            if ext == "heic":
                if len(payload) >= len(sig) + 4 and payload[4:4 + len(sig)] == sig:
                    return ext
            else:
                if payload.startswith(sig):
                    return ext
    return None


def _check_filename(name: str) -> None:
    if not name:
        return
    if "/" in name or "\\" in name:
        raise UploadError(
            "filename contains path separator",
            http_status=400,
            code="bad_filename_path",
        )
    if ".." in name:
        raise UploadError(
            "filename contains parent-directory traversal",
            http_status=400,
            code="bad_filename_traversal",
        )
    if name.startswith("."):
        raise UploadError(
            "filename starts with dot (hidden file)",
            http_status=400,
            code="bad_filename_dotfile",
        )


def upload_validate(
    payload: bytes,
    *,
    declared_filename: str = "",
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowed_extensions: Iterable[str] = ("jpeg", "png", "heic"),
) -> UploadResult:
    """Validate a single uploaded payload.

    Raises UploadError if any check fails:
      - 413 too large
      - 415 unsupported content type (no magic match)
      - 400 bad filename (path separator, traversal, dotfile)

    Returns the canonical UploadResult on success. Caller renames to
    UUID before persisting; the result's `extension` is for that path.
    """
    if not payload:
        raise UploadError("empty payload", http_status=400, code="empty_payload")

    if len(payload) > max_bytes:
        raise UploadError(
            f"upload exceeds limit ({len(payload)} > {max_bytes} bytes)",
            http_status=413,
            code="too_large",
        )

    _check_filename(declared_filename)

    ext = _check_magic(payload, allowed_extensions)
    if ext is None:
        raise UploadError(
            "content type not in allowlist (magic-byte sniff failed)",
            http_status=415,
            code="unsupported_media",
        )

    return UploadResult(
        extension=ext,
        media_type=EXT_TO_MEDIA[ext],
        size_bytes=len(payload),
    )
