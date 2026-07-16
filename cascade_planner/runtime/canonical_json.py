"""Small canonical-JSON helpers shared across runtime and display adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_sha256(value: Any) -> str:
    """Return the legacy-compatible stable digest for a JSON-like value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON without lossy fallbacks or non-finite numbers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(strict_canonical_json_bytes(value)).hexdigest()


__all__ = [
    "canonical_json_sha256",
    "strict_canonical_json_bytes",
    "strict_canonical_json_sha256",
]
