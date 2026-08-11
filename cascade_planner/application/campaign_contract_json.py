"""Pure JSON and digest helpers for the immutable campaign contract."""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


def bound_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = plain_json(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = digest(row)
    return row


def normalized_strings(values: Any) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values or () if str(value).strip()}))


def freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("campaign contract values must be JSON-compatible")


def plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("campaign contract values must be JSON-compatible")


def is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "bound_row",
    "digest",
    "freeze_json",
    "is_sha256",
    "normalized_strings",
    "plain_json",
]
