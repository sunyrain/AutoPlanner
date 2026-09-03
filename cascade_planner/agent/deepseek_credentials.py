"""Dependency-free DeepSeek credential normalization for active callers."""
from __future__ import annotations


PLACEHOLDER_DEEPSEEK_KEY = "replace_with_your_deepseek_key"


def normalize_deepseek_key_value(value: str | None) -> str:
    normalized = str(value or "").strip()
    for quote in ('"', "'"):
        if normalized.endswith(quote):
            normalized = normalized[:-1]
        if normalized.startswith(quote):
            normalized = normalized[1:]
    return normalized.strip()


def is_placeholder_deepseek_key(value: str | None) -> bool:
    return normalize_deepseek_key_value(value) == PLACEHOLDER_DEEPSEEK_KEY


__all__ = [
    "PLACEHOLDER_DEEPSEEK_KEY",
    "is_placeholder_deepseek_key",
    "normalize_deepseek_key_value",
]
