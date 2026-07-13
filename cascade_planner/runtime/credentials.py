"""Credential discovery for new runtime components.

No repository-relative credential file is ever consulted implicitly.  Secret
values remain outside artifacts, metrics, manifests, and error messages.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Credential:
    value: str
    source: str

    def __repr__(self) -> str:
        return f"Credential(value=<redacted>, source={self.source!r})"


def resolve_codex_credential(
    *,
    environ: Mapping[str, str] | None = None,
    explicit_path: str | os.PathLike[str] | None = None,
) -> Credential | None:
    env = dict(os.environ if environ is None else environ)
    for name in (
        "AUTOPLANNER_CODEX_API_KEY",
        "OPENAI_API_KEY",
    ):
        value = _normalize_secret(env.get(name))
        if value:
            return Credential(value=value, source=f"env:{name}")
    configured_path = explicit_path or str(
        env.get("AUTOPLANNER_CODEX_KEY_PATH") or ""
    ).strip()
    if not configured_path:
        return None
    path = Path(configured_path).expanduser().resolve()
    try:
        value = _normalize_secret(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return Credential(value=value, source="configured_file") if value else None


def _normalize_secret(value: str | None) -> str:
    normalized = str(value or "").strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {
        "'",
        '"',
    }:
        normalized = normalized[1:-1].strip()
    return normalized


__all__ = ["Credential", "resolve_codex_credential"]
