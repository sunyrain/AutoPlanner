"""Integrity contract for concise, exact-source retrosynthesis dossiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .replay_contract import digest


CASE_DOSSIER_SCHEMA = "retrosynthesis_case_dossier.v1"


class CaseDossierError(ValueError):
    """A dossier cannot be compiled into a scientifically closed pack."""


def json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def load_case_dossier(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        dossier = json_copy(value)
    else:
        dossier = json.loads(
            Path(value).expanduser().resolve().read_text(encoding="utf-8")
        )
    if not isinstance(dossier, dict):
        raise CaseDossierError("case_dossier_must_be_an_object")
    if dossier.get("schema_version") != CASE_DOSSIER_SCHEMA:
        raise CaseDossierError("case_dossier_schema_invalid")
    supplied = str(dossier.get("content_sha256") or "")
    computed = digest(
        {key: item for key, item in dossier.items() if key != "content_sha256"}
    )
    if supplied and supplied != computed:
        raise CaseDossierError("case_dossier_digest_invalid")
    dossier["content_sha256"] = computed
    return dossier


def with_case_dossier_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    dossier = json_copy(value)
    dossier.pop("content_sha256", None)
    dossier["content_sha256"] = digest(dossier)
    return dossier


__all__ = [
    "CASE_DOSSIER_SCHEMA",
    "CaseDossierError",
    "json_copy",
    "load_case_dossier",
    "with_case_dossier_digest",
]
