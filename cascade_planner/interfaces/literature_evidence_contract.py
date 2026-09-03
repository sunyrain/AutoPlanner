"""Deterministic contracts for literature discovery observations."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _compact_discovery_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Bound verbose procedure text while preserving source-binding fields."""

    row = dict(source)
    procedures: list[dict[str, Any]] = []
    for raw in row.get("procedure_inventory") or []:
        if not isinstance(raw, Mapping):
            continue
        procedure = dict(raw)
        for key in ("procedure_excerpt", "procedure", "text", "source_grounding"):
            if key in procedure:
                procedure[key] = str(procedure.get(key) or "")[:800]
        procedures.append(procedure)
        if len(procedures) >= 12:
            break
    row["procedure_inventory"] = procedures
    row["semantics"] = {
        **dict(row.get("semantics") or {}),
        "discovery_procedure_inventory_bounded": True,
        "full_source_artifact_remains_hash_bound": True,
    }
    return row


def _route_binding_eligible(
    source: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> bool:
    return bool(
        source.get("procedure_inventory")
        and source.get("source_fulltext_sha256")
        and any(
            isinstance(row, Mapping)
            and row.get("current_host_reaction_validated") is True
            and str(row.get("product_smiles") or "")
            for row in request.get("edges") or []
        )
    )




__all__: list[str] = []
