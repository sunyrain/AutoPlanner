"""Logical example identity and upgrade merging for self-evolution memory."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def merge_template_example(
    existing: Mapping[str, Mapping[str, Any]],
    *,
    exact: Mapping[str, Any],
    edge_id: str,
    edge: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Deduplicate record revisions and retain the richest source observation."""

    merged: dict[str, dict[str, Any]] = {}
    changed = False
    for stored_key, raw in sorted(existing.items()):
        row = dict(raw)
        logical_key = _logical_key(row)
        prior = merged.get(logical_key)
        merged[logical_key] = _preferred(prior, row)
        changed |= stored_key != logical_key or prior is not None
    candidate = {
        "record_id": str(exact.get("record_id") or ""),
        "claim_scope_id": str(exact.get("claim_scope_id") or ""),
        "edge_id": str(edge_id),
        "edge_digest": str(edge.get("edge_digest") or ""),
        "proof_digest": str(proof.get("proof_digest") or ""),
        "source_ref": str(exact.get("source_ref") or ""),
        "independence_group": str(exact.get("independence_group") or ""),
        "location_refs": list(exact.get("location_refs") or []),
        "conditions": dict(exact.get("conditions") or {}),
        "condition_completeness": dict(exact.get("condition_completeness") or {}),
        "procedure_authority_scope": str(
            exact.get("procedure_authority_scope") or ""
        ),
        "product_smiles": str(edge.get("product_smiles") or ""),
        "precursor_smiles": list(edge.get("precursor_smiles") or []),
    }
    logical_key = _logical_key(candidate)
    prior = merged.get(logical_key)
    preferred = _preferred(prior, candidate)
    merged[logical_key] = preferred
    changed |= prior != preferred
    return merged, changed


def _logical_key(value: Mapping[str, Any]) -> str:
    locations = sorted(
        str(item)
        for item in value.get("location_refs") or []
        if str(item) and not str(item).startswith(("image_sha256:", "pdf_sha256:"))
    )
    identity = {
        "edge_digest": str(value.get("edge_digest") or ""),
        "independence_group": str(value.get("independence_group") or "").casefold(),
        "source_ref": str(value.get("source_ref") or "").casefold(),
        "locations": locations,
    }
    return "example:" + _digest(identity)[:24]


def _preferred(
    prior: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if prior is None:
        return dict(candidate)
    rows = [dict(prior), dict(candidate)]
    rows.sort(
        key=lambda row: (
            int(dict(row.get("condition_completeness") or {}).get("complete") is True),
            len(dict(row.get("conditions") or {})),
            len(row.get("location_refs") or []),
            _digest(row),
        ),
        reverse=True,
    )
    return rows[0]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["merge_template_example"]
