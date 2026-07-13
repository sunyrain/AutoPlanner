"""Version lifecycle for deterministic host reaction proofs."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


CURRENT_REACTION_VALIDATOR_VERSION = "autoplanner.reaction_step_verifier.v7"
REACTION_PROOF_VERSION_AUDIT_SCHEMA = "reaction_proof_version_audit.v1"


def active_reaction_proofs(values: Any) -> list[dict[str, Any]]:
    """Return legacy proofs or proofs produced by the current host verifier."""

    rows = [dict(value) for value in values or [] if isinstance(value, Mapping)]
    versioned = [row for row in rows if str(row.get("validator_version") or "")]
    if not versioned:
        return rows
    return [
        row
        for row in versioned
        if row.get("validator_version") == CURRENT_REACTION_VALIDATOR_VERSION
    ]


def compile_reaction_proof_version_audit(
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose stale verifier output without treating it as active science."""

    current_count = 0
    stale_count = 0
    legacy_count = 0
    stale_only_edge_ids: list[str] = []
    for edge_id, raw_edge in dict(graph.get("edges") or {}).items():
        edge = dict(raw_edge) if isinstance(raw_edge, Mapping) else {}
        proofs = [
            dict(value)
            for value in edge.get("reaction_proofs") or []
            if isinstance(value, Mapping)
        ]
        current = [
            value
            for value in proofs
            if value.get("validator_version") == CURRENT_REACTION_VALIDATOR_VERSION
        ]
        stale = [
            value
            for value in proofs
            if str(value.get("validator_version") or "")
            and value.get("validator_version") != CURRENT_REACTION_VALIDATOR_VERSION
        ]
        legacy = [value for value in proofs if not value.get("validator_version")]
        current_count += len(current)
        stale_count += len(stale)
        legacy_count += len(legacy)
        if stale and not current:
            stale_only_edge_ids.append(str(edge_id))
    row = {
        "schema_version": REACTION_PROOF_VERSION_AUDIT_SCHEMA,
        "current_validator_version": CURRENT_REACTION_VALIDATOR_VERSION,
        "current_proof_count": current_count,
        "stale_versioned_proof_count": stale_count,
        "legacy_unversioned_proof_count": legacy_count,
        "stale_only_edge_ids": sorted(stale_only_edge_ids),
        "requires_revalidation": bool(stale_only_edge_ids),
        "semantics": {
            "stale_proof_never_grants_current_validation": True,
            "audit_does_not_mutate_terminal_runs": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


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


__all__ = [
    "CURRENT_REACTION_VALIDATOR_VERSION",
    "REACTION_PROOF_VERSION_AUDIT_SCHEMA",
    "active_reaction_proofs",
    "compile_reaction_proof_version_audit",
]
