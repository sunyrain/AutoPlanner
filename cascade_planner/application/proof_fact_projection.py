"""Lifecycle-aware canonical fact projection for edge proof stitching."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cascade_planner.application.fact_lifecycle import graph_fact_lifecycle_state
from cascade_planner.application.reaction_proof_versions import active_reaction_proofs


def collect_edge_authority_facts(
    graph: Mapping[str, Any], edge: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    inactive: dict[tuple[str, str], dict[str, Any]] = {}
    reaction_proofs: list[dict[str, Any]] = []
    exact_records: list[dict[str, Any]] = []
    procedure_records: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    source_groups: set[str] = set()

    for value in active_reaction_proofs(edge.get("reaction_proofs") or []):
        if not isinstance(value, Mapping) or not _valid_reaction_proof(value):
            continue
        proof = dict(value)
        proof_id = str(proof.get("proof_digest") or "")
        if _active(
            graph,
            "reaction_proof",
            proof_id,
            proof,
            inactive=inactive,
            reasons=reasons,
        ):
            reaction_proofs.append(proof)

    aliases = dict(graph.get("source_aliases") or {})
    sources = dict(graph.get("source_bindings") or {})
    for record_id in edge.get("exact_record_ids") or []:
        record = dict(dict(graph.get("exact_records") or {}).get(record_id) or {})
        if not record or not _valid_content_digest(record):
            reasons.append(f"exact_record_invalid:{record_id}")
            continue
        if str(record.get("edge_digest") or "") != str(edge.get("edge_digest") or ""):
            reasons.append(f"exact_record_edge_mismatch:{record_id}")
            continue
        if not _active(
            graph,
            "exact_record",
            str(record_id),
            record,
            inactive=inactive,
            reasons=reasons,
        ):
            continue
        external_source_id = str(record.get("source_binding_id") or "")
        source_id = str(aliases.get(external_source_id) or "")
        source = dict(sources.get(source_id) or {})
        if not source or not _valid_content_digest(source):
            reasons.append(f"exact_record_source_binding_invalid:{record_id}")
            continue
        if not _active(
            graph,
            "source_binding",
            source_id,
            source,
            inactive=inactive,
            reasons=reasons,
        ):
            continue
        exact_records.append(record)
        source_ids.add(source_id)
        group = str(source.get("independence_group") or record.get("independence_group") or "")
        if group and group != "codex_model":
            source_groups.add(group)

    exact_ids = {str(value.get("record_id") or "") for value in exact_records}
    for record_id in edge.get("procedure_record_ids") or []:
        record = dict(dict(graph.get("procedure_records") or {}).get(record_id) or {})
        if not record or not _valid_content_digest(record):
            reasons.append(f"procedure_record_invalid:{record_id}")
            continue
        if str(record.get("edge_digest") or "") != str(edge.get("edge_digest") or ""):
            reasons.append(f"procedure_record_edge_mismatch:{record_id}")
            continue
        if not _active(
            graph,
            "procedure_record",
            str(record_id),
            record,
            inactive=inactive,
            reasons=reasons,
        ):
            continue
        if str(record.get("exact_record_id") or "") not in exact_ids:
            reasons.append(f"procedure_record_exact_binding_invalid:{record_id}")
            continue
        source_id = str(aliases.get(str(record.get("source_binding_id") or "")) or "")
        source = dict(sources.get(source_id) or {})
        if source and not _active(
            graph,
            "source_binding",
            source_id,
            source,
            inactive=inactive,
            reasons=reasons,
        ):
            continue
        procedure_records.append(record)

    return {
        "reaction_proofs": reaction_proofs,
        "exact_records": exact_records,
        "procedure_records": procedure_records,
        "source_binding_ids": sorted(source_ids),
        "independent_source_groups": sorted(source_groups),
        "inactive_facts": [inactive[key] for key in sorted(inactive)],
        "reasons": sorted(set(reasons)),
    }


def lifecycle_impact(
    graph: Mapping[str, Any],
    subject_kind: str,
    subject_id: str,
    subject: Mapping[str, Any],
) -> dict[str, Any]:
    state = graph_fact_lifecycle_state(graph, subject_kind, subject_id, subject)
    if state.get("active") is True:
        return {}
    return {
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "status": str(state.get("status") or "inactive"),
        "lifecycle_event_id": str(state.get("latest_event_id") or ""),
        "effective_at": str(state.get("effective_at") or ""),
        "reason_codes": list(state.get("reason_codes") or []),
        "authority_scope": str(state.get("authority_scope") or ""),
    }


def _active(
    graph: Mapping[str, Any],
    kind: str,
    subject_id: str,
    subject: Mapping[str, Any],
    *,
    inactive: dict[tuple[str, str], dict[str, Any]],
    reasons: list[str],
) -> bool:
    impact = lifecycle_impact(graph, kind, subject_id, subject)
    if not impact:
        return True
    reasons.append(f"{kind}_{impact['status']}:{subject_id}")
    inactive[(kind, subject_id)] = impact
    return False


def _valid_reaction_proof(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("proof_digest", ""))
    return bool(
        supplied
        and supplied == _digest(row)
        and row.get("schema_version") == "reaction_step_proof.v1"
    )


def _valid_content_digest(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return bool(supplied and supplied == _digest(row))


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


__all__ = ["collect_edge_authority_facts", "lifecycle_impact"]
