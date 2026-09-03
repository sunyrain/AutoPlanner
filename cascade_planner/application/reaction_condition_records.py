"""Canonical source-procedure records and deterministic completeness audits."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


SOURCE_PROCEDURE_RECORD_SCHEMA = "source_reaction_procedure_record.v1"
CONDITION_COMPLETENESS_SCHEMA = "reaction_condition_completeness.v1"

_CONDITION_KEYS = {
    "addition_order",
    "agitation",
    "agitation_program",
    "atmosphere",
    "base",
    "buffer",
    "catalyst",
    "concentration",
    "conversion_percent",
    "conversion_percent_range",
    "equivalents",
    "medium",
    "oxidant",
    "ph",
    "ph_program",
    "pressure",
    "purification",
    "reductant",
    "reagents",
    "scale",
    "solvent",
    "temperature",
    "temperature_c",
    "temperature_program",
    "time",
    "time_program",
    "workup",
    "yield",
    "yield_percent",
    "yield_percent_range",
}
_ALIASES = {
    "duration": "time",
    "reported_yield": "yield",
}
_REQUIRED_GROUPS = {
    "agents": {"reagents", "base", "catalyst", "oxidant", "reductant"},
    "solvent": {"solvent"},
    "temperature": {"temperature", "temperature_c"},
    "time": {"time"},
}


def normalize_source_conditions(value: Any) -> dict[str, Any]:
    """Preserve only operational condition fields using one canonical vocabulary."""

    source = dict(value) if isinstance(value, Mapping) else {}
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in source.items():
        if raw_value in (None, "", [], {}):
            continue
        key = _ALIASES.get(str(raw_key), str(raw_key))
        if key == "reagent":
            key = "reagents"
            raw_value = [raw_value] if isinstance(raw_value, str) else raw_value
        if key not in _CONDITION_KEYS:
            continue
        normalized[key] = _json_value(raw_value)
    return {key: normalized[key] for key in sorted(normalized)}


def audit_condition_completeness(conditions: Mapping[str, Any]) -> dict[str, Any]:
    """Report missing operational groups without inferring unreported values."""

    present = {
        str(key)
        for key, value in dict(conditions).items()
        if value not in (None, "", [], {})
    }
    missing = sorted(
        name
        for name, alternatives in _REQUIRED_GROUPS.items()
        if not (present & alternatives)
    )
    return {
        "schema_version": CONDITION_COMPLETENESS_SCHEMA,
        "complete": not missing,
        "missing_required_groups": missing,
        "present_fields": sorted(present),
        "yield_reported": bool(
            present & {"yield", "yield_percent", "yield_percent_range"}
        ),
        "workup_reported": "workup" in present,
        "purification_reported": "purification" in present,
    }


def build_source_procedure_record(
    *,
    exact_record: Mapping[str, Any],
    extraction_row: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    extraction_artifact_sha256: str,
) -> dict[str, Any] | None:
    """Create an authoritative procedure entity only from a hash-bound fragment."""

    evidence_refs = sorted(
        {str(value).strip() for value in extraction_row.get("evidence_refs") or [] if str(value).strip()}
    )
    procedure_sha256, digest_kind = procedure_text_digest(evidence_refs)
    location_refs = sorted(
        {str(value).strip() for value in exact_record.get("location_refs") or [] if str(value).strip()}
    )
    source_ref = str(exact_record.get("source_ref") or "")
    if not procedure_sha256 or not location_refs or not source_ref:
        return None
    conditions = normalize_source_conditions(
        extraction_row.get("condition_candidate")
        or extraction_row.get("conditions")
        or {}
    )
    completeness = audit_condition_completeness(conditions)
    identity = {
        "edge_digest": str(exact_record.get("edge_digest") or ""),
        "source_binding_id": str(exact_record.get("source_binding_id") or ""),
        "procedure_text_sha256": procedure_sha256,
        "location_refs": location_refs,
    }
    record = {
        "schema_version": SOURCE_PROCEDURE_RECORD_SCHEMA,
        "procedure_record_id": f"procedure:{_digest(identity)[:24]}",
        "exact_record_id": str(exact_record.get("record_id") or ""),
        **identity,
        "source_ref": source_ref,
        "independence_group": str(exact_record.get("independence_group") or ""),
        "conditions": conditions,
        "condition_completeness": completeness,
        "procedure_authority_scope": "source_exact_reaction_procedure",
        "procedure_status": (
            "condition_complete"
            if completeness["complete"]
            else "condition_partial"
            if conditions
            else "procedure_located_condition_unparsed"
        ),
        "source_fragment": {
            "digest_kind": digest_kind,
            "procedure_text_sha256": procedure_sha256,
            "source_artifact_sha256": _source_artifact_sha256(
                source_binding, evidence_refs
            ),
            "extraction_artifact_sha256": str(extraction_artifact_sha256),
            "evidence_refs": evidence_refs,
            "procedure_text_stored": False,
        },
        "semantics": {
            "procedure_is_distinct_from_structure_observation": True,
            "empty_conditions_do_not_invalidate_source_location": True,
            "missing_condition_fields_are_not_inferred": True,
        },
    }
    record["content_sha256"] = _digest(record)
    return record


def procedure_text_digest(evidence_refs: list[str]) -> tuple[str, str]:
    for kind in ("procedure-text-sha256", "text_sha256"):
        prefix = f"{kind}:"
        for value in evidence_refs:
            digest = value.removeprefix(prefix) if value.startswith(prefix) else ""
            if re.fullmatch(r"[0-9a-f]{64}", digest.lower()):
                return digest.lower(), kind
    return "", ""


def _source_artifact_sha256(
    source_binding: Mapping[str, Any], evidence_refs: list[str]
) -> str:
    supplied = str(source_binding.get("artifact_sha256") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", supplied):
        return supplied
    for kind in ("xml_sha256", "html_sha256", "pdf_sha256"):
        prefix = f"{kind}:"
        for value in evidence_refs:
            digest = value.removeprefix(prefix) if value.startswith(prefix) else ""
            if re.fullmatch(r"[0-9a-f]{64}", digest.lower()):
                return digest.lower()
    return ""


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


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
    "CONDITION_COMPLETENESS_SCHEMA",
    "SOURCE_PROCEDURE_RECORD_SCHEMA",
    "audit_condition_completeness",
    "build_source_procedure_record",
    "normalize_source_conditions",
    "procedure_text_digest",
]
