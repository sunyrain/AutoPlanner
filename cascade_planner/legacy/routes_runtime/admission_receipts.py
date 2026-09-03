"""Current-host provenance receipts for durable external route admissions.

The fused route graph is caller-controlled advisory state.  In particular,
``source_channels`` and serialized ``authority_bound`` flags are not
capabilities.  This module accepts complete source material, replays the
deterministic host adapter, and emits a deliberately narrow receipt that can
authorize *search admission only*.

The issued receipt contains no reaction proof, stock result, or solved state.
Those authorities remain with their dedicated current-host replay boundaries.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

from cascade_planner.legacy.routes_runtime.signatures import exact_edge_signature
from cascade_planner.harness.route_verifier import replay_route_proof_bank_entry
from cascade_planner.harness.stitched_route import (
    is_materialized_source_bound_literature_step,
    is_validated_source_detail_literature_step,
)
from cascade_planner.routes.domain import canonicalize_smiles


EXACT_LITERATURE_ADMISSION_MATERIAL_SCHEMA = (
    "external_hyperedge_exact_literature_material.v1"
)
MATERIALIZED_LITERATURE_SEARCH_ADMISSION_SCHEMA = (
    "materialized_literature_search_admission.v1"
)
CHEMENZY_ADMISSION_MATERIAL_SCHEMA = "external_hyperedge_chemenzy_material.v1"
EXTERNAL_HYPEREDGE_PROVENANCE_RECEIPT_SCHEMA = (
    "external_hyperedge_provenance_receipt.v1"
)

_EXACT_MATERIAL_KEYS = {
    "schema_version",
    "source_kind",
    "source_row",
    "content_sha256",
}
_CHEMENZY_MATERIAL_KEYS = {
    "schema_version",
    "source_kind",
    "route_bank",
    "source_entry_id",
    "source_step_index",
    "artifact_ref",
    "content_sha256",
}
_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "source_kind",
    "exact_edge_signature",
    "source_material_schema_version",
    "source_material_sha256",
    "source_binding",
    "semantics",
    "content_sha256",
}
_EXACT_BINDING_KEYS = {
    "schema_version",
    "source_ref",
    "source_template_id",
    "source_evidence_sha256",
}
_MATERIALIZED_LITERATURE_BINDING_KEYS = {
    "schema_version",
    "source_ref",
    "source_template_id",
    "document_ids",
    "page_numbers",
    "artifact_sha256",
    "source_evidence_sha256",
}
_CHEMENZY_BINDING_KEYS = {
    "schema_version",
    "source_bank_sha256",
    "source_entry_id",
    "source_entry_sha256",
    "source_step_index",
    "artifact_ref",
}
_RECEIPT_SEMANTICS = {
    "search_admission_only": True,
    "producer_authority_not_transferred": True,
    "dedicated_authority_replay_required": True,
}
_FORBIDDEN_RECEIPT_KEYS = {
    "accepted",
    "completion",
    "executable",
    "in_stock",
    "proof",
    "proof_level",
    "solved",
    "stock",
    "validated",
}


def make_exact_literature_admission_material(
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Return replayable material only for a host-validated exact source row."""

    row = _json_value(dict(source_row))
    if not is_validated_source_detail_literature_step(row):
        return {}
    product, precursors = _source_row_edge(row)
    if not exact_edge_signature(product, precursors):
        return {}
    material = {
        "schema_version": EXACT_LITERATURE_ADMISSION_MATERIAL_SCHEMA,
        "source_kind": "validated_exact_literature_adapter",
        "source_row": row,
    }
    material["content_sha256"] = _digest(material)
    return material


def make_materialized_literature_search_admission_material(
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Return L0-only material for a host-replayed source-bound claim."""

    row = _json_value(dict(source_row))
    if not is_materialized_source_bound_literature_step(row):
        return {}
    product, precursors = _source_row_edge(row)
    if not exact_edge_signature(product, precursors):
        return {}
    material = {
        "schema_version": MATERIALIZED_LITERATURE_SEARCH_ADMISSION_SCHEMA,
        "source_kind": "materialized_literature_search_admission",
        "source_row": row,
    }
    material["content_sha256"] = _digest(material)
    return material


def make_chemenzy_admission_material(
    route_bank: Mapping[str, Any],
    *,
    source_entry_id: str,
    source_step_index: int,
    artifact_ref: str = "",
) -> dict[str, Any]:
    """Return replayable material for one current-host ChemEnzy bank step."""

    material = {
        "schema_version": CHEMENZY_ADMISSION_MATERIAL_SCHEMA,
        "source_kind": "current_host_replayed_chemenzy_bank",
        "route_bank": _json_value(dict(route_bank)),
        "source_entry_id": str(source_entry_id or ""),
        "source_step_index": int(source_step_index),
        "artifact_ref": str(artifact_ref or ""),
    }
    material["content_sha256"] = _digest(material)
    receipt, _ = issue_external_hyperedge_provenance_receipt(material)
    return material if receipt else {}


def issue_external_hyperedge_provenance_receipt(
    material: Mapping[str, Any],
    *,
    expected_product_smiles: str = "",
    expected_precursor_smiles: Iterable[Any] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Replay source material and issue a strict, authority-free receipt."""

    row = _json_value(dict(material)) if isinstance(material, Mapping) else {}
    schema = str(row.get("schema_version") or "")
    if schema == EXACT_LITERATURE_ADMISSION_MATERIAL_SCHEMA:
        source_kind = "validated_exact_literature_adapter"
        product, precursors, binding, reasons = _replay_exact_material(row)
    elif schema == MATERIALIZED_LITERATURE_SEARCH_ADMISSION_SCHEMA:
        source_kind = "materialized_literature_search_admission"
        product, precursors, binding, reasons = (
            _replay_materialized_literature_search_material(row)
        )
    elif schema == CHEMENZY_ADMISSION_MATERIAL_SCHEMA:
        source_kind = "current_host_replayed_chemenzy_bank"
        product, precursors, binding, reasons = _replay_chemenzy_material(row)
    else:
        return {}, ["admission_material_schema_not_allowlisted"]

    expected_signature = exact_edge_signature(
        expected_product_smiles,
        expected_precursor_smiles,
    )
    actual_signature = exact_edge_signature(product, precursors)
    if not actual_signature:
        reasons.append("admission_material_exact_edge_invalid")
    if expected_signature and actual_signature != expected_signature:
        reasons.append("admission_material_exact_edge_mismatch")
    if reasons:
        return {}, sorted(set(reasons))

    identity_payload = {
        "schema_version": EXTERNAL_HYPEREDGE_PROVENANCE_RECEIPT_SCHEMA,
        "source_kind": source_kind,
        "exact_edge_signature": actual_signature,
        "source_material_schema_version": schema,
        "source_material_sha256": str(row.get("content_sha256") or ""),
        "source_binding": binding,
        "semantics": dict(_RECEIPT_SEMANTICS),
    }
    identity = _digest(identity_payload)
    receipt = {
        **identity_payload,
        "receipt_id": f"external-admission-receipt:sha256:{identity}",
    }
    receipt["content_sha256"] = _digest(receipt)
    validation_reasons = validate_external_hyperedge_provenance_receipt(
        receipt,
        expected_exact_edge_signature=actual_signature,
    )
    return ({}, validation_reasons) if validation_reasons else (receipt, [])


def validate_external_hyperedge_provenance_receipt(
    value: Mapping[str, Any],
    *,
    expected_exact_edge_signature: str = "",
) -> list[str]:
    """Validate an issued receipt without granting any downstream authority."""

    receipt = _json_value(dict(value)) if isinstance(value, Mapping) else {}
    reasons: list[str] = []
    if set(receipt) != _RECEIPT_KEYS:
        reasons.append("provenance_receipt_fields_invalid")
    digest_payload = dict(receipt)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    if not _valid_sha256(recorded_digest) or recorded_digest != _digest(
        digest_payload
    ):
        reasons.append("provenance_receipt_content_digest_invalid")
    if receipt.get("schema_version") != (
        EXTERNAL_HYPEREDGE_PROVENANCE_RECEIPT_SCHEMA
    ):
        reasons.append("provenance_receipt_schema_invalid")
    source_kind = str(receipt.get("source_kind") or "")
    source_schema = str(receipt.get("source_material_schema_version") or "")
    binding = receipt.get("source_binding")
    binding_row = dict(binding) if isinstance(binding, Mapping) else {}
    if not isinstance(binding, Mapping):
        reasons.append("provenance_receipt_binding_not_object")
    if source_kind == "materialized_literature_search_admission":
        if source_schema != MATERIALIZED_LITERATURE_SEARCH_ADMISSION_SCHEMA:
            reasons.append("provenance_receipt_source_schema_mismatch")
        document_ids = binding_row.get("document_ids")
        page_numbers = binding_row.get("page_numbers")
        artifact_sha256 = binding_row.get("artifact_sha256")
        if (
            set(binding_row) != _MATERIALIZED_LITERATURE_BINDING_KEYS
            or binding_row.get("schema_version")
            != "materialized_literature_search_binding.v1"
            or not str(binding_row.get("source_ref") or "").strip()
            or not str(binding_row.get("source_template_id") or "").startswith(
                "source_detail_exact_step:"
            )
            or not isinstance(document_ids, list)
            or document_ids != sorted(set(document_ids))
            or not document_ids
            or not all(isinstance(item, str) and item for item in document_ids)
            or not isinstance(page_numbers, list)
            or page_numbers != sorted(set(page_numbers))
            or not page_numbers
            or not all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in page_numbers
            )
            or not isinstance(artifact_sha256, list)
            or artifact_sha256 != sorted(set(artifact_sha256))
            or len(artifact_sha256) < 3
            or not all(_valid_sha256(item) for item in artifact_sha256)
            or not _valid_sha256(binding_row.get("source_evidence_sha256"))
        ):
            reasons.append(
                "provenance_receipt_materialized_literature_binding_invalid"
            )
    elif source_kind == "validated_exact_literature_adapter":
        if source_schema != EXACT_LITERATURE_ADMISSION_MATERIAL_SCHEMA:
            reasons.append("provenance_receipt_source_schema_mismatch")
        if (
            set(binding_row) != _EXACT_BINDING_KEYS
            or binding_row.get("schema_version")
            != "exact_literature_source_binding.v1"
            or not str(binding_row.get("source_ref") or "").strip()
            or not str(binding_row.get("source_template_id") or "").startswith(
                "source_detail_exact_step:"
            )
            or not _valid_sha256(binding_row.get("source_evidence_sha256"))
        ):
            reasons.append("provenance_receipt_exact_binding_invalid")
    elif source_kind == "current_host_replayed_chemenzy_bank":
        if source_schema != CHEMENZY_ADMISSION_MATERIAL_SCHEMA:
            reasons.append("provenance_receipt_source_schema_mismatch")
        step_index = binding_row.get("source_step_index")
        if (
            set(binding_row) != _CHEMENZY_BINDING_KEYS
            or binding_row.get("schema_version") != "chemenzy_source_binding.v1"
            or not _valid_sha256(binding_row.get("source_bank_sha256"))
            or not str(binding_row.get("source_entry_id") or "")
            or not _valid_sha256(binding_row.get("source_entry_sha256"))
            or not isinstance(step_index, int)
            or isinstance(step_index, bool)
            or step_index < 0
        ):
            reasons.append("provenance_receipt_chemenzy_binding_invalid")
    else:
        reasons.append("provenance_receipt_source_kind_not_allowlisted")
    edge_signature = str(receipt.get("exact_edge_signature") or "")
    if not edge_signature.startswith("edge:sha256:"):
        reasons.append("provenance_receipt_exact_edge_signature_invalid")
    if expected_exact_edge_signature and edge_signature != str(
        expected_exact_edge_signature
    ):
        reasons.append("provenance_receipt_exact_edge_signature_mismatch")
    if not _valid_sha256(receipt.get("source_material_sha256")):
        reasons.append("provenance_receipt_material_digest_invalid")
    if receipt.get("semantics") != _RECEIPT_SEMANTICS:
        reasons.append("provenance_receipt_semantics_invalid")
    if _contains_forbidden_authority_key(receipt):
        reasons.append("provenance_receipt_carries_authority_field")
    identity_payload = {
        key: receipt.get(key)
        for key in (
            "schema_version",
            "source_kind",
            "exact_edge_signature",
            "source_material_schema_version",
            "source_material_sha256",
            "source_binding",
            "semantics",
        )
    }
    identity = _digest(identity_payload)
    if receipt.get("receipt_id") != (
        f"external-admission-receipt:sha256:{identity}"
    ):
        reasons.append("provenance_receipt_identity_invalid")
    return sorted(set(reasons))


def _replay_exact_material(
    material: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any], list[str]]:
    reasons: list[str] = []
    payload = dict(material)
    recorded_digest = str(payload.pop("content_sha256", ""))
    if set(material) != _EXACT_MATERIAL_KEYS:
        reasons.append("exact_literature_material_fields_invalid")
    if (
        material.get("source_kind") != "validated_exact_literature_adapter"
        or not _valid_sha256(recorded_digest)
        or recorded_digest != _digest(payload)
    ):
        reasons.append("exact_literature_material_digest_or_kind_invalid")
    raw_source = material.get("source_row")
    source = dict(raw_source) if isinstance(raw_source, Mapping) else {}
    if not isinstance(raw_source, Mapping):
        reasons.append("exact_literature_source_row_not_object")
    elif not is_validated_source_detail_literature_step(source):
        reasons.append("exact_literature_source_row_host_replay_failed")
    product, precursors = _source_row_edge(source)
    evidence = [
        dict(row)
        for row in source.get("source_evidence") or []
        if isinstance(row, Mapping)
    ]
    binding = {
        "schema_version": "exact_literature_source_binding.v1",
        "source_ref": str(source.get("source_ref") or ""),
        "source_template_id": str(source.get("source_template_id") or ""),
        "source_evidence_sha256": _digest(evidence),
    }
    return product, precursors, binding, reasons


def _replay_materialized_literature_search_material(
    material: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any], list[str]]:
    reasons: list[str] = []
    payload = dict(material)
    recorded_digest = str(payload.pop("content_sha256", ""))
    if set(material) != _EXACT_MATERIAL_KEYS:
        reasons.append("materialized_literature_material_fields_invalid")
    if (
        material.get("source_kind")
        != "materialized_literature_search_admission"
        or not _valid_sha256(recorded_digest)
        or recorded_digest != _digest(payload)
    ):
        reasons.append("materialized_literature_material_digest_or_kind_invalid")
    raw_source = material.get("source_row")
    source = dict(raw_source) if isinstance(raw_source, Mapping) else {}
    if not isinstance(raw_source, Mapping):
        reasons.append("materialized_literature_source_row_not_object")
    elif not is_materialized_source_bound_literature_step(source):
        reasons.append("materialized_literature_source_binding_replay_failed")
    product, precursors = _source_row_edge(source)
    evidence = [
        dict(row)
        for row in source.get("source_evidence") or []
        if isinstance(row, Mapping)
    ]
    document_ids = sorted(
        {
            str(row.get("document_id") or "")
            for row in evidence
            if str(row.get("document_id") or "")
        }
    )
    page_numbers = sorted(
        {
            int(row.get("page_number") or 0)
            for row in evidence
            if int(row.get("page_number") or 0) > 0
        }
    )
    artifact_sha256 = sorted(
        {
            str(row.get(field) or "").lower()
            for row in evidence
            for field in ("manifest_sha256", "source_pdf_sha256", "image_sha256")
            if _valid_sha256(row.get(field))
        }
    )
    binding = {
        "schema_version": "materialized_literature_search_binding.v1",
        "source_ref": str(source.get("source_ref") or ""),
        "source_template_id": str(source.get("source_template_id") or ""),
        "document_ids": document_ids,
        "page_numbers": page_numbers,
        "artifact_sha256": artifact_sha256,
        "source_evidence_sha256": _digest(evidence),
    }
    return product, precursors, binding, reasons


def _replay_chemenzy_material(
    material: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any], list[str]]:
    reasons: list[str] = []
    payload = dict(material)
    recorded_digest = str(payload.pop("content_sha256", ""))
    if set(material) != _CHEMENZY_MATERIAL_KEYS:
        reasons.append("chemenzy_material_fields_invalid")
    if (
        material.get("source_kind") != "current_host_replayed_chemenzy_bank"
        or not _valid_sha256(recorded_digest)
        or recorded_digest != _digest(payload)
    ):
        reasons.append("chemenzy_material_digest_or_kind_invalid")
    raw_bank = material.get("route_bank")
    bank = dict(raw_bank) if isinstance(raw_bank, Mapping) else {}
    entry_id = str(material.get("source_entry_id") or "")
    step_index = material.get("source_step_index")
    if not isinstance(raw_bank, Mapping):
        reasons.append("chemenzy_source_bank_not_object")
    if (
        not isinstance(step_index, int)
        or isinstance(step_index, bool)
        or step_index < 0
    ):
        reasons.append("chemenzy_source_step_index_invalid")
        step_index = -1
    replay = replay_route_proof_bank_entry(
        bank,
        proof_id=entry_id,
        expected_target_smiles=canonicalize_smiles(bank.get("target_smiles")),
    )
    if replay.get("accepted") is not True:
        reasons.extend(
            f"chemenzy_source_bank_replay:{reason}"
            for reason in replay.get("reasons") or ["rejected"]
        )
    entries = [
        dict(row)
        for row in bank.get("entries") or []
        if isinstance(row, Mapping) and str(row.get("proof_id") or "") == entry_id
    ]
    if len(entries) != 1:
        reasons.append("chemenzy_source_entry_not_unique")
        entry: dict[str, Any] = {}
    else:
        entry = entries[0]
    steps = [
        dict(row)
        for row in (entry.get("materialized_route") or {}).get("steps") or []
        if isinstance(row, Mapping)
    ]
    if step_index < 0 or step_index >= len(steps):
        reasons.append("chemenzy_source_step_missing")
        step: dict[str, Any] = {}
    else:
        step = steps[step_index]
    product, precursors = _materialized_step_edge(step)
    binding = {
        "schema_version": "chemenzy_source_binding.v1",
        "source_bank_sha256": str(bank.get("content_hash") or ""),
        "source_entry_id": entry_id,
        "source_entry_sha256": str(entry.get("content_hash") or ""),
        "source_step_index": max(0, int(step_index)),
        "artifact_ref": str(material.get("artifact_ref") or ""),
    }
    return product, precursors, binding, reasons


def _source_row_edge(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    reaction = str(row.get("reaction_smiles") or "").strip()
    if ">>" in reaction:
        left, right = reaction.split(">>", 1)
        product = canonicalize_smiles(right)
        precursors = [
            canonicalize_smiles(value)
            for value in left.split(".")
            if str(value or "").strip()
        ]
    else:
        raw_product = (
            row.get("product_smiles")
            or row.get("products")
            or row.get("product")
        )
        if isinstance(raw_product, list):
            raw_product = raw_product[0] if raw_product else ""
        raw_precursors = (
            row.get("reactant_smiles")
            or row.get("precursor_smiles")
            or row.get("reactants")
            or row.get("main_reactant_smiles")
            or []
        )
        values = (
            list(raw_precursors)
            if isinstance(raw_precursors, (list, tuple))
            else [raw_precursors]
        )
        product = canonicalize_smiles(raw_product)
        precursors = [canonicalize_smiles(value) for value in values]
    return product, sorted(value for value in precursors if value)


def _materialized_step_edge(step: Mapping[str, Any]) -> tuple[str, list[str]]:
    product = canonicalize_smiles(
        step.get("product_smiles") or step.get("product")
    )
    raw = (
        step.get("reactant_smiles")
        or step.get("precursor_smiles")
        or step.get("reactants")
        or []
    )
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    precursors = sorted(
        value for value in (canonicalize_smiles(item) for item in values) if value
    )
    return product, precursors


def _contains_forbidden_authority_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key or "").strip().lower()
            if normalized in _FORBIDDEN_RECEIPT_KEYS:
                return True
            if _contains_forbidden_authority_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_authority_key(item) for item in value)
    return False


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


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


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
