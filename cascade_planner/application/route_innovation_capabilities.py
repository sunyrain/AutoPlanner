"""Data-driven capability matching for route innovation discovery.

Capability records describe a supported *class* of net transformations.  They
are evidence-backed search priors, never reaction or enzyme validation.  The
matcher contains generic structure arithmetic only; target names and route
labels are intentionally absent.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.route_structure_matching import (
    match_structure_capability,
    normalize_structure_match,
    structure_match_input_valid,
    structure_transition,
)


BIOCATALYSIS_CAPABILITY_SCHEMA = "biocatalysis_capability.v1"
BIOCATALYSIS_CAPABILITY_CATALOG_SCHEMA = "biocatalysis_capability_catalog.v1"


def normalize_biocatalysis_capability(
    value: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    raw = dict(value or {})
    capability_id = str(raw.get("capability_id") or "").strip()
    enzyme = dict(raw.get("enzyme") or {})
    match = normalize_structure_match(raw.get("match"))
    motif_delta = dict(match["net_motif_delta"])
    refs = _strings(raw.get("precedent_refs") or [])
    reasons: list[str] = []
    if not structure_match_input_valid(raw.get("match")):
        reasons.append("biocatalysis_capability_structure_match_invalid")
    if not capability_id:
        reasons.append("biocatalysis_capability_id_missing")
    if not motif_delta:
        reasons.append("biocatalysis_capability_net_transform_missing")
    if not (
        _strings(enzyme.get("classes") or [])
        or _strings(enzyme.get("ec_numbers") or [])
        or _strings(enzyme.get("candidate_ids") or [])
    ):
        reasons.append("biocatalysis_capability_enzyme_missing")
    if not refs:
        reasons.append("biocatalysis_capability_precedent_missing")
    objective = str(raw.get("selectivity_objective") or "").strip()
    if not objective:
        reasons.append("biocatalysis_capability_selectivity_objective_missing")
    if reasons:
        return {}, sorted(set(reasons))
    record = {
        "schema_version": BIOCATALYSIS_CAPABILITY_SCHEMA,
        "capability_id": capability_id,
        "label": str(raw.get("label") or capability_id),
        "enzyme": {
            "classes": _strings(enzyme.get("classes") or []),
            "ec_numbers": _strings(enzyme.get("ec_numbers") or []),
            "candidate_ids": _strings(enzyme.get("candidate_ids") or []),
        },
        "match": match,
        "selectivity_objective": objective,
        "substrate_scope_basis": str(raw.get("substrate_scope_basis") or "analogy only"),
        "cofactor_requirements": dict(raw.get("cofactor_requirements") or {}),
        "cofactor_regenerations": dict(raw.get("cofactor_regenerations") or {}),
        "precedent_refs": refs,
        "authority_scope": "search_prior_only",
        "not_reaction_proof": True,
        "exact_substrate_validated": False,
    }
    record["content_sha256"] = _digest(record)
    return record, []


def normalize_biocatalysis_catalog(
    value: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = (
        list(value.get("capabilities") or [])
        if isinstance(value, Mapping)
        else list(value)
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in rows:
        record, reasons = normalize_biocatalysis_capability(raw)
        if reasons:
            rejected.append(
                {
                    "capability_id": str(dict(raw).get("capability_id") or ""),
                    "reasons": reasons,
                }
            )
        elif record:
            accepted.append(record)
    accepted.sort(key=lambda row: row["capability_id"])
    return accepted, rejected


def match_biocatalysis_capability(
    capability: Mapping[str, Any],
    precursor_smiles: str,
    product_smiles: str,
    *,
    window_steps: int,
) -> dict[str, Any]:
    """Compatibility name for the execution-domain-neutral structure matcher."""

    return match_structure_capability(
        capability,
        precursor_smiles,
        product_smiles,
        window_steps=window_steps,
    )


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return sorted({str(item).strip() for item in value or [] if str(item).strip()})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "BIOCATALYSIS_CAPABILITY_CATALOG_SCHEMA",
    "BIOCATALYSIS_CAPABILITY_SCHEMA",
    "match_biocatalysis_capability",
    "match_structure_capability",
    "normalize_biocatalysis_capability",
    "normalize_biocatalysis_catalog",
    "normalize_structure_match",
    "structure_transition",
]
