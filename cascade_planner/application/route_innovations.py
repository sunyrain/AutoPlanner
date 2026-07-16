"""Typed innovation records for non-literature route improvement.

The canonical reaction hypergraph identifies chemistry by product and complete
precursor multiset.  Execution ideas are deliberately orthogonal annotations:
the same connectivity can have chemical and enzymatic execution options, while
a biocatalytic superstep can also span the boundary of several reported
chemical operations.

These records are proposals.  They never grant reaction or source authority.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


ROUTE_INNOVATION_SCHEMA = "route_innovation.v1"
BIOCATALYTIC_STEP = "biocatalytic_step"
BIOCATALYTIC_SUPERSTEP = "biocatalytic_superstep"
MECHANISM_EXTRAPOLATION = "mechanism_extrapolation"
BIOCATALYTIC_KINDS = {BIOCATALYTIC_STEP, BIOCATALYTIC_SUPERSTEP}
INNOVATION_KINDS = {*BIOCATALYTIC_KINDS, MECHANISM_EXTRAPOLATION}


def normalize_route_innovation(
    value: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize one proposal-only innovation record and return reject reasons."""

    raw = dict(value or {})
    if not raw:
        return {}, []
    nested = raw.get("route_innovation") or raw.get("innovation")
    row = (
        {
            **{
                key: raw[key]
                for key in ("route_family_id", "route_family_ids")
                if key in raw
            },
            **dict(nested),
        }
        if isinstance(nested, Mapping)
        else raw
    )
    kind = str(
        row.get("kind")
        or row.get("innovation_kind")
        or row.get("proposal_basis")
        or ""
    ).strip().lower()
    if kind in {"enzyme", "enzymatic", "biocatalysis"}:
        kind = BIOCATALYTIC_STEP
    if kind == "enzyme_superstep":
        kind = BIOCATALYTIC_SUPERSTEP
    if kind in {"mechanism", "mechanistic", "mechanistic_extrapolation"}:
        kind = MECHANISM_EXTRAPOLATION
    if not kind:
        return {}, []
    if kind not in INNOVATION_KINDS:
        return {}, ["route_innovation_kind_invalid"]

    if kind in BIOCATALYTIC_KINDS:
        record, reasons = _biocatalytic_record(
            row,
            require_multiple=kind == BIOCATALYTIC_SUPERSTEP,
        )
    else:
        record, reasons = _mechanism_record(row)
    if reasons:
        return {}, sorted(set(reasons))
    record = {
        "schema_version": ROUTE_INNOVATION_SCHEMA,
        "kind": kind,
        **record,
        "authority_scope": "proposal_only",
        "not_reaction_proof": True,
        "not_exact_source_evidence": True,
    }
    identity = {key: value for key, value in record.items() if key != "innovation_id"}
    record["innovation_id"] = f"innovation:{_digest(identity)[:24]}"
    record["content_sha256"] = _digest(record)
    return record, []


def merge_route_innovations(
    existing: Any,
    incoming: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge valid proposal options without converting them into proof."""

    rows: dict[str, dict[str, Any]] = {}
    for raw in [*(existing or []), *incoming]:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        innovation_id = str(row.get("innovation_id") or "")
        if not innovation_id:
            normalized, reasons = normalize_route_innovation(row)
            if reasons or not normalized:
                continue
            row = normalized
            innovation_id = str(row["innovation_id"])
        rows[innovation_id] = row
    return [rows[key] for key in sorted(rows)]


def innovation_proof_gate(
    innovations: Iterable[Mapping[str, Any]],
    reaction_proofs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the additional L2 gate required by selected execution ideas.

    A conventional host reaction proof is sufficient for a one-hop mechanism
    proposal.  An enzyme-labelled edge additionally needs a digest-bound
    biocatalysis validation; atom mapping or a generic EC label is not enough.
    """

    options = [dict(value) for value in innovations if isinstance(value, Mapping)]
    enzyme_options = [
        value for value in options if value.get("kind") in BIOCATALYTIC_KINDS
    ]
    if not enzyme_options:
        return {
            "required": False,
            "accepted": True,
            "reasons": [],
            "validated_innovation_ids": [],
            "generic_validation": False,
        }
    validated_ids: set[str] = set()
    generic_validation = False
    for raw_proof in reaction_proofs:
        if not isinstance(raw_proof, Mapping) or raw_proof.get("accepted") is not True:
            continue
        validation = raw_proof.get("biocatalysis_validation")
        if not isinstance(validation, Mapping) or validation.get("accepted") is not True:
            continue
        innovation_id = str(validation.get("innovation_id") or "")
        if innovation_id:
            validated_ids.add(innovation_id)
        else:
            generic_validation = True
    accepted = generic_validation or any(
        str(value.get("innovation_id") or "") in validated_ids
        for value in enzyme_options
    )
    return {
        "required": True,
        "accepted": accepted,
        "reasons": [] if accepted else ["biocatalysis_validation_missing"],
        "validated_innovation_ids": sorted(validated_ids),
        "generic_validation": generic_validation,
    }


def route_innovation_summary(
    graph: Mapping[str, Any],
    edge_ids: Iterable[str],
    *,
    route_family_id: str = "",
) -> dict[str, Any]:
    """Summarize route step compression and low-evidence extrapolations."""

    route_edge_ids = [str(value) for value in edge_ids]
    selected: list[dict[str, Any]] = []
    for edge_id in route_edge_ids:
        edge = dict(dict(graph.get("edges") or {}).get(str(edge_id)) or {})
        options = [
            dict(value)
            for value in edge.get("route_innovations") or []
            if isinstance(value, Mapping)
        ]
        matching = [
            value
            for value in options
            if not route_family_id
            or not value.get("route_family_ids")
            or route_family_id in value.get("route_family_ids", [])
        ]
        # A route family can propose several execution options for one topology.
        # Prefer the greatest explicit compression, deterministically.
        matching.sort(
            key=lambda value: (
                -int(value.get("chemical_step_equivalent_count") or 1),
                str(value.get("innovation_id") or ""),
            )
        )
        if matching:
            selected.append({"edge_id": str(edge_id), **matching[0]})
    enzyme = [value for value in selected if value.get("kind") in BIOCATALYTIC_KINDS]
    supersteps = [
        value for value in enzyme if value.get("kind") == BIOCATALYTIC_SUPERSTEP
    ]
    mechanism = [
        value for value in selected if value.get("kind") == MECHANISM_EXTRAPOLATION
    ]
    equivalents = sum(
        max(1, int(value.get("chemical_step_equivalent_count") or 1))
        for value in enzyme
    ) + max(0, len(route_edge_ids) - len(enzyme))
    savings = sum(max(0, int(value.get("step_savings") or 0)) for value in enzyme)
    return {
        "schema_version": "route_innovation_summary.v1",
        "execution_option_ids": sorted(
            str(value.get("innovation_id") or "") for value in selected
        ),
        "biocatalytic_edge_ids": sorted(value["edge_id"] for value in enzyme),
        "mechanism_extrapolation_edge_ids": sorted(
            value["edge_id"] for value in mechanism
        ),
        "biocatalytic_step_count": len(enzyme),
        "biocatalytic_superstep_count": len(supersteps),
        "mechanism_extrapolation_count": len(mechanism),
        "chemical_step_equivalent_count": equivalents,
        "net_step_savings": savings,
        "selected_options": selected,
        "semantics": {
            "physical_edges_and_chemical_equivalents_are_distinct": True,
            "proposal_options_do_not_grant_reaction_proof": True,
            "mechanism_extrapolation_is_not_source_reported": True,
        },
    }


def _biocatalytic_record(
    row: Mapping[str, Any],
    *,
    require_multiple: bool,
) -> tuple[dict[str, Any], list[str]]:
    enzyme = dict(row.get("enzyme") or {}) if isinstance(row.get("enzyme"), Mapping) else {}
    ec_numbers = _strings(
        row.get("ec_numbers")
        or row.get("enzyme_ec_numbers")
        or enzyme.get("ec_numbers")
        or ([row.get("ec_number")] if row.get("ec_number") else [])
    )
    enzyme_classes = _strings(
        row.get("enzyme_classes")
        or enzyme.get("classes")
        or ([row.get("enzyme_class")] if row.get("enzyme_class") else [])
    )
    candidate_ids = _strings(
        row.get("enzyme_candidate_ids") or enzyme.get("candidate_ids") or []
    )
    replaced = _ordered_strings(
        row.get("replaced_step_ids") or row.get("replaced_edge_ids") or []
    )
    equivalent_count = _positive_int(
        row.get("chemical_step_equivalent_count")
        or row.get("collapsed_step_count")
        or len(replaced)
        or 1
    )
    reasons: list[str] = []
    if require_multiple and equivalent_count < 2:
        reasons.append("biocatalytic_superstep_requires_multiple_chemical_steps")
    elif equivalent_count < 1:
        reasons.append("biocatalytic_step_equivalent_count_invalid")
    if not (ec_numbers or enzyme_classes or candidate_ids):
        reasons.append("biocatalytic_superstep_enzyme_hypothesis_missing")
    selectivity = str(row.get("selectivity_objective") or "").strip()
    if not selectivity:
        reasons.append("biocatalytic_superstep_selectivity_objective_missing")
    record = {
        "proposal_basis": "biocatalysis_hypothesis",
        "transformation_mode": "biocatalytic",
        "chemical_step_equivalent_count": equivalent_count,
        "step_savings": max(0, equivalent_count - 1),
        "replaced_step_ids": replaced,
        "enzyme": {
            "ec_numbers": ec_numbers,
            "classes": enzyme_classes,
            "candidate_ids": candidate_ids,
        },
        "selectivity_objective": selectivity,
        "substrate_scope_basis": str(row.get("substrate_scope_basis") or "unresolved"),
        "cofactor_requirements": _json_mapping(row.get("cofactor_requirements")),
        "cofactor_regenerations": _json_mapping(row.get("cofactor_regenerations")),
        "precedent_refs": _strings(row.get("precedent_refs") or []),
        "validation_status": str(row.get("validation_status") or "proposed"),
        "route_family_ids": _route_family_ids(row),
        "evidence_grade": "low_until_biocatalysis_validated",
        "l2_requires_biocatalysis_validation": True,
    }
    return record, reasons


def _mechanism_record(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    anchor = dict(row.get("anchor") or {}) if isinstance(row.get("anchor"), Mapping) else {}
    anchor_edge_ids = _strings(row.get("anchor_edge_ids") or anchor.get("edge_ids") or [])
    anchor_source_ids = _strings(
        row.get("anchor_source_binding_ids") or anchor.get("source_binding_ids") or []
    )
    anchor_source_refs = _strings(
        row.get("anchor_source_refs") or anchor.get("source_refs") or []
    )
    depth = _positive_int(row.get("hypothesis_depth") or row.get("extension_depth") or 1)
    rationale = str(
        row.get("mechanistic_rationale")
        or row.get("transformation_hypothesis")
        or ""
    ).strip()
    checks = _strings(row.get("falsifiable_checks") or row.get("validation_checks") or [])
    reasons: list[str] = []
    if not (anchor_edge_ids or anchor_source_ids or anchor_source_refs):
        reasons.append("mechanism_extrapolation_literature_anchor_missing")
    if depth != 1:
        reasons.append("mechanism_extrapolation_must_be_one_hop")
    if len(rationale) < 12:
        reasons.append("mechanism_extrapolation_rationale_missing")
    if not checks:
        reasons.append("mechanism_extrapolation_falsifiable_check_missing")
    record = {
        "proposal_basis": "mechanism_extrapolation",
        "transformation_mode": "chemical_hypothesis",
        "hypothesis_depth": depth,
        "anchor": {
            "edge_ids": anchor_edge_ids,
            "source_binding_ids": anchor_source_ids,
            "source_refs": anchor_source_refs,
        },
        "mechanistic_rationale": rationale[:4000],
        "elementary_steps": _strings(row.get("elementary_steps") or []),
        "analogy_refs": _strings(row.get("analogy_refs") or []),
        "falsifiable_checks": checks,
        "route_family_ids": _route_family_ids(row),
        "evidence_grade": "low_mechanistic_hypothesis",
        "unvalidated_proof_ceiling": "L1_structural_materialized",
        "reported_in_anchor_source": False,
    }
    return record, reasons


def _route_family_ids(row: Mapping[str, Any]) -> list[str]:
    values = _strings(row.get("route_family_ids") or [])
    if row.get("route_family_id"):
        values.append(str(row["route_family_id"]))
    return sorted(set(values))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return sorted({str(item).strip() for item in value or [] if str(item).strip()})


def _ordered_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return list(
        dict.fromkeys(str(item).strip() for item in value or [] if str(item).strip())
    )


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _json_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))


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
    "BIOCATALYTIC_KINDS",
    "BIOCATALYTIC_STEP",
    "BIOCATALYTIC_SUPERSTEP",
    "INNOVATION_KINDS",
    "MECHANISM_EXTRAPOLATION",
    "ROUTE_INNOVATION_SCHEMA",
    "innovation_proof_gate",
    "merge_route_innovations",
    "normalize_route_innovation",
    "route_innovation_summary",
]
