"""Data contracts for whole-cell and hybrid route execution capabilities."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.route_structure_matching import (
    normalize_structure_match,
    structure_match_input_valid,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_EXECUTION_CAPABILITY_SCHEMA = "program_execution_capability.v1"
PROGRAM_EXECUTION_CAPABILITY_CATALOG_SCHEMA = (
    "program_execution_capability_catalog.v1"
)
EXECUTION_DOMAINS = {"whole_cell", "hybrid"}
OPERATION_DOMAINS = {"chemical", "enzymatic", "whole_cell"}
OPERATION_KINDS = {
    "chemical_reaction",
    "enzyme_reaction",
    "whole_cell_preparation",
    "whole_cell_biotransformation",
    "cofactor_regeneration",
    "workup",
    "separation",
}


def normalize_program_execution_capability(
    value: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    raw = dict(value or {})
    capability_id = str(raw.get("capability_id") or "").strip()
    execution_domain = str(raw.get("execution_domain") or "").strip().lower()
    match = normalize_structure_match(raw.get("match"))
    actors = _actors(raw.get("actors"))
    operations, operation_reasons = _operations(raw.get("operation_blueprints"))
    refs = _strings(raw.get("precedent_refs"))
    selectivity = str(raw.get("selectivity_objective") or "").strip()
    reasons = list(operation_reasons)
    if not structure_match_input_valid(raw.get("match")):
        reasons.append("program_execution_structure_match_invalid")
    supplied_schema = str(raw.get("schema_version") or "").strip()
    if supplied_schema and supplied_schema != PROGRAM_EXECUTION_CAPABILITY_SCHEMA:
        reasons.append("program_execution_capability_schema_invalid")
    if not capability_id:
        reasons.append("program_execution_capability_id_missing")
    if execution_domain not in EXECUTION_DOMAINS:
        reasons.append("program_execution_domain_invalid")
    if not match["net_motif_delta"]:
        reasons.append("program_execution_net_transform_missing")
    if not refs:
        reasons.append("program_execution_precedent_missing")
    if not selectivity:
        reasons.append("program_execution_selectivity_objective_missing")
    reasons.extend(_domain_reasons(execution_domain, actors, operations))
    if reasons:
        return {}, sorted(set(reasons))
    record = {
        "schema_version": PROGRAM_EXECUTION_CAPABILITY_SCHEMA,
        "capability_id": capability_id,
        "label": str(raw.get("label") or capability_id),
        "execution_domain": execution_domain,
        "actors": actors,
        "match": match,
        "operation_blueprints": operations,
        "selectivity_objective": selectivity,
        "substrate_scope_basis": str(
            raw.get("substrate_scope_basis") or "analogy only"
        ),
        "cofactor_requirements": _mapping(raw.get("cofactor_requirements")),
        "cofactor_regenerations": _mapping(raw.get("cofactor_regenerations")),
        "carrier_requirements": _mapping(raw.get("carrier_requirements")),
        "carrier_regenerations": _mapping(raw.get("carrier_regenerations")),
        "precedent_refs": refs,
        "validation_requirements": _validation_requirements(
            execution_domain, operations
        ),
        "authority_scope": "search_prior_only",
        "not_reaction_proof": True,
        "exact_substrate_validated": False,
    }
    record["content_sha256"] = strict_canonical_json_sha256(record)
    return record, []


def normalize_program_execution_catalog(
    value: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(value, Mapping):
        supplied_schema = str(value.get("schema_version") or "").strip()
        if (
            supplied_schema
            and supplied_schema != PROGRAM_EXECUTION_CAPABILITY_CATALOG_SCHEMA
        ):
            return [], [
                {
                    "capability_id": "",
                    "reasons": ["program_execution_capability_catalog_schema_invalid"],
                }
            ]
    rows = (
        list(value.get("capabilities") or [])
        if isinstance(value, Mapping)
        else list(value)
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in rows:
        record, reasons = normalize_program_execution_capability(raw)
        if reasons:
            rejected.append(
                {
                    "capability_id": str(dict(raw).get("capability_id") or ""),
                    "reasons": reasons,
                }
            )
        elif record:
            accepted.append(record)
    return sorted(accepted, key=lambda row: row["capability_id"]), rejected


def _actors(value: Any) -> dict[str, Any]:
    raw = dict(value or {}) if isinstance(value, Mapping) else {}
    organism = dict(raw.get("organism") or {})
    enzyme = dict(raw.get("enzyme") or {})
    return {
        "organism": {
            "strain_ids": _strings(organism.get("strain_ids")),
            "taxa": _strings(organism.get("taxa")),
            "preparation_modes": _strings(organism.get("preparation_modes")),
        },
        "enzyme": {
            "classes": _strings(enzyme.get("classes")),
            "ec_numbers": _strings(enzyme.get("ec_numbers")),
            "candidate_ids": _strings(enzyme.get("candidate_ids")),
        },
        "catalyst_classes": _strings(raw.get("catalyst_classes")),
    }


def _operations(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, list):
        return [], ["program_execution_operations_invalid"]
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            reasons.append("program_execution_operation_not_object")
            continue
        operation_kind = str(raw.get("operation_kind") or "").strip().lower()
        operation_domain = str(raw.get("execution_domain") or "").strip().lower()
        description = str(raw.get("description") or "").strip()
        if operation_kind not in OPERATION_KINDS:
            reasons.append("program_execution_operation_kind_invalid")
        if operation_domain not in OPERATION_DOMAINS:
            reasons.append("program_execution_operation_domain_invalid")
        if not description:
            reasons.append("program_execution_operation_description_missing")
        if not isinstance(raw.get("isolated_operation"), bool):
            reasons.append("program_execution_operation_isolated_flag_invalid")
        if not isinstance(raw.get("contributes_to_net_transform"), bool):
            reasons.append("program_execution_operation_transform_flag_invalid")
        rows.append(
            {
                "sequence_index": index,
                "operation_kind": operation_kind,
                "execution_domain": operation_domain,
                "isolated_operation": raw.get("isolated_operation") is True,
                "contributes_to_net_transform": (
                    raw.get("contributes_to_net_transform") is True
                ),
                "description": description,
            }
        )
    if not rows or not any(row["isolated_operation"] for row in rows):
        reasons.append("program_execution_isolated_operation_missing")
    if not any(row["contributes_to_net_transform"] for row in rows):
        reasons.append("program_execution_transform_operation_missing")
    return rows, reasons


def _domain_reasons(
    domain: str, actors: Mapping[str, Any], operations: list[dict[str, Any]]
) -> list[str]:
    reasons: list[str] = []
    transform_domains = {
        row["execution_domain"]
        for row in operations
        if row["contributes_to_net_transform"]
    }
    kinds = {row["operation_kind"] for row in operations}
    organism = dict(actors["organism"])
    enzyme = dict(actors["enzyme"])
    organism_present = bool(organism["strain_ids"] or organism["taxa"])
    if domain == "whole_cell":
        if not organism_present or not organism["preparation_modes"]:
            reasons.append("whole_cell_organism_or_preparation_missing")
        if not {
            "whole_cell_preparation",
            "whole_cell_biotransformation",
        }.issubset(kinds):
            reasons.append("whole_cell_operation_sequence_incomplete")
        if transform_domains != {"whole_cell"}:
            reasons.append("whole_cell_transform_domain_invalid")
    if domain == "hybrid":
        biological = transform_domains.intersection({"enzymatic", "whole_cell"})
        if "chemical" not in transform_domains or not biological:
            reasons.append("hybrid_requires_chemical_and_biological_transforms")
        if "whole_cell" in biological and (
            not organism_present or not organism["preparation_modes"]
        ):
            reasons.append("hybrid_whole_cell_actor_missing")
        if "enzymatic" in biological and not (
            enzyme["classes"] or enzyme["ec_numbers"] or enzyme["candidate_ids"]
        ):
            reasons.append("hybrid_enzyme_actor_missing")
    return reasons


def _validation_requirements(
    domain: str, operations: list[dict[str, Any]]
) -> list[str]:
    checks = {
        "condition_envelope_and_record_complete",
        "exact_product_identity",
        "conversion_and_mass_balance",
        "selectivity_and_byproduct_profile",
        "preparative_reproducibility",
    }
    if domain == "whole_cell" or any(
        row["execution_domain"] == "whole_cell" for row in operations
    ):
        checks.update(
            {
                "organism_identity_and_viability",
                "intracellular_extracellular_product_distribution",
            }
        )
    if domain == "hybrid":
        checks.update(
            {
                "ordered_operation_compatibility",
                "solvent_ph_temperature_and_carryover_compatibility",
            }
        )
    return sorted(checks)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return sorted({str(item).strip() for item in value or [] if str(item).strip()})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "EXECUTION_DOMAINS",
    "PROGRAM_EXECUTION_CAPABILITY_CATALOG_SCHEMA",
    "PROGRAM_EXECUTION_CAPABILITY_SCHEMA",
    "normalize_program_execution_capability",
    "normalize_program_execution_catalog",
]
