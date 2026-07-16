"""Translate retained ChemEnzy enzyme metadata into route innovation options."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.route_innovations import (
    BIOCATALYTIC_STEP,
    BIOCATALYTIC_SUPERSTEP,
    normalize_route_innovation,
)


def route_innovation_from_chemenzy_step(
    value: Mapping[str, Any],
    *,
    route_family_id: str = "",
) -> dict[str, Any]:
    step = dict(value)
    metadata = (
        dict(step.get("raw_backend_metadata") or {})
        if isinstance(step.get("raw_backend_metadata"), Mapping)
        else {}
    )
    enzyme_annotations = [
        dict(item)
        for item in step.get("enzyme_ec_annotations") or []
        if isinstance(item, Mapping)
    ]
    catalyst_annotations = [
        dict(item)
        for item in step.get("catalyst_annotations") or []
        if isinstance(item, Mapping)
    ]
    ec_numbers = _strings(
        [
            *(item.get("ec_number") for item in enzyme_annotations),
            *(item.get("ec_number") for item in catalyst_annotations),
            *(metadata.get("ec_numbers") or []),
        ]
    )
    enzyme_classes = _strings(
        [
            *(item.get("enzyme_class") for item in enzyme_annotations),
            *(item.get("catalyst_class") for item in catalyst_annotations),
            metadata.get("enzyme_class"),
        ]
    )
    enzyme_classes = [item for item in enzyme_classes if item != "enzyme"]
    if not (
        ec_numbers
        or enzyme_classes
        or step.get("is_enzymatic") is True
        or metadata.get("is_enzymatic") is True
    ):
        return {}
    replaced = _strings(
        step.get("replaced_step_ids") or metadata.get("replaced_step_ids") or []
    )
    equivalent_count = _positive_int(
        step.get("chemical_step_equivalent_count")
        or metadata.get("chemical_step_equivalent_count")
        or metadata.get("collapsed_step_count")
        or len(replaced)
        or 1
    )
    sp = metadata.get("enzyme_sp_verifier_v1")
    sp = dict(sp) if isinstance(sp, Mapping) else {}
    innovation, reasons = normalize_route_innovation(
        {
            "kind": (
                BIOCATALYTIC_SUPERSTEP
                if equivalent_count >= 2
                else BIOCATALYTIC_STEP
            ),
            "route_family_id": route_family_id,
            "chemical_step_equivalent_count": equivalent_count,
            "replaced_step_ids": replaced,
            "ec_numbers": ec_numbers,
            "enzyme_classes": enzyme_classes or ["enzyme class unresolved"],
            "selectivity_objective": str(
                step.get("selectivity_objective")
                or metadata.get("selectivity_objective")
                or "match proposed product regio- and stereochemistry"
            ),
            "substrate_scope_basis": (
                "chemenzy_sp_v1_supported"
                if sp.get("accepted") is True
                else "chemenzy_candidate_unvalidated"
            ),
            "cofactor_requirements": metadata.get("cofactor_requirements") or {},
            "cofactor_regenerations": metadata.get("cofactor_regenerations") or {},
            "precedent_refs": metadata.get("precedent_refs") or [],
            "validation_status": (
                "scope_predicted" if sp.get("accepted") is True else "proposed"
            ),
        }
    )
    return innovation if not reasons else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return sorted({str(item).strip() for item in value or [] if str(item).strip()})


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = ["route_innovation_from_chemenzy_step"]
