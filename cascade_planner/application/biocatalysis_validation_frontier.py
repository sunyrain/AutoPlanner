"""Compile unvalidated enzyme Program proposals into executable assay plans."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.application.biocatalytic_program_contracts import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA,
    BiocatalyticProgramError,
    strict_object,
    with_digest,
)
from cascade_planner.application.program_validation_frontier_contracts import (
    ProgramValidationFrontierError,
    program_validation_state_snapshots,
    validate_program_validation_frontier_inputs,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


BIOCATALYSIS_VALIDATION_FRONTIER_SCHEMA = "biocatalysis_validation_frontier.v1"
BIOCATALYSIS_VALIDATION_PLAN_SCHEMA = "biocatalysis_validation_plan.v1"

_REQUIRED_ASSAYS = (
    ("exact_substrate_identity", "Confirm every input state before enzyme exposure."),
    ("endpoint_conversion", "Measure conversion to the exact requested output state."),
    (
        "regio_and_stereoselectivity",
        "Resolve product site and stereochemistry against the stated selectivity objective.",
    ),
    (
        "preserved_functionality",
        "Confirm that non-target functional groups and recorded boundary states are preserved.",
    ),
    (
        "cofactor_regeneration_closure",
        "Record cofactor use, regeneration, mass balance, and controls.",
    ),
    (
        "condition_record_binding",
        "Bind enzyme, loading, medium, time, temperature, analytics, and outcome to one record.",
    ),
)


def compile_biocatalysis_validation_frontier(
    graph: Mapping[str, Any],
    discovery: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce plans for every visible proposal lacking accepted exact validation."""

    graph_value = strict_object(graph, "graph")
    discovery_value = strict_object(discovery, "discovery")
    bundle_value = strict_object(bundle, "bundle")
    _validate_inputs(discovery_value, bundle_value)
    candidates = {
        str(row.get("candidate_id") or ""): row
        for row in discovery_value.get("candidates") or []
        if isinstance(row, dict) and row.get("candidate_kind") == "enzyme_window"
    }
    plans: dict[str, dict[str, Any]] = {}
    for program_id, proposal in sorted(
        dict(bundle_value.get("program_proposals") or {}).items()
    ):
        if dict(proposal.get("validation_gate") or {}).get("accepted") is True:
            continue
        candidate_id = str(proposal.get("source_candidate_id") or "")
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise BiocatalyticProgramError("validation_frontier_candidate_missing")
        innovation = dict(candidate.get("route_innovation") or {})
        boundary = dict(candidate.get("boundary") or {})
        plan_id = f"biocatalysis-plan:{strict_canonical_json_sha256({'program_id': program_id, 'candidate_id': candidate_id})[:24]}"
        plan = {
            "schema_version": BIOCATALYSIS_VALIDATION_PLAN_SCHEMA,
            "plan_id": plan_id,
            "program_id": program_id,
            "innovation_id": str(proposal.get("source_innovation_id") or ""),
            "candidate_id": candidate_id,
            "capability_id": str(candidate.get("capability_id") or ""),
            "canonical_context": {
                "replaced_edge_ids": list(
                    proposal.get("equivalent_reference_span") or []
                ),
            },
            "exact_boundary": {
                "input_states": _state_snapshots(
                    graph_value, proposal.get("input_state_ids") or []
                ),
                "output_states": _state_snapshots(
                    graph_value, proposal.get("output_state_ids") or []
                ),
                "principal_input_smiles": str(boundary.get("precursor_smiles") or ""),
                "principal_output_smiles": str(boundary.get("product_smiles") or ""),
            },
            "screen_matrix": {
                "enzyme_candidates": dict(innovation.get("enzyme") or {}),
                "cofactor_requirements": dict(
                    innovation.get("cofactor_requirements") or {}
                ),
                "cofactor_regenerations": dict(
                    innovation.get("cofactor_regenerations") or {}
                ),
            },
            "selectivity_objective": str(
                innovation.get("selectivity_objective") or ""
            ),
            "required_assays": [
                {"assay_id": assay_id, "objective": objective, "required": True}
                for assay_id, objective in _REQUIRED_ASSAYS
            ],
            "evidence_frontier": {
                "precedent_refs": list(innovation.get("precedent_refs") or []),
                "substrate_scope_basis": str(
                    innovation.get("substrate_scope_basis") or ""
                ),
                "exact_substrate_claim_refs": [],
                "missing_gate": "specialized_biocatalysis_validation",
            },
            "required_output_contract": {
                "schema_version": BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
                "must_bind_exact_input_and_output_states": True,
                "must_assess_selectivity": True,
                "must_close_cofactor_ledger": bool(
                    innovation.get("cofactor_requirements")
                ),
                "must_bind_claim_and_condition_refs": True,
            },
            "status": "experiment_required",
            "grants_validation": False,
            "eligible_for_shadow_admission": False,
            "eligible_for_route_completion": False,
        }
        plans[plan_id] = with_digest(plan)
    return with_digest(
        {
            "schema_version": BIOCATALYSIS_VALIDATION_FRONTIER_SCHEMA,
            "run_id": str(graph_value.get("run_id") or ""),
            "route_id": str(discovery_value.get("route_id") or ""),
            "source_discovery_sha256": str(discovery_value["content_sha256"]),
            "source_bundle_sha256": str(bundle_value["content_sha256"]),
            "plans": plans,
            "counts": {"experiment_required": len(plans), "validation_granted": 0},
            "semantics": {
                "plans_are_read_only": True,
                "literature_analogy_does_not_satisfy_exact_substrate_validation": True,
                "plans_do_not_grant_admission_or_route_completion": True,
                "target_names_are_not_plan_inputs": True,
            },
        }
    )


def _validate_inputs(discovery: dict[str, Any], bundle: dict[str, Any]) -> None:
    try:
        validate_program_validation_frontier_inputs(
            discovery,
            bundle,
            expected_bundle_schema=BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA,
        )
    except ProgramValidationFrontierError as exc:
        raise BiocatalyticProgramError(str(exc)) from exc


def _state_snapshots(graph: dict[str, Any], state_ids: Sequence[Any]) -> list[dict[str, str]]:
    try:
        return program_validation_state_snapshots(graph, state_ids)
    except ProgramValidationFrontierError as exc:
        raise BiocatalyticProgramError(str(exc)) from exc


__all__ = [
    "BIOCATALYSIS_VALIDATION_FRONTIER_SCHEMA",
    "BIOCATALYSIS_VALIDATION_PLAN_SCHEMA",
    "compile_biocatalysis_validation_frontier",
]
