"""Project multi-edge Program proposals as a separate route-graph layer."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


PROGRAM_OVERLAY_SCHEMA = "route_program_overlay.v1"


def project_program_overlays(
    reviews: Iterable[Mapping[str, Any]],
    *,
    step_id_by_branch_edge: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """Return display-only superstep spans without changing canonical edges."""

    overlays: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_review in reviews:
        if not isinstance(raw_review, Mapping):
            continue
        review = dict(raw_review)
        bundle = dict(review.get("program_bundle") or {})
        branch_id = str(bundle.get("source_route_id") or review.get("route_id") or "")
        plans = {
            str(value.get("program_id") or ""): dict(value)
            for value in dict(
                dict(review.get("validation_frontier") or {}).get("plans") or {}
            ).values()
            if isinstance(value, Mapping) and str(value.get("program_id") or "")
        }
        proposals = dict(bundle.get("program_proposals") or {})
        for raw_proposal in proposals.values():
            if not isinstance(raw_proposal, Mapping):
                continue
            proposal = dict(raw_proposal)
            program_id = str(proposal.get("program_id") or "")
            replaced_edge_ids = [
                str(value)
                for value in proposal.get("equivalent_reference_span") or []
                if str(value)
            ]
            # Single-edge enzyme options remain in the reaction inspector.  A
            # graph overlay is reserved for a real multi-edge Program boundary.
            if not program_id or program_id in seen or len(replaced_edge_ids) < 2:
                continue
            replaced_step_ids = [
                step_id_by_branch_edge.get((branch_id, edge_id), "")
                for edge_id in replaced_edge_ids
            ]
            if not branch_id or any(not value for value in replaced_step_ids):
                continue
            plan = plans.get(program_id, {})
            boundary = dict(plan.get("exact_boundary") or {})
            input_states = _boundary_states(boundary.get("input_states"))
            output_states = _boundary_states(boundary.get("output_states"))
            if not input_states or not output_states:
                continue
            screen = dict(plan.get("screen_matrix") or {})
            enzymes = dict(screen.get("enzyme_candidates") or {})
            overlay = {
                "schema_version": PROGRAM_OVERLAY_SCHEMA,
                "program_id": program_id,
                "program_kind": str(proposal.get("proposal_kind") or "biocatalytic_superstep"),
                "branch_id": branch_id,
                "source_capability_id": str(proposal.get("source_capability_id") or ""),
                "replaced_edge_ids": replaced_edge_ids,
                "replaced_step_ids": replaced_step_ids,
                "input_molecule_node_ids": [row["molecule_id"] for row in input_states],
                "output_molecule_node_ids": [row["molecule_id"] for row in output_states],
                "input_states": input_states,
                "output_states": output_states,
                "chemical_step_equivalent_count": int(
                    proposal.get("chemical_step_equivalent_count") or len(replaced_edge_ids)
                ),
                "isolated_operation_count": int(proposal.get("isolated_operation_count") or 1),
                "net_step_savings": int(proposal.get("net_step_savings") or 0),
                "status": str(proposal.get("status") or "proposal_only"),
                "validation_status": str(plan.get("status") or "experiment_required"),
                "warning_codes": [
                    str(value) for value in proposal.get("warning_codes") or [] if str(value)
                ],
                "candidate_enzyme_ids": [
                    str(value) for value in enzymes.get("candidate_ids") or [] if str(value)
                ],
                "enzyme_classes": [
                    str(value) for value in enzymes.get("classes") or [] if str(value)
                ],
                "enzyme_ec_numbers": [
                    str(value) for value in enzymes.get("ec_numbers") or [] if str(value)
                ],
                "cofactor_and_carrier_ledger": dict(
                    proposal.get("cofactor_and_carrier_ledger") or {}
                ),
                "selectivity_constraints": [
                    str(value)
                    for value in proposal.get("selectivity_constraints") or []
                    if str(value)
                ],
                "precedent_refs": [
                    str(value)
                    for value in dict(proposal.get("claim_refs") or {}).get("precedent_refs") or []
                    if str(value)
                ],
                "analogy_only": dict(proposal.get("claim_refs") or {}).get("analogy_only")
                is True,
                "required_assays": [
                    dict(value)
                    for value in plan.get("required_assays") or []
                    if isinstance(value, Mapping)
                ],
                "validation_gate": dict(proposal.get("validation_gate") or {}),
                "fallback_retained": True,
                "eligible_for_route_completion": False,
                "semantics": {
                    "display_only_shadow_layer": True,
                    "not_a_canonical_reaction_edge": True,
                    "does_not_inherit_replaced_edge_proof": True,
                    "canonical_chemical_steps_remain_visible_fallback": True,
                    "cannot_grant_route_completion": True,
                },
            }
            overlays.append(overlay)
            seen.add(program_id)
    return sorted(overlays, key=lambda row: (row["branch_id"], row["program_id"]))


def program_overlay_integrity_reasons(
    overlays: Any,
    *,
    nodes: Any,
    steps: Any,
    branches: Any,
    scope: str,
) -> list[str]:
    """Validate overlay boundaries against the delivered canonical fallback."""

    if overlays is None:
        return []
    if not isinstance(overlays, list):
        return [f"{scope}_program_overlays_not_list"]
    node_ids = {
        str(value.get("node_id") or "")
        for value in nodes or []
        if isinstance(value, Mapping)
    }
    step_rows = {
        str(value.get("step_id") or ""): dict(value)
        for value in steps or []
        if isinstance(value, Mapping)
    }
    branch_ids = {
        str(value.get("branch_id") or "")
        for value in branches or []
        if isinstance(value, Mapping)
    }
    reasons: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(overlays):
        prefix = f"{scope}_program_overlay:{index}"
        if not isinstance(raw, Mapping):
            reasons.append(f"{prefix}_not_object")
            continue
        row = dict(raw)
        program_id = str(row.get("program_id") or "")
        branch_id = str(row.get("branch_id") or "")
        replaced = row.get("replaced_step_ids")
        if row.get("schema_version") != PROGRAM_OVERLAY_SCHEMA:
            reasons.append(f"{prefix}_schema_invalid")
        if not program_id or program_id in seen:
            reasons.append(f"{prefix}_program_id_invalid_or_duplicate")
        seen.add(program_id)
        if branch_id not in branch_ids:
            reasons.append(f"{prefix}_branch_unknown")
        if not isinstance(replaced, list) or len(replaced) < 2:
            reasons.append(f"{prefix}_fallback_span_invalid")
            replaced = []
        for step_id in replaced:
            step = step_rows.get(str(step_id))
            if step is None:
                reasons.append(f"{prefix}_fallback_step_unknown:{step_id}")
            elif str(step.get("branch_id") or "") != branch_id:
                reasons.append(f"{prefix}_fallback_step_branch_mismatch:{step_id}")
        for field in ("input_molecule_node_ids", "output_molecule_node_ids"):
            values = row.get(field)
            if not isinstance(values, list) or not values:
                reasons.append(f"{prefix}_{field}_invalid")
                continue
            for node_id in values:
                if str(node_id) not in node_ids:
                    reasons.append(f"{prefix}_{field}_unknown:{node_id}")
        if row.get("fallback_retained") is not True:
            reasons.append(f"{prefix}_fallback_not_retained")
        if row.get("eligible_for_route_completion") is not False:
            reasons.append(f"{prefix}_route_completion_authority_invalid")
    return reasons


def _boundary_states(value: Any) -> list[dict[str, str]]:
    return [
        {
            "state_id": str(row.get("state_id") or ""),
            "molecule_id": str(row.get("molecule_id") or ""),
            "canonical_smiles": str(row.get("canonical_smiles") or ""),
        }
        for row in value or []
        if isinstance(row, Mapping) and str(row.get("molecule_id") or "")
    ]


__all__ = [
    "PROGRAM_OVERLAY_SCHEMA",
    "program_overlay_integrity_reasons",
    "project_program_overlays",
]
