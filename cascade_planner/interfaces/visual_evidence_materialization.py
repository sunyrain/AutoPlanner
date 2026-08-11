"""Materialize visual source observations through canonical host admission."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.reaction_condition_records import (
    audit_condition_completeness,
    normalize_source_conditions,
)
from cascade_planner.application.retrosynthesis_workers import (
    materialization_commands_for_proposals,
)
from cascade_planner.interfaces.visual_evidence_contract import (
    materialization_stage as _materialization_stage,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate

def materialize_visual_evidence_candidates(
    service: Any,
    *,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit a visually extracted literature chain as L0/L1 proposals.

    Page-bound visual extraction is useful chemistry generation, but it is not
    deterministic source proof.  Every step therefore goes through the same
    host materialization gates as ChemEnzy/Codex and remains below L3 until an
    independent structured extractor or curator supplies exact rows.
    """

    steps = [
        dict(row)
        for row in observation.get("candidate_steps") or []
        if isinstance(row, Mapping) and row.get("admission_eligible") is True
    ]
    if not steps:
        return _materialization_stage("not_needed", reason="visual_candidate_steps_missing")
    if not (
        int(observation.get("matched_current_edge_count") or 0)
        or int(observation.get("frontier_anchored_step_count") or 0)
        or int(observation.get("target_anchored_step_count") or 0)
    ):
        return _materialization_stage(
            "not_needed",
            reason="visual_chain_not_connected_to_canonical_target_or_frontier",
            proposal_count=len(steps),
            observation_ref=str(observation.get("content_sha256") or ""),
            semantics={
                "disconnected_visual_chain_retained_as_source_observation": True,
                "disconnected_visual_chain_not_added_to_target_graph": True,
                "visual_chain_grants_exact_evidence": False,
            },
        )
    source_ref = str(observation.get("source_ref") or "")
    graph = service.graph_store.load()
    existing_visual_origins = {
        (
            str(edge.get("edge_digest") or ""),
            str(origin.get("origin_ref") or ""),
            str(origin.get("proposal_id") or ""),
        )
        for edge in dict(graph.get("edges") or {}).values()
        if isinstance(edge, Mapping)
        for origin in edge.get("origin_records") or []
        if isinstance(origin, Mapping)
        and str(origin.get("origin_kind") or "")
        == "literature_visual_extraction"
    }
    proposals = []
    for row in steps:
        condition = dict(row.get("condition_candidate") or {})
        normalized_conditions = normalize_source_conditions(condition)
        edge_digest = str(
            audit_retrosynthetic_candidate(
                row.get("product_smiles"),
                row.get("precursor_smiles") or [],
            ).get("edge_digest")
            or ""
        )
        proposal_id = str(row.get("candidate_id") or "")
        if (edge_digest, source_ref, proposal_id) in existing_visual_origins:
            continue
        proposals.append(
            {
                "product_smiles": str(row.get("product_smiles") or ""),
                "precursor_smiles": list(row.get("precursor_smiles") or []),
                "reagent_smiles": list(
                    row.get("spectator_reactant_smiles") or []
                ),
                "origin_kind": "literature_visual_extraction",
                "origin_ref": source_ref,
                "proposal_id": proposal_id,
                "transformation_hypothesis": (
                    "page-bound literature structure-chain extraction"
                ),
                "condition_predictions": (
                    [
                        {
                            **condition,
                            "conditions": normalized_conditions,
                            "condition_completeness": audit_condition_completeness(
                                normalized_conditions
                            ),
                            "source_ref": source_ref,
                            "source_locator": str(row.get("source_locator") or ""),
                            "authority_scope": "model_extracted_source_condition_candidate",
                            "not_reaction_proof": True,
                            "exact_structure_binding_candidate": bool(
                                row.get("exact_structure_binding_candidate")
                            ),
                            "matched_current_edge_id": str(
                                row.get("matched_current_edge_id") or ""
                            ),
                        }
                    ]
                    if condition
                    else []
                ),
            }
        )
    if not proposals:
        return _materialization_stage(
            "reused_or_empty",
            reason="visual_source_binding_already_materialized",
            proposal_count=0,
            observation_step_count=len(steps),
            exact_structure_binding_candidate_count=sum(
                bool(row.get("exact_structure_binding_candidate")) for row in steps
            ),
            matched_current_edge_ids=sorted(
                {
                    str(row.get("matched_current_edge_id") or "")
                    for row in steps
                    if str(row.get("matched_current_edge_id") or "")
                }
            ),
        )
    revision = service.kernel.revision
    commands = materialization_commands_for_proposals(
        proposals,
        run_id=service.kernel.spec.run_id,
        input_revision=revision.graph_revision,
        dependency_revisions={
            "graph_revision": revision.graph_revision,
            "evidence_revision": revision.evidence_revision,
        },
        # Re-run an already-known identity through the canonical worker so
        # its literature origin and page-bound condition are merged onto the
        # existing edge.  Passing the digest as an exclusion would silently
        # discard the newly acquired source binding.
        existing_edge_digests=(),
    )
    if not commands:
        return _materialization_stage(
            "reused_or_empty",
            reason="visual_chain_already_materialized_or_ineligible",
            proposal_count=len(proposals),
        )
    execution = service.execute_commands(
        commands,
        idempotency_key=f"visual-chain:{str(observation.get('content_sha256') or '')}",
    )
    return _materialization_stage(
        "completed" if execution.get("changed") else "partial",
        proposal_count=len(proposals),
        exact_structure_binding_candidate_count=sum(
            bool(row.get("exact_structure_binding_candidate")) for row in steps
        ),
        matched_current_edge_ids=sorted(
            {
                str(row.get("matched_current_edge_id") or "")
                for row in steps
                if str(row.get("matched_current_edge_id") or "")
            }
        ),
        command_count=len(commands),
        execution=execution,
        material_events=(
            ["visual_literature_chain_materialized"]
            if execution.get("changed")
            else []
        ),
        semantics={
            "visual_chain_enters_canonical_hypergraph": True,
            "visual_chain_grants_exact_evidence": False,
            "host_validation_still_required": True,
        },
    )

__all__ = ["materialize_visual_evidence_candidates"]
