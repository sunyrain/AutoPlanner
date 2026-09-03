"""Compile literature/analogy evidence into search-consumable proposals.

This layer is intentionally advisory.  It translates reaction ideas, process
anchors, visual connectivity hints, and broad templates into bounded precursor
proposals that downstream search can expand recursively.  It never creates a
solved claim and it does not promote analogy to parent-route proof.
"""
from __future__ import annotations

import hashlib
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.legacy.harness_runtime.recursive_hypothesis_tasks import (
    RECURSIVE_HYPOTHESIS_TASK_SCHEMA,
)


RDLogger.DisableLog("rdApp.*")

REACTION_IDEA_CARD_SCHEMA = "reaction_idea_card.v1"
RETROSYNTHETIC_PROPOSAL_SCHEMA = "retrosynthetic_proposal.v1"
PROPOSAL_COMPILE_REPORT_SCHEMA = "retrosynthetic_proposal_compile_report.v1"


def compile_retrosynthetic_proposal_bus(
    blackboard: dict[str, Any],
    *,
    max_cards: int = 32,
    max_proposals: int = 32,
    max_recursive_tasks: int = 12,
) -> dict[str, Any]:
    """Return reaction idea cards, proposals, and recursive search tasks."""
    cards = _dedupe_rows(
        [
            *_cards_from_target_side_hypotheses(blackboard),
            *_cards_from_analogical_hypotheses(blackboard),
            *_cards_from_process_evidence(blackboard),
            *_cards_from_broad_templates(blackboard),
            *_cards_from_template_applications(blackboard),
            *_cards_from_visual_chains(blackboard),
            *_cards_from_exact_rows(blackboard),
            *_cards_from_route_objectives(blackboard),
        ],
        key="card_id",
    )[: max(1, int(max_cards or 32))]
    raw_proposals = [
            *_proposals_from_template_applications(blackboard),
            *_proposals_from_visual_chains(blackboard),
            *_proposals_from_exact_rows(blackboard),
            *_proposals_from_process_evidence(blackboard),
            *_proposals_from_analogical_reaction_pairs(blackboard),
            *_proposals_from_failure_feedback(blackboard),
            *_concrete_proposals_from_reaction_idea_cards(blackboard, cards),
            *_strategic_proposals_from_cards(blackboard, cards),
        ]
    proposals = deduplicate_retrosynthetic_proposals(raw_proposals)
    proposals = sorted(proposals, key=_proposal_sort_key)[: max(1, int(max_proposals or 32))]
    recursive_tasks = recursive_tasks_from_retrosynthetic_proposals(
        blackboard,
        proposals,
        max_tasks=max_recursive_tasks,
    )
    return {
        "schema_version": PROPOSAL_COMPILE_REPORT_SCHEMA,
        "accepted": bool(cards or proposals),
        "reaction_idea_cards": cards,
        "retrosynthetic_proposals": proposals,
        "recursive_hypothesis_tasks": recursive_tasks,
        "counts": {
            "reaction_idea_cards": len(cards),
            "retrosynthetic_proposals": len(proposals),
            "recursive_hypothesis_tasks": len(recursive_tasks),
            "executable_or_semi_executable_proposals": sum(
                1 for row in proposals if str(row.get("proposal_type") or "") in {"exact_executable", "semi_executable"}
            ),
            "strategic_proposals": sum(1 for row in proposals if str(row.get("proposal_type") or "") == "strategic"),
            "raw_projection_proposals": len(raw_proposals),
            "semantic_edge_proposals": len(proposals),
            "duplicate_projection_count": max(0, len(raw_proposals) - len(proposals)),
        },
        "projection_deduplication": {
            "schema_version": "retrosynthetic_proposal_projection_deduplication.v1",
            "key_contract": "canonical_target_plus_canonical_precursor_set; strategic_rows_include_transform_identity",
            "raw_count": len(raw_proposals),
            "deduplicated_count": len(proposals),
            "duplicate_count": max(0, len(raw_proposals) - len(proposals)),
        },
        "allowed_use": "proposal_bus_and_recursive_search_seed_only",
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }


def retrosynthetic_proposal_semantic_key(row: dict[str, Any]) -> str:
    """Return a stable chemical-edge key shared across proposal channels."""
    target = _canonical_smiles(str(row.get("target_smiles") or row.get("product_smiles") or ""))
    precursor = _canonical_smiles(
        str(row.get("precursor_smiles") or row.get("reactant_smiles") or row.get("precursors_smiles") or "")
    )
    if target and precursor:
        return "edge:" + _short_hash(f"{target}>>{precursor}")
    # Strategic cards without structures are not chemical edges.  Keep their
    # transform identity so unrelated high-level ideas do not collapse.
    transform = " ".join(
        str(
            row.get("transformation_idea")
            or row.get("proposal_label")
            or row.get("source_type")
            or row.get("proposal_id")
            or ""
        )
        .strip()
        .lower()
        .split()
    )
    granularity = str(row.get("proposal_granularity") or row.get("proposal_type") or "").strip().lower()
    return "strategic:" + _short_hash("|".join([target, precursor, granularity, transform]))


def deduplicate_retrosynthetic_proposals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate projections without treating corroboration as proof.

    The same target→precursor edge can enter through an exact-row adapter, a
    visual-chain adapter and a derived reaction-idea card.  UI/graph consumers
    should see one edge with multiple projection sources, not several routes.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        key = retrosynthetic_proposal_semantic_key(row)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    merged: list[dict[str, Any]] = []
    for key in order:
        projections = groups[key]
        primary = dict(sorted(projections, key=_projection_primary_sort_key)[0])
        primary["semantic_edge_key"] = key
        projection_ids = _dedupe(
            [
                str(item)
                for row in projections
                for item in [str(row.get("proposal_id") or ""), *[str(value) for value in row.get("projection_proposal_ids") or []]]
                if str(item or "").strip()
            ]
        )
        projection_sources = _dedupe(
            [
                str(item)
                for row in projections
                for item in [str(row.get("source_type") or ""), *[str(value) for value in row.get("projection_source_types") or []]]
                if str(item or "").strip()
            ]
        )
        prior_projection_count = max(
            [int(row.get("projection_count") or 0) for row in projections] or [0]
        )
        primary["projection_count"] = max(len(projections), len(projection_ids), prior_projection_count)
        primary["projection_proposal_ids"] = projection_ids
        primary["projection_source_types"] = projection_sources
        for field in ("evidence_refs", "risk_flags", "required_verification"):
            primary[field] = _dedupe(
                [
                    str(item)
                    for row in projections
                    for item in row.get(field) or []
                    if str(item or "").strip()
                ]
            )
        primary["reaction_families"] = _dedupe(
            [
                str(item)
                for row in projections
                for item in [
                    str(row.get("reaction_family") or ""),
                    *[str(value) for value in row.get("reaction_families") or []],
                ]
                if str(item or "").strip()
            ]
        )
        primary["product_retron_types"] = _dedupe(
            [
                str(item)
                for row in projections
                for item in [
                    str(row.get("product_retron_type") or ""),
                    str(row.get("derived_from_retron") or ""),
                    *[str(value) for value in row.get("product_retron_types") or []],
                ]
                if str(item or "").strip()
            ]
        )
        if not str(primary.get("reaction_family") or "").strip() and primary["reaction_families"]:
            primary["reaction_family"] = primary["reaction_families"][0]
        if not str(primary.get("product_retron_type") or "").strip() and len(primary["product_retron_types"]) == 1:
            primary["product_retron_type"] = primary["product_retron_types"][0]
        primary["derived_from_retron"] = str(primary.get("product_retron_type") or "")
        primary["retron_authority"] = "advisory_search_prior_only"
        primary["score"] = max(int(row.get("score") or 0) for row in projections)
        primary["executable"] = any(bool(row.get("executable")) for row in projections)
        primary["recursive_expandable"] = any(bool(row.get("recursive_expandable")) for row in projections)
        primary["not_exact_literature_segment"] = all(
            bool(row.get("not_exact_literature_segment", True)) for row in projections
        )
        primary["projection_deduplicated"] = primary["projection_count"] > 1
        primary["projection_support_is_not_independent_proof"] = True
        merged.append(primary)
    return merged


def _projection_primary_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    source = str(row.get("source_type") or "").lower()
    source_priority = 5
    if "exact" in source:
        source_priority = 0
    elif source == "analogical_reaction_pair_transfer":
        source_priority = 1
    elif "template_application" in source or "process" in source:
        source_priority = 2
    elif "visual" in source:
        source_priority = 3
    elif "reaction_idea" in source or "strategic" in source:
        source_priority = 4
    score_key = _proposal_sort_key(row)
    return (source_priority, score_key[0], score_key[1])


def recursive_tasks_from_retrosynthetic_proposals(
    blackboard: dict[str, Any],
    proposals: list[dict[str, Any]],
    *,
    max_tasks: int = 12,
) -> list[dict[str, Any]]:
    """Create first-frontier recursive tasks from precursor-bearing proposals."""
    target = dict(blackboard.get("target_profile") or {})
    target_smiles = _canonical_smiles(str(target.get("target_smiles") or target.get("canonical_smiles") or ""))
    existing = _existing_recursive_frontier(blackboard)
    tasks: list[dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        if not bool(proposal.get("recursive_expandable")):
            continue
        precursor = _canonical_smiles(str(proposal.get("precursor_smiles") or ""))
        if not precursor or precursor == target_smiles or precursor in existing:
            continue
        components = _precursor_components(precursor)
        if len(components) > 1:
            for idx, component in enumerate(components, start=1):
                if not component or component == target_smiles or component in existing:
                    continue
                task = _recursive_task_from_proposal(
                    proposal=proposal,
                    target_smiles=target_smiles,
                    precursor=component,
                    precursor_set=precursor,
                    component_index=idx,
                    component_count=len(components),
                    task_scope="precursor_component",
                )
                tasks.append(task)
                existing.add(component)
                if len(tasks) >= max(1, int(max_tasks or 12)):
                    return tasks
            continue
        task = _recursive_task_from_proposal(
            proposal=proposal,
            target_smiles=target_smiles,
            precursor=precursor,
            precursor_set="",
            component_index=0,
            component_count=1,
            task_scope="precursor",
        )
        tasks.append(task)
        existing.add(precursor)
        if len(tasks) >= max(1, int(max_tasks or 12)):
            break
    return tasks


def _recursive_task_from_proposal(
    *,
    proposal: dict[str, Any],
    target_smiles: str,
    precursor: str,
    precursor_set: str,
    component_index: int,
    component_count: int,
    task_scope: str,
) -> dict[str, Any]:
    proposal_id = str(proposal.get("proposal_id") or "")
    label = str(proposal.get("proposal_label") or proposal.get("source_type") or "proposal precursor")
    component_suffix = f":component:{component_index}" if task_scope == "precursor_component" else ""
    risk_flags = [
        "proposal_hypothesis_only",
        "requires_route_expansion_verifier",
        *[str(item) for item in proposal.get("risk_flags") or []],
    ]
    if task_scope == "precursor_component":
        risk_flags.extend(
            [
                "multi_component_precursor_component",
                "requires_precursor_set_stitching",
            ]
        )
    return {
        "schema_version": RECURSIVE_HYPOTHESIS_TASK_SCHEMA,
        "task_id": "recursive_hypothesis:proposal:" + _short_hash(
            "|".join([proposal_id, target_smiles, precursor, task_scope, str(component_index)])
        ),
        "task_type": "recursive_hypothesis_frontier_expansion",
        "task_scope": task_scope,
        "status": "pending",
        "source": "retrosynthetic_proposal",
        "proposal_id": proposal_id,
        "parent_candidate_id": proposal_id,
        "parent_subgoal_name": label,
        "parent_smiles": target_smiles,
        "precursor_smiles": precursor,
        "precursor_set_smiles": precursor_set,
        "precursor_component_index": int(component_index or 0),
        "precursor_component_count": int(component_count or 1),
        "multi_component_precursor_set": int(component_count or 1) > 1,
        "requires_precursor_set_stitching": task_scope == "precursor_component",
        "sibling_precursor_smiles": [
            item
            for item in _precursor_components(precursor_set)
            if item and item != precursor
        ],
        "name": f"{label}{component_suffix}",
        "recursive_depth": 1,
        "operation_idea": str(proposal.get("transformation_idea") or ""),
        "variant_type": str(proposal.get("source_type") or "proposal_precursor"),
        "reaction_family": str(
            proposal.get("reaction_family") or proposal.get("proposal_label") or ""
        ),
        "reaction_families": _dedupe(
            [
                str(proposal.get("reaction_family") or ""),
                *[str(item) for item in proposal.get("reaction_families") or []],
            ]
        ),
        "product_retron_type": str(proposal.get("product_retron_type") or ""),
        "product_retron_types": _dedupe(
            [
                str(proposal.get("product_retron_type") or ""),
                str(proposal.get("derived_from_retron") or ""),
                *[str(item) for item in proposal.get("product_retron_types") or []],
            ]
        ),
        "derived_from_retron": str(
            proposal.get("product_retron_type")
            or proposal.get("derived_from_retron")
            or ""
        ),
        "retron_authority": "advisory_search_prior_only",
        "proposal_granularity": str(proposal.get("proposal_granularity") or "hypothesis"),
        "proposal_score": int(proposal.get("score") or 0),
        "route_objective_type": str(proposal.get("route_objective_type") or ""),
        "failure_response_policy": dict(proposal.get("failure_response_policy") or {}),
        "failure_reasons": [],
        "risk_flags": _dedupe(risk_flags),
        "allowed_use": "route_expansion_subgoal_hint_only",
        "not_exact_literature_segment": bool(proposal.get("not_exact_literature_segment", True)),
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "child_route_cannot_promote_parent": True,
        "no_solved_claim": True,
    }


def _cards_from_target_side_hypotheses(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    for row in (blackboard.get("target_side_disconnection_hypotheses") or {}).get("hypotheses") or []:
        if not isinstance(row, dict):
            continue
        card_id = "idea:target_side:" + _short_hash(str(row.get("hypothesis_id") or row))
        out.append(
            _idea_card(
                card_id=card_id,
                source_type="target_side_disconnection",
                target_handle=str(row.get("target_handle") or ""),
                transformation_idea=str(
                    row.get("proposed_disconnection_region")
                    or row.get("expected_precursor_type")
                    or "target-side disconnection hypothesis"
                ),
                preserved_core=row.get("must_preserve_substructure") or [],
                expected_precursor_type=str(row.get("expected_precursor_type") or ""),
                evidence_refs=[str(row.get("hypothesis_id") or ""), *[str(item) for item in row.get("related_source_evidence") or []]],
                confidence=str(row.get("confidence") or "medium"),
                risk_flags=[str(item) for item in row.get("risk_flags") or []],
                required_verification=[str(item) for item in row.get("required_verification") or []],
                target_smiles=str(target.get("target_smiles") or ""),
            )
        )
    return out


def _cards_from_analogical_hypotheses(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    for row in _analogical_hypothesis_rows(blackboard):
        source_product = _canonical_smiles(_first_text(row.get("source_product_smiles"), row.get("product_smiles")))
        source_reactants = _smiles_values(
            row.get("source_reactant_smiles"),
            row.get("source_reactants_smiles"),
            row.get("source_precursor_smiles"),
            row.get("reactant_smiles"),
        )
        transfer_classes = _source_pair_transfer_classes(source_product, source_reactants)
        if not transfer_classes:
            continue
        out.append(
            _idea_card(
                card_id="idea:analogical_pair:" + _short_hash(str(row.get("hypothesis_id") or row)),
                source_type="analogical_reaction_pair",
                target_handle=str(row.get("reaction_family") or row.get("target_handle") or "analogical_reaction_center"),
                transformation_idea=_join_idea(
                    str(row.get("reaction_family") or "analogical reaction-center transfer"),
                    ", ".join(transfer_classes),
                ),
                preserved_core=row.get("must_preserve_substructure") or row.get("must_preserve") or ["target_core_connectivity"],
                expected_precursor_type="target precursor with the analog source pair reaction-center state",
                evidence_refs=[
                    str(row.get("hypothesis_id") or ""),
                    str(row.get("source_ref") or ""),
                    *[str(item) for item in row.get("evidence_refs") or []],
                ],
                confidence=str(row.get("analogy_strength") or row.get("confidence") or "medium"),
                risk_flags=[
                    *[str(item) for item in row.get("risk_flags") or []],
                    "analogical_reaction_pair_transfer",
                    "source_reaction_center_inferred",
                ],
                required_verification=[
                    "route_expansion_verifier",
                    "parent_bridge_connectivity",
                    "analogy_not_parent_route_proof",
                ],
                target_smiles=str(target.get("target_smiles") or ""),
            )
        )
    return out


def _cards_from_process_evidence(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    evidence = dict(blackboard.get("literature_evidence") or {})
    for row in evidence.get("process_evidence_rows") or []:
        if not isinstance(row, dict):
            continue
        endpoint = str((row.get("endpoint_labels") or ["process endpoint"])[0] or "process endpoint")
        substrates = [str(item) for item in row.get("substrate_or_feedstock_labels") or [] if str(item).strip()]
        process = str(row.get("process_type") or "process_literature_endpoint")
        out.append(
            _idea_card(
                card_id="idea:process:" + _short_hash(str(row.get("row_id") or row)),
                source_type="process_or_biotransformation",
                target_handle="semisynthesis_or_biotransformation_anchor",
                transformation_idea=f"use {process.replace('_', ' ')} precedent to reach {endpoint}",
                preserved_core=["natural_product_or_steroid_core"],
                expected_precursor_type=", ".join(substrates) or "reported process feedstock or endpoint precursor",
                evidence_refs=[str(row.get("row_id") or ""), str(row.get("source_ref") or ""), *[str(item) for item in row.get("evidence_refs") or []]],
                confidence=str(row.get("confidence") or "medium"),
                risk_flags=[str(item) for item in row.get("risk_flags") or []],
                required_verification=[str(item) for item in row.get("verification_required") or []],
                target_smiles=str(target.get("target_smiles") or ""),
            )
        )
    return out


def _cards_from_broad_templates(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    for row in blackboard.get("broad_transform_templates") or []:
        if not isinstance(row, dict):
            continue
        out.append(
            _idea_card(
                card_id="idea:broad_template:" + _short_hash(str(row.get("template_id") or row)),
                source_type="broad_transform_template",
                target_handle=str(row.get("target_handle") or row.get("objective_type") or ""),
                transformation_idea=str(row.get("template_idea") or row.get("transformation_idea") or row.get("summary") or ""),
                preserved_core=row.get("must_preserve_substructure") or row.get("preserved_core") or [],
                expected_precursor_type=str(row.get("expected_precursor_type") or row.get("precursor_class") or ""),
                evidence_refs=[str(row.get("template_id") or ""), *[str(item) for item in row.get("evidence_refs") or []]],
                confidence=str(row.get("confidence") or "low"),
                risk_flags=[str(item) for item in row.get("risk_flags") or ["broad_template_scope"]],
                required_verification=[str(item) for item in row.get("required_verification") or ["route_expansion_verifier"]],
                target_smiles=str(target.get("target_smiles") or ""),
            )
        )
    return out


def _cards_from_template_applications(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    for app in blackboard.get("template_applications") or []:
        if not isinstance(app, dict):
            continue
        route_hypothesis = dict(app.get("hypothetical_route_hypothesis") or {})
        out.append(
            _idea_card(
                card_id="idea:template_application:" + _short_hash(str(app.get("application_id") or app)),
                source_type="analogical_template_application",
                target_handle=str(app.get("product_retron_type") or ""),
                transformation_idea=str(
                    route_hypothesis.get("reaction_center_idea")
                    or route_hypothesis.get("template_application")
                    or "analogical template application"
                ),
                preserved_core=["target_core_connectivity"],
                expected_precursor_type=str(app.get("product_retron_type") or "same-core precursor"),
                evidence_refs=[str(app.get("application_id") or ""), str(app.get("template_id") or ""), *[str(item) for item in app.get("evidence_refs") or []]],
                confidence="medium" if app.get("accepted") else "low",
                risk_flags=[str(item) for item in route_hypothesis.get("risk_flags") or []],
                required_verification=["template_application_validation", "route_expansion_verifier"],
                target_smiles=str(target.get("target_smiles") or ""),
            )
        )
    return out


def _cards_from_visual_chains(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    evidence = dict(blackboard.get("literature_evidence") or {})
    for chain in evidence.get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        for idx, step in enumerate(chain.get("steps") or [], start=1):
            if not isinstance(step, dict):
                continue
            precursor = str(step.get("main_reactant_smiles") or "")
            if not precursor and step.get("reactant_smiles"):
                precursor = str((step.get("reactant_smiles") or [""])[0] or "")
            out.append(
                _idea_card(
                    card_id="idea:visual:" + _short_hash("|".join([str(chain.get("chain_id") or chain.get("artifact_ref") or ""), str(idx), precursor])),
                    source_type="visual_connectivity",
                    target_handle="visual_precursor_connectivity",
                    transformation_idea="visual evidence suggests a precursor/product connectivity relationship",
                    preserved_core=["source_visual_core_connectivity"],
                    expected_precursor_type=str((step.get("reactant_labels") or ["visual precursor"])[0] or "visual precursor"),
                    evidence_refs=[str(chain.get("artifact_ref") or ""), str(chain.get("source_ref") or ""), *[str(item) for item in step.get("evidence_refs") or []]],
                    confidence=str(step.get("confidence") or chain.get("confidence") or "low"),
                    risk_flags=[*[str(item) for item in step.get("risk_flags") or []], "visual_connectivity_approximation"],
                    required_verification=["structure_resolution", "route_expansion_verifier"],
                    target_smiles=str(step.get("product_smiles") or ""),
                )
            )
    return out


def _cards_from_exact_rows(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    evidence = dict(blackboard.get("literature_evidence") or {})
    for row in evidence.get("exact_rows") or []:
        if not isinstance(row, dict):
            continue
        if row.get("not_exact_literature_segment"):
            continue
        if not _first_text(
            row.get("product_smiles"),
            row.get("main_reactant_smiles"),
            row.get("reactant_smiles"),
            row.get("reactants_smiles"),
        ):
            continue
        out.append(
            _idea_card(
                card_id="idea:exact_row:" + _short_hash(str(row.get("row_id") or row)),
                source_type="exact_literature_row",
                target_handle=str(row.get("reaction_family") or row.get("step_role") or ""),
                transformation_idea="source-detail exact literature row can seed a guarded local precursor proposal",
                preserved_core=["source_detail_product_identity"],
                expected_precursor_type=str(row.get("reactant_label") or row.get("main_reactant_label") or "source-detail reactant"),
                evidence_refs=[str(row.get("row_id") or ""), str(row.get("source_ref") or "")],
                confidence=str(row.get("confidence") or "high"),
                risk_flags=[str(item) for item in row.get("risk_flags") or []],
                required_verification=["target_equivalence", "parent_route_verifier", "literature_segment_connectivity"],
                target_smiles=str(row.get("product_smiles") or ""),
            )
        )
    return out


def _cards_from_route_objectives(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    for row in (blackboard.get("route_objective_summary") or {}).get("selected_objectives") or []:
        if not isinstance(row, dict):
            continue
        objective = str(row.get("objective_type") or "")
        out.append(
            _idea_card(
                card_id="idea:route_objective:" + _short_hash(str(row.get("objective_id") or row)),
                source_type="route_objective",
                target_handle=objective,
                transformation_idea=str(row.get("rationale") or objective.replace("_", " ")),
                preserved_core=row.get("must_preserve_substructure") or [],
                expected_precursor_type=str(row.get("expected_endpoint_type") or row.get("objective_type") or ""),
                evidence_refs=[str(row.get("objective_id") or ""), *[str(item) for item in row.get("evidence_refs") or []]],
                confidence=str(row.get("confidence") or "medium"),
                risk_flags=[str(item) for item in row.get("risk_flags") or []],
                required_verification=[str(item) for item in row.get("required_verification") or ["objective_specific_verification"]],
                target_smiles=str(target.get("target_smiles") or ""),
            )
        )
    return out


def _proposals_from_template_applications(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    target_smiles = str(target.get("target_smiles") or "")
    for app in blackboard.get("template_applications") or []:
        if not isinstance(app, dict):
            continue
        route_hypothesis = dict(app.get("hypothetical_route_hypothesis") or {})
        for hint in app.get("hypothetical_precursor_hints") or []:
            if not isinstance(hint, dict):
                continue
            precursor = str(hint.get("precursor_smiles") or "").strip()
            if not precursor:
                continue
            out.append(
                _proposal(
                    proposal_type="semi_executable",
                    source_type="analogical_template_hint",
                    target_smiles=str(hint.get("target_smiles") or target_smiles),
                    precursor_smiles=precursor,
                    transformation_idea=str(
                        route_hypothesis.get("reaction_center_idea")
                        or route_hypothesis.get("template_application")
                        or hint.get("derived_from_retron")
                        or "same-core analogical transformation"
                    ),
                    proposal_label=str(hint.get("precursor_role") or hint.get("hypothesis_type") or "same_core_precursor"),
                    confidence=_confidence_from_risk_flags([*[str(item) for item in route_hypothesis.get("risk_flags") or []], *[str(item) for item in hint.get("risk_flags") or []]]),
                    evidence_refs=[str(app.get("application_id") or ""), str(app.get("template_id") or ""), *[str(item) for item in app.get("evidence_refs") or []]],
                    risk_flags=[*[str(item) for item in route_hypothesis.get("risk_flags") or []], *[str(item) for item in hint.get("risk_flags") or []], "analogy_not_proof"],
                    required_verification=["route_expansion_verifier", "parent_bridge_connectivity"],
                    executable=True,
                    recursive_expandable=True,
                    not_exact_literature_segment=True,
                )
            )
    return out


def _proposals_from_visual_chains(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    evidence = dict(blackboard.get("literature_evidence") or {})
    for chain in evidence.get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        for step in chain.get("steps") or []:
            if not isinstance(step, dict):
                continue
            precursor = str(step.get("main_reactant_smiles") or "").strip()
            if not precursor:
                reactants = [str(item) for item in step.get("reactant_smiles") or [] if str(item).strip()]
                precursor = reactants[0] if reactants else ""
            if not precursor:
                continue
            out.append(
                _proposal(
                    proposal_type="semi_executable",
                    source_type="visual_connectivity_candidate",
                    target_smiles=str(step.get("product_smiles") or target.get("target_smiles") or ""),
                    precursor_smiles=precursor,
                    transformation_idea="visual source suggests a same-core precursor; use as connectivity-only recursive search seed",
                    proposal_label=str((step.get("reactant_labels") or ["visual precursor"])[0] or "visual precursor"),
                    confidence=str(step.get("confidence") or chain.get("confidence") or "low"),
                    evidence_refs=[str(chain.get("artifact_ref") or ""), str(chain.get("source_ref") or ""), *[str(item) for item in step.get("evidence_refs") or []]],
                    risk_flags=[*[str(item) for item in chain.get("reasons") or []], *[str(item) for item in step.get("risk_flags") or []], "visual_connectivity_approximation", "stereochemistry_unresolved"],
                    required_verification=["structure_resolution", "route_expansion_verifier", "stereochemistry_recovery"],
                    executable=True,
                    recursive_expandable=True,
                    not_exact_literature_segment=True,
                )
            )
    return out


def _proposals_from_exact_rows(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    evidence = dict(blackboard.get("literature_evidence") or {})
    for row in evidence.get("exact_rows") or []:
        if not isinstance(row, dict):
            continue
        precursor = _first_text(row.get("main_reactant_smiles"), row.get("reactant_smiles"), row.get("reactants_smiles"))
        if isinstance(row.get("reactant_smiles"), list):
            precursor = _first_text(*row.get("reactant_smiles"))
        if not precursor:
            continue
        out.append(
            _proposal(
                proposal_type="exact_executable",
                source_type="exact_literature_row",
                target_smiles=str(row.get("product_smiles") or ""),
                precursor_smiles=precursor,
                transformation_idea="source-detail exact row proposes a guarded precursor; still requires target equivalence and parent bridge proof",
                proposal_label=str(row.get("row_id") or "exact_literature_precursor"),
                confidence=str(row.get("confidence") or "high"),
                evidence_refs=[str(row.get("row_id") or ""), str(row.get("source_ref") or "")],
                risk_flags=[str(item) for item in row.get("risk_flags") or []],
                required_verification=["target_equivalence", "parent_route_verifier", "stock_audit"],
                executable=True,
                recursive_expandable=True,
                not_exact_literature_segment=False,
            )
        )
    return out


def _proposals_from_process_evidence(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    evidence = dict(blackboard.get("literature_evidence") or {})
    for row in evidence.get("process_evidence_rows") or []:
        if not isinstance(row, dict):
            continue
        endpoint = str((row.get("endpoint_labels") or ["process endpoint"])[0] or "process endpoint")
        substrates = [str(item) for item in row.get("substrate_or_feedstock_labels") or [] if str(item).strip()]
        out.append(
            _proposal(
                proposal_type="strategic",
                source_type="process_or_biotransformation_anchor",
                target_smiles=str(target.get("target_smiles") or ""),
                precursor_smiles="",
                transformation_idea=f"treat {endpoint} as a process/semisynthesis endpoint; search feedstock or organism evidence before small-molecule closure",
                proposal_label=endpoint,
                confidence=str(row.get("confidence") or "medium"),
                evidence_refs=[str(row.get("row_id") or ""), str(row.get("source_ref") or "")],
                risk_flags=[*[str(item) for item in row.get("risk_flags") or []], "process_evidence_not_reaction_smiles"],
                required_verification=[str(item) for item in row.get("verification_required") or ["process_endpoint_acceptability"]],
                executable=False,
                recursive_expandable=False,
                not_exact_literature_segment=True,
            )
        )
        for anchor in blackboard.get("semisynthesis_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            anchor_smiles = str(anchor.get("smiles") or "").strip()
            if not anchor_smiles:
                continue
            out.append(
                _proposal(
                    proposal_type="semi_executable",
                    source_type="semisynthesis_anchor",
                    target_smiles=str(target.get("target_smiles") or ""),
                    precursor_smiles=anchor_smiles,
                    transformation_idea="expand reported semisynthesis anchor as a non-proof precursor frontier",
                    proposal_label=str(anchor.get("name") or endpoint),
                    confidence=str(row.get("confidence") or "medium"),
                    evidence_refs=[str(anchor.get("anchor_id") or ""), str(row.get("row_id") or "")],
                    risk_flags=["semisynthesis_anchor_not_parent_proof"],
                    required_verification=["anchor_identity_audit", "route_expansion_verifier"],
                    executable=True,
                    recursive_expandable=True,
                    not_exact_literature_segment=True,
                )
            )
        for idx, feedstock_smiles in enumerate(
            _smiles_values(
                row.get("substrate_or_feedstock_smiles"),
                row.get("feedstock_smiles"),
                row.get("substrate_smiles"),
                row.get("starting_material_smiles"),
                row.get("precursor_smiles"),
            ),
            start=1,
        ):
            out.append(
                _proposal(
                    proposal_type="semi_executable",
                    source_type="process_feedstock_anchor",
                    target_smiles=str(target.get("target_smiles") or ""),
                    precursor_smiles=feedstock_smiles,
                    transformation_idea=(
                        "expand a reported process or biotransformation feedstock as a "
                        "non-proof semisynthesis frontier"
                    ),
                    proposal_label=str((substrates or [f"process feedstock {idx}"])[min(idx - 1, len(substrates) - 1)]),
                    confidence=str(row.get("confidence") or "medium"),
                    evidence_refs=[str(row.get("row_id") or ""), str(row.get("source_ref") or "")],
                    risk_flags=[
                        *[str(item) for item in row.get("risk_flags") or []],
                        "process_feedstock_not_parent_proof",
                        "biotransformation_or_process_scope_hypothetical",
                    ],
                    required_verification=[
                        "feedstock_identity_audit",
                        "process_endpoint_acceptability",
                        "route_expansion_verifier",
                    ],
                    executable=True,
                    recursive_expandable=True,
                    not_exact_literature_segment=True,
                )
            )
    return out


def _proposals_from_analogical_reaction_pairs(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    target = dict(blackboard.get("target_profile") or {})
    target_smiles = str(target.get("target_smiles") or target.get("canonical_smiles") or "")
    target_canonical = _canonical_smiles(target_smiles)
    if not target_canonical:
        return []
    target_variants = _deterministic_precursor_variants_from_target(target_canonical, limit=32)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _analogical_hypothesis_rows(blackboard):
        source_product = _canonical_smiles(_first_text(row.get("source_product_smiles"), row.get("product_smiles")))
        source_reactants = _smiles_values(
            row.get("source_reactant_smiles"),
            row.get("source_reactants_smiles"),
            row.get("source_precursor_smiles"),
            row.get("reactant_smiles"),
        )
        transfer_classes = set(_source_pair_transfer_classes(source_product, source_reactants))
        if not transfer_classes:
            continue
        evidence_refs = [
            str(row.get("hypothesis_id") or ""),
            str(row.get("source_template_id") or ""),
            str(row.get("source_ref") or ""),
            *[str(item) for item in row.get("evidence_refs") or []],
        ]
        for variant in target_variants:
            variant_type = str(variant.get("variant_type") or "")
            if variant_type not in transfer_classes:
                continue
            precursor = str(variant.get("precursor_smiles") or "")
            if not precursor:
                continue
            marker = "|".join([variant_type, _canonical_smiles(precursor) or precursor])
            if marker in seen:
                continue
            seen.add(marker)
            out.append(
                _proposal(
                    proposal_type="semi_executable",
                    source_type="analogical_reaction_pair_transfer",
                    target_smiles=target_canonical,
                    precursor_smiles=precursor,
                    transformation_idea=_join_idea(
                        "source analog reactant/product pair supports this reaction-center direction",
                        str(variant.get("operation_idea") or ""),
                    ),
                    proposal_label=variant_type,
                    confidence=_analogical_pair_transfer_confidence(row),
                    evidence_refs=evidence_refs,
                    risk_flags=_dedupe(
                        [
                            *[str(item) for item in row.get("risk_flags") or []],
                            *[str(item) for item in variant.get("risk_flags") or []],
                            "analogical_reaction_pair_transfer",
                            "source_reaction_center_inferred",
                            "analogy_not_proof",
                            "not_exact_literature_segment",
                        ]
                    ),
                    required_verification=[
                        "route_expansion_verifier",
                        "parent_bridge_connectivity",
                        "target_core_retention",
                        "analogy_not_parent_route_proof",
                    ],
                    executable=True,
                    recursive_expandable=True,
                    not_exact_literature_segment=True,
                )
            )
    return out


def _proposals_from_failure_feedback(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = dict(blackboard.get("target_profile") or {})
    existing_precursors = {
        _canonical_smiles(str(row.get("precursor_smiles") or ""))
        for row in blackboard.get("retrosynthetic_proposals") or []
        if isinstance(row, dict)
    }
    existing_precursors.discard("")
    for feedback in blackboard.get("proposal_failure_feedback") or []:
        if not isinstance(feedback, dict):
            continue
        parent_target = _canonical_smiles(str(feedback.get("parent_smiles") or target.get("target_smiles") or ""))
        for variant in _refined_precursor_sets_from_failure_feedback(feedback):
            precursor_set = str(variant.get("precursor_smiles") or "")
            canonical_set = _canonical_smiles(precursor_set)
            if not canonical_set or canonical_set == parent_target or canonical_set in existing_precursors:
                continue
            out.append(
                _proposal(
                    proposal_type="semi_executable",
                    source_type="failure_driven_proposal_refinement",
                    target_smiles=str(feedback.get("parent_smiles") or target.get("target_smiles") or ""),
                    precursor_smiles=canonical_set,
                    transformation_idea=_join_idea(
                        "previous hypothesis component failed route expansion; change the precursor granularity or activation state",
                        str(variant.get("operation_idea") or ""),
                    ),
                    proposal_label=str(variant.get("variant_type") or "failure_refined_precursor_set"),
                    confidence="low",
                    evidence_refs=[
                        str(feedback.get("feedback_id") or ""),
                        str(feedback.get("proposal_id") or ""),
                        str(feedback.get("recursive_hypothesis_task_id") or ""),
                    ],
                    risk_flags=_dedupe(
                        [
                            "failure_driven_refinement",
                            "prior_component_frontier_failed",
                            "hypothesis_granularity_changed",
                            *[str(item) for item in variant.get("risk_flags") or []],
                            *[f"previous_failure:{item}" for item in feedback.get("failure_reasons") or []],
                        ]
                    ),
                    required_verification=[
                        "route_expansion_verifier",
                        "parent_bridge_connectivity",
                        "precursor_set_stitching",
                    ],
                    executable=True,
                    recursive_expandable=True,
                    not_exact_literature_segment=True,
                )
            )
            existing_precursors.add(canonical_set)
    return out


def _refined_precursor_sets_from_failure_feedback(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    failed = _canonical_smiles(str(feedback.get("failed_precursor_smiles") or ""))
    if not failed:
        return []
    precursor_set = _canonical_smiles(str(feedback.get("precursor_set_smiles") or ""))
    components = _precursor_components(precursor_set)
    siblings = [
        _canonical_smiles(str(item))
        for item in feedback.get("sibling_precursor_smiles") or []
        if _canonical_smiles(str(item))
    ]
    if components:
        siblings = [item for item in components if item != failed]
    variants: list[dict[str, Any]] = []
    for variant in _component_refinement_variants(failed):
        replacement = _canonical_smiles(str(variant.get("smiles") or ""))
        if not replacement or replacement == failed:
            continue
        if components:
            refined_components = [replacement if item == failed else item for item in components]
        elif siblings:
            refined_components = [replacement, *siblings]
        else:
            refined_components = [replacement]
        refined_set = _canonical_smiles(".".join(refined_components))
        if not refined_set or refined_set == precursor_set:
            continue
        variants.append(
            {
                "variant_type": str(variant.get("variant_type") or "failure_refined_component"),
                "precursor_smiles": refined_set,
                "operation_idea": _join_idea(
                    str(variant.get("operation_idea") or ""),
                    "reuse the sibling components from the failed precursor set" if siblings else "",
                ),
                "risk_flags": _dedupe(
                    [
                        *[str(item) for item in variant.get("risk_flags") or []],
                        "failure_feedback_not_literature_exact",
                        "requires_precursor_set_stitching",
                    ]
                ),
            }
        )
    return _dedupe_rows(variants, key="precursor_smiles")


def _component_refinement_variants(smiles: str) -> list[dict[str, Any]]:
    rules = [
        {
            "rule_id": "failed_carboxylic_acid_to_acid_chloride_component",
            "smarts": "[C:1](=[O:2])[OX2H:3]>>[C:1](=[O:2])Cl",
            "operation_idea": "replace a failed carboxylic acid component with the corresponding acid chloride activation state",
            "risk_flags": ["alternate_acyl_activation_state", "acid_chloride_scope_hypothetical"],
        },
        {
            "rule_id": "failed_carboxylic_acid_to_methyl_ester_component",
            "smarts": "[C:1](=[O:2])[OX2H:3]>>[C:1](=[O:2])OC",
            "operation_idea": "replace a failed carboxylic acid component with a methyl ester or masked acid frontier",
            "risk_flags": ["masked_acid_frontier_hypothetical", "ester_exchange_direction_not_proven"],
        },
        {
            "rule_id": "failed_carboxylic_acid_to_mixed_anhydride_component",
            "smarts": "[C:1](=[O:2])[OX2H:3]>>[C:1](=[O:2])OC(C)=O",
            "operation_idea": "replace a failed carboxylic acid component with a mixed-anhydride-like activated acyl donor",
            "risk_flags": ["alternate_acyl_activation_state", "mixed_anhydride_scope_hypothetical"],
        },
        {
            "rule_id": "failed_acid_chloride_to_carboxylic_acid_component",
            "smarts": "[C:1](=[O:2])Cl>>[C:1](=[O:2])O",
            "operation_idea": "fall back from a failed acid chloride component to the carboxylic acid level",
            "risk_flags": ["alternate_acyl_activation_state", "leaving_group_choice_revised"],
        },
        {
            "rule_id": "failed_acid_chloride_to_methyl_ester_component",
            "smarts": "[C:1](=[O:2])Cl>>[C:1](=[O:2])OC",
            "operation_idea": "replace a failed acid chloride component with a methyl ester or masked acyl frontier",
            "risk_flags": ["masked_acid_frontier_hypothetical", "leaving_group_choice_revised"],
        },
        {
            "rule_id": "failed_enone_to_saturated_ketone_component",
            "smarts": "[C:1](=[O:4])[C:2]=[C:3]>>[C:1](=[O:4])[C:2][C:3]",
            "operation_idea": "step back from an enone frontier to the corresponding saturated ketone oxidation state",
            "risk_flags": ["enone_redox_state_hypothetical", "selectivity_not_encoded"],
        },
        {
            "rule_id": "failed_ketone_to_secondary_alcohol_component",
            "smarts": "[C:1](=[O:2])([#6:3])[#6:4]>>[CH:1]([O:2])([#6:3])[#6:4]",
            "operation_idea": "replace a failed ketone frontier with the corresponding secondary alcohol redox state",
            "risk_flags": ["redox_state_changed", "stereochemistry_not_encoded"],
        },
        {
            "rule_id": "failed_secondary_alcohol_to_ketone_component",
            "smarts": "[CH:1]([OX2H:2])([#6:3])[#6:4]>>[C:1](=[O:2])([#6:3])[#6:4]",
            "operation_idea": "replace a failed secondary alcohol frontier with the corresponding ketone redox state",
            "risk_flags": ["redox_state_changed", "oxidation_selectivity_hypothetical"],
        },
        {
            "rule_id": "failed_primary_alcohol_to_aldehyde_component",
            "smarts": "[CH2:1][OX2H:2]>>[CH:1]=[O:2]",
            "operation_idea": "replace a failed primary alcohol frontier with the corresponding aldehyde redox state",
            "risk_flags": ["redox_state_changed", "aldehyde_frontier_hypothetical"],
        },
        {
            "rule_id": "failed_aldehyde_to_primary_alcohol_component",
            "smarts": "[CH:1]=[O:2]>>[CH2:1][OX2H:2]",
            "operation_idea": "replace a failed aldehyde frontier with the corresponding primary alcohol redox state",
            "risk_flags": ["redox_state_changed", "forward_direction_not_proven"],
        },
        {
            "rule_id": "failed_alkyl_chloride_to_alcohol_component",
            "smarts": "[CX4:1][Cl:2]>>[C:1][OH]",
            "operation_idea": "replace a failed alkyl chloride frontier with the corresponding alcohol substitution state",
            "risk_flags": ["leaving_group_state_changed", "substitution_stereochemistry_not_encoded"],
        },
        {
            "rule_id": "failed_aliphatic_alcohol_to_alkyl_chloride_component",
            "smarts": "[CX4:1][OX2H:2]>>[C:1]Cl",
            "exclude_smarts": ["[CX3](=O)[OX2H]"],
            "operation_idea": "replace a failed aliphatic alcohol frontier with an alkyl chloride activation state",
            "risk_flags": ["leaving_group_state_changed", "substitution_stereochemistry_not_encoded"],
        },
        {
            "rule_id": "failed_alcohol_to_acetate_component",
            "smarts": "[OX2H:1]>>[O:1]C(C)=O",
            "exclude_smarts": ["[CX3](=O)[OX2H]"],
            "operation_idea": "protect or activate a failed alcohol or phenol component as the acetate",
            "risk_flags": ["protecting_group_choice_hypothetical", "component_protection_state_changed"],
        },
        {
            "rule_id": "failed_acetate_to_alcohol_component",
            "smarts": "[OX2:1]C(C)=O>>[OX2H:1]",
            "exclude_smarts": ["[CX3](=O)[OX2H]"],
            "operation_idea": "fall back from a failed acetate component to the free alcohol or phenol level",
            "risk_flags": ["deprotection_direction_hypothetical", "component_protection_state_changed"],
        },
    ]
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        if _rule_excluded(mol, [str(item) for item in rule.get("exclude_smarts") or []]):
            continue
        try:
            rxn = AllChem.ReactionFromSmarts(str(rule["smarts"]))
            products = rxn.RunReactants((mol,))
        except Exception:
            products = ()
        for product_set in products:
            product = _smiles_from_product_set(product_set)
            canonical = _canonical_smiles(product)
            if not canonical or canonical in seen or canonical == _canonical_smiles(smiles):
                continue
            seen.add(canonical)
            rows.append(
                {
                    "variant_type": str(rule["rule_id"]),
                    "smiles": canonical,
                    "operation_idea": str(rule["operation_idea"]),
                    "risk_flags": [str(item) for item in rule.get("risk_flags") or []],
                }
            )
    return rows


def _analogical_hypothesis_rows(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in blackboard.get("analogical_hypotheses") or []:
        if isinstance(raw, dict):
            rows.append(dict(raw))
    artifact = blackboard.get("analogical_retrosynthesis_hypotheses")
    if isinstance(artifact, dict):
        rows.extend(dict(raw) for raw in artifact.get("hypotheses") or [] if isinstance(raw, dict))
    ranking = blackboard.get("analogical_hypothesis_ranking")
    if isinstance(ranking, dict):
        rows.extend(dict(raw) for raw in ranking.get("ranked_hypotheses") or [] if isinstance(raw, dict))
        rows.extend(dict(raw) for raw in ranking.get("selected_hypotheses") or [] if isinstance(raw, dict))
    return _dedupe_rows(rows, key="hypothesis_id")


def _source_pair_transfer_classes(source_product: str, source_reactants: list[str]) -> list[str]:
    product = Chem.MolFromSmiles(str(source_product or ""))
    reactants = [Chem.MolFromSmiles(str(item or "")) for item in source_reactants]
    reactants = [mol for mol in reactants if mol is not None]
    if product is None or not reactants:
        return []
    classes: list[str] = []
    if _mol_has(product, "[C](=O)([#6])[#6]") and _any_mol_has(reactants, "[CH]([OX2H])([#6])[#6]"):
        classes.append("ketone_to_secondary_alcohol_precursor")
    if _mol_has(product, "[CH]([OX2H])([#6])[#6]") and _any_mol_has(reactants, "[C](=O)([#6])[#6]"):
        classes.append("secondary_alcohol_to_ketone_precursor")
    if _mol_has(product, "[CH2][OX2H]") and _any_mol_has(reactants, "[CH]=O"):
        classes.append("primary_alcohol_to_aldehyde_precursor")
    if _mol_has(product, "[CH]=O") and _any_mol_has(reactants, "[CH2][OX2H]"):
        classes.append("aldehyde_to_primary_alcohol_precursor")
    if _mol_has(product, "[C](=O)[C]=[C]") and _any_mol_has(reactants, "[C](=O)[C][C]"):
        classes.append("enone_to_saturated_ketone_precursor")
    if _mol_has(product, "[C](=O)[O][#6]"):
        if _any_mol_has(reactants, "[C](=O)[OX2H]") and _any_mol_has(reactants, "[OX2H]"):
            classes.append("ester_to_carboxylic_acid_alcohol_precursors")
        if _any_mol_has(reactants, "[C](=O)Cl") and _any_mol_has(reactants, "[OX2H]"):
            classes.append("ester_to_acid_chloride_alcohol_precursors")
    if _mol_has(product, "[CX4]Cl") and _any_mol_has(reactants, "[CX4][OX2H]"):
        classes.append("alkyl_chloride_to_alcohol_precursor")
    if _mol_has(product, "[CX4][OX2H]") and _any_mol_has(reactants, "[CX4]Cl"):
        classes.append("aliphatic_alcohol_to_alkyl_chloride_precursor")
    return _dedupe(classes)


def _analogical_pair_transfer_confidence(row: dict[str, Any]) -> str:
    value = str(row.get("analogy_strength") or row.get("confidence") or "").lower()
    if value in {"high", "medium_high"}:
        return "medium"
    if value == "medium":
        return "medium"
    return "low"


def _any_mol_has(mols: list[Chem.Mol], smarts: str) -> bool:
    return any(_mol_has(mol, smarts) for mol in mols)


def _mol_has(mol: Chem.Mol | None, smarts: str) -> bool:
    if mol is None:
        return False
    query = Chem.MolFromSmarts(str(smarts or ""))
    return bool(query is not None and mol.HasSubstructMatch(query))


def _rule_excluded(mol: Chem.Mol, exclude_smarts: list[str]) -> bool:
    for smarts in exclude_smarts:
        if not smarts:
            continue
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            return True
    return False


def _concrete_proposals_from_reaction_idea_cards(
    blackboard: dict[str, Any],
    cards: list[dict[str, Any]],
    *,
    max_variants_per_card: int = 6,
) -> list[dict[str, Any]]:
    """Concretize broad reaction ideas into bounded precursor SMILES.

    This is not template proof.  It is a deterministic bridge from advisory
    chemistry ideas to recursive search seeds when the current target has a
    common functional handle that can be moved one oxidation/protection state
    upstream or disconnected into a bounded multi-component precursor set.
    """
    target = dict(blackboard.get("target_profile") or {})
    fallback_target = str(target.get("target_smiles") or target.get("canonical_smiles") or "")
    out: list[dict[str, Any]] = []
    seen_precursor_variants: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        source_type = str(card.get("source_type") or "")
        if source_type in {"process_or_biotransformation", "exact_literature_row", "visual_connectivity", "analogical_reaction_pair"}:
            continue
        target_smiles = str(card.get("target_smiles") or fallback_target)
        variants = _deterministic_precursor_variants_from_target(target_smiles, limit=max_variants_per_card)
        for variant in variants:
            precursor = str(variant.get("precursor_smiles") or "")
            if not precursor:
                continue
            variant_key = "|".join([str(variant.get("variant_type") or ""), _canonical_smiles(precursor) or precursor])
            if variant_key in seen_precursor_variants:
                continue
            seen_precursor_variants.add(variant_key)
            output_source_type = (
                "analogical_reaction_idea_concretization"
                if source_type == "analogical_template_application"
                else "deterministic_reaction_idea_concretization"
            )
            analogy_risk_flags = (
                ["analogy_concretized_without_exact_precursor_hint", "analogy_not_proof"]
                if source_type == "analogical_template_application"
                else []
            )
            out.append(
                _proposal(
                    proposal_type="semi_executable",
                    source_type=output_source_type,
                    target_smiles=target_smiles,
                    precursor_smiles=precursor,
                    transformation_idea=_join_idea(
                        str(card.get("transformation_idea") or ""),
                        str(variant.get("operation_idea") or ""),
                    ),
                    proposal_label=str(variant.get("variant_type") or card.get("expected_precursor_type") or "concretized precursor"),
                    confidence=_concretized_confidence(str(card.get("confidence") or "low")),
                    evidence_refs=[str(card.get("card_id") or ""), *[str(item) for item in card.get("evidence_refs") or []]],
                    risk_flags=_dedupe(
                        [
                            *[str(item) for item in card.get("risk_flags") or []],
                            *[str(item) for item in variant.get("risk_flags") or []],
                            *analogy_risk_flags,
                            "deterministic_broad_transform_not_literature_exact",
                            "reaction_idea_concretization_requires_verifier",
                        ]
                    ),
                    required_verification=_dedupe(
                        [
                            *[str(item) for item in card.get("required_verification") or []],
                            "route_expansion_verifier",
                            "parent_bridge_connectivity",
                        ]
                    ),
                    executable=True,
                    recursive_expandable=True,
                    not_exact_literature_segment=True,
                )
            )
    return out


def _deterministic_precursor_variants_from_target(smiles: str, *, limit: int) -> list[dict[str, Any]]:
    rules = [
        {
            "rule_id": "amide_to_carboxylic_acid_amine_precursors",
            "smarts": "[C:1](=[O:2])[N:3]([#6:4])>>[C:1](=[O:2])O.[N:3][#6:4]",
            "operation_idea": "disconnect an amide into carboxylic acid plus amine precursor components",
            "risk_flags": ["acyl_disconnection_hypothetical", "multi_component_precursor_set", "amide_coupling_direction_not_proven"],
        },
        {
            "rule_id": "amide_to_acid_chloride_amine_precursors",
            "smarts": "[C:1](=[O:2])[N:3]([#6:4])>>[C:1](=[O:2])Cl.[N:3][#6:4]",
            "operation_idea": "try an activated acid chloride plus amine precursor set for an amide disconnection",
            "risk_flags": [
                "acyl_disconnection_hypothetical",
                "multi_component_precursor_set",
                "leaving_group_choice_hypothetical",
                "amide_coupling_direction_not_proven",
            ],
        },
        {
            "rule_id": "ester_to_carboxylic_acid_alcohol_precursors",
            "smarts": "[C:1](=[O:2])[O:3][#6:4]>>[C:1](=[O:2])O.[O:3][#6:4]",
            "operation_idea": "disconnect an ester into carboxylic acid plus alcohol or phenol precursor components",
            "risk_flags": ["acyl_disconnection_hypothetical", "multi_component_precursor_set", "esterification_direction_not_proven"],
        },
        {
            "rule_id": "ester_to_acid_chloride_alcohol_precursors",
            "smarts": "[C:1](=[O:2])[O:3][#6:4]>>[C:1](=[O:2])Cl.[O:3][#6:4]",
            "operation_idea": "try an activated acid chloride plus alcohol or phenol precursor set for an ester disconnection",
            "risk_flags": [
                "acyl_disconnection_hypothetical",
                "multi_component_precursor_set",
                "leaving_group_choice_hypothetical",
                "esterification_direction_not_proven",
            ],
        },
        {
            "rule_id": "carbonate_to_alcohol_chloroformate_precursors",
            "smarts": "[O:1][C:2](=[O:3])[O:4]>>[O:1].[Cl][C:2](=[O:3])[O:4]",
            "operation_idea": "disconnect a carbonate into alcohol plus chloroformate-like precursor components",
            "risk_flags": [
                "acyl_disconnection_hypothetical",
                "multi_component_precursor_set",
                "leaving_group_choice_hypothetical",
                "carbonate_scope_hypothetical",
            ],
        },
        {
            "rule_id": "enone_to_saturated_ketone_precursor",
            "smarts": "[C:1](=[O:4])[C:2]=[C:3]>>[C:1](=[O:4])[C:2][C:3]",
            "operation_idea": "move an enone one redox state upstream to a saturated ketone precursor",
            "risk_flags": ["enone_redox_state_hypothetical", "selectivity_not_encoded"],
        },
        {
            "rule_id": "ketone_to_secondary_alcohol_precursor",
            "smarts": "[C:1](=[O:2])([#6:3])[#6:4]>>[CH:1]([O:2])([#6:3])[#6:4]",
            "operation_idea": "move a ketone one redox state upstream to a secondary alcohol precursor",
            "risk_flags": ["redox_direction_hypothetical", "stereochemistry_not_encoded"],
        },
        {
            "rule_id": "secondary_alcohol_to_ketone_precursor",
            "smarts": "[CH:1]([OX2H:2])([#6:3])[#6:4]>>[C:1](=[O:2])([#6:3])[#6:4]",
            "operation_idea": "move a secondary alcohol one oxidation state upstream to a ketone precursor",
            "risk_flags": ["redox_direction_hypothetical", "oxidation_selectivity_hypothetical"],
        },
        {
            "rule_id": "alkyl_chloride_to_alcohol_precursor",
            "smarts": "[CX4:1][Cl:2]>>[C:1][OH]",
            "operation_idea": "try the corresponding alcohol before an alkyl chloride substitution frontier",
            "risk_flags": ["leaving_group_state_hypothetical", "substitution_stereochemistry_not_encoded"],
        },
        {
            "rule_id": "aliphatic_alcohol_to_alkyl_chloride_precursor",
            "smarts": "[CX4:1][OX2H:2]>>[C:1]Cl",
            "exclude_smarts": ["[CX3](=O)[OX2H]"],
            "operation_idea": "try an alkyl chloride activation state for an aliphatic alcohol frontier",
            "risk_flags": ["leaving_group_state_hypothetical", "substitution_stereochemistry_not_encoded"],
        },
        {
            "rule_id": "primary_alcohol_to_aldehyde_precursor",
            "smarts": "[CH2:1][OX2H:2]>>[CH:1]=[O:2]",
            "operation_idea": "move a primary alcohol one oxidation state upstream to an aldehyde precursor",
            "risk_flags": ["redox_direction_hypothetical", "selectivity_not_encoded"],
        },
        {
            "rule_id": "alcohol_to_acetate_protected_precursor",
            "smarts": "[OX2H:1]>>[O:1]C(C)=O",
            "operation_idea": "try an acetate-protected alcohol precursor before recursive expansion",
            "risk_flags": ["protecting_group_choice_hypothetical"],
        },
        {
            "rule_id": "acetate_to_free_alcohol_precursor",
            "smarts": "[OX2:1]C(C)=O>>[OX2H:1]",
            "operation_idea": "remove acetate protection and continue from the free alcohol frontier",
            "risk_flags": ["deprotection_direction_hypothetical"],
        },
        {
            "rule_id": "aldehyde_to_primary_alcohol_precursor",
            "smarts": "[CH:1]=[O:2]>>[CH2:1][OX2H:2]",
            "operation_idea": "move an aldehyde one oxidation state downstream to an alcohol precursor hypothesis",
            "risk_flags": ["redox_direction_hypothetical", "forward_direction_not_proven"],
        },
    ]
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        if _rule_excluded(mol, [str(item) for item in rule.get("exclude_smarts") or []]):
            continue
        rxn = AllChem.ReactionFromSmarts(str(rule["smarts"]))
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            products = ()
        for product_set in products:
            if not product_set:
                continue
            precursor = _smiles_from_product_set(product_set)
            if not precursor:
                continue
            canonical = _canonical_smiles(precursor)
            if not canonical or canonical == _canonical_smiles(smiles) or canonical in seen:
                continue
            seen.add(canonical)
            rows.append(
                {
                    "variant_type": str(rule["rule_id"]),
                    "precursor_smiles": canonical,
                    "operation_idea": str(rule["operation_idea"]),
                    "risk_flags": [str(item) for item in rule.get("risk_flags") or []],
                }
            )
            if len(rows) >= max(1, int(limit or 1)):
                return rows
    return rows


def _smiles_from_product_set(product_set: tuple[Chem.Mol, ...]) -> str:
    parts: list[str] = []
    for product in product_set:
        try:
            Chem.SanitizeMol(product)
        except Exception:
            return ""
        text = Chem.MolToSmiles(product, isomericSmiles=True)
        if text:
            parts.append(text)
    if not parts:
        return ""
    return ".".join(parts)


def _join_idea(primary: str, secondary: str) -> str:
    first = str(primary or "").strip()
    second = str(secondary or "").strip()
    if first and second:
        return f"{first}; {second}"
    return first or second


def _concretized_confidence(card_confidence: str) -> str:
    text = str(card_confidence or "").lower()
    if text in {"high", "medium_high"}:
        return "medium"
    if text == "medium":
        return "medium"
    return "low"


def _strategic_proposals_from_cards(blackboard: dict[str, Any], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = dict(blackboard.get("target_profile") or {})
    out: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        if str(card.get("source_type") or "") in {"visual_connectivity", "exact_literature_row", "analogical_template_application"}:
            continue
        out.append(
            _proposal(
                proposal_type="strategic",
                source_type=str(card.get("source_type") or "reaction_idea"),
                target_smiles=str(card.get("target_smiles") or target.get("target_smiles") or ""),
                precursor_smiles="",
                transformation_idea=str(card.get("transformation_idea") or ""),
                proposal_label=str(card.get("expected_precursor_type") or card.get("target_handle") or "strategic proposal"),
                confidence=str(card.get("confidence") or "low"),
                evidence_refs=[str(card.get("card_id") or ""), *[str(item) for item in card.get("evidence_refs") or []]],
                risk_flags=[*[str(item) for item in card.get("risk_flags") or []], "strategic_not_executable"],
                required_verification=[str(item) for item in card.get("required_verification") or ["proposal_compilation_to_precursor"]],
                executable=False,
                recursive_expandable=False,
                not_exact_literature_segment=True,
            )
        )
    return out


def _idea_card(
    *,
    card_id: str,
    source_type: str,
    target_handle: str,
    transformation_idea: str,
    preserved_core: list[Any],
    expected_precursor_type: str,
    evidence_refs: list[str],
    confidence: str,
    risk_flags: list[str],
    required_verification: list[str],
    target_smiles: str,
) -> dict[str, Any]:
    return {
        "schema_version": REACTION_IDEA_CARD_SCHEMA,
        "card_id": card_id,
        "source_type": source_type,
        "target_smiles": target_smiles,
        "target_handle": target_handle,
        "transformation_idea": transformation_idea,
        "must_preserve_substructure": _dedupe([str(item) for item in preserved_core if str(item).strip()]),
        "expected_precursor_type": expected_precursor_type,
        "evidence_refs": _dedupe([str(item) for item in evidence_refs if str(item).strip()]),
        "confidence": confidence or "low",
        "risk_flags": _dedupe([str(item) for item in risk_flags if str(item).strip()]),
        "required_verification": _dedupe([str(item) for item in required_verification if str(item).strip()]),
        "allowed_use": "proposal_compilation_and_search_policy_hint_only",
        "not_exact_literature_segment": source_type != "exact_literature_row",
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }


def _proposal(
    *,
    proposal_type: str,
    source_type: str,
    target_smiles: str,
    precursor_smiles: str,
    transformation_idea: str,
    proposal_label: str,
    confidence: str,
    evidence_refs: list[str],
    risk_flags: list[str],
    required_verification: list[str],
    executable: bool,
    recursive_expandable: bool,
    not_exact_literature_segment: bool,
) -> dict[str, Any]:
    canonical_target = _canonical_smiles(target_smiles)
    canonical_precursor = _canonical_smiles(precursor_smiles)
    precursor_component_count = len([part for part in canonical_precursor.split(".") if part]) if canonical_precursor else 0
    proposal_granularity = _proposal_granularity(
        proposal_type=proposal_type,
        source_type=source_type,
        risk_flags=risk_flags,
        not_exact_literature_segment=not_exact_literature_segment,
        has_precursor=bool(canonical_precursor),
    )
    route_objective_type = _route_objective_type_for_proposal(
        proposal_granularity=proposal_granularity,
        source_type=source_type,
        risk_flags=risk_flags,
    )
    proposal_id = "proposal:" + _short_hash(
        "|".join([proposal_type, source_type, canonical_target or target_smiles, canonical_precursor or precursor_smiles, transformation_idea])
    )
    return {
        "schema_version": RETROSYNTHETIC_PROPOSAL_SCHEMA,
        "proposal_id": proposal_id,
        "proposal_type": proposal_type,
        "source_type": source_type,
        "proposal_label": proposal_label,
        "reaction_family": proposal_label,
        "reaction_families": _dedupe([proposal_label]),
        "product_retron_type": "",
        "product_retron_types": [],
        "derived_from_retron": "",
        "retron_authority": "not_supplied",
        "target_smiles": canonical_target or target_smiles,
        "precursor_smiles": canonical_precursor or precursor_smiles,
        "precursor_component_count": precursor_component_count,
        "multi_component_precursor_set": precursor_component_count > 1,
        "transformation_idea": transformation_idea,
        "proposal_granularity": proposal_granularity,
        "route_objective_type": route_objective_type,
        "failure_response_policy": _failure_response_policy_for_proposal(
            proposal_granularity=proposal_granularity,
            route_objective_type=route_objective_type,
            multi_component_precursor_set=precursor_component_count > 1,
        ),
        "confidence": confidence or "low",
        "score": _proposal_score(
            proposal_type,
            confidence,
            risk_flags,
            bool(canonical_precursor),
            proposal_granularity=proposal_granularity,
            route_objective_type=route_objective_type,
        ),
        "executable": bool(executable and canonical_precursor),
        "recursive_expandable": bool(recursive_expandable and canonical_precursor),
        "evidence_refs": _dedupe([str(item) for item in evidence_refs if str(item).strip()]),
        "risk_flags": _dedupe([str(item) for item in risk_flags if str(item).strip()]),
        "required_verification": _dedupe([str(item) for item in required_verification if str(item).strip()]),
        "allowed_use": "recursive_search_seed_and_guided_policy_hint_only",
        "not_exact_literature_segment": bool(not_exact_literature_segment),
        "not_parent_route_proof": True,
        "requires_verifier": True,
        "child_route_cannot_promote_parent": True,
        "no_solved_claim": True,
    }


def _proposal_granularity(
    *,
    proposal_type: str,
    source_type: str,
    risk_flags: list[str],
    not_exact_literature_segment: bool,
    has_precursor: bool,
) -> str:
    source = str(source_type or "").lower()
    flags = " ".join(str(item or "").lower() for item in risk_flags)
    if str(proposal_type or "") == "exact_executable" or (not not_exact_literature_segment and has_precursor):
        return "exact"
    if "process" in source or "biotransformation" in source or "feedstock" in source or "semisynthesis_anchor" in source:
        return "process"
    if "same_core" in flags or "semisynthesis" in flags or "core" in flags or "redox" in flags or "protection" in flags:
        return "same_core"
    if "analogical" in source or "broad" in source or "template" in source or "mechanism" in flags:
        return "mechanism"
    if str(proposal_type or "") == "strategic":
        return "process" if "process" in source or "objective" in source else "fallback"
    return "fallback"


def _route_objective_type_for_proposal(
    *,
    proposal_granularity: str,
    source_type: str,
    risk_flags: list[str],
) -> str:
    source = str(source_type or "").lower()
    flags = " ".join(str(item or "").lower() for item in risk_flags)
    if proposal_granularity == "exact":
        return "literature_exact_frontier"
    if proposal_granularity == "process":
        if "feedstock" in source:
            return "process_feedstock_route"
        if "semisynthesis" in source:
            return "semisynthesis_anchor_route"
        return "process_or_biotransformation_route"
    if proposal_granularity == "same_core":
        return "same_core_redox_or_protection_route"
    if proposal_granularity == "mechanism":
        return "mechanism_transfer_route"
    if "multi_component" in flags:
        return "fragment_coupling_hypothesis_route"
    return "hypothesis_route"


def _failure_response_policy_for_proposal(
    *,
    proposal_granularity: str,
    route_objective_type: str,
    multi_component_precursor_set: bool,
) -> dict[str, Any]:
    policy = {
        "schema_version": "proposal_failure_response_policy.v1",
        "on_no_route": "expand_precursor_recursively",
        "on_verifier_rejection": "create_failure_feedback_and_try_neighboring_frontier",
        "on_large_atom_jump": "coarsen_to_same_core_or_process_anchor",
        "on_structure_uncertainty": "allow_stereo_relaxed_hypothesis_without_proof",
        "on_repeated_failure": "change_granularity_or_stop_branch",
        "no_solved_claim": True,
    }
    if proposal_granularity == "exact":
        policy["on_no_route"] = "try_upstream_terminal_synthesis_before_relaxing_to_same_core"
    elif proposal_granularity == "process":
        policy["on_no_route"] = "validate_feedstock_or_semisynthesis_anchor_before_small_molecule_closure"
        policy["on_large_atom_jump"] = "preserve_process_endpoint_and_search_feedstock_bridge"
    elif proposal_granularity == "mechanism":
        policy["on_no_route"] = "relax_reaction_center_or_try_same_core_precursor"
    elif proposal_granularity == "fallback":
        policy["on_no_route"] = "try_one_recursive_depth_then_lower_priority"
    if multi_component_precursor_set:
        policy["on_component_failure"] = "replace_failed_component_activation_state_and_keep_siblings"
    policy["route_objective_type"] = route_objective_type
    return policy


def _proposal_score(
    proposal_type: str,
    confidence: str,
    risk_flags: list[str],
    has_precursor: bool,
    *,
    proposal_granularity: str = "fallback",
    route_objective_type: str = "",
) -> int:
    score = {"exact_executable": 80, "semi_executable": 60, "strategic": 35}.get(proposal_type, 20)
    score += {"high": 15, "medium_high": 12, "medium": 8, "low": 0}.get(str(confidence or "").lower(), 0)
    score += {"exact": 18, "process": 14, "same_core": 12, "mechanism": 6, "fallback": 0}.get(
        str(proposal_granularity or "fallback"),
        0,
    )
    if str(route_objective_type or "").startswith(("semisynthesis", "process", "same_core")):
        score += 5
    if has_precursor:
        score += 10
    score -= min(25, 3 * len([item for item in risk_flags if str(item).strip()]))
    return score


def _proposal_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    return (-int(row.get("score") or 0), str(row.get("proposal_id") or ""))


def _existing_recursive_frontier(blackboard: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for task in blackboard.get("recursive_hypothesis_tasks") or []:
        if not isinstance(task, dict):
            continue
        smiles = _canonical_smiles(str(task.get("precursor_smiles") or ""))
        if smiles:
            out.add(smiles)
    return out


def _dedupe_rows(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        marker = str(raw.get(key) or "").strip()
        if not marker:
            marker = _short_hash(str(raw))
        if marker in seen:
            continue
        seen.add(marker)
        out.append(raw)
    return out


def _canonical_smiles(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _precursor_components(smiles: str) -> list[str]:
    canonical = _canonical_smiles(smiles)
    if not canonical:
        return []
    return _dedupe([part for part in canonical.split(".") if part])


def _confidence_from_risk_flags(risk_flags: list[str]) -> str:
    flags = [str(item).lower() for item in risk_flags]
    if any("visual" in item or "stereo" in item or "broad" in item for item in flags):
        return "low"
    if flags:
        return "medium"
    return "medium_high"


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, list):
            text = _first_text(*value)
            if text:
                return text
            continue
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _smiles_values(*values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            if isinstance(candidate, dict):
                text = _first_text(candidate.get("smiles"), candidate.get("canonical_smiles"), candidate.get("target_smiles"))
            else:
                text = str(candidate or "").strip()
            canonical = _canonical_smiles(text)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            out.append(canonical)
    return out


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _short_hash(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:12]
