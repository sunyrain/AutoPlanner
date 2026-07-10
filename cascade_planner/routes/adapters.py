"""Adapters that bring blackboard route channels into the canonical consensus domain."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.routes.graph import assemble_route_consensus_graph, make_route_consensus_expansion


def rebuild_consensus_graph_from_blackboard(
    blackboard: Mapping[str, Any],
    *,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Fuse Codex, planner, ChemEnzy, template, and exact-row proposal records."""
    board = dict(blackboard)
    target_profile = dict(board.get("target_profile") or {})
    target_smiles = str(
        target_profile.get("target_smiles")
        or target_profile.get("canonical_smiles")
        or (board.get("target") or {}).get("smiles")
        or ""
    )
    case_id = str(board.get("case_id") or "")
    candidates = [
        *_candidates_from_consensus(dict(board.get("route_consensus") or {})),
        *_candidates_from_legacy_proposals(board.get("retrosynthetic_proposals") or [], target_smiles=target_smiles),
        *_candidates_from_exact_rows(
            (board.get("literature_evidence") or {}).get("exact_rows")
            or (board.get("literature_evidence") or {}).get("exact_literature_rows")
            or [],
            target_smiles=target_smiles,
        ),
    ]
    consensus = fuse_route_candidates(candidates, case_id=case_id, target_smiles=target_smiles)
    team = dict(board.get("codex_agent_team") or {})
    prior_expansions = [
        dict(row)
        for row in team.get("route_consensus_expansions") or []
        if isinstance(row, Mapping)
    ]
    root_expansion = make_route_consensus_expansion(
        consensus,
        requested_product_smiles=target_smiles,
        consensus_ref="blackboard:route_consensus_fused",
        agent_run_ref=str((team.get("coordinator") or {}).get("run_record_ref") or ""),
        depth=0,
    )
    expansions = [
        root_expansion,
        *[
            row
            for row in prior_expansions
            if int(row.get("depth") or 0) > 0
        ],
    ]
    graph = assemble_route_consensus_graph(
        expansions,
        case_id=case_id,
        target_smiles=target_smiles,
        max_depth=max(1, int(max_depth or 1)),
    )
    return {
        "schema_version": "blackboard_route_consensus_rebuild.v1",
        "accepted": bool(consensus.get("accepted")),
        "candidate_count": len(candidates),
        "consensus": consensus,
        "expansions": expansions,
        "graph": graph,
        "semantics": {
            "advisory_only": True,
            "no_solved_claim": True,
            "deterministic_parent_proof_required": True,
        },
    }


def _candidates_from_consensus(consensus: dict[str, Any]) -> list[dict[str, Any]]:
    if consensus.get("schema_version") != "route_consensus.v1":
        return []
    candidates: list[dict[str, Any]] = []
    for proposal in consensus.get("proposals") or []:
        if not isinstance(proposal, Mapping):
            continue
        proposal = dict(proposal)
        source_records = [dict(row) for row in proposal.get("source_records") or [] if isinstance(row, Mapping)]
        if not source_records:
            source_records = [
                {
                    "candidate_id": str(proposal.get("consensus_id") or "consensus"),
                    "source_channel": "other",
                    "evidence_level": str(proposal.get("evidence_level") or "model_only"),
                    "confidence": str(proposal.get("confidence") or "low"),
                    "source_refs": list(proposal.get("source_refs") or []),
                    "evidence_refs": list(proposal.get("evidence_refs") or []),
                    "report_ref": "",
                }
            ]
        for record in source_records:
            candidates.append(
                _candidate(
                    candidate_id=str(record.get("candidate_id") or proposal.get("consensus_id") or "consensus"),
                    product_smiles=str(proposal.get("product_smiles") or ""),
                    precursor_smiles=list(proposal.get("precursor_smiles") or []),
                    reaction_family=str(proposal.get("reaction_family") or "unspecified"),
                    rationale=" | ".join(str(value) for value in proposal.get("rationales") or []),
                    source_channel=str(record.get("source_channel") or "other"),
                    source_refs=list(record.get("source_refs") or []),
                    evidence_refs=list(record.get("evidence_refs") or []),
                    evidence_level=_safe_evidence_level(record.get("evidence_level")),
                    confidence=str(record.get("confidence") or proposal.get("confidence") or "low"),
                    conditions=list(proposal.get("conditions") or []),
                    catalyst="",
                    enzyme="",
                    limitations=list(proposal.get("limitations") or []),
                    required_validation=list(proposal.get("required_validation") or []),
                    report_ref=str(record.get("report_ref") or ""),
                )
            )
    return candidates


def _candidates_from_legacy_proposals(values: Any, *, target_smiles: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(values if isinstance(values, list) else []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if str(row.get("source_type") or "") == "multi_source_consensus" or str(row.get("proposal_id") or "").startswith("consensus:"):
            continue
        precursor = row.get("precursor_smiles") or row.get("precursor_set_smiles") or row.get("precursors")
        if not precursor:
            continue
        source_channel = _legacy_source_channel(row)
        evidence_level = _legacy_evidence_level(row, source_channel=source_channel)
        candidates.append(
            _candidate(
                candidate_id=str(row.get("proposal_id") or f"legacy:{index}"),
                product_smiles=str(row.get("product_smiles") or row.get("target_smiles") or target_smiles),
                precursor_smiles=precursor if isinstance(precursor, list) else [str(precursor)],
                reaction_family=str(row.get("proposal_label") or row.get("reaction_family") or row.get("proposal_type") or "unspecified"),
                rationale=str(row.get("transformation_idea") or row.get("transformation_rationale") or ""),
                source_channel=source_channel,
                source_refs=list(row.get("source_refs") or []),
                evidence_refs=list(row.get("evidence_refs") or []),
                evidence_level=evidence_level,
                confidence=str(row.get("confidence") or "low"),
                conditions=list(row.get("conditions") or []),
                catalyst=str(row.get("catalyst") or ""),
                enzyme=str(row.get("enzyme") or ""),
                limitations=list(row.get("risk_flags") or row.get("limitations") or []),
                required_validation=list(row.get("required_verification") or row.get("required_validation") or []),
                report_ref=str(row.get("artifact_ref") or ""),
            )
        )
    return candidates


def _candidates_from_exact_rows(values: Any, *, target_smiles: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(values if isinstance(values, list) else []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if row.get("accepted") is not True and row.get("validated") is not True and row.get("validation_status") != "accepted":
            continue
        product = row.get("product_smiles") or row.get("products") or row.get("product")
        reactants = row.get("reactant_smiles") or row.get("reactants") or row.get("main_reactant_smiles")
        if isinstance(product, list):
            product = product[0] if product else ""
        if not product or not reactants:
            continue
        refs = [
            str(value)
            for value in [row.get("source_ref"), *(row.get("source_refs") or []), *(row.get("evidence_refs") or [])]
            if str(value or "").strip()
        ]
        if not refs:
            continue
        candidates.append(
            _candidate(
                candidate_id=str(row.get("step_id") or row.get("row_id") or f"exact:{index}"),
                product_smiles=str(product or target_smiles),
                precursor_smiles=reactants if isinstance(reactants, list) else [str(reactants)],
                reaction_family=str(row.get("reaction_family") or row.get("reaction_class") or "literature exact step"),
                rationale="validated exact literature row",
                source_channel="literature_exact",
                source_refs=refs,
                evidence_refs=list(row.get("evidence_refs") or []),
                evidence_level="literature_exact",
                confidence="high",
                conditions=[],
                catalyst="",
                enzyme="",
                limitations=[],
                required_validation=["parent_route_connection"],
                report_ref=str(row.get("artifact_ref") or ""),
            )
        )
    return candidates


def _candidate(**values: Any) -> dict[str, Any]:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": str(values["candidate_id"]),
        "product_smiles": str(values["product_smiles"]),
        "precursor_smiles": list(values["precursor_smiles"]),
        "reaction_family": str(values["reaction_family"]),
        "transformation_rationale": str(values["rationale"]),
        "source_channel": str(values["source_channel"]),
        "source_refs": list(values["source_refs"]),
        "evidence_refs": list(values["evidence_refs"]),
        "evidence_level": str(values["evidence_level"]),
        "confidence": str(values["confidence"]),
        "conditions": list(values["conditions"]),
        "catalyst": str(values["catalyst"]),
        "enzyme": str(values["enzyme"]),
        "limitations": list(values["limitations"]),
        "required_validation": list(values["required_validation"]),
        "report_ref": str(values["report_ref"]),
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def _legacy_source_channel(row: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("source_type") or ""),
            str(row.get("proposal_type") or ""),
            str(row.get("origin") or ""),
        ]
    ).lower()
    if "chem" in text or "enzyme" in text:
        return "chem_enzy"
    if "literature" in text or "exact" in text:
        return "literature_analogy"
    if "template" in text:
        return "template"
    if "stock" in text:
        return "stock"
    if "human" in text:
        return "human"
    return "other"


def _legacy_evidence_level(row: dict[str, Any], *, source_channel: str) -> str:
    if source_channel == "chem_enzy":
        return "computational"
    if source_channel == "template":
        return "analogy"
    if source_channel == "literature_analogy":
        return "analogy"
    return "model_only"


def _safe_evidence_level(value: Any) -> str:
    text = str(value or "model_only")
    # Rebuilding is not a validation authority. Preserve trusted lower levels
    # and downgrade any historical self-claimed validation.
    return "computational" if text == "validated" else text
