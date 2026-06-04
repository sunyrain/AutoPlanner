"""Hybrid route-package assembly and audit for SMILES-first P0 workflow."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.agent.evidence_cards import EvidenceCard, validate_evidence_card
from cascade_planner.agent.strategic_candidate_generation import (
    LiteratureCandidate,
    validate_literature_candidate,
)
from cascade_planner.agent.target_profile import TargetProfile


RDLogger.DisableLog("rdApp.*")
ROUTE_PACKAGE_SCHEMA = "hybrid_route_package.v1"
ROUTE_PACKAGE_VALIDATION_SCHEMA = "route_package_validation.v1"


def build_hybrid_route_package(
    *,
    profile: TargetProfile,
    frontier_report: dict[str, Any],
    evidence_cards: list[EvidenceCard],
    candidates: list[LiteratureCandidate],
    baseline_routes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frontier = _primary_frontier(frontier_report, profile)
    candidate_rows = [candidate.to_dict() for candidate in candidates]
    route_status = _minimal_route_status(evidence_cards, candidates)
    return {
        "schema_version": ROUTE_PACKAGE_SCHEMA,
        "case_id": profile.case_id,
        "target": {
            "name": profile.target_name,
            "smiles": profile.isomeric_smiles or profile.input_smiles,
            "profile_ref": "target_profile.json",
        },
        "baseline": _baseline_section(baseline_routes or {}),
        "frontier": frontier,
        "literature_evidence_refs": [card.evidence_id for card in evidence_cards],
        "literature_candidates": candidate_rows,
        "strategy_templates": [
            candidate.strategy_template
            for candidate in candidates
            if candidate.strategy_template
        ],
        "route_graph": _route_graph(profile, frontier, candidates),
        "route_status": route_status,
        "status_contract": (
            "P0 route package is planning material. It cannot be solved without "
            "separate stock closure and product/route audit proof."
        ),
    }


def validate_route_package(
    package: dict[str, Any],
    *,
    evidence_cards: list[EvidenceCard],
    candidates: list[LiteratureCandidate],
) -> dict[str, Any]:
    evidence_by_id = {card.evidence_id: card for card in evidence_cards}
    evidence_results = [validate_evidence_card(card) for card in evidence_cards]
    candidate_results = [validate_literature_candidate(candidate) for candidate in candidates]
    reasons: list[str] = []

    target_smiles = ((package.get("target") or {}).get("smiles") or "")
    if not _valid_smiles(target_smiles):
        reasons.append("invalid_package_target_smiles")
    if package.get("route_status") == "solved":
        reasons.append("p0_package_must_not_claim_solved")
    if not evidence_cards:
        reasons.append("missing_evidence_cards")
    if not candidates and not any(item["accepted"] for item in evidence_results):
        reasons.append("literature_gap_no_candidates")

    for result in evidence_results:
        if not result["accepted"]:
            reasons.append(f"evidence_rejected:{result['evidence_id']}")
    for candidate, result in zip(candidates, candidate_results):
        if not result["accepted"]:
            reasons.append(f"candidate_rejected:{candidate.candidate_id}")
        for evidence_ref in candidate.evidence_refs:
            if evidence_ref not in evidence_by_id:
                reasons.append(f"candidate_missing_evidence_ref:{candidate.candidate_id}:{evidence_ref}")
        if candidate.candidate_kind == "route_anchor" and candidate.rxn_smiles:
            reasons.append(f"route_anchor_has_rxn:{candidate.candidate_id}")
        if candidate.candidate_kind == "forward_surrogate" and not candidate.not_lab_procedure:
            reasons.append(f"surrogate_missing_not_lab_procedure:{candidate.candidate_id}")

    all_analogy = bool(evidence_cards) and all(card.target_relation == "analogy_only" for card in evidence_cards)
    if all_analogy and package.get("route_status") == "ready_for_guided_rerun":
        reasons.append("analogy_only_evidence_cannot_guided_rerun")

    route_status = package.get("route_status") or "invalid_package"
    if reasons:
        route_status = "literature_gap" if "missing_evidence_cards" in reasons else "invalid_package"
    return {
        "schema_version": ROUTE_PACKAGE_VALIDATION_SCHEMA,
        "case_id": package.get("case_id"),
        "accepted": not reasons,
        "route_status": route_status,
        "reasons": sorted(set(reasons)),
        "evidence_validation": evidence_results,
        "candidate_validation": candidate_results,
        "guards": {
            "route_anchor_not_stock": True,
            "forward_surrogate_not_lab_procedure": True,
            "p0_not_solved_without_stock_audit": True,
        },
    }


def render_summary(package: dict[str, Any], validation: dict[str, Any]) -> str:
    target = package.get("target") or {}
    frontier = package.get("frontier") or {}
    lines = [
        "# SMILES-First Literature Route Package Summary",
        "",
        f"- Case: `{package.get('case_id')}`",
        f"- Target: `{target.get('name') or ''}` `{target.get('smiles') or ''}`",
        f"- Frontier: `{frontier.get('frontier_smiles') or ''}`",
        f"- Route status: `{validation.get('route_status') or package.get('route_status')}`",
        f"- Accepted: `{bool(validation.get('accepted'))}`",
        "",
        "## Candidate Types",
    ]
    counts: dict[str, int] = {}
    for candidate in package.get("literature_candidates") or []:
        kind = candidate.get("candidate_kind")
        counts[kind] = counts.get(kind, 0) + 1
    for kind in ("exact_fragment_retro", "forward_surrogate", "route_anchor"):
        lines.append(f"- `{kind}`: {counts.get(kind, 0)}")

    lines.extend(["", "## Strategic Candidates"])
    for candidate in package.get("literature_candidates") or []:
        lines.append(
            f"- `{candidate.get('candidate_id')}` `{candidate.get('candidate_kind')}` "
            f"{candidate.get('reaction_class') or ''}"
        )
        if candidate.get("strategic_bond"):
            lines.append(f"  - Strategic bond: {candidate.get('strategic_bond')}")
        if candidate.get("literature_basis"):
            lines.append(f"  - Literature basis: {candidate.get('literature_basis')}")
        if candidate.get("candidate_kind") == "forward_surrogate":
            lines.append("  - Guard: not a lab procedure; planning surrogate only.")
        if candidate.get("candidate_kind") == "route_anchor":
            lines.append("  - Guard: multi-step anchor, not stock closure.")

    lines.extend(["", "## Validation"])
    if validation.get("reasons"):
        for reason in validation["reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- P0 package validation passed.")
    lines.extend([
        "",
        "## Next Step",
        "- Review evidence cards and candidate templates before compiling any guided ChemEnzy policy.",
    ])
    return "\n".join(lines) + "\n"


def render_route_map_svg(package: dict[str, Any]) -> str:
    nodes = _route_map_nodes(package)
    width = 1180
    height = 130 + len(nodes) * 104
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        '<polygon points="0 0, 10 3.5, 0 7" fill="#334155"/>',
        "</marker>",
        "<style>",
        ".title{font:700 22px Arial;fill:#111827}.label{font:700 14px Arial;fill:#111827}",
        ".body{font:12px Arial;fill:#475569}.box{fill:#fff;stroke:#cbd5e1;stroke-width:1.2;rx:4}",
        ".guard{fill:#fff7ed;stroke:#fdba74}.arrow{stroke:#334155;stroke-width:1.8;marker-end:url(#arrowhead)}",
        "</style></defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="40" y="38" class="title">{_esc("SMILES-first hybrid route map")}</text>',
        f'<text x="40" y="62" class="body">{_esc(str(package.get("status_contract") or ""))}</text>',
    ]
    x = 64
    y = 96
    box_w = 1000
    box_h = 72
    for idx, node in enumerate(nodes):
        cls = "box guard" if node.get("guard") else "box"
        chunks.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{box_w}" height="{box_h}"/>')
        chunks.append(f'<text x="{x + 18}" y="{y + 26}" class="label">{_esc(node["label"])}</text>')
        chunks.append(f'<text x="{x + 18}" y="{y + 50}" class="body">{_esc(node["body"][:150])}</text>')
        if idx < len(nodes) - 1:
            y2 = y + box_h + 22
            chunks.append(f'<line class="arrow" x1="{x + box_w / 2}" y1="{y + box_h}" x2="{x + box_w / 2}" y2="{y2}"/>')
        y += 104
    chunks.append("</svg>")
    return "\n".join(chunks)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _primary_frontier(frontier_report: dict[str, Any], profile: TargetProfile) -> dict[str, Any]:
    frontiers = frontier_report.get("frontiers") or []
    if frontiers:
        return frontiers[0]
    return {
        "frontier_smiles": profile.isomeric_smiles or profile.input_smiles,
        "source": "target_as_frontier",
        "flags": ["no_baseline_frontier_available"],
    }


def _baseline_section(baseline_routes: dict[str, Any]) -> dict[str, Any]:
    if not baseline_routes:
        return {
            "status": "not_run",
            "ordinary_steps": [],
            "note": "P0 fallback can proceed with manual frontier.",
        }
    return {
        "status": baseline_routes.get("status") or "provided",
        "ordinary_steps": baseline_routes.get("ordinary_steps") or [],
        "route_count": len(baseline_routes.get("routes") or []),
        "solved": bool(baseline_routes.get("solved")),
    }


def _route_graph(profile: TargetProfile, frontier: dict[str, Any], candidates: list[LiteratureCandidate]) -> dict[str, Any]:
    nodes = [
        {"id": "target", "kind": "target", "smiles": profile.isomeric_smiles or profile.input_smiles},
        {"id": "frontier", "kind": "advanced_frontier", "smiles": frontier.get("frontier_smiles")},
    ]
    edges = [{"from": "target", "to": "frontier", "role": "ordinary_or_manual_frontier"}]
    for candidate in candidates:
        nodes.append({
            "id": candidate.candidate_id,
            "kind": candidate.candidate_kind,
            "smiles": candidate.product_smiles,
            "guard": candidate.candidate_kind in {"forward_surrogate", "route_anchor"},
        })
        edges.append({"from": "frontier", "to": candidate.candidate_id, "role": candidate.candidate_kind})
    return {"nodes": nodes, "edges": edges}


def _minimal_route_status(evidence_cards: list[EvidenceCard], candidates: list[LiteratureCandidate]) -> str:
    if not evidence_cards:
        return "literature_gap"
    if not candidates:
        return "literature_gap"
    if any(candidate.candidate_kind == "route_anchor" for candidate in candidates):
        return "partial_anchor"
    return "ready_for_guided_rerun"


def _route_map_nodes(package: dict[str, Any]) -> list[dict[str, Any]]:
    target = package.get("target") or {}
    frontier = package.get("frontier") or {}
    nodes = [
        {
            "label": "Target",
            "body": f"{target.get('name') or ''} {target.get('smiles') or ''}",
        },
        {
            "label": "Ordinary planning / manual frontier",
            "body": f"frontier={frontier.get('frontier_smiles') or ''}; flags={frontier.get('flags') or []}",
        },
    ]
    for candidate in package.get("literature_candidates") or []:
        nodes.append({
            "label": f"{candidate.get('candidate_kind')} - {candidate.get('reaction_class') or ''}",
            "body": candidate.get("literature_basis") or candidate.get("candidate_id") or "",
            "guard": candidate.get("candidate_kind") in {"forward_surrogate", "route_anchor"},
        })
    nodes.append({
        "label": "P0 validation",
        "body": f"route_status={package.get('route_status')}; no solved claim without stock/audit proof",
        "guard": True,
    })
    return nodes


def _valid_smiles(smiles: str) -> bool:
    return bool(smiles and Chem.MolFromSmiles(str(smiles)) is not None)


def _esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)
