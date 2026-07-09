"""Compile hypothesis-only retrosynthesis sketches from an agent blackboard."""
from __future__ import annotations

import hashlib
from typing import Any

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def compile_hypothesis_only_retrosynthesis_report(
    *,
    blackboard: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    max_candidates: int = 12,
) -> dict[str, Any]:
    """Return advisory precursor sketches without creating a solved claim.

    The report intentionally stores candidate molecules and route ideas, not
    executable reaction SMILES. It is useful when exact literature rows or
    stereochemical structure recovery are incomplete but the run has gathered
    enough analogy/template evidence to propose a chemist-reviewable path.
    """
    del artifacts
    target = dict(blackboard.get("target_profile") or {})
    case_id = str(blackboard.get("case_id") or target.get("target_name") or "case")
    target_smiles = str(target.get("target_smiles") or "")
    candidates = _dedupe_candidates(
        [
            *_candidates_from_template_applications(blackboard, target_smiles=target_smiles),
            *_candidates_from_visual_chains(blackboard, target_smiles=target_smiles),
        ]
    )
    candidates = sorted(candidates, key=_candidate_sort_key)[: max(1, int(max_candidates or 12))]
    route_sketches = [_route_sketch_from_candidate(row, idx=idx) for idx, row in enumerate(candidates, start=1)]
    return {
        "schema_version": "hypothesis_only_retrosynthesis_report.v1",
        "case_id": case_id,
        "accepted": bool(candidates),
        "route_status": "hypothesis_only_not_solved" if candidates else "no_hypothesis_candidates",
        "solved": False,
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "not_exact_literature_segment": True,
        "final_verdict_authority": "none",
        "allowed_use": "chemist_review_and_guided_search_seed_only",
        "target": {
            "name": str(target.get("target_name") or ""),
            "smiles": target_smiles,
            "achiral_canonical_smiles": _canonical_smiles(target_smiles, isomeric=False),
            "heavy_atoms": _heavy_atoms(target_smiles),
        },
        "stereochemistry_policy": {
            "preserve_stereo_when_available": True,
            "achiral_connectivity_candidates_allowed": True,
            "achiral_connectivity_is_not_identity_proof": True,
            "requires_stereo_recovery_before_exact_literature_row": True,
        },
        "candidate_precursor_count": len(candidates),
        "candidate_precursors": candidates,
        "route_sketch_count": len(route_sketches),
        "route_sketches": route_sketches,
        "evidence_summary": _evidence_summary(blackboard),
        "failure_summary": _failure_summary(blackboard),
        "required_before_solved": [
            "recover_or_curate_exact_stereochemical_structures",
            "verify_product_equivalence_to_target",
            "verify_parent_route_without_large_atom_jump",
            "connect_child_or_precursor_route_to_parent_bridge",
            "pass_stock_audit",
            "compile_parent_route_proof",
        ],
    }


def _candidates_from_template_applications(blackboard: dict[str, Any], *, target_smiles: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
                _candidate_row(
                    source_type="analogical_template_hint",
                    target_smiles=str(hint.get("target_smiles") or target_smiles),
                    precursor_smiles=precursor,
                    precursor_role=str(hint.get("precursor_role") or hint.get("hypothesis_type") or "same_core_precursor"),
                    operation_idea=str(
                        route_hypothesis.get("reaction_center_idea")
                        or route_hypothesis.get("template_application")
                        or hint.get("derived_from_retron")
                        or "same-core analogical transformation"
                    ),
                    confidence=_confidence_from_risk_flags(hint.get("risk_flags") or []),
                    evidence_refs=[
                        str(app.get("application_id") or ""),
                        str(app.get("template_id") or ""),
                        *[str(item) for item in app.get("evidence_refs") or []],
                    ],
                    risk_flags=[
                        *[str(item) for item in route_hypothesis.get("risk_flags") or []],
                        *[str(item) for item in hint.get("risk_flags") or []],
                        "hypothesis_only_not_literature_exact",
                    ],
                    stereochemistry_status="partial_or_unverified",
                    source_locator=str(route_hypothesis.get("template_application") or ""),
                )
            )
    return out


def _candidates_from_visual_chains(blackboard: dict[str, Any], *, target_smiles: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
                _candidate_row(
                    source_type="visual_connectivity_candidate",
                    target_smiles=str(step.get("product_smiles") or target_smiles),
                    precursor_smiles=precursor,
                    precursor_role=str((step.get("reactant_labels") or ["visual precursor"])[0] or "visual precursor"),
                    operation_idea="visual source suggests a same-core or steroid-core precursor; use as connectivity-only search seed",
                    confidence=str(step.get("confidence") or chain.get("confidence") or "low"),
                    evidence_refs=[
                        str(chain.get("artifact_ref") or ""),
                        str(chain.get("source_ref") or ""),
                        *[str(item) for item in step.get("evidence_refs") or []],
                    ],
                    risk_flags=[
                        *[str(item) for item in chain.get("reasons") or []],
                        *[str(item) for item in step.get("risk_flags") or []],
                        "visual_connectivity_approximation",
                        "stereochemistry_unresolved",
                    ],
                    stereochemistry_status=str(step.get("stereochemistry_status") or "unspecified_or_partial"),
                    source_locator=str(step.get("source_locator") or chain.get("source_locator") or ""),
                    source_ref=str(chain.get("source_ref") or step.get("source_ref") or ""),
                )
            )
    return out


def _candidate_row(
    *,
    source_type: str,
    target_smiles: str,
    precursor_smiles: str,
    precursor_role: str,
    operation_idea: str,
    confidence: str,
    evidence_refs: list[str],
    risk_flags: list[str],
    stereochemistry_status: str,
    source_locator: str = "",
    source_ref: str = "",
) -> dict[str, Any]:
    achiral_target = _canonical_smiles(target_smiles, isomeric=False)
    achiral_precursor = _canonical_smiles(precursor_smiles, isomeric=False)
    candidate_id = "hyp_route:" + _safe_hash([source_type, precursor_role, achiral_precursor or precursor_smiles])
    return {
        "schema_version": "hypothesis_only_precursor_candidate.v1",
        "candidate_id": candidate_id,
        "source_type": source_type,
        "allowed_use": "guided_search_seed_only",
        "route_status": "hypothesis_only_not_solved",
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "not_exact_literature_segment": True,
        "target_smiles": target_smiles,
        "target_achiral_canonical_smiles": achiral_target,
        "precursor_smiles": precursor_smiles,
        "precursor_achiral_canonical_smiles": achiral_precursor,
        "precursor_role": precursor_role,
        "operation_idea": operation_idea,
        "confidence": confidence or "low",
        "stereochemistry_status": stereochemistry_status or "unverified",
        "same_achiral_connectivity_as_target": bool(achiral_target and achiral_target == achiral_precursor),
        "heavy_atom_delta": abs(_heavy_atoms(target_smiles) - _heavy_atoms(precursor_smiles)),
        "evidence_refs": _dedupe_strings(evidence_refs),
        "source_ref": source_ref,
        "source_locator": source_locator,
        "risk_flags": _dedupe_strings(
            [
                *risk_flags,
                "candidate_requires_route_verifier",
                "candidate_requires_parent_route_proof",
            ]
        ),
    }


def _route_sketch_from_candidate(candidate: dict[str, Any], *, idx: int) -> dict[str, Any]:
    return {
        "schema_version": "hypothesis_only_route_sketch.v1",
        "sketch_id": f"hypothesis_route_sketch_{idx}",
        "status": "hypothesis_only_not_solved",
        "no_solved_claim": True,
        "not_parent_route_proof": True,
        "not_exact_literature_segment": True,
        "route_order": "retro_target_to_precursor",
        "target_smiles": str(candidate.get("target_smiles") or ""),
        "precursor_smiles": str(candidate.get("precursor_smiles") or ""),
        "precursor_role": str(candidate.get("precursor_role") or ""),
        "operation_idea": str(candidate.get("operation_idea") or ""),
        "stereochemistry_status": str(candidate.get("stereochemistry_status") or ""),
        "confidence": str(candidate.get("confidence") or "low"),
        "evidence_refs": [str(item) for item in candidate.get("evidence_refs") or []],
        "required_verification": [
            "structure_identity_or_achiral_connectivity_audit",
            "condition_and_selectivity_review",
            "route_verifier",
            "parent_route_proof",
        ],
        "risk_flags": [str(item) for item in candidate.get("risk_flags") or []],
    }


def _evidence_summary(blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(blackboard.get("literature_evidence") or {})
    exploratory_visual = [
        row
        for row in evidence.get("visual_chains") or []
        if isinstance(row, dict) and row.get("exploratory_accepted")
    ]
    return {
        "schema_version": "hypothesis_route_evidence_summary.v1",
        "source_candidate_count": len(evidence.get("source_candidates") or []),
        "pdf_structure_evidence_count": len(evidence.get("pdf_structure_evidence") or []),
        "visual_chain_count": len(evidence.get("visual_chains") or []),
        "exploratory_visual_chain_count": len(exploratory_visual),
        "exact_row_count": len(evidence.get("exact_rows") or []),
        "analogical_template_count": len(blackboard.get("analogical_templates") or []),
        "template_application_count": len(blackboard.get("template_applications") or []),
        "no_solved_claim": True,
    }


def _failure_summary(blackboard: dict[str, Any]) -> dict[str, Any]:
    failures = [dict(row) for row in blackboard.get("route_failures") or [] if isinstance(row, dict)]
    reasons = _dedupe_strings([str(row.get("reason") or "") for row in failures])
    proof = dict(blackboard.get("parent_route_proof") or {})
    return {
        "schema_version": "hypothesis_route_failure_summary.v1",
        "failure_count": len(failures),
        "failure_reasons": reasons,
        "parent_proof_accepted": bool(proof.get("accepted")),
        "parent_proof_reasons": [str(item) for item in proof.get("reasons") or []],
        "no_solved_claim": True,
    }


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for row in candidates:
        key = str(row.get("precursor_achiral_canonical_smiles") or row.get("precursor_smiles") or "")
        if not key:
            continue
        if key in seen:
            existing = out[seen[key]]
            existing["source_type"] = _merge_label(existing.get("source_type"), row.get("source_type"))
            existing["evidence_refs"] = _dedupe_strings(
                [*list(existing.get("evidence_refs") or []), *list(row.get("evidence_refs") or [])]
            )
            existing["risk_flags"] = _dedupe_strings(
                [*list(existing.get("risk_flags") or []), *list(row.get("risk_flags") or [])]
            )
            existing["source_ref"] = _merge_label(existing.get("source_ref"), row.get("source_ref"))
            existing["source_locator"] = _merge_label(existing.get("source_locator"), row.get("source_locator"))
            if str(existing.get("confidence") or "") == "low" and str(row.get("confidence") or "") in {"medium", "high"}:
                existing["confidence"] = str(row.get("confidence"))
            continue
        seen[key] = len(out)
        out.append(row)
    return out


def _merge_label(left: Any, right: Any) -> str:
    return "+".join(_dedupe_strings([str(left or ""), str(right or "")]))


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, str]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(str(candidate.get("confidence") or "low"), 3)
    return (confidence_rank, int(candidate.get("heavy_atom_delta") or 999), str(candidate.get("candidate_id") or ""))


def _confidence_from_risk_flags(risk_flags: list[Any]) -> str:
    flags = {str(item) for item in risk_flags}
    if "visual_connectivity_approximation" in flags or "stereochemistry_unresolved" in flags:
        return "low"
    if "selectivity_not_proven" in flags or "broad_template_scope" in flags:
        return "low"
    return "medium"


def _canonical_smiles(smiles: str, *, isomeric: bool) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    if not isomeric:
        mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _safe_hash(values: list[str]) -> str:
    payload = "|".join(str(item or "") for item in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _dedupe_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
