"""Instantiate and validate executable literature template candidates."""
from __future__ import annotations

from collections import Counter
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.agent.literature_templates import (
    EXECUTABLE_TEMPLATE_CANDIDATE_SCHEMA,
    LITERATURE_TEMPLATE_PLUGIN_MODEL,
    LITERATURE_TEMPLATE_PLUGIN_SOURCE,
    LiteratureTemplateCard,
    TemplateApplicabilityReport,
    TemplateValidationReport,
    ExecutableTemplateCandidate,
    applicability_report_from_dict,
    template_card_from_dict,
    validate_executable_template_candidate,
)
from cascade_planner.agent.literature_segments import (
    SegmentStepCandidate,
    segment_step_from_dict,
    validate_segment_step,
)
from cascade_planner.agent.template_applicability import (
    EXECUTABLE_ALLOWED_USE,
    assess_template_applicability,
)


RDLogger.DisableLog("rdApp.*")


def instantiate_literature_template(
    product_smiles: str,
    template_card: LiteratureTemplateCard | dict[str, Any],
    *,
    target_smiles: str | None = None,
) -> ExecutableTemplateCandidate:
    """Convert a matched literature retron into a deterministic one-step row."""
    card = template_card if isinstance(template_card, LiteratureTemplateCard) else template_card_from_dict(template_card)
    applicability = assess_template_applicability(
        target_smiles=target_smiles or product_smiles,
        frontier_smiles=product_smiles,
        template_card=card,
    )
    if applicability.allowed_use != EXECUTABLE_ALLOWED_USE:
        candidate = _rejected_candidate(product_smiles, card, applicability, "applicability_disallowed")
        return candidate
    reactants = _reactants_from_applicability(applicability)
    roles = assign_fragment_roles(
        reactants,
        reaction_class=card.reaction_class,
        retron_type=applicability.retron_type,
    )
    rxn_smiles = ".".join(reactants) + f">>{applicability.frontier_smiles or product_smiles}"
    candidate = ExecutableTemplateCandidate(
        product_smiles=applicability.frontier_smiles or product_smiles,
        reactant_smiles=reactants,
        rxn_smiles=rxn_smiles,
        atom_mapping_status="unmapped_fragment_cut",
        template_smarts=str((card.product_retron or {}).get("smarts") or ""),
        source_template_id=card.template_id,
        not_lab_procedure=True,
        proposal_source=LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        evidence_refs=list(card.evidence_refs),
        precursor_roles=roles,
        applicability_report=applicability.to_dict(),
        literature_template_trace={
            "schema_version": "literature_template_trace.v1",
            "source_model": LITERATURE_TEMPLATE_PLUGIN_MODEL,
            "source_template_id": card.template_id,
            "evidence_refs": list(card.evidence_refs),
            "retron_type": applicability.retron_type,
            "not_lab_procedure": True,
            "requires_audit": True,
            "no_solved_claim": True,
        },
        requires_audit=True,
        condition_source=card.condition_source or "unknown",
        schema_version=EXECUTABLE_TEMPLATE_CANDIDATE_SCHEMA,
    )
    validation = validate_template_candidate(candidate)
    candidate.validation_report = validation.to_dict()
    return candidate


def executable_candidate_from_segment_step(
    step_or_data: SegmentStepCandidate | dict[str, Any],
    *,
    source_template_id: str = "",
    reaction_class: str = "literature_route_segment_step",
) -> ExecutableTemplateCandidate:
    """Compile a validated structured literature step into one-step material.

    This path accepts only structured product/reactant fields from
    SegmentStepCandidate. It intentionally ignores any raw reaction string in
    the input payload.
    """
    step = step_or_data if isinstance(step_or_data, SegmentStepCandidate) else segment_step_from_dict(step_or_data)
    step_validation = validate_segment_step(step)
    template_id = source_template_id or f"segment_step:{step.step_id}"
    rxn_smiles = ".".join(step.reactant_smiles) + f">>{step.product_smiles}"
    applicability = {
        "schema_version": "template_applicability_report.v1",
        "target_smiles": step.product_smiles,
        "frontier_smiles": step.product_smiles,
        "matched_retron_atoms": [],
        "matched_bonds": [],
        "match_confidence": "high" if step_validation.get("accepted") else "none",
        "mismatch_reasons": list(step_validation.get("reasons") or []),
        "allowed_use": "executable_candidate" if step_validation.get("accepted") else "forbidden",
        "ambiguity_count": 0,
        "selected_bond": {"source": "structured_literature_segment_step", "step_id": step.step_id},
        "cut_fragments": list(step.reactant_smiles),
        "retron_type": reaction_class,
        "template_id": template_id,
    }
    candidate = ExecutableTemplateCandidate(
        product_smiles=step.product_smiles,
        reactant_smiles=list(step.reactant_smiles),
        rxn_smiles=rxn_smiles,
        atom_mapping_status="structured_literature_step_unmapped",
        template_smarts="",
        source_template_id=template_id,
        not_lab_procedure=True,
        proposal_source=LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        evidence_refs=list(step.evidence_refs),
        precursor_roles=[
            {
                "role": f"segment_reactant_{idx + 1}",
                "smiles": smiles,
                "heavy_atoms": _heavy_atoms_without_dummy(smiles),
            }
            for idx, smiles in enumerate(step.reactant_smiles)
        ],
        applicability_report=applicability,
        literature_template_trace={
            "schema_version": "literature_template_trace.v1",
            "source_model": LITERATURE_TEMPLATE_PLUGIN_MODEL,
            "source_template_id": template_id,
            "source_ref": step.source_ref,
            "evidence_refs": list(step.evidence_refs),
            "not_lab_procedure": True,
            "requires_audit": True,
            "no_solved_claim": True,
            "structured_segment_step": True,
        },
        requires_audit=True,
        condition_source=step.source_ref or "literature_segment",
    )
    validation = validate_template_candidate(candidate)
    candidate.validation_report = validation.to_dict()
    return candidate


def assign_fragment_roles(
    reactant_smiles: list[str],
    *,
    reaction_class: str,
    retron_type: str,
) -> list[dict[str, Any]]:
    roles = _role_labels(reaction_class, retron_type)
    ranked = sorted(reactant_smiles, key=lambda smi: (-_heavy_atoms_without_dummy(smi), smi))
    out: list[dict[str, Any]] = []
    for idx, smiles in enumerate(ranked):
        role = roles[idx] if idx < len(roles) else f"fragment_{idx + 1}"
        out.append(
            {
                "role": role,
                "smiles": smiles,
                "heavy_atoms": _heavy_atoms_without_dummy(smiles),
                "dummy_attachment_count": _dummy_count(smiles),
            }
        )
    return out


def validate_template_candidate(candidate_or_data: ExecutableTemplateCandidate | dict[str, Any]) -> TemplateValidationReport:
    candidate = (
        candidate_or_data
        if isinstance(candidate_or_data, ExecutableTemplateCandidate)
        else _candidate_from_data(candidate_or_data)
    )
    reasons: list[str] = []
    base_validation = validate_executable_template_candidate(candidate)
    if not base_validation["accepted"]:
        reasons.extend(base_validation["reasons"])
    reconstruction = forward_reconstruction_audit(candidate)
    chemical_sanity = basic_chemical_sanity(candidate)
    if not reconstruction.get("passed"):
        reasons.extend(str(item) for item in reconstruction.get("reasons") or ["reconstruction_failed"])
    if not chemical_sanity.get("passed"):
        reasons.extend(str(item) for item in chemical_sanity.get("reasons") or ["chemical_sanity_failed"])
    confidence = "high" if not reasons and reconstruction.get("connectivity_recoverable") else "medium" if not reasons else "none"
    return TemplateValidationReport(
        accepted=not reasons,
        reasons=sorted(set(reasons)),
        confidence=confidence,
        allowed_for_one_step_source=not reasons,
        source_template_id=candidate.source_template_id,
        reconstruction_report=reconstruction,
        chemical_sanity=chemical_sanity,
        audit_required=True,
        no_solved_claim=True,
    )


def forward_reconstruction_audit(candidate_or_data: ExecutableTemplateCandidate | dict[str, Any]) -> dict[str, Any]:
    candidate = (
        candidate_or_data
        if isinstance(candidate_or_data, ExecutableTemplateCandidate)
        else _candidate_from_data(candidate_or_data)
    )
    reasons: list[str] = []
    product = Chem.MolFromSmiles(str(candidate.product_smiles or ""))
    if product is None:
        return {
            "schema_version": "template_forward_reconstruction_audit.v1",
            "passed": False,
            "reasons": ["invalid_product_smiles"],
        }
    if (candidate.literature_template_trace or {}).get("structured_segment_step"):
        return _structured_segment_step_reconstruction_audit(candidate)
    fragments = [Chem.MolFromSmiles(smi) for smi in candidate.reactant_smiles]
    if not fragments or any(mol is None for mol in fragments):
        reasons.append("invalid_reactant_fragment_smiles")
    dummy_counts = [_dummy_count(smi) for smi in candidate.reactant_smiles]
    if sum(dummy_counts) < 2:
        reasons.append("missing_dummy_attachment_points")
    product_heavy = _heavy_atoms_without_dummy(candidate.product_smiles)
    reactant_heavy = sum(_heavy_atoms_without_dummy(smi) for smi in candidate.reactant_smiles)
    if product_heavy != reactant_heavy:
        reasons.append("heavy_atom_accounting_mismatch")
    product_elements = _element_counts_without_dummy(candidate.product_smiles)
    reactant_elements = Counter()
    for smi in candidate.reactant_smiles:
        reactant_elements.update(_element_counts_without_dummy(smi))
    if product_elements != reactant_elements:
        reasons.append("element_accounting_mismatch")
    app = applicability_report_from_dict(candidate.applicability_report or {})
    selected = app.selected_bond or {}
    connectivity_recoverable = bool(selected and sum(dummy_counts) >= 2 and not reasons)
    return {
        "schema_version": "template_forward_reconstruction_audit.v1",
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "connectivity_recoverable": connectivity_recoverable,
        "selected_bond": selected,
        "product_heavy_atoms": product_heavy,
        "reactant_heavy_atoms": reactant_heavy,
        "dummy_attachment_count": sum(dummy_counts),
        "explanation": "fragment cut preserves product atoms and records the matched bond for deterministic recombination",
    }


def basic_chemical_sanity(candidate_or_data: ExecutableTemplateCandidate | dict[str, Any]) -> dict[str, Any]:
    candidate = (
        candidate_or_data
        if isinstance(candidate_or_data, ExecutableTemplateCandidate)
        else _candidate_from_data(candidate_or_data)
    )
    reasons: list[str] = []
    product = Chem.MolFromSmiles(str(candidate.product_smiles or ""))
    if product is None:
        reasons.append("invalid_product_smiles")
    reactants = [Chem.MolFromSmiles(smi) for smi in candidate.reactant_smiles or []]
    if not reactants or any(mol is None for mol in reactants):
        reasons.append("invalid_reactant_smiles")
    if candidate.product_smiles in set(candidate.reactant_smiles or []):
        if not (candidate.literature_template_trace or {}).get("structured_segment_step"):
            reasons.append("reactant_equals_product")
    product_heavy = _heavy_atoms_without_dummy(candidate.product_smiles)
    largest_reactant_heavy = max([_heavy_atoms_without_dummy(smi) for smi in candidate.reactant_smiles] or [0])
    app = applicability_report_from_dict(candidate.applicability_report or {})
    intramolecular_ring_opening = bool((app.selected_bond or {}).get("bond_in_ring") and sum(_dummy_count(smi) for smi in candidate.reactant_smiles) >= 2)
    structured_step = bool((candidate.literature_template_trace or {}).get("structured_segment_step"))
    if largest_reactant_heavy >= product_heavy and not intramolecular_ring_opening and not structured_step:
        reasons.append("no_complexity_drop")
    if sum(_heavy_atoms_without_dummy(smi) for smi in candidate.reactant_smiles) > product_heavy + 2:
        reasons.append("unexplained_large_skeleton_growth")
    if not candidate.not_lab_procedure:
        reasons.append("missing_not_lab_procedure_guard")
    if not candidate.requires_audit:
        reasons.append("missing_route_audit_requirement")
    return {
        "schema_version": "template_basic_chemical_sanity.v1",
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "product_heavy_atoms": product_heavy,
        "largest_reactant_heavy_atoms": largest_reactant_heavy,
        "complexity_drop": product_heavy - largest_reactant_heavy,
        "intramolecular_ring_opening": intramolecular_ring_opening,
        "not_lab_procedure": bool(candidate.not_lab_procedure),
        "requires_audit": bool(candidate.requires_audit),
    }


def _structured_segment_step_reconstruction_audit(candidate: ExecutableTemplateCandidate) -> dict[str, Any]:
    reasons: list[str] = []
    product = Chem.MolFromSmiles(str(candidate.product_smiles or ""))
    reactants = [Chem.MolFromSmiles(smi) for smi in candidate.reactant_smiles or []]
    if product is None:
        reasons.append("invalid_product_smiles")
    if not reactants or any(mol is None for mol in reactants):
        reasons.append("invalid_reactant_smiles")
    product_elements = _element_counts_without_dummy(candidate.product_smiles)
    reactant_elements = Counter()
    for smi in candidate.reactant_smiles:
        reactant_elements.update(_element_counts_without_dummy(smi))
    for element, count in product_elements.items():
        if reactant_elements.get(element, 0) < count:
            reasons.append("segment_step_atom_accounting_failed")
            break
    return {
        "schema_version": "template_forward_reconstruction_audit.v1",
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "connectivity_recoverable": not reasons,
        "selected_bond": dict((candidate.applicability_report or {}).get("selected_bond") or {}),
        "product_heavy_atoms": _heavy_atoms_without_dummy(candidate.product_smiles),
        "reactant_heavy_atoms": sum(_heavy_atoms_without_dummy(smi) for smi in candidate.reactant_smiles),
        "dummy_attachment_count": sum(_dummy_count(smi) for smi in candidate.reactant_smiles),
        "explanation": "structured literature segment step preserves source-grounded product/reactant accounting",
    }


def candidate_to_one_step_row(candidate_or_data: ExecutableTemplateCandidate | dict[str, Any], *, score: float = 0.62) -> dict[str, Any]:
    candidate = (
        candidate_or_data
        if isinstance(candidate_or_data, ExecutableTemplateCandidate)
        else _candidate_from_data(candidate_or_data)
    )
    validation = dict(candidate.validation_report or validate_template_candidate(candidate).to_dict())
    allowed = bool(validation.get("allowed_for_one_step_source"))
    template_payload = {
        "model_full_name": LITERATURE_TEMPLATE_PLUGIN_MODEL,
        "source": LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        "source_model": LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        "template_id": candidate.source_template_id,
        "evidence_refs": list(candidate.evidence_refs),
        "not_lab_procedure": bool(candidate.not_lab_procedure),
        "requires_audit": True,
        "template_validation_report": validation,
        "template_applicability_report": dict(candidate.applicability_report or {}),
        "literature_template_trace": dict(candidate.literature_template_trace or {}),
        "condition_source": candidate.condition_source,
        "no_solved_claim": True,
        "source_policy_decision": "enabled_literature_template_plugin",
    }
    return {
        "reactants": ".".join(candidate.reactant_smiles),
        "scores": float(score if allowed else 0.0),
        "costs": None,
        "template": template_payload,
        "templates": template_payload,
        "model_full_name": LITERATURE_TEMPLATE_PLUGIN_MODEL,
        "weight": 1.0,
        "reaction_domains": _domain_for_candidate(candidate),
        "literature_template_trace": dict(candidate.literature_template_trace or {}),
        "source_policy_decision": "enabled_literature_template_plugin",
    }


def candidate_to_provider_row(candidate_or_data: ExecutableTemplateCandidate | dict[str, Any], *, rank: int = 1) -> dict[str, Any]:
    candidate = (
        candidate_or_data
        if isinstance(candidate_or_data, ExecutableTemplateCandidate)
        else _candidate_from_data(candidate_or_data)
    )
    validation = dict(candidate.validation_report or validate_template_candidate(candidate).to_dict())
    return {
        "product_smiles": candidate.product_smiles,
        "main_reactant": _largest_fragment(candidate.reactant_smiles),
        "aux_reactants": [smi for smi in candidate.reactant_smiles if smi != _largest_fragment(candidate.reactant_smiles)],
        "reactant_smiles": list(candidate.reactant_smiles),
        "rxn_smiles": candidate.rxn_smiles,
        "reaction_smiles": candidate.rxn_smiles,
        "source": LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        "source_model": LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        "score": 0.62 if validation.get("allowed_for_one_step_source") else 0.0,
        "rank": int(rank),
        "candidate_count": 1,
        "type": "literature_executable_template",
        "proposal_type": "literature_template_plugin",
        "template": candidate.source_template_id,
        "model_full_name": LITERATURE_TEMPLATE_PLUGIN_MODEL,
        "cost": None,
        "weight": 1.0,
        "teacher_one_step": True,
        "teacher_source": LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        "evidence_refs": list(candidate.evidence_refs),
        "not_lab_procedure": bool(candidate.not_lab_procedure),
        "requires_audit": True,
        "template_validation_report": validation,
        "template_applicability_report": dict(candidate.applicability_report or {}),
        "literature_template_trace": dict(candidate.literature_template_trace or {}),
        "source_policy_decision": "enabled_literature_template_plugin",
        "condition_source": candidate.condition_source,
        "no_solved_claim": True,
    }


def _rejected_candidate(
    product_smiles: str,
    card: LiteratureTemplateCard,
    applicability: TemplateApplicabilityReport,
    reason: str,
) -> ExecutableTemplateCandidate:
    validation = TemplateValidationReport(
        accepted=False,
        reasons=[reason, *list(applicability.mismatch_reasons or [])],
        confidence="none",
        allowed_for_one_step_source=False,
        source_template_id=card.template_id,
        reconstruction_report={"passed": False, "reasons": [reason]},
        chemical_sanity={"passed": False, "reasons": [reason]},
    )
    return ExecutableTemplateCandidate(
        product_smiles=str(product_smiles or applicability.frontier_smiles or ""),
        reactant_smiles=[],
        rxn_smiles=f">>{product_smiles}",
        atom_mapping_status="not_applicable",
        template_smarts=str((card.product_retron or {}).get("smarts") or ""),
        source_template_id=card.template_id,
        not_lab_procedure=True,
        proposal_source=LITERATURE_TEMPLATE_PLUGIN_SOURCE,
        evidence_refs=list(card.evidence_refs),
        applicability_report=applicability.to_dict(),
        validation_report=validation.to_dict(),
        requires_audit=True,
        condition_source=card.condition_source or "unknown",
    )


def _reactants_from_applicability(applicability: TemplateApplicabilityReport) -> list[str]:
    fragments = [smi for smi in applicability.cut_fragments or [] if smi]
    if not fragments:
        selected = dict(applicability.selected_bond or {})
        fragments = list(selected.get("cut_fragments") or [])
    return sorted(fragments, key=lambda smi: (-_heavy_atoms_without_dummy(smi), smi))


def _role_labels(reaction_class: str, retron_type: str) -> list[str]:
    text = f"{reaction_class} {retron_type}".lower()
    if "glycos" in text and "c_gly" not in text:
        return ["sugar_donor_or_precursor", "aglycone_acceptor"]
    if "c_gly" in text:
        return ["sugar_coupling_partner", "aryl_acceptor"]
    if "bufadienolide" in text or "pyrone" in text or "c_c_coupling" in text:
        return ["steroid_core", "pyrone_coupling_partner"]
    if "taxane" in text:
        return ["taxane_core", "side_chain_fragment"]
    if "macrolactone" in text:
        return ["seco_acid", "hydroxy_acid_fragment"]
    if "corey" in text:
        return ["corey_lactone", "side_chain_fragment"]
    return ["major_fragment", "minor_fragment"]


def _domain_for_candidate(candidate: ExecutableTemplateCandidate) -> str:
    text = " ".join([candidate.source_template_id, candidate.condition_source]).lower()
    if "enzyme" in text or "bio" in text:
        return "literature_biocatalytic"
    return "literature_chemical"


def _candidate_from_data(data: dict[str, Any]) -> ExecutableTemplateCandidate:
    return ExecutableTemplateCandidate(
        product_smiles=str(data.get("product_smiles") or ""),
        reactant_smiles=[str(item) for item in data.get("reactant_smiles") or []],
        rxn_smiles=str(data.get("rxn_smiles") or ""),
        atom_mapping_status=str(data.get("atom_mapping_status") or ""),
        template_smarts=str(data.get("template_smarts") or ""),
        source_template_id=str(data.get("source_template_id") or ""),
        not_lab_procedure=bool(data.get("not_lab_procedure")),
        proposal_source=str(data.get("proposal_source") or LITERATURE_TEMPLATE_PLUGIN_SOURCE),
        evidence_refs=[str(item) for item in data.get("evidence_refs") or []],
        precursor_roles=[dict(item) for item in data.get("precursor_roles") or [] if isinstance(item, dict)],
        applicability_report=dict(data.get("applicability_report") or {}),
        validation_report=dict(data.get("validation_report") or {}),
        literature_template_trace=dict(data.get("literature_template_trace") or {}),
        requires_audit=bool(data.get("requires_audit", True)),
        condition_source=str(data.get("condition_source") or "unknown"),
        schema_version=str(data.get("schema_version") or EXECUTABLE_TEMPLATE_CANDIDATE_SCHEMA),
    )


def _dummy_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def _heavy_atoms_without_dummy(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _element_counts_without_dummy(smiles: str) -> Counter[str]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    counts: Counter[str] = Counter()
    if mol is None:
        return counts
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            counts[atom.GetSymbol()] += 1
    return counts


def _largest_fragment(smiles: list[str]) -> str:
    if not smiles:
        return ""
    return max(smiles, key=lambda smi: (_heavy_atoms_without_dummy(smi), smi))
