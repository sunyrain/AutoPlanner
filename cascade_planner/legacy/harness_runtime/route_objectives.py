"""Route-objective layer for agentic retrosynthesis blackboard runs.

This module classifies what kind of retrosynthesis endpoint is appropriate
before the planner decides which route-search action to spend budget on.  It
is intentionally evidence-seeking and non-executable: no reaction SMILES and
no solved verdicts are emitted here.
"""
from __future__ import annotations

from typing import Any

from cascade_planner.legacy.harness_runtime.parent_route_proof import (
    is_solved_parent_route_proof,
)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
except Exception:  # pragma: no cover - only for stripped test environments.
    Chem = None  # type: ignore[assignment]
    RDLogger = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]

if RDLogger is not None:  # pragma: no branch
    RDLogger.DisableLog("rdApp.*")


ROUTE_OBJECTIVE_SUMMARY_SCHEMA = "route_objective_summary.v1"
ROUTE_OBJECTIVE_HYPOTHESIS_SCHEMA = "route_objective_hypothesis.v1"
ENDPOINT_CANDIDATE_SCHEMA = "endpoint_candidate.v1"
BROAD_TRANSFORM_TEMPLATE_SCHEMA = "broad_transform_template.v1"
ROUTE_PROOF_BUNDLE_SCHEMA = "route_proof_bundle.v1"


OBJECTIVE_TYPES = {
    "small_molecule_stock_closure",
    "advanced_intermediate_anchor",
    "semisynthesis_from_natural_product",
    "biotransformation_endpoint",
    "biosynthetic_or_fermentation_endpoint",
    "literature_known_scaffold_anchor",
    "platform_intermediate_route",
    "chiral_pool_route",
    "parent_to_impurity_or_metabolite",
    "assembly_or_late_stage_modification",
}


def classify_route_objectives(
    *,
    target_smiles: str,
    target_name: str = "",
    family_hint: str = "",
    failure_reasons: list[str] | None = None,
    source_evidence_refs: list[str] | None = None,
    case_id: str = "",
) -> dict[str, Any]:
    """Classify route objectives from molecular features and failure memory."""
    features = _target_features(target_smiles, target_name=target_name, family_hint=family_hint)
    failure_set = {str(item) for item in failure_reasons or [] if str(item or "").strip()}
    evidence_refs = _dedupe([str(item) for item in source_evidence_refs or []])
    reasons: list[str] = []
    if not features.get("valid"):
        reasons.append("invalid_target_smiles")

    objectives = [
        _stock_objective(features, failure_set, evidence_refs),
        _advanced_intermediate_objective(features, failure_set, evidence_refs),
        _semisynthesis_objective(features, failure_set, evidence_refs),
        _biotransformation_objective(features, failure_set, evidence_refs),
        _biosynthetic_objective(features, failure_set, evidence_refs),
        _literature_anchor_objective(features, failure_set, evidence_refs),
        _platform_objective(features, failure_set, evidence_refs),
        _chiral_pool_objective(features, failure_set, evidence_refs),
        _parent_conversion_objective(features, failure_set, evidence_refs),
        _assembly_objective(features, failure_set, evidence_refs),
    ]
    ranked = sorted(
        objectives,
        key=lambda row: (-int(row.get("score") or 0), str(row.get("objective_type") or "")),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    selected = [row for row in ranked if int(row.get("score") or 0) >= 70][:4]
    if not selected:
        selected = [row for row in ranked if int(row.get("score") or 0) >= 45][:3] or ranked[:2]
    endpoint_candidates = _endpoint_candidates_from_objectives(selected, features, case_id=case_id)
    route_scope = _route_scope_from_objectives(selected, features, failure_set)
    return {
        "schema_version": ROUTE_OBJECTIVE_SUMMARY_SCHEMA,
        "accepted": bool(features.get("valid")) and bool(ranked),
        "case_id": case_id,
        "target": {
            "name": target_name,
            "smiles": target_smiles,
            "features": features,
        },
        "objectives": ranked,
        "selected_objectives": selected,
        "endpoint_candidates": endpoint_candidates,
        "route_scope": route_scope,
        "ranking_factors": [
            "objective_fit",
            "endpoint_realism",
            "core_retention",
            "transformation_plausibility",
            "evidence_strength",
            "availability",
            "selectivity_or_stereo_risk",
            "verification_cost",
            "prior_failure_penalty",
        ],
        "source_search_guidance": _source_search_guidance(selected, features),
        "semantics": {
            "raw_reaction_output_allowed": False,
            "solved_claim_allowed": False,
            "objective_selection_is_not_route_proof": True,
        },
        "no_solved_claim": True,
        "requires_verifier": True,
        "reasons": sorted(set(reasons)),
    }


def build_broad_transform_templates_from_blackboard(blackboard: dict[str, Any]) -> dict[str, Any]:
    """Derive broad transform ideas from objectives, hypotheses, and source hints."""
    objectives = [
        dict(row)
        for row in (blackboard.get("route_objective_summary") or {}).get("selected_objectives") or []
        if isinstance(row, dict)
    ]
    target_side = [
        dict(row)
        for row in (blackboard.get("target_side_disconnection_hypotheses") or {}).get("hypotheses") or []
        if isinstance(row, dict)
    ]
    templates: list[dict[str, Any]] = []
    for objective in objectives:
        objective_type = str(objective.get("objective_type") or "")
        if objective_type in {"semisynthesis_from_natural_product", "advanced_intermediate_anchor", "platform_intermediate_route"}:
            templates.append(
                _broad_template(
                    template_id=f"broad_template:{objective_type}:same_core_state_adjustment",
                    objective_type=objective_type,
                    reaction_center="peripheral functional handle or oxidation/protection-state difference on a retained core",
                    preserved_scaffold="largest ring/scaffold system should remain connected",
                    transform_logic="Use same-scaffold precedent to move between oxidation, hydroxylation, protection, or side-chain states.",
                    required_verification=[
                        "same_core_identity",
                        "target_equivalence_after_forward_application",
                        "source_or_model_condition_support",
                    ],
                    risk_flags=["broad_scope_template", "selectivity_and_stereochemistry_not_proven"],
                )
            )
        if objective_type == "biotransformation_endpoint":
            templates.append(
                _broad_template(
                    template_id="broad_template:biotransformation_endpoint:site_selective_functionalization",
                    objective_type=objective_type,
                    reaction_center="late-stage site-selective oxidation, reduction, hydrolysis, or side-chain editing",
                    preserved_scaffold="native-like scaffold and stereochemical core",
                    transform_logic="Prefer enzyme, whole-cell, or biocatalytic precedent when direct chemical selectivity is weak.",
                    required_verification=[
                        "biocatalyst_or_whole_cell_source",
                        "substrate_scope_or_same_family_precedent",
                        "product_identity_audit",
                    ],
                    risk_flags=["biotransformation_scope_unknown", "condition_transfer_required"],
                )
            )
    for row in target_side[:8]:
        handle = str(row.get("target_handle") or "target_handle")
        templates.append(
            _broad_template(
                template_id=f"broad_template:target_handle:{_safe_token(handle)}",
                objective_type="target_side_handle_transfer",
                reaction_center=str(row.get("proposed_disconnection_region") or handle),
                preserved_scaffold=", ".join(str(item) for item in row.get("must_preserve_substructure") or []) or "target core",
                transform_logic=str(row.get("expected_precursor_type") or "target-proximal precursor search"),
                required_verification=[str(item) for item in row.get("required_verification") or []],
                risk_flags=[str(item) for item in row.get("risk_flags") or []],
            )
        )
    for row in _visual_template_hints_from_blackboard(blackboard)[:8]:
        templates.append(
            _broad_template(
                template_id=str(row.get("template_id") or ""),
                objective_type="visual_advisory_semisynthesis_template",
                reaction_center=str(row.get("reaction_center") or ""),
                preserved_scaffold=str(row.get("preserved_scaffold") or "source visual scaffold"),
                transform_logic=str(row.get("transform_logic") or ""),
                required_verification=[
                    "RDKit-valid reactant/product reconstruction",
                    "same-core or intended-side-chain audit",
                    "source/detail or model rerun verification",
                ],
                risk_flags=[
                    "visual_template_not_exact_row",
                    "stereochemistry_may_be_partial",
                    *[str(item) for item in row.get("risk_flags") or [] if str(item or "").strip()],
                ],
                source_refs=[str(item) for item in row.get("source_refs") or [] if str(item or "").strip()],
            )
        )
    templates = _dedupe_templates(templates)
    return {
        "schema_version": "broad_transform_template_report.v1",
        "accepted": bool(templates),
        "case_id": str(blackboard.get("case_id") or ""),
        "templates": templates,
        "template_count": len(templates),
        "allowed_use": "planner_priority_and_guided_search_hint_only",
        "no_solved_claim": True,
        "requires_verifier": True,
        "reasons": [] if templates else ["no_objective_or_hypothesis_material_for_broad_templates"],
    }


def _visual_template_hints_from_blackboard(blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[str] = set()
    evidence = dict((blackboard or {}).get("literature_evidence") or {})
    for chain in evidence.get("visual_chains") or []:
        if not isinstance(chain, dict):
            continue
        source_ref = str(chain.get("source_ref") or chain.get("source_title") or chain.get("artifact_ref") or "")
        for step in chain.get("steps") or []:
            if not isinstance(step, dict):
                continue
            product = str(step.get("product_label") or step.get("product_smiles") or "").strip()
            reactants = [
                str(item).strip()
                for item in step.get("reactant_labels") or step.get("reactant_smiles") or []
                if str(item or "").strip()
            ]
            if not product and not reactants:
                continue
            key = f"{source_ref}:{product}:{'|'.join(reactants[:3])}"
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                {
                    "template_id": f"broad_template:visual:{_safe_token(source_ref or 'source')}:{len(hints) + 1}",
                    "reaction_center": str(
                        step.get("reaction_center")
                        or step.get("disconnection")
                        or step.get("transform_label")
                        or step.get("step_id")
                        or "visible literature transform"
                    ),
                    "preserved_scaffold": str(
                        chain.get("preserved_scaffold")
                        or step.get("preserved_scaffold")
                        or "taxane/core scaffold if present"
                    ),
                    "transform_logic": str(
                        step.get("reaction_class")
                        or step.get("transform_logic")
                        or f"{', '.join(reactants[:3]) or 'visible precursor'} -> {product or 'visible product'}"
                    ),
                    "risk_flags": [str(item) for item in step.get("risk_flags") or [] if str(item or "").strip()],
                    "source_refs": [
                        str(source_ref),
                        str(step.get("source_locator") or ""),
                    ],
                }
            )
    return hints


def compile_route_objective_proof_bundle(
    *,
    blackboard: dict[str, Any],
    parent_route_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile objective-specific proof status without upgrading hypotheses."""
    proof = dict(parent_route_proof or blackboard.get("parent_route_proof") or {})
    summary = dict(blackboard.get("route_objective_summary") or {})
    selected = [
        dict(row)
        for row in summary.get("selected_objectives") or []
        if isinstance(row, dict)
    ]
    evidence = dict(blackboard.get("literature_evidence") or {})
    broad_templates = [
        dict(row)
        for row in blackboard.get("broad_transform_templates") or []
        if isinstance(row, dict)
    ]
    objective_proofs = []
    for objective in selected:
        objective_proofs.append(
            _objective_proof(
                objective,
                parent_route_proof=proof,
                expected_target_smiles=str((blackboard.get("target_profile") or {}).get("target_smiles") or ""),
                evidence=evidence,
                broad_templates=broad_templates,
            )
        )
    accepted = any(row.get("accepted") and row.get("solved") for row in objective_proofs)
    route_status = "solved" if accepted else _best_objective_status(objective_proofs)
    return {
        "schema_version": ROUTE_PROOF_BUNDLE_SCHEMA,
        "accepted": bool(accepted),
        "solved": bool(accepted),
        "case_id": str(blackboard.get("case_id") or ""),
        "selected_objective_count": len(selected),
        "objective_proofs": objective_proofs,
        "route_status": route_status,
        "source_policy": {
            "objective_specific_proof_required": True,
            "analogy_is_not_proof": True,
            "hypothesis_route_can_be_plausible_not_solved": True,
            "stock_audit_required_only_for_stock_objective": True,
        },
        "no_solved_claim": not bool(accepted),
        "reasons": _dedupe(
            [
                reason
                for row in objective_proofs
                for reason in row.get("reasons") or []
            ]
        ),
    }


def _target_features(target_smiles: str, *, target_name: str, family_hint: str) -> dict[str, Any]:
    text = f"{target_name} {family_hint}".lower()
    if Chem is None:
        return {
            "schema_version": "route_objective_target_features.v1",
            "valid": False,
            "text_hints": _text_hints(text),
        }
    mol = Chem.MolFromSmiles(str(target_smiles or ""))
    if mol is None:
        return {
            "schema_version": "route_objective_target_features.v1",
            "valid": False,
            "text_hints": _text_hints(text),
        }
    atoms = list(mol.GetAtoms())
    heavy_atoms = mol.GetNumHeavyAtoms()
    carbon_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() == 6)
    hetero_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() not in {1, 6})
    oxygen_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() == 8)
    nitrogen_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() == 7)
    halogen_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() in {9, 17, 35, 53})
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    rings = mol.GetRingInfo().NumRings()
    fused_ring_system_rings = _largest_ring_system_ring_count(mol)
    carbon_fraction = carbon_atoms / max(1, heavy_atoms)
    formula = rdMolDescriptors.CalcMolFormula(mol) if rdMolDescriptors is not None else ""
    statin_process_like = any(
        token in text
        for token in ("statin", "atorvastatin", "rosuvastatin", "fluvastatin", "pitavastatin", "hmg-coa", "hmg coa")
    )
    flags = {
        "steroid_like_polycyclic_scaffold": _steroid_like_polycyclic(mol, carbon_fraction=carbon_fraction, text=text),
        "natural_product_like": (not statin_process_like)
        and (
            (fused_ring_system_rings >= 3 and (chiral_centers >= 2 or carbon_fraction >= 0.72))
            or (rings >= 3 and chiral_centers >= 4 and carbon_fraction >= 0.70)
        ),
        "high_scaffold_complexity": fused_ring_system_rings >= 3 or heavy_atoms >= 40 or (heavy_atoms >= 28 and chiral_centers >= 3),
        "fused_polycyclic_ring_system": fused_ring_system_rings >= 3,
        "statin_process_like": statin_process_like,
        "peptide_or_amide_rich": _has(mol, "[NX3][CX3](=O)") and nitrogen_atoms >= 2,
        "glycoside_or_sugar_like": _has(mol, "[OX2][C;R][C;R][C;R][O;R]") or "glycos" in text,
        "oligo_or_bioconjugate_like": any(token in text for token in ("oligo", "peptide", "nucleotide", "rna", "dna", "bioconjugate")),
        "impurity_or_metabolite_hint": any(token in text for token in ("impurity", "metabolite", "degrad", "analog", "analogue")),
        "biotransformation_hint": any(token in text for token in ("enzyme", "enzym", "biotrans", "fermentation", "microbial", "whole-cell")),
        "late_stage_functionalization_handles": oxygen_atoms >= 2 or halogen_atoms > 0 or _has(mol, "[CX3]=[OX1]"),
    }
    return {
        "schema_version": "route_objective_target_features.v1",
        "valid": True,
        "heavy_atoms": int(heavy_atoms),
        "rings": int(rings),
        "largest_fused_ring_system_rings": int(fused_ring_system_rings),
        "chiral_centers": int(chiral_centers),
        "carbon_atoms": int(carbon_atoms),
        "hetero_atoms": int(hetero_atoms),
        "oxygen_atoms": int(oxygen_atoms),
        "nitrogen_atoms": int(nitrogen_atoms),
        "halogen_atoms": int(halogen_atoms),
        "carbon_fraction": round(carbon_fraction, 3),
        "formula": formula,
        "text_hints": _text_hints(text),
        "flags": flags,
    }


def _stock_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    score = 70
    heavy = int(features.get("heavy_atoms") or 0)
    rings = int(features.get("rings") or 0)
    chiral = int(features.get("chiral_centers") or 0)
    score -= max(0, heavy - 22)
    score -= max(0, rings - 2) * 10
    score -= max(0, chiral - 2) * 6
    if "large_atom_jump" in failure_set:
        score -= 35
    if (features.get("flags") or {}).get("high_scaffold_complexity"):
        score -= 25
    return _objective(
        "small_molecule_stock_closure",
        score,
        rationale="Use when the target is small enough that complete construction from stock is plausible.",
        endpoint_acceptance=["commercial_or_common_stock_precursors", "strict_stock_audit", "route_verifier_acceptance"],
        evidence_requirements=["stock_availability", "no_unexplained_large_atom_jump"],
        allowed_actions=["run_guided_chemenzy", "stitch_parent_route"],
        proof_type="stock_route_proof",
        risk_flags=["de_novo_core_construction_can_fake_close_large_scaffolds"],
        evidence_refs=evidence_refs,
    )


def _advanced_intermediate_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 35 + int(features.get("rings") or 0) * 8 + int(features.get("chiral_centers") or 0) * 4
    if flags.get("high_scaffold_complexity"):
        score += 20
    if "large_atom_jump" in failure_set:
        score += 20
    return _objective(
        "advanced_intermediate_anchor",
        score,
        rationale="Stop at an advanced same-scaffold intermediate when full stock closure is unrealistic.",
        endpoint_acceptance=["advanced_intermediate_identity", "availability_or_literature_precedent", "parent_conversion_connected"],
        evidence_requirements=["same_scaffold_intermediate_evidence", "conversion_to_target_or_template"],
        allowed_actions=["search_literature", "extract_visual_literature_chain", "derive_broad_reaction_template", "run_guided_chemenzy"],
        proof_type="advanced_intermediate_proof",
        risk_flags=["intermediate_availability_may_be_unknown"],
        evidence_refs=evidence_refs,
    )


def _semisynthesis_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 20
    if flags.get("natural_product_like"):
        score += 30
    if flags.get("steroid_like_polycyclic_scaffold"):
        score += 25
    if int(features.get("chiral_centers") or 0) >= 3:
        score += 15
    if "large_atom_jump" in failure_set:
        score += 25
    return _objective(
        "semisynthesis_from_natural_product",
        score,
        rationale="Preserve a naturally available or biosynthetically accessible scaffold instead of rebuilding it.",
        endpoint_acceptance=["natural_product_or_feedstock_anchor", "same_core_retention", "late_stage_conversion_evidence"],
        evidence_requirements=["anchor_identity", "source_or_database_support", "target_side_conversion_logic"],
        allowed_actions=["search_literature", "extract_visual_literature_chain", "derive_broad_reaction_template", "run_guided_chemenzy"],
        proof_type="semisynthesis_anchor_proof",
        risk_flags=["anchor_is_not_full_route_without_conversion_proof"],
        evidence_refs=evidence_refs,
    )


def _biotransformation_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 20
    if flags.get("biotransformation_hint"):
        score += 35
    if flags.get("steroid_like_polycyclic_scaffold"):
        score += 25
    if flags.get("late_stage_functionalization_handles"):
        score += 15
    if int(features.get("chiral_centers") or 0) >= 2:
        score += 10
    if "large_atom_jump" in failure_set:
        score += 15
    return _objective(
        "biotransformation_endpoint",
        score,
        rationale="Use when selectivity suggests enzyme, microbial, or whole-cell transformations may be the practical endpoint.",
        endpoint_acceptance=["substrate_identity", "biocatalyst_or_strain_precedent", "product_identity_or_scope_support"],
        evidence_requirements=["biotransformation_source", "same_family_substrate_scope", "product_identity_audit"],
        allowed_actions=["search_literature", "derive_broad_reaction_template", "run_guided_chemenzy"],
        proof_type="biotransformation_proof",
        risk_flags=["whole_cell_scope_not_implied_by_similarity_alone"],
        evidence_refs=evidence_refs,
    )


def _biosynthetic_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 15
    if flags.get("natural_product_like"):
        score += 20
    if flags.get("biotransformation_hint"):
        score += 20
    return _objective(
        "biosynthetic_or_fermentation_endpoint",
        score,
        rationale="Consider when the target resembles a natural product or pathway product.",
        endpoint_acceptance=["producer_or_pathway_evidence", "isolation_or_engineered_biosynthesis_precedent"],
        evidence_requirements=["biosynthetic_source", "product_or_close_analog_identity"],
        allowed_actions=["search_literature"],
        proof_type="biosynthetic_pathway_proof",
        risk_flags=["biosynthesis_may_not_be_preparative_route"],
        evidence_refs=evidence_refs,
    )


def _literature_anchor_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 35
    if int(features.get("rings") or 0) >= 3:
        score += 20
    if int(features.get("heavy_atoms") or 0) >= 25:
        score += 10
    if flags.get("statin_process_like"):
        score += 20
    if evidence_refs:
        score += 15
    if "large_atom_jump" in failure_set:
        score += 10
    return _objective(
        "literature_known_scaffold_anchor",
        score,
        rationale="Search for same-scaffold or same-series routes before inventing distant de novo closures.",
        endpoint_acceptance=["source_detail_or_validated_source", "same_scaffold_anchor", "connectivity_to_target"],
        evidence_requirements=["source_metadata", "local_or_online_source", "extracted_intermediate_or_transform"],
        allowed_actions=["search_literature", "extract_pdf_literature_structures", "extract_visual_literature_chain", "compile_exact_literature_rows"],
        proof_type="literature_scaffold_proof",
        risk_flags=["similar_literature_is_evidence_not_proof_until_connected"],
        evidence_refs=evidence_refs,
    )


def _platform_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 25
    if flags.get("impurity_or_metabolite_hint"):
        score += 20
    if flags.get("statin_process_like"):
        score += 15
    if int(features.get("rings") or 0) >= 3:
        score += 20
    return _objective(
        "platform_intermediate_route",
        score,
        rationale="Use a common scaffold platform for analog or series-level synthesis.",
        endpoint_acceptance=["platform_intermediate_identity", "divergent_step_to_target"],
        evidence_requirements=["series_or_analog_precedent", "late_stage_divergence_logic"],
        allowed_actions=["search_literature", "derive_broad_reaction_template", "run_guided_chemenzy"],
        proof_type="platform_route_proof",
        risk_flags=["platform_may_not_have_target_selectivity"],
        evidence_refs=evidence_refs,
    )


def _chiral_pool_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    chiral = int(features.get("chiral_centers") or 0)
    heavy = int(features.get("heavy_atoms") or 0)
    score = 15 + chiral * 8
    if heavy <= 35 and chiral >= 1:
        score += 15
    return _objective(
        "chiral_pool_route",
        score,
        rationale="Use when preserving existing stereochemical information is more realistic than asymmetric de novo construction.",
        endpoint_acceptance=["chiral_pool_precursor_identity", "stereochemical_mapping"],
        evidence_requirements=["commercial_or_common_chiral_pool_source", "stereo_retention_or_inversion_logic"],
        allowed_actions=["search_literature", "run_guided_chemenzy"],
        proof_type="chiral_pool_proof",
        risk_flags=["stereochemical_mapping_must_be_explicit"],
        evidence_refs=evidence_refs,
    )


def _parent_conversion_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 10 + (45 if flags.get("impurity_or_metabolite_hint") else 0)
    return _objective(
        "parent_to_impurity_or_metabolite",
        score,
        rationale="For impurities, metabolites, degradants, or analogs, start from the parent/API or close analog when justified.",
        endpoint_acceptance=["parent_or_close_analog_identity", "conversion_to_target"],
        evidence_requirements=["parent_relation_evidence", "late_stage_conversion_or_degradation_precedent"],
        allowed_actions=["search_literature", "derive_broad_reaction_template", "run_guided_chemenzy"],
        proof_type="impurity_parent_conversion_proof",
        risk_flags=["parent_relation_must_not_be_name_assumption_only"],
        evidence_refs=evidence_refs,
    )


def _assembly_objective(features: dict[str, Any], failure_set: set[str], evidence_refs: list[str]) -> dict[str, Any]:
    flags = dict(features.get("flags") or {})
    score = 10
    if flags.get("peptide_or_amide_rich"):
        score += 20
    if flags.get("glycoside_or_sugar_like"):
        score += 20
    if flags.get("oligo_or_bioconjugate_like"):
        score += 40
    return _objective(
        "assembly_or_late_stage_modification",
        score,
        rationale="Use modular assembly objectives for oligomers, conjugates, glycosides, salts, prodrugs, or materials-like targets.",
        endpoint_acceptance=["monomer_or_fragment_identity", "assembly_order", "deprotection_or_final_modification"],
        evidence_requirements=["fragment_scope", "assembly_protocol_or_precedent"],
        allowed_actions=["search_literature", "compile_exact_literature_rows", "run_guided_chemenzy"],
        proof_type="assembly_route_proof",
        risk_flags=["fragment_assembly_is_domain_specific"],
        evidence_refs=evidence_refs,
    )


def _objective(
    objective_type: str,
    score: int,
    *,
    rationale: str,
    endpoint_acceptance: list[str],
    evidence_requirements: list[str],
    allowed_actions: list[str],
    proof_type: str,
    risk_flags: list[str],
    evidence_refs: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": ROUTE_OBJECTIVE_HYPOTHESIS_SCHEMA,
        "objective_id": f"route_objective:{objective_type}",
        "objective_type": objective_type,
        "score": max(0, min(100, int(score))),
        "rank": 0,
        "rationale": rationale,
        "endpoint_acceptance": _dedupe(endpoint_acceptance),
        "evidence_requirements": _dedupe(evidence_requirements),
        "allowed_actions": _dedupe(allowed_actions),
        "proof_type": proof_type,
        "risk_flags": _dedupe(risk_flags),
        "related_source_evidence": _dedupe(evidence_refs),
        "no_solved_claim": True,
    }


def _endpoint_candidates_from_objectives(
    objectives: list[dict[str, Any]],
    features: dict[str, Any],
    *,
    case_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for objective in objectives:
        objective_type = str(objective.get("objective_type") or "")
        if objective_type == "semisynthesis_from_natural_product":
            rows.append(
                _endpoint(
                    case_id,
                    objective_type,
                    "natural_product_or_feedstock_same_scaffold_pool",
                    "Natural product, fermentation product, or isolated scaffold pool retaining the major ring/stereo framework.",
                    ["source_confirms_anchor_identity", "same_core_retention", "conversion_to_target_required"],
                )
            )
        if objective_type == "biotransformation_endpoint":
            rows.append(
                _endpoint(
                    case_id,
                    objective_type,
                    "same_core_biotransformation_substrate",
                    "Same-family substrate where a catalyst, strain, enzyme, or whole-cell system performs the hard selectivity step.",
                    ["biocatalyst_precedent", "substrate_scope_check", "product_identity_required"],
                )
            )
        if objective_type == "advanced_intermediate_anchor":
            rows.append(
                _endpoint(
                    case_id,
                    objective_type,
                    "same_scaffold_advanced_intermediate",
                    "Commercial, source-reported, or plausibly isolable same-scaffold intermediate with fewer unresolved edits than the target.",
                    ["availability_or_source_evidence", "target_conversion_logic"],
                )
            )
        if objective_type == "literature_known_scaffold_anchor":
            rows.append(
                _endpoint(
                    case_id,
                    objective_type,
                    "source_reported_same_series_intermediate",
                    "Intermediate, analog, or scaffold route extracted from literature and then connected to the target.",
                    ["source_detail_extraction", "connectivity_or_template_bridge"],
                )
            )
        if objective_type == "parent_to_impurity_or_metabolite":
            rows.append(
                _endpoint(
                    case_id,
                    objective_type,
                    "parent_api_or_close_analog",
                    "Known parent compound or close analog converted by late-stage functionalization, metabolism, or degradation logic.",
                    ["parent_relation_evidence", "conversion_selectivity_check"],
                )
            )
    if not rows:
        rows.append(
            _endpoint(
                case_id,
                str(objectives[0].get("objective_type") or "small_molecule_stock_closure") if objectives else "unknown",
                "stock_or_nearest_supported_precursor",
                "Nearest endpoint supported by stock, source, or objective-specific evidence.",
                ["endpoint_identity", "route_verifier_or_objective_proof"],
            )
        )
    return _dedupe_endpoints(rows)


def _endpoint(case_id: str, objective_type: str, endpoint_type: str, description: str, required_verification: list[str]) -> dict[str, Any]:
    return {
        "schema_version": ENDPOINT_CANDIDATE_SCHEMA,
        "endpoint_id": f"endpoint:{objective_type}:{endpoint_type}",
        "case_id": case_id,
        "objective_type": objective_type,
        "endpoint_type": endpoint_type,
        "description": description,
        "source_status": "requires_discovery",
        "required_verification": _dedupe(required_verification),
        "allowed_use": "route_objective_endpoint_hint_only",
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }


def _route_scope_from_objectives(
    selected: list[dict[str, Any]],
    features: dict[str, Any],
    failure_set: set[str],
) -> dict[str, Any]:
    top_type = str((selected[0] if selected else {}).get("objective_type") or "unknown")
    stock = next((row for row in selected if row.get("objective_type") == "small_molecule_stock_closure"), {})
    stock_score = int(stock.get("score") or 0)
    non_stock_score = max(
        [int(row.get("score") or 0) for row in selected if row.get("objective_type") != "small_molecule_stock_closure"] or [0]
    )
    deprioritize_stock = bool(
        "large_atom_jump" in failure_set
        or (features.get("flags") or {}).get("high_scaffold_complexity")
        or non_stock_score >= stock_score + 15
    )
    return {
        "schema_version": "target_route_scope.v1",
        "route_scope": top_type,
        "selected_objective_types": [str(row.get("objective_type") or "") for row in selected],
        "de_novo_core_construction_deprioritized": deprioritize_stock,
        "small_molecule_stock_closure_deprioritized": deprioritize_stock,
        "objective_evidence_validation_required": any(
            str(row.get("objective_type") or "") != "small_molecule_stock_closure" for row in selected[:2]
        ),
        "preferred_endpoint_types": _dedupe(
            [
                endpoint
                for row in selected
                for endpoint in row.get("endpoint_acceptance") or []
            ]
        )[:8],
        "required_route_evidence": _dedupe(
            [
                requirement
                for row in selected
                for requirement in row.get("evidence_requirements") or []
            ]
        )[:10],
        "no_solved_claim": True,
    }


def _source_search_guidance(selected: list[dict[str, Any]], features: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for objective in selected:
        objective_type = str(objective.get("objective_type") or "")
        terms = [
            objective_type.replace("_", " "),
            *[str(item).replace("_", " ") for item in objective.get("evidence_requirements") or []],
        ]
        if (features.get("flags") or {}).get("steroid_like_polycyclic_scaffold"):
            terms.extend(["steroid scaffold", "same core intermediate", "biotransformation"])
        elif (features.get("flags") or {}).get("statin_process_like"):
            terms.extend(["statin process chemistry", "Paal-Knorr intermediate", "advanced ketal ester"])
        elif (features.get("flags") or {}).get("natural_product_like"):
            terms.extend(["natural product scaffold", "semisynthesis", "same core"])
        rows.append(
            {
                "schema_version": "route_objective_source_search_guidance.v1",
                "objective_type": objective_type,
                "search_terms": _dedupe(terms)[:8],
                "evidence_class": "route_objective_search_guidance",
                "allowed_use": "source_acquisition_query_hint_only",
                "no_solved_claim": True,
            }
        )
    return rows


def _broad_template(
    *,
    template_id: str,
    objective_type: str,
    reaction_center: str,
    preserved_scaffold: str,
    transform_logic: str,
    required_verification: list[str],
    risk_flags: list[str],
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": BROAD_TRANSFORM_TEMPLATE_SCHEMA,
        "template_id": template_id,
        "objective_type": objective_type,
        "reaction_center": reaction_center,
        "preserved_scaffold": preserved_scaffold,
        "transform_logic": transform_logic,
        "required_verification": _dedupe(required_verification),
        "risk_flags": _dedupe(risk_flags),
        "allowed_use": "planner_priority_and_guided_search_hint_only",
        "not_exact_literature_segment": True,
        "not_parent_route_proof": True,
        "no_solved_claim": True,
    }
    refs = _dedupe([str(item) for item in source_refs or [] if str(item or "").strip()])
    if refs:
        row["source_refs"] = refs
    return row


def _objective_proof(
    objective: dict[str, Any],
    *,
    parent_route_proof: dict[str, Any],
    expected_target_smiles: str,
    evidence: dict[str, Any],
    broad_templates: list[dict[str, Any]],
) -> dict[str, Any]:
    objective_type = str(objective.get("objective_type") or "")
    parent_solved = is_solved_parent_route_proof(
        parent_route_proof,
        expected_target_smiles=expected_target_smiles,
    )
    evidence_backed = bool(
        evidence.get("exact_rows")
        or evidence.get("visual_chains")
        or evidence.get("process_evidence_rows")
        or evidence.get("source_candidates")
        or broad_templates
    )
    reasons: list[str] = []
    if not parent_solved:
        reasons.append("deterministic_connected_route_not_proven")
    if objective_type != "small_molecule_stock_closure" and not evidence_backed:
        reasons.append("objective_endpoint_evidence_missing")
    solved = parent_solved
    status = "proof_solved" if solved else ("plausible_hypothesis_route" if evidence_backed else "unresolved")
    return {
        "schema_version": "objective_specific_route_proof.v1",
        "objective_id": str(objective.get("objective_id") or ""),
        "objective_type": objective_type,
        "proof_type": str(objective.get("proof_type") or ""),
        "accepted": solved,
        "solved": solved,
        "route_status": status,
        "proof_clauses": {
            "deterministic_connected_route_proven": parent_solved,
            "objective_endpoint_evidence_present": evidence_backed,
            "analogy_used_only_as_rationale": True,
        },
        "reasons": reasons,
        "no_solved_claim": not solved,
    }


def _best_objective_status(rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("route_status") or "") for row in rows}
    if "plausible_hypothesis_route" in statuses:
        return "plausible_hypothesis_route"
    if rows:
        return "unresolved"
    return "no_route_objective_selected"


def _steroid_like_polycyclic(mol: Any, *, carbon_fraction: float, text: str) -> bool:
    if any(token in text for token in ("steroid", "sterol", "pregn", "androst", "chol", "cardenolide", "bufadienolide")):
        return True
    return bool(
        _largest_ring_system_ring_count(mol) >= 3
        and mol.GetNumHeavyAtoms() >= 18
        and carbon_fraction >= 0.72
        and not _has(mol, "[NX3][CX3](=O)[NX3]")
    )


def _largest_ring_system_ring_count(mol: Any) -> int:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    if not rings:
        return 0
    seen: set[int] = set()
    best = 0
    for start in range(len(rings)):
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            for jdx, other in enumerate(rings):
                if jdx not in component and rings[idx] & other:
                    stack.append(jdx)
        seen.update(component)
        best = max(best, len(component))
    return best


def _has(mol: Any, smarts: str) -> bool:
    if Chem is None or mol is None:
        return False
    query = Chem.MolFromSmarts(smarts)
    return bool(query is not None and mol.HasSubstructMatch(query))


def _text_hints(text: str) -> list[str]:
    hints = []
    for token in (
        "steroid",
        "natural product",
        "semisynthesis",
        "biotransformation",
        "enzyme",
        "impurity",
        "metabolite",
        "glycoside",
        "peptide",
        "oligo",
        "statin",
        "atorvastatin",
    ):
        if token in text:
            hints.append(token.replace(" ", "_"))
    return hints


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


def _dedupe_endpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("endpoint_id") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe_templates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("template_id") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _safe_token(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").lower()).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "item"
