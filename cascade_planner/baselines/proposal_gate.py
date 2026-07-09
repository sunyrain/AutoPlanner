"""Conservative proposal gate for obvious material-sanity artifacts.

This module is intentionally not a full reaction validator.  It only blocks or
labels retrosynthetic proposals where the product gains a large unexplained
material source relative to listed reactants and small condition reagents.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.baselines.route_contract import RouteStepCandidate
from cascade_planner.baselines.route_plausibility import (
    DEFAULT_MAX_CARBON_GAIN,
    DEFAULT_MAX_HEAVY_GAIN,
    DEFAULT_MAX_HETERO_GAIN,
    audit_step_plausibility,
    element_counts,
    has_unsupported_biosynthetic_prenyl_terminal,
    heavy_atom_count,
    hetero_atom_count,
    looks_like_long_prenyl_chain,
    route_condition_risk_warnings,
    terminal_reactants_from_route,
)


RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = "proposal_material_gate.v1"
HARD_REJECT_REASONS = {
    "invalid_product_smiles",
    "invalid_or_missing_reactants",
    "large_unexplained_heavy_atom_gain",
    "large_unexplained_carbon_gain",
    "large_unexplained_hetero_atom_gain",
    "unexplained_new_element_source",
    "non_mild_predicted_temperature",
    "strong_hydride_reagent_predicted",
    "unsupported_biosynthetic_prenyl_terminal",
}


@dataclass(frozen=True)
class ProposalGateConfig:
    mode: str = "hard_reject"
    max_heavy_gain: int = DEFAULT_MAX_HEAVY_GAIN
    max_carbon_gain: int = DEFAULT_MAX_CARBON_GAIN
    max_hetero_gain: int = DEFAULT_MAX_HETERO_GAIN
    complex_core_heavy_gain: int = 8
    complex_core_min_rings: int = 3
    complex_core_min_chiral_centers: int = 3


def normalize_proposal_gate_mode(value: Any) -> str:
    raw = str(value or "hard_reject").strip().lower()
    aliases = {
        "on": "hard_reject",
        "true": "hard_reject",
        "1": "hard_reject",
        "gate": "hard_reject",
        "strict": "hard_reject",
        "debug": "warn",
        "debug_show_rejected": "warn",
        "show_rejected": "warn",
        "annotation": "warn",
        "annotation_only": "warn",
        "warn_only": "warn",
        "false": "off",
        "0": "off",
        "none": "off",
    }
    mode = aliases.get(raw, raw)
    return mode if mode in {"off", "warn", "hard_reject"} else "hard_reject"


def evaluate_step_candidate(
    *,
    product_smiles: str,
    reactant_smiles: list[str],
    rxn_smiles: str = "",
    condition_predictions: list[dict[str, Any]] | None = None,
    source_model: str = "",
    config: ProposalGateConfig | None = None,
) -> dict[str, Any]:
    config = config or ProposalGateConfig()
    step = RouteStepCandidate(
        product_smiles=str(product_smiles or ""),
        reactant_smiles=[str(item) for item in reactant_smiles or [] if str(item or "")],
        rxn_smiles=str(rxn_smiles or _reaction_smiles(reactant_smiles, product_smiles)),
        source_model=str(source_model or ""),
        condition_predictions=list(condition_predictions or []),
    )
    return gate_step_candidate(step, config=config)


def gate_step_candidate(step: RouteStepCandidate, *, config: ProposalGateConfig | None = None) -> dict[str, Any]:
    config = config or ProposalGateConfig()
    audit = audit_step_plausibility(
        step,
        max_heavy_gain=config.max_heavy_gain,
        max_carbon_gain=config.max_carbon_gain,
        max_hetero_gain=config.max_hetero_gain,
    )
    reasons = [str(reason) for reason in audit.get("reasons") or []]
    hard_reasons = [reason for reason in reasons if reason in HARD_REJECT_REASONS]
    product_profile = molecule_profile(step.product_smiles)
    reactant_profiles = [molecule_profile(smi) for smi in step.reactant_smiles or []]
    recognized_roles = recognized_reagent_roles(step.reactant_smiles, step.condition_predictions)
    if _is_complex_core_jump(audit, product_profile, config):
        if "unexplained_complex_core_growth" not in hard_reasons:
            hard_reasons.append("unexplained_complex_core_growth")
        if "unexplained_complex_core_growth" not in reasons:
            reasons.append("unexplained_complex_core_growth")

    decision = "reject" if hard_reasons else "keep"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "hard_reject": bool(hard_reasons),
        "reasons": reasons,
        "hard_reasons": hard_reasons,
        "frontier_reason": "complex_core_unresolved" if hard_reasons else "",
        "source_model": step.source_model,
        "rxn_smiles": step.rxn_smiles,
        "product_profile": product_profile,
        "reactant_profiles": reactant_profiles,
        "recognized_reagent_roles": recognized_roles,
        "material_audit": audit,
        "contract": (
            "conservative pre-display proposal gate; rejects only obvious "
            "unexplained material/core growth artifacts"
        ),
    }


def gate_web_route(route: dict[str, Any], *, config: ProposalGateConfig | None = None) -> dict[str, Any]:
    config = config or ProposalGateConfig()
    step_reports = []
    reason_counts: Counter[str] = Counter()
    first_frontier: dict[str, Any] | None = None
    route_hard_reasons = _route_level_hard_reasons(route)
    for idx, step in enumerate(route.get("steps") or []):
        if not isinstance(step, dict):
            continue
        report = gate_web_step(step, config=config)
        step["proposal_gate"] = report
        step_reports.append(report)
        reason_counts.update(str(reason) for reason in report.get("hard_reasons") or [])
        if report.get("hard_reject") and first_frontier is None:
            first_frontier = {
                "smiles": str(step.get("product") or ""),
                "reason": report.get("frontier_reason") or "bad_proposal_rejected",
                "last_valid_step": max(0, idx - 1),
                "rejected_step": idx,
                "proposal_reasons": list(report.get("hard_reasons") or []),
                "rxn_smiles": step.get("reaction_smiles") or step.get("rxn_smiles") or "",
            }
    reason_counts.update(route_hard_reasons)
    if route_hard_reasons and first_frontier is None:
        first_frontier = _route_level_frontier(route, route_hard_reasons)
    decision = "reject" if route_hard_reasons or any(report.get("hard_reject") for report in step_reports) else "keep"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "hard_reject": decision == "reject",
        "mode": normalize_proposal_gate_mode(config.mode),
        "step_count": len(step_reports),
        "rejected_step_count": sum(1 for report in step_reports if report.get("hard_reject")),
        "route_hard_reasons": route_hard_reasons,
        "reason_counts": dict(sorted(reason_counts.items())),
        "frontier": first_frontier,
        "steps": step_reports,
    }


def gate_web_step(step: dict[str, Any], *, config: ProposalGateConfig | None = None) -> dict[str, Any]:
    product = str(step.get("product") or "")
    reactants = []
    main = str(step.get("main_reactant") or "")
    if main:
        reactants.append(main)
    reactants.extend(str(item) for item in step.get("aux_reactants") or [] if str(item or ""))
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    if (not product or not reactants) and ">>" in rxn:
        lhs, rhs = rxn.split(">>", 1)
        product = product or rhs
        reactants = reactants or [part for part in lhs.split(".") if part]
    return evaluate_step_candidate(
        product_smiles=product,
        reactant_smiles=reactants,
        rxn_smiles=rxn,
        condition_predictions=list(step.get("condition_predictions") or []),
        source_model=str(step.get("reaction_type") or step.get("source") or ""),
        config=config,
    )


def summarize_route_gate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    route_decisions: Counter[str] = Counter(str(report.get("decision") or "unknown") for report in reports)
    reason_counts: Counter[str] = Counter()
    for report in reports:
        reason_counts.update({str(k): int(v) for k, v in (report.get("reason_counts") or {}).items()})
    return {
        "route_decision_counts": dict(sorted(route_decisions.items())),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def molecule_profile(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return {"available": False, "smiles": str(smiles or "")}
    counts = element_counts(str(smiles or ""))
    return {
        "available": True,
        "smiles": str(smiles or ""),
        "heavy_atoms": heavy_atom_count(counts),
        "carbon_atoms": int(counts.get("C", 0)),
        "hetero_atoms": hetero_atom_count(counts),
        "rings": int(mol.GetRingInfo().NumRings()),
        "chiral_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }


def recognized_reagent_roles(
    reactant_smiles: list[str],
    condition_predictions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for role_source, smi in _candidate_reagent_smiles(reactant_smiles, condition_predictions):
        role = _reagent_role(smi)
        if role:
            rows.append({"smiles": smi, "source": role_source, "role": role})
    return rows


def _candidate_reagent_smiles(
    reactant_smiles: list[str],
    condition_predictions: list[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    out = [("reactant", str(smi)) for smi in reactant_smiles or [] if str(smi or "")]
    for row in condition_predictions or []:
        if not isinstance(row, dict):
            continue
        for key in ("Reagent", "reagent", "Catalyst", "catalyst"):
            for item in str(row.get(key) or "").replace(";", ".").split("."):
                smi = item.strip()
                if smi:
                    out.append((key.lower(), smi))
    return out


def _reagent_role(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    text = str(smiles or "")
    counts = element_counts(text)
    heavy = heavy_atom_count(counts)
    if heavy > 16:
        return ""
    if "[Li]" in text or "[Mg]" in text:
        return "organometallic_base_or_transfer_reagent"
    if "Si" in counts:
        return "silyl_protecting_group_reagent"
    if "S" in counts and int(counts.get("O", 0)) >= 3:
        return "sulfonate_or_sulfate_reagent"
    if "P" in counts and int(counts.get("Cl", 0)) > 0:
        return "chlorinating_phosphorus_reagent"
    return ""


def _is_complex_core_jump(audit: dict[str, Any], product_profile: dict[str, Any], config: ProposalGateConfig) -> bool:
    if not product_profile.get("available"):
        return False
    heavy_gain = int(audit.get("heavy_atom_gain") or 0)
    if heavy_gain < int(config.complex_core_heavy_gain):
        return False
    return (
        int(product_profile.get("rings") or 0) >= int(config.complex_core_min_rings)
        or int(product_profile.get("chiral_centers") or 0) >= int(config.complex_core_min_chiral_centers)
    )


def _route_level_hard_reasons(route: dict[str, Any]) -> list[str]:
    reasons = list(route_condition_risk_warnings(route))
    if has_unsupported_biosynthetic_prenyl_terminal(route):
        reasons.append("unsupported_biosynthetic_prenyl_terminal")
    return sorted(set(reasons))


def _route_level_frontier(route: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    terminal_smiles = ""
    terminals = terminal_reactants_from_route(route)
    if "unsupported_biosynthetic_prenyl_terminal" in reasons:
        for smiles in terminals:
            if looks_like_long_prenyl_chain(smiles):
                terminal_smiles = smiles
                break
    for smiles in terminals:
        if terminal_smiles:
            break
        terminal_smiles = smiles
    product = ""
    if steps:
        product = str(steps[-1].get("product") or steps[-1].get("product_smiles") or "")
    return {
        "smiles": terminal_smiles or product,
        "reason": reasons[0] if reasons else "route_level_proposal_rejected",
        "last_valid_step": max(0, len(steps) - 1),
        "rejected_step": None,
        "proposal_reasons": list(reasons),
        "rxn_smiles": "",
    }


def _reaction_smiles(reactants: list[str], product: str) -> str:
    return ".".join(str(item) for item in reactants or [] if str(item or "")) + f">>{product}"
