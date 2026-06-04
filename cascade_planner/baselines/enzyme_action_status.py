"""Structured enzyme action and route-level enzyme status helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


RDLogger.DisableLog("rdApp.*")

ENZYME_ACTION_SCHEMA = "structured_enzyme_action.v1"
ROUTE_ENZYME_STATUS_SCHEMA = "route_enzyme_status.v1"
ENZYMATIC_SOURCES = {
    "autoplanner.enzyme_precedent",
    "enzyme_precedent",
    "chem_enzy_onmt",
    "chem_enzy_bionav",
    "enzyformer",
    "enzexpand",
    "v3_retrieval",
    "retrorules",
    "rhea",
    "rhea_template",
    "retrieval",
    "enzymatic",
}


@dataclass
class StructuredEnzymeAction:
    action_id: str
    substrate_smiles: str
    product_smiles: str
    ec_numbers: list[str] = field(default_factory=list)
    reaction_center: dict[str, Any] = field(default_factory=dict)
    precedent: dict[str, Any] = field(default_factory=dict)
    verifier_score: float | None = None
    verifier_threshold: float | None = None
    verifier_accepted: bool | None = None
    source_evidence: dict[str, Any] = field(default_factory=dict)
    cofactor_flags: dict[str, Any] = field(default_factory=dict)
    common_metabolite_flags: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    step_index: int = 0
    validation_status: str = "unknown"
    schema_version: str = ENZYME_ACTION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def structured_enzyme_action_from_slot(slot: Any) -> dict[str, Any] | None:
    """Build a normalized enzyme-action payload from a CascadeBoard slot."""
    source = str(getattr(slot, "source", "") or "")
    evidence = getattr(slot, "evidence", {}) or {}
    if not isinstance(evidence, dict):
        evidence = {}
    ec_numbers = _ec_numbers(slot, evidence)
    if not ec_numbers and not _is_enzyme_source(source):
        return None
    substrate = str(getattr(slot, "main_reactant", "") or "")
    product = str(getattr(slot, "product", "") or "")
    sp_payload = evidence.get("enzyme_sp_verifier_v1") if isinstance(evidence.get("enzyme_sp_verifier_v1"), dict) else {}
    quality = evidence.get("enzyme_step_quality_v1") if isinstance(evidence.get("enzyme_step_quality_v1"), dict) else {}
    action = StructuredEnzymeAction(
        action_id=f"enzyme_action_step_{int(getattr(slot, 'index', 0) or 0)}",
        substrate_smiles=substrate,
        product_smiles=product,
        ec_numbers=ec_numbers,
        reaction_center={
            "reaction_smiles": str(getattr(slot, "reaction_smiles", "") or ""),
            "reaction_type": str(getattr(slot, "reaction_type", "") or ""),
            "canonical_substrate": canonical_smiles(substrate) or "",
            "canonical_product": canonical_smiles(product) or "",
        },
        precedent=_precedent_payload(evidence),
        verifier_score=_float_or_none(sp_payload.get("score")),
        verifier_threshold=_float_or_none(sp_payload.get("threshold")),
        verifier_accepted=bool(sp_payload.get("accepted")) if sp_payload else None,
        source_evidence={
            "source": source,
            "source_gate": evidence.get("source_gate") or {},
            "quality_decision": quality.get("decision") or "",
            "quality_flags": list(quality.get("flags") or []),
            "candidate_provenance": _is_enzyme_source(source),
            "posthoc_ec_annotation_only": bool(ec_numbers and not _is_enzyme_source(source)),
        },
        cofactor_flags=_cofactor_flags(slot, evidence),
        common_metabolite_flags=_common_metabolite_flags(slot),
        source=source,
        step_index=int(getattr(slot, "index", 0) or 0),
        validation_status=_enzyme_action_status(source=source, ec_numbers=ec_numbers, sp_payload=sp_payload, quality=quality),
    )
    return action.to_dict()


def route_level_enzyme_status(board: Any) -> dict[str, Any]:
    actions = [
        action for slot in getattr(board, "slots", []) or []
        for action in [structured_enzyme_action_from_slot(slot)]
        if action is not None
    ]
    counts: dict[str, int] = {}
    for action in actions:
        status = str(action.get("validation_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    if not actions:
        route_status = "unknown"
    elif any(action.get("validation_status") == "rejected" for action in actions):
        route_status = "rejected"
    elif any(action.get("validation_status") == "validated" for action in actions):
        route_status = "validated"
    elif any(action.get("validation_status") == "generic_ec_only" for action in actions):
        route_status = "generic_ec_only"
    else:
        route_status = "unknown"
    partial_evidence = _partial_evidence_route(board)
    return {
        "schema_version": ROUTE_ENZYME_STATUS_SCHEMA,
        "route_enzyme_status": route_status,
        "enzyme_action_count": len(actions),
        "status_counts": counts,
        "validated_enzyme_steps": counts.get("validated", 0),
        "generic_ec_only_steps": counts.get("generic_ec_only", 0),
        "rejected_enzyme_steps": counts.get("rejected", 0),
        "unknown_enzyme_steps": counts.get("unknown", 0),
        "actions": actions,
        "partial_evidence_only": partial_evidence,
        "production_solved_allowed": (
            not partial_evidence
            and route_status in {"validated", "unknown"}
            and counts.get("rejected", 0) == 0
        ),
    }


def _enzyme_action_status(
    *,
    source: str,
    ec_numbers: list[str],
    sp_payload: dict[str, Any],
    quality: dict[str, Any],
) -> str:
    if quality.get("decision") == "reject" or sp_payload.get("accepted") is False:
        return "rejected"
    if sp_payload.get("accepted") is True and (_is_enzyme_source(source) or ec_numbers):
        return "validated"
    if ec_numbers and not _is_enzyme_source(source):
        return "generic_ec_only"
    if _is_enzyme_source(source):
        return "unknown"
    return "unknown"


def _ec_numbers(slot: Any, evidence: dict[str, Any]) -> list[str]:
    values = []
    ec = str(getattr(slot, "ec", "") or "")
    if ec:
        values.append(ec)
    source_gate = evidence.get("source_gate") or {}
    flags = source_gate.get("molecule_flags") if isinstance(source_gate, dict) else {}
    raw = (flags or {}).get("bridge_gate_ec_numbers") or []
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    values.extend(str(item) for item in raw if item)
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _precedent_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "selected_enzyme_precedent",
        "literature_precedent",
        "doi",
        "pmid",
        "rhea_ids",
        "source_db",
        "substrate_similarity",
        "reaction_center_similarity",
    )
    return {key: evidence.get(key) for key in keys if evidence.get(key) not in (None, "", [], {})}


def _cofactor_flags(slot: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(item or "") for item in [
        getattr(slot, "reaction_smiles", ""),
        getattr(slot, "catalyst", ""),
        evidence.get("cofactor"),
        evidence.get("cofactor_required"),
    ]).lower()
    return {
        "requires_cofactor": any(token in text for token in ("nad", "nadp", "fad", "sam", "coa")),
        "cofactor": evidence.get("cofactor") or evidence.get("cofactor_required") or "",
    }


def _common_metabolite_flags(slot: Any) -> dict[str, Any]:
    smiles = [str(getattr(slot, "main_reactant", "") or ""), *list(getattr(slot, "aux_reactants", []) or [])]
    common = {"O", "N", "C", "CO", "CCO", "O=O", "[H][H]"}
    hits = [smi for smi in smiles if smi in common or _heavy_atoms(smi) <= 3]
    return {"common_metabolite_like_reactants": hits, "count": len(hits)}


def _partial_evidence_route(board: Any) -> bool:
    constraints = getattr(board, "global_constraints", {}) or {}
    if not isinstance(constraints, dict):
        return False
    for key in ("p5_partial_evidence", "partial_evidence_route", "diagnostic_only", "p5_diagnostic_only"):
        if bool(constraints.get(key)):
            return True
    evidence_scope = str(constraints.get("evidence_scope") or constraints.get("route_evidence_scope") or "").lower()
    return evidence_scope in {"p5_partial", "partial_evidence", "diagnostic_only"}


def _is_enzyme_source(source: str) -> bool:
    source_l = str(source or "").lower()
    return source_l in ENZYMATIC_SOURCES or "enzyme" in source_l or "bionav" in source_l


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out
