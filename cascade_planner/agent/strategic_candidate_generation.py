"""Strategic candidate generation for SMILES-first literature workflow."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger

from cascade_planner.agent.evidence_cards import EvidenceCard


RDLogger.DisableLog("rdApp.*")
CANDIDATE_SCHEMA_VERSION = "literature_candidate.v1"
ALLOWED_CANDIDATE_KINDS = {
    "exact_fragment_retro",
    "forward_surrogate",
    "route_anchor",
}


@dataclass
class LiteratureCandidate:
    candidate_id: str
    case_id: str
    candidate_kind: str
    target_smiles: str
    product_smiles: str
    precursor_smiles: list[str] = field(default_factory=list)
    rxn_smiles: str = ""
    reaction_class: str = ""
    strategic_bond: str = ""
    literature_basis: str = ""
    use_case: str = "planning_material"
    confidence: str = "medium"
    evidence_refs: list[str] = field(default_factory=list)
    source_record_refs: list[str] = field(default_factory=list)
    not_lab_procedure: bool = False
    surrogate_reason: str = ""
    route_anchor_role: str = ""
    strategy_template: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "draft"
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def normalize(self) -> "LiteratureCandidate":
        if self.candidate_kind not in ALLOWED_CANDIDATE_KINDS:
            self.candidate_kind = "forward_surrogate"
        if self.candidate_kind == "forward_surrogate":
            self.not_lab_procedure = True
            if not self.surrogate_reason:
                self.surrogate_reason = "literature strategy instantiated as parseable planning surrogate"
        if self.candidate_kind == "route_anchor":
            self.use_case = "multi_step_anchor_planning_material"
            self.rxn_smiles = ""
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalize())


def generate_literature_candidates(
    *,
    case_id: str,
    target_smiles: str,
    frontier_smiles: str,
    evidence_cards: list[EvidenceCard],
) -> list[LiteratureCandidate]:
    usable = [
        card for card in evidence_cards
        if card.target_relation != "analogy_only" and card.validation_status != "draft_only"
    ]
    if not usable:
        return []

    strategic_cards = [card for card in usable if card.route_role == "strategic_disconnection"]
    anchor_cards = [card for card in usable if card.route_role == "route_anchor"]
    candidates: list[LiteratureCandidate] = []

    if strategic_cards:
        candidates.extend(_exact_fragment_candidates(case_id, target_smiles, frontier_smiles, strategic_cards[:2]))
        candidates.extend(_forward_surrogate_candidates(case_id, target_smiles, frontier_smiles, strategic_cards[:3]))
    for card in anchor_cards[:3]:
        candidates.append(_route_anchor_candidate(case_id, target_smiles, frontier_smiles, card))
    if not anchor_cards:
        for card in strategic_cards[:1]:
            candidates.append(_implicit_anchor_candidate(case_id, target_smiles, frontier_smiles, card))
    return [candidate.normalize() for candidate in candidates]


def validate_literature_candidate(candidate_or_data: LiteratureCandidate | dict[str, Any]) -> dict[str, Any]:
    candidate = candidate_or_data if isinstance(candidate_or_data, LiteratureCandidate) else candidate_from_dict(candidate_or_data)
    reasons: list[str] = []
    if candidate.candidate_kind not in ALLOWED_CANDIDATE_KINDS:
        reasons.append("invalid_candidate_kind")
    if not candidate.candidate_id:
        reasons.append("missing_candidate_id")
    if not candidate.evidence_refs:
        reasons.append("missing_evidence_refs")
    if not _valid_smiles(candidate.target_smiles):
        reasons.append("invalid_target_smiles")
    if candidate.product_smiles and not _valid_smiles(candidate.product_smiles):
        reasons.append("invalid_product_smiles")
    for smi in candidate.precursor_smiles:
        if smi and not _valid_smiles(smi):
            reasons.append("invalid_precursor_smiles")
            break
    if candidate.candidate_kind == "forward_surrogate":
        if not candidate.not_lab_procedure:
            reasons.append("forward_surrogate_missing_not_lab_procedure")
        if not candidate.surrogate_reason:
            reasons.append("forward_surrogate_missing_surrogate_reason")
        if not _valid_rxn_smiles(candidate.rxn_smiles):
            reasons.append("invalid_forward_surrogate_rxn_smiles")
    elif candidate.candidate_kind == "route_anchor":
        if candidate.rxn_smiles:
            reasons.append("route_anchor_must_not_have_single_step_rxn")
    elif candidate.candidate_kind == "exact_fragment_retro":
        if not candidate.precursor_smiles:
            reasons.append("exact_fragment_retro_missing_precursors")
        if candidate.rxn_smiles and not _valid_rxn_smiles(candidate.rxn_smiles, allow_dummy=True):
            reasons.append("invalid_exact_fragment_rxn_smiles")

    accepted = not reasons
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_kind": candidate.candidate_kind,
        "accepted": accepted,
        "validation_status": "validated" if accepted else "rejected",
        "reasons": sorted(set(reasons)),
    }


def candidate_from_dict(data: dict[str, Any]) -> LiteratureCandidate:
    allowed = set(LiteratureCandidate.__dataclass_fields__)
    kwargs = {key: value for key, value in data.items() if key in allowed}
    return LiteratureCandidate(**kwargs).normalize()


def write_candidates_jsonl(candidates: Iterable[LiteratureCandidate], path: str | Path) -> None:
    rows = []
    for candidate in candidates:
        validation = validate_literature_candidate(candidate)
        candidate.validation_status = validation["validation_status"]
        rows.append(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True))
    Path(path).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def load_candidates_jsonl(path: str | Path) -> list[LiteratureCandidate]:
    candidates: list[LiteratureCandidate] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            candidates.append(candidate_from_dict(json.loads(line)))
    return candidates


def _exact_fragment_candidates(
    case_id: str,
    target_smiles: str,
    frontier_smiles: str,
    cards: list[EvidenceCard],
) -> list[LiteratureCandidate]:
    cuts = _cut_frontier(frontier_smiles)
    if not cuts:
        return []
    fragments, bond_label = cuts
    candidates = []
    for idx, card in enumerate(cards, start=1):
        candidates.append(LiteratureCandidate(
            candidate_id=f"{case_id}_exact_fragment_retro_{idx}",
            case_id=case_id,
            candidate_kind="exact_fragment_retro",
            target_smiles=target_smiles,
            product_smiles=frontier_smiles,
            precursor_smiles=fragments,
            rxn_smiles=".".join(fragments) + ">>" + frontier_smiles,
            reaction_class=_reaction_class(card),
            strategic_bond=bond_label,
            literature_basis=card.source_title,
            use_case="topology_exact_frontier_disconnection",
            confidence=card.confidence,
            evidence_refs=[card.evidence_id],
            source_record_refs=[card.source_record_id] if card.source_record_id else [],
            strategy_template=_template_from_card(card, "exact_fragment_retro"),
        ))
    return candidates


def _forward_surrogate_candidates(
    case_id: str,
    target_smiles: str,
    frontier_smiles: str,
    cards: list[EvidenceCard],
) -> list[LiteratureCandidate]:
    candidates = []
    for idx, card in enumerate(cards, start=1):
        rxn = _surrogate_rxn(card)
        lhs, rhs = rxn.split(">>", 1)
        candidates.append(LiteratureCandidate(
            candidate_id=f"{case_id}_forward_surrogate_{idx}",
            case_id=case_id,
            candidate_kind="forward_surrogate",
            target_smiles=target_smiles,
            product_smiles=rhs,
            precursor_smiles=[part for part in lhs.split(".") if part],
            rxn_smiles=rxn,
            reaction_class=_reaction_class(card),
            strategic_bond=_strategic_bond(card),
            literature_basis=card.source_title,
            use_case="planner_policy_or_template_seed_only",
            confidence=card.confidence,
            evidence_refs=[card.evidence_id],
            source_record_refs=[card.source_record_id] if card.source_record_id else [],
            not_lab_procedure=True,
            surrogate_reason=(
                "Representative parseable forward reaction for policy/template tests; "
                "not claimed to be the literature substrate or lab procedure."
            ),
            strategy_template=_template_from_card(card, "forward_surrogate"),
        ))
    return candidates


def _route_anchor_candidate(
    case_id: str,
    target_smiles: str,
    frontier_smiles: str,
    card: EvidenceCard,
) -> LiteratureCandidate:
    record = card.source_metadata.get("record") or {}
    anchor_smiles = str(record.get("smiles") or "")
    return LiteratureCandidate(
        candidate_id=f"{case_id}_route_anchor_{card.source_record_id or card.evidence_id}",
        case_id=case_id,
        candidate_kind="route_anchor",
        target_smiles=target_smiles,
        product_smiles=frontier_smiles,
        precursor_smiles=[anchor_smiles] if anchor_smiles else [],
        rxn_smiles="",
        reaction_class="route_anchor",
        strategic_bond="multi_step_anchor",
        literature_basis=card.source_title,
        use_case="multi_step_anchor_planning_material",
        confidence=card.confidence,
        evidence_refs=[card.evidence_id],
        source_record_refs=[card.source_record_id] if card.source_record_id else [],
        route_anchor_role=card.route_role_detail or "literature_anchor",
        strategy_template=_template_from_card(card, "route_anchor"),
    )


def _implicit_anchor_candidate(
    case_id: str,
    target_smiles: str,
    frontier_smiles: str,
    card: EvidenceCard,
) -> LiteratureCandidate:
    return LiteratureCandidate(
        candidate_id=f"{case_id}_route_anchor_from_{card.source_record_id or card.evidence_id}",
        case_id=case_id,
        candidate_kind="route_anchor",
        target_smiles=target_smiles,
        product_smiles=frontier_smiles,
        precursor_smiles=[],
        rxn_smiles="",
        reaction_class="route_anchor",
        strategic_bond="literature_upstream_anchor",
        literature_basis=card.source_title,
        use_case="multi_step_anchor_planning_material",
        confidence=card.confidence,
        evidence_refs=[card.evidence_id],
        source_record_refs=[card.source_record_id] if card.source_record_id else [],
        route_anchor_role="implicit_anchor_from_strategic_disconnection",
        strategy_template=_template_from_card(card, "route_anchor"),
    )


def _cut_frontier(smiles: str) -> tuple[list[str], str] | None:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    candidate_bonds = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        one_ring = a.IsInRing() ^ b.IsInRing()
        candidate_bonds.append((0 if one_ring else 1, bond.GetIdx(), a.GetIdx(), b.GetIdx()))
    if not candidate_bonds:
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                candidate_bonds.append((1, bond.GetIdx(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    if not candidate_bonds:
        return None
    _, bond_idx, a_idx, b_idx = sorted(candidate_bonds)[0]
    frag = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
    parts = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    fragments = [Chem.MolToSmiles(part, isomericSmiles=True) for part in parts if part.GetNumAtoms() > 0]
    if len(fragments) < 2:
        return None
    ring_note = "side_chain" if ring_info.NumRings() else "acyclic"
    return fragments, f"{ring_note}_bond_atoms_{a_idx}_{b_idx}"


def _reaction_class(card: EvidenceCard) -> str:
    text = json.dumps(card.source_metadata.get("record") or {}, ensure_ascii=False).lower()
    if "pyrone" in text or "coupling" in text:
        return "C_C_coupling"
    if "taxane" in card.family_id.lower() or "paclitaxel" in text or "baccatin" in text:
        return "taxane_side_chain_acylation"
    if "artemisinin" in card.family_id.lower() or "peroxide" in text or "photooxidation" in text:
        return "late_stage_peroxide_formation"
    if "natural_statin" in card.family_id.lower() or "fermentation core" in text:
        return "statin_semisynthesis"
    if "synthetic_statin" in card.family_id.lower() or "syn-3,5" in text or "hwe" in text:
        return "statin_side_chain_convergence"
    if "prostaglandin" in card.family_id.lower() or "corey" in text:
        return "corey_lactone_sidechain_installation"
    if "macrolactonization" in text or "lactone" in text:
        return "macrolactonization"
    if "glycos" in text or "sugar" in text:
        return "glycosylation"
    if "pictet" in text:
        return "Pictet-Spengler"
    if "aldol" in text:
        return "aldol_fragment_assembly"
    return "strategic_disconnection"


def _strategic_bond(card: EvidenceCard) -> str:
    move = (card.source_metadata.get("record") or {}).get("retrosynthetic_move") or {}
    bonds = move.get("break_bonds") or []
    return str(bonds[0]) if bonds else "literature_strategic_bond"


def _surrogate_rxn(card: EvidenceCard) -> str:
    family = str(card.family_id or "").lower()
    text = json.dumps(card.source_metadata.get("record") or {}, ensure_ascii=False).lower()
    if "bufadienolide" in family or "pyrone" in text:
        return "C=CBr.O=C1OC=CC=C1>>C=Cc1ccoc(=O)c1"
    if "taxane" in family or "paclitaxel" in text or "baccatin" in text:
        return "CC(=O)O.O=C(O)C(N)C1=CC=CC=C1>>CC(=O)OC(=O)C(N)C1=CC=CC=C1"
    if "artemisinin" in family or "peroxide" in text:
        return "CC(C)=CCC1CC(=O)OC1>>CC(C)C1OC2OOCC1CC2=O"
    if "natural_statin" in family or "fermentation core" in text:
        return "CC(=O)O.CC1CCOC(=O)C1>>CC(=O)OC1CCOC(=O)C1"
    if "synthetic_statin" in family or "syn-3,5" in text or "hwe" in text:
        return "O=CCC(O)CC(O)C(=O)O.Cc1ccccc1>>Cc1ccccc1C=CCC(O)CC(O)C(=O)O"
    if "prostaglandin" in family or "corey" in text:
        return "O=CC1CCC(=O)O1.CCCCCCCC=O>>CCCCCCCC=CC1CCC(=O)O1"
    if "macrocycle" in family or "macrolactonization" in text:
        return "O=C(O)CCCCCCCCCCO>>O=C1OCCCCCCCCCC1"
    if "glycoside" in family or "glycos" in text or "sugar" in text:
        return "OC1COC(O)C(O)C1O.Oc1ccccc1>>Oc1ccccc1OC1COC(O)C(O)C1O"
    if "alkaloid" in family or "pictet" in text:
        return "NCCc1ccc(O)cc1.O=CO>>c1cc(O)ccc1C1NCCC1"
    return "CC(=O)O.CCO>>CC(=O)OCC"


def _template_from_card(card: EvidenceCard, candidate_kind: str) -> dict[str, Any]:
    record = card.source_metadata.get("record") or {}
    move = record.get("retrosynthetic_move") or {}
    return {
        "template_schema": "advisory_strategy_template.v1",
        "template_level": "advisory_strategy",
        "candidate_kind": candidate_kind,
        "family_id": card.family_id,
        "source_record_id": card.source_record_id,
        "reaction_class": _reaction_class(card),
        "break_bonds": move.get("break_bonds") or [],
        "forward_logic": move.get("forward_logic") or [],
        "suggested_precursor_roles": move.get("suggested_precursor_roles") or [],
        "planner_hint": move.get("planner_hint") or card.route_role_detail,
        "use_policy": record.get("use_policy") or {},
        "direct_one_step_consumption_allowed": False,
        "executable_template_candidate": False,
        "requires_product_specific_applicability_report": True,
        "not_raw_reaction_injection": True,
    }


def _valid_smiles(smiles: str, *, allow_dummy: bool = True) -> bool:
    if not smiles:
        return True
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    if not allow_dummy and any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return False
    return True


def _valid_rxn_smiles(rxn_smiles: str, *, allow_dummy: bool = False) -> bool:
    if not rxn_smiles or rxn_smiles.count(">>") != 1:
        return False
    lhs, rhs = rxn_smiles.split(">>", 1)
    if not lhs or not rhs:
        return False
    parts = [part for side in (lhs, rhs) for part in side.split(".") if part]
    return bool(parts) and all(_valid_smiles(part, allow_dummy=allow_dummy) for part in parts)
