"""Structured route export and lightweight route-level diagnostics.

This module is the factual contract for CLI JSON, benchmark artifacts, and
future LLM route critique. It intentionally exports only fields present in
RouteResult/CascadeBoard plus deterministic diagnostics derived from them.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from cascade_planner.cascadeboard import CascadeBoard, RouteExplanation, RouteResult, Slot
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles


StockChecker = Callable[[str], bool]

ORGANIC_SOLVENT_RISK = {
    "dmf", "dmso", "dcm", "ch2cl2", "chloroform", "toluene", "thf",
    "acetonitrile", "mecn", "dioxane", "hexane", "ethyl acetate",
}
METAL_TOKENS = {"pd", "ni", "cu", "ru", "rh", "ir", "pt", "ag", "au"}
OXIDANT_TOKENS = {"h2o2", "mcpba", "tempo", "naio4", "kmno4", "o2", "oxygen"}
REDUCTANT_TOKENS = {"nabh4", "libh4", "lialh4", "h2", "nadh", "nadph", "bh4"}
COFACTOR_OX = {"nad+", "nadp+", "nad(p)+"}
COFACTOR_RED = {"nadh", "nadph", "nad(p)h"}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _jsonable(value: Any) -> Any:
    """Convert common non-JSON-native values to plain Python objects."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _candidate_to_dict(cand: dict[str, Any]) -> dict[str, Any]:
    out = {
        "main_reactant": cand.get("main_reactant", ""),
        "aux_reactants": list(cand.get("aux_reactants") or []),
        "reaction_smiles": cand.get("rxn_smiles", cand.get("reaction_smiles", "")),
        "reaction_type": cand.get("type", ""),
        "ec": cand.get("ec", ""),
        "enzyme_uid": cand.get("enzyme_uid", ""),
        "catalyst": cand.get("catalyst", ""),
        "T": _safe_float(cand.get("T")),
        "pH": _safe_float(cand.get("pH")),
        "solvent": cand.get("solvent", ""),
        "source": cand.get("source", ""),
        "score": _safe_float(cand.get("score")),
        "value_score": _safe_float(cand.get("value_score")),
        "value_probability": _safe_float(cand.get("value_probability")),
        "candidate_ranker_score": _safe_float(cand.get("candidate_ranker_score")),
    }
    evidence = cand.get("evidence") or {}
    for key in (
        "uniprot_accession",
        "uniprot_status",
        "organism",
        "tax_id",
        "sequence",
        "sequence_length",
        "protein_existence",
        "reviewed",
        "rhea_ids",
        "cofactor",
        "cofactor_regeneration_mode",
        "doi",
        "pmid",
        "literature_title",
        "cascade_id",
        "step_id",
        "substrate_similarity",
        "reaction_center_similarity",
        "condition_match",
        "literature_precedent",
        "enzyme_name",
        "biocatalyst_format",
        "engineering_status",
        "source_db",
    ):
        value = cand.get(key, evidence.get(key))
        if value not in (None, "", [], {}):
            out[key] = _jsonable(value)
    if evidence:
        out["evidence"] = _jsonable(evidence)
    return out


def _candidate_reaction_key(cand: dict[str, Any]) -> str:
    rxn = cand.get("rxn_smiles") or cand.get("reaction_smiles") or ""
    return canonical_reaction(rxn)


def _candidate_reactant_set_key(cand: dict[str, Any]) -> str:
    rxn = cand.get("rxn_smiles") or cand.get("reaction_smiles") or ""
    parts: tuple[str, ...] = ()
    if rxn and ">>" in rxn:
        parts = canonical_side(rxn.split(">>", 1)[0])
    if not parts:
        reactants = [canonical_smiles(cand.get("main_reactant") or "")]
        reactants.extend(canonical_smiles(smi) for smi in cand.get("aux_reactants") or [])
        parts = tuple(sorted(smi for smi in reactants if smi))
    return ".".join(parts)


def _key_stats(keys: list[str]) -> dict[str, Any]:
    valid = [key for key in keys if key]
    unique = len(set(valid))
    return {
        "valid": len(valid),
        "unique": unique,
        "coverage": round(len(valid) / len(keys), 4) if keys else None,
        "duplicate_fraction": round(1.0 - unique / len(valid), 4) if valid else None,
        "unique_fraction": round(unique / len(valid), 4) if valid else None,
    }


def candidate_pool_summary(slot: Slot, limit: int = 20) -> dict[str, Any]:
    cands = list(slot.candidates or [])
    reactions = [_candidate_reaction_key(c) for c in cands]
    main_reactants = [canonical_smiles(c.get("main_reactant") or "") for c in cands]
    reactant_sets = [_candidate_reactant_set_key(c) for c in cands]
    reaction_stats = _key_stats(reactions)
    main_stats = _key_stats(main_reactants)
    reactant_set_stats = _key_stats(reactant_sets)
    diversity_terms = [
        x
        for x in (
            reaction_stats["unique_fraction"],
            main_stats["unique_fraction"],
            reactant_set_stats["unique_fraction"],
        )
        if x is not None
    ]
    return {
        "n_candidates": len(cands),
        "source_counts": dict(Counter(c.get("source") or "unknown" for c in cands)),
        "unique_reactions": reaction_stats["unique"],
        "unique_main_reactants": main_stats["unique"],
        "unique_reactant_sets": reactant_set_stats["unique"],
        "reaction_coverage": reaction_stats["coverage"],
        "main_reactant_coverage": main_stats["coverage"],
        "reactant_set_coverage": reactant_set_stats["coverage"],
        "duplicate_reaction_fraction": reaction_stats["duplicate_fraction"],
        "duplicate_main_reactant_fraction": main_stats["duplicate_fraction"],
        "duplicate_reactant_set_fraction": reactant_set_stats["duplicate_fraction"],
        "pool_diversity_score": round(sum(diversity_terms) / len(diversity_terms), 4) if diversity_terms else None,
        "diversity_flags": {
            "has_duplicate_reactions": bool(
                reaction_stats["duplicate_fraction"] is not None
                and reaction_stats["duplicate_fraction"] > 0.0
            ),
            "has_duplicate_main_reactants": bool(
                main_stats["duplicate_fraction"] is not None
                and main_stats["duplicate_fraction"] > 0.0
            ),
            "has_duplicate_reactant_sets": bool(
                reactant_set_stats["duplicate_fraction"] is not None
                and reactant_set_stats["duplicate_fraction"] > 0.0
            ),
            "single_reactant_set_pool": bool(
                len(cands) > 1
                and reactant_set_stats["valid"] > 1
                and reactant_set_stats["unique"] == 1
            ),
        },
        "top_candidates": [_candidate_to_dict(c) for c in cands[:limit]],
    }


def _mol_features(smiles: str | None) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {
            "valid": False,
            "heavy_atoms": None,
            "hetero_atoms": None,
            "ring_count": None,
            "formula": "",
        }
    heavy_atoms = mol.GetNumHeavyAtoms()
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    return {
        "valid": True,
        "heavy_atoms": heavy_atoms,
        "hetero_atoms": hetero_atoms,
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }


def _ec_principle(ec: str | None) -> str:
    if not ec:
        return ""
    ec1 = str(ec).split(".", 1)[0]
    return {
        "1": "EC 1 oxidoreductase: electron or hydrogen transfer; often oxidation/reduction with NAD(P), flavin, oxygen, or metal cofactors.",
        "2": "EC 2 transferase: transfers a functional group such as glycosyl, acyl, methyl, amino, or phosphate from a donor.",
        "3": "EC 3 hydrolase: cleaves a bond by addition of water, commonly ester, amide, glycosidic, or phosphate hydrolysis.",
        "4": "EC 4 lyase: forms or cleaves bonds without hydrolysis or redox, often elimination/addition across unsaturation.",
        "5": "EC 5 isomerase: rearranges atoms within the same molecular formula.",
        "6": "EC 6 ligase: joins two fragments with nucleotide-triphosphate energy input.",
        "7": "EC 7 translocase: couples transport to a chemical energy process; unusual for small-molecule synthesis.",
    }.get(ec1, "")


def _reaction_type_principle(reaction_type: str | None) -> str:
    key = (reaction_type or "").strip().lower().replace("_", " ")
    rules = [
        ("glycosyl", "Glycosylation: forms a glycosidic C-O, C-N, C-S, or C-C linkage by transferring a sugar unit from an activated donor."),
        ("hydrolysis", "Hydrolysis: water cleaves a labile bond; common targets are esters, amides, glycosides, acetals, and phosphates."),
        ("reduction", "Reduction: adds hydride, hydrogen, or electrons and lowers oxidation state; carbonyl-to-alcohol and alkene hydrogenation are common cases."),
        ("oxidation", "Oxidation: removes hydrogen/electrons or introduces oxygen, raising oxidation state."),
        ("amination", "Amination: installs or exchanges a nitrogen substituent, usually through C-N bond formation."),
        ("transamination", "Transamination: transfers an amino group between a donor and an acceptor, typically PLP-dependent."),
        ("acyl", "Acyl transfer/acylation: transfers an acyl group to O, N, S, or C nucleophiles."),
        ("ester", "Esterification/transesterification: forms or exchanges an ester linkage between an alcohol and an acyl donor."),
        ("alkyl", "Alkylation: installs an alkyl group, commonly through substitution or transferase chemistry."),
        ("methyl", "Methylation: transfers a methyl group, often from SAM or a chemical methyl donor."),
        ("phosph", "Phosphorylation: installs or transfers a phosphate group."),
        ("coupling", "Coupling: joins two fragments, often by forming a C-C, C-N, C-O, or C-S bond."),
        ("c-c", "C-C bond formation: joins carbon frameworks through coupling, aldol-type addition, alkylation, or enzyme-mediated ligation."),
        ("deprotect", "Deprotection: removes a protecting group to reveal a functional group for later chemistry."),
        ("isomer", "Isomerization: rearranges connectivity or stereochemistry without large atom-count change."),
        ("other", "Other transformation: the model found a candidate transformation but the reaction class is too broad for a specific mechanism label."),
    ]
    for token, text in rules:
        if token in key:
            return text
    return "General transformation: inspect the reaction SMILES and candidate evidence before treating the mechanism as reliable."


def reaction_interpretation(slot: Slot) -> dict[str, Any]:
    """Return conservative, deterministic chemistry notes for a rendered step."""
    rtype = slot.reaction_type or ""
    main = slot.main_reactant or ""
    product = slot.product or ""
    aux = list(slot.aux_reactants or [])
    main_features = _mol_features(main)
    product_features = _mol_features(product)

    heavy_delta = None
    hetero_delta = None
    ring_delta = None
    if main_features["valid"] and product_features["valid"]:
        heavy_delta = int(product_features["heavy_atoms"] - main_features["heavy_atoms"])
        hetero_delta = int(product_features["hetero_atoms"] - main_features["hetero_atoms"])
        ring_delta = int(product_features["ring_count"] - main_features["ring_count"])

    atom_change_notes: list[str] = []
    if heavy_delta is not None:
        if heavy_delta > 0:
            atom_change_notes.append(f"product has {heavy_delta} more heavy atom(s) than the main precursor")
        elif heavy_delta < 0:
            atom_change_notes.append(f"product has {-heavy_delta} fewer heavy atom(s) than the main precursor")
        else:
            atom_change_notes.append("main precursor and product have the same heavy-atom count")
    if hetero_delta:
        direction = "more" if hetero_delta > 0 else "fewer"
        atom_change_notes.append(f"product has {abs(hetero_delta)} {direction} hetero atom(s)")
    if ring_delta:
        direction = "more" if ring_delta > 0 else "fewer"
        atom_change_notes.append(f"product has {abs(ring_delta)} {direction} ring(s)")

    added_or_removed: list[str] = []
    if aux:
        added_or_removed.append("auxiliary reactant/coupling partner: " + " . ".join(aux))
    if heavy_delta and heavy_delta > 0 and not aux:
        added_or_removed.append("additional atoms likely come from an implicit reagent, donor, protecting group, or unmapped partner")
    if heavy_delta and heavy_delta < 0:
        added_or_removed.append("the displayed main precursor is larger than the product; check whether this is a degradation, protecting-group removal, or candidate artifact")

    catalyst_notes: list[str] = []
    if slot.ec:
        catalyst_notes.append(_ec_principle(slot.ec))
    if slot.catalyst:
        catalyst_notes.append(f"catalyst field: {slot.catalyst}")
    if slot.solvent:
        catalyst_notes.append(f"solvent: {slot.solvent}")
    if slot.T is not None or slot.pH is not None:
        catalyst_notes.append(f"conditions: T={_safe_float(slot.T)} C, pH={_safe_float(slot.pH)}")

    return {
        "reaction_class": rtype or "unknown",
        "forward_summary": f"{rtype or 'Transformation'} converts the displayed precursor into the product.",
        "reaction_principle": _reaction_type_principle(rtype),
        "ec_principle": _ec_principle(slot.ec),
        "atom_change": {
            "main_precursor": main_features,
            "product": product_features,
            "heavy_atom_delta": heavy_delta,
            "hetero_atom_delta": hetero_delta,
            "ring_delta": ring_delta,
            "notes": atom_change_notes,
        },
        "likely_added_or_removed": added_or_removed,
        "catalysis_and_conditions": [x for x in catalyst_notes if x],
        "confidence_note": "Heuristic explanation from exported route fields; verify against mapped reaction chemistry before execution.",
    }


def slot_to_dict(slot: Slot, stock_checker: StockChecker | None = None) -> dict[str, Any]:
    """Export a CascadeBoard slot with candidate provenance and stock status."""
    terminal_reactants = []
    if slot.main_reactant:
        terminal_reactants.append(slot.main_reactant)
    terminal_reactants.extend(slot.aux_reactants or [])

    stock_status = None
    if stock_checker is not None and terminal_reactants:
        stock_status = {
            smi: bool(stock_checker(smi))
            for smi in terminal_reactants
            if smi
        }

    return {
        "index": slot.index,
        "product": slot.product,
        "main_reactant": slot.main_reactant,
        "aux_reactants": list(slot.aux_reactants or []),
        "reaction_smiles": slot.reaction_smiles,
        "reaction_type": slot.reaction_type,
        "ec": slot.ec,
        "enzyme_uid": slot.enzyme_uid,
        "catalyst": slot.catalyst,
        "T": _safe_float(slot.T),
        "pH": _safe_float(slot.pH),
        "solvent": slot.solvent,
        "evidence": _jsonable(slot.evidence),
        "source": slot.source or "",
        "scores": {
            "retro": _safe_float(slot.e_retro),
            "enzyme": _safe_float(slot.e_enzyme),
            "condition": _safe_float(slot.e_condition),
            "confidence": _safe_float(slot.confidence),
        },
        "fixed_fields": sorted(slot.fixed_fields),
        "is_filled": bool(slot.reaction_smiles),
        "is_enzymatic": bool(slot.ec),
        "stock_status": stock_status,
        "reaction_interpretation": reaction_interpretation(slot),
        "candidate_pool": candidate_pool_summary(slot),
    }


def condition_window_metrics(
    board: CascadeBoard,
    max_delta_T: float = 15.0,
    max_delta_pH: float = 1.5,
) -> dict[str, Any]:
    """Compute simple cross-step T/pH compatibility diagnostics."""
    Ts = [_safe_float(s.T) for s in board.slots if _safe_float(s.T) is not None]
    pHs = [_safe_float(s.pH) for s in board.slots if _safe_float(s.pH) is not None]

    adjacent_delta_T = []
    adjacent_delta_pH = []
    for a, b in zip(board.slots, board.slots[1:]):
        aT, bT = _safe_float(a.T), _safe_float(b.T)
        apH, bpH = _safe_float(a.pH), _safe_float(b.pH)
        if aT is not None and bT is not None:
            adjacent_delta_T.append(abs(aT - bT))
        if apH is not None and bpH is not None:
            adjacent_delta_pH.append(abs(apH - bpH))

    max_adj_T = max(adjacent_delta_T) if adjacent_delta_T else None
    max_adj_pH = max(adjacent_delta_pH) if adjacent_delta_pH else None
    T_span = (max(Ts) - min(Ts)) if len(Ts) >= 2 else 0.0 if Ts else None
    pH_span = (max(pHs) - min(pHs)) if len(pHs) >= 2 else 0.0 if pHs else None

    has_all_T = len(Ts) == board.n_steps
    has_all_pH = len(pHs) == board.n_steps
    T_ok = has_all_T and (max_adj_T is None or max_adj_T <= max_delta_T)
    pH_ok = has_all_pH and (max_adj_pH is None or max_adj_pH <= max_delta_pH)

    return {
        "has_all_T": has_all_T,
        "has_all_pH": has_all_pH,
        "T_span": T_span,
        "pH_span": pH_span,
        "max_adjacent_delta_T": max_adj_T,
        "max_adjacent_delta_pH": max_adj_pH,
        "max_delta_T_threshold": max_delta_T,
        "max_delta_pH_threshold": max_delta_pH,
        "condition_window_success": bool(T_ok and pH_ok),
    }


def enzyme_evidence_metrics(board: CascadeBoard) -> dict[str, Any]:
    """Summarize evidence coverage for enzymatic slots."""
    enzymatic = [s for s in board.slots if s.ec]
    if not enzymatic:
        return {
            "n_enzymatic_steps": 0,
            "enzyme_evidence_coverage": None,
            "supported_steps": 0,
            "steps": [],
        }

    rows = []
    supported = 0
    scores = []
    for s in enzymatic:
        source = s.source or ""
        evidence = s.evidence or {}
        has_candidate_source = source in {"enzyformer", "v3_retrieval", "enzexpand", "enzymatic"}
        has_score = (s.e_enzyme is not None and s.e_enzyme > 0) or source == "v3_retrieval"
        uniprot = evidence.get("uniprot_accession") or evidence.get("uniprot_id") or s.enzyme_uid
        organism = evidence.get("organism") or evidence.get("uniprot_lookup_organism")
        cofactor = evidence.get("cofactor") or evidence.get("cofactor_required")
        doi = evidence.get("doi")
        precedent = bool(
            evidence.get("literature_precedent")
            or evidence.get("source_db") in {"v3_cascade", "rhea", "brenda", "retrobiocat", "enzymemap"}
            or evidence.get("rhea_ids")
            or doi
        )
        sequence = bool(evidence.get("sequence") or evidence.get("sequence_length"))
        substrate_similarity = evidence.get("substrate_similarity")
        condition_match = evidence.get("condition_match")
        is_supported = bool(s.ec and (has_candidate_source or has_score or uniprot or precedent))
        ec_parts = [p for p in (s.ec or "").split(".") if p and p != "x"]
        dimensions = {
            "ec": bool(s.ec),
            "ec_depth": len(ec_parts),
            "candidate_provenance": bool(has_candidate_source),
            "enzyme_uid": bool(uniprot),
            "uniprot": bool(uniprot),
            "organism": bool(organism),
            "sequence": sequence,
            "cofactor": bool(cofactor) or _contains_any(_slot_text(s), COFACTOR_OX | COFACTOR_RED),
            "substrate_similarity": substrate_similarity is not None,
            "literature_precedent": precedent,
            "condition_match": condition_match is not None,
            "condition": bool(_safe_float(s.T) is not None and _safe_float(s.pH) is not None),
        }
        score = (
            0.18 * float(dimensions["ec"])
            + 0.10 * min(dimensions["ec_depth"], 4) / 4.0
            + 0.12 * float(dimensions["candidate_provenance"])
            + 0.15 * float(dimensions["uniprot"])
            + 0.10 * float(dimensions["organism"])
            + 0.10 * float(dimensions["sequence"])
            + 0.10 * float(dimensions["cofactor"])
            + 0.10 * float(dimensions["substrate_similarity"])
            + 0.10 * float(dimensions["literature_precedent"])
            + 0.05 * float(dimensions["condition"] or dimensions["condition_match"])
        )
        score = min(1.0, score)
        scores.append(score)
        supported += int(is_supported)
        rows.append({
            "step": s.index,
            "ec": s.ec,
            "enzyme_uid": s.enzyme_uid,
            "source": source,
            "e_enzyme": _safe_float(s.e_enzyme),
            "supported": is_supported,
            "evidence_dimensions": dimensions,
            "evidence": _jsonable({
                k: evidence.get(k)
                for k in (
                    "uniprot_accession",
                    "uniprot_status",
                    "organism",
                    "tax_id",
                    "sequence",
                    "sequence_length",
                    "protein_existence",
                    "reviewed",
                    "rhea_ids",
                    "cofactor",
                    "cofactor_regeneration_mode",
                    "doi",
                    "pmid",
                    "literature_title",
                    "substrate_similarity",
                    "condition_match",
                    "literature_precedent",
                    "source_db",
                )
                if evidence.get(k) not in (None, "", [], {})
            }),
            "enzyme_evidence_score": round(score, 3),
        })

    return {
        "n_enzymatic_steps": len(enzymatic),
        "enzyme_evidence_coverage": supported / len(enzymatic),
        "enzyme_evidence_score": round(sum(scores) / len(scores), 3) if scores else None,
        "supported_steps": supported,
        "steps": rows,
    }


def _contains_any(text: str, tokens: set[str]) -> bool:
    low = text.lower()
    return any(tok in low for tok in tokens)


def _slot_text(slot: Slot) -> str:
    return " ".join(str(x or "") for x in [
        slot.reaction_smiles,
        slot.catalyst,
        slot.solvent,
        slot.enzyme_uid,
        slot.ec,
    ])


def _reaction_product_set(rxn_smiles: str | None) -> set[str]:
    if not rxn_smiles or ">>" not in rxn_smiles:
        return set()
    return set(canonical_side(rxn_smiles.split(">>", 1)[1]))


def _heavy_atoms(smiles: str | None) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _slot_reactant_heavy_atoms(slot: Slot) -> int:
    total = _heavy_atoms(slot.main_reactant)
    for smi in slot.aux_reactants or []:
        total += _heavy_atoms(smi)
    return total


def _leaf_reactants(board: CascadeBoard) -> list[str]:
    leaves: list[str] = []
    expanded_products = {
        canonical_smiles(slot.product)
        for slot in board.slots[1:]
        if slot.product
    }
    for slot in board.slots:
        reactants = []
        if slot.main_reactant:
            reactants.append(slot.main_reactant)
        reactants.extend(slot.aux_reactants or [])
        for smi in reactants:
            can = canonical_smiles(smi)
            if can and can in expanded_products:
                continue
            leaves.append(smi)

    seen: set[str] = set()
    out: list[str] = []
    for smi in leaves:
        can = canonical_smiles(smi)
        key = can or smi
        if smi and key not in seen:
            seen.add(key)
            out.append(smi)
    return out


def _stock_overrides(board: CascadeBoard) -> dict[str, bool]:
    raw = (board.global_constraints or {}).get("stock_overrides") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, bool] = {}
    for smi, value in raw.items():
        if not smi:
            continue
        key = canonical_smiles(str(smi)) or str(smi)
        out[key] = bool(value)
    return out


def _stock_status_for_smiles(smiles: str, *, stock_checker: StockChecker | None, overrides: dict[str, bool]) -> bool | None:
    key = canonical_smiles(smiles) or smiles
    if key in overrides:
        return bool(overrides[key])
    if stock_checker is None:
        return None
    return bool(stock_checker(smiles))


def retrosynthesis_progress_metrics(
    board: CascadeBoard,
    stock_checker: StockChecker | None = None,
) -> dict[str, Any]:
    """Measure whether a filled route actually disconnects the target.

    ``filled_route`` only means every slot has a candidate. This diagnostic
    asks a stricter retrosynthesis question: did the main chain become
    substantially simpler, and did the leaves reach stock or small fragments?
    """
    target = board.slots[0].product if board.slots else None
    terminal_main = board.slots[-1].main_reactant if board.slots else None
    target_atoms = _heavy_atoms(target)
    terminal_atoms = _heavy_atoms(terminal_main)
    leaf_reactants = _leaf_reactants(board)
    leaf_atoms = [_heavy_atoms(smi) for smi in leaf_reactants]
    largest_leaf_atoms = max(leaf_atoms) if leaf_atoms else None
    simplified_leaf_threshold = max(10, int(target_atoms * 0.35)) if target_atoms else 8

    step_rows = []
    progressive_steps = 0
    for slot in board.slots:
        product_atoms = _heavy_atoms(slot.product)
        main_atoms = _heavy_atoms(slot.main_reactant)
        delta = product_atoms - main_atoms if product_atoms and main_atoms else None
        frac = (delta / product_atoms) if delta is not None and product_atoms else None
        is_progressive = bool(
            delta is not None
            and delta >= max(1, int(product_atoms * 0.08))
        )
        progressive_steps += int(is_progressive)
        step_rows.append({
            "step": slot.index,
            "product_heavy_atoms": product_atoms,
            "main_reactant_heavy_atoms": main_atoms,
            "delta_heavy_atoms": delta,
            "reduction_fraction": round(frac, 3) if frac is not None else None,
            "progressive": is_progressive,
        })

    main_chain_reduction = None
    if target_atoms and terminal_atoms:
        main_chain_reduction = max(0.0, (target_atoms - terminal_atoms) / target_atoms)
    progressive_step_fraction = progressive_steps / max(board.n_steps, 1)
    terminal_simplified = bool(
        terminal_atoms
        and (
            terminal_atoms <= 8
            or (target_atoms and terminal_atoms <= simplified_leaf_threshold)
        )
    )
    leaf_simplified = bool(
        largest_leaf_atoms
        and (
            largest_leaf_atoms <= 8
            or (target_atoms and largest_leaf_atoms <= simplified_leaf_threshold)
        )
    )
    largest_leaf_reduction = None
    if target_atoms and largest_leaf_atoms:
        largest_leaf_reduction = max(0.0, (target_atoms - largest_leaf_atoms) / target_atoms)

    terminal_stock_solve = None
    stock_status = None
    overrides = _stock_overrides(board)
    if (stock_checker is not None or overrides) and leaf_reactants:
        stock_status = {
            smi: _stock_status_for_smiles(smi, stock_checker=stock_checker, overrides=overrides)
            for smi in leaf_reactants
            if smi
        }
        terminal_stock_solve = all(value is True for value in stock_status.values()) if stock_status else None

    strong_terminal_simplification = bool(
        main_chain_reduction is not None
        and main_chain_reduction >= 0.65
        and terminal_simplified
        and leaf_simplified
        and progressive_steps >= 1
    )
    progress_success = bool(
        main_chain_reduction is not None
        and main_chain_reduction >= 0.35
        and leaf_simplified
        and (progressive_step_fraction >= 0.5 or strong_terminal_simplification)
    )
    return {
        "target_heavy_atoms": target_atoms,
        "terminal_main_heavy_atoms": terminal_atoms,
        "largest_leaf_heavy_atoms": largest_leaf_atoms,
        "main_chain_reduction": round(main_chain_reduction, 3) if main_chain_reduction is not None else None,
        "largest_leaf_reduction": round(largest_leaf_reduction, 3) if largest_leaf_reduction is not None else None,
        "progressive_steps": progressive_steps,
        "progressive_step_fraction": round(progressive_step_fraction, 3),
        "retrosynthesis_progress_success": progress_success,
        "strong_terminal_simplification": strong_terminal_simplification,
        "terminal_simplified": terminal_simplified,
        "leaf_simplified": leaf_simplified,
        "simplified_leaf_threshold": simplified_leaf_threshold,
        "terminal_stock_solve": terminal_stock_solve,
        "leaf_stock_status": stock_status,
        "step_progress": step_rows,
    }


def route_naturalness_metrics(board: CascadeBoard) -> dict[str, Any]:
    """Detect structural route artifacts that look unnatural in a linear route.

    These checks do not prove synthetic feasibility. They catch common planner
    artifacts: self-loops, revisiting a previous main-chain node, repeated main
    reactants, and reaction SMILES whose product side does not contain the slot
    product.
    """
    seen_main: set[str] = set()
    seen_products: set[str] = set()
    self_loop_steps = 0
    repeated_main_reactants = 0
    revisited_main_chain_nodes = 0
    product_mismatch_steps = 0
    atom_balance_violations = 0
    unfilled_steps = 0
    issues_by_step: list[dict[str, Any]] = []

    for slot in board.slots:
        product = canonical_smiles(slot.product)
        main = canonical_smiles(slot.main_reactant)
        step_issues: list[str] = []
        if not slot.reaction_smiles or not main:
            unfilled_steps += 1
            step_issues.append("unfilled")
        if product and main and product == main:
            self_loop_steps += 1
            step_issues.append("self_loop")
        if main and main in seen_main:
            repeated_main_reactants += 1
            step_issues.append("repeated_main_reactant")
        if main and main in seen_products:
            revisited_main_chain_nodes += 1
            step_issues.append("revisited_main_chain_node")
        rxn_products = _reaction_product_set(slot.reaction_smiles)
        if product and rxn_products and product not in rxn_products:
            product_mismatch_steps += 1
            step_issues.append("product_mismatch")
        product_atoms = _heavy_atoms(slot.product)
        reactant_atoms = _slot_reactant_heavy_atoms(slot)
        if product_atoms > 10 and reactant_atoms and reactant_atoms < max(4, int(product_atoms * 0.45)):
            atom_balance_violations += 1
            step_issues.append("atom_balance_violation")
        if step_issues:
            issues_by_step.append({"step": slot.index, "issues": step_issues})
        if main:
            seen_main.add(main)
        if product:
            seen_products.add(product)

    n = max(board.n_steps, 1)
    weighted_bad = (
        1.0 * unfilled_steps
        + 1.0 * self_loop_steps
        + 0.8 * repeated_main_reactants
        + 0.8 * revisited_main_chain_nodes
        + 1.0 * product_mismatch_steps
        + 1.0 * atom_balance_violations
    )
    naturalness_score = max(0.0, 1.0 - weighted_bad / n)
    return {
        "naturalness_score": round(naturalness_score, 3),
        "self_loop_steps": self_loop_steps,
        "repeated_main_reactants": repeated_main_reactants,
        "revisited_main_chain_nodes": revisited_main_chain_nodes,
        "product_mismatch_steps": product_mismatch_steps,
        "atom_balance_violations": atom_balance_violations,
        "unfilled_steps": unfilled_steps,
        "issues_by_step": issues_by_step,
    }


def compatibility_dimension_checks(board: CascadeBoard) -> dict[str, Any]:
    has_enzyme = any(s.ec for s in board.slots)
    all_text = " ".join(_slot_text(s) for s in board.slots).lower()
    solvents = sorted({(s.solvent or "").lower() for s in board.slots if s.solvent})
    enzymatic_solvents = [
        (s.solvent or "").lower()
        for s in board.slots
        if s.ec and s.solvent
    ]
    enzymatic_organic_risk = any(
        any(tok in solvent for tok in ORGANIC_SOLVENT_RISK)
        for solvent in enzymatic_solvents
    )
    metal_tokens = sorted(tok for tok in METAL_TOKENS if tok in all_text)
    oxidant_tokens = sorted(tok for tok in OXIDANT_TOKENS if tok in all_text)
    reductant_tokens = sorted(tok for tok in REDUCTANT_TOKENS if tok in all_text)
    cofactor_ox_tokens = sorted(tok for tok in COFACTOR_OX if tok in all_text)
    cofactor_red_tokens = sorted(tok for tok in COFACTOR_RED if tok in all_text)
    oxygen_conflict = "anaerobic" in all_text and (
        "oxygen" in all_text or " o2 " in f" {all_text} "
    )
    water_sensitivity = "water sensitive" in all_text or "moisture sensitive" in all_text

    return {
        "solvent": {
            "has_solvent_fields": bool(solvents),
            "solvents": solvents,
            "enzymatic_organic_solvent_risk": bool(has_enzyme and enzymatic_organic_risk),
        },
        "metal_enzyme": {
            "has_enzyme": has_enzyme,
            "metal_tokens": metal_tokens,
            "metal_enzyme_conflict": bool(has_enzyme and metal_tokens),
        },
        "oxidant_reductant": {
            "oxidant_tokens": oxidant_tokens,
            "reductant_tokens": reductant_tokens,
            "oxidant_reductant_conflict": bool(oxidant_tokens and reductant_tokens),
        },
        "cofactor": {
            "oxidized_cofactors": cofactor_ox_tokens,
            "reduced_cofactors": cofactor_red_tokens,
            "cofactor_cross_talk": bool(cofactor_ox_tokens and cofactor_red_tokens),
        },
        "oxygen": {
            "mentions_anaerobic": "anaerobic" in all_text,
            "mentions_oxygen": "oxygen" in all_text or " o2 " in f" {all_text} ",
            "oxygen_requirement_conflict": oxygen_conflict,
        },
        "water": {
            "water_sensitivity": water_sensitivity,
        },
    }


def cascade_compatibility_metrics(board: CascadeBoard) -> dict[str, Any]:
    """Summarize route-level cascade feasibility from deterministic fields.

    This is intentionally conservative. It is a factual diagnostic, not a
    learned claim of experimental success.
    """
    cond = condition_window_metrics(board)
    enz = enzyme_evidence_metrics(board)
    dimensions = compatibility_dimension_checks(board)
    natural = route_naturalness_metrics(board)

    issues = []
    if not cond["has_all_T"]:
        issues.append("missing_temperature")
    if not cond["has_all_pH"]:
        issues.append("missing_pH")
    if cond["max_adjacent_delta_T"] is not None and cond["max_adjacent_delta_T"] > cond["max_delta_T_threshold"]:
        issues.append("temperature_window_mismatch")
    if cond["max_adjacent_delta_pH"] is not None and cond["max_adjacent_delta_pH"] > cond["max_delta_pH_threshold"]:
        issues.append("pH_window_mismatch")

    enzyme_cov = enz.get("enzyme_evidence_coverage")
    if enzyme_cov is not None and enzyme_cov < 1.0:
        issues.append("incomplete_enzyme_evidence")

    filled_steps = sum(1 for s in board.slots if s.reaction_smiles and s.main_reactant)
    if filled_steps < board.n_steps:
        issues.append("unfilled_reaction_step")
    if natural["self_loop_steps"] or natural["revisited_main_chain_nodes"]:
        issues.append("route_cycle")
    if natural["product_mismatch_steps"]:
        issues.append("reaction_product_mismatch")
    if natural["atom_balance_violations"]:
        issues.append("atom_balance_violation")

    if dimensions["solvent"]["enzymatic_organic_solvent_risk"]:
        issues.append("solvent_incompatibility")
    if dimensions["metal_enzyme"]["metal_enzyme_conflict"]:
        issues.append("metal_enzyme_conflict")
    if dimensions["oxidant_reductant"]["oxidant_reductant_conflict"]:
        issues.append("oxidant_reductant_conflict")
    if dimensions["cofactor"]["cofactor_cross_talk"]:
        issues.append("cofactor_cross_talk")
    if dimensions["oxygen"]["oxygen_requirement_conflict"]:
        issues.append("oxygen_requirement_conflict")
    if dimensions["water"]["water_sensitivity"]:
        issues.append("water_sensitivity")

    # Heuristic operation mode suggestion from condition spread.
    if any(x in issues for x in {
        "temperature_window_mismatch",
        "pH_window_mismatch",
        "metal_enzyme_conflict",
        "oxidant_reductant_conflict",
        "oxygen_requirement_conflict",
        "route_cycle",
        "reaction_product_mismatch",
        "atom_balance_violation",
    }):
        suggested_mode = "sequential_or_telescoped"
    elif "solvent_incompatibility" in issues:
        suggested_mode = "sequential_or_compartmentalized"
    elif board.n_steps <= 1:
        suggested_mode = "single_step"
    else:
        suggested_mode = "one_pot_candidate"

    return {
        "cascade_compatibility_success": len(issues) == 0,
        "issues": issues,
        "issue_counts": dict(Counter(issues)),
        "dimension_checks": dimensions,
        "route_naturalness": natural,
        "suggested_operation_mode": suggested_mode,
    }


def operation_transition_metrics(board: CascadeBoard) -> dict[str, Any]:
    """Estimate operational burden across consecutive chemistry/enzyme steps."""
    classes = [slot_operation_class(slot) for slot in board.slots]
    transitions = []
    chemo_bio = 0
    temp_shifts = 0
    ph_shifts = 0
    solvent_switches = 0
    issues = []
    for idx, (left, right) in enumerate(zip(board.slots, board.slots[1:]), start=1):
        left_class = classes[idx - 1]
        right_class = classes[idx]
        class_switch = left_class != right_class and "unknown" not in {left_class, right_class}
        if class_switch:
            chemo_bio += 1
        t_delta = None
        if left.T is not None and right.T is not None:
            t_delta = abs(float(left.T) - float(right.T))
            if t_delta > 15.0:
                temp_shifts += 1
        ph_delta = None
        if left.pH is not None and right.pH is not None:
            ph_delta = abs(float(left.pH) - float(right.pH))
            if ph_delta > 2.0:
                ph_shifts += 1
        left_solvent = (left.solvent or "").strip().lower()
        right_solvent = (right.solvent or "").strip().lower()
        solvent_switch = bool(left_solvent and right_solvent and left_solvent != right_solvent)
        if solvent_switch:
            solvent_switches += 1
        transition_issues = []
        if class_switch:
            transition_issues.append("chemo_bio_transition")
        if t_delta is not None and t_delta > 15.0:
            transition_issues.append("temperature_shift")
        if ph_delta is not None and ph_delta > 2.0:
            transition_issues.append("pH_shift")
        if solvent_switch:
            transition_issues.append("solvent_switch")
        transitions.append({
            "from_step": idx - 1,
            "to_step": idx,
            "from_class": left_class,
            "to_class": right_class,
            "temperature_delta": round(t_delta, 3) if t_delta is not None else None,
            "pH_delta": round(ph_delta, 3) if ph_delta is not None else None,
            "solvent_switch": solvent_switch,
            "issues": transition_issues,
        })
        issues.extend(transition_issues)

    denom = max(board.n_steps - 1, 1)
    raw_cost = chemo_bio + 0.5 * (temp_shifts + ph_shifts + solvent_switches)
    operation_score = max(0.0, 1.0 - raw_cost / denom)
    return {
        "step_classes": classes,
        "transitions": transitions,
        "chemo_bio_transitions": chemo_bio,
        "temperature_shifts": temp_shifts,
        "pH_shifts": ph_shifts,
        "solvent_switches": solvent_switches,
        "operation_cost": round(raw_cost, 3),
        "operation_score": round(operation_score, 3),
        "issues": sorted(set(issues)),
    }


def slot_operation_class(slot: Slot) -> str:
    source = (slot.source or "").lower()
    if slot.ec or source in {"enzyformer", "enzexpand", "v3_retrieval"}:
        return "enzymatic"
    if source in {"retrochimera", "chemical", "template", "retrosim"}:
        return "chemical"
    if slot.catalyst and any(token in (slot.catalyst or "").lower() for token in METAL_TOKENS):
        return "chemical"
    return "unknown"


def route_candidate_pool_metrics(board: CascadeBoard) -> dict[str, Any]:
    """Aggregate per-step candidate-pool quality across a route."""
    summaries = [candidate_pool_summary(slot, limit=0) for slot in board.slots]
    non_empty = [row for row in summaries if row.get("n_candidates", 0) > 0]
    source_counts = Counter()
    for row in summaries:
        source_counts.update(row.get("source_counts") or {})

    def mean_field(key: str) -> float | None:
        values = [row.get(key) for row in non_empty]
        values = [float(x) for x in values if isinstance(x, (int, float))]
        return round(sum(values) / len(values), 4) if values else None

    diversity_values = [
        float(row["pool_diversity_score"])
        for row in non_empty
        if isinstance(row.get("pool_diversity_score"), (int, float))
    ]
    total_candidates = sum(int(row.get("n_candidates") or 0) for row in summaries)
    return {
        "steps_with_candidates": len(non_empty),
        "candidate_pool_coverage": round(len(non_empty) / max(board.n_steps, 1), 4),
        "total_candidates": total_candidates,
        "avg_candidates_per_step": round(total_candidates / max(board.n_steps, 1), 4),
        "candidate_pool_source_counts": dict(source_counts),
        "avg_pool_diversity_score": mean_field("pool_diversity_score"),
        "min_pool_diversity_score": round(min(diversity_values), 4) if diversity_values else None,
        "avg_duplicate_reaction_fraction": mean_field("duplicate_reaction_fraction"),
        "avg_duplicate_main_reactant_fraction": mean_field("duplicate_main_reactant_fraction"),
        "avg_duplicate_reactant_set_fraction": mean_field("duplicate_reactant_set_fraction"),
        "single_reactant_set_steps": sum(
            int(((row.get("diversity_flags") or {}).get("single_reactant_set_pool")) is True)
            for row in non_empty
        ),
    }


def route_metrics(board: CascadeBoard, stock_checker: StockChecker | None = None) -> dict[str, Any]:
    """Compute deterministic route-level metrics from a board."""
    steps = board.slots
    filled_steps = sum(1 for s in steps if s.reaction_smiles and s.main_reactant)
    source_counts = Counter(s.source or "unknown" for s in steps)
    terminal_reactants = _leaf_reactants(board)

    strict_stock_solve = None
    stock_overrides = _stock_overrides(board)
    stock_override_hits = 0
    if (stock_checker is not None or stock_overrides) and terminal_reactants:
        values = []
        for smi in terminal_reactants:
            if not smi:
                continue
            key = canonical_smiles(smi) or smi
            stock_override_hits += int(key in stock_overrides)
            values.append(_stock_status_for_smiles(smi, stock_checker=stock_checker, overrides=stock_overrides))
        strict_stock_solve = all(value is True for value in values) if values else None

    cond = condition_window_metrics(board)
    enz = enzyme_evidence_metrics(board)
    natural = route_naturalness_metrics(board)
    progress = retrosynthesis_progress_metrics(board, stock_checker=stock_checker)
    compat = cascade_compatibility_metrics(board)
    operation = operation_transition_metrics(board)
    candidate_pool = route_candidate_pool_metrics(board)
    route_solved = bool(
        filled_steps == board.n_steps
        and natural.get("naturalness_score") == 1.0
        and (
            strict_stock_solve is True
            or (
                strict_stock_solve is None
                and progress.get("retrosynthesis_progress_success")
                and progress.get("terminal_simplified")
            )
        )
    )

    return {
        "n_steps": board.n_steps,
        "filled_steps": filled_steps,
        "filled_route": filled_steps == board.n_steps,
        "progressive_route": bool(
            filled_steps == board.n_steps
            and natural.get("naturalness_score") == 1.0
            and progress.get("retrosynthesis_progress_success")
        ),
        "route_solved": route_solved,
        "candidate_source_counts": dict(source_counts),
        "terminal_reactants": terminal_reactants,
        "strict_stock_solve": strict_stock_solve,
        "stock_override_count": stock_override_hits,
        "condition": cond,
        "enzyme_evidence": enz,
        "route_naturalness": natural,
        "retrosynthesis_progress": progress,
        "cascade_compatibility": compat,
        "operation_transitions": operation,
        "candidate_pool": candidate_pool,
    }


def explanation_to_dict(exp: RouteExplanation | None) -> dict[str, Any]:
    if exp is None:
        return {}
    return {
        "why_selected": exp.why_selected,
        "what_was_changed": list(exp.what_was_changed),
        "constraints_satisfied": _jsonable(exp.constraints_satisfied),
        "constraints_at_risk": _jsonable(exp.constraints_at_risk),
        "global_condition_window": exp.global_condition_window,
        "evidence_table": _jsonable(exp.evidence_table),
        "edited_slots": list(exp.edited_slots),
        "alternative_edits": list(exp.alternative_edits),
        "minimal_relaxation": exp.minimal_relaxation,
        "uncertainty_table": _jsonable(exp.uncertainty_table),
    }


def route_result_to_dict(
    result: RouteResult,
    stock_checker: StockChecker | None = None,
) -> dict[str, Any]:
    """Export a RouteResult as the canonical JSON route object."""
    board = result.board
    return {
        "score": _safe_float(result.score),
        "confidence": _safe_float(result.confidence),
        "n_steps": board.n_steps,
        "quality_vector": _jsonable(result.quality_vector),
        "risk_vector": _jsonable(result.risk_vector),
        "constraint_report": _jsonable(result.constraint_report),
        "bottleneck_slot": result.bottleneck_slot,
        "bottleneck_reason": result.bottleneck_reason,
        "global_constraints": _jsonable(board.global_constraints),
        "steps": [slot_to_dict(s, stock_checker=stock_checker) for s in board.slots],
        "metrics": route_metrics(board, stock_checker=stock_checker),
        "explanation": explanation_to_dict(result.explanation),
    }


def route_set_diversity_metrics(routes: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure whether the returned route set contains genuinely different plans."""
    n_routes = len(routes)
    type_sequences = [tuple(step.get("reaction_type") or "" for step in route.get("steps") or []) for route in routes]
    source_sequences = [tuple(step.get("source") or "" for step in route.get("steps") or []) for route in routes]
    ec_sequences = [tuple(_ec1(step.get("ec")) for step in route.get("steps") or []) for route in routes]
    terminal_sets = [terminal_reactant_set(route) for route in routes]
    full_signatures = [
        (
            type_sequences[idx],
            source_sequences[idx],
            ec_sequences[idx],
            tuple(sorted(terminal_sets[idx])),
        )
        for idx in range(n_routes)
    ]
    return {
        "n_routes": n_routes,
        "unique_type_sequences": len(set(type_sequences)),
        "unique_source_sequences": len(set(source_sequences)),
        "unique_ec1_sequences": len(set(ec_sequences)),
        "unique_terminal_reactant_sets": len({tuple(sorted(values)) for values in terminal_sets}),
        "unique_full_signatures": len(set(full_signatures)),
        "duplicate_route_fraction": round(1.0 - (len(set(full_signatures)) / max(n_routes, 1)), 4) if n_routes else None,
        "mean_pairwise_type_distance": round(mean_pairwise_sequence_distance(type_sequences), 4) if n_routes > 1 else None,
        "mean_pairwise_terminal_jaccard_distance": round(mean_pairwise_jaccard_distance(terminal_sets), 4) if n_routes > 1 else None,
    }


def diversify_ranked_route_results(results: list[RouteResult]) -> list[RouteResult]:
    """Keep base quality order, but push exact route-signature duplicates later."""
    seen = set()
    unique = []
    duplicates = []
    for result in results:
        try:
            sig = route_result_diversity_signature(result)
        except Exception:
            sig = ("opaque_result", id(result))
        if sig in seen:
            duplicates.append(result)
        else:
            seen.add(sig)
            unique.append(result)
    return unique + duplicates


def route_result_diversity_signature(result: RouteResult) -> tuple:
    board = result.board
    types = tuple(slot.reaction_type or "" for slot in board.slots)
    sources = tuple(slot.source or "" for slot in board.slots)
    ec1s = tuple(_ec1(slot.ec) for slot in board.slots)
    terminals = []
    for slot in board.slots:
        if slot.main_reactant:
            terminals.append(canonical_smiles(slot.main_reactant))
        for smi in slot.aux_reactants or []:
            can = canonical_smiles(smi)
            if can:
                terminals.append(can)
    return (types, sources, ec1s, tuple(sorted(set(terminals))))


def terminal_reactant_set(route: dict[str, Any]) -> set[str]:
    metrics = route.get("metrics") or {}
    terminals = metrics.get("terminal_reactants") or []
    values = {canonical_smiles(smi) for smi in terminals if canonical_smiles(smi)}
    if values:
        return values
    for step in route.get("steps") or []:
        if step.get("main_reactant"):
            values.add(canonical_smiles(step["main_reactant"]))
        for smi in step.get("aux_reactants") or []:
            can = canonical_smiles(smi)
            if can:
                values.add(can)
    return values


def mean_pairwise_sequence_distance(sequences: list[tuple[str, ...]]) -> float:
    values = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            denom = max(len(sequences[i]), len(sequences[j]), 1)
            values.append(_sequence_edit_distance(sequences[i], sequences[j]) / denom)
    return sum(values) / max(len(values), 1)


def mean_pairwise_jaccard_distance(sets: list[set[str]]) -> float:
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i].union(sets[j])
            inter = sets[i].intersection(sets[j])
            values.append(1.0 - (len(inter) / max(len(union), 1)))
    return sum(values) / max(len(values), 1)


def _sequence_edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, 1):
        cur = [i]
        for j, bv in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if av == bv else 1),
            ))
        prev = cur
    return prev[-1]


def _ec1(ec: Any) -> str:
    return str(ec or "").split(".", 1)[0] if ec else ""


def route_results_payload(
    target: str,
    results: list[RouteResult],
    *,
    objective: str = "balanced",
    constraints: dict[str, Any] | None = None,
    elapsed_s: float | None = None,
    stock_checker: StockChecker | None = None,
) -> dict[str, Any]:
    """Export a complete planner response payload."""
    routes = [
        route_result_to_dict(r, stock_checker=stock_checker)
        for r in results
    ]
    return {
        "target": target,
        "objective": objective,
        "constraints": constraints,
        "n_results": len(results),
        "time_s": round(elapsed_s, 3) if elapsed_s is not None else None,
        "routes": routes,
        "route_set_metrics": {
            "diversity": route_set_diversity_metrics(routes),
        },
    }
