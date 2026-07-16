"""Execution-domain-neutral structural scope matching for route windows."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")
_MOTIF_SMARTS = {
    "alkene": "[CX3]=[CX3]",
    "carbonyl": "[CX3]=[OX1]",
    "carboxyl": "[CX3](=O)[OX1H0-,OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "hydroxyl": "[OX2H]",
    "silyl_ether": "[OX2][Si]",
}


def match_structure_capability(
    capability: Mapping[str, Any],
    precursor_smiles: str,
    product_smiles: str,
    *,
    window_steps: int,
) -> dict[str, Any]:
    transition = structure_transition(precursor_smiles, product_smiles)
    raw_match = capability.get("match")
    match = normalize_structure_match(raw_match)
    reasons: list[str] = []
    if not structure_match_input_valid(raw_match):
        reasons.append("innovation_structure_match_input_invalid")
    if not transition.get("valid"):
        reasons.append("innovation_boundary_structure_invalid")
    if not (
        int(match.get("min_window_steps") or 1)
        <= window_steps
        <= int(match.get("max_window_steps") or 8)
    ):
        reasons.append("innovation_window_length_out_of_capability_scope")
    observed_motifs = dict(transition.get("motif_delta") or {})
    expected_motifs = dict(match.get("net_motif_delta") or {})
    if any(observed_motifs.get(key, 0) != value for key, value in expected_motifs.items()):
        reasons.append("innovation_net_motif_delta_mismatch")
    if any(observed_motifs.get(key, 0) != 0 for key in match.get("preserved_motifs") or []):
        reasons.append("innovation_preserved_motif_changed")
    if match.get("reject_unlisted_motif_changes") and any(
        value != 0 and key not in expected_motifs
        for key, value in observed_motifs.items()
    ):
        reasons.append("innovation_unlisted_motif_change")
    if any(
        dict(transition.get("element_delta") or {}).get(key, 0) != value
        for key, value in dict(match.get("element_delta") or {}).items()
    ):
        reasons.append("innovation_element_delta_mismatch")
    if float(transition.get("scaffold_similarity") or 0.0) < float(
        match.get("min_scaffold_similarity") or 0.0
    ):
        reasons.append("innovation_scaffold_similarity_below_scope")
    if abs(int(transition.get("heavy_atom_delta") or 0)) > int(
        match.get("max_abs_heavy_atom_delta") or 0
    ):
        reasons.append("innovation_heavy_atom_delta_out_of_scope")
    if int(transition.get("substrate_carbon_count") or 0) < int(
        match.get("min_substrate_carbons") or 0
    ):
        reasons.append("innovation_substrate_too_small_for_scope")
    if int(transition.get("substrate_ring_count") or 0) < int(
        match.get("min_substrate_rings") or 0
    ):
        reasons.append("innovation_substrate_ring_scope_mismatch")
    if not _smarts_any(precursor_smiles, match.get("substrate_smarts_any") or []):
        reasons.append("innovation_substrate_smarts_mismatch")
    if not _smarts_any(product_smiles, match.get("product_smarts_any") or []):
        reasons.append("innovation_product_smarts_mismatch")
    specificity = min(
        0.12,
        0.02
        * (
            len(match.get("substrate_smarts_any") or [])
            + len(match.get("product_smarts_any") or [])
        )
        + 0.01 * len(match.get("preserved_motifs") or []),
    )
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "transition": transition,
        "match_score": round(
            min(
                1.0,
                0.6 * float(transition.get("scaffold_similarity") or 0.0)
                + 0.4 * min(1.0, max(0.0, (window_steps - 1) / 5.0))
                + specificity,
            ),
            6,
        ),
        "capability_specificity_bonus": round(specificity, 6),
    }


def normalize_structure_match(value: Any) -> dict[str, Any]:
    """Normalize structural scope shared by all execution domains."""

    match = dict(value or {}) if isinstance(value, Mapping) else {}
    motif_delta, _ = _integer_mapping(
        match.get("net_motif_delta"), allowed_keys=set(_MOTIF_SMARTS)
    )
    element_delta, _ = _integer_mapping(match.get("element_delta"))
    return {
        "net_motif_delta": {key: value for key, value in motif_delta.items() if value},
        "preserved_motifs": _strings(match.get("preserved_motifs") or []),
        "substrate_smarts_any": _strings(match.get("substrate_smarts_any") or []),
        "product_smarts_any": _strings(match.get("product_smarts_any") or []),
        "element_delta": element_delta,
        "min_scaffold_similarity": _float(
            match.get("min_scaffold_similarity"), 0.45
        ),
        "max_abs_heavy_atom_delta": _integer(
            match.get("max_abs_heavy_atom_delta"), 4
        ),
        "min_substrate_carbons": _integer(match.get("min_substrate_carbons"), 0),
        "min_substrate_rings": _integer(match.get("min_substrate_rings"), 0),
        "min_window_steps": max(1, _integer(match.get("min_window_steps"), 1)),
        "max_window_steps": max(1, _integer(match.get("max_window_steps"), 8)),
        "reject_unlisted_motif_changes": bool(
            match.get("reject_unlisted_motif_changes", False)
        ),
    }


def structure_match_input_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    _, motif_valid = _integer_mapping(
        value.get("net_motif_delta"), allowed_keys=set(_MOTIF_SMARTS)
    )
    _, element_valid = _integer_mapping(value.get("element_delta"))
    sequence_fields_valid = all(
        value.get(key) is None
        or isinstance(value.get(key), (str, list, tuple, set))
        for key in ("preserved_motifs", "substrate_smarts_any", "product_smarts_any")
    )
    rejection_flag_valid = value.get(
        "reject_unlisted_motif_changes"
    ) is None or isinstance(value.get("reject_unlisted_motif_changes"), bool)
    return motif_valid and element_valid and sequence_fields_valid and rejection_flag_valid


def structure_transition(precursor_smiles: str, product_smiles: str) -> dict[str, Any]:
    precursor = Chem.MolFromSmiles(str(precursor_smiles or ""))
    product = Chem.MolFromSmiles(str(product_smiles or ""))
    if precursor is None or product is None:
        return {"valid": False, "motif_delta": {}, "element_delta": {}}
    precursor_motifs = _motif_counts(precursor)
    product_motifs = _motif_counts(product)
    precursor_elements = _element_counts(precursor)
    product_elements = _element_counts(product)
    return {
        "valid": True,
        "precursor_smiles": Chem.MolToSmiles(precursor, isomericSmiles=True),
        "product_smiles": Chem.MolToSmiles(product, isomericSmiles=True),
        "motif_delta": {
            key: product_motifs[key] - precursor_motifs[key] for key in _MOTIF_SMARTS
        },
        "element_delta": {
            key: product_elements.get(key, 0) - precursor_elements.get(key, 0)
            for key in sorted(set(precursor_elements) | set(product_elements))
        },
        "heavy_atom_delta": product.GetNumHeavyAtoms() - precursor.GetNumHeavyAtoms(),
        "substrate_carbon_count": precursor_elements.get("C", 0),
        "substrate_ring_count": precursor.GetRingInfo().NumRings(),
        "scaffold_similarity": round(_similarity(precursor, product), 6),
    }


def _motif_counts(molecule: Any) -> dict[str, int]:
    return {
        key: len(molecule.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
        for key, smarts in _MOTIF_SMARTS.items()
    }


def _element_counts(molecule: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in molecule.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    return counts


def _similarity(left: Any, right: Any) -> float:
    left_fp = AllChem.GetMorganFingerprintAsBitVect(left, 2, nBits=2048)
    right_fp = AllChem.GetMorganFingerprintAsBitVect(right, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _smarts_any(smiles: str, patterns: Iterable[str]) -> bool:
    values = list(patterns)
    if not values:
        return True
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    return bool(
        molecule is not None
        and any(
            query is not None and molecule.HasSubstructMatch(query)
            for query in (Chem.MolFromSmarts(value) for value in values)
        )
    )


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(item).strip() for item in value or [] if str(item).strip()})


def _integer_mapping(
    value: Any, *, allowed_keys: set[str] | None = None
) -> tuple[dict[str, int], bool]:
    if value is None:
        return {}, True
    if not isinstance(value, Mapping):
        return {}, False
    normalized: dict[str, int] = {}
    valid = True
    for key, amount in value.items():
        name = str(key).strip()
        if not name or (allowed_keys is not None and name not in allowed_keys):
            valid = False
            continue
        try:
            if isinstance(amount, bool):
                raise ValueError
            normalized[name] = int(amount)
        except (TypeError, ValueError):
            valid = False
    return normalized, valid


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "match_structure_capability",
    "normalize_structure_match",
    "structure_match_input_valid",
    "structure_transition",
]
