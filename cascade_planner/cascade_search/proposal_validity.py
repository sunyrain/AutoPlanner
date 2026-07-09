"""Conservative validity filters for generated retrosynthesis proposals."""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem, RDLogger

from cascade_planner.cascadeboard.route_recovery import canonical_side, canonical_smiles


RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class ProposalValidityConfig:
    reject_invalid_molecules: bool = True
    reject_empty_side: bool = True
    reject_self_reaction: bool = True
    dedupe_canonical_sides: bool = True
    max_reactant_to_product_heavy_ratio: float | None = None


@dataclass
class ProposalValidityReport:
    kept: list[str] = field(default_factory=list)
    rejected: list[dict[str, object]] = field(default_factory=list)


def filter_reactant_predictions(
    predictions: list[str],
    *,
    product_smiles: str,
    config: ProposalValidityConfig | None = None,
) -> ProposalValidityReport:
    config = config or ProposalValidityConfig()
    seen: set[tuple[str, ...]] = set()
    report = ProposalValidityReport()
    for prediction in predictions:
        text = str(prediction or "").replace(" ", "")
        reason = proposal_rejection_reason(text, product_smiles=product_smiles, config=config)
        side_key = canonical_side(text)
        if reason is None and config.dedupe_canonical_sides:
            if side_key in seen:
                reason = "duplicate_canonical_side"
            else:
                seen.add(side_key)
        if reason is None:
            report.kept.append(text)
        else:
            report.rejected.append({"prediction": text, "reason": reason})
    return report


def proposal_rejection_reason(
    reactant_side: str,
    *,
    product_smiles: str,
    config: ProposalValidityConfig | None = None,
) -> str | None:
    config = config or ProposalValidityConfig()
    parts = [part for part in str(reactant_side or "").replace(" ", "").split(".") if part]
    if config.reject_empty_side and not parts:
        return "empty_reactant_side"
    if config.reject_invalid_molecules:
        invalid = [part for part in parts if Chem.MolFromSmiles(part) is None]
        if invalid:
            return "invalid_reactant_molecule"
    side_key = canonical_side(".".join(parts))
    if config.reject_empty_side and not side_key:
        return "empty_canonical_side"
    if config.reject_self_reaction and _is_self_reaction(product_smiles, list(side_key)):
        return "self_reaction"
    ratio = config.max_reactant_to_product_heavy_ratio
    if ratio is not None and ratio > 0:
        product_heavy = _heavy_atoms(product_smiles)
        reactants_heavy = sum(_heavy_atoms(part) for part in side_key)
        if product_heavy > 0 and reactants_heavy / product_heavy > float(ratio):
            return "reactant_atom_count_exceeds_ratio"
    return None


def canonicalize_reactant_side_for_output(reactant_side: str) -> str:
    return ".".join(canonical_side(str(reactant_side or "").replace(" ", "")))


def _is_self_reaction(product_smiles: str, reactants: list[str]) -> bool:
    product_key = canonical_smiles(product_smiles)
    if not product_key:
        return False
    return any(canonical_smiles(reactant) == product_key for reactant in reactants if reactant)


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))
