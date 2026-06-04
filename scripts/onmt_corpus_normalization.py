"""Strict SMILES-side normalization for ChemEnzy/OpenNMT proposal corpora."""
from __future__ import annotations

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def canonicalize_product_and_reactants(product: str, reactants: list[str]) -> tuple[str, list[str]]:
    """Return canonical product and sorted canonical reactant side, or empty values.

    The older route-recovery helper intentionally preserves unparsable SMILES as
    text for diagnostics. Training corpora need stricter behavior: invalid
    source or target strings should not become supervised sequence targets.
    """
    product_side = canonicalize_side_strict(product)
    reactant_side = canonicalize_side_strict(".".join(str(item) for item in reactants if item))
    if not product_side or not reactant_side:
        return "", []
    return ".".join(product_side), list(reactant_side)


def canonicalize_side_strict(side: str) -> tuple[str, ...]:
    out: list[str] = []
    for raw_part in str(side or "").split("."):
        part = raw_part.strip()
        if not part:
            continue
        canonical = canonicalize_smiles_strict(part)
        if not canonical:
            return ()
        out.append(canonical)
    return tuple(sorted(out))


def canonicalize_smiles_strict(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)
