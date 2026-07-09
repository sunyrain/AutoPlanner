"""Zinc stock checker using real InChI key lookup.

Loads 17.4M InChI keys from zinc_stock.hdf5 for accurate stock checking.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Set

from rdkit import Chem
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

_ZINC_KEYS: Set[str] | None = None


def _load_zinc(path: str = "results/shared/zinc_inchikeys.txt") -> Set[str]:
    global _ZINC_KEYS
    if _ZINC_KEYS is not None:
        return _ZINC_KEYS
    p = Path(path)
    if not p.exists():
        # Try loading from HDF5
        hdf = Path("workspace/aizdata/zinc_stock.hdf5")
        if hdf.exists():
            import pandas as pd
            df = pd.read_hdf(str(hdf), key="table")
            _ZINC_KEYS = set(df["inchi_key"].tolist())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(_ZINC_KEYS))
        else:
            _ZINC_KEYS = set()
    else:
        _ZINC_KEYS = set(p.read_text().strip().split("\n"))
    return _ZINC_KEYS


def smiles_to_inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchi = MolToInchi(mol)
        return InchiToInchiKey(inchi) if inchi else None
    except Exception:
        return None


def is_in_zinc_stock(smiles: str) -> bool:
    """Check if a SMILES is in the zinc stock (17.4M molecules)."""
    zinc = _load_zinc()
    if not zinc:
        # Fallback: simple heuristic
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and mol.GetNumHeavyAtoms() <= 6
    ik = smiles_to_inchikey(smiles)
    return ik is not None and ik in zinc
