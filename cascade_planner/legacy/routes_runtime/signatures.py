"""Canonical structure signatures shared by current and legacy route adapters."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from rdkit import Chem


def exact_edge_signature(product_smiles: Any, precursor_smiles: Iterable[Any]) -> str:
    """Return a collision-resistant exact-structure retrosynthetic edge key."""

    product = _canonical_smiles(product_smiles)
    precursors = sorted(
        value
        for value in (_canonical_smiles(item) for item in precursor_smiles)
        if value
    )
    if not product or not precursors:
        return ""
    payload = {
        "product_canonical_isomeric_smiles": product,
        "reactant_canonical_isomeric_smiles": precursors,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"edge:sha256:{digest}"


def _canonical_smiles(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


__all__ = ["exact_edge_signature"]
