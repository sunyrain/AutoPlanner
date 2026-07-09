"""Shared scoring features for proposal preference models."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.route_recovery import canonical_side, canonical_smiles


RDLogger.DisableLog("rdApp.*")

ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si")


def score_candidate(artifact: dict[str, Any], *, product: str, reactants: str) -> float:
    model = artifact["model"]
    n_bits = int(artifact.get("n_bits") or 128)
    x = candidate_vector(product, reactants, n_bits=n_bits)[None, :]
    proba = np.asarray(model.predict_proba(x), dtype=float)
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    classes = list(getattr(estimator, "classes_", []))
    if 1 in classes:
        return float(proba[0, classes.index(1)])
    return float(proba[0, -1])


def candidate_vector(product: str, reactants: str, *, n_bits: int) -> np.ndarray:
    product_fp = morgan_fp(product, n_bits=n_bits)
    reactant_fp = morgan_fp(reactants, n_bits=n_bits)
    common = np.minimum(product_fp, reactant_fp)
    diff = np.abs(product_fp - reactant_fp)
    numeric = np.asarray(numeric_features(product, reactants), dtype=np.float32)
    return np.concatenate([product_fp, reactant_fp, common, diff, numeric]).astype(np.float32)


def numeric_features(product: str, reactants: str) -> list[float]:
    product_stats = mol_stats(product)
    reactant_stats = mol_stats(reactants)
    reactant_parts = [part for part in str(reactants or "").split(".") if part]
    part_stats = [mol_stats(part) for part in reactant_parts]
    product_heavy = max(product_stats["heavy_atoms"], 1.0)
    max_part_heavy = max((stats["heavy_atoms"] for stats in part_stats), default=0.0)
    min_part_heavy = min((stats["heavy_atoms"] for stats in part_stats), default=0.0)
    product_can = canonical_smiles(product)
    reactant_side = canonical_side(reactants)
    contains_product = float(bool(product_can and product_can in reactant_side))
    side_equals_product = float(reactant_side == canonical_side(product))
    tanimoto = tanimoto_similarity(product, reactants)
    values = [
        float(len(reactant_parts)),
        product_stats["heavy_atoms"] / 100.0,
        reactant_stats["heavy_atoms"] / 100.0,
        max_part_heavy / 100.0,
        min_part_heavy / 100.0,
        reactant_stats["heavy_atoms"] / product_heavy,
        max_part_heavy / product_heavy,
        abs(reactant_stats["heavy_atoms"] - product_stats["heavy_atoms"]) / product_heavy,
        product_stats["rings"] / 10.0,
        reactant_stats["rings"] / 10.0,
        product_stats["hetero_atoms"] / product_heavy,
        reactant_stats["hetero_atoms"] / max(reactant_stats["heavy_atoms"], 1.0),
        product_stats["aromatic_atoms"] / product_heavy,
        reactant_stats["aromatic_atoms"] / max(reactant_stats["heavy_atoms"], 1.0),
        tanimoto,
        contains_product,
        side_equals_product,
        float(product_stats["invalid"]),
        float(reactant_stats["invalid"]),
    ]
    for element in ELEMENTS:
        values.append((reactant_stats["elements"].get(element, 0.0) - product_stats["elements"].get(element, 0.0)) / product_heavy)
    return values


def mol_stats(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {
            "heavy_atoms": 0.0,
            "hetero_atoms": 0.0,
            "rings": 0.0,
            "aromatic_atoms": 0.0,
            "invalid": 1.0,
            "elements": {},
        }
    elements = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    heavy_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))
    return {
        "heavy_atoms": heavy_atoms,
        "hetero_atoms": float(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"})),
        "rings": float(mol.GetRingInfo().NumRings()),
        "aromatic_atoms": float(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())),
        "invalid": 0.0,
        "elements": {key: float(value) for key, value in elements.items()},
    }


def morgan_fp(smiles: str, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(int(n_bits), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(n_bits))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_similarity(left: str, right: str) -> float:
    left_mol = Chem.MolFromSmiles(left or "")
    right_mol = Chem.MolFromSmiles(right or "")
    if left_mol is None or right_mol is None:
        return 0.0
    left_fp = AllChem.GetMorganFingerprintAsBitVect(left_mol, 2, nBits=256)
    right_fp = AllChem.GetMorganFingerprintAsBitVect(right_mol, 2, nBits=256)
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def feature_names(n_bits: int) -> list[str]:
    names: list[str] = []
    for prefix in ("product_fp", "reactants_fp", "fp_common", "fp_abs_diff"):
        names.extend(f"{prefix}_{idx}" for idx in range(int(n_bits)))
    names.extend([
        "n_reactants",
        "product_heavy_atoms_scaled",
        "reactants_heavy_atoms_scaled",
        "max_reactant_heavy_scaled",
        "min_reactant_heavy_scaled",
        "reactants_to_product_heavy_ratio",
        "max_reactant_to_product_heavy_ratio",
        "heavy_atom_abs_delta_ratio",
        "product_rings_scaled",
        "reactants_rings_scaled",
        "product_hetero_fraction",
        "reactants_hetero_fraction",
        "product_aromatic_fraction",
        "reactants_aromatic_fraction",
        "product_reactants_tanimoto",
        "contains_product",
        "side_equals_product",
        "invalid_product",
        "invalid_reactants",
    ])
    names.extend(f"element_delta_{element}" for element in ELEMENTS)
    return names


def is_tie(value: float) -> bool:
    return math.isclose(float(value), 0.0)
