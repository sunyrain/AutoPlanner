"""Stable checkpoint and feature contract for skeleton reranking."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class SkeletonReranker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden, max(48, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(48, hidden // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def skeleton_row_features(
    row: dict[str, Any],
    schema: dict[str, Any],
) -> np.ndarray:
    n_bits = int(schema.get("n_bits") or 128)
    target_fp = _fingerprint(row.get("target_smiles"), n_bits=n_bits)
    max_steps = int(schema.get("max_steps") or 8)
    types = [str(value or "NONE") for value in row.get("type_sequence") or []]
    ec1s = [str(value or "NONE") for value in row.get("ec1_sequence") or []]
    type_positions = _position_one_hot(
        types,
        schema.get("type_vocab") or [],
        max_steps=max_steps,
    )
    ec_positions = _position_one_hot(
        ec1s,
        schema.get("ec1_vocab") or [],
        max_steps=max_steps,
    )
    molecule = Chem.MolFromSmiles(row.get("target_smiles") or "")
    heavy = float(molecule.GetNumHeavyAtoms()) if molecule is not None else 0.0
    depth = int(row.get("depth") or len(types) or 0)
    known_ec = sum(1 for value in ec1s if value not in {"", "NONE", "0"})
    scalar = np.asarray(
        [
            depth / max(max_steps, 1),
            heavy / 100.0,
            len(set(types)) / max(max_steps, 1),
            known_ec / max(depth, 1),
        ],
        dtype=np.float32,
    )
    return np.concatenate([target_fp, type_positions, ec_positions, scalar])


def _fingerprint(smiles: str | None, *, n_bits: int) -> np.ndarray:
    array = np.zeros(n_bits, dtype=np.float32)
    molecule = Chem.MolFromSmiles(smiles or "")
    if molecule is None:
        return array
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        2,
        nBits=n_bits,
    )
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def _position_one_hot(
    values: list[str],
    vocabulary: list[str],
    *,
    max_steps: int,
) -> np.ndarray:
    index = {token: position for position, token in enumerate(vocabulary)}
    array = np.zeros(max_steps * len(vocabulary), dtype=np.float32)
    for position, value in enumerate(values[:max_steps]):
        vocabulary_index = index.get(value)
        if vocabulary_index is None:
            continue
        array[position * len(vocabulary) + vocabulary_index] = 1.0
    return array


__all__ = ["SkeletonReranker", "skeleton_row_features"]
