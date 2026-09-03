"""Stable checkpoint and feature contract for planner failure risk."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


DOMAIN_VALUES = [
    "all_chemical",
    "all_enzymatic",
    "chemoenzymatic",
    "hybrid_mimetic",
    "whole_cell_biocatalytic",
]
METRIC_KEYS = [
    "plan",
    "filled_route_any",
    "strict_stock_solve_any",
    "condition_window_success_any",
    "cascade_compatibility_success_any",
    "terminal_GT_reactant_in_top5",
    "filled_type_GT@1",
    "filled_type_GT@5",
    "skeleton_type_GT@1",
    "skeleton_type_GT@5",
]


class FailureClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 2), out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def failure_row_features(row: dict[str, Any], *, n_bits: int) -> np.ndarray:
    target_fp = _fingerprint(row.get("target_smiles"), n_bits=n_bits)
    molecule = Chem.MolFromSmiles(row.get("target_smiles") or "")
    heavy = float(molecule.GetNumHeavyAtoms()) if molecule is not None else 0.0
    domain = row.get("route_domain") or ""
    domain_vector = [1.0 if domain == value else 0.0 for value in DOMAIN_VALUES]
    metrics = row.get("metrics") or {}
    metric_vector = [_boolean_metric(metrics.get(key)) for key in METRIC_KEYS]
    scalar = [
        heavy / 80.0,
        float(row.get("depth") or 0.0) / 10.0,
        float(row.get("n_routes") or 0.0) / 10.0,
        float(bool(row.get("has_failure_label"))),
    ]
    return np.concatenate(
        [
            target_fp,
            np.asarray(domain_vector + metric_vector + scalar, dtype=np.float32),
        ]
    )


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


def _boolean_metric(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False:
        return -1.0
    return 0.0


__all__ = ["FailureClassifier", "failure_row_features"]
