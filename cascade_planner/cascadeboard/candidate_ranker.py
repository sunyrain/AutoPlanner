"""Runtime inference for the pack-trained candidate reranker."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.value_function import candidate_value_features


RDLogger.DisableLog("rdApp.*")

DEFAULT_RANKER_PATH = Path("results/shared/candidate_ranker/pack_candidate_ranker_20260507.pt")
DEFAULT_RANKER_WEIGHT = 0.75
_RANKER_CACHE: dict[str, "PackCandidateRankerInference | None"] = {}


class _PackCandidateRankerModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden, max(32, hidden // 3)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 3), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PackCandidateRankerInference:
    def __init__(self, path: str | Path = DEFAULT_RANKER_PATH):
        ckpt = torch.load(str(path), map_location="cpu")
        schema = ckpt.get("feature_schema") or {}
        self.n_bits = int(schema.get("n_bits") or 128)
        self.numeric_features = list(schema.get("numeric_features") or [])
        self.source_values = list(schema.get("source_values") or [])
        self.metadata_features = list(schema.get("metadata_features") or ["has_ec", "has_type", "has_doi"])
        self.feature_dim = int(schema.get("feature_dim") or 0)
        hidden = int(ckpt.get("hidden") or 192)
        self.model = _PackCandidateRankerModel(self.feature_dim, hidden=hidden)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def score_candidate(
        self,
        product: str,
        candidate: dict[str, Any],
        *,
        stock_checker: Callable[[str], bool] | None = None,
    ) -> float:
        features = _candidate_feature_vector(
            product,
            candidate,
            n_bits=self.n_bits,
            numeric_features=self.numeric_features,
            source_values=self.source_values,
            metadata_features=self.metadata_features,
            feature_dim=self.feature_dim,
            stock_checker=stock_checker,
        )
        with torch.no_grad():
            logit = self.model(torch.tensor(features[None, :], dtype=torch.float32))
            return float(torch.sigmoid(logit)[0].item())


def default_candidate_ranker() -> PackCandidateRankerInference | None:
    if str(os.environ.get("AUTOPLANNER_DISABLE_CANDIDATE_RANKER") or "").lower() in {"1", "true", "yes"}:
        return None
    key = str(DEFAULT_RANKER_PATH)
    if key in _RANKER_CACHE:
        return _RANKER_CACHE[key]
    if not DEFAULT_RANKER_PATH.exists():
        _RANKER_CACHE[key] = None
        return None
    try:
        _RANKER_CACHE[key] = PackCandidateRankerInference(DEFAULT_RANKER_PATH)
    except Exception:
        _RANKER_CACHE[key] = None
    return _RANKER_CACHE[key]


def candidate_ranker_weight(default: float = DEFAULT_RANKER_WEIGHT) -> float:
    raw = os.environ.get("AUTOPLANNER_CANDIDATE_RANKER_WEIGHT")
    if raw in (None, ""):
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return max(0.0, min(value, 3.0))


def _candidate_feature_vector(
    product: str,
    candidate: dict[str, Any],
    *,
    n_bits: int,
    numeric_features: list[str],
    source_values: list[str],
    metadata_features: list[str],
    feature_dim: int,
    stock_checker: Callable[[str], bool] | None = None,
) -> np.ndarray:
    product_fp = _fp(product, n_bits=n_bits)
    reactants = [candidate.get("main_reactant") or ""]
    reactants.extend(candidate.get("aux_reactants") or [])
    reactant_fp = _fp(".".join(x for x in reactants if x), n_bits=n_bits)
    exported = candidate_value_features(product, candidate, stock_checker=stock_checker)
    numeric = [float(exported.get(name) or 0.0) for name in numeric_features]
    rank = float(candidate.get("rank") or 1.0)
    rank_features = [
        1.0 / max(rank, 1.0),
        np.log1p(rank) / 5.0,
        0.0,
    ]
    source = str(candidate.get("source") or "").lower()
    source_features = [1.0 if source == value else 0.0 for value in source_values]
    metadata = _candidate_metadata_features(candidate, metadata_features)
    vec = np.concatenate([
        product_fp,
        reactant_fp,
        np.asarray(numeric + rank_features + source_features + metadata, dtype=np.float32),
    ]).astype(np.float32)
    if feature_dim and len(vec) != feature_dim:
        fixed = np.zeros(feature_dim, dtype=np.float32)
        fixed[: min(feature_dim, len(vec))] = vec[: min(feature_dim, len(vec))]
        return fixed
    return vec


def _candidate_metadata_features(candidate: dict[str, Any], feature_names: list[str]) -> list[float]:
    evidence = candidate.get("evidence") or {}
    t_value = _safe_float(candidate.get("T"))
    ph_value = _safe_float(candidate.get("pH"))
    values = {
        "has_ec": float(bool(candidate.get("ec"))),
        "has_type": float(bool(candidate.get("type") or candidate.get("reaction_type"))),
        "has_doi": float(bool(candidate.get("doi") or evidence.get("doi"))),
        "has_uniprot": float(bool(candidate.get("uniprot_accession") or evidence.get("uniprot_accession"))),
        "has_T": float(t_value is not None),
        "has_pH": float(ph_value is not None),
        "has_T_and_pH": float(t_value is not None and ph_value is not None),
        "T_scaled": float(t_value or 0.0) / 100.0,
        "pH_scaled": float(ph_value or 0.0) / 14.0,
        "has_solvent": float(bool(candidate.get("solvent"))),
        "has_catalyst": float(bool(candidate.get("catalyst"))),
        "has_enzyme_uid": float(bool(candidate.get("enzyme_uid"))),
        "has_cofactor": float(bool(candidate.get("cofactor") or evidence.get("cofactor"))),
        "has_condition_match": float(bool(candidate.get("condition_match") or evidence.get("condition_match"))),
    }
    return [values.get(name, 0.0) for name in feature_names]


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr
