"""Runtime helper for GraphFP top-N candidate reranking."""
from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascadeboard.value_function import candidate_value_features, heavy_atoms


RDLogger.DisableLog("rdApp.*")

DEFAULT_GRAPHFP_RERANKER_PATH = Path(
    "results/shared/graphfp_reranker_20260530/model/graphfp_lightgbm_reranker.pkl"
)
NUMERIC_FEATURES = [
    "candidate_score",
    "stock_fraction",
    "main_reduction",
    "has_ec",
    "has_evidence",
    "large_aux_penalty",
    "self_loop",
    "rank",
    "inverse_rank",
    "candidate_count",
    "num_reactants",
    "product_heavy_atoms",
    "largest_reactant_heavy_atoms",
    "total_reactant_heavy_atoms",
    "template_length",
    "template_has_chiral",
    "template_has_ring",
]


class GraphFPLightGBMReranker:
    def __init__(self, path: str | Path = DEFAULT_GRAPHFP_RERANKER_PATH):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            artifact = pickle.load(handle)
        self.model = artifact["model"]
        schema = artifact.get("feature_schema") or {}
        self.n_bits = int(schema.get("n_bits") or 256)
        self.best_blend_alpha = artifact.get("best_blend_alpha")

    @classmethod
    def from_env(cls) -> "GraphFPLightGBMReranker | None":
        raw_path = os.environ.get("AUTOPLANNER_GRAPHFP_RERANKER_PATH")
        path = Path(raw_path) if raw_path else DEFAULT_GRAPHFP_RERANKER_PATH
        if not path.exists():
            return None
        try:
            return cls(path)
        except Exception:
            return None

    def rerank(
        self,
        product: str,
        candidates: list[dict[str, Any]],
        *,
        output_k: int | None = None,
        blend_alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        x = np.asarray([candidate_features(product, item, n_bits=self.n_bits) for item in candidates], dtype=np.float32)
        scores = self.model.predict(x)
        alpha = self.best_blend_alpha if blend_alpha is None else blend_alpha
        reranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["graphfp_reranker_score"] = float(score)
            item["graphfp_reranker_blend_alpha"] = alpha
            item["graphfp_reranker_combined_score"] = combined_score(item, float(score), alpha)
            reranked.append(item)
        reranked.sort(key=lambda item: float(item.get("graphfp_reranker_combined_score") or 0.0), reverse=True)
        for idx, item in enumerate(reranked, start=1):
            item["graphfp_original_rank"] = item.get("rank")
            item["rank"] = idx
        return reranked[:output_k] if output_k else reranked


def candidate_features(product: str, candidate: dict[str, Any], *, n_bits: int) -> np.ndarray:
    product_fp = _morgan_fp(product, n_bits=n_bits)
    reactants = [str(item) for item in candidate.get("reactant_smiles") or [] if str(item or "")]
    if not reactants:
        reactants = [candidate.get("main_reactant") or "", *(candidate.get("aux_reactants") or [])]
    reactants_text = ".".join(str(item) for item in reactants if str(item or ""))
    reactant_fp = _morgan_fp(reactants_text, n_bits=n_bits)
    shared = np.logical_and(product_fp > 0, reactant_fp > 0).astype(np.float32)
    exported = candidate_value_features(product, candidate)
    template = str(candidate.get("template") or "")
    product_atoms = heavy_atoms(product)
    reactant_atoms = [heavy_atoms(smi) for smi in reactants]
    candidate_count = float(candidate.get("candidate_count") or len(reactants) or 0.0)
    rank = float(candidate.get("rank") or 1.0)
    values = {
        **exported,
        "rank": rank,
        "inverse_rank": 1.0 / max(rank, 1.0),
        "candidate_count": candidate_count,
        "num_reactants": float(len(reactants)),
        "product_heavy_atoms": float(product_atoms),
        "largest_reactant_heavy_atoms": float(max(reactant_atoms, default=0)),
        "total_reactant_heavy_atoms": float(sum(reactant_atoms)),
        "template_length": float(len(template)),
        "template_has_chiral": float("@" in template),
        "template_has_ring": float(any(ch.isdigit() for ch in template)),
    }
    numeric = [float(values.get(name) or candidate.get(name) or 0.0) for name in NUMERIC_FEATURES]
    score = float(candidate.get("score") or 0.0)
    numeric.extend([math.log(max(score, 1e-12)), _tanimoto(product_fp, reactant_fp)])
    return np.concatenate([product_fp, reactant_fp, shared, np.asarray(numeric, dtype=np.float32)])


def combined_score(candidate: dict[str, Any], model_score: float, blend_alpha: float | None) -> float:
    if blend_alpha is None:
        return float(model_score)
    base = math.log(max(float(candidate.get("score") or 0.0), 1e-12))
    return float(model_score) + float(blend_alpha) * base


def _morgan_fp(smiles: str | None, *, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union else 0.0
