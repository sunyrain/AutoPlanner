"""Inference API for the frozen EnzExpand reranker.

Given a list of candidate reactant sets for a target product, return the
reranker's score for each. Uses the exact same FEATURE_COLS as training.

This module stays lightweight (no torch) so it can be loaded inside
``cascade_planner.multistep`` pipelines without slow imports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cascade_planner.expand.reranker import FEATURE_COLS, candidate_features


class EnzReranker:
    def __init__(self, model_path: str | Path, meta_path: str | Path | None = None):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(model_path))
        if meta_path is None:
            meta_path = Path(model_path).with_suffix(".meta.json")
        self.meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
        # sanity
        expected = self.meta.get("feature_cols") or FEATURE_COLS
        if expected != FEATURE_COLS:
            raise ValueError(
                f"Saved reranker feature_cols {expected} do not match runtime "
                f"FEATURE_COLS {FEATURE_COLS}. Retrain with matching schema."
            )

    def score(
        self,
        *,
        product_smi: str,
        candidates: list[dict],
    ) -> np.ndarray:
        """Score a list of candidate dicts.

        Each candidate dict must contain:
          mlp_rank: int          (0-based rank within the MLP top-N)
          mlp_logit: float
          tpl_freq: int          (template training frequency)
          tpl_ec1_prob: float    (per-template EC1 prior; 0.0 if unknown)
          tpl_tx_prob: float     (per-template transformation prior)
          cand_reactants: frozenset[str] of canonical reactant SMILES
        """
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        feat_mat = np.zeros((len(candidates), len(FEATURE_COLS)), dtype=np.float32)
        for i, c in enumerate(candidates):
            f = candidate_features(
                mlp_rank=int(c["mlp_rank"]),
                mlp_logit=float(c["mlp_logit"]),
                tpl_freq=int(c["tpl_freq"]),
                tpl_ec1_prob=float(c.get("tpl_ec1_prob", 0.0)),
                tpl_tx_prob=float(c.get("tpl_tx_prob", 0.0)),
                cand_reactants=frozenset(c["cand_reactants"]),
                product_smi=product_smi,
            )
            for j, col in enumerate(FEATURE_COLS):
                feat_mat[i, j] = f[col]
        return self.booster.predict(feat_mat).astype(np.float32)

    def rerank(
        self,
        *,
        product_smi: str,
        candidates: list[dict],
    ) -> list[tuple[int, float]]:
        """Return list of (original_index, score) sorted by score desc."""
        scores = self.score(product_smi=product_smi, candidates=candidates)
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]


def load(model_path: str = "results/shared/reranker_frozen_mf2_ns.txt") -> EnzReranker:
    return EnzReranker(model_path)
