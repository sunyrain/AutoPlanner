"""Optional runtime skeleton reranker.

The current checkpoint is research-grade and disabled by default. Enable only
for controlled A/B runs.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from cascade_planner.eval.train_skeleton_reranker import SkeletonReranker, row_features


DEFAULT_SKELETON_RERANKER_PATH = Path("results/shared/skeleton_reranker/hard_negative_v2_metadata_20260507.pt")
ENABLE_ENV = "AUTOPLANNER_ENABLE_SKELETON_RERANKER"
PATH_ENV = "AUTOPLANNER_SKELETON_RERANKER_PATH"
WEIGHT_ENV = "AUTOPLANNER_SKELETON_RERANKER_WEIGHT"


class SkeletonRerankerInference:
    def __init__(self, model_path: str | Path = DEFAULT_SKELETON_RERANKER_PATH):
        self.model_path = Path(model_path)
        payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.schema = payload["feature_schema"]
        hidden = int(payload.get("hidden") or 192)
        self.model = SkeletonReranker(int(self.schema["feature_dim"]), hidden=hidden)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def score_skeleton(self, *, target_smiles: str, skeleton: Any) -> float:
        row = {
            "target_smiles": target_smiles,
            "depth": len(getattr(skeleton, "types", []) or []),
            "type_sequence": list(getattr(skeleton, "types", []) or []),
            "ec1_sequence": [str(value or "") for value in getattr(skeleton, "ec1s", []) or []],
        }
        x = torch.tensor(row_features(row, self.schema), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(x))[0].item())


def skeleton_reranker_enabled() -> bool:
    return str(os.environ.get(ENABLE_ENV) or "").lower() in {"1", "true", "yes"}


def skeleton_reranker_weight(default: float = 0.25) -> float:
    try:
        value = float(os.environ.get(WEIGHT_ENV, default))
    except (TypeError, ValueError):
        return default
    return max(0.0, min(value, 2.0))


def skeleton_reranker_metadata() -> dict[str, Any]:
    return {
        "enabled": skeleton_reranker_enabled(),
        "path": str(os.environ.get(PATH_ENV) or DEFAULT_SKELETON_RERANKER_PATH),
        "enable_env": ENABLE_ENV,
        "path_env": PATH_ENV,
        "weight_env": WEIGHT_ENV,
        "note": "research artifact; disabled by default until scaffold-heldout metrics improve",
    }


def rerank_skeletons_with_model(
    skeletons: list[Any],
    *,
    target_smiles: str,
    weight: float | None = None,
    ranker: SkeletonRerankerInference | None = None,
) -> list[Any]:
    if ranker is None and not skeleton_reranker_enabled():
        return skeletons
    ranker = ranker or default_skeleton_reranker()
    if ranker is None:
        return skeletons
    rerank_weight = skeleton_reranker_weight() if weight is None else max(0.0, float(weight))
    out = list(skeletons)
    for skel in out:
        try:
            score = ranker.score_skeleton(target_smiles=target_smiles, skeleton=skel)
        except Exception:
            continue
        setattr(skel, "skeleton_reranker_score", score)
        skel.log_prob = float(getattr(skel, "log_prob", 0.0) or 0.0) + rerank_weight * score
    out.sort(key=lambda skel: float(getattr(skel, "log_prob", 0.0) or 0.0), reverse=True)
    return out


@lru_cache(maxsize=2)
def default_skeleton_reranker() -> SkeletonRerankerInference | None:
    path = Path(os.environ.get(PATH_ENV) or DEFAULT_SKELETON_RERANKER_PATH)
    if not path.exists():
        return None
    try:
        return SkeletonRerankerInference(path)
    except Exception:
        return None
