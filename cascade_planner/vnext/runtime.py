"""Optional vNext inference hooks used by search controllers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from cascade_planner.vnext.features import candidate_feature_dim, candidate_feature_vector
from cascade_planner.vnext.models import CandidatePoolCrossAttentionRanker


DEFAULT_VNEXT_CANDIDATE_RANKER = Path("results/shared/vnext/candidate_pool_ranker.pt")
_RUNTIME_CACHE: dict[str, "VNextRuntime | None"] = {}


class VNextRuntime:
    """Load optional vNext models for route search scoring.

    The first runtime-supported model is the candidate-pool cross-attention
    ranker. It can score one candidate or annotate a full candidate pool; both
    paths use the same checkpoint and feature schema.
    """

    def __init__(self, path: str | Path = DEFAULT_VNEXT_CANDIDATE_RANKER):
        ckpt = torch.load(str(path), map_location="cpu")
        schema = ckpt.get("feature_schema") or {}
        meta = ckpt.get("metadata") or {}
        model_kind = schema.get("model_kind") or meta.get("model_kind")
        if model_kind != "candidate_pool_ranker":
            raise ValueError(f"unsupported vNext runtime model kind: {model_kind}")
        self.path = Path(path)
        self.n_bits = int(schema.get("n_bits") or 256)
        self.max_candidates = int(schema.get("max_candidates") or 32)
        self.feature_dim = int(schema.get("candidate_feature_dim") or candidate_feature_dim(self.n_bits))
        config = ckpt.get("model_config") or {}
        self.model = CandidatePoolCrossAttentionRanker(
            candidate_feature_dim=self.feature_dim,
            d_model=int(config.get("d_model") or 128),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def score_candidate(
        self,
        product: str,
        candidate: dict[str, Any],
        *,
        rank: int | float | None = None,
        stock_checker: Callable[[str], bool] | None = None,
    ) -> float:
        features = candidate_feature_vector(
            product,
            candidate,
            rank=rank,
            n_bits=self.n_bits,
            stock_checker=stock_checker,
        )
        if len(features) != self.feature_dim:
            features = _resize_vector(features, self.feature_dim)
        x = torch.tensor(features[None, None, :], dtype=torch.float32)
        mask = torch.ones(1, 1, dtype=torch.bool)
        with torch.no_grad():
            logit = self.model(x, mask)["candidate_logits"][0, 0]
            return float(torch.sigmoid(logit).item())

    def annotate_candidate_pool(
        self,
        product: str,
        candidates: list[dict[str, Any]],
        *,
        stock_checker: Callable[[str], bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        rows = candidates[: self.max_candidates]
        x = []
        for idx, candidate in enumerate(rows, start=1):
            features = candidate_feature_vector(
                product,
                candidate,
                rank=candidate.get("rank") or idx,
                n_bits=self.n_bits,
                stock_checker=stock_checker,
            )
            x.append(_resize_vector(features, self.feature_dim))
        features_t = torch.tensor(np.asarray([x], dtype=np.float32), dtype=torch.float32)
        mask = torch.ones(1, len(rows), dtype=torch.bool)
        with torch.no_grad():
            scores = torch.sigmoid(self.model(features_t, mask)["candidate_logits"])[0].cpu().tolist()
        mean_score = float(sum(scores) / max(len(scores), 1))
        order = {idx: rank for rank, idx in enumerate(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), start=1)}
        out = []
        for idx, (candidate, score) in enumerate(zip(rows, scores)):
            item = dict(candidate)
            item["vnext_candidate_score"] = float(score)
            item["vnext_pool_mean_score"] = mean_score
            item["vnext_pool_rank"] = order.get(idx)
            out.append(item)
        if len(candidates) > len(rows):
            out.extend(candidates[len(rows):])
        return out


def default_vnext_runtime() -> VNextRuntime | None:
    if not _env_truthy("AUTOPLANNER_ENABLE_VNEXT"):
        return None
    path = Path(os.environ.get("AUTOPLANNER_VNEXT_CANDIDATE_RANKER") or DEFAULT_VNEXT_CANDIDATE_RANKER)
    key = str(path)
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    if not path.exists():
        _RUNTIME_CACHE[key] = None
        return None
    try:
        _RUNTIME_CACHE[key] = VNextRuntime(path)
    except Exception:
        _RUNTIME_CACHE[key] = None
    return _RUNTIME_CACHE[key]


def vnext_candidate_weight(default: float = 0.75) -> float:
    raw = os.environ.get("AUTOPLANNER_VNEXT_CANDIDATE_WEIGHT")
    if raw in (None, ""):
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return max(0.0, min(value, 3.0))


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _resize_vector(values, size: int):
    if len(values) == size:
        return values
    arr = np.zeros(size, dtype=np.float32)
    arr[: min(size, len(values))] = values[: min(size, len(values))]
    return arr
