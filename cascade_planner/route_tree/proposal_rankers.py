"""Runtime source-specific proposal rankers.

These rankers only reorder candidate rows emitted by proposal tools. They do
not select route actions; route-tree search still owns action choice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from cascade_planner.route_tree.source_gate import source_group
from cascade_planner.vnext.features import candidate_feature_dim, candidate_feature_vector
from cascade_planner.vnext.models import CandidatePoolCrossAttentionRanker


DEFAULT_PROPOSAL_RANKER_DIR = Path("results/shared/proposal_rankers/full_20260508")
RANKER_FILENAMES = {
    "chemical": "chemical_proposal_ranker.pt",
    "enzymatic": "enzymatic_proposal_ranker.pt",
    "rhea_retrorules": "rhea_retrorules_ranker.pt",
}
_RANKER_CACHE: dict[str, "SourceSpecificProposalRankers | None"] = {}


@dataclass
class _LoadedRanker:
    group: str
    path: Path
    model: CandidatePoolCrossAttentionRanker
    n_bits: int
    max_candidates: int
    feature_dim: int


class SourceSpecificProposalRankers:
    """Load and apply one candidate-pool ranker per proposal source group."""

    def __init__(self, paths: dict[str, str | Path]):
        self.rankers: dict[str, _LoadedRanker] = {}
        for group, path in paths.items():
            loaded = self._load_one(group, Path(path))
            if loaded is not None:
                self.rankers[group] = loaded

    @property
    def active(self) -> bool:
        return bool(self.rankers)

    def request_k(self, source: str, top_k: int) -> int:
        loaded = self.rankers.get(source_group(source))
        if loaded is None:
            return max(1, int(top_k or 1))
        return max(1, int(top_k or 1), loaded.max_candidates)

    def rerank(
        self,
        product: str,
        source: str,
        candidates: list[dict[str, Any]],
        *,
        limit: int,
        stock_checker: Callable[[str], bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        loaded = self.rankers.get(source_group(source))
        if loaded is None:
            return [dict(row) for row in candidates[:limit]]
        scored_rows = [dict(row) for row in candidates[: loaded.max_candidates]]
        if not scored_rows:
            return []
        features = []
        for idx, row in enumerate(scored_rows, start=1):
            row.setdefault("proposal_original_rank", row.get("rank") or idx)
            feat = candidate_feature_vector(
                product,
                row,
                rank=row.get("rank") or idx,
                gt_available=False,
                n_bits=loaded.n_bits,
                stock_checker=stock_checker,
            )
            features.append(_resize_vector(feat, loaded.feature_dim))
        x = torch.tensor(np.asarray([features], dtype=np.float32), dtype=torch.float32)
        mask = torch.ones(1, len(scored_rows), dtype=torch.bool)
        with torch.no_grad():
            logits = loaded.model(x, mask)["candidate_logits"][0].cpu()
            scores = torch.sigmoid(logits).tolist()
        order = sorted(range(len(scored_rows)), key=lambda idx: (scores[idx], -idx), reverse=True)
        out: list[dict[str, Any]] = []
        for rank, idx in enumerate(order, start=1):
            row = scored_rows[idx]
            row["rank"] = rank
            row["proposal_ranker_score"] = float(scores[idx])
            row["proposal_ranker_rank"] = rank
            row["proposal_ranker_group"] = loaded.group
            row["proposal_ranker_model"] = str(loaded.path)
            out.append(row)
        if len(candidates) > len(scored_rows):
            out.extend(dict(row) for row in candidates[len(scored_rows):])
        return out[: max(0, int(limit or 0))]

    @staticmethod
    def _load_one(group: str, path: Path) -> _LoadedRanker | None:
        if not path.exists():
            return None
        ckpt = torch.load(str(path), map_location="cpu")
        schema = ckpt.get("feature_schema") or {}
        meta = ckpt.get("metadata") or {}
        model_kind = schema.get("model_kind") or meta.get("model_kind")
        if model_kind not in {"proposal_ranker", "candidate_pool_ranker"}:
            raise ValueError(f"unsupported proposal ranker kind at {path}: {model_kind}")
        source_group_name = str(schema.get("source_group") or meta.get("source_group") or group)
        n_bits = int(schema.get("n_bits") or meta.get("n_bits") or 128)
        max_candidates = int(schema.get("max_candidates") or meta.get("max_candidates") or 8)
        feature_dim = int(schema.get("candidate_feature_dim") or meta.get("candidate_feature_dim") or candidate_feature_dim(n_bits))
        config = ckpt.get("model_config") or {}
        model = CandidatePoolCrossAttentionRanker(
            candidate_feature_dim=feature_dim,
            d_model=int(config.get("d_model") or meta.get("d_model") or 128),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return _LoadedRanker(
            group=source_group_name,
            path=path,
            model=model,
            n_bits=n_bits,
            max_candidates=max_candidates,
            feature_dim=feature_dim,
        )


def default_proposal_rankers() -> SourceSpecificProposalRankers | None:
    if not _env_truthy("AUTOPLANNER_ENABLE_PROPOSAL_RANKERS"):
        return None
    ranker_dir = Path(os.environ.get("AUTOPLANNER_PROPOSAL_RANKER_DIR") or DEFAULT_PROPOSAL_RANKER_DIR)
    paths = {
        group: Path(os.environ.get(f"AUTOPLANNER_{group.upper()}_PROPOSAL_RANKER") or ranker_dir / filename)
        for group, filename in RANKER_FILENAMES.items()
    }
    key = "|".join(f"{group}={paths[group]}" for group in sorted(paths))
    cached = _RANKER_CACHE.get(key)
    if cached is not None or key in _RANKER_CACHE:
        return cached
    try:
        rankers = SourceSpecificProposalRankers(paths)
    except Exception:
        rankers = None
    if rankers is not None and not rankers.active:
        rankers = None
    _RANKER_CACHE[key] = rankers
    return rankers


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _resize_vector(values: np.ndarray, size: int) -> np.ndarray:
    if values.shape[-1] == size:
        return values.astype(np.float32)
    if values.shape[-1] > size:
        return values[:size].astype(np.float32)
    out = np.zeros(size, dtype=np.float32)
    out[: values.shape[-1]] = values
    return out
