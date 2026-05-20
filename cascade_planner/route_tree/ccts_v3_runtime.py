"""Runtime-safe CCTS-v3 action scorer for route-tree search."""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.eval.audit_candidate_specific_evidence import _train_bank
from cascade_planner.eval.build_ccts_v3_runtime_evidence_cache import _runtime_evidence_scores
from cascade_planner.eval.train_ccts_v3_runtime_pairwise_ranker import FittedRuntimeModel, _feature_row
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState


DEFAULT_CCTS_V3_MODEL_NAME = "runtime_pairwise_block_supported_positive_label__runtime_evidence_only"
_CCTS_V3_CACHE: dict[str, "CCTSV3Runtime | UnavailableCCTSV3Runtime | None"] = {}


@dataclass
class CCTSV3ScoreResult:
    scores: list[float]
    normalized_scores: list[float]
    active: bool
    reason: str
    selected_score: str = ""
    score_columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)


class UnavailableCCTSV3Runtime:
    def __init__(self, reason: str):
        self.reason = reason

    def score_actions(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        max_depth: int,
    ) -> CCTSV3ScoreResult:
        del state, leaf, actions, max_depth
        return CCTSV3ScoreResult(scores=[], normalized_scores=[], active=False, reason=self.reason)


class CCTSV3Runtime:
    """Score already-proposed actions with runtime-safe CCTS-v3 evidence."""

    def __init__(
        self,
        model_pickle: str | Path,
        *,
        program_manifest: str | Path,
        model_name: str = DEFAULT_CCTS_V3_MODEL_NAME,
        score_name: str = "model",
    ):
        self.model_pickle = Path(model_pickle)
        self.program_manifest = Path(program_manifest)
        self.model_name = str(model_name or DEFAULT_CCTS_V3_MODEL_NAME)
        self.score_name = str(score_name or "model")
        self.model_bundle = _load_model_bundle(self.model_pickle, self.model_name)
        self.fitted: FittedRuntimeModel = self.model_bundle["model"]
        self.train_bank = _train_bank(self.program_manifest)
        self.product_sim_cache: dict[tuple[str, str], list[float]] = {}

    def score_actions(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        max_depth: int,
    ) -> CCTSV3ScoreResult:
        del max_depth
        if not actions:
            return CCTSV3ScoreResult(scores=[], normalized_scores=[], active=True, reason="no_actions")
        try:
            previous_transform = _previous_transform(state)
            rows = []
            model_scores = []
            raw_any = []
            raw_pair = []
            chem_rank = []
            for idx, action in enumerate(actions, start=1):
                row = _runtime_candidate_row(
                    leaf,
                    action,
                    rank=idx,
                    previous_transform=previous_transform,
                    train_bank=self.train_bank,
                    product_sim_cache=self.product_sim_cache,
                )
                rows.append(row)
                model_scores.append(_model_score(self.fitted, row))
                raw_any.append(float(row.get("runtime_nearest_any_transition_sim") or 0.0))
                raw_pair.append(float(row.get("runtime_nearest_pair_compatible_sim") or 0.0))
                chem_rank.append(-float(row.get("candidate_rank") or idx))
            score_columns = {
                "model": np.asarray(model_scores, dtype=np.float64),
                "runtime_any": np.asarray(raw_any, dtype=np.float64),
                "runtime_pair": np.asarray(raw_pair, dtype=np.float64),
                "chem_rank": np.asarray(chem_rank, dtype=np.float64),
            }
            selected = self.score_name if self.score_name in score_columns else "model"
            scores = score_columns[selected]
            normalized = _standardize(scores)
            ranks = _rank_maps(score_columns)
            compact_rows = []
            for idx, row in enumerate(rows):
                compact_rows.append(
                    {
                        "action_key": actions[idx].canonical_key,
                        "candidate_rank": row.get("candidate_rank"),
                        "candidate_source": row.get("candidate_source"),
                        "candidate_model": row.get("candidate_model"),
                        "candidate_type": row.get("candidate_type"),
                        "candidate_main_reactant": row.get("candidate_main_reactant"),
                        "runtime_inferred_transform": row.get("runtime_inferred_transform"),
                        "runtime_any": round(float(row.get("runtime_nearest_any_transition_sim") or 0.0), 6),
                        "runtime_pair": round(float(row.get("runtime_nearest_pair_compatible_sim") or 0.0), 6),
                        "selected_score": round(float(scores[idx]), 6),
                        "selected_score_z": round(float(normalized[idx]), 6),
                        "selected_rank": int(ranks.get(selected, {}).get(idx, idx + 1)),
                        "chem_rank": int(ranks.get("chem_rank", {}).get(idx, idx + 1)),
                    }
                )
            return CCTSV3ScoreResult(
                scores=[float(x) for x in scores],
                normalized_scores=[float(x) for x in normalized],
                active=True,
                reason="ccts_v3_runtime",
                selected_score=selected,
                score_columns=sorted(score_columns),
                rows=compact_rows,
            )
        except Exception as exc:
            return CCTSV3ScoreResult(
                scores=[],
                normalized_scores=[],
                active=False,
                reason=f"runtime_error:{type(exc).__name__}",
            )


def ccts_v3_runtime_from_env() -> CCTSV3Runtime | UnavailableCCTSV3Runtime | None:
    model_path = os.environ.get("AUTOPLANNER_CCTS_V3_RUNTIME_MODEL") or ""
    if not model_path:
        return None
    program_manifest = os.environ.get("AUTOPLANNER_CCTS_V3_RUNTIME_PROGRAM_MANIFEST") or ""
    model_name = os.environ.get("AUTOPLANNER_CCTS_V3_RUNTIME_MODEL_NAME") or DEFAULT_CCTS_V3_MODEL_NAME
    score_name = os.environ.get("AUTOPLANNER_CCTS_V3_RUNTIME_SCORE") or "model"
    key = f"{model_path}::{program_manifest}::{model_name}::{score_name}"
    if key in _CCTS_V3_CACHE:
        return _CCTS_V3_CACHE[key]
    if not Path(model_path).exists():
        runtime: CCTSV3Runtime | UnavailableCCTSV3Runtime | None = UnavailableCCTSV3Runtime("missing_model_pickle")
        _CCTS_V3_CACHE[key] = runtime
        return runtime
    if not program_manifest or not Path(program_manifest).exists():
        runtime = UnavailableCCTSV3Runtime("missing_program_manifest")
        _CCTS_V3_CACHE[key] = runtime
        return runtime
    try:
        runtime = CCTSV3Runtime(
            model_path,
            program_manifest=program_manifest,
            model_name=model_name,
            score_name=score_name,
        )
    except Exception as exc:
        runtime = UnavailableCCTSV3Runtime(f"{type(exc).__name__}:load_failed")
    _CCTS_V3_CACHE[key] = runtime
    return runtime


def _load_model_bundle(path: Path, model_name: str) -> dict[str, Any]:
    # The training script may have been executed as __main__; make the class
    # available there so old pickle payloads can be loaded.
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "FittedRuntimeModel"):
        setattr(main_mod, "FittedRuntimeModel", FittedRuntimeModel)
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    models = payload.get("models") or {}
    if model_name not in models:
        raise KeyError(f"model {model_name!r} not found; available={sorted(models)}")
    return {"model": models[model_name], "metadata": payload.get("metadata") or {}}


def _runtime_candidate_row(
    leaf: str,
    action: CandidateAction,
    *,
    rank: int,
    previous_transform: str,
    train_bank: dict[str, Any],
    product_sim_cache: dict[tuple[str, str], list[float]],
) -> dict[str, Any]:
    evidence = _runtime_evidence_scores(
        product=str(leaf or action.product or ""),
        candidate_main=str(action.main_reactant or ""),
        previous_transform=previous_transform,
        next_transform="",
        train_bank=train_bank,
        product_sim_cache=product_sim_cache,
    )
    metadata = dict(action.metadata or {})
    return {
        "product_smiles": str(leaf or action.product or ""),
        "candidate_rank": int(action.rank or rank or 0) or int(rank),
        "candidate_score": float(action.raw_score or 0.0),
        "candidate_source": str(action.source or ""),
        "candidate_model": str(metadata.get("model_full_name") or metadata.get("teacher_source") or metadata.get("model_name") or ""),
        "candidate_type": str(action.reaction_type or metadata.get("proposal_type") or metadata.get("type") or ""),
        "candidate_reactants": list(action.reactants or ()),
        "candidate_main_reactant": str(action.main_reactant or ""),
        **evidence,
    }


def _model_score(fitted: FittedRuntimeModel, row: dict[str, Any]) -> float:
    x = np.asarray([_feature_row(row)], dtype=np.float32)[:, fitted.feature_indices]
    return float(fitted.model.decision_function((x - fitted.mean) / fitted.std)[0])


def _previous_transform(state: RouteTreeState) -> str:
    if not state.steps:
        return ""
    action = state.steps[-1].action
    return str(action.reaction_type or (action.metadata or {}).get("proposal_type") or "")


def _standardize(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    std = float(arr.std())
    if std < 1e-9:
        return arr * 0.0
    return (arr - float(arr.mean())) / std


def _rank_maps(score_columns: dict[str, np.ndarray]) -> dict[str, dict[int, int]]:
    out = {}
    for name, scores in score_columns.items():
        order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
        out[name] = {idx: rank + 1 for rank, idx in enumerate(order)}
    return out
