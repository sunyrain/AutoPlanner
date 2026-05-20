"""Runtime adapter for the CCTS-v0 transition ranker.

The scorer is intentionally non-generative: it only scores an already proposed
route-tree action pool.  Search may use the score as a soft priority term, but
candidate generation and validity filtering remain owned by the existing
proposal providers.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_side, canonical_smiles
from cascade_planner.eval.train_ccts_v0_transition_ranker import (
    _baseline_scores,
    _build_evidence_bank,
    _feature_groups,
    _feature_vector,
    _product_evidence_pool,
    _read_json,
    _standardize,
)
from cascade_planner.route_tree.schema import CandidateAction, RouteTreeState


_CCTS_CACHE: dict[str, "CCTSV0Runtime | UnavailableCCTSV0Runtime | None"] = {}


@dataclass
class CCTSV0ScoreResult:
    scores: list[float]
    normalized_scores: list[float]
    active: bool
    reason: str
    selected_score: str = ""
    score_columns: list[str] = field(default_factory=list)
    blend_spec: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)


class UnavailableCCTSV0Runtime:
    def __init__(self, reason: str):
        self.reason = reason

    def score_actions(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        max_depth: int,
    ) -> CCTSV0ScoreResult:
        del state, leaf, actions, max_depth
        return CCTSV0ScoreResult(scores=[], normalized_scores=[], active=False, reason=self.reason)


class CCTSV0Runtime:
    """Score route-tree child actions with a trained CCTS-v0 model bundle."""

    def __init__(
        self,
        model_bundle: str | Path,
        *,
        train_coverage: str | Path,
        report_path: str | Path | None = None,
        blend_name: str | None = None,
        score_name: str | None = None,
        evidence_pool_size: int = 80,
    ):
        self.model_bundle = Path(model_bundle)
        self.train_coverage = Path(train_coverage)
        self.report_path = Path(report_path) if report_path else _default_report_path(self.model_bundle)
        self.blend_name = str(blend_name or "").strip()
        self.score_name = str(score_name or "").strip()
        self.evidence_pool_size = max(1, int(evidence_pool_size or 80))

        with self.model_bundle.open("rb") as fh:
            bundle = pickle.load(fh)
        self.models = dict(bundle.get("models") or {})
        self.feature_schema = dict(bundle.get("feature_schema") or {})
        self.feature_names = list(self.feature_schema.get("feature_names") or [])
        self.schema = {key: value for key, value in self.feature_schema.items() if key not in {"feature_names", "chem_feature_names"}}
        if not self.models:
            raise ValueError("CCTS-v0 model bundle has no models")
        if not self.schema:
            raise ValueError("CCTS-v0 model bundle has no feature schema")

        train_payload = _read_json(self.train_coverage)
        train_transitions = [row for row in train_payload.get("transitions") or [] if isinstance(row, dict)]
        self.evidence_bank = _build_evidence_bank(train_transitions)
        self.feature_groups = _feature_groups(self.feature_names)
        self.blend_spec = _select_blend(_read_optional_json(self.report_path), blend_name=self.blend_name)
        self.selected_score = self._select_runtime_score()

    def score_actions(
        self,
        state: RouteTreeState,
        leaf: str,
        actions: list[CandidateAction],
        *,
        max_depth: int,
    ) -> CCTSV0ScoreResult:
        if not actions:
            return CCTSV0ScoreResult(scores=[], normalized_scores=[], active=True, reason="no_actions")
        try:
            transition = _runtime_transition(state, leaf, max_depth=max_depth)
            evidence_pool = _product_evidence_pool(str(transition.get("product_smiles") or ""), self.evidence_bank, limit=self.evidence_pool_size)
            rows = []
            x_rows = []
            for idx, action in enumerate(actions, start=1):
                row = _runtime_candidate_row(transition, action, rank=idx)
                rows.append(row)
                x_rows.append(_feature_vector(transition, row, evidence_pool=evidence_pool, schema=self.schema))
            if not x_rows:
                return CCTSV0ScoreResult(scores=[], normalized_scores=[], active=False, reason="no_feature_rows")
            x = np.asarray(x_rows, dtype=np.float32)
            score_columns: dict[str, np.ndarray] = {"chem_rank": _baseline_scores(rows)}
            for name, model in self.models.items():
                indices = self._indices_for_model(name)
                if not indices:
                    continue
                score_columns[name] = model.predict(x[:, indices], num_iteration=getattr(model, "best_iteration_", None))
            if self.blend_spec and self.blend_spec["base_model"] in score_columns and self.blend_spec["aux_model"] in score_columns:
                score_columns[self.blend_spec["name"]] = _standardize(score_columns[self.blend_spec["base_model"]]) + float(
                    self.blend_spec["alpha"]
                ) * _standardize(score_columns[self.blend_spec["aux_model"]])
            selected = self.selected_score if self.selected_score in score_columns else _default_score_name(score_columns)
            scores = np.asarray(score_columns[selected], dtype=np.float64)
            normalized = _standardize(scores)
            ranked = _rank_maps(score_columns, rows)
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
                        "selected_score": round(float(scores[idx]), 6),
                        "selected_score_z": round(float(normalized[idx]), 6),
                        "selected_rank": int(ranked.get(selected, {}).get(idx, idx + 1)),
                        "chem_rank": int(ranked.get("chem_rank", {}).get(idx, idx + 1)),
                    }
                )
            return CCTSV0ScoreResult(
                scores=[float(x) for x in scores],
                normalized_scores=[float(x) for x in normalized],
                active=True,
                reason="ccts_v0",
                selected_score=selected,
                score_columns=sorted(score_columns),
                blend_spec=dict(self.blend_spec or {}),
                rows=compact_rows,
            )
        except Exception as exc:
            return CCTSV0ScoreResult(
                scores=[],
                normalized_scores=[],
                active=False,
                reason=f"runtime_error:{type(exc).__name__}",
            )

    def _indices_for_model(self, model_name: str) -> list[int]:
        if model_name == "chem_scalar":
            return list(self.feature_groups.get("chem_scalar") or [])
        if model_name == "chem_only":
            return [idx for idx, name in enumerate(self.feature_names) if name.startswith("chem__")]
        if model_name == "context_evidence_only":
            return list(self.feature_groups.get("ccts_scalar") or [])
        if model_name == "chem_scalar_plus_context_evidence":
            return list(self.feature_groups.get("chem_scalar") or []) + list(self.feature_groups.get("ccts_scalar") or [])
        if model_name == "ccts_evidence":
            return list(range(len(self.feature_names)))
        return list(range(len(self.feature_names)))

    def _select_runtime_score(self) -> str:
        available = {"chem_rank", *self.models.keys()}
        if self.blend_spec and self.blend_spec.get("name"):
            available.add(str(self.blend_spec["name"]))
        if self.score_name and self.score_name in available:
            return self.score_name
        if not self.score_name and "context_evidence_only" in self.models:
            return "context_evidence_only"
        if self.blend_spec and self.blend_spec.get("name"):
            return str(self.blend_spec["name"])
        return _default_score_name({"chem_rank": np.asarray([]), **{name: np.asarray([]) for name in self.models}})


def ccts_v0_runtime_from_env() -> CCTSV0Runtime | UnavailableCCTSV0Runtime | None:
    model_path = os.environ.get("AUTOPLANNER_CCTS_V0_MODEL") or ""
    if not model_path:
        return None
    train_coverage = os.environ.get("AUTOPLANNER_CCTS_V0_TRAIN_COVERAGE") or ""
    report_path = os.environ.get("AUTOPLANNER_CCTS_V0_REPORT") or ""
    blend_name = os.environ.get("AUTOPLANNER_CCTS_V0_BLEND") or ""
    score_name = os.environ.get("AUTOPLANNER_CCTS_V0_SCORE") or ""
    evidence_pool = os.environ.get("AUTOPLANNER_CCTS_V0_EVIDENCE_POOL") or "80"
    key = f"{model_path}::{train_coverage}::{report_path}::{blend_name}::{score_name}::{evidence_pool}"
    if key in _CCTS_CACHE:
        return _CCTS_CACHE[key]
    if not Path(model_path).exists():
        runtime: CCTSV0Runtime | UnavailableCCTSV0Runtime | None = UnavailableCCTSV0Runtime("missing_model_bundle")
        _CCTS_CACHE[key] = runtime
        return runtime
    if not train_coverage or not Path(train_coverage).exists():
        runtime = UnavailableCCTSV0Runtime("missing_train_coverage")
        _CCTS_CACHE[key] = runtime
        return runtime
    try:
        runtime = CCTSV0Runtime(
            model_path,
            train_coverage=train_coverage,
            report_path=report_path or None,
            blend_name=blend_name or None,
            score_name=score_name or None,
            evidence_pool_size=int(evidence_pool or 80),
        )
    except Exception as exc:
        runtime = UnavailableCCTSV0Runtime(f"{type(exc).__name__}:load_failed")
    _CCTS_CACHE[key] = runtime
    return runtime


def _runtime_transition(state: RouteTreeState, leaf: str, *, max_depth: int) -> dict[str, Any]:
    return {
        "transition_id": f"{state.canonical_id}:{canonical_smiles(leaf) or leaf}",
        "target_smiles": state.target,
        "product_smiles": leaf,
        "route_domain": "runtime",
        "step_pos": int(state.depth or 0),
        "remaining_steps": max(0, int(max_depth or 0) - int(state.depth or 0)),
        "previous_transformation_superclass": _previous_transform(state),
        "quality_tier": "",
    }


def _runtime_candidate_row(transition: dict[str, Any], action: CandidateAction, *, rank: int) -> dict[str, Any]:
    rxn = canonical_reaction(action.rxn_smiles)
    reactants = [canonical_smiles(smi) or smi for smi in action.reactants if smi]
    if rxn and ">>" in rxn:
        reactants.extend(canonical_side(rxn.split(">>", 1)[0]))
    reactants = sorted({smi for smi in reactants if smi})
    metadata = dict(action.metadata or {})
    candidate_model = str(
        metadata.get("model_full_name")
        or metadata.get("teacher_source")
        or metadata.get("model_name")
        or ""
    )
    candidate_type = str(action.reaction_type or metadata.get("proposal_type") or metadata.get("type") or "")
    return {
        "transition_id": transition.get("transition_id"),
        "target_smiles": transition.get("target_smiles"),
        "product_smiles": transition.get("product_smiles"),
        "route_domain": transition.get("route_domain"),
        "step_pos": transition.get("step_pos"),
        "remaining_steps": transition.get("remaining_steps"),
        "previous_transformation_superclass": transition.get("previous_transformation_superclass") or "",
        "candidate_rank": int(action.rank or rank or 0) or int(rank),
        "candidate_score": float(action.raw_score or 0.0),
        "candidate_source": str(action.source or ""),
        "candidate_model": candidate_model,
        "candidate_type": candidate_type,
        "candidate_reaction_smiles": rxn,
        "candidate_reactants": reactants,
        "candidate_main_reactant": canonical_smiles(action.main_reactant) or action.main_reactant,
        "exact_label": False,
        "reactant_set_label": False,
        "main_reactant_label": False,
        "any_reactant_label": False,
        "similar_label": False,
        "reactant_similarity": 0.0,
        "positive_label": False,
    }


def _previous_transform(state: RouteTreeState) -> str:
    if not state.steps:
        return ""
    action = state.steps[-1].action
    return str(action.reaction_type or (action.metadata or {}).get("proposal_type") or "")


def _rank_maps(score_columns: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> dict[str, dict[int, int]]:
    out = {}
    for name, scores in score_columns.items():
        order = sorted(range(len(rows)), key=lambda idx: (-float(scores[idx]), int(rows[idx].get("candidate_rank") or 10**9)))
        out[name] = {idx: rank + 1 for rank, idx in enumerate(order)}
    return out


def _default_report_path(model_bundle: Path) -> Path | None:
    candidate = model_bundle.with_name("ccts_v0_report.json")
    return candidate if candidate.exists() else None


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _select_blend(report: dict[str, Any] | None, *, blend_name: str | None) -> dict[str, Any] | None:
    blends = (report or {}).get("blends") or {}
    if not blends:
        return None
    selected_name = str(blend_name or "").strip()
    if not selected_name:
        selected_name = max(blends, key=lambda name: float(blends[name].get("val_selection_score") or 0.0))
    row = blends.get(selected_name)
    if not row:
        return None
    return {
        "name": selected_name,
        "base_model": "chem_only",
        "aux_model": str(row.get("aux_model") or "").strip(),
        "alpha": float(row.get("alpha_selected_on_val") or 0.0),
        "val_selection_score": row.get("val_selection_score"),
    }


def _default_score_name(score_columns: dict[str, np.ndarray]) -> str:
    for name in ("chem_only_plus_context_evidence_only", "chem_only", "chem_scalar", "ccts_evidence", "chem_rank"):
        if name in score_columns:
            return name
    return sorted(score_columns)[0]
