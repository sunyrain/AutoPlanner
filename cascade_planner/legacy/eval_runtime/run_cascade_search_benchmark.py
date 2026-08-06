"""Frozen model adapters for the active CascadeProgramSearch benchmark shell."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.legacy.cascade_search_runtime import (
    LearnedCascadePairScorer,
    LearnedCascadeValueModel,
    LoadedCascadeActionValueModel,
    LoadedCascadeTransitionValueModel,
    RuleCascadePairScorer,
)
from cascade_planner.eval.run_cascade_search_benchmark import (
    BenchmarkRuntimeOverrides,
    build_parser,
    run_from_args,
)


class RouteBlockValueFinalReranker:
    """Frozen final reranker for already generated search results."""

    def __init__(self, model_pickle: Path):
        self.model_pickle = str(model_pickle)
        with Path(model_pickle).open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(f"expected route/block value model payload dict: {model_pickle}")
        self.model = payload["model"]
        self.feature_names = [str(name) for name in payload.get("feature_names") or []]
        if not self.feature_names:
            raise ValueError(f"route/block value final reranker has no feature_names: {model_pickle}")
        self.mean = np.asarray(payload.get("mean"), dtype=np.float32)
        self.std = np.asarray(payload.get("std"), dtype=np.float32)
        if self.mean.shape[0] != len(self.feature_names) or self.std.shape[0] != len(
            self.feature_names
        ):
            raise ValueError(
                f"route/block value scaler shape does not match feature schema: {model_pickle}"
            )
        self.metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    def rerank(
        self,
        results: list[Any],
        *,
        search_elapsed_s: float | None = None,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        if not results:
            return results, []
        scored: list[tuple[float, float, int, Any, dict[str, Any]]] = []
        for native_rank, result in enumerate(results):
            row = self._row_from_result(
                result,
                native_rank=native_rank,
                search_elapsed_s=search_elapsed_s,
            )
            vector = np.asarray(
                [_nested_feature(row, name) for name in self.feature_names],
                dtype=np.float32,
            )
            score = float(
                self.model.decision_function(((vector - self.mean) / self.std).reshape(1, -1))[0]
            )
            diagnostics = {
                "original_rank": int(native_rank + 1),
                "route_block_value_score": round(score, 6),
                "feature_groups": row["feature_groups"],
                "model_pickle": self.model_pickle,
                "positive_task": self.metadata.get("positive_task"),
                "negative_task": self.metadata.get("negative_task"),
                "contract": "runtime final rerank of generated result programs; no expert labels",
            }
            scored.append(
                (
                    score,
                    float(getattr(result, "score", 0.0) or 0.0),
                    -native_rank,
                    result,
                    diagnostics,
                )
            )
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        ordered = []
        diagnostics = []
        for new_index, (_score, _native_score, _tie, result, detail) in enumerate(
            scored,
            start=1,
        ):
            result.diagnostics.setdefault("route_block_value_final_rerank", {})
            result.diagnostics["route_block_value_final_rerank"] = {
                **detail,
                "new_rank": int(new_index),
            }
            ordered.append(result)
            diagnostics.append({**detail, "new_rank": int(new_index)})
        return ordered, diagnostics

    def _row_from_result(
        self,
        result: Any,
        *,
        native_rank: int,
        search_elapsed_s: float | None,
    ) -> dict[str, Any]:
        state = getattr(result, "state", None)
        steps = list(getattr(state, "step_annotations", []) or []) if state is not None else []
        scores = [_float(getattr(step, "score", None)) for step in steps]
        source_models = {str(getattr(step, "source_model", "") or "unknown") for step in steps}
        reaction_types = {str(getattr(step, "reaction_type", "") or "unknown") for step in steps}
        condition_scores = _runtime_condition_scores(steps)
        enzyme_scores = _runtime_enzyme_scores(steps)
        learned = _runtime_learned_ccts_features(state)
        stock_closed = bool(getattr(state, "stock_closed", False)) if state is not None else False
        feature_groups = {
            "native": {
                "native_rank": float(native_rank),
                "native_inv_rank": 1.0 / float(native_rank + 1),
                "native_score": _mean(
                    [value for value in scores if value is not None],
                    default=_float(getattr(result, "score", 0.0)),
                ),
                "n_steps": float(len(steps)),
            },
            "stock_route": {
                "stock_closed": float(stock_closed),
                "route_solved": float(bool(getattr(result, "solved", False))),
                "strict_stock_solve": float(stock_closed),
                "terminal_max_heavy_atoms": 0.0,
                "terminal_similarity_to_product": 0.0,
            },
            "condition_enzyme": {
                "condition_score_count": float(len(condition_scores)),
                "condition_score_mean": _mean(condition_scores, default=0.0),
                "condition_score_max": max(condition_scores) if condition_scores else 0.0,
                "enzyme_confidence_count": float(len(enzyme_scores)),
                "enzyme_confidence_mean": _mean(enzyme_scores, default=0.0),
                "enzyme_confidence_max": max(enzyme_scores) if enzyme_scores else 0.0,
            },
            "learned_ccts": learned,
            "route_context": {
                "source_model_count": float(len(source_models)),
                "reaction_type_count": float(len(reaction_types)),
                "n_input_species": 0.0,
                "n_output_species": 0.0,
                "n_substrate_scope_entries": 0.0,
                "overall_ee": 0.0,
                "overall_yield": 0.0,
                "search_time_s": float(search_elapsed_s or 0.0),
                "total_reaction_time": 0.0,
            },
        }
        return {"feature_groups": feature_groups}


def build_legacy_runtime_overrides(args: argparse.Namespace) -> BenchmarkRuntimeOverrides:
    value_model_path = _optional_path(args.cascade_value_model)
    transition_model_path = _optional_path(args.cascade_transition_model)
    action_value_model_path = _optional_path(args.cascade_action_value_model)
    pair_scorer_path = _optional_path(args.cascade_pair_scorer)
    final_reranker_path = _optional_path(args.route_block_value_final_reranker)
    cascade_context = _json_mapping(args.chem_enzy_cascade_context_json)
    cascade_cost_model = _json_mapping(args.chem_enzy_cascade_cost_json)
    cascade_source_policy = _json_mapping(args.chem_enzy_cascade_source_policy_json)
    _validate_legacy_inputs(
        explicit_paths={
            "cascade_value_model": value_model_path,
            "cascade_transition_model": transition_model_path,
            "cascade_action_value_model": action_value_model_path,
            "cascade_pair_scorer": pair_scorer_path,
            "route_block_value_final_reranker": final_reranker_path,
        },
        configured_models={
            "chem_enzy_cascade_cost_model": cascade_cost_model,
            "chem_enzy_cascade_source_policy": cascade_source_policy,
        },
    )

    search_flags: dict[str, Any] = {}
    if args.chem_enzy_cascade_cost or cascade_cost_model:
        search_flags["use_cascade_cost_model"] = True
        search_flags["cascade_cost_model"] = dict(cascade_cost_model or {"enabled": True})
        search_flags["cascade_cost_model"].setdefault("enabled", True)
    if (
        args.chem_enzy_cascade_cost
        or args.chem_enzy_cascade_source_policy
        or cascade_context
        or cascade_cost_model
        or cascade_source_policy
        or args.chem_enzy_cascade_context_from_row
    ):
        search_flags["cascade_search_context"] = dict(cascade_context or {"enabled": True})
        search_flags["cascade_search_context"].setdefault("enabled", True)
    if args.chem_enzy_cascade_source_policy or cascade_source_policy:
        search_flags["use_cascade_source_policy"] = True
        search_flags["cascade_source_policy"] = dict(
            cascade_source_policy or {"enabled": True}
        )
        search_flags["cascade_source_policy"].setdefault("enabled", True)

    pair_scorer = None
    if pair_scorer_path:
        pair_scorer = LearnedCascadePairScorer(pair_scorer_path)
    elif args.cascade_rule_pair_scorer:
        pair_scorer = RuleCascadePairScorer()

    cascade_metadata = {
        "legacy_cascade_value_model": str(value_model_path) if value_model_path else None,
        "transition_value_model": str(transition_model_path) if transition_model_path else None,
        "action_value_model": str(action_value_model_path) if action_value_model_path else None,
        "pair_scorer": (
            str(pair_scorer_path)
            if pair_scorer_path
            else ("rule" if args.cascade_rule_pair_scorer else None)
        ),
        "route_block_value_final_reranker": (
            str(final_reranker_path) if final_reranker_path else None
        ),
    }
    chem_enzy_metadata = {
        "cascade_cost_model": {
            "enabled": bool(search_flags.get("use_cascade_cost_model")),
            "context": search_flags.get("cascade_search_context"),
            "cost_model": search_flags.get("cascade_cost_model"),
            "context_from_row": bool(args.chem_enzy_cascade_context_from_row),
            "context_policy": (
                args.chem_enzy_cascade_context_policy
                if args.chem_enzy_cascade_context_from_row
                else None
            ),
        },
        "cascade_source_policy": {
            "enabled": bool(search_flags.get("use_cascade_source_policy")),
            "policy": search_flags.get("cascade_source_policy"),
        },
    }
    return BenchmarkRuntimeOverrides(
        value_model=LearnedCascadeValueModel(value_model_path) if value_model_path else None,
        value_model_label="legacy_learned_cascade_value" if value_model_path else None,
        transition_model=(
            LoadedCascadeTransitionValueModel(transition_model_path)
            if transition_model_path
            else None
        ),
        action_value_model=(
            LoadedCascadeActionValueModel(action_value_model_path)
            if action_value_model_path
            else None
        ),
        pair_scorer=pair_scorer,
        final_reranker=(
            RouteBlockValueFinalReranker(final_reranker_path) if final_reranker_path else None
        ),
        pair_reward_weight=args.cascade_pair_reward_weight,
        pair_reward_mode=args.cascade_pair_reward_mode,
        pair_reward_tie_epsilon=args.cascade_pair_reward_tie_epsilon,
        chem_enzy_search_flags=search_flags,
        chem_enzy_context_from_row=args.chem_enzy_cascade_context_from_row,
        chem_enzy_context_policy=args.chem_enzy_cascade_context_policy,
        metadata={
            "chem_enzy": chem_enzy_metadata,
            "cascade_search": cascade_metadata,
        },
    )


def main() -> None:
    parser = build_parser()
    _add_legacy_arguments(parser)
    args = parser.parse_args()
    runtime_overrides = (
        BenchmarkRuntimeOverrides()
        if args.merge is not None or args.merge_traces is not None
        else build_legacy_runtime_overrides(args)
    )
    run_from_args(args, runtime_overrides=runtime_overrides)


def _add_legacy_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("frozen benchmark adapters")
    group.add_argument("--cascade-value-model", default=None)
    group.add_argument("--cascade-transition-model", default=None)
    group.add_argument("--cascade-action-value-model", default=None)
    group.add_argument("--cascade-pair-scorer", default=None)
    group.add_argument("--cascade-rule-pair-scorer", action="store_true")
    group.add_argument("--cascade-pair-reward-weight", type=float, default=0.0)
    group.add_argument(
        "--cascade-pair-reward-mode",
        default="additive",
        choices=["additive", "guarded_tie_break"],
    )
    group.add_argument("--cascade-pair-reward-tie-epsilon", type=float, default=0.0)
    group.add_argument("--route-block-value-final-reranker", default=None)
    group.add_argument("--chem-enzy-cascade-cost", action="store_true")
    group.add_argument("--chem-enzy-cascade-source-policy", action="store_true")
    group.add_argument("--chem-enzy-cascade-context-json", default=None)
    group.add_argument("--chem-enzy-cascade-cost-json", default=None)
    group.add_argument("--chem-enzy-cascade-source-policy-json", default=None)
    group.add_argument("--chem-enzy-cascade-context-from-row", action="store_true")
    group.add_argument(
        "--chem-enzy-cascade-context-policy",
        default="safe",
        choices=["safe", "strict"],
    )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _json_mapping(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("JSON value must be an object")
    return payload


def _validate_legacy_inputs(
    *,
    explicit_paths: dict[str, Path | None],
    configured_models: dict[str, dict[str, Any] | None],
) -> None:
    for label, path in explicit_paths.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    for label, payload in configured_models.items():
        for dotted_key, value in _iter_configured_model_paths(payload):
            path = Path(str(value))
            if not path.is_file():
                raise FileNotFoundError(f"{label}.{dotted_key} not found: {path}")


def _iter_configured_model_paths(payload: Any, prefix: str = ""):
    model_path_keys = {
        "action_value_model_path",
        "source_value_model_path",
        "transition_value_model_path",
        "cascade_value_model_path",
        "cascade_pair_scorer_path",
        "pair_scorer_model_path",
        "model_path",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key in model_path_keys and value:
                yield dotted, value
            elif isinstance(value, (dict, list)):
                yield from _iter_configured_model_paths(value, dotted)
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            if isinstance(value, (dict, list)):
                yield from _iter_configured_model_paths(value, f"{prefix}[{idx}]")


def _nested_feature(row: dict[str, Any], name: str) -> float:
    group, key = str(name).split(".", 1)
    values = (row.get("feature_groups") or {}).get(group) or {}
    return _float(values.get(key))


def _runtime_condition_scores(steps: list[Any]) -> list[float]:
    values = []
    for step in steps:
        condition = getattr(step, "condition", None)
        confidence = getattr(condition, "confidence", None) if condition is not None else None
        if confidence is not None:
            values.append(_float(confidence))
        raw = getattr(step, "raw_metadata", {}) or {}
        scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
        if scores.get("condition") is not None:
            values.append(_float(scores.get("condition")))
        for item in raw.get("condition_predictions") or []:
            if isinstance(item, dict):
                values.append(_float(item.get("Score", item.get("score", item.get("confidence")))))
    return values


def _runtime_enzyme_scores(steps: list[Any]) -> list[float]:
    values = []
    for step in steps:
        confidence = getattr(step, "evidence_confidence", None)
        if confidence is not None:
            values.append(_float(confidence))
        raw = getattr(step, "raw_metadata", {}) or {}
        scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
        if scores.get("enzyme") is not None:
            values.append(_float(scores.get("enzyme")))
        for item in raw.get("enzyme_ec_annotations") or []:
            if isinstance(item, dict):
                values.append(_float(item.get("confidence", item.get("Confidence"))))
    return values


def _runtime_learned_ccts_features(state: Any | None) -> dict[str, float]:
    values = []
    if state is not None:
        raw_state = getattr(state, "raw_metadata", {})
        for key in ("cascade_pair_summary", "cascade_action_value_summary"):
            summary = raw_state.get(key) if isinstance(raw_state, dict) else {}
            if isinstance(summary, dict):
                for name in ("mean_reward", "total_reward", "mean_score", "max_score"):
                    if summary.get(name) is not None:
                        values.append(_float(summary.get(name)))
        for step in getattr(state, "step_annotations", []) or []:
            raw = getattr(step, "raw_metadata", {}) or {}
            for key in ("ccts_v3_runtime_model_max", "ccts_v3_runtime_model_mean"):
                if raw.get(key) is not None:
                    values.append(_float(raw.get(key)))
    return {
        "ccts_v3_runtime_model_max": max(values) if values else 0.0,
        "ccts_v3_runtime_model_mean": _mean(values, default=0.0),
    }


def _mean(values: Any, *, default: float = 0.0) -> float:
    rows = [float(value) for value in values if value is not None]
    return sum(rows) / len(rows) if rows else float(default)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    from cascade_planner.legacy.guard import require_legacy_research_enabled

    require_legacy_research_enabled("frozen CascadeProgramSearch benchmark adapters")
    main()
