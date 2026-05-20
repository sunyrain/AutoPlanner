"""Run ChemEnzy-backed CascadeProgramSearch benchmarks.

This runner evaluates the new cascade-native search architecture. ChemEnzy is
used as a proposal provider, while CascadeProgramSearch owns state, failures,
repairs, cofactor/stage bookkeeping, and final ranking.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from cascade_planner.baselines.chem_enzy_adapter import (
    BACKEND_NAME,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
    ChemEnzyBackendAdapter,
)
from cascade_planner.baselines.route_contract import BaselineRunResult, RouteSearchConfig
from cascade_planner.cascade_search import (
    CascadeProgramSearch,
    CascadeSearchConfig,
    CascadeSearchController,
    HeuristicCascadeValueModel,
    LearnedCascadePairScorer,
    LearnedCascadeValueModel,
    LoadedCascadeTransitionValueModel,
    LoadedCascadeActionValueModel,
    SubgoalHintActionScorer,
    RuleCascadePairScorer,
    CascadeRetrievalProposalProvider,
    CascadeSubgoalEvidenceProvider,
    ChemicalTemplateProposalProvider,
    RetroChimeraProposalProvider,
    StaticProposalProvider,
    TemplateRelevanceProposalProvider,
    CascadeAction,
    CascadeActionType,
    StepAnnotation,
    route_step_candidate_to_action,
    score_cascade_state,
)
from cascade_planner.cascade_search.trace import CascadeTraceCollector, TRACE_SCHEMA_VERSION
from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_smiles,
    gt_reactants,
    reaction_reactants,
)
from cascade_planner.eval.product_route_feasibility_audit import (
    ROUTE_CLASS_ORDER,
    _audit_route as _audit_product_route,
    _mol_props as _product_mol_props,
    product_audit_risk_order,
)


def run_cascade_search_benchmark(
    *,
    benchmark_path: Path,
    output_path: Path,
    vendor_root: Path,
    stock_names: list[str] | None = None,
    one_step_models: list[str] | None = None,
    iterations: int = 10,
    chem_enzy_max_depth: int = 6,
    expansion_topk: int = 50,
    gpu: int = -1,
    limit: int | None = None,
    enable_condition_prediction: bool = False,
    enable_enzyme_assignment: bool = False,
    condition_model: str = "rcr",
    cascade_max_depth: int = 6,
    cascade_branch_factor: int = 12,
    cascade_leaf_beam_size: int = 1,
    cascade_diverse_leaf_reserve: int = 0,
    cascade_proposal_topk: int | None = None,
    cascade_expansion_budget: int = 100,
    cascade_result_limit: int = 5,
    cascade_min_step_score: float = 0.05,
    cascade_value_model_path: Path | None = None,
    cascade_transition_model_path: Path | None = None,
    cascade_action_value_model_path: Path | None = None,
    cascade_subgoal_provider_model_path: Path | None = None,
    cascade_subgoal_program_manifest: Path | None = None,
    use_cascade_subgoal_action_scorer: bool = False,
    cascade_subgoal_action_max_bonus: float = 0.20,
    cascade_pair_scorer_path: Path | None = None,
    use_rule_pair_scorer: bool = False,
    cascade_pair_reward_weight: float = 0.0,
    cascade_pair_reward_mode: str = "additive",
    cascade_pair_reward_tie_epsilon: float = 0.0,
    route_block_value_final_reranker_path: Path | None = None,
    use_product_audit_final_reranker: bool = False,
    use_chem_enzy_cascade_cost: bool = False,
    use_chem_enzy_cascade_source_policy: bool = False,
    chem_enzy_cascade_context: dict[str, Any] | None = None,
    chem_enzy_cascade_cost_model: dict[str, Any] | None = None,
    chem_enzy_cascade_source_policy: dict[str, Any] | None = None,
    chem_enzy_cascade_context_from_row: bool = False,
    chem_enzy_cascade_context_policy: str = "safe",
    reuse_planner: bool = True,
    dry_run: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    trace_output_path: Path | None = None,
    chem_enzy_expansion_trace_output_path: Path | None = None,
    use_chem_enzy_expansion_proposals: bool = False,
    chem_enzy_expansion_proposal_topk_per_leaf: int | None = 50,
    use_cascade_retrieval_proposals: bool = False,
    cascade_retrieval_program_manifest: Path | None = None,
    cascade_retrieval_mode: str = "block_downstream_product",
    cascade_retrieval_min_similarity: float = 0.20,
    cascade_retrieval_require_downstream_transform: bool = False,
    cascade_retrieval_topk: int = 8,
    use_retrochimera_proposals: bool = False,
    retrochimera_model_dir: Path = Path("data_external/retrochimera_model"),
    retrochimera_device: str | None = None,
    retrochimera_topk: int = 8,
    use_chemical_template_proposals: bool = False,
    chemical_template_topk: int = 12,
    chemical_template_prefer_preselector: bool = True,
    use_template_relevance_proposals: bool = False,
    template_relevance_models: list[str] | None = None,
    template_relevance_topk: int = 8,
    include_route_outcomes: bool = False,
) -> dict[str, Any]:
    rows = _read_rows(benchmark_path)
    if limit is not None:
        rows = rows[: int(limit)]
    num_shards = max(1, int(num_shards or 1))
    shard_index = int(shard_index or 0)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    unsharded_count = len(rows)
    rows = rows[shard_index::num_shards]
    _validate_model_inputs(
        cascade_value_model_path=cascade_value_model_path,
        cascade_transition_model_path=cascade_transition_model_path,
        cascade_action_value_model_path=cascade_action_value_model_path,
        cascade_subgoal_provider_model_path=cascade_subgoal_provider_model_path,
        cascade_pair_scorer_path=cascade_pair_scorer_path,
        route_block_value_final_reranker_path=route_block_value_final_reranker_path,
        chem_enzy_cascade_cost_model=chem_enzy_cascade_cost_model,
        chem_enzy_cascade_source_policy=chem_enzy_cascade_source_policy,
    )
    adapter = ChemEnzyBackendAdapter(
        vendor_root=vendor_root,
        gpu=gpu,
        enable_condition_prediction=enable_condition_prediction,
        enable_enzyme_assignment=enable_enzyme_assignment,
    )
    _validate_stock_inputs(adapter.config_path, stock_names or DEFAULT_STOCKS)
    base_chem_enzy_search_flags: dict[str, Any] = {"gpu": gpu, "condition_model": condition_model}
    if (
        use_chem_enzy_cascade_cost
        or chem_enzy_cascade_cost_model
    ):
        base_chem_enzy_search_flags["use_cascade_cost_model"] = True
        base_chem_enzy_search_flags["cascade_cost_model"] = dict(chem_enzy_cascade_cost_model or {"enabled": True})
        base_chem_enzy_search_flags["cascade_cost_model"].setdefault("enabled", True)
    if (
        use_chem_enzy_cascade_cost
        or use_chem_enzy_cascade_source_policy
        or chem_enzy_cascade_context
        or chem_enzy_cascade_cost_model
        or chem_enzy_cascade_source_policy
        or chem_enzy_cascade_context_from_row
    ):
        base_chem_enzy_search_flags["cascade_search_context"] = dict(chem_enzy_cascade_context or {"enabled": True})
        base_chem_enzy_search_flags["cascade_search_context"].setdefault("enabled", True)
    if use_chem_enzy_cascade_source_policy or chem_enzy_cascade_source_policy:
        base_chem_enzy_search_flags["use_cascade_source_policy"] = True
        base_chem_enzy_search_flags["cascade_source_policy"] = dict(
            chem_enzy_cascade_source_policy or {"enabled": True}
        )
        base_chem_enzy_search_flags["cascade_source_policy"].setdefault("enabled", True)
    if chem_enzy_expansion_trace_output_path is not None or use_chem_enzy_expansion_proposals:
        base_chem_enzy_search_flags["include_cascade_expansion_trace"] = True

    configs = [
        RouteSearchConfig(
            target_smiles=str(row["target_smiles"]),
            stock_names=stock_names or DEFAULT_STOCKS,
            max_iterations=iterations,
            max_depth=chem_enzy_max_depth,
            expansion_topk=expansion_topk,
            one_step_models=one_step_models or DEFAULT_ONE_STEP_MODELS,
            search_flags=_chem_enzy_search_flags_for_row(
                row,
                base_chem_enzy_search_flags,
                context_from_row=chem_enzy_cascade_context_from_row,
                context_policy=chem_enzy_cascade_context_policy,
            ),
        )
        for row in rows
    ]

    started = time.monotonic()
    chem_results = adapter.run_targets(configs, dry_run=dry_run, reuse_planner=reuse_planner)
    chem_trace_path = None
    if chem_enzy_expansion_trace_output_path is not None:
        chem_trace_path = _sharded_output_path(
            chem_enzy_expansion_trace_output_path,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        _write_chem_enzy_expansion_trace(chem_results, chem_trace_path)
    cascade_value_model = LearnedCascadeValueModel(cascade_value_model_path) if cascade_value_model_path else None
    cascade_transition_model = (
        LoadedCascadeTransitionValueModel(cascade_transition_model_path)
        if cascade_transition_model_path
        else None
    )
    cascade_action_value_model: Any | None = (
        LoadedCascadeActionValueModel(cascade_action_value_model_path)
        if cascade_action_value_model_path
        else None
    )
    if use_cascade_subgoal_action_scorer:
        cascade_action_value_model = SubgoalHintActionScorer(max_bonus=cascade_subgoal_action_max_bonus)
    cascade_pair_scorer = None
    if cascade_pair_scorer_path:
        cascade_pair_scorer = LearnedCascadePairScorer(cascade_pair_scorer_path)
    elif use_rule_pair_scorer:
        cascade_pair_scorer = RuleCascadePairScorer()
    route_block_value_final_reranker = (
        RouteBlockValueFinalReranker(route_block_value_final_reranker_path)
        if route_block_value_final_reranker_path
        else None
    )
    chem_by_target = {result.target_smiles: result for result in chem_results}
    trace_fh = None
    trace_path = None
    if trace_output_path is not None:
        trace_path = _sharded_output_path(trace_output_path, num_shards=num_shards, shard_index=shard_index)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = trace_path.open("w", encoding="utf-8")
    target_payloads = []
    try:
        for idx, row in enumerate(rows):
            target = str(row.get("target_smiles") or "")
            chem_result = chem_by_target.get(target) or BaselineRunResult(target_smiles=target, backend=BACKEND_NAME)
            target_payload, trace_rows = _run_one_target(
                row,
                chem_result,
                cascade_config=CascadeSearchConfig(
                    max_depth=cascade_max_depth,
                    branch_factor=cascade_branch_factor,
                    leaf_beam_size=cascade_leaf_beam_size,
                    diverse_leaf_reserve=cascade_diverse_leaf_reserve,
                    proposal_top_k=cascade_proposal_topk,
                    expansion_budget=cascade_expansion_budget,
                    min_step_score=cascade_min_step_score,
                    pair_reward_weight=cascade_pair_reward_weight,
                    pair_reward_mode=cascade_pair_reward_mode,
                    pair_reward_tie_epsilon=cascade_pair_reward_tie_epsilon,
                ),
                cascade_value_model=cascade_value_model,
                cascade_transition_model=cascade_transition_model,
                cascade_action_value_model=cascade_action_value_model,
                cascade_subgoal_provider_model_path=cascade_subgoal_provider_model_path,
                cascade_subgoal_program_manifest=cascade_subgoal_program_manifest,
                cascade_pair_scorer=cascade_pair_scorer,
                route_block_value_final_reranker=route_block_value_final_reranker,
                product_audit_final_reranker=(
                    ProductAuditFinalReranker() if use_product_audit_final_reranker else None
                ),
                vendor_root=vendor_root,
                collect_trace=trace_fh is not None,
                include_route_outcomes=include_route_outcomes,
                use_chem_enzy_expansion_proposals=use_chem_enzy_expansion_proposals,
                chem_enzy_expansion_proposal_topk_per_leaf=chem_enzy_expansion_proposal_topk_per_leaf,
                use_cascade_retrieval_proposals=use_cascade_retrieval_proposals,
                cascade_retrieval_program_manifest=cascade_retrieval_program_manifest,
                cascade_retrieval_mode=cascade_retrieval_mode,
                cascade_retrieval_min_similarity=cascade_retrieval_min_similarity,
                cascade_retrieval_require_downstream_transform=cascade_retrieval_require_downstream_transform,
                cascade_retrieval_topk=cascade_retrieval_topk,
                use_retrochimera_proposals=use_retrochimera_proposals,
                retrochimera_model_dir=retrochimera_model_dir,
                retrochimera_device=retrochimera_device,
                retrochimera_topk=retrochimera_topk,
                use_chemical_template_proposals=use_chemical_template_proposals,
                chemical_template_topk=chemical_template_topk,
                chemical_template_prefer_preselector=chemical_template_prefer_preselector,
                chemical_template_predict_conditions=enable_condition_prediction,
                condition_model=condition_model,
                use_template_relevance_proposals=use_template_relevance_proposals,
                template_relevance_models=template_relevance_models,
                template_relevance_topk=template_relevance_topk,
                cascade_result_limit=cascade_result_limit,
            )
            target_payloads.append(target_payload)
            if trace_fh is not None:
                if trace_rows:
                    for event in trace_rows:
                        trace_fh.write(
                            json.dumps(
                                {
                                    "schema_version": TRACE_SCHEMA_VERSION,
                                    "benchmark": str(benchmark_path),
                                    "benchmark_index": idx,
                                    "target_smiles": target_payload["target_smiles"],
                                    "doi": row.get("doi"),
                                    "cascade_id": row.get("cascade_id"),
                                    "route_domain": row.get("route_domain"),
                                    "gt_route": row.get("gt_route") or [],
                                    "planner_error": None,
                                    "elapsed_s": target_payload["cascade_search"]["elapsed_s"],
                                    "n_routes": target_payload["cascade_search"]["n_results"],
                                    "route_metrics": [],
                                    "route_model_active": bool(cascade_value_model is not None),
                                    "transition_model_active": bool(cascade_transition_model is not None),
                                    "action_value_model_active": bool(cascade_action_value_model is not None),
                                    "subgoal_action_scorer_active": bool(use_cascade_subgoal_action_scorer),
                                    "event": event,
                                    "outcome": target_payload["cascade_search"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                else:
                    trace_fh.write(
                        json.dumps(
                            {
                                "schema_version": TRACE_SCHEMA_VERSION,
                                "benchmark": str(benchmark_path),
                                "benchmark_index": idx,
                                "target_smiles": target_payload["target_smiles"],
                                "doi": row.get("doi"),
                                "cascade_id": row.get("cascade_id"),
                                "route_domain": row.get("route_domain"),
                                "gt_route": row.get("gt_route") or [],
                                "planner_error": None,
                                "elapsed_s": target_payload["cascade_search"]["elapsed_s"],
                                "n_routes": target_payload["cascade_search"]["n_results"],
                                "route_metrics": [],
                                "route_model_active": bool(cascade_value_model is not None),
                                "transition_model_active": bool(cascade_transition_model is not None),
                                    "action_value_model_active": bool(cascade_action_value_model is not None),
                                    "subgoal_action_scorer_active": bool(use_cascade_subgoal_action_scorer),
                                "event": None,
                                "outcome": target_payload["cascade_search"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if trace_fh is not None:
            trace_fh.close()

    payload = {
        "metadata": {
            "runner": "cascade_search_benchmark",
            "benchmark": str(benchmark_path),
            "proposal_backend": BACKEND_NAME,
            "vendor_root": str(vendor_root),
            "dry_run": dry_run,
            "n_requested": len(rows),
            "n_unsharded": unsharded_count,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "chem_enzy": {
                "stock_names": stock_names or DEFAULT_STOCKS,
                "one_step_models": one_step_models or DEFAULT_ONE_STEP_MODELS,
                "iterations": iterations,
                "max_depth": chem_enzy_max_depth,
                "expansion_topk": expansion_topk,
                "reuse_planner": reuse_planner,
                "condition_prediction": enable_condition_prediction,
                "enzyme_assignment": enable_enzyme_assignment,
                "condition_model": condition_model if enable_condition_prediction else None,
                "cascade_cost_model": {
                    "enabled": bool(base_chem_enzy_search_flags.get("use_cascade_cost_model")),
                    "context": base_chem_enzy_search_flags.get("cascade_search_context"),
                    "cost_model": base_chem_enzy_search_flags.get("cascade_cost_model"),
                    "context_from_row": chem_enzy_cascade_context_from_row,
                    "context_policy": chem_enzy_cascade_context_policy if chem_enzy_cascade_context_from_row else None,
                },
                "cascade_source_policy": {
                    "enabled": bool(base_chem_enzy_search_flags.get("use_cascade_source_policy")),
                    "policy": base_chem_enzy_search_flags.get("cascade_source_policy"),
                },
            },
            "cascade_search": {
                "max_depth": cascade_max_depth,
                "branch_factor": cascade_branch_factor,
                "proposal_top_k": cascade_proposal_topk,
                "expansion_budget": cascade_expansion_budget,
                "result_limit": cascade_result_limit,
                "min_step_score": cascade_min_step_score,
                "value_model": str(cascade_value_model_path) if cascade_value_model_path else "heuristic",
                "transition_value_model": (
                    str(cascade_transition_model_path) if cascade_transition_model_path else None
                ),
                "action_value_model": (
                    str(cascade_action_value_model_path) if cascade_action_value_model_path else None
                ),
                "subgoal_action_scorer": bool(use_cascade_subgoal_action_scorer),
                "subgoal_action_max_bonus": cascade_subgoal_action_max_bonus,
                "subgoal_provider_model": (
                    str(cascade_subgoal_provider_model_path) if cascade_subgoal_provider_model_path else None
                ),
                "subgoal_program_manifest": (
                    str(cascade_subgoal_program_manifest) if cascade_subgoal_program_manifest else None
                ),
                "pair_scorer": (
                    str(cascade_pair_scorer_path)
                    if cascade_pair_scorer_path
                    else ("rule" if use_rule_pair_scorer else None)
                ),
                "pair_reward_weight": cascade_pair_reward_weight,
                "pair_reward_mode": cascade_pair_reward_mode,
                "pair_reward_tie_epsilon": cascade_pair_reward_tie_epsilon,
                "route_block_value_final_reranker": (
                    str(route_block_value_final_reranker_path) if route_block_value_final_reranker_path else None
                ),
                "product_audit_final_reranker": bool(use_product_audit_final_reranker),
                "include_route_outcomes": include_route_outcomes,
                "use_chem_enzy_expansion_proposals": use_chem_enzy_expansion_proposals,
                "chem_enzy_expansion_proposal_topk_per_leaf": chem_enzy_expansion_proposal_topk_per_leaf,
                "use_cascade_retrieval_proposals": use_cascade_retrieval_proposals,
                "cascade_retrieval_program_manifest": (
                    str(cascade_retrieval_program_manifest) if cascade_retrieval_program_manifest else None
                ),
                "cascade_retrieval_mode": cascade_retrieval_mode,
                "cascade_retrieval_min_similarity": cascade_retrieval_min_similarity,
                "cascade_retrieval_require_downstream_transform": cascade_retrieval_require_downstream_transform,
                "cascade_retrieval_topk": cascade_retrieval_topk,
                "use_retrochimera_proposals": use_retrochimera_proposals,
                "retrochimera_model_dir": str(retrochimera_model_dir) if use_retrochimera_proposals else None,
                "retrochimera_device": retrochimera_device if use_retrochimera_proposals else None,
                "retrochimera_topk": retrochimera_topk if use_retrochimera_proposals else None,
                "use_chemical_template_proposals": use_chemical_template_proposals,
                "chemical_template_topk": chemical_template_topk if use_chemical_template_proposals else None,
                "chemical_template_prefer_preselector": (
                    chemical_template_prefer_preselector if use_chemical_template_proposals else None
                ),
                "chemical_template_condition_prediction": (
                    enable_condition_prediction if use_chemical_template_proposals else None
                ),
                "use_template_relevance_proposals": use_template_relevance_proposals,
                "template_relevance_models": list(template_relevance_models or []) if use_template_relevance_proposals else None,
                "template_relevance_topk": template_relevance_topk if use_template_relevance_proposals else None,
            },
            "trace_output": str(trace_path) if trace_path is not None else None,
            "chem_enzy_expansion_trace_output": str(chem_trace_path) if chem_trace_path is not None else None,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "summary": _summarize(target_payloads),
        "targets": target_payloads,
    }
    output_path = _sharded_output_path(output_path, num_shards=num_shards, shard_index=shard_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


class RouteBlockValueFinalReranker:
    """Runtime final reranker for already generated CascadeProgramSearch results."""

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
        if self.mean.shape[0] != len(self.feature_names) or self.std.shape[0] != len(self.feature_names):
            raise ValueError(f"route/block value scaler shape does not match feature schema: {model_pickle}")
        self.metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    def rerank(self, results: list[Any], *, search_elapsed_s: float | None = None) -> tuple[list[Any], list[dict[str, Any]]]:
        if not results:
            return results, []
        scored: list[tuple[float, float, int, Any, dict[str, Any]]] = []
        for native_rank, result in enumerate(results):
            row = self._row_from_result(result, native_rank=native_rank, search_elapsed_s=search_elapsed_s)
            vector = np.asarray([_nested_feature(row, name) for name in self.feature_names], dtype=np.float32)
            score = float(self.model.decision_function(((vector - self.mean) / self.std).reshape(1, -1))[0])
            diagnostics = {
                "original_rank": int(native_rank + 1),
                "route_block_value_score": round(score, 6),
                "feature_groups": row["feature_groups"],
                "model_pickle": self.model_pickle,
                "positive_task": self.metadata.get("positive_task"),
                "negative_task": self.metadata.get("negative_task"),
                "contract": "runtime final rerank of generated result programs; no expert labels",
            }
            scored.append((score, float(getattr(result, "score", 0.0) or 0.0), -native_rank, result, diagnostics))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        ordered = []
        diagnostics = []
        for new_index, (_score, _native_score, _tie, result, detail) in enumerate(scored, start=1):
            result.diagnostics.setdefault("route_block_value_final_rerank", {})
            result.diagnostics["route_block_value_final_rerank"] = {**detail, "new_rank": int(new_index)}
            ordered.append(result)
            diagnostics.append({**detail, "new_rank": int(new_index)})
        return ordered, diagnostics

    def _row_from_result(self, result: Any, *, native_rank: int, search_elapsed_s: float | None) -> dict[str, Any]:
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
                "native_score": _mean([value for value in scores if value is not None], default=_float(getattr(result, "score", 0.0))),
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


class ProductAuditFinalReranker:
    """No-label conservative final reranker based on product-audit route sanity."""

    contract = (
        "runtime final rerank of generated result programs; product-audit weak supervision only; "
        "no GT labels and no expert labels"
    )

    def rerank(self, results: list[Any], *, target_smiles: str) -> tuple[list[Any], list[dict[str, Any]]]:
        if not results:
            return results, []
        product_props = _product_mol_props(target_smiles)
        scored: list[tuple[tuple[int, int, int], float, int, Any, dict[str, Any]]] = []
        for native_rank, result in enumerate(results):
            audit = _audit_product_route(
                _product_audit_route_from_result(result),
                rank=native_rank + 1,
                product_family="general_product",
                product_props=product_props,
            )
            issues = list(audit.get("issues") or [])
            tags = list(audit.get("tags") or [])
            severe = int(product_audit_risk_order(audit) >= 40)
            # Conservative selector: only hard-demote severe artifacts, then use a
            # late-stage/product-audit hint as a small no-label tie-breaker.
            key = (
                severe,
                -int("late_stage_derivatization" in tags),
                native_rank,
            )
            diagnostics = {
                "original_rank": int(native_rank + 1),
                "product_audit_key": list(key),
                "route_class": audit.get("route_class"),
                "issues": issues,
                "tags": tags,
                "risk_order": int(product_audit_risk_order(audit)),
                "route_class_order": int(ROUTE_CLASS_ORDER.get(str(audit.get("route_class")), 99)),
                "contract": self.contract,
            }
            scored.append((key, -float(getattr(result, "score", 0.0) or 0.0), native_rank, result, diagnostics))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        ordered = []
        diagnostics = []
        for new_index, (_key, _score_tie, _native_rank, result, detail) in enumerate(scored, start=1):
            result.diagnostics.setdefault("product_audit_final_rerank", {})
            result.diagnostics["product_audit_final_rerank"] = {**detail, "new_rank": int(new_index)}
            ordered.append(result)
            diagnostics.append({**detail, "new_rank": int(new_index)})
        return ordered, diagnostics


def _product_audit_route_from_result(result: Any) -> dict[str, Any]:
    state = getattr(result, "state", None)
    steps = list(getattr(state, "step_annotations", []) or []) if state is not None else []
    converted_steps = []
    terminal_reactants = []
    stock_map: dict[str, bool | None] = {}
    for idx, step in enumerate(steps):
        reactants = [str(smi) for smi in getattr(step, "reactant_smiles", []) or [] if smi]
        terminal_reactants.extend(reactants)
        stock_map.update(dict(getattr(step, "stock_status", {}) or {}))
        converted_steps.append(
            {
                "index": idx,
                "product": getattr(step, "product_smiles", ""),
                "main_reactant": reactants[0] if reactants else "",
                "aux_reactants": reactants[1:],
                "reaction_smiles": getattr(step, "rxn_smiles", ""),
                "reaction_type": getattr(step, "reaction_type", "") or "unknown",
                "source": getattr(step, "source_model", "") or "runtime",
                "scores": {"retro": getattr(step, "score", None), "confidence": getattr(step, "score", None)},
                "stock_status": dict(getattr(step, "stock_status", {}) or {}),
                "reaction_interpretation": {
                    "reaction_class": getattr(step, "reaction_type", "") or "unknown",
                },
            }
        )
    terminal_reactants = list(dict.fromkeys(terminal_reactants))
    stock_closed = bool(getattr(state, "stock_closed", False)) if state is not None else bool(getattr(result, "solved", False))
    return {
        "score": getattr(result, "score", None),
        "steps": converted_steps,
        "metrics": {
            "strict_stock_solve": stock_closed,
            "route_solved": bool(getattr(result, "solved", False)),
            "filled_route": bool(converted_steps),
            "terminal_reactants": terminal_reactants,
            "route_naturalness": {},
        },
        "quality_vector": {"stock_closed": float(stock_closed), "route_solved": float(bool(getattr(result, "solved", False)))},
    }


def merge_cascade_search_outputs(input_paths: list[Path], output_path: Path) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    targets = []
    for payload in payloads:
        targets.extend(payload.get("targets") or [])
    targets.sort(key=lambda row: str(row.get("target_smiles") or ""))
    merged = {
        "metadata": {
            "runner": "cascade_search_benchmark_merge",
            "inputs": [str(path) for path in input_paths],
            "n_inputs": len(input_paths),
            "n_targets": len(targets),
        },
        "summary": _summarize(targets),
        "targets": targets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return merged


def merge_cascade_search_trace_outputs(input_paths: list[Path], output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in input_paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(
        key=lambda row: (
            str(row.get("target_smiles") or ""),
            str(row.get("state_id") or (row.get("event") or {}).get("state_id") or ""),
            int((row.get("event") or {}).get("depth") or 0),
            str((row.get("event") or {}).get("expanded_leaf") or ""),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "metadata": {
            "runner": "cascade_search_trace_merge",
            "inputs": [str(path) for path in input_paths],
            "n_inputs": len(input_paths),
            "n_rows": len(rows),
        },
        "schema_version": TRACE_SCHEMA_VERSION,
        "output_path": str(output_path),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _write_chem_enzy_expansion_trace(results: list[BaselineRunResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for result in results:
            trace = (result.raw_backend_metadata.get("cascade_expansion_trace") or {}).get("rows") or []
            for row in trace:
                payload = {
                    "schema_version": "chem_enzy_cascade_expansion_trace.v1",
                    "target_smiles": result.target_smiles,
                    "backend": result.backend,
                    **row,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sharded_output_path(path: Path, *, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return Path(path)
    path = Path(path)
    suffix = "".join(path.suffixes) if path.suffixes else path.suffix
    stem = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}_shard{shard_index}of{num_shards}{suffix}")


def _run_one_target(
    row: dict[str, Any],
    chem_result: BaselineRunResult,
    *,
    cascade_config: CascadeSearchConfig,
    cascade_value_model: Any | None = None,
    cascade_transition_model: Any | None = None,
    cascade_action_value_model: Any | None = None,
    cascade_subgoal_provider_model_path: Path | None = None,
    cascade_subgoal_program_manifest: Path | None = None,
    cascade_pair_scorer: Any | None = None,
    route_block_value_final_reranker: Any | None = None,
    product_audit_final_reranker: Any | None = None,
    vendor_root: Path = Path("vendor/ChemEnzyRetroPlanner"),
    collect_trace: bool = False,
    include_route_outcomes: bool = False,
    use_chem_enzy_expansion_proposals: bool = False,
    chem_enzy_expansion_proposal_topk_per_leaf: int | None = 50,
    use_cascade_retrieval_proposals: bool = False,
    cascade_retrieval_program_manifest: Path | None = None,
    cascade_retrieval_mode: str = "block_downstream_product",
    cascade_retrieval_min_similarity: float = 0.20,
    cascade_retrieval_require_downstream_transform: bool = False,
    cascade_retrieval_topk: int = 8,
    use_retrochimera_proposals: bool = False,
    retrochimera_model_dir: Path = Path("data_external/retrochimera_model"),
    retrochimera_device: str | None = None,
    retrochimera_topk: int = 8,
    use_chemical_template_proposals: bool = False,
    chemical_template_topk: int = 12,
    chemical_template_prefer_preselector: bool = True,
    chemical_template_predict_conditions: bool = False,
    condition_model: str = "rcr",
    use_template_relevance_proposals: bool = False,
    template_relevance_models: list[str] | None = None,
    template_relevance_topk: int = 8,
    cascade_result_limit: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = str(row.get("target_smiles") or chem_result.target_smiles or "")
    route_proposals_by_leaf = _proposal_cache_from_chem_enzy(chem_result)
    proposals_by_leaf = {
        leaf: list(actions)
        for leaf, actions in route_proposals_by_leaf.items()
    }
    expansion_proposals_by_leaf: dict[str, list[Any]] = {}
    expansion_proposal_count = 0
    if use_chem_enzy_expansion_proposals:
        expansion_proposals_by_leaf = _expansion_proposal_cache_from_chem_enzy(
            chem_result,
            topk_per_leaf=chem_enzy_expansion_proposal_topk_per_leaf,
        )
        expansion_proposal_count = sum(len(actions) for actions in expansion_proposals_by_leaf.values())
        proposals_by_leaf = _merge_proposal_caches(proposals_by_leaf, expansion_proposals_by_leaf)
        proposals_by_leaf = _sort_proposal_cache(proposals_by_leaf)
    stock_map = _stock_map(chem_result, row)
    stock_checker = _stock_checker(stock_map)
    provider = StaticProposalProvider(proposals_by_leaf)
    providers: list[Any] = [provider]
    if cascade_subgoal_provider_model_path is not None and cascade_subgoal_program_manifest is not None:
        providers.append(
            CascadeSubgoalEvidenceProvider(
                cascade_subgoal_program_manifest,
                cascade_subgoal_provider_model_path,
            )
        )
    if use_cascade_retrieval_proposals and cascade_retrieval_program_manifest is not None:
        providers.append(
            CascadeRetrievalProposalProvider(
                cascade_retrieval_program_manifest,
                mode=cascade_retrieval_mode,
                min_similarity=cascade_retrieval_min_similarity,
                require_downstream_transform_context=cascade_retrieval_require_downstream_transform,
            )
        )
    if use_retrochimera_proposals:
        providers.append(
            TopKProvider(
                RetroChimeraProposalProvider(
                    model_dir=retrochimera_model_dir,
                    device=retrochimera_device,
                ),
                top_k=retrochimera_topk,
            )
        )
    if use_chemical_template_proposals:
        providers.append(
            TopKProvider(
                ChemicalTemplateProposalProvider(
                    vendor_root=vendor_root,
                    prefer_preselector=chemical_template_prefer_preselector,
                    expansion_topk=chemical_template_topk,
                    predict_conditions=chemical_template_predict_conditions,
                    condition_model=condition_model,
                ),
                top_k=chemical_template_topk,
            )
        )
    if use_template_relevance_proposals:
        providers.append(
            TopKProvider(
                TemplateRelevanceProposalProvider(
                    vendor_root=vendor_root,
                    models=tuple(template_relevance_models or ["template_relevance.reaxys_biocatalysis"]),
                    expansion_topk=template_relevance_topk,
                ),
                top_k=template_relevance_topk,
            )
        )
    started = time.monotonic()
    trace_collector = CascadeTraceCollector() if collect_trace else None
    controller = CascadeSearchController(
        value_model=cascade_value_model or HeuristicCascadeValueModel(),
        transition_value_model=cascade_transition_model,
        action_value_model=cascade_action_value_model,
        pair_scorer=cascade_pair_scorer,
    )
    planner = CascadeProgramSearch(
        providers,
        stock_checker=stock_checker,
        config=cascade_config,
        controller=controller,
        trace_collector=trace_collector,
    )
    results = planner.search(target, n_results=max(1, int(cascade_result_limit or 1)))
    elapsed_s = time.monotonic() - started
    route_block_value_rerank_diagnostics = []
    if route_block_value_final_reranker is not None and results:
        results, route_block_value_rerank_diagnostics = route_block_value_final_reranker.rerank(
            results,
            search_elapsed_s=elapsed_s,
        )
        for result, detail in zip(results, route_block_value_rerank_diagnostics):
            if isinstance(getattr(result, "diagnostics", None), dict):
                result.diagnostics.setdefault("route_block_value_final_rerank", detail)
    product_audit_rerank_diagnostics = []
    if product_audit_final_reranker is not None and results:
        results, product_audit_rerank_diagnostics = product_audit_final_reranker.rerank(
            results,
            target_smiles=target,
        )
        for result, detail in zip(results, product_audit_rerank_diagnostics):
            if isinstance(getattr(result, "diagnostics", None), dict):
                result.diagnostics.setdefault("product_audit_final_rerank", detail)
    best = results[0] if results else None
    best_state = best.state if best is not None else None
    best_failures = best.failures if best is not None else []
    route_rxns = {
        canonical_reaction(step.rxn_smiles) or step.rxn_smiles
        for step in (best_state.step_annotations if best_state is not None else [])
        if step.rxn_smiles
    }
    route_reactants = {
        canonical_smiles(reactant) or reactant
        for step in (best_state.step_annotations if best_state is not None else [])
        for reactant in step.reactant_smiles
        if reactant
    }
    for step in best_state.step_annotations if best_state is not None else []:
        route_reactants.update(reaction_reactants(step.rxn_smiles))
    proposal_rxns, proposal_reactants = _proposal_pool_keys(proposals_by_leaf)
    gt_rxns = _gt_reactions(row)
    gt_reactant_set = gt_reactants(row)
    proposal_exact_hits = proposal_rxns & gt_rxns
    proposal_reactant_hits = proposal_reactants & gt_reactant_set
    route_exact_hits = route_rxns & gt_rxns
    route_reactant_hits = route_reactants & gt_reactant_set
    cost = score_cascade_state(best_state).to_dict() if best_state is not None else None
    result_programs = (
        _cascade_result_programs(results, gt_rxns=gt_rxns, gt_reactants_set=gt_reactant_set)
        if include_route_outcomes
        else None
    )
    result_exact_hit_count = 0
    result_reactant_hit_count = 0
    if result_programs is not None:
        result_exact_hit_count = sum(
            int(bool(program.get("exact_reaction_hit_count"))) for program in result_programs
        )
        result_reactant_hit_count = sum(
            int(bool(program.get("gt_reactant_hit_count"))) for program in result_programs
        )
    payload = {
        "target_smiles": target,
        "route_domain": row.get("route_domain"),
        "gap_bucket": row.get("gap_bucket"),
        "depth": row.get("depth"),
        "gt_route": row.get("gt_route") or [],
        "chem_enzy": {
            "solved": chem_result.solved,
            "route_count": chem_result.route_count,
            "failures": [failure.to_dict() for failure in chem_result.failures],
            "proposal_leaf_count": len(proposals_by_leaf),
            "proposal_step_count": sum(len(values) for values in proposals_by_leaf.values()),
            "route_pool_leaf_count": len(route_proposals_by_leaf),
            "route_pool_step_count": sum(len(values) for values in route_proposals_by_leaf.values()),
            "expansion_proposal_leaf_count": len(expansion_proposals_by_leaf),
            "expansion_proposal_step_count": expansion_proposal_count,
            "cascade_retrieval_proposals_enabled": bool(use_cascade_retrieval_proposals),
            "cascade_retrieval_program_manifest": (
                str(cascade_retrieval_program_manifest) if cascade_retrieval_program_manifest else None
            ),
            "cascade_cost_summary": _chem_enzy_cascade_cost_summary(chem_result),
            "cascade_expansion_trace_count": (
                chem_result.raw_backend_metadata.get("cascade_expansion_trace") or {}
            ).get("count", 0),
        },
        "cascade_search": {
            "solved": bool(best and best.solved),
            "n_results": len(results),
            "score": best.score if best is not None else None,
            "elapsed_s": round(elapsed_s, 3),
            "stock_closed": bool(best_state and best_state.stock_closed),
            "cofactor_closed": bool(best_state and not best_state.cofactor_ledger.unclosed_requirements()),
            "condition_conflict_free": not any(failure.category == "ConditionConflict" for failure in best_failures),
            "enzyme_evidence_sufficient": not any(failure.category == "EnzymeEvidenceWeak" for failure in best_failures),
            "stage_count": best_state.stage_graph.n_stages if best_state is not None else 0,
            "step_count": len(best_state.step_annotations) if best_state is not None else 0,
            "failure_categories": [failure.category for failure in best_failures],
            "cost": cost,
            "stats": planner.stats.to_dict(),
            "route_rxns": sorted(route_rxns),
        },
        "recovery": {
            "gt_step_count": len(gt_rxns),
            "exact_gt_route_recovered": bool(gt_rxns and gt_rxns.issubset(route_rxns)),
            "partial_gt_step_overlap": bool(gt_rxns and route_rxns & gt_rxns),
            "gt_step_overlap_count": len(route_rxns & gt_rxns),
            "gt_step_overlap_fraction": round(len(route_rxns & gt_rxns) / len(gt_rxns), 4) if gt_rxns else None,
            "exact_reaction_in_route_pool": bool(route_exact_hits),
            "exact_reaction_hit_count": len(route_exact_hits),
            "gt_reactant_in_route_pool": bool(route_reactant_hits),
            "gt_reactant_hit_count": len(route_reactant_hits),
            "candidate_exact_reaction_in_pool": bool(proposal_exact_hits),
            "candidate_exact_reaction_hit_count": len(proposal_exact_hits),
            "candidate_gt_reactant_in_pool": bool(proposal_reactant_hits),
            "candidate_gt_reactant_hit_count": len(proposal_reactant_hits),
            "result_exact_reaction_in_pool": bool(result_exact_hit_count),
            "result_exact_reaction_hit_count": result_exact_hit_count,
            "result_gt_reactant_in_pool": bool(result_reactant_hit_count),
            "result_gt_reactant_hit_count": result_reactant_hit_count,
            "proposal_pool_reaction_count": len(proposal_rxns),
            "proposal_pool_reactant_count": len(proposal_reactants),
        },
    }
    if result_programs is not None:
        payload["cascade_search"]["result_programs"] = result_programs
    if route_block_value_rerank_diagnostics:
        payload["cascade_search"]["route_block_value_final_rerank"] = {
            "enabled": True,
            "changed_top_route": bool(
                route_block_value_rerank_diagnostics
                and int(route_block_value_rerank_diagnostics[0].get("original_rank") or 0) != 1
            ),
            "model_pickle": route_block_value_rerank_diagnostics[0].get("model_pickle"),
            "scores": route_block_value_rerank_diagnostics,
        }
    if product_audit_rerank_diagnostics:
        payload["cascade_search"]["product_audit_final_rerank"] = {
            "enabled": True,
            "changed_top_route": bool(
                product_audit_rerank_diagnostics
                and int(product_audit_rerank_diagnostics[0].get("original_rank") or 0) != 1
            ),
            "contract": ProductAuditFinalReranker.contract,
            "scores": product_audit_rerank_diagnostics,
        }
    trace_rows = trace_collector.to_rows() if trace_collector is not None else []
    return payload, trace_rows


def _cascade_result_programs(
    results: list[Any],
    *,
    gt_rxns: set[str],
    gt_reactants_set: set[str],
) -> list[dict[str, Any]]:
    programs = []
    for rank, result in enumerate(results, start=1):
        state = result.state
        route_rxns = {
            canonical_reaction(step.rxn_smiles) or step.rxn_smiles
            for step in (state.step_annotations if state is not None else [])
            if step.rxn_smiles
        }
        route_reactants = {
            canonical_smiles(reactant) or reactant
            for step in (state.step_annotations if state is not None else [])
            for reactant in step.reactant_smiles
            if reactant
        }
        for step in state.step_annotations if state is not None else []:
            route_reactants.update(reaction_reactants(step.rxn_smiles))
        exact_hits = route_rxns & gt_rxns
        reactant_hits = route_reactants & gt_reactants_set
        try:
            cost = score_cascade_state(state).to_dict() if state is not None else None
        except Exception:
            cost = None
        programs.append(
            {
                "rank": rank,
                "original_rank": int((result.diagnostics.get("route_block_value_final_rerank") or {}).get("original_rank") or rank)
                if isinstance(getattr(result, "diagnostics", None), dict)
                else rank,
                "solved": bool(result.solved),
                "score": result.score,
                "route_block_value_score": (
                    (result.diagnostics.get("route_block_value_final_rerank") or {}).get("route_block_value_score")
                    if isinstance(getattr(result, "diagnostics", None), dict)
                    else None
                ),
                "product_audit_original_rank": int(
                    (result.diagnostics.get("product_audit_final_rerank") or {}).get("original_rank") or rank
                )
                if isinstance(getattr(result, "diagnostics", None), dict)
                else rank,
                "product_audit_route_class": (
                    (result.diagnostics.get("product_audit_final_rerank") or {}).get("route_class")
                    if isinstance(getattr(result, "diagnostics", None), dict)
                    else None
                ),
                "product_audit_issues": (
                    (result.diagnostics.get("product_audit_final_rerank") or {}).get("issues")
                    if isinstance(getattr(result, "diagnostics", None), dict)
                    else None
                ),
                "product_audit_tags": (
                    (result.diagnostics.get("product_audit_final_rerank") or {}).get("tags")
                    if isinstance(getattr(result, "diagnostics", None), dict)
                    else None
                ),
                "route_rxns": sorted(route_rxns),
                "route_reactants": sorted(route_reactants),
                "exact_reaction_hit_count": len(exact_hits),
                "gt_reactant_hit_count": len(reactant_hits),
                "exact_gt_route_recovered": bool(gt_rxns and gt_rxns.issubset(route_rxns)),
                "partial_gt_step_overlap": bool(gt_rxns and exact_hits),
                "gt_reactant_in_route": bool(reactant_hits),
                "route_outcome_value": _route_outcome_value(
                    solved=bool(result.solved),
                    exact_gt_route=bool(gt_rxns and gt_rxns.issubset(route_rxns)),
                    exact_hit=bool(exact_hits),
                    reactant_hit=bool(reactant_hits),
                    rank=rank,
                ),
                "cost": cost,
                "failure_categories": [failure.category for failure in result.failures],
            }
        )
    return programs


def _route_outcome_value(
    *,
    solved: bool,
    exact_gt_route: bool,
    exact_hit: bool,
    reactant_hit: bool,
    rank: int,
) -> float:
    if exact_gt_route:
        value = 1.0
    elif exact_hit:
        value = 0.8
    elif reactant_hit:
        value = 0.45
    elif solved:
        value = 0.10
    else:
        value = 0.0
    rank_decay = 1.0 / max(1.0, float(rank))
    return round(value * (0.75 + 0.25 * rank_decay), 6)


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
        for key in ("cascade_pair_summary", "cascade_action_value_summary"):
            summary = getattr(state, "raw_metadata", {}).get(key) if isinstance(getattr(state, "raw_metadata", {}), dict) else {}
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


def _proposal_cache_from_chem_enzy(result: BaselineRunResult) -> dict[str, list[Any]]:
    proposals: dict[str, list[Any]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for route in result.routes:
        for step in route.steps:
            if not step.product_smiles or not step.rxn_smiles:
                continue
            key = (step.product_smiles, canonical_reaction(step.rxn_smiles) or step.rxn_smiles)
            if key in seen:
                continue
            seen.add(key)
            proposals[step.product_smiles].append(route_step_candidate_to_action(step, provider_name=BACKEND_NAME))
    return dict(proposals)


def _expansion_proposal_cache_from_chem_enzy(
    result: BaselineRunResult,
    *,
    topk_per_leaf: int | None = 50,
) -> dict[str, list[Any]]:
    trace = (result.raw_backend_metadata.get("cascade_expansion_trace") or {}).get("rows") or []
    proposals: dict[str, list[Any]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in trace:
        action = _expansion_trace_row_to_action(row)
        if action is None or action.step is None:
            continue
        key = (action.target_leaf, canonical_reaction(action.step.rxn_smiles) or action.step.rxn_smiles)
        if key in seen:
            continue
        seen.add(key)
        proposals[action.target_leaf].append(action)
    sorted_proposals = _sort_proposal_cache(dict(proposals))
    if topk_per_leaf is None or int(topk_per_leaf) <= 0:
        return sorted_proposals
    limit = int(topk_per_leaf)
    return {leaf: actions[:limit] for leaf, actions in sorted_proposals.items()}


def _merge_proposal_caches(*caches: dict[str, list[Any]]) -> dict[str, list[Any]]:
    merged: dict[str, list[Any]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for cache in caches:
        for leaf, actions in cache.items():
            for action in actions:
                step = getattr(action, "step", None)
                rxn_smiles = getattr(step, "rxn_smiles", "") if step is not None else ""
                key = (str(leaf), canonical_reaction(rxn_smiles) or rxn_smiles or json.dumps(str(action)))
                if key in seen:
                    continue
                seen.add(key)
                merged[str(leaf)].append(action)
    return dict(merged)


def _sort_proposal_cache(cache: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {
        leaf: sorted(actions, key=_proposal_sort_key)
        for leaf, actions in cache.items()
    }


def _proposal_sort_key(action: Any) -> tuple[float, float, int]:
    step = getattr(action, "step", None)
    raw = getattr(step, "raw_metadata", {}) if step is not None else {}
    cascade_cost = raw.get("cascade_cost") if isinstance(raw, dict) else {}
    total_cost = _safe_sort_float((cascade_cost or {}).get("total_cost"), default=1e9)
    score = _safe_sort_float(getattr(step, "score", None), default=0.0) if step is not None else 0.0
    candidate_index = _safe_sort_int((cascade_cost or {}).get("candidate_index"), default=10**9)
    return (total_cost, -score, candidate_index)


class TopKProvider:
    """Cap a runtime proposal provider without changing global search top-k."""

    def __init__(self, inner: Any, *, top_k: int):
        self.inner = inner
        self.top_k = max(1, int(top_k or 1))
        self.provider_name = str(getattr(inner, "provider_name", type(inner).__name__))
        self.last_diagnostics = getattr(inner, "last_diagnostics", None)

    def propose(self, request: Any, *args: Any, **kwargs: Any) -> list[Any]:
        if hasattr(request, "top_k"):
            request = dataclass_replace(request, top_k=min(int(getattr(request, "top_k") or self.top_k), self.top_k))
        else:
            kwargs["top_k"] = min(int(kwargs.get("top_k") or self.top_k), self.top_k)
        rows = list(self.inner.propose(request, *args, **kwargs) or [])
        self.last_diagnostics = getattr(self.inner, "last_diagnostics", None)
        return rows


def dataclass_replace(value: Any, **kwargs: Any) -> Any:
    from dataclasses import replace

    return replace(value, **kwargs)


def _safe_sort_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_sort_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _expansion_trace_row_to_action(row: dict[str, Any]) -> Any | None:
    parent = str(row.get("parent_mol") or "")
    reactants = [str(smi) for smi in row.get("reactants") or [] if smi]
    if not parent or not reactants:
        return None
    rxn_smiles = ".".join(reactants) + ">>" + parent
    base_score = row.get("base_score")
    score = None
    try:
        score = float(base_score)
    except (TypeError, ValueError):
        pass
    source_model = str(row.get("source_model") or BACKEND_NAME)
    raw_metadata = {
        "source": "chem_enzy_expansion_trace",
        "parent_depth": row.get("parent_depth"),
        "candidate_index": row.get("candidate_index"),
        "template": row.get("template"),
        "cascade_cost": {
            "parent_mol": parent,
            "parent_depth": row.get("parent_depth"),
            "candidate_index": row.get("candidate_index"),
            "source_model": source_model,
            "reaction_domain": row.get("reaction_domain"),
            "base_score": row.get("base_score"),
            "base_cost": row.get("base_cost"),
            "cascade_adjustment": row.get("cascade_adjustment"),
            "total_cost": row.get("total_cost"),
            "components": row.get("components") or {},
            "context_features": row.get("context_features") or {},
            "source_policy_decision": row.get("source_policy_decision"),
            "action_value_score": row.get("action_value_score"),
            "active_failure_modes": row.get("active_failure_modes") or [],
        },
    }
    step = StepAnnotation(
        product_smiles=parent,
        reactant_smiles=reactants,
        rxn_smiles=rxn_smiles,
        source_model=source_model,
        score=score,
        reaction_type=str(row.get("reaction_domain") or ""),
        raw_metadata=raw_metadata,
    )
    return CascadeAction(
        CascadeActionType.RETROSYNTHETIC_STEP,
        target_leaf=parent,
        step=step,
        source=BACKEND_NAME,
    )


def _proposal_pool_keys(proposals_by_leaf: dict[str, list[Any]]) -> tuple[set[str], set[str]]:
    rxns: set[str] = set()
    reactants: set[str] = set()
    for actions in proposals_by_leaf.values():
        for action in actions:
            step = getattr(action, "step", None)
            if step is None and isinstance(action, dict):
                step = action
            rxn_smiles = ""
            step_reactants: list[str] = []
            if isinstance(step, dict):
                rxn_smiles = str(step.get("rxn_smiles") or step.get("reaction_smiles") or "")
                step_reactants = [str(smi) for smi in step.get("reactant_smiles") or step.get("reactants") or []]
            elif step is not None:
                rxn_smiles = str(getattr(step, "rxn_smiles", "") or "")
                step_reactants = [str(smi) for smi in getattr(step, "reactant_smiles", []) or []]
            if rxn_smiles:
                key = canonical_reaction(rxn_smiles) or rxn_smiles
                if key:
                    rxns.add(key)
                reactants.update(reaction_reactants(rxn_smiles))
            for smi in step_reactants:
                key = canonical_smiles(smi) or smi
                if key:
                    reactants.add(key)
    return rxns, reactants


def _chem_enzy_cascade_cost_summary(result: BaselineRunResult) -> dict[str, Any]:
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    adjustments = []
    total_costs = []
    annotated_steps = 0
    for route in result.routes:
        for step in route.steps:
            cascade_cost = step.raw_backend_metadata.get("cascade_cost") if step.raw_backend_metadata else None
            if not isinstance(cascade_cost, dict):
                continue
            annotated_steps += 1
            domain = str(cascade_cost.get("reaction_domain") or "unknown")
            source = str(cascade_cost.get("source_model") or step.source_model or "unknown")
            domain_counts[domain] += 1
            source_counts[source] += 1
            if cascade_cost.get("cascade_adjustment") is not None:
                adjustments.append(float(cascade_cost["cascade_adjustment"]))
            if cascade_cost.get("total_cost") is not None:
                total_costs.append(float(cascade_cost["total_cost"]))
    return {
        "annotated_steps": annotated_steps,
        "domain_counts": dict(domain_counts),
        "source_counts": dict(source_counts),
        "avg_adjustment": round(sum(adjustments) / len(adjustments), 4) if adjustments else None,
        "avg_total_cost": round(sum(total_costs) / len(total_costs), 4) if total_costs else None,
    }


def _stock_map(result: BaselineRunResult, row: dict[str, Any]) -> dict[str, bool]:
    stock: dict[str, bool] = {}
    for route in result.routes:
        stock.update({str(k): bool(v) for k, v in route.stock_status.items() if v is not None})
        for step in route.steps:
            stock.update({str(k): bool(v) for k, v in step.stock_status.items() if v is not None})
    for item in row.get("starting_materials") or []:
        smi = str(item.get("smiles") or item.get("smiles_string") or "")
        if smi:
            stock[smi] = True
    return stock


def _stock_checker(stock_map: dict[str, bool]) -> Callable[[str], bool]:
    canonical = {canonical_smiles(key) or key: bool(value) for key, value in stock_map.items()}

    def check(smiles: str) -> bool:
        return bool(stock_map.get(smiles) or canonical.get(canonical_smiles(smiles) or smiles))

    return check


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    failure_counts = Counter()
    chem_failures = Counter()
    for row in rows:
        failure_counts.update(row["cascade_search"].get("failure_categories") or [])
        for failure in row["chem_enzy"].get("failures") or []:
            category = failure.get("category")
            if category:
                chem_failures[category] += 1

    def rate(path: tuple[str, ...]) -> float | None:
        if not n:
            return None
        total = 0
        for row in rows:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            total += int(bool(value))
        return total / n

    def top_result_rate(program_field: str, fallback_path: tuple[str, ...]) -> float | None:
        if not n:
            return None
        total = 0
        for row in rows:
            programs = (row.get("cascade_search") or {}).get("result_programs") or []
            if programs:
                total += int(bool((programs[0] or {}).get(program_field)))
                continue
            value: Any = row
            for key in fallback_path:
                value = value.get(key) if isinstance(value, dict) else None
            total += int(bool(value))
        return total / n

    elapsed = [float(row["cascade_search"].get("elapsed_s") or 0.0) for row in rows]
    stages = [float(row["cascade_search"].get("stage_count") or 0.0) for row in rows if row["cascade_search"].get("stage_count")]
    return {
        "n_targets": n,
        "chem_enzy_solved_rate": rate(("chem_enzy", "solved")),
        "cascade_solved_rate": rate(("cascade_search", "solved")),
        "stock_closed_rate": rate(("cascade_search", "stock_closed")),
        "cofactor_closed_rate": rate(("cascade_search", "cofactor_closed")),
        "condition_conflict_free_rate": rate(("cascade_search", "condition_conflict_free")),
        "enzyme_evidence_sufficient_rate": rate(("cascade_search", "enzyme_evidence_sufficient")),
        "exact_gt_route_recovered_rate": rate(("recovery", "exact_gt_route_recovered")),
        "partial_gt_step_overlap_rate": rate(("recovery", "partial_gt_step_overlap")),
        "exact_reaction_in_route_pool": rate(("recovery", "exact_reaction_in_route_pool")),
        "gt_reactant_in_route_pool": rate(("recovery", "gt_reactant_in_route_pool")),
        "candidate_exact_reaction_in_pool": rate(("recovery", "candidate_exact_reaction_in_pool")),
        "candidate_gt_reactant_in_pool": rate(("recovery", "candidate_gt_reactant_in_pool")),
        "top_result_exact_reaction_in_pool": top_result_rate(
            "exact_reaction_hit_count",
            ("recovery", "exact_reaction_in_route_pool"),
        ),
        "top_result_gt_reactant_in_pool": top_result_rate(
            "gt_reactant_hit_count",
            ("recovery", "gt_reactant_in_route_pool"),
        ),
        "result_exact_reaction_in_pool": rate(("recovery", "result_exact_reaction_in_pool")),
        "result_gt_reactant_in_pool": rate(("recovery", "result_gt_reactant_in_pool")),
        "avg_gt_step_overlap_fraction": _avg(
            row["recovery"].get("gt_step_overlap_fraction")
            for row in rows
            if row["recovery"].get("gt_step_overlap_fraction") is not None
        ),
        "avg_cascade_search_time_s": _avg(elapsed),
        "avg_stage_count": _avg(stages),
        "cascade_failure_counts": dict(failure_counts),
        "chem_enzy_failure_counts": dict(chem_failures),
    }


def _avg(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _mean(values: Any, *, default: float = 0.0) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return float(default)
    return float(sum(clean) / len(clean))


def _float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gt_reactions(row: dict[str, Any]) -> set[str]:
    out = set()
    for step in row.get("gt_route") or []:
        rxn = step.get("rxn_smiles")
        if rxn:
            out.add(canonical_reaction(rxn) or rxn)
    return out


def _read_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    rows = [row for row in data if isinstance(row, dict) and row.get("target_smiles")]
    if not rows:
        raise ValueError(f"no targets in benchmark: {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ChemEnzy-backed CascadeProgramSearch benchmark")
    ap.add_argument("--merge", nargs="*", default=None, help="Merge shard JSON outputs instead of running targets")
    ap.add_argument("--merge-traces", nargs="*", default=None, help="Merge shard trace JSONL outputs instead of running targets")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace-output", default=None)
    ap.add_argument("--chem-enzy-expansion-trace-output", default=None)
    ap.add_argument("--use-chem-enzy-expansion-proposals", action="store_true")
    ap.add_argument("--chem-enzy-expansion-proposal-topk-per-leaf", type=int, default=50)
    ap.add_argument("--use-cascade-retrieval-proposals", action="store_true")
    ap.add_argument("--cascade-retrieval-program-manifest", default=None)
    ap.add_argument("--cascade-retrieval-mode", default="block_downstream_product")
    ap.add_argument("--cascade-retrieval-min-similarity", type=float, default=0.20)
    ap.add_argument("--cascade-retrieval-require-downstream-transform", action="store_true")
    ap.add_argument("--cascade-retrieval-topk", type=int, default=8)
    ap.add_argument("--use-retrochimera-proposals", action="store_true")
    ap.add_argument("--retrochimera-model-dir", default="data_external/retrochimera_model")
    ap.add_argument("--retrochimera-device", default=None)
    ap.add_argument("--retrochimera-topk", type=int, default=8)
    ap.add_argument("--use-chemical-template-proposals", action="store_true")
    ap.add_argument("--chemical-template-topk", type=int, default=12)
    ap.add_argument(
        "--chemical-template-no-preselector",
        action="store_true",
        help="Use USPTO template ranking without the local lightweight product-to-template preselector.",
    )
    ap.add_argument("--use-template-relevance-proposals", action="store_true")
    ap.add_argument(
        "--template-relevance-model",
        action="append",
        default=[],
        help="Template relevance model name(s), e.g. template_relevance.reaxys_biocatalysis",
    )
    ap.add_argument("--template-relevance-topk", type=int, default=8)
    ap.add_argument("--include-route-outcomes", action="store_true")
    ap.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    ap.add_argument("--stock", action="append", default=[])
    ap.add_argument("--one-step-model", action="append", default=[])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--chem-enzy-max-depth", type=int, default=6)
    ap.add_argument("--expansion-topk", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--enable-condition-prediction", action="store_true")
    ap.add_argument("--condition-model", default="rcr", choices=["rcr", "parrot"])
    ap.add_argument("--enable-enzyme-assignment", action="store_true")
    ap.add_argument("--cascade-max-depth", type=int, default=6)
    ap.add_argument("--cascade-branch-factor", type=int, default=12)
    ap.add_argument("--cascade-leaf-beam-size", type=int, default=1)
    ap.add_argument("--cascade-diverse-leaf-reserve", type=int, default=0)
    ap.add_argument("--cascade-proposal-topk", type=int, default=None)
    ap.add_argument("--cascade-expansion-budget", type=int, default=100)
    ap.add_argument("--cascade-result-limit", type=int, default=5)
    ap.add_argument("--cascade-min-step-score", type=float, default=0.05)
    ap.add_argument("--cascade-value-model", default=None)
    ap.add_argument("--cascade-transition-model", default=None)
    ap.add_argument("--cascade-action-value-model", default=None)
    ap.add_argument("--cascade-subgoal-provider-model", default=None)
    ap.add_argument("--cascade-subgoal-program-manifest", default=None)
    ap.add_argument("--cascade-subgoal-action-scorer", action="store_true")
    ap.add_argument("--cascade-subgoal-action-max-bonus", type=float, default=0.20)
    ap.add_argument("--cascade-pair-scorer", default=None)
    ap.add_argument("--cascade-rule-pair-scorer", action="store_true")
    ap.add_argument("--cascade-pair-reward-weight", type=float, default=0.0)
    ap.add_argument(
        "--cascade-pair-reward-mode",
        default="additive",
        choices=["additive", "guarded_tie_break"],
        help="How pair scorer rewards are applied inside CascadeProgramSearch.",
    )
    ap.add_argument(
        "--cascade-pair-reward-tie-epsilon",
        type=float,
        default=0.0,
        help="Base-score tie window for guarded_tie_break pair reward mode.",
    )
    ap.add_argument(
        "--route-block-value-final-reranker",
        default=None,
        help="Optional route_block_value_model.pkl used only to rerank generated result programs after search.",
    )
    ap.add_argument(
        "--product-audit-final-reranker",
        action="store_true",
        help="Rerank generated result programs with a conservative no-label product-audit guard.",
    )
    ap.add_argument("--chem-enzy-cascade-cost", action="store_true")
    ap.add_argument("--chem-enzy-cascade-source-policy", action="store_true")
    ap.add_argument("--chem-enzy-cascade-context-json", default=None)
    ap.add_argument("--chem-enzy-cascade-cost-json", default=None)
    ap.add_argument("--chem-enzy-cascade-source-policy-json", default=None)
    ap.add_argument("--chem-enzy-cascade-context-from-row", action="store_true")
    ap.add_argument("--chem-enzy-cascade-context-policy", default="safe", choices=["safe", "strict"])
    ap.add_argument("--no-reuse-planner", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()

    if args.merge is not None:
        payload = merge_cascade_search_outputs([Path(path) for path in args.merge], Path(args.output))
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
        return
    if args.merge_traces is not None:
        payload = merge_cascade_search_trace_outputs([Path(path) for path in args.merge_traces], Path(args.output))
        print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
        return

    payload = run_cascade_search_benchmark(
        benchmark_path=Path(args.benchmark),
        output_path=Path(args.output),
        trace_output_path=Path(args.trace_output) if args.trace_output else None,
        chem_enzy_expansion_trace_output_path=(
            Path(args.chem_enzy_expansion_trace_output) if args.chem_enzy_expansion_trace_output else None
        ),
        use_chem_enzy_expansion_proposals=args.use_chem_enzy_expansion_proposals,
        chem_enzy_expansion_proposal_topk_per_leaf=args.chem_enzy_expansion_proposal_topk_per_leaf,
        use_cascade_retrieval_proposals=args.use_cascade_retrieval_proposals,
        cascade_retrieval_program_manifest=(
            Path(args.cascade_retrieval_program_manifest) if args.cascade_retrieval_program_manifest else None
        ),
        cascade_retrieval_mode=args.cascade_retrieval_mode,
        cascade_retrieval_min_similarity=args.cascade_retrieval_min_similarity,
        cascade_retrieval_require_downstream_transform=args.cascade_retrieval_require_downstream_transform,
        cascade_retrieval_topk=args.cascade_retrieval_topk,
        use_retrochimera_proposals=args.use_retrochimera_proposals,
        retrochimera_model_dir=Path(args.retrochimera_model_dir),
        retrochimera_device=args.retrochimera_device,
        retrochimera_topk=args.retrochimera_topk,
        use_chemical_template_proposals=args.use_chemical_template_proposals,
        chemical_template_topk=args.chemical_template_topk,
        chemical_template_prefer_preselector=not args.chemical_template_no_preselector,
        use_template_relevance_proposals=args.use_template_relevance_proposals,
        template_relevance_models=args.template_relevance_model or None,
        template_relevance_topk=args.template_relevance_topk,
        vendor_root=Path(args.vendor_root),
        stock_names=args.stock or DEFAULT_STOCKS,
        one_step_models=args.one_step_model or DEFAULT_ONE_STEP_MODELS,
        iterations=args.iterations,
        chem_enzy_max_depth=args.chem_enzy_max_depth,
        expansion_topk=args.expansion_topk,
        gpu=args.gpu,
        limit=args.limit,
        enable_condition_prediction=args.enable_condition_prediction,
        enable_enzyme_assignment=args.enable_enzyme_assignment,
        condition_model=args.condition_model,
        cascade_max_depth=args.cascade_max_depth,
        cascade_branch_factor=args.cascade_branch_factor,
        cascade_leaf_beam_size=args.cascade_leaf_beam_size,
        cascade_diverse_leaf_reserve=args.cascade_diverse_leaf_reserve,
        cascade_proposal_topk=args.cascade_proposal_topk,
        cascade_expansion_budget=args.cascade_expansion_budget,
        cascade_result_limit=args.cascade_result_limit,
        cascade_min_step_score=args.cascade_min_step_score,
        cascade_value_model_path=Path(args.cascade_value_model) if args.cascade_value_model else None,
        cascade_transition_model_path=Path(args.cascade_transition_model) if args.cascade_transition_model else None,
        cascade_action_value_model_path=(
            Path(args.cascade_action_value_model) if args.cascade_action_value_model else None
        ),
        cascade_subgoal_provider_model_path=(
            Path(args.cascade_subgoal_provider_model) if args.cascade_subgoal_provider_model else None
        ),
        cascade_subgoal_program_manifest=(
            Path(args.cascade_subgoal_program_manifest) if args.cascade_subgoal_program_manifest else None
        ),
        use_cascade_subgoal_action_scorer=args.cascade_subgoal_action_scorer,
        cascade_subgoal_action_max_bonus=args.cascade_subgoal_action_max_bonus,
        cascade_pair_scorer_path=Path(args.cascade_pair_scorer) if args.cascade_pair_scorer else None,
        use_rule_pair_scorer=args.cascade_rule_pair_scorer,
        cascade_pair_reward_weight=args.cascade_pair_reward_weight,
        cascade_pair_reward_mode=args.cascade_pair_reward_mode,
        cascade_pair_reward_tie_epsilon=args.cascade_pair_reward_tie_epsilon,
        route_block_value_final_reranker_path=(
            Path(args.route_block_value_final_reranker) if args.route_block_value_final_reranker else None
        ),
        use_product_audit_final_reranker=args.product_audit_final_reranker,
        use_chem_enzy_cascade_cost=args.chem_enzy_cascade_cost,
        use_chem_enzy_cascade_source_policy=args.chem_enzy_cascade_source_policy,
        chem_enzy_cascade_context=_json_mapping(args.chem_enzy_cascade_context_json),
        chem_enzy_cascade_cost_model=_json_mapping(args.chem_enzy_cascade_cost_json),
        chem_enzy_cascade_source_policy=_json_mapping(args.chem_enzy_cascade_source_policy_json),
        chem_enzy_cascade_context_from_row=args.chem_enzy_cascade_context_from_row,
        chem_enzy_cascade_context_policy=args.chem_enzy_cascade_context_policy,
        reuse_planner=not args.no_reuse_planner,
        dry_run=args.dry_run,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        include_route_outcomes=args.include_route_outcomes,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


def _json_mapping(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("JSON value must be an object")
    return payload


def _validate_model_inputs(
    *,
    cascade_value_model_path: Path | None = None,
    cascade_transition_model_path: Path | None = None,
    cascade_action_value_model_path: Path | None = None,
    cascade_subgoal_provider_model_path: Path | None = None,
    cascade_pair_scorer_path: Path | None = None,
    route_block_value_final_reranker_path: Path | None = None,
    chem_enzy_cascade_cost_model: dict[str, Any] | None = None,
    chem_enzy_cascade_source_policy: dict[str, Any] | None = None,
) -> None:
    """Fail before expensive ChemEnzy search when configured model files are missing."""
    explicit_paths = {
        "cascade_value_model": cascade_value_model_path,
        "cascade_transition_model": cascade_transition_model_path,
        "cascade_action_value_model": cascade_action_value_model_path,
        "cascade_subgoal_provider_model": cascade_subgoal_provider_model_path,
        "cascade_pair_scorer": cascade_pair_scorer_path,
        "route_block_value_final_reranker": route_block_value_final_reranker_path,
    }
    for label, path in explicit_paths.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    for label, payload in (
        ("chem_enzy_cascade_cost_model", chem_enzy_cascade_cost_model),
        ("chem_enzy_cascade_source_policy", chem_enzy_cascade_source_policy),
    ):
        if payload:
            for dotted_key, value in _iter_configured_model_paths(payload):
                path = Path(str(value))
                if not path.is_file():
                    raise FileNotFoundError(f"{label}.{dotted_key} not found: {path}")


def _validate_stock_inputs(config_path: Path, stock_names: list[str]) -> None:
    if not config_path.is_file():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    available = set((config.get("stocks") or {}).keys())
    missing = [name for name in stock_names if name not in available]
    if missing:
        raise ValueError(
            "selected stock names not found in ChemEnzy config: "
            f"{missing}; available stocks: {sorted(available)}"
        )


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


def _chem_enzy_search_flags_for_row(
    row: dict[str, Any],
    base_flags: dict[str, Any],
    *,
    context_from_row: bool,
    context_policy: str = "safe",
) -> dict[str, Any]:
    flags = dict(base_flags)
    if not context_from_row or not (
        flags.get("use_cascade_cost_model") or flags.get("use_cascade_source_policy")
    ):
        return flags
    context = dict(flags.get("cascade_search_context") or {})
    context.update(_route_domain_cascade_context(str(row.get("route_domain") or ""), policy=context_policy))
    context["context_source"] = "route_domain"
    context["context_policy"] = context_policy
    context["route_domain"] = row.get("route_domain")
    flags["cascade_search_context"] = context
    return flags


def _route_domain_cascade_context(route_domain: str, *, policy: str = "safe") -> dict[str, Any]:
    domain = route_domain.strip().lower()
    if policy == "strict":
        if domain in {"all_chemical", "hybrid_mimetic"}:
            return {
                "preferred_reaction_domains": ["chemical"],
                "active_failure_modes": [],
                "penalize_unpreferred_domain": True,
            }
        if domain in {"all_enzymatic", "whole_cell_biocatalytic"}:
            return {
                "preferred_reaction_domains": ["enzymatic"],
                "active_failure_modes": ["EnzymeEvidenceWeak"],
                "penalize_unpreferred_domain": True,
            }
        if domain == "chemoenzymatic":
            return {
                "preferred_reaction_domains": ["chemical", "enzymatic"],
                "active_failure_modes": ["EnzymeEvidenceWeak"],
                "penalize_unpreferred_domain": False,
            }
        return {}

    if domain in {"all_chemical", "hybrid_mimetic"}:
        return {
            "preferred_reaction_domains": [],
            "active_failure_modes": [],
            "penalize_unpreferred_domain": False,
        }
    if domain in {"all_enzymatic", "whole_cell_biocatalytic"}:
        return {
            "preferred_reaction_domains": ["enzymatic"],
            "active_failure_modes": ["EnzymeEvidenceWeak"],
            "penalize_unpreferred_domain": False,
        }
    if domain == "chemoenzymatic":
        return {
            "preferred_reaction_domains": ["enzymatic"],
            "active_failure_modes": ["EnzymeEvidenceWeak"],
            "penalize_unpreferred_domain": False,
        }
    return {}


if __name__ == "__main__":
    main()
