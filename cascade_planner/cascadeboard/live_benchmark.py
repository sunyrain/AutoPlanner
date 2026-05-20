"""Full live CascadeBoard benchmark runner.

Unlike integrated_benchmark.py, this runner executes the live skeleton-to-fill
path and records filled-route, stock, condition, compatibility, and candidate
source diagnostics separately.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from cascade_planner.cascadeboard.live_retro import build_live_retro_engine, retro_engine_cache_stats
from cascade_planner.cascadeboard import CascadeBoard
from cascade_planner.cascadeboard.route_export import (
    diversify_ranked_route_results,
    route_metrics,
    route_results_payload,
)
from cascade_planner.cascadeboard.skeleton_inpainter import (
    generate_multiple_skeletons,
    load_model as load_skeleton_model,
)
from cascade_planner.cascadeboard.skeleton_retrieval_prior import (
    augment_skeletons_with_retrieval_prior,
    skeleton_retrieval_prior_metadata,
)
from cascade_planner.cascadeboard.skeleton_reranker import (
    rerank_skeletons_with_model,
    skeleton_reranker_metadata,
)
from cascade_planner.cascadeboard.skeleton_planner import RouteSkeleton, plan_with_skeleton
from cascade_planner.cascadeboard.cc_aostar import plan_with_cc_aostar
from cascade_planner.cascadeboard.route_recovery import target_recovery_metrics
from cascade_planner.agent.prior_bridge import (
    get_prior_for_target,
    load_prior_cache,
    rank_skeletons_with_prior,
    summarize_prior,
    write_prior_cache,
)
from cascade_planner.agent.route_critic import critique_route_payload
from cascade_planner.agent.failure_policy import predict_failure_risk


def _canonical(smi: str | None) -> str:
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi


def _gt_types(entry: dict[str, Any]) -> list[str]:
    return [s.get("transformation", "") for s in entry.get("gt_route", [])]


def _retro_gt_types(entry: dict[str, Any]) -> list[str]:
    return list(reversed(_gt_types(entry)))


def _threshold_for_depth(depth: int) -> float:
    if depth <= 2:
        return 1.0
    if depth == 3:
        return 2.0 / 3.0
    return 0.5


def _type_match(route: dict[str, Any], gt_types: list[str], require_filled: bool = False) -> bool:
    steps = route.get("steps", [])
    if len(steps) != len(gt_types) or not gt_types:
        return False
    if require_filled and not route.get("metrics", {}).get("filled_route"):
        return False
    pred_types = [s.get("reaction_type") or "" for s in steps]
    matches = sum(1 for p, g in zip(pred_types, gt_types) if p == g)
    return matches / len(gt_types) >= _threshold_for_depth(len(gt_types))


def _route_tree_type_metric_order(search_mode: str) -> str:
    return "retro" if search_mode == "route_tree" else "forward"


def _terminal_gt_in_route(route: dict[str, Any], entry: dict[str, Any]) -> bool:
    """Weak full-route check: terminal predicted starting material appears in GT."""
    gt_reactants = set()
    for step in entry.get("gt_route", []):
        rxn = step.get("rxn_smiles") or ""
        if ">>" not in rxn:
            continue
        lhs = rxn.split(">>", 1)[0]
        for part in lhs.split("."):
            can = _canonical(part.strip())
            if can:
                gt_reactants.add(can)

    terminals = route.get("metrics", {}).get("terminal_reactants") or []
    pred = {_canonical(smi) for smi in terminals if smi}
    return bool(gt_reactants and pred and pred.intersection(gt_reactants))


def _build_stock_checker(enabled: bool):
    if not enabled:
        return None
    try:
        from cascade_planner.cascadeboard.zinc_stock import is_in_zinc_stock
        return is_in_zinc_stock
    except Exception:
        return None


def _load_benchmark_entries(
    bench_path: str,
    limit: int | None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    bench = []
    for original_index, entry in enumerate(json.loads(Path(bench_path).read_text())):
        row = dict(entry)
        row["_benchmark_index"] = original_index
        bench.append(row)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if num_shards > 1:
        bench = bench[shard_index::num_shards]
    if limit is not None:
        bench = bench[:limit]
    return bench


def _emit_target_log(mode: str, event: str, payload: dict[str, Any]) -> None:
    if mode == "none":
        return
    record = {"event": event, **payload}
    if mode == "json":
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)
        return
    prefix = "[benchmark]"
    if event == "start":
        print(
            (
                f"{prefix} start {record.get('ordinal')}/{record.get('total')} "
                f"idx={record.get('index')} domain={record.get('domain')} "
                f"depth={record.get('depth')} target={record.get('target')}"
            ),
            file=sys.stderr,
            flush=True,
        )
        return
    print(
        (
            f"{prefix} done {record.get('ordinal')}/{record.get('total')} "
            f"idx={record.get('index')} domain={record.get('domain')} "
            f"depth={record.get('depth')} routes={record.get('routes')} "
            f"plan={int(bool(record.get('plan')))} "
            f"type@1={int(bool(record.get('type_at_1')))} "
            f"type@5={int(bool(record.get('type_at_5')))} "
            f"stock={record.get('stock')} cond={int(bool(record.get('condition')))} "
            f"compat={int(bool(record.get('compatibility')))} "
            f"exact_pool={int(bool(record.get('exact_reaction_in_route_pool')))} "
            f"gt_pool={int(bool(record.get('gt_reactant_in_route_pool')))} "
            f"cand_exact={int(bool(record.get('candidate_exact_reaction_in_pool')))} "
            f"cand_gt={int(bool(record.get('candidate_gt_reactant_in_pool')))} "
            f"elapsed={record.get('elapsed_s')}s "
            f"error={record.get('error') or ''}"
        ),
        file=sys.stderr,
        flush=True,
    )


def _to_route_skeleton(skel) -> RouteSkeleton:
    return RouteSkeleton(
        n_steps=len(skel.types),
        types=skel.types,
        ec1s=skel.ec1s,
        Ts=skel.Ts,
        pHs=skel.pHs,
        compatibility=skel.compat_pred,
        operation_mode=skel.opmode_pred,
        issues=skel.issues_pred,
        pairwise_modes=[],
        log_prob=float(getattr(skel, "log_prob", 0.0) or 0.0),
    )


def _prior_adjusted_budget(
    *,
    n_results: int,
    n_candidates_per_skeleton: int,
    skeleton_samples: int | None,
    search_budget: int | None,
    search_mode: str,
    agent_prior: dict[str, Any] | None,
) -> tuple[int, int]:
    skeleton_budget = max(skeleton_samples or n_results, search_budget or 0, n_results, 1)
    candidate_budget = n_candidates_per_skeleton
    if search_mode not in {"critic_control", "cc_aostar", "hybrid", "stock_rescue", "route_tree"} or not agent_prior:
        return skeleton_budget, candidate_budget

    risk_count = len(agent_prior.get("condition_risks") or [])
    enzyme_weight = sum(float(x.get("weight") or 0.0) for x in agent_prior.get("enzyme_priors") or [])
    cascade_weight = sum(
        float(x.get("weight") or 0.0)
        for x in agent_prior.get("route_mode_priors") or []
        if x.get("mode") in {"chemoenzymatic_cascade", "enzymatic_only", "enzymatic_late_stage"}
    )
    skeleton_budget = max(skeleton_budget, n_results + min(6, risk_count + int(cascade_weight > 0.5)))
    if risk_count or enzyme_weight > 0.75 or cascade_weight > 0.75:
        candidate_budget = max(candidate_budget, 3)
    return skeleton_budget, candidate_budget


def _result_is_acceptable(result, stock_checker=None) -> bool:
    metrics = route_metrics(result.board, stock_checker=stock_checker)
    if not metrics.get("filled_route"):
        return False
    stock = metrics.get("strict_stock_solve")
    if stock is False:
        return False
    cond = metrics.get("condition") or {}
    if cond.get("condition_window_success") is False:
        return False
    compat = metrics.get("cascade_compatibility") or {}
    return compat.get("cascade_compatibility_success") is not False


def _needs_stock_rescue(results: list, *, stock_checker=None) -> bool:
    if not results or stock_checker is None:
        return False
    for result in results:
        metrics = route_metrics(result.board, stock_checker=stock_checker)
        if metrics.get("strict_stock_solve") is True:
            return False
    return True


def _fixed_slots_from_constraints(constraints: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    fixed: dict[int, dict[str, Any]] = {}
    for item in (constraints or {}).get("fixed_steps", []) or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        values = item.get("values") or {}
        if isinstance(values, dict):
            fixed[idx] = dict(values)
    return fixed


def _plan_one_target(
    skeleton_model,
    retro_engine: dict,
    *,
    target: str,
    depth: int,
    domain: str,
    model_device: str,
    n_results: int,
    n_candidates_per_skeleton: int,
    skeleton_samples: int | None = None,
    agent_prior: dict[str, Any] | None = None,
    prior_weight: float = 1.0,
    search_mode: str = "rerank",
    search_budget: int | None = None,
    stock_checker=None,
    constraints: dict[str, Any] | None = None,
    trace_collector: Any | None = None,
) -> list:
    skeleton_budget, candidate_budget = _prior_adjusted_budget(
        n_results=n_results,
        n_candidates_per_skeleton=n_candidates_per_skeleton,
        skeleton_samples=skeleton_samples,
        search_budget=search_budget,
        search_mode=search_mode,
        agent_prior=agent_prior,
    )
    if search_mode == "route_tree" and str(domain or "").lower() in {"enzymatic", "biosynthetic", "bio"}:
        skeleton_budget = max(
            skeleton_budget,
            _env_int("AUTOPLANNER_ROUTE_TREE_ENZYMATIC_MIN_SKELETONS", 4),
        )
    skeletons = generate_multiple_skeletons(
        skeleton_model,
        target,
        n_steps=depth,
        k=skeleton_budget,
        domain=domain,
        objective="balanced",
        temperature=0.8,
        fixed_slots=_fixed_slots_from_constraints(constraints),
        device=model_device,
    )
    skeletons = rank_skeletons_with_prior(skeletons, agent_prior, prior_weight=prior_weight)
    skeletons = augment_skeletons_with_retrieval_prior(
        skeletons,
        target_smiles=target,
        depth=depth,
        domain=domain,
        max_new=max(1, min(2, skeleton_budget)),
    )[:skeleton_budget]
    skeletons = rerank_skeletons_with_model(skeletons, target_smiles=target)[:skeleton_budget]
    route_skeletons = [_to_route_skeleton(skel) for skel in skeletons]
    if search_mode == "hybrid":
        from cascade_planner.cascadeboard.stock_andor import plan_stock_closed_andor

        hybrid_candidate_budget = max(candidate_budget, n_candidates_per_skeleton * 4, 8)
        hybrid_expansion_budget = max(
            search_budget or 0,
            skeleton_budget * hybrid_candidate_budget * 2,
            64,
        )
        stock_depth = max(depth, min(depth + 2, 6))
        andor_results = plan_stock_closed_andor(
            target=target,
            skeletons=route_skeletons,
            retro_engine=retro_engine,
            stock_checker=stock_checker,
            max_depth=stock_depth,
            n_results=n_results,
            branch_factor=hybrid_candidate_budget,
            expansion_budget=hybrid_expansion_budget,
            constraints=constraints,
        )
        results = list(andor_results)
        if not any(_result_is_acceptable(result, stock_checker=stock_checker) for result in andor_results):
            results.extend(
                plan_with_cc_aostar(
                    target=target,
                    skeletons=route_skeletons,
                    retro_engine=retro_engine,
                    n_results=n_results,
                    candidate_budget=hybrid_candidate_budget,
                    expansion_budget=hybrid_expansion_budget,
                    stock_checker=stock_checker,
                    constraints=constraints,
                )
            )
        return _rank_results(
            results,
            search_mode="critic_control" if stock_checker else "rerank",
            stock_checker=stock_checker,
        )[:n_results]
    if search_mode == "cc_aostar":
        cc_candidate_budget = max(candidate_budget, n_candidates_per_skeleton * 4, 8)
        return plan_with_cc_aostar(
            target=target,
            skeletons=route_skeletons,
            retro_engine=retro_engine,
            n_results=n_results,
            candidate_budget=cc_candidate_budget,
            expansion_budget=max(search_budget or 0, skeleton_budget * cc_candidate_budget, n_results),
            stock_checker=stock_checker,
            constraints=constraints,
        )
    if search_mode == "stock_rescue":
        from cascade_planner.cascadeboard.stock_andor import plan_stock_closed_andor

        base_candidate_budget = max(candidate_budget, n_candidates_per_skeleton * 4, 8)
        base_expansion_budget = max(search_budget or 0, skeleton_budget * base_candidate_budget, n_results)
        rescue_candidate_budget = max(candidate_budget, n_candidates_per_skeleton * 2, 4)
        rescue_expansion_budget = max(
            search_budget or 0,
            skeleton_budget * rescue_candidate_budget,
            32,
        )
        results = plan_with_cc_aostar(
            target=target,
            skeletons=route_skeletons,
            retro_engine=retro_engine,
            n_results=n_results,
            candidate_budget=base_candidate_budget,
            expansion_budget=base_expansion_budget,
            stock_checker=stock_checker,
            constraints=constraints,
        )
        if _needs_stock_rescue(results, stock_checker=stock_checker):
            results.extend(
                plan_stock_closed_andor(
                    target=target,
                    skeletons=route_skeletons,
                    retro_engine=retro_engine,
                    stock_checker=stock_checker,
                    max_depth=max(depth, min(depth + 2, 6)),
                    n_results=n_results,
                    branch_factor=rescue_candidate_budget,
                    expansion_budget=rescue_expansion_budget,
                    constraints=constraints,
                )
            )
        return _rank_results(
            results,
            search_mode="critic_control" if stock_checker else "rerank",
            stock_checker=stock_checker,
        )[:n_results]
    if search_mode == "route_tree":
        from cascade_planner.route_tree.search import plan_with_route_tree
        from cascade_planner.route_tree.bounded_reservoir import append_bounded_native_reservoir, bounded_reservoir_enabled

        route_tree_branch_factor = max(candidate_budget, n_candidates_per_skeleton * 4, 8)
        route_tree_branch_factor = max(
            route_tree_branch_factor,
            _env_int("AUTOPLANNER_ROUTE_TREE_MIN_BRANCH_FACTOR", route_tree_branch_factor),
        )
        route_tree_budget = max(search_budget or 0, skeleton_budget * route_tree_branch_factor * 2, 64)
        route_tree_budget = max(
            route_tree_budget,
            _env_int("AUTOPLANNER_ROUTE_TREE_MIN_EXPANSION_BUDGET", route_tree_budget),
        )
        route_tree_results = plan_with_route_tree(
            target=target,
            skeletons=route_skeletons,
            retro_engine=retro_engine,
            stock_checker=stock_checker,
            max_depth=max(depth, min(depth + 2, 6)),
            n_results=n_results,
            branch_factor=route_tree_branch_factor,
            expansion_budget=route_tree_budget,
            constraints=constraints,
            trace_collector=trace_collector,
        )
        if not route_tree_results and _env_truthy("AUTOPLANNER_ROUTE_TREE_EMPTY_FALLBACK"):
            fallback_results = plan_with_cc_aostar(
                target=target,
                skeletons=route_skeletons,
                retro_engine=retro_engine,
                n_results=n_results,
                candidate_budget=route_tree_branch_factor,
                expansion_budget=route_tree_budget,
                stock_checker=stock_checker,
                constraints=constraints,
            )
            if not fallback_results and route_skeletons:
                skeleton_results: list = []
                for skeleton in route_skeletons[: max(1, min(2, n_results))]:
                    try:
                        skeleton_results = plan_with_skeleton(
                            target=target,
                            retro_engine=retro_engine,
                            stock_checker=stock_checker,
                            constraints=constraints,
                            objective="balanced",
                            n_steps=skeleton.n_steps,
                            n_candidates=max(2, n_candidates_per_skeleton * 2),
                            device=model_device,
                            skeleton=skeleton,
                        )
                    except Exception:
                        skeleton_results = []
                    if skeleton_results:
                        fallback_results = skeleton_results
                        break
            if fallback_results:
                route_tree_results = fallback_results
        if _env_truthy("AUTOPLANNER_ROUTE_TREE_FINAL_RERANK"):
            ranked = _rank_results(
                route_tree_results,
                search_mode="critic_control" if stock_checker else "rerank",
                stock_checker=stock_checker,
            )[:n_results]
            if bounded_reservoir_enabled():
                return append_bounded_native_reservoir(
                    target=target,
                    results=ranked,
                    stock_checker=stock_checker,
                    max_depth=max(depth, min(depth + 2, 6)),
                    n_results=n_results,
                )
            return ranked
        base_results = route_tree_results[:n_results]
        if bounded_reservoir_enabled():
            return append_bounded_native_reservoir(
                target=target,
                results=base_results,
                stock_checker=stock_checker,
                max_depth=max(depth, min(depth + 2, 6)),
                n_results=n_results,
            )
        return base_results
    if search_mode == "rerank":
        skeletons = skeletons[:max(n_results, 1)]

    results = []
    for skel in skeletons:
        results.extend(
            plan_with_skeleton(
                target=target,
                skeleton=_to_route_skeleton(skel),
                retro_engine=retro_engine,
                constraints=constraints,
                objective="balanced",
                n_steps=depth,
                starting_material=(constraints or {}).get("starting_material"),
                n_candidates=candidate_budget,
                stock_checker=stock_checker if search_mode in {"stock_aware", "critic_control"} else None,
                device=model_device,
            )
        )
        if search_mode == "critic_control":
            ranked = _rank_results(results, search_mode=search_mode, stock_checker=stock_checker)
            if len(ranked) >= n_results and any(
                _result_is_acceptable(r, stock_checker=stock_checker)
                for r in ranked[:n_results]
            ):
                return ranked[:n_results]

    results = _rank_results(results, search_mode=search_mode, stock_checker=stock_checker)
    return results[:n_results]


def _plan_one_target_with_policy_retry(
    skeleton_model,
    retro_engine: dict,
    *,
    target: str,
    depth: int,
    domain: str,
    model_device: str,
    n_results: int,
    n_candidates_per_skeleton: int,
    skeleton_samples: int | None = None,
    agent_prior: dict[str, Any] | None = None,
    prior_weight: float = 1.0,
    search_budget: int | None = None,
    stock_checker=None,
    constraints: dict[str, Any] | None = None,
    trace_collector: Any | None = None,
) -> tuple[list, dict[str, Any]]:
    """Run a base AO* search, then one learned bounded retry if appropriate."""
    base_search_mode = "cc_aostar"
    t0 = time.time()
    base_results = _plan_one_target(
        skeleton_model,
        retro_engine,
        target=target,
        depth=depth,
        domain=domain,
        model_device=model_device,
        n_results=n_results,
        n_candidates_per_skeleton=n_candidates_per_skeleton,
        skeleton_samples=skeleton_samples,
        agent_prior=agent_prior,
        prior_weight=prior_weight,
        search_mode=base_search_mode,
        search_budget=search_budget,
        stock_checker=stock_checker,
        constraints=constraints,
        trace_collector=trace_collector if base_search_mode == "route_tree" else None,
    )
    base_elapsed = time.time() - t0
    base_payload = route_results_payload(
        target,
        base_results,
        objective="balanced",
        constraints=constraints,
        elapsed_s=base_elapsed,
        stock_checker=stock_checker,
    )
    _annotate_policy_payload(
        base_payload,
        depth=depth,
        domain=domain,
        n_results=n_results,
        skeleton_samples=skeleton_samples,
        candidate_budget=n_candidates_per_skeleton,
        expansion_budget=search_budget,
    )
    failure_risk = predict_failure_risk(base_payload)
    retry_policy = failure_risk.get("retry_policy") or {}
    retry_meta = {
        "base_search_mode": base_search_mode,
        "base_elapsed_s": round(base_elapsed, 3),
        "retry_executed": False,
        "failure_risk": failure_risk,
        "retry_policy": retry_policy,
    }
    if not retry_policy.get("automatic_retry_safe"):
        ranked = _rank_results(
            base_results,
            search_mode="critic_control" if stock_checker else "rerank",
            stock_checker=stock_checker,
        )
        return ranked[:n_results], retry_meta

    adjusted = retry_policy.get("adjusted_settings") or {}
    retry_depth = max(depth, int(adjusted.get("max_steps") or depth))
    retry_n_results = int(adjusted.get("n_results") or n_results)
    retry_candidates = int(adjusted.get("candidate_budget") or n_candidates_per_skeleton)
    retry_skeletons = int(adjusted.get("skeleton_samples") or (skeleton_samples or n_results))
    retry_search_mode = str(adjusted.get("retry_search_mode") or base_search_mode)
    if retry_search_mode == "stock_rescue" and stock_checker is None:
        retry_search_mode = base_search_mode
    retry_plan_candidates = retry_candidates
    if retry_search_mode == "stock_rescue":
        retry_plan_candidates = min(retry_candidates, max(n_candidates_per_skeleton, 2))
    requested_retry_budget = int(adjusted.get("expansion_budget") or (search_budget or 0)) or None
    auto_retry_budget_cap = max(
        search_budget or 0,
        retry_depth * retry_skeletons * retry_plan_candidates,
        64,
    )
    retry_budget = (
        min(requested_retry_budget, auto_retry_budget_cap)
        if requested_retry_budget is not None
        else auto_retry_budget_cap
    )
    t1 = time.time()
    retry_results = _plan_one_target(
        skeleton_model,
        retro_engine,
        target=target,
        depth=retry_depth,
        domain=domain,
        model_device=model_device,
        n_results=retry_n_results,
        n_candidates_per_skeleton=retry_plan_candidates,
        skeleton_samples=retry_skeletons,
        agent_prior=agent_prior,
        prior_weight=prior_weight,
        search_mode=retry_search_mode,
        search_budget=retry_budget,
        stock_checker=stock_checker,
        constraints=constraints,
        trace_collector=trace_collector if retry_search_mode == "route_tree" else None,
    )
    retry_meta.update({
        "retry_executed": True,
        "retry_elapsed_s": round(time.time() - t1, 3),
        "retry_depth": retry_depth,
        "retry_n_results": retry_n_results,
        "retry_candidates": retry_candidates,
        "retry_plan_candidates": retry_plan_candidates,
        "retry_skeletons": retry_skeletons,
        "retry_budget": retry_budget,
        "requested_retry_budget": requested_retry_budget,
        "auto_retry_budget_cap": auto_retry_budget_cap,
        "retry_search_mode": retry_search_mode,
    })
    merged = _rank_results(
        base_results + retry_results,
        search_mode="critic_control" if stock_checker else "rerank",
        stock_checker=stock_checker,
    )
    return merged[:n_results], retry_meta


def _annotate_policy_payload(
    payload: dict[str, Any],
    *,
    depth: int,
    domain: str,
    n_results: int,
    skeleton_samples: int | None,
    candidate_budget: int,
    expansion_budget: int | None,
) -> None:
    routes = payload.get("routes") or []
    solved = any(
        bool((route.get("metrics") or {}).get("route_solved"))
        and bool((route.get("metrics") or {}).get("progressive_route"))
        for route in routes
    )
    payload["n_results"] = n_results
    payload["ui_metadata"] = {
        "search_mode": "policy_retry",
        "planner_mode": "policy_retry",
        "min_steps": depth,
        "max_steps": depth,
        "domain": domain,
        "skeleton_samples": skeleton_samples or n_results,
        "candidate_budget": candidate_budget,
        "expansion_budget": expansion_budget or max(depth * (skeleton_samples or n_results) * candidate_budget * 4, n_results),
    }
    payload["search_status"] = {
        "mode": "policy_retry",
        "status": "solved" if solved else "failed",
        "solved": solved,
        "best_depth": depth,
    }


def _rank_results(results: list, *, search_mode: str = "rerank", stock_checker=None) -> list:
    if search_mode not in {"stock_aware", "critic_control"}:
        return diversify_ranked_route_results(sorted(results, key=lambda r: r.score, reverse=True))

    def control_score(result) -> float:
        metrics = route_metrics(result.board, stock_checker=stock_checker)
        cond = metrics.get("condition") or {}
        compat = metrics.get("cascade_compatibility") or {}
        enz = metrics.get("enzyme_evidence") or {}
        progress = metrics.get("retrosynthesis_progress") or {}
        natural = metrics.get("route_naturalness") or {}
        operation = metrics.get("operation_transitions") or {}
        candidate_pool = metrics.get("candidate_pool") or {}
        score = 0.0
        score += 50.0 * float(bool(metrics.get("filled_route")))
        if metrics.get("strict_stock_solve") is True:
            score += 80.0
        elif metrics.get("strict_stock_solve") is False:
            score -= 25.0
        score += 35.0 * float(bool(cond.get("condition_window_success")))
        if cond.get("condition_window_success") is False:
            score -= 20.0
        score += 35.0 * float(bool(compat.get("cascade_compatibility_success")))
        if compat.get("cascade_compatibility_success") is False:
            score -= 20.0
        score += 20.0 * float(bool(metrics.get("progressive_route")))
        score += 10.0 * float(progress.get("main_chain_reduction") or 0.0)
        score += 5.0 * float(natural.get("naturalness_score") or 0.0)
        if operation.get("operation_score") is not None:
            score += 8.0 * float(operation.get("operation_score") or 0.0)
            score -= 2.0 * len(operation.get("issues") or [])
        if candidate_pool.get("avg_pool_diversity_score") is not None:
            score += 3.0 * float(candidate_pool.get("avg_pool_diversity_score") or 0.0)
        if candidate_pool.get("candidate_pool_coverage") is not None:
            score += 2.0 * float(candidate_pool.get("candidate_pool_coverage") or 0.0)
        if candidate_pool.get("avg_duplicate_reactant_set_fraction") is not None:
            score -= 2.0 * float(candidate_pool.get("avg_duplicate_reactant_set_fraction") or 0.0)
        if enz.get("enzyme_evidence_score") is not None:
            score += 10.0 * float(enz["enzyme_evidence_score"])
        if search_mode == "critic_control":
            score -= 10.0 * len(compat.get("issues") or [])
        score += 0.01 * float(result.score or 0.0)
        return score

    return diversify_ranked_route_results(sorted(results, key=control_score, reverse=True))


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _route_tree_diagnostics_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for route in payload.get("routes") or []:
        table = ((route.get("explanation") or {}).get("uncertainty_table") or {})
        if table.get("search_mode") == "route_tree" or table.get("route_tree_version"):
            return table
    return {}


def _time_bucket(elapsed_s: float) -> str:
    if elapsed_s < 1:
        return "<1"
    if elapsed_s < 10:
        return "1-10"
    if elapsed_s < 30:
        return "10-30"
    if elapsed_s < 60:
        return "30-60"
    if elapsed_s < 120:
        return "60-120"
    return ">120"


def summarize_target_results(target_results: list[dict[str, Any]], check_stock: bool = False) -> dict[str, Any]:
    counters = Counter()
    denominators = Counter()
    source_counts = Counter()
    candidate_pool_source_counts = Counter()
    recovery_bottleneck_counts = Counter()
    runtime_bottleneck_counts = Counter()
    route_tree_stop_counts = Counter()
    time_bucket_counts = Counter()
    route_tree_source_call_counts = Counter()
    route_tree_source_latency_ms = defaultdict(float)
    prior_source_counts = Counter()
    per_domain = defaultdict(Counter)
    numeric_recovery = defaultdict(list)
    numeric_metrics = defaultdict(list)
    slow_targets: list[dict[str, Any]] = []
    total_time = 0.0

    for target in target_results:
        domain = target.get("route_domain") or "unknown"
        metrics = target.get("metrics") or {}
        payload = target.get("planner_output") or {}
        routes = payload.get("routes") or []
        elapsed_s = float(payload.get("time_s") or 0.0)
        total_time += elapsed_s
        numeric_metrics["route_count"].append(float(len(routes)))
        time_bucket_counts[_time_bucket(elapsed_s)] += 1
        route_tree_diag = _route_tree_diagnostics_from_payload(payload)
        if route_tree_diag:
            route_tree_stop_counts[str(route_tree_diag.get("search_stop_reason") or "unknown")] += 1
            for label in route_tree_diag.get("route_tree_runtime_bottlenecks") or []:
                runtime_bottleneck_counts[str(label)] += 1
            for source, row in (route_tree_diag.get("proposal_source_stats") or {}).items():
                route_tree_source_call_counts[str(source)] += int(row.get("calls") or 0)
                route_tree_source_latency_ms[str(source)] += float(row.get("latency_ms_total") or 0.0)
            slow_targets.append({
                "index": target.get("index"),
                "route_domain": domain,
                "target_smiles": target.get("target_smiles"),
                "elapsed_s": round(elapsed_s, 3),
                "routes": len(routes),
                "stock": metrics.get("strict_stock_solve_any"),
                "candidate_gt_reactant_in_pool": (target.get("route_recovery") or {}).get("candidate_gt_reactant_in_pool"),
                "gt_reactant_in_route_pool": (target.get("route_recovery") or {}).get("gt_reactant_in_route_pool"),
                "expansions": route_tree_diag.get("expansions"),
                "generated_actions": route_tree_diag.get("generated_actions"),
                "proposal_calls": route_tree_diag.get("proposal_calls"),
                "proposal_cache_hits": route_tree_diag.get("proposal_cache_hits"),
                "model_calls": route_tree_diag.get("model_calls"),
                "stop_reason": route_tree_diag.get("search_stop_reason"),
                "runtime_bottlenecks": route_tree_diag.get("route_tree_runtime_bottlenecks") or [],
            })
        for key, value in metrics.items():
            if key == "strict_stock_solve_any":
                continue
            if isinstance(value, bool):
                counters[key] += int(value)
                per_domain[domain][key] += int(value)
            elif isinstance(value, (int, float)) and value is not None:
                numeric_metrics[key].append(float(value))
        recovery = target.get("route_recovery") or {}
        bottleneck = recovery.get("recovery_bottleneck")
        if bottleneck:
            recovery_bottleneck_counts[str(bottleneck)] += 1
            per_domain[domain][f"recovery_bottleneck:{bottleneck}"] += 1
        for key, value in recovery.items():
            if isinstance(value, bool):
                metric_key = f"recovery_{key}"
                denominators[metric_key] += 1
                counters[metric_key] += int(value)
                per_domain[domain][metric_key] += int(value)
            elif isinstance(value, (int, float)) and value is not None:
                numeric_recovery[key].append(float(value))
        counters["n"] += 1
        per_domain[domain]["n"] += 1
        for route in routes:
            route_metrics_row = route.get("metrics", {}) or {}
            source_counts.update(route_metrics_row.get("candidate_source_counts") or {})
            candidate_pool = route_metrics_row.get("candidate_pool") or {}
            candidate_pool_source_counts.update(candidate_pool.get("candidate_pool_source_counts") or {})
            for key in (
                "candidate_pool_coverage",
                "total_candidates",
                "avg_candidates_per_step",
                "avg_pool_diversity_score",
                "min_pool_diversity_score",
                "avg_duplicate_reaction_fraction",
                "avg_duplicate_main_reactant_fraction",
                "avg_duplicate_reactant_set_fraction",
                "single_reactant_set_steps",
            ):
                value = candidate_pool.get(key)
                if isinstance(value, (int, float)) and value is not None:
                    numeric_metrics[f"route_candidate_pool_{key}"].append(float(value))
        diversity = ((payload.get("route_set_metrics") or {}).get("diversity") or {})
        for key in (
            "unique_type_sequences",
            "unique_source_sequences",
            "unique_ec1_sequences",
            "unique_terminal_reactant_sets",
            "unique_full_signatures",
            "duplicate_route_fraction",
            "mean_pairwise_type_distance",
            "mean_pairwise_terminal_jaccard_distance",
        ):
            value = diversity.get(key)
            if isinstance(value, (int, float)) and value is not None:
                numeric_metrics[f"route_set_{key}"].append(float(value))
        prior = target.get("agent_prior") or {}
        if prior:
            prior_source_counts.update([prior.get("source") or "unknown"])

        stock_values = [
            (route.get("metrics") or {}).get("strict_stock_solve")
            for route in routes
            if (route.get("metrics") or {}).get("strict_stock_solve") is not None
        ]
        if stock_values:
            counters["strict_stock_solve_any"] += int(any(stock_values))
            per_domain[domain]["strict_stock_solve_any"] += int(any(stock_values))

    n = max(counters["n"], 1)
    def recovery_rate(key: str) -> float | None:
        metric_key = f"recovery_{key}"
        denom = denominators[metric_key]
        if denom <= 0:
            return None
        return counters[metric_key] / denom

    def recovery_mean(key: str) -> float | None:
        values = numeric_recovery.get(key) or []
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    def metric_mean(key: str) -> float | None:
        values = numeric_metrics.get(key) or []
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    return {
        "n_targets": counters["n"],
        "plan_rate": counters["plan"] / n,
        "skeleton_type_GT@1": counters["skeleton_type_GT@1"] / n,
        "skeleton_type_GT@5": counters["skeleton_type_GT@5"] / n,
        "forward_skeleton_type_GT@1": counters["forward_skeleton_type_GT@1"] / n,
        "forward_skeleton_type_GT@5": counters["forward_skeleton_type_GT@5"] / n,
        "retro_skeleton_type_GT@1": counters["retro_skeleton_type_GT@1"] / n,
        "retro_skeleton_type_GT@5": counters["retro_skeleton_type_GT@5"] / n,
        "filled_route_any": counters["filled_route_any"] / n,
        "filled_type_GT@1": counters["filled_type_GT@1"] / n,
        "filled_type_GT@5": counters["filled_type_GT@5"] / n,
        "forward_filled_type_GT@1": counters["forward_filled_type_GT@1"] / n,
        "forward_filled_type_GT@5": counters["forward_filled_type_GT@5"] / n,
        "retro_filled_type_GT@1": counters["retro_filled_type_GT@1"] / n,
        "retro_filled_type_GT@5": counters["retro_filled_type_GT@5"] / n,
        "terminal_GT_reactant_in_top5": counters["terminal_GT_reactant_in_top5"] / n,
        "exact_reaction_in_route_pool": recovery_rate("exact_reaction_in_route_pool"),
        "exact_route_reaction_match_any": recovery_rate("exact_route_reaction_match_any"),
        "gt_reactant_in_route_pool": recovery_rate("gt_reactant_in_route_pool"),
        "candidate_exact_reaction_in_pool": recovery_rate("candidate_exact_reaction_in_pool"),
        "candidate_gt_reactant_in_pool": recovery_rate("candidate_gt_reactant_in_pool"),
        "avg_best_exact_reaction_fraction": recovery_mean("best_exact_reaction_fraction"),
        "avg_best_reaction_edit_distance": recovery_mean("best_reaction_edit_distance"),
        "avg_best_type_edit_distance": recovery_mean("best_type_edit_distance"),
        "avg_candidate_exact_reaction_best_rank": recovery_mean("candidate_exact_reaction_best_candidate_rank"),
        "avg_candidate_gt_reactant_best_rank": recovery_mean("candidate_gt_reactant_best_candidate_rank"),
        "strict_stock_solve_any": counters["strict_stock_solve_any"] / n if check_stock else None,
        "avg_strict_stock_first_rank": metric_mean("strict_stock_first_rank") if check_stock else None,
        "condition_window_success_any": counters["condition_window_success_any"] / n,
        "cascade_compatibility_success_any": counters["cascade_compatibility_success_any"] / n,
        "candidate_source_counts": dict(source_counts),
        "candidate_pool_source_counts": dict(candidate_pool_source_counts),
        "recovery_bottleneck_counts": dict(recovery_bottleneck_counts),
        "runtime_time_bucket_counts": dict(time_bucket_counts),
        "route_tree_runtime_bottleneck_counts": dict(runtime_bottleneck_counts),
        "route_tree_stop_reason_counts": dict(route_tree_stop_counts),
        "route_tree_source_call_counts": dict(route_tree_source_call_counts),
        "route_tree_source_latency_ms": {
            source: round(value, 3)
            for source, value in sorted(route_tree_source_latency_ms.items())
        },
        "route_tree_slow_targets_top20": sorted(slow_targets, key=lambda row: row["elapsed_s"], reverse=True)[:20],
        "avg_candidate_pool_coverage": metric_mean("route_candidate_pool_candidate_pool_coverage"),
        "avg_candidate_pool_total_candidates": metric_mean("route_candidate_pool_total_candidates"),
        "avg_candidate_pool_candidates_per_step": metric_mean("route_candidate_pool_avg_candidates_per_step"),
        "avg_candidate_pool_diversity_score": metric_mean("route_candidate_pool_avg_pool_diversity_score"),
        "avg_candidate_pool_min_diversity_score": metric_mean("route_candidate_pool_min_pool_diversity_score"),
        "avg_candidate_pool_duplicate_reaction_fraction": metric_mean("route_candidate_pool_avg_duplicate_reaction_fraction"),
        "avg_candidate_pool_duplicate_main_reactant_fraction": metric_mean("route_candidate_pool_avg_duplicate_main_reactant_fraction"),
        "avg_candidate_pool_duplicate_reactant_set_fraction": metric_mean("route_candidate_pool_avg_duplicate_reactant_set_fraction"),
        "avg_candidate_pool_single_reactant_set_steps": metric_mean("route_candidate_pool_single_reactant_set_steps"),
        "agent_prior_source_counts": dict(prior_source_counts),
        "avg_route_count": metric_mean("route_count"),
        "avg_route_set_unique_full_signatures": metric_mean("route_set_unique_full_signatures"),
        "avg_route_set_duplicate_route_fraction": metric_mean("route_set_duplicate_route_fraction"),
        "avg_route_set_pairwise_type_distance": metric_mean("route_set_mean_pairwise_type_distance"),
        "avg_route_set_terminal_jaccard_distance": metric_mean("route_set_mean_pairwise_terminal_jaccard_distance"),
        "check_stock": check_stock,
        "total_time_s": round(total_time, 3),
        "avg_time_per_target_s": round(total_time / n, 3),
        "per_domain": {
            d: {
                k: (v / max(stats["n"], 1) if k != "n" else v)
                for k, v in stats.items()
            }
            for d, stats in per_domain.items()
        },
    }


def run_live_benchmark(
    bench_path: str = "data/benchmark_v2_100.json",
    output_path: str = "results/v2/live_benchmark.json",
    model_path: str = "results/shared/skeleton_inpainter/best.pt",
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    n_results: int = 5,
    n_candidates_per_skeleton: int = 2,
    skeleton_samples: int | None = None,
    device: str = "cpu",
    check_stock: bool = False,
    prior_provider: str = "none",
    prior_weight: float = 1.0,
    prior_cache_path: str | None = None,
    search_mode: str = "rerank",
    search_budget: int | None = None,
    constraints: dict[str, Any] | None = None,
    target_log: str = "brief",
    trace_output_path: str | None = None,
) -> dict[str, Any]:
    bench = _load_benchmark_entries(bench_path, limit, shard_index, num_shards)
    prior_cache = load_prior_cache(prior_cache_path)

    stock_checker = _build_stock_checker(check_stock)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        retro_engine = build_live_retro_engine()
        skeleton_model = load_skeleton_model(model_path, device=device)

    target_results = []
    trace_fh = None
    if trace_output_path:
        trace_path = Path(trace_output_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = trace_path.open("w", encoding="utf-8")

    t_all = time.time()
    try:
      for idx, entry in enumerate(bench):
        target = entry["target_smiles"]
        depth = int(entry.get("depth") or len(entry.get("gt_route", [])) or 3)
        domain = entry.get("route_domain") or "chemoenzymatic"
        gt_types = _gt_types(entry)
        retro_gt_types = list(reversed(gt_types))
        type_metric_order = _route_tree_type_metric_order(search_mode)
        primary_gt_types = retro_gt_types if type_metric_order == "retro" else gt_types
        benchmark_index = entry.get("_benchmark_index", idx)

        _emit_target_log(target_log, "start", {
            "ordinal": idx + 1,
            "total": len(bench),
            "index": benchmark_index,
            "target": target,
            "domain": domain,
            "depth": depth,
            "search_mode": search_mode,
        })

        t0 = time.time()
        error = None
        results = []
        agent_prior = None
        policy_retry_meta = None
        route_tree_trace = None
        if trace_fh is not None and search_mode in {"route_tree", "policy_retry"}:
            try:
                from cascade_planner.route_tree.trace import RouteTreeTraceCollector

                route_tree_trace = RouteTreeTraceCollector()
            except Exception:
                route_tree_trace = None
        try:
            agent_prior = get_prior_for_target(target, provider=prior_provider, cache=prior_cache)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if search_mode == "policy_retry":
                    results, policy_retry_meta = _plan_one_target_with_policy_retry(
                        skeleton_model,
                        retro_engine,
                        target=target,
                        depth=depth,
                        domain=domain,
                        model_device=device,
                        n_results=n_results,
                        n_candidates_per_skeleton=n_candidates_per_skeleton,
                        skeleton_samples=skeleton_samples,
                        agent_prior=agent_prior,
                        prior_weight=prior_weight,
                        search_budget=search_budget,
                        stock_checker=stock_checker,
                        constraints=constraints,
                        trace_collector=route_tree_trace,
                    )
                else:
                    results = _plan_one_target(
                        skeleton_model,
                        retro_engine,
                        target=target,
                        depth=depth,
                        domain=domain,
                        model_device=device,
                        n_results=n_results,
                        n_candidates_per_skeleton=n_candidates_per_skeleton,
                        skeleton_samples=skeleton_samples,
                        agent_prior=agent_prior,
                        prior_weight=prior_weight,
                        search_mode=search_mode,
                        search_budget=search_budget,
                        stock_checker=stock_checker,
                        constraints=constraints,
                        trace_collector=route_tree_trace if search_mode == "route_tree" else None,
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - t0
        payload = route_results_payload(
            target,
            results,
            objective="balanced",
            constraints=constraints,
            elapsed_s=elapsed,
            stock_checker=stock_checker,
        )
        if search_mode == "route_tree":
            try:
                from cascade_planner.route_tree.bounded_reservoir import annotate_bounded_reservoir_payload

                annotate_bounded_reservoir_payload(payload)
            except Exception:
                pass
        if policy_retry_meta:
            payload["policy_retry"] = policy_retry_meta

        routes = payload["routes"]
        forward_skel_hit_1 = bool(routes and _type_match(routes[0], gt_types, require_filled=False))
        forward_skel_hit_5 = any(_type_match(r, gt_types, require_filled=False) for r in routes[:5])
        forward_filled_hit_1 = bool(routes and _type_match(routes[0], gt_types, require_filled=True))
        forward_filled_hit_5 = any(_type_match(r, gt_types, require_filled=True) for r in routes[:5])
        retro_skel_hit_1 = bool(routes and _type_match(routes[0], retro_gt_types, require_filled=False))
        retro_skel_hit_5 = any(_type_match(r, retro_gt_types, require_filled=False) for r in routes[:5])
        retro_filled_hit_1 = bool(routes and _type_match(routes[0], retro_gt_types, require_filled=True))
        retro_filled_hit_5 = any(_type_match(r, retro_gt_types, require_filled=True) for r in routes[:5])
        skel_hit_1 = bool(routes and _type_match(routes[0], primary_gt_types, require_filled=False))
        skel_hit_5 = any(_type_match(r, primary_gt_types, require_filled=False) for r in routes[:5])
        filled_hit_1 = bool(routes and _type_match(routes[0], primary_gt_types, require_filled=True))
        filled_hit_5 = any(_type_match(r, primary_gt_types, require_filled=True) for r in routes[:5])
        terminal_gt_hit = any(_terminal_gt_in_route(r, entry) for r in routes[:5])
        recovery = target_recovery_metrics(routes, entry)

        filled_any = any(r.get("metrics", {}).get("filled_route") for r in routes)
        condition_ok = any(
            (r.get("metrics", {}).get("condition") or {}).get("condition_window_success")
            for r in routes
        )
        cascade_ok = any(
            (r.get("metrics", {}).get("cascade_compatibility") or {}).get("cascade_compatibility_success")
            for r in routes
        )
        enzyme_cov = [
            (r.get("metrics", {}).get("enzyme_evidence") or {}).get("enzyme_evidence_coverage")
            for r in routes
        ]
        enzyme_cov = [x for x in enzyme_cov if x is not None]

        stock_solved = [
            r.get("metrics", {}).get("strict_stock_solve")
            for r in routes
            if r.get("metrics", {}).get("strict_stock_solve") is not None
        ]
        strict_stock_first_rank = None
        for ridx, route in enumerate(routes, 1):
            if (route.get("metrics") or {}).get("strict_stock_solve") is True:
                strict_stock_first_rank = ridx
                break

        metrics = {
            "plan": bool(routes),
            "skeleton_type_GT@1": skel_hit_1,
            "skeleton_type_GT@5": skel_hit_5,
            "forward_skeleton_type_GT@1": forward_skel_hit_1,
            "forward_skeleton_type_GT@5": forward_skel_hit_5,
            "retro_skeleton_type_GT@1": retro_skel_hit_1,
            "retro_skeleton_type_GT@5": retro_skel_hit_5,
            "filled_route_any": filled_any,
            "filled_type_GT@1": filled_hit_1,
            "filled_type_GT@5": filled_hit_5,
            "forward_filled_type_GT@1": forward_filled_hit_1,
            "forward_filled_type_GT@5": forward_filled_hit_5,
            "retro_filled_type_GT@1": retro_filled_hit_1,
            "retro_filled_type_GT@5": retro_filled_hit_5,
            "type_metric_order_retro": type_metric_order == "retro",
            "terminal_GT_reactant_in_top5": terminal_gt_hit,
            "condition_window_success_any": condition_ok,
            "cascade_compatibility_success_any": cascade_ok,
            "strict_stock_solve_any": any(stock_solved) if stock_solved else None,
            "strict_stock_first_rank": strict_stock_first_rank,
            "best_enzyme_evidence_coverage": max(enzyme_cov) if enzyme_cov else None,
            "error": error,
        }

        _emit_target_log(target_log, "done", {
            "ordinal": idx + 1,
            "total": len(bench),
            "index": benchmark_index,
            "target": target,
            "domain": domain,
            "depth": depth,
            "elapsed_s": round(elapsed, 3),
            "routes": len(routes),
            "plan": metrics["plan"],
            "type_at_1": skel_hit_1,
            "type_at_5": skel_hit_5,
            "type_order": type_metric_order,
            "filled": filled_any,
            "stock": metrics["strict_stock_solve_any"],
            "condition": condition_ok,
            "compatibility": cascade_ok,
            "exact_reaction_in_route_pool": recovery.get("exact_reaction_in_route_pool"),
            "gt_reactant_in_route_pool": recovery.get("gt_reactant_in_route_pool"),
            "candidate_exact_reaction_in_pool": recovery.get("candidate_exact_reaction_in_pool"),
            "candidate_gt_reactant_in_pool": recovery.get("candidate_gt_reactant_in_pool"),
            "error": error,
        })

        target_results.append({
            "index": benchmark_index,
            "shard_local_index": idx,
            "doi": entry.get("doi"),
            "cascade_id": entry.get("cascade_id"),
            "target_smiles": target,
            "route_domain": domain,
            "depth": depth,
            "gt_types": gt_types,
            "gt_types_retro": retro_gt_types,
            "type_metric_order": type_metric_order,
            "gt_route": entry.get("gt_route", []),
            "agent_prior": summarize_prior(agent_prior, requested_provider=prior_provider),
            "metrics": metrics,
            "route_recovery": recovery,
            "route_critiques": critique_route_payload(payload),
            "policy_retry": policy_retry_meta,
            "planner_output": payload,
        })
        if trace_fh is not None and route_tree_trace is not None:
            trace_context = {
                "schema_version": "route_tree_trace.v1",
                "benchmark": bench_path,
                "benchmark_index": benchmark_index,
                "shard_local_index": idx,
                "target_smiles": target,
                "doi": entry.get("doi"),
                "cascade_id": entry.get("cascade_id"),
                "route_domain": domain,
                "gt_route": entry.get("gt_route", []),
                "planner_error": error,
                "elapsed_s": round(elapsed, 3),
                "n_routes": len(routes),
                "route_metrics": [route.get("metrics") or {} for route in routes],
                "route_model_active": False,
                "same_run_benchmark_trace": True,
            }
            rows = route_tree_trace.to_rows()
            if not rows:
                trace_fh.write(json.dumps({**trace_context, "event": None}, ensure_ascii=False) + "\n")
            else:
                for event in rows:
                    trace_fh.write(json.dumps({**trace_context, "event": event}, ensure_ascii=False) + "\n")
            trace_fh.flush()
    finally:
        if trace_fh is not None:
            trace_fh.close()

    write_prior_cache(prior_cache_path, prior_cache)
    summary = summarize_target_results(target_results, check_stock=bool(stock_checker))
    summary["wall_time_s"] = round(time.time() - t_all, 3)

    output = {
        "metadata": {
            "benchmark": bench_path,
            "model_path": model_path,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "n_results": n_results,
            "n_candidates_per_skeleton": n_candidates_per_skeleton,
            "skeleton_samples": skeleton_samples or n_results,
            "device": device,
            "prior_provider": prior_provider,
            "prior_weight": prior_weight,
            "prior_cache_path": prior_cache_path,
            "search_mode": search_mode,
            "search_budget": search_budget,
            "policy_retry": search_mode == "policy_retry",
            "constraints": constraints,
            "target_log": target_log,
            "skeleton_retrieval_prior": skeleton_retrieval_prior_metadata(),
            "skeleton_reranker": skeleton_reranker_metadata(),
            "retro_cache_stats": retro_engine_cache_stats(retro_engine),
            "metric_note": (
                "filled_type_GT uses filled routes plus type-sequence threshold. "
                "For route_tree, skeleton_type_GT/filled_type_GT use retrosynthesis-order "
                "GT types; forward_* and retro_* type metrics are also reported."
            ),
        },
        "summary": summary,
        "targets": target_results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    return output


def merge_benchmark_outputs(paths: list[str], output_path: str) -> dict[str, Any]:
    merged_targets = []
    metadata = []
    check_stock = False
    for p in paths:
        data = json.loads(Path(p).read_text())
        metadata.append(data.get("metadata", {}))
        merged_targets.extend(data.get("targets", []))
        check_stock = check_stock or bool(data.get("summary", {}).get("check_stock"))

    merged_targets.sort(key=lambda x: x.get("index", 0))
    refresh_recovery_metrics(merged_targets)
    stock_checker = _build_stock_checker(True) if check_stock else None
    refresh_route_metrics(merged_targets, stock_checker=stock_checker)
    output = {
        "metadata": {
            "merged_from": paths,
            "source_metadata": metadata,
            "metric_note": (
                "Merged live benchmark shards. Metrics are recomputed from "
                "target-level route artifacts."
            ),
        },
        "summary": summarize_target_results(merged_targets, check_stock=check_stock),
        "targets": merged_targets,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    return output


def _board_from_exported_route(route: dict[str, Any]) -> CascadeBoard:
    steps = route.get("steps") or []
    target = steps[0].get("product") if steps else None
    board = CascadeBoard.from_n_steps(len(steps), target)
    for key, value in (route.get("global_constraints") or {}).items():
        board.set_global_constraint(str(key), value)
    for idx, step in enumerate(steps):
        slot = board.slots[idx]
        slot.product = step.get("product")
        slot.main_reactant = step.get("main_reactant")
        slot.aux_reactants = list(step.get("aux_reactants") or [])
        slot.reaction_smiles = step.get("reaction_smiles")
        slot.reaction_type = step.get("reaction_type")
        slot.ec = step.get("ec")
        slot.enzyme_uid = step.get("enzyme_uid")
        slot.catalyst = step.get("catalyst")
        slot.T = step.get("T")
        slot.pH = step.get("pH")
        slot.solvent = step.get("solvent")
        slot.evidence = dict(step.get("evidence") or {})
        slot.source = step.get("source") or ""
        scores = step.get("scores") or {}
        slot.e_retro = scores.get("retro")
        slot.e_enzyme = scores.get("enzyme")
        slot.e_condition = scores.get("condition")
        slot.confidence = scores.get("confidence") or 1.0
        slot.fixed_fields = set(step.get("fixed_fields") or [])
        pool = step.get("candidate_pool") or {}
        slot.candidates = list(pool.get("top_candidates") or [])
    return board


def refresh_route_metrics(targets: list[dict[str, Any]], stock_checker=None) -> None:
    """Recompute deterministic route metrics from exported route slots."""
    for target in targets:
        payload = target.get("planner_output") or {}
        routes = payload.get("routes") or []
        for route in routes:
            board = _board_from_exported_route(route)
            route["metrics"] = route_metrics(board, stock_checker=stock_checker)
            if stock_checker is not None:
                for step in route.get("steps") or []:
                    status = {}
                    reactants = []
                    if step.get("main_reactant"):
                        reactants.append(step["main_reactant"])
                    reactants.extend(step.get("aux_reactants") or [])
                    for smi in reactants:
                        status[smi] = bool(stock_checker(smi))
                    if status:
                        step["stock_status"] = status

        stock_values = [
            (route.get("metrics") or {}).get("strict_stock_solve")
            for route in routes
            if (route.get("metrics") or {}).get("strict_stock_solve") is not None
        ]
        first_stock_rank = next(
            (
                idx
                for idx, route in enumerate(routes, 1)
                if (route.get("metrics") or {}).get("strict_stock_solve") is True
            ),
            None,
        )
        enzyme_cov = [
            (route.get("metrics", {}).get("enzyme_evidence") or {}).get("enzyme_evidence_coverage")
            for route in routes
        ]
        enzyme_cov = [x for x in enzyme_cov if x is not None]
        metrics = target.setdefault("metrics", {})
        metrics["filled_route_any"] = any((r.get("metrics") or {}).get("filled_route") for r in routes)
        metrics["condition_window_success_any"] = any(
            ((r.get("metrics") or {}).get("condition") or {}).get("condition_window_success")
            for r in routes
        )
        metrics["cascade_compatibility_success_any"] = any(
            ((r.get("metrics") or {}).get("cascade_compatibility") or {}).get("cascade_compatibility_success")
            for r in routes
        )
        metrics["strict_stock_solve_any"] = any(stock_values) if stock_values else None
        metrics["strict_stock_first_rank"] = first_stock_rank
        metrics["best_enzyme_evidence_coverage"] = max(enzyme_cov) if enzyme_cov else None
        target["route_critiques"] = critique_route_payload(payload)


def refresh_recovery_metrics(targets: list[dict[str, Any]]) -> None:
    """Recompute recovery metrics when merged artifacts carry ground truth routes."""
    for target in targets:
        gt_route = target.get("gt_route")
        if not gt_route:
            continue
        payload = target.get("planner_output") or {}
        routes = payload.get("routes") or []
        target["route_recovery"] = target_recovery_metrics(routes, target)


def apply_stock_to_targets(targets: list[dict[str, Any]]) -> bool:
    """Annotate merged route artifacts with stock solve if ZINC is available."""
    stock_checker = _build_stock_checker(True)
    if stock_checker is None:
        return False
    refresh_route_metrics(targets, stock_checker=stock_checker)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full live CascadeBoard benchmark")
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--output", default="results/v2/live_benchmark.json")
    ap.add_argument("--merge", nargs="+", default=None, help="Merge existing shard JSON files and exit")
    ap.add_argument("--model", default="results/shared/skeleton_inpainter/best.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-results", type=int, default=5)
    ap.add_argument("--n-candidates-per-skeleton", type=int, default=2)
    ap.add_argument("--skeleton-samples", type=int, default=None, help="Number of skeletons sampled before prior reranking")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check-stock", action="store_true")
    ap.add_argument("--prior-provider", default="none", choices=["none", "deterministic", "deepseek"])
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--prior-cache", default=None, help="Optional JSON cache for structured priors; contains no API key")
    ap.add_argument(
        "--search-mode",
        default="rerank",
        choices=["rerank", "stock_aware", "critic_control", "cc_aostar", "hybrid", "stock_rescue", "route_tree", "policy_retry"],
    )
    ap.add_argument("--search-budget", type=int, default=None, help="Skeleton budget for stock-aware, critic-control, or cc_aostar modes")
    ap.add_argument("--constraints-json", default=None, help="Optional JSON file with fixed steps, starting material, or node constraints")
    ap.add_argument("--trace-output", default=None, help="Optional route_tree trace JSONL from the same benchmark run")
    ap.add_argument(
        "--target-log",
        default="brief",
        choices=["none", "brief", "json"],
        help="Emit per-target benchmark progress to stderr",
    )
    args = ap.parse_args()

    if args.merge:
        result = merge_benchmark_outputs(args.merge, args.output)
        if args.check_stock and apply_stock_to_targets(result["targets"]):
            result["summary"] = summarize_target_results(result["targets"], check_stock=True)
            Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result["summary"], indent=2))
        return

    constraints = None
    if args.constraints_json:
        constraints = json.loads(Path(args.constraints_json).read_text(encoding="utf-8"))

    result = run_live_benchmark(
        bench_path=args.bench,
        output_path=args.output,
        model_path=args.model,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        n_results=args.n_results,
        n_candidates_per_skeleton=args.n_candidates_per_skeleton,
        skeleton_samples=args.skeleton_samples,
        device=args.device,
        check_stock=args.check_stock,
        prior_provider=args.prior_provider,
        prior_weight=args.prior_weight,
        prior_cache_path=args.prior_cache,
        search_mode=args.search_mode,
        search_budget=args.search_budget,
        constraints=constraints,
        target_log=args.target_log,
        trace_output_path=args.trace_output,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
