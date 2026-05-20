"""Replay CCTS-v3 evidence reranking on a controller benchmark run.

This offline ablation keeps the generated controller/Hybrid-D route pool fixed.
It only changes the route order within each target, then recomputes top-k route
quality diagnostics from the already exported per-route metrics.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import RDLogger

from cascade_planner.eval.audit_candidate_specific_evidence import _train_bank
from cascade_planner.eval.replay_ccts_v3_on_route_pool import _score_route, _transform_pair_stats


SCHEMA_VERSION = "ccts_v3_controller_run_replay.v1"


def replay_ccts_v3_on_controller_run(
    *,
    run_json: Path,
    program_manifest: Path,
    output_dir: Path,
    alpha_values: tuple[float, ...] = (0.10, 0.30, 0.50),
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
    best_policy: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads(run_json.read_text(encoding="utf-8"))
    train = _train_bank(program_manifest)
    pair_stats = _transform_pair_stats(train)
    product_sim_cache: dict[tuple[str, str], list[float]] = {}

    target_rows = []
    scored_routes = []
    for target in run.get("targets") or []:
        target_scored = []
        routes = ((target.get("planner_output") or {}).get("routes") or [])
        per_route = ((target.get("route_recovery") or {}).get("per_route") or [])
        for native_rank, route in enumerate(routes):
            route_record = _route_record(target, route, native_rank=native_rank)
            scored = _score_route(
                route_record,
                train_bank=train,
                pair_stats=pair_stats,
                product_sim_cache=product_sim_cache,
            )
            scored = _add_controller_route_scores(scored)
            target_scored.append(
                {
                    "route_index": native_rank,
                    "route_id": route_record["route_id"],
                    "scores": _score_subset(scored),
                    "metrics": route.get("metrics") or {},
                    "recovery": per_route[native_rank] if native_rank < len(per_route) and isinstance(per_route[native_rank], dict) else {},
                    "source": _route_source(route),
                    "n_steps": len(route.get("steps") or []),
                }
            )
            scored_routes.append(scored)
        target_rows.append({"target": target, "routes": target_scored})

    policies = _policy_orders(target_rows, alpha_values=alpha_values)
    summaries = {
        name: _policy_summary(target_rows, orders=orders, top_ks=top_ks)
        for name, orders in policies.items()
    }
    deltas = {
        name: _policy_delta(summary, summaries["native"])
        for name, summary in summaries.items()
        if name != "native"
    }
    selected_policy = best_policy or _select_best_policy(summaries)
    selected_run = _reranked_run(run, policies[selected_policy], selected_policy=selected_policy)

    scored_path = output_dir / "hybrid_d_ccts_v3_scored_routes.jsonl"
    _write_jsonl(scored_path, scored_routes)
    selected_run_path = output_dir / f"hybrid_d_ccts_v3_{selected_policy}_run.json"
    selected_run_path.write_text(json.dumps(selected_run, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "run_json": str(run_json),
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "alpha_values": list(alpha_values),
            "top_ks": list(top_ks),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "fixed Hybrid-D route pool; CCTS-v3 train-split evidence only; no generation and no hard pruning",
        },
        "counts": _counts(target_rows),
        "score_summary": _score_summary(scored_routes),
        "policy_summaries": summaries,
        "deltas_vs_native": deltas,
        "selected_policy": selected_policy,
        "outputs": {
            "scored_routes_jsonl": str(scored_path),
            "selected_reranked_run_json": str(selected_run_path),
        },
        "interpretation": _interpretation(summaries, deltas, selected_policy),
    }
    report_path = output_dir / "hybrid_d_ccts_v3_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _route_record(target: dict[str, Any], route: dict[str, Any], *, native_rank: int) -> dict[str, Any]:
    target_index = target.get("index")
    steps = []
    for idx, step in enumerate(route.get("steps") or []):
        if not isinstance(step, dict):
            continue
        product = str(step.get("product") or step.get("product_smiles") or target.get("target_smiles") or "")
        main = str(step.get("main_reactant") or "")
        aux = [str(value) for value in step.get("aux_reactants") or [] if value]
        reactants = [value for value in [main, *aux] if value]
        rxn = str(step.get("reaction_smiles") or "")
        transform = _step_transform(step)
        steps.append(
            {
                "step_index": int(step.get("index") if step.get("index") is not None else idx) + 1,
                "step_id": f"step_{idx + 1}",
                "product_smiles": product,
                "products": [product] if product else [],
                "main_reactant": main,
                "reactants": reactants,
                "rxn_smiles": rxn,
                "transformation_superclass": transform,
                "source": step.get("source"),
            }
        )
    return {
        "route_id": f"target_{target_index}_route_{native_rank}",
        "target_id": str(target_index),
        "target_smiles": target.get("target_smiles"),
        "native_rank": native_rank,
        "original_native_rank": native_rank,
        "native_score": route.get("score"),
        "stock_closed": ((route.get("metrics") or {}).get("strict_stock_solve")),
        "route_source": _route_source(route),
        "n_steps": len(steps),
        "steps": steps,
        "terminal_reactants": ((route.get("metrics") or {}).get("terminal_reactants") or []),
    }


def _step_transform(step: dict[str, Any]) -> str:
    for key in ("reaction_type", "transformation_superclass"):
        value = str(step.get(key) or "").strip()
        if value and value.lower() != "unknown":
            return value
    interp = step.get("reaction_interpretation") or {}
    value = str(interp.get("reaction_class") or "").strip()
    return value if value and value.lower() != "unknown" else "unknown"


def _route_source(route: dict[str, Any]) -> str:
    broad = route.get("broad_reservoir") or {}
    if broad.get("source"):
        return str(broad.get("source"))
    sources = Counter(str(step.get("source") or "") for step in route.get("steps") or [] if isinstance(step, dict))
    return sources.most_common(1)[0][0] if sources else "unknown"


def _add_controller_route_scores(scored: dict[str, Any]) -> dict[str, Any]:
    row = dict(scored)
    evidence = row.get("ccts_v3_route_evidence") or {}
    step_scores = evidence.get("step_scores") or []
    block_scores = evidence.get("block_scores") or []
    any_step = [float(step.get("candidate_nearest_any_transition_sim") or 0.0) for step in step_scores if isinstance(step, dict)]
    ctx_step = [float(step.get("candidate_nearest_context_transform_sim") or 0.0) for step in step_scores if isinstance(step, dict)]
    ctx_or_any = [
        float(step.get("candidate_nearest_context_transform_sim") or step.get("candidate_nearest_any_transition_sim") or 0.0)
        for step in step_scores
        if isinstance(step, dict)
    ]
    any_block = [float(block.get("any_structural_score") or 0.0) for block in block_scores if isinstance(block, dict)]
    row["ccts_v3_step_any_mean"] = _mean(any_step)
    row["ccts_v3_step_any_max"] = max(any_step, default=0.0)
    row["ccts_v3_step_context_or_any_mean"] = _mean(ctx_or_any)
    row["ccts_v3_step_context_mean"] = _mean(ctx_step)
    row["ccts_v3_best_block_any_sim"] = max(any_block, default=0.0)
    row["ccts_v3_best_route_evidence"] = max(
        float(row.get("ccts_v3_best_block_context_sim") or 0.0),
        float(row.get("ccts_v3_step_context_or_any_mean") or 0.0),
        float(row.get("ccts_v3_best_block_any_sim") or 0.0),
    )
    return row


def _score_subset(scored: dict[str, Any]) -> dict[str, float]:
    keys = (
        "ccts_v3_best_block_context_sim",
        "ccts_v3_best_block_any_sim",
        "ccts_v3_step_context_mean",
        "ccts_v3_step_context_or_any_mean",
        "ccts_v3_step_any_mean",
        "ccts_v3_step_any_max",
        "ccts_v3_best_inferred_pair_score",
        "ccts_v3_best_route_evidence",
    )
    return {key: float(scored.get(key) or 0.0) for key in keys}


def _policy_orders(target_rows: list[dict[str, Any]], *, alpha_values: tuple[float, ...]) -> dict[str, dict[int, list[int]]]:
    policies: dict[str, dict[int, list[int]]] = {"native": {}}
    for score_key in (
        "ccts_v3_step_any_mean",
        "ccts_v3_step_context_or_any_mean",
        "ccts_v3_best_block_any_sim",
        "ccts_v3_best_route_evidence",
    ):
        policies[f"evidence_only__{score_key}"] = {}
    for alpha in alpha_values:
        tag = f"alpha{int(round(alpha * 100)):03d}"
        for score_key in ("ccts_v3_step_any_mean", "ccts_v3_step_context_or_any_mean", "ccts_v3_best_route_evidence"):
            policies[f"blend__{score_key}__{tag}"] = {}
            policies[f"stock_guarded_blend__{score_key}__{tag}"] = {}
            policies[f"native_stock_guarded_blend__{score_key}__{tag}"] = {}
            policies[f"native_quality_guarded_blend__{score_key}__{tag}"] = {}
    for target_idx, row in enumerate(target_rows):
        routes = row["routes"]
        policies["native"][target_idx] = [route["route_index"] for route in sorted(routes, key=lambda item: item["route_index"])]
        for score_key in (
            "ccts_v3_step_any_mean",
            "ccts_v3_step_context_or_any_mean",
            "ccts_v3_best_block_any_sim",
            "ccts_v3_best_route_evidence",
        ):
            policies[f"evidence_only__{score_key}"][target_idx] = _sort_indices(routes, score_key=score_key)
        native_values = np.asarray([-float(route["route_index"]) for route in routes], dtype=np.float64)
        native_z = _standardize(native_values)
        score_arrays = {
            score_key: _standardize(np.asarray([float(route["scores"].get(score_key) or 0.0) for route in routes], dtype=np.float64))
            for score_key in ("ccts_v3_step_any_mean", "ccts_v3_step_context_or_any_mean", "ccts_v3_best_route_evidence")
        }
        for alpha in alpha_values:
            tag = f"alpha{int(round(alpha * 100)):03d}"
            for score_key, z_scores in score_arrays.items():
                blended = native_z + float(alpha) * z_scores
                policies[f"blend__{score_key}__{tag}"][target_idx] = _sort_indices_with_values(routes, blended)
                policies[f"stock_guarded_blend__{score_key}__{tag}"][target_idx] = _sort_indices_with_values(routes, blended, stock_guard=True)
                policies[f"native_stock_guarded_blend__{score_key}__{tag}"][target_idx] = _sort_indices_with_values(
                    routes,
                    blended,
                    native_stock_guard=True,
                )
                policies[f"native_quality_guarded_blend__{score_key}__{tag}"][target_idx] = _sort_indices_with_values(
                    routes,
                    blended,
                    native_quality_guard=True,
                )
    return policies


def _sort_indices(routes: list[dict[str, Any]], *, score_key: str) -> list[int]:
    return [
        route["route_index"]
        for route in sorted(
            routes,
            key=lambda route: (-float(route["scores"].get(score_key) or 0.0), route["route_index"], route["route_id"]),
        )
    ]


def _sort_indices_with_values(
    routes: list[dict[str, Any]],
    values: np.ndarray,
    *,
    stock_guard: bool = False,
    native_stock_guard: bool = False,
    native_quality_guard: bool = False,
) -> list[int]:
    enriched = []
    native = min(routes, key=lambda route: route["route_index"]) if routes else {}
    native_stock = _route_metric(native, "strict_stock_solve") is True
    native_condition = (((native.get("metrics") or {}).get("condition") or {}).get("condition_window_success") is True)
    native_cascade = (((native.get("metrics") or {}).get("cascade_compatibility") or {}).get("cascade_compatibility_success") is True)
    for route, value in zip(routes, values):
        stock = bool((route.get("metrics") or {}).get("strict_stock_solve"))
        condition = (((route.get("metrics") or {}).get("condition") or {}).get("condition_window_success") is True)
        cascade = (((route.get("metrics") or {}).get("cascade_compatibility") or {}).get("cascade_compatibility_success") is True)
        native_stock_penalty = int(native_stock and not stock)
        native_quality_penalty = (
            int(native_stock and not stock)
            + int(native_condition and not condition)
            + int(native_cascade and not cascade)
        )
        enriched.append((route, float(value), stock, native_stock_penalty, native_quality_penalty))
    return [
        item[0]["route_index"]
        for item in sorted(
            enriched,
            key=lambda item: (
                -int(item[2]) if stock_guard else 0,
                item[3] if native_stock_guard else 0,
                item[4] if native_quality_guard else 0,
                -item[1],
                item[0]["route_index"],
                item[0]["route_id"],
            ),
        )
    ]


def _policy_summary(target_rows: list[dict[str, Any]], *, orders: dict[int, list[int]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    n_targets = len(target_rows)
    summary: dict[str, Any] = {
        "n_targets": n_targets,
        "top1_changed_targets": 0,
        "top1_change_counts": {},
        "top1_source_counts": {},
        "avg_selected_original_rank_at1": 0.0,
    }
    change_counts = Counter()
    top1_sources = Counter()
    selected_ranks = []
    for target_idx, row in enumerate(target_rows):
        order = orders.get(target_idx) or []
        if not order:
            continue
        selected_ranks.append(order[0])
        top1_sources[str(_route_by_index(row, order[0]).get("source") or "unknown")] += 1
        if order[0] != 0:
            summary["top1_changed_targets"] += 1
            change_counts.update(_top1_change_labels(row, order[0]))
    summary["top1_change_counts"] = dict(change_counts)
    summary["top1_source_counts"] = dict(top1_sources)
    summary["avg_selected_original_rank_at1"] = round(_mean(selected_ranks), 6)
    for k in top_ks:
        summary.update(_topk_metrics(target_rows, orders=orders, k=k))
    return summary


def _topk_metrics(target_rows: list[dict[str, Any]], *, orders: dict[int, list[int]], k: int) -> dict[str, Any]:
    counts = Counter()
    exact_fracs = []
    for target_idx, row in enumerate(target_rows):
        selected = [_route_by_index(row, idx) for idx in (orders.get(target_idx) or [])[:k]]
        if not selected:
            continue
        counts[f"targets_with_routes_at{k}"] += 1
        counts[f"strict_stock_solve_at{k}"] += int(any(_route_metric(route, "strict_stock_solve") is True for route in selected))
        counts[f"route_solved_at{k}"] += int(any(_route_metric(route, "route_solved") is True for route in selected))
        counts[f"condition_window_success_at{k}"] += int(any(((route.get("metrics") or {}).get("condition") or {}).get("condition_window_success") is True for route in selected))
        counts[f"cascade_compatibility_success_at{k}"] += int(
            any(((route.get("metrics") or {}).get("cascade_compatibility") or {}).get("cascade_compatibility_success") is True for route in selected)
        )
        counts[f"exact_reaction_hit_at{k}"] += int(any(_recovery_bool(route, "exact_reaction_hits") for route in selected))
        counts[f"exact_route_match_at{k}"] += int(any(_recovery_bool(route, "exact_route_reaction_match") for route in selected))
        counts[f"gt_reactant_hit_at{k}"] += int(any(_recovery_bool(route, "gt_reactant_hit") for route in selected))
        counts[f"candidate_exact_reaction_hit_at{k}"] += int(any(_recovery_bool(route, "candidate_exact_reaction_hit") for route in selected))
        counts[f"candidate_gt_reactant_hit_at{k}"] += int(any(_recovery_bool(route, "candidate_gt_reactant_hit") for route in selected))
        top1_recovery = selected[0].get("recovery") or {}
        exact_fracs.append(float(top1_recovery.get("exact_reaction_fraction") or 0.0))
    out = {}
    denom = max(len(target_rows), 1)
    for key, value in counts.items():
        out[key] = round(float(value) / denom, 6) if key != f"targets_with_routes_at{k}" else int(value)
    if k == 1:
        out["top1_mean_exact_reaction_fraction"] = round(_mean(exact_fracs), 6)
    return out


def _route_by_index(target_row: dict[str, Any], route_index: int) -> dict[str, Any]:
    for route in target_row["routes"]:
        if route["route_index"] == route_index:
            return route
    return {}


def _route_metric(route: dict[str, Any], key: str) -> Any:
    return (route.get("metrics") or {}).get(key)


def _recovery_bool(route: dict[str, Any], key: str) -> bool:
    value = (route.get("recovery") or {}).get(key)
    if key.endswith("_hits") and isinstance(value, int):
        return value > 0
    return bool(value)


def _top1_change_labels(target_row: dict[str, Any], selected_idx: int) -> list[str]:
    native = _route_by_index(target_row, 0)
    selected = _route_by_index(target_row, selected_idx)
    checks = {
        "stock": (_route_metric(native, "strict_stock_solve") is True, _route_metric(selected, "strict_stock_solve") is True),
        "gt_reactant": (_recovery_bool(native, "gt_reactant_hit"), _recovery_bool(selected, "gt_reactant_hit")),
        "exact_reaction": (_recovery_bool(native, "exact_reaction_hits"), _recovery_bool(selected, "exact_reaction_hits")),
        "cascade_compat": (
            (((native.get("metrics") or {}).get("cascade_compatibility") or {}).get("cascade_compatibility_success") is True),
            (((selected.get("metrics") or {}).get("cascade_compatibility") or {}).get("cascade_compatibility_success") is True),
        ),
    }
    labels = []
    for name, (old, new) in checks.items():
        if old is False and new is True:
            labels.append(f"{name}_gain")
        elif old is True and new is False:
            labels.append(f"{name}_loss")
    return labels or ["changed_neutral"]


def _policy_delta(summary: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "strict_stock_solve_at1",
        "strict_stock_solve_at3",
        "gt_reactant_hit_at1",
        "gt_reactant_hit_at3",
        "exact_reaction_hit_at1",
        "exact_reaction_hit_at3",
        "cascade_compatibility_success_at1",
        "cascade_compatibility_success_at3",
        "top1_mean_exact_reaction_fraction",
        "avg_selected_original_rank_at1",
    ]
    return {
        key: round(float(summary.get(key) or 0.0) - float(native.get(key) or 0.0), 6)
        for key in keys
        if key in summary or key in native
    }


def _select_best_policy(summaries: dict[str, Any]) -> str:
    native = summaries.get("native") or {}
    native_stock = float(native.get("strict_stock_solve_at1") or 0.0)
    native_cascade = float(native.get("cascade_compatibility_success_at1") or 0.0)
    native_exact1 = float(native.get("exact_reaction_hit_at1") or 0.0)
    candidates = []
    for name, summary in summaries.items():
        stock = float(summary.get("strict_stock_solve_at1") or 0.0)
        cascade = float(summary.get("cascade_compatibility_success_at1") or 0.0)
        exact1 = float(summary.get("exact_reaction_hit_at1") or 0.0)
        no_regression = stock >= native_stock and cascade >= native_cascade and exact1 >= native_exact1
        candidates.append(
            (
                int(no_regression),
                int(stock >= native_stock),
                int(cascade >= native_cascade),
                int(exact1 >= native_exact1),
                float(summary.get("gt_reactant_hit_at3") or 0.0),
                float(summary.get("gt_reactant_hit_at1") or 0.0),
                float(summary.get("exact_reaction_hit_at3") or 0.0),
                exact1,
                -float(summary.get("avg_selected_original_rank_at1") or 0.0),
                name,
            )
        )
    return max(candidates)[-1] if candidates else "native"


def _reranked_run(run: dict[str, Any], orders: dict[int, list[int]], *, selected_policy: str) -> dict[str, Any]:
    out = copy.deepcopy(run)
    out.setdefault("metadata", {})["ccts_v3_rerank_policy"] = selected_policy
    for target_idx, target in enumerate(out.get("targets") or []):
        routes = ((target.get("planner_output") or {}).get("routes") or [])
        order = orders.get(target_idx) or list(range(len(routes)))
        by_idx = {idx: route for idx, route in enumerate(routes)}
        reranked = []
        for new_rank, old_idx in enumerate(order):
            route = by_idx.get(old_idx)
            if route is None:
                continue
            route["ccts_v3_rerank"] = {"policy": selected_policy, "original_rank": old_idx, "new_rank": new_rank}
            reranked.append(route)
        if target.get("planner_output"):
            target["planner_output"]["routes"] = reranked
    return out


def _counts(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    routes = [route for row in target_rows for route in row["routes"]]
    return {
        "targets": len(target_rows),
        "routes": len(routes),
        "single_step_routes": sum(1 for route in routes if int(route.get("n_steps") or 0) <= 1),
        "multi_step_routes": sum(1 for route in routes if int(route.get("n_steps") or 0) > 1),
        "stock_closed_routes": sum(1 for route in routes if _route_metric(route, "strict_stock_solve") is True),
        "source_counts": dict(Counter(str(route.get("source") or "unknown") for route in routes)),
    }


def _score_summary(scored_routes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "ccts_v3_step_any_mean",
        "ccts_v3_step_context_or_any_mean",
        "ccts_v3_best_block_any_sim",
        "ccts_v3_best_route_evidence",
    )
    return {key: _numeric([float(route.get(key) or 0.0) for route in scored_routes]) for key in keys}


def _numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p75": round(float(np.quantile(arr, 0.75)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _interpretation(summaries: dict[str, Any], deltas: dict[str, Any], selected_policy: str) -> dict[str, Any]:
    selected = summaries.get(selected_policy) or {}
    native = summaries.get("native") or {}
    selected_delta = deltas.get(selected_policy) or {}
    return {
        "selected_policy": selected_policy,
        "stock_top1_regressed": float(selected_delta.get("strict_stock_solve_at1") or 0.0) < 0,
        "cascade_top1_regressed": float(selected_delta.get("cascade_compatibility_success_at1") or 0.0) < 0,
        "exact_top1_regressed": float(selected_delta.get("exact_reaction_hit_at1") or 0.0) < 0,
        "gt_top3_delta": selected_delta.get("gt_reactant_hit_at3"),
        "gt_top1_delta": selected_delta.get("gt_reactant_hit_at1"),
        "exact_top3_delta": selected_delta.get("exact_reaction_hit_at3"),
        "exact_top1_delta": selected_delta.get("exact_reaction_hit_at1"),
        "native_top1_gt": native.get("gt_reactant_hit_at1"),
        "selected_top1_gt": selected.get("gt_reactant_hit_at1"),
        "native_top3_gt": native.get("gt_reactant_hit_at3"),
        "selected_top3_gt": selected.get("gt_reactant_hit_at3"),
        "safe_claim": (
            "CCTS reranking improves or preserves selected top-k diagnostics on this fixed Hybrid-D pool"
            if float(selected_delta.get("gt_reactant_hit_at3") or 0.0) >= 0
            and float(selected_delta.get("strict_stock_solve_at1") or 0.0) >= 0
            and float(selected_delta.get("cascade_compatibility_success_at1") or 0.0) >= 0
            and float(selected_delta.get("exact_reaction_hit_at1") or 0.0) >= 0
            else "CCTS reranking is not yet a safe replacement for Hybrid-D native ordering"
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hybrid-D + CCTS-v3 Replay",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Selected Policy",
        "",
        "```json",
        json.dumps(report.get("interpretation") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Policy Comparison",
        "",
        "| Policy | stock@1 | GT@1 | GT@3 | exact@1 | exact@3 | cascade@1 | changed top1 | avg original rank@1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted((report.get("policy_summaries") or {}).items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row.get("strict_stock_solve_at1")),
                    str(row.get("gt_reactant_hit_at1")),
                    str(row.get("gt_reactant_hit_at3")),
                    str(row.get("exact_reaction_hit_at1")),
                    str(row.get("exact_reaction_hit_at3")),
                    str(row.get("cascade_compatibility_success_at1")),
                    str(row.get("top1_changed_targets")),
                    str(row.get("avg_selected_original_rank_at1")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Deltas Vs Native", "", "```json", json.dumps(report.get("deltas_vs_native") or {}, indent=2, ensure_ascii=False)[:10000], "```", ""])
    return "\n".join(lines)


def _standardize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-9:
        return values * 0.0
    return (values - float(values.mean())) / std


def _mean(values: list[float] | np.ndarray) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    vals = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    return vals or (0.10, 0.30, 0.50)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    vals = tuple(sorted({int(part.strip()) for part in str(value).split(",") if part.strip()}))
    return vals or (1, 3, 5, 10)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Replay CCTS-v3 evidence reranking on a controller run")
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--alphas", default="0.10,0.30,0.50")
    ap.add_argument("--top-ks", default="1,3,5,10")
    ap.add_argument("--best-policy")
    args = ap.parse_args()
    report = replay_ccts_v3_on_controller_run(
        run_json=Path(args.run_json),
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        alpha_values=_parse_float_tuple(args.alphas),
        top_ks=_parse_int_tuple(args.top_ks),
        best_policy=args.best_policy,
    )
    print(json.dumps({"selected_policy": report["selected_policy"], "interpretation": report["interpretation"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
