"""Replay runtime-safe CCTS-v3 reranking on a ChemEnzy native route pool.

This is an offline selector experiment.  The ChemEnzy generated route pool is
kept fixed; CCTS only changes route order inside each target.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import RDLogger

from cascade_planner.eval.audit_candidate_specific_evidence import _train_bank
from cascade_planner.eval.build_ccts_v3_runtime_evidence_cache import _runtime_evidence_scores
from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit
from cascade_planner.eval.rerank_native_routes_with_v4_value import (
    _audit_delta,
    _audit_summary,
    _gt_recovery,
    _ranked_product_metrics,
    _read_rows,
    _routes_for_target,
)
from cascade_planner.eval.train_ccts_v3_runtime_pairwise_ranker import FittedRuntimeModel, _feature_row


SCHEMA_VERSION = "ccts_v3_runtime_native_pool_replay.v1"
DEFAULT_MODEL_NAME = "runtime_pairwise_block_supported_positive_label__runtime_evidence_only"


def replay_ccts_v3_runtime_on_native_pool(
    *,
    native_pool: Path,
    program_manifest: Path,
    model_pickle: Path,
    output_dir: Path,
    benchmark: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    alpha_values: tuple[float, ...] = (0.10, 0.30, 0.50),
    top_k: int | None = None,
    best_policy: str | None = None,
    write_all_policy_runs: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_rows = _read_rows(benchmark) if benchmark else None
    native_run = _load_native_pool(native_pool, top_k=top_k)
    train = _train_bank(program_manifest)
    model_payload = _load_model(model_pickle, model_name)
    product_sim_cache: dict[tuple[str, str], list[float]] = {}

    scored_targets = []
    scored_routes = []
    for target_idx, target in enumerate(native_run.get("targets") or []):
        routes = []
        for native_rank, route in enumerate(_routes_for_target(target)):
            scored = _score_route_runtime(
                route,
                native_rank=native_rank,
                train_bank=train,
                product_sim_cache=product_sim_cache,
                model_payload=model_payload,
            )
            routes.append(scored)
            scored_routes.append(
                {
                    "target_index": target_idx,
                    "target_id": target.get("target_id") or target.get("cascade_id") or target.get("index"),
                    "target_smiles": target.get("target_smiles"),
                    "route_id": scored.get("route_id"),
                    "native_rank": native_rank,
                    "scores": _score_subset(scored),
                    "n_steps": len(scored.get("steps") or []),
                    "stock_closed": scored.get("stock_closed"),
                }
            )
        target_out = dict(target)
        target_out["routes"] = routes
        target_out["route_count"] = len(routes)
        scored_targets.append(target_out)

    scored_run = {
        "metadata": {
            **(native_run.get("metadata") or {}),
            "reranker": "ccts_v3_runtime_native_pool",
            "source_native_pool": str(native_pool),
            "program_manifest": str(program_manifest),
            "model_pickle": str(model_pickle),
            "model_name": model_name,
            "top_k_input": top_k,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": _run_summary(scored_targets),
        "targets": scored_targets,
    }
    scored_path = output_dir / "ccts_v3_runtime_scored_native_pool.json"
    scored_path.write_text(json.dumps(scored_run, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(output_dir / "ccts_v3_runtime_scored_routes.jsonl", scored_routes)

    policies = _policy_orders(scored_targets, alpha_values=alpha_values)
    policy_runs = {name: _reranked_run(scored_run, orders, selected_policy=name) for name, orders in policies.items()}
    policy_summaries = {
        name: _evaluate_policy(run, benchmark_rows=benchmark_rows)
        for name, run in policy_runs.items()
    }
    deltas = {
        name: _policy_delta(summary, policy_summaries["native"])
        for name, summary in policy_summaries.items()
        if name != "native"
    }
    selected_policy = best_policy or _select_best_policy(policy_summaries)
    selected_run_path = output_dir / f"ccts_v3_runtime_{selected_policy}_reranked.json"
    policy_runs[selected_policy]["metadata"]["selected_policy"] = selected_policy
    selected_run_path.write_text(json.dumps(policy_runs[selected_policy], indent=2, ensure_ascii=False), encoding="utf-8")
    policy_run_paths: dict[str, str] = {}
    if write_all_policy_runs:
        policy_dir = output_dir / "policy_runs"
        policy_dir.mkdir(parents=True, exist_ok=True)
        for name, run in policy_runs.items():
            safe_name = name.replace("/", "_")
            path = policy_dir / f"{safe_name}.json"
            run["metadata"]["selected_policy"] = name
            path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
            policy_run_paths[name] = str(path)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "native_pool": str(native_pool),
            "program_manifest": str(program_manifest),
            "model_pickle": str(model_pickle),
            "model_name": model_name,
            "benchmark": str(benchmark) if benchmark else None,
            "output_dir": str(output_dir),
            "alpha_values": list(alpha_values),
            "top_k": top_k,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "fixed ChemEnzy native route pool; runtime-safe CCTS evidence/model only; no generation",
            "selection_gate": "artifact/trivial top3 must not increase; generic top3 may increase by at most 0.05; then maximize top3/top1 usable",
        },
        "counts": _counts(scored_run),
        "score_summary": _score_summary(scored_routes),
        "policy_summaries": policy_summaries,
        "deltas_vs_native": deltas,
        "selected_policy": selected_policy,
        "outputs": {
            "scored_run": str(scored_path),
            "scored_routes_jsonl": str(output_dir / "ccts_v3_runtime_scored_routes.jsonl"),
            "selected_reranked_run": str(selected_run_path),
            "policy_runs": policy_run_paths,
        },
        "interpretation": _interpretation(policy_summaries, deltas, selected_policy),
    }
    report_path = output_dir / "ccts_v3_runtime_native_pool_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _load_native_pool(path: Path, *, top_k: int | None) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                route = json.loads(line)
                if not isinstance(route, dict):
                    continue
                key = str(route.get("target_id") or route.get("target_smiles") or len(groups))
                groups[key].append(route)
        targets = []
        for idx, (key, routes) in enumerate(sorted(groups.items(), key=lambda item: _group_sort_key(item[0]))):
            sorted_routes = sorted(routes, key=lambda route: int(route.get("native_rank") or 0))
            if top_k is not None and top_k > 0:
                sorted_routes = sorted_routes[:top_k]
            target_smiles = str((sorted_routes[0] if sorted_routes else {}).get("target_smiles") or "")
            targets.append(
                {
                    "index": idx,
                    "target_id": key,
                    "target_smiles": target_smiles,
                    "routes": sorted_routes,
                    "route_count": len(sorted_routes),
                }
            )
        return {
            "metadata": {
                "source": str(path),
                "dataset": "native_route_pool_jsonl",
                "top_k": top_k,
            },
            "summary": _run_summary(targets),
            "targets": targets,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object or JSONL route pool: {path}")
    if top_k is None or top_k <= 0:
        return data
    targets = []
    for target in data.get("targets") or []:
        payload = dict(target)
        routes = _routes_for_target(target)[:top_k]
        if isinstance((payload.get("planner_output") or {}).get("routes"), list):
            planner = dict(payload.get("planner_output") or {})
            planner["routes"] = routes
            planner["n_results"] = len(routes)
            payload["planner_output"] = planner
        payload["routes"] = routes
        payload["route_count"] = len(routes)
        targets.append(payload)
    return {**data, "summary": _run_summary(targets), "targets": targets}


def _score_route_runtime(
    route: dict[str, Any],
    *,
    native_rank: int,
    train_bank: dict[str, Any],
    product_sim_cache: dict[tuple[str, str], list[float]],
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    out = dict(route)
    out["original_native_rank"] = int(route.get("original_native_rank", route.get("native_rank", native_rank)) or 0)
    out["native_rank"] = native_rank
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    step_scores = []
    for idx, step in enumerate(steps):
        previous_transform = _step_transform(steps[idx - 1]) if idx > 0 else ""
        next_transform = _step_transform(steps[idx + 1]) if idx + 1 < len(steps) else ""
        product = _step_product(step)
        main = _main_reactant(step)
        evidence = _runtime_evidence_scores(
            product=product,
            candidate_main=main,
            previous_transform=previous_transform,
            next_transform=next_transform,
            train_bank=train_bank,
            product_sim_cache=product_sim_cache,
        )
        feature_row = {
            "candidate_rank": native_rank + 1,
            "candidate_score": _step_or_route_score(step, route),
            "candidate_reactants": _step_reactants(step),
            **evidence,
        }
        model_score = _runtime_model_score(model_payload, feature_row)
        step_scores.append(
            {
                "step_index": step.get("step_index") or idx + 1,
                "step_id": step.get("step_id") or f"step_{idx + 1}",
                "product_smiles": product,
                "candidate_main_reactant": main,
                "previous_transform": previous_transform,
                "next_transform": next_transform,
                "runtime_model_score": model_score,
                **evidence,
            }
        )
    any_scores = [float(row.get("runtime_nearest_any_transition_sim") or 0.0) for row in step_scores]
    pair_scores = [float(row.get("runtime_nearest_pair_compatible_sim") or 0.0) for row in step_scores]
    model_scores = [float(row.get("runtime_model_score") or 0.0) for row in step_scores]
    out["ccts_v3_runtime_route_evidence"] = {
        "n_steps": len(steps),
        "step_any_mean": _mean(any_scores),
        "step_any_max": max(any_scores, default=0.0),
        "step_pair_mean": _mean(pair_scores),
        "step_pair_max": max(pair_scores, default=0.0),
        "step_model_mean": _mean(model_scores),
        "step_model_max": max(model_scores, default=0.0),
        "step_scores": step_scores,
    }
    out["ccts_v3_runtime_step_any_mean"] = _mean(any_scores)
    out["ccts_v3_runtime_step_any_max"] = max(any_scores, default=0.0)
    out["ccts_v3_runtime_step_pair_mean"] = _mean(pair_scores)
    out["ccts_v3_runtime_step_pair_max"] = max(pair_scores, default=0.0)
    out["ccts_v3_runtime_model_mean"] = _mean(model_scores)
    out["ccts_v3_runtime_model_max"] = max(model_scores, default=0.0)
    out["ccts_v3_runtime_best_route_evidence"] = max(
        out["ccts_v3_runtime_step_any_mean"],
        out["ccts_v3_runtime_step_pair_max"],
        out["ccts_v3_runtime_model_mean"],
    )
    return out


def _policy_orders(targets: list[dict[str, Any]], *, alpha_values: tuple[float, ...]) -> dict[str, dict[int, list[int]]]:
    score_keys = (
        "ccts_v3_runtime_step_any_mean",
        "ccts_v3_runtime_step_pair_max",
        "ccts_v3_runtime_model_mean",
        "ccts_v3_runtime_best_route_evidence",
    )
    policies: dict[str, dict[int, list[int]]] = {"native": {}}
    for score_key in score_keys:
        policies[f"evidence_only__{score_key}"] = {}
    policies["trained_selected__chem_any030_model005"] = {}
    policies["trained_blockmrr__chem_model030"] = {}
    for alpha in alpha_values:
        tag = f"alpha{int(round(alpha * 100)):03d}"
        for score_key in score_keys:
            policies[f"blend__{score_key}__{tag}"] = {}
    for target_idx, target in enumerate(targets):
        routes = _routes_for_target(target)
        policies["native"][target_idx] = [idx for idx, _route in enumerate(routes)]
        for score_key in score_keys:
            policies[f"evidence_only__{score_key}"][target_idx] = _sort_indices(routes, score_key=score_key)
        native_z = _standardize(np.asarray([-idx for idx, _route in enumerate(routes)], dtype=np.float64))
        any_z = _standardize(np.asarray([float(route.get("ccts_v3_runtime_step_any_mean") or 0.0) for route in routes], dtype=np.float64))
        model_z = _standardize(np.asarray([float(route.get("ccts_v3_runtime_model_mean") or 0.0) for route in routes], dtype=np.float64))
        trained_selected = _standardize(native_z + 0.30 * any_z) + 0.05 * model_z
        trained_blockmrr = native_z + 0.30 * model_z
        policies["trained_selected__chem_any030_model005"][target_idx] = _sort_indices_with_values(routes, trained_selected)
        policies["trained_blockmrr__chem_model030"][target_idx] = _sort_indices_with_values(routes, trained_blockmrr)
        for alpha in alpha_values:
            tag = f"alpha{int(round(alpha * 100)):03d}"
            for score_key in score_keys:
                values = np.asarray([float(route.get(score_key) or 0.0) for route in routes], dtype=np.float64)
                blended = native_z + float(alpha) * _standardize(values)
                policies[f"blend__{score_key}__{tag}"][target_idx] = _sort_indices_with_values(routes, blended)
    return policies


def _reranked_run(run: dict[str, Any], orders: dict[int, list[int]], *, selected_policy: str) -> dict[str, Any]:
    targets = []
    for target_idx, target in enumerate(run.get("targets") or []):
        routes = _routes_for_target(target)
        ordered = [routes[idx] for idx in orders.get(target_idx, []) if 0 <= idx < len(routes)]
        reranked = []
        for rank, route in enumerate(ordered):
            payload = dict(route)
            payload["native_rank"] = rank
            payload["rerank_policy"] = selected_policy
            reranked.append(payload)
        target_out = dict(target)
        target_out["routes"] = reranked
        target_out["route_count"] = len(reranked)
        if isinstance((target_out.get("planner_output") or {}).get("routes"), list):
            planner = dict(target_out.get("planner_output") or {})
            planner["routes"] = reranked
            planner["n_results"] = len(reranked)
            target_out["planner_output"] = planner
        targets.append(target_out)
    return {
        "metadata": {**(run.get("metadata") or {}), "rerank_policy": selected_policy},
        "summary": _run_summary(targets),
        "targets": targets,
    }


def _evaluate_policy(run: dict[str, Any], *, benchmark_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    audit = build_product_route_feasibility_audit(run, benchmark_rows=benchmark_rows)
    return {
        "summary": run.get("summary") or {},
        "audit_summary": _audit_summary(audit),
        "ranked_product_metrics": _ranked_product_metrics(audit),
        "gt_recovery": _gt_recovery(run, benchmark_rows),
    }


def _policy_delta(summary: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    out = {"audit": _audit_delta(native.get("audit_summary") or {}, summary.get("audit_summary") or {})}
    ranked = summary.get("ranked_product_metrics") or {}
    native_ranked = native.get("ranked_product_metrics") or {}
    for key in (
        "top1_product_usable_rate",
        "top3_product_usable_rate",
        "top5_product_usable_rate",
        "top1_generic_route_rate",
        "top3_generic_route_rate",
        "top5_generic_route_rate",
        "top1_artifact_rate",
        "top3_artifact_rate",
        "top1_trivial_stock_closure_rate",
        "top3_trivial_stock_closure_rate",
    ):
        if isinstance(ranked.get(key), (int, float)) and isinstance(native_ranked.get(key), (int, float)):
            out[key] = round(float(ranked[key]) - float(native_ranked[key]), 6)
    gt = summary.get("gt_recovery") or {}
    native_gt = native.get("gt_recovery") or {}
    for key in ("exact_gt_route_recovered_rate", "partial_gt_step_overlap_rate", "gt_reactant_in_route_pool_rate"):
        if isinstance(gt.get(key), (int, float)) and isinstance(native_gt.get(key), (int, float)):
            out[key] = round(float(gt[key]) - float(native_gt[key]), 6)
    return out


def _select_best_policy(policy_summaries: dict[str, Any]) -> str:
    def key(item: tuple[str, Any]) -> tuple[float, float, float, float, float, str]:
        name, summary = item
        ranked = summary.get("ranked_product_metrics") or {}
        return (
            float(ranked.get("top3_product_usable_rate") or 0.0),
            float(ranked.get("top1_product_usable_rate") or 0.0),
            -float(ranked.get("top3_artifact_rate") or 0.0),
            -float(ranked.get("top3_trivial_stock_closure_rate") or 0.0),
            -float(ranked.get("top3_generic_route_rate") or 0.0),
            "0" if name == "native" else name,
        )

    native_ranked = (policy_summaries.get("native") or {}).get("ranked_product_metrics") or {}
    native_generic = float(native_ranked.get("top3_generic_route_rate") or 0.0)
    native_artifact = float(native_ranked.get("top3_artifact_rate") or 0.0)
    native_trivial = float(native_ranked.get("top3_trivial_stock_closure_rate") or 0.0)
    native_usable = float(native_ranked.get("top3_product_usable_rate") or 0.0)

    gated = []
    for name, summary in policy_summaries.items():
        ranked = summary.get("ranked_product_metrics") or {}
        generic = float(ranked.get("top3_generic_route_rate") or 0.0)
        artifact = float(ranked.get("top3_artifact_rate") or 0.0)
        trivial = float(ranked.get("top3_trivial_stock_closure_rate") or 0.0)
        usable = float(ranked.get("top3_product_usable_rate") or 0.0)
        if artifact <= native_artifact + 1e-9 and trivial <= native_trivial + 1e-9 and generic <= native_generic + 0.05 + 1e-9 and usable >= native_usable:
            gated.append((name, summary))
    if gated:
        return max(gated, key=key)[0]
    return max(policy_summaries.items(), key=key)[0]


def _sort_indices(routes: list[dict[str, Any]], *, score_key: str) -> list[int]:
    return [
        idx
        for idx, _route in sorted(
            enumerate(routes),
            key=lambda item: (-float(item[1].get(score_key) or 0.0), int(item[1].get("original_native_rank", item[0]) or item[0]), str(item[1].get("route_id") or "")),
        )
    ]


def _sort_indices_with_values(routes: list[dict[str, Any]], values: np.ndarray) -> list[int]:
    return [
        idx
        for idx, _route in sorted(
            enumerate(routes),
            key=lambda item: (-float(values[item[0]]), int(item[1].get("original_native_rank", item[0]) or item[0]), str(item[1].get("route_id") or "")),
        )
    ]


def _standardize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(values.mean())) / std


def _load_model(path: Path, model_name: str) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    models = payload.get("models") or {}
    if model_name not in models:
        raise KeyError(f"model {model_name!r} not found in {path}; available={sorted(models)}")
    return {"model": models[model_name], "model_name": model_name}


def _runtime_model_score(model_payload: dict[str, Any], row: dict[str, Any]) -> float:
    fitted = model_payload["model"]
    x = np.asarray([_feature_row(row)], dtype=np.float32)[:, fitted.feature_indices]
    score = fitted.model.decision_function((x - fitted.mean) / fitted.std)
    return float(score[0])


def _step_product(step: dict[str, Any]) -> str:
    if step.get("product_smiles"):
        return str(step.get("product_smiles"))
    products = step.get("products")
    if isinstance(products, list) and products:
        return str(products[-1])
    rxn = str(step.get("rxn_smiles") or step.get("reaction_smiles") or "")
    if ">>" in rxn:
        return rxn.split(">>", 1)[1].split(".")[0].strip()
    return ""


def _main_reactant(step: dict[str, Any]) -> str:
    if step.get("main_reactant"):
        return str(step.get("main_reactant"))
    reactants = _step_reactants(step)
    return max(reactants, key=len) if reactants else ""


def _step_reactants(step: dict[str, Any]) -> list[str]:
    for key in ("reactants", "reactant_smiles"):
        values = step.get(key)
        if isinstance(values, list) and values:
            return [str(value) for value in values if value]
    rxn = str(step.get("rxn_smiles") or step.get("reaction_smiles") or "")
    if ">>" in rxn:
        return [part.strip() for part in rxn.split(">>", 1)[0].split(".") if part.strip()]
    values = [step.get("main_reactant"), *(step.get("aux_reactants") or [])]
    return [str(value) for value in values if value]


def _step_transform(step: dict[str, Any]) -> str:
    for key in ("transformation_superclass", "transformation_name", "reaction_type", "step_mode"):
        value = str(step.get(key) or "").strip().lower()
        if value and value not in {"unknown", "none", "not_specified"}:
            return value
    return ""


def _step_or_route_score(step: dict[str, Any], route: dict[str, Any]) -> float:
    for value in (step.get("native_step_score"), step.get("score"), route.get("native_score")):
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _score_subset(route: dict[str, Any]) -> dict[str, float]:
    keys = (
        "ccts_v3_runtime_step_any_mean",
        "ccts_v3_runtime_step_any_max",
        "ccts_v3_runtime_step_pair_mean",
        "ccts_v3_runtime_step_pair_max",
        "ccts_v3_runtime_model_mean",
        "ccts_v3_runtime_model_max",
        "ccts_v3_runtime_best_route_evidence",
    )
    return {key: float(route.get(key) or 0.0) for key in keys}


def _run_summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = [len(_routes_for_target(target)) for target in targets]
    return {
        "targets": len(targets),
        "plan_rate": round(sum(1 for value in route_counts if value > 0) / max(len(targets), 1), 6),
        "total_routes": sum(route_counts),
        "avg_route_count": round(sum(route_counts) / max(len(targets), 1), 6),
    }


def _counts(run: dict[str, Any]) -> dict[str, Any]:
    route_counts = [len(_routes_for_target(target)) for target in run.get("targets") or []]
    n_steps = 0
    stock_closed = 0
    for target in run.get("targets") or []:
        for route in _routes_for_target(target):
            n_steps += len(route.get("steps") or [])
            stock_closed += int(bool(route.get("stock_closed")))
    return {
        "targets": len(route_counts),
        "routes": sum(route_counts),
        "steps": n_steps,
        "avg_routes_per_target": round(sum(route_counts) / max(len(route_counts), 1), 6),
        "stock_closed_routes": stock_closed,
    }


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in (
        "ccts_v3_runtime_step_any_mean",
        "ccts_v3_runtime_step_pair_max",
        "ccts_v3_runtime_model_mean",
        "ccts_v3_runtime_best_route_evidence",
    ):
        values = [float((row.get("scores") or {}).get(key) or 0.0) for row in rows]
        if values:
            out[key] = {
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "mean": round(sum(values) / len(values), 6),
            }
    return out


def _interpretation(policy_summaries: dict[str, Any], deltas: dict[str, Any], selected_policy: str) -> dict[str, Any]:
    native = policy_summaries.get("native") or {}
    selected = policy_summaries.get(selected_policy) or {}
    native_ranked = native.get("ranked_product_metrics") or {}
    selected_ranked = selected.get("ranked_product_metrics") or {}
    return {
        "selected_policy": selected_policy,
        "native_top3_product_usable_rate": native_ranked.get("top3_product_usable_rate"),
        "selected_top3_product_usable_rate": selected_ranked.get("top3_product_usable_rate"),
        "delta_vs_native": deltas.get(selected_policy, {}) if selected_policy != "native" else {},
        "verdict": "offline selector only; promote only if selected policy improves usable/cascade metrics without increasing artifact/trivial rates",
    }


def _markdown(report: dict[str, Any]) -> str:
    native = report.get("policy_summaries", {}).get("native", {})
    selected = report.get("policy_summaries", {}).get(report.get("selected_policy"), {})
    delta = report.get("deltas_vs_native", {}).get(report.get("selected_policy"), {}) if report.get("selected_policy") != "native" else {}
    native_ranked = native.get("ranked_product_metrics") or {}
    selected_ranked = selected.get("ranked_product_metrics") or {}
    lines = [
        "# CCTS-v3 Runtime Native Pool Replay",
        "",
        f"- Native pool: `{report['metadata']['native_pool']}`",
        f"- Routes: `{report['counts']['routes']}`",
        f"- Selected policy: `{report['selected_policy']}`",
        "",
        "## Native vs Selected",
        "",
        "| metric | native | selected | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "top1_product_usable_rate",
        "top3_product_usable_rate",
        "top5_product_usable_rate",
        "top1_generic_route_rate",
        "top3_generic_route_rate",
        "top3_artifact_rate",
        "top3_trivial_stock_closure_rate",
    ):
        lines.append(f"| {key} | {_fmt(native_ranked.get(key))} | {_fmt(selected_ranked.get(key))} | {_fmt(delta.get(key))} |")
    lines += [
        "",
        "## GT Recovery",
        "",
        "```json",
        json.dumps(
            {
                "native": native.get("gt_recovery"),
                "selected": selected.get("gt_recovery"),
                "delta": {key: value for key, value in delta.items() if key.endswith("_rate") and key.startswith(("exact", "partial", "gt_"))},
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## Top Policies",
        "",
        "| policy | top3 usable | top1 usable | top3 generic | top3 artifact | top3 trivial |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in sorted(report.get("policy_summaries", {}).items(), key=lambda item: _policy_table_key(item), reverse=True)[:12]:
        ranked = summary.get("ranked_product_metrics") or {}
        lines.append(
            f"| {name} | {_fmt(ranked.get('top3_product_usable_rate'))} | {_fmt(ranked.get('top1_product_usable_rate'))} | "
            f"{_fmt(ranked.get('top3_generic_route_rate'))} | {_fmt(ranked.get('top3_artifact_rate'))} | {_fmt(ranked.get('top3_trivial_stock_closure_rate'))} |"
        )
    return "\n".join(lines) + "\n"


def _policy_table_key(item: tuple[str, Any]) -> tuple[float, float, float, float, float]:
    ranked = (item[1].get("ranked_product_metrics") or {})
    return (
        float(ranked.get("top3_product_usable_rate") or 0.0),
        float(ranked.get("top1_product_usable_rate") or 0.0),
        -float(ranked.get("top3_artifact_rate") or 0.0),
        -float(ranked.get("top3_trivial_stock_closure_rate") or 0.0),
        -float(ranked.get("top3_generic_route_rate") or 0.0),
    )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):09d}")
    except Exception:
        return (1, value)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Replay runtime-safe CCTS-v3 on a ChemEnzy native route pool")
    ap.add_argument("--native-pool", required=True)
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--model-pickle", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--benchmark")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--alpha-values", default="0.10,0.30,0.50")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--best-policy")
    ap.add_argument("--write-all-policy-runs", action="store_true")
    args = ap.parse_args()
    report = replay_ccts_v3_runtime_on_native_pool(
        native_pool=Path(args.native_pool),
        program_manifest=Path(args.program_manifest),
        model_pickle=Path(args.model_pickle),
        output_dir=Path(args.output_dir),
        benchmark=Path(args.benchmark) if args.benchmark else None,
        model_name=args.model_name,
        alpha_values=_parse_float_tuple(args.alpha_values),
        top_k=args.top_k,
        best_policy=args.best_policy,
        write_all_policy_runs=args.write_all_policy_runs,
    )
    print(json.dumps(report["interpretation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
