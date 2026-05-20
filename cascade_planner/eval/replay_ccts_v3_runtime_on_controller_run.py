"""Replay runtime-safe CCTS-v3 reranking on a controller benchmark run."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import RDLogger

from cascade_planner.eval.audit_candidate_specific_evidence import _train_bank
from cascade_planner.eval.build_ccts_v3_runtime_evidence_cache import _runtime_evidence_scores
from cascade_planner.eval.replay_ccts_v3_on_controller_run import (
    _counts,
    _parse_float_tuple,
    _parse_int_tuple,
    _policy_delta,
    _policy_summary,
    _reranked_run,
    _route_record,
    _score_summary,
    _select_best_policy,
    _sort_indices,
    _sort_indices_with_values,
    _standardize,
    _write_jsonl,
)
from cascade_planner.eval.train_ccts_v3_runtime_pairwise_ranker import FittedRuntimeModel, _feature_row


SCHEMA_VERSION = "ccts_v3_runtime_controller_run_replay.v1"
DEFAULT_MODEL = "runtime_pairwise_block_supported_positive_label__runtime_evidence_only"


def replay_ccts_v3_runtime_on_controller_run(
    *,
    run_json: Path,
    program_manifest: Path,
    model_pickle: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    alpha_values: tuple[float, ...] = (0.10, 0.30, 0.50),
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
    best_policy: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads(run_json.read_text(encoding="utf-8"))
    train = _train_bank(program_manifest)
    model_payload = _load_model(model_pickle, model_name)
    product_sim_cache: dict[tuple[str, str], list[float]] = {}
    target_rows = []
    scored_routes = []
    for target in run.get("targets") or []:
        target_scored = []
        routes = ((target.get("planner_output") or {}).get("routes") or [])
        per_route = ((target.get("route_recovery") or {}).get("per_route") or [])
        for native_rank, route in enumerate(routes):
            route_record = _route_record(target, route, native_rank=native_rank)
            scored = _score_route_runtime(
                route_record,
                train_bank=train,
                product_sim_cache=product_sim_cache,
                model_payload=model_payload,
            )
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
    summaries = {name: _policy_summary(target_rows, orders=orders, top_ks=top_ks) for name, orders in policies.items()}
    deltas = {name: _policy_delta(summary, summaries["native"]) for name, summary in summaries.items() if name != "native"}
    selected_policy = best_policy or _select_best_policy(summaries)
    selected_run = _reranked_run(run, policies[selected_policy], selected_policy=selected_policy)

    scored_path = output_dir / "hybrid_d_ccts_v3_runtime_scored_routes.jsonl"
    _write_jsonl(scored_path, scored_routes)
    selected_run_path = output_dir / f"hybrid_d_ccts_v3_runtime_{selected_policy}_run.json"
    selected_run_path.write_text(json.dumps(selected_run, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "run_json": str(run_json),
            "program_manifest": str(program_manifest),
            "model_pickle": str(model_pickle),
            "model_name": model_name,
            "output_dir": str(output_dir),
            "alpha_values": list(alpha_values),
            "top_ks": list(top_ks),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "fixed Hybrid-D route pool; runtime-safe CCTS model/evidence only; no generation and no pruning",
        },
        "counts": _counts(target_rows),
        "score_summary": _score_summary(scored_routes),
        "runtime_score_summary": _runtime_score_summary(scored_routes),
        "policy_summaries": summaries,
        "deltas_vs_native": deltas,
        "selected_policy": selected_policy,
        "outputs": {
            "scored_routes_jsonl": str(scored_path),
            "selected_reranked_run_json": str(selected_run_path),
        },
        "interpretation": _interpretation(summaries, deltas, selected_policy),
    }
    report_path = output_dir / "hybrid_d_ccts_v3_runtime_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _score_route_runtime(
    route: dict[str, Any],
    *,
    train_bank: dict[str, Any],
    product_sim_cache: dict[tuple[str, str], list[float]],
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    out = dict(route)
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    step_scores = []
    for idx, step in enumerate(steps):
        previous_transform = _step_transform(steps[idx - 1]) if idx > 0 else ""
        next_transform = _step_transform(steps[idx + 1]) if idx + 1 < len(steps) else ""
        evidence = _runtime_evidence_scores(
            product=str(step.get("product_smiles") or ""),
            candidate_main=str(step.get("main_reactant") or ""),
            previous_transform=previous_transform,
            next_transform=next_transform,
            train_bank=train_bank,
            product_sim_cache=product_sim_cache,
        )
        feature_row = {
            "candidate_rank": 1,
            "candidate_score": route.get("native_score"),
            "candidate_reactants": step.get("reactants") or [],
            **evidence,
        }
        model_score = _runtime_model_score(model_payload, feature_row)
        step_scores.append(
            {
                "step_index": step.get("step_index") or idx + 1,
                "step_id": step.get("step_id") or f"step_{idx + 1}",
                "product_smiles": step.get("product_smiles"),
                "candidate_main_reactant": step.get("main_reactant"),
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


def _policy_orders(target_rows: list[dict[str, Any]], *, alpha_values: tuple[float, ...]) -> dict[str, dict[int, list[int]]]:
    score_keys = (
        "ccts_v3_runtime_step_any_mean",
        "ccts_v3_runtime_step_pair_max",
        "ccts_v3_runtime_model_mean",
        "ccts_v3_runtime_best_route_evidence",
    )
    policies: dict[str, dict[int, list[int]]] = {"native": {}}
    for score_key in score_keys:
        policies[f"evidence_only__{score_key}"] = {}
    for alpha in alpha_values:
        tag = f"alpha{int(round(alpha * 100)):03d}"
        for score_key in score_keys:
            policies[f"blend__{score_key}__{tag}"] = {}
            policies[f"native_quality_guarded_blend__{score_key}__{tag}"] = {}
    for target_idx, row in enumerate(target_rows):
        routes = row["routes"]
        policies["native"][target_idx] = [route["route_index"] for route in sorted(routes, key=lambda item: item["route_index"])]
        for score_key in score_keys:
            policies[f"evidence_only__{score_key}"][target_idx] = _sort_indices(routes, score_key=score_key)
        native_values = np.asarray([-float(route["route_index"]) for route in routes], dtype=np.float64)
        native_z = _standardize(native_values)
        for alpha in alpha_values:
            tag = f"alpha{int(round(alpha * 100)):03d}"
            for score_key in score_keys:
                score_values = np.asarray([float(route["scores"].get(score_key) or 0.0) for route in routes], dtype=np.float64)
                blended = native_z + float(alpha) * _standardize(score_values)
                policies[f"blend__{score_key}__{tag}"][target_idx] = _sort_indices_with_values(routes, blended)
                policies[f"native_quality_guarded_blend__{score_key}__{tag}"][target_idx] = _sort_indices_with_values(
                    routes,
                    blended,
                    native_quality_guard=True,
                )
    return policies


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


def _route_source(route: dict[str, Any]) -> str:
    broad = route.get("broad_reservoir") or {}
    if broad.get("source"):
        return str(broad.get("source"))
    sources = Counter(str(step.get("source") or "") for step in route.get("steps") or [] if isinstance(step, dict))
    return sources.most_common(1)[0][0] if sources else "unknown"


def _step_transform(step: dict[str, Any]) -> str:
    value = str(step.get("transformation_superclass") or "").strip()
    return value if value and value.lower() != "unknown" else ""


def _runtime_score_summary(scored_routes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "ccts_v3_runtime_step_any_mean",
        "ccts_v3_runtime_step_pair_max",
        "ccts_v3_runtime_model_mean",
        "ccts_v3_runtime_best_route_evidence",
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
        "native_stock_at1": native.get("strict_stock_solve_at1"),
        "selected_stock_at1": selected.get("strict_stock_solve_at1"),
        "native_gt_at1": native.get("gt_reactant_hit_at1"),
        "selected_gt_at1": selected.get("gt_reactant_hit_at1"),
        "native_gt_at3": native.get("gt_reactant_hit_at3"),
        "selected_gt_at3": selected.get("gt_reactant_hit_at3"),
        "native_exact_at1": native.get("exact_reaction_hit_at1"),
        "selected_exact_at1": selected.get("exact_reaction_hit_at1"),
        "native_exact_at3": native.get("exact_reaction_hit_at3"),
        "selected_exact_at3": selected.get("exact_reaction_hit_at3"),
        "delta": selected_delta,
        "safe_claim": (
            "runtime-safe CCTS replay improves at least one GT/exact top-k metric without selected top1 stock/cascade/exact regression"
            if float(selected_delta.get("strict_stock_solve_at1") or 0.0) >= 0
            and float(selected_delta.get("cascade_compatibility_success_at1") or 0.0) >= 0
            and float(selected_delta.get("exact_reaction_hit_at1") or 0.0) >= 0
            and (
                float(selected_delta.get("gt_reactant_hit_at1") or 0.0) > 0
                or float(selected_delta.get("gt_reactant_hit_at3") or 0.0) > 0
                or float(selected_delta.get("exact_reaction_hit_at3") or 0.0) > 0
            )
            else "runtime-safe CCTS replay is not yet a safe route-order replacement"
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hybrid-D + Runtime-Safe CCTS-v3 Replay",
        "",
        "## Interpretation",
        "",
        "```json",
        json.dumps(report.get("interpretation") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False),
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
    return "\n".join(lines) + "\n"


def _mean(values: list[float] | np.ndarray) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Replay runtime-safe CCTS-v3 reranking on a controller run")
    ap.add_argument("--run-json", default="results/shared/phase2_20260515/full100_abcd_gate30/D/run.json")
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--model-pickle", default="results/shared/cascadebench_strict_20260516/ccts_v3_runtime_pairwise_ranker/ccts_v3_runtime_pairwise_ranker_models.pkl")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default="results/shared/cascadebench_strict_20260516/hybrid_d_ccts_v3_runtime_full100_replay")
    ap.add_argument("--alpha-values", default="0.10,0.30,0.50")
    ap.add_argument("--top-ks", default="1,3,5,10")
    ap.add_argument("--best-policy", default=None)
    args = ap.parse_args()
    report = replay_ccts_v3_runtime_on_controller_run(
        run_json=Path(args.run_json),
        program_manifest=Path(args.program_manifest),
        model_pickle=Path(args.model_pickle),
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        alpha_values=_parse_float_tuple(args.alpha_values),
        top_ks=_parse_int_tuple(args.top_ks),
        best_policy=args.best_policy,
    )
    print(
        json.dumps(
            {
                "selected_policy": report["selected_policy"],
                "interpretation": report["interpretation"],
                "outputs": report["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
