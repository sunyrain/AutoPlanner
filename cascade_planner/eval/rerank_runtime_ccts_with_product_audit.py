"""Product-audit guarded rerank using runtime CCTS route-pool scores."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit, product_audit_guard_key
from cascade_planner.eval.rerank_native_routes_with_v4_value import (
    _audit_delta,
    _audit_summary,
    _gt_recovery,
    _ranked_product_metrics,
    _read_rows,
    _routes_for_target,
)
from cascade_planner.eval.train_route_pool_ranker import _float


SCHEMA_VERSION = "runtime_ccts_product_audit_guarded_rerank.v1"
RUNTIME_TO_RANKER_FEATURE = {
    "ccts_best_route_evidence": "ccts_v3_runtime_best_route_evidence",
    "ccts_model_max": "ccts_v3_runtime_model_max",
    "ccts_model_mean": "ccts_v3_runtime_model_mean",
    "ccts_step_any_max": "ccts_v3_runtime_step_any_max",
    "ccts_step_any_mean": "ccts_v3_runtime_step_any_mean",
    "ccts_step_pair_max": "ccts_v3_runtime_step_pair_max",
    "ccts_step_pair_mean": "ccts_v3_runtime_step_pair_mean",
}


def rerank_runtime_ccts_with_product_audit(
    *,
    scored_native_pool: Path,
    ranker_pickle: Path,
    output: Path,
    report: Path,
    benchmark: Path | None = None,
    top_k: int | None = None,
    ranking_mode: str = "audit_guarded_ccts_ranker",
) -> dict[str, Any]:
    run = _load_run(scored_native_pool, top_k=top_k)
    benchmark_rows = _read_rows(benchmark) if benchmark else None
    audit = build_product_route_feasibility_audit(run, benchmark_rows=benchmark_rows)
    ranker = _load_ranker(ranker_pickle)
    targets_out = []
    route_rows = []
    for target_index, target in enumerate(run.get("targets") or []):
        audit_target = (audit.get("targets") or [{}])[target_index] if target_index < len(audit.get("targets") or []) else {}
        audit_by_rank = {
            int(row.get("rank") or 0) - 1: row
            for row in audit_target.get("routes") or []
            if row.get("rank") is not None
        }
        scored_routes = []
        for native_rank, route in enumerate(_routes_for_target(target)):
            audit_row = audit_by_rank.get(native_rank, {})
            ccts_score = _ranker_score(route, ranker)
            payload = dict(route)
            payload["native_rank"] = native_rank
            payload["runtime_ccts_route_pool_ranker_score"] = ccts_score
            payload["runtime_ccts_product_audit_features"] = {
                "route_class": audit_row.get("route_class"),
                "issues": audit_row.get("issues") or [],
                "tags": audit_row.get("tags") or [],
                "route_plausibility": audit_row.get("route_plausibility") or {},
            }
            scored_routes.append(payload)
            route_rows.append(
                {
                    "target_index": target_index,
                    "target_smiles": target.get("target_smiles"),
                    "native_rank": native_rank,
                    "route_class": audit_row.get("route_class"),
                    "issues": audit_row.get("issues") or [],
                    "ccts_ranker_score": ccts_score,
                }
            )
        reranked = sorted(scored_routes, key=lambda route: _ranking_key(route, ranking_mode=ranking_mode))
        reranked = [_with_new_rank(route, rank, ranking_mode) for rank, route in enumerate(reranked)]
        target_out = dict(target)
        target_out["routes"] = reranked
        target_out["route_count"] = len(reranked)
        if isinstance((target_out.get("planner_output") or {}).get("routes"), list):
            planner = dict(target_out.get("planner_output") or {})
            planner["routes"] = reranked
            planner["n_results"] = len(reranked)
            target_out["planner_output"] = planner
        targets_out.append(target_out)
    output_run = {
        "metadata": {
            **(run.get("metadata") or {}),
            "reranker": "runtime_ccts_product_audit_guarded",
            "source_scored_native_pool": str(scored_native_pool),
            "ranker_pickle": str(ranker_pickle),
            "ranking_mode": ranking_mode,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": _run_summary(targets_out),
        "targets": targets_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_run, indent=2, ensure_ascii=False), encoding="utf-8")
    reranked_audit = build_product_route_feasibility_audit(output_run, benchmark_rows=benchmark_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "scored_native_pool": str(scored_native_pool),
            "ranker_pickle": str(ranker_pickle),
            "benchmark": str(benchmark) if benchmark else None,
            "output": str(output),
            "report": str(report),
            "top_k": top_k,
            "ranking_mode": ranking_mode,
            "contract": "fixed ChemEnzy pool; product-audit route class is a guard; runtime CCTS ranker only tie-breaks inside class",
        },
        "summary": output_run["summary"],
        "native_audit_summary": _audit_summary(audit),
        "guarded_audit_summary": _audit_summary(reranked_audit),
        "native_ranked_product_metrics": _ranked_product_metrics(audit),
        "guarded_ranked_product_metrics": _ranked_product_metrics(reranked_audit),
        "native_gt_recovery": _gt_recovery(run, benchmark_rows),
        "guarded_gt_recovery": _gt_recovery(output_run, benchmark_rows),
        "delta_guarded_minus_native": _audit_delta(audit, reranked_audit),
        "score_summary": _score_summary(route_rows),
        "route_rows": route_rows[:200],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _load_run(path: Path, *, top_k: int | None) -> dict[str, Any]:
    run = json.loads(path.read_text(encoding="utf-8"))
    if top_k is None or top_k <= 0:
        return run
    targets = []
    for target in run.get("targets") or []:
        payload = dict(target)
        routes = _routes_for_target(target)[: int(top_k)]
        payload["routes"] = routes
        payload["route_count"] = len(routes)
        if isinstance((payload.get("planner_output") or {}).get("routes"), list):
            planner = dict(payload.get("planner_output") or {})
            planner["routes"] = routes
            planner["n_results"] = len(routes)
            payload["planner_output"] = planner
        targets.append(payload)
    return {**run, "summary": _run_summary(targets), "targets": targets}


def _load_ranker(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    feature_names = [str(name) for name in payload.get("feature_names") or []]
    missing = [name for name in feature_names if name not in RUNTIME_TO_RANKER_FEATURE]
    if missing:
        raise ValueError(
            "runtime guarded CCTS rerank currently supports ccts-only rankers; "
            f"unsupported features in {path}: {missing}"
        )
    return payload


def _ranker_score(route: dict[str, Any], ranker: dict[str, Any]) -> float:
    names = [str(name) for name in ranker.get("feature_names") or []]
    x = np.asarray([[_float(route.get(RUNTIME_TO_RANKER_FEATURE[name])) for name in names]], dtype=np.float32)
    mean = np.asarray(ranker.get("mean"), dtype=np.float32)
    std = np.asarray(ranker.get("std"), dtype=np.float32)
    model = ranker["model"]
    return float(model.decision_function((x - mean) / std)[0])


def _ranking_key(route: dict[str, Any], *, ranking_mode: str) -> tuple[Any, ...]:
    audit = route.get("runtime_ccts_product_audit_features") or {}
    ccts_value = -float(route.get("runtime_ccts_route_pool_ranker_score") or 0.0)
    native_rank = int(route.get("native_rank") or 10**9)
    if ranking_mode == "audit_guarded_ccts_ranker":
        return (*product_audit_guard_key(audit), ccts_value, native_rank)
    if ranking_mode == "ccts_ranker_only":
        return (ccts_value, native_rank)
    if ranking_mode == "audit_guarded_native":
        return (*product_audit_guard_key(audit), native_rank)
    raise ValueError(f"unknown ranking_mode: {ranking_mode}")


def _with_new_rank(route: dict[str, Any], rank: int, policy: str) -> dict[str, Any]:
    payload = dict(route)
    payload["native_rank"] = int(rank)
    payload["rerank_policy"] = policy
    return payload


def _run_summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = [len(_routes_for_target(target)) for target in targets]
    return {
        "targets": len(targets),
        "plan_rate": round(sum(1 for count in route_counts if count > 0) / max(len(route_counts), 1), 6),
        "total_routes": sum(route_counts),
        "avg_route_count": round(sum(route_counts) / max(len(route_counts), 1), 6),
    }


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("ccts_ranker_score") or 0.0) for row in rows]
    classes = Counter(str(row.get("route_class") or "unknown") for row in rows)
    if not values:
        return {"routes": 0}
    return {
        "routes": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "route_class_counts": dict(sorted(classes.items())),
    }


def _markdown(report: dict[str, Any]) -> str:
    native = report.get("native_ranked_product_metrics") or {}
    guarded = report.get("guarded_ranked_product_metrics") or {}
    lines = [
        "# Runtime CCTS Product-Audit Guarded Rerank",
        "",
        f"- Scored pool: `{report['metadata']['scored_native_pool']}`",
        f"- Ranking mode: `{report['metadata']['ranking_mode']}`",
        f"- Routes: `{report['summary']['total_routes']}`",
        "",
        "## Ranked Product Metrics",
        "",
        "| metric | native | guarded |",
        "|---|---:|---:|",
    ]
    for key in (
        "top1_product_usable_rate",
        "top3_product_usable_rate",
        "top5_product_usable_rate",
        "top3_artifact_rate",
        "top3_trivial_stock_closure_rate",
        "top3_generic_route_rate",
    ):
        lines.append(f"| {key} | {_fmt(native.get(key))} | {_fmt(guarded.get(key))} |")
    lines.extend(["", "## GT Recovery", "", "```json", json.dumps({"native": report.get("native_gt_recovery"), "guarded": report.get("guarded_gt_recovery")}, indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerank runtime-scored native pools with product-audit guarded CCTS ranker")
    ap.add_argument("--scored-native-pool", required=True)
    ap.add_argument("--ranker-pickle", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--benchmark")
    ap.add_argument("--top-k", type=int)
    ap.add_argument(
        "--ranking-mode",
        default="audit_guarded_ccts_ranker",
        choices=["audit_guarded_ccts_ranker", "ccts_ranker_only", "audit_guarded_native"],
    )
    args = ap.parse_args()
    result = rerank_runtime_ccts_with_product_audit(
        scored_native_pool=Path(args.scored_native_pool),
        ranker_pickle=Path(args.ranker_pickle),
        output=Path(args.output),
        report=Path(args.report),
        benchmark=Path(args.benchmark) if args.benchmark else None,
        top_k=args.top_k,
        ranking_mode=args.ranking_mode,
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "native": result["native_ranked_product_metrics"],
                "guarded": result["guarded_ranked_product_metrics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
