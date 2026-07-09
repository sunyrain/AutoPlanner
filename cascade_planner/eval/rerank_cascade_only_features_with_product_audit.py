"""Product-audit guarded rerank with full cascade-only route-pool features."""
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


SCHEMA_VERSION = "cascade_only_product_audit_guarded_rerank.v1"
FEATURE_NAMES = [
    "block_conservative_coherence",
    "block_low_count_lt_0_25",
    "block_max",
    "block_mean",
    "block_min",
    "block_rerank_score",
    "block_route_coherence",
    "ccts_best_route_evidence",
    "ccts_model_max",
    "ccts_model_mean",
    "ccts_step_any_max",
    "ccts_step_any_mean",
    "ccts_step_pair_max",
    "ccts_step_pair_mean",
    "n_blocks",
    "v4_step_accepted_rate",
    "v4_step_matched_rate",
    "v4_step_similarity_max",
    "v4_step_similarity_mean",
    "v4_step_similarity_min",
]


def rerank_cascade_only_features_with_product_audit(
    *,
    scored_native_pool: Path,
    enriched_route_pool_jsonl: Path,
    block_replay_jsonl: Path,
    ranker_pickle: Path,
    output: Path,
    report: Path,
    benchmark: Path | None = None,
    top_k: int | None = None,
    ranking_mode: str = "audit_guarded_cascade_ranker",
) -> dict[str, Any]:
    run = _load_run(scored_native_pool, top_k=top_k)
    enriched = _index_jsonl(enriched_route_pool_jsonl, top_k=top_k)
    blocks = _index_jsonl(block_replay_jsonl, top_k=top_k)
    _validate_alignment(run, enriched, blocks)
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
            key = _key_for_route(target, route, native_rank=native_rank, target_index=target_index)
            feature = _feature_row(route, enriched[key], blocks[key], route_count=len(_routes_for_target(target)))
            cascade_score = _ranker_score(feature, ranker)
            audit_row = audit_by_rank.get(native_rank, {})
            payload = dict(route)
            payload["native_rank"] = native_rank
            payload["cascade_only_route_pool_ranker_score"] = cascade_score
            payload["cascade_only_route_pool_features"] = feature
            payload["cascade_only_product_audit_features"] = {
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
                    "cascade_ranker_score": cascade_score,
                    "feature": feature,
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
            "reranker": "cascade_only_product_audit_guarded",
            "source_scored_native_pool": str(scored_native_pool),
            "enriched_route_pool_jsonl": str(enriched_route_pool_jsonl),
            "block_replay_jsonl": str(block_replay_jsonl),
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
            "enriched_route_pool_jsonl": str(enriched_route_pool_jsonl),
            "block_replay_jsonl": str(block_replay_jsonl),
            "ranker_pickle": str(ranker_pickle),
            "benchmark": str(benchmark) if benchmark else None,
            "output": str(output),
            "report": str(report),
            "top_k": top_k,
            "ranking_mode": ranking_mode,
            "contract": "fixed ChemEnzy pool; product-audit route class is a guard; full cascade-only ranker tie-breaks inside class",
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


def _index_jsonl(path: Path, *, top_k: int | None) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rank = int(row.get("native_rank") or 0)
            if top_k is not None and top_k > 0 and rank >= top_k:
                continue
            key = (str(row.get("target_id") or ""), rank)
            out[key] = row
    return out


def _validate_alignment(run: dict[str, Any], enriched: dict[tuple[str, int], dict[str, Any]], blocks: dict[tuple[str, int], dict[str, Any]]) -> None:
    missing_enriched = []
    missing_blocks = []
    for target_index, target in enumerate(run.get("targets") or []):
        for native_rank, route in enumerate(_routes_for_target(target)):
            key = _key_for_route(target, route, native_rank=native_rank, target_index=target_index)
            if key not in enriched:
                missing_enriched.append(key)
            if key not in blocks:
                missing_blocks.append(key)
    if missing_enriched or missing_blocks:
        raise ValueError(
            "cascade-only feature alignment failed; "
            f"missing_enriched={missing_enriched[:5]} count={len(missing_enriched)}; "
            f"missing_blocks={missing_blocks[:5]} count={len(missing_blocks)}"
        )


def _key_for_route(target: dict[str, Any], route: dict[str, Any], *, native_rank: int, target_index: int | None = None) -> tuple[str, int]:
    target_id = str(
        _first_present(
            target.get("target_id"),
            target.get("cascade_id"),
            target.get("index"),
            route.get("target_id"),
            target_index,
            target.get("target_smiles"),
            "",
        )
    )
    return (target_id, int(native_rank))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        return value
    return ""


def _load_ranker(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    names = [str(name) for name in payload.get("feature_names") or []]
    if names != FEATURE_NAMES:
        raise ValueError(f"expected cascade_only feature names {FEATURE_NAMES}, got {names}")
    return payload


def _feature_row(route: dict[str, Any], enriched: dict[str, Any], block: dict[str, Any], *, route_count: int) -> dict[str, float]:
    native_rank = int(route.get("native_rank") or enriched.get("native_rank") or block.get("native_rank") or 0)
    route_count = max(route_count, native_rank + 1, 1)
    block_summary = block.get("block_coherence") or {}
    return {
        "block_conservative_coherence": _float(block_summary.get("conservative_route_coherence_score")),
        "block_low_count_lt_0_25": _float(block_summary.get("low_block_count_lt_0_25")),
        "block_max": _float(block_summary.get("max")),
        "block_mean": _float(block_summary.get("mean")),
        "block_min": _float(block_summary.get("min")),
        "block_rerank_score": _float(block_summary.get("rerank_score")),
        "block_route_coherence": _float(block_summary.get("route_coherence_score")),
        "ccts_best_route_evidence": _float(route.get("ccts_v3_runtime_best_route_evidence")),
        "ccts_model_max": _float(route.get("ccts_v3_runtime_model_max")),
        "ccts_model_mean": _float(route.get("ccts_v3_runtime_model_mean")),
        "ccts_step_any_max": _float(route.get("ccts_v3_runtime_step_any_max")),
        "ccts_step_any_mean": _float(route.get("ccts_v3_runtime_step_any_mean")),
        "ccts_step_pair_max": _float(route.get("ccts_v3_runtime_step_pair_max")),
        "ccts_step_pair_mean": _float(route.get("ccts_v3_runtime_step_pair_mean")),
        "n_blocks": _float(block_summary.get("n_blocks")),
        **_step_enrichment_features(enriched),
    }


def _step_enrichment_features(route: dict[str, Any]) -> dict[str, float]:
    sims = []
    accepted = 0
    matched = 0
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    for step in steps:
        ev = step.get("v4_step_evidence") or {}
        if ev.get("matched"):
            matched += 1
        if ev.get("accepted"):
            accepted += 1
        if ev.get("similarity") is not None:
            sims.append(_float(ev.get("similarity")))
    n_steps = max(1, len(steps))
    return {
        "v4_step_matched_rate": float(matched) / n_steps,
        "v4_step_accepted_rate": float(accepted) / n_steps,
        "v4_step_similarity_mean": float(np.mean(sims)) if sims else 0.0,
        "v4_step_similarity_min": float(np.min(sims)) if sims else 0.0,
        "v4_step_similarity_max": float(np.max(sims)) if sims else 0.0,
    }


def _ranker_score(feature: dict[str, float], ranker: dict[str, Any]) -> float:
    names = [str(name) for name in ranker.get("feature_names") or []]
    x = np.asarray([[_float(feature.get(name)) for name in names]], dtype=np.float32)
    mean = np.asarray(ranker.get("mean"), dtype=np.float32)
    std = np.asarray(ranker.get("std"), dtype=np.float32)
    return float(ranker["model"].decision_function((x - mean) / std)[0])


def _ranking_key(route: dict[str, Any], *, ranking_mode: str) -> tuple[Any, ...]:
    audit = route.get("cascade_only_product_audit_features") or {}
    score = -float(route.get("cascade_only_route_pool_ranker_score") or 0.0)
    native_rank = int(route.get("native_rank") or 10**9)
    if ranking_mode == "audit_guarded_cascade_ranker":
        return (*product_audit_guard_key(audit), score, native_rank)
    if ranking_mode == "cascade_ranker_only":
        return (score, native_rank)
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
    values = [float(row.get("cascade_ranker_score") or 0.0) for row in rows]
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
        "# Cascade-Only Product-Audit Guarded Rerank",
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
    ap = argparse.ArgumentParser(description="Rerank with full cascade-only features behind product-audit guard")
    ap.add_argument("--scored-native-pool", required=True)
    ap.add_argument("--enriched-route-pool-jsonl", required=True)
    ap.add_argument("--block-replay-jsonl", required=True)
    ap.add_argument("--ranker-pickle", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--benchmark")
    ap.add_argument("--top-k", type=int)
    ap.add_argument(
        "--ranking-mode",
        default="audit_guarded_cascade_ranker",
        choices=["audit_guarded_cascade_ranker", "cascade_ranker_only", "audit_guarded_native"],
    )
    args = ap.parse_args()
    result = rerank_cascade_only_features_with_product_audit(
        scored_native_pool=Path(args.scored_native_pool),
        enriched_route_pool_jsonl=Path(args.enriched_route_pool_jsonl),
        block_replay_jsonl=Path(args.block_replay_jsonl),
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
