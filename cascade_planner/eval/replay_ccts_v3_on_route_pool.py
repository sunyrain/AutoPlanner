"""Replay CCTS-v3 candidate evidence on a ChemEnzy route pool.

This is a route-level offline ablation.  It keeps the ChemEnzy-generated route
pool fixed, scores each route step against train-split v4 cascade transition
evidence, reranks routes within each target, and then optionally runs the
held-out v4 block-recovery audit on the reranked pools.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import RDLogger

from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.eval.audit_candidate_specific_evidence import _candidate_evidence_scores, _train_bank
from cascade_planner.eval.audit_v4_heldout_block_recovery import audit_v4_heldout_block_recovery, _connected_step_pairs
from cascade_planner.eval.replay_block_coherence_on_route_pool import _main_reactant


SCHEMA_VERSION = "ccts_v3_route_pool_replay.v1"


def replay_ccts_v3_on_route_pool(
    *,
    route_pool: Path,
    program_manifest: Path,
    output_dir: Path,
    split: str = "test",
    alpha: float = 0.30,
    analog_similarity: float = 0.55,
    top_ks: tuple[int, ...] = (1, 3, 5, 10, 50, 100),
    run_block_audit: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = _read_jsonl(route_pool)
    train_bank = _train_bank(program_manifest)
    pair_stats = _transform_pair_stats(train_bank)
    product_sim_cache: dict[tuple[str, str], list[float]] = {}
    scored = [
        _score_route(route, train_bank=train_bank, pair_stats=pair_stats, product_sim_cache=product_sim_cache)
        for route in routes
    ]
    scored_path = output_dir / "ccts_v3_scored_routes.jsonl"
    _write_jsonl(scored_path, scored)

    rerank_specs = {
        "native": ("native_rank", 0.0),
        "evidence_best_block": ("ccts_v3_best_block_context_sim", 1.0),
        "evidence_inferred_pair": ("ccts_v3_best_inferred_pair_score", 1.0),
        "blend_best_block_alpha030": ("ccts_v3_best_block_blend_score", 1.0),
        "blend_inferred_pair_alpha030": ("ccts_v3_inferred_pair_blend_score", 1.0),
        "blend_step_mean_alpha030": ("ccts_v3_step_mean_blend_score", 1.0),
    }
    reranked_paths: dict[str, str] = {}
    for name, (score_key, _) in rerank_specs.items():
        if name == "native":
            rows = _with_native_rank(scored)
        else:
            rows = _rerank(scored, score_key=score_key)
        path = output_dir / f"{name}_route_pool.jsonl"
        _write_jsonl(path, rows)
        reranked_paths[name] = str(path)
    relabel_specs = {
        "native_inferred_transform_relabel": _with_native_rank(scored),
        "evidence_best_block_inferred_transform_relabel": _rerank(scored, score_key="ccts_v3_best_block_context_sim"),
        "evidence_inferred_pair_inferred_transform_relabel": _rerank(scored, score_key="ccts_v3_best_inferred_pair_score"),
    }
    for name, rows in relabel_specs.items():
        relabeled = [_relabel_steps_with_inferred_transform(row) for row in rows]
        path = output_dir / f"{name}_route_pool.jsonl"
        _write_jsonl(path, relabeled)
        reranked_paths[name] = str(path)

    recovery_reports: dict[str, Any] = {}
    if run_block_audit:
        for name, path in reranked_paths.items():
            report = audit_v4_heldout_block_recovery(
                route_pool=Path(path),
                program_manifest=program_manifest,
                split=split,
                output_json=output_dir / f"{name}_block_recovery.json",
                analog_similarity=analog_similarity,
                top_ks=top_ks,
            )
            recovery_reports[name] = report

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool": str(route_pool),
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "split": split,
            "alpha": alpha,
            "analog_similarity": analog_similarity,
            "top_ks": list(top_ks),
            "elapsed_s": round(time.monotonic() - started, 3),
            "score_contract": "fixed ChemEnzy route pool; CCTS-v3 uses train-split candidate-specific transition evidence only",
        },
        "counts": _counts(scored),
        "score_summary": _score_summary(scored),
        "top1_changes": _top1_changes(scored),
        "outputs": {
            "scored_routes": str(scored_path),
            "reranked_route_pools": reranked_paths,
        },
        "block_recovery_comparison": _recovery_comparison(recovery_reports),
    }
    report_path = output_dir / "ccts_v3_route_pool_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _score_route(
    route: dict[str, Any],
    *,
    train_bank: dict[str, Any],
    pair_stats: dict[str, Any],
    product_sim_cache: dict[tuple[str, str], list[float]],
) -> dict[str, Any]:
    out = dict(route)
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    step_scores = []
    for idx, step in enumerate(steps):
        previous_transform = _step_transform(steps[idx - 1]) if idx > 0 else ""
        next_transform = _step_transform(steps[idx + 1]) if idx + 1 < len(steps) else ""
        product = _step_product(step)
        main = _main_reactant(step)
        transform = _step_transform(step)
        scores = _candidate_evidence_scores(
            product=product,
            candidate_main=main,
            context_transform=transform,
            previous_transform=previous_transform,
            next_transform=next_transform,
            train_bank=train_bank,
            product_sim_cache=product_sim_cache,
        )
        step_scores.append(
            {
                "step_index": step.get("step_index") or idx + 1,
                "step_id": step.get("step_id") or f"step_{idx + 1}",
                "product_smiles": product,
                "candidate_main_reactant": main,
                "transform": transform,
                "previous_transform": previous_transform,
                "next_transform": next_transform,
                **scores,
            }
        )
    block_records = _block_score_records(steps, step_scores, pair_stats=pair_stats)
    block_scores = [float(row.get("context_structural_score") or 0.0) for row in block_records]
    inferred_pair_scores = [float(row.get("inferred_pair_score") or 0.0) for row in block_records]
    native_rank = _native_rank(route)
    native_score = 1.0 / float(native_rank + 1)
    step_context_scores = [float(row.get("candidate_nearest_context_transform_sim") or 0.0) for row in step_scores]
    step_match_scores = [float(row.get("candidate_inferred_transform_match_score") or 0.0) for row in step_scores]
    best_block = max(block_scores, default=0.0)
    step_mean = float(np.mean(step_context_scores)) if step_context_scores else 0.0
    out["ccts_v3_route_evidence"] = {
        "n_steps": len(steps),
        "n_blocks": len(block_scores),
        "native_rank_score": native_score,
        "step_context_mean": step_mean,
        "step_context_max": max(step_context_scores, default=0.0),
        "step_context_min": min(step_context_scores, default=0.0) if step_context_scores else 0.0,
        "step_match_mean": float(np.mean(step_match_scores)) if step_match_scores else 0.0,
        "best_block_context_sim": best_block,
        "mean_block_context_sim": float(np.mean(block_scores)) if block_scores else 0.0,
        "best_inferred_pair_score": max(inferred_pair_scores, default=0.0),
        "mean_inferred_pair_score": float(np.mean(inferred_pair_scores)) if inferred_pair_scores else 0.0,
        "best_inferred_pair": _best_pair_record(block_records),
        "step_scores": step_scores,
        "block_scores": block_records,
    }
    out["original_native_rank"] = native_rank
    out["ccts_v3_best_block_context_sim"] = best_block
    out["ccts_v3_step_mean_context_sim"] = step_mean
    out["ccts_v3_best_inferred_pair_score"] = max(inferred_pair_scores, default=0.0)
    out["ccts_v3_mean_inferred_pair_score"] = float(np.mean(inferred_pair_scores)) if inferred_pair_scores else 0.0
    # Filled after target-wise standardization.
    out["ccts_v3_best_block_blend_score"] = None
    out["ccts_v3_step_mean_blend_score"] = None
    out["ccts_v3_inferred_pair_blend_score"] = None
    return out


def _block_score_records(steps: list[dict[str, Any]], step_scores: list[dict[str, Any]], *, pair_stats: dict[str, Any]) -> list[dict[str, Any]]:
    if len(steps) < 2:
        return []
    by_identity = {id(step): score for step, score in zip(steps, step_scores)}
    records = []
    for left, right in _connected_step_pairs(steps):
        left_score = by_identity.get(id(left), {})
        right_score = by_identity.get(id(right), {})
        # _connected_step_pairs returns (downstream, upstream).  The reference
        # block direction used by the audit is upstream -> downstream.
        downstream_score = left_score
        upstream_score = right_score
        upstream_context = float(upstream_score.get("candidate_nearest_context_transform_sim") or 0.0)
        downstream_context = float(downstream_score.get("candidate_nearest_context_transform_sim") or 0.0)
        upstream_any = float(upstream_score.get("candidate_nearest_any_transition_sim") or 0.0)
        downstream_any = float(downstream_score.get("candidate_nearest_any_transition_sim") or 0.0)
        upstream_inferred = str(upstream_score.get("candidate_inferred_transform") or "unknown").lower()
        downstream_inferred = str(downstream_score.get("candidate_inferred_transform") or "unknown").lower()
        inferred_pair = f"{upstream_inferred}->{downstream_inferred}"
        pair_count = int((pair_stats.get("pair_counts") or {}).get(inferred_pair, 0))
        pair_prior = float((pair_stats.get("pair_prior") or {}).get(inferred_pair, 0.0))
        structural_any = min(upstream_any, downstream_any)
        structural_context = min(upstream_context, downstream_context)
        records.append(
            {
                "context_structural_score": structural_context,
                "any_structural_score": structural_any,
                "inferred_transform_pair": inferred_pair,
                "train_pair_count": pair_count,
                "train_pair_prior": pair_prior,
                "inferred_pair_score": structural_any * pair_prior,
            }
        )
    return records


def _best_pair_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(records, key=lambda row: float(row.get("inferred_pair_score") or 0.0))


def _transform_pair_stats(train_bank: dict[str, Any]) -> dict[str, Any]:
    by_prev = train_bank.get("by_prev_pair") or {}
    counts = {str(key).lower(): len(value or []) for key, value in by_prev.items()}
    max_count = max(counts.values(), default=0)
    if max_count <= 0:
        return {"pair_counts": {}, "pair_prior": {}}
    denom = float(np.log1p(max_count))
    prior = {key: float(np.log1p(value)) / denom for key, value in counts.items()}
    return {"pair_counts": counts, "pair_prior": prior, "max_count": max_count}


def _rerank(routes: list[dict[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in _with_group_blend_scores(routes):
        grouped[_target_key(route)].append(route)
    out = []
    for group in grouped.values():
        sorted_group = sorted(group, key=lambda row: (-float(row.get(score_key) or -1.0), _native_rank(row), str(row.get("route_id") or "")))
        for rank, route in enumerate(sorted_group):
            row = dict(route)
            row["native_rank"] = rank
            row["rerank_policy"] = score_key
            out.append(row)
    return out


def _with_native_rank(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for route in routes:
        row = dict(route)
        row["native_rank"] = _native_rank(route)
        row["rerank_policy"] = "native_rank"
        rows.append(row)
    return rows


def _relabel_steps_with_inferred_transform(route: dict[str, Any]) -> dict[str, Any]:
    row = dict(route)
    evidence = row.get("ccts_v3_route_evidence") or {}
    step_scores = evidence.get("step_scores") or []
    steps = []
    for idx, step in enumerate(row.get("steps") or []):
        new_step = dict(step)
        score = step_scores[idx] if idx < len(step_scores) and isinstance(step_scores[idx], dict) else {}
        inferred = str(score.get("candidate_inferred_transform") or "").strip()
        if inferred and inferred.lower() != "unknown":
            new_step["ccts_v3_original_transformation_superclass"] = new_step.get("transformation_superclass")
            new_step["ccts_v3_inferred_transform"] = inferred
            new_step["transformation_superclass"] = inferred
        steps.append(new_step)
    row["steps"] = steps
    row["transform_relabel_policy"] = "ccts_v3_candidate_inferred_transform"
    return row


def _with_group_blend_scores(routes: list[dict[str, Any]], *, alpha: float = 0.30) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        grouped[_target_key(route)].append(route)
    out = []
    for group in grouped.values():
        native = np.asarray([-float(_native_rank(route)) for route in group], dtype=np.float64)
        best_block = np.asarray([float(route.get("ccts_v3_best_block_context_sim") or 0.0) for route in group], dtype=np.float64)
        step_mean = np.asarray([float(route.get("ccts_v3_step_mean_context_sim") or 0.0) for route in group], dtype=np.float64)
        inferred_pair = np.asarray([float(route.get("ccts_v3_best_inferred_pair_score") or 0.0) for route in group], dtype=np.float64)
        best_blend = _standardize(native) + float(alpha) * _standardize(best_block)
        mean_blend = _standardize(native) + float(alpha) * _standardize(step_mean)
        inferred_pair_blend = _standardize(native) + float(alpha) * _standardize(inferred_pair)
        for route, best_score, mean_score, pair_score in zip(group, best_blend, mean_blend, inferred_pair_blend):
            row = dict(route)
            row["ccts_v3_best_block_blend_score"] = float(best_score)
            row["ccts_v3_step_mean_blend_score"] = float(mean_score)
            row["ccts_v3_inferred_pair_blend_score"] = float(pair_score)
            out.append(row)
    return out


def _target_key(route: dict[str, Any]) -> str:
    target = canonical_smiles(str(route.get("target_smiles") or "")) or str(route.get("target_smiles") or "")
    return target or str(route.get("target_id") or "")


def _step_product(step: dict[str, Any]) -> str:
    product = step.get("product_smiles")
    if not product and isinstance(step.get("products"), list) and step.get("products"):
        product = step["products"][0]
    return canonical_smiles(str(product or "")) or str(product or "")


def _step_transform(step: dict[str, Any]) -> str:
    transform = str(step.get("transformation_superclass") or "").strip()
    if transform and transform.lower() != "unknown":
        return transform
    evidence = step.get("v4_step_evidence") or {}
    fallback = str(evidence.get("source_transform") or "").strip()
    return fallback or transform or "unknown"


def _native_rank(route: dict[str, Any]) -> int:
    try:
        return int(route.get("original_native_rank") if route.get("original_native_rank") is not None else route.get("native_rank"))
    except (TypeError, ValueError):
        return 10**9


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-9:
        return values * 0.0
    return (values - float(values.mean())) / std


def _counts(routes: list[dict[str, Any]]) -> dict[str, Any]:
    target_keys = {_target_key(route) for route in routes}
    return {
        "routes": len(routes),
        "targets": len(target_keys),
        "single_step_routes": sum(1 for route in routes if len(route.get("steps") or []) <= 1),
        "multi_step_routes": sum(1 for route in routes if len(route.get("steps") or []) > 1),
        "steps": sum(len(route.get("steps") or []) for route in routes),
        "blocks": sum(len((route.get("ccts_v3_route_evidence") or {}).get("block_scores") or []) for route in routes),
    }


def _score_summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "ccts_v3_best_block_context_sim",
        "ccts_v3_step_mean_context_sim",
        "ccts_v3_best_inferred_pair_score",
    ]
    out = {}
    for key in keys:
        out[key] = _numeric_summary([float(route.get(key) or 0.0) for route in routes])
    return out


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "p25": round(float(np.quantile(arr, 0.25)), 6),
        "median": round(float(np.median(arr)), 6),
        "p75": round(float(np.quantile(arr, 0.75)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _top1_changes(routes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in _with_group_blend_scores(routes):
        grouped[_target_key(route)].append(route)
    rows = []
    for target, group in grouped.items():
        native = sorted(group, key=lambda row: (_native_rank(row), str(row.get("route_id") or "")))[0]
        evidence = sorted(group, key=lambda row: (-float(row.get("ccts_v3_best_block_context_sim") or -1.0), _native_rank(row)))[0]
        blend = sorted(group, key=lambda row: (-float(row.get("ccts_v3_best_block_blend_score") or -1.0), _native_rank(row)))[0]
        rows.append(
            {
                "target": target,
                "native_route": native.get("route_id"),
                "native_rank": native.get("native_rank"),
                "evidence_route": evidence.get("route_id"),
                "evidence_original_rank": evidence.get("original_native_rank"),
                "blend_route": blend.get("route_id"),
                "blend_original_rank": blend.get("original_native_rank"),
                "evidence_changed": native.get("route_id") != evidence.get("route_id"),
                "blend_changed": native.get("route_id") != blend.get("route_id"),
            }
        )
    return {
        "targets": len(rows),
        "evidence_changed": sum(1 for row in rows if row.get("evidence_changed")),
        "blend_changed": sum(1 for row in rows if row.get("blend_changed")),
        "examples": rows[:30],
    }


def _recovery_comparison(reports: dict[str, Any]) -> dict[str, Any]:
    out = {}
    keys = [
        "routed_analog_recovery_at_1",
        "routed_analog_recovery_at_3",
        "routed_analog_recovery_at_5",
        "routed_analog_recovery_at_10",
        "routed_analog_recovery_at_50",
        "routed_transform_consistent_analog_recovery_at_10",
        "targets_with_analog_any",
        "targets_with_transform_consistent_analog_any",
    ]
    for name, report in reports.items():
        summary = report.get("summary") or {}
        out[name] = {key: summary.get(key) for key in keys}
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v3 Route-Pool Replay",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Block Recovery Comparison",
        "",
        "| Policy | routed analog @1 | @3 | @5 | @10 | @50 | strict analog @10 | strict any targets | targets analog any |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (report.get("block_recovery_comparison") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row.get("routed_analog_recovery_at_1")),
                    str(row.get("routed_analog_recovery_at_3")),
                    str(row.get("routed_analog_recovery_at_5")),
                    str(row.get("routed_analog_recovery_at_10")),
                    str(row.get("routed_analog_recovery_at_50")),
                    str(row.get("routed_transform_consistent_analog_recovery_at_10")),
                    str(row.get("targets_with_transform_consistent_analog_any")),
                    str(row.get("targets_with_analog_any")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Top-1 Changes",
            "",
            "```json",
            json.dumps(report.get("top1_changes") or {}, indent=2, ensure_ascii=False)[:6000],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_top_ks(value: str) -> tuple[int, ...]:
    out = tuple(sorted({int(item.strip()) for item in str(value).split(",") if item.strip()}))
    return out or (1, 3, 5, 10, 50, 100)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Replay CCTS-v3 candidate evidence on a fixed ChemEnzy route pool")
    ap.add_argument("--route-pool", required=True)
    ap.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--analog-similarity", type=float, default=0.55)
    ap.add_argument("--top-ks", default="1,3,5,10,50,100")
    ap.add_argument("--no-block-audit", action="store_true")
    args = ap.parse_args()
    report = replay_ccts_v3_on_route_pool(
        route_pool=Path(args.route_pool),
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        split=args.split,
        alpha=args.alpha,
        analog_similarity=args.analog_similarity,
        top_ks=_parse_top_ks(args.top_ks),
        run_block_audit=not args.no_block_audit,
    )
    print(json.dumps({"counts": report["counts"], "block_recovery_comparison": report["block_recovery_comparison"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
