"""Replay a CascadeBlock coherence scorer on ChemEnzy route pools."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.eval.audit_candidate_specific_evidence import _train_bank
from cascade_planner.eval.train_cascade_block_coherence import (
    _dataset,
    _read_jsonl,
    _transform_pair,
)


ROUTE_POOL_REPLAY_SCHEMA_VERSION = "cascade_block_coherence_route_pool_replay.v1"


def replay_block_coherence_on_route_pool(
    *,
    route_pool: Path,
    model_pickle: Path,
    output_jsonl: Path,
    report_json: Path,
    model_name: str = "structure_plus_context",
    reranked_output_jsonl: Path | None = None,
    program_manifest: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    with model_pickle.open("rb") as fh:
        payload = pickle.load(fh)
    models = payload.get("models") or {}
    if model_name not in models:
        raise ValueError(f"model {model_name!r} not found in {model_pickle}; available={sorted(models)}")
    model = models[model_name]
    feature_schema = payload.get("feature_schema") or {}
    feature_names = feature_schema.get("feature_names") or []
    model_specs = feature_schema.get("model_specs") or {}
    feature_indices = [feature_names.index(name) for name in model_specs[model_name]]
    evidence = _empty_evidence(
        program_manifest=program_manifest,
        runtime_required=bool(feature_schema.get("runtime_evidence_features_enabled")),
    )
    routes = _read_jsonl(route_pool)
    scored_routes = []
    for route in routes:
        scored_routes.append(
            _score_route(route, model=model, feature_indices=feature_indices, feature_schema=feature_schema, evidence=evidence)
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in scored_routes:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = _report(
        scored_routes,
        route_pool=route_pool,
        model_pickle=model_pickle,
        output_jsonl=output_jsonl,
        report_json=report_json,
        model_name=model_name,
        elapsed_s=round(time.monotonic() - started, 3),
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    if reranked_output_jsonl is not None:
        _write_reranked_route_pool(scored_routes, reranked_output_jsonl)
        report.setdefault("outputs", {})["reranked_route_pool"] = str(reranked_output_jsonl)
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _score_route(
    route: dict[str, Any],
    *,
    model: Any,
    feature_indices: list[int],
    feature_schema: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    blocks = _route_blocks(route)
    if blocks:
        data = _dataset(blocks, schema=feature_schema, evidence=evidence)
        scores = model.predict_proba(data["x"][:, feature_indices], num_iteration=model.best_iteration_)[:, 1]
        for block, score in zip(blocks, scores):
            block["block_coherence_score"] = float(score)
    else:
        scores = np.asarray([], dtype=np.float32)
    summary = _route_score_summary(route, blocks=blocks, scores=scores)
    return {
        "route_id": route.get("route_id"),
        "target_id": route.get("target_id"),
        "target_smiles": route.get("target_smiles"),
        "original_native_rank": route.get("original_native_rank", route.get("native_rank")),
        "native_rank": route.get("native_rank"),
        "native_score": route.get("native_score"),
        "stock_closed": route.get("stock_closed"),
        "route_source": route.get("route_source"),
        "route_domain": route.get("route_domain"),
        "n_steps": len(route.get("steps") or []),
        "steps": route.get("steps") or [],
        "terminal_reactants": route.get("terminal_reactants") or [],
        "block_coherence": summary,
        "blocks": [_compact_block(block) for block in blocks],
    }


def _route_blocks(route: dict[str, Any]) -> list[dict[str, Any]]:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    blocks = []
    for idx, (left, right) in enumerate(zip(steps, steps[1:])):
        blocks.append(
            {
                "block_id": f"{route.get('route_id')}::{idx}",
                "positive_block_id": f"{route.get('route_id')}::{idx}",
                "program_id": route.get("route_id"),
                "doi": "",
                "cascade_id": route.get("target_id"),
                "target_smiles": route.get("target_smiles"),
                "route_domain": route.get("route_domain") or "unknown",
                "anchor_index": idx,
                "label": 0,
                "example_type": "route_pool_block",
                "compatibility_label": ((route.get("compatibility") or {}).get("compatibility_label") or ""),
                "compatibility_evidence_strength": ((route.get("compatibility") or {}).get("evidence_strength") or ""),
                "compatibility_issue_types": (route.get("compatibility") or {}).get("issue_types") or [],
                "compatibility_mitigation_strategies": (route.get("compatibility") or {}).get("mitigation_strategies") or [],
                "left_step": _normalize_step(left, idx),
                "right_step": _normalize_step(right, idx + 1),
            }
        )
    return blocks


def _normalize_step(step: dict[str, Any], fallback_pos: int) -> dict[str, Any]:
    cats = step.get("catalyst_classes")
    if cats is None:
        cats = sorted(
            {
                str(cat.get("catalyst_class"))
                for cat in step.get("catalyst_components") or []
                if isinstance(cat, dict) and cat.get("catalyst_class")
            }
        )
    return {
        "transition_id": step.get("step_id") or f"step_{fallback_pos}",
        "step_id": step.get("step_id") or f"step_{fallback_pos}",
        "step_index": step.get("step_index") or fallback_pos + 1,
        "step_pos": fallback_pos,
        "remaining_steps": 0,
        "rxn_smiles": step.get("rxn_smiles") or step.get("reaction_smiles") or "",
        "product_smiles": step.get("product_smiles") or ((step.get("products") or [""])[0] if isinstance(step.get("products"), list) else ""),
        "reactants": step.get("reactants") or [],
        "main_reactant": _main_reactant(step),
        "transformation_name": step.get("transformation_name") or "unknown",
        "transformation_superclass": step.get("transformation_superclass") or "unknown",
        "step_mode": step.get("step_mode") or "unknown",
        "pairwise_mode": step.get("pairwise_mode") or "unknown",
        "intermediate_isolated": step.get("intermediate_isolated"),
        "condition_tokens": _condition_tokens(step.get("step_conditions") or {}),
        "catalyst_classes": cats or [],
        "ec1_values": [],
        "enzyme_families": [],
        "cofactors": [],
        "metal_identities": [],
    }


def _main_reactant(step: dict[str, Any]) -> str:
    if step.get("main_reactant"):
        return str(step.get("main_reactant"))
    reactants = [str(value) for value in step.get("reactants") or [] if value]
    return max(reactants, key=len) if reactants else ""


def _condition_tokens(cond: dict[str, Any]) -> list[str]:
    tokens = []
    for key in ("solvent", "cosolvent", "buffer_name", "atmosphere", "reactor_type", "mixing_method"):
        value = str(cond.get(key) or "").strip().lower()
        if value and value != "not_specified":
            tokens.append(f"{key}:{value}")
    return sorted(set(tokens))


def _route_score_summary(route: dict[str, Any], *, blocks: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    if len(scores) == 0:
        return {
            "n_blocks": 0,
            "mean": None,
            "min": None,
            "max": None,
            "low_block_count_lt_0_25": 0,
            "transform_pairs": [],
            "route_coherence_score": None,
            "rerank_score": None,
            "eligible_for_block_rerank": False,
        }
    mean = float(np.mean(scores))
    min_score = float(np.min(scores))
    return {
        "n_blocks": int(len(scores)),
        "mean": mean,
        "min": min_score,
        "max": float(np.max(scores)),
        "low_block_count_lt_0_25": int(np.sum(scores < 0.25)),
        "transform_pairs": [_transform_pair(block.get("left_step") or {}, block.get("right_step") or {}) for block in blocks],
        "route_coherence_score": mean,
        "conservative_route_coherence_score": mean - 0.5 * max(0.0, 0.5 - min_score),
        "rerank_score": mean + 0.02 * _native_rank_score(route),
        "eligible_for_block_rerank": True,
    }


def _native_rank_score(route: dict[str, Any]) -> float:
    rank = _native_rank(route)
    return 1.0 / float(rank + 1)


def _compact_block(block: dict[str, Any]) -> dict[str, Any]:
    left = block.get("left_step") or {}
    right = block.get("right_step") or {}
    return {
        "block_id": block.get("block_id"),
        "score": block.get("block_coherence_score"),
        "left_step_id": left.get("step_id"),
        "right_step_id": right.get("step_id"),
        "left_transform": left.get("transformation_superclass"),
        "right_transform": right.get("transformation_superclass"),
        "transform_pair": _transform_pair(left, right),
        "left_product": left.get("product_smiles"),
        "right_product": right.get("product_smiles"),
    }


def _empty_evidence(*, program_manifest: Path | None = None, runtime_required: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "transform_pairs": Counter(),
        "catalyst_pairs": Counter(),
        "left_transform": Counter(),
        "right_transform": Counter(),
    }
    if runtime_required:
        if program_manifest is None:
            raise ValueError("runtime-evidence block coherence model requires --program-manifest")
        out["runtime_train_bank"] = _train_bank(program_manifest)
        out["runtime_product_sim_cache"] = {}
    return out


def _report(
    rows: list[dict[str, Any]],
    *,
    route_pool: Path,
    model_pickle: Path,
    output_jsonl: Path,
    report_json: Path,
    model_name: str,
    elapsed_s: float,
) -> dict[str, Any]:
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target_groups[str(row.get("target_id") or "")].append(row)
    top_changes = []
    top_changes_multistep = []
    for target_id, group in target_groups.items():
        native_top = sorted(group, key=lambda row: _native_rank(row))[0]
        reranked_top = sorted(group, key=lambda row: _sort_key(row))[0]
        top_changes.append(
            {
                "target_id": target_id,
                "native_top_route_id": native_top.get("route_id"),
                "native_top_rank": native_top.get("native_rank"),
                "native_top_n_steps": native_top.get("n_steps"),
                "native_top_score": (native_top.get("block_coherence") or {}).get("rerank_score"),
                "reranked_top_route_id": reranked_top.get("route_id"),
                "reranked_top_native_rank": reranked_top.get("native_rank"),
                "reranked_top_n_steps": reranked_top.get("n_steps"),
                "reranked_top_score": (reranked_top.get("block_coherence") or {}).get("rerank_score"),
                "changed": native_top.get("route_id") != reranked_top.get("route_id"),
            }
        )
        multistep_group = [row for row in group if (row.get("block_coherence") or {}).get("eligible_for_block_rerank")]
        if multistep_group:
            native_top_multi = sorted(multistep_group, key=lambda row: _native_rank(row))[0]
            reranked_top_multi = sorted(multistep_group, key=lambda row: _sort_key(row))[0]
            top_changes_multistep.append(
                {
                    "target_id": target_id,
                    "native_top_route_id": native_top_multi.get("route_id"),
                    "native_top_rank": native_top_multi.get("native_rank"),
                    "native_top_n_steps": native_top_multi.get("n_steps"),
                    "native_top_score": (native_top_multi.get("block_coherence") or {}).get("rerank_score"),
                    "reranked_top_route_id": reranked_top_multi.get("route_id"),
                    "reranked_top_native_rank": reranked_top_multi.get("native_rank"),
                    "reranked_top_n_steps": reranked_top_multi.get("n_steps"),
                    "reranked_top_score": (reranked_top_multi.get("block_coherence") or {}).get("rerank_score"),
                    "changed": native_top_multi.get("route_id") != reranked_top_multi.get("route_id"),
                }
            )
    block_scores = [
        block.get("score")
        for row in rows
        for block in row.get("blocks") or []
        if block.get("score") is not None
    ]
    route_scores = [
        (row.get("block_coherence") or {}).get("route_coherence_score")
        for row in rows
        if (row.get("block_coherence") or {}).get("route_coherence_score") is not None
    ]
    return {
        "schema_version": ROUTE_POOL_REPLAY_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "route_pool": str(route_pool),
            "model_pickle": str(model_pickle),
            "model_name": model_name,
            "output_jsonl": str(output_jsonl),
            "report_json": str(report_json),
            "elapsed_s": elapsed_s,
        },
        "counts": {
            "routes": len(rows),
            "targets": len(target_groups),
            "single_step_routes": sum(1 for row in rows if int(row.get("n_steps") or 0) <= 1),
            "multi_step_routes": sum(1 for row in rows if int(row.get("n_steps") or 0) > 1),
            "blocks": len(block_scores),
            "top1_changed_targets_all_routes": sum(1 for row in top_changes if row.get("changed")),
            "top1_changed_targets_multistep_only": sum(1 for row in top_changes_multistep if row.get("changed")),
        },
        "score_summary": {
            "block": _numeric_summary(block_scores),
            "route": _numeric_summary(route_scores),
            "by_n_steps": _by_n_steps(rows),
        },
        "top_changes": top_changes,
        "top_changes_multistep_only": top_changes_multistep,
        "route_examples": sorted([row for row in rows if (row.get("block_coherence") or {}).get("eligible_for_block_rerank")], key=lambda row: _sort_key(row))[:30],
    }


def _sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    score = (row.get("block_coherence") or {}).get("rerank_score")
    if score is None:
        score = -1.0
    return (-float(score), _native_rank(row), str(row.get("route_id") or ""))


def _write_reranked_route_pool(rows: list[dict[str, Any]], output_jsonl: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("target_id") or row.get("target_smiles") or "")].append(row)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for _target_id, group in sorted(groups.items()):
            for new_rank, row in enumerate(sorted(group, key=_sort_key)):
                out = dict(row)
                out["original_native_rank"] = row.get("original_native_rank", row.get("native_rank"))
                out["native_rank"] = new_rank
                out["rerank_policy"] = "cascade_block_coherence"
                fh.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")


def _native_rank(row: dict[str, Any]) -> int:
    value = row.get("native_rank")
    if value is None:
        return 10**9
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10**9


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _by_n_steps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        score = (row.get("block_coherence") or {}).get("route_coherence_score")
        if score is not None:
            grouped[int(row.get("n_steps") or 0)].append(float(score))
    return {str(key): _numeric_summary(values) for key, values in sorted(grouped.items())}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Block Coherence Route-Pool Replay",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Score Summary",
        "",
        "```json",
        json.dumps(report.get("score_summary") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top-1 Changes",
        "",
        "| Target | Native Top | Reranked Top | Changed |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("top_changes") or []:
        lines.append(
            f"| {row.get('target_id')} | {row.get('native_top_rank')} | {row.get('reranked_top_native_rank')} | {row.get('changed')} |"
        )
    lines.extend(
        [
            "",
            "## Top-1 Changes, Multi-Step Only",
            "",
            "| Target | Native Multi-Step Top | Reranked Multi-Step Top | Changed |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report.get("top_changes_multistep_only") or []:
        lines.append(
            f"| {row.get('target_id')} | {row.get('native_top_rank')} | {row.get('reranked_top_native_rank')} | {row.get('changed')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay CascadeBlock coherence scorer on a route pool JSONL")
    ap.add_argument("--route-pool", required=True)
    ap.add_argument("--model-pickle", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--model-name", default="structure_plus_context")
    ap.add_argument("--reranked-output-jsonl")
    ap.add_argument("--program-manifest")
    args = ap.parse_args()
    report = replay_block_coherence_on_route_pool(
        route_pool=Path(args.route_pool),
        model_pickle=Path(args.model_pickle),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        model_name=args.model_name,
        reranked_output_jsonl=Path(args.reranked_output_jsonl) if args.reranked_output_jsonl else None,
        program_manifest=Path(args.program_manifest) if args.program_manifest else None,
    )
    print(json.dumps({"counts": report["counts"], "score_summary": report["score_summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
