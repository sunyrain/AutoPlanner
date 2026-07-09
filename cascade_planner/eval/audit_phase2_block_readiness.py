"""Audit whether Phase-II should be route-rerank, reaction injection, or block-centric.

This diagnostic deliberately avoids training a new scorer.  It compares three
facts on the same held-out v4 cascade blocks:

1. whether ChemEnzy route pools already contain transform-consistent cascade
   blocks;
2. whether v4-train evidence can retrieve the held-out block's cascade type and
   structural analogue;
3. whether the evidence is strong enough for direct reaction injection or only
   for block/type priors.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.cascade_retrieval_provider import CascadeRetrievalProvider
from cascade_planner.cascade_search.v4_product_value import canonical_smiles


SCHEMA_VERSION = "phase2_block_readiness_audit.v1"
TOP_KS = (1, 3, 5, 10, 20)


def audit_phase2_block_readiness(
    *,
    program_manifest: Path,
    output_json: Path,
    report: Path,
    split: str = "test",
    route_pool_recovery: Path | None = None,
    limit: int = 20,
    min_similarity: float = 0.20,
    weak_analog_similarity: float = 0.40,
    strong_analog_similarity: float = 0.55,
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = _read_json(program_manifest)
    train_programs = _read_jsonl(Path((manifest.get("outputs") or {})["train"]))
    heldout_programs = _read_jsonl(Path((manifest.get("outputs") or {})[split]))
    train_blocks = _program_blocks(train_programs)
    heldout_blocks = _program_blocks(heldout_programs)
    route_pool = _route_pool_index(route_pool_recovery) if route_pool_recovery else {}

    provider = CascadeRetrievalProvider(program_manifest)
    modes = ("block_downstream_product", "block_downstream_transition")
    contexts = ("none", "downstream_transform")
    retrieval_reports = {}
    for mode in modes:
        for context in contexts:
            key = f"{mode}:{context}"
            rows = [
                _retrieve_one(
                    provider,
                    block,
                    mode=mode,
                    limit=limit,
                    min_similarity=min_similarity,
                    weak_analog_similarity=weak_analog_similarity,
                    strong_analog_similarity=strong_analog_similarity,
                    condition_on_downstream_transform=(context == "downstream_transform"),
                )
                for block in heldout_blocks
            ]
            retrieval_reports[key] = _retrieval_summary(rows, train_blocks=train_blocks, route_pool=route_pool, limit=limit)

    support = _support_summary(train_blocks, heldout_blocks)
    route_pool_summary = _route_pool_summary(route_pool, heldout_blocks)
    decision = _decision(route_pool_summary, retrieval_reports, limit=limit)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "split": split,
            "route_pool_recovery": str(route_pool_recovery) if route_pool_recovery else None,
            "limit": limit,
            "min_similarity": min_similarity,
            "weak_analog_similarity": weak_analog_similarity,
            "strong_analog_similarity": strong_analog_similarity,
            "elapsed_s": round(time.monotonic() - started, 3),
            "leakage_guard": "retrieval provider indexes train split only; labels and blocks are read from the requested held-out split",
        },
        "counts": {
            "train_programs": len(train_programs),
            "train_blocks": len(train_blocks),
            "heldout_programs": len(heldout_programs),
            "heldout_blocks": len(heldout_blocks),
            "heldout_targets": len({row["target_smiles"] for row in heldout_blocks}),
        },
        "train_support": support,
        "route_pool": route_pool_summary,
        "retrieval": retrieval_reports,
        "decision": decision,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown(result), encoding="utf-8")
    return result


def _program_blocks(programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    for program in programs:
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        for idx, (left, right) in enumerate(zip(steps, steps[1:])):
            upstream_transform = _norm_transform(left.get("transformation_superclass"))
            downstream_transform = _norm_transform(right.get("transformation_superclass"))
            upstream_product = _canon(left.get("product_smiles"))
            upstream_main = _canon(left.get("main_reactant"))
            downstream_product = _canon(right.get("product_smiles"))
            downstream_main = _canon(right.get("main_reactant"))
            target = _canon(program.get("target_smiles"))
            blocks.append(
                {
                    "block_id": f"{program.get('program_id')}::{idx}",
                    "program_id": str(program.get("program_id") or ""),
                    "doi": str(program.get("doi") or ""),
                    "cascade_id": str(program.get("cascade_id") or ""),
                    "cascade_type": str(program.get("cascade_type") or "unknown"),
                    "quality_tier": str(program.get("quality_tier") or ""),
                    "target_smiles": target,
                    "upstream_product": upstream_product,
                    "upstream_main_reactant": upstream_main,
                    "downstream_product": downstream_product,
                    "downstream_main_reactant": downstream_main,
                    "upstream_transform": upstream_transform,
                    "downstream_transform": downstream_transform,
                    "transform_pair": f"{upstream_transform}->{downstream_transform}",
                    "catalyst_pair": _catalyst_pair(left, right),
                    "left_catalyst_classes": _norm_list(left.get("catalyst_classes")),
                    "right_catalyst_classes": _norm_list(right.get("catalyst_classes")),
                    "left_condition_tokens": _norm_list(left.get("condition_tokens")),
                    "right_condition_tokens": _norm_list(right.get("condition_tokens")),
                    "hidden_or_nonisolated": bool(left.get("intermediate_isolated") is False),
                    "upstream_fp": _transition_fp(upstream_product, upstream_main),
                    "downstream_fp": _transition_fp(downstream_product, downstream_main),
                }
            )
    return blocks


def _retrieve_one(
    provider: CascadeRetrievalProvider,
    block: dict[str, Any],
    *,
    mode: str,
    limit: int,
    min_similarity: float,
    weak_analog_similarity: float,
    strong_analog_similarity: float,
    condition_on_downstream_transform: bool,
) -> dict[str, Any]:
    kwargs = {
        "mode": mode,
        "limit": limit,
        "min_similarity": min_similarity,
        "required_downstream_transform": block["downstream_transform"] if condition_on_downstream_transform else None,
        "exclude_program_ids": {str(block.get("program_id") or "")},
    }
    if mode.endswith("_transition"):
        hits = provider.retrieve_for_transition(block["downstream_product"], block["downstream_main_reactant"], **kwargs)
    else:
        hits = provider.retrieve_for_product(block["downstream_product"], **kwargs)
    hit_rows = []
    for rank, hit in enumerate(hits, start=1):
        upstream_sim = _fp_similarity(block.get("upstream_fp"), _transition_fp(hit.product_smiles, hit.main_reactant))
        hit_rows.append(
            {
                "rank": rank,
                "hit_id": hit.hit_id,
                "similarity": round(float(hit.similarity), 6),
                "upstream_similarity": round(float(upstream_sim), 6),
                "transform_pair": hit.transform_pair,
                "transform_pair_hit": hit.transform_pair == block["transform_pair"],
                "upstream_transform_hit": hit.transformation_superclass == block["upstream_transform"],
                "weak_analog_hit": upstream_sim >= weak_analog_similarity,
                "strong_analog_hit": upstream_sim >= strong_analog_similarity,
                "doi": hit.doi,
                "program_id": hit.program_id,
            }
        )
    return {
        "block_id": block["block_id"],
        "target_smiles": block["target_smiles"],
        "transform_pair": block["transform_pair"],
        "upstream_transform": block["upstream_transform"],
        "downstream_transform": block["downstream_transform"],
        "cascade_type": block["cascade_type"],
        "hidden_or_nonisolated": block["hidden_or_nonisolated"],
        "n_hits": len(hit_rows),
        "best_similarity": hit_rows[0]["similarity"] if hit_rows else None,
        "best_upstream_similarity": max((row["upstream_similarity"] for row in hit_rows), default=None),
        "hit_at": _hit_at(hit_rows, limit=limit),
        "top_hits": hit_rows[:5],
    }


def _hit_at(hit_rows: list[dict[str, Any]], *, limit: int) -> dict[str, dict[str, bool]]:
    ks = sorted({k for k in TOP_KS if k <= limit} | {limit})
    out = {}
    for k in ks:
        top = hit_rows[:k]
        out[str(k)] = {
            "transform_pair": any(row["transform_pair_hit"] for row in top),
            "upstream_transform": any(row["upstream_transform_hit"] for row in top),
            "weak_analog": any(row["weak_analog_hit"] for row in top),
            "strong_analog": any(row["strong_analog_hit"] for row in top),
            "pair_and_weak_analog": any(row["transform_pair_hit"] and row["weak_analog_hit"] for row in top),
            "pair_and_strong_analog": any(row["transform_pair_hit"] and row["strong_analog_hit"] for row in top),
        }
    return out


def _retrieval_summary(
    rows: list[dict[str, Any]],
    *,
    train_blocks: list[dict[str, Any]],
    route_pool: dict[str, dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    pair_counts = Counter(row["transform_pair"] for row in train_blocks)
    row_groups = {
        "all": rows,
        "train_pair_seen": [row for row in rows if pair_counts.get(row["transform_pair"], 0) > 0],
        "train_pair_unseen": [row for row in rows if pair_counts.get(row["transform_pair"], 0) == 0],
        "routed_targets": [row for row in rows if _has_route_pool(_route_row(route_pool, row.get("target_smiles")))],
        "hidden_or_nonisolated": [row for row in rows if row.get("hidden_or_nonisolated")],
    }
    return {
        "summary": {name: _metric_summary(group, limit=limit) for name, group in row_groups.items()},
        "top_transform_pair_misses": _top_pair_misses(rows, metric="pair_and_weak_analog", k=limit),
        "diagnostic_examples": _diagnostic_examples(rows, route_pool=route_pool, limit=limit),
    }


def _metric_summary(rows: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    total = len(rows)
    out: dict[str, Any] = {
        "blocks": total,
        "with_hits": sum(1 for row in rows if row["n_hits"] > 0),
        "with_hits_rate": _rate(sum(1 for row in rows if row["n_hits"] > 0), total),
    }
    for k in sorted({value for value in TOP_KS if value <= limit} | {limit}):
        key = str(k)
        for metric in (
            "transform_pair",
            "upstream_transform",
            "weak_analog",
            "strong_analog",
            "pair_and_weak_analog",
            "pair_and_strong_analog",
        ):
            out[f"{metric}_at_{k}"] = _rate(
                sum(1 for row in rows if (row.get("hit_at") or {}).get(key, {}).get(metric)),
                total,
            )
    return out


def _support_summary(train_blocks: list[dict[str, Any]], heldout_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    train_pairs = Counter(row["transform_pair"] for row in train_blocks)
    train_up = Counter(row["upstream_transform"] for row in train_blocks)
    train_down = Counter(row["downstream_transform"] for row in train_blocks)
    train_catalyst_pairs = Counter(row["catalyst_pair"] for row in train_blocks)
    heldout_pairs = Counter(row["transform_pair"] for row in heldout_blocks)
    return {
        "heldout_blocks": len(heldout_blocks),
        "train_blocks": len(train_blocks),
        "transform_pair_seen_rate": _rate(sum(1 for row in heldout_blocks if train_pairs.get(row["transform_pair"], 0) > 0), len(heldout_blocks)),
        "upstream_transform_seen_rate": _rate(sum(1 for row in heldout_blocks if train_up.get(row["upstream_transform"], 0) > 0), len(heldout_blocks)),
        "downstream_transform_seen_rate": _rate(sum(1 for row in heldout_blocks if train_down.get(row["downstream_transform"], 0) > 0), len(heldout_blocks)),
        "catalyst_pair_seen_rate": _rate(sum(1 for row in heldout_blocks if train_catalyst_pairs.get(row["catalyst_pair"], 0) > 0), len(heldout_blocks)),
        "top_heldout_transform_pairs": dict(heldout_pairs.most_common(20)),
        "top_unseen_transform_pairs": dict(Counter(row["transform_pair"] for row in heldout_blocks if train_pairs.get(row["transform_pair"], 0) == 0).most_common(20)),
    }


def _route_pool_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = _read_json(path)
    out = {}
    for row in payload.get("targets") or []:
        target = _canon(row.get("target_smiles"))
        if target:
            out[target] = row
    return out


def _route_pool_summary(route_pool: dict[str, dict[str, Any]], heldout_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    targets = sorted({row["target_smiles"] for row in heldout_blocks})
    routed_rows = [_route_row(route_pool, target) for target in targets]
    routed_rows = [row for row in routed_rows if _has_route_pool(row)]
    transform_consistent = sum(1 for row in routed_rows if row.get("best_transform_consistent_analog_native_rank") is not None)
    exact = sum(1 for row in routed_rows if row.get("best_exact_native_rank") is not None)
    analog = sum(1 for row in routed_rows if row.get("best_analog_native_rank") is not None)
    return {
        "heldout_targets": len(targets),
        "targets_with_route_pool": len(routed_rows),
        "route_pool_target_rate": _rate(len(routed_rows), len(targets)),
        "avg_routes_on_routed_targets": round(sum(float(row.get("route_count") or 0) for row in routed_rows) / max(len(routed_rows), 1), 6),
        "routed_targets_with_exact_block": exact,
        "routed_targets_with_analog_block": analog,
        "routed_targets_with_transform_consistent_analog_block": transform_consistent,
        "routed_transform_consistent_rate": _rate(transform_consistent, len(routed_rows)),
    }


def _decision(route_pool: dict[str, Any], retrieval_reports: dict[str, Any], *, limit: int) -> dict[str, Any]:
    best_key = ""
    best_pair = -1.0
    best_pair_and_strong = -1.0
    for key, payload in retrieval_reports.items():
        summary = ((payload.get("summary") or {}).get("all") or {})
        pair = float(summary.get(f"transform_pair_at_{limit}") or 0.0)
        pair_and_strong = float(summary.get(f"pair_and_strong_analog_at_{limit}") or 0.0)
        if pair > best_pair:
            best_key = key
            best_pair = pair
            best_pair_and_strong = pair_and_strong
    route_tc = float(route_pool.get("routed_transform_consistent_rate") or 0.0)
    route_count = int(route_pool.get("targets_with_route_pool") or 0)
    conclusions = []
    if route_count and route_tc <= 0.05 and best_pair >= 0.40:
        conclusions.append("same_route_pool_rerank_is_wrong_target")
    if best_pair >= 0.40 and best_pair_and_strong < 0.20:
        conclusions.append("use_block_or_transform_prior_before_direct_reaction_injection")
    if best_pair_and_strong >= 0.20:
        conclusions.append("reaction_level_retrieval_injection_may_be_testable_but_must_be_guarded")
    if best_pair < 0.25:
        conclusions.append("train_evidence_is_too_sparse_for_retrieval_only")
    return {
        "best_retrieval_setting": best_key,
        "best_transform_pair_at_limit": round(best_pair, 6),
        "best_pair_and_strong_analog_at_limit": round(best_pair_and_strong, 6),
        "route_pool_transform_consistent_rate_on_routed_targets": round(route_tc, 6),
        "conclusions": conclusions,
        "recommended_next_experiment": (
            "Build a block-centric candidate layer: retrieve/score cascade block types from v4 train, "
            "then ask ChemEnzy to solve entry/peripheral substrates. Do not keep training a reranker over "
            "unchanged ChemEnzy complete-route pools until transform-consistent block coverage is improved."
        ),
    }


def _top_pair_misses(rows: list[dict[str, Any]], *, metric: str, k: int) -> dict[str, int]:
    counter = Counter()
    key = str(k)
    for row in rows:
        if not (row.get("hit_at") or {}).get(key, {}).get(metric):
            counter[row.get("transform_pair") or "unknown"] += 1
    return dict(counter.most_common(20))


def _diagnostic_examples(rows: list[dict[str, Any]], *, route_pool: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples = []
    key = str(limit)
    for row in rows:
        route_row = _route_row(route_pool, row.get("target_smiles"))
        if not _has_route_pool(route_row):
            continue
        hit_at = (row.get("hit_at") or {}).get(key, {})
        route_has_tc = route_row.get("best_transform_consistent_analog_native_rank") is not None
        if route_has_tc:
            continue
        if hit_at.get("transform_pair") or hit_at.get("pair_and_weak_analog"):
            examples.append(
                {
                    "block_id": row.get("block_id"),
                    "target_smiles": row.get("target_smiles"),
                    "transform_pair": row.get("transform_pair"),
                    "route_count": route_row.get("route_count"),
                    "route_best_analog_rank": route_row.get("best_analog_native_rank"),
                    "retrieval_pair_hit": bool(hit_at.get("transform_pair")),
                    "retrieval_pair_and_weak_analog": bool(hit_at.get("pair_and_weak_analog")),
                    "top_hits": row.get("top_hits") or [],
                }
            )
        if len(examples) >= 10:
            break
    return examples


def _route_row(route_pool: dict[str, dict[str, Any]], target_smiles: Any) -> dict[str, Any] | None:
    if not route_pool:
        return None
    return route_pool.get(_canon(target_smiles))


def _has_route_pool(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    return int(row.get("route_count") or 0) > 0


def _catalyst_pair(left: dict[str, Any], right: dict[str, Any]) -> str:
    return f"{'+'.join(_norm_list(left.get('catalyst_classes'))) or 'none'}->{'+'.join(_norm_list(right.get('catalyst_classes'))) or 'none'}"


def _transition_fp(product_smiles: Any, main_reactant: Any):
    product_fp = _fp(product_smiles)
    reactant_fp = _fp(main_reactant)
    if product_fp is None and reactant_fp is None:
        return None
    if product_fp is None:
        return reactant_fp
    if reactant_fp is None:
        return product_fp
    arr_product = np.zeros((2048,), dtype=np.int8)
    arr_reactant = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(product_fp, arr_product)
    DataStructs.ConvertToNumpyArray(reactant_fp, arr_reactant)
    arr = np.maximum(arr_product, arr_reactant)
    fp = DataStructs.ExplicitBitVect(2048)
    for bit in np.nonzero(arr)[0]:
        fp.SetBit(int(bit))
    return fp


def _fp(smiles: Any):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _fp_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _canon(value: Any) -> str:
    return canonical_smiles(str(value or "")) or str(value or "")


def _norm_transform(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _norm_list(values: Any) -> list[str]:
    return sorted({str(value).strip().lower() for value in (values or []) if str(value).strip()})


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    route_pool = result.get("route_pool") or {}
    decision = result.get("decision") or {}
    lines = [
        "# Phase-II Block Readiness Audit",
        "",
        "## 结论",
        "",
        f"- 最佳 retrieval 设置: `{decision.get('best_retrieval_setting')}`",
        f"- 最佳 transform-pair@limit: `{decision.get('best_transform_pair_at_limit')}`",
        f"- 最佳 pair+strong-analog@limit: `{decision.get('best_pair_and_strong_analog_at_limit')}`",
        f"- ChemEnzy routed target 中 transform-consistent block rate: `{decision.get('route_pool_transform_consistent_rate_on_routed_targets')}`",
        f"- 判断标签: `{', '.join(decision.get('conclusions') or [])}`",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Train Evidence Support",
        "",
        "```json",
        json.dumps(result.get("train_support") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## ChemEnzy Route Pool",
        "",
        "```json",
        json.dumps(route_pool, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Retrieval Readiness",
        "",
        "| Setting | Blocks | With Hits | Pair@20 | Weak Analog@20 | Strong Analog@20 | Pair+Weak@20 | Pair+Strong@20 | Routed Pair@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, payload in (result.get("retrieval") or {}).items():
        all_summary = ((payload.get("summary") or {}).get("all") or {})
        routed_summary = ((payload.get("summary") or {}).get("routed_targets") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    str(all_summary.get("blocks")),
                    str(all_summary.get("with_hits_rate")),
                    str(all_summary.get("transform_pair_at_20")),
                    str(all_summary.get("weak_analog_at_20")),
                    str(all_summary.get("strong_analog_at_20")),
                    str(all_summary.get("pair_and_weak_analog_at_20")),
                    str(all_summary.get("pair_and_strong_analog_at_20")),
                    str(routed_summary.get("transform_pair_at_20")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "ChemEnzy complete-route pool can solve many targets, but current routed held-out targets still have near-zero transform-consistent cascade-block recovery. "
            "By contrast, v4 train evidence often recovers the held-out transform-pair/type signal, while structural pair+analog recovery remains much lower.",
            "",
            "Therefore the immediate research object should not be a final route reranker over unchanged ChemEnzy pools. "
            "The next model should operate on cascade blocks or transform-pair priors, then use ChemEnzy for entry-substrate/peripheral synthesis.",
            "",
            "## Recommended Next Experiment",
            "",
            str(decision.get("recommended_next_experiment") or ""),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Audit Phase-II block-centric readiness")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--route-pool-recovery")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min-similarity", type=float, default=0.20)
    ap.add_argument("--weak-analog-similarity", type=float, default=0.40)
    ap.add_argument("--strong-analog-similarity", type=float, default=0.55)
    args = ap.parse_args()
    result = audit_phase2_block_readiness(
        program_manifest=Path(args.program_manifest),
        output_json=Path(args.output_json),
        report=Path(args.report),
        split=args.split,
        route_pool_recovery=Path(args.route_pool_recovery) if args.route_pool_recovery else None,
        limit=args.limit,
        min_similarity=args.min_similarity,
        weak_analog_similarity=args.weak_analog_similarity,
        strong_analog_similarity=args.strong_analog_similarity,
    )
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "route_pool": result["route_pool"],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
