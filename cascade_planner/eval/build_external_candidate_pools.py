"""Build listwise candidate pools from external single-step positives.

The external step-pair importer creates independent positive reactions. This
module turns those rows into ranking/search-policy supervision by pairing each
true precursor set with deterministic hard negatives sampled from similar
source/type/EC buckets.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cascade_planner.vnext.features import read_jsonl, stable_id, write_jsonl


def build_external_candidate_pools(
    *,
    external_step_pairs: Path,
    output_dir: Path,
    max_pools: int | None = None,
    max_candidates: int = 8,
    seed: int = 20260507,
) -> dict[str, Any]:
    rows = read_jsonl(external_step_pairs)
    rows = [row for row in rows if row.get("product") and row.get("candidate")]
    if max_pools is not None and len(rows) > max_pools:
        rows = sorted(rows, key=lambda row: stable_id(seed, row.get("step_id"), row.get("reaction_smiles")))[:max_pools]

    buckets = _build_negative_buckets(rows)
    pools = [
        _pool_from_positive(row, idx=idx, buckets=buckets, max_candidates=max_candidates, seed=seed)
        for idx, row in enumerate(rows)
    ]
    pools = [pool for pool in pools if pool and int(pool.get("positive_count") or 0) > 0 and len(pool.get("candidates") or []) >= 2]
    transitions = [_search_transition_from_pool(pool) for pool in pools]

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "external_candidate_pools.jsonl"
    transition_path = output_dir / "external_search_transitions.jsonl"
    write_jsonl(pool_path, pools)
    write_jsonl(transition_path, transitions)

    manifest = {
        "schema_version": "external_candidate_pools.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_external_step_pairs": str(external_step_pairs),
        "files": {
            "external_candidate_pools": str(pool_path),
            "external_search_transitions": str(transition_path),
            "manifest": str(output_dir / "manifest.json"),
            "report": str(output_dir / "report.md"),
        },
        "counts": {
            "source_step_pairs": len(rows),
            "candidate_pools": len(pools),
            "search_transitions": len(transitions),
        },
        "quality": _quality_summary(pools),
        "max_candidates": max_candidates,
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_report_markdown(manifest), encoding="utf-8")
    return manifest


def _build_negative_buckets(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source") or "external")
        typ = str(row.get("reaction_type") or "")
        ec1 = _ec1(row.get("ec"))
        keys = [
            (source, typ, ec1),
            (source, typ, ""),
            (source, "", ec1),
            ("", typ, ec1),
            ("", typ, ""),
            ("", "", ""),
        ]
        for key in keys:
            buckets[key].append(row)
    return buckets


def _pool_from_positive(
    row: dict[str, Any],
    *,
    idx: int,
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]],
    max_candidates: int,
    seed: int,
) -> dict[str, Any] | None:
    positive = _candidate_item(row, label=1.0, label_type="external_positive", rank=1)
    negatives = _sample_negatives(row, buckets=buckets, n=max(1, max_candidates - 1), seed=seed)
    candidates = [positive, *negatives]
    rng = random.Random(int(stable_id(seed, row.get("step_id"), row.get("product")), 16))
    rng.shuffle(candidates)
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        item["candidate"]["rank"] = rank
    product = row.get("product") or ""
    pool_id = stable_id("external_pool", row.get("step_id"), product, len(candidates))
    return {
        "pool_id": pool_id,
        "route_id": f"external:{pool_id}",
        "target_smiles": row.get("target_smiles") or product,
        "product": product,
        "step_index": 0,
        "source": "external_step_pair_hard_negatives",
        "external_step_id": row.get("step_id") or "",
        "external_source": row.get("source") or "",
        "reaction_type": row.get("reaction_type") or "",
        "ec": row.get("ec") or "",
        "candidates": candidates,
        "has_exact_gt": True,
        "positive_count": sum(1 for item in candidates if float(item.get("label") or 0.0) >= 0.75),
    }


def _sample_negatives(
    row: dict[str, Any],
    *,
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    source = str(row.get("source") or "external")
    typ = str(row.get("reaction_type") or "")
    ec1 = _ec1(row.get("ec"))
    keys = [
        (source, typ, ec1),
        (source, typ, ""),
        (source, "", ec1),
        ("", typ, ec1),
        ("", typ, ""),
        ("", "", ""),
    ]
    forbidden_ids = {row.get("step_id")}
    forbidden_rxns = {row.get("reaction_smiles")}
    product = row.get("product") or ""
    selected: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    rng = random.Random(int(stable_id("neg", seed, row.get("step_id"), product), 16))
    for key in keys:
        bucket = buckets.get(key) or []
        if not bucket:
            continue
        attempts = 0
        while len(selected) < n and attempts < max(50, n * 40):
            attempts += 1
            cand_row = bucket[rng.randrange(len(bucket))]
            if cand_row.get("step_id") in forbidden_ids or cand_row.get("reaction_smiles") in forbidden_rxns:
                continue
            if cand_row.get("product") == product:
                continue
            cand = cand_row.get("candidate") or {}
            sig = cand.get("rxn_smiles") or cand.get("reaction_smiles") or cand_row.get("reaction_smiles") or cand_row.get("step_id")
            if sig in seen_candidates:
                continue
            seen_candidates.add(str(sig))
            selected.append(_candidate_item(cand_row, label=0.0, label_type="external_hard_negative", rank=len(selected) + 2))
        if len(selected) >= n:
            break
    return selected[:n]


def _candidate_item(row: dict[str, Any], *, label: float, label_type: str, rank: int) -> dict[str, Any]:
    candidate = dict(row.get("candidate") or {})
    rxn = row.get("reaction_smiles") or candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or ""
    candidate.setdefault("rxn_smiles", rxn)
    candidate.setdefault("reaction_smiles", rxn)
    candidate.setdefault("source", row.get("source") or "external")
    candidate.setdefault("reaction_type", row.get("reaction_type") or "")
    candidate.setdefault("type", row.get("reaction_type") or "")
    candidate.setdefault("ec", row.get("ec") or "")
    candidate["external_step_id"] = row.get("step_id") or ""
    candidate["rank"] = rank
    return {
        "candidate_id": stable_id("external_candidate", row.get("step_id"), label_type, rxn),
        "rank": rank,
        "label": float(label),
        "label_type": label_type,
        "weight": float(row.get("weight") or 1.0) if label > 0 else 0.75,
        "gt_available": True,
        "candidate": candidate,
    }


def _search_transition_from_pool(pool: dict[str, Any]) -> dict[str, Any]:
    candidates = pool.get("candidates") or []
    labels = [float(item.get("label") or 0.0) for item in candidates]
    best_index = max(range(len(labels)), key=lambda idx: labels[idx]) if labels else -1
    if best_index < 0 or labels[best_index] <= 0.0:
        best_index = -1
    return {
        "transition_id": stable_id("external_transition", pool.get("pool_id")),
        "source": "external_candidate_pool_distillation",
        "route_id": pool.get("route_id", ""),
        "pool_id": pool.get("pool_id", ""),
        "target_smiles": pool.get("target_smiles", ""),
        "product": pool.get("product", ""),
        "step_index": pool.get("step_index", 0),
        "action_count": len(candidates),
        "best_action_index": best_index,
        "best_action_label": labels[best_index] if best_index >= 0 else 0.0,
        "action_labels": labels,
        "action_candidate_ids": [item.get("candidate_id", "") for item in candidates],
        "reward": max(labels) if labels else 0.0,
        "has_positive_action": any(label >= 0.75 for label in labels),
    }


def _quality_summary(pools: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [len(pool.get("candidates") or []) for pool in pools]
    positives = [int(pool.get("positive_count") or 0) for pool in pools]
    return {
        "sources": dict(Counter(pool.get("external_source") for pool in pools)),
        "reaction_types": dict(Counter(pool.get("reaction_type") for pool in pools).most_common(20)),
        "with_ec": sum(1 for pool in pools if pool.get("ec")),
        "avg_candidates": round(sum(candidate_counts) / max(len(candidate_counts), 1), 3),
        "min_candidates": min(candidate_counts, default=0),
        "max_candidates": max(candidate_counts, default=0),
        "positive_count_distribution": dict(Counter(positives)),
    }


def _report_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest.get("counts") or {}
    quality = manifest.get("quality") or {}
    return "\n".join([
        "# External Candidate Pools",
        "",
        f"- source step pairs: `{counts.get('source_step_pairs', 0)}`",
        f"- candidate pools: `{counts.get('candidate_pools', 0)}`",
        f"- search transitions: `{counts.get('search_transitions', 0)}`",
        f"- average candidates: `{quality.get('avg_candidates', 0)}`",
        f"- sources: `{quality.get('sources', {})}`",
        "",
        "Each pool contains a known external single-step precursor as a positive action plus deterministic hard negatives from similar source/type/EC buckets.",
        "",
    ])


def _ec1(value: Any) -> str:
    text = str(value or "")
    return text.split(".", 1)[0] if text else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external candidate pools from external step pairs")
    parser.add_argument("--external-step-pairs", default="results/shared/external_step_pairs/full_20260507/external_step_pairs.jsonl")
    parser.add_argument("--output-dir", default="results/shared/external_candidate_pools/current")
    parser.add_argument("--max-pools", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()
    manifest = build_external_candidate_pools(
        external_step_pairs=Path(args.external_step_pairs),
        output_dir=Path(args.output_dir),
        max_pools=args.max_pools,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    print(json.dumps({"counts": manifest["counts"], "quality": manifest["quality"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
