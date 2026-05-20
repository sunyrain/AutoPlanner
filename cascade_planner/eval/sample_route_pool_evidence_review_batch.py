"""Sample route-pool cascade evidence cases for expert/LLM review.

The input audits are diagnostic outputs from
``audit_route_pool_cascade_evidence``.  This sampler converts high-evidence and
low-evidence route examples into a compact review table with empty expert
fields.  Diagnostic labels remain separate from expert labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_evidence_review_batch.v1"

DEFAULT_CLASSES = (
    "same_pair_analog_supported",
    "any_analog_supported",
    "multistep_without_observed_pair",
)

EXAMPLE_KEY_TO_CLASS = {
    "same_pair_analog_supported_routes": "same_pair_analog_supported",
    "any_analog_supported_routes": "any_analog_supported",
    "multistep_without_observed_pair_examples": "multistep_without_observed_pair",
}


def sample_route_pool_evidence_review_batch(
    *,
    audit_jsons: list[Path],
    output_jsonl: Path,
    report_json: Path,
    output_csv: Path | None = None,
    per_class: int = 25,
    seed: int = 42,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> dict[str, Any]:
    rows = _candidate_rows(audit_jsons)
    rows = _dedupe_with_class_priority(rows, classes=classes)
    rng = random.Random(int(seed))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cls = str(row.get("evidence_class") or "")
        if cls in classes:
            grouped[cls].append(row)

    selected = []
    for cls in classes:
        candidates = sorted(grouped.get(cls) or [], key=_stable_sort_key)
        rng.shuffle(candidates)
        selected.extend(candidates[: max(0, int(per_class))])
    selected.sort(key=lambda row: (row["evidence_class"], row["source_pool"], row["review_id"]))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if output_csv is not None:
        _write_csv(selected, output_csv)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "audit_jsons": [str(path) for path in audit_jsons],
            "output_jsonl": str(output_jsonl),
            "output_csv": str(output_csv) if output_csv else None,
            "report_json": str(report_json),
            "per_class": per_class,
            "seed": seed,
            "classes": list(classes),
        },
        "summary": {
            "source_rows": len(rows),
            "sampled_rows": len(selected),
            "source_classes": dict(Counter(str(row.get("evidence_class") or "") for row in rows)),
            "sampled_classes": dict(Counter(str(row.get("evidence_class") or "") for row in selected)),
            "sampled_pools": dict(Counter(str(row.get("source_pool") or "") for row in selected)),
        },
        "review_contract": {
            "diagnostic_labels_are_not_ground_truth": True,
            "expert_fields_to_fill": [
                "expert_route_plausible",
                "expert_block_transform_correct",
                "expert_support_precedent_relevant",
                "expert_cascade_coherent",
                "expert_priority",
                "expert_reject_reason",
                "expert_comments",
            ],
        },
        "examples": selected[:10],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _candidate_rows(audit_jsons: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in audit_jsons:
        payload = _read_json(path)
        source_pool = _source_pool_name(path, payload)
        meta = payload.get("metadata") or {}
        for example_key, evidence_class in EXAMPLE_KEY_TO_CLASS.items():
            for route in (payload.get("examples") or {}).get(example_key) or []:
                row = _review_row(route, evidence_class=evidence_class, source_pool=source_pool, source_audit=path, metadata=meta)
                if row:
                    out.append(row)
    return out


def _review_row(
    route: dict[str, Any],
    *,
    evidence_class: str,
    source_pool: str,
    source_audit: Path,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    block = _class_specific_block(route, evidence_class=evidence_class)
    if not isinstance(block, dict):
        block = {}
    support_any = block.get("best_any_support") or {}
    support_same_pair = block.get("best_same_pair_support") or {}
    review_id = _review_id(source_pool, route, block, evidence_class)
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": review_id,
        "source_pool": source_pool,
        "source_audit": str(source_audit),
        "evidence_class": evidence_class,
        "target_id": route.get("target_id"),
        "target_smiles": route.get("target_smiles"),
        "route_id": route.get("route_id"),
        "native_rank": route.get("native_rank"),
        "native_score": route.get("native_score"),
        "stock_closed": route.get("stock_closed"),
        "n_steps": route.get("n_steps"),
        "n_blocks": route.get("n_blocks"),
        "transform_pairs": route.get("transform_pairs") or [],
        "diagnostic_labels": {
            "has_observed_pair_block": bool(route.get("has_observed_pair_block")),
            "has_any_analog_block": bool(route.get("has_any_analog_block")),
            "has_same_pair_analog_block": bool(route.get("has_same_pair_analog_block")),
            "observed_pair_block_count": route.get("observed_pair_block_count"),
            "any_analog_block_count": route.get("any_analog_block_count"),
            "same_pair_analog_block_count": route.get("same_pair_analog_block_count"),
        },
        "diagnostic_scores": {
            "known_transform_step_fraction": route.get("known_transform_step_fraction"),
            "best_any_block_min_sim": route.get("best_any_block_min_sim"),
            "best_same_pair_block_min_sim": route.get("best_same_pair_block_min_sim"),
            "audit_analog_similarity": metadata.get("analog_similarity"),
        },
        "review_block": {
            "route_block_index": block.get("route_block_index"),
            "upstream_rxn": block.get("upstream_rxn"),
            "downstream_rxn": block.get("downstream_rxn"),
            "upstream_product": block.get("upstream_product"),
            "downstream_product": block.get("downstream_product"),
            "upstream_main_reactant": block.get("upstream_main_reactant"),
            "downstream_main_reactant": block.get("downstream_main_reactant"),
            "upstream_transform": block.get("upstream_transform"),
            "downstream_transform": block.get("downstream_transform"),
            "transform_pair": block.get("transform_pair"),
            "pair_count_in_evidence": block.get("pair_count_in_evidence"),
            "pair_observed_in_evidence": block.get("pair_observed_in_evidence"),
            "best_any_block_min_sim": block.get("best_any_block_min_sim"),
            "best_any_block_mean_sim": block.get("best_any_block_mean_sim"),
            "best_same_pair_block_min_sim": block.get("best_same_pair_block_min_sim"),
            "best_same_pair_block_mean_sim": block.get("best_same_pair_block_mean_sim"),
            "any_analog_supported": block.get("any_analog_supported"),
            "same_pair_analog_supported": block.get("same_pair_analog_supported"),
        },
        "support_any": _compact_support(support_any),
        "support_same_pair": _compact_support(support_same_pair),
        "expert_route_plausible": None,
        "expert_block_transform_correct": None,
        "expert_support_precedent_relevant": None,
        "expert_cascade_coherent": None,
        "expert_priority": None,
        "expert_reject_reason": None,
        "expert_comments": None,
    }


def _class_specific_block(route: dict[str, Any], *, evidence_class: str) -> dict[str, Any] | None:
    blocks = [block for block in route.get("blocks") or [] if isinstance(block, dict)]
    if evidence_class == "same_pair_analog_supported":
        candidates = [block for block in blocks if block.get("same_pair_analog_supported")]
        return max(
            candidates,
            key=lambda row: (float(row.get("best_same_pair_block_min_sim") or 0.0), float(row.get("best_any_block_min_sim") or 0.0)),
            default=route.get("top_supported_block") or _best_block(blocks),
        )
    if evidence_class == "any_analog_supported":
        candidates = [block for block in blocks if block.get("any_analog_supported")]
        return max(
            candidates,
            key=lambda row: (float(row.get("best_any_block_min_sim") or 0.0), float(row.get("best_any_block_mean_sim") or 0.0)),
            default=route.get("top_supported_block") or _best_block(blocks),
        )
    if evidence_class == "multistep_without_observed_pair":
        candidates = [block for block in blocks if not block.get("pair_observed_in_evidence")]
        return min(
            candidates or blocks,
            key=lambda row: (
                float(row.get("best_any_block_min_sim") or 0.0),
                int(row.get("pair_count_in_evidence") or 0),
            ),
            default=route.get("top_supported_block") or _best_block(blocks),
        )
    return route.get("top_supported_block") or _best_block(blocks)


def _best_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not blocks:
        return None
    return max(
        blocks,
        key=lambda row: (
            int(bool(row.get("same_pair_analog_supported"))),
            float(row.get("best_same_pair_block_min_sim") or 0.0),
            int(bool(row.get("any_analog_supported"))),
            float(row.get("best_any_block_min_sim") or 0.0),
            int(row.get("pair_count_in_evidence") or 0),
        ),
    )


def _dedupe_with_class_priority(rows: list[dict[str, Any]], *, classes: tuple[str, ...]) -> list[dict[str, Any]]:
    priority = {cls: idx for idx, cls in enumerate(classes)}
    rows = sorted(rows, key=lambda row: (priority.get(str(row.get("evidence_class") or ""), 10**6), _stable_sort_key(row)))
    seen = set()
    out = []
    for row in rows:
        key = (row.get("source_pool"), row.get("route_id"), (row.get("review_block") or {}).get("route_block_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _compact_support(support: dict[str, Any]) -> dict[str, Any]:
    return {
        key: support.get(key)
        for key in (
            "program_id",
            "doi",
            "cascade_id",
            "quality_tier",
            "cascade_type",
            "transform_pair",
            "upstream_product",
            "downstream_product",
        )
        if key in support
    }


def _source_pool_name(path: Path, payload: dict[str, Any]) -> str:
    parent = path.parent.name
    if parent.startswith("route_pool_cascade_evidence_"):
        return parent.replace("route_pool_cascade_evidence_", "")
    route_pool = str(((payload.get("metadata") or {}).get("route_pool")) or "")
    if "statin" in route_pool:
        return "statin"
    if "full100" in route_pool:
        return "full100"
    if "route_pool_20" in route_pool or "test20" in route_pool:
        return "v4_test20"
    return parent or path.stem


def _review_id(source_pool: str, route: dict[str, Any], block: dict[str, Any], evidence_class: str) -> str:
    material = json.dumps(
        {
            "source_pool": source_pool,
            "class": evidence_class,
            "route_id": route.get("route_id"),
            "target": route.get("target_smiles"),
            "block": block.get("route_block_index"),
            "pair": block.get("transform_pair"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _stable_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    try:
        rank = int(row.get("native_rank") or 10**9)
    except (TypeError, ValueError):
        rank = 10**9
    return (str(row.get("target_smiles") or ""), rank, str(row.get("review_id") or ""))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "source_pool",
        "evidence_class",
        "target_smiles",
        "route_id",
        "native_rank",
        "stock_closed",
        "n_steps",
        "transform_pairs",
        "diagnostic_labels",
        "diagnostic_scores",
        "review_block",
        "support_any",
        "support_same_pair",
        "expert_route_plausible",
        "expert_block_transform_correct",
        "expert_support_precedent_relevant",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_comments",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("transform_pairs", "diagnostic_labels", "diagnostic_scores", "review_block", "support_any", "support_same_pair"):
                flat[key] = json.dumps(flat.get(key), ensure_ascii=False)
            writer.writerow({key: flat.get(key) for key in fieldnames})


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Route Pool Evidence Review Batch",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "Diagnostic evidence classes are not ground truth. Fill expert fields before using this as supervised data.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample route-pool cascade evidence review cases")
    parser.add_argument("--audit-json", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    args = parser.parse_args()
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    report = sample_route_pool_evidence_review_batch(
        audit_jsons=[Path(path) for path in args.audit_json],
        output_jsonl=Path(args.output_jsonl),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        report_json=Path(args.report_json),
        per_class=int(args.per_class),
        seed=int(args.seed),
        classes=classes,
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl, "report_json": args.report_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
