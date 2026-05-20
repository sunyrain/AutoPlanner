"""Build ChemEnzy trace-generation benchmarks from dataset_v4_release."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import canonical_smiles


def build_v4_trace_benchmark(
    *,
    v4_jsonl: Path,
    benchmark_path: Path,
    output_path: Path,
    report_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    candidates, candidate_report = collect_v4_trace_candidates(
        v4_jsonl=v4_jsonl,
        benchmark_path=benchmark_path,
    )
    selected = _balanced_limit(candidates, limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "benchmark_path": str(benchmark_path),
            "output_path": str(output_path),
            "limit": limit,
        },
        "counts": {
            "candidate_rows": len(candidates),
            "selected_rows": len(selected),
            "skipped": dict(candidate_report["counts"]["skipped"]),
        },
        "candidate_report": candidate_report,
        "selected_route_domain_counts": dict(Counter(row.get("route_domain") for row in selected)),
        "selected_depth_counts": dict(Counter(len(row.get("gt_route") or []) for row in selected)),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def collect_v4_trace_candidates(
    *,
    v4_jsonl: Path,
    benchmark_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_rows = _read_benchmark(benchmark_path)
    excluded_keys = {
        (_norm(row.get("doi")), _norm(row.get("cascade_id")))
        for row in benchmark_rows
    }
    excluded_targets = {
        canonical_smiles(str(row.get("target_smiles") or "")) or str(row.get("target_smiles") or "")
        for row in benchmark_rows
        if row.get("target_smiles")
    }
    candidates = []
    skipped = Counter()
    source_rows = 0
    for row in _read_jsonl(v4_jsonl):
        source_rows += 1
        payload, reason = _row_to_benchmark_payload(row, excluded_keys=excluded_keys, excluded_targets=excluded_targets)
        if payload is None:
            skipped[reason or "unknown"] += 1
            continue
        candidates.append(payload)
    report = {
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "benchmark_path": str(benchmark_path),
            "full100_exclusion_policy": "exclude DOI/cascade_id and canonical target overlap",
        },
        "counts": {
            "source_rows": source_rows,
            "candidate_rows": len(candidates),
            "skipped": dict(skipped),
            "unique_candidate_targets": len({row.get("target_smiles") for row in candidates if row.get("target_smiles")}),
            "unique_candidate_doi": len({_norm(row.get("doi")) for row in candidates if _norm(row.get("doi"))}),
        },
        "route_domain_counts": dict(Counter(row.get("route_domain") for row in candidates)),
        "depth_counts": dict(Counter(len(row.get("gt_route") or []) for row in candidates)),
        "quality_tier_counts": dict(Counter(row.get("quality_tier") for row in candidates)),
        "compatibility_label_counts": dict(Counter(row.get("compatibility_label") for row in candidates)),
    }
    return candidates, report


def _row_to_benchmark_payload(
    row: dict[str, Any],
    *,
    excluded_keys: set[tuple[str, str]],
    excluded_targets: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    doi = _norm(row.get("doi"))
    cascade_id = _norm(row.get("cascade_id"))
    if (doi, cascade_id) in excluded_keys:
        return None, "benchmark_key_overlap"
    if not _truthy(row.get("trainable_recommended")):
        return None, "not_trainable_recommended"
    target_raw = str(row.get("target_product_smiles") or "").strip()
    if not target_raw or ";" in target_raw:
        return None, "missing_or_multi_target"
    target = canonical_smiles(target_raw)
    if not target:
        return None, "invalid_target_smiles"
    if target in excluded_targets:
        return None, "benchmark_target_overlap"
    steps = []
    for step in row.get("steps") or []:
        rxn = str(step.get("rxn_smiles") or "").strip()
        if not rxn or ">>" not in rxn:
            continue
        steps.append(
            {
                "step_index": step.get("step_index"),
                "rxn_smiles": rxn,
                "condition": step.get("step_conditions") or {},
                "ec_number": _first_ec(step),
                "catalyst_classes": [
                    cat.get("catalyst_class")
                    for cat in step.get("catalyst_components") or []
                    if cat.get("catalyst_class")
                ],
            }
        )
    if not steps:
        return None, "no_usable_gt_rxn"
    return {
        "target_smiles": target,
        "doi": row.get("doi"),
        "cascade_id": row.get("cascade_id"),
        "route_domain": row.get("cascade_type") or row.get("route_domain"),
        "depth": len(steps),
        "gt_route": steps,
        "source_dataset": "dataset_v4_release",
        "quality_tier": row.get("quality_tier"),
        "compatibility_label": (row.get("compatibility") or {}).get("compatibility_label"),
    }, None


def _balanced_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("route_domain") or "unknown")].append(row)
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit and any(buckets.values()):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def _first_ec(step: dict[str, Any]) -> str | None:
    for cat in step.get("catalyst_components") or []:
        if cat.get("ec_number"):
            return str(cat.get("ec_number"))
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _read_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("targets", "items", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported benchmark format: {path}")
    return [row for row in data if isinstance(row, dict)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build trace benchmark rows from dataset_v4_release")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    report = build_v4_trace_benchmark(
        v4_jsonl=Path(args.v4_jsonl),
        benchmark_path=Path(args.benchmark),
        output_path=Path(args.output),
        report_path=Path(args.report) if args.report else None,
        limit=args.limit,
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
