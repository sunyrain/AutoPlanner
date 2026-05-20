"""Build route-level v4 cascade product-value feature packs."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.v4_product_value import (
    ROUTE_LABEL_NAMES,
    iter_jsonl,
    route_record_from_trace_candidate,
    route_record_from_v4,
)


PACK_SCHEMA_VERSION = "v4_cascade_product_value_pack.v1"


def build_v4_cascade_product_value_pack(
    *,
    v4_jsonl: Path,
    split_dir: Path | None,
    output_dir: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    split_rows = _read_split_rows(split_dir) if split_dir is not None else {}
    raw_by_key = {
        _record_key(row): row
        for row in iter_jsonl(v4_jsonl)
        if _record_key(row)
    }
    rows: list[dict[str, Any]] = []
    skipped = Counter()

    if split_rows:
        for split, candidates in split_rows.items():
            for candidate in candidates:
                key = _record_key(candidate)
                raw = raw_by_key.get(key)
                if raw is not None:
                    row = route_record_from_v4(
                        raw,
                        split=split,
                        split_group_id=str(candidate.get("split_group_id") or ""),
                    )
                else:
                    row = route_record_from_trace_candidate(candidate)
                    row["split"] = split
                if not row.get("steps"):
                    skipped["empty_steps"] += 1
                    continue
                rows.append(row)
    else:
        for raw in raw_by_key.values():
            if not _truthy(raw.get("trainable_recommended")):
                skipped["not_trainable_recommended"] += 1
                continue
            if ";" in str(raw.get("target_product_smiles") or ""):
                skipped["multi_target"] += 1
                continue
            row = route_record_from_v4(raw, split="train")
            if not row.get("steps"):
                skipped["empty_steps"] += 1
                continue
            rows.append(row)

    rows.sort(key=lambda row: (str(row.get("split") or ""), str(row.get("doi") or ""), str(row.get("cascade_id") or "")))
    if max_rows is not None and max_rows > 0:
        rows = rows[: int(max_rows)]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": output_dir / "cascade_v4_route_feature_pack_all.jsonl",
        "train": output_dir / "cascade_v4_route_feature_pack_train.jsonl",
        "val": output_dir / "cascade_v4_route_feature_pack_val.jsonl",
        "test": output_dir / "cascade_v4_route_feature_pack_test.jsonl",
    }
    _write_jsonl(paths["all"], rows)
    for split in ("train", "val", "test"):
        _write_jsonl(paths[split], [row for row in rows if row.get("split") == split])

    report = {
        "schema_version": PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "v4_jsonl": str(v4_jsonl),
            "split_dir": str(split_dir) if split_dir else None,
            "output_dir": str(output_dir),
            "max_rows": max_rows,
            "label_names": list(ROUTE_LABEL_NAMES),
            "training_contract": "route_level_v4_cascade_product_value_not_autoplanner_trace.v1",
        },
        "counts": {
            "rows": len(rows),
            "skipped": dict(skipped),
        },
        "splits": _split_summary(rows),
        "quality_tier_counts": dict(Counter(row.get("quality_tier") for row in rows)),
        "route_domain_counts": dict(Counter(row.get("route_domain") for row in rows)),
        "label_rates": _label_rates(rows),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    (output_dir / "cascade_v4_route_feature_pack_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(report), encoding="utf-8")
    return report


def _read_split_rows(split_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    if split_dir is None:
        return {}
    out = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"v4_trace_{split}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                out[split] = [row for row in data if isinstance(row, dict)]
    return out


def _record_key(row: dict[str, Any]) -> tuple[str, str] | None:
    doi = str(row.get("doi") or "").strip().lower()
    cascade_id = str(row.get("cascade_id") or "").strip().lower()
    if not doi or not cascade_id:
        return None
    return doi, cascade_id


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test", ""):
        split_rows = [row for row in rows if str(row.get("split") or "") == split]
        key = split or "unspecified"
        if not split_rows:
            continue
        out[key] = {
            "rows": len(split_rows),
            "unique_targets": len({row.get("target_smiles") for row in split_rows if row.get("target_smiles")}),
            "unique_doi": len({row.get("doi") for row in split_rows if row.get("doi")}),
            "quality_tier_counts": dict(Counter(row.get("quality_tier") for row in split_rows)),
            "route_domain_counts": dict(Counter(row.get("route_domain") for row in split_rows)),
        }
    return out


def _label_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    out = {}
    for label in ROUTE_LABEL_NAMES:
        out[label] = round(
            sum(float((row.get("labels") or {}).get(label) or 0.0) for row in rows) / len(rows),
            6,
        )
    return out


def _readme(report: dict[str, Any]) -> str:
    lines = [
        "# v4 Cascade Product-Value Feature Pack",
        "",
        "This pack is built from dataset_v4_release gold/silver cascade records.",
        "It is route-level supervision for a ChemEnzy native route reranker; it does not use AutoPlanner route-tree traces.",
        "",
        "## Counts",
        "",
        f"- Rows: `{report['counts']['rows']}`",
        f"- Skipped: `{report['counts']['skipped']}`",
        "",
        "## Splits",
        "",
        "```json",
        json.dumps(report.get("splits") or {}, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build route-level v4 cascade product-value feature packs")
    ap.add_argument("--v4-jsonl", default="dataset_v4_release/cascade_v4_high_quality.jsonl")
    ap.add_argument("--split-dir", default=None, help="Directory produced by build_v4_training_splits")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-rows", type=int)
    args = ap.parse_args()
    report = build_v4_cascade_product_value_pack(
        v4_jsonl=Path(args.v4_jsonl),
        split_dir=Path(args.split_dir) if args.split_dir else None,
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
    )
    print(json.dumps({"rows": report["counts"]["rows"], "outputs": report["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
