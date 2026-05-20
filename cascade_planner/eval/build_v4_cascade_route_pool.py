"""Normalize ChemEnzy native route pools into a v4-aware route pool JSONL."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.v4_product_value import route_record_from_native_route


def build_v4_cascade_route_pool(
    *,
    native_pool: Path,
    output_jsonl: Path,
    report_json: Path | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    run = json.loads(Path(native_pool).read_text(encoding="utf-8"))
    rows = []
    for target_index, target in enumerate(run.get("targets") or []):
        target_smiles = str(target.get("target_smiles") or "")
        target_id = str(target.get("cascade_id") or target.get("target_id") or target.get("index") or target_index)
        routes = target.get("routes") or []
        for native_rank, route in enumerate(routes):
            if top_k is not None and top_k > 0 and native_rank >= int(top_k):
                break
            rows.append(
                route_record_from_native_route(
                    route,
                    target_smiles=target_smiles,
                    target_id=target_id,
                    native_rank=native_rank,
                    dataset=str(run.get("metadata", {}).get("dataset") or "native_route_pool"),
                )
            )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "v4_cascade_route_pool.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "native_pool": str(native_pool),
            "output_jsonl": str(output_jsonl),
            "top_k": top_k,
        },
        "counts": {
            "routes": len(rows),
            "targets": len(run.get("targets") or []),
            "route_source_counts": dict(Counter(row.get("route_source") for row in rows)),
        },
        "outputs": {
            "jsonl": str(output_jsonl),
        },
    }
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize native ChemEnzy route pools into a v4-aware route pool")
    ap.add_argument("--native-pool", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report-json")
    ap.add_argument("--top-k", type=int)
    args = ap.parse_args()
    report = build_v4_cascade_route_pool(
        native_pool=Path(args.native_pool),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json) if args.report_json else None,
        top_k=args.top_k,
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
