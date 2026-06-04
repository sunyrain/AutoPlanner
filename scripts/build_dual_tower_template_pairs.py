#!/usr/bin/env python3
"""Build positive product-template pairs for a dual-tower template retriever."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "dual_tower_template_pairs.v1"


def main() -> None:
    args = _parse_args()
    template2idx, _idx2template = torch.load(args.templates_index, map_location="cpu")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    counts: Counter[str] = Counter()
    rows_written = 0
    with args.reaction_csv.open(encoding="utf-8") as src, args.output_jsonl.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(src)
        for row_idx, row in enumerate(reader):
            if row_idx < int(args.offset):
                continue
            template = str(row.get("templates") or "")
            template_id = template2idx.get(template)
            if template_id is None:
                counts["missing_template"] += 1
                continue
            product = str(row.get("prod_smiles") or "")
            reactants = str(row.get("react_smiles") or "")
            if not product or not reactants:
                counts["missing_smiles"] += 1
                continue
            out.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "row_idx": row_idx,
                        "product": product,
                        "reactants": reactants,
                        "template": template,
                        "template_id": int(template_id),
                        "source": args.source_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows_written += 1
            counts["written"] += 1
            if args.limit is not None and rows_written >= int(args.limit):
                break
            if args.progress_every and rows_written % int(args.progress_every) == 0:
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "written": rows_written,
                            "elapsed_s": round(elapsed, 3),
                            "rows_per_s": round(rows_written / max(elapsed, 1e-9), 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "reaction_csv": str(args.reaction_csv),
            "templates_index": str(args.templates_index),
            "source_name": args.source_name,
            "limit": args.limit,
            "offset": args.offset,
        },
        "summary": {
            "rows_written": rows_written,
            "template_classes": len(template2idx),
            "elapsed_s": round(time.monotonic() - started, 3),
            "counts": dict(counts),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_json.with_suffix(".md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "output_jsonl": str(args.output_jsonl)}, indent=2, ensure_ascii=False))


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dual Tower Template Pairs",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| rows_written | {summary['rows_written']} |",
        f"| template_classes | {summary['template_classes']} |",
        f"| elapsed_s | {summary['elapsed_s']} |",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reaction-csv",
        type=Path,
        default=Path(
            "vendor/ChemEnzyRetroPlanner/retro_planner/packages/graph_retrosyn/graph_retrosyn/data/raw/USPTO-remapped_tpl_prod_react.csv"
        ),
    )
    parser.add_argument(
        "--templates-index",
        type=Path,
        default=Path(
            "vendor/ChemEnzyRetroPlanner/retro_planner/packages/graph_retrosyn/graph_retrosyn/data/raw/templates_index.pkl"
        ),
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--source-name", default="chem_enzy_uspto_full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
