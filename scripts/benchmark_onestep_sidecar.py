#!/usr/bin/env python
"""Evaluate a provider-neutral one-step sidecar on product/reactant CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.sidecars.one_step import (  # noqa: E402
    build_one_step_request,
    run_one_step_sidecar,
)


def _canonical_side(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for fragment in str(text or "").split("."):
        mol = Chem.MolFromSmiles(fragment.strip())
        if mol is not None:
            values.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return tuple(sorted(values))


def _read_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if not {"input", "output"}.issubset(fieldnames):
            raise ValueError("dataset must contain input and output columns")
        rows = []
        for row in reader:
            if row.get("input") and row.get("output"):
                rows.append({"input": row["input"], "output": row["output"]})
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("dataset contains no usable rows")
    return rows


def _metrics(rows: list[dict[str, str]], response: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(result["query_id"]): result for result in response["results"]}
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures = 0
    for index, row in enumerate(rows, start=1):
        result = by_id[f"uspto50k-{index}"]
        expected = _canonical_side(row["output"])
        predicted = [tuple(candidate["reactant_smiles"]) for candidate in result["candidates"]]
        match_rank = next((rank for rank, value in enumerate(predicted, start=1) if value == expected), None)
        for k in hits:
            if match_rank is not None and match_rank <= k:
                hits[k] += 1
        elapsed = float(result.get("diagnostics", {}).get("elapsed_s") or 0.0)
        latencies.append(elapsed)
        if result.get("status") != "ok":
            failures += 1
        details.append(
            {
                "query_id": result["query_id"],
                "product_smiles": result.get("product_smiles"),
                "ground_truth_reactants": list(expected),
                "match_rank": match_rank,
                "candidate_count": len(predicted),
                "elapsed_s": elapsed,
            }
        )
    count = len(rows)
    return {
        "schema_version": "autoplanner.one_step_sidecar_benchmark.v1",
        "dataset": "USPTO50K tdc_test.csv",
        "sample_count": count,
        "metrics": {f"top_{k}_exact": hits[k] / count for k in sorted(hits)},
        "hit_counts": {f"top_{k}_exact": hits[k] for k in sorted(hits)},
        "failure_count": failures,
        "mean_query_latency_s": statistics.fmean(latencies),
        "median_query_latency_s": statistics.median(latencies),
        "sidecar": {
            "provider": response.get("provider"),
            "diagnostics": response.get("diagnostics"),
            "semantics": response.get("semantics"),
        },
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        default=os.environ.get("AUTOPLANNER_AIZYNTH_SIDECAR_PYTHON", ""),
        help="Python interpreter from the isolated AiZynthFinder environment",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data_external" / "uspto50k" / "tdc_test.csv",
    )
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "workspace" / "aizdata" / "uspto_model.onnx")
    parser.add_argument(
        "--templates",
        type=Path,
        default=REPO_ROOT / "workspace" / "aizdata" / "uspto_templates.csv.gz",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.python:
        parser.error("--python or AUTOPLANNER_AIZYNTH_SIDECAR_PYTHON is required")

    rows = _read_rows(args.dataset, max(1, int(args.limit)))
    request = build_one_step_request(
        [
            {
                "query_id": f"uspto50k-{index}",
                "product_smiles": row["input"],
                "top_k": max(1, int(args.top_k)),
            }
            for index, row in enumerate(rows, start=1)
        ]
    )
    command = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "run_aizynthfinder_onestep_sidecar.py"),
        "--model",
        str(args.model),
        "--templates",
        str(args.templates),
    ]
    response = run_one_step_sidecar(command, request, timeout_s=args.timeout_s)
    report = _metrics(rows, response)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
