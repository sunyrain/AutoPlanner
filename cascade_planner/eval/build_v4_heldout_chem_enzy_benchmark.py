"""Build ChemEnzy target files from held-out CascadeProgramPack splits."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.v4_product_value import canonical_smiles, stable_id


SCHEMA_VERSION = "v4_heldout_chem_enzy_benchmark.v1"


def build_v4_heldout_chem_enzy_benchmark(
    *,
    program_manifest: Path,
    split: str,
    output: Path,
    report: Path,
    limit: int | None = None,
    min_steps: int = 2,
) -> dict[str, Any]:
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    split_path = Path((manifest.get("outputs") or {})[split])
    programs = _read_jsonl(split_path)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for program in programs:
        steps = _program_steps(program)
        if len(steps) < int(min_steps):
            continue
        target = canonical_smiles(str(program.get("target_smiles") or "")) or str(program.get("target_smiles") or "")
        if not target:
            continue
        by_target[target].append(program)

    rows = []
    for target in sorted(by_target, key=lambda smi: stable_id(split, smi)):
        refs = by_target[target]
        rows.append(
            {
                "target_smiles": target,
                "target_id": stable_id("v4_heldout_target", split, target),
                "split": split,
                "program_count": len(refs),
                "program_ids": [_program_id(row) for row in refs],
                "cascade_ids": [row.get("cascade_id") for row in refs],
                "dois": sorted({str(row.get("doi") or "") for row in refs if row.get("doi")}),
                "max_steps": max(len(_program_steps(row)) for row in refs),
                "adjacency_count": sum(_adjacency_count(row) for row in refs),
                "cascade_types": sorted({str(row.get("cascade_type") or row.get("route_domain") or "unknown") for row in refs}),
            }
        )
    if limit is not None and limit > 0:
        rows = rows[: int(limit)]

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "program_manifest": str(program_manifest),
            "split": split,
            "limit": limit,
            "min_steps": min_steps,
        },
        "targets": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": payload["metadata"]["generated_at"],
        "metadata": payload["metadata"] | {"output": str(output), "report": str(report)},
        "counts": {
            "source_programs": len(programs),
            "unique_targets_available": len(by_target),
            "targets_written": len(rows),
            "program_refs_written": sum(row.get("program_count") or 0 for row in rows),
            "adjacency_refs_written": sum(row.get("adjacency_count") or 0 for row in rows),
            "cascade_type_counts": dict(Counter(t for row in rows for t in row.get("cascade_types") or [])),
        },
        "outputs": {
            "benchmark": str(output),
            "report": str(report),
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"unsupported split format: {path}")
        return [row for row in payload if isinstance(row, dict)]
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _program_steps(program: dict[str, Any]) -> list[dict[str, Any]]:
    steps = program.get("steps")
    if isinstance(steps, list) and steps:
        return [step for step in steps if isinstance(step, dict)]
    gt_route = program.get("gt_route")
    if isinstance(gt_route, list):
        return [step for step in gt_route if isinstance(step, dict)]
    return []


def _program_id(program: dict[str, Any]) -> str:
    return str(program.get("program_id") or program.get("cascade_id") or program.get("doi") or "")


def _adjacency_count(program: dict[str, Any]) -> int:
    adjacencies = program.get("adjacencies")
    if isinstance(adjacencies, list) and adjacencies:
        return len(adjacencies)
    return max(0, len(_program_steps(program)) - 1)


def _markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    meta = report.get("metadata") or {}
    return "\n".join(
        [
            "# v4 Held-Out ChemEnzy Benchmark",
            "",
            f"- split: `{meta.get('split')}`",
            f"- targets written: `{counts.get('targets_written', 0)}`",
            f"- program refs written: `{counts.get('program_refs_written', 0)}`",
            f"- adjacency refs written: `{counts.get('adjacency_refs_written', 0)}`",
            f"- cascade types: `{counts.get('cascade_type_counts', {})}`",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build held-out v4 target files for ChemEnzy native search")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-steps", type=int, default=2)
    args = ap.parse_args()
    result = build_v4_heldout_chem_enzy_benchmark(
        program_manifest=Path(args.program_manifest),
        split=args.split,
        output=Path(args.output),
        report=Path(args.report),
        limit=args.limit,
        min_steps=args.min_steps,
    )
    print(json.dumps({"counts": result["counts"], "outputs": result["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
