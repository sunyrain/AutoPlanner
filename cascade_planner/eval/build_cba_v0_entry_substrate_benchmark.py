"""Build a ChemEnzy benchmark for CBA-v0 guarded sketch entry substrates."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.v4_product_value import canonical_smiles, stable_id


SCHEMA_VERSION = "cba_v0_entry_substrate_benchmark.v1"


def build_cba_v0_entry_substrate_benchmark(
    *,
    sketch_pack: Path,
    output: Path,
    report: Path,
    max_targets: int | None = None,
    routed_only: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    sketches = _read_jsonl(sketch_pack)
    candidates = [row for row in sketches if row.get("enter_next_stage")]
    if routed_only:
        candidates = [row for row in candidates if (row.get("chem_enzy_context") or {}).get("routed_by_chem_enzy")]
    by_entry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sketch in candidates:
        entry = canonical_smiles(str((sketch.get("prototype") or {}).get("upstream_main_reactant") or ""))
        if not entry:
            continue
        by_entry[entry].append(sketch)
    rows = []
    for entry, refs in sorted(by_entry.items(), key=lambda item: stable_id("entry", item[0])):
        best = sorted(refs, key=_sketch_sort_key)[0]
        rows.append(
            {
                "target_smiles": entry,
                "target_id": stable_id("cba_entry_substrate", entry),
                "benchmark_role": "cba_v0_entry_substrate",
                "source_target_smiles": best.get("target_smiles"),
                "source_transform_pair": best.get("transform_pair"),
                "source_guard_level": best.get("guard_level"),
                "source_pair_rank": best.get("pair_rank"),
                "source_pair_score": best.get("pair_score"),
                "source_prototype_upstream_similarity": best.get("prototype_upstream_similarity"),
                "source_prototype_target_similarity": best.get("prototype_target_similarity"),
                "source_sketch_ids": [row.get("sketch_id") for row in refs],
                "source_block_ids": [row.get("block_id") for row in refs],
                "source_dois": sorted({str(row.get("doi") or "") for row in refs if row.get("doi")}),
                "source_targets": sorted({str(row.get("target_smiles") or "") for row in refs if row.get("target_smiles")}),
                "source_guard_levels": dict(Counter(row.get("guard_level") for row in refs)),
                "routed_by_chem_enzy_source": any((row.get("chem_enzy_context") or {}).get("routed_by_chem_enzy") for row in refs),
            }
        )
    if max_targets is not None and max_targets > 0:
        rows = rows[: int(max_targets)]
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sketch_pack": str(sketch_pack),
            "max_targets": max_targets,
            "routed_only": routed_only,
        },
        "targets": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": payload["metadata"]["generated_at"],
        "metadata": {
            **payload["metadata"],
            "output": str(output),
            "report": str(report),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            "sketch_rows": len(sketches),
            "next_stage_sketch_rows": len([row for row in sketches if row.get("enter_next_stage")]),
            "candidate_sketch_rows": len(candidates),
            "unique_entry_substrates": len(by_entry),
            "targets_written": len(rows),
            "guard_levels_written": dict(Counter(row.get("source_guard_level") for row in rows)),
            "source_targets_written": len({target for row in rows for target in row.get("source_targets") or []}),
        },
        "outputs": {"benchmark": str(output), "report": str(report)},
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _sketch_sort_key(row: dict[str, Any]) -> tuple[int, int, float, float]:
    guard_order = {"strong_analog": 0, "weak_analog": 1, "type_only": 2}
    return (
        guard_order.get(str(row.get("guard_level")), 9),
        int(row.get("pair_rank") or 999),
        -float(row.get("prototype_upstream_similarity") or 0.0),
        -float(row.get("prototype_target_similarity") or 0.0),
    )


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
    return "\n".join(
        [
            "# CBA-v0 Entry Substrate Benchmark",
            "",
            "## Counts",
            "",
            "```json",
            json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Outputs",
            "",
            f"- benchmark: `{(result.get('outputs') or {}).get('benchmark')}`",
            f"- report: `{(result.get('outputs') or {}).get('report')}`",
            "",
            "## Interpretation",
            "",
            "This benchmark asks ChemEnzy to solve entry substrates from guarded CBA sketches. "
            "It is not a final route benchmark; it measures whether guarded cascade-block sketches can be connected to synthesizable/peripheral substrates.",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CBA-v0 entry-substrate benchmark")
    ap.add_argument("--sketch-pack", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-targets", type=int)
    ap.add_argument("--routed-only", action="store_true")
    args = ap.parse_args()
    result = build_cba_v0_entry_substrate_benchmark(
        sketch_pack=Path(args.sketch_pack),
        output=Path(args.output),
        report=Path(args.report),
        max_targets=args.max_targets,
        routed_only=args.routed_only,
    )
    print(json.dumps({"counts": result["counts"], "outputs": result["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
