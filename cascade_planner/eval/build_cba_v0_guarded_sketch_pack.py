"""Build guarded cascade route-sketch candidates from CBA-v0 audit output."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cba_v0_guarded_sketch_pack.v1"


def build_cba_v0_guarded_sketch_pack(
    *,
    sketch_audit: Path,
    output_jsonl: Path,
    report: Path,
    max_sketches_per_target: int = 5,
    min_pair_rank: int = 10,
    weak_analog_similarity: float = 0.40,
    strong_analog_similarity: float = 0.55,
    min_target_similarity: float = 0.20,
) -> dict[str, Any]:
    started = time.monotonic()
    audit = json.loads(sketch_audit.read_text(encoding="utf-8"))
    block_rows = [row for row in audit.get("blocks") or [] if isinstance(row, dict)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in block_rows:
        grouped[str(row.get("target_smiles") or "")].append(row)
    sketches = []
    for target, rows in grouped.items():
        target_sketches = []
        for row in rows:
            sketch = _best_guarded_sketch(
                row,
                min_pair_rank=min_pair_rank,
                weak_analog_similarity=weak_analog_similarity,
                strong_analog_similarity=strong_analog_similarity,
                min_target_similarity=min_target_similarity,
            )
            if sketch:
                target_sketches.append(sketch)
        target_sketches.sort(key=lambda item: _sort_key(item))
        for idx, sketch in enumerate(target_sketches[: max(1, int(max_sketches_per_target))], start=1):
            sketch["target_sketch_rank"] = idx
            sketch["target_smiles"] = target
            sketches.append(sketch)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, sketches)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "sketch_audit": str(sketch_audit),
            "output_jsonl": str(output_jsonl),
            "report": str(report),
            "max_sketches_per_target": max_sketches_per_target,
            "min_pair_rank": min_pair_rank,
            "weak_analog_similarity": weak_analog_similarity,
            "strong_analog_similarity": strong_analog_similarity,
            "min_target_similarity": min_target_similarity,
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": "guarded sketches are not executable routes; they are block-type/prototype hypotheses for later ChemEnzy entry-substrate solving",
        },
        "counts": _counts(sketches, target_count=len(grouped), block_count=len(block_rows)),
        "top_transform_pairs": dict(Counter(row.get("transform_pair") for row in sketches).most_common(20)),
        "examples": sketches[:40],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown(result), encoding="utf-8")
    output_jsonl.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _best_guarded_sketch(
    row: dict[str, Any],
    *,
    min_pair_rank: int,
    weak_analog_similarity: float,
    strong_analog_similarity: float,
    min_target_similarity: float,
) -> dict[str, Any] | None:
    pair_rank = row.get("pair_rank")
    if pair_rank is None or int(pair_rank) > int(min_pair_rank):
        return None
    supports = [item for item in row.get("top_predicted_supports") or [] if item.get("transform_pair") == row.get("true_transform_pair")]
    if not supports:
        return None
    support = supports[0]
    sim = float(support.get("best_upstream_similarity") or 0.0)
    target_sim = float(support.get("best_target_similarity") or 0.0)
    if sim >= strong_analog_similarity:
        guard = "strong_analog"
    elif sim >= weak_analog_similarity:
        guard = "weak_analog"
    else:
        guard = "type_only"
    actionable = guard in {"strong_analog", "weak_analog"} and target_sim >= float(min_target_similarity)
    best_support = support.get("best_support") or {}
    return {
        "sketch_id": f"{row.get('block_id')}::{row.get('true_transform_pair')}::{guard}",
        "block_id": row.get("block_id"),
        "program_id": row.get("program_id"),
        "doi": row.get("doi"),
        "transform_pair": row.get("true_transform_pair"),
        "pair_rank": int(pair_rank),
        "pair_score": support.get("score"),
        "guard_level": guard,
        "enter_next_stage": actionable,
        "actionable_reason": "analog_and_target_supported" if actionable else _not_actionable_reason(guard, target_sim, min_target_similarity),
        "prototype_support_count": support.get("support_count"),
        "prototype_upstream_similarity": round(sim, 6),
        "prototype_target_similarity": round(target_sim, 6),
        "prototype": {
            "program_id": best_support.get("program_id"),
            "doi": best_support.get("doi"),
            "target_smiles": best_support.get("target_smiles"),
            "upstream_product": best_support.get("upstream_product"),
            "upstream_main_reactant": best_support.get("upstream_main_reactant"),
            "upstream_similarity": best_support.get("upstream_similarity"),
            "target_similarity": best_support.get("target_similarity"),
        },
        "chem_enzy_context": {
            "routed_by_chem_enzy": row.get("routed_by_chem_enzy"),
            "chem_enzy_route_count": row.get("chem_enzy_route_count"),
            "chem_enzy_transform_consistent_rank": row.get("chem_enzy_transform_consistent_rank"),
        },
    }


def _not_actionable_reason(guard: str, target_sim: float, min_target_similarity: float) -> str:
    if guard == "type_only":
        return "type_only_without_structural_analog"
    if target_sim < float(min_target_similarity):
        return "target_similarity_below_threshold"
    return "not_actionable"


def _sort_key(item: dict[str, Any]) -> tuple[int, int, float, float]:
    guard_order = {"strong_analog": 0, "weak_analog": 1, "type_only": 2}
    return (
        guard_order.get(str(item.get("guard_level")), 9),
        int(item.get("pair_rank") or 999),
        -float(item.get("prototype_upstream_similarity") or 0.0),
        -float(item.get("pair_score") or 0.0),
    )


def _counts(sketches: list[dict[str, Any]], *, target_count: int, block_count: int) -> dict[str, Any]:
    targets_with = {row.get("target_smiles") for row in sketches}
    next_stage = [row for row in sketches if row.get("enter_next_stage")]
    routed = [row for row in sketches if (row.get("chem_enzy_context") or {}).get("routed_by_chem_enzy")]
    return {
        "audit_targets": target_count,
        "audit_blocks": block_count,
        "sketches": len(sketches),
        "targets_with_sketch": len(targets_with),
        "targets_with_sketch_rate": round(len(targets_with) / max(target_count, 1), 6),
        "next_stage_sketches": len(next_stage),
        "targets_with_next_stage_sketch": len({row.get("target_smiles") for row in next_stage}),
        "targets_with_next_stage_sketch_rate": round(len({row.get("target_smiles") for row in next_stage}) / max(target_count, 1), 6),
        "guard_level_counts": dict(Counter(row.get("guard_level") for row in sketches)),
        "enter_next_stage_counts": dict(Counter(bool(row.get("enter_next_stage")) for row in sketches)),
        "routed_chem_enzy_sketches": len(routed),
        "routed_chem_enzy_next_stage_sketches": sum(1 for row in routed if row.get("enter_next_stage")),
        "routed_chem_enzy_targets_with_next_stage_sketch": len({row.get("target_smiles") for row in routed if row.get("enter_next_stage")}),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CBA-v0 Guarded Sketch Pack",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top Transform Pairs",
        "",
        "```json",
        json.dumps(result.get("top_transform_pairs") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "The pack contains guarded cascade-block sketches. "
        "`strong_analog` and `weak_analog` sketches may enter ChemEnzy entry-substrate/peripheral solving; "
        "`type_only` sketches are retained as type priors but should not be injected as reaction prototypes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build guarded CBA-v0 sketch pack")
    ap.add_argument("--sketch-audit", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-sketches-per-target", type=int, default=5)
    ap.add_argument("--min-pair-rank", type=int, default=10)
    ap.add_argument("--weak-analog-similarity", type=float, default=0.40)
    ap.add_argument("--strong-analog-similarity", type=float, default=0.55)
    ap.add_argument("--min-target-similarity", type=float, default=0.20)
    args = ap.parse_args()
    result = build_cba_v0_guarded_sketch_pack(
        sketch_audit=Path(args.sketch_audit),
        output_jsonl=Path(args.output_jsonl),
        report=Path(args.report),
        max_sketches_per_target=args.max_sketches_per_target,
        min_pair_rank=args.min_pair_rank,
        weak_analog_similarity=args.weak_analog_similarity,
        strong_analog_similarity=args.strong_analog_similarity,
        min_target_similarity=args.min_target_similarity,
    )
    print(json.dumps({"counts": result["counts"], "top_transform_pairs": result["top_transform_pairs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
