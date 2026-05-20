"""Audit rank movements in a CCTS-v0 replay JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_rank_delta(
    *,
    replay_jsonl: Path,
    selected_score: str,
    output: Path,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in replay_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {
        "schema_version": "ccts_v0_rank_delta_audit.v1",
        "metadata": {
            "replay_jsonl": str(replay_jsonl),
            "selected_score": selected_score,
            "base_score": "chem_rank",
        },
        "labels": {
            "positive": _label_delta(rows, selected_score=selected_score, rank_key="best_positive_rank"),
            "exact": _label_delta(rows, selected_score=selected_score, rank_key="best_exact_rank"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _label_delta(rows: list[dict[str, Any]], *, selected_score: str, rank_key: str) -> dict[str, Any]:
    covered = []
    for row in rows:
        rank_map = row.get(rank_key) or {}
        chem = rank_map.get("chem_rank")
        selected = rank_map.get(selected_score)
        if chem is not None and selected is not None:
            covered.append((row, int(chem), int(selected)))
    improved = [(row, chem, selected) for row, chem, selected in covered if selected < chem]
    same = [(row, chem, selected) for row, chem, selected in covered if selected == chem]
    worsened = [(row, chem, selected) for row, chem, selected in covered if selected > chem]
    deltas = [selected - chem for _, chem, selected in covered]
    return {
        "covered": len(covered),
        "improved": len(improved),
        "same": len(same),
        "worsened": len(worsened),
        "lift_to_1": sum(1 for _, chem, selected in covered if chem > 1 and selected <= 1),
        "lift_to_3": sum(1 for _, chem, selected in covered if chem > 3 and selected <= 3),
        "lift_to_5": sum(1 for _, chem, selected in covered if chem > 5 and selected <= 5),
        "fall_from_5": sum(1 for _, chem, selected in covered if chem <= 5 and selected > 5),
        "mean_delta": round(sum(deltas) / max(len(deltas), 1), 6) if deltas else None,
        "examples_improved": [_example(row, chem, selected) for row, chem, selected in improved[:10]],
        "examples_worsened": [_example(row, chem, selected) for row, chem, selected in worsened[:10]],
    }


def _example(row: dict[str, Any], chem_rank: int, selected_rank: int) -> dict[str, Any]:
    return {
        "transition_id": row.get("transition_id"),
        "product_smiles": row.get("product_smiles"),
        "route_domain": row.get("route_domain"),
        "chem_rank": chem_rank,
        "selected_rank": selected_rank,
        "delta": selected_rank - chem_rank,
        "chem_top_main_reactant": ((row.get("chem_top_candidate") or {}).get("candidate_main_reactant")),
        "selected_top_main_reactant": ((row.get("selected_top_candidate") or {}).get("candidate_main_reactant")),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CCTS-v0 Rank Delta Audit",
        "",
        f"- Selected score: `{(result.get('metadata') or {}).get('selected_score')}`",
        "",
        "| Label | Covered | Improved | Same | Worsened | Lift >5 to <=5 | Fall <=5 to >5 | Mean delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (result.get("labels") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(metrics.get("covered")),
                    str(metrics.get("improved")),
                    str(metrics.get("same")),
                    str(metrics.get("worsened")),
                    str(metrics.get("lift_to_5")),
                    str(metrics.get("fall_from_5")),
                    str(metrics.get("mean_delta")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit CCTS-v0 replay rank deltas")
    ap.add_argument("--replay-jsonl", required=True)
    ap.add_argument("--selected-score", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = audit_rank_delta(
        replay_jsonl=Path(args.replay_jsonl),
        selected_score=args.selected_score,
        output=Path(args.output),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
