#!/usr/bin/env python
"""Diagnose context-ONMT reactant-completion predictions by corruption type."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side  # noqa: E402


RDLogger.DisableLog("rdApp.*")
SCHEMA_VERSION = "reactant_completion_prediction_diagnostic.v1"


def diagnose_completion_predictions(
    *,
    exact_recall_json: Path,
    metadata_jsonl: Path,
    output_json: Path,
    output_md: Path | None = None,
    model_selector: str = "last",
    topk: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(exact_recall_json).read_text(encoding="utf-8"))
    result = _select_result(payload.get("results") or [], model_selector=model_selector)
    meta_rows = _read_jsonl(metadata_jsonl, limit=limit or result.get("limit"))
    result_rows = list(result.get("rows") or [])
    if limit is not None:
        result_rows = result_rows[: int(limit)]
    if len(result_rows) != len(meta_rows):
        raise ValueError(f"row/meta length mismatch: {len(result_rows)} != {len(meta_rows)}")
    rows = [_diagnose_row(row, meta, topk=topk) for row, meta in zip(result_rows, meta_rows)]
    out = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "exact_recall_json": str(exact_recall_json),
        "metadata_jsonl": str(metadata_jsonl),
        "model_selector": model_selector,
        "model_path": result.get("model_path"),
        "topk": int(topk),
        "summary": _summary(rows),
        "by_corruption_type": _by_corruption_type(rows),
        "decision": _decision(rows),
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(out), encoding="utf-8")
    return out


def _diagnose_row(row: dict[str, Any], meta: dict[str, Any], *, topk: int) -> dict[str, Any]:
    target_side = canonical_side(row.get("target_reactants") or meta.get("chosen_reactants") or "")
    given_side = canonical_side(meta.get("given_reactants") or "")
    target_fp = _fp_side(target_side)
    predictions = []
    for rank, pred in enumerate((row.get("predictions") or [])[:topk], 1):
        pred_side = canonical_side(str(pred or ""))
        pred_fp = _fp_side(pred_side)
        overlap = sorted(set(pred_side) & set(target_side))
        missing = sorted(set(target_side) - set(pred_side))
        copied_given = pred_side == given_side
        predictions.append({
            "rank": rank,
            "prediction": pred,
            "canonical_side": list(pred_side),
            "exact_target": pred_side == target_side,
            "copies_given_side": copied_given,
            "all_molecules_valid": bool(pred_side) and all(_is_valid_mol(smi) for smi in pred_side),
            "target_molecule_overlap": overlap,
            "target_molecule_overlap_count": len(overlap),
            "missing_target_molecules": missing,
            "side_similarity": round(_similarity(target_fp, pred_fp), 6),
        })
    return {
        "idx": row.get("idx"),
        "product": meta.get("product") or row.get("product"),
        "corruption_type": meta.get("corruption_type"),
        "given_reactants": meta.get("given_reactants"),
        "target_reactants": ".".join(target_side),
        "given_side": list(given_side),
        "target_side": list(target_side),
        "top1_exact": bool(predictions and predictions[0]["exact_target"]),
        "topk_exact": any(pred["exact_target"] for pred in predictions),
        "topk_copies_given": any(pred["copies_given_side"] for pred in predictions),
        "best_overlap": max((pred["target_molecule_overlap_count"] for pred in predictions), default=0),
        "best_similarity": max((pred["side_similarity"] for pred in predictions), default=0.0),
        "any_invalid": any(not pred["all_molecules_valid"] for pred in predictions),
        "predictions": predictions,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate(rows)


def _by_corruption_type(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("corruption_type") or "unknown")].append(row)
    return {key: _aggregate(group_rows) for key, group_rows in sorted(grouped.items())}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    labels = Counter()
    for row in rows:
        if row.get("topk_exact"):
            labels["topk_exact"] += 1
        if row.get("topk_copies_given"):
            labels["topk_copies_given"] += 1
        if row.get("any_invalid"):
            labels["invalid_in_topk"] += 1
        if int(row.get("best_overlap") or 0) == 0:
            labels["no_target_overlap"] += 1
        elif not row.get("topk_exact"):
            labels["partial_target_overlap"] += 1
    return {
        "n_rows": n,
        "top1_exact": sum(1 for row in rows if row.get("top1_exact")),
        "topk_exact": sum(1 for row in rows if row.get("topk_exact")),
        "rows_with_any_target_overlap": sum(1 for row in rows if int(row.get("best_overlap") or 0) > 0),
        "rows_copying_given_side": sum(1 for row in rows if row.get("topk_copies_given")),
        "rows_with_invalid_topk": sum(1 for row in rows if row.get("any_invalid")),
        "avg_best_similarity": _avg([float(row.get("best_similarity") or 0.0) for row in rows]),
        "label_counts": dict(labels),
    }


def _decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    summary = _summary(rows)
    if summary["topk_exact"] > 0:
        return {"status": "completion_exact_signal_present", "reason": "At least one exact completion appears in top-k."}
    if summary["rows_copying_given_side"] >= max(1, summary["n_rows"] // 2):
        return {"status": "candidate_copying_without_completion", "reason": "The model often repeats the corrupted candidate side."}
    if summary["rows_with_invalid_topk"] >= max(1, summary["n_rows"] // 2):
        return {"status": "invalid_completion_generation", "reason": "Most rows contain invalid molecules in top-k completions."}
    return {"status": "no_exact_completion_signal", "reason": "No exact top-k completion observed in the evaluated rows."}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reactant Completion Prediction Diagnostic",
        "",
        f"created_at: `{payload['created_at']}`",
        f"model_path: `{payload['model_path']}`",
        "",
        "## Decision",
        "",
        f"- status: `{payload['decision']['status']}`",
        f"- reason: {payload['decision']['reason']}",
        "",
        "## Summary",
        "",
    ]
    summary = payload["summary"]
    for key in ("n_rows", "top1_exact", "topk_exact", "rows_with_any_target_overlap", "rows_copying_given_side", "rows_with_invalid_topk", "avg_best_similarity"):
        lines.append(f"- {key}: {summary.get(key)}")
    lines.extend(["", "## By Corruption Type", "", "| type | n | topk exact | overlap | copied given | invalid topk | avg sim |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for key, row in payload["by_corruption_type"].items():
        lines.append(
            f"| `{key}` | {row['n_rows']} | {row['topk_exact']} | {row['rows_with_any_target_overlap']} | "
            f"{row['rows_copying_given_side']} | {row['rows_with_invalid_topk']} | {row['avg_best_similarity']} |"
        )
    lines.extend(["", "## Worst Rows", "", "| idx | type | given | target | best sim | top1 |", "| ---: | --- | --- | --- | ---: | --- |"])
    ranked = sorted(payload["rows"], key=lambda row: (float(row.get("best_similarity") or 0.0), int(row.get("best_overlap") or 0)))
    for row in ranked[:20]:
        top1 = (row.get("predictions") or [{}])[0]
        lines.append(
            "| {idx} | `{typ}` | `{given}` | `{target}` | {sim:.3f} | `{top1}` |".format(
                idx=row.get("idx"),
                typ=row.get("corruption_type"),
                given=_truncate(row.get("given_reactants"), 60),
                target=_truncate(row.get("target_reactants"), 80),
                sim=float(row.get("best_similarity") or 0.0),
                top1=_truncate(top1.get("prediction"), 100),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _select_result(results: list[dict[str, Any]], *, model_selector: str) -> dict[str, Any]:
    if not results:
        raise ValueError("exact recall JSON has no results")
    if model_selector == "first":
        return results[0]
    if model_selector == "last":
        return results[-1]
    try:
        return results[int(model_selector)]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unsupported model selector: {model_selector}") from exc


def _read_jsonl(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _fp_side(side: tuple[str, ...]) -> Any:
    fps = []
    for smi in side:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if not fps:
        return None
    out = fps[0]
    for fp in fps[1:]:
        out |= fp
    return out


def _is_valid_mol(smiles: str) -> bool:
    return bool(smiles and Chem.MolFromSmiles(smiles) is not None)


def _similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _avg(values: list[float]) -> float:
    return round(sum(values) / max(len(values), 1), 6)


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exact-recall", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--model-selector", default="last")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    payload = diagnose_completion_predictions(
        exact_recall_json=args.exact_recall,
        metadata_jsonl=args.metadata,
        output_json=args.output,
        output_md=args.markdown_output,
        model_selector=args.model_selector,
        topk=args.topk,
        limit=args.limit,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
