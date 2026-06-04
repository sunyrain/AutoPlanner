#!/usr/bin/env python
"""Diagnose ChemEnzy ONMT exact-recall prediction error modes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side


RDLogger.DisableLog("rdApp.*")
SCHEMA_VERSION = "onmt_prediction_error_diagnostic.v1"


def diagnose_prediction_errors(
    *,
    exact_recall_json: Path,
    output_json: Path,
    output_md: Path | None = None,
    model_selector: str = "last",
    topk: int = 5,
) -> dict[str, Any]:
    payload = json.loads(Path(exact_recall_json).read_text(encoding="utf-8"))
    result = _select_result(payload.get("results") or [], model_selector=model_selector)
    rows = [_diagnose_row(row, topk=topk) for row in result.get("rows") or []]
    out = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "exact_recall_json": str(exact_recall_json),
        "model_selector": model_selector,
        "model_path": result.get("model_path"),
        "topk": topk,
        "summary": _summary(rows),
        "rows": rows,
        "decision": _decision(rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(out), encoding="utf-8")
    return out


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


def _diagnose_row(row: dict[str, Any], *, topk: int) -> dict[str, Any]:
    target = str(row.get("target_reactants") or "")
    target_side = canonical_side(target)
    target_fp = _fp_side(target_side)
    preds = []
    for rank, pred in enumerate((row.get("predictions") or [])[:topk], 1):
        pred_side = canonical_side(str(pred or ""))
        pred_fp = _fp_side(pred_side)
        valid_mols = [_is_valid_mol(smi) for smi in pred_side]
        reactant_overlap = sorted(set(pred_side) & set(target_side))
        preds.append(
            {
                "rank": rank,
                "prediction": pred,
                "canonical_side": list(pred_side),
                "n_molecules": len(pred_side),
                "n_target_molecules": len(target_side),
                "all_molecules_valid": bool(pred_side) and all(valid_mols),
                "any_invalid_molecule": bool(pred_side) and not all(valid_mols),
                "target_molecule_overlap": reactant_overlap,
                "target_molecule_overlap_count": len(reactant_overlap),
                "side_similarity": round(_similarity(target_fp, pred_fp), 6),
                "atom_count": _side_atom_count(pred_side),
                "target_atom_count": _side_atom_count(target_side),
                "too_many_atoms_ratio": _too_many_atoms_ratio(pred_side, target_side),
            }
        )
    best = max((float(pred.get("side_similarity") or 0.0) for pred in preds), default=0.0)
    overlap_best = max((int(pred.get("target_molecule_overlap_count") or 0) for pred in preds), default=0)
    labels = _row_labels(preds, target_side=target_side)
    return {
        "idx": row.get("idx"),
        "product": row.get("product"),
        "target_reactants": target,
        "target_side": list(target_side),
        "target_n_molecules": len(target_side),
        "target_atom_count": _side_atom_count(target_side),
        "top1_exact": bool(row.get("top1_exact")),
        "topk_exact": bool(row.get("top5_exact") or row.get(f"top{topk}_exact")),
        "best_side_similarity": round(best, 6),
        "best_target_molecule_overlap_count": overlap_best,
        "labels": labels,
        "predictions": preds,
    }


def _row_labels(preds: list[dict[str, Any]], *, target_side: tuple[str, ...]) -> list[str]:
    if not preds:
        return ["no_predictions"]
    labels: list[str] = []
    top1 = preds[0]
    if any(pred.get("any_invalid_molecule") for pred in preds):
        labels.append("invalid_molecule_in_topk")
    if not any(pred.get("target_molecule_overlap_count") for pred in preds):
        labels.append("no_gt_reactant_overlap")
    elif not any(set(pred.get("canonical_side") or []) == set(target_side) for pred in preds):
        labels.append("partial_gt_reactant_overlap")
    if len(top1.get("canonical_side") or []) != len(target_side):
        labels.append("top1_molecule_count_mismatch")
    if float(top1.get("too_many_atoms_ratio") or 0.0) >= 2.0:
        labels.append("top1_atom_count_explosion")
    if float(top1.get("side_similarity") or 0.0) < 0.30:
        labels.append("top1_low_similarity")
    return labels or ["near_miss_unclassified"]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(label for row in rows for label in row.get("labels") or [])
    return {
        "n_rows": len(rows),
        "top1_exact": sum(1 for row in rows if row.get("top1_exact")),
        "topk_exact": sum(1 for row in rows if row.get("topk_exact")),
        "rows_with_any_gt_reactant_overlap": sum(1 for row in rows if int(row.get("best_target_molecule_overlap_count") or 0) > 0),
        "avg_best_side_similarity": _avg([float(row.get("best_side_similarity") or 0.0) for row in rows]),
        "label_counts": dict(label_counts),
    }


def _decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    summary = _summary(rows)
    labels = Counter(summary.get("label_counts") or {})
    if summary["topk_exact"]:
        return {"status": "exact_recall_present", "reason": "At least one exact top-k reactant-set hit exists."}
    if labels.get("no_gt_reactant_overlap", 0) >= max(1, len(rows) // 2):
        return {
            "status": "reactant_set_generation_failure",
            "reason": "Most examples have no GT reactant molecule overlap in top-k predictions.",
        }
    if labels.get("partial_gt_reactant_overlap", 0):
        return {
            "status": "reaction_completion_failure",
            "reason": "The model often recovers part of the GT side but fails to complete the reactant set.",
        }
    return {"status": "mixed_prediction_errors", "reason": "No single dominant exact-recall failure mode."}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# ONMT Prediction Error Diagnostic",
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
        f"- n_rows: {summary['n_rows']}",
        f"- top1_exact: {summary['top1_exact']}",
        f"- topk_exact: {summary['topk_exact']}",
        f"- rows_with_any_gt_reactant_overlap: {summary['rows_with_any_gt_reactant_overlap']}",
        f"- avg_best_side_similarity: {summary['avg_best_side_similarity']}",
        "",
        "## Label Counts",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted((summary.get("label_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Worst Rows",
        "",
        "| idx | labels | best sim | overlap | target | top1 |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ])
    ranked = sorted(payload["rows"], key=lambda row: (float(row.get("best_side_similarity") or 0.0), int(row.get("best_target_molecule_overlap_count") or 0)))
    for row in ranked[:20]:
        top1 = (row.get("predictions") or [{}])[0]
        lines.append(
            "| {idx} | `{labels}` | {sim:.3f} | {overlap} | `{target}` | `{top1}` |".format(
                idx=row.get("idx"),
                labels=",".join(row.get("labels") or []),
                sim=float(row.get("best_side_similarity") or 0.0),
                overlap=row.get("best_target_molecule_overlap_count"),
                target=_truncate(row.get("target_reactants"), 80),
                top1=_truncate(top1.get("prediction"), 120),
            )
        )
    lines.append("")
    return "\n".join(lines)


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


def _side_atom_count(side: tuple[str, ...]) -> int:
    total = 0
    for smi in side:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            total += int(mol.GetNumAtoms())
    return total


def _too_many_atoms_ratio(pred_side: tuple[str, ...], target_side: tuple[str, ...]) -> float:
    target_atoms = _side_atom_count(target_side)
    if target_atoms <= 0:
        return 0.0
    return round(_side_atom_count(pred_side) / target_atoms, 6)


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
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--model-selector", default="last")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    payload = diagnose_prediction_errors(
        exact_recall_json=args.exact_recall,
        output_json=args.output,
        output_md=args.markdown_output,
        model_selector=args.model_selector,
        topk=args.topk,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
