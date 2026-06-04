#!/usr/bin/env python
"""Post-filter existing ONMT exact-recall JSONs with conservative proposal validity rules."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.cascadeboard.route_recovery import canonical_side  # noqa: E402
from cascade_planner.cascade_search.proposal_validity import (  # noqa: E402
    ProposalValidityConfig,
    filter_reactant_predictions,
)


SCHEMA_VERSION = "onmt_exact_recall_validity_filter.v1"


def filter_exact_recall_predictions(
    *,
    exact_recall_json: Path,
    output_json: Path,
    output_md: Path | None = None,
    topk: int | None = None,
    max_reactant_to_product_heavy_ratio: float | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(exact_recall_json).read_text(encoding="utf-8"))
    results = [
        _filter_result(
            result,
            topk=topk or int(result.get("topk") or 5),
            max_reactant_to_product_heavy_ratio=max_reactant_to_product_heavy_ratio,
        )
        for result in payload.get("results") or []
    ]
    out = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "exact_recall_json": str(exact_recall_json),
        "topk": topk,
        "max_reactant_to_product_heavy_ratio": max_reactant_to_product_heavy_ratio,
        "results": results,
        "summary": _summary(results),
        "decision": _decision(results),
        "contract": (
            "Post-hoc validity filtering only. It does not rerun ONMT and does not prove generator improvement."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(out), encoding="utf-8")
    return out


def _filter_result(
    result: dict[str, Any],
    *,
    topk: int,
    max_reactant_to_product_heavy_ratio: float | None,
) -> dict[str, Any]:
    rows = []
    rejected_counts: Counter[str] = Counter()
    for row in result.get("rows") or []:
        product_smiles = _product_from_row(row)
        target_key = canonical_side(row.get("target_reactants") or "")
        predictions = [str(item or "").replace(" ", "") for item in (row.get("predictions") or [])]
        scores = list(row.get("scores") or [])
        report = filter_reactant_predictions(
            predictions,
            product_smiles=product_smiles,
            config=ProposalValidityConfig(max_reactant_to_product_heavy_ratio=max_reactant_to_product_heavy_ratio),
        )
        for rejected in report.rejected:
            rejected_counts[str(rejected.get("reason") or "unknown")] += 1
        filtered_scores = _align_scores(predictions, scores, report.kept)
        filtered_keys = [canonical_side(prediction) for prediction in report.kept]
        filtered_top1 = bool(filtered_keys and filtered_keys[0] == target_key)
        filtered_topk = target_key in filtered_keys[:topk]
        rows.append({
            "idx": row.get("idx"),
            "product": row.get("product"),
            "product_smiles": product_smiles,
            "target_reactants": row.get("target_reactants"),
            "raw_predictions": predictions[:topk],
            "raw_scores": scores[:topk],
            "filtered_predictions": report.kept[:topk],
            "filtered_scores": filtered_scores[:topk],
            "filtered_rejected": report.rejected,
            "raw_top1_exact": bool(row.get("top1_exact")),
            "raw_topk_exact": bool(row.get(f"top{topk}_exact") or row.get("top5_exact")),
            "filtered_top1_exact": filtered_top1,
            f"filtered_top{topk}_exact": filtered_topk,
        })
    n = len(rows)
    filtered_top1 = sum(1 for row in rows if row["filtered_top1_exact"])
    filtered_topk = sum(1 for row in rows if row[f"filtered_top{topk}_exact"])
    raw_top1 = sum(1 for row in rows if row["raw_top1_exact"])
    raw_topk = sum(1 for row in rows if row["raw_topk_exact"])
    filtered_nonempty = sum(1 for row in rows if row["filtered_predictions"])
    return {
        "model_path": result.get("model_path"),
        "src_path": result.get("src_path"),
        "tgt_path": result.get("tgt_path"),
        "tokenizer": result.get("tokenizer"),
        "topk": topk,
        "n_examples": n,
        "raw_top1_exact": raw_top1,
        "raw_topk_exact": raw_topk,
        "filtered_top1_exact": filtered_top1,
        "filtered_topk_exact": filtered_topk,
        "raw_top1_rate": round(raw_top1 / max(n, 1), 6),
        "raw_topk_rate": round(raw_topk / max(n, 1), 6),
        "filtered_top1_rate": round(filtered_top1 / max(n, 1), 6),
        "filtered_topk_rate": round(filtered_topk / max(n, 1), 6),
        "filtered_nonempty": filtered_nonempty,
        "filtered_rejected_total": sum(rejected_counts.values()),
        "filtered_rejected_reason_counts": dict(rejected_counts),
        "rows": rows,
    }


def _summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "model_path",
        "n_examples",
        "raw_top1_exact",
        "raw_topk_exact",
        "filtered_top1_exact",
        "filtered_topk_exact",
        "filtered_nonempty",
        "filtered_rejected_total",
        "filtered_rejected_reason_counts",
    ]
    return [{key: result.get(key) for key in keys} for result in results]


def _decision(results: list[dict[str, Any]]) -> dict[str, str]:
    if len(results) >= 2:
        native, adapter = results[0], results[-1]
        delta = int(adapter.get("filtered_topk_exact") or 0) - int(native.get("filtered_topk_exact") or 0)
        if delta > 0:
            return {"status": "filtered_exact_lift_present", "reason": f"filtered top-k delta is {delta}."}
        return {"status": "filtered_no_exact_lift", "reason": f"filtered top-k delta is {delta}."}
    if any(int(result.get("filtered_rejected_total") or 0) for result in results):
        return {"status": "filtered_predictions_available", "reason": "Validity filter removed at least one proposal."}
    return {"status": "no_filter_effect", "reason": "No proposals were removed."}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ONMT Exact Recall Validity Filter",
        "",
        f"created_at: `{payload['created_at']}`",
        "",
        "## Decision",
        "",
        f"- status: `{payload['decision']['status']}`",
        f"- reason: {payload['decision']['reason']}",
        "",
        "## Summary",
        "",
        "| model | n | raw topk | filtered topk | filtered nonempty | rejected | reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in payload.get("results") or []:
        reasons = ", ".join(f"{key}:{value}" for key, value in sorted((result.get("filtered_rejected_reason_counts") or {}).items()))
        lines.append(
            "| `{model}` | {n} | {raw} | {filtered} | {nonempty} | {rejected} | {reasons} |".format(
                model=_basename(result.get("model_path")),
                n=result.get("n_examples"),
                raw=result.get("raw_topk_exact"),
                filtered=result.get("filtered_topk_exact"),
                nonempty=result.get("filtered_nonempty"),
                rejected=result.get("filtered_rejected_total"),
                reasons=reasons or "-",
            )
        )
    lines.extend(["", "## Contract", "", payload["contract"], ""])
    return "\n".join(lines)


def _product_from_row(row: dict[str, Any]) -> str:
    text = str(row.get("product") or "").strip()
    if "<product>" not in text:
        return text.replace(" ", "")
    after = text.split("<product>", 1)[1].strip()
    if "<candidate>" in after:
        after = after.split("<candidate>", 1)[0].strip()
    return after.replace(" ", "")


def _align_scores(predictions: list[str], scores: list[Any], kept: list[str]) -> list[float]:
    out: list[float] = []
    used: set[int] = set()
    for item in kept:
        for idx, prediction in enumerate(predictions):
            if idx in used:
                continue
            if prediction == item:
                used.add(idx)
                if idx < len(scores):
                    try:
                        out.append(float(scores[idx]))
                    except (TypeError, ValueError):
                        pass
                break
    return out


def _basename(path: Any) -> str:
    return Path(str(path or "")).name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exact-recall", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path)
    ap.add_argument("--topk", type=int)
    ap.add_argument("--max-reactant-to-product-heavy-ratio", type=float)
    args = ap.parse_args()
    payload = filter_exact_recall_predictions(
        exact_recall_json=args.exact_recall,
        output_json=args.output,
        output_md=args.markdown_output,
        topk=args.topk,
        max_reactant_to_product_heavy_ratio=args.max_reactant_to_product_heavy_ratio,
    )
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
