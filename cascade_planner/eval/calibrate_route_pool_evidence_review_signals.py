"""Calibrate route-pool evidence diagnostics against review labels.

This is a pre-training gate.  It checks whether diagnostic signals such as
analog support scores and transform-label sanity warnings actually correlate
with human/LLM review labels.  It does not train a scorer.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "route_pool_evidence_signal_calibration.v1"

NUMERIC_SIGNALS = (
    "native_rank",
    "best_any_block_min_sim",
    "best_same_pair_block_min_sim",
    "known_transform_step_fraction",
)

CATEGORICAL_SIGNALS = (
    "evidence_class",
    "source_pool",
    "stock_closed",
    "has_observed_pair_block",
    "has_any_analog_block",
    "has_same_pair_analog_block",
    "transform_label_warning",
)


def calibrate_route_pool_evidence_review_signals(
    *,
    labels_jsonl: Path,
    output_json: Path,
    output_md: Path | None = None,
    min_rows: int = 30,
    min_positive: int = 5,
    min_negative: int = 5,
    min_auc: float = 0.65,
) -> dict[str, Any]:
    rows = _read_jsonl(labels_jsonl)
    labeled = [_labeled_row(row) for row in rows]
    determinate = [row for row in labeled if row["label"] in {"positive", "negative"}]
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "labels_jsonl": str(labels_jsonl),
            "output_json": str(output_json),
            "output_md": str(output_md or output_json.with_suffix(".md")),
            "min_rows": min_rows,
            "min_positive": min_positive,
            "min_negative": min_negative,
            "min_auc": min_auc,
        },
        "summary": _summary(labeled, determinate),
        "numeric_signals": _numeric_signal_calibration(determinate),
        "categorical_signals": _categorical_signal_calibration(determinate),
        "decision": _decision(determinate, min_rows=min_rows, min_positive=min_positive, min_negative=min_negative, min_auc=min_auc),
        "contract": {
            "pre_training_gate_only": True,
            "does_not_train_model": True,
            "requires_real_review_labels": True,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _summary(rows: list[dict[str, Any]], determinate: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "determinate_rows": len(determinate),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "determinate_classes": dict(Counter(str(row.get("evidence_class") or "") for row in determinate)),
        "determinate_pools": dict(Counter(str(row.get("source_pool") or "") for row in determinate)),
    }


def _labeled_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if _usable_positive(row):
        label = "positive"
    elif _usable_negative(row):
        label = "negative"
    elif _has_unclear(row):
        label = "unclear"
    else:
        label = "unusable"
    out["label"] = label
    out["signals"] = _signals(row)
    return out


def _signals(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("diagnostic_scores") or {}
    labels = row.get("diagnostic_labels") or {}
    sanity = row.get("transform_sanity") or {}
    return {
        "native_rank": _to_float(row.get("native_rank")),
        "best_any_block_min_sim": _to_float(scores.get("best_any_block_min_sim")),
        "best_same_pair_block_min_sim": _to_float(scores.get("best_same_pair_block_min_sim")),
        "known_transform_step_fraction": _to_float(scores.get("known_transform_step_fraction")),
        "evidence_class": row.get("evidence_class"),
        "source_pool": row.get("source_pool"),
        "stock_closed": _boolish(row.get("stock_closed")),
        "has_observed_pair_block": _boolish(labels.get("has_observed_pair_block")),
        "has_any_analog_block": _boolish(labels.get("has_any_analog_block")),
        "has_same_pair_analog_block": _boolish(labels.get("has_same_pair_analog_block")),
        "transform_label_warning": _boolish(sanity.get("block_has_label_mismatch")),
    }


def _numeric_signal_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for signal in NUMERIC_SIGNALS:
        pos = [_signal(row, signal) for row in rows if row["label"] == "positive"]
        neg = [_signal(row, signal) for row in rows if row["label"] == "negative"]
        pos_vals = _finite(pos)
        neg_vals = _finite(neg)
        auc = _auc(pos_vals, neg_vals)
        out[signal] = {
            "positive": _numeric_summary(pos_vals),
            "negative": _numeric_summary(neg_vals),
            "mean_delta_positive_minus_negative": _mean_delta(pos_vals, neg_vals),
            "auc_higher_is_positive": auc,
            "abs_auc_delta_from_random": round(abs(float(auc) - 0.5), 6) if auc is not None else None,
        }
    return out


def _categorical_signal_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for signal in CATEGORICAL_SIGNALS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(_signal(row, signal))].append(row)
        out[signal] = {
            value: {
                "rows": len(group),
                "positive_rows": sum(1 for row in group if row["label"] == "positive"),
                "negative_rows": sum(1 for row in group if row["label"] == "negative"),
                "positive_rate": _rate(sum(1 for row in group if row["label"] == "positive"), len(group)),
                "negative_rate": _rate(sum(1 for row in group if row["label"] == "negative"), len(group)),
            }
            for value, group in sorted(grouped.items())
        }
    return out


def _decision(
    rows: list[dict[str, Any]],
    *,
    min_rows: int,
    min_positive: int,
    min_negative: int,
    min_auc: float,
) -> dict[str, Any]:
    positive = sum(1 for row in rows if row["label"] == "positive")
    negative = sum(1 for row in rows if row["label"] == "negative")
    numeric = _numeric_signal_calibration(rows)
    best_numeric = _best_numeric_signal(numeric)
    checks = [
        {"name": "minimum_determinate_rows", "ok": len(rows) >= int(min_rows), "value": len(rows), "threshold": int(min_rows)},
        {"name": "minimum_positive_rows", "ok": positive >= int(min_positive), "value": positive, "threshold": int(min_positive)},
        {"name": "minimum_negative_rows", "ok": negative >= int(min_negative), "value": negative, "threshold": int(min_negative)},
        {
            "name": "minimum_numeric_auc_signal",
            "ok": bool(best_numeric and float(best_numeric.get("best_oriented_auc") or 0.0) >= float(min_auc)),
            "value": best_numeric,
            "threshold": float(min_auc),
        },
    ]
    ready = all(row["ok"] for row in checks)
    return {
        "ready_for_proxy_training": ready,
        "recommendation": (
            "diagnostic signals pass the preliminary calibration gate; scorer training can be considered"
            if ready
            else "do not train a scorer from these review labels yet; collect enough real labels and confirm signal calibration"
        ),
        "determinate_rows": len(rows),
        "positive_rows": positive,
        "negative_rows": negative,
        "best_numeric_signal": best_numeric,
        "checks": checks,
    }


def _best_numeric_signal(numeric: dict[str, Any]) -> dict[str, Any] | None:
    best = None
    for name, row in numeric.items():
        auc = row.get("auc_higher_is_positive")
        if auc is None:
            continue
        oriented = max(float(auc), 1.0 - float(auc))
        candidate = {
            "signal": name,
            "auc_higher_is_positive": round(float(auc), 6),
            "best_oriented_auc": round(oriented, 6),
            "orientation": "higher_is_positive" if float(auc) >= 0.5 else "lower_is_positive",
        }
        if best is None or candidate["best_oriented_auc"] > best["best_oriented_auc"]:
            best = candidate
    return best


def _review(row: dict[str, Any]) -> dict[str, Any]:
    review = row.get("expert_review") or {}
    return review if isinstance(review, dict) else {}


def _usable_positive(row: dict[str, Any]) -> bool:
    review = _review(row)
    return (
        review.get("route_plausible") == "yes"
        and review.get("block_transform_correct") == "yes"
        and review.get("support_precedent_relevant") == "yes"
        and review.get("cascade_coherent") == "yes"
        and review.get("priority") in {"high", "medium"}
    )


def _usable_negative(row: dict[str, Any]) -> bool:
    review = _review(row)
    return (
        review.get("priority") == "reject"
        or review.get("route_plausible") == "no"
        or review.get("block_transform_correct") == "no"
        or review.get("support_precedent_relevant") == "no"
        or review.get("cascade_coherent") == "no"
    )


def _has_unclear(row: dict[str, Any]) -> bool:
    review = _review(row)
    return any(review.get(field) == "unclear" for field in ("route_plausible", "block_transform_correct", "support_precedent_relevant", "cascade_coherent"))


def _signal(row: dict[str, Any], signal: str) -> Any:
    return (row.get("signals") or {}).get(signal)


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            out.append(fval)
    return out


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _mean_delta(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    return round(sum(pos) / len(pos) - sum(neg) / len(neg), 6)


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for pval in pos:
        for nval in neg:
            total += 1
            if pval > nval:
                wins += 1.0
            elif pval == nval:
                wins += 0.5
    return round(wins / total, 6) if total else None


def _to_float(value: Any) -> float | None:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    return fval if math.isfinite(fval) else None


def _boolish(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _rate(num: int | float, denom: int | float) -> float:
    return round(float(num) / float(denom), 6) if denom else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Route Pool Evidence Signal Calibration",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(result.get("decision") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (result.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Contract", "", "This is a pre-training calibration gate. It does not train a scorer.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate route-pool evidence diagnostics against review labels")
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--min-negative", type=int, default=5)
    parser.add_argument("--min-auc", type=float, default=0.65)
    args = parser.parse_args()
    result = calibrate_route_pool_evidence_review_signals(
        labels_jsonl=Path(args.labels_jsonl),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        min_rows=int(args.min_rows),
        min_positive=int(args.min_positive),
        min_negative=int(args.min_negative),
        min_auc=float(args.min_auc),
    )
    print(json.dumps({"summary": result["summary"], "decision": result["decision"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
