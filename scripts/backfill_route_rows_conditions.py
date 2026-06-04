"""Backfill missing per-step condition predictions in benchmark route rows."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.proposals import (  # noqa: E402
    _cached_condition_predictor,
    _normalize_condition_prediction_rows,
)


ENZYMATIC_SOURCES = {
    "chem_enzy_bionav",
    "chem_enzy_onmt",
    "enzyme_precedent",
    "enzexpand",
    "enzyformer",
    "enzymatic",
    "retrorules",
    "rhea",
    "rhea_retrorules",
    "rhea_template",
    "v3_retrieval",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing condition_predictions in native_vs_enhanced route rows."
    )
    parser.add_argument(
        "--rows",
        default=(
            "results/shared/statin_enhanced_formal_depth20_budget480_top20_20260602/"
            "enhanced_all9/native_vs_enhanced_route_rows.jsonl"
        ),
    )
    parser.add_argument("--output", default=None, help="Output JSONL. Defaults to *.rcr_backfilled.jsonl.")
    parser.add_argument("--summary-json", default=None, help="Backfill summary JSON path.")
    parser.add_argument("--summary-md", default=None, help="Backfill summary Markdown path.")
    parser.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    parser.add_argument("--model", choices=["rcr"], default="rcr")
    parser.add_argument("--condition-topk", type=int, default=1)
    parser.add_argument("--max-unique-reactions", type=int, default=0, help="0 means all candidates.")
    parser.add_argument("--include-enzymatic", action="store_true", help="Also run RCR on enzymatic-looking steps.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace existing condition_predictions/step_conditions/conditions.",
    )
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Use the predictor subprocess path instead of the faster in-process batch path.",
    )
    args = parser.parse_args()

    rows_path = Path(args.rows)
    output_path = Path(args.output) if args.output else _default_output_path(rows_path)
    summary_json = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    summary_md = Path(args.summary_md) if args.summary_md else output_path.with_suffix(".summary.md")

    if not args.subprocess:
        os.environ.setdefault("AUTOPLANNER_RCR_CONDITION_INPROCESS", "1")

    started = time.monotonic()
    rows = _read_jsonl(rows_path)
    plan = _collect_backfill_plan(
        rows,
        include_enzymatic=bool(args.include_enzymatic),
        overwrite_existing=bool(args.overwrite_existing),
        max_unique_reactions=max(0, int(args.max_unique_reactions or 0)),
    )
    raw_by_rxn, prediction_errors = _predict_reactions(
        plan["candidate_rxns"],
        vendor_root=Path(args.vendor_root),
        model=str(args.model),
        top_k=max(1, int(args.condition_topk or 1)),
    )
    predictions, normalization_errors = _normalize_predictions(raw_by_rxn)
    apply_stats = _apply_predictions(
        rows,
        predictions=predictions,
        candidate_rxns=set(plan["candidate_rxns"]),
        include_enzymatic=bool(args.include_enzymatic),
        overwrite_existing=bool(args.overwrite_existing),
    )
    elapsed_s = time.monotonic() - started
    summary = _build_summary(
        rows=rows,
        rows_path=rows_path,
        output_path=output_path,
        vendor_root=Path(args.vendor_root),
        model=str(args.model),
        condition_topk=max(1, int(args.condition_topk or 1)),
        include_enzymatic=bool(args.include_enzymatic),
        overwrite_existing=bool(args.overwrite_existing),
        subprocess_mode=bool(args.subprocess),
        collect_plan=plan,
        predictions=predictions,
        prediction_errors=prediction_errors,
        normalization_errors=normalization_errors,
        apply_stats=apply_stats,
        elapsed_s=elapsed_s,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary_json": str(summary_json), "summary_md": str(summary_md)}, indent=2))


def _default_output_path(rows_path: Path) -> Path:
    if rows_path.suffix == ".jsonl":
        return rows_path.with_name(rows_path.stem + ".rcr_backfilled.jsonl")
    return rows_path.with_name(rows_path.name + ".rcr_backfilled.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collect_backfill_plan(
    rows: list[dict[str, Any]],
    *,
    include_enzymatic: bool,
    overwrite_existing: bool,
    max_unique_reactions: int,
) -> dict[str, Any]:
    candidate_rxns: list[str] = []
    seen_rxns: set[str] = set()
    stats: Counter[str] = Counter()
    candidate_steps_by_source: Counter[str] = Counter()
    candidate_unique_by_source: Counter[str] = Counter()
    skipped_enzymatic_by_source: Counter[str] = Counter()

    for _row, _route, step in _iter_steps(rows):
        stats["steps_total"] += 1
        source = str(step.get("source") or "missing")
        if _has_condition(step) and not overwrite_existing:
            stats["skipped_existing_condition"] += 1
            continue
        rxn = _step_reaction_smiles(step)
        if ">>" not in rxn:
            stats["skipped_missing_reaction_smiles"] += 1
            continue
        if not include_enzymatic and _is_enzymatic_step(step):
            stats["skipped_enzymatic"] += 1
            skipped_enzymatic_by_source[source] += 1
            continue
        if max_unique_reactions and len(candidate_rxns) >= max_unique_reactions and rxn not in seen_rxns:
            stats["skipped_unique_limit"] += 1
            continue
        stats["candidate_steps"] += 1
        candidate_steps_by_source[source] += 1
        if rxn not in seen_rxns:
            seen_rxns.add(rxn)
            candidate_rxns.append(rxn)
            candidate_unique_by_source[source] += 1

    return {
        "candidate_rxns": candidate_rxns,
        "stats": dict(sorted(stats.items())),
        "candidate_steps_by_source": dict(sorted(candidate_steps_by_source.items())),
        "candidate_unique_by_source": dict(sorted(candidate_unique_by_source.items())),
        "skipped_enzymatic_by_source": dict(sorted(skipped_enzymatic_by_source.items())),
    }


def _predict_reactions(
    rxns: list[str],
    *,
    vendor_root: Path,
    model: str,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not rxns:
        return {}, {}
    predictor = _cached_condition_predictor(vendor_root, model)
    if hasattr(predictor, "predict_many"):
        try:
            return dict(predictor.predict_many(rxns, top_k=top_k) or {}), {}
        except Exception as exc:  # pragma: no cover - depends on optional vendor env.
            return {}, {rxn: f"{type(exc).__name__}: {exc}" for rxn in rxns}

    raw_by_rxn: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for rxn in rxns:
        try:
            if hasattr(predictor, "predict"):
                raw_by_rxn[rxn] = predictor.predict(rxn, top_k=top_k)
            elif hasattr(predictor, "get_n_conditions"):
                raw_by_rxn[rxn] = predictor.get_n_conditions(rxn, n=top_k, return_scores=True)
            else:
                raw_by_rxn[rxn] = predictor(rxn, top_k=top_k)
        except Exception as exc:  # pragma: no cover - depends on optional vendor env.
            errors[rxn] = f"{type(exc).__name__}: {exc}"
    return raw_by_rxn, errors


def _normalize_predictions(raw_by_rxn: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for rxn, raw in raw_by_rxn.items():
        try:
            rows = _normalize_condition_prediction_rows(raw)
        except Exception as exc:  # pragma: no cover - normalization should be defensive.
            errors[rxn] = f"{type(exc).__name__}: {exc}"
            continue
        normalized = [_annotate_prediction(row) for row in rows if isinstance(row, dict)]
        if normalized:
            predictions[rxn] = normalized
    return predictions, errors


def _annotate_prediction(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("source", "rcr_posthoc_backfill")
    out.setdefault("condition_model", "rcr")
    out.setdefault("condition_prediction_source", "ChemEnzyRCR")
    out.setdefault("condition_prediction_trust", "weak_posthoc_prediction")
    out.setdefault("condition_prediction_enabled_by", "route_rows_posthoc_backfill")
    out.setdefault("condition_label", "RCR model prediction")
    return out


def _apply_predictions(
    rows: list[dict[str, Any]],
    *,
    predictions: dict[str, list[dict[str, Any]]],
    candidate_rxns: set[str],
    include_enzymatic: bool,
    overwrite_existing: bool,
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    filled_by_source: Counter[str] = Counter()
    still_missing_by_source: Counter[str] = Counter()
    for _row, _route, step in _iter_steps(rows):
        source = str(step.get("source") or "missing")
        if _has_condition(step) and not overwrite_existing:
            stats["existing_condition_steps"] += 1
            continue
        rxn = _step_reaction_smiles(step)
        if rxn not in candidate_rxns:
            if not _has_condition(step):
                stats["not_candidate_missing_steps"] += 1
                still_missing_by_source[source] += 1
            continue
        if not include_enzymatic and _is_enzymatic_step(step):
            stats["skipped_enzymatic_steps"] += 1
            if not _has_condition(step):
                still_missing_by_source[source] += 1
            continue
        predicted = predictions.get(rxn) or []
        if predicted:
            step["condition_predictions"] = [dict(row) for row in predicted]
            stats["filled_steps"] += 1
            filled_by_source[source] += 1
        elif not _has_condition(step):
            stats["prediction_empty_or_failed_steps"] += 1
            still_missing_by_source[source] += 1
    return {
        "stats": dict(sorted(stats.items())),
        "filled_steps_by_source": dict(sorted(filled_by_source.items())),
        "still_missing_steps_by_source": dict(sorted(still_missing_by_source.items())),
    }


def _build_summary(
    *,
    rows: list[dict[str, Any]],
    rows_path: Path,
    output_path: Path,
    vendor_root: Path,
    model: str,
    condition_topk: int,
    include_enzymatic: bool,
    overwrite_existing: bool,
    subprocess_mode: bool,
    collect_plan: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    prediction_errors: dict[str, str],
    normalization_errors: dict[str, str],
    apply_stats: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    final_counts = _final_condition_counts(rows)
    values = _prediction_value_summary(predictions)
    return {
        "schema_version": "route_rows_condition_backfill.v1",
        "inputs": {
            "rows": str(rows_path),
            "output": str(output_path),
            "vendor_root": str(vendor_root),
            "model": model,
            "condition_topk": int(condition_topk),
            "include_enzymatic": bool(include_enzymatic),
            "overwrite_existing": bool(overwrite_existing),
            "subprocess_mode": bool(subprocess_mode),
        },
        "summary": {
            "targets": len(rows),
            "routes": sum(1 for row in rows for route in row.get("routes") or [] if isinstance(route, dict)),
            "steps": sum(1 for _row, _route, _step in _iter_steps(rows)),
            "candidate_unique_reactions": len(collect_plan.get("candidate_rxns") or []),
            "unique_predictions": len(predictions),
            "unique_prediction_errors": len(prediction_errors),
            "unique_normalization_errors": len(normalization_errors),
            "filled_steps": int((apply_stats.get("stats") or {}).get("filled_steps") or 0),
            "elapsed_s": round(float(elapsed_s), 3),
            **final_counts,
        },
        "collection": {
            "stats": collect_plan.get("stats") or {},
            "candidate_steps_by_source": collect_plan.get("candidate_steps_by_source") or {},
            "candidate_unique_by_source": collect_plan.get("candidate_unique_by_source") or {},
            "skipped_enzymatic_by_source": collect_plan.get("skipped_enzymatic_by_source") or {},
        },
        "application": apply_stats,
        "prediction_value_summary": values,
        "prediction_error_examples": _error_examples(prediction_errors),
        "normalization_error_examples": _error_examples(normalization_errors),
    }


def _final_condition_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_condition = 0
    missing = 0
    by_source: Counter[str] = Counter()
    missing_by_source: Counter[str] = Counter()
    for _row, _route, step in _iter_steps(rows):
        source = str(step.get("source") or "missing")
        if _has_condition(step):
            with_condition += 1
            by_source[source] += 1
        else:
            missing += 1
            missing_by_source[source] += 1
    total = with_condition + missing
    return {
        "steps_with_condition": with_condition,
        "steps_missing_condition": missing,
        "condition_coverage": round(with_condition / total, 4) if total else 0.0,
        "steps_with_condition_by_source": dict(sorted(by_source.items())),
        "steps_missing_condition_by_source": dict(sorted(missing_by_source.items())),
    }


def _prediction_value_summary(predictions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    scores: list[float] = []
    temps: list[float] = []
    issue_counts: Counter[str] = Counter()
    for rows in predictions.values():
        if not rows:
            continue
        top = rows[0]
        score = _safe_float(top.get("Score") or top.get("score"))
        temp = _safe_float(top.get("Temperature") or top.get("temperature") or top.get("T"))
        if score is not None:
            scores.append(score)
        if temp is not None:
            temps.append(temp)
        issue_counts.update(str(issue) for issue in top.get("condition_prediction_issues") or [])
    return {
        "score": _numeric_summary(scores),
        "temperature_c": _numeric_summary(temps),
        "top1_low_score_lt_0_10": sum(1 for value in scores if value < 0.10),
        "top1_low_temperature_lt_minus_20": sum(1 for value in temps if value < -20.0),
        "top1_high_temperature_gt_100": sum(1 for value in temps if value > 100.0),
        "condition_prediction_issue_counts": dict(sorted(issue_counts.items())),
    }


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "max": round(max(values), 4),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    s = summary.get("summary") or {}
    value = summary.get("prediction_value_summary") or {}
    collection = summary.get("collection") or {}
    application = summary.get("application") or {}
    lines = [
        "# Route Rows Condition Backfill",
        "",
        "## Summary",
        "",
        f"- Targets: `{s.get('targets')}`",
        f"- Routes: `{s.get('routes')}`",
        f"- Steps: `{s.get('steps')}`",
        f"- Candidate unique reactions: `{s.get('candidate_unique_reactions')}`",
        f"- Unique predictions: `{s.get('unique_predictions')}`",
        f"- Filled steps: `{s.get('filled_steps')}`",
        f"- Steps with condition: `{s.get('steps_with_condition')}`",
        f"- Steps missing condition: `{s.get('steps_missing_condition')}`",
        f"- Condition coverage: `{s.get('condition_coverage')}`",
        f"- Prediction errors: `{s.get('unique_prediction_errors')}`",
        f"- Normalization errors: `{s.get('unique_normalization_errors')}`",
        f"- Elapsed seconds: `{s.get('elapsed_s')}`",
        "",
        "## Sources",
        "",
        f"- Candidate steps by source: `{json.dumps(collection.get('candidate_steps_by_source'), sort_keys=True)}`",
        f"- Filled steps by source: `{json.dumps((application.get('filled_steps_by_source') or {}), sort_keys=True)}`",
        f"- Still missing by source: `{json.dumps((application.get('still_missing_steps_by_source') or {}), sort_keys=True)}`",
        "",
        "## Prediction Values",
        "",
        f"- Top-1 score: `{json.dumps(value.get('score'), sort_keys=True)}`",
        f"- Top-1 temperature C: `{json.dumps(value.get('temperature_c'), sort_keys=True)}`",
        f"- Top-1 score < 0.10: `{value.get('top1_low_score_lt_0_10')}`",
        f"- Top-1 temperature < -20 C: `{value.get('top1_low_temperature_lt_minus_20')}`",
        f"- Top-1 temperature > 100 C: `{value.get('top1_high_temperature_gt_100')}`",
        f"- Prediction issue counts: `{json.dumps(value.get('condition_prediction_issue_counts'), sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def _iter_steps(rows: list[dict[str, Any]]):
    for row in rows:
        for route in row.get("routes") or []:
            if not isinstance(route, dict):
                continue
            for step in route.get("steps") or []:
                if isinstance(step, dict):
                    yield row, route, step


def _has_condition(step: dict[str, Any]) -> bool:
    return bool(step.get("condition_predictions") or step.get("step_conditions") or step.get("conditions"))


def _step_reaction_smiles(step: dict[str, Any]) -> str:
    return str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")


def _is_enzymatic_step(step: dict[str, Any]) -> bool:
    source = str(step.get("source") or "").lower()
    text = " ".join(
        str(value or "")
        for value in [
            step.get("reaction_type"),
            step.get("source_model"),
            step.get("ec"),
            (step.get("reaction_interpretation") or {}).get("reaction_class")
            if isinstance(step.get("reaction_interpretation"), dict)
            else "",
        ]
    ).lower()
    return bool(
        step.get("is_enzymatic")
        or step.get("ec")
        or step.get("enzyme_uid")
        or step.get("enzyme_ec_annotations")
        or source in ENZYMATIC_SOURCES
        or "enzym" in text
    )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _error_examples(errors: dict[str, str], *, limit: int = 5) -> list[dict[str, str]]:
    return [
        {"reaction_smiles": rxn, "error": error}
        for rxn, error in list(sorted(errors.items()))[: max(0, int(limit))]
    ]


if __name__ == "__main__":
    main()
