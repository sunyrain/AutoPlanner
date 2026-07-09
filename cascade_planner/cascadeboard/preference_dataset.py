"""Preference-pair data contract for objective-specific CascadeBoard learning."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

VALID_OBJECTIVES = {"balanced", "industrial", "green", "novelty"}
VALID_LABELS = {"a_preferred", "b_preferred", "tie", "incomparable"}
REQUIRED_FIELDS = {
    "pair_id",
    "objective",
    "route_a",
    "route_b",
    "label",
    "annotator_type",
}


SCHEMA = {
    "description": "Objective-specific route preference pairs for learned Bradley-Terry training.",
    "required_fields": sorted(REQUIRED_FIELDS),
    "valid_objectives": sorted(VALID_OBJECTIVES),
    "valid_labels": sorted(VALID_LABELS),
    "route_fields_minimum": [
        "route_id",
        "target_smiles",
        "steps",
        "quality_vector",
        "risk_vector",
        "candidate_sources",
    ],
    "notes": [
        "Use label=incomparable for Pareto-incomparable routes; these pairs are excluded from BT loss.",
        "Do not convert synthetic clean-vs-corrupted labels into objective-specific human preference labels.",
        "Keep annotator_type separate: expert, chemist, process_engineer, computed_rule, or user_feedback.",
        "Include rationale/free-text when available, but model training should not require it.",
    ],
}


def validate_pairs(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        pairs = data.get("pairs", [])
    else:
        pairs = data

    errors = []
    counts = {
        "n_pairs": len(pairs),
        "by_objective": {},
        "by_label": {},
        "bt_trainable": 0,
        "incomparable": 0,
    }
    for idx, pair in enumerate(pairs):
        missing = REQUIRED_FIELDS - set(pair)
        if missing:
            errors.append({"index": idx, "error": "missing_fields", "fields": sorted(missing)})
            continue
        obj = pair.get("objective")
        label = pair.get("label")
        counts["by_objective"][obj] = counts["by_objective"].get(obj, 0) + 1
        counts["by_label"][label] = counts["by_label"].get(label, 0) + 1
        if obj not in VALID_OBJECTIVES:
            errors.append({"index": idx, "error": "invalid_objective", "value": obj})
        if label not in VALID_LABELS:
            errors.append({"index": idx, "error": "invalid_label", "value": label})
        if label in {"a_preferred", "b_preferred"}:
            counts["bt_trainable"] += 1
        elif label == "incomparable":
            counts["incomparable"] += 1
        for side in ("route_a", "route_b"):
            route = pair.get(side) or {}
            if not route.get("route_id"):
                errors.append({"index": idx, "error": "missing_route_id", "side": side})
            if not route.get("steps"):
                errors.append({"index": idx, "error": "missing_route_steps", "side": side})

    return {
        "metadata": {"date": time.strftime("%Y-%m-%d"), "path": str(p)},
        "valid": not errors,
        "counts": counts,
        "errors": errors[:200],
        "schema": SCHEMA,
    }


def write_schema(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(SCHEMA, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-output", default="data/cascadeboard_preference_pairs.schema.json")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--report", default="results/v2/cascadeboard_preference_data_report.json")
    args = ap.parse_args()
    write_schema(args.schema_output)
    if args.pairs:
        report = validate_pairs(args.pairs)
    else:
        report = {
            "metadata": {"date": time.strftime("%Y-%m-%d")},
            "valid": False,
            "counts": {"n_pairs": 0, "bt_trainable": 0},
            "errors": [{"error": "no_preference_pairs_provided"}],
            "schema": SCHEMA,
        }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "schema_output": args.schema_output,
        "report": args.report,
        "valid": report["valid"],
        "n_pairs": report["counts"].get("n_pairs", 0),
        "bt_trainable": report["counts"].get("bt_trainable", 0),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
