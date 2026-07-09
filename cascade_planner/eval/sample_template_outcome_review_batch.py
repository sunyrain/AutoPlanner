"""Sample balanced review batches from template outcome supervision packs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "template_outcome_review_batch.v1"
DEFAULT_CLASSES = (
    "pair_and_analog_positive",
    "analog_only_positive",
    "pair_only_near_miss",
    "high_score_hard_negative",
)


def sample_template_outcome_review_batch(
    *,
    supervision_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    output_csv: Path | None = None,
    per_class: int = 25,
    seed: int = 42,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> dict[str, Any]:
    rows = _read_jsonl(supervision_jsonl)
    rng = random.Random(int(seed))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cls = str(row.get("supervision_class") or "")
        if cls in classes:
            grouped[cls].append(row)

    selected = []
    for cls in classes:
        candidates = list(grouped.get(cls) or [])
        candidates.sort(key=_stable_sort_key)
        rng.shuffle(candidates)
        selected.extend(_review_row(row) for row in candidates[: max(0, int(per_class))])
    selected.sort(key=lambda row: (row["supervision_class"], row["review_id"]))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if output_csv is not None:
        _write_csv(selected, output_csv)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "supervision_jsonl": str(supervision_jsonl),
            "output_jsonl": str(output_jsonl),
            "output_csv": str(output_csv) if output_csv else None,
            "report_json": str(report_json),
            "per_class": per_class,
            "seed": seed,
            "classes": list(classes),
        },
        "summary": {
            "source_rows": len(rows),
            "sampled_rows": len(selected),
            "sampled_classes": dict(Counter(row["supervision_class"] for row in selected)),
            "source_classes": dict(Counter(str(row.get("supervision_class") or "") for row in rows)),
        },
        "review_contract": {
            "diagnostic_labels_are_not_ground_truth": True,
            "expert_fields_to_fill": [
                "expert_template_applicable",
                "expert_outcome_plausible",
                "expert_cascade_coherent",
                "expert_priority",
                "expert_reject_reason",
                "expert_comments",
            ],
        },
        "examples": selected[:10],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _review_row(row: dict[str, Any]) -> dict[str, Any]:
    labels = row.get("labels") or {}
    similarities = row.get("similarities") or {}
    reference = row.get("reference") or {}
    features = row.get("features") or {}
    review_id = _review_id(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "review_id": review_id,
        "source_pool": row.get("source_pool"),
        "split": row.get("split"),
        "target_smiles": row.get("target_smiles"),
        "supervision_class": row.get("supervision_class"),
        "proposal_rank": row.get("proposal_rank"),
        "proposal_score": row.get("proposal_score"),
        "downstream_rank": row.get("downstream_rank"),
        "connector": row.get("connector"),
        "template_transform_pair": row.get("template_transform_pair"),
        "template": row.get("template"),
        "reactants": row.get("reactants") or [],
        "main_reactant": row.get("main_reactant"),
        "diagnostic_labels": {
            "pair_hit": bool(labels.get("pair_hit")),
            "analog_hit": bool(labels.get("analog_hit")),
            "pair_and_analog": bool(labels.get("pair_and_analog")),
        },
        "diagnostic_similarities": {
            "upstream_similarity": similarities.get("upstream_similarity"),
            "downstream_similarity": similarities.get("downstream_similarity"),
        },
        "reference": {
            "best_reference_block_id": reference.get("best_reference_block_id"),
            "reference_transform_pair": reference.get("reference_transform_pair"),
        },
        "feature_highlights": {
            key: features.get(key)
            for key in (
                "app_connector_main_similarity",
                "app_template_example_best_transition_sim",
                "rc_inherited_atom_fraction",
                "rc_new_atom_fraction",
                "rc_template_matched_fraction",
            )
            if key in features
        },
        "expert_template_applicable": None,
        "expert_outcome_plausible": None,
        "expert_cascade_coherent": None,
        "expert_priority": None,
        "expert_reject_reason": None,
        "expert_comments": None,
    }


def _review_id(row: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "source_pool": row.get("source_pool"),
            "target": row.get("target_smiles"),
            "connector": row.get("connector"),
            "template": row.get("template"),
            "reactants": row.get("reactants"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _stable_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("target_smiles") or ""), str(row.get("proposal_rank") or ""), _review_id(row))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "supervision_class",
        "source_pool",
        "split",
        "target_smiles",
        "connector",
        "template_transform_pair",
        "reactants",
        "main_reactant",
        "proposal_score",
        "proposal_rank",
        "downstream_rank",
        "diagnostic_labels",
        "diagnostic_similarities",
        "feature_highlights",
        "expert_template_applicable",
        "expert_outcome_plausible",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_comments",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("reactants", "diagnostic_labels", "diagnostic_similarities", "feature_highlights"):
                flat[key] = json.dumps(flat.get(key), ensure_ascii=False)
            writer.writerow({key: flat.get(key) for key in fieldnames})


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Template Outcome Review Batch",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Contract", "", "Diagnostic labels are not expert ground truth. Fill expert fields before using this as supervised training data.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a balanced template outcome review batch")
    parser.add_argument("--supervision-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    args = parser.parse_args()
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    report = sample_template_outcome_review_batch(
        supervision_jsonl=Path(args.supervision_jsonl),
        output_jsonl=Path(args.output_jsonl),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        report_json=Path(args.report_json),
        per_class=args.per_class,
        seed=args.seed,
        classes=classes,
    )
    print(json.dumps({"summary": report["summary"], "output_jsonl": args.output_jsonl, "report_json": args.report_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
