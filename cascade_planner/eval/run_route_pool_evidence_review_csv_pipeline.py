"""Run the human CSV route-pool evidence review pipeline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cascade_planner.eval.calibrate_route_pool_evidence_review_signals import calibrate_route_pool_evidence_review_signals
from cascade_planner.eval.gate_route_pool_evidence_review_promotion import gate_route_pool_evidence_review_promotion
from cascade_planner.eval.ingest_route_pool_evidence_review_csv import ingest_route_pool_evidence_review_csv
from cascade_planner.eval.summarize_route_pool_evidence_review_labels import summarize_route_pool_evidence_review_labels


SCHEMA_VERSION = "route_pool_evidence_review_csv_pipeline.v1"


def run_route_pool_evidence_review_csv_pipeline(
    *,
    review_csv: Path,
    output_dir: Path,
    prefix: str = "human_route_pool_evidence_review",
    min_rows: int = 30,
    min_usable_positive: int = 5,
    min_usable_negative: int = 5,
    max_unclear_rate: float = 0.50,
    min_evidence_classes: int = 2,
    min_auc: float = 0.65,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "labels_jsonl": output_dir / f"{prefix}_labels.jsonl",
        "labels_report_json": output_dir / f"{prefix}_labels_report.json",
        "invalid_jsonl": output_dir / f"{prefix}_labels_invalid.jsonl",
        "unreviewed_jsonl": output_dir / f"{prefix}_labels_unreviewed.jsonl",
        "label_summary_json": output_dir / f"{prefix}_label_summary.json",
        "signal_calibration_json": output_dir / f"{prefix}_signal_calibration.json",
        "promotion_gate_json": output_dir / f"{prefix}_promotion_gate.json",
        "pipeline_manifest_json": output_dir / f"{prefix}_csv_pipeline_manifest.json",
    }
    label_report = ingest_route_pool_evidence_review_csv(
        review_csv=review_csv,
        output_jsonl=paths["labels_jsonl"],
        report_json=paths["labels_report_json"],
        invalid_jsonl=paths["invalid_jsonl"],
        unreviewed_jsonl=paths["unreviewed_jsonl"],
    )
    label_summary = summarize_route_pool_evidence_review_labels(
        labels_jsonl=paths["labels_jsonl"],
        output_json=paths["label_summary_json"],
    )
    signal_calibration = calibrate_route_pool_evidence_review_signals(
        labels_jsonl=paths["labels_jsonl"],
        output_json=paths["signal_calibration_json"],
        min_rows=min_rows,
        min_positive=min_usable_positive,
        min_negative=min_usable_negative,
        min_auc=min_auc,
    )
    gate = gate_route_pool_evidence_review_promotion(
        label_summary_json=paths["label_summary_json"],
        output_json=paths["promotion_gate_json"],
        min_rows=min_rows,
        min_usable_positive=min_usable_positive,
        min_usable_negative=min_usable_negative,
        max_unclear_rate=max_unclear_rate,
        min_evidence_classes=min_evidence_classes,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "review_csv": str(review_csv),
            "output_dir": str(output_dir),
            "prefix": prefix,
            "elapsed_s": round(time.monotonic() - started, 3),
            "thresholds": {
                "min_rows": min_rows,
                "min_usable_positive": min_usable_positive,
                "min_usable_negative": min_usable_negative,
                "max_unclear_rate": max_unclear_rate,
                "min_evidence_classes": min_evidence_classes,
                "min_auc": min_auc,
            },
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "summaries": {
            "labels": label_report.get("summary") or {},
            "label_summary": label_summary.get("summary") or {},
            "signal_calibration": {
                "ready_for_proxy_training": (signal_calibration.get("decision") or {}).get("ready_for_proxy_training"),
                "recommendation": (signal_calibration.get("decision") or {}).get("recommendation"),
            },
            "promotion_gate": {
                "ready_for_training": gate.get("ready_for_training"),
                "recommendation": gate.get("recommendation"),
            },
        },
    }
    paths["pipeline_manifest_json"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["pipeline_manifest_json"].with_suffix(".md").write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def _markdown(manifest: dict[str, Any]) -> str:
    summaries = manifest.get("summaries") or {}
    lines = [
        "# Route Pool Evidence Human CSV Review Pipeline",
        "",
        "## Labels",
        "",
        "```json",
        json.dumps(summaries.get("labels") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Signal Calibration",
        "",
        "```json",
        json.dumps(summaries.get("signal_calibration") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Promotion Gate",
        "",
        "```json",
        json.dumps(summaries.get("promotion_gate") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run human CSV route-pool evidence review pipeline")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="human_route_pool_evidence_review")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-usable-positive", type=int, default=5)
    parser.add_argument("--min-usable-negative", type=int, default=5)
    parser.add_argument("--max-unclear-rate", type=float, default=0.50)
    parser.add_argument("--min-evidence-classes", type=int, default=2)
    parser.add_argument("--min-auc", type=float, default=0.65)
    args = parser.parse_args()
    manifest = run_route_pool_evidence_review_csv_pipeline(
        review_csv=Path(args.review_csv),
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        min_rows=int(args.min_rows),
        min_usable_positive=int(args.min_usable_positive),
        min_usable_negative=int(args.min_usable_negative),
        max_unclear_rate=float(args.max_unclear_rate),
        min_evidence_classes=int(args.min_evidence_classes),
        min_auc=float(args.min_auc),
    )
    print(json.dumps({"summaries": manifest["summaries"], "outputs": manifest["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
