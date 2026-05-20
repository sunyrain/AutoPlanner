"""Run the full route-pool evidence review pipeline.

Pipeline:
  optional prompt build -> LLM/dry-run responses -> validated labels
  -> label summary -> promotion gate.

The default is dry-run, so this command is safe to execute without API keys.
Use ``--no-dry-run`` only when real review calls should be made and
``DEEPSEEK_API_KEY`` is present in the environment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cascade_planner.eval.build_route_pool_evidence_review_prompts import build_route_pool_evidence_review_prompts
from cascade_planner.eval.calibrate_route_pool_evidence_review_signals import calibrate_route_pool_evidence_review_signals
from cascade_planner.eval.gate_route_pool_evidence_review_promotion import gate_route_pool_evidence_review_promotion
from cascade_planner.eval.ingest_route_pool_evidence_review_results import ingest_route_pool_evidence_review_results
from cascade_planner.eval.run_route_pool_evidence_llm_review import run_route_pool_evidence_llm_review
from cascade_planner.eval.summarize_route_pool_evidence_review_labels import summarize_route_pool_evidence_review_labels


SCHEMA_VERSION = "route_pool_evidence_review_pipeline.v1"


def run_route_pool_evidence_review_pipeline(
    *,
    prompts_jsonl: Path,
    output_dir: Path,
    review_jsonl: Path | None = None,
    transform_sanity_json: Path | None = None,
    prefix: str = "route_pool_evidence_review",
    cache_path: Path | None = None,
    model: str | None = None,
    max_rows: int | None = None,
    start_index: int = 0,
    dry_run: bool = True,
    resume: bool = False,
    continue_on_error: bool = True,
    workers: int = 1,
    sleep_s: float = 0.0,
    min_rows: int = 30,
    min_usable_positive: int = 5,
    min_usable_negative: int = 5,
    max_unclear_rate: float = 0.50,
    min_evidence_classes: int = 2,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prompts_jsonl": prompts_jsonl,
        "prompt_report_json": output_dir / f"{prefix}_prompts_report.json",
        "responses_jsonl": output_dir / f"{prefix}_responses.jsonl",
        "run_report_json": output_dir / f"{prefix}_run_report.json",
        "labels_jsonl": output_dir / f"{prefix}_labels.jsonl",
        "label_report_json": output_dir / f"{prefix}_labels_report.json",
        "label_summary_json": output_dir / f"{prefix}_label_summary.json",
        "signal_calibration_json": output_dir / f"{prefix}_signal_calibration.json",
        "promotion_gate_json": output_dir / f"{prefix}_promotion_gate.json",
        "pipeline_manifest_json": output_dir / f"{prefix}_pipeline_manifest.json",
    }
    prompt_report = {}
    if review_jsonl is not None:
        prompt_report = build_route_pool_evidence_review_prompts(
            review_jsonl=review_jsonl,
            output_jsonl=prompts_jsonl,
            report_json=paths["prompt_report_json"],
            transform_sanity_json=transform_sanity_json,
        )
    run_report = run_route_pool_evidence_llm_review(
        prompts_jsonl=prompts_jsonl,
        output_jsonl=paths["responses_jsonl"],
        report_json=paths["run_report_json"],
        cache_path=cache_path,
        model=model,
        max_rows=max_rows,
        start_index=start_index,
        dry_run=dry_run,
        resume=resume,
        continue_on_error=continue_on_error,
        workers=workers,
        sleep_s=sleep_s,
    )
    label_report = ingest_route_pool_evidence_review_results(
        prompts_jsonl=prompts_jsonl,
        responses_jsonl=paths["responses_jsonl"],
        output_jsonl=paths["labels_jsonl"],
        report_json=paths["label_report_json"],
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
            "prompts_jsonl": str(prompts_jsonl),
            "review_jsonl": str(review_jsonl) if review_jsonl else None,
            "transform_sanity_json": str(transform_sanity_json) if transform_sanity_json else None,
            "output_dir": str(output_dir),
            "prefix": prefix,
            "dry_run": dry_run,
            "resume": resume,
            "workers": workers,
            "max_rows": max_rows,
            "start_index": start_index,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "summaries": {
            "prompts": prompt_report.get("summary") or {},
            "run": run_report.get("summary") or {},
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
        "# Route Pool Evidence Review Pipeline",
        "",
        "## Run",
        "",
        "```json",
        json.dumps(summaries.get("run") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Labels",
        "",
        "```json",
        json.dumps(summaries.get("labels") or {}, indent=2, ensure_ascii=False),
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
    parser = argparse.ArgumentParser(description="Run the route-pool evidence review pipeline")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--review-jsonl")
    parser.add_argument("--transform-sanity-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="route_pool_evidence_review")
    parser.add_argument("--cache")
    parser.add_argument("--model")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-dry-run", action="store_true")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-usable-positive", type=int, default=5)
    parser.add_argument("--min-usable-negative", type=int, default=5)
    parser.add_argument("--max-unclear-rate", type=float, default=0.50)
    parser.add_argument("--min-evidence-classes", type=int, default=2)
    args = parser.parse_args()
    manifest = run_route_pool_evidence_review_pipeline(
        prompts_jsonl=Path(args.prompts_jsonl),
        output_dir=Path(args.output_dir),
        review_jsonl=Path(args.review_jsonl) if args.review_jsonl else None,
        transform_sanity_json=Path(args.transform_sanity_json) if args.transform_sanity_json else None,
        prefix=args.prefix,
        cache_path=Path(args.cache) if args.cache else None,
        model=args.model,
        max_rows=args.max_rows,
        start_index=args.start_index,
        dry_run=not args.no_dry_run,
        resume=bool(args.resume),
        continue_on_error=not bool(args.fail_fast),
        workers=int(args.workers),
        sleep_s=float(args.sleep_s),
        min_rows=int(args.min_rows),
        min_usable_positive=int(args.min_usable_positive),
        min_usable_negative=int(args.min_usable_negative),
        max_unclear_rate=float(args.max_unclear_rate),
        min_evidence_classes=int(args.min_evidence_classes),
    )
    print(json.dumps({"summaries": manifest["summaries"], "outputs": manifest["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
