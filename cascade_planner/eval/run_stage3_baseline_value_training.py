"""Close out the baseline-only Stage 3 value-model training chain.

This runner is deliberately narrower than the full v4 manifest.  It consumes
completed baseline train/val ChemEnzy traces, builds action/source and
transition packs, then trains the three Stage 3 models.  It does not read the
locked full100 benchmark and should not be used for final evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade_planner.eval.build_cascade_action_value_pack import build_cascade_action_value_pack
from cascade_planner.eval.build_cascade_transition_pack import build_cascade_transition_pack
from cascade_planner.eval.run_cascade_search_benchmark import merge_cascade_search_outputs
from cascade_planner.eval.train_cascade_action_value import train_cascade_action_value
from cascade_planner.eval.train_cascade_source_value import train_cascade_source_value
from cascade_planner.eval.train_cascade_transition_value import train_cascade_transition_value


TRACE_SPLITS = ("train", "val")


@dataclass(frozen=True)
class Stage3BaselineInputs:
    output_root: Path
    split_dir: Path
    num_shards: int
    runtime_shards: dict[str, list[Path]]
    expansion_traces: dict[str, list[Path]]
    cascade_traces: dict[str, list[Path]]
    merged_runtimes: dict[str, Path]
    benchmark_all: Path


@dataclass(frozen=True)
class Stage3BaselineOutputs:
    action_pack_dir: Path
    transition_pack_dir: Path
    action_model: Path
    source_model: Path
    transition_model: Path
    action_report: Path
    source_report: Path
    transition_report: Path
    summary_report: Path


def expected_baseline_inputs(
    *,
    output_root: Path,
    split_dir: Path,
    num_shards: int = 8,
) -> Stage3BaselineInputs:
    output_root = Path(output_root)
    split_dir = Path(split_dir)
    runtime_shards: dict[str, list[Path]] = {}
    expansion_traces: dict[str, list[Path]] = {}
    cascade_traces: dict[str, list[Path]] = {}
    merged_runtimes: dict[str, Path] = {}
    for split in TRACE_SPLITS:
        base_dir = output_root / "traces" / "baseline" / split
        runtime_shards[split] = [
            base_dir / f"runtime_shard{shard}of{num_shards}.json"
            for shard in range(num_shards)
        ]
        expansion_traces[split] = [
            base_dir / f"chem_enzy_expansion_shard{shard}of{num_shards}.jsonl"
            for shard in range(num_shards)
        ]
        cascade_traces[split] = [
            base_dir / f"cascade_trace_shard{shard}of{num_shards}.jsonl"
            for shard in range(num_shards)
        ]
        merged_runtimes[split] = output_root / "merged" / f"baseline_{split}_runtime.json"
    return Stage3BaselineInputs(
        output_root=output_root,
        split_dir=split_dir,
        num_shards=num_shards,
        runtime_shards=runtime_shards,
        expansion_traces=expansion_traces,
        cascade_traces=cascade_traces,
        merged_runtimes=merged_runtimes,
        benchmark_all=split_dir / "v4_trace_candidates_all.json",
    )


def expected_baseline_outputs(output_root: Path) -> Stage3BaselineOutputs:
    output_root = Path(output_root)
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    return Stage3BaselineOutputs(
        action_pack_dir=output_root / "packs" / "v4_baseline_action_value_pack",
        transition_pack_dir=output_root / "packs" / "v4_baseline_transition_value_pack",
        action_model=models_dir / "cascade_action_value_route_quality_baseline.pt",
        source_model=models_dir / "cascade_source_value_baseline.pt",
        transition_model=models_dir / "cascade_transition_value_baseline.pt",
        action_report=reports_dir / "cascade_action_value_route_quality_baseline.json",
        source_report=reports_dir / "cascade_source_value_baseline.json",
        transition_report=reports_dir / "cascade_transition_value_baseline.json",
        summary_report=reports_dir / "stage3_baseline_value_training_summary.json",
    )


def missing_input_paths(inputs: Stage3BaselineInputs) -> list[Path]:
    required = [inputs.benchmark_all]
    for split in TRACE_SPLITS:
        required.extend(inputs.runtime_shards[split])
        required.extend(inputs.expansion_traces[split])
        required.extend(inputs.cascade_traces[split])
    return [path for path in required if not Path(path).exists()]


def run_stage3_baseline_value_training(
    *,
    output_root: Path,
    split_dir: Path,
    num_shards: int = 8,
    action_epochs: int = 20,
    source_epochs: int = 20,
    transition_epochs: int = 8,
    device: str = "cpu",
    validate_only: bool = False,
) -> dict[str, Any]:
    inputs = expected_baseline_inputs(
        output_root=output_root,
        split_dir=split_dir,
        num_shards=num_shards,
    )
    outputs = expected_baseline_outputs(output_root)
    missing = missing_input_paths(inputs)
    validation = {
        "output_root": str(inputs.output_root),
        "split_dir": str(inputs.split_dir),
        "num_shards": inputs.num_shards,
        "required_inputs": {
            "benchmark_all": str(inputs.benchmark_all),
            "runtime_shards": {
                split: [str(path) for path in inputs.runtime_shards[split]]
                for split in TRACE_SPLITS
            },
            "expansion_traces": {
                split: [str(path) for path in inputs.expansion_traces[split]]
                for split in TRACE_SPLITS
            },
            "cascade_traces": {
                split: [str(path) for path in inputs.cascade_traces[split]]
                for split in TRACE_SPLITS
            },
        },
        "missing_inputs": [str(path) for path in missing],
    }
    if validate_only:
        return {"validated": not missing, **validation}
    if missing:
        examples = "\n".join(str(path) for path in missing[:20])
        more = f"\n... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"baseline Stage3 trace inputs are incomplete:\n{examples}{more}")

    started = time.monotonic()
    outputs.summary_report.parent.mkdir(parents=True, exist_ok=True)

    merge_reports = {}
    for split in TRACE_SPLITS:
        merge_reports[split] = merge_cascade_search_outputs(
            inputs.runtime_shards[split],
            inputs.merged_runtimes[split],
        )["summary"]

    runtime_paths = [inputs.merged_runtimes[split] for split in TRACE_SPLITS]
    cascade_trace_paths = [
        path
        for split in TRACE_SPLITS
        for path in inputs.cascade_traces[split]
    ]

    action_pack_report = build_cascade_action_value_pack(
        trace_paths=cascade_trace_paths,
        benchmark_path=inputs.benchmark_all,
        output_dir=outputs.action_pack_dir,
        runtime_paths=runtime_paths,
        preserve_benchmark_splits=True,
    )
    transition_pack_report = build_cascade_transition_pack(
        trace_paths=cascade_trace_paths,
        output_dir=outputs.transition_pack_dir,
    )

    action_report = train_cascade_action_value(
        pack_dir=outputs.action_pack_dir,
        model_output=outputs.action_model,
        report_output=outputs.action_report,
        md_output=outputs.action_report.with_suffix(".md"),
        epochs=action_epochs,
        label_name="route_quality_action_value",
        loss_mode="pairwise",
        selection_metric="pairwise_positive_state_accuracy",
        device=device,
    )
    source_report = train_cascade_source_value(
        pack_dir=outputs.action_pack_dir,
        model_output=outputs.source_model,
        report_output=outputs.source_report,
        md_output=outputs.source_report.with_suffix(".md"),
        epochs=source_epochs,
        loss_mode="pairwise",
        selection_metric="top1_positive_state_hit_rate",
        device=device,
    )
    transition_report = train_cascade_transition_value(
        pack_dir=outputs.transition_pack_dir,
        model_output=outputs.transition_model,
        report_output=outputs.transition_report,
        md_output=outputs.transition_report.with_suffix(".md"),
        epochs=transition_epochs,
        device=device,
    )

    summary = {
        "schema_version": "stage3_baseline_value_training.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.monotonic() - started, 3),
        "training_use": "train_val_only_no_full100",
        "validation": validation,
        "merged_runtime_summaries": merge_reports,
        "action_pack": {
            "dir": str(outputs.action_pack_dir),
            "summary": action_pack_report.get("summary") or {},
            "splits": action_pack_report.get("splits") or {},
        },
        "transition_pack": {
            "dir": str(outputs.transition_pack_dir),
            "counts": transition_pack_report.get("counts") or {},
            "label_bins": transition_pack_report.get("label_bins") or {},
        },
        "models": {
            "action_value": str(outputs.action_model),
            "source_value": str(outputs.source_model),
            "transition_value": str(outputs.transition_model),
        },
        "reports": {
            "action_value": str(outputs.action_report),
            "source_value": str(outputs.source_report),
            "transition_value": str(outputs.transition_report),
        },
        "model_metrics": {
            "action_value": action_report.get("best_checkpoint") or {},
            "source_value": source_report.get("best_checkpoint") or {},
            "transition_value": transition_report.get("final_metrics") or {},
        },
    }
    outputs.summary_report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline-only Stage 3 value-model training")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--action-epochs", type=int, default=20)
    ap.add_argument("--source-epochs", type=int, default=20)
    ap.add_argument("--transition-epochs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    report = run_stage3_baseline_value_training(
        output_root=Path(args.output_root),
        split_dir=Path(args.split_dir),
        num_shards=args.num_shards,
        action_epochs=args.action_epochs,
        source_epochs=args.source_epochs,
        transition_epochs=args.transition_epochs,
        device=args.device,
        validate_only=args.validate_only,
    )
    if args.validate_only:
        print(json.dumps({
            "validated": report["validated"],
            "missing_count": len(report["missing_inputs"]),
        }, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({
            "elapsed_s": report["elapsed_s"],
            "models": report["models"],
            "action_pack": report["action_pack"],
            "transition_pack": report["transition_pack"],
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
