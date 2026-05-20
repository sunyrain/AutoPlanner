"""Strict project report card with separated metric families."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _candidate_dataset_summary(path: str | Path) -> dict[str, Any] | None:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        return None
    meta = dict(obj.get("metadata") or {})
    meta["examples_omitted_from_report_card"] = True
    return {
        "metadata": meta,
        "overall": obj.get("overall"),
        "by_domain": obj.get("by_domain"),
        "by_ec1": obj.get("by_ec1"),
    }


def _float(v: Any) -> float | None:
    try:
        if v in ("", None):
            return None
        return float(v)
    except Exception:
        return None


def build_report_card(output_json: str, output_md: str) -> dict[str, Any]:
    single_step = {
        "hybrid_multi_audited": _load_csv("results/v2/hybrid_multi_audited_overall.csv"),
        "condition_diagnosis": _load_csv("results/v2/condition_diagnosis.csv"),
        "k2_summary_partial": _load_json("results/v2/k2_uspto50k_summary_partial.json"),
        "k2_summary_full": _load_json("results/v2/k2_uspto50k_summary_full.json"),
    }

    multi_step = {
        "benchmark_v2_100_summary": _load_csv("results/v2/benchmark_v2_100_summary.csv"),
        "benchmark_v2_100_summary_hybrid": _load_csv("results/v2/benchmark_v2_100_summary_hybrid.csv"),
        "aizynthfinder_step_eval": "results/v2/aizynthfinder_full_gpu_step_eval.csv",
    }

    controlled = {
        "constraint_and_repair": _load_json("results/v2/cascadeboard_benchmarks.json"),
        "policy_benchmark_v7": _load_json("results/v2/cascadeboard_policy_benchmark_v7.json"),
    }

    strict = {
        "retrochimera_only": _load_json("results/v2/cascadeboard_real_benchmark.json"),
        "retrochimera_plus_enzexpand_dualtower": _load_json("results/v2/cascadeboard_real_benchmark_enzexpand.json"),
        "candidate_supervision_dataset": _candidate_dataset_summary("results/shared/cascadeboard_candidate_supervision_v1.json"),
        "candidate_supervision_training": _load_json("results/v2/cascadeboard_candidate_supervision_report.json"),
    }

    warnings = [
        "K6 pH is not met unless a result explicitly shows pH MAE <= 0.50 on a held-out split.",
        "CascadeBoard mock/controlled repair metrics are not substitutes for strict no-mock real-candidate metrics.",
        "Synthetic Bradley-Terry preference loss is not objective-specific human/expert preference learning.",
        "Candidate nearest labels are weak supervision and must be reported separately from exact GT-in-pool.",
    ]

    result = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "purpose": "separate single-step, multi-step, controlled CascadeBoard, and strict no-mock CascadeBoard metrics",
        },
        "single_step_metrics": single_step,
        "multi_step_mcts_metrics": multi_step,
        "cascadeboard_controlled_metrics": controlled,
        "cascadeboard_strict_no_mock_metrics": strict,
        "reporting_warnings": warnings,
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# CascadeBoard++ Strict Report Card",
        "",
        f"Date: {result['metadata']['date']}",
        "",
        "## Reporting Warnings",
        "",
    ]
    lines.extend(f"- {w}" for w in warnings)
    lines.extend(["", "## Single-Step Metrics", ""])
    for name, value in single_step.items():
        if value:
            lines.append(f"- `{name}`: present")
        else:
            lines.append(f"- `{name}`: missing")
    lines.extend(["", "## Multi-Step MCTS Metrics", ""])
    for name, value in multi_step.items():
        lines.append(f"- `{name}`: {'present' if value else 'missing'}")
    lines.extend(["", "## CascadeBoard Controlled Metrics", ""])
    for name, value in controlled.items():
        lines.append(f"- `{name}`: {'present' if value else 'missing'}")
    lines.extend(["", "## CascadeBoard Strict No-Mock Metrics", ""])
    for name, value in strict.items():
        if isinstance(value, dict) and value.get("overall"):
            ov = value["overall"]
            if "plan_rate" in ov:
                lines.append(
                    f"- `{name}`: plan_rate={ov.get('plan_rate')}, "
                    f"GT@5={ov.get('gt_at_5')}, strict_stock_solve={ov.get('strict_stock_solve')}"
                )
            else:
                lines.append(
                    f"- `{name}`: pool_coverage={ov.get('candidate_pool_coverage')}, "
                    f"exact_GT_in_pool={ov.get('exact_gt_in_pool')}, exact_GT@5={ov.get('exact_gt_at_5')}"
                )
        elif isinstance(value, dict) and value.get("best_val_loss") is not None:
            lines.append(f"- `{name}`: best_val_loss={value.get('best_val_loss')}")
        else:
            lines.append(f"- `{name}`: {'present' if value else 'missing'}")
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", default="results/v2/cascadeboard_report_card.json")
    ap.add_argument("--output-md", default="results/v2/cascadeboard_report_card.md")
    args = ap.parse_args()
    result = build_report_card(args.output_json, args.output_md)
    print(json.dumps({
        "output_json": args.output_json,
        "output_md": args.output_md,
        "warnings": result["reporting_warnings"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
