"""Audit reservoir-distilled controller artifacts against the implementation plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_PACK_FILES = [
    "reservoir_distill_pack_train.jsonl",
    "reservoir_distill_pack_val.jsonl",
    "reservoir_distill_pack_eval_full100.jsonl",
    "reservoir_distill_manifest.json",
    "teacher_report.json",
    "reservoir_distilled_controller.pt",
    "reservoir_distilled_controller.json",
]


def build_completion_audit(
    *,
    distill_dir: Path,
    acceptance_dir: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    distill_dir = Path(distill_dir)
    acceptance_dir = Path(acceptance_dir)
    checks = []

    manifest = _load_json(distill_dir / "reservoir_distill_manifest.json")
    teacher = _load_json(distill_dir / "teacher_report.json")
    train_rows = _jsonl_count(distill_dir / "reservoir_distill_pack_train.jsonl")
    val_rows = _jsonl_count(distill_dir / "reservoir_distill_pack_val.jsonl")
    eval_rows = _jsonl_count(distill_dir / "reservoir_distill_pack_eval_full100.jsonl")
    eval_flags = _eval_flag_counts(distill_dir / "reservoir_distill_pack_eval_full100.jsonl")
    train_flags = _eval_flag_counts(distill_dir / "reservoir_distill_pack_train.jsonl")
    val_flags = _eval_flag_counts(distill_dir / "reservoir_distill_pack_val.jsonl")

    checks.append(_check(
        "teacher_pack_files",
        all((distill_dir / name).exists() for name in REQUIRED_PACK_FILES),
        {"required_files": REQUIRED_PACK_FILES},
    ))
    checks.append(_check(
        "train_val_eval_counts",
        train_rows > 0 and val_rows > 0 and eval_rows > 0,
        {"train_rows": train_rows, "val_rows": val_rows, "eval_rows": eval_rows, "manifest_counts": manifest.get("counts")},
    ))
    checks.append(_check(
        "full100_eval_only_isolation",
        eval_flags.get("eval_true", 0) == eval_rows and train_flags.get("eval_true", 0) == 0 and val_flags.get("eval_true", 0) == 0,
        {"train_flags": train_flags, "val_flags": val_flags, "eval_flags": eval_flags},
    ))
    checks.append(_check(
        "teacher_has_stock_exact_gt_labels",
        (teacher.get("counts") or {}).get("teacher_stock_closed", 0) > 0
        and (teacher.get("counts") or {}).get("teacher_exact_hit", 0) > 0
        and (teacher.get("counts") or {}).get("teacher_gt_reactant_hit", 0) > 0,
        {"teacher_counts": teacher.get("counts") or {}},
    ))

    controller_report = _load_json(distill_dir / "reservoir_distilled_controller.json")
    metadata = controller_report.get("metadata") or {}
    checks.append(_check(
        "controller_checkpoint_trained",
        (distill_dir / "reservoir_distilled_controller.pt").exists()
        and metadata.get("model_kind") == "reservoir_distilled_controller"
        and metadata.get("input_dim"),
        {
            "checkpoint": str(distill_dir / "reservoir_distilled_controller.pt"),
            "model_kind": metadata.get("model_kind"),
            "input_dim": metadata.get("input_dim"),
            "best_val_loss": controller_report.get("best_val_loss"),
        },
    ))

    acceptance_manifest = _load_json(acceptance_dir / "reservoir_acceptance_manifest.json")
    checks.append(_check(
        "full100_acceptance_manifest",
        bool(acceptance_manifest.get("commands")) and {"A", "B", "C", "D"}.issubset(set(acceptance_manifest.get("matrix") or {})),
        {
            "manifest": str(acceptance_dir / "reservoir_acceptance_manifest.json"),
            "matrix": acceptance_manifest.get("matrix") or {},
            "command_count": len(acceptance_manifest.get("commands") or []),
        },
    ))

    c_run = _load_json(acceptance_dir / "C" / "run.json")
    d_run = _load_json(acceptance_dir / "D" / "run.json")
    external_smoke = _load_json(distill_dir / "external_smokes" / "external_smoke_summary.json")
    checks.append(_check(
        "controller_smoke_runs",
        (len(c_run.get("targets") or []) >= 3 and len(d_run.get("targets") or []) >= 3)
        or bool(external_smoke.get("ready")),
        {
            "C_targets": len(c_run.get("targets") or []),
            "D_targets": len(d_run.get("targets") or []),
            "C_summary": _metric_subset(c_run.get("summary") or {}),
            "D_summary": _metric_subset(d_run.get("summary") or {}),
            "external_smoke_ready": external_smoke.get("ready"),
            "external_smoke_executed": external_smoke.get("executed") or [],
        },
    ))

    d_report = _load_json(acceptance_dir / "reports" / "D" / "reservoir_distill_report.json")
    d_source_report = _load_json(acceptance_dir / "reports" / "D" / "source_policy_report.json")
    d_summary = d_run.get("summary") or {}
    checks.append(_check(
        "reports_include_latency_and_gates",
        bool(d_report)
        and bool(d_source_report)
        and "avg_time_per_target_s" in d_summary
        and "route_tree_source_latency_ms" in d_summary,
        {
            "reservoir_report": str(acceptance_dir / "reports" / "D" / "reservoir_distill_report.json"),
            "source_policy_report": str(acceptance_dir / "reports" / "D" / "source_policy_report.json"),
            "avg_time_per_target_s": d_summary.get("avg_time_per_target_s"),
            "latency_breakdown": d_summary.get("route_tree_source_latency_ms"),
            "promotion_gates": (_load_json(acceptance_dir / "reservoir_acceptance_manifest.json").get("promotion_gates") or {}),
        },
    ))

    always_on = _load_json(distill_dir / "bounded_reservoir_smoke" / "D_always_on" / "run.json")
    broad = ((always_on.get("targets") or [{}])[0].get("planner_output") or {}).get("broad_reservoir") or {}
    checks.append(_check(
        "bounded_reservoir_records_broad_reservoir",
        int(broad.get("native_route_count") or 0) > 0,
        {"broad_reservoir": broad},
    ))

    external = _external_benchmark_evidence(distill_dir=distill_dir, acceptance_dir=acceptance_dir)
    checks.append(_check(
        "external_benchmarks_available",
        bool(external.get("ready")),
        external,
        required_for_completion=True,
    ))

    full_matrix = _full_matrix_status(acceptance_dir)
    checks.append(_check(
        "full100_A_D_runs_complete",
        full_matrix["complete"],
        full_matrix,
        required_for_completion=True,
    ))
    promotion = _promotion_gate_status(acceptance_dir)
    checks.append(_check(
        "full100_D_promotion_gates_met",
        promotion["promotable"],
        promotion,
        required_for_completion=True,
    ))

    passed = [row for row in checks if row["pass"]]
    failed = [row for row in checks if not row["pass"]]
    blocking = [row for row in failed if row.get("required_for_completion")]
    audit = {
        "schema_version": "reservoir_completion_audit.v1",
        "distill_dir": str(distill_dir),
        "acceptance_dir": str(acceptance_dir),
        "checks": checks,
        "passed": len(passed),
        "failed": len(failed),
        "blocking_incomplete": [row["name"] for row in blocking],
        "complete": not blocking and not failed,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_audit_markdown(audit), encoding="utf-8")
    return audit


def _check(name: str, ok: bool, evidence: dict[str, Any], *, required_for_completion: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(ok),
        "required_for_completion": bool(required_for_completion),
        "evidence": evidence,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def _eval_flag_counts(path: Path) -> dict[str, int]:
    out = {"eval_true": 0, "eval_false": 0}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("eval_only"):
            out["eval_true"] += 1
        else:
            out["eval_false"] += 1
    return out


def _metric_subset(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n_targets",
        "plan_rate",
        "strict_stock_solve_any",
        "candidate_gt_reactant_in_pool",
        "exact_reaction_in_route_pool",
        "gt_reactant_in_route_pool",
        "avg_time_per_target_s",
    ]
    return {key: summary.get(key) for key in keys}


def _full_matrix_status(acceptance_dir: Path) -> dict[str, Any]:
    labels = ["A", "B", "C", "D"]
    runs = {}
    for label in labels:
        path = acceptance_dir / label / "run.json"
        run = _load_json(path)
        runs[label] = {
            "path": str(path),
            "exists": path.exists(),
            "n_targets": len(run.get("targets") or []),
        }
    full100_complete = all(row["n_targets"] >= 100 for row in runs.values())
    report_path = acceptance_dir / "reports" / "reservoir_distill_matrix_manifest.json"
    return {
        "complete": full100_complete and report_path.exists(),
        "runs": runs,
        "matrix_report": str(report_path),
        "matrix_report_exists": report_path.exists(),
        "note": "3-target smoke reports do not satisfy the full100 A-D acceptance gate.",
    }


def _promotion_gate_status(acceptance_dir: Path) -> dict[str, Any]:
    manifest = _load_json(acceptance_dir / "reservoir_acceptance_manifest.json")
    gates = manifest.get("promotion_gates") or {}
    reference_diagnostics = manifest.get("reference_recall_diagnostics") or {}
    b_summary = _load_json(acceptance_dir / "B" / "run.json").get("summary") or {}
    c_summary = _load_json(acceptance_dir / "C" / "run.json").get("summary") or {}
    d_summary = _load_json(acceptance_dir / "D" / "run.json").get("summary") or {}
    d_append_summary = _load_json(acceptance_dir / "D_APPEND" / "run.json").get("summary") or {}

    metric_checks = {
        "plan_rate": _gte(d_summary.get("plan_rate"), gates.get("plan_rate")),
        "strict_stock_solve_any": _gte(d_summary.get("strict_stock_solve_any"), gates.get("strict_stock_solve_any")),
        "avg_time_per_target_s": _lte(
            d_summary.get("avg_time_per_target_s"),
            gates.get("avg_time_per_target_s_max"),
        ),
    }
    reference_recall_checks = {
        metric: _gte(d_summary.get(metric), reference_diagnostics.get(metric))
        for metric in (
            "candidate_gt_reactant_in_pool",
            "exact_reaction_in_route_pool",
            "gt_reactant_in_route_pool",
        )
        if reference_diagnostics.get(metric) is not None
    }
    reference_recall_vs_B = {
        "candidate_gt_vs_B": _gte(
            d_summary.get("candidate_gt_reactant_in_pool"),
            b_summary.get("candidate_gt_reactant_in_pool"),
        ),
        "exact_vs_B": _gte(
            d_summary.get("exact_reaction_in_route_pool"),
            b_summary.get("exact_reaction_in_route_pool"),
        ),
        "gt_vs_B": _gte(
            d_summary.get("gt_reactant_in_route_pool"),
            b_summary.get("gt_reactant_in_route_pool"),
        ),
    }
    return {
        "promotable": all(metric_checks.values()),
        "gates": gates,
        "reference_recall_diagnostics": reference_diagnostics,
        "D_summary": _metric_subset(d_summary),
        "C_summary": _metric_subset(c_summary),
        "B_summary": _metric_subset(b_summary),
        "metric_checks": metric_checks,
        "reference_recall_checks": reference_recall_checks,
        "reference_recall_vs_B": reference_recall_vs_B,
        "no_coverage_regression_vs_B": reference_recall_vs_B,
        "append_only_diagnostic": _append_only_promotion_diagnostic(
            summary=d_append_summary,
            baseline_summary=c_summary,
            teacher_summary=b_summary,
            gates=gates,
        ),
        "note": "D promotion uses plan, stock, runtime, and route-quality review. Exact/GT metrics are reference-route recall diagnostics and do not block usability promotion by themselves.",
    }


def _append_only_promotion_diagnostic(
    *,
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    teacher_summary: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    if not summary:
        return {
            "available": False,
            "online_promotable": False,
            "note": "D_APPEND run not present.",
        }
    effect_metrics = ["plan_rate", "strict_stock_solve_any"]
    reference_metrics = [
        "candidate_gt_reactant_in_pool",
        "exact_reaction_in_route_pool",
        "gt_reactant_in_route_pool",
    ]
    gate_key = {"avg_time_per_target_s": "avg_time_per_target_s_max"}
    metric_checks = {
        metric: _gte(summary.get(metric), gates.get(gate_key.get(metric, metric)))
        for metric in effect_metrics
    }
    no_regression_vs_c = {
        metric: _gte(summary.get(metric), baseline_summary.get(metric))
        for metric in effect_metrics
        if baseline_summary.get(metric) is not None
    }
    no_regression_vs_b = {
        metric: _gte(summary.get(metric), teacher_summary.get(metric))
        for metric in effect_metrics
        if teacher_summary.get(metric) is not None
    }
    reference_recall_vs_b = {
        metric: _gte(summary.get(metric), teacher_summary.get(metric))
        for metric in reference_metrics
        if teacher_summary.get(metric) is not None
    }
    gains_vs_c = {
        metric: (_delta(summary.get(metric), baseline_summary.get(metric)) or 0.0) > 0.0
        for metric in effect_metrics
        if baseline_summary.get(metric) is not None
    }
    avg_time_source = summary.get("avg_time_source")
    effect_gate_pass = bool(metric_checks) and all(metric_checks.values()) and all(no_regression_vs_c.values())
    return {
        "available": True,
        "summary": _metric_subset(summary),
        "avg_time_source": avg_time_source,
        "metric_checks_without_online_runtime": metric_checks,
        "no_regression_vs_C": no_regression_vs_c,
        "no_regression_vs_B": no_regression_vs_b,
        "reference_recall_vs_B": reference_recall_vs_b,
        "gains_vs_C": gains_vs_c,
        "effect_gate_pass": effect_gate_pass,
        "hybrid_append_only_candidate": effect_gate_pass and any(gains_vs_c.values()),
        "online_promotable": False,
        "note": "D_APPEND freezes C then appends native routes offline; it can support a hybrid architecture decision but is not an online runtime promotion gate.",
    }


def _gte(value: Any, threshold: Any) -> bool:
    if value is None or threshold is None:
        return False
    try:
        return float(value) >= float(threshold)
    except (TypeError, ValueError):
        return False


def _lte(value: Any, threshold: Any) -> bool:
    if value is None or threshold is None:
        return False
    try:
        return float(value) <= float(threshold)
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(value: Any, baseline: Any) -> float | None:
    left = _safe_float(value)
    right = _safe_float(baseline)
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _external_benchmark_evidence(*, distill_dir: Path, acceptance_dir: Path) -> dict[str, Any]:
    summaries = _external_smoke_summaries(distill_dir)
    complete_summaries = [
        row for row in summaries
        if {"paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"}.issubset(set(row.get("executed") or []))
    ]
    if complete_summaries:
        pure = _best_external_summary(row for row in complete_summaries if not row.get("native_stock_aligned"))
        aligned = _best_external_summary(row for row in complete_summaries if row.get("native_stock_aligned"))
        return {
            "schema_version": "reservoir_external_benchmark_evidence.v1",
            "source": "external_smoke_summary_scan",
            "ready": True,
            "mode": "external_smoke_summary_scan",
            "required": ["paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"],
            "summaries": complete_summaries,
            "pure_strict_summary": pure,
            "native_stock_aligned_summary": aligned,
            "adaptive_cd_comparison": _best_adaptive_cd_comparison(complete_summaries),
            "append_only_external_comparison": _best_append_only_external_comparison(complete_summaries),
        }
    external = _load_json(acceptance_dir / "external_benchmark_audit.json")
    return external


def _external_smoke_summaries(distill_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(distill_dir).glob("external*/external_smoke_summary.json")):
        summary = _load_json(path)
        if not summary:
            continue
        rows = summary.get("rows") or []
        executed = _external_summary_dataset_keys(summary, rows, key="executed")
        required = _external_summary_dataset_keys(summary, rows, key="required")
        row_stock = {
            str(row.get("label")): {
                "stock": row.get("strict_stock_solve_any"),
                "avg_time_per_target_s": row.get("avg_time_per_target_s"),
                "broad_reservoir_runtime_stock_routes": row.get("broad_reservoir_runtime_stock_routes"),
                "native_payload_runtime_stock_routes": row.get("native_payload_runtime_stock_routes"),
            }
            for row in rows
        }
        parent_name = path.parent.name
        out.append({
            "path": str(path),
            "name": parent_name,
            "ready": bool(summary.get("ready")),
            "executed": executed,
            "required": required,
            "native_stock_aligned": "native_stock" in parent_name or _manifest_trusts_native_stock(path.parent),
            "row_stock": row_stock,
            "paired_config_deltas": summary.get("paired_config_deltas") or [],
        })
    for path in sorted(Path(distill_dir).glob("external*/external_smoke_summary_adaptive_cd.json")):
        summary = _load_json(path)
        if not summary:
            continue
        rows = summary.get("rows") or []
        executed = sorted({_external_dataset_key(row.get("dataset")) for row in rows if row.get("dataset")})
        row_stock = {
            f"{row.get('config')}:{row.get('dataset')}": {
                "stock": row.get("strict_stock_solve_any"),
                "avg_time_per_target_s": row.get("avg_time_per_target_s"),
                "candidate_gt_reactant_in_pool": row.get("candidate_gt_reactant_in_pool"),
                "exact_reaction_in_route_pool": row.get("exact_reaction_in_route_pool"),
                "gt_reactant_in_route_pool": row.get("gt_reactant_in_route_pool"),
            }
            for row in rows
        }
        parent_name = path.parent.name
        out.append({
            "path": str(path),
            "name": parent_name + ":adaptive_cd",
            "ready": {"paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"}.issubset(set(executed)),
            "executed": executed,
            "required": ["paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like"],
            "native_stock_aligned": "native_stock" in parent_name or _manifest_trusts_native_stock(path.parent),
            "row_stock": row_stock,
            "aggregates": summary.get("aggregates") or [],
            "paired_config_deltas": summary.get("paired_config_deltas") or [],
        })
    return out


def _best_external_summary(rows) -> dict[str, Any] | None:
    rows = list(rows)
    if not rows:
        return None
    return max(rows, key=_external_summary_quality)


def _external_summary_quality(row: dict[str, Any]) -> tuple[int, int, int, str]:
    row_stock = row.get("row_stock") or {}
    known_stock = sum(1 for item in row_stock.values() if item.get("stock") is not None)
    prebuilt = int("prebuild" in str(row.get("name") or ""))
    ready = int(bool(row.get("ready")))
    return known_stock, prebuilt, ready, str(row.get("name") or "")


def _best_adaptive_cd_comparison(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    comparisons = [
        comparison
        for summary in summaries
        for comparison in [_adaptive_cd_comparison(summary)]
        if comparison is not None
    ]
    if not comparisons:
        return None
    return max(comparisons, key=lambda row: (int(bool(row.get("effect_first_pass"))), row.get("effect_gain_count") or 0, row.get("runtime_margin_s") or -999.0))


def _best_append_only_external_comparison(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    comparisons = [
        comparison
        for summary in summaries
        for comparison in [_append_only_external_comparison(summary)]
        if comparison is not None
    ]
    if not comparisons:
        return None
    return max(
        comparisons,
        key=lambda row: (
            int(bool(row.get("effect_isolation_pass"))),
            row.get("effect_gain_count") or 0,
            -(row.get("coverage_loss_count") or 0),
            str(row.get("source_summary") or ""),
        ),
    )


def _append_only_external_comparison(summary: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        row
        for row in summary.get("paired_config_deltas") or []
        if str(row.get("candidate_config") or "") == "D_APPEND"
    ]
    if not rows:
        return None
    metrics = [
        "strict_stock_solve_any",
        "candidate_gt_reactant_in_pool",
        "candidate_exact_reaction_in_pool",
        "exact_reaction_in_route_pool",
        "gt_reactant_in_route_pool",
    ]
    gains = []
    losses = []
    by_dataset = {}
    for row in rows:
        deltas = row.get("metric_deltas") or {}
        dataset = str(row.get("dataset_label") or "")
        by_dataset[dataset] = {
            "coverage_gains": row.get("coverage_gains") or [],
            "coverage_losses": row.get("coverage_losses") or [],
            "likely_change_cause": row.get("likely_change_cause"),
            "broad_routes": row.get("candidate_broad_reservoir_routes"),
        }
        for metric in metrics:
            delta = _safe_float(deltas.get(metric))
            if delta is None:
                continue
            if delta > 0:
                gains.append((dataset, metric))
            elif delta < 0:
                losses.append((dataset, metric))
    return {
        "source_summary": summary.get("path"),
        "datasets": sorted(by_dataset),
        "by_dataset": by_dataset,
        "effect_gain_count": len(gains),
        "coverage_loss_count": len(losses),
        "effect_gains": [{"dataset": dataset, "metric": metric} for dataset, metric in gains],
        "coverage_losses": [{"dataset": dataset, "metric": metric} for dataset, metric in losses],
        "effect_isolation_pass": bool(gains) and not losses,
        "online_runtime_measured": False,
        "note": "D_APPEND paired deltas freeze C and append native routes offline; this isolates effect but does not measure online runtime.",
    }


def _adaptive_cd_comparison(summary: dict[str, Any], *, runtime_gate_s: float = 30.0) -> dict[str, Any] | None:
    aggregates = summary.get("aggregates") or []
    if not aggregates:
        return None
    c_row = _find_aggregate_config(aggregates, "C")
    d_row = _find_aggregate_config(aggregates, "D")
    if not c_row or not d_row:
        return None
    metrics = [
        "strict_stock_solve_any",
        "candidate_gt_reactant_in_pool",
        "exact_reaction_in_route_pool",
        "gt_reactant_in_route_pool",
    ]
    deltas = {metric: _delta(d_row.get(metric), c_row.get(metric)) for metric in metrics}
    no_regression = {
        metric: _gte(d_row.get(metric), c_row.get(metric))
        for metric in metrics
    }
    effect_gain = {
        metric: (deltas.get(metric) is not None and float(deltas[metric]) > 0.0)
        for metric in metrics
    }
    runtime_value = _safe_float(d_row.get("avg_time_per_target_s"))
    runtime_pass = runtime_value is not None and runtime_value <= float(runtime_gate_s)
    return {
        "source_summary": summary.get("path"),
        "runtime_gate_s": float(runtime_gate_s),
        "C": c_row,
        "D": d_row,
        "deltas": deltas,
        "no_regression_vs_C": no_regression,
        "effect_gain": effect_gain,
        "effect_gain_count": sum(1 for value in effect_gain.values() if value),
        "runtime_pass": runtime_pass,
        "runtime_margin_s": round(float(runtime_gate_s) - runtime_value, 3) if runtime_value is not None else None,
        "effect_first_pass": runtime_pass and all(no_regression.values()) and any(effect_gain.values()),
    }


def _find_aggregate_config(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    prefix = str(prefix or "").lower()
    for row in rows:
        config = str(row.get("config") or "").strip().lower()
        if config == prefix or config.startswith(prefix + " "):
            return row
    return None


def _external_dataset_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    if "paroutes" in text and "n1" in text:
        return "paroutes_n1"
    if "paroutes" in text and "n5" in text:
        return "paroutes_n5"
    if "uspto" in text and "190" in text:
        return "uspto_190"
    if "bionavi" in text or "bio_navi" in text:
        return "bionavi_like"
    return text


def _external_summary_dataset_keys(summary: dict[str, Any], rows: list[dict[str, Any]], *, key: str) -> list[str]:
    values = list(summary.get(key) or [])
    if not values:
        values = [
            row.get("dataset_label") or row.get("dataset") or row.get("label")
            for row in rows
        ]
    normalized = sorted({
        item
        for item in (_external_dataset_key(value) for value in values)
        if item
    })
    return normalized


def _manifest_trusts_native_stock(root: Path) -> bool:
    manifest = _load_json(root / "external_smoke_manifest.json")
    if manifest.get("trust_native_stock"):
        return True
    for command in manifest.get("commands") or []:
        if "AUTOPLANNER_RESERVOIR_TRUST_NATIVE_STOCK=1" in str(command.get("cmd") or ""):
            return True
    return False


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Reservoir Completion Audit",
        "",
        f"Distill dir: `{audit['distill_dir']}`",
        f"Acceptance dir: `{audit['acceptance_dir']}`",
        "",
        "| Check | Pass | Blocking | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in audit["checks"]:
        evidence = row.get("evidence") or {}
        compact = json.dumps(evidence, ensure_ascii=False)
        if len(compact) > 220:
            compact = compact[:217] + "..."
        lines.append(
            f"| `{row['name']}` | {row['pass']} | {row.get('required_for_completion', False)} | `{compact}` |"
        )
    lines.extend([
        "",
        f"Complete: **{audit['complete']}**",
        f"Blocking incomplete: {', '.join(audit['blocking_incomplete']) or 'none'}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit reservoir-distilled completion state")
    ap.add_argument("--distill-dir", default="results/shared/reservoir_distill_20260513")
    ap.add_argument("--acceptance-dir", default="results/shared/reservoir_distill_20260513/full100_acceptance")
    ap.add_argument("--output-json", default="results/shared/reservoir_distill_20260513/completion_audit.json")
    ap.add_argument("--output-md", default="results/shared/reservoir_distill_20260513/completion_audit.md")
    args = ap.parse_args()
    audit = build_completion_audit(
        distill_dir=Path(args.distill_dir),
        acceptance_dir=Path(args.acceptance_dir),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
    )
    print(json.dumps({"complete": audit["complete"], "blocking_incomplete": audit["blocking_incomplete"]}, indent=2))


if __name__ == "__main__":
    main()
