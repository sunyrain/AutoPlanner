"""Publication-readiness audit for reservoir-distilled controller results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMARY_METRICS = (
    "plan_rate",
    "strict_stock_solve_any",
    "candidate_gt_reactant_in_pool",
    "candidate_exact_reaction_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
    "avg_time_per_target_s",
    "avg_route_count",
)
PROMOTION_METRICS = (
    "plan_rate",
    "strict_stock_solve_any",
    "avg_time_per_target_s",
)
REFERENCE_RECALL_METRICS = (
    "candidate_gt_reactant_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
)
REQUIRED_EXTERNAL_DATASETS = ("paroutes_n1", "paroutes_n5", "uspto_190", "bionavi_like")


def build_publication_readiness_report(
    *,
    distill_dir: Path,
    acceptance_dir: Path,
    output_json: Path,
    output_md: Path,
    external_min_targets: int = 10,
) -> dict[str, Any]:
    distill_dir = Path(distill_dir)
    acceptance_dir = Path(acceptance_dir)
    completion = _load_json(acceptance_dir / "completion_audit.json")
    manifest = _load_json(acceptance_dir / "reservoir_acceptance_manifest.json")
    gates = manifest.get("promotion_gates") or {}
    runs = {
        label: _run_summary(acceptance_dir / label / "run.json")
        for label in ("A", "B", "C", "D", "D_APPEND")
    }
    external = _external_summary_status(distill_dir, min_targets=external_min_targets)
    internal = _internal_status(completion=completion, gates=gates, runs=runs)
    statistical = _statistical_evidence_status(distill_dir)
    criteria = _publication_criteria(internal=internal, external=external, acceptance_dir=acceptance_dir, statistical=statistical)
    claims = _claim_guidance(criteria=criteria, internal=internal)
    report = {
        "schema_version": "reservoir_publication_readiness.v1",
        "distill_dir": str(distill_dir),
        "acceptance_dir": str(acceptance_dir),
        "external_min_targets": int(external_min_targets),
        "internal": internal,
        "external": external,
        "statistical": statistical,
        "criteria": criteria,
        "claims": claims,
        "source_artifacts": {
            "completion_audit": str(acceptance_dir / "completion_audit.json"),
            "acceptance_manifest": str(acceptance_dir / "reservoir_acceptance_manifest.json"),
            "matrix_comparison": str(acceptance_dir / "reports" / "comparison.md"),
            "statistical_report": statistical.get("path"),
        },
        "next_steps": _next_steps(criteria),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _run_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary") or {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "n_targets": len(payload.get("targets") or []),
        "summary": {metric: summary.get(metric) for metric in PRIMARY_METRICS},
        "avg_time_source": summary.get("avg_time_source"),
    }


def _internal_status(*, completion: dict[str, Any], gates: dict[str, Any], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    d_summary = (runs.get("D") or {}).get("summary") or {}
    c_summary = (runs.get("C") or {}).get("summary") or {}
    b_summary = (runs.get("B") or {}).get("summary") or {}
    d_append_summary = (runs.get("D_APPEND") or {}).get("summary") or {}
    d_checks = _gate_checks(d_summary, gates)
    c_checks = _gate_checks(c_summary, gates)
    reference_recall_vs_b = {
        metric: _gte(d_summary.get(metric), b_summary.get(metric))
        for metric in REFERENCE_RECALL_METRICS
        if b_summary.get(metric) is not None
    }
    return {
        "completion_audit_complete": bool(completion.get("complete")),
        "completion_blocking": completion.get("blocking_incomplete") or [],
        "runs": runs,
        "gates": gates,
        "D_gate_checks": d_checks,
        "D_promotable_online": bool(d_checks) and all(d_checks.values()),
        "D_reference_recall_vs_B": reference_recall_vs_b,
        "D_no_regression_vs_B": reference_recall_vs_b,
        "C_student_only_gate_checks": c_checks,
        "C_student_only_promotable": bool(c_checks) and all(c_checks.values()),
        "D_APPEND_available": bool((runs.get("D_APPEND") or {}).get("exists")),
        "D_APPEND_effect_diagnostic": _append_diagnostic(d_append_summary=d_append_summary, c_summary=c_summary, b_summary=b_summary),
    }


def _gate_checks(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    if not summary or not gates:
        return {}
    return {
        "plan_rate": _gte(summary.get("plan_rate"), gates.get("plan_rate")),
        "strict_stock_solve_any": _gte(summary.get("strict_stock_solve_any"), gates.get("strict_stock_solve_any")),
        "avg_time_per_target_s": _lte(summary.get("avg_time_per_target_s"), gates.get("avg_time_per_target_s_max")),
    }


def _append_diagnostic(*, d_append_summary: dict[str, Any], c_summary: dict[str, Any], b_summary: dict[str, Any]) -> dict[str, Any]:
    if not d_append_summary:
        return {"available": False, "hybrid_effect_gain": False}
    metrics = ("plan_rate", "strict_stock_solve_any")
    gains_vs_c = {
        metric: (_delta(d_append_summary.get(metric), c_summary.get(metric)) or 0.0) > 0.0
        for metric in metrics
        if c_summary.get(metric) is not None
    }
    no_regression_vs_b = {
        metric: _gte(d_append_summary.get(metric), b_summary.get(metric))
        for metric in metrics
        if b_summary.get(metric) is not None
    }
    reference_recall_vs_b = {
        metric: _gte(d_append_summary.get(metric), b_summary.get(metric))
        for metric in REFERENCE_RECALL_METRICS
        if b_summary.get(metric) is not None
    }
    return {
        "available": True,
        "summary": {metric: d_append_summary.get(metric) for metric in PRIMARY_METRICS},
        "avg_time_source": d_append_summary.get("avg_time_source"),
        "gains_vs_C": gains_vs_c,
        "no_regression_vs_B": no_regression_vs_b,
        "reference_recall_vs_B": reference_recall_vs_b,
        "hybrid_effect_gain": any(gains_vs_c.values()) and all(no_regression_vs_b.values()),
        "online_promotable": False,
    }


def _external_summary_status(distill_dir: Path, *, min_targets: int) -> dict[str, Any]:
    summaries = []
    for path in sorted(Path(distill_dir).glob("external*/external_smoke_summary.json")):
        payload = _load_json(path)
        if not payload:
            continue
        rows = payload.get("rows") or []
        dataset_counts: dict[str, int] = {}
        for row in rows:
            dataset = _dataset_key(row.get("dataset_label") or row.get("dataset") or row.get("label"))
            if dataset not in REQUIRED_EXTERNAL_DATASETS:
                continue
            try:
                count = int(row.get("n_run_targets") or 0)
            except (TypeError, ValueError):
                count = 0
            dataset_counts[dataset] = max(dataset_counts.get(dataset, 0), count)
        complete_required = set(REQUIRED_EXTERNAL_DATASETS).issubset(set(dataset_counts))
        scale_ready = complete_required and all(dataset_counts.get(name, 0) >= min_targets for name in REQUIRED_EXTERNAL_DATASETS)
        summaries.append(
            {
                "path": str(path),
                "ready": bool(payload.get("ready")),
                "dataset_counts": dataset_counts,
                "complete_required": complete_required,
                "scale_ready": scale_ready,
                "paired_config_deltas": payload.get("paired_config_deltas") or [],
            }
        )
    best_scale = max(summaries, key=lambda row: (int(row["scale_ready"]), len(row["dataset_counts"]), min(row["dataset_counts"].values() or [0])), default=None)
    any_smoke_ready = any(row.get("ready") and row.get("complete_required") for row in summaries)
    return {
        "required": list(REQUIRED_EXTERNAL_DATASETS),
        "min_targets_per_dataset_for_publication_audit": int(min_targets),
        "smoke_ready": any_smoke_ready,
        "scale_ready": bool(best_scale and best_scale.get("scale_ready")),
        "best_summary": best_scale,
        "summaries_scanned": summaries,
    }


def _publication_criteria(*, internal: dict[str, Any], external: dict[str, Any], acceptance_dir: Path, statistical: dict[str, Any]) -> dict[str, Any]:
    matrix_exists = (acceptance_dir / "reports" / "comparison.md").exists()
    statistical_repeats = bool(statistical.get("ready"))
    criteria = {
        "internal_full100_promoted": bool(internal.get("completion_audit_complete") and internal.get("D_promotable_online")),
        "ablation_matrix_complete": all((internal.get("runs") or {}).get(label, {}).get("exists") for label in ("A", "B", "C", "D", "D_APPEND")) and matrix_exists,
        "student_only_claim_supported": bool(internal.get("C_student_only_promotable")),
        "hybrid_claim_supported": bool(internal.get("D_promotable_online")),
        "append_only_effect_supported": bool((internal.get("D_APPEND_effect_diagnostic") or {}).get("hybrid_effect_gain")),
        "external_smoke_available": bool(external.get("smoke_ready")),
        "external_scale_sufficient": bool(external.get("scale_ready")),
        "statistical_repeats_available": statistical_repeats,
    }
    criteria["internal_or_technical_report_ready"] = (
        criteria["internal_full100_promoted"]
        and criteria["ablation_matrix_complete"]
        and criteria["external_smoke_available"]
    )
    criteria["limited_preprint_ready"] = (
        criteria["internal_or_technical_report_ready"]
        and criteria["hybrid_claim_supported"]
        and not criteria["student_only_claim_supported"]
    )
    criteria["publication_ready_strict"] = (
        criteria["internal_full100_promoted"]
        and criteria["ablation_matrix_complete"]
        and criteria["hybrid_claim_supported"]
        and criteria["external_scale_sufficient"]
        and criteria["statistical_repeats_available"]
    )
    return criteria


def _statistical_evidence_status(distill_dir: Path) -> dict[str, Any]:
    candidates = []
    for path in sorted(Path(distill_dir).glob("reservoir_statistical_report*.json")):
        payload = _load_json(path)
        if not payload:
            continue
        coverage = payload.get("coverage") or {}
        candidates.append(
            {
                "path": str(path),
                "ready": bool(payload.get("ready")),
                "bootstrap_iterations": int(payload.get("bootstrap_iterations") or 0),
                "coverage": coverage,
            }
        )
    if not candidates:
        return {"ready": False, "path": None, "candidates": []}
    best = max(
        candidates,
        key=lambda row: (
            int(bool(row.get("ready"))),
            int(row.get("bootstrap_iterations") or 0),
            int(bool((row.get("coverage") or {}).get("full100"))),
            int(bool((row.get("coverage") or {}).get("external"))),
        ),
    )
    return {**best, "candidates": candidates}


def _claim_guidance(*, criteria: dict[str, Any], internal: dict[str, Any]) -> dict[str, Any]:
    allowed = []
    blocked = []
    if criteria.get("hybrid_claim_supported"):
        allowed.append("Online D hybrid controller meets full100 promotion gates.")
        allowed.append("Reference-route exact/GT recall is reported as a diagnostic; non-GT stock-closed routes require route-quality review before usability claims.")
    if criteria.get("append_only_effect_supported"):
        allowed.append("D_APPEND is an offline effect-isolation diagnostic showing append-only native safety routes can improve stock without coverage loss.")
    if not criteria.get("student_only_claim_supported"):
        blocked.append("Do not claim student-only distillation is sufficient or promoted.")
    if not criteria.get("external_scale_sufficient"):
        blocked.append("Do not claim broad external benchmark generalization beyond smoke evidence.")
    if not criteria.get("statistical_repeats_available"):
        blocked.append("Do not claim statistical robustness or significance yet.")
    return {
        "allowed_claims": allowed,
        "blocked_or_needs_more_evidence": blocked,
        "recommended_framing": (
            "Hybrid controller/ranker distillation with bounded native reservoir fallback; "
            "not a monolithic reaction generator and not a pure student-only replacement."
        ),
        "student_only_promotable": internal.get("C_student_only_promotable"),
    }


def _next_steps(criteria: dict[str, Any]) -> list[dict[str, str]]:
    steps = []
    if not criteria.get("external_scale_sufficient"):
        steps.append({
            "priority": "P0",
            "step": "Run scaled external C/D/D_APPEND benchmarks.",
            "reason": "Current external evidence is smoke-scale; strict publication readiness needs larger PaRoutes n1/n5, USPTO-190, and BioNavi-like coverage.",
        })
    if not criteria.get("statistical_repeats_available"):
        steps.append({
            "priority": "P0",
            "step": "Add repeated runs or bootstrap confidence intervals for full100 and external metrics.",
            "reason": "Publication claims need stability estimates, not only one deterministic run.",
        })
    if not criteria.get("student_only_claim_supported"):
        steps.append({
            "priority": "P1",
            "step": "Keep student-only claims scoped; improve provider/proposal coverage before training more rerank losses.",
            "reason": "C misses stock/exact/GT gates, so the student alone is not the promoted system.",
        })
    steps.append({
        "priority": "P1",
        "step": "Prepare paper tables around A/B/C/D/D_APPEND and explicitly label D_APPEND as offline diagnostic.",
        "reason": "The current strongest evidence supports a hybrid system claim.",
    })
    return steps


def _markdown(report: dict[str, Any]) -> str:
    internal = report.get("internal") or {}
    external = report.get("external") or {}
    statistical = report.get("statistical") or {}
    criteria = report.get("criteria") or {}
    claims = report.get("claims") or {}
    runs = internal.get("runs") or {}
    lines = [
        "# Reservoir Publication Readiness",
        "",
        f"Acceptance dir: `{report['acceptance_dir']}`",
        f"Distill dir: `{report['distill_dir']}`",
        "",
        "## Verdict",
        "",
        f"- Internal / technical-report ready: `{criteria.get('internal_or_technical_report_ready')}`",
        f"- Limited preprint ready: `{criteria.get('limited_preprint_ready')}`",
        f"- Strict publication ready: `{criteria.get('publication_ready_strict')}`",
        "",
        "## Full100 Matrix",
        "",
        "| Config | Targets | Plan | Stock | Cand GT | Cand Exact | Exact | Route GT | Avg s | Avg routes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ("A", "B", "C", "D", "D_APPEND"):
        run = runs.get(label) or {}
        summary = run.get("summary") or {}
        lines.append(
            "| {label} | {targets} | {plan} | {stock} | {cand_gt} | {cand_exact} | {exact} | {gt} | {time} | {routes} |".format(
                label=label,
                targets=run.get("n_targets"),
                plan=_fmt(summary.get("plan_rate")),
                stock=_fmt(summary.get("strict_stock_solve_any")),
                cand_gt=_fmt(summary.get("candidate_gt_reactant_in_pool")),
                cand_exact=_fmt(summary.get("candidate_exact_reaction_in_pool")),
                exact=_fmt(summary.get("exact_reaction_in_route_pool")),
                gt=_fmt(summary.get("gt_reactant_in_route_pool")),
                time=_fmt(summary.get("avg_time_per_target_s")),
                routes=_fmt(summary.get("avg_route_count")),
            )
        )
    best_external = external.get("best_summary") or {}
    if best_external:
        lines.extend(
            [
                "",
                "## External Summary",
                "",
                f"Best summary: `{best_external.get('path')}`",
                "",
                "| Dataset | Targets |",
                "| --- | ---: |",
            ]
        )
        for dataset, count in sorted((best_external.get("dataset_counts") or {}).items()):
            lines.append(f"| `{dataset}` | {count} |")
        lines.extend(
            [
                "",
                "| Dataset | Candidate | Delta stock | Delta exact | Delta route GT | Delta avg s |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for delta in best_external.get("paired_config_deltas") or []:
            metrics = delta.get("metric_deltas") or {}
            lines.append(
                "| {dataset} | {candidate} | {stock} | {exact} | {gt} | {time} |".format(
                    dataset=delta.get("dataset_label"),
                    candidate=delta.get("candidate_config"),
                    stock=_fmt(metrics.get("strict_stock_solve_any")),
                    exact=_fmt(metrics.get("exact_reaction_in_route_pool")),
                    gt=_fmt(metrics.get("gt_reactant_in_route_pool")),
                    time=_fmt(metrics.get("avg_time_per_target_s")),
                )
            )
    lines.extend(
        [
            "",
            "## Statistical Evidence",
            "",
            f"Best report: `{statistical.get('path')}`",
            f"Ready: `{statistical.get('ready')}`",
            f"Bootstrap iterations: `{statistical.get('bootstrap_iterations')}`",
            "",
            "| Scope | Pass |",
            "| --- | ---: |",
        ]
    )
    for key, value in (statistical.get("coverage") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Criteria", "", "| Criterion | Pass |", "| --- | ---: |"])
    for key, value in criteria.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Claim Guidance", "", "Allowed claims:"])
    for claim in claims.get("allowed_claims") or []:
        lines.append(f"- {claim}")
    lines.append("")
    lines.append("Blocked or needs more evidence:")
    for claim in claims.get("blocked_or_needs_more_evidence") or []:
        lines.append(f"- {claim}")
    lines.extend(["", "## Next Steps", ""])
    for step in report.get("next_steps") or []:
        lines.append(f"- `{step.get('priority')}` {step.get('step')} {step.get('reason')}")
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gte(value: Any, threshold: Any) -> bool:
    left = _safe_float(value)
    right = _safe_float(threshold)
    return left is not None and right is not None and left >= right


def _lte(value: Any, threshold: Any) -> bool:
    left = _safe_float(value)
    right = _safe_float(threshold)
    return left is not None and right is not None and left <= right


def _delta(value: Any, baseline: Any) -> float | None:
    left = _safe_float(value)
    right = _safe_float(baseline)
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _dataset_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "paroutes" in text and "n1" in text:
        return "paroutes_n1"
    if "paroutes" in text and "n5" in text:
        return "paroutes_n5"
    if "uspto" in text and "190" in text:
        return "uspto_190"
    if "bionavi" in text or "bio_navi" in text:
        return "bionavi_like"
    return text


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit publication readiness for reservoir-distilled controller results")
    ap.add_argument("--distill-dir", default="results/shared/reservoir_distill_20260513")
    ap.add_argument("--acceptance-dir", default="results/shared/reservoir_distill_20260513/full100_acceptance_real_v2")
    ap.add_argument("--output-json", default="results/shared/reservoir_distill_20260513/publication_readiness_20260514.json")
    ap.add_argument("--output-md", default="results/shared/reservoir_distill_20260513/publication_readiness_20260514.md")
    ap.add_argument("--external-min-targets", type=int, default=10)
    args = ap.parse_args()
    report = build_publication_readiness_report(
        distill_dir=Path(args.distill_dir),
        acceptance_dir=Path(args.acceptance_dir),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        external_min_targets=args.external_min_targets,
    )
    print(json.dumps({
        "internal_or_technical_report_ready": report["criteria"]["internal_or_technical_report_ready"],
        "limited_preprint_ready": report["criteria"]["limited_preprint_ready"],
        "publication_ready_strict": report["criteria"]["publication_ready_strict"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()
