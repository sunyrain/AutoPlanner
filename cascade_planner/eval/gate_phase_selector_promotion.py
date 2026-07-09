"""Gate selector promotion for the current ChemEnzy+CCTS phase.

This gate is intentionally conservative.  It distinguishes:

* product default: a selector that improves over product-audit rule-post without
  increasing artifact/trivial/generic route rates;
* research evidence: strict fixed-pool controls showing cascade-context signal;
* no-promote branches: learned post-rerank variants that do not beat rule-post.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "phase_selector_promotion_gate.v1"
METRIC_KEYS = (
    "top1_product_usable_rate",
    "top3_product_usable_rate",
    "top3_artifact_rate",
    "top3_trivial_stock_closure_rate",
    "top3_generic_route_rate",
)


def gate_phase_selector_promotion(
    *,
    plausibility_v10_summary: Path,
    context_control_summary: Path,
    runtime_ccts_comparison: Path,
    cascade_only_comparison: Path,
    output_json: Path,
    output_md: Path | None = None,
    risk_guarded_comparison: Path | None = None,
    risk_guarded_takeaway: Path | None = None,
    min_context_mrr_delta: float = 0.05,
    min_product_usable_gain: float = 0.01,
    max_quality_regression: float = 1e-9,
) -> dict[str, Any]:
    plausibility = _read_json(plausibility_v10_summary)
    context = _read_json(context_control_summary)
    runtime = _read_json(runtime_ccts_comparison)
    cascade = _read_json(cascade_only_comparison)
    risk_guarded = _read_json(risk_guarded_comparison) if risk_guarded_comparison else {}
    risk_takeaway = _read_json(risk_guarded_takeaway) if risk_guarded_takeaway else {}
    context_gate = _context_gate(context, min_delta=min_context_mrr_delta)
    candidate_gates = [
        _candidate_gate(
            "v10_audit_guarded",
            _rows_from_plausibility_v10(plausibility),
            candidate_prefix="v10",
            baseline_prefix="rule",
            min_product_usable_gain=min_product_usable_gain,
            max_quality_regression=max_quality_regression,
        ),
        _candidate_gate(
            "runtime_ccts_audit_guarded",
            runtime.get("rows") or [],
            candidate_prefix="audit_ccts",
            baseline_prefix="rule",
            min_product_usable_gain=min_product_usable_gain,
            max_quality_regression=max_quality_regression,
        ),
        _candidate_gate(
            "cascade_only_audit_guarded",
            cascade.get("rows") or [],
            candidate_prefix="cascade_only",
            baseline_prefix="rule",
            min_product_usable_gain=min_product_usable_gain,
            max_quality_regression=max_quality_regression,
        ),
    ]
    risk_guarded_rows = _risk_guarded_rows(risk_guarded)
    if risk_guarded_rows:
        candidate_gates.extend(
            [
                _candidate_gate(
                    "runtime_ccts_risk_guarded",
                    risk_guarded_rows,
                    candidate_prefix="runtime_ccts",
                    baseline_prefix="rule",
                    min_product_usable_gain=min_product_usable_gain,
                    max_quality_regression=max_quality_regression,
                ),
                _candidate_gate(
                    "cascade_only_risk_guarded",
                    [row for row in risk_guarded_rows if any(key.startswith("cascade_only_") for key in row)],
                    candidate_prefix="cascade_only",
                    baseline_prefix="rule",
                    min_product_usable_gain=min_product_usable_gain,
                    max_quality_regression=max_quality_regression,
                ),
            ]
        )
    promotable = [row for row in candidate_gates if row["promote_product_default"]]
    risk_guarded_summary = _risk_guarded_summary(risk_guarded, risk_takeaway)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "plausibility_v10_summary": str(plausibility_v10_summary),
            "context_control_summary": str(context_control_summary),
            "runtime_ccts_comparison": str(runtime_ccts_comparison),
            "cascade_only_comparison": str(cascade_only_comparison),
            "risk_guarded_comparison": str(risk_guarded_comparison) if risk_guarded_comparison else None,
            "risk_guarded_takeaway": str(risk_guarded_takeaway) if risk_guarded_takeaway else None,
            "output_json": str(output_json),
            "output_md": str(output_md or output_json.with_suffix(".md")),
            "thresholds": {
                "min_context_mrr_delta": float(min_context_mrr_delta),
                "min_product_usable_gain": float(min_product_usable_gain),
                "max_quality_regression": float(max_quality_regression),
            },
        },
        "product_default": {
            "selector": "product_audit_risk_guarded_rule_post",
            "reason": (
                "no learned CCTS post-rerank candidate beats risk-guarded rule-post under product gates"
                if not promotable
                else "one or more candidates passed product gates; inspect candidate_gates"
            ),
        },
        "risk_guarded_selector": risk_guarded_summary,
        "research_claim": {
            "cascade_context_fixed_pool_signal": bool(context_gate["passed"]),
            "scope": (
                "strict fixed ChemEnzy route-pool ranking only"
                if context_gate["passed"]
                else "not supported by current controls"
            ),
            "context_gate": context_gate,
        },
        "candidate_gates": candidate_gates,
        "promoted_candidates": [row["candidate"] for row in promotable],
        "decision": {
            "promote_product_default": bool(promotable),
            "continue_post_rerank_tie_breakers": False,
            "next_direction": (
                "search-time cascade integration or stronger route-quality supervision; "
                "risk-guarded post-rerank is now a safety default, but do not keep "
                "optimizing learned post-rerank tie-breakers without new labels"
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _context_gate(context: dict[str, Any], *, min_delta: float) -> dict[str, Any]:
    diagnostics = context.get("diagnostics") or {}
    deltas = {
        "original_minus_feature_shuffle_mrr": _float(diagnostics.get("cascade_original_minus_feature_shuffle_mrr")),
        "original_minus_label_shuffle_mrr": _float(diagnostics.get("cascade_original_minus_label_shuffle_mrr")),
        "original_minus_native_rank_mrr": _float(diagnostics.get("cascade_original_minus_native_rank_mrr")),
    }
    checks = [
        {
            "name": key,
            "ok": value >= float(min_delta),
            "actual": round(value, 6),
            "required_min": float(min_delta),
        }
        for key, value in deltas.items()
    ]
    return {
        "passed": all(row["ok"] for row in checks),
        "checks": checks,
        "source_interpretation": diagnostics.get("interpretation"),
    }


def _candidate_gate(
    candidate: str,
    rows: list[dict[str, Any]],
    *,
    candidate_prefix: str,
    baseline_prefix: str,
    min_product_usable_gain: float,
    max_quality_regression: float,
) -> dict[str, Any]:
    checks = []
    row_summaries = []
    for row in rows:
        dataset = str(row.get("dataset") or "unknown")
        baseline = _extract_metrics(row, baseline_prefix)
        candidate_metrics = _extract_metrics(row, candidate_prefix)
        usable_gain = max(
            candidate_metrics["top1_product_usable_rate"] - baseline["top1_product_usable_rate"],
            candidate_metrics["top3_product_usable_rate"] - baseline["top3_product_usable_rate"],
        )
        quality_regressions = {
            "top3_artifact_rate": candidate_metrics["top3_artifact_rate"] - baseline["top3_artifact_rate"],
            "top3_trivial_stock_closure_rate": (
                candidate_metrics["top3_trivial_stock_closure_rate"]
                - baseline["top3_trivial_stock_closure_rate"]
            ),
            "top3_generic_route_rate": candidate_metrics["top3_generic_route_rate"] - baseline["top3_generic_route_rate"],
        }
        dataset_checks = [
            {
                "name": "usable_gain",
                "ok": usable_gain >= float(min_product_usable_gain),
                "actual": round(float(usable_gain), 6),
                "required_min": float(min_product_usable_gain),
            },
            *[
                {
                    "name": f"no_regression_{key}",
                    "ok": value <= float(max_quality_regression),
                    "actual": round(float(value), 6),
                    "required_max": float(max_quality_regression),
                }
                for key, value in quality_regressions.items()
            ],
        ]
        row_summaries.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "candidate": candidate_metrics,
                "usable_gain": round(float(usable_gain), 6),
                "quality_regressions": {key: round(float(value), 6) for key, value in quality_regressions.items()},
                "checks": dataset_checks,
            }
        )
        checks.extend({"dataset": dataset, **check} for check in dataset_checks)
    promote = bool(rows) and all(row["ok"] for row in checks)
    return {
        "candidate": candidate,
        "baseline": baseline_prefix,
        "datasets": len(rows),
        "promote_product_default": promote,
        "reason": (
            "passes usable-gain and no-regression gates across all datasets"
            if promote
            else "does not beat rule-post under conservative product gates"
        ),
        "checks": checks,
        "rows": row_summaries,
    }


def _extract_metrics(row: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        key: _float(row.get(f"{prefix}_{key}"))
        for key in METRIC_KEYS
    }


def _rows_from_plausibility_v10(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in payload.get("same_pool_top100_comparison") or []:
        converted = {
            "dataset": row.get("dataset"),
            "rule_top1_product_usable_rate": row.get("rule_top1_usable"),
            "rule_top3_product_usable_rate": row.get("rule_top3_usable"),
            "rule_top3_artifact_rate": row.get("rule_top3_artifact"),
            "rule_top3_trivial_stock_closure_rate": row.get("rule_top3_trivial"),
            "rule_top3_generic_route_rate": row.get("rule_top3_generic"),
            "v10_top1_product_usable_rate": row.get("v10_top1_usable"),
            "v10_top3_product_usable_rate": row.get("v10_top3_usable"),
            "v10_top3_artifact_rate": row.get("v10_top3_artifact"),
            "v10_top3_trivial_stock_closure_rate": row.get("v10_top3_trivial"),
            "v10_top3_generic_route_rate": row.get("v10_top3_generic"),
        }
        out.append(converted)
    return out


def _risk_guarded_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in payload.get("rows") or []:
        dataset = row.get("dataset")
        converted: dict[str, Any] = {"dataset": dataset}
        for prefix in ("rule", "runtime_ccts", "cascade_only"):
            metrics = row.get(prefix)
            if not isinstance(metrics, dict):
                continue
            for key in METRIC_KEYS:
                converted[f"{prefix}_{key}"] = metrics.get(key)
        out.append(converted)
    return out


def _risk_guarded_summary(comparison: dict[str, Any], takeaway: dict[str, Any]) -> dict[str, Any]:
    if not comparison:
        return {
            "available": False,
            "interpretation": "risk-guarded selector comparison was not provided to this gate",
        }
    rows = _risk_guarded_rows(comparison)
    cascade_rows = [row for row in rows if any(key.startswith("cascade_only_") for key in row)]
    return {
        "available": True,
        "contract": "product-audit route class plus concrete issue-risk bucket before learned tie-breakers",
        "datasets": [row.get("dataset") for row in rows],
        "runtime_ccts_no_regression_vs_rule": _no_quality_regression(rows, candidate_prefix="runtime_ccts"),
        "cascade_only_no_regression_vs_rule": _no_quality_regression(cascade_rows, candidate_prefix="cascade_only"),
        "top3_usable_gain_any": max(
            [
                _float(row.get(f"{prefix}_top3_product_usable_rate"))
                - _float(row.get("rule_top3_product_usable_rate"))
                for row in rows
                for prefix in ("runtime_ccts", "cascade_only")
                if f"{prefix}_top3_product_usable_rate" in row
            ]
            or [0.0]
        ),
        "interpretation": (takeaway.get("interpretation") or {}).get(
            "limit",
            "risk guard is a safety-contract fix; it does not by itself prove learned product-level improvement",
        ),
    }


def _no_quality_regression(rows: list[dict[str, Any]], *, candidate_prefix: str) -> bool:
    quality_keys = (
        "top3_artifact_rate",
        "top3_trivial_stock_closure_rate",
        "top3_generic_route_rate",
    )
    if not rows:
        return False
    for row in rows:
        for key in quality_keys:
            if _float(row.get(f"{candidate_prefix}_{key}")) > _float(row.get(f"rule_{key}")) + 1e-9:
                return False
    return True


def _float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _markdown(result: dict[str, Any]) -> str:
    risk = result.get("risk_guarded_selector") or {}
    lines = [
        "# Phase Selector Promotion Gate",
        "",
        f"- Product default: `{(result.get('product_default') or {}).get('selector')}`",
        f"- Promote product default: `{(result.get('decision') or {}).get('promote_product_default')}`",
        f"- Continue post-rerank tie-breakers: `{(result.get('decision') or {}).get('continue_post_rerank_tie_breakers')}`",
        f"- Research fixed-pool context signal: `{((result.get('research_claim') or {}).get('cascade_context_fixed_pool_signal'))}`",
        f"- Risk-guarded selector evidence: `{risk.get('available')}`",
        "",
        "## Candidate Gates",
        "",
        "| candidate | promote | reason |",
        "|---|---:|---|",
    ]
    for gate in result.get("candidate_gates") or []:
        lines.append(f"| `{gate.get('candidate')}` | `{gate.get('promote_product_default')}` | {gate.get('reason')} |")
    lines.extend(["", "## Risk Guard", "", "```json", json.dumps(risk, indent=2, ensure_ascii=False), "```", ""])
    lines.extend(["", "## Context Gate", "", "```json", json.dumps((result.get("research_claim") or {}).get("context_gate") or {}, indent=2, ensure_ascii=False), "```", ""])
    lines.extend(["## Next Direction", "", str((result.get("decision") or {}).get("next_direction")), ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate selector promotion for the current model-focus phase")
    ap.add_argument("--plausibility-v10-summary", required=True)
    ap.add_argument("--context-control-summary", required=True)
    ap.add_argument("--runtime-ccts-comparison", required=True)
    ap.add_argument("--cascade-only-comparison", required=True)
    ap.add_argument("--risk-guarded-comparison")
    ap.add_argument("--risk-guarded-takeaway")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    ap.add_argument("--min-context-mrr-delta", type=float, default=0.05)
    ap.add_argument("--min-product-usable-gain", type=float, default=0.01)
    ap.add_argument("--max-quality-regression", type=float, default=1e-9)
    args = ap.parse_args()
    result = gate_phase_selector_promotion(
        plausibility_v10_summary=Path(args.plausibility_v10_summary),
        context_control_summary=Path(args.context_control_summary),
        runtime_ccts_comparison=Path(args.runtime_ccts_comparison),
        cascade_only_comparison=Path(args.cascade_only_comparison),
        risk_guarded_comparison=Path(args.risk_guarded_comparison) if args.risk_guarded_comparison else None,
        risk_guarded_takeaway=Path(args.risk_guarded_takeaway) if args.risk_guarded_takeaway else None,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        min_context_mrr_delta=float(args.min_context_mrr_delta),
        min_product_usable_gain=float(args.min_product_usable_gain),
        max_quality_regression=float(args.max_quality_regression),
    )
    print(
        json.dumps(
            {
                "product_default": result["product_default"],
                "research_claim": result["research_claim"],
                "decision": result["decision"],
                "promoted_candidates": result["promoted_candidates"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
