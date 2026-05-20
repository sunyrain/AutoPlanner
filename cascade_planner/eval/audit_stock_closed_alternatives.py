"""Audit stock-closed routes that do not match the reference route.

Reference routes are useful diagnostics, but retrosynthesis can have many
valid alternatives. This audit separates "missed the benchmark GT" from
"looks chemically unsupported or route-quality risky" using the route metrics
already exported by CascadeBoard.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PLAUSIBLE_CLASSES = {"plausible_alternative", "plausible_needs_condition_review"}
REVIEW_CLASSES = PLAUSIBLE_CLASSES | {"weakly_supported_alternative"}
CRITICAL_CLASSES = {"invalid_or_open_route", "suspicious_stock_shortcut"}
REFERENCE_RECALL_METRICS = (
    "candidate_gt_reactant_in_pool",
    "candidate_exact_reaction_in_pool",
    "exact_reaction_in_route_pool",
    "gt_reactant_in_route_pool",
)


def build_stock_closed_alternative_audit(
    run: dict[str, Any],
    *,
    sample_size: int = 30,
    non_gt_only: bool = True,
) -> dict[str, Any]:
    targets = run.get("targets") or []
    rows: list[dict[str, Any]] = []
    target_class_counts: Counter[str] = Counter()
    route_class_counts: Counter[str] = Counter()
    route_issue_counts: Counter[str] = Counter()
    stock_closed_targets = 0
    reference_gt_stock_targets = 0

    for target in targets:
        metrics = target.get("metrics") or {}
        recovery = target.get("route_recovery") or {}
        if not bool(metrics.get("strict_stock_solve_any")):
            continue
        stock_closed_targets += 1
        if bool(recovery.get("gt_reactant_in_route_pool")):
            reference_gt_stock_targets += 1
            if non_gt_only:
                continue
        routes = (target.get("planner_output") or {}).get("routes") or []
        classified_routes = [_classify_route(route, rank=rank) for rank, route in enumerate(routes, start=1)]
        stock_routes = [row for row in classified_routes if row.get("stock_closed")]
        if not stock_routes:
            continue
        stock_routes = sorted(stock_routes, key=_route_class_sort_key)
        for row in stock_routes:
            route_class_counts[str(row["class"])] += 1
            for issue in row.get("issues") or []:
                route_issue_counts[str(issue)] += 1
        best = _best_route_class(stock_routes)
        target_class_counts[best] += 1
        rows.append(
            {
                "index": target.get("index"),
                "target_smiles": target.get("target_smiles"),
                "route_domain": target.get("route_domain"),
                "gt_n_reactions": len(target.get("gt_route") or []),
                "reference_recall": {
                    metric: bool(recovery.get(metric))
                    for metric in REFERENCE_RECALL_METRICS
                },
                "best_route_class": best,
                "best_route_review_pass": best in REVIEW_CLASSES,
                "best_route_plausible": best in PLAUSIBLE_CLASSES,
                "route_count": len(routes),
                "stock_closed_route_count": len(stock_routes),
                "routes": stock_routes[:5],
            }
        )

    rows.sort(key=_sample_sort_key)
    sample = rows[: max(0, int(sample_size))]
    n_reviewed = len(rows)
    n_review_pass = sum(1 for row in rows if row["best_route_review_pass"])
    n_plausible = sum(1 for row in rows if row["best_route_plausible"])
    report = {
        "schema_version": "stock_closed_alternative_audit.v1",
        "scope": "stock_closed_non_reference_gt" if non_gt_only else "all_stock_closed",
        "n_targets": len(targets),
        "n_stock_closed_targets": stock_closed_targets,
        "n_stock_closed_reference_gt_targets": reference_gt_stock_targets,
        "n_reviewed_targets": n_reviewed,
        "n_routes_reviewed": sum(route_class_counts.values()),
        "target_best_class_counts": dict(sorted(target_class_counts.items())),
        "route_class_counts": dict(sorted(route_class_counts.items())),
        "route_issue_counts": dict(sorted(route_issue_counts.items())),
        "review_pass_rate": _rate(n_review_pass, n_reviewed),
        "plausible_rate": _rate(n_plausible, n_reviewed),
        "critical_or_suspicious_rate": _rate(
            sum(1 for row in rows if row["best_route_class"] in CRITICAL_CLASSES),
            n_reviewed,
        ),
        "sample": sample,
        "interpretation": {
            "review_pass_rate": "Fraction of non-reference-GT stock-closed targets whose best stock-closed route has no critical route-quality flags.",
            "plausible_rate": "Stricter subset with route support or progress; missing T/pH is review-needed, not an automatic failure.",
            "reference_gt": "GT route recall is reported separately because it is one valid literature route, not the only acceptable retrosynthesis.",
        },
    }
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Stock-Closed Alternative Route Audit",
        "",
        f"Scope: `{report.get('scope')}`",
        "",
        "## Summary",
        "",
        f"- Stock-closed targets: `{report.get('n_stock_closed_targets')}`",
        f"- Stock-closed targets that also hit reference GT reactants: `{report.get('n_stock_closed_reference_gt_targets')}`",
        f"- Reviewed non-reference stock-closed targets: `{report.get('n_reviewed_targets')}`",
        f"- Reviewed stock-closed routes: `{report.get('n_routes_reviewed')}`",
        f"- Review-pass rate: `{_fmt(report.get('review_pass_rate'))}`",
        f"- Plausible-alternative rate: `{_fmt(report.get('plausible_rate'))}`",
        f"- Critical/suspicious best-route rate: `{_fmt(report.get('critical_or_suspicious_rate'))}`",
        "",
        "## Best Route Classes",
        "",
        "| Class | Targets |",
        "| --- | ---: |",
    ]
    for key, value in (report.get("target_best_class_counts") or {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Route Issues", "", "| Issue | Routes |", "| --- | ---: |"])
    for key, value in (report.get("route_issue_counts") or {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Sample Cases",
            "",
            "| Index | Domain | Best class | Routes | Reference GT | Target | Top stock-closed route | Issues |",
            "| ---: | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in report.get("sample") or []:
        top = (row.get("routes") or [{}])[0]
        lines.append(
            "| {idx} | {domain} | `{klass}` | {routes} | `{gt}` | `{target}` | `{route}` | {issues} |".format(
                idx=row.get("index"),
                domain=row.get("route_domain") or "",
                klass=row.get("best_route_class"),
                routes=row.get("stock_closed_route_count"),
                gt=(row.get("reference_recall") or {}).get("gt_reactant_in_route_pool"),
                target=row.get("target_smiles"),
                route=_route_signature(top),
                issues=", ".join(f"`{issue}`" for issue in top.get("issues") or []) or "",
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _classify_route(route: dict[str, Any], *, rank: int) -> dict[str, Any]:
    metrics = route.get("metrics") or {}
    naturalness = metrics.get("route_naturalness") or {}
    progress = metrics.get("retrosynthesis_progress") or {}
    condition = metrics.get("condition") or {}
    cascade = metrics.get("cascade_compatibility") or {}
    candidate_pool = metrics.get("candidate_pool") or {}
    steps = route.get("steps") or []
    stock_closed = bool(metrics.get("strict_stock_solve") or (route.get("quality_vector") or {}).get("stock_closed"))
    filled = bool(metrics.get("filled_route", True))
    solved = bool(metrics.get("route_solved", stock_closed))
    critical_counts = {
        "unfilled_steps": int(naturalness.get("unfilled_steps") or 0),
        "product_mismatch_steps": int(naturalness.get("product_mismatch_steps") or 0),
        "atom_balance_violations": int(naturalness.get("atom_balance_violations") or 0),
        "self_loop_steps": int(naturalness.get("self_loop_steps") or 0),
    }
    issues: list[str] = []
    if not stock_closed or not filled or not solved:
        issues.append("open_or_unfilled_route")
    for key, value in critical_counts.items():
        if value > 0:
            issues.append(key)
    progressive = bool(
        metrics.get("progressive_route")
        or progress.get("retrosynthesis_progress_success")
        or _safe_float(progress.get("progressive_step_fraction"), 0.0) >= 0.5
        or _safe_float(progress.get("main_chain_reduction"), 0.0) >= 0.25
    )
    confidence_values = [_step_probability(step) for step in steps]
    measured_conf = [value for value in confidence_values if value is not None]
    min_conf = min(measured_conf) if measured_conf else None
    low_conf_steps = sum(1 for value in measured_conf if value <= 0.005)
    pool_supported = _safe_float(candidate_pool.get("total_candidates"), 0.0) > 0
    named_source_supported = any(_is_named_provider(step.get("source")) for step in steps)
    native_low_conf = low_conf_steps > 0 and not pool_supported and not named_source_supported
    if native_low_conf:
        issues.append("low_confidence_native_template")
    if not progressive:
        issues.append("weak_retrosynthetic_progress")
    if condition and not condition.get("condition_window_success"):
        missing = cascade.get("issues") or []
        if "missing_temperature" in missing or "missing_pH" in missing:
            issues.append("missing_condition_window")
        else:
            issues.append("condition_window_review")

    if not stock_closed or not filled or not solved or any(critical_counts.values()):
        klass = "invalid_or_open_route"
    elif native_low_conf and not progressive:
        klass = "suspicious_stock_shortcut"
    elif native_low_conf or not progressive:
        klass = "weakly_supported_alternative"
    elif "missing_condition_window" in issues or "condition_window_review" in issues:
        klass = "plausible_needs_condition_review"
    else:
        klass = "plausible_alternative"

    return {
        "rank": rank,
        "class": klass,
        "stock_closed": stock_closed,
        "filled_route": filled,
        "route_solved": solved,
        "progressive": progressive,
        "min_step_probability": min_conf,
        "low_confidence_steps": low_conf_steps,
        "candidate_pool_supported": pool_supported,
        "named_source_supported": named_source_supported,
        "n_steps": len(steps),
        "route_score": route.get("score"),
        "route_cost": (route.get("quality_vector") or {}).get("route_cost"),
        "terminal_reactants": metrics.get("terminal_reactants") or [],
        "issues": issues,
        "steps": [_step_summary(step) for step in steps[:5]],
    }


def _best_route_class(routes: list[dict[str, Any]]) -> str:
    return min(routes, key=_route_class_sort_key).get("class", "unknown")


def _route_class_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    order = {
        "plausible_alternative": 0,
        "plausible_needs_condition_review": 1,
        "weakly_supported_alternative": 2,
        "suspicious_stock_shortcut": 3,
        "invalid_or_open_route": 4,
    }
    return (order.get(str(row.get("class")), 99), int(row.get("rank") or 999))


def _sample_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    priority = {
        "suspicious_stock_shortcut": 0,
        "invalid_or_open_route": 1,
        "weakly_supported_alternative": 2,
        "plausible_needs_condition_review": 3,
        "plausible_alternative": 4,
    }
    return (priority.get(str(row.get("best_route_class")), 9), int(row.get("index") or 10**9))


def _step_probability(step: dict[str, Any]) -> float | None:
    scores = step.get("scores") or {}
    for key in ("confidence", "retro", "probability", "score"):
        value = _safe_float(scores.get(key), None)
        if value is not None and 0.0 <= value <= 1.0:
            return value
    value = _safe_float(step.get("score"), None)
    if value is not None and 0.0 <= value <= 1.0:
        return value
    return None


def _is_named_provider(source: Any) -> bool:
    text = str(source or "").lower()
    named = ("chemenzyretroplanner", "retrochimera", "enzyformer", "enzexpand", "uspto_template", "v3_retrieval")
    return any(name in text for name in named)


def _step_summary(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "reaction_smiles": step.get("reaction_smiles") or step.get("rxn_smiles"),
        "source": step.get("source"),
        "probability": _step_probability(step),
        "stock_status": step.get("stock_status") or {},
    }


def _route_signature(route: dict[str, Any]) -> str:
    steps = route.get("steps") or []
    if not steps:
        return ""
    first = steps[0]
    rxn = str(first.get("reaction_smiles") or "")
    if len(rxn) > 80:
        rxn = rxn[:77] + "..."
    if len(steps) == 1:
        return rxn
    return f"{rxn} (+{len(steps) - 1} step)"


def _rate(num: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return round(float(num) / float(denom), 4)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit stock-closed routes that do not match reference GT routes")
    ap.add_argument("--run", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--sample-size", type=int, default=30)
    ap.add_argument("--include-reference-gt", action="store_true")
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    report = build_stock_closed_alternative_audit(
        run,
        sample_size=args.sample_size,
        non_gt_only=not args.include_reference_gt,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, Path(args.output_md))
    print(json.dumps({"output_json": str(output_json), "n_reviewed_targets": report["n_reviewed_targets"]}, indent=2))


if __name__ == "__main__":
    main()
