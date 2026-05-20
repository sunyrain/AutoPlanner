"""Compare native, rule-post, and learned rerankers on the same route pool."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "top1_product_usable_rate",
    "top3_product_usable_rate",
    "top5_product_usable_rate",
    "top3_generic_route_rate",
    "top3_artifact_rate",
    "top3_trivial_stock_closure_rate",
)

GT_KEYS = (
    "n_targets_with_gt",
    "exact_gt_route_recovered_rate",
    "partial_gt_step_overlap_rate",
    "gt_reactant_in_route_pool_rate",
)


def compare_same_pool_route_selectors(
    *,
    comparisons: list[str],
    output_json: Path,
    output_md: Path | None = None,
) -> dict[str, Any]:
    rows = []
    for spec in comparisons:
        label, rule_path, learned_path = _parse_spec(spec)
        rule_report = _read_json(rule_path)
        learned_report = _read_json(learned_path)
        rows.append(_comparison_row(label, rule_path, learned_path, rule_report, learned_report))

    report = {
        "schema_version": "same_pool_route_selector_comparison.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": "A/B/C share the same ChemEnzy native route pool; only route ordering changes.",
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown(report), encoding="utf-8")
    return report


def _comparison_row(
    label: str,
    rule_path: Path,
    learned_path: Path,
    rule_report: dict[str, Any],
    learned_report: dict[str, Any],
) -> dict[str, Any]:
    native = rule_report.get("native_ranked_product_metrics") or learned_report.get("native_ranked_product_metrics") or {}
    rule = rule_report.get("rule_ranked_product_metrics") or {}
    learned = learned_report.get("learned_ranked_product_metrics") or {}
    native_gt = rule_report.get("native_gt_recovery") or learned_report.get("native_gt_recovery") or {}
    rule_gt = rule_report.get("rule_gt_recovery") or {}
    learned_gt = learned_report.get("learned_gt_recovery") or {}
    return {
        "label": label,
        "rule_report": str(rule_path),
        "learned_report": str(learned_path),
        "n_targets": native.get("n_targets") or rule.get("n_targets") or learned.get("n_targets"),
        "summary": {
            "native": _pick(native, METRIC_KEYS),
            "rule_post": _pick(rule, METRIC_KEYS),
            "learned_v8_guarded": _pick(learned, METRIC_KEYS),
        },
        "gt_recovery": {
            "native": _pick(native_gt, GT_KEYS),
            "rule_post": _pick(rule_gt, GT_KEYS),
            "learned_v8_guarded": _pick(learned_gt, GT_KEYS),
        },
        "deltas": {
            "rule_minus_native": _delta(rule, native, METRIC_KEYS),
            "learned_minus_native": _delta(learned, native, METRIC_KEYS),
            "learned_minus_rule": _delta(learned, rule, METRIC_KEYS),
        },
    }


def _parse_spec(spec: str) -> tuple[str, Path, Path]:
    parts = [part.strip() for part in spec.split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--comparison must be label,rule_report,learned_report")
    return parts[0], Path(parts[1]), Path(parts[2])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys if key in row}


def _delta(left: dict[str, Any], right: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        a = left.get(key)
        b = right.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[key] = round(float(a) - float(b), 6)
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Same-Pool Route Selector Comparison",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Contract: {report['contract']}",
        "",
        "| label | n | A native top3 usable | B rule top3 usable | C v8 top3 usable | C-B usable | A top3 generic | B top3 generic | C top3 generic | C top3 artifact | C top3 trivial |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows") or []:
        summary = row.get("summary") or {}
        native = summary.get("native") or {}
        rule = summary.get("rule_post") or {}
        learned = summary.get("learned_v8_guarded") or {}
        deltas = (row.get("deltas") or {}).get("learned_minus_rule") or {}
        lines.append(
            "| {label} | {n} | {a3} | {b3} | {c3} | {cb3} | {ag} | {bg} | {cg} | {ca} | {ct} |".format(
                label=row.get("label"),
                n=_fmt(row.get("n_targets")),
                a3=_fmt(native.get("top3_product_usable_rate")),
                b3=_fmt(rule.get("top3_product_usable_rate")),
                c3=_fmt(learned.get("top3_product_usable_rate")),
                cb3=_fmt(deltas.get("top3_product_usable_rate")),
                ag=_fmt(native.get("top3_generic_route_rate")),
                bg=_fmt(rule.get("top3_generic_route_rate")),
                cg=_fmt(learned.get("top3_generic_route_rate")),
                ca=_fmt(learned.get("top3_artifact_rate")),
                ct=_fmt(learned.get("top3_trivial_stock_closure_rate")),
            )
        )
    lines.extend(["", "## Inputs", ""])
    for row in report.get("rows") or []:
        lines.append(f"- `{row.get('label')}` rule: `{row.get('rule_report')}`")
        lines.append(f"- `{row.get('label')}` learned: `{row.get('learned_report')}`")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare A/B/C route selector reports from the same ChemEnzy pool")
    ap.add_argument("--comparison", action="append", required=True, help="label,rule_report,learned_report")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    args = ap.parse_args()
    report = compare_same_pool_route_selectors(
        comparisons=list(args.comparison or []),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
    )
    print(json.dumps({"rows": len(report["rows"]), "output": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
