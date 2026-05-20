"""Aggregate v4 cascade rerank reports into a compact comparison table."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def compare_v4_cascade_rerank_reports(*, reports: list[Path], output_md: Path, output_json: Path | None = None) -> dict[str, Any]:
    rows = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        native = data.get("native_audit_summary") or {}
        method = "learned"
        learned = data.get("learned_audit_summary") or {}
        delta = data.get("delta_learned_minus_native") or {}
        ranked = data.get("learned_ranked_product_metrics") or {}
        if not learned and data.get("rule_audit_summary"):
            method = "rule_post"
            learned = data.get("rule_audit_summary") or {}
            delta = data.get("delta_rule_minus_native") or {}
            ranked = data.get("rule_ranked_product_metrics") or {}
        native_ranked = data.get("native_ranked_product_metrics") or {}
        rows.append(
            {
                "name": path.stem.replace("_v4_rerank_report", "").replace("_v4_rerank_top50_report", ""),
                "method": method,
                "path": str(path),
                "targets": (data.get("summary") or {}).get("targets"),
                "routes": (data.get("summary") or {}).get("total_routes"),
                "native_stock": native.get("strict_stock_solve_rate"),
                "learned_stock": learned.get("strict_stock_solve_rate"),
                "native_top3_triage": native.get("top3_triage_signal_rate"),
                "learned_top3_triage": learned.get("top3_triage_signal_rate"),
                "delta_top3_triage": delta.get("top3_triage_signal_rate"),
                "native_triage": native.get("triage_signal_rate"),
                "learned_triage": learned.get("triage_signal_rate"),
                "delta_triage": delta.get("triage_signal_rate"),
                "route_value_mean": (data.get("route_value_summary") or {}).get("mean"),
                "native_top1_product_usable": native_ranked.get("top1_product_usable_rate"),
                "method_top1_product_usable": ranked.get("top1_product_usable_rate"),
                "native_top5_product_usable": native_ranked.get("top5_product_usable_rate"),
                "method_top5_product_usable": ranked.get("top5_product_usable_rate"),
                "method_top3_artifact": ranked.get("top3_artifact_rate"),
                "method_top3_trivial": ranked.get("top3_trivial_stock_closure_rate"),
                "method_top3_generic": ranked.get("top3_generic_route_rate"),
            }
        )
    summary = {
        "schema_version": "v4_cascade_rerank_comparison.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
        "promotion_readiness": _promotion_readiness(rows),
    }
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(summary), encoding="utf-8")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _promotion_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row.get("delta_top3_triage") for row in rows if isinstance(row.get("delta_top3_triage"), (int, float))]
    negative = [row for row in rows if isinstance(row.get("delta_top3_triage"), (int, float)) and row["delta_top3_triage"] < 0]
    positive = [row for row in rows if isinstance(row.get("delta_top3_triage"), (int, float)) and row["delta_top3_triage"] > 0]
    return {
        "ready_for_promotion": bool(deltas) and not negative and bool(positive),
        "positive_reports": [row["name"] for row in positive],
        "negative_reports": [row["name"] for row in negative],
        "interpretation": (
            "not ready: at least one benchmark has negative top3 triage delta"
            if negative
            else "candidate: no negative top3 triage delta in supplied reports"
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# v4 Cascade Rerank Comparison",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        "",
        "## Promotion Readiness",
        "",
        "```json",
        json.dumps(summary.get("promotion_readiness") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Benchmarks",
        "",
        "| benchmark | method | targets | routes | native stock | method stock | native top1 | method top1 | native top3 | method top3 | delta top3 | native top5 | method top5 | method artifact top3 | method trivial top3 | method generic top3 | delta triage | value mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("rows") or []:
        lines.append(
            "| {name} | {method} | {targets} | {routes} | {native_stock} | {learned_stock} | {native_top1} | {method_top1} | {native_top3} | {learned_top3} | {delta_top3} | {native_top5} | {method_top5} | {method_artifact} | {method_trivial} | {method_generic} | {delta_triage} | {value_mean} |".format(
                name=row.get("name"),
                method=row.get("method"),
                targets=_fmt(row.get("targets")),
                routes=_fmt(row.get("routes")),
                native_stock=_fmt(row.get("native_stock")),
                learned_stock=_fmt(row.get("learned_stock")),
                native_top1=_fmt(row.get("native_top1_product_usable")),
                method_top1=_fmt(row.get("method_top1_product_usable")),
                native_top3=_fmt(row.get("native_top3_triage")),
                learned_top3=_fmt(row.get("learned_top3_triage")),
                delta_top3=_fmt(row.get("delta_top3_triage")),
                native_top5=_fmt(row.get("native_top5_product_usable")),
                method_top5=_fmt(row.get("method_top5_product_usable")),
                method_artifact=_fmt(row.get("method_top3_artifact")),
                method_trivial=_fmt(row.get("method_top3_trivial")),
                method_generic=_fmt(row.get("method_top3_generic")),
                delta_triage=_fmt(row.get("delta_triage")),
                value_mean=_fmt(row.get("route_value_mean")),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare v4 cascade rerank reports")
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json")
    args = ap.parse_args()
    summary = compare_v4_cascade_rerank_reports(
        reports=[Path(path) for path in args.report],
        output_md=Path(args.output_md),
        output_json=Path(args.output_json) if args.output_json else None,
    )
    print(json.dumps(summary["promotion_readiness"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
