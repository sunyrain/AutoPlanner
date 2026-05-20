"""Select union routes only when product-audit evidence is not worse than native."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit
from cascade_planner.eval.rerank_native_routes_with_v4_value import _read_rows, _routes_for_target
from cascade_planner.eval.rerank_native_routes_with_product_audit import _audit_delta, _audit_summary, _ranked_product_metrics


TRIAGE_CLASSES = {"triage_semisynthesis", "triage_late_stage", "triage_fragment"}


def select_safe_union_routes(
    *,
    native_run: Path,
    union_run: Path,
    output: Path,
    report: Path,
    benchmark: Path | None = None,
) -> dict[str, Any]:
    native_payload = json.loads(native_run.read_text(encoding="utf-8"))
    union_payload = json.loads(union_run.read_text(encoding="utf-8"))
    benchmark_rows = _read_rows(benchmark) if benchmark else None

    native_audit = build_product_route_feasibility_audit(native_payload, benchmark_rows=benchmark_rows)
    union_audit = build_product_route_feasibility_audit(union_payload, benchmark_rows=benchmark_rows)
    union_index = _target_index(union_payload.get("targets") or [])
    union_audit_index = _target_index(union_audit.get("targets") or [])

    selected_targets = []
    decisions = []
    for ordinal, native_target in enumerate(native_payload.get("targets") or []):
        target_key = _target_key(native_target, ordinal)
        union_target = union_index.get(target_key)
        native_audit_target = (native_audit.get("targets") or [{}])[ordinal] if ordinal < len(native_audit.get("targets") or []) else {}
        union_audit_target = union_audit_index.get(target_key)
        if union_target is None or union_audit_target is None:
            selected = _with_routes(native_target, _routes_for_target(native_target))
            decision = {"target_key": target_key, "selected": "native", "reason": "union_missing"}
        else:
            keep_union, reason = _keep_union(native_audit_target, union_audit_target)
            selected_source = union_target if keep_union else native_target
            selected = _with_routes(selected_source, _routes_for_target(selected_source))
            decision = {"target_key": target_key, "selected": "union" if keep_union else "native", "reason": reason}
        selected_targets.append(selected)
        decisions.append(decision)

    selected_payload = {
        "metadata": {
            **(union_payload.get("metadata") or native_payload.get("metadata") or {}),
            "reranker": "safe_union_selector",
            "native_run": str(native_run),
            "union_run": str(union_run),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection_contract": "target_level_product_audit_no_regression.v1",
        },
        "summary": {
            "targets": len(selected_targets),
            "total_routes": sum(len(_routes_for_target(target)) for target in selected_targets),
            "selected_union_targets": sum(1 for row in decisions if row["selected"] == "union"),
            "selected_native_targets": sum(1 for row in decisions if row["selected"] == "native"),
        },
        "targets": selected_targets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    selected_audit = build_product_route_feasibility_audit(selected_payload, benchmark_rows=benchmark_rows)
    result = {
        "schema_version": "safe_union_selection_report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "native_run": str(native_run),
            "union_run": str(union_run),
            "output": str(output),
            "benchmark": str(benchmark) if benchmark else None,
        },
        "summary": selected_payload["summary"],
        "native_audit_summary": _audit_summary(native_audit),
        "union_audit_summary": _audit_summary(union_audit),
        "selected_audit_summary": _audit_summary(selected_audit),
        "native_ranked_product_metrics": _ranked_product_metrics(native_audit),
        "union_ranked_product_metrics": _ranked_product_metrics(union_audit),
        "selected_ranked_product_metrics": _ranked_product_metrics(selected_audit),
        "delta_selected_minus_native": _audit_delta(native_audit, selected_audit),
        "decision_counts": dict(Counter(row["selected"] for row in decisions)),
        "reason_counts": dict(Counter(row["reason"] for row in decisions)),
        "decisions": decisions,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def _keep_union(native_target: dict[str, Any], union_target: dict[str, Any]) -> tuple[bool, str]:
    native_metrics = _target_top_metrics(native_target)
    union_metrics = _target_top_metrics(union_target)
    if union_metrics["stock_any"] < native_metrics["stock_any"]:
        return False, "stock_regression"
    if union_metrics["top3_artifact"] > native_metrics["top3_artifact"]:
        return False, "artifact_regression"
    if union_metrics["top3_trivial"] > native_metrics["top3_trivial"]:
        return False, "trivial_regression"
    if union_metrics["top3_usable"] < native_metrics["top3_usable"]:
        return False, "triage_regression"
    if union_metrics["top3_usable"] > native_metrics["top3_usable"]:
        return True, "triage_gain"
    if union_metrics["top3_generic"] < native_metrics["top3_generic"]:
        return True, "generic_reduction"
    return False, "native_tie"


def _target_top_metrics(audit_target: dict[str, Any]) -> dict[str, int]:
    routes = sorted(audit_target.get("routes") or [], key=lambda row: int(row.get("rank") or 10**9))[:3]
    return {
        "stock_any": int(bool(audit_target.get("strict_stock_solve_any"))),
        "top3_usable": int(any(route.get("route_class") in TRIAGE_CLASSES for route in routes)),
        "top3_artifact": int(any(route.get("route_class") == "reject_artifact" for route in routes)),
        "top3_trivial": int(any("trivial_stock_closure" in (route.get("issues") or []) for route in routes)),
        "top3_generic": int(any("generic_reaction_sequence" in (route.get("issues") or []) for route in routes)),
    }


def _with_routes(target: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(target)
    if isinstance((target.get("planner_output") or {}).get("routes"), list):
        planner = dict(target.get("planner_output") or {})
        planner["routes"] = routes
        planner["n_results"] = len(routes)
        out["planner_output"] = planner
    out["routes"] = routes
    out["route_count"] = len(routes)
    return out


def _target_index(targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_target_key(target, idx): target for idx, target in enumerate(targets)}


def _target_key(target: dict[str, Any], ordinal: int) -> str:
    return str(target.get("target_smiles") or target.get("target_id") or target.get("cascade_id") or target.get("index") or ordinal)


def _markdown(report: dict[str, Any]) -> str:
    metrics = report.get("selected_ranked_product_metrics") or {}
    native = report.get("native_ranked_product_metrics") or {}
    lines = [
        "# Safe Union Selection Report",
        "",
        f"- Targets: `{report['summary']['targets']}`",
        f"- Selected union targets: `{report['summary']['selected_union_targets']}`",
        f"- Selected native targets: `{report['summary']['selected_native_targets']}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps({"native": native, "selected": metrics, "decision_counts": report.get("decision_counts")}, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Select union reranked routes when product-audit metrics do not regress")
    ap.add_argument("--native-run", required=True)
    ap.add_argument("--union-run", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--benchmark")
    args = ap.parse_args()
    result = select_safe_union_routes(
        native_run=Path(args.native_run),
        union_run=Path(args.union_run),
        output=Path(args.output),
        report=Path(args.report),
        benchmark=Path(args.benchmark) if args.benchmark else None,
    )
    print(json.dumps(result["delta_selected_minus_native"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
