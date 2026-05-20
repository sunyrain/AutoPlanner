"""Refresh product/condition audit metadata for an exported route-pool JSON."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.eval.product_route_feasibility_audit import (
    build_product_route_feasibility_audit,
    product_audit_guard_key,
    product_audit_risk_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute product and condition audit fields for exported routes.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-id", default="web_target")
    parser.add_argument("--mode", choices=["preserve", "rerank"], default="preserve")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    refreshed, summary = refresh_route_pool_audit(payload, target_id=args.target_id, mode=args.mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(refreshed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def refresh_route_pool_audit(payload: dict[str, Any], *, target_id: str, mode: str = "preserve") -> tuple[dict[str, Any], dict[str, Any]]:
    routes = [route for route in payload.get("routes") or [] if isinstance(route, dict)]
    target_smiles = str(payload.get("target") or payload.get("target_smiles") or "")
    audit_run = {
        "metadata": {"source": "reaudit_route_pool", "mode": mode},
        "targets": [
            {
                "index": 0,
                "target_id": target_id,
                "target_smiles": target_smiles,
                "planner_output": {"routes": routes},
                "metrics": {
                    "strict_stock_solve_any": any(
                        bool((route.get("metrics") or {}).get("strict_stock_solve"))
                        for route in routes
                    )
                },
            }
        ],
    }
    audit = build_product_route_feasibility_audit(audit_run)
    audit_target = (audit.get("targets") or [{}])[0]
    audit_by_index = {
        int(row.get("rank") or 0) - 1: row
        for row in audit_target.get("routes") or []
        if row.get("rank") is not None
    }
    refreshed_routes = []
    for idx, route in enumerate(routes):
        row = audit_by_index.get(idx) or {}
        route_out = dict(route)
        risk = product_audit_risk_order(row)
        route_out["product_audit"] = _compact_audit(row, risk)
        route_out["rule_post_rank_metadata"] = {
            **(route_out.get("rule_post_rank_metadata") or {}),
            "route_class": route_out["product_audit"].get("route_class"),
            "risk_order": risk,
            "issues": route_out["product_audit"].get("issues") or [],
            "tags": route_out["product_audit"].get("tags") or [],
            "condition_audit": route_out["product_audit"].get("condition_audit") or {},
            "route_plausibility": route_out["product_audit"].get("route_plausibility") or {},
        }
        refreshed_routes.append({"route": route_out, "audit": row, "key": (*product_audit_guard_key(row), idx)})
    if mode == "rerank":
        refreshed_routes.sort(key=lambda item: item["key"])
    output_routes = [item["route"] for item in refreshed_routes]
    for rank, route in enumerate(output_routes):
        route["post_filter_rank"] = rank
        route["route_rank"] = rank
    out = dict(payload)
    out["routes"] = output_routes
    out["n_results"] = len(output_routes)
    out.setdefault("ui_metadata", {})["route_pool_reaudit"] = {
        "schema_version": "route_pool_reaudit.v1",
        "mode": mode,
        "target_id": target_id,
    }
    diversity = out.setdefault("route_set_metrics", {}).setdefault("diversity", {})
    diversity["n_routes"] = len(output_routes)
    diversity["unique_full_signatures"] = len({_route_signature(route) for route in output_routes})
    summary = _summary(output_routes)
    out["route_condition_audit_summary"] = summary["condition"]
    out["route_product_audit_summary"] = summary["product"]
    return out, summary


def _compact_audit(row: dict[str, Any], risk: int) -> dict[str, Any]:
    return {
        "schema_version": "route_product_audit.v2",
        "route_class": row.get("route_class"),
        "risk_order": risk,
        "autonomous_route_candidate": bool(row.get("autonomous_route_candidate")),
        "stock_closed": bool(row.get("stock_closed")),
        "route_solved": bool(row.get("route_solved")),
        "filled_route": bool(row.get("filled_route")),
        "issues": list(row.get("issues") or []),
        "tags": list(row.get("tags") or []),
        "terminal_profile": row.get("terminal_profile") or {},
        "reaction_profile": row.get("reaction_profile") or {},
        "condition_audit": row.get("condition_audit") or {},
        "route_plausibility": row.get("route_plausibility") or {},
    }


def _summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    product_classes: Counter[str] = Counter()
    product_issues: Counter[str] = Counter()
    condition_risks: Counter[str] = Counter()
    condition_issues: Counter[str] = Counter()
    for route in routes:
        audit = route.get("product_audit") or {}
        product_classes[str(audit.get("route_class") or "missing")] += 1
        product_issues.update(str(issue) for issue in audit.get("issues") or [])
        cond = audit.get("condition_audit") or {}
        condition_risks[str(cond.get("route_risk") or "missing")] += 1
        condition_issues.update(
            str(issue)
            for issue, count in (cond.get("issue_counts") or {}).items()
            for _ in range(int(count or 0))
        )
    return {
        "product": {
            "route_class_counts": dict(sorted(product_classes.items())),
            "issue_counts": dict(sorted(product_issues.items())),
        },
        "condition": {
            "route_risk_counts": dict(sorted(condition_risks.items())),
            "issue_counts": dict(sorted(condition_issues.items())),
        },
    }


def _route_signature(route: dict[str, Any]) -> str:
    return "|".join(str(step.get("reaction_smiles") or "") for step in route.get("steps") or [] if isinstance(step, dict))


if __name__ == "__main__":
    main()
