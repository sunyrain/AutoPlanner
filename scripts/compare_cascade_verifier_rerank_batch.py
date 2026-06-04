#!/usr/bin/env python3
"""Batch compare original route ordering with cascade-verifier reranking."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.rerank_routes_with_cascade_verifier import LEARNED_VERIFIER_POLICIES, rerank_routes_with_verifier


SCHEMA_VERSION = "cascade_verifier_rerank_batch_comparison.v1"


def main() -> None:
    args = _parse_args()
    result = compare_batch(
        inputs=[Path(path) for path in args.input],
        output=args.output,
        output_dir=args.output_dir,
        learned_verifier_model=args.learned_verifier_model,
        learned_verifier_policy=args.learned_verifier_policy,
        max_routes_per_target=args.max_routes_per_target,
        default_stage_mode=args.default_stage_mode,
    )
    if args.markdown:
        _write_markdown(result, args.markdown)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def compare_batch(
    *,
    inputs: list[Path],
    output: Path,
    output_dir: Path,
    learned_verifier_model: Path | None = None,
    learned_verifier_policy: str = "annotation_only",
    max_routes_per_target: int | None = None,
    default_stage_mode: str = "stepwise",
) -> dict[str, Any]:
    rows = []
    reason_counts: Counter[str] = Counter()
    audit_bucket_counts: Counter[str] = Counter()
    audit_bucket_feasible_counts: Counter[str] = Counter()
    total_routes = 0
    total_feasible = 0
    top1_changed = 0
    top1_improved = 0
    top1_regressed = 0
    learned_top1_changed = 0
    learned_top1_differs_from_rule = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        rule_out = output_dir / f"{path.stem}.rule_verifier_rerank.json"
        learned_out = output_dir / f"{path.stem}.learned_verifier_rerank.json" if learned_verifier_model else None
        rule_result = rerank_routes_with_verifier(
            input_path=path,
            output_path=rule_out,
            drop_infeasible=False,
            max_routes_per_target=max_routes_per_target,
            default_stage_mode=default_stage_mode,
        )
        learned_result = None
        if learned_out is not None:
            learned_result = rerank_routes_with_verifier(
                input_path=path,
                output_path=learned_out,
                learned_verifier_model=learned_verifier_model,
                learned_verifier_policy=learned_verifier_policy,
                drop_infeasible=False,
                max_routes_per_target=max_routes_per_target,
                default_stage_mode=default_stage_mode,
            )
        row = _compare_one(path, rule_result, learned_result)
        rows.append(row)
        reason_counts.update(row["rule_reason_counts"])
        audit_bucket_counts.update(row["audit_bucket_counts"])
        audit_bucket_feasible_counts.update(row["audit_bucket_feasible_counts"])
        total_routes += row["n_routes_input"]
        total_feasible += row["rule_feasible_routes"]
        top1_changed += int(row["rule_top1_changed"])
        top1_improved += int(row["rule_top1_audit_delta"] > 0)
        top1_regressed += int(row["rule_top1_audit_delta"] < 0)
        learned_top1_changed += int(bool(row.get("learned_top1_changed")))
        learned_top1_differs_from_rule += int(bool(row.get("learned_top1_differs_from_rule")))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_inputs": len(inputs),
        "n_targets": len(rows),
        "n_routes": total_routes,
        "rule_feasible_routes": total_feasible,
        "rule_feasible_fraction": round(total_feasible / total_routes, 4) if total_routes else None,
        "rule_top1_changed": top1_changed,
        "rule_top1_audit_improved": top1_improved,
        "rule_top1_audit_regressed": top1_regressed,
        "learned_top1_changed": learned_top1_changed if learned_verifier_model else None,
        "learned_top1_differs_from_rule": learned_top1_differs_from_rule if learned_verifier_model else None,
        "learned_verifier_model": str(learned_verifier_model) if learned_verifier_model else None,
        "learned_verifier_policy": learned_verifier_policy if learned_verifier_model else None,
        "default_stage_mode": default_stage_mode,
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "audit_bucket_counts": dict(sorted(audit_bucket_counts.items())),
        "audit_bucket_feasible_counts": dict(sorted(audit_bucket_feasible_counts.items())),
        "audit_bucket_feasible_fraction": {
            key: round(float(audit_bucket_feasible_counts.get(key, 0)) / float(count), 4)
            for key, count in sorted(audit_bucket_counts.items())
            if count
        },
        "promotion_decision": _promotion_decision(
            total_routes=total_routes,
            rule_feasible=total_feasible,
            rule_top1_changed=top1_changed,
            rule_top1_regressed=top1_regressed,
            learned_top1_changed=learned_top1_changed,
            learned_top1_differs_from_rule=learned_top1_differs_from_rule,
            learned_enabled=learned_verifier_model is not None,
            learned_verifier_policy=learned_verifier_policy if learned_verifier_model else None,
        ),
        "contract": (
            "Compares reranking of existing route pools only. It does not measure proposal recall "
            "or recover routes absent from the candidate pool."
        ),
    }
    result = {"summary": summary, "targets": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _compare_one(path: Path, rule_result: dict[str, Any], learned_result: dict[str, Any] | None) -> dict[str, Any]:
    target = rule_result["targets"][0] if rule_result.get("targets") else {"routes": [], "target_smiles": ""}
    rule_routes = target.get("routes") or []
    original_order = sorted(rule_routes, key=lambda route: int((route.get("cascade_verifier_rerank") or {}).get("original_rank", 0)))
    original_top = original_order[0] if original_order else {}
    rule_top = rule_routes[0] if rule_routes else {}
    original_top_id = _route_id(original_top)
    rule_top_id = _route_id(rule_top)
    learned_top_id = None
    learned_top_changed = None
    learned_top_differs_from_rule = None
    if learned_result and learned_result.get("targets"):
        learned_routes = learned_result["targets"][0].get("routes") or []
        learned_top_id = _route_id(learned_routes[0]) if learned_routes else None
        learned_top_changed = learned_top_id != original_top_id if learned_top_id is not None else None
        learned_top_differs_from_rule = learned_top_id != rule_top_id if learned_top_id is not None else None
    rule_reason_counts = Counter()
    audit_bucket_counts: Counter[str] = Counter()
    audit_bucket_feasible_counts: Counter[str] = Counter()
    for route in rule_routes:
        rule_reason_counts.update((route.get("cascade_verifier_rerank") or {}).get("reason_counts") or {})
        bucket = _audit_bucket(route)
        audit_bucket_counts[bucket] += 1
        if (route.get("cascade_verifier_rerank") or {}).get("rule_feasible"):
            audit_bucket_feasible_counts[bucket] += 1
    return {
        "input": str(path),
        "target_smiles": target.get("target_smiles"),
        "target_name": _target_name(path),
        "n_routes_input": int(rule_result["summary"]["n_routes_input"]),
        "n_routes_output": int(rule_result["summary"]["n_routes_output"]),
        "rule_feasible_routes": int(rule_result["summary"]["n_feasible_by_rule"]),
        "rule_top1_changed": rule_top_id != original_top_id,
        "original_top1_id": original_top_id,
        "rule_top1_id": rule_top_id,
        "learned_top1_id": learned_top_id,
        "learned_top1_changed": learned_top_changed,
        "learned_top1_differs_from_rule": learned_top_differs_from_rule,
        "original_top1_audit_rank": _audit_rank(original_top),
        "rule_top1_audit_rank": _audit_rank(rule_top),
        "rule_top1_audit_delta": _audit_rank(original_top) - _audit_rank(rule_top),
        "rule_top1_rule_score": (rule_top.get("cascade_verifier_rerank") or {}).get("rule_score"),
        "rule_top1_reasons": (rule_top.get("cascade_verifier_rerank") or {}).get("reason_counts") or {},
        "rule_reason_counts": dict(sorted(rule_reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "audit_bucket_counts": dict(sorted(audit_bucket_counts.items())),
        "audit_bucket_feasible_counts": dict(sorted(audit_bucket_feasible_counts.items())),
    }


def _route_id(route: dict[str, Any]) -> str:
    return str(route.get("id") or route.get("rank") or route.get("display_rank") or _signature(route))


def _signature(route: dict[str, Any]) -> str:
    return "|".join(str(step.get("reaction_smiles") or step.get("rxn_smiles") or "") for step in route.get("steps") or [])


def _audit_rank(route: dict[str, Any]) -> int:
    audit = route.get("product_audit") or {}
    if "risk_order" in audit:
        try:
            return int(audit["risk_order"])
        except (TypeError, ValueError):
            pass
    route_class = str(audit.get("route_class") or "")
    order = {
        "product_exact": 0,
        "late_stage_derivatization": 10,
        "semisynthesis": 20,
        "triage_fragment": 30,
        "needs_chemist_review": 40,
        "reject_artifact": 100,
    }
    return order.get(route_class, 50)


def _audit_bucket(route: dict[str, Any]) -> str:
    audit = route.get("product_audit") or {}
    route_class = str(audit.get("route_class") or "unknown")
    risk_order = audit.get("risk_order")
    if risk_order not in (None, ""):
        try:
            return f"{route_class}:risk{int(risk_order)}"
        except (TypeError, ValueError):
            pass
    return route_class


def _promotion_decision(
    *,
    total_routes: int,
    rule_feasible: int,
    rule_top1_changed: int,
    rule_top1_regressed: int,
    learned_top1_changed: int,
    learned_top1_differs_from_rule: int,
    learned_enabled: bool,
    learned_verifier_policy: str | None,
) -> dict[str, Any]:
    rule_feasible_fraction = float(rule_feasible) / float(total_routes) if total_routes else 0.0
    return {
        "rule_verifier": (
            "promote_as_conservative_gate"
            if rule_top1_regressed == 0
            else "hold_due_to_top1_regression"
        ),
        "rule_feasible_fraction": round(rule_feasible_fraction, 4) if total_routes else None,
        "rule_top1_changed": int(rule_top1_changed),
        "rule_top1_regressed": int(rule_top1_regressed),
        "learned_verifier": (
            "annotation_only_not_ranked"
            if learned_enabled and learned_verifier_policy == "annotation_only"
            else "hold_as_experimental_reranker"
            if learned_enabled and learned_top1_differs_from_rule
            else "not_evaluated"
            if not learned_enabled
            else "calibrated_no_extra_top1_effect_observed"
        ),
        "learned_top1_changed": int(learned_top1_changed) if learned_enabled else None,
        "learned_top1_differs_from_rule": int(learned_top1_differs_from_rule) if learned_enabled else None,
        "rationale": (
            "Rule verifier is interpretable and non-regressive on audit-ranked top1 in this pool. "
            "Learned verifier is the promoted annotation/reason-evidence path; learned reranking is only promotion-ready "
            "if it adds validated improvements beyond the rule verifier."
        ),
    }


def _target_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_top3_routes", "_top5_routes", "_routes"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Cascade Verifier Rerank Batch Comparison",
        "",
        f"- Targets: `{summary['n_targets']}`",
        f"- Routes: `{summary['n_routes']}`",
        f"- Rule-feasible fraction: `{summary['rule_feasible_fraction']}`",
        f"- Rule top1 changed: `{summary['rule_top1_changed']}`",
        f"- Rule top1 audit improved/regressed: `{summary['rule_top1_audit_improved']}` / `{summary['rule_top1_audit_regressed']}`",
        f"- Learned top1 changed/differs-from-rule: `{summary.get('learned_top1_changed')}` / `{summary.get('learned_top1_differs_from_rule')}`",
        "",
        "## Targets",
        "",
        "| Target | Routes | Feasible | Rule top1 changed | Learned top1 changed | Learned differs from rule | Audit delta | Rule top1 score | Top1 reasons |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in result["targets"]:
        lines.append(
            "| {target} | {n} | {feasible} | {changed} | {learned_changed} | {learned_differs} | {delta} | {score} | {reasons} |".format(
                target=row["target_name"],
                n=row["n_routes_input"],
                feasible=row["rule_feasible_routes"],
                changed=row["rule_top1_changed"],
                learned_changed=row["learned_top1_changed"],
                learned_differs=row["learned_top1_differs_from_rule"],
                delta=row["rule_top1_audit_delta"],
                score=row["rule_top1_rule_score"],
                reasons=", ".join(row["rule_top1_reasons"].keys()) or "-",
            )
        )
    lines.extend(["", "## Reasons", "", "| Reason | Count |", "| --- | ---: |"])
    for reason, count in summary["reason_counts"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Audit Buckets", "", "| Bucket | Routes | Rule feasible | Fraction |", "| --- | ---: | ---: | ---: |"])
    bucket_counts = summary.get("audit_bucket_counts") or {}
    bucket_feasible = summary.get("audit_bucket_feasible_counts") or {}
    bucket_fraction = summary.get("audit_bucket_feasible_fraction") or {}
    for bucket, count in bucket_counts.items():
        lines.append(
            f"| `{bucket}` | {count} | {bucket_feasible.get(bucket, 0)} | {bucket_fraction.get(bucket)} |"
        )
    lines.extend([
        "",
        "## Promotion Decision",
        "",
        "```json",
        json.dumps(summary.get("promotion_decision") or {}, indent=2, ensure_ascii=False),
        "```",
    ])
    lines.extend(["", "## Contract", "", summary["contract"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compare cascade-verifier route reranking")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--learned-verifier-model", type=Path)
    parser.add_argument(
        "--learned-verifier-policy",
        choices=LEARNED_VERIFIER_POLICIES,
        default="annotation_only",
    )
    parser.add_argument("--max-routes-per-target", type=int)
    parser.add_argument("--default-stage-mode", choices=["stepwise", "single"], default="stepwise")
    return parser.parse_args()


if __name__ == "__main__":
    main()
