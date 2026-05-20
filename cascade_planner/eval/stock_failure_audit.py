"""Audit strict-stock failures in live benchmark result JSON files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_stock_failure_audit(run: dict[str, Any]) -> dict[str, Any]:
    targets = list(run.get("targets") or [])
    failures = []
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()
    bottleneck_counts: Counter[str] = Counter()
    for target in targets:
        metrics = target.get("metrics") or {}
        if metrics.get("strict_stock_solve_any") is True:
            continue
        recovery = target.get("route_recovery") or {}
        reason = _stock_failure_reason(target)
        domain = str(target.get("route_domain") or "unknown")
        reason_counts[reason] += 1
        domain_counts[domain][reason] += 1
        for label in recovery.get("recovery_bottleneck_labels") or []:
            bottleneck_counts[str(label)] += 1
        failures.append(
            {
                "index": target.get("index"),
                "target_smiles": target.get("target_smiles"),
                "route_domain": domain,
                "reason": reason,
                "plan": bool(metrics.get("plan")),
                "candidate_exact_reaction_in_pool": bool(recovery.get("candidate_exact_reaction_in_pool")),
                "candidate_gt_reactant_in_pool": bool(recovery.get("candidate_gt_reactant_in_pool")),
                "exact_reaction_in_route_pool": bool(recovery.get("exact_reaction_in_route_pool")),
                "gt_reactant_in_route_pool": bool(recovery.get("gt_reactant_in_route_pool")),
                "condition_window_success_any": metrics.get("condition_window_success_any"),
                "cascade_compatibility_success_any": metrics.get("cascade_compatibility_success_any"),
                "recovery_bottleneck_labels": list(recovery.get("recovery_bottleneck_labels") or []),
                "nonstock_terminal_examples": _nonstock_terminal_examples(target),
                "route_count": len(((target.get("planner_output") or {}).get("routes") or [])),
            }
        )
    return {
        "n_targets": len(targets),
        "stock_failure_count": len(failures),
        "stock_failure_rate": round(len(failures) / max(len(targets), 1), 6),
        "reason_counts": dict(reason_counts),
        "bottleneck_counts": dict(bottleneck_counts),
        "domain_reason_counts": {domain: dict(counter) for domain, counter in sorted(domain_counts.items())},
        "failures": failures,
    }


def write_stock_failure_markdown(audit: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stock Failure Audit",
        "",
        f"- targets: {audit.get('n_targets')}",
        f"- stock failures: {audit.get('stock_failure_count')} ({audit.get('stock_failure_rate')})",
        "",
        "## Reasons",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for reason, count in sorted((audit.get("reason_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Domains", "", "| domain | reasons |", "| --- | --- |"])
    for domain, counts in (audit.get("domain_reason_counts") or {}).items():
        summary = ", ".join(f"{reason}:{count}" for reason, count in sorted(counts.items()))
        lines.append(f"| {domain} | {summary} |")
    lines.extend(["", "## Failures", "", "| idx | domain | reason | cand_GT | route_GT | nonstock examples |", "| ---: | --- | --- | ---: | ---: | --- |"])
    for item in audit.get("failures") or []:
        examples = ", ".join(item.get("nonstock_terminal_examples") or [])[:120]
        lines.append(
            "| {idx} | {domain} | {reason} | {cand} | {route} | {examples} |".format(
                idx=item.get("index"),
                domain=item.get("route_domain"),
                reason=item.get("reason"),
                cand=int(bool(item.get("candidate_gt_reactant_in_pool"))),
                route=int(bool(item.get("gt_reactant_in_route_pool"))),
                examples=examples,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stock_failure_reason(target: dict[str, Any]) -> str:
    metrics = target.get("metrics") or {}
    recovery = target.get("route_recovery") or {}
    if not metrics.get("plan"):
        return "no_plan"
    if not recovery.get("candidate_gt_reactant_in_pool"):
        return "generator_gt_reactant_miss"
    if not recovery.get("gt_reactant_in_route_pool"):
        return "selector_missed_gt_reactant_candidate"
    if recovery.get("gt_reactant_in_route_pool") and not metrics.get("strict_stock_solve_any"):
        return "stock_closure_after_route_hit"
    return "other_stock_failure"


def _nonstock_terminal_examples(target: dict[str, Any], *, limit: int = 6) -> list[str]:
    seen: list[str] = []
    for route in ((target.get("planner_output") or {}).get("routes") or []):
        progress = ((route.get("metrics") or {}).get("retrosynthesis_progress") or {})
        status = progress.get("leaf_stock_status") or {}
        for smi, in_stock in status.items():
            if not in_stock and smi not in seen:
                seen.append(str(smi))
                if len(seen) >= limit:
                    return seen
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit strict-stock failures in benchmark result JSON")
    ap.add_argument("--run", required=True, help="Benchmark run.json")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    audit = build_stock_failure_audit(run)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_stock_failure_markdown(audit, Path(args.output_md))
    print(json.dumps({
        "stock_failure_count": audit["stock_failure_count"],
        "reason_counts": audit["reason_counts"],
        "output_json": str(output_json),
        "output_md": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()
