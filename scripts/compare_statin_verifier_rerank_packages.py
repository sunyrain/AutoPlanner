#!/usr/bin/env python3
"""Compare verifier reranking across statin report packages."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.compare_cascade_verifier_rerank_batch import compare_batch
from scripts.rerank_routes_with_cascade_verifier import LEARNED_VERIFIER_POLICIES


SCHEMA_VERSION = "statin_verifier_rerank_package_matrix.v1"


def main() -> None:
    args = _parse_args()
    result = compare_statin_packages(
        package_dirs=[Path(path) for path in args.package_dir],
        output=args.output,
        output_dir=args.output_dir,
        markdown=args.markdown,
        learned_verifier_model=args.learned_verifier_model,
        learned_verifier_policy=args.learned_verifier_policy,
        default_stage_mode=args.default_stage_mode,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def compare_statin_packages(
    *,
    package_dirs: list[Path],
    output: Path,
    output_dir: Path,
    markdown: Path | None = None,
    learned_verifier_model: Path | None = None,
    learned_verifier_policy: str = "annotation_only",
    default_stage_mode: str = "stepwise",
) -> dict[str, Any]:
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for package_dir in package_dirs:
        route_docs = package_dir / "route_docs"
        inputs = sorted(route_docs.glob("*_top*_routes.json"))
        if not inputs:
            continue
        package_name = package_dir.name
        package_out_dir = output_dir / package_name / "routes"
        comparison_path = output_dir / package_name / "comparison.json"
        comparison_md = output_dir / package_name / "comparison.md"
        comparison = compare_batch(
            inputs=inputs,
            output=comparison_path,
            output_dir=package_out_dir,
            learned_verifier_model=learned_verifier_model,
            learned_verifier_policy=learned_verifier_policy,
            default_stage_mode=default_stage_mode,
        )
        summary = comparison["summary"]
        rows.append(
            {
                "package": package_name,
                "package_dir": str(package_dir),
                "comparison": str(comparison_path),
                "comparison_markdown": str(comparison_md),
                "n_targets": summary["n_targets"],
                "n_routes": summary["n_routes"],
                "rule_feasible_routes": summary["rule_feasible_routes"],
                "rule_feasible_fraction": summary["rule_feasible_fraction"],
                "rule_top1_changed": summary["rule_top1_changed"],
                "rule_top1_audit_improved": summary["rule_top1_audit_improved"],
                "rule_top1_audit_regressed": summary["rule_top1_audit_regressed"],
                "learned_top1_changed": summary["learned_top1_changed"],
                "learned_top1_differs_from_rule": summary.get("learned_top1_differs_from_rule"),
                "promotion_decision": summary.get("promotion_decision") or {},
                "audit_bucket_feasible_fraction": summary.get("audit_bucket_feasible_fraction") or {},
                "reason_counts": summary["reason_counts"],
            }
        )
        # Keep the per-package markdown close to its JSON even when no matrix
        # markdown was requested.
        from scripts.compare_cascade_verifier_rerank_batch import _write_markdown

        _write_markdown(comparison, comparison_md)

    total_routes = sum(int(row["n_routes"]) for row in rows)
    total_feasible = sum(int(row["rule_feasible_routes"]) for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_packages": len(rows),
        "n_routes": total_routes,
        "rule_feasible_routes": total_feasible,
        "rule_feasible_fraction": round(total_feasible / total_routes, 4) if total_routes else None,
        "learned_verifier_model": str(learned_verifier_model) if learned_verifier_model else None,
        "learned_verifier_policy": learned_verifier_policy if learned_verifier_model else None,
        "default_stage_mode": default_stage_mode,
        "promotion_decision": _matrix_promotion_decision(
            rows,
            learned_verifier_policy=learned_verifier_policy if learned_verifier_model else None,
        ),
        "contract": (
            "Package matrix summarizes verifier reranking on curated statin route docs. "
            "It is not a proposal-recall benchmark."
        ),
    }
    result = {"summary": summary, "packages": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if markdown:
        _write_matrix_markdown(result, markdown)
    return result


def _write_matrix_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Statin Verifier Rerank Package Matrix",
        "",
        f"- Packages: `{summary['n_packages']}`",
        f"- Routes: `{summary['n_routes']}`",
        f"- Rule-feasible fraction: `{summary['rule_feasible_fraction']}`",
        f"- Learned verifier model: `{summary['learned_verifier_model']}`",
        f"- Learned verifier policy: `{summary['learned_verifier_policy']}`",
        "",
        "| Package | Routes | Rule feasible | Rule top1 changed | Learned top1 changed | Learned differs from rule | Audit improved/regressed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["packages"]:
        lines.append(
            "| {package} | {routes} | {feasible} | {rule_changed} | {learned_changed} | {learned_differs} | {improved}/{regressed} |".format(
                package=row["package"],
                routes=row["n_routes"],
                feasible=row["rule_feasible_fraction"],
                rule_changed=row["rule_top1_changed"],
                learned_changed=row["learned_top1_changed"],
                learned_differs=row.get("learned_top1_differs_from_rule"),
                improved=row["rule_top1_audit_improved"],
                regressed=row["rule_top1_audit_regressed"],
            )
        )
    lines.extend([
        "",
        "## Promotion Decision",
        "",
        "```json",
        json.dumps(summary.get("promotion_decision") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Contract",
        "",
        summary["contract"],
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matrix_promotion_decision(rows: list[dict[str, Any]], *, learned_verifier_policy: str | None = None) -> dict[str, Any]:
    rule_regressions = sum(int(row["rule_top1_audit_regressed"]) for row in rows)
    learned_changes = sum(int(row["learned_top1_changed"] or 0) for row in rows)
    learned_extra_changes = sum(int(row.get("learned_top1_differs_from_rule") or 0) for row in rows)
    if learned_verifier_policy == "annotation_only":
        learned_decision = "annotation_only_not_ranked"
    elif learned_extra_changes:
        learned_decision = "hold_as_experimental_reranker"
    else:
        learned_decision = "calibrated_no_extra_top1_effect_observed"
    return {
        "rule_verifier": "promote_as_default_metric_and_optional_gate" if rule_regressions == 0 else "hold",
        "learned_verifier": learned_decision,
        "rule_top1_audit_regressions": rule_regressions,
        "learned_top1_changes": learned_changes,
        "learned_top1_differs_from_rule": learned_extra_changes,
        "rationale": (
            "Across statin packages, rule verifier filters clear atom-balance artifacts without audit-ranked top1 regression. "
            "Learned verifier is the promoted annotation/reason-evidence path; learned reranking should only be promoted "
            "if it adds validated improvements beyond the rule verifier."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare verifier reranking across statin report packages")
    parser.add_argument("--package-dir", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--learned-verifier-model", type=Path)
    parser.add_argument(
        "--learned-verifier-policy",
        choices=LEARNED_VERIFIER_POLICIES,
        default="annotation_only",
    )
    parser.add_argument("--default-stage-mode", choices=["stepwise", "single"], default="stepwise")
    return parser.parse_args()


if __name__ == "__main__":
    main()
