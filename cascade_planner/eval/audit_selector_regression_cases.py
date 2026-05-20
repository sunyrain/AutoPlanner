"""Compare selector outputs target-by-target and surface product-audit regressions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cascade_planner.eval.product_route_feasibility_audit import build_product_route_feasibility_audit
from cascade_planner.eval.rerank_native_routes_with_v4_value import _read_rows


SCHEMA_VERSION = "selector_regression_case_audit.v1"
TRIAGE_CLASSES = {"triage_semisynthesis", "triage_late_stage", "triage_fragment"}
ISSUE_KEYS = {
    "generic": "generic_reaction_sequence",
    "trivial": "trivial_stock_closure",
    "artifact_class": "reject_artifact",
}


def audit_selector_regression_cases(
    *,
    baseline_run: Path,
    candidate_run: Path,
    output_json: Path,
    output_md: Path | None = None,
    benchmark: Path | None = None,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    top_k: int = 3,
) -> dict[str, Any]:
    benchmark_rows = _read_rows(benchmark) if benchmark else None
    baseline_payload = _read_json(baseline_run)
    candidate_payload = _read_json(candidate_run)
    baseline_audit = build_product_route_feasibility_audit(baseline_payload, benchmark_rows=benchmark_rows)
    candidate_audit = build_product_route_feasibility_audit(candidate_payload, benchmark_rows=benchmark_rows)
    cases = _compare_audits(
        baseline_audit,
        candidate_audit,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        top_k=top_k,
    )
    summary = _summary(cases)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "baseline_run": str(baseline_run),
            "candidate_run": str(candidate_run),
            "benchmark": str(benchmark) if benchmark else None,
            "baseline_name": baseline_name,
            "candidate_name": candidate_name,
            "top_k": int(top_k),
        },
        "summary": summary,
        "cases": cases,
        "interpretation": _interpretation(summary, baseline_name=baseline_name, candidate_name=candidate_name),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = output_md or output_json.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _compare_audits(
    baseline_audit: dict[str, Any],
    candidate_audit: dict[str, Any],
    *,
    baseline_name: str,
    candidate_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    baseline_targets = baseline_audit.get("targets") or []
    candidate_targets = candidate_audit.get("targets") or []
    cases = []
    for idx, (base_target, cand_target) in enumerate(zip(baseline_targets, candidate_targets)):
        base_top = _top_routes(base_target, top_k=top_k)
        cand_top = _top_routes(cand_target, top_k=top_k)
        base_flags = _top_flags(base_top)
        cand_flags = _top_flags(cand_top)
        deltas = {
            "usable": int(cand_flags["usable"]) - int(base_flags["usable"]),
            "artifact": int(cand_flags["artifact"]) - int(base_flags["artifact"]),
            "trivial": int(cand_flags["trivial"]) - int(base_flags["trivial"]),
            "generic": int(cand_flags["generic"]) - int(base_flags["generic"]),
        }
        case = {
            "target_index": idx,
            "target_id": cand_target.get("target_id") or base_target.get("target_id") or "",
            "target_smiles": cand_target.get("target_smiles") or base_target.get("target_smiles") or "",
            "baseline_name": baseline_name,
            "candidate_name": candidate_name,
            "baseline_top": [_compact_route(row) for row in base_top],
            "candidate_top": [_compact_route(row) for row in cand_top],
            "baseline_flags": base_flags,
            "candidate_flags": cand_flags,
            "deltas": deltas,
            "tags": _case_tags(base_flags, cand_flags, deltas),
        }
        cases.append(case)
    return cases


def _top_routes(target: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    return sorted(target.get("routes") or [], key=lambda row: int(row.get("rank") or 10**9))[:top_k]


def _top_flags(routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "usable": any(row.get("route_class") in TRIAGE_CLASSES for row in routes),
        "artifact": any(row.get("route_class") == "reject_artifact" for row in routes),
        "trivial": any(ISSUE_KEYS["trivial"] in (row.get("issues") or []) for row in routes),
        "generic": any(ISSUE_KEYS["generic"] in (row.get("issues") or []) for row in routes),
        "classes": [str(row.get("route_class") or "unknown") for row in routes],
        "issues": sorted({str(issue) for row in routes for issue in (row.get("issues") or [])}),
        "tags": sorted({str(tag) for row in routes for tag in (row.get("tags") or [])}),
    }


def _case_tags(base: dict[str, Any], cand: dict[str, Any], deltas: dict[str, int]) -> list[str]:
    tags = []
    if deltas["usable"] > 0:
        tags.append("usable_gain")
    if deltas["usable"] < 0:
        tags.append("usable_loss")
    if deltas["generic"] > 0:
        tags.append("generic_regression")
    if deltas["generic"] < 0:
        tags.append("generic_improvement")
    if deltas["trivial"] > 0:
        tags.append("trivial_regression")
    if deltas["trivial"] < 0:
        tags.append("trivial_improvement")
    if deltas["artifact"] > 0:
        tags.append("artifact_regression")
    if deltas["artifact"] < 0:
        tags.append("artifact_improvement")
    if base["classes"] != cand["classes"]:
        tags.append("top_class_changed")
    if not tags:
        tags.append("no_topk_flag_change")
    return tags


def _compact_route(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row.get("rank"),
        "route_class": row.get("route_class"),
        "issues": row.get("issues") or [],
        "tags": row.get("tags") or [],
        "n_steps": row.get("n_steps"),
        "route_score": row.get("route_score"),
        "stock_closed": row.get("stock_closed"),
        "reaction_classes": ((row.get("reaction_profile") or {}).get("classes") or [])[:8],
        "generic_fraction": (row.get("reaction_profile") or {}).get("generic_fraction"),
        "terminal_profile": {
            key: (row.get("terminal_profile") or {}).get(key)
            for key in (
                "max_terminal_heavy_atoms",
                "max_terminal_similarity_to_product",
                "all_terminals_small",
                "product_like_terminal",
            )
        },
        "step_summaries": row.get("step_summaries") or [],
    }


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "targets": len(cases),
        "usable_gain_targets": 0,
        "usable_loss_targets": 0,
        "generic_regression_targets": 0,
        "generic_improvement_targets": 0,
        "trivial_regression_targets": 0,
        "trivial_improvement_targets": 0,
        "artifact_regression_targets": 0,
        "artifact_improvement_targets": 0,
        "top_class_changed_targets": 0,
        "no_topk_flag_change_targets": 0,
    }
    for case in cases:
        tags = set(case.get("tags") or [])
        for key in list(out):
            if key == "targets":
                continue
            tag = key.removesuffix("_targets")
            out[key] += int(tag in tags)
    out["regression_case_target_ids"] = [
        case.get("target_id")
        for case in cases
        if set(case.get("tags") or []) & {"generic_regression", "trivial_regression", "artifact_regression", "usable_loss"}
    ]
    return out


def _interpretation(summary: dict[str, Any], *, baseline_name: str, candidate_name: str) -> dict[str, Any]:
    regressions = int(summary.get("generic_regression_targets") or 0) + int(summary.get("artifact_regression_targets") or 0) + int(summary.get("trivial_regression_targets") or 0)
    improvements = int(summary.get("generic_improvement_targets") or 0) + int(summary.get("artifact_improvement_targets") or 0) + int(summary.get("trivial_improvement_targets") or 0)
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "verdict": (
            "candidate has quality regressions; inspect cases before promotion"
            if regressions > improvements
            else "candidate has no net quality-regression warning under this top-k audit"
        ),
        "note": "This is product-audit flag attribution, not expert chemical feasibility review.",
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    lines = [
        "# Selector Regression Case Audit",
        "",
        f"- Baseline: `{(result.get('metadata') or {}).get('baseline_name')}`",
        f"- Candidate: `{(result.get('metadata') or {}).get('candidate_name')}`",
        f"- Top-k: `{(result.get('metadata') or {}).get('top_k')}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Regression Cases",
        "",
        "| target | tags | baseline classes | candidate classes | baseline issues | candidate issues |",
        "|---|---|---|---|---|---|",
    ]
    for case in result.get("cases") or []:
        tags = set(case.get("tags") or [])
        if not tags & {"generic_regression", "trivial_regression", "artifact_regression", "usable_loss"}:
            continue
        base = case.get("baseline_flags") or {}
        cand = case.get("candidate_flags") or {}
        lines.append(
            "| `{}` | {} | `{}` | `{}` | `{}` | `{}` |".format(
                case.get("target_id") or case.get("target_index"),
                ", ".join(f"`{tag}`" for tag in case.get("tags") or []),
                ", ".join(base.get("classes") or []),
                ", ".join(cand.get("classes") or []),
                ", ".join(base.get("issues") or []),
                ", ".join(cand.get("issues") or []),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit target-level regressions between two selector outputs")
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--candidate-run", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md")
    ap.add_argument("--benchmark")
    ap.add_argument("--baseline-name", default="baseline")
    ap.add_argument("--candidate-name", default="candidate")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    result = audit_selector_regression_cases(
        baseline_run=Path(args.baseline_run),
        candidate_run=Path(args.candidate_run),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        benchmark=Path(args.benchmark) if args.benchmark else None,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        top_k=args.top_k,
    )
    print(json.dumps({"summary": result["summary"], "interpretation": result["interpretation"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
