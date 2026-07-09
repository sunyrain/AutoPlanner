"""Compare live CascadeBoard runs with different structured-prior providers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from AUTOPLANNRELLM.deepseek_client import is_placeholder_deepseek_key, normalize_deepseek_key_value
from cascade_planner.cascadeboard.live_benchmark import run_live_benchmark


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _value(value: Any, key: str) -> str:
    if value is None:
        return "n/a"
    if key in {
        "avg_time_per_target_s",
        "avg_best_reaction_edit_distance",
        "avg_best_type_edit_distance",
        "avg_candidate_exact_reaction_best_rank",
        "avg_candidate_gt_reactant_best_rank",
        "avg_strict_stock_first_rank",
    }:
        return "n/a" if value is None else f"{float(value):.3f}"
    return _pct(value)


def _provider_output_path(output_prefix: str, provider: str) -> str:
    safe = provider.replace("-", "_")
    return f"{output_prefix}_{safe}.json"


def write_report(rows: list[dict[str, Any]], report_path: str, metadata: dict[str, Any]) -> None:
    lines = [
        "# Prior Benchmark Comparison",
        "",
        f"Benchmark: `{metadata['bench_path']}`",
        f"Limit: `{metadata['limit']}`",
        f"n_results: `{metadata['n_results']}`",
        f"skeleton_samples: `{metadata['skeleton_samples']}`",
        f"device: `{metadata['device']}`",
        f"check_stock: `{metadata['check_stock']}`",
        f"search_mode: `{metadata.get('search_mode')}`",
        f"search_budget: `{metadata.get('search_budget')}`",
        "",
        "| Provider | Status | Source counts | Plan | Filled GT@5 | Exact rxn | Cand exact rxn | Cand rxn rank | Exact route | GT reactant | Cand GT reactant | Cand reactant rank | Stock | Stock rank | Condition | Compatibility | s/target |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary = row.get("summary") or {}
        source_counts = summary.get("agent_prior_source_counts") or {}
        source_text = ", ".join(f"{k}:{v}" for k, v in sorted(source_counts.items())) or "-"
        lines.append(
            f"| `{row['provider']}` | {row['status']} | {source_text} | "
            f"{_value(summary.get('plan_rate'), 'plan_rate')} | "
            f"{_value(summary.get('filled_type_GT@5'), 'filled_type_GT@5')} | "
            f"{_value(summary.get('exact_reaction_in_route_pool'), 'exact_reaction_in_route_pool')} | "
            f"{_value(summary.get('candidate_exact_reaction_in_pool'), 'candidate_exact_reaction_in_pool')} | "
            f"{_value(summary.get('avg_candidate_exact_reaction_best_rank'), 'avg_candidate_exact_reaction_best_rank')} | "
            f"{_value(summary.get('exact_route_reaction_match_any'), 'exact_route_reaction_match_any')} | "
            f"{_value(summary.get('gt_reactant_in_route_pool'), 'gt_reactant_in_route_pool')} | "
            f"{_value(summary.get('candidate_gt_reactant_in_pool'), 'candidate_gt_reactant_in_pool')} | "
            f"{_value(summary.get('avg_candidate_gt_reactant_best_rank'), 'avg_candidate_gt_reactant_best_rank')} | "
            f"{_value(summary.get('strict_stock_solve_any'), 'strict_stock_solve_any')} | "
            f"{_value(summary.get('avg_strict_stock_first_rank'), 'avg_strict_stock_first_rank')} | "
            f"{_value(summary.get('condition_window_success_any'), 'condition_window_success_any')} | "
            f"{_value(summary.get('cascade_compatibility_success_any'), 'cascade_compatibility_success_any')} | "
            f"{_value(summary.get('avg_time_per_target_s'), 'avg_time_per_target_s')} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append("- `none` is the no-agent baseline.")
    lines.append("- Prior providers only rerank generated skeletons; they do not create accepted reaction facts.")
    lines.append("- DeepSeek rows require `DEEPSEEK_API_KEY` at runtime. The key is never written to the report.")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n")


def run_prior_comparison(
    *,
    providers: list[str],
    bench_path: str,
    output_prefix: str,
    report_path: str,
    model_path: str,
    limit: int | None,
    n_results: int,
    n_candidates_per_skeleton: int,
    skeleton_samples: int | None,
    device: str,
    check_stock: bool,
    prior_weight: float,
    search_mode: str,
    search_budget: int | None,
    allow_deepseek_fallback: bool,
) -> dict[str, Any]:
    rows = []
    for provider in providers:
        provider = provider.strip()
        if not provider:
            continue
        key = normalize_deepseek_key_value(os.environ.get("DEEPSEEK_API_KEY"))
        unusable_deepseek_key = not key or is_placeholder_deepseek_key(key)
        if provider == "deepseek" and unusable_deepseek_key and not allow_deepseek_fallback:
            rows.append({
                "provider": provider,
                "status": "skipped: DEEPSEEK_API_KEY not configured",
                "summary": {},
                "output": None,
            })
            continue

        output_path = _provider_output_path(output_prefix, provider)
        prior_cache = _provider_output_path(f"{output_prefix}_prior_cache", provider)
        result = run_live_benchmark(
            bench_path=bench_path,
            output_path=output_path,
            model_path=model_path,
            limit=limit,
            n_results=n_results,
            n_candidates_per_skeleton=n_candidates_per_skeleton,
            skeleton_samples=skeleton_samples,
            device=device,
            check_stock=check_stock,
            prior_provider=provider,
            prior_weight=prior_weight,
            prior_cache_path=prior_cache if provider != "none" else None,
            search_mode=search_mode,
            search_budget=search_budget,
        )
        rows.append({
            "provider": provider,
            "status": "ok",
            "summary": result.get("summary", {}),
            "output": output_path,
        })

    output = {"rows": rows}
    metadata = {
        "bench_path": bench_path,
        "limit": limit,
        "n_results": n_results,
        "skeleton_samples": skeleton_samples,
        "device": device,
        "check_stock": check_stock,
        "prior_weight": prior_weight,
        "search_mode": search_mode,
        "search_budget": search_budget,
    }
    write_report(rows, report_path, metadata)
    output["metadata"] = metadata
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare no-agent and prior-guided live benchmarks")
    ap.add_argument("--providers", nargs="+", default=["none", "deterministic", "deepseek"])
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--output-prefix", default="results/v2/prior_benchmark_compare")
    ap.add_argument("--report", default="results/v2/prior_benchmark_compare.md")
    ap.add_argument("--model", default="results/shared/skeleton_inpainter/best.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-results", type=int, default=5)
    ap.add_argument("--n-candidates-per-skeleton", type=int, default=2)
    ap.add_argument("--skeleton-samples", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check-stock", action="store_true")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--search-mode", default="rerank", choices=["rerank", "stock_aware", "critic_control", "cc_aostar"])
    ap.add_argument("--search-budget", type=int, default=None)
    ap.add_argument("--allow-deepseek-fallback", action="store_true")
    args = ap.parse_args()

    result = run_prior_comparison(
        providers=args.providers,
        bench_path=args.bench,
        output_prefix=args.output_prefix,
        report_path=args.report,
        model_path=args.model,
        limit=args.limit,
        n_results=args.n_results,
        n_candidates_per_skeleton=args.n_candidates_per_skeleton,
        skeleton_samples=args.skeleton_samples,
        device=args.device,
        check_stock=args.check_stock,
        prior_weight=args.prior_weight,
        search_mode=args.search_mode,
        search_budget=args.search_budget,
        allow_deepseek_fallback=args.allow_deepseek_fallback,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
