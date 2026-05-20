"""Generate manuscript-ready benchmark snippets from live benchmark JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    if x is None:
        return "n/a"
    return f"{100 * float(x):.1f}%"


PERCENT_KEYS = {
    "plan_rate",
    "skeleton_type_GT@1",
    "skeleton_type_GT@5",
    "filled_route_any",
    "filled_type_GT@1",
    "filled_type_GT@5",
    "terminal_GT_reactant_in_top5",
    "exact_reaction_in_route_pool",
    "candidate_exact_reaction_in_pool",
    "exact_route_reaction_match_any",
    "gt_reactant_in_route_pool",
    "candidate_gt_reactant_in_pool",
    "strict_stock_solve_any",
    "condition_window_success_any",
    "cascade_compatibility_success_any",
    "avg_best_exact_reaction_fraction",
}


def format_value(key, value):
    if value is None:
        return "n/a"
    if key in PERCENT_KEYS:
        return pct(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def generate(path: str) -> str:
    data = json.loads(Path(path).read_text())
    s = data["summary"]
    lines = []
    lines.append("# Live Benchmark Results")
    lines.append("")
    lines.append(f"Source: `{path}`")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in [
        "n_targets",
        "plan_rate",
        "skeleton_type_GT@1",
        "skeleton_type_GT@5",
        "filled_route_any",
        "filled_type_GT@1",
        "filled_type_GT@5",
        "terminal_GT_reactant_in_top5",
        "exact_reaction_in_route_pool",
        "candidate_exact_reaction_in_pool",
        "exact_route_reaction_match_any",
        "gt_reactant_in_route_pool",
        "candidate_gt_reactant_in_pool",
        "avg_best_exact_reaction_fraction",
        "avg_best_reaction_edit_distance",
        "avg_best_type_edit_distance",
        "avg_candidate_exact_reaction_best_rank",
        "avg_candidate_gt_reactant_best_rank",
        "strict_stock_solve_any",
        "avg_strict_stock_first_rank",
        "condition_window_success_any",
        "cascade_compatibility_success_any",
        "avg_time_per_target_s",
    ]:
        lines.append(f"| `{key}` | {format_value(key, s.get(key))} |")

    lines.append("")
    lines.append("## Candidate Sources")
    lines.append("")
    lines.append("| Source | Count |")
    lines.append("|---|---:|")
    for source, count in sorted((s.get("candidate_source_counts") or {}).items()):
        lines.append(f"| `{source}` | {count} |")

    lines.append("")
    lines.append("## Per Domain")
    lines.append("")
    lines.append("| Domain | n | Plan | Filled type GT@5 | Stock | Condition | Compatibility |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for domain, row in sorted((s.get("per_domain") or {}).items()):
        lines.append(
            f"| `{domain}` | {row.get('n')} | {pct(row.get('plan'))} | "
            f"{pct(row.get('filled_type_GT@5'))} | "
            f"{pct(row.get('strict_stock_solve_any'))} | "
            f"{pct(row.get('condition_window_success_any'))} | "
            f"{pct(row.get('cascade_compatibility_success_any'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    Path(args.output).write_text(generate(args.input))


if __name__ == "__main__":
    main()
