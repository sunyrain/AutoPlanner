"""Generate a compact comparison table from multiple live benchmark JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PERCENT_KEYS = {
    "filled_type_GT@5",
    "exact_reaction_in_route_pool",
    "candidate_exact_reaction_in_pool",
    "exact_route_reaction_match_any",
    "gt_reactant_in_route_pool",
    "candidate_gt_reactant_in_pool",
    "strict_stock_solve_any",
    "condition_window_success_any",
    "cascade_compatibility_success_any",
}


def format_value(key: str, value) -> str:
    if value is None:
        return "n/a"
    if key in PERCENT_KEYS:
        return f"{100.0 * float(value):.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_input(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = value
        return Path(path).stem, path
    label, path = value.split("=", 1)
    return label, path


def generate(inputs: list[str]) -> str:
    rows = []
    for item in inputs:
        label, path = parse_input(item)
        data = json.loads(Path(path).read_text())
        rows.append((label, path, data.get("summary") or {}))

    metrics = [
        "filled_type_GT@5",
        "exact_reaction_in_route_pool",
        "candidate_exact_reaction_in_pool",
        "exact_route_reaction_match_any",
        "gt_reactant_in_route_pool",
        "candidate_gt_reactant_in_pool",
        "strict_stock_solve_any",
        "avg_strict_stock_first_rank",
        "avg_best_reaction_edit_distance",
        "avg_candidate_exact_reaction_best_rank",
        "avg_candidate_gt_reactant_best_rank",
        "avg_time_per_target_s",
    ]
    lines = ["# Live Benchmark Comparison", ""]
    lines.append("| Run | Source | " + " | ".join(f"`{m}`" for m in metrics) + " |")
    lines.append("|---|---|" + "|".join("---:" for _ in metrics) + "|")
    for label, path, summary in rows:
        values = [format_value(m, summary.get(m)) for m in metrics]
        lines.append(f"| `{label}` | `{path}` | " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="label=path entries or plain paths")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    Path(args.output).write_text(generate(args.input))


if __name__ == "__main__":
    main()
