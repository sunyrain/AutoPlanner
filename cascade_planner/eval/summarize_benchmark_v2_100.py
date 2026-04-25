"""Post-hoc summariser for benchmark_v2_100_solvebench.csv.

Breaks down by route_domain × policy and writes a markdown digest suitable
for pasting into STATUS_REPORT.md. Must produce honest numbers only:
    solve_rate  = mean(solve)
    gt@1/gt@5   = mean(gt_hit_at1 / gt_hit_at5)
    overlap     = mean(gt_overlap_best)
    mean_depth  = mean(best_depth) over solved routes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cascade_planner.paths import results_dir


def _fmt_pct(x):
    return "-" if pd.isna(x) else f"{x*100:.1f}%"


def summarize(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        return "empty"

    md = [f"# Benchmark v2 100 summary ({csv_path.name})", "",
          f"total rows: {len(df)}  ·  unique targets: {df['target'].nunique()}",
          "",
          "## Overall by policy",
          "| policy | n | solve | gt@1 | gt@5 | overlap | mean_depth_solved | mean_time_s |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, g in df.groupby("policy"):
        s = g[g["solve"] == 1]
        md.append("| {p} | {n} | {solve} | {g1} | {g5} | {ov:.2f} | {md:.2f} | {tt:.1f} |".format(
            p=policy, n=len(g),
            solve=_fmt_pct(g["solve"].mean()),
            g1=_fmt_pct(g["gt_hit_at1"].mean()),
            g5=_fmt_pct(g["gt_hit_at5"].mean()),
            ov=g["gt_overlap_best"].mean(),
            md=s["best_depth"].mean() if len(s) else float("nan"),
            tt=g["search_time_s"].mean(),
        ))

    md += ["", "## By route_domain × policy",
           "| domain | policy | n | solve | gt@1 | gt@5 | overlap |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for (dom, policy), g in df.groupby(["route_domain", "policy"]):
        md.append("| {d} | {p} | {n} | {solve} | {g1} | {g5} | {ov:.2f} |".format(
            d=dom, p=policy, n=len(g),
            solve=_fmt_pct(g["solve"].mean()),
            g1=_fmt_pct(g["gt_hit_at1"].mean()),
            g5=_fmt_pct(g["gt_hit_at5"].mean()),
            ov=g["gt_overlap_best"].mean(),
        ))

    md += ["", "## By GT depth × policy",
           "| gt_depth | policy | n | solve | gt@5 |",
           "|---:|---|---:|---:|---:|"]
    for (d, policy), g in df.groupby(["gt_depth", "policy"]):
        md.append("| {d} | {p} | {n} | {solve} | {g5} |".format(
            d=int(d), p=policy, n=len(g),
            solve=_fmt_pct(g["solve"].mean()),
            g5=_fmt_pct(g["gt_hit_at5"].mean()),
        ))

    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default=str(results_dir() / "benchmark_v2_100_solvebench.csv"))
    ap.add_argument("--out",
                    default=str(results_dir() / "benchmark_v2_100_summary.md"))
    args = ap.parse_args()

    csv_p = Path(args.csv)
    if not csv_p.exists():
        raise SystemExit(f"not found: {csv_p}")
    md = summarize(csv_p)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
