"""Run the frozen 100-target benchmark through AiZ MCTS.

Two policies are compared (same K-budget):
  - AiZ USPTO-only (baseline)
  - AiZ USPTO + EnzExpand expansion (hybrid)

For each target we record:
  solve(0/1), n_routes, best_depth, best_in_stock_frac,
  search_time_s, gt_depth.

Output:
  results/v2/benchmark_v2_100_solvebench.csv
  results/v2/benchmark_v2_100_solvebench.md  (honest summary by route_domain)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from cascade_planner.multistep.plan_route import call_mcts
from cascade_planner.paths import results_dir

ROOT = Path(__file__).resolve().parent.parent.parent


def _canon(smi: str) -> str | None:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def _gt_step_products(gt_route: list[dict]) -> set[str]:
    """Set of canonical product SMILES across every GT step (intermediate + final)."""
    out = set()
    for s in gt_route:
        rxn = s.get("rxn_smiles") or ""
        if ">>" not in rxn:
            continue
        for frag in rxn.split(">>")[1].split("."):
            c = _canon(frag)
            if c:
                out.add(c)
    return out


def _route_intermediates(tree: dict | None) -> set[str]:
    """All molecule SMILES appearing in an AiZ route tree."""
    if not tree:
        return set()
    out = set()
    stack = [tree]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        smi = node.get("smiles")
        if smi:
            c = _canon(smi)
            if c:
                out.add(c)
        for c in node.get("children", []) or []:
            stack.append(c)
    return out


def run_target(target: str, cfg: Path, policies, weights, max_iter, max_depth, n_routes, timeout,
               gt_intermediates: set[str] | None = None):
    t0 = time.time()
    try:
        out = call_mcts(
            target, max_iter=max_iter, max_depth=max_depth, n_routes=n_routes,
            use_filter=True, timeout=timeout, config_path=cfg,
            policies=policies, policy_weights=weights,
        )
    except Exception as e:
        return dict(solve=0, error=str(e)[:200], search_time_s=time.time() - t0)
    routes = out.get("routes", [])
    solved_routes = [r for r in routes if r.get("is_solved")]
    best = (max(solved_routes, key=lambda r: (r.get("in_stock_frac") or 0.0))
            if solved_routes else
            (max(routes, key=lambda r: (r.get("in_stock_frac") or 0.0)) if routes else {}))

    # GT@K: does any of the top-K routes share intermediate molecules with GT?
    gt_hit_at1 = gt_hit_at5 = 0
    gt_overlap_best = 0.0
    if gt_intermediates:
        sorted_routes = sorted(routes, key=lambda r: -(r.get("in_stock_frac") or 0.0))
        for idx, r in enumerate(sorted_routes[:5]):
            inter = _route_intermediates(r.get("tree"))
            if not inter:
                continue
            overlap = len(inter & gt_intermediates) / max(1, len(gt_intermediates))
            if idx == 0:
                gt_overlap_best = overlap
            # hit if at least half the GT intermediates are in the route
            if overlap >= 0.5:
                gt_hit_at5 = 1
                if idx == 0:
                    gt_hit_at1 = 1

    return dict(
        solve=int(len(solved_routes) > 0),
        n_routes=out.get("n_routes_total", 0),
        n_solved_routes=len(solved_routes),
        best_score=best.get("score"),
        best_depth=best.get("depth"),
        in_stock_frac=best.get("in_stock_frac"),
        gt_hit_at1=gt_hit_at1,
        gt_hit_at5=gt_hit_at5,
        gt_overlap_best=round(gt_overlap_best, 3),
        search_time_s=out.get("search_time_s") or (time.time() - t0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark_v2_100.json")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-routes", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--config-uspto", default="aizdata/config.yml")
    ap.add_argument("--config-hybrid", default="aizdata/config_hybrid.yml")
    ap.add_argument("--n-targets", type=int, default=0,
                    help="0 = all; else first N targets (for smoke tests)")
    ap.add_argument("--skip-hybrid", action="store_true")
    ap.add_argument("--skip-uspto", action="store_true",
                    help="Skip the USPTO-only run (useful when re-running hybrid only).")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to output CSV/MD names, e.g. '_hybrid'.")
    args = ap.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    if args.n_targets:
        benchmark = benchmark[: args.n_targets]

    out_dir = results_dir()
    rows: list[dict] = []
    cfg_uspto = ROOT / args.config_uspto
    cfg_hybrid = ROOT / args.config_hybrid

    for i, item in enumerate(benchmark):
        target = item["target_smiles"]
        doi = item["doi"]
        dom = item["route_domain"]
        gt_depth = item["depth"]
        gt_inter = _gt_step_products(item.get("gt_route") or [])
        # add precursors (LHS of first step) too
        if item.get("gt_route"):
            first_rxn = item["gt_route"][0].get("rxn_smiles") or ""
            if ">>" in first_rxn:
                for frag in first_rxn.split(">>")[0].split("."):
                    c = _canon(frag)
                    if c:
                        gt_inter.add(c)
        print(f"\n=== [{i+1}/{len(benchmark)}] {dom} d={gt_depth} {target[:60]}...")

        # USPTO-only
        if not args.skip_uspto:
            r_u = run_target(target, cfg_uspto, ["uspto"], None,
                             args.max_iter, args.max_depth, args.n_routes, args.timeout,
                             gt_intermediates=gt_inter)
            rows.append(dict(doi=doi, target=target, route_domain=dom, gt_depth=gt_depth,
                             policy="uspto", **r_u))
            print(f"  uspto: solve={r_u.get('solve')}  gt@5={r_u.get('gt_hit_at5')}  "
                  f"overlap={r_u.get('gt_overlap_best',0):.2f}  t={r_u.get('search_time_s',0):.0f}s")

        # Hybrid
        if not args.skip_hybrid and cfg_hybrid.exists():
            r_h = run_target(target, cfg_hybrid, ["uspto", "enzexpand"], [0.5, 0.5],
                             args.max_iter, args.max_depth, args.n_routes, args.timeout,
                             gt_intermediates=gt_inter)
            rows.append(dict(doi=doi, target=target, route_domain=dom, gt_depth=gt_depth,
                             policy="uspto+enz", **r_h))
            print(f"  hybrid: solve={r_h.get('solve')}  gt@5={r_h.get('gt_hit_at5')}  "
                  f"t={r_h.get('search_time_s',0):.0f}s")

        # incremental save
        pd.DataFrame(rows).to_csv(out_dir / f"benchmark_v2_100_solvebench{args.out_suffix}.csv", index=False)

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"benchmark_v2_100_solvebench{args.out_suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    # summary by policy and domain
    def _row(d, policy, subset):
        return dict(
            policy=policy,
            subset=subset,
            n=len(d),
            solve_rate=d["solve"].mean() if len(d) else float("nan"),
            gt_at1=d.get("gt_hit_at1", pd.Series(dtype=float)).mean() if "gt_hit_at1" in d else float("nan"),
            gt_at5=d.get("gt_hit_at5", pd.Series(dtype=float)).mean() if "gt_hit_at5" in d else float("nan"),
            gt_overlap=d.get("gt_overlap_best", pd.Series(dtype=float)).mean() if "gt_overlap_best" in d else float("nan"),
            mean_depth=d["best_depth"].dropna().mean() if d.get("best_depth") is not None and d["best_depth"].notna().any() else float("nan"),
            mean_time_s=d["search_time_s"].mean() if len(d) else float("nan"),
        )
    summary_rows = []
    for policy in df["policy"].unique():
        d = df[df["policy"] == policy]
        summary_rows.append(_row(d, policy, "ALL"))
        for dom in sorted(d["route_domain"].unique()):
            dd = d[d["route_domain"] == dom]
            if len(dd) >= 5:
                summary_rows.append(_row(dd, policy, dom))
    df_sum = pd.DataFrame(summary_rows)
    sum_csv = out_dir / f"benchmark_v2_100_summary{args.out_suffix}.csv"
    df_sum.to_csv(sum_csv, index=False)
    print("\n=== summary ===")
    print(df_sum.round(3).to_string(index=False))

    md = ["# 100-target multi-step benchmark — honest evaluation", "",
          f"max_iter={args.max_iter}  max_depth={args.max_depth}  n_routes={args.n_routes}  timeout={args.timeout}s",
          "",
          "`solve-rate`: fraction of targets with ≥1 route whose leaves are all in ZINC stock.",
          "`GT@K`: fraction with any of top-K routes overlapping ≥50% of GT intermediates.",
          "",
          "| policy | subset | n | solve-rate | GT@1 | GT@5 | mean GT-overlap | mean depth | mean t(s) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        md.append(
            f"| {r['policy']} | {r['subset']} | {r['n']} | "
            f"{r['solve_rate']:.2%} | {r['gt_at1']:.2%} | {r['gt_at5']:.2%} | "
            f"{r['gt_overlap']:.2f} | {r['mean_depth']:.2f} | {r['mean_time_s']:.0f} |"
        )
    md_path = out_dir / f"benchmark_v2_100_solvebench{args.out_suffix}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[save] {md_path}")


if __name__ == "__main__":
    main()
