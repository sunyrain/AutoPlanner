"""Multi-step solve-rate benchmark.

Picks N targets from the canonical dataset (cascade final products),
runs AiZ MCTS (USPTO-only AND USPTO+EnzExpand hybrid), records:

   target_smi, doi, n_steps_gt, policy, solve, n_routes, best_score,
   best_depth, in_stock_frac, search_time_s

Output: results/multistep_solvebench.csv

Designed to fill PROPOSAL.md's K3 (solve rate) baseline column.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.multistep.plan_route import call_mcts

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


def _final_product(steps_in_cascade):
    """Last step's main product is the cascade final product."""
    if not steps_in_cascade:
        return None
    last = steps_in_cascade[-1]
    rxn = last.rxn_smiles
    if not rxn or ">>" not in rxn:
        return None
    rhs = rxn.split(">>", 1)[1]
    best, best_n = None, -1
    for s in rhs.split("."):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n = m.GetNumHeavyAtoms()
        if n > best_n:
            best, best_n = Chem.MolToSmiles(m), n
    return best


def collect_targets(steps, n, seed=42):
    """Group steps by (doi, cascade) by treating same DOI's steps as one cascade."""
    by_doi = {}
    for s in steps:
        by_doi.setdefault(s.doi, []).append(s)
    items = []
    for doi, sts in by_doi.items():
        prod = _final_product(sts)
        if prod and 6 <= len(prod) <= 200:
            items.append((doi, prod, len(sts)))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def run_one(target, doi, n_gt, policies, weights, max_iter, max_depth, n_routes, timeout, config):
    try:
        out = call_mcts(target, max_iter=max_iter, max_depth=max_depth,
                        n_routes=n_routes, use_filter=True, timeout=timeout,
                        config_path=config, policies=policies,
                        policy_weights=weights)
    except Exception as e:
        return dict(target=target, doi=doi, n_steps_gt=n_gt,
                    policy="+".join(policies), solve=False, error=str(e)[:200])
    routes = out.get("routes", [])
    solve = len(routes) > 0
    best = max(routes, key=lambda r: (r.get("in_stock_frac") or 0)) if solve else {}
    return dict(target=target, doi=doi, n_steps_gt=n_gt,
                policy="+".join(policies),
                solve=int(solve),
                n_routes=out.get("n_routes_total", 0),
                best_score=best.get("score"),
                best_depth=best.get("depth"),
                in_stock_frac=best.get("in_stock_frac"),
                search_time_s=out.get("search_time_s"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-routes", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="aizdata/config_hybrid.yml")
    ap.add_argument("--out", default="results/multistep_solvebench.csv")
    ap.add_argument("--mode", choices=["uspto", "hybrid", "both"], default="both")
    args = ap.parse_args()

    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    targets = collect_targets(steps, args.n_targets, seed=args.seed)
    print(f"  picked {len(targets)} targets")

    rows = []
    out_path = ROOT / args.out
    cfg = ROOT / args.config

    for i, (doi, target, n_gt) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {doi}  gt_steps={n_gt}  target={target[:60]}")
        # USPTO only
        if args.mode in ("uspto", "both"):
            t0 = time.time()
            r = run_one(target, doi, n_gt, ["uspto"], None,
                        args.max_iter, args.max_depth, args.n_routes,
                        args.timeout, cfg)
            print(f"  uspto  : solve={r.get('solve')} routes={r.get('n_routes')} "
                  f"depth={r.get('best_depth')} stock={r.get('in_stock_frac')} "
                  f"time={r.get('search_time_s')}s wall={time.time()-t0:.0f}s")
            rows.append(r)
        # hybrid USPTO+EnzExpand
        if args.mode in ("hybrid", "both"):
            t0 = time.time()
            r = run_one(target, doi, n_gt, ["uspto", "enzexpand"], [0.7, 0.3],
                        args.max_iter, args.max_depth, args.n_routes,
                        args.timeout, cfg)
            print(f"  hybrid : solve={r.get('solve')} routes={r.get('n_routes')} "
                  f"depth={r.get('best_depth')} stock={r.get('in_stock_frac')} "
                  f"time={r.get('search_time_s')}s wall={time.time()-t0:.0f}s")
            rows.append(r)
        # save after each target so we don't lose progress
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")
    print("\n=========== SUMMARY ===========")
    summ = df.groupby("policy").agg(
        n=("target", "count"),
        solve_rate=("solve", "mean"),
        mean_depth=("best_depth", "mean"),
        mean_stock=("in_stock_frac", "mean"),
        mean_time=("search_time_s", "mean"),
    ).round(3)
    print(summ)


if __name__ == "__main__":
    main()
