"""Multi-step solve-rate benchmark — HARD variant.

Picks N targets from the canonical dataset filtered to:
  - heavy_atoms >= --min-atoms (default 22)
  - NOT in the ZINC stock (so search must actually disconnect)

Otherwise identical to multistep_solvebench.py.

Output: results/multistep_solvebench_hard.csv
"""
from __future__ import annotations

import argparse
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


def _final_product(steps):
    last = steps[-1]
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
    return best, best_n


def _load_stock_inchikeys(stock_h5):
    """Read aizynthfinder hdf5 zinc_stock — keys are inchikeys."""
    try:
        import h5py
        out = set()
        with h5py.File(stock_h5, "r") as f:
            for k in f.keys():
                out.add(k)
        return out
    except Exception as e:
        print(f"  [warn] could not load stock {stock_h5}: {e}")
        return set()


def collect_targets(steps, n, min_atoms, seed, stock_keys):
    by_doi = {}
    for s in steps:
        by_doi.setdefault(s.doi, []).append(s)
    items = []
    for doi, sts in by_doi.items():
        fp = _final_product(sts)
        if not fp:
            continue
        smi, n_atoms = fp
        if n_atoms < min_atoms or n_atoms > 80:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        ik = Chem.MolToInchiKey(m)
        if ik in stock_keys:
            continue  # trivial — already in stock
        items.append((doi, smi, len(sts), n_atoms))
    rng = random.Random(seed)
    rng.shuffle(items)
    print(f"  candidate pool after filter: {len(items)}")
    return items[:n]


def run_one(target, doi, n_gt, n_atoms, policies, weights,
            max_iter, max_depth, n_routes, timeout, config):
    try:
        out = call_mcts(target, max_iter=max_iter, max_depth=max_depth,
                        n_routes=n_routes, use_filter=True, timeout=timeout,
                        config_path=config, policies=policies,
                        policy_weights=weights)
    except Exception as e:
        return dict(target=target, doi=doi, n_steps_gt=n_gt, n_atoms=n_atoms,
                    policy="+".join(policies), solve=False, error=str(e)[:200])
    routes = out.get("routes", [])
    solve = len(routes) > 0
    best = max(routes, key=lambda r: (r.get("in_stock_frac") or 0)) if solve else {}
    return dict(target=target, doi=doi, n_steps_gt=n_gt, n_atoms=n_atoms,
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
    ap.add_argument("--min-atoms", type=int, default=22)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--n-routes", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="aizdata/config_hybrid.yml")
    ap.add_argument("--stock", default="aizdata/zinc_stock.hdf5")
    ap.add_argument("--out", default="results/multistep_solvebench_hard.csv")
    ap.add_argument("--mode", choices=["uspto", "hybrid", "both"], default="both")
    args = ap.parse_args()

    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    print(f"[load stock] {args.stock}")
    stock_keys = _load_stock_inchikeys(ROOT / args.stock)
    print(f"  stock entries: {len(stock_keys)}")
    targets = collect_targets(steps, args.n_targets, args.min_atoms,
                              args.seed, stock_keys)
    print(f"  picked {len(targets)} targets (min_atoms={args.min_atoms})")

    rows = []
    out_path = ROOT / args.out
    cfg = ROOT / args.config

    for i, (doi, target, n_gt, na) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {doi}  gt_steps={n_gt}  atoms={na}  target={target[:60]}")
        if args.mode in ("uspto", "both"):
            t0 = time.time()
            r = run_one(target, doi, n_gt, na, ["uspto"], None,
                        args.max_iter, args.max_depth, args.n_routes,
                        args.timeout, cfg)
            print(f"  uspto  : solve={r.get('solve')} routes={r.get('n_routes')} "
                  f"depth={r.get('best_depth')} stock={r.get('in_stock_frac')} "
                  f"time={r.get('search_time_s')}s wall={time.time()-t0:.0f}s")
            rows.append(r)
        if args.mode in ("hybrid", "both"):
            t0 = time.time()
            r = run_one(target, doi, n_gt, na, ["uspto", "enzexpand"], [0.7, 0.3],
                        args.max_iter, args.max_depth, args.n_routes,
                        args.timeout, cfg)
            print(f"  hybrid : solve={r.get('solve')} routes={r.get('n_routes')} "
                  f"depth={r.get('best_depth')} stock={r.get('in_stock_frac')} "
                  f"time={r.get('search_time_s')}s wall={time.time()-t0:.0f}s")
            rows.append(r)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")
    print("\n=========== SUMMARY (HARD) ===========")
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
