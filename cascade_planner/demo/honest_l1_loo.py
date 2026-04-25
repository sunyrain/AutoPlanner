"""HONEST LOO confirmation of L1 generalization win.

Per query: drop DOI, train ONE EnzExpand-EM (union pool, 20 ep), then
score with generalize=0 (baseline) vs generalize=1 (the proposed fix)
and 4 matchers (exact / no_cofactor / main_only / scaffold).

Lighter than final_integrated_eval (no USPTO, no EnzExpand-A, no
condition heads) so it finishes in ~25 min.
"""
from __future__ import annotations

import argparse
import collections
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.expand.enz_template import (apply_template_to_product,
                                                  canon_set, main_product,
                                                  morgan2, predict_topk,
                                                  train as train_mlp)
from cascade_planner.demo.final_integrated_eval import build_template_pool
from cascade_planner.demo.cofactor_regrader import (_main_frag_set,
                                                     _scaffold_set,
                                                     _strip_cofactors)
from cascade_planner.expand.enzymemap_loader import (extract_templates_from_enzymemap,
                                                      load_filtered)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


def expand_with_mlp(model, tpls_sorted, prod, k, max_outcomes=5, generalize=0):
    if model is None or not prod:
        return []
    x = morgan2(prod).reshape(1, -1)
    tk, _ = predict_topk(model, x, k=k)
    out = []
    seen = set()
    for lid in tk[0]:
        tpl = tpls_sorted[int(lid)]
        for cs in apply_template_to_product(tpl, prod, max_outcomes=max_outcomes,
                                             generalize=generalize):
            fs = frozenset(cs)
            if fs not in seen:
                seen.add(fs); out.append(fs)
    return out


def first_match(pred_sets, truth, transform):
    truth_t = transform(truth)
    if not truth_t:
        return None
    for i, fs in enumerate(pred_sets, 1):
        if transform(fs) == truth_t:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--n-queries", type=int, default=24)
    ap.add_argument("--topk-tpl", type=int, default=100)
    ap.add_argument("--max-outcomes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print(" HONEST LOO confirmation: L0 vs L1 generalization")
    print("=" * 70)
    steps, _, _ = load_v2(args.data)
    em_df = load_filtered(min_quality=0.95, only_single=True, ec1_balance=4000)
    em_rows = extract_templates_from_enzymemap(em_df)

    # Same query selection as final_integrated_eval
    rng = np.random.default_rng(args.seed)
    by_ec1 = collections.defaultdict(list)
    for s in steps:
        if not s.ec_number: continue
        e = s.ec_number.split(".")[0]
        p = main_product(s.rxn_smiles)
        if p and len(p) > 4:
            by_ec1[e].append(s)
    chem = [s for s in steps if not s.ec_number
            and main_product(s.rxn_smiles) and len(main_product(s.rxn_smiles)) > 4]
    queries = []
    per_ec = max(1, (args.n_queries - 4) // 5)
    for e in ["1","2","3","4","5"]:
        if not by_ec1[e]: continue
        idxs = rng.choice(len(by_ec1[e]),
                          size=min(per_ec, len(by_ec1[e])), replace=False)
        for i in idxs:
            queries.append(("enz_EC"+e, by_ec1[e][int(i)]))
    chem_idxs = rng.choice(len(chem), size=min(4, len(chem)), replace=False)
    for i in chem_idxs:
        queries.append(("chem", chem[int(i)]))
    queries = queries[:args.n_queries]
    print(f"queries: n={len(queries)}")

    rows = []
    for qi, (tag, sq) in enumerate(queries):
        prod = main_product(sq.rxn_smiles)
        truth = canon_set(sq.rxn_smiles.split(">>",1)[0]) if ">>" in sq.rxn_smiles else frozenset()
        held = sq.doi
        train_steps = [s for s in steps if s.doi != held]
        train_enz = [s for s in train_steps if s.ec_number]

        t0 = time.time()
        Xb, yb, tplsB, _ = build_template_pool(train_enz, em_rows)
        if Xb is None:
            continue
        modelB = train_mlp(Xb, yb, n_tpl=len(tplsB), epochs=20,
                            hidden=1024, dropout=0.4, batch=512,
                            seed=0, verbose=False)
        result = {"qi": qi, "tag": tag, "doi": held, "n_pool": Xb.shape[0],
                  "n_tpl": len(tplsB)}
        for level in (0, 1):
            preds = expand_with_mlp(modelB, tplsB, prod, k=args.topk_tpl,
                                     max_outcomes=args.max_outcomes,
                                     generalize=level)
            result[f"L{level}_n"] = len(preds)
            for mname, mfn in [("exact", lambda s: s),
                               ("noCF",  _strip_cofactors),
                               ("main",  _main_frag_set),
                               ("scaf",  _scaffold_set)]:
                r = first_match(preds, truth, mfn)
                result[f"L{level}_{mname}"] = r
        rows.append(result)
        dt = time.time() - t0
        print(f"  [Q{qi+1:02d} {tag:8s}] {held[:34]:34s} {dt:5.1f}s "
              f"L0(n={result['L0_n']:3d} ex={result['L0_exact']} sc={result['L0_scaf']}) "
              f"L1(n={result['L1_n']:3d} ex={result['L1_exact']} sc={result['L1_scaf']})")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "honest_l1_loo.csv", index=False)
    print(f"\n[save] results/honest_l1_loo.csv")

    print("\n" + "=" * 70)
    print(" HONEST LOO TOP-K RECALL  (per-query model dropped DOI)")
    print("=" * 70)
    print(f"{'slice':10s}{'level':6s}{'exact_top1':>11s}{'exact_top10':>13s}"
          f"{'scaf_top1':>11s}{'scaf_top10':>12s}{'avg_n':>8s}")
    for slc, sub in [("enz", out[out["tag"].str.startswith("enz")]),
                     ("chem", out[out["tag"]=="chem"]),
                     ("ALL", out)]:
        for L in (0, 1):
            ex = sub[f"L{L}_exact"]
            sc = sub[f"L{L}_scaf"]
            an = sub[f"L{L}_n"]
            line = f"{slc:10s}L{L}    "
            line += f"{(ex.notna()&(ex<=1)).mean()*100:9.0f}% "
            line += f"{(ex.notna()&(ex<=10)).mean()*100:11.0f}% "
            line += f"{(sc.notna()&(sc<=1)).mean()*100:9.0f}% "
            line += f"{(sc.notna()&(sc<=10)).mean()*100:10.0f}% "
            line += f"{an.mean():7.1f}"
            print(line)


if __name__ == "__main__":
    main()
