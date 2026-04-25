"""EnzExpand ablation: sweep min_freq × topk on the same DOI-grouped folds.

Reuses cascade_planner.expand.enz_template.run() but iterates a small grid.

Output: results/enzexpand_ablation.csv with columns
   min_freq, topk, n_templates, n_train_samples, top1, top5, top10, top50, runtime_s
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/enzexpand_ablation.csv")
    args = ap.parse_args()

    grid = []
    for mf in (1, 2, 3, 5):
        for tk in (10, 25, 50, 100):
            grid.append((mf, tk))

    rows = []
    for mf, tk in grid:
        print(f"\n[ablate] min_freq={mf} topk={tk}")
        t0 = time.time()
        cmd = [
            sys.executable, "-u", "-m", "cascade_planner.expand.enz_template",
            "--data", args.data,
            "--min-freq", str(mf), "--folds", str(args.folds),
            "--epochs", str(args.epochs), "--topk", str(tk),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")
            continue
        rt = time.time() - t0
        # Parse the summary line from enzexpand_summary.csv (overwritten each run)
        summ = pd.read_csv(RESULTS / "enzexpand_summary.csv")
        last = summ.iloc[-1]
        row = dict(min_freq=mf, topk=tk,
                   n_templates_kept=int(last.get("n_templates_kept", 0)),
                   n_steps_eval=int(last.get("n_steps_eval", 0)),
                   top1=float(last.get("top1", 0)),
                   top5=float(last.get("top5", 0)),
                   top10=float(last.get("top10", 0)),
                   top50=float(last.get("top50", 0)),
                   runtime_s=round(rt, 1))
        print(f"  -> top1={row['top1']:.3f} top10={row['top10']:.3f} "
              f"n_tpl={row['n_templates_kept']} t={rt:.0f}s")
        rows.append(row)
        # save after every run (in case we get interrupted)
        pd.DataFrame(rows).to_csv(ROOT / args.out, index=False)

    df = pd.DataFrame(rows)
    out = ROOT / args.out
    df.to_csv(out, index=False)
    print(f"\n[save] {out}")
    print("\n=========== ablation grid ===========")
    piv = df.pivot_table(index="min_freq", columns="topk",
                         values="top1", aggfunc="mean").round(3)
    print("top1 by (min_freq, topk):")
    print(piv)
    piv2 = df.pivot_table(index="min_freq", columns="topk",
                          values="top10", aggfunc="mean").round(3)
    print("\ntop10 by (min_freq, topk):")
    print(piv2)


if __name__ == "__main__":
    main()
