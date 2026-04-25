"""Train the EnzExpand reranker on ALL candidate rows (no CV) and freeze.

Produces a LightGBM booster + metadata JSON usable at inference time by a
custom AiZynthFinder expansion strategy (see
``cascade_planner/expand/aiz_enz_policy.py``).

Usage
-----
    python -m cascade_planner.expand.reranker_freeze \\
        --candidates results/v2/reranker/candidates_v2_mf2.csv \\
        --out shared/reranker_frozen_mf2.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade_planner.expand.reranker import FEATURE_COLS
from cascade_planner.paths import shared_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-rounds", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (shared_dir() / "reranker_frozen.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidates)
    df["step_uid"] = df["doi"].astype(str) + "||" + df["step_id"].astype(str)
    df = df.drop_duplicates(
        subset=["step_uid", "template_id_global", "tanimoto_product_cand", "n_reactants"]
    ).reset_index(drop=True)
    df = df.sort_values("step_uid").reset_index(drop=True)

    group_sizes = df.groupby("step_uid", sort=False).size().values
    print(f"[train] rows={len(df)}  groups={len(group_sizes)}  features={len(FEATURE_COLS)}")

    import lightgbm as lgb
    dtr = lgb.Dataset(
        df[FEATURE_COLS].values.astype(np.float32),
        label=df["hit"].values,
        group=group_sizes,
    )
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 5, 10],
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=10,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
    )
    booster = lgb.train(params, dtr, num_boost_round=args.n_rounds)

    booster.save_model(str(out))
    meta = {
        "feature_cols": FEATURE_COLS,
        "n_rows": int(len(df)),
        "n_groups": int(len(group_sizes)),
        "num_leaves": args.num_leaves,
        "n_rounds": args.n_rounds,
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[save] {out}")
    print(f"[save] {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
