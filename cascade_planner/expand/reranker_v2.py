"""EnzExpand reranker v2 — adds DRFP + richer EC/tx priors.

Reuses the candidate CSV from reranker v1 (``results/v2/reranker/candidates_{tag}.csv``) and
augments it with:
  * DRFP (512-bit) of ``reactants>>product`` for each candidate (positional SMILES).
  * EC1/EC2/EC3/EC4 one-hot indicator columns for the step's EC number.
  * transformation-superclass one-hot.

Trains LightGBM LambdaRank with the same 5-fold DOI CV and hypergeometric
random-in-pool baseline.

This module stays independent from v1 so v1 results remain reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cascade_planner.paths import results_dir


# ---------------------------------------------------------------- features

def _drfp_one(rxn: str, length: int = 512) -> np.ndarray:
    from drfp import DrfpEncoder
    fp = DrfpEncoder.encode([rxn], n_folded_length=length)[0]
    return np.asarray(fp, dtype=np.float32)


def _cand_rxn(cand_reactants: str, product: str) -> str:
    # candidate CSV stores joined-sorted reactants in feature dicts already,
    # but we didn't persist them; rebuild from row fields in main()
    return f"{cand_reactants}>>{product}"


def _ec_onehot(ec: str | None) -> dict:
    """Return 4 indicator columns for each EC level presence (not one-hot; indicator)."""
    out = {"ec_l1_any": 0, "ec_l2_any": 0, "ec_l3_any": 0, "ec_l4_any": 0}
    if not ec:
        return out
    parts = str(ec).split(".")
    out["ec_l1_any"] = 1 if len(parts) >= 1 and parts[0] and parts[0] != "-" else 0
    out["ec_l2_any"] = 1 if len(parts) >= 2 and parts[1] and parts[1] != "-" else 0
    out["ec_l3_any"] = 1 if len(parts) >= 3 and parts[2] and parts[2] != "-" else 0
    out["ec_l4_any"] = 1 if len(parts) >= 4 and parts[3] and parts[3] != "-" else 0
    return out


def _ec_hash(ec: str | None, buckets: int = 32) -> int:
    if not ec:
        return 0
    h = int(hashlib.md5(str(ec).encode()).hexdigest(), 16)
    return h % buckets


def _tx_hash(tx: str | None, buckets: int = 32) -> int:
    if not tx:
        return 0
    h = int(hashlib.md5(str(tx).encode()).hexdigest(), 16)
    return h % buckets


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="candidates CSV from reranker v1 (results/v2/reranker/candidates_*.csv)")
    ap.add_argument("--full-data", default="cascade_dataset_v2.normalized.json",
                    help="needed to look up product SMILES per step_id (reranker v1 CSV doesn't carry them)")
    ap.add_argument("--tag", default="v2_mf2_drfp")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--drfp-length", type=int, default=256,
                    help="DRFP folded length (smaller = faster)")
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-rounds", type=int, default=400)
    args = ap.parse_args()

    out_dir = results_dir() / "reranker" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidates)
    print(f"[load] {len(df)} candidate rows from {args.candidates}")

    # re-load v2 data to fetch product SMILES per step uid
    from cascade_planner.data.loader_v2 import load_v2
    from cascade_planner.expand.enz_template import main_product
    steps, _, _ = load_v2(args.full_data)
    step_map = {(s.doi, s.step_id): s for s in steps}

    # build candidate rxn SMILES: we need the candidate reactants string.
    # reranker v1 CSV doesn't include the raw SMILES — we'll approximate by
    # hashing (template_id_global + step_uid + mlp_rank) into a stable id
    # and using the *product* SMILES only for the DRFP input; we take
    # advantage of the product-only Morgan baseline plus the template-id to
    # proxy reactants. For a true rxn-level DRFP we'd need to re-dump; see
    # `dump_with_rxn` CLI below for a fresh dump.

    # Check if the candidates CSV has a 'cand_reactants' column; if not,
    # the DRFP will fall back to product-only Morgan via tanimoto (already in v1).
    has_cand = "cand_reactants" in df.columns
    if not has_cand:
        print("[warn] candidates CSV lacks 'cand_reactants' column; DRFP features disabled.")
        print("       Falling back to v1 features + EC/tx hash buckets only.")

    # enrich with EC/tx hash buckets (cheap and helps v1 generalize per-EC)
    df["step_uid"] = df["doi"].astype(str) + "||" + df["step_id"].astype(str)
    df["ec_bucket"] = df["ec1"].apply(lambda x: _ec_hash(x, 16)).astype(np.int32)
    df["tx_bucket"] = df["transformation"].apply(lambda x: _tx_hash(x, 16)).astype(np.int32)
    ec_ind = df["ec1"].apply(_ec_onehot).apply(pd.Series)
    df = pd.concat([df, ec_ind], axis=1)

    # product-level DRFP: one per step_uid (cached)
    if has_cand:
        print("[drfp] encoding per-candidate rxn ...")
        from drfp import DrfpEncoder
        t0 = time.time()
        smis = df.apply(lambda r: f"{r['cand_reactants']}>>{r.get('product_smi','')}", axis=1).tolist()
        fps = DrfpEncoder.encode(smis, n_folded_length=args.drfp_length)
        drfp_arr = np.asarray(fps, dtype=np.float32)
        print(f"  DRFP shape={drfp_arr.shape}  t={time.time()-t0:.0f}s")
        drfp_cols = [f"drfp_{i}" for i in range(args.drfp_length)]
        df_drfp = pd.DataFrame(drfp_arr, columns=drfp_cols, index=df.index)
        df = pd.concat([df, df_drfp], axis=1)
    else:
        drfp_cols = []
        # product-only DRFP per step (product is the same for all candidates of a step)
        print("[drfp] encoding per-step product-only Morgan→DRFP surrogate ...")
        from rdkit import Chem
        from rdkit.Chem import AllChem
        uids = df["step_uid"].unique()
        prod_map = {}
        for uid in uids:
            doi, sid = uid.split("||", 1)
            s = step_map.get((doi, sid))
            if s is None:
                continue
            prod = main_product(s.rxn_smiles)
            prod_map[uid] = prod
        # simple 256-bit Morgan of product — already redundant with n_heavy etc
        # but provides a dense feature the ranker can learn from
        L = args.drfp_length
        fp_mat = np.zeros((len(df), L), dtype=np.float32)
        prod_fp_cache: dict[str, np.ndarray] = {}
        for i, uid in enumerate(df["step_uid"].values):
            prod = prod_map.get(uid)
            if not prod:
                continue
            if prod in prod_fp_cache:
                fp_mat[i] = prod_fp_cache[prod]
                continue
            m = Chem.MolFromSmiles(prod)
            if m is None:
                continue
            bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=L)
            arr = np.zeros(L, dtype=np.float32)
            from rdkit import DataStructs
            DataStructs.ConvertToNumpyArray(bv, arr)
            prod_fp_cache[prod] = arr
            fp_mat[i] = arr
        drfp_cols = [f"prodfp_{i}" for i in range(L)]
        df_drfp = pd.DataFrame(fp_mat, columns=drfp_cols, index=df.index)
        df = pd.concat([df, df_drfp], axis=1)

    base_cols = [
        "mlp_rank", "mlp_logit", "tpl_freq", "tpl_ec1_prob", "tpl_tx_prob",
        "n_reactants", "total_heavy", "tanimoto_product_cand",
        "tanimoto_product_max_reactant",
    ]
    extra_cols = ["ec_bucket", "tx_bucket", "ec_l1_any", "ec_l2_any", "ec_l3_any", "ec_l4_any"]
    feat_cols = base_cols + extra_cols + drfp_cols
    print(f"[features] total cols={len(feat_cols)}")

    # dedupe
    dedupe_keys = ["step_uid", "template_id_global", "tanimoto_product_cand", "n_reactants"]
    df = df.drop_duplicates(subset=dedupe_keys).reset_index(drop=True)

    import lightgbm as lgb

    groups = df["doi"].values
    n_g = len(set(groups))
    n_folds = min(args.folds, n_g)
    print(f"[cv] {n_folds}-fold DOI GroupKFold  ({n_g} DOIs)")

    Ks = (1, 5, 10, 50)

    def _rand(pool_n: int, n_hits: int, k: int) -> float:
        if pool_n == 0 or n_hits == 0:
            return 0.0
        k = min(k, pool_n)
        if n_hits >= pool_n:
            return 1.0
        return 1.0 - comb(pool_n - n_hits, k) / comb(pool_n, k)

    per_step_rows: list[dict] = []
    for fold, (tr_idx, te_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(df, df["hit"], groups=groups)
    ):
        print(f"\n[fold {fold}] train={len(tr_idx)} test={len(te_idx)}")
        tr = df.iloc[tr_idx].sort_values("step_uid").reset_index(drop=True)
        te = df.iloc[te_idx]
        group_sizes = tr.groupby("step_uid", sort=False).size().values
        params = dict(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 5, 10],
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=10,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
        )
        dtr = lgb.Dataset(tr[feat_cols].values.astype(np.float32),
                          label=tr["hit"].values,
                          group=group_sizes)
        booster = lgb.train(params, dtr, num_boost_round=args.n_rounds)
        te = te.assign(rr_score=booster.predict(te[feat_cols].values.astype(np.float32)))

        for uid, grp in te.groupby("step_uid", sort=False):
            grp_b = grp.sort_values("mlp_rank", ascending=True)
            grp_r = grp.sort_values("rr_score", ascending=False)
            pool = len(grp)
            hits = int(grp["hit"].sum())
            rec = dict(
                fold=fold,
                step_uid=uid,
                doi=grp["doi"].iloc[0],
                step_id=grp["step_id"].iloc[0],
                ec1=grp["ec1"].iloc[0],
                transformation=grp["transformation"].iloc[0],
                pool_size=pool,
                n_hits=hits,
                has_gt_in_pool=int(hits > 0),
            )
            for K in Ks:
                rec[f"base_top{K}"] = int(grp_b.head(K)["hit"].any())
                rec[f"rr_top{K}"] = int(grp_r.head(K)["hit"].any())
                rec[f"rand_top{K}"] = _rand(pool, hits, K)
            per_step_rows.append(rec)

    per_step = pd.DataFrame(per_step_rows)
    per_step.to_csv(out_dir / "reranker_per_step.csv", index=False)

    def _agg(d: pd.DataFrame, label: str):
        out = []
        for K in Ks:
            b = d[f"base_top{K}"].mean() if len(d) else float("nan")
            r = d[f"rr_top{K}"].mean() if len(d) else float("nan")
            rn = d[f"rand_top{K}"].mean() if len(d) else float("nan")
            out.append(dict(subset=label, n=len(d), K=K,
                            base=b, reranker=r, random=rn,
                            base_lift=b / rn if rn else float("nan"),
                            rr_lift=r / rn if rn else float("nan"),
                            delta_pp=(r - b) * 100))
        return out

    rows = []
    rows += _agg(per_step, "ALL")
    rows += _agg(per_step[per_step["has_gt_in_pool"] == 1], "GT_in_pool")
    rows += _agg(per_step[per_step["pool_size"] >= 5], "pool_ge5")
    sumdf = pd.DataFrame(rows)
    sumdf.to_csv(out_dir / "reranker_summary.csv", index=False)
    print("\n=== reranker v2 summary ===")
    print(sumdf.round(4).to_string(index=False))

    md = ["# EnzExpand reranker v2 — DRFP + EC/tx features", "",
          f"candidates: {len(df)} rows, {per_step['step_uid'].nunique()} unique steps",
          f"feature dim: {len(feat_cols)}",
          "",
          "| subset | K | base | reranker | random | base lift | rr lift | Δpp |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['subset']} (n={r['n']}) | {r['K']} | {r['base']:.3f} | "
                  f"{r['reranker']:.3f} | {r['random']:.3f} | "
                  f"{r['base_lift']:.2f} | {r['rr_lift']:.2f} | {r['delta_pp']:+.1f} |")
    (out_dir / "reranker_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[save] {out_dir/'reranker_report.md'}")


if __name__ == "__main__":
    main()
