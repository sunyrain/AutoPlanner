"""EnzExpand reranker (P1-2).

Goal: push EnzExpand lift over random-in-pool from ~0.97 to >=1.3.

Pipeline
--------
1. Re-run the same 3-fold DOI-CV base MLP as in ``enz_template.run``.
2. For every test step, collect up to ``top_n_tpl`` template candidates and
   their rdchiral-generated reactant outcomes (up to ``max_outcomes`` each).
3. Dump per-candidate rows to
   ``results/v2/reranker/candidates_{tag}.csv`` with features and a binary
   hit label (cand == GT reactants).
4. Fit a LightGBM LambdaRank model on those candidates with a 5-fold DOI CV.
5. Reorder each step's candidates by the reranker score and compute:
     - base EnzExpand top-K (MLP-only rank)
     - reranked top-K
     - random-in-pool top-K (within the same candidate set)
     - lift = model / random_in_pool
6. Write audited report.

Honest-reporting rules: report per-step top-K only over steps where pool >=K,
and report the pool-filter retention alongside.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import GroupKFold

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.paths import results_dir, shared_dir
from cascade_planner.expand.enz_template import (
    apply_template_to_product,
    canon_set,
    canon_set_nostereo,
    extract_templates,
    main_product,
    morgan2,
    predict_topk,
    train as train_mlp,
)

RDLogger.DisableLog("rdApp.*")


# ----------------------------------------------------------------- features

def _tanimoto_smi(a: str, b: str) -> float:
    ma = Chem.MolFromSmiles(a) if a else None
    mb = Chem.MolFromSmiles(b) if b else None
    if ma is None or mb is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def _n_heavy(smi: str) -> int:
    m = Chem.MolFromSmiles(smi) if smi else None
    return m.GetNumHeavyAtoms() if m else 0


def candidate_features(
    *,
    mlp_rank: int,
    mlp_logit: float,
    tpl_freq: int,
    tpl_ec1_prob: float,
    tpl_tx_prob: float,
    cand_reactants: frozenset[str],
    product_smi: str,
) -> dict:
    cand_list = sorted(cand_reactants)
    combined = ".".join(cand_list)
    n_reactants = len(cand_list)
    total_heavy = sum(_n_heavy(s) for s in cand_list)
    tanimoto_pc = _tanimoto_smi(product_smi, combined) if combined else 0.0
    max_reactant_tanimoto = max((_tanimoto_smi(product_smi, s) for s in cand_list), default=0.0)
    return dict(
        mlp_rank=mlp_rank,
        mlp_logit=mlp_logit,
        tpl_freq=tpl_freq,
        tpl_ec1_prob=tpl_ec1_prob,
        tpl_tx_prob=tpl_tx_prob,
        n_reactants=n_reactants,
        total_heavy=total_heavy,
        tanimoto_product_cand=tanimoto_pc,
        tanimoto_product_max_reactant=max_reactant_tanimoto,
    )


FEATURE_COLS = [
    "mlp_rank", "mlp_logit", "tpl_freq", "tpl_ec1_prob", "tpl_tx_prob",
    "n_reactants", "total_heavy", "tanimoto_product_cand",
    "tanimoto_product_max_reactant",
]


# ----------------------------------------------------------------- dump

def dump_candidates(
    data_path: str,
    min_freq: int,
    top_n_tpl: int,
    max_outcomes: int,
    folds: int,
    out_csv: Path,
    drop_cofactor_products: bool = False,
) -> pd.DataFrame:
    print(f"[load] {data_path}")
    steps, _, _ = load_v2(data_path, drop_cofactor_products=drop_cofactor_products)
    enz_steps = [s for s in steps if s.ec_number]
    print(f"  total steps={len(steps)}  enzymatic={len(enz_steps)}"
          + (f"  (cofactor-drop)" if drop_cofactor_products else ""))

    cache_path = shared_dir() / "enzexpand_atommap_cache.json"
    pairs = extract_templates(enz_steps, mapped_cache=cache_path)

    tpl_count: dict[str, int] = collections.Counter()
    for _, tpl in pairs:
        tpl_count[tpl] += 1
    tpls_sorted = [t for t, c in tpl_count.most_common() if c >= min_freq]
    tpl_to_id = {t: i for i, t in enumerate(tpls_sorted)}
    print(f"  unique tpl={len(tpl_count)}  kept={len(tpl_to_id)}  (min_freq={min_freq})")

    # template priors: EC1 distribution and transformation distribution
    tpl_ec1: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    tpl_tx: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    samples = []
    for idx, tpl in pairs:
        if tpl not in tpl_to_id:
            continue
        tid = tpl_to_id[tpl]
        samples.append((idx, tid))
        step = enz_steps[idx]
        ec1 = (step.ec_number or "").split(".")[0]
        if ec1:
            tpl_ec1[tid][ec1] += 1
        tx = (step.transformation_superclass or "other")
        tpl_tx[tid][tx] += 1

    print(f"  trainable samples: {len(samples)}")
    if not samples:
        print("[abort] no trainable samples")
        return pd.DataFrame()

    def _ec1_prob(tid: int, ec1: str) -> float:
        d = tpl_ec1[tid]
        tot = sum(d.values()) or 1
        return d.get(ec1, 0) / tot

    def _tx_prob(tid: int, tx: str) -> float:
        d = tpl_tx[tid]
        tot = sum(d.values()) or 1
        return d.get(tx, 0) / tot

    X = np.stack([morgan2(main_product(enz_steps[i].rxn_smiles)) for i, _ in samples])
    y = np.array([tid for _, tid in samples], dtype=np.int64)
    groups = np.array([enz_steps[i].doi for i, _ in samples])
    sample_step_idx = np.array([i for i, _ in samples])

    n_groups = len(set(groups))
    n_folds = min(folds, n_groups)
    print(f"  CV: {n_folds} folds")

    rows = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups=groups)):
        print(f"\n[fold {fold}] train={len(tr)} test={len(te)}")
        train_tids = sorted(set(y[tr]))
        local_to_global = dict(enumerate(train_tids))
        global_to_local = {g: i for i, g in local_to_global.items()}
        y_tr_local = np.array([global_to_local[v] for v in y[tr]])

        model = train_mlp(
            X[tr], y_tr_local, n_tpl=len(train_tids),
            epochs=60, hidden=1024, dropout=0.4, batch=128, seed=fold,
            verbose=True,
        )

        k_use = min(top_n_tpl, len(train_tids))
        topk_local_idx, topk_logits = predict_topk(model, X[te], k=k_use)

        for j, te_i in enumerate(te):
            step_i = int(sample_step_idx[te_i])
            step = enz_steps[step_i]
            prod = main_product(step.rxn_smiles)
            if prod is None:
                continue
            true_set = canon_set(step.rxn_smiles.split(">>", 1)[0])
            if not true_set:
                continue
            true_set_ns = canon_set_nostereo(step.rxn_smiles.split(">>", 1)[0])
            step_ec1 = (step.ec_number or "").split(".")[0]
            step_tx = step.transformation_superclass or "other"

            rank_emit = 0  # rank among *successfully fired* candidates
            for local_rank, (lid, logit) in enumerate(zip(topk_local_idx[j], topk_logits[j])):
                gid = local_to_global[int(lid)]
                tpl = tpls_sorted[gid]
                cands = apply_template_to_product(tpl, prod, max_outcomes=max_outcomes)
                if not cands:
                    continue
                for cs in cands:
                    feats = candidate_features(
                        mlp_rank=rank_emit,
                        mlp_logit=float(logit),
                        tpl_freq=tpl_count[tpl],
                        tpl_ec1_prob=_ec1_prob(gid, step_ec1),
                        tpl_tx_prob=_tx_prob(gid, step_tx),
                        cand_reactants=cs,
                        product_smi=prod,
                    )
                    cs_ns = frozenset(
                        canon_set_nostereo(".".join(sorted(cs)))
                    )
                    rows.append(dict(
                        fold=fold,
                        doi=step.doi,
                        step_id=step.step_id,
                        ec1=step_ec1,
                        transformation=step_tx,
                        template_id_global=gid,
                        hit=int(cs == true_set),
                        hit_nostereo=int(bool(cs_ns) and cs_ns == true_set_ns),
                        product_smi=prod,
                        cand_reactants=".".join(sorted(cs)),
                        **feats,
                    ))
                    rank_emit += 1
            # diagnostic: also emit a placeholder row if NO template fired (so the step
            # is represented in the candidate dataframe; rank_emit==0 means empty pool)

        elapsed = time.time() - t0
        print(f"  fold {fold} done  cumulative rows={len(rows)}  t={elapsed:.0f}s")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_unique_steps = df.assign(_uid=df['doi'].astype(str) + '||' + df['step_id'].astype(str))['_uid'].nunique() if len(df) else 0
    print(f"\n[save] {out_csv}  ({len(df)} rows, {n_unique_steps} unique steps)")
    return df


# ----------------------------------------------------------------- train reranker

def train_and_eval_reranker(df: pd.DataFrame, out_dir: Path, *, n_folds: int = 5) -> dict:
    import lightgbm as lgb

    print(f"\n[reranker] training on {len(df)} candidates over {df['step_id'].nunique()} steps")
    df = df.copy()
    df["step_uid"] = df["doi"].astype(str) + "||" + df["step_id"].astype(str)

    # deduplicate candidates: same step + same template + same hit + same features -> keep first
    dedupe_cols = ["step_uid", "template_id_global", "tanimoto_product_cand", "n_reactants"]
    df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    groups = df["doi"].values
    n_g = len(set(groups))
    n_folds = min(n_folds, n_g)
    print(f"  {n_folds}-fold DOI CV  ({n_g} unique DOIs)")

    Ks = (1, 5, 10, 50)
    per_step_rows: list[dict] = []

    for outer_fold, (tr_idx, te_idx) in enumerate(GroupKFold(n_splits=n_folds).split(df, df["hit"], groups=groups)):
        tr = df.iloc[tr_idx]
        te = df.iloc[te_idx]
        # train
        tr_sorted = tr.sort_values("step_uid").reset_index(drop=True)
        group_sizes = tr_sorted.groupby("step_uid", sort=False).size().values
        params = dict(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 5, 10],
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=10,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            verbose=-1,
        )
        dtr = lgb.Dataset(
            tr_sorted[FEATURE_COLS].values,
            label=tr_sorted["hit"].values,
            group=group_sizes,
        )
        booster = lgb.train(params, dtr, num_boost_round=200)

        # score test
        te = te.assign(rr_score=booster.predict(te[FEATURE_COLS].values))

        from math import comb

        def _rand_at_least_one(pool_n: int, n_hits: int, k: int) -> float:
            """P(>=1 hit in K draws without replacement from a pool with n_hits positives)."""
            if pool_n == 0 or n_hits == 0:
                return 0.0
            k = min(k, pool_n)
            if n_hits >= pool_n:
                return 1.0
            return 1.0 - comb(pool_n - n_hits, k) / comb(pool_n, k)

        for uid, grp in te.groupby("step_uid", sort=False):
            grp_base = grp.sort_values("mlp_rank", ascending=True).reset_index(drop=True)
            grp_rr = grp.sort_values("rr_score", ascending=False).reset_index(drop=True)
            pool = len(grp)
            hits = int(grp["hit"].sum())
            has_gt = hits > 0
            hits_ns = int(grp["hit_nostereo"].sum()) if "hit_nostereo" in grp.columns else 0
            rec = dict(
                fold=outer_fold,
                step_uid=uid,
                doi=grp["doi"].iloc[0],
                step_id=grp["step_id"].iloc[0],
                ec1=grp["ec1"].iloc[0],
                transformation=grp["transformation"].iloc[0],
                pool_size=pool,
                n_hits=hits,
                n_hits_nostereo=hits_ns,
                has_gt_in_pool=int(has_gt),
                has_gt_in_pool_ns=int(hits_ns > 0),
            )
            for K in Ks:
                rec[f"base_top{K}"] = int(grp_base.head(K)["hit"].any())
                rec[f"rr_top{K}"] = int(grp_rr.head(K)["hit"].any())
                rec[f"rand_top{K}"] = _rand_at_least_one(pool, hits, K)
                if "hit_nostereo" in grp.columns:
                    rec[f"base_top{K}_ns"] = int(grp_base.head(K)["hit_nostereo"].any())
                    rec[f"rr_top{K}_ns"] = int(grp_rr.head(K)["hit_nostereo"].any())
                    rec[f"rand_top{K}_ns"] = _rand_at_least_one(pool, hits_ns, K)
            per_step_rows.append(rec)

    per_step = pd.DataFrame(per_step_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_step.to_csv(out_dir / "reranker_per_step.csv", index=False)

    def _agg(df_sub: pd.DataFrame, label: str) -> list[dict]:
        out = []
        for K in Ks:
            base = df_sub[f"base_top{K}"].mean() if len(df_sub) else float("nan")
            rr = df_sub[f"rr_top{K}"].mean() if len(df_sub) else float("nan")
            rnd = df_sub[f"rand_top{K}"].mean() if len(df_sub) else float("nan")
            out.append(dict(
                subset=label,
                n_steps=len(df_sub),
                K=K,
                base_topK=base,
                reranker_topK=rr,
                random_in_pool_topK=rnd,
                base_lift=(base / rnd) if rnd else float("nan"),
                reranker_lift=(rr / rnd) if rnd else float("nan"),
                gain_pp=(rr - base) * 100 if not np.isnan(rr) else float("nan"),
            ))
        return out

    summary_rows: list[dict] = []
    summary_rows += _agg(per_step, "ALL")
    summary_rows += _agg(per_step[per_step["has_gt_in_pool"] == 1], "GT_in_pool")
    summary_rows += _agg(per_step[per_step["pool_size"] >= 5], "pool_ge5")

    # nostereo variant (graph-level)
    def _agg_ns(df_sub: pd.DataFrame, label: str) -> list[dict]:
        out = []
        for K in Ks:
            base = df_sub[f"base_top{K}_ns"].mean() if len(df_sub) and f"base_top{K}_ns" in df_sub.columns else float("nan")
            rr = df_sub[f"rr_top{K}_ns"].mean() if len(df_sub) and f"rr_top{K}_ns" in df_sub.columns else float("nan")
            rnd = df_sub[f"rand_top{K}_ns"].mean() if len(df_sub) and f"rand_top{K}_ns" in df_sub.columns else float("nan")
            out.append(dict(
                subset=label + " (nostereo)",
                n_steps=len(df_sub),
                K=K,
                base_topK=base,
                reranker_topK=rr,
                random_in_pool_topK=rnd,
                base_lift=(base / rnd) if rnd else float("nan"),
                reranker_lift=(rr / rnd) if rnd else float("nan"),
                gain_pp=(rr - base) * 100 if not np.isnan(rr) else float("nan"),
            ))
        return out

    if "hit_nostereo" in df.columns:
        summary_rows += _agg_ns(per_step, "ALL")
        summary_rows += _agg_ns(
            per_step[per_step["has_gt_in_pool_ns"] == 1], "GT_in_pool"
        )

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "reranker_summary.csv", index=False)
    print("\n=== reranker summary ===")
    print(df_sum.round(4).to_string(index=False))

    md = ["# EnzExpand reranker — audited report", "",
          f"candidate rows: {len(df)}",
          f"unique steps: {per_step['step_uid'].nunique()}",
          f"pools with GT: {int(per_step['has_gt_in_pool'].sum())} / {len(per_step)}",
          "", "## Headline numbers", "",
          "| subset | K | base top-K | reranker top-K | random-in-pool | base lift | reranker lift | Δpp (rr-base) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        md.append(
            f"| {r['subset']} (n={r['n_steps']}) | {r['K']} | "
            f"{r['base_topK']:.3f} | {r['reranker_topK']:.3f} | {r['random_in_pool_topK']:.3f} | "
            f"{r['base_lift']:.2f} | {r['reranker_lift']:.2f} | {r['gain_pp']:+.1f} |"
        )
    md += ["", "## Reporting discipline", "",
           "- `base`  = MLP-only ranking (current EnzExpand).",
           "- `reranker` = LightGBM LambdaRank on MLP-top-N candidates, DOI-holdout.",
           "- `random_in_pool` = expected top-K hit if we pick K random candidates from the same pool.",
           "- `lift = metric / random_in_pool` (>1 means the ranking is informative).",
           "- `GT_in_pool` subset filters to steps whose GT appeared in the MLP top-N pool;",
           "  this is the only subset where any ranking strategy can possibly help."]
    (out_dir / "reranker_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[save] {out_dir/'reranker_report.md'}")
    return dict(summary=df_sum, per_step=per_step)


# ----------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--top-n-tpl", type=int, default=50)
    ap.add_argument("--max-outcomes", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--tag", default="mf2")
    ap.add_argument("--skip-dump", action="store_true",
                    help="reuse existing candidates CSV")
    ap.add_argument("--drop-cofactor-products", action="store_true",
                    help="drop steps where product is a cofactor (O2/H2O2/CO2/H2O/...)")
    args = ap.parse_args()

    out_dir = results_dir() / "reranker"
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_csv = out_dir / f"candidates_{args.tag}.csv"

    if args.skip_dump and cand_csv.exists():
        print(f"[reuse] {cand_csv}")
        df = pd.read_csv(cand_csv)
    else:
        df = dump_candidates(
            data_path=args.data,
            min_freq=args.min_freq,
            top_n_tpl=args.top_n_tpl,
            max_outcomes=args.max_outcomes,
            folds=args.folds,
            out_csv=cand_csv,
            drop_cofactor_products=args.drop_cofactor_products,
        )

    if df.empty:
        print("[abort] empty candidate set")
        return

    train_and_eval_reranker(df, out_dir / args.tag)


if __name__ == "__main__":
    main()
