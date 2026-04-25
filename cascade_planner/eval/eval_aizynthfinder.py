"""Evaluate AiZynthFinder as an external single-step retrosynthesis baseline
on our cascade dataset.

For each step in each trainable cascade we:
  1. Take the product side of the annotated rxn_smiles (canonicalized).
  2. Run AiZynthFinder's expansion policy as a single-step expander.
  3. Take top-K predicted reactant sets; canonicalize each.
  4. Mark hit if any prediction's canonical reactant-set equals the
     annotated reactants' canonical set.

Outputs:
  - results/aizynthfinder_step_eval.csv (per-step rows)
  - results/aizynthfinder_summary.csv   (top-K aggregates)

Usage
-----
  .\\.venv_aizynth\\Scripts\\python.exe -m cascade_planner.eval.eval_aizynthfinder \\
      --config aizdata/config.yml --max-steps 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def canon_set(smiles_dot: str) -> frozenset[str]:
    parts = []
    for s in smiles_dot.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            return frozenset()
        parts.append(Chem.MolToSmiles(m))
    return frozenset(parts)


def split_rxn(rxn: str) -> tuple[frozenset[str], str | None]:
    """Returns (reactant_set_canonical, product_canonical_smiles_dot).

    Product is returned as canonical 'A.B' joined string (multi-product allowed,
    we'll usually pass the largest/main product to the expander)."""
    if not rxn or ">>" not in rxn:
        return frozenset(), None
    lhs, rhs = rxn.split(">>", 1)
    react_set = canon_set(lhs)
    prod_set = canon_set(rhs)
    if not react_set or not prod_set:
        return frozenset(), None
    # main product = largest by atom count
    if len(prod_set) == 1:
        prod = next(iter(prod_set))
    else:
        prod = max(prod_set, key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms())
    return react_set, prod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="aizdata/config.yml",
                    help="Path to AiZynthFinder config.yml")
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="Max steps to evaluate (cap for runtime)")
    ap.add_argument("--top-k", type=int, default=50,
                    help="How many expansion suggestions to draw per step")
    ap.add_argument("--policy", default="uspto",
                    help="Expansion policy name in config.yml")
    ap.add_argument("--out", default="results/aizynthfinder_step_eval.csv")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cascade_planner.data.loader_v2 import load_v2  # type: ignore

    print(f"[load] {args.data}")
    steps, _, _ = load_v2(args.data)
    # Filter to steps with valid rxn_smiles, take in order
    rows_to_eval = []
    for s in steps:
        rs, prod = split_rxn(s.rxn_smiles)
        if not rs or not prod:
            continue
        rows_to_eval.append((s, rs, prod))
        if len(rows_to_eval) >= args.max_steps:
            break
    print(f"[load] eval steps: {len(rows_to_eval)}")

    print(f"[init] AiZynthFinder config={args.config}  policy={args.policy}")
    from aizynthfinder.aizynthfinder import AiZynthFinder
    finder = AiZynthFinder(configfile=args.config)
    finder.expansion_policy.select(args.policy)
    if hasattr(finder, "filter_policy") and finder.filter_policy.items:
        try:
            finder.filter_policy.select_first()
        except Exception:
            pass

    # Use the policy directly as a single-step expander
    policy = finder.expansion_policy
    from aizynthfinder.chem import TreeMolecule

    out_rows = []
    t0 = time.time()
    for i, (step, react_set, prod_smi) in enumerate(rows_to_eval):
        try:
            tm = TreeMolecule(parent=None, smiles=prod_smi)
            actions, _priors = policy.get_actions([tm])
            actions = list(actions)[: args.top_k]
            preds: list[frozenset[str]] = []
            preds_text: list[str] = []
            for act in actions:
                try:
                    rs_mols = act.reactants  # tuple of tuples of TreeMolecule
                except Exception:
                    continue
                for rmols in rs_mols:
                    smis = ".".join(m.smiles for m in rmols)
                    cs = canon_set(smis)
                    if cs:
                        preds.append(cs)
                        preds_text.append(".".join(sorted(cs)))
            # rank-of-correct
            rank = next((k + 1 for k, p in enumerate(preds) if p == react_set), None)
            top1 = int(bool(preds) and preds[0] == react_set)
            top5 = int(any(p == react_set for p in preds[:5]))
            top10 = int(any(p == react_set for p in preds[:10]))
            top50 = int(rank is not None and rank <= 50)
            out_rows.append(dict(
                doi=step.doi, cascade_id=step.cascade_id, step_id=step.step_id,
                step_index=step.step_index, n_predictions=len(preds),
                rank_of_truth=rank, top1=top1, top5=top5, top10=top10, top50=top50,
                product=prod_smi, truth_reactants=".".join(sorted(react_set)),
                top1_pred=preds_text[0] if preds_text else "",
                ec_number=step.ec_number, transformation=step.transformation_superclass,
            ))
        except Exception as e:
            out_rows.append(dict(
                doi=step.doi, cascade_id=step.cascade_id, step_id=step.step_id,
                step_index=step.step_index, n_predictions=0,
                rank_of_truth=None, top1=0, top5=0, top10=0, top50=0,
                product=prod_smi, truth_reactants=".".join(sorted(react_set)),
                top1_pred="", ec_number=step.ec_number,
                transformation=step.transformation_superclass,
                error=f"{type(e).__name__}: {str(e)[:100]}",
            ))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(rows_to_eval)}] {elapsed:.1f}s  "
                  f"top1={sum(r['top1'] for r in out_rows)} "
                  f"top10={sum(r['top10'] for r in out_rows)}")

    df = pd.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}  ({len(df)} rows)")

    # ---- summary ----
    print("\n==== AiZynthFinder cascade-step evaluation ====")
    print(f"Steps evaluated      : {len(df)}")
    print(f"Steps with any pred  : {(df['n_predictions'] > 0).sum()}")
    print(f"Mean # predictions   : {df['n_predictions'].mean():.1f}")
    for k in [1, 5, 10, 50]:
        col = f"top{k}"
        print(f"top-{k:<2d} accuracy        : {df[col].mean()*100:5.1f}%  ({df[col].sum()}/{len(df)})")
    if "error" in df.columns:
        nerr = df["error"].notna().sum()
        print(f"Errors               : {nerr}")

    # by enzymatic vs chemical
    print("\nBy enzymatic vs chemical step:")
    for label, mask in [("enzymatic", df["ec_number"].notna() & (df["ec_number"] != "")),
                        ("chemical",  df["ec_number"].isna() | (df["ec_number"] == ""))]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        print(f"  {label:9s} N={len(sub):3d}  top1={sub['top1'].mean()*100:5.1f}%  "
              f"top10={sub['top10'].mean()*100:5.1f}%  top50={sub['top50'].mean()*100:5.1f}%")

    # by transformation
    print("\nTop-10 by transformation_superclass:")
    g = (df.groupby("transformation")
           .agg(N=("top1", "size"), top1=("top1", "mean"),
                top10=("top10", "mean"))
           .sort_values("N", ascending=False).head(10))
    print(g.round(3).to_string())

    summary_path = out_path.with_name(out_path.stem.replace("_step_eval", "_summary") + ".csv")
    summary = {
        "n_steps": len(df),
        "top1": float(df["top1"].mean()),
        "top5": float(df["top5"].mean()),
        "top10": float(df["top10"].mean()),
        "top50": float(df["top50"].mean()),
        "mean_n_predictions": float(df["n_predictions"].mean()),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"\n[save] {summary_path}")


if __name__ == "__main__":
    main()
