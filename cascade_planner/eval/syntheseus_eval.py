"""Evaluate syntheseus single-step models on the cascade benchmark.

Runs 2–3 models over every step (product -> reactants) and saves per-step top-K
hit indicators, joinable with `aizynthfinder_full_gpu_step_eval.csv` and
`enzexpand_step_eval_*.csv` for a unified head-to-head / routed / union analysis.

Output: results/syntheseus_step_eval_<model>.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem

from syntheseus import Molecule

# Older checkpoints (OpenNMT RootAligned, MEGAN, etc.) were saved without
# weights_only semantics. torch>=2.6 defaults to weights_only=True which
# refuses to load them. Monkey-patch torch.load to fall back to the legacy
# behaviour for these trusted local files.
import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(*args, **kwargs):
    # PL/lightning passes weights_only=True explicitly — override.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
_torch.load = _patched_torch_load

# Lazy import — each wrapper has heavy deps (dgl / OpenNMT / TF / hopfield-layers).
# Importing them all at once causes failures if any single one is missing.
def _load_class(name: str):
    mod = "syntheseus.reaction_prediction.inference"
    if name == "localretro":
        from syntheseus.reaction_prediction.inference import LocalRetroModel as C
    elif name == "chemformer":
        from syntheseus.reaction_prediction.inference import ChemformerModel as C
    elif name == "mhnreact":
        from syntheseus.reaction_prediction.inference import MHNreactModel as C
    elif name == "rootaligned":
        from syntheseus.reaction_prediction.inference import RootAlignedModel as C
    elif name == "megan":
        from syntheseus.reaction_prediction.inference import MEGANModel as C
    else:
        raise KeyError(name)
    return C

MODEL_NAMES = {"localretro", "chemformer", "mhnreact", "rootaligned", "megan"}

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


def canonical(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m)


def canonical_multi(smi: str) -> str:
    return ".".join(sorted(canonical(p) for p in smi.split(".") if p))


def load_steps(path: Path):
    """Use the canonical loader_v2 so we evaluate on the same set of
    rxn_smiles steps as the AiZynthFinder / EnzExpand evaluators."""
    import sys
    sys.path.insert(0, str(ROOT))
    from cascade_planner.data.loader_v2 import load_v2  # type: ignore
    steps_v2, _, _ = load_v2(str(path))
    rows = []
    for s in steps_v2:
        rxn = s.rxn_smiles
        if not rxn or ">>" not in rxn:
            continue
        lhs, rhs = rxn.split(">>", 1)
        lhs = lhs.strip(); rhs = rhs.strip()
        if not lhs or not rhs:
            continue
        # main product = largest fragment by heavy atoms
        prod_parts = [p for p in rhs.split(".") if p]
        try:
            mols = [(p, Chem.MolFromSmiles(p)) for p in prod_parts]
            mols = [(p, m) for p, m in mols if m is not None]
            if not mols:
                continue
            prod = max(mols, key=lambda pm: pm[1].GetNumHeavyAtoms())[0]
        except Exception:
            continue
        rows.append({
            "doi": s.doi,
            "cascade_id": s.cascade_id,
            "step_id": s.step_id,
            "step_index": s.step_index,
            "product": prod,
            "truth_reactants": lhs,
            "ec_number": s.ec_number or "",
            "transformation": s.transformation_superclass or "",
        })
    return rows


def evaluate(model_name: str, model, steps, num_results: int = 50):
    out_rows = []
    t0 = time.time()
    for i, s in enumerate(steps):
        prod_c = canonical(s["product"])
        truth_c = canonical_multi(s["truth_reactants"])
        try:
            [predictions] = model([Molecule(prod_c)], num_results=num_results)
        except Exception as e:
            out_rows.append({**{k: s[k] for k in ("doi", "cascade_id", "step_id", "step_index", "ec_number", "transformation")},
                             "n_predictions": 0, "rank_of_truth": None,
                             "top1": 0, "top5": 0, "top10": 0, "top50": 0,
                             "product": prod_c, "truth_reactants": truth_c,
                             "top1_pred": "", "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        pred_smis = []
        rank = None
        for j, p in enumerate(predictions, 1):
            pr = ".".join(sorted(canonical(r.smiles) for r in p.reactants))
            pred_smis.append(pr)
            if rank is None and pr == truth_c:
                rank = j
        row = {**{k: s[k] for k in ("doi", "cascade_id", "step_id", "step_index", "ec_number", "transformation")},
               "n_predictions": len(predictions),
               "rank_of_truth": rank,
               "top1": int(rank is not None and rank <= 1),
               "top5": int(rank is not None and rank <= 5),
               "top10": int(rank is not None and rank <= 10),
               "top50": int(rank is not None and rank <= 50),
               "product": prod_c,
               "truth_reactants": truth_c,
               "top1_pred": pred_smis[0] if pred_smis else "",
               "error": None}
        out_rows.append(row)
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            n1 = sum(r["top1"] for r in out_rows)
            n10 = sum(r["top10"] for r in out_rows)
            print(f"  [{model_name} {i+1}/{len(steps)}] {dt:.1f}s  top1={n1} top10={n10}")
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--models", nargs="+", default=["localretro"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--num-results", type=int, default=50)
    args = ap.parse_args()

    steps = load_steps(ROOT / args.data)
    if args.limit > 0:
        steps = steps[: args.limit]
    print(f"[data] loaded {len(steps)} steps from {args.data}")

    for mname in args.models:
        if mname not in MODEL_NAMES:
            raise SystemExit(f"unknown model {mname}; choose from {MODEL_NAMES}")
        cls = _load_class(mname)
        print(f"\n[model] loading {mname} ({cls.__name__})")
        t0 = time.time()
        model = cls()
        print(f"  loaded in {time.time()-t0:.1f}s  model_dir={getattr(model, 'model_dir', '?')}")

        rows = evaluate(mname, model, steps, num_results=args.num_results)
        df = pd.DataFrame(rows)
        out = RESULTS / f"syntheseus_step_eval_{mname}.csv"
        df.to_csv(out, index=False)
        print(f"[save] {out}  n={len(df)}")
        for k in (1, 5, 10, 50):
            print(f"  top-{k:<2d} = {df[f'top{k}'].mean()*100:5.2f}%")
        print("  errors:", df["error"].notna().sum())


if __name__ == "__main__":
    main()
