"""Diagnostic re-grader: dumps EnzExpand-EM top-50 predictions on the
24 LOO queries and re-scores with 4 different matchers:

  exact       — frozenset(reactants) == frozenset(truth_reactants)
  no_cofactor — both sides stripped of common cofactors then compared
  main_only   — only the largest fragment by atom count compared (canonical)
  scaffold    — Murcko scaffold of largest fragment compared

This is a DIAGNOSTIC (single global model, NOT leave-one-DOI-out). It
exists to answer: "does the model's chemistry make sense once we stop
penalising cofactor accounting?".
"""
from __future__ import annotations

import collections
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.expand.enz_template import (apply_template_to_product,
                                                  canon_set, main_product,
                                                  morgan2, predict_topk,
                                                  train as train_mlp)
from cascade_planner.demo.final_integrated_eval import (build_template_pool,
                                                         expand_with_mlp)
from cascade_planner.expand.enzymemap_loader import (extract_templates_from_enzymemap,
                                                      load_filtered)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"

# Cofactor canon — canonical SMILES of common biological cofactors / spectator
# molecules to STRIP before comparing reactant sets.
_COFACTOR_SMILES_RAW = [
    "O", "[OH-]", "[H+]", "O=O", "[O-2]",
    "OC(=O)O", "O=C=O",                              # carbonate / CO2
    "N", "[NH4+]", "N#N",
    "[Na+]", "[K+]", "[Cl-]", "Cl",
    # NAD / NADP family (mapped/unmapped variants common in datasets)
    "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1",
    "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1",
    # ATP / ADP / AMP simplified
    "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O",
    "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O",
    "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O",
    # SAM / SAH / methyl donors (very rough)
    "C[S+](CCC(N)C(=O)O)CC1OC(n2cnc3c(N)ncnc32)C(O)C1O",
    # CoA core
    "CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(OP(=O)(O)O)C1O)C(O)C(=O)NCCC(=O)NCCS",
    # PLP / PMP
    "Cc1ncc(COP(=O)(O)O)c(C=O)c1O",
    "Cc1ncc(COP(=O)(O)O)c(CN)c1O",
]


def _canon(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except Exception:
        return smi


COFACTORS = set()
for s in _COFACTOR_SMILES_RAW:
    c = _canon(s)
    if c:
        COFACTORS.add(c)


def _strip_cofactors(reactant_set):
    return frozenset(r for r in reactant_set
                     if _canon(r) not in COFACTORS and r not in COFACTORS)


def _main_frag(smi):
    """Return canonical SMILES of the largest fragment by heavy-atom count."""
    if not smi:
        return ""
    if "." not in smi:
        return _canon(smi)
    try:
        frags = smi.split(".")
        mols = [(Chem.MolFromSmiles(f), f) for f in frags]
        mols = [(m, f) for m, f in mols if m is not None]
        if not mols:
            return _canon(smi)
        mols.sort(key=lambda x: x[0].GetNumHeavyAtoms(), reverse=True)
        return _canon(mols[0][1])
    except Exception:
        return _canon(smi)


def _main_frag_set(reactant_set):
    """Take the largest fragment from each reactant, after stripping cofactors."""
    stripped = _strip_cofactors(reactant_set)
    return frozenset(_main_frag(r) for r in stripped if r)


def _scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc, canonical=True) if sc else ""
    except Exception:
        return ""


def _scaffold_set(reactant_set):
    stripped = _strip_cofactors(reactant_set)
    return frozenset(s for s in (_scaffold(r) for r in stripped) if s)


def first_match(pred_iter, truth, transform):
    seen = []
    truth_t = transform(truth)
    if not truth_t:
        return None
    for cs in pred_iter:
        fs = frozenset(cs)
        if fs in seen:
            continue
        seen.append(fs)
        if transform(fs) == truth_t:
            return len(seen)
    return None


def main():
    print("=" * 70)
    print(" COFACTOR-AWARE RE-GRADER — diagnostic")
    print("=" * 70)
    df = pd.read_csv(RESULTS / "final_loo_eval.csv")
    print(f"loaded {len(df)} queries from final_loo_eval.csv")

    print(f"\n[cofactor canon] {len(COFACTORS)} entries")
    for c in sorted(COFACTORS, key=len)[:8]:
        print(f"  {c[:60]}")

    print("\n[load] snapshot + EnzymeMap")
    steps, _, _ = load_v2("cascade_dataset_v2.normalized.json")
    em_df = load_filtered(min_quality=0.95, only_single=True, ec1_balance=4000)
    em_rows = extract_templates_from_enzymemap(em_df)

    print("\n[train] ONE global EnzExpand-EM (diagnostic, NOT LOO)")
    train_enz = [s for s in steps if s.ec_number]
    Xb, yb, tplsB, _ = build_template_pool(train_enz, em_rows)
    print(f"  pool: {Xb.shape[0]} samples / {len(tplsB)} templates")
    modelB = train_mlp(Xb, yb, n_tpl=len(tplsB), epochs=25,
                       hidden=1024, dropout=0.4, batch=512, seed=0, verbose=True)

    print("\n[predict + regrade]")
    rows = []
    for _, q in df.iterrows():
        prod = q["product"]
        truth_str = q.get("truth_precursors", "") or ""
        truth = frozenset(truth_str.split(".")) if truth_str else frozenset()
        if not prod or not truth:
            continue

        # Materialise all top-K predictions once.
        preds = list(expand_with_mlp(modelB, tplsB, prod, k=50, max_outcomes=3))
        pred_sets = []
        seen = set()
        for cs in preds:
            fs = frozenset(cs)
            if fs not in seen:
                seen.add(fs)
                pred_sets.append(fs)

        ranks = {}
        for name, fn in [("exact", lambda s: s),
                         ("no_cofactor", _strip_cofactors),
                         ("main_only", _main_frag_set),
                         ("scaffold", _scaffold_set)]:
            ranks[name] = first_match(pred_sets, truth, fn)

        rows.append(dict(
            tag=q["tag"], doi=q["doi"],
            n_pred=len(pred_sets),
            truth=truth_str[:60],
            **{f"r_{k}": v for k, v in ranks.items()},
        ))
        print(f"  [{q['tag']:8s}] {q['doi'][:38]:38s}  "
              f"exact={ranks['exact']}  noCF={ranks['no_cofactor']}  "
              f"main={ranks['main_only']}  scaf={ranks['scaffold']}  "
              f"(n_pred={len(pred_sets)})")

    out = pd.DataFrame(rows)
    out_csv = RESULTS / "regrade_cofactor.csv"
    out.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")

    print("\n" + "=" * 70)
    print(" RE-GRADED TOP-K RECALL  (LEAKY diagnostic, single global model)")
    print("=" * 70)
    print(f"{'slice':12s}{'metric':14s}{'top1':>8s}{'top5':>8s}{'top10':>8s}{'top25':>8s}{'top50':>8s}")
    for slc, sub in [("enz", out[out["tag"].str.startswith("enz")]),
                     ("chem", out[out["tag"] == "chem"]),
                     ("ALL",  out)]:
        for m in ("exact", "no_cofactor", "main_only", "scaffold"):
            col = sub[f"r_{m}"]
            line = f"{slc:12s}{m:14s}"
            for k in (1, 5, 10, 25, 50):
                hit = (col.notna() & (col <= k)).mean() * 100
                line += f"{hit:7.0f}%"
            print(line)


if __name__ == "__main__":
    main()
