"""Spot-check why rdchiral template_fires_fail: inspect actual SMILES mismatch."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger

from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.expand.enz_template import (
    apply_template_to_product, canon_set, extract_templates, main_product,
)
from cascade_planner.paths import shared_dir

RDLogger.DisableLog("rdApp.*")


def main():
    df = pd.read_csv("results/v2/recall_diag/mf2_top50/recall_per_step.csv")
    fails = df[df["bucket"] == "template_fires_fail"].copy()
    print(f"fires_fail steps: {len(fails)}")
    print(fails.groupby("ec1").size())

    steps, _, _ = load_v2("cascade_dataset_v2.normalized.json")
    enz = {(s.doi, s.step_id): s for s in steps if s.ec_number}
    pairs = extract_templates([s for s in steps if s.ec_number],
                              mapped_cache=shared_dir() / "enzexpand_atommap_cache.json")
    # build step -> tpl
    enz_list = [s for s in steps if s.ec_number]
    tpl_by_idx = {i: t for i, t in pairs}

    examples = []
    for _, row in fails.head(30).iterrows():
        key = (row["doi"], str(row["step_id"]))
        s = enz.get(key)
        if s is None:
            # step_id may be int
            for k in enz:
                if k[0] == row["doi"] and str(k[1]) == str(row["step_id"]):
                    s = enz[k]; break
        if s is None:
            continue
        # find index
        idx = next((i for i, x in enumerate(enz_list) if x is s), None)
        if idx is None:
            continue
        tpl = tpl_by_idx.get(idx)
        if not tpl:
            continue
        prod = main_product(s.rxn_smiles)
        gt_reactants = canon_set(s.rxn_smiles.split(">>", 1)[0])
        outs = apply_template_to_product(tpl, prod, max_outcomes=50)
        examples.append(dict(
            ec1=row["ec1"],
            product=prod,
            gt_reactants=".".join(sorted(gt_reactants)) if gt_reactants else "",
            n_outcomes=len(outs),
            first_outcome=".".join(sorted(next(iter(outs)))) if outs else "",
        ))

    exdf = pd.DataFrame(examples)
    out = Path("results/v2/recall_diag/fires_fail_examples.csv")
    exdf.to_csv(out, index=False)
    print(f"[save] {out}")
    print("\n--- EC1 breakdown of outcome counts ---")
    print(exdf.groupby("ec1")["n_outcomes"].describe().round(1).to_string())
    print("\n--- Sample rows ---")
    for _, r in exdf.head(10).iterrows():
        print(f"\nEC{r['ec1']}  prod={r['product'][:50]}")
        print(f"  GT:   {r['gt_reactants'][:80]}")
        print(f"  gen:  {r['first_outcome'][:80]}")


if __name__ == "__main__":
    main()
