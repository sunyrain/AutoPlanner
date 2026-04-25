"""Freeze a 100-target multi-step benchmark from cascade_dataset_v2.strict.json.

Selection rules:
  - Cascade has >=2 valid steps (multi-step is the point).
  - All steps in cascade are in the strict subset (no rxn_status issues, no
    multi-EC etc — i.e. cascade survives strict filter intact).
  - Take the FINAL output_species smiles (canonicalized) as the target.
  - GT route = the ordered list of (rxn_smiles, ec_number, transformation).
  - Stratify across route_domain (chemoenzymatic, biocatalytic, chemical) and
    cascade depth (2, 3, 4+) where possible.
  - DOI-disjoint from any held-out used in earlier work (best effort: random
    seed 0, cap 1 cascade per DOI for diversity).

Output:
  data/benchmark_v2_100.json — list of {target_smiles, doi, cascade_id,
                                gt_route, route_domain, depth}
  data/benchmark_v2_100.csv — flat table for inspection
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon(smi: str) -> str | None:
    if not smi:
        return None
    parts = []
    for s in smi.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        parts.append(Chem.MolToSmiles(m, canonical=True))
    return ".".join(sorted(parts)) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="cascade_dataset_v2.strict.json")
    ap.add_argument("--out-json", default="data/benchmark_v2_100.json")
    ap.add_argument("--out-csv", default="data/benchmark_v2_100.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    cands = []
    for art in data.get("records_kept", []):
        doi = art.get("doi") or art.get("title", "unknown")
        for c in art.get("cascades", []):
            steps = c.get("steps", []) or []
            if len(steps) < 2:
                continue
            # final target = last step's LARGEST heavy-atom product (canonical)
            # (first-product selection picked byproducts like water/CO2)
            last = steps[-1]
            tgt = None
            best_ha = -1
            outs = last.get("output_species") or []
            for o in outs:
                role = (o.get("species_role") or "").lower()
                if role != "product":
                    continue
                sm = canon(o.get("smiles") or "")
                if not sm:
                    continue
                m = Chem.MolFromSmiles(sm)
                if m is None:
                    continue
                ha = m.GetNumHeavyAtoms()
                if ha > best_ha:
                    tgt, best_ha = sm, ha
            if not tgt or best_ha < 6:
                # fallback to RHS of last rxn (largest fragment)
                rxn = last.get("rxn_smiles") or ""
                if ">>" in rxn:
                    for s in rxn.split(">>")[1].split("."):
                        sm = canon(s)
                        if not sm:
                            continue
                        m = Chem.MolFromSmiles(sm)
                        if m is None:
                            continue
                        ha = m.GetNumHeavyAtoms()
                        if ha > best_ha:
                            tgt, best_ha = sm, ha
            if not tgt or best_ha < 6:
                continue
            route = []
            for s in steps:
                ec = None
                cats = s.get("catalyst_components") or []
                if cats:
                    ec = cats[0].get("ec_number")
                route.append({
                    "rxn_smiles": s.get("rxn_smiles"),
                    "ec_number": ec,
                    "transformation": s.get("transformation_superclass"),
                    "step_role": s.get("step_role"),
                })
            cands.append({
                "doi": doi,
                "cascade_id": c.get("cascade_id"),
                "target_smiles": tgt,
                "route_domain": c.get("route_domain"),
                "operation_mode": c.get("operation_mode"),
                "depth": len(steps),
                "gt_route": route,
            })

    print(f"[scan] {len(cands)} multi-step cascades pass strict filter")

    # cap 1 per DOI
    random.seed(args.seed)
    random.shuffle(cands)
    seen_doi = set()
    uniq = []
    for c in cands:
        if c["doi"] in seen_doi:
            continue
        seen_doi.add(c["doi"])
        uniq.append(c)
    print(f"[uniq-doi] {len(uniq)} candidates after 1-per-DOI cap")

    # stratify by depth bucket and route_domain
    buckets = defaultdict(list)
    for c in uniq:
        d = c["depth"]
        bk = (c.get("route_domain") or "unknown",
              "2" if d == 2 else "3" if d == 3 else "4+")
        buckets[bk].append(c)
    print(f"[strata] {len(buckets)} (route_domain, depth) buckets")

    # round-robin draw
    keys = sorted(buckets.keys())
    out = []
    while len(out) < args.n:
        progress = False
        for k in keys:
            if buckets[k] and len(out) < args.n:
                out.append(buckets[k].pop()); progress = True
        if not progress:
            break
    print(f"[freeze] {len(out)} targets selected")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat = [{
        "doi": c["doi"], "cascade_id": c["cascade_id"], "depth": c["depth"],
        "route_domain": c.get("route_domain"),
        "target_smiles": c["target_smiles"],
        "ec_chain": ",".join((s.get("ec_number") or "-") for s in c["gt_route"]),
        "tx_chain": ",".join((s.get("transformation") or "-") for s in c["gt_route"]),
    } for c in out]
    pd.DataFrame(flat).to_csv(args.out_csv, index=False)
    print(f"[save] {args.out_json}\n[save] {args.out_csv}")

    # summary
    print("\n=== composition ===")
    print(pd.DataFrame(flat).groupby(["route_domain", "depth"]).size().to_string())


if __name__ == "__main__":
    main()
