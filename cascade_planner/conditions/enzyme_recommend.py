"""Enzyme recommendation from enriched cascade snapshot.

Given a target SMILES (or a transformation_superclass), returns the most
common (EC, organism, Uniprot accession) tuples observed in the dataset
for similar reactions, ranked by frequency. Uses the Uniprot-enriched
snapshot produced by `cascade_planner.data.enrich_uniprot`.

Usage:
    python -m cascade_planner.conditions.enzyme_recommend \
        --data cascade_dataset_v2.normalized.json \
        --transform "ester hydrolysis" --topk 5
    python -m cascade_planner.conditions.enzyme_recommend \
        --data cascade_dataset_v2.normalized.json \
        --product "OC(C(=O)O)c1ccccc1" --topk 5
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def _mol(smi: str):
    if not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _fp(mol):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def collect_records(data: dict) -> list[dict]:
    rows = []
    for r in data.get("records_kept", []):
        for c in r.get("cascades", []):
            for s in c.get("steps", []):
                cats = s.get("catalyst_components", []) or []
                for cat in cats:
                    if (cat.get("catalyst_class") or "").lower() == "enzyme":
                        rows.append({
                            "doi": r.get("doi"),
                            "step_id": s.get("step_id"),
                            "transform": s.get("transformation_name"),
                            "transform_super": s.get("transformation_superclass"),
                            "rxn_smi": s.get("rxn_smiles"),
                            "ec": cat.get("ec_number"),
                            "organism": cat.get("organism"),
                            "uniprot_id": cat.get("uniprot_id"),
                            "uniprot_entry_name": cat.get("uniprot_entry_name"),
                            "uniprot_protein_name": cat.get("uniprot_protein_name"),
                            "uniprot_match_quality": cat.get("uniprot_match_quality"),
                            "engineering_status": cat.get("engineering_status"),
                            "T": (s.get("step_conditions") or {}).get("temperature_c"),
                            "pH": (s.get("step_conditions") or {}).get("ph"),
                            "yield": (s.get("step_outcome") or {}).get("step_yield_percent"),
                            "ee": (s.get("step_outcome") or {}).get("step_ee_percent"),
                        })
    return rows


def by_transform(rows: list[dict], q: str, topk: int) -> list[dict]:
    q = q.lower()
    matched = [r for r in rows
               if (r["transform"] and q in r["transform"].lower())
               or (r["transform_super"] and q in r["transform_super"].lower())]
    counter = collections.Counter()
    examples: dict = collections.defaultdict(list)
    for r in matched:
        key = (r["ec"], r["organism"], r["uniprot_id"])
        counter[key] += 1
        examples[key].append(r)
    items = []
    for (ec, org, up), cnt in counter.most_common(topk):
        ex = examples[(ec, org, up)][0]
        items.append({
            "ec": ec, "organism": org, "uniprot_id": up,
            "protein_name": ex.get("uniprot_protein_name"),
            "uniprot_match_quality": ex.get("uniprot_match_quality"),
            "n_records": cnt,
            "example_doi": ex["doi"],
            "example_rxn": ex["rxn_smi"],
            "T_C": ex["T"], "pH": ex["pH"],
        })
    return items


def by_product(rows: list[dict], product_smi: str, topk: int) -> list[dict]:
    qmol = _mol(product_smi)
    qfp = _fp(qmol)
    if qfp is None:
        return []
    scored = []
    for r in rows:
        rxn = r.get("rxn_smi") or ""
        if ">>" not in rxn:
            continue
        rhs = rxn.split(">>", 1)[1]
        # take main fragment by atom count
        frags = [_mol(x) for x in rhs.split(".") if x]
        frags = [m for m in frags if m is not None]
        if not frags:
            continue
        main = max(frags, key=lambda m: m.GetNumAtoms())
        fp = _fp(main)
        if fp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(qfp, fp)
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    counter = collections.Counter()
    best_sim: dict = {}
    examples: dict = collections.defaultdict(list)
    for sim, r in scored[:200]:  # consider top-200 nearest
        key = (r["ec"], r["organism"], r["uniprot_id"])
        counter[key] += 1
        if key not in best_sim or sim > best_sim[key]:
            best_sim[key] = sim
        examples[key].append(r)
    # rank by best_sim then by count
    keys = sorted(counter.keys(), key=lambda k: (-best_sim.get(k, 0), -counter[k]))[:topk]
    out = []
    for k in keys:
        ec, org, up = k
        ex = examples[k][0]
        out.append({
            "ec": ec, "organism": org, "uniprot_id": up,
            "protein_name": ex.get("uniprot_protein_name"),
            "uniprot_match_quality": ex.get("uniprot_match_quality"),
            "best_tanimoto": round(best_sim[k], 3),
            "n_supporting_records": counter[k],
            "example_doi": ex["doi"],
            "example_rxn": ex["rxn_smi"],
            "T_C": ex["T"], "pH": ex["pH"], "ee_pct": ex["ee"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--transform", default=None,
                    help="Substring of transformation_name or _superclass (e.g. 'ester hydrolysis')")
    ap.add_argument("--product", default=None,
                    help="Target product SMILES — ranks by morgan2 Tanimoto to dataset main products")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows = collect_records(data)
    print(f"[recommend] enzyme records in dataset: {len(rows)}")
    n_with_up = sum(1 for r in rows if r["uniprot_id"])
    print(f"[recommend] with Uniprot ID: {n_with_up} ({100*n_with_up/max(1,len(rows)):.1f}%)")

    if args.transform:
        items = by_transform(rows, args.transform, args.topk)
        print(f"\n--- TOP-{args.topk} by transformation '{args.transform}' ---")
    elif args.product:
        items = by_product(rows, args.product, args.topk)
        print(f"\n--- TOP-{args.topk} by product similarity to {args.product} ---")
    else:
        print("Provide --transform or --product")
        return

    for i, it in enumerate(items, 1):
        print(f"\n#{i}  EC={it['ec']}  organism={it['organism']}")
        print(f"    Uniprot: {it.get('uniprot_id')}  ({it.get('uniprot_match_quality')})")
        print(f"    Protein: {it.get('protein_name')}")
        if "best_tanimoto" in it:
            print(f"    sim={it['best_tanimoto']}  records={it['n_supporting_records']}")
        else:
            print(f"    n_records={it['n_records']}")
        print(f"    example: {it['example_doi']}  T={it['T_C']}  pH={it['pH']}")

    if args.out:
        Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
