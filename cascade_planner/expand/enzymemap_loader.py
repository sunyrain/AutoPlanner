"""Load EnzymeMap (Heid et al. 2023, Zenodo 8254726) and extract retro-templates.

EnzymeMap CSV columns we use:
  - mapped       : atom-mapped rxn SMILES (already mapped → no rxnmapper needed)
  - unmapped     : clean rxn SMILES
  - ec_num       : EC number, e.g. "1.1.1.1"
  - quality      : 0..1 confidence
  - steps        : "single" / "multi"

Output format compatible with cascade_planner.expand.enz_template:
  list of dicts {product_smi, template_smarts, ec_num, source}.
"""
from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
ENZYMEMAP_CSV = ROOT / "data_external" / "enzymemap" / "enzymemap_v2_brenda2023.csv.gz"
CACHE_TPL = ROOT / "results" / "enzymemap_templates.json.gz"


def _main_product(rxn: str) -> str | None:
    if not rxn or ">>" not in rxn:
        return None
    rhs = rxn.split(">>", 1)[1]
    best, best_n = None, -1
    for s in rhs.split("."):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n = m.GetNumHeavyAtoms()
        if n > best_n:
            best, best_n = Chem.MolToSmiles(m), n
    return best


def load_filtered(min_quality: float = 0.95, only_single: bool = True,
                  ec1_balance: int | None = 4000) -> pd.DataFrame:
    """Load EnzymeMap CSV with quality+single-step filter, optionally cap per EC1."""
    df = pd.read_csv(ENZYMEMAP_CSV,
                     usecols=["mapped", "unmapped", "ec_num", "quality", "steps", "natural"])
    n0 = len(df)
    df = df[df["quality"] >= min_quality]
    if only_single:
        df = df[df["steps"] == "single"]
    df = df.dropna(subset=["mapped", "ec_num"])
    df = df.drop_duplicates(subset=["unmapped"])
    if ec1_balance:
        df["_ec1"] = df["ec_num"].astype(str).str.split(".").str[0]
        df = (df.groupby("_ec1", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), ec1_balance), random_state=42)))
        df = df.drop(columns=["_ec1"], errors="ignore")
    print(f"[enzymemap] {n0} → {len(df)} after filter "
          f"(quality>={min_quality}, single={only_single}, cap_per_EC1={ec1_balance})")
    return df.reset_index(drop=True)


def extract_templates_from_enzymemap(df: pd.DataFrame, *, cache_path: Path = CACHE_TPL,
                                      max_rows: int | None = None) -> list[dict]:
    """Run rdchiral template extraction on EnzymeMap mapped rxns. Cached on disk."""
    from rdchiral.template_extractor import extract_from_reaction

    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[enzymemap] loaded cache: {len(cached)} templates from {cache_path}")
        return cached

    rows = []
    n_total = len(df) if max_rows is None else min(len(df), max_rows)
    print(f"[enzymemap] extracting templates from {n_total} rxns ...")
    for i, r in enumerate(df.itertuples(index=False)):
        if max_rows is not None and i >= max_rows:
            break
        mapped = r.mapped
        if not mapped or ">>" not in mapped:
            continue
        lhs, rhs = mapped.split(">>", 1)
        try:
            res = extract_from_reaction({"_id": f"em{i}", "reactants": lhs, "products": rhs})
            tpl = res.get("reaction_smarts") if res else None
            if not tpl:
                continue
            prod = _main_product(r.unmapped)
            if not prod:
                continue
            rows.append(dict(product=prod, template=tpl,
                             ec_num=str(r.ec_num), source="enzymemap"))
        except Exception:
            continue
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n_total}  → {len(rows)} ok")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        json.dump(rows, f)
    print(f"[enzymemap] {len(rows)} templates → {cache_path}")
    return rows


if __name__ == "__main__":
    df = load_filtered(min_quality=0.95, only_single=True, ec1_balance=4000)
    rows = extract_templates_from_enzymemap(df)
    import collections
    cnt = collections.Counter(r["ec_num"].split(".")[0] for r in rows)
    print(f"\nTemplates by EC1: {dict(cnt)}")
    cnt_t = collections.Counter(r["template"] for r in rows)
    print(f"Unique templates: {len(cnt_t)}; top-5 freq: {cnt_t.most_common(5)[:5]}")
