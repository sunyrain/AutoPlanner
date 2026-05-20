"""Build and audit real candidate caches for CascadeBoard strict benchmarks."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.paths import aizdata_dir, shared_dir

RDLogger.DisableLog("rdApp.*")


def canon_smiles(smiles: str | None, *, nostereo: bool = False) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if nostereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not nostereo)


def canon_set(dot_smiles: str | None, *, nostereo: bool = False) -> frozenset[str]:
    out: set[str] = set()
    for part in (dot_smiles or "").split("."):
        c = canon_smiles(part.strip(), nostereo=nostereo)
        if c:
            out.add(c)
    return frozenset(out)


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def split_reactants(reactant_dot: str) -> tuple[str, list[str]]:
    parts = sorted(canon_set(reactant_dot), key=lambda s: (-_heavy_atoms(s), s))
    if not parts:
        return "", []
    return parts[0], parts[1:]


def morgan_fp(smiles: str | None, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return arr
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TemplateRow:
    template_code: int
    retro_template: str
    classification: str
    library_occurrence: int


class EnzExpandONNX:
    """Small inference wrapper around the exported EnzExpand ONNX policy."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        template_path: str | Path | None = None,
    ) -> None:
        import onnxruntime as ort

        root = aizdata_dir()
        self.model_path = Path(model_path) if model_path else root / "enzexpand_model.onnx"
        self.template_path = Path(template_path) if template_path else root / "enzexpand_templates.csv.gz"
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.templates = self._load_templates(self.template_path)

    @staticmethod
    def _load_templates(path: Path) -> list[TemplateRow]:
        rows: list[TemplateRow] = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(TemplateRow(
                    template_code=int(row.get("template_code") or len(rows)),
                    retro_template=row["retro_template"],
                    classification=row.get("classification") or "enz.unrecognized",
                    library_occurrence=int(float(row.get("library_occurence") or 0)),
                ))
        return rows

    def top_templates(self, product: str, k: int) -> list[tuple[TemplateRow, float]]:
        fp = morgan_fp(product).reshape(1, -1)
        scores = self.session.run(None, {self.input_name: fp})[0][0]
        k = min(k, len(scores))
        idx = np.argsort(-scores)[:k]
        return [(self.templates[int(i)], float(scores[int(i)])) for i in idx]

    def predict(
        self,
        product: str,
        *,
        topk: int = 50,
        max_outcomes: int = 3,
        generalize: int = 1,
        drop_identity: bool = True,
    ) -> list[dict[str, Any]]:
        from cascade_planner.expand.enz_template import apply_template_to_product

        product_c = canon_smiles(product)
        if not product_c:
            return []

        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        rank = 0
        for tpl, score in self.top_templates(product_c, topk):
            outcomes = apply_template_to_product(
                tpl.retro_template,
                product_c,
                max_outcomes=max_outcomes,
                generalize=generalize,
            )
            for outcome in outcomes:
                key = tuple(sorted(outcome))
                if not key or key in seen:
                    continue
                if drop_identity and canon_set(".".join(key)) == canon_set(product_c):
                    continue
                seen.add(key)
                reactant_dot = ".".join(key)
                main, aux = split_reactants(reactant_dot)
                rxn = f"{reactant_dot}>>{product_c}"
                cls = tpl.classification.replace("enz.", "")
                if cls in {"unrecognized", "unknown", ""}:
                    cls = "other"
                rows.append({
                    "product": product_c,
                    "main_reactant": main,
                    "aux_reactants": aux,
                    "reaction_smiles": rxn,
                    "reaction_type": cls,
                    "ec": None,
                    "score": score,
                    "source": "enzexpand",
                    "rank": rank,
                    "template_code": tpl.template_code,
                    "template_classification": tpl.classification,
                    "template_occurrence": tpl.library_occurrence,
                    "generalize": generalize,
                })
                rank += 1
        return rows


def benchmark_seed_products(bench_path: str | Path) -> list[str]:
    bench = load_json(bench_path)
    seeds = []
    for item in bench:
        c = canon_smiles(item.get("target_smiles"))
        if c:
            seeds.append(c)
    return sorted(set(seeds))


def rc_cache_products(rc_cache_path: str | Path) -> list[str]:
    cache = load_json(rc_cache_path)
    products = []
    for key, rows in cache.items():
        c = canon_smiles(key)
        if c:
            products.append(c)
        for row in rows or []:
            c = canon_smiles(row.get("product"))
            if c:
                products.append(c)
    return sorted(set(products))


def build_enzexpand_cache(
    *,
    bench_path: str,
    rc_cache_path: str | None,
    output: str,
    topk: int = 50,
    max_outcomes: int = 3,
    max_depth: int = 2,
    max_products: int = 600,
    include_rc_products: bool = True,
    generalize: int = 1,
    drop_identity: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Build an EnzExpand cache without reading benchmark GT routes."""
    seeds = set(benchmark_seed_products(bench_path))
    if rc_cache_path and include_rc_products:
        seeds.update(rc_cache_products(rc_cache_path))

    engine = EnzExpandONNX()
    queue = deque((s, 0) for s in sorted(seeds))
    seen_products: set[str] = set()
    cache: dict[str, list[dict[str, Any]]] = {}

    t0 = time.time()
    while queue and len(seen_products) < max_products:
        product, depth = queue.popleft()
        product = canon_smiles(product) or ""
        if not product or product in seen_products:
            continue
        seen_products.add(product)
        rows = engine.predict(
            product,
            topk=topk,
            max_outcomes=max_outcomes,
            generalize=generalize,
            drop_identity=drop_identity,
        )
        cache[product] = rows
        if depth < max_depth:
            for row in rows[:10]:
                nxt = canon_smiles(row.get("main_reactant"))
                if nxt and nxt not in seen_products:
                    queue.append((nxt, depth + 1))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    nonempty = sum(bool(v) for v in cache.values())
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "bench_path": bench_path,
        "rc_cache_path": rc_cache_path,
        "output": output,
        "topk": topk,
        "max_outcomes": max_outcomes,
        "max_depth": max_depth,
        "max_products": max_products,
        "generalize": generalize,
        "drop_identity": drop_identity,
        "n_products": len(cache),
        "n_products_nonempty": nonempty,
        "n_candidates": sum(len(v) for v in cache.values()),
        "candidate_sources": {"enzexpand": sum(len(v) for v in cache.values())},
        "elapsed_s": round(time.time() - t0, 3),
    }
    Path(str(out) + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cache


def merge_candidate_caches(*caches: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for cache in caches:
        for product, rows in cache.items():
            product_c = canon_smiles(product)
            if not product_c:
                continue
            bucket = merged.setdefault(product_c, [])
            seen = {
                (
                    canon_smiles(r.get("main_reactant")) or "",
                    tuple(sorted(canon_smiles(x) or x for x in r.get("aux_reactants", []))),
                    r.get("source", ""),
                )
                for r in bucket
            }
            for row in rows or []:
                key = (
                    canon_smiles(row.get("main_reactant")) or "",
                    tuple(sorted(canon_smiles(x) or x for x in row.get("aux_reactants", []))),
                    row.get("source", ""),
                )
                if not key[0] or key in seen:
                    continue
                seen.add(key)
                copied = dict(row)
                copied["product"] = product_c
                copied["main_reactant"] = key[0]
                copied["aux_reactants"] = list(key[1])
                bucket.append(copied)
    for rows in merged.values():
        rows.sort(key=lambda r: (r.get("source") != "enzexpand", -float(r.get("score", 0.0))))
    return merged


def cache_summary(cache: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    source_counts = Counter()
    nonempty = 0
    for rows in cache.values():
        if rows:
            nonempty += 1
        for row in rows:
            source_counts[row.get("source", "unknown")] += 1
    return {
        "n_products": len(cache),
        "n_products_nonempty": nonempty,
        "n_candidates": sum(len(v) for v in cache.values()),
        "source_counts": dict(source_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--rc-cache", default="results/shared/retrochimera_candidates_depth2.json")
    ap.add_argument("--output", default=str(shared_dir() / "enzexpand_candidates_100.json"))
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--max-outcomes", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-products", type=int, default=600)
    ap.add_argument("--generalize", type=int, default=1)
    ap.add_argument("--no-rc-products", action="store_true")
    ap.add_argument("--keep-identity", action="store_true")
    args = ap.parse_args()
    cache = build_enzexpand_cache(
        bench_path=args.bench,
        rc_cache_path=args.rc_cache,
        output=args.output,
        topk=args.topk,
        max_outcomes=args.max_outcomes,
        max_depth=args.max_depth,
        max_products=args.max_products,
        include_rc_products=not args.no_rc_products,
        generalize=args.generalize,
        drop_identity=not args.keep_identity,
    )
    print(json.dumps(cache_summary(cache), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
