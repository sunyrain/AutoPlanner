"""Current candidate-cache normalization helpers.

The V4 benchmark and supervision paths use this module only for deterministic
SMILES normalization, cache merging, and cache summaries. Model-specific cache
builders belong to explicit research or legacy namespaces.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from rdkit import Chem


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
        canonical = canon_smiles(part.strip(), nostereo=nostereo)
        if canonical:
            out.add(canonical)
    return frozenset(out)


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
                    canon_smiles(row.get("main_reactant")) or "",
                    tuple(sorted(canon_smiles(item) or item for item in row.get("aux_reactants", []))),
                    row.get("source", ""),
                )
                for row in bucket
            }
            for row in rows or []:
                key = (
                    canon_smiles(row.get("main_reactant")) or "",
                    tuple(sorted(canon_smiles(item) or item for item in row.get("aux_reactants", []))),
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
        rows.sort(
            key=lambda row: (
                -_candidate_score(row),
                str(row.get("source") or ""),
                str(row.get("main_reactant") or ""),
                tuple(str(item) for item in row.get("aux_reactants", [])),
            )
        )
    return merged


def _candidate_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
        "n_candidates": sum(len(rows) for rows in cache.values()),
        "source_counts": dict(source_counts),
    }


__all__ = ["cache_summary", "canon_set", "canon_smiles", "merge_candidate_caches"]
