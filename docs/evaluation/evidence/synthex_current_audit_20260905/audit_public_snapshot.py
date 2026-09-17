"""Read-only, reproducible audit of downloaded public SynthAtlas JSON.

This measures public field coverage, not reaction feasibility or SynthEx success.
Run from the repository root with Python + RDKit.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from rdkit import Chem

BASE = Path(__file__).resolve().parent


def read(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def inspect(route_id, leaf_index, route_index):
    route = read(route_id + ".json")
    steps = route["steps"]
    products = {s["product"] for s in steps}
    precursors = {r for s in steps for r in s["reactants"]}
    leaves = sorted(precursors - products)
    cats = {leaf: leaf_index.get(leaf, {}).get("cat", "missing") for leaf in leaves}
    invalid = []
    target_products = []
    for step in steps:
        parts = step["rxn_smiles"].split(">")
        if len(parts) != 3 or canonical(parts[0]) is None or canonical(parts[-1]) is None:
            invalid.append(step["idx"])
        elif canonical(parts[-1]) == canonical(route["target_smiles"]):
            target_products.append(step["idx"])
    return {
        "id": route_id,
        "name": route["name"],
        "url": "https://synthatlas.epfl.ch/#/route/" + route_id,
        "feasibility": route["feasibility"],
        "solved": route["solved"],
        "productive": route["productive"],
        "index_productive": route_index[route_id]["productive"],
        "productive_index_detail_mismatch": route["productive"] != route_index[route_id]["productive"],
        "steps": len(steps),
        "key_steps": sum(bool(s["is_key"]) for s in steps),
        "null_step_critic_verdict": sum(s.get("critic_verdict") is None for s in steps),
        "nonempty_step_critic_reason": sum(bool(s.get("critic_reason")) for s in steps),
        "nonempty_conditions": sum(bool(s.get("conditions")) for s in steps),
        "condition_assessment": dict(collections.Counter(s.get("assessment", "") for s in steps)),
        "close_reaxys_positive": sum(s.get("reaxys_close_count", 0) > 0 for s in steps),
        "related_only_reaxys": sum(s.get("reaxys_close_count", 0) == 0 and s.get("reaxys_related_count", 0) > 0 for s in steps),
        "key_steps_without_close_reaxys": sum(s["is_key"] and s.get("reaxys_close_count", 0) == 0 for s in steps),
        "step_field_union": sorted(set().union(*(s.keys() for s in steps))),
        "route_fields": sorted(route),
        "invalid_reaction_smiles": invalid,
        "exact_stereo_target_product_steps": target_products,
        "leaf_categories": cats,
        "all_actual_leaves_buyable_or_obtainable": bool(cats) and all(c in {"buyable", "obtainable"} for c in cats.values()),
        "all_actual_leaves_buyable_obtainable_or_small": bool(cats) and all(c in {"buyable", "obtainable", "small"} for c in cats.values()),
    }


def aggregate(rows):
    summed = ["steps", "key_steps", "null_step_critic_verdict", "nonempty_step_critic_reason", "nonempty_conditions", "close_reaxys_positive", "related_only_reaxys", "key_steps_without_close_reaxys"]
    return {
        "routes": len(rows),
        **{key: sum(r[key] for r in rows) for key in summed},
        "routes_with_parse_error": sum(bool(r["invalid_reaction_smiles"]) for r in rows),
        "routes_with_exact_stereo_target_product": sum(bool(r["exact_stereo_target_product_steps"]) for r in rows),
        "condition_assessment": dict(sum((collections.Counter(r["condition_assessment"]) for r in rows), collections.Counter())),
        "productive_routes": sum(r["productive"] for r in rows),
        "productive_index_detail_mismatches": [r["id"] for r in rows if r["productive_index_detail_mismatch"]],
        "productive_routes_with_small_actual_leaf": [r["id"] for r in rows if r["productive"] and "small" in r["leaf_categories"].values()],
        "productive_routes_with_unavailable_actual_leaf": [r["id"] for r in rows if r["productive"] and "unavailable" in r["leaf_categories"].values()],
        "routes_whose_leaves_are_buyable_or_obtainable_but_solved_false": [r["id"] for r in rows if r["all_actual_leaves_buyable_or_obtainable"] and not r["solved"]],
    }


def main():
    index = read("index.json")
    leaf_index = read("leaf_routes.json")
    plan = read("sample-plan.json")
    route_index = {r["id"]: r for r in index}
    random_rows = [inspect(r, leaf_index, route_index) for r in plan["random_ids"]]
    example_rows = [inspect(r, leaf_index, route_index) for r in plan["example_ids"]]
    result = {
        "scope": "Current public snapshot field audit; no experimental or chemical success-rate inference.",
        "data_version": read("manifest.json")["dataVer"],
        "index": {
            "routes": len(index), "targets": len({r["target"] for r in index}),
            "steps": sum(r["total_steps"] for r in index),
            "solved_true": sum(r["solved"] for r in index),
            "productive_true": sum(r["productive"] for r in index),
            "feasibility": dict(collections.Counter(r["feasibility"] for r in index)),
        },
        "leaf_categories": dict(collections.Counter(r.get("cat") for r in leaf_index.values())),
        "sample_plan": plan,
        "random_sample": aggregate(random_rows),
        "purposive_examples": aggregate(example_rows),
        "random_rows": random_rows,
        "example_rows": example_rows,
        "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE.iterdir()) if p.suffix in {".json", ".txt", ".js"} and p.name not in {"audit-results.json"}},
    }
    (BASE / "audit-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["index", "leaf_categories", "random_sample", "purposive_examples"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
