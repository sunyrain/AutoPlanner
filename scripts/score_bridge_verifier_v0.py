"""Score candidate chemo-enzymatic bridge pairs with bridge verifier v0.

Input rows must contain:

* chemical_smiles
* enzyme_smiles

Optional columns:

* chemical_inchikey
* enzyme_inchikey
* bridge_direction
* enzyme_ec_sample_json, enzyme_ec, ec
* label, label_type

Supported input/output suffixes: .jsonl, .json, .csv, .parquet.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

from train_bridge_verifier_v0 import MoleculeCache, build_matrix


RDLogger.DisableLog("rdApp.*")

DEFAULT_DIRECTION = "chemical_product_to_similar_enzyme_substrate"


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(".json input must be a list of objects")
        return data
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    raise ValueError(f"unsupported input suffix: {path}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return
    if suffix == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if suffix == ".csv":
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    if suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)
        return
    raise ValueError(f"unsupported output suffix: {path}")


def canonical_inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    return MolToInchiKey(mol)


def normalize_ec_json(row: dict[str, Any]) -> str:
    existing = row.get("enzyme_ec_sample_json")
    if existing:
        try:
            parsed = json.loads(str(existing))
            if isinstance(parsed, list):
                return json.dumps([str(item) for item in parsed if item])
        except Exception:
            pass
    for key in ["enzyme_ec", "ec", "ec_number"]:
        value = row.get(key)
        if value:
            tokens = []
            for sep in [";", ",", "|"]:
                if sep in str(value):
                    tokens = [token.strip() for token in str(value).split(sep) if token.strip()]
                    break
            if not tokens:
                tokens = [str(value).strip()]
            return json.dumps(tokens)
    return "[]"


def prepare_rows(rows: list[dict[str, Any]], cache: MoleculeCache) -> list[dict[str, Any]]:
    prepared = []
    for idx, row in enumerate(rows):
        out = dict(row)
        if not out.get("chemical_smiles") or not out.get("enzyme_smiles"):
            raise ValueError(f"row {idx} missing chemical_smiles or enzyme_smiles")
        out.setdefault("chemical_inchikey", canonical_inchikey(out["chemical_smiles"]))
        out.setdefault("enzyme_inchikey", canonical_inchikey(out["enzyme_smiles"]))
        out.setdefault("bridge_direction", DEFAULT_DIRECTION)
        out["enzyme_ec_sample_json"] = normalize_ec_json(out)
        out.setdefault("label", 0)
        out.setdefault("label_type", "candidate")
        out.setdefault("label_weight", 1.0)
        chem = cache.get(out["chemical_smiles"])
        enz = cache.get(out["enzyme_smiles"])
        if chem["fp"] is not None and enz["fp"] is not None:
            out["tanimoto"] = float(DataStructs.TanimotoSimilarity(chem["fp"], enz["fp"]))
        else:
            out["tanimoto"] = 0.0
        prepared.append(out)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Score bridge candidates with verifier v0")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib")
    parser.add_argument("--threshold", type=float, default=0.8409896871324669, help="default is valid-derived ~0.99 precision threshold")
    args = parser.parse_args()

    model_obj = joblib.load(args.model)
    model = model_obj["model"]
    cache = MoleculeCache()
    rows = prepare_rows(read_rows(Path(args.input)), cache)
    matrix, *_ = build_matrix(rows, cache)
    scores = model.predict_proba(matrix)[:, 1]
    out = []
    for row, score in zip(rows, scores):
        scored = dict(row)
        scored["verifier_score"] = float(score)
        scored["verifier_pass"] = bool(score >= args.threshold)
        scored["verifier_threshold"] = float(args.threshold)
        out.append(scored)
    write_rows(Path(args.output), out)
    print(json.dumps({
        "input_rows": len(rows),
        "output": args.output,
        "threshold": args.threshold,
        "passed": int(np.sum(scores >= args.threshold)),
        "score_min": float(np.min(scores)) if len(scores) else None,
        "score_mean": float(np.mean(scores)) if len(scores) else None,
        "score_max": float(np.max(scores)) if len(scores) else None,
    }, indent=2))


if __name__ == "__main__":
    main()
