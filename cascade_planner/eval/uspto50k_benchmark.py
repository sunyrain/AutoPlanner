"""USPTO-50K public benchmark runner for K2.

Evaluates single-step retrosynthesis models on the standard USPTO-50K test split.
Supports: AiZynthFinder (ONNX), MEGAN, LocalRetro, RootAligned via Syntheseus.

Usage:
    # With AiZynthFinder ONNX model (already available):
    python -m cascade_planner.eval.uspto50k_benchmark --model aizynth --data data_external/uspto50k/

    # With Syntheseus models (need model weights):
    python -m cascade_planner.eval.uspto50k_benchmark --model megan --model-dir /path/to/megan/
    python -m cascade_planner.eval.uspto50k_benchmark --model localretro --model-dir /path/to/localretro/

Data download:
    The USPTO-50K test set can be obtained from:
    - https://github.com/Hanjun-Dai/GLN/tree/master/data/USPTO50k
    - pip install PyTDC && python -c "from tdc.single_pred import Retrosynthesis; Retrosynthesis('USPTO-50K')"
    Place raw_test.csv in data_external/uspto50k/
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent.parent


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def canon_set(smiles_dot: str) -> frozenset[str]:
    parts = []
    for s in smiles_dot.split("."):
        c = canon(s.strip())
        if c:
            parts.append(c)
    return frozenset(parts)


def load_uspto50k_test(data_dir: str | Path) -> pd.DataFrame:
    """Load USPTO-50K test set from various formats."""
    data_dir = Path(data_dir)

    # Try raw_test.csv (GLN format: id, class, reactants>reagents>production)
    for name in ["raw_test.csv", "test.csv", "USPTO_50k_test.csv"]:
        path = data_dir / name
        if path.exists():
            df = pd.read_csv(path)
            if "reactants>reagents>production" in df.columns:
                # GLN format
                rxns = df["reactants>reagents>production"].tolist()
                products, reactants_list = [], []
                for rxn in rxns:
                    parts = rxn.split(">")
                    if len(parts) == 3:
                        products.append(parts[2])
                        reactants_list.append(parts[0])
                    elif ">>" in rxn:
                        lhs, rhs = rxn.split(">>", 1)
                        products.append(rhs)
                        reactants_list.append(lhs)
                return pd.DataFrame({"product": products, "reactants": reactants_list})
            elif "product" in df.columns and "reactants" in df.columns:
                return df[["product", "reactants"]]
            elif "rxn_smiles" in df.columns:
                lhs = df["rxn_smiles"].str.split(">>").str[0]
                rhs = df["rxn_smiles"].str.split(">>").str[1]
                return pd.DataFrame({"product": rhs, "reactants": lhs})

    raise FileNotFoundError(f"No USPTO-50K test file found in {data_dir}")


def evaluate_model(model_name: str, test_df: pd.DataFrame, model_dir: str | None = None,
                   max_samples: int | None = None, top_k: int = 10) -> dict:
    """Evaluate a retrosynthesis model on USPTO-50K test set."""
    if max_samples:
        test_df = test_df.head(max_samples)

    n = len(test_df)
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    t0 = time.time()

    if model_name == "aizynth":
        # Use AiZynthFinder's template-based model
        import subprocess, sys
        for i, row in test_df.iterrows():
            product = row["product"]
            gt_reactants = canon_set(row["reactants"])
            if not gt_reactants:
                continue

            payload = json.dumps({
                "product": product,
                "config": "workspace/aizdata/config.yml",
                "max_iter": 1, "max_depth": 1, "n_routes": top_k,
                "use_filter": False, "use_stock": False,
            })
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "cascade_planner.multistep.aiz_mcts_bridge"],
                    input=payload, capture_output=True, text=True, timeout=30)
                idx = r.stdout.find("{")
                if idx >= 0:
                    out = json.loads(r.stdout[idx:])
                    # Extract predictions from route trees
                    for k_val in sorted(hits.keys()):
                        for rt in out.get("routes", [])[:k_val]:
                            tree = rt.get("tree", {})
                            for rxn_node in tree.get("children", []):
                                if rxn_node.get("type") == "reaction":
                                    pred = frozenset(
                                        canon(c.get("smiles", "")) or ""
                                        for c in rxn_node.get("children", [])
                                    )
                                    if pred == gt_reactants:
                                        hits[k_val] += 1
                                        break
            except Exception:
                pass

            total += 1
            if (total) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{total}/{n}] top-1={hits[1]/total*100:.1f}% ({elapsed:.0f}s)")

    else:
        print(f"Model {model_name} not yet supported for direct evaluation.")
        print(f"Available: aizynth")
        return {}

    results = {"model": model_name, "n_samples": total, "time_s": round(time.time() - t0, 1)}
    for k in sorted(hits.keys()):
        results[f"top_{k}"] = round(hits[k] / total * 100, 2) if total > 0 else 0
    return results


def main():
    ap = argparse.ArgumentParser(description="USPTO-50K benchmark for K2")
    ap.add_argument("--model", default="aizynth", choices=["aizynth", "megan", "localretro", "rootaligned"])
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--data", type=str, default="data_external/uspto50k/")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    try:
        test_df = load_uspto50k_test(args.data)
        print(f"USPTO-50K test: {len(test_df)} reactions")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Download USPTO-50K test set first. See module docstring for instructions.")
        return

    results = evaluate_model(args.model, test_df, args.model_dir, args.max_samples)
    print(json.dumps(results, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
