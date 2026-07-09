"""USPTO-50K K2 evaluator using Syntheseus single-step models.

Runs MEGAN / RootAligned / LocalRetro (any subset) on USPTO-50K test
and reports top-1/3/5/10/50 reactant-set match accuracy.

Optionally combines per-model predictions into a RetroChimera-style ensemble
via softmax-normalized score sum.

Usage:
    # single model, smoke
    python -m cascade_planner.eval.uspto50k_syntheseus \
        --model megan --max-samples 100

    # ensemble all 3, full set
    python -m cascade_planner.eval.uspto50k_syntheseus \
        --model megan rootaligned localretro --ensemble \
        --output results/v2/k2_uspto50k_ensemble.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Patch torch.load for legacy checkpoints (same trick as syntheseus_eval.py)
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

from syntheseus import Molecule


ROOT = Path(__file__).resolve().parent.parent.parent


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def canon_set(smiles_dot: str) -> frozenset[str]:
    return frozenset(canon(s) for s in smiles_dot.split(".") if s)


def load_class(name: str):
    name = name.lower()
    from syntheseus.reaction_prediction.inference import (
        MEGANModel, RootAlignedModel, LocalRetroModel,
        ChemformerModel, MHNreactModel,
    )
    return {
        "megan": MEGANModel,
        "rootaligned": RootAlignedModel,
        "localretro": LocalRetroModel,
        "chemformer": ChemformerModel,
        "mhnreact": MHNreactModel,
    }[name]


def load_model(name: str, model_dir: str | None, device: str | None = None):
    Cls = load_class(name)
    kwargs = {}
    if model_dir:
        kwargs["model_dir"] = model_dir
    if device:
        kwargs["device"] = device
    return Cls(**kwargs)


def load_uspto50k_test(data_dir: str | Path, max_samples: int | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir)
    # Prefer TDC format: input,output  (input = product, output = reactants)
    for name in ("tdc_test.csv",):
        p = data_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if {"input", "output"} <= set(df.columns):
                df = df.rename(columns={"input": "product", "output": "reactants"})
                if max_samples:
                    df = df.head(max_samples)
                return df[["product", "reactants"]].reset_index(drop=True)
    # Fallback: GLN raw_test.csv
    for name in ("raw_test.csv", "test.csv"):
        p = data_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if "reactants>reagents>production" in df.columns:
                rxns = df["reactants>reagents>production"].tolist()
                rows = []
                for rxn in rxns:
                    parts = rxn.split(">")
                    if len(parts) == 3:
                        rows.append({"product": parts[2], "reactants": parts[0]})
                d = pd.DataFrame(rows)
                if max_samples:
                    d = d.head(max_samples)
                return d.reset_index(drop=True)
    raise FileNotFoundError(f"No USPTO-50K test CSV in {data_dir}")


def run_model(model, products: list[str], top_k: int, batch_size: int = 1, debug: bool = False):
    """Return list[list[(reactants_set:frozenset, score:float)]]."""
    out = []
    err_seen = 0
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        mols = [Molecule(canon(p)) for p in batch]
        try:
            preds = model(mols, num_results=top_k)
        except Exception as e:
            err_seen += 1
            if debug and err_seen <= 3:
                import traceback
                print(f"[run_model] error on batch starting idx={i}: {type(e).__name__}: {e}")
                traceback.print_exc()
            for _ in batch:
                out.append([])
            continue
        for plist in preds:
            ranked = []
            for rxn in plist:
                key = frozenset(canon(r.smiles) for r in rxn.reactants)
                meta = getattr(rxn, "metadata", {}) or {}
                if "probability" in meta:
                    s = float(meta["probability"])
                elif hasattr(rxn, "log_prob") and rxn.log_prob is not None:
                    s = float(np.exp(rxn.log_prob))
                else:
                    s = 1.0 / (len(ranked) + 1)
                ranked.append((key, s))
            out.append(ranked)
    return out


def compute_hits(preds: list[list[tuple[frozenset, float]]], truths: list[frozenset], ks=(1, 3, 5, 10, 50)):
    hits = {k: 0 for k in ks}
    n = len(preds)
    rank_of_truth = []
    for plist, gt in zip(preds, truths):
        rank = None
        for idx, (pred_set, _) in enumerate(plist):
            if pred_set == gt:
                rank = idx + 1
                break
        rank_of_truth.append(rank)
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
    return {f"top_{k}": round(hits[k] / n * 100, 2) for k in ks}, rank_of_truth


def ensemble_combine(per_model_preds: dict[str, list[list[tuple[frozenset, float]]]],
                     weights: dict[str, float] | None = None,
                     top_k_final: int = 50) -> list[list[tuple[frozenset, float]]]:
    """Combine multi-model predictions: weighted score sum on canonical reactant key."""
    n = len(next(iter(per_model_preds.values())))
    names = list(per_model_preds.keys())
    if weights is None:
        weights = {n_: 1.0 / len(names) for n_ in names}

    combined = []
    for i in range(n):
        agg: dict[frozenset, float] = defaultdict(float)
        for name in names:
            plist = per_model_preds[name][i]
            # Renormalize to a probability distribution for fair fusion
            total = sum(s for _, s in plist) or 1.0
            for k, s in plist:
                agg[k] += weights[name] * (s / total)
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:top_k_final]
        combined.append(ranked)
    return combined


def _save_preds(path: Path, preds: list[list[tuple[frozenset, float]]]):
    serial = [[[sorted(list(k)), s] for k, s in plist] for plist in preds]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial))


def _load_preds(path: Path) -> list[list[tuple[frozenset, float]]]:
    raw = json.loads(path.read_text())
    return [[(frozenset(k), float(s)) for k, s in plist] for plist in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=[],
                    choices=["megan", "rootaligned", "localretro", "chemformer", "mhnreact"])
    ap.add_argument("--model-dir", action="append", nargs=2, metavar=("NAME", "DIR"), default=[])
    ap.add_argument("--data", default="data_external/uspto50k/")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ensemble", action="store_true",
                    help="Combine per-model preds into one ensemble (also reports per-model)")
    ap.add_argument("--cache-dir", type=str, default="results/v2/k2_preds",
                    help="Where to dump/load per-model prediction JSONs")
    ap.add_argument("--load-cached", nargs="+", default=[],
                    help="Model names whose predictions to load from cache instead of running")
    ap.add_argument("--device", type=str, default=None,
                    help="Override torch device (e.g. cpu, cuda:0)")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)

    model_dirs = {n: d for n, d in args.model_dir}

    test = load_uspto50k_test(args.data, args.max_samples)
    print(f"[K2] USPTO-50K test: {len(test)} rxns")

    products = test["product"].tolist()
    truths = [canon_set(r) for r in test["reactants"]]

    per_model_preds: dict[str, list] = {}
    per_model_results: dict[str, dict] = {}

    # Load cached predictions first
    for name in args.load_cached:
        cpath = cache_dir / f"{name}_n{len(test)}.json"
        print(f"[K2] loading cached preds for {name} from {cpath}", flush=True)
        preds = _load_preds(cpath)
        per_model_preds[name] = preds
        metrics, _ = compute_hits(preds, truths)
        metrics["n"] = len(test)
        per_model_results[name] = metrics
        print(f"[K2]   {name} (cached): {metrics}", flush=True)

    for name in args.model:
        if name in per_model_preds:
            continue
        print(f"[K2] loading {name} ...", flush=True)
        t0 = time.time()
        model = load_model(name, model_dirs.get(name), args.device)
        print(f"[K2]   loaded in {time.time()-t0:.1f}s; predicting top-{args.top_k} ...", flush=True)
        t0 = time.time()
        preds = run_model(model, products, args.top_k)
        elapsed = time.time() - t0
        per_model_preds[name] = preds
        cpath = cache_dir / f"{name}_n{len(test)}.json"
        _save_preds(cpath, preds)
        print(f"[K2]   cached preds -> {cpath}", flush=True)
        metrics, _ = compute_hits(preds, truths)
        metrics["n"] = len(test)
        metrics["elapsed_s"] = round(elapsed, 1)
        per_model_results[name] = metrics
        print(f"[K2]   {name}: {metrics}", flush=True)
        del model
        try:
            _torch.cuda.empty_cache()
        except Exception:
            pass

    summary = {
        "n_samples": len(test),
        "models": list(per_model_preds.keys()),
        "per_model": per_model_results,
    }

    if args.ensemble and len(per_model_preds) > 1:
        combined = ensemble_combine(per_model_preds)
        ens_metrics, _ = compute_hits(combined, truths)
        ens_metrics["n"] = len(test)
        summary["ensemble"] = ens_metrics
        print(f"[K2] ENSEMBLE: {ens_metrics}", flush=True)

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"[K2] saved {args.output}")


if __name__ == "__main__":
    main()
