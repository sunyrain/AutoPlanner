"""Chemical retrosynthesis ensemble via Syntheseus pre-trained models.

Wraps MEGAN, LocalRetro, RootAligned (and optionally Chemformer, MHNreact)
through the Syntheseus framework as a unified ensemble with RetroChimera-style
score combination.

Usage:
    python -m cascade_planner.expand.chem_ensemble --smiles "CC(=O)Oc1ccccc1C(=O)O" --top-k 10
    python -m cascade_planner.expand.chem_ensemble --eval --data cascade_dataset_v2.normalized.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Syntheseus availability check
# ---------------------------------------------------------------------------

_SYNTHESEUS_AVAILABLE = False
try:
    from syntheseus.interface.molecule import Molecule
    from syntheseus.interface.reaction import SingleProductReaction
    _SYNTHESEUS_AVAILABLE = True
except ImportError:
    pass


def _check_syntheseus():
    if not _SYNTHESEUS_AVAILABLE:
        print(
            "[chem_ensemble] syntheseus not installed.\n"
            "  pip install 'syntheseus[all]'\n"
            "  This installs wrappers for MEGAN, LocalRetro, RootAligned, etc.",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Canonical SMILES helper
# ---------------------------------------------------------------------------

def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    reactants: list[str]
    score: float
    source: str  # model name


@dataclass
class EnsembleConfig:
    models: list[str] = field(default_factory=lambda: ["megan", "rootaligned"])
    weights: dict[str, float] = field(default_factory=dict)
    top_k_per_model: int = 10
    top_k_final: int = 50
    model_dirs: dict[str, str] = field(default_factory=dict)


class ChemEnsemble:
    """Unified chemical retrosynthesis ensemble."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self._models: dict[str, Any] = {}
        self._loaded = False

    def load_models(self):
        if not _check_syntheseus():
            return
        if self._loaded:
            return

        for name in self.config.models:
            try:
                model = self._load_one(name)
                if model is not None:
                    self._models[name] = model
                    print(f"[chem_ensemble] loaded {name}")
            except Exception as e:
                print(f"[chem_ensemble] failed to load {name}: {e}", file=sys.stderr)

        if not self.config.weights:
            n = len(self._models)
            self.config.weights = {k: 1.0 / n for k in self._models} if n else {}

        self._loaded = True

    def _load_one(self, name: str) -> Any:
        model_dir = self.config.model_dirs.get(name)

        if name == "megan":
            try:
                from syntheseus.search.mol.inventory import SmilesListInventory
                from syntheseus.reaction_prediction.inference.megan import MEGANModel
                kwargs = {"model_dir": model_dir} if model_dir else {}
                return MEGANModel(**kwargs)
            except (ImportError, Exception) as e:
                print(f"[chem_ensemble] MEGAN unavailable: {e}", file=sys.stderr)
                return None

        if name in ("rootaligned", "root_aligned"):
            try:
                from syntheseus.reaction_prediction.inference.root_aligned import RootAlignedModel
                kwargs = {"model_dir": model_dir} if model_dir else {}
                return RootAlignedModel(**kwargs)
            except (ImportError, Exception) as e:
                print(f"[chem_ensemble] RootAligned unavailable: {e}", file=sys.stderr)
                return None

        if name == "localretro":
            try:
                from syntheseus.reaction_prediction.inference.local_retro import LocalRetroModel
                kwargs = {"model_dir": model_dir} if model_dir else {}
                return LocalRetroModel(**kwargs)
            except (ImportError, Exception) as e:
                print(f"[chem_ensemble] LocalRetro unavailable: {e}", file=sys.stderr)
                return None

        if name == "chemformer":
            try:
                from syntheseus.reaction_prediction.inference.chemformer import ChemformerModel
                kwargs = {"model_dir": model_dir} if model_dir else {}
                return ChemformerModel(**kwargs)
            except (ImportError, Exception) as e:
                print(f"[chem_ensemble] Chemformer unavailable: {e}", file=sys.stderr)
                return None

        print(f"[chem_ensemble] unknown model: {name}", file=sys.stderr)
        return None

    def predict(self, product_smiles: str, top_k: int | None = None) -> list[Prediction]:
        """Run all models and combine predictions."""
        if not self._loaded:
            self.load_models()
        if not self._models:
            return []

        top_k = top_k or self.config.top_k_final
        k_per = self.config.top_k_per_model

        all_preds: dict[str, float] = defaultdict(float)
        pred_sources: dict[str, list[str]] = defaultdict(list)

        for name, model in self._models.items():
            w = self.config.weights.get(name, 1.0 / len(self._models))
            try:
                results = self._run_model(model, product_smiles, k_per)
                for rxn_key, score in results:
                    all_preds[rxn_key] += w * score
                    pred_sources[rxn_key].append(name)
            except Exception as e:
                print(f"[chem_ensemble] {name} failed on {product_smiles[:40]}: {e}",
                      file=sys.stderr)

        ranked = sorted(all_preds.items(), key=lambda x: -x[1])[:top_k]
        return [
            Prediction(
                reactants=rxn_key.split("."),
                score=score,
                source="+".join(pred_sources[rxn_key]),
            )
            for rxn_key, score in ranked
        ]

    def _run_model(self, model, product_smiles: str, top_k: int) -> list[tuple[str, float]]:
        """Run a single Syntheseus model, return [(canonical_reactants_key, score)]."""
        if not _SYNTHESEUS_AVAILABLE:
            return []

        mol = Molecule(product_smiles)
        reactions = model.get_reactions([mol], num_results=top_k)

        results = []
        for rxn_list in reactions:
            for rxn in rxn_list:
                reactant_smiles = sorted(
                    _canon(str(r)) or str(r) for r in rxn.reactants
                )
                key = ".".join(reactant_smiles)
                score = float(getattr(rxn, "log_prob", 0.0))
                score = np.exp(score) if score < 0 else score
                results.append((key, score))

        return results

    def predict_batch(self, products: list[str], top_k: int | None = None) -> list[list[Prediction]]:
        return [self.predict(p, top_k) for p in products]


# ---------------------------------------------------------------------------
# Standalone fallback (no Syntheseus) using existing AiZynth/synth_weights
# ---------------------------------------------------------------------------

class FallbackEnsemble:
    """Use pre-computed eval CSVs from existing engines when Syntheseus is unavailable."""

    def __init__(self, results_dir: Path | None = None):
        self.results_dir = results_dir or ROOT / "results" / "v2"

    def load_eval_results(self) -> dict[str, dict[str, list[str]]]:
        """Load pre-computed step-level eval CSVs from existing engines."""
        import pandas as pd
        engines = {}
        for name in ["aizynthfinder_full_gpu_step_eval",
                      "syntheseus_step_eval_megan",
                      "syntheseus_step_eval_rootaligned"]:
            csv_path = self.results_dir / f"{name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                engines[name] = df
                print(f"[fallback] loaded {name}: {len(df)} rows")
        return engines


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    data_path: str,
    config: EnsembleConfig | None = None,
    max_steps: int | None = None,
) -> dict:
    """Evaluate ensemble on AutoPlanner chemical steps."""
    from cascade_planner.data.loader_v2 import load_v2

    steps, _, _ = load_v2(data_path)
    chem_steps = [s for s in steps if not s.ec_number]
    if max_steps:
        chem_steps = chem_steps[:max_steps]

    print(f"[eval] {len(chem_steps)} chemical steps")

    ensemble = ChemEnsemble(config)
    ensemble.load_models()

    if not ensemble._models:
        print("[eval] no models loaded, using fallback eval from CSVs")
        fb = FallbackEnsemble()
        engines = fb.load_eval_results()
        if not engines:
            print("[eval] no eval CSVs found either")
            return {}
        print(f"[eval] loaded {len(engines)} engine CSVs for offline analysis")
        return {"mode": "fallback", "engines": list(engines.keys())}

    hits = {k: 0 for k in [1, 5, 10, 50]}
    total = 0

    for step in chem_steps:
        rxn = step.rxn_smiles
        if ">>" not in rxn:
            continue
        product = rxn.split(">>")[1].strip()
        gt_lhs = rxn.split(">>")[0].strip()
        gt_reactants = frozenset(_canon(s) or s for s in gt_lhs.split("."))

        preds = ensemble.predict(product, top_k=50)
        total += 1

        for k in hits:
            for pred in preds[:k]:
                pred_set = frozenset(_canon(r) or r for r in pred.reactants)
                if pred_set == gt_reactants:
                    hits[k] += 1
                    break

    results = {}
    for k in hits:
        acc = hits[k] / total if total else 0
        results[f"top_{k}"] = round(acc * 100, 2)

    results["n_steps"] = total
    results["n_models"] = len(ensemble._models)
    results["models"] = list(ensemble._models.keys())
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chemical retrosynthesis ensemble")
    parser.add_argument("--smiles", type=str, help="Product SMILES for single prediction")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval", action="store_true", help="Evaluate on AutoPlanner data")
    parser.add_argument("--data", type=str, default="cascade_dataset_v2.normalized.json")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["megan", "rootaligned"],
                        help="Models to include")
    parser.add_argument("--model-dir", type=str, action="append", nargs=2,
                        metavar=("NAME", "DIR"), default=[],
                        help="Model directory override: --model-dir megan /path/to/megan")
    args = parser.parse_args()

    model_dirs = {name: d for name, d in args.model_dir}
    config = EnsembleConfig(
        models=args.models,
        model_dirs=model_dirs,
        top_k_per_model=args.top_k,
    )

    if args.eval:
        results = evaluate_ensemble(args.data, config, args.max_steps)
        print(json.dumps(results, indent=2))
        return

    if args.smiles:
        ensemble = ChemEnsemble(config)
        preds = ensemble.predict(args.smiles, args.top_k)
        for i, p in enumerate(preds):
            print(f"  {i+1:3d}  {'.'.join(p.reactants):60s}  "
                  f"score={p.score:.4f}  src={p.source}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
