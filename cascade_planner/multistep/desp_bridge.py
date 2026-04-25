"""DESP (Double-Ended Synthesis Planning) bridge for AutoPlanner.

Drop-in alternative to ``aiz_mcts_bridge.py`` that uses DESP's
bidirectional goal-constrained search (NeurIPS 2024 Spotlight) instead
of AiZynthFinder UCT-MCTS.

Reads/writes the same JSON envelope so downstream evaluation scripts
(``multistep_solvebench.py``, ``plan_route.py``, etc.) work unchanged.

DESP paper:  https://github.com/coleygroup/desp
Pre-trained: https://figshare.com/articles/preprint/25956076

When DESP is not installed the module falls back to AiZynthFinder MCTS
with a warning, keeping the pipeline functional.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lazy / optional imports — DESP may not be installed yet
# ---------------------------------------------------------------------------

_DESP_AVAILABLE = False
try:
    from desp.search import DESPSearch  # type: ignore[import-untyped]
    from desp.models.backward import BackwardModel  # type: ignore[import-untyped]
    from desp.models.forward import ForwardModel  # type: ignore[import-untyped]
    from desp.models.value import ValueModel  # type: ignore[import-untyped]

    _DESP_AVAILABLE = True
except ImportError:
    pass

# RDKit — always available in this project
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Project-internal imports (lightweight; no torch at module level)
from cascade_planner.paths import ROOT, shared_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_DESP_DIR = ROOT / "desp_models"


@dataclass
class DESPConfig:
    """All knobs needed to run a DESP search."""

    retro_model_path: str = str(_DEFAULT_DESP_DIR / "retro_model")
    forward_model_path: str = str(_DEFAULT_DESP_DIR / "forward_model")
    value_model_path: str = str(_DEFAULT_DESP_DIR / "value_model")
    building_blocks_path: str = str(_DEFAULT_DESP_DIR / "building_blocks.csv.gz")
    max_depth: int = 6
    max_iterations: int = 500
    n_routes: int = 5
    use_enzexpand: bool = True
    # EnzExpand model paths (auto-resolved from results/shared/ when empty)
    enzexpand_mlp_path: str = ""
    enzexpand_templates_path: str = ""
    reranker_path: str = str(shared_dir() / "reranker_frozen_mf2_ns.txt")

    def resolve_enzexpand_paths(self) -> None:
        """Fill in default EnzExpand artefact paths if left blank."""
        sd = shared_dir()
        if not self.enzexpand_mlp_path:
            self.enzexpand_mlp_path = str(sd / "enzexpand_mlp.pt")
        if not self.enzexpand_templates_path:
            self.enzexpand_templates_path = str(sd / "enzexpand_templates.csv")


# ---------------------------------------------------------------------------
# EnzExpandBackwardModel — wraps AutoPlanner's MLP + reranker for DESP
# ---------------------------------------------------------------------------

class EnzExpandBackwardModel:
    """DESP-compatible backward (retro) model backed by EnzExpand.

    Loads the EnzExpand MLP, template library, and optional LightGBM
    reranker so it can be registered as an additional backward model
    inside a DESP search.
    """

    def __init__(
        self,
        mlp_path: str,
        templates_path: str,
        reranker_path: str | None = None,
        top_k_templates: int = 50,
        max_outcomes: int = 5,
        generalize_level: int = 1,
    ) -> None:
        import torch
        import pandas as pd
        from cascade_planner.expand.enz_template import TemplateMLP, morgan2  # noqa: F811

        self.top_k_templates = top_k_templates
        self.max_outcomes = max_outcomes
        self.generalize_level = generalize_level
        self._morgan2 = morgan2

        # --- load template library ---
        tpl_df = pd.read_csv(templates_path)
        self.templates: list[str] = tpl_df["template"].tolist()
        self.tpl_freq: list[int] = (
            tpl_df["n_examples"].tolist()
            if "n_examples" in tpl_df.columns
            else [1] * len(self.templates)
        )
        # EC1 prior per template (used by reranker features)
        self.tpl_ec1_prob: list[float] = (
            tpl_df["ec1_prob"].tolist()
            if "ec1_prob" in tpl_df.columns
            else [0.0] * len(self.templates)
        )
        self.tpl_tx_prob: list[float] = (
            tpl_df["tx_prob"].tolist()
            if "tx_prob" in tpl_df.columns
            else [0.0] * len(self.templates)
        )

        # --- load MLP ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_tpl = len(self.templates)
        state = torch.load(mlp_path, map_location=device, weights_only=True)
        # infer hidden dim from first layer weight shape
        hidden = state["net.0.weight"].shape[0]
        in_dim = state["net.0.weight"].shape[1]
        self.model = TemplateMLP(in_dim, n_tpl, hidden=hidden)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        self.device = device

        # --- load reranker (optional) ---
        self.reranker = None
        if reranker_path and Path(reranker_path).exists():
            from cascade_planner.expand.reranker_infer import EnzReranker

            self.reranker = EnzReranker(reranker_path)

    # ---- public API expected by DESP ----

    def predict(
        self,
        product_smiles: str,
        top_k: int = 50,
    ) -> list[dict]:
        """Return retro-predictions for *product_smiles*.

        Each entry: ``{reactants: list[str], score: float, template: str}``.
        """
        import torch
        from cascade_planner.expand.enz_template import (
            apply_template_to_product,
            canon_set,
        )

        fp = self._morgan2(product_smiles).reshape(1, -1)
        xt = torch.tensor(fp, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(xt)[0]
            topk = torch.topk(logits, k=min(self.top_k_templates, logits.shape[0]))
        tpl_indices = topk.indices.cpu().numpy()
        tpl_logits = topk.values.cpu().numpy()

        # Apply templates via rdchiral
        candidates: list[dict] = []
        for rank, (tid, logit) in enumerate(zip(tpl_indices, tpl_logits)):
            tid = int(tid)
            if tid >= len(self.templates):
                continue
            tpl = self.templates[tid]
            outcomes = apply_template_to_product(
                tpl,
                product_smiles,
                max_outcomes=self.max_outcomes,
                generalize=self.generalize_level,
            )
            for reactant_set in outcomes:
                candidates.append({
                    "reactants": sorted(reactant_set),
                    "cand_reactants": reactant_set,
                    "score": float(logit),
                    "template": tpl,
                    "mlp_rank": rank,
                    "mlp_logit": float(logit),
                    "tpl_freq": self.tpl_freq[tid],
                    "tpl_ec1_prob": self.tpl_ec1_prob[tid],
                    "tpl_tx_prob": self.tpl_tx_prob[tid],
                })

        if not candidates:
            return []

        # Rerank if reranker is available
        if self.reranker is not None:
            ranked = self.reranker.rerank(
                product_smi=product_smiles,
                candidates=candidates,
            )
            reranked: list[dict] = []
            for orig_idx, score in ranked:
                c = candidates[orig_idx]
                c["score"] = float(score)
                reranked.append(c)
            candidates = reranked

        # Deduplicate by reactant set, keep best score
        seen: set[frozenset[str]] = set()
        deduped: list[dict] = []
        for c in candidates:
            key = frozenset(c["reactants"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append({
                "reactants": c["reactants"],
                "score": c["score"],
                "template": c["template"],
            })

        return deduped[:top_k]


# ---------------------------------------------------------------------------
# Installation helper
# ---------------------------------------------------------------------------

_INSTALL_MSG = """\
DESP is not installed. To install:

  pip install desp

Or clone from source:

  git clone https://github.com/coleygroup/desp.git
  cd desp && pip install -e .

Pre-trained models (retro, forward, value, building blocks) can be
downloaded from:

  https://figshare.com/articles/preprint/25956076

Place them under {desp_dir}/ with the following layout:

  {desp_dir}/
    retro_model/
    forward_model/
    value_model/
    building_blocks.csv.gz
"""


def install_desp(desp_dir: str | Path | None = None) -> bool:
    """Check whether DESP is importable. Print instructions if not.

    Returns True if DESP is available, False otherwise.
    """
    if _DESP_AVAILABLE:
        return True
    target = desp_dir or _DEFAULT_DESP_DIR
    print(_INSTALL_MSG.format(desp_dir=target), file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def _convert_desp_routes(
    desp_result: Any,
    target_smiles: str,
    n_routes: int,
) -> list[dict]:
    """Convert DESP's native route objects to AutoPlanner's format.

    Each route becomes::

        {"score": float, "depth": int, "in_stock_frac": float,
         "is_solved": bool,
         "reactions": [{"product": str, "reactants": [str], ...}, ...],
         "tree": <nested dict matching aiz_mcts_bridge output>}
    """
    routes_out: list[dict] = []

    raw_routes = []
    if hasattr(desp_result, "routes"):
        raw_routes = desp_result.routes
    elif isinstance(desp_result, dict):
        raw_routes = desp_result.get("routes", [])
    elif isinstance(desp_result, (list, tuple)):
        raw_routes = list(desp_result)

    for route in raw_routes[:n_routes]:
        reactions: list[dict] = []

        # DESP routes expose reactions as a list of step dicts
        steps = []
        if hasattr(route, "reactions"):
            steps = route.reactions
        elif isinstance(route, dict):
            steps = route.get("reactions", route.get("steps", []))

        for step in steps:
            if isinstance(step, dict):
                rxn_info = {
                    "product": step.get("product", ""),
                    "reactants": step.get("reactants", []),
                    "template": step.get("template", ""),
                    "score": step.get("score", 0.0),
                }
            else:
                # Object with attributes
                rxn_info = {
                    "product": getattr(step, "product", ""),
                    "reactants": list(getattr(step, "reactants", [])),
                    "template": getattr(step, "template", ""),
                    "score": float(getattr(step, "score", 0.0)),
                }
            reactions.append(rxn_info)

        depth = len(reactions)

        # Score
        route_score = 0.0
        if hasattr(route, "score"):
            route_score = float(route.score)
        elif isinstance(route, dict):
            route_score = float(route.get("score", 0.0))

        # In-stock fraction
        in_stock_frac = 0.0
        if hasattr(route, "in_stock_fraction"):
            in_stock_frac = float(route.in_stock_fraction)
        elif isinstance(route, dict):
            in_stock_frac = float(route.get("in_stock_fraction", 0.0))

        is_solved = in_stock_frac >= 1.0 and depth > 0

        # Build a nested tree dict compatible with aiz_mcts_bridge output
        tree = _build_tree_dict(target_smiles, reactions)

        routes_out.append({
            "score": route_score,
            "depth": depth,
            "in_stock_frac": in_stock_frac,
            "is_solved": is_solved,
            "tree": tree,
        })

    return routes_out


def _build_tree_dict(target: str, reactions: list[dict]) -> dict:
    """Build a nested mol/reaction tree dict from a flat reaction list.

    Mimics the AiZynthFinder tree format so ``plan_route.walk_reactions``
    can consume it directly.
    """
    if not reactions:
        return {"type": "mol", "smiles": target, "in_stock": False, "children": []}

    # Index reactions by product for quick lookup
    by_product: dict[str, list[dict]] = {}
    for rxn in reactions:
        prod = rxn.get("product", "")
        by_product.setdefault(prod, []).append(rxn)

    def _mol_node(smi: str, visited: set[str] | None = None) -> dict:
        if visited is None:
            visited = set()
        if smi in visited:
            return {"type": "mol", "smiles": smi, "in_stock": False, "children": []}
        visited = visited | {smi}

        rxns_for_mol = by_product.get(smi, [])
        if not rxns_for_mol:
            # Leaf — building block
            return {"type": "mol", "smiles": smi, "in_stock": True, "children": []}

        rxn = rxns_for_mol[0]
        children_mols = [
            _mol_node(r, visited) for r in rxn.get("reactants", [])
        ]
        rxn_node = {
            "type": "reaction",
            "metadata": {
                "template": rxn.get("template", ""),
                "policy_probability": rxn.get("score"),
                "policy_name": "desp",
            },
            "children": children_mols,
        }
        return {
            "type": "mol",
            "smiles": smi,
            "in_stock": False,
            "children": [rxn_node],
        }

    return _mol_node(target)


def run_desp_search(target_smiles: str, config: DESPConfig) -> dict:
    """Run DESP bidirectional search on a single target.

    Returns the standard AutoPlanner route envelope::

        {"target": str, "search_time_s": float, "n_routes": int,
         "routes": [...]}

    Falls back to AiZynthFinder MCTS if DESP is unavailable.
    """
    if not _DESP_AVAILABLE:
        warnings.warn(
            "DESP not installed — falling back to AiZynthFinder MCTS. "
            "Run `install_desp()` for setup instructions.",
            stacklevel=2,
        )
        return _fallback_mcts(target_smiles, config)

    # Validate that model files exist
    for label, path in [
        ("retro_model", config.retro_model_path),
        ("forward_model", config.forward_model_path),
        ("value_model", config.value_model_path),
        ("building_blocks", config.building_blocks_path),
    ]:
        if not Path(path).exists():
            warnings.warn(
                f"DESP {label} not found at {path} — falling back to MCTS.",
                stacklevel=2,
            )
            return _fallback_mcts(target_smiles, config)

    # Build DESP search
    search_kwargs: dict[str, Any] = {
        "retro_model_path": config.retro_model_path,
        "forward_model_path": config.forward_model_path,
        "value_model_path": config.value_model_path,
        "building_blocks_path": config.building_blocks_path,
        "max_depth": config.max_depth,
        "max_iterations": config.max_iterations,
        "num_routes": config.n_routes,
    }

    searcher = DESPSearch(**search_kwargs)

    # Register EnzExpand as additional backward model
    if config.use_enzexpand:
        config.resolve_enzexpand_paths()
        try:
            enz_model = EnzExpandBackwardModel(
                mlp_path=config.enzexpand_mlp_path,
                templates_path=config.enzexpand_templates_path,
                reranker_path=config.reranker_path,
            )
            if hasattr(searcher, "add_backward_model"):
                searcher.add_backward_model(enz_model)
            elif hasattr(searcher, "backward_models"):
                searcher.backward_models.append(enz_model)
            else:
                print(
                    "[desp] Warning: could not register EnzExpand — "
                    "DESP API does not expose backward model registration.",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"[desp] Warning: EnzExpand init failed ({exc}); "
                "proceeding with DESP retro model only.",
                file=sys.stderr,
            )

    t0 = time.time()
    try:
        result = searcher.search(target_smiles)
    except Exception as exc:
        print(f"[desp] search failed: {exc}", file=sys.stderr)
        return {
            "target": target_smiles,
            "search_time_s": round(time.time() - t0, 2),
            "n_routes_total": 0,
            "n_routes_returned": 0,
            "error": str(exc),
            "routes": [],
        }
    search_time = time.time() - t0

    routes = _convert_desp_routes(result, target_smiles, config.n_routes)

    return {
        "target": target_smiles,
        "search_time_s": round(search_time, 2),
        "n_routes_total": len(routes),
        "n_routes_returned": len(routes),
        "stats": {
            "engine": "desp",
            "max_iter": config.max_iterations,
            "max_depth": config.max_depth,
            "use_enzexpand": config.use_enzexpand,
        },
        "routes": routes,
    }


# ---------------------------------------------------------------------------
# Fallback to AiZynthFinder MCTS
# ---------------------------------------------------------------------------

def _fallback_mcts(target_smiles: str, config: DESPConfig) -> dict:
    """Run AiZynthFinder MCTS as a fallback when DESP is unavailable."""
    from cascade_planner.multistep.plan_route import call_mcts

    return call_mcts(
        target_smiles,
        max_iter=config.max_iterations,
        max_depth=config.max_depth,
        n_routes=config.n_routes,
    )


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_batch(
    targets: list[str],
    config: DESPConfig,
    n_workers: int = 1,
) -> list[dict]:
    """Run DESP search on multiple targets.

    When *n_workers* > 1, targets are processed in parallel using
    ``ProcessPoolExecutor``.  Each worker gets its own DESP instance.
    """
    if n_workers <= 1:
        results: list[dict] = []
        for i, smi in enumerate(targets):
            print(f"[desp] target {i + 1}/{len(targets)}: {smi[:80]}")
            results.append(run_desp_search(smi, config))
        return results

    # Parallel execution
    results_map: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(run_desp_search, smi, config): idx
            for idx, smi in enumerate(targets)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_map[idx] = future.result()
            except Exception as exc:
                print(f"[desp] target {idx} failed: {exc}", file=sys.stderr)
                results_map[idx] = {
                    "target": targets[idx],
                    "search_time_s": 0.0,
                    "n_routes_total": 0,
                    "n_routes_returned": 0,
                    "error": str(exc),
                    "routes": [],
                }

    return [results_map[i] for i in range(len(targets))]


# ---------------------------------------------------------------------------
# stdin/stdout bridge mode (matches aiz_mcts_bridge.py protocol)
# ---------------------------------------------------------------------------

def _bridge_main() -> None:
    """Read JSON from stdin, run DESP, write JSON to stdout.

    Payload schema is identical to ``aiz_mcts_bridge.py``::

        {"product": "SMILES", "config": "...",
         "max_iter": 500, "max_depth": 6, "n_routes": 5, ...}
    """
    payload = json.load(sys.stdin)
    product = payload["product"]

    cfg = DESPConfig(
        max_iterations=int(payload.get("max_iter", 500)),
        max_depth=int(payload.get("max_depth", 6)),
        n_routes=int(payload.get("n_routes", 5)),
        use_enzexpand=bool(payload.get("use_enzexpand", True)),
    )

    # Override model paths if provided
    desp_dir = payload.get("desp_dir")
    if desp_dir:
        d = Path(desp_dir)
        cfg.retro_model_path = str(d / "retro_model")
        cfg.forward_model_path = str(d / "forward_model")
        cfg.value_model_path = str(d / "value_model")
        cfg.building_blocks_path = str(d / "building_blocks.csv.gz")

    result = run_desp_search(product, cfg)
    json.dump(result, sys.stdout, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="DESP bidirectional retrosynthesis search (AutoPlanner bridge)",
    )
    ap.add_argument("--target", type=str, default=None, help="Single target SMILES")
    ap.add_argument(
        "--targets-file",
        type=str,
        default=None,
        help="JSON file with list of SMILES (or list of {smiles: ...} dicts)",
    )
    ap.add_argument(
        "--desp-dir",
        type=str,
        default=str(_DEFAULT_DESP_DIR),
        help="Directory containing DESP pre-trained models",
    )
    ap.add_argument("--use-enzexpand", action="store_true", default=True)
    ap.add_argument("--no-enzexpand", dest="use_enzexpand", action="store_false")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-routes", type=int, default=5)
    ap.add_argument("--n-workers", type=int, default=1, help="Parallel workers for batch mode")
    ap.add_argument("--output", type=str, default=None, help="Output JSON path")
    ap.add_argument(
        "--bridge",
        action="store_true",
        help="Run in bridge mode (read JSON from stdin, write to stdout)",
    )
    args = ap.parse_args()

    # Bridge mode — stdin/stdout protocol
    if args.bridge:
        _bridge_main()
        return

    if args.target is None and args.targets_file is None:
        ap.print_help()
        sys.exit(1)

    # Build config
    d = Path(args.desp_dir)
    cfg = DESPConfig(
        retro_model_path=str(d / "retro_model"),
        forward_model_path=str(d / "forward_model"),
        value_model_path=str(d / "value_model"),
        building_blocks_path=str(d / "building_blocks.csv.gz"),
        max_depth=args.max_depth,
        max_iterations=args.max_iter,
        n_routes=args.n_routes,
        use_enzexpand=args.use_enzexpand,
    )

    # Check DESP availability
    if not install_desp(args.desp_dir):
        print(
            "[desp] DESP not available. Will fall back to AiZynthFinder MCTS.",
            file=sys.stderr,
        )

    # Single target
    if args.target:
        print(f"[desp] target: {args.target}")
        result = run_desp_search(args.target, cfg)
        n_solved = sum(1 for r in result.get("routes", []) if r.get("is_solved"))
        print(
            f"[desp] done in {result.get('search_time_s', '?')}s — "
            f"{result.get('n_routes_returned', 0)} routes, {n_solved} solved"
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8",
            )
            print(f"[desp] saved -> {args.output}")
        else:
            json.dump(result, sys.stdout, indent=2, default=str)
            print()
        return

    # Batch mode
    targets_path = Path(args.targets_file)
    raw = json.loads(targets_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        targets = [
            (item["smiles"] if isinstance(item, dict) else str(item))
            for item in raw
        ]
    elif isinstance(raw, dict):
        targets = [
            (item["smiles"] if isinstance(item, dict) else str(item))
            for item in raw.get("targets", raw.get("smiles", []))
        ]
    else:
        print(f"[desp] cannot parse targets file: {targets_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[desp] batch: {len(targets)} targets, {args.n_workers} workers")
    results = run_batch(targets, cfg, n_workers=args.n_workers)

    n_total_solved = sum(
        1
        for res in results
        for r in res.get("routes", [])
        if r.get("is_solved")
    )
    print(
        f"[desp] batch done — {len(results)} targets, "
        f"{n_total_solved} solved routes total"
    )

    out_path = args.output or "results/desp_batch_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    print(f"[desp] saved -> {out_path}")


if __name__ == "__main__":
    main()
