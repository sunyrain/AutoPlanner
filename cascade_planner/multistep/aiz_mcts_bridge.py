"""Subprocess bridge to AiZynthFinder MCTS multi-step search.

Runs INSIDE .venv_aizynth env. Reads JSON from stdin:
    {"product": "SMILES", "config": "path/to/config.yml",
     "max_iter": 100, "max_depth": 5, "n_routes": 5,
     "use_filter": true, "use_stock": true}

Writes JSON to stdout:
    {"target": "...", "search_time_s": 12.3, "n_routes": 5,
     "stats": {...},
     "routes": [
        {"score": 0.92, "depth": 3, "in_stock_frac": 1.0,
         "tree": {<recursive route dict from aizynth>}},
        ...
     ]}
"""
from __future__ import annotations

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def main():
    payload = json.load(sys.stdin)
    product = payload["product"]
    config = payload.get("config", "aizdata/config.yml")
    max_iter = int(payload.get("max_iter", 100))
    max_depth = int(payload.get("max_depth", 5))
    n_routes = int(payload.get("n_routes", 5))
    use_filter = bool(payload.get("use_filter", True))
    policies = payload.get("policies", ["uspto"])
    if isinstance(policies, str):
        policies = [policies]
    policy_weights = payload.get("policy_weights")  # optional list, will be normalized

    from aizynthfinder.aizynthfinder import AiZynthFinder
    from aizynthfinder.context.policy.expansion_strategies import MultiExpansionStrategy
    from aizynthfinder.chem.reaction import TemplatedRetroReaction

    # Patch _apply so a malformed template (e.g. EnzymeMap-derived enzexpand
    # SMARTS that rdchiral can't parse) returns no reactants instead of raising
    # an exception that would terminate the MCTS iteration.
    _orig_apply = TemplatedRetroReaction._apply
    def _safe_apply(self):
        try:
            return _orig_apply(self)
        except Exception as e:
            print(f"[bridge] template apply failed: {type(e).__name__}", file=sys.stderr)
            return tuple()
    TemplatedRetroReaction._apply = _safe_apply

    finder = AiZynthFinder(configfile=config)
    avail = list(finder.expansion_policy.items)
    sel = [p for p in policies if p in avail] or avail[:1]

    # Redirect stdout at OS level during policy setup — MultiExpansionStrategy
    # prints diagnostics via C/Cython that bypass Python sys.stdout.
    import contextlib, io as _io, os as _os
    _saved_fd = _os.dup(1)
    _devnull = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(_devnull, 1)
    _os.close(_devnull)

    if len(sel) == 1:
        finder.expansion_policy.select(sel[0])
        active = sel[0]
    else:
        multi_key = "_runtime_multi"
        ms_kwargs = dict(
            expansion_strategies=sel,
            additive_expansion=True,
            cutoff_number=int(payload.get("multi_cutoff", 100)),
        )
        if policy_weights and len(policy_weights) == len(sel):
            w = [float(x) for x in policy_weights]
            s = sum(w)
            if s > 0:
                w = [x / s for x in w]
                # nudge last to fix float rounding so sum == 1.0 exactly
                w[-1] = 1.0 - sum(w[:-1])
                ms_kwargs["expansion_strategy_weights"] = w
                print(f"[bridge] policy weights = {dict(zip(sel, w))}", file=sys.stderr)
        with contextlib.redirect_stdout(_io.StringIO()) as _buf:
            multi = MultiExpansionStrategy(
                multi_key, finder.config, **ms_kwargs,
            )
            finder.expansion_policy.load(multi)
            finder.expansion_policy.select(multi_key)
        if _buf.getvalue():
            print(_buf.getvalue(), file=sys.stderr, end="")
        active = f"multi[{','.join(sel)}]"

    # Restore stdout fd and dump captured noise to stderr
    _os.dup2(_saved_fd, 1)
    _os.close(_saved_fd)

    print(f"[bridge] expansion policies = {active}", file=sys.stderr)
    if use_filter and "uspto" in finder.filter_policy.items:
        finder.filter_policy.select("uspto")
    if "zinc" in finder.stock.items:
        finder.stock.select("zinc")

    # MCTS hyper-parameters
    finder.config.search.iteration_limit = max_iter
    finder.config.search.max_transforms = max_depth
    finder.config.search.return_first = False

    finder.target_smiles = product
    finder.prepare_tree()
    t0 = time.time()
    finder.tree_search()
    search_time = time.time() - t0
    finder.build_routes()

    routes_out = []
    n = min(n_routes, len(finder.routes))
    for i in range(n):
        try:
            d = finder.routes.dicts[i]
        except Exception:
            d = None
        try:
            score = float(finder.routes.scores[i])
        except Exception:
            score = None
        try:
            tree = finder.routes.reaction_trees[i]
            # count reactions as tree depth proxy (robust vs n.depth attr)
            try:
                rxns = list(tree.reactions())
                depth = len(rxns)
            except Exception:
                depth = max((getattr(n, "depth", 0) for n in tree.molecules()), default=0)
            mols = list(tree.molecules())
            in_stock = sum(1 for m in mols if tree.in_stock(m))
            frac = in_stock / max(1, len(mols))
            # leaves-only in-stock check: a route is "solved" iff every leaf is in stock.
            leaves = [m for m in mols if not any(
                r.reactants and m in r.reactants[0] for r in tree.reactions()
            )] or mols
            try:
                leaves = [m for m in mols if tree.depth(m) == max(tree.depth(x) for x in mols)]
            except Exception:
                pass
            leaves_in_stock = all(tree.in_stock(m) for m in leaves) if leaves else False
            is_solved = bool(leaves_in_stock and depth > 0)
        except Exception:
            depth = None
            frac = None
            is_solved = False
        routes_out.append({
            "score": score,
            "depth": depth,
            "in_stock_frac": frac,
            "is_solved": is_solved,
            "tree": d,
        })

    out = {
        "target": product,
        "search_time_s": round(search_time, 2),
        "n_routes_total": len(finder.routes),
        "n_routes_returned": len(routes_out),
        "stats": {
            "iterations": finder.search_stats.get("iterations") if hasattr(finder, "search_stats") else None,
            "max_iter": max_iter,
            "max_depth": max_depth,
        },
        "routes": routes_out,
    }
    json.dump(out, sys.stdout, default=str)


if __name__ == "__main__":
    main()
