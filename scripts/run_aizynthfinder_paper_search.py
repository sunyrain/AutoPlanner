"""Run the exact AiZynthFinder baseline or leaf-completion budget locally.

Use this script with ``.venv_aizynth``.  It is deliberately provider-free and
never invokes Codex; its output can therefore be used as the deterministic
template baseline or as a short-tail fragment returned to the host planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aizynthfinder.aizynthfinder import AiZynthFinder


_MODES = {
    "baseline": {"max_transforms": 25, "iterations": 1500, "timeout_s": 1800},
    "short_tail": {"max_transforms": 6, "iterations": 500, "timeout_s": 1200},
    "canary": {"max_transforms": 3, "iterations": 12, "timeout_s": 60},
}


def _flatten_route(route: dict[str, Any], *, route_index: int) -> dict[str, Any]:
    """Project one AiZ molecule/reaction tree into host-ingestible steps."""

    steps: list[dict[str, Any]] = []
    terminal_leaves: list[dict[str, Any]] = []

    def visit_molecule(node: dict[str, Any]) -> None:
        product_smiles = str(node.get("smiles") or "").strip()
        reaction_nodes = [
            dict(child)
            for child in node.get("children") or []
            if isinstance(child, dict) and child.get("type") == "reaction"
        ]
        expanded = False
        for reaction_node in reaction_nodes:
            reactant_nodes = [
                dict(child)
                for child in reaction_node.get("children") or []
                if isinstance(child, dict) and child.get("type") == "mol"
            ]
            reactant_smiles = [
                str(child.get("smiles") or "").strip()
                for child in reactant_nodes
                if str(child.get("smiles") or "").strip()
            ]
            if not product_smiles or not reactant_smiles:
                continue
            expanded = True
            metadata = dict(reaction_node.get("metadata") or {})
            policy_name = str(metadata.get("policy_name") or "template")
            steps.append(
                {
                    "step_index": len(steps) + 1,
                    "product_smiles": product_smiles,
                    "reactant_smiles": reactant_smiles,
                    "precursor_smiles": reactant_smiles,
                    "rxn_smiles": f"{product_smiles}>>{'.'.join(reactant_smiles)}",
                    "source_model": f"AiZynthFinder:{policy_name}",
                    "policy_probability": float(
                        metadata.get("policy_probability") or 0.0
                    ),
                    "policy_probability_rank": metadata.get(
                        "policy_probability_rank"
                    ),
                    "template_hash": str(metadata.get("template_hash") or ""),
                    "template_code": metadata.get("template_code"),
                    "classification": str(metadata.get("classification") or ""),
                    "mapped_reaction_smiles": str(
                        metadata.get("mapped_reaction_smiles") or ""
                    ),
                    "reactant_stock_status": [
                        bool(child.get("in_stock")) for child in reactant_nodes
                    ],
                    "raw_backend_metadata": metadata,
                }
            )
            for child in reactant_nodes:
                visit_molecule(child)
        if not expanded:
            terminal_leaves.append(node)

    visit_molecule(route)
    digest = hashlib.sha256(
        json.dumps(route, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "route_index": route_index,
        "route_trace_id": f"aizynthfinder:{digest[:24]}",
        "raw_route_sha256": digest,
        "target_smiles": str(route.get("smiles") or ""),
        "steps": steps,
        "step_count": len(steps),
        "terminal_leaf_count": len(terminal_leaves),
        "terminal_leaf_stock_status": [
            bool(leaf.get("in_stock")) for leaf in terminal_leaves
        ],
        "all_leaves_in_provider_stock": bool(terminal_leaves)
        and all(leaf.get("in_stock") is True for leaf in terminal_leaves),
    }


def run_search(
    smiles: str,
    *,
    mode: str,
    config_path: Path,
) -> dict[str, Any]:
    budget = dict(_MODES[mode])
    finder = AiZynthFinder(configfile=str(config_path))
    finder.expansion_policy.select(["uspto", "ringbreaker"])
    finder.filter_policy.select("uspto")
    finder.stock.select("paper_zinc_emolecules")
    finder.config.search.max_transforms = int(budget["max_transforms"])
    finder.config.search.iteration_limit = int(budget["iterations"])
    finder.config.search.time_limit = int(budget["timeout_s"])
    finder.config.search.return_first = True
    finder.target_smiles = smiles

    started = time.monotonic()
    finder.prepare_tree()
    search_time = finder.tree_search(show_progress=False)
    finder.build_routes()
    elapsed = time.monotonic() - started
    statistics = dict(finder.extract_statistics())
    routes = list(finder.routes.dicts if finder.routes is not None else [])
    proposal_routes = [
        _flatten_route(dict(route), route_index=index)
        for index, route in enumerate(routes, start=1)
    ]
    return {
        "schema_version": "aizynthfinder_paper_search.v1",
        "engine": "AiZynthFinder 4.4.1",
        "mode": mode,
        "target_smiles": smiles,
        "budget": budget,
        "selection": {
            "expansion": ["uspto", "ringbreaker"],
            "filter": ["uspto"],
            "stock": ["paper_zinc_emolecules"],
        },
        "solved": bool(statistics.get("is_solved")),
        "search_time_s": float(search_time),
        "elapsed_s": float(elapsed),
        "statistics": statistics,
        "routes": routes,
        "proposal_routes": proposal_routes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--mode", choices=sorted(_MODES), default="canary")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/aizynthfinder.paper.yml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = run_search(
        args.smiles,
        mode=args.mode,
        config_path=args.config,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
