"""Run the current AutoPlanner route_tree baseline on the cascade gold smoke set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cascade_planner.cascadeboard.live_benchmark import run_live_benchmark


DEFAULT_ROUTE_TREE_POLICY = (
    "results/shared/controller_v2_20260512/fullrun/train_v8/open_leaf_policy/stock_aware_search_policy.pt"
)
DEFAULT_PROPOSAL_RANKER_DIR = "results/shared/proposal_rankers/full_20260508"
DEFAULT_SOURCE_GATE = "results/shared/proposal_rankers/full_20260508/source_gate.pt"


def _set_if_present(env_name: str, value: str | None) -> None:
    if value:
        os.environ[env_name] = value


def _existing_or_empty(path: str) -> str:
    return path if Path(path).exists() else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Run same-target AutoPlanner route_tree smoke baseline")
    ap.add_argument("--benchmark", default="data/benchmark_cascade_gold_smoke_v1.json")
    ap.add_argument("--output", default="results/shared/chem_enzy_baseline/route_tree_gold_smoke.json")
    ap.add_argument("--model", default="results/shared/skeleton_inpainter/best.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-results", type=int, default=5)
    ap.add_argument("--n-candidates-per-skeleton", type=int, default=2)
    ap.add_argument("--skeleton-samples", type=int, default=5)
    ap.add_argument("--search-budget", type=int, default=None)
    ap.add_argument("--check-stock", action="store_true")
    ap.add_argument("--target-log", default="brief", choices=["none", "brief", "json"])
    ap.add_argument("--route-tree-policy", default=_existing_or_empty(DEFAULT_ROUTE_TREE_POLICY))
    ap.add_argument("--proposal-ranker-dir", default=_existing_or_empty(DEFAULT_PROPOSAL_RANKER_DIR))
    ap.add_argument("--source-gate", default=_existing_or_empty(DEFAULT_SOURCE_GATE))
    ap.add_argument("--disable-v3-retrieval-proposals", action="store_true")
    args = ap.parse_args()

    _set_if_present("AUTOPLANNER_ENABLE_ROUTE_TREE_PLANNER", "1" if args.route_tree_policy else "")
    _set_if_present("AUTOPLANNER_ROUTE_TREE_POLICY", args.route_tree_policy)
    _set_if_present("AUTOPLANNER_PROPOSAL_RANKER_DIR", args.proposal_ranker_dir)
    _set_if_present("AUTOPLANNER_SOURCE_GATE", args.source_gate)
    if args.proposal_ranker_dir:
        os.environ["AUTOPLANNER_ENABLE_PROPOSAL_RANKERS"] = "1"
    if not args.disable_v3_retrieval_proposals:
        os.environ["AUTOPLANNER_ENABLE_V3_RETRIEVAL_PROPOSALS"] = "1"

    result = run_live_benchmark(
        bench_path=args.benchmark,
        output_path=args.output,
        model_path=args.model,
        limit=args.limit,
        n_results=args.n_results,
        n_candidates_per_skeleton=args.n_candidates_per_skeleton,
        skeleton_samples=args.skeleton_samples,
        device=args.device,
        check_stock=args.check_stock,
        prior_provider="none",
        search_mode="route_tree",
        search_budget=args.search_budget,
        target_log=args.target_log,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
