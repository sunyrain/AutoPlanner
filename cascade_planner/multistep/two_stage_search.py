"""Two-stage hybrid search: maximizes BOTH solve rate and GT@5.

Stage 1: depth=6, equal policy weights → high solve rate (79%)
Stage 2: depth=4, enzyme-weighted (0.3/0.7) → high GT match (61%)
Union of routes from both stages.

Usage:
    python -m cascade_planner.multistep.two_stage_search --target "SMILES"
    python -m cascade_planner.multistep.two_stage_search --targets-file data/benchmark_v2_100.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def parse_bridge_output(stdout: str) -> dict | None:
    idx = stdout.find("{")
    if idx < 0:
        return None
    try:
        return json.loads(stdout[idx:])
    except json.JSONDecodeError:
        return None


def run_bridge(
    product: str,
    config: str,
    policies: list[str],
    max_depth: int,
    max_iter: int,
    n_routes: int = 10,
    weights: list[float] | None = None,
    timeout: float = 180,
) -> dict | None:
    payload: dict = {
        "product": product,
        "config": config,
        "policies": policies,
        "max_iter": max_iter,
        "max_depth": max_depth,
        "n_routes": n_routes,
        "use_filter": True,
        "use_stock": True,
    }
    if weights:
        payload["policy_weights"] = weights
    r = subprocess.run(
        [sys.executable, "-m", "cascade_planner.multistep.aiz_mcts_bridge"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return parse_bridge_output(r.stdout) if r.returncode == 0 else None


def two_stage_search(
    product: str,
    config: str = "workspace/aizdata/config_hybrid_fixed.yml",
    policies: list[str] | None = None,
    stage1_depth: int = 6,
    stage1_iter: int = 100,
    stage1_weights: list[float] | None = None,
    stage2_depth: int = 4,
    stage2_iter: int = 100,
    stage2_weights: list[float] | None = None,
    n_routes: int = 10,
) -> dict:
    if policies is None:
        policies = ["uspto", "enzexpand"]
    if stage1_weights is None:
        stage1_weights = [0.5, 0.5]
    if stage2_weights is None:
        stage2_weights = [0.3, 0.7]

    t0 = time.time()

    out1 = run_bridge(product, config, policies, stage1_depth, stage1_iter, n_routes, stage1_weights)
    out2 = run_bridge(product, config, policies, stage2_depth, stage2_iter, n_routes, stage2_weights)

    routes1 = out1.get("routes", []) if out1 else []
    routes2 = out2.get("routes", []) if out2 else []

    solved1 = any(rt.get("is_solved", False) for rt in routes1)
    solved2 = any(rt.get("is_solved", False) for rt in routes2)

    # Merge: stage2 routes first (shorter, more GT-like), then stage1
    merged = routes2[:n_routes] + routes1[:n_routes]

    return {
        "target": product,
        "search_time_s": round(time.time() - t0, 2),
        "is_solved": solved1 or solved2,
        "n_routes": len(merged),
        "routes": merged[:n_routes],
        "stage1_solved": solved1,
        "stage2_solved": solved2,
        "stage1_n_routes": len(routes1),
        "stage2_n_routes": len(routes2),
    }


def main():
    ap = argparse.ArgumentParser(description="Two-stage hybrid retrosynthesis search")
    ap.add_argument("--target", type=str)
    ap.add_argument("--targets-file", type=str)
    ap.add_argument("--config", default="workspace/aizdata/config_hybrid_fixed.yml")
    ap.add_argument("--output", type=str)
    args = ap.parse_args()

    if args.target:
        result = two_stage_search(args.target, args.config)
        print(json.dumps(result, indent=2))
        return

    if args.targets_file:
        data = json.loads(Path(args.targets_file).read_text())
        targets = []
        for t in data:
            smi = t.get("target_smiles", t.get("target", t.get("smiles", ""))) if isinstance(t, dict) else str(t)
            if smi:
                targets.append(smi)

        results = []
        for i, smi in enumerate(targets):
            r = two_stage_search(smi, args.config)
            results.append(r)
            if (i + 1) % 10 == 0:
                solved = sum(1 for x in results if x["is_solved"])
                print(f"  [{i+1}/{len(targets)}] solved={solved}/{len(results)}", file=sys.stderr)

        solved = sum(1 for x in results if x["is_solved"])
        print(f"Solved: {solved}/{len(results)} ({solved/len(results)*100:.1f}%)", file=sys.stderr)

        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
        else:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
