"""CascadeBoard ablation runner.

The ablations here are strict/no-mock where possible. Mock fallback is only run
as an explicitly labeled dev-only control.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.real_benchmark import run_real_benchmark


def run_ablations(
    *,
    bench: str,
    rc_cache: str,
    enz_cache: str | None,
    enz_cache_no_dualtower: str | None,
    output: str,
    n_targets: int = 30,
    n_particles: int = 16,
    branch_factor: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    base_kwargs = {
        "bench_path": bench,
        "rc_cache_path": rc_cache,
        "enz_cache_path": enz_cache,
        "n_targets": n_targets,
        "n_particles": n_particles,
        "n_final": 5,
        "branch_factor": branch_factor,
        "seed": seed,
    }
    variants = {
        "baseline_strict": {},
        "no_compatibility_energy": {"energy_weights": {**EnergyAPI().weights, "compatibility": 0.0}},
        "no_pareto_resampling": {"resample": False},
        "retrochimera_only": {"enz_cache_path": None},
    }
    if enz_cache_no_dualtower:
        variants["enzexpand_without_dual_tower"] = {"enz_cache_path": enz_cache_no_dualtower}

    rows = {}
    for name, overrides in variants.items():
        kwargs = {**base_kwargs, **overrides}
        out_path = f"results/v2/ablation_{name}.json"
        result = run_real_benchmark(output=out_path, **kwargs)
        rows[name] = {
            "output": out_path,
            "overall": result["overall"],
            "by_domain": result["by_domain"],
            "metadata": {
                "strict_no_mock": True,
                "energy_weights": result["metadata"].get("energy_weights"),
                "candidate_sources": result["metadata"].get("candidate_sources"),
                "resample": result["metadata"].get("resample"),
            },
        }

    baseline = rows["baseline_strict"]["overall"]
    deltas = {}
    for name, row in rows.items():
        ov = row["overall"]
        deltas[name] = {
            "delta_gt_at_5": ov.get("gt_at_5", 0) - baseline.get("gt_at_5", 0),
            "delta_strict_stock_solve": ov.get("strict_stock_solve", 0) - baseline.get("strict_stock_solve", 0),
            "delta_mean_overlap_top1": ov.get("mean_overlap_top1", 0) - baseline.get("mean_overlap_top1", 0),
        }

    result = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "bench": bench,
            "rc_cache": rc_cache,
            "enz_cache": enz_cache,
            "enz_cache_no_dualtower": enz_cache_no_dualtower,
            "n_targets": n_targets,
            "n_particles": n_particles,
            "branch_factor": branch_factor,
            "seed": seed,
            "note": "Mock fallback is not included; all variants here are strict no-mock.",
        },
        "variants": rows,
        "deltas_vs_baseline": deltas,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--rc-cache", default="results/shared/retrochimera_candidates_depth2.json")
    ap.add_argument("--enz-cache", default="results/shared/enzexpand_dualtower_candidates_100.json")
    ap.add_argument("--enz-cache-no-dualtower", default="results/shared/enzexpand_candidates_100.json")
    ap.add_argument("--output", default="results/v2/cascadeboard_ablation_report.json")
    ap.add_argument("--n-targets", type=int, default=30)
    ap.add_argument("--n-particles", type=int, default=16)
    ap.add_argument("--branch-factor", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = run_ablations(
        bench=args.bench,
        rc_cache=args.rc_cache,
        enz_cache=args.enz_cache,
        enz_cache_no_dualtower=args.enz_cache_no_dualtower,
        output=args.output,
        n_targets=args.n_targets,
        n_particles=args.n_particles,
        branch_factor=args.branch_factor,
        seed=args.seed,
    )
    print(json.dumps(result["deltas_vs_baseline"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
