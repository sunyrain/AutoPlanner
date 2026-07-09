"""Integrated Benchmark: OA-ARM Skeleton → Fill → Learned Scorer.

Runs the 100-target benchmark with the new 3-layer architecture:
  Layer 1: OA-ARM Skeleton Inpainter (generates K skeletons)
  Layer 2: Enzyformer/RetroChimera fill (greedy, from skeleton_planner)
  Layer 3: Learned Route Scorer (ranks filled routes)

Metrics: GT@5, plan_rate, constraint_satisfaction
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from cascade_planner.cascadeboard.skeleton_inpainter import (
    load_model as load_inpainter,
    generate_multiple_skeletons,
    SkeletonResult,
    RTYPE_TO_ID, REACTION_TYPE_VOCAB,
    morgan_fp,
)
from cascade_planner.cascadeboard.learned_scorer import (
    load_scorer,
    score_route,
    build_scorer_dataset,
    collate_scorer_batch,
    SlotFeatures,
    ScoreResult,
)


@dataclass
class BenchmarkResult:
    target_smiles: str
    domain: str
    depth: int
    gt_route: list[dict]
    predicted_routes: list[dict]
    gt_match_at: int  # -1 if no match in top-K
    plan_time: float
    n_routes_generated: int


def _rxn_to_product(rxn_smiles: str) -> str:
    """Extract product from rxn_smiles."""
    if ">>" in rxn_smiles:
        return rxn_smiles.split(">>")[-1].strip()
    return ""


def _canonical(smi: str) -> str:
    """Canonicalize SMILES."""
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol)


def _gt_match(predicted_route: list[dict], gt_route: list[dict], threshold: float = 0.5) -> bool:
    """Check if predicted route matches ground truth.

    Match criteria: ≥threshold of GT steps have matching reaction type
    AND product overlap (Tanimoto > 0.5 on Morgan FP).
    """
    if len(predicted_route) != len(gt_route):
        return False

    matches = 0
    for pred_step, gt_step in zip(predicted_route, gt_route):
        # Type match
        pred_type = pred_step.get("reaction_type", "")
        gt_type = gt_step.get("transformation", "")
        type_match = pred_type == gt_type

        # Product overlap (Tanimoto)
        pred_prod = _canonical(pred_step.get("product", ""))
        gt_prod = _canonical(_rxn_to_product(gt_step.get("rxn_smiles", "")))

        if pred_prod and gt_prod:
            fp1 = morgan_fp(pred_prod)
            fp2 = morgan_fp(gt_prod)
            dot = np.dot(fp1, fp2)
            norm = np.sqrt(np.dot(fp1, fp1) * np.dot(fp2, fp2))
            tanimoto = dot / max(norm, 1e-8)
            prod_match = tanimoto > 0.5
        else:
            prod_match = False

        if type_match or prod_match:
            matches += 1

    return matches / len(gt_route) >= threshold


def _skeleton_to_slots(skel: SkeletonResult, n_steps: int) -> list[SlotFeatures]:
    """Convert a SkeletonResult to SlotFeatures for scoring."""
    slots = []
    for i in range(n_steps):
        sf = SlotFeatures(
            rtype_id=RTYPE_TO_ID.get(skel.types[i], 0),
            ec1_id=skel.ec1s[i],
            T_norm=(skel.Ts[i] - 37) / 30,
            pH_norm=(skel.pHs[i] - 7) / 3,
            is_observed=1,
        )
        slots.append(sf)
    return slots


def run_benchmark(
    bench_path: str = "data/benchmark_v2_100.json",
    inpainter_path: str = "results/shared/skeleton_inpainter/best.pt",
    scorer_path: str = "results/shared/learned_scorer/best.pt",
    k_skeletons: int = 5,
    device: str = "cuda",
    output_path: str = "results/v2/integrated_benchmark_results.json",
) -> dict:
    """Run the integrated benchmark."""
    print("Loading models...")
    inpainter = load_inpainter(inpainter_path, device=device)
    scorer = load_scorer(scorer_path, device=device)

    bench = json.loads(Path(bench_path).read_text())
    print(f"Benchmark: {len(bench)} targets")

    results = []
    gt_at_1 = gt_at_5 = 0
    plan_count = 0
    total_time = 0.0

    for idx, entry in enumerate(bench):
        target = entry["target_smiles"]
        domain = entry["route_domain"]
        depth = int(entry["depth"])
        gt_route = entry["gt_route"]

        t0 = time.time()

        # Layer 1: Generate K skeletons
        skeletons = generate_multiple_skeletons(
            inpainter, target, n_steps=depth, k=k_skeletons,
            domain=domain, objective="balanced",
            temperature=0.8, device=device,
        )

        # Layer 2: For now, we don't have live RetroChimera/EnzExpand fill.
        # We score the skeletons directly using the learned scorer.
        # This tests skeleton quality + scorer ranking.
        scored_routes = []
        for skel in skeletons:
            slots = _skeleton_to_slots(skel, depth)
            sr = score_route(scorer, target, slots, domain=domain, device=device)
            route_info = {
                "types": skel.types,
                "ec1s": skel.ec1s,
                "Ts": skel.Ts,
                "pHs": skel.pHs,
                "score": sr.route_score,
                "compat": sr.compat_pred,
                "opmode": sr.opmode_pred,
                "issues": sr.issues_pred,
                "skel_log_prob": skel.log_prob,
            }
            # Build predicted route for GT matching
            predicted_steps = []
            for i in range(depth):
                predicted_steps.append({
                    "reaction_type": skel.types[i],
                    "ec1": skel.ec1s[i],
                    "product": target if i == 0 else "",  # only first step product known
                    "T": skel.Ts[i],
                    "pH": skel.pHs[i],
                })
            route_info["steps"] = predicted_steps
            scored_routes.append(route_info)

        # Sort by scorer score (descending)
        scored_routes.sort(key=lambda r: r["score"], reverse=True)

        elapsed = time.time() - t0
        total_time += elapsed

        # Check GT match (type-only matching since we don't have filled molecules)
        gt_match_at = -1
        for rank, route in enumerate(scored_routes[:5]):
            # Type-sequence match
            pred_types = route["types"]
            gt_types = [s.get("transformation", "") for s in gt_route]
            if len(pred_types) == len(gt_types):
                type_matches = sum(1 for p, g in zip(pred_types, gt_types) if p == g)
                if type_matches / len(gt_types) >= 0.5:
                    gt_match_at = rank
                    break

        if gt_match_at == 0:
            gt_at_1 += 1
        if gt_match_at >= 0:
            gt_at_5 += 1
        if scored_routes:
            plan_count += 1

        results.append(BenchmarkResult(
            target_smiles=target,
            domain=domain,
            depth=depth,
            gt_route=gt_route,
            predicted_routes=scored_routes[:5],
            gt_match_at=gt_match_at,
            plan_time=elapsed,
            n_routes_generated=len(scored_routes),
        ))

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/100] GT@1={gt_at_1}/{idx+1} GT@5={gt_at_5}/{idx+1} time={total_time/(idx+1):.2f}s/target")

    # Summary
    n = len(bench)
    summary = {
        "n_targets": n,
        "plan_rate": plan_count / n,
        "GT@1": gt_at_1 / n,
        "GT@5": gt_at_5 / n,
        "avg_time_per_target": total_time / n,
        "total_time": total_time,
    }

    # Per-domain breakdown
    from collections import Counter, defaultdict
    domain_stats = defaultdict(lambda: {"n": 0, "gt1": 0, "gt5": 0})
    for r in results:
        d = r.domain
        domain_stats[d]["n"] += 1
        if r.gt_match_at == 0:
            domain_stats[d]["gt1"] += 1
        if r.gt_match_at >= 0:
            domain_stats[d]["gt5"] += 1

    summary["per_domain"] = {
        d: {
            "n": s["n"],
            "GT@1": s["gt1"] / s["n"],
            "GT@5": s["gt5"] / s["n"],
        }
        for d, s in domain_stats.items()
    }

    print(f"\n{'='*60}")
    print(f"INTEGRATED BENCHMARK RESULTS (OA-ARM + Learned Scorer)")
    print(f"{'='*60}")
    print(f"  Plan rate:  {summary['plan_rate']*100:.0f}%")
    print(f"  GT@1:       {summary['GT@1']*100:.1f}%")
    print(f"  GT@5:       {summary['GT@5']*100:.1f}%")
    print(f"  Avg time:   {summary['avg_time_per_target']:.2f}s/target")
    print(f"\n  Per domain:")
    for d, s in summary["per_domain"].items():
        print(f"    {d:25s} n={s['n']:2d}  GT@1={s['GT@1']*100:5.1f}%  GT@5={s['GT@5']*100:5.1f}%")

    # Save results
    output = {
        "summary": summary,
        "results": [
            {
                "target": r.target_smiles,
                "domain": r.domain,
                "depth": r.depth,
                "gt_match_at": r.gt_match_at,
                "plan_time": r.plan_time,
                "n_routes": r.n_routes_generated,
                "top_route": r.predicted_routes[0] if r.predicted_routes else None,
            }
            for r in results
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to {output_path}")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--inpainter", default="results/shared/skeleton_inpainter/best.pt")
    ap.add_argument("--scorer", default="results/shared/learned_scorer/best.pt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", default="results/v2/integrated_benchmark_results.json")
    args = ap.parse_args()

    run_benchmark(
        bench_path=args.bench,
        inpainter_path=args.inpainter,
        scorer_path=args.scorer,
        k_skeletons=args.k,
        output_path=args.output,
    )
