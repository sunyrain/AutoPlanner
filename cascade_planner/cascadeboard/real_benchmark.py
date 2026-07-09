"""Strict real-candidate benchmark for CascadeBoard++.

This runner is publication-oriented: it never falls back to mock candidates.
Missing cached candidates are reported as candidate coverage failures.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from cascade_planner.cascadeboard import CascadeBoard
from cascade_planner.cascadeboard.candidate_graph import (
    CandidateHypergraph, CandidateReaction,
)
from cascade_planner.cascadeboard.constraint_compiler import ConstraintCompiler
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.planner import ScoredBoard, resample_pareto_diverse
from cascade_planner.cascadeboard.candidate_cache import (
    cache_summary, canon_smiles, merge_candidate_caches,
)


class StrictCachedGraph(CandidateHypergraph):
    """Candidate graph backed only by cache entries; no mock fallback."""

    def __init__(self, cache: dict[str, list[dict]], **kwargs):
        super().__init__(**kwargs)
        self._cache = cache
        self.cache_misses: set[str] = set()

    def _get_candidates(self, product, depth, compiled):
        product_key = canon_smiles(product) or product
        rows = self._cache.get(product_key) or self._cache.get(product) or []
        if not rows:
            self.cache_misses.add(product)
            return []
        cands = []
        for row in rows:
            cands.append(CandidateReaction(
                product=row.get("product") or product,
                main_reactant=row.get("main_reactant", ""),
                aux_reactants=row.get("aux_reactants", []),
                reaction_smiles=row.get("reaction_smiles") or row.get("rxn_smiles", ""),
                reaction_type=row.get("reaction_type") or row.get("type", ""),
                ec=row.get("ec"),
                enzyme_uid=row.get("enzyme_uid"),
                score=float(row.get("score", 0.0)),
                source=row.get("source", "cache"),
                metadata={
                    k: row[k]
                    for k in (
                        "rank", "T", "pH", "solvent", "e_enzyme",
                        "dual_tower_score", "enzyme_source",
                    )
                    if k in row
                },
            ))
        return sorted(cands, key=lambda c: -c.score)


def _git_status_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _canon(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _gt_molecule_set(gt_route: list[dict]) -> set[str]:
    mols = set()
    for step in gt_route or []:
        rxn = step.get("rxn_smiles") or ""
        if ">>" not in rxn:
            continue
        lhs, rhs = rxn.split(">>", 1)
        for part in lhs.split(".") + rhs.split("."):
            c = _canon(part.strip())
            if c:
                mols.add(c)
    return mols


def _board_molecule_set(board) -> set[str]:
    mols = set()
    for slot in board.slots:
        for smi in [slot.product, slot.main_reactant, *slot.aux_reactants]:
            c = _canon(smi)
            if c:
                mols.add(c)
    return mols


def _route_overlap(board, gt_route: list[dict]) -> float:
    gt = _gt_molecule_set(gt_route)
    if not gt:
        return 0.0
    pred = _board_molecule_set(board)
    return len(gt & pred) / len(gt)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + pow(2.718281828, -x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


PLATT_PARAMS: dict[str, tuple[float, float]] = {
    "gt_at_5": (-0.5860, -6.6242),
    "strict_stock_solve": (-2.7645, -24.8630),
}


def _platt_confidence(energy: float, outcome_key: str) -> float:
    """Calibrated confidence via Platt scaling: P = sigmoid(coef*energy + intercept)."""
    coef, intercept = PLATT_PARAMS.get(outcome_key, (0.0, 0.0))
    return _sigmoid(coef * energy + intercept)


def _ece(rows: list[dict], outcome_key: str, *, bins: int = 10, calibrated: bool = False) -> float | None:
    usable = [r for r in rows if r.get("best_energy") is not None and outcome_key in r]
    if not usable:
        return None
    total = len(usable)
    err = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bucket = []
        for r in usable:
            e = float(r["best_energy"])
            if calibrated:
                conf = _platt_confidence(e, outcome_key)
            else:
                conf = _sigmoid(-e)
            if lo <= conf < hi or (b == bins - 1 and conf <= hi):
                bucket.append((conf, 1.0 if r.get(outcome_key) else 0.0))
        if not bucket:
            continue
        avg_conf = sum(x for x, _ in bucket) / len(bucket)
        avg_acc = sum(y for _, y in bucket) / len(bucket)
        err += (len(bucket) / total) * abs(avg_conf - avg_acc)
    return round(err, 4)


def run_real_benchmark(
    *,
    bench_path: str,
    rc_cache_path: str,
    output: str,
    enz_cache_path: str | None = None,
    n_targets: int | None = None,
    n_particles: int = 32,
    n_final: int = 5,
    branch_factor: int = 10,
    energy_weights: dict[str, float] | None = None,
    resample: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    bench = json.loads(Path(bench_path).read_text())
    if n_targets is not None:
        bench = bench[:n_targets]
    rc_cache = json.loads(Path(rc_cache_path).read_text())
    caches = [rc_cache]
    candidate_sources = ["retrochimera_cache"]
    if enz_cache_path:
        enz_cache = json.loads(Path(enz_cache_path).read_text())
        caches.append(enz_cache)
        candidate_sources.append("enzexpand_cache")
    cache = merge_candidate_caches(*caches)
    cache_meta = cache_summary(cache)
    api = EnergyAPI(weights=energy_weights)
    compiler = ConstraintCompiler()

    rows = []
    t0 = time.time()
    for item in bench:
        target = item["target_smiles"]
        depth = min(int(item.get("depth") or 3), 6)
        row = {
            "target": target,
            "domain": item.get("route_domain", ""),
            "depth": depth,
            "planned": False,
            "strict_stock_solve": False,
            "gt_at_1": False,
            "gt_at_5": False,
            "overlap_top1": 0.0,
            "failure_reason": "",
            "time_s": 0.0,
            "candidate_source": "+".join(candidate_sources),
            "mock_fallback": False,
        }
        start = time.time()
        try:
            template = CascadeBoard.from_n_steps(depth, target)
            compiled = compiler.compile(template)
            graph = StrictCachedGraph(
                cache,
                max_depth=depth,
                branch_factor=branch_factor,
            )
            graph.build(target, compiled)
            graph.propagate_constraints(compiled)
            source_counts = Counter()
            if graph.root:
                stack = [graph.root]
                while stack:
                    node = stack.pop()
                    for cand, child in node.children:
                        source_counts[cand.source or "unknown"] += 1
                        stack.append(child)
            row["candidate_source_counts"] = dict(source_counts)
            if graph.is_empty():
                row["failure_reason"] = "candidate_missing"
                row["cache_misses"] = sorted(graph.cache_misses)[:10]
                rows.append(row)
                continue

            paths = graph.sample_paths(n=n_particles)
            if not paths:
                row["failure_reason"] = "no_sampled_paths"
                row["cache_misses"] = sorted(graph.cache_misses)[:10]
                rows.append(row)
                continue

            scored = []
            for path in paths:
                board = graph.path_to_board(path, target)
                energy = api.compute_energy(board, compiled)
                quality = api.compute_quality(board)
                risk = api.compute_risk(board)
                scored.append(ScoredBoard(
                    board=board,
                    energy=energy,
                    quality=quality,
                    risk=risk,
                    posterior=-energy,
                ))
            if resample:
                scored = resample_pareto_diverse(scored, K=n_final)
            else:
                scored.sort(key=lambda x: x.posterior, reverse=True)
                scored = scored[:n_final]
            scored.sort(key=lambda x: x.posterior, reverse=True)

            overlaps = [_route_overlap(s.board, item.get("gt_route", [])) for s in scored]
            row["planned"] = bool(scored)
            row["strict_stock_solve"] = bool(scored and api.score_stock(scored[0].board) >= 1.0)
            row["overlap_top1"] = round(overlaps[0], 4) if overlaps else 0.0
            row["gt_at_1"] = bool(overlaps and overlaps[0] >= 0.5)
            row["gt_at_5"] = bool(any(v >= 0.5 for v in overlaps[:5]))
            row["n_routes"] = len(scored)
            row["best_energy"] = round(scored[0].energy, 4) if scored else None
            row["failure_reason"] = "" if row["planned"] else "no_routes"
            row["cache_misses"] = sorted(graph.cache_misses)[:10]
        except Exception as exc:
            row["failure_reason"] = f"error:{type(exc).__name__}"
            row["error"] = str(exc)[:500]
        finally:
            row["time_s"] = round(time.time() - start, 3)
            rows.append(row)

    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)

    def summarize(subrows: list[dict]) -> dict[str, Any]:
        n = len(subrows)
        return {
            "n": n,
            "plan_rate": sum(r["planned"] for r in subrows) / max(n, 1),
            "strict_stock_solve": sum(r["strict_stock_solve"] for r in subrows) / max(n, 1),
            "gt_at_1": sum(r["gt_at_1"] for r in subrows) / max(n, 1),
            "gt_at_5": sum(r["gt_at_5"] for r in subrows) / max(n, 1),
            "mean_overlap_top1": sum(r["overlap_top1"] for r in subrows) / max(n, 1),
            "mean_time_s": sum(r["time_s"] for r in subrows) / max(n, 1),
            "failures": dict(Counter(r["failure_reason"] or "planned" for r in subrows)),
            "calibration": {
                "confidence_proxy": "sigmoid(-best_energy)",
                "ece_gt_at_5_raw": _ece(subrows, "gt_at_5"),
                "ece_strict_stock_solve_raw": _ece(subrows, "strict_stock_solve"),
                "ece_gt_at_5_calibrated": _ece(subrows, "gt_at_5", calibrated=True),
                "ece_strict_stock_solve_calibrated": _ece(subrows, "strict_stock_solve", calibrated=True),
                "calibration_method": "platt_scaling",
            },
        }

    result = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "bench_path": bench_path,
            "rc_cache_path": rc_cache_path,
            "enz_cache_path": enz_cache_path,
            "n_targets": len(rows),
            "n_particles": n_particles,
            "n_final": n_final,
            "branch_factor": branch_factor,
            "seed": seed,
            "energy_weights": energy_weights or api.weights,
            "resample": resample,
            "candidate_sources": candidate_sources,
            "candidate_cache_summary": cache_meta,
            "mock_fallback": False,
            "git_status_short": _git_status_short(),
            "elapsed_s": round(time.time() - t0, 3),
        },
        "overall": summarize(rows),
        "by_domain": {domain: summarize(subrows) for domain, subrows in by_domain.items()},
        "rows": rows,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--rc-cache", default="results/shared/retrochimera_candidates_depth2.json")
    ap.add_argument("--enz-cache", default=None)
    ap.add_argument("--output", default="results/v2/cascadeboard_real_benchmark.json")
    ap.add_argument("--n-targets", type=int, default=None)
    ap.add_argument("--n-particles", type=int, default=32)
    ap.add_argument("--n-final", type=int, default=5)
    ap.add_argument("--branch-factor", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-compatibility-energy", action="store_true")
    ap.add_argument("--no-resample", action="store_true")
    args = ap.parse_args()
    result = run_real_benchmark(
        bench_path=args.bench,
        rc_cache_path=args.rc_cache,
        enz_cache_path=args.enz_cache,
        output=args.output,
        n_targets=args.n_targets,
        n_particles=args.n_particles,
        n_final=args.n_final,
        branch_factor=args.branch_factor,
        seed=args.seed,
        energy_weights=(
            {**EnergyAPI().weights, "compatibility": 0.0}
            if args.no_compatibility_energy else None
        ),
        resample=not args.no_resample,
    )
    print(json.dumps({
        "overall": result["overall"],
        "by_domain": result["by_domain"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
