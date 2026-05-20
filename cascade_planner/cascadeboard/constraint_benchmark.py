"""Constraint Satisfaction Benchmark for CascadeBoard++.

COMPLETE spec §增强7: give target + constraints → measure
hard_constraint_success_rate, soft_constraint_score, conflict detection accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard import CascadeBoard, CompiledConstraints
from cascade_planner.cascadeboard.benchmarks import _make_cached_graph
from cascade_planner.cascadeboard.constraint_compiler import (
    ConstraintCompiler, EC1_T_RANGES, EC1_PH_RANGES,
)
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.planner import (
    CascadeBoardPlanner, apply_edit, propose_edits, propose_neural_edits,
    load_cascadeboard_model, resample_pareto_diverse, ScoredBoard,
)

import numpy as np


# ---------------------------------------------------------------------------
# Constraint scenario generators
# ---------------------------------------------------------------------------

def _scenario_fix_ec(item: dict, rng: random.Random) -> dict | None:
    """Fix one step's EC class from GT."""
    gt = item.get("gt_route", [])
    ec_steps = [(i, s["ec_number"]) for i, s in enumerate(gt) if s.get("ec_number")]
    if not ec_steps:
        return None
    idx, ec = rng.choice(ec_steps)
    ec1 = ec.split(".")[0]
    return {
        "name": "fix_ec",
        "description": f"Step {idx} must use EC {ec1}.x",
        "constraints": {"fixed_steps": [{"index": idx, "values": {"ec": f"{ec1}.x"}}]},
        "check": lambda board: _check_ec_prefix(board, idx, ec1),
    }


def _scenario_fix_n_steps(item: dict, rng: random.Random) -> dict | None:
    """Fix the number of steps."""
    depth = item.get("depth", 2)
    if depth < 2:
        return None
    return {
        "name": "fix_n_steps",
        "description": f"Route must have exactly {depth} steps",
        "constraints": {"n_steps": depth},
        "check": lambda board: board.n_steps == depth,
    }


def _scenario_max_delta_T(item: dict, rng: random.Random) -> dict | None:
    """Max temperature difference between adjacent steps."""
    delta = rng.choice([10, 15, 20])
    return {
        "name": f"max_delta_T_{delta}",
        "description": f"Adjacent steps must have ΔT ≤ {delta}°C",
        "constraints": {"max_delta_T": delta},
        "check": lambda board: _check_max_delta_T(board, delta),
    }


def _scenario_max_delta_pH(item: dict, rng: random.Random) -> dict | None:
    """Max pH difference between adjacent steps."""
    delta = rng.choice([1.0, 1.5, 2.0])
    return {
        "name": f"max_delta_pH_{delta}",
        "description": f"Adjacent steps must have ΔpH ≤ {delta}",
        "constraints": {"max_delta_pH": delta},
        "check": lambda board: _check_max_delta_pH(board, delta),
    }


def _scenario_prefer_enzymatic(item: dict, rng: random.Random) -> dict | None:
    """Prefer enzymatic steps."""
    return {
        "name": "prefer_enzymatic",
        "description": "Prefer enzymatic catalysis",
        "constraints": {"prefer_enzymatic": True},
        "check": lambda board: sum(1 for s in board.slots if s.is_enzymatic()) > 0,
    }


def _scenario_one_pot(item: dict, rng: random.Random) -> dict | None:
    """One-pot constraint."""
    return {
        "name": "one_pot",
        "description": "One-pot cascade (no intermediate purification)",
        "constraints": {"one_pot": True},
        "check": lambda board: _check_one_pot(board),
    }


def _scenario_exclude_reaction_type(item: dict, rng: random.Random) -> dict | None:
    """Exclude a specific reaction type that is NOT in the GT."""
    gt_types = {s.get("transformation") for s in item.get("gt_route", []) if s.get("transformation")}
    all_types = {"oxidation", "reduction", "acylation", "hydrolysis", "amination", "C_C_coupling"}
    excludable = all_types - gt_types
    if not excludable:
        return None
    excluded = rng.choice(sorted(excludable))
    return {
        "name": f"exclude_{excluded}",
        "description": f"No {excluded} steps allowed",
        "constraints": {"fixed_steps": [], "exclude_reaction_type": excluded},
        "check": lambda board: all(s.reaction_type != excluded for s in board.slots),
    }


def _scenario_combined(item: dict, rng: random.Random) -> dict | None:
    """Combine fix_n_steps + max_delta_T."""
    depth = item.get("depth", 2)
    if depth < 2:
        return None
    delta = 15
    return {
        "name": "combined_steps_deltaT",
        "description": f"Exactly {depth} steps AND ΔT ≤ {delta}°C",
        "constraints": {"n_steps": depth, "max_delta_T": delta},
        "check": lambda board: board.n_steps == depth and _check_max_delta_T(board, delta),
    }


SCENARIO_GENERATORS = [
    _scenario_fix_ec,
    _scenario_fix_n_steps,
    _scenario_max_delta_T,
    _scenario_max_delta_pH,
    _scenario_prefer_enzymatic,
    _scenario_one_pot,
    _scenario_exclude_reaction_type,
    _scenario_combined,
]


# ---------------------------------------------------------------------------
# Constraint checkers
# ---------------------------------------------------------------------------

def _check_ec_prefix(board: CascadeBoard, slot_idx: int, ec1: str) -> bool:
    if slot_idx >= board.n_steps:
        return False
    ec = board.slots[slot_idx].ec
    if not ec:
        return False
    return ec.startswith(ec1 + ".")


def _check_max_delta_T(board: CascadeBoard, max_delta: float) -> bool:
    for i in range(board.n_steps - 1):
        t_a = board.slots[i].T
        t_b = board.slots[i + 1].T
        if t_a is not None and t_b is not None:
            if abs(t_a - t_b) > max_delta:
                return False
    return True


def _check_max_delta_pH(board: CascadeBoard, max_delta: float) -> bool:
    for i in range(board.n_steps - 1):
        ph_a = board.slots[i].pH
        ph_b = board.slots[i + 1].pH
        if ph_a is not None and ph_b is not None:
            if abs(ph_a - ph_b) > max_delta:
                return False
    return True


def _check_one_pot(board: CascadeBoard) -> bool:
    for i in range(board.n_steps - 1):
        t_a = board.slots[i].T
        t_b = board.slots[i + 1].T
        ph_a = board.slots[i].pH
        ph_b = board.slots[i + 1].pH
        if t_a is not None and t_b is not None and abs(t_a - t_b) > 10:
            return False
        if ph_a is not None and ph_b is not None and abs(ph_a - ph_b) > 1.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------

def _run_scenario(
    item: dict,
    scenario: dict,
    rc_cache: dict,
    enz_cache: dict | None,
    api: EnergyAPI,
    model=None,
    device: str = "cpu",
    n_particles: int = 16,
    n_refine: int = 3,
) -> dict[str, Any]:
    smi = item["target_smiles"]
    depth = item.get("depth", 2)
    constraints = scenario["constraints"]

    n_steps = constraints.get("n_steps", depth)
    template = CascadeBoard.from_n_steps(n_steps, smi)

    for fc in constraints.get("fixed_steps", []):
        template.fix(fc["index"], **fc["values"])
    for k in ("one_pot", "max_delta_T", "max_delta_pH", "prefer_enzymatic"):
        if k in constraints:
            template.set_global_constraint(k, constraints[k])

    compiler = ConstraintCompiler()
    compiled = compiler.compile(template, raw_constraints=constraints)

    graph = _make_cached_graph(rc_cache, max_depth=min(n_steps, 3), branch_factor=8)
    graph.build(smi, compiled)
    graph.propagate_constraints(compiled)

    if graph.is_empty():
        return {
            "planned": False,
            "hard_satisfied": False,
            "soft_score": 0.0,
            "reason": "empty_graph",
        }

    paths = graph.sample_paths(n=n_particles)
    if not paths:
        return {
            "planned": False,
            "hard_satisfied": False,
            "soft_score": 0.0,
            "reason": "no_paths",
        }

    particles: list[ScoredBoard] = []
    for path in paths:
        board = graph.path_to_board(path, smi)
        for k, v in template.global_constraints.items():
            board.set_global_constraint(k, v)
        for i, slot in enumerate(template.slots):
            if i < board.n_steps:
                for fld in slot.fixed_fields:
                    val = getattr(slot, fld)
                    setattr(board.slots[i], fld, val)
                    board.slots[i].fixed_fields.add(fld)

        energy = api.compute_energy(board, compiled)
        particles.append(ScoredBoard(board=board, energy=energy, posterior=-energy))

    for _iter in range(n_refine):
        proposals: list[ScoredBoard] = []
        for p in particles:
            if model is not None:
                edits = propose_neural_edits(p.board, model, api, compiled, device=device, m=4)
            else:
                edits = propose_edits(p.board, api, compiled, m=4)
            for edit in edits:
                new_board = apply_edit(p.board, edit)
                if not compiled.hard_satisfied(new_board):
                    continue
                energy = api.compute_energy(new_board, compiled)
                proposals.append(ScoredBoard(board=new_board, energy=energy, posterior=-energy))
        particles = resample_pareto_diverse(particles + proposals, K=n_particles)

    particles.sort(key=lambda p: p.posterior, reverse=True)
    best = particles[0] if particles else None

    if best is None:
        return {"planned": False, "hard_satisfied": False, "soft_score": 0.0, "reason": "no_particles"}

    board = best.board
    check_fn = scenario["check"]
    hard_ok = check_fn(board)
    hard_compiled_ok = compiled.hard_satisfied(board)

    soft_score = _compute_soft_score(board, constraints)

    return {
        "planned": True,
        "hard_satisfied": bool(hard_ok and hard_compiled_ok),
        "soft_score": round(soft_score, 4),
        "energy": round(best.energy, 4),
        "n_steps": board.n_steps,
        "n_enzymatic": sum(1 for s in board.slots if s.is_enzymatic()),
    }


def _compute_soft_score(board: CascadeBoard, constraints: dict) -> float:
    scores = []
    if "max_delta_T" in constraints:
        max_dt = constraints["max_delta_T"]
        for i in range(board.n_steps - 1):
            ta, tb = board.slots[i].T, board.slots[i + 1].T
            if ta is not None and tb is not None:
                dt = abs(ta - tb)
                scores.append(max(0.0, 1.0 - dt / max_dt))
    if "max_delta_pH" in constraints:
        max_dp = constraints["max_delta_pH"]
        for i in range(board.n_steps - 1):
            pa, pb = board.slots[i].pH, board.slots[i + 1].pH
            if pa is not None and pb is not None:
                dp = abs(pa - pb)
                scores.append(max(0.0, 1.0 - dp / max_dp))
    if constraints.get("one_pot"):
        scores.append(1.0 if _check_one_pot(board) else 0.0)
    if constraints.get("prefer_enzymatic"):
        n_enz = sum(1 for s in board.slots if s.is_enzymatic())
        scores.append(min(1.0, n_enz / max(board.n_steps, 1)))
    return float(np.mean(scores)) if scores else 1.0


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_constraint_benchmark(
    *,
    bench_path: str,
    rc_cache_path: str,
    enz_cache_path: str | None = None,
    output: str,
    checkpoint: str | None = None,
    n_targets: int = 50,
    scenarios_per_target: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    bench = json.loads(Path(bench_path).read_text())[:n_targets]
    rc_cache = json.loads(Path(rc_cache_path).read_text())
    enz_cache = json.loads(Path(enz_cache_path).read_text()) if enz_cache_path else None
    api = EnergyAPI()
    rng = random.Random(seed)
    model = load_cascadeboard_model(checkpoint, device=device) if checkpoint else None

    rows: list[dict] = []
    for item in bench:
        available = []
        for gen in SCENARIO_GENERATORS:
            sc = gen(item, rng)
            if sc is not None:
                available.append(sc)
        if not available:
            continue
        chosen = rng.sample(available, min(scenarios_per_target, len(available)))
        for scenario in chosen:
            t0 = time.time()
            result = _run_scenario(item, scenario, rc_cache, enz_cache, api, model, device)
            result.update({
                "target": item["target_smiles"],
                "domain": item.get("route_domain", ""),
                "scenario": scenario["name"],
                "scenario_desc": scenario["description"],
                "time_s": round(time.time() - t0, 3),
            })
            rows.append(result)

    # Aggregate
    n_total = len(rows)
    n_planned = sum(r["planned"] for r in rows)
    n_hard_ok = sum(r["hard_satisfied"] for r in rows)
    soft_scores = [r["soft_score"] for r in rows if r["planned"]]

    by_scenario: dict[str, dict] = {}
    for name in sorted({r["scenario"] for r in rows}):
        sub = [r for r in rows if r["scenario"] == name]
        n = len(sub)
        by_scenario[name] = {
            "n": n,
            "plan_rate": sum(r["planned"] for r in sub) / max(n, 1),
            "hard_success_rate": sum(r["hard_satisfied"] for r in sub) / max(n, 1),
            "mean_soft_score": float(np.mean([r["soft_score"] for r in sub if r["planned"]])) if any(r["planned"] for r in sub) else 0.0,
        }

    by_domain: dict[str, dict] = {}
    for domain in sorted({r["domain"] for r in rows}):
        sub = [r for r in rows if r["domain"] == domain]
        n = len(sub)
        by_domain[domain] = {
            "n": n,
            "plan_rate": sum(r["planned"] for r in sub) / max(n, 1),
            "hard_success_rate": sum(r["hard_satisfied"] for r in sub) / max(n, 1),
            "mean_soft_score": float(np.mean([r["soft_score"] for r in sub if r["planned"]])) if any(r["planned"] for r in sub) else 0.0,
        }

    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "bench_path": bench_path,
            "rc_cache_path": rc_cache_path,
            "checkpoint": checkpoint,
            "n_targets": n_targets,
            "scenarios_per_target": scenarios_per_target,
            "seed": seed,
        },
        "overall": {
            "n_scenarios": n_total,
            "plan_rate": n_planned / max(n_total, 1),
            "hard_constraint_success_rate": n_hard_ok / max(n_total, 1),
            "mean_soft_score": float(np.mean(soft_scores)) if soft_scores else 0.0,
        },
        "by_scenario": by_scenario,
        "by_domain": by_domain,
        "rows": rows,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="CascadeBoard constraint satisfaction benchmark")
    ap.add_argument("--bench", default="data/benchmark_v2_100.json")
    ap.add_argument("--rc-cache", default="results/shared/retrochimera_candidates_depth2.json")
    ap.add_argument("--enz-cache", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default="results/v2/cascadeboard_constraint_benchmark.json")
    ap.add_argument("--n-targets", type=int, default=50)
    ap.add_argument("--scenarios-per-target", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    report = run_constraint_benchmark(
        bench_path=args.bench,
        rc_cache_path=args.rc_cache,
        enz_cache_path=args.enz_cache,
        output=args.output,
        checkpoint=args.checkpoint,
        n_targets=args.n_targets,
        scenarios_per_target=args.scenarios_per_target,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({
        "overall": report["overall"],
        "by_scenario": report["by_scenario"],
    }, indent=2))


if __name__ == "__main__":
    main()
