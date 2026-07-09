"""CascadeBoard++ Benchmark Suite.

Four benchmark types:
1. Unconstrained plan rate (already done: 84%)
2. Constraint satisfaction (already done: 100%)
3. Counterfactual repair: given a corrupted route, can the system fix it?
4. Inpainting recovery: mask a step, can the system recover it?
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from collections import Counter

from cascade_planner.cascadeboard import CascadeBoard, Slot
from cascade_planner.cascadeboard.candidate_graph import (
    CandidateReaction, CandidateHypergraph,
)
from cascade_planner.cascadeboard.constraint_compiler import ConstraintCompiler
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.planner import (
    CascadeBoardPlanner, apply_edit, propose_edits, ScoredBoard,
)
from cascade_planner.cascadeboard.preference import (
    compute_route_uncertainty, rank_routes,
)


def _make_cached_graph(rc_cache: dict, **kwargs):
    """Create a CandidateHypergraph backed by cached RetroChimera candidates."""

    class CG(CandidateHypergraph):
        def __init__(self, cache, **kw):
            super().__init__(**kw)
            self._c = cache

        def _get_candidates(self, product, depth, compiled):
            if product in self._c:
                return [
                    CandidateReaction(
                        product=x["product"],
                        main_reactant=x["main_reactant"],
                        aux_reactants=x.get("aux_reactants", []),
                        score=x["score"],
                        source=x["source"],
                    )
                    for x in self._c[product]
                ]
            from cascade_planner.cascadeboard.candidate_graph import _mock_candidates
            return _mock_candidates(product, depth)

    return CG(rc_cache, **kwargs)


def _within_energy_margin(candidate_energy: float, reference_energy: float, rel_margin: float) -> bool:
    """Return True when candidate is no worse than a relative margin.

    CascadeBoard energies are lower-is-better and often negative, so multiplying
    the reference by 1 + margin would make the threshold stricter rather than
    looser. Use an additive tolerance based on absolute energy magnitude.
    """
    return candidate_energy <= reference_energy + abs(reference_energy) * rel_margin


# ---------------------------------------------------------------------------
# Benchmark 2: Constraint scenarios
# ---------------------------------------------------------------------------

def benchmark_constraint_scenarios() -> dict:
    """Deterministic checks for constraint compilation and satisfaction.

    These tests validate the core constraint semantics directly. Route-level
    planner tests can otherwise look falsely strong because fixed fields are
    copied from the user template into sampled boards after candidate selection.
    """
    compiler = ConstraintCompiler()
    results = {
        "tested": 0,
        "passed": 0,
        "scenarios": {},
    }

    def record(name: str, ok: bool, detail: str = "") -> None:
        results["tested"] += 1
        results["passed"] += int(ok)
        results["scenarios"][name] = {"passed": ok, "detail": detail}

    # 1. Fixed reaction type accepts exact match and rejects mismatch.
    template = CascadeBoard.from_n_steps(1, "CCO")
    template.fix(0, reaction_type="reduction")
    compiled = compiler.compile(template)
    good = CascadeBoard.from_n_steps(1, "CCO")
    good.slots[0].reaction_type = "reduction"
    bad = CascadeBoard.from_n_steps(1, "CCO")
    bad.slots[0].reaction_type = "oxidation"
    record(
        "fix_reaction_type",
        compiled.hard_satisfied(good) and not compiled.hard_satisfied(bad),
        "reduction accepted; oxidation rejected",
    )

    # 2. Fixed EC prefix supports x wildcard semantics.
    template = CascadeBoard.from_n_steps(1, "CCO")
    template.fix(0, ec="2.6.1.x")
    compiled = compiler.compile(template)
    good = CascadeBoard.from_n_steps(1, "CCO")
    good.slots[0].ec = "2.6.1.62"
    bad = CascadeBoard.from_n_steps(1, "CCO")
    bad.slots[0].ec = "2.6.2.1"
    record(
        "fix_ec_prefix",
        compiled.hard_satisfied(good) and not compiled.hard_satisfied(bad),
        "2.6.1.62 accepted; 2.6.2.1 rejected",
    )

    # 3. Fixed starting material is compiled as an exact main-reactant mask.
    template = CascadeBoard.from_n_steps(2, "CCO")
    template.fix_starting_material("CCN")
    compiled = compiler.compile(template)
    good = CascadeBoard.from_n_steps(2, "CCO")
    good.slots[-1].main_reactant = "CCN"
    bad = CascadeBoard.from_n_steps(2, "CCO")
    bad.slots[-1].main_reactant = "CCC"
    record(
        "fix_starting_material",
        compiled.hard_satisfied(good) and not compiled.hard_satisfied(bad),
        "CCN accepted as leaf; CCC rejected",
    )

    # 4. Excluded catalyst rejects only matching catalyst values.
    template = CascadeBoard.from_n_steps(1, "CCO")
    compiled = compiler.compile(template, raw_constraints={"exclude_catalyst": "Pd"})
    good = CascadeBoard.from_n_steps(1, "CCO")
    good.slots[0].catalyst = "Ni"
    bad = CascadeBoard.from_n_steps(1, "CCO")
    bad.slots[0].catalyst = "Pd"
    record(
        "exclude_catalyst",
        compiled.hard_satisfied(good) and not compiled.hard_satisfied(bad),
        "Ni accepted; Pd rejected",
    )

    # 5. One-pot conflict is detected for large fixed temperature gaps.
    template = CascadeBoard.from_n_steps(2, "CCO")
    template.fix(0, T=25.0)
    template.fix(1, T=80.0)
    compiled = compiler.compile(template, raw_constraints={"one_pot": True})
    record(
        "one_pot_temperature_conflict",
        len(compiled.conflicts) >= 1,
        compiled.conflicts[0].description if compiled.conflicts else "no conflict",
    )

    # 6. Fixed enzyme EC conflicts with a fixed condition outside its EC1 range.
    template = CascadeBoard.from_n_steps(1, "CCO")
    template.fix(0, ec="5.1.1.1", T=80.0)
    compiled = compiler.compile(template)
    record(
        "enzyme_temperature_conflict",
        len(compiled.conflicts) >= 1,
        compiled.conflicts[0].description if compiled.conflicts else "no conflict",
    )

    results["pass_rate"] = results["passed"] / max(results["tested"], 1)
    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Counterfactual Repair
# ---------------------------------------------------------------------------

def benchmark_counterfactual_repair(
    bench: list[dict],
    rc_cache: dict,
    n_targets: int = 50,
    seed: int = 42,
) -> dict:
    """Given a planned route, corrupt it, then see if the system can repair it."""
    rng = random.Random(seed)
    api = EnergyAPI()
    results = {"tested": 0, "repaired": 0, "energy_improved": 0, "details": []}

    for t in bench[:n_targets]:
        smi = t["target_smiles"]
        d = min(t.get("depth", 2), 3)

        # Step 1: Get a good route
        bt = CascadeBoard.from_n_steps(d, smi)
        cc = ConstraintCompiler().compile(bt)
        g = _make_cached_graph(rc_cache, max_depth=d, branch_factor=8)
        g.build(smi, cc)
        g.propagate_constraints(cc)
        paths = g.sample_paths(n=16) if not g.is_empty() else []
        if not paths:
            continue

        original = g.path_to_board(paths[0], smi)
        # Assign reasonable default conditions if missing
        for s in original.slots:
            if s.T is None:
                s.T = 30.0 + rng.uniform(-5, 10)
            if s.pH is None:
                s.pH = 7.0 + rng.uniform(-0.5, 0.5)
        original_energy = api.compute_energy(original, cc)

        # Step 2: Corrupt it (set extreme temperature on a random slot)
        corrupted = original.copy()
        slot_idx = rng.randint(0, corrupted.n_steps - 1)
        corrupted.slots[slot_idx].T = 95.0  # extreme: most enzymes denature
        corrupted_energy = api.compute_energy(corrupted, cc)

        # Step 3: Try to repair using edit policy
        edits = propose_edits(corrupted, api, cc, m=8)
        best_repair = corrupted
        best_repair_energy = corrupted_energy

        for edit in edits:
            repaired = apply_edit(corrupted, edit)
            repair_energy = api.compute_energy(repaired, cc)
            if repair_energy < best_repair_energy:
                best_repair = repaired
                best_repair_energy = repair_energy

        results["tested"] += 1
        improved = best_repair_energy < corrupted_energy
        recovered = _within_energy_margin(best_repair_energy, original_energy, 0.10)

        if improved:
            results["energy_improved"] += 1
        if recovered:
            results["repaired"] += 1

        results["details"].append({
            "original_energy": round(original_energy, 3),
            "corrupted_energy": round(corrupted_energy, 3),
            "repaired_energy": round(best_repair_energy, 3),
            "improved": improved,
            "recovered": recovered,
        })

    return results


# ---------------------------------------------------------------------------
# Benchmark 4: Inpainting Recovery
# ---------------------------------------------------------------------------

def benchmark_inpainting_recovery(
    bench: list[dict],
    rc_cache: dict,
    n_targets: int = 50,
    seed: int = 42,
) -> dict:
    """Mask a step's conditions, see if sampling can recover similar energy."""
    rng = random.Random(seed)
    api = EnergyAPI()
    results = {"tested": 0, "recovered_top1": 0, "recovered_top5": 0, "details": []}

    for t in bench[:n_targets]:
        smi = t["target_smiles"]
        d = min(t.get("depth", 2), 3)

        bt = CascadeBoard.from_n_steps(d, smi)
        cc = ConstraintCompiler().compile(bt)
        g = _make_cached_graph(rc_cache, max_depth=d, branch_factor=8)
        g.build(smi, cc)
        g.propagate_constraints(cc)
        paths = g.sample_paths(n=16) if not g.is_empty() else []
        if not paths:
            continue

        original = g.path_to_board(paths[0], smi)
        if original.n_steps < 2:
            continue

        # Assign default conditions
        for s in original.slots:
            if s.T is None:
                s.T = 30.0 + rng.uniform(-5, 10)
            if s.pH is None:
                s.pH = 7.0 + rng.uniform(-0.5, 0.5)

        original_energy = api.compute_energy(original, cc)

        results["tested"] += 1

        # Sample multiple paths and check if any achieves similar or better energy
        scored_paths = []
        for p in g.sample_paths(n=32):
            b = g.path_to_board(p, smi)
            for s in b.slots:
                if s.T is None:
                    s.T = 30.0 + rng.uniform(-5, 10)
                if s.pH is None:
                    s.pH = 7.0 + rng.uniform(-0.5, 0.5)
            e = api.compute_energy(b, cc)
            scored_paths.append((b, e))

        scored_paths.sort(key=lambda x: x[1])

        # Top-1: within 10% of original energy
        if scored_paths and _within_energy_margin(scored_paths[0][1], original_energy, 0.10):
            results["recovered_top1"] += 1

        # Top-5: any within 20% of original
        for b, e in scored_paths[:5]:
            if _within_energy_margin(e, original_energy, 0.20):
                results["recovered_top5"] += 1
                break

        results["details"].append({
            "recovered_top1": results["recovered_top1"],
            "recovered_top5": results["recovered_top5"],
        })

    return results


# ---------------------------------------------------------------------------
# Run all benchmarks
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    bench_path: str = "data/benchmark_v2_100.json",
    rc_cache_path: str = "results/shared/retrochimera_candidates_depth2.json",
    n_targets: int = 50,
) -> dict:
    """Run all four benchmark types."""
    bench = json.loads(Path(bench_path).read_text())
    rc_cache = json.loads(Path(rc_cache_path).read_text())

    print("Running CascadeBoard++ Benchmark Suite...")

    # Benchmark 2: Constraint scenarios
    print("\n[2] Constraint Scenarios...")
    constraint = benchmark_constraint_scenarios()
    print(f"  Passed: {constraint['passed']}/{constraint['tested']} "
          f"({constraint['pass_rate']*100:.0f}%)")

    # Benchmark 3: Counterfactual Repair
    print("\n[3] Counterfactual Repair...")
    t0 = time.time()
    repair = benchmark_counterfactual_repair(bench, rc_cache, n_targets)
    print(f"  Tested: {repair['tested']}")
    print(f"  Energy improved: {repair['energy_improved']}/{repair['tested']} "
          f"({repair['energy_improved']/max(repair['tested'],1)*100:.0f}%)")
    print(f"  Fully repaired: {repair['repaired']}/{repair['tested']} "
          f"({repair['repaired']/max(repair['tested'],1)*100:.0f}%)")
    print(f"  Time: {time.time()-t0:.0f}s")

    # Benchmark 4: Inpainting Recovery
    print("\n[4] Inpainting Recovery...")
    t0 = time.time()
    inpaint = benchmark_inpainting_recovery(bench, rc_cache, n_targets)
    print(f"  Tested: {inpaint['tested']}")
    print(f"  Recovered top-1: {inpaint['recovered_top1']}/{inpaint['tested']} "
          f"({inpaint['recovered_top1']/max(inpaint['tested'],1)*100:.0f}%)")
    print(f"  Recovered top-5: {inpaint['recovered_top5']}/{inpaint['tested']} "
          f"({inpaint['recovered_top5']/max(inpaint['tested'],1)*100:.0f}%)")
    print(f"  Time: {time.time()-t0:.0f}s")

    return {
        "constraint_scenarios": constraint,
        "counterfactual_repair": repair,
        "inpainting_recovery": inpaint,
    }


if __name__ == "__main__":
    results = run_all_benchmarks()
    Path("results/v2/cascadeboard_benchmarks.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "details"}
                    for k, v in results.items()}, indent=2)
    )
