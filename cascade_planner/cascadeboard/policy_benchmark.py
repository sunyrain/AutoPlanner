"""Compare rule-only vs neural edit policy on controlled repair tasks."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.benchmarks import _make_cached_graph, _within_energy_margin
from cascade_planner.cascadeboard import CascadeBoard
from cascade_planner.cascadeboard.constraint_compiler import ConstraintCompiler
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.planner import (
    apply_edit, load_cascadeboard_model, propose_edits, propose_neural_edits,
)


def _repair_once(item, rc_cache, api, rng, policy: str, model=None, device="cpu") -> dict[str, Any] | None:
    smi = item["target_smiles"]
    depth = min(int(item.get("depth") or 2), 3)
    template = CascadeBoard.from_n_steps(depth, smi)
    compiled = ConstraintCompiler().compile(template)
    graph = _make_cached_graph(rc_cache, max_depth=depth, branch_factor=8)
    graph.build(smi, compiled)
    graph.propagate_constraints(compiled)
    paths = graph.sample_paths(n=16) if not graph.is_empty() else []
    if not paths:
        return None

    original = graph.path_to_board(paths[0], smi)
    for slot in original.slots:
        if slot.T is None:
            slot.T = 30.0 + rng.uniform(-5, 10)
        if slot.pH is None:
            slot.pH = 7.0 + rng.uniform(-0.5, 0.5)
    original_energy = api.compute_energy(original, compiled)

    corrupted = original.copy()
    slot_idx = rng.randint(0, corrupted.n_steps - 1)
    corrupted.slots[slot_idx].T = 95.0
    corrupted_energy = api.compute_energy(corrupted, compiled)

    if policy == "neural":
        edits = propose_neural_edits(corrupted, model, api, compiled, device=device, m=8)
    else:
        edits = propose_edits(corrupted, api, compiled, m=8)
        for edit in edits:
            edit.metadata["source"] = "rule"

    best_energy = corrupted_energy
    best_source = None
    for edit in edits:
        repaired = apply_edit(corrupted, edit)
        if not compiled.hard_satisfied(repaired):
            continue
        energy = api.compute_energy(repaired, compiled)
        if energy < best_energy:
            best_energy = energy
            best_source = edit.metadata.get("source") or policy

    return {
        "original_energy": round(original_energy, 4),
        "corrupted_energy": round(corrupted_energy, 4),
        "best_energy": round(best_energy, 4),
        "improved": bool(best_energy < corrupted_energy),
        "repaired": bool(_within_energy_margin(best_energy, original_energy, 0.10)),
        "best_source": best_source,
        "n_edits": int(len(edits)),
    }


def run_policy_benchmark(
    *,
    bench_path: str,
    rc_cache_path: str,
    output: str,
    checkpoint: str | None,
    n_targets: int = 50,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    bench = json.loads(Path(bench_path).read_text())[:n_targets]
    rc_cache = json.loads(Path(rc_cache_path).read_text())
    api = EnergyAPI()
    rng = random.Random(seed)
    model = load_cascadeboard_model(checkpoint, device=device) if checkpoint else None

    rows = []
    for item in bench:
        for policy in ("rule", "neural"):
            if policy == "neural" and model is None:
                continue
            result = _repair_once(item, rc_cache, api, rng, policy, model, device)
            if result is None:
                continue
            result.update({
                "target": item["target_smiles"],
                "domain": item.get("route_domain", ""),
                "policy": policy,
            })
            rows.append(result)

    def summarize(policy: str) -> dict[str, Any]:
        sub = [r for r in rows if r["policy"] == policy]
        n = len(sub)
        return {
            "n": n,
            "improved_rate": sum(r["improved"] for r in sub) / max(n, 1),
            "repaired_rate": sum(r["repaired"] for r in sub) / max(n, 1),
            "mean_n_edits": sum(r["n_edits"] for r in sub) / max(n, 1),
            "best_sources": {
                k: sum(1 for r in sub if str(r.get("best_source")) == k)
                for k in sorted({str(r.get("best_source")) for r in sub})
            },
        }

    result = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "bench_path": bench_path,
            "rc_cache_path": rc_cache_path,
            "checkpoint": checkpoint,
            "n_targets": n_targets,
            "seed": seed,
            "device": device,
        },
        "summary": {
            "rule": summarize("rule"),
            "neural": summarize("neural"),
        },
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
    ap.add_argument("--checkpoint", default="results/shared/cascadeboard_model_v7/best.pt")
    ap.add_argument("--output", default="results/v2/cascadeboard_policy_benchmark.json")
    ap.add_argument("--n-targets", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    result = run_policy_benchmark(
        bench_path=args.bench,
        rc_cache_path=args.rc_cache,
        output=args.output,
        checkpoint=args.checkpoint,
        n_targets=args.n_targets,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
