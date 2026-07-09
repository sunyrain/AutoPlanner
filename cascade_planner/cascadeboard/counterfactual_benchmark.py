"""Counterfactual Repair Benchmark for CascadeBoard++.

COMPLETE spec §增强7: Input corrupted routes with known corruption types,
measure problem detection accuracy, repair success rate, minimal edit distance.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cascade_planner.cascadeboard import CascadeBoard, EditType
from cascade_planner.cascadeboard.benchmarks import _make_cached_graph
from cascade_planner.cascadeboard.constraint_compiler import ConstraintCompiler
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.planner import (
    apply_edit, propose_edits, propose_neural_edits,
    load_cascadeboard_model,
)
from cascade_planner.cascadeboard.route_encoder import (
    board_to_tensors, REACTION_TYPE_TO_ID,
)
from cascade_planner.cascadeboard.training_data import (
    load_cascade_routes, route_to_board,
    corrupt_replace_type, corrupt_shift_T, corrupt_shift_pH,
    corrupt_replace_ec, corrupt_swap_order,
)


# ---------------------------------------------------------------------------
# Corruption → expected repair mapping
# ---------------------------------------------------------------------------

CORRUPTION_TO_REPAIR = {
    "replace_type": EditType.REPLACE_STEP,
    "shift_T": EditType.ADJUST_CONDITION,
    "shift_pH": EditType.ADJUST_CONDITION,
    "replace_ec": EditType.REPLACE_ENZYME,
    "swap_order": EditType.SWAP_ORDER,
}

CORRUPTION_FNS = {
    "replace_type": corrupt_replace_type,
    "shift_T": corrupt_shift_T,
    "shift_pH": corrupt_shift_pH,
    "replace_ec": corrupt_replace_ec,
    "swap_order": corrupt_swap_order,
}


# ---------------------------------------------------------------------------
# Detection: does the model identify the corrupted slot?
# ---------------------------------------------------------------------------

def _detect_issue(
    board: CascadeBoard,
    model,
    device: str = "cpu",
) -> tuple[int | None, str | None]:
    """Use model issue_scores and edit heads to detect the problem slot and type."""
    if model is None:
        return None, None
    model.eval()
    tensors = board_to_tensors(board, device=device)
    with torch.no_grad():
        out = model(**tensors)
    n = board.n_steps
    issue_scores = out["issue_scores"][0, :n].cpu().numpy()
    pred_slot = int(np.argmax(issue_scores))
    edit_logits = out["edit_type_logits"][0].cpu().numpy()
    pred_edit_idx = int(np.argmax(edit_logits))
    pred_edit = list(EditType)[pred_edit_idx].name
    return pred_slot, pred_edit


# ---------------------------------------------------------------------------
# Repair: apply edits and check if the route is restored
# ---------------------------------------------------------------------------

def _repair_board(
    corrupted: CascadeBoard,
    original: CascadeBoard,
    api: EnergyAPI,
    compiled,
    model=None,
    device: str = "cpu",
    max_edits: int = 8,
) -> dict[str, Any]:
    """Try to repair a corrupted board back toward the original."""
    if model is not None:
        edits = propose_neural_edits(corrupted, model, api, compiled, device=device, m=max_edits)
    else:
        edits = propose_edits(corrupted, api, compiled, m=max_edits)

    original_energy = api.compute_energy(original, compiled)
    corrupted_energy = api.compute_energy(corrupted, compiled)
    best_energy = corrupted_energy
    best_board = corrupted
    best_source = None
    n_applied = 0

    for edit in edits:
        repaired = apply_edit(corrupted, edit)
        if not compiled.hard_satisfied(repaired):
            continue
        energy = api.compute_energy(repaired, compiled)
        n_applied += 1
        if energy < best_energy:
            best_energy = energy
            best_board = repaired
            best_source = edit.metadata.get("source", "unknown")

    improved = best_energy < corrupted_energy
    margin = 0.10
    repaired_to_original = abs(best_energy - original_energy) <= margin * max(abs(original_energy), 1.0)

    type_match = _type_sequence_match(best_board, original)
    condition_mae = _condition_mae(best_board, original)

    return {
        "original_energy": round(original_energy, 4),
        "corrupted_energy": round(corrupted_energy, 4),
        "best_energy": round(best_energy, 4),
        "improved": bool(improved),
        "repaired": bool(repaired_to_original),
        "type_match": bool(type_match),
        "condition_mae_T": round(condition_mae[0], 2),
        "condition_mae_pH": round(condition_mae[1], 2),
        "best_source": best_source,
        "n_edits_tried": len(edits),
        "n_edits_applied": n_applied,
    }


def _type_sequence_match(a: CascadeBoard, b: CascadeBoard) -> bool:
    if a.n_steps != b.n_steps:
        return False
    return all(
        a.slots[i].reaction_type == b.slots[i].reaction_type
        for i in range(a.n_steps)
    )


def _condition_mae(a: CascadeBoard, b: CascadeBoard) -> tuple[float, float]:
    t_diffs, ph_diffs = [], []
    n = min(a.n_steps, b.n_steps)
    for i in range(n):
        if a.slots[i].T is not None and b.slots[i].T is not None:
            t_diffs.append(abs(a.slots[i].T - b.slots[i].T))
        if a.slots[i].pH is not None and b.slots[i].pH is not None:
            ph_diffs.append(abs(a.slots[i].pH - b.slots[i].pH))
    return (
        float(np.mean(t_diffs)) if t_diffs else 0.0,
        float(np.mean(ph_diffs)) if ph_diffs else 0.0,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_counterfactual_benchmark(
    *,
    data_path: str,
    rc_cache_path: str,
    output: str,
    checkpoint: str | None = None,
    n_routes: int = 100,
    corruptions_per_route: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    routes = load_cascade_routes(data_path)
    rng = random.Random(seed)
    rng.shuffle(routes)
    routes = routes[:n_routes]

    rc_cache = json.loads(Path(rc_cache_path).read_text())
    api = EnergyAPI()
    model = load_cascadeboard_model(checkpoint, device=device) if checkpoint else None
    compiler = ConstraintCompiler()

    corruption_names = list(CORRUPTION_FNS.keys())
    rows: list[dict] = []

    for route in routes:
        board = route_to_board(route)
        if board.n_steps < 2:
            continue

        compiled = compiler.compile(board)

        for _ in range(corruptions_per_route):
            ctype = rng.choice(corruption_names)
            fn = CORRUPTION_FNS[ctype]
            sample = fn(board, rng)

            corrupted = route_to_board({
                "doi": route.get("doi", ""),
                "target": sample.target_smiles,
                "steps": [
                    {
                        "product": s.get("product", ""),
                        "reactant": s.get("main_reactant", ""),
                        "rxn_smiles": s.get("rxn_smiles", ""),
                        "reaction_type": s.get("reaction_type", ""),
                        "ec": s.get("ec"),
                        "enzyme_uid": s.get("enzyme_uid"),
                        "T": s.get("T"),
                        "pH": s.get("pH"),
                        "solvent": s.get("solvent"),
                    }
                    for s in sample.slots
                ],
            })

            # Detection
            pred_slot, pred_edit = _detect_issue(corrupted, model, device)
            gt_slot = sample.edit_slot
            gt_edit = sample.edit_action
            expected_repair = CORRUPTION_TO_REPAIR.get(ctype)

            slot_detected = pred_slot == gt_slot if (pred_slot is not None and gt_slot is not None) else False
            edit_detected = (
                pred_edit == gt_edit
                if (pred_edit is not None and gt_edit is not None)
                else False
            )

            # Repair
            repair_result = _repair_board(
                corrupted, board, api, compiled, model, device,
            )

            row = {
                "corruption": ctype,
                "gt_slot": gt_slot,
                "gt_edit": gt_edit,
                "pred_slot": pred_slot,
                "pred_edit": pred_edit,
                "slot_detected": bool(slot_detected),
                "edit_detected": bool(edit_detected),
                "domain": route.get("route_domain", ""),
                **repair_result,
            }
            rows.append(row)

    # Aggregate
    n = len(rows)

    def _rate(key):
        return sum(r[key] for r in rows) / max(n, 1)

    by_corruption: dict[str, dict] = {}
    for ctype in sorted({r["corruption"] for r in rows}):
        sub = [r for r in rows if r["corruption"] == ctype]
        ns = len(sub)
        by_corruption[ctype] = {
            "n": ns,
            "slot_detection_acc": sum(r["slot_detected"] for r in sub) / max(ns, 1),
            "edit_detection_acc": sum(r["edit_detected"] for r in sub) / max(ns, 1),
            "repair_rate": sum(r["repaired"] for r in sub) / max(ns, 1),
            "improved_rate": sum(r["improved"] for r in sub) / max(ns, 1),
            "mean_T_mae": float(np.mean([r["condition_mae_T"] for r in sub])),
            "mean_pH_mae": float(np.mean([r["condition_mae_pH"] for r in sub])),
        }

    report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "data_path": data_path,
            "checkpoint": checkpoint,
            "n_routes": n_routes,
            "corruptions_per_route": corruptions_per_route,
            "seed": seed,
        },
        "overall": {
            "n_scenarios": n,
            "slot_detection_acc": _rate("slot_detected"),
            "edit_detection_acc": _rate("edit_detected"),
            "repair_rate": _rate("repaired"),
            "improved_rate": _rate("improved"),
            "type_match_rate": _rate("type_match"),
            "mean_T_mae": float(np.mean([r["condition_mae_T"] for r in rows])) if rows else 0,
            "mean_pH_mae": float(np.mean([r["condition_mae_pH"] for r in rows])) if rows else 0,
        },
        "by_corruption": by_corruption,
        "rows": rows,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="CascadeBoard counterfactual repair benchmark")
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--rc-cache", default="results/shared/retrochimera_candidates_depth2.json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default="results/v2/cascadeboard_counterfactual_benchmark.json")
    ap.add_argument("--n-routes", type=int, default=100)
    ap.add_argument("--corruptions-per-route", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    report = run_counterfactual_benchmark(
        data_path=args.data,
        rc_cache_path=args.rc_cache,
        output=args.output,
        checkpoint=args.checkpoint,
        n_routes=args.n_routes,
        corruptions_per_route=args.corruptions_per_route,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({
        "overall": report["overall"],
        "by_corruption": report["by_corruption"],
    }, indent=2))


if __name__ == "__main__":
    main()
