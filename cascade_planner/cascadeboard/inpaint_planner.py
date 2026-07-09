"""BERT-style masked inpainting planner using v20 CascadeBoardTransformer.

Uses the existing v20 checkpoint's type/EC/cond prediction heads for
iterative masked completion. User-fixed slots are unmasked; unknown
slots are predicted iteratively (unmask most confident first).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cascade_planner.cascadeboard import CascadeBoard, Slot, RouteResult, RouteExplanation
from cascade_planner.cascadeboard.route_encoder import (
    CascadeBoardTransformer,
    board_to_tensors,
    smiles_to_morgan_fp,
    REACTION_TYPE_TO_ID,
    OBJECTIVE_TO_ID,
    NUM_EC1_CLASSES,
    COMPAT_VOCAB,
    OPMODE_VOCAB,
    ISSUE_TYPE_VOCAB,
    PAIRWISE_VOCAB,
)
from cascade_planner.cascadeboard.planner import load_cascadeboard_model

logger = logging.getLogger(__name__)

ID_TO_TYPE = {v: k for k, v in REACTION_TYPE_TO_ID.items() if k}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_type(logits: torch.Tensor, temperature: float = 1.0) -> tuple[int, float]:
    """Decode type logits to (type_id, confidence). logits shape: (NUM_TYPES,)."""
    if temperature <= 0:
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        idx = torch.multinomial(probs, 1).item()
    confidence = F.softmax(logits, dim=-1)[idx].item()
    return idx, confidence


def _decode_ec1(logits: torch.Tensor, temperature: float = 1.0) -> tuple[int, float]:
    """Decode EC1 logits to (ec1_id, confidence). logits shape: (NUM_EC1,)."""
    if temperature <= 0:
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        idx = torch.multinomial(probs, 1).item()
    confidence = F.softmax(logits, dim=-1)[idx].item()
    return idx, confidence


def _decode_conditions(cond_pred: torch.Tensor) -> tuple[float, float]:
    """Denormalize condition predictions to (T, pH)."""
    T_norm = cond_pred[0].item()
    pH_norm = cond_pred[1].item()
    T = max(0.0, min(120.0, 37.0 + 30.0 * T_norm))
    pH = max(1.0, min(14.0, 7.0 + 3.0 * pH_norm))
    return T, pH


# ---------------------------------------------------------------------------
# Core inpainting function
# ---------------------------------------------------------------------------

def inpaint_skeleton(
    model: CascadeBoardTransformer,
    target: str,
    n_steps: int,
    fixed_slots: dict[int, dict] | None = None,
    objective: str = "balanced",
    device: str = "cpu",
    n_iterations: int = 5,
    temperature: float = 1.0,
) -> CascadeBoard:
    """Iterative masked inpainting using the CascadeBoardTransformer.

    Args:
        model: Loaded CascadeBoardTransformer (eval mode).
        target: Target molecule SMILES.
        n_steps: Number of synthesis steps (slots).
        fixed_slots: Dict mapping slot index to fixed field values,
            e.g. {2: {"ec": "1.1.1.1", "T": 37}}.
        objective: Planning objective (balanced/industrial/green/novelty).
        device: Torch device string.
        n_iterations: Max iterations for iterative unmasking.
        temperature: Sampling temperature (0 = greedy argmax).

    Returns:
        Completed CascadeBoard with predicted fields filled in.
    """
    model.eval()
    fixed_slots = fixed_slots or {}

    # Build initial board with target and fixed slots
    board = CascadeBoard.from_n_steps(n_steps, target)

    # Apply fixed slot values
    for slot_idx, values in fixed_slots.items():
        if slot_idx >= n_steps:
            continue
        slot = board.slots[slot_idx]
        for key, val in values.items():
            if key == "ec":
                slot.ec = val
                slot.fixed_fields.add("ec")
            elif key == "T":
                slot.T = val
                slot.fixed_fields.add("T")
            elif key == "pH":
                slot.pH = val
                slot.fixed_fields.add("pH")
            elif key == "reaction_type":
                slot.reaction_type = val
                slot.fixed_fields.add("reaction_type")
            elif key in Slot.ALL_FIELDS:
                setattr(slot, key, val)
                slot.fixed_fields.add(key)

    # Track which slots still need prediction
    def _slot_needs_prediction(slot: Slot) -> bool:
        has_type = slot.reaction_type is not None and "reaction_type" in slot.fixed_fields
        has_ec = slot.ec is not None and "ec" in slot.fixed_fields
        has_T = slot.T is not None and "T" in slot.fixed_fields
        has_pH = slot.pH is not None and "pH" in slot.fixed_fields
        return not (has_type and has_ec and has_T and has_pH)

    # Iterative unmasking: predict all masked slots, unmask the most confident one
    for iteration in range(n_iterations):
        # Find slots that still need prediction
        masked_indices = [i for i in range(n_steps) if _slot_needs_prediction(board.slots[i])]
        if not masked_indices:
            break

        # Forward pass
        tensors = board_to_tensors(board, objective=objective, device=device)
        with torch.no_grad():
            out = model(**tensors)

        # For each masked slot, compute confidence and predicted values
        best_idx = -1
        best_confidence = -1.0
        best_predictions: dict[str, Any] = {}

        for slot_idx in masked_indices:
            slot = board.slots[slot_idx]

            # Decode predictions for this slot
            type_logits = out["type_logits"][0, slot_idx]
            ec1_logits = out["ec1_logits"][0, slot_idx]
            cond_pred = out["cond_preds"][0, slot_idx]

            type_id, type_conf = _decode_type(type_logits, temperature)
            ec1_id, ec1_conf = _decode_ec1(ec1_logits, temperature)
            T, pH = _decode_conditions(cond_pred)

            # Overall confidence for this slot = average of classification confidences
            # Only count fields that are not already fixed
            confs = []
            preds: dict[str, Any] = {}
            if "reaction_type" not in slot.fixed_fields:
                confs.append(type_conf)
                preds["reaction_type"] = ID_TO_TYPE.get(type_id, "")
            if "ec" not in slot.fixed_fields:
                confs.append(ec1_conf)
                preds["ec1_id"] = ec1_id
            if "T" not in slot.fixed_fields:
                preds["T"] = T
            if "pH" not in slot.fixed_fields:
                preds["pH"] = pH

            slot_confidence = sum(confs) / len(confs) if confs else 1.0

            if slot_confidence > best_confidence:
                best_confidence = slot_confidence
                best_idx = slot_idx
                best_predictions = preds

        if best_idx < 0:
            break

        # Unmask the most confident slot
        slot = board.slots[best_idx]
        if "reaction_type" in best_predictions and "reaction_type" not in slot.fixed_fields:
            slot.reaction_type = best_predictions["reaction_type"]
            slot.fixed_fields.add("reaction_type")
        if "ec1_id" in best_predictions and "ec" not in slot.fixed_fields:
            ec1_id = best_predictions["ec1_id"]
            if ec1_id > 0:
                slot.ec = f"{ec1_id}.x"
            else:
                slot.ec = None
            slot.fixed_fields.add("ec")
        if "T" in best_predictions and "T" not in slot.fixed_fields:
            slot.T = best_predictions["T"]
            slot.fixed_fields.add("T")
        if "pH" in best_predictions and "pH" not in slot.fixed_fields:
            slot.pH = best_predictions["pH"]
            slot.fixed_fields.add("pH")

        slot.source = "inpainted"
        slot.confidence = best_confidence

    # Final pass: fill any remaining slots that weren't reached by iteration limit
    remaining = [i for i in range(n_steps) if _slot_needs_prediction(board.slots[i])]
    if remaining:
        tensors = board_to_tensors(board, objective=objective, device=device)
        with torch.no_grad():
            out = model(**tensors)

        for slot_idx in remaining:
            slot = board.slots[slot_idx]
            type_logits = out["type_logits"][0, slot_idx]
            ec1_logits = out["ec1_logits"][0, slot_idx]
            cond_pred = out["cond_preds"][0, slot_idx]

            if "reaction_type" not in slot.fixed_fields:
                type_id, _ = _decode_type(type_logits, temperature)
                slot.reaction_type = ID_TO_TYPE.get(type_id, "")
            if "ec" not in slot.fixed_fields:
                ec1_id, _ = _decode_ec1(ec1_logits, temperature)
                slot.ec = f"{ec1_id}.x" if ec1_id > 0 else None
            if "T" not in slot.fixed_fields:
                T, _ = _decode_conditions(cond_pred)
                slot.T = T
            if "pH" not in slot.fixed_fields:
                _, pH = _decode_conditions(cond_pred)
                slot.pH = pH
            slot.source = "inpainted"

    return board


# ---------------------------------------------------------------------------
# High-level planning function
# ---------------------------------------------------------------------------

def plan_with_inpainting(
    target: str,
    n_steps: int = 3,
    fixed_slots: dict[int, dict] | None = None,
    objective: str = "balanced",
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    n_results: int = 5,
    n_iterations: int = 5,
) -> list[RouteResult]:
    """High-level inpainting planner: load model, generate multiple skeletons.

    Generates diverse routes by using temperature sampling across multiple
    runs, then returns them sorted by model feasibility score.

    Args:
        target: Target molecule SMILES.
        n_steps: Number of synthesis steps.
        fixed_slots: User-fixed slot constraints.
        objective: Planning objective.
        checkpoint_path: Path to v20 checkpoint. Defaults to best.pt in v20 dir.
        device: Torch device.
        n_results: Number of diverse routes to generate.
        n_iterations: Iterations per route for iterative unmasking.

    Returns:
        List of RouteResult sorted by score (best first).
    """
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path("results/shared/cascadeboard_model_v20/best.pt")
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    model = load_cascadeboard_model(checkpoint_path, device=device)

    # Generate multiple routes with varying temperature
    results: list[RouteResult] = []
    temperatures = _sample_temperatures(n_results)

    for i, temp in enumerate(temperatures):
        board = inpaint_skeleton(
            model=model,
            target=target,
            n_steps=n_steps,
            fixed_slots=fixed_slots,
            objective=objective,
            device=device,
            n_iterations=n_iterations,
            temperature=temp,
        )

        # Score using model's feasibility head
        tensors = board_to_tensors(board, objective=objective, device=device)
        with torch.no_grad():
            out = model(**tensors)

        feasibility = out["feasibility"].item()
        energy = out["energy"].item()

        # Decode global diagnostics
        compat = ""
        if "compat_logits" in out:
            compat = COMPAT_VOCAB[out["compat_logits"][0].argmax().item()]
        opmode = ""
        if "opmode_logits" in out:
            opmode = OPMODE_VOCAB[out["opmode_logits"][0].argmax().item()]
        issues = []
        if "issue_type_logits" in out:
            issue_probs = torch.sigmoid(out["issue_type_logits"][0])
            issues = [ISSUE_TYPE_VOCAB[j] for j, p in enumerate(issue_probs) if p > 0.3]
        pairwise = []
        if out.get("pairwise_logits") is not None and n_steps >= 2:
            pw = out["pairwise_logits"][0]
            for j in range(min(pw.shape[0], n_steps - 1)):
                pairwise.append(PAIRWISE_VOCAB[pw[j].argmax().item()])

        # Build explanation
        types_str = " -> ".join(
            s.reaction_type or "?" for s in board.slots
        )
        explanation = RouteExplanation(
            why_selected=f"Inpainted {n_steps}-step route (T={temp:.2f}): {types_str}",
            constraints_satisfied={},
            global_condition_window=_format_condition_window(board),
            uncertainty_table={
                "predicted_compatibility": compat,
                "predicted_operation_mode": opmode,
                "predicted_issues": issues,
                "predicted_pairwise": pairwise,
            },
        )

        results.append(RouteResult(
            board=board,
            score=feasibility,
            confidence=feasibility,
            explanation=explanation,
        ))

    # Deduplicate by reaction type sequence, keep highest scoring
    results = _deduplicate_results(results)

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:n_results]


def _format_condition_window(board: CascadeBoard) -> str:
    """Format the T/pH range across all slots."""
    Ts = [s.T for s in board.slots if s.T is not None]
    pHs = [s.pH for s in board.slots if s.pH is not None]
    if not Ts and not pHs:
        return ""
    parts = []
    if Ts:
        parts.append(f"T: {min(Ts):.0f}-{max(Ts):.0f} C")
    if pHs:
        parts.append(f"pH: {min(pHs):.1f}-{max(pHs):.1f}")
    return ", ".join(parts)


def _sample_temperatures(n: int) -> list[float]:
    """Generate a spread of temperatures for diverse sampling."""
    if n == 1:
        return [0.0]  # greedy
    # First result is greedy, rest use increasing temperature
    temps = [0.0]
    for i in range(1, n):
        temps.append(0.3 + 0.4 * i / (n - 1))  # range [0.3, 0.7]
    return temps


def _deduplicate_results(results: list[RouteResult]) -> list[RouteResult]:
    """Remove duplicate routes (same type+EC sequence), keep best score."""
    seen: dict[str, RouteResult] = {}
    for r in results:
        key = "|".join(
            f"{s.reaction_type or '?'}:{s.ec or '-'}"
            for s in r.board.slots
        )
        if key not in seen or r.score > seen[key].score:
            seen[key] = r
    return list(seen.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BERT-style masked inpainting planner for CascadeBoard"
    )
    parser.add_argument("--target", required=True, help="Target SMILES")
    parser.add_argument("--n-steps", type=int, default=3, help="Number of synthesis steps")
    parser.add_argument("--objective", default="balanced", choices=list(OBJECTIVE_TO_ID.keys()))
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    parser.add_argument("--n-results", type=int, default=5, help="Number of routes to generate")
    parser.add_argument("--n-iterations", type=int, default=5, help="Unmasking iterations per route")
    parser.add_argument(
        "--fixed", default=None,
        help='JSON string of fixed slots, e.g. \'{"0": {"ec": "1.1.1.1", "T": 37}}\'',
    )
    args = parser.parse_args()

    # Parse fixed slots
    fixed_slots = None
    if args.fixed:
        import json
        raw = json.loads(args.fixed)
        fixed_slots = {int(k): v for k, v in raw.items()}

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = plan_with_inpainting(
        target=args.target,
        n_steps=args.n_steps,
        fixed_slots=fixed_slots,
        objective=args.objective,
        checkpoint_path=args.checkpoint,
        device=args.device,
        n_results=args.n_results,
        n_iterations=args.n_iterations,
    )

    print(f"\n{'='*60}")
    print(f"Inpainting results for: {args.target}")
    print(f"Objective: {args.objective}, Steps: {args.n_steps}")
    print(f"{'='*60}\n")

    for i, r in enumerate(results):
        print(f"--- Route {i+1} (score={r.score:.3f}) ---")
        print(r.board.summary())
        if r.explanation.global_condition_window:
            print(f"  Conditions: {r.explanation.global_condition_window}")
        diag = r.explanation.uncertainty_table
        if diag.get("predicted_compatibility"):
            print(f"  Compatibility: {diag['predicted_compatibility']}")
        if diag.get("predicted_operation_mode"):
            print(f"  Operation mode: {diag['predicted_operation_mode']}")
        if diag.get("predicted_issues"):
            print(f"  Issues: {diag['predicted_issues']}")
        if diag.get("predicted_pairwise"):
            print(f"  Pairwise: {diag['predicted_pairwise']}")
        print()


if __name__ == "__main__":
    main()
