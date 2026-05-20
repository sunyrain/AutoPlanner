"""CascadeBoard++ Planner: complete inference pipeline.

Layer 6: Particle refinement + full planning pipeline.
Integrates all layers: constraint compiler → candidate graph →
energy API → route encoder → edit policy → particle sampling.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cascade_planner.cascadeboard import (
    CascadeBoard, Slot, EditType, EditAction,
    CompiledConstraints, RouteResult, RouteExplanation,
)
from cascade_planner.cascadeboard.constraint_compiler import (
    ConstraintCompiler, EC1_T_RANGES, EC1_PH_RANGES,
)
from cascade_planner.cascadeboard.candidate_graph import CandidateHypergraph
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.route_encoder import (
    CascadeBoardTransformer, board_to_tensors, REACTION_TYPE_TO_ID,
)

# Lazy-loaded real candidate cache for production use
_REAL_CACHE: dict | None = None
_ZINC_CHECKER = None


def _load_real_cache() -> dict:
    """Load and merge RetroChimera + EnzExpand candidate caches."""
    global _REAL_CACHE
    if _REAL_CACHE is not None:
        return _REAL_CACHE
    import json
    from pathlib import Path
    from cascade_planner.cascadeboard.candidate_cache import merge_candidate_caches, canon_smiles

    merged = {}
    for cache_path in [
        "results/shared/retrochimera_candidates_depth2.json",
        "results/shared/enzexpand_dualtower_candidates_100.json",
        "results/shared/enzexpand_v3_expanded_cache.json",
    ]:
        p = Path(cache_path)
        if p.exists():
            data = json.loads(p.read_text())
            for smi, cands in data.items():
                key = canon_smiles(smi) or smi
                for c in cands:
                    # Ensure reaction_smiles is populated
                    if not c.get("reaction_smiles") and c.get("main_reactant") and c.get("product"):
                        reactants = c["main_reactant"]
                        if c.get("aux_reactants"):
                            reactants = ".".join([reactants] + c["aux_reactants"])
                        c["reaction_smiles"] = f"{reactants}>>{c['product']}"
                merged.setdefault(key, []).extend(cands)
    _REAL_CACHE = merged
    return _REAL_CACHE


def _get_zinc_checker():
    """Get ZINC stock checker, lazy-loaded."""
    global _ZINC_CHECKER
    if _ZINC_CHECKER is not None:
        return _ZINC_CHECKER
    try:
        from cascade_planner.cascadeboard.zinc_stock import is_in_zinc_stock
        _ZINC_CHECKER = is_in_zinc_stock
    except Exception:
        _ZINC_CHECKER = lambda s: False
    return _ZINC_CHECKER


# ---------------------------------------------------------------------------
# Scored particle
# ---------------------------------------------------------------------------

@dataclass
class ScoredBoard:
    board: CascadeBoard
    energy: float = float("inf")
    quality: dict[str, float] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)
    posterior: float = float("-inf")


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------

def apply_edit(board: CascadeBoard, edit: EditAction) -> CascadeBoard:
    """Apply an edit action to a board, returning a new board."""
    new = board.copy()

    if edit.edit_type == EditType.FILL_FIELD:
        if edit.field and edit.field not in new.slots[edit.slot_index].fixed_fields:
            setattr(new.slots[edit.slot_index], edit.field, edit.new_value)

    elif edit.edit_type == EditType.REPLACE_STEP:
        slot = new.slots[edit.slot_index]
        if "reaction_smiles" not in slot.fixed_fields:
            cand = edit.new_value  # should be a candidate dict
            if isinstance(cand, dict):
                for k in ("reaction_smiles", "reaction_type", "ec", "main_reactant"):
                    if k not in slot.fixed_fields and k in cand:
                        setattr(slot, k, cand[k])
                slot.e_retro = cand.get("score", slot.e_retro)
                slot.source = cand.get("source", "edit")

    elif edit.edit_type == EditType.REPLACE_ENZYME:
        slot = new.slots[edit.slot_index]
        if "enzyme_uid" not in slot.fixed_fields:
            slot.enzyme_uid = edit.new_value

    elif edit.edit_type == EditType.ADJUST_CONDITION:
        slot = new.slots[edit.slot_index]
        if isinstance(edit.new_value, dict):
            if "T" in edit.new_value and "T" not in slot.fixed_fields:
                slot.T = edit.new_value["T"]
            if "pH" in edit.new_value and "pH" not in slot.fixed_fields:
                slot.pH = edit.new_value["pH"]

    elif edit.edit_type == EditType.INSERT_STEP:
        idx = edit.slot_index
        new_slot = Slot(index=idx)
        if isinstance(edit.new_value, dict):
            for k, v in edit.new_value.items():
                if hasattr(new_slot, k):
                    setattr(new_slot, k, v)
        new.slots.insert(idx, new_slot)
        # Re-index
        for i, s in enumerate(new.slots):
            s.index = i

    elif edit.edit_type == EditType.DELETE_STEP:
        idx = edit.slot_index
        if 0 <= idx < len(new.slots) and not new.slots[idx].is_fully_fixed():
            new.slots.pop(idx)
            for i, s in enumerate(new.slots):
                s.index = i

    elif edit.edit_type == EditType.SWAP_ORDER:
        idx = edit.slot_index
        if 0 <= idx < len(new.slots) - 1:
            a, b = new.slots[idx], new.slots[idx + 1]
            if not a.is_fully_fixed() and not b.is_fully_fixed():
                new.slots[idx], new.slots[idx + 1] = b, a
                new.slots[idx].index = idx
                new.slots[idx + 1].index = idx + 1

    new.edit_history.append(edit)
    return new


# ---------------------------------------------------------------------------
# Propose edits (rule-based MVP, upgradeable to neural edit policy)
# ---------------------------------------------------------------------------

def propose_edits(
    board: CascadeBoard,
    energy_api: EnergyAPI,
    compiled: CompiledConstraints | None = None,
    m: int = 8,
) -> list[EditAction]:
    """Propose m edit actions for a board. Rule-based MVP."""
    edits: list[EditAction] = []

    # 1. Find worst slot
    idx, reason = energy_api.diagnose_bottleneck(board)
    if idx is None:
        return edits

    slot = board.slots[idx]

    # 2. REPLACE_STEP: try other candidates
    if slot.candidates and "reaction_smiles" not in slot.fixed_fields:
        for cand in slot.candidates[:3]:
            edits.append(EditAction(
                EditType.REPLACE_STEP, slot_index=idx,
                new_value=cand, reason=f"replace worst step {idx}",
            ))

    # 3. ADJUST_CONDITION: try nearby conditions
    if slot.T is not None and "T" not in slot.fixed_fields:
        condition_targets: list[dict[str, float | None]] = []

        # Return extreme conditions directly to enzyme-compatible or mild defaults.
        if slot.ec:
            ec1 = slot.ec.split(".")[0]
            if ec1 in EC1_T_RANGES:
                lo, hi = EC1_T_RANGES[ec1]
                condition_targets.append({"T": (lo + hi) / 2, "pH": slot.pH})
            if ec1 in EC1_PH_RANGES:
                lo, hi = EC1_PH_RANGES[ec1]
                condition_targets.append({"T": slot.T, "pH": (lo + hi) / 2})

        condition_targets.append({"T": 37.0, "pH": 7.0})

        # Match adjacent condition windows when compatibility is the bottleneck.
        neighbors = []
        if idx > 0:
            neighbors.append(board.slots[idx - 1])
        if idx + 1 < board.n_steps:
            neighbors.append(board.slots[idx + 1])
        neighbor_T = [s.T for s in neighbors if s.T is not None]
        neighbor_pH = [s.pH for s in neighbors if s.pH is not None]
        if neighbor_T or neighbor_pH:
            condition_targets.append({
                "T": sum(neighbor_T) / len(neighbor_T) if neighbor_T else slot.T,
                "pH": sum(neighbor_pH) / len(neighbor_pH) if neighbor_pH else slot.pH,
            })

        for delta in [-10, -5, 5, 10]:
            condition_targets.append({"T": slot.T + delta, "pH": slot.pH})

        seen_conditions = set()
        for target in condition_targets:
            t_val = target.get("T")
            ph_val = target.get("pH")
            if t_val is not None:
                t_val = max(0.0, min(120.0, float(t_val)))
            if ph_val is not None:
                ph_val = max(1.0, min(14.0, float(ph_val)))
            key = (round(t_val, 2) if t_val is not None else None,
                   round(ph_val, 2) if ph_val is not None else None)
            if key in seen_conditions:
                continue
            seen_conditions.add(key)
            edits.append(EditAction(
                EditType.ADJUST_CONDITION, slot_index=idx,
                new_value={"T": t_val, "pH": ph_val},
                reason="adjust condition toward compatible window",
            ))

    # 4. SWAP_ORDER: try swapping with neighbor
    if idx > 0 and not board.slots[idx - 1].is_fully_fixed():
        edits.append(EditAction(
            EditType.SWAP_ORDER, slot_index=idx - 1,
            reason=f"swap steps {idx-1} and {idx}",
        ))

    return edits[:m]


def _condition_from_model(cond_pred: torch.Tensor) -> dict[str, float]:
    """Denormalize model condition head output to T/pH values."""
    t_norm = float(cond_pred[0].detach().cpu().item())
    ph_norm = float(cond_pred[1].detach().cpu().item())
    return {
        "T": max(0.0, min(120.0, 37.0 + 30.0 * t_norm)),
        "pH": max(1.0, min(14.0, 7.0 + 3.0 * ph_norm)),
    }


def _candidate_to_dict(cand: Any) -> dict[str, Any]:
    if isinstance(cand, dict):
        return cand
    out: dict[str, Any] = {}
    for key in ("reaction_smiles", "reaction_type", "ec", "main_reactant", "score", "source"):
        if hasattr(cand, key):
            out[key] = getattr(cand, key)
    return out


def load_cascadeboard_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    strict: bool = False,
) -> CascadeBoardTransformer:
    """Load a CascadeBoardTransformer checkpoint for planner inference.

    Auto-detects architecture params (bottleneck, d_model, n_layers) from the
    state dict so checkpoints with different configs load correctly.
    """
    state = torch.load(checkpoint_path, map_location=device)

    # Infer architecture from state dict keys
    has_bottleneck = any("route_bottleneck" in k for k in state)
    d_model = state.get("pos_emb.weight", state.get("route_latent_emb")).shape[-1]
    n_layers = sum(1 for k in state if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight"))

    kwargs: dict = {"d_model": d_model, "n_layers": n_layers}
    if has_bottleneck:
        # Infer bottleneck dim from weight shape
        bn_weight = state["route_bottleneck.0.weight"]
        kwargs["route_latent_dim"] = bn_weight.shape[0]

    model = CascadeBoardTransformer(**kwargs).to(device)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def propose_neural_edits(
    board: CascadeBoard,
    model: CascadeBoardTransformer,
    energy_api: EnergyAPI,
    compiled: CompiledConstraints | None = None,
    *,
    objective: str = "balanced",
    device: str = "cpu",
    m: int = 8,
) -> list[EditAction]:
    """Propose edits using the trained edit policy head, with actionable fallbacks.

    The current checkpoint predicts edit type and target slot but does not yet
    train candidate attention. This function turns the top policy decisions into
    executable edits and then appends rule-based edits for coverage.
    """
    if board.n_steps == 0:
        return []

    model.eval()
    tensors = board_to_tensors(board, objective=objective, device=device)
    with torch.no_grad():
        out = model(**tensors)

    n_slots = board.n_steps
    edit_probs = torch.softmax(out["edit_type_logits"][0], dim=-1)
    target_probs = torch.softmax(out["edit_target_logits"][0, :n_slots], dim=-1)
    edit_order = torch.argsort(edit_probs, descending=True).detach().cpu().tolist()
    target_order = torch.argsort(target_probs, descending=True).detach().cpu().tolist()

    edits: list[EditAction] = []
    seen = set()

    def add(edit: EditAction) -> None:
        key = (
            edit.edit_type.name,
            edit.slot_index,
            edit.field,
            repr(edit.new_value),
        )
        if key in seen:
            return
        seen.add(key)
        edits.append(edit)

    for edit_idx in edit_order[:4]:
        edit_type = list(EditType)[edit_idx]
        for slot_idx in target_order[:3]:
            if slot_idx >= n_slots:
                continue
            slot = board.slots[slot_idx]

            if edit_type == EditType.FILL_FIELD:
                cond = _condition_from_model(out["cond_preds"][0, slot_idx])
                if slot.T is None and "T" not in slot.fixed_fields:
                    add(EditAction(
                        EditType.FILL_FIELD, slot_index=slot_idx, field="T",
                        new_value=cond["T"], reason="neural_policy: fill T",
                        metadata={"source": "neural"},
                    ))
                if slot.pH is None and "pH" not in slot.fixed_fields:
                    add(EditAction(
                        EditType.FILL_FIELD, slot_index=slot_idx, field="pH",
                        new_value=cond["pH"], reason="neural_policy: fill pH",
                        metadata={"source": "neural"},
                    ))

            elif edit_type == EditType.ADJUST_CONDITION:
                add(EditAction(
                    EditType.ADJUST_CONDITION, slot_index=slot_idx,
                    new_value=_condition_from_model(out["cond_preds"][0, slot_idx]),
                    reason="neural_policy: adjust condition",
                    metadata={"source": "neural"},
                ))

            elif edit_type == EditType.REPLACE_STEP and slot.candidates:
                for cand in slot.candidates[:2]:
                    add(EditAction(
                        EditType.REPLACE_STEP, slot_index=slot_idx,
                        new_value=_candidate_to_dict(cand),
                        reason="neural_policy: replace step",
                        metadata={"source": "neural"},
                    ))

            elif edit_type == EditType.SWAP_ORDER and slot_idx < n_slots - 1:
                add(EditAction(
                    EditType.SWAP_ORDER, slot_index=slot_idx,
                    reason="neural_policy: swap order",
                    metadata={"source": "neural"},
                ))

            elif edit_type == EditType.DELETE_STEP and n_slots > 1:
                add(EditAction(
                    EditType.DELETE_STEP, slot_index=slot_idx,
                    reason="neural_policy: delete step",
                    metadata={"source": "neural"},
                ))

            elif edit_type == EditType.INSERT_STEP:
                add(EditAction(
                    EditType.INSERT_STEP, slot_index=slot_idx,
                    new_value={"reaction_type": "other", "T": 37.0, "pH": 7.0},
                    reason="neural_policy: insert placeholder step",
                    metadata={"source": "neural"},
                ))

            if len(edits) >= m:
                return edits[:m]

    # Coverage fallback: current neural checkpoint has no trained candidate head.
    for edit in propose_edits(board, energy_api, compiled, m=m):
        edit.reason = f"rule_fallback: {edit.reason}"
        edit.metadata["source"] = "rule_fallback"
        add(edit)
        if len(edits) >= m:
            break

    return edits[:m]


# ---------------------------------------------------------------------------
# Particle resampling
# ---------------------------------------------------------------------------

def resample_pareto_diverse(
    proposals: list[ScoredBoard],
    K: int = 32,
) -> list[ScoredBoard]:
    """Resample top-K particles with diversity preservation."""
    if not proposals:
        return []

    # Sort by posterior
    proposals.sort(key=lambda p: p.posterior, reverse=True)

    # Take top-K but ensure diversity (no two boards with identical reaction types)
    selected = []
    seen_signatures = set()

    for p in proposals:
        sig = tuple(s.reaction_type for s in p.board.slots)
        if sig not in seen_signatures or len(selected) < K // 2:
            selected.append(p)
            seen_signatures.add(sig)
        if len(selected) >= K:
            break

    # Fill remaining with top-scoring
    if len(selected) < K:
        for p in proposals:
            if p not in selected:
                selected.append(p)
            if len(selected) >= K:
                break

    return selected[:K]


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

class CascadeBoardPlanner:
    """CascadeBoard++ complete planning pipeline."""

    def __init__(
        self,
        energy_api: EnergyAPI | None = None,
        model: CascadeBoardTransformer | None = None,
        retro_engine: dict | None = None,
        stock_checker=None,
        candidate_cache: dict | None = None,
        device: str = "cpu",
        use_real_candidates: bool = True,
        use_live_retro: bool = False,
    ):
        self.energy_api = energy_api or EnergyAPI()
        self.model = model
        self.stock_checker = stock_checker or (_get_zinc_checker() if use_real_candidates else None)
        self.candidate_cache = candidate_cache or (_load_real_cache() if use_real_candidates and not use_live_retro else None)
        self.device = device
        self.compiler = ConstraintCompiler()
        if self.model is not None:
            self.model.to(self.device)

        # Live retro engine: real-time RetroChimera + EnzExpand inference
        if use_live_retro:
            from cascade_planner.cascadeboard.live_retro import build_live_retro_engine
            self.retro_engine = build_live_retro_engine()
            self.candidate_cache = None  # don't use cache when live retro is available
        else:
            self.retro_engine = retro_engine

    def plan(
        self,
        target: str,
        constraints: dict[str, Any] | None = None,
        objective: str = "balanced",
        n_steps: int | None = None,
        n_particles: int = 32,
        n_refine: int = 5,
        n_final: int = 5,
    ) -> list[RouteResult]:
        """
        Full CascadeBoard++ planning pipeline.

        Args:
            target: target molecule SMILES
            constraints: user constraints dict
            objective: "balanced" / "industrial" / "green" / "novelty"
            n_steps: fixed number of steps (None = auto-estimate)
            n_particles: number of parallel particles
            n_refine: number of refinement iterations
            n_final: number of final routes to return
        """
        t0 = time.time()

        # ===== Layer 0: Compile constraints =====
        board_template = CascadeBoard.from_n_steps(n_steps or 3, target)

        # Apply user constraints to template
        if constraints:
            for fc in constraints.get("fixed_steps", []):
                board_template.fix(fc["index"], **fc["values"])
            if "starting_material" in constraints:
                board_template.fix_starting_material(constraints["starting_material"])
            if "n_steps" in constraints:
                board_template.fix_n_steps(constraints["n_steps"])
            for k in ("one_pot", "max_delta_T", "max_delta_pH",
                       "exclude_catalyst", "exclude_solvent", "prefer_enzymatic"):
                if k in constraints:
                    board_template.set_global_constraint(k, constraints[k])

        compiled = self.compiler.compile(
            board_template,
            raw_constraints=constraints,
            objective=objective,
        )

        # Check for conflicts
        if compiled.conflicts:
            return [RouteResult(
                board=board_template,
                explanation=RouteExplanation(
                    why_selected="CONFLICT DETECTED",
                    constraints_at_risk={
                        c.description: "; ".join(c.suggested_relaxations)
                        for c in compiled.conflicts
                    },
                ),
            )]

        # ===== Layer 1: Build candidate hypergraph =====
        # Multi-depth parallel planning: try multiple depths and merge
        if n_steps is None and not constraints:
            depths_to_try = [2, 3, 4]
        else:
            depths_to_try = [board_template.n_steps]

        all_particles: list[ScoredBoard] = []
        for depth_try in depths_to_try:
            template_d = CascadeBoard.from_n_steps(depth_try, target)
            if constraints:
                for fc in constraints.get("fixed_steps", []):
                    if fc["index"] < depth_try:
                        template_d.fix(fc["index"], **fc["values"])
                for k in ("one_pot", "max_delta_T", "max_delta_pH",
                           "exclude_catalyst", "exclude_solvent", "prefer_enzymatic"):
                    if k in constraints:
                        template_d.set_global_constraint(k, constraints[k])
            compiled_d = self.compiler.compile(template_d, raw_constraints=constraints, objective=objective)

            if self.candidate_cache:
                from cascade_planner.cascadeboard.real_benchmark import StrictCachedGraph
                graph = StrictCachedGraph(
                    cache=self.candidate_cache,
                    stock_checker=self.stock_checker,
                    max_depth=max(depth_try + 1, 3),
                    branch_factor=15,
                )
            else:
                graph = CandidateHypergraph(
                    retro_engine=self.retro_engine,
                    stock_checker=self.stock_checker,
                    max_depth=max(depth_try + 1, 3),
                    branch_factor=15,
                )
            graph.build(target, compiled_d, min_depth=depth_try)
            graph.propagate_constraints(compiled_d)

            if graph.is_empty():
                continue

            paths = graph.sample_paths(n=n_particles // len(depths_to_try), target_depth=depth_try)
            for path in paths:
                board = graph.path_to_board(path, target)
                for key, value in template_d.global_constraints.items():
                    board.set_global_constraint(key, value)
                for i, slot in enumerate(template_d.slots):
                    if i < board.n_steps:
                        for fld in slot.fixed_fields:
                            val = getattr(slot, fld)
                            setattr(board.slots[i], fld, val)
                            board.slots[i].fixed_fields.add(fld)
                energy = self.energy_api.compute_energy(board, compiled_d)
                quality = self.energy_api.compute_quality(board)
                risk = self.energy_api.compute_risk(board)
                all_particles.append(ScoredBoard(
                    board=board, energy=energy, quality=quality, risk=risk, posterior=-energy,
                ))

        particles = all_particles
        if not particles:
            return [RouteResult(
                board=board_template,
                explanation=RouteExplanation(
                    why_selected="NO CANDIDATES",
                    alternative_edits=["Increase max_depth", "Relax constraints"],
                ),
            )]

        # Trim to n_particles best before refinement
        particles = resample_pareto_diverse(particles, K=n_particles)

        # ===== Layer 4+5+6: Iterative refinement =====
        for iteration in range(n_refine):
            new_proposals: list[ScoredBoard] = []

            for particle in particles:
                # Propose edits. Prefer the neural edit policy when a model is
                # loaded, but keep rule-based edits as fallback/ablation.
                if self.model is not None:
                    edits = propose_neural_edits(
                        particle.board, self.model, self.energy_api, compiled,
                        objective=objective, device=self.device, m=4,
                    )
                else:
                    edits = propose_edits(
                        particle.board, self.energy_api, compiled, m=4,
                    )
                    for edit in edits:
                        edit.metadata["source"] = "rule"

                for edit in edits:
                    # When user specified n_steps, don't allow step count changes
                    if n_steps is not None and edit.edit_type in (
                        EditType.DELETE_STEP, EditType.INSERT_STEP,
                    ):
                        continue
                    edit.metadata["pre_energy"] = particle.energy
                    new_board = apply_edit(particle.board, edit)

                    # Check hard constraints
                    if not compiled.hard_satisfied(new_board):
                        edit.metadata["hard_satisfied"] = False
                        continue

                    # Score
                    energy = self.energy_api.compute_energy(new_board, compiled)
                    quality = self.energy_api.compute_quality(new_board)
                    risk = self.energy_api.compute_risk(new_board)
                    if new_board.edit_history:
                        new_board.edit_history[-1].metadata.update({
                            "post_energy": energy,
                            "delta_energy": energy - particle.energy,
                            "hard_satisfied": True,
                        })

                    new_proposals.append(ScoredBoard(
                        board=new_board, energy=energy,
                        quality=quality, risk=risk,
                        posterior=-energy,
                    ))

            # Keep original particles + new proposals
            all_candidates = particles + new_proposals
            particles = resample_pareto_diverse(all_candidates, K=n_particles)

        # ===== Build results =====
        particles.sort(key=lambda p: p.posterior, reverse=True)
        results: list[RouteResult] = []

        for p in particles[:n_final]:
            board = p.board
            idx, reason = self.energy_api.diagnose_bottleneck(board)

            # Use model's real-label heads for richer diagnostics
            model_diagnostics = {}
            if self.model is not None:
                try:
                    tensors = board_to_tensors(board, objective=objective, device=self.device)
                    with torch.no_grad():
                        out = self.model(**tensors)
                    # Compatibility prediction
                    from cascade_planner.cascadeboard.route_encoder import COMPAT_VOCAB, OPMODE_VOCAB, ISSUE_TYPE_VOCAB, PAIRWISE_VOCAB
                    compat_probs = torch.softmax(out["compat_logits"][0], dim=-1).cpu().numpy()
                    model_diagnostics["predicted_compatibility"] = COMPAT_VOCAB[int(compat_probs.argmax())]
                    model_diagnostics["compatibility_confidence"] = float(compat_probs.max())
                    # Operation mode prediction
                    opmode_probs = torch.softmax(out["opmode_logits"][0], dim=-1).cpu().numpy()
                    model_diagnostics["predicted_operation_mode"] = OPMODE_VOCAB[int(opmode_probs.argmax())]
                    # Issue type detection
                    issue_probs = torch.sigmoid(out["issue_type_logits"][0]).cpu().numpy()
                    detected_issues = [ISSUE_TYPE_VOCAB[i] for i, p in enumerate(issue_probs) if p > 0.3]
                    model_diagnostics["detected_issues"] = detected_issues
                    # Pairwise compatibility
                    if out.get("pairwise_logits") is not None and out["pairwise_logits"].shape[1] > 0:
                        pw_preds = torch.softmax(out["pairwise_logits"][0], dim=-1).cpu().numpy()
                        model_diagnostics["pairwise_modes"] = [
                            PAIRWISE_VOCAB[int(pw_preds[i].argmax())]
                            for i in range(min(pw_preds.shape[0], board.n_steps - 1))
                        ]
                    # Use compatibility confidence to adjust overall confidence
                    compat_conf = float(compat_probs[0])  # prob of "empirically_compatible"
                except Exception:
                    compat_conf = 0.5
            else:
                compat_conf = 0.5

            # Build explanation
            explanation = RouteExplanation(
                why_selected=self._explain_selection(board, compiled),
                what_was_changed=[str(e) for e in board.edit_history[-3:]],
                constraints_satisfied=self._check_constraints(board, compiled),
                constraints_at_risk=self._constraints_at_risk(board, compiled),
                global_condition_window=self._condition_window(board),
                evidence_table=self._evidence_table(board),
                edited_slots=[f"Step {e.slot_index}" for e in board.edit_history],
                alternative_edits=self._alternative_edits(board, compiled),
                minimal_relaxation=self._minimal_relaxation(board, compiled),
                uncertainty_table={**(p.risk or {}), **model_diagnostics},
            )

            # Confidence: blend energy-based risk with model's compatibility prediction
            energy_conf = 1.0 - np.mean(list(p.risk.values())) if p.risk else 0.5
            confidence = 0.6 * energy_conf + 0.4 * compat_conf

            results.append(RouteResult(
                board=board,
                quality_vector=p.quality,
                risk_vector=p.risk,
                score=p.posterior,
                confidence=confidence,
                bottleneck_slot=idx,
                bottleneck_reason=reason,
                explanation=explanation,
            ))

        return results

    # ------------------------------------------------------------------
    # Explanation helpers
    # ------------------------------------------------------------------

    def _explain_selection(self, board: CascadeBoard, compiled: CompiledConstraints) -> str:
        parts = [f"{board.n_steps}-step route"]
        n_enz = sum(1 for s in board.slots if s.is_enzymatic())
        if n_enz > 0:
            parts.append(f"{n_enz} enzymatic steps")
        n_fixed = sum(1 for s in board.slots if s.fixed_fields)
        if n_fixed > 0:
            parts.append(f"{n_fixed} user-fixed steps")
        if board.compatibility_scores:
            min_compat = min(board.compatibility_scores)
            parts.append(f"min compatibility={min_compat:.2f}")
        return "; ".join(parts)

    def _check_constraints(self, board: CascadeBoard, compiled: CompiledConstraints) -> dict[str, str]:
        report = {}
        report["hard_constraints"] = "all satisfied" if compiled.hard_satisfied(board) else "VIOLATED"
        for f in compiled.soft_factors:
            report[f.name] = "applied"
        return report

    def _condition_window(self, board: CascadeBoard) -> str:
        temps = [s.T for s in board.slots if s.T is not None]
        phs = [s.pH for s in board.slots if s.pH is not None]
        parts = []
        if temps:
            parts.append(f"T: {min(temps):.0f}-{max(temps):.0f}°C")
        if phs:
            parts.append(f"pH: {min(phs):.1f}-{max(phs):.1f}")
        return ", ".join(parts) or "unknown"

    def _constraints_at_risk(self, board: CascadeBoard, compiled: CompiledConstraints) -> dict:
        at_risk = {}
        for i in range(board.n_steps - 1):
            sa, sb = board.slots[i], board.slots[i + 1]
            if sa.T is not None and sb.T is not None:
                dt = abs(sa.T - sb.T)
                if dt > 15:
                    at_risk[f"ΔT(step{i}-{i+1})"] = f"{dt:.0f}°C (>15°C threshold)"
            if sa.pH is not None and sb.pH is not None:
                dp = abs(sa.pH - sb.pH)
                if dp > 1.5:
                    at_risk[f"ΔpH(step{i}-{i+1})"] = f"{dp:.1f} (>1.5 threshold)"
        for s in board.slots:
            if s.is_enzymatic() and (s.e_enzyme or 0) < 0.2:
                at_risk[f"enzyme_step{s.index}"] = f"low enzyme compatibility ({s.e_enzyme or 0:.2f})"
            if (s.e_retro or 0) < 0.2:
                at_risk[f"retro_step{s.index}"] = f"low retro score ({s.e_retro or 0:.2f})"
        return at_risk

    def _evidence_table(self, board: CascadeBoard) -> list[dict]:
        table = []
        for s in board.slots:
            table.append({
                "step": s.index,
                "reaction_type": s.reaction_type or "unknown",
                "ec": s.ec or "-",
                "e_retro": round(s.e_retro or 0, 3),
                "e_enzyme": round(s.e_enzyme or 0, 3),
                "e_condition": round(s.e_condition or 0, 3),
                "source": s.source or "unknown",
                "in_stock": bool(s.main_reactant and self.stock_checker and self.stock_checker(s.main_reactant)),
            })
        return table

    def _alternative_edits(self, board: CascadeBoard, compiled: CompiledConstraints) -> list[str]:
        alts = []
        idx, reason = self.energy_api.diagnose_bottleneck(board)
        if idx is not None and idx < board.n_steps:
            slot = board.slots[idx]
            if (slot.e_retro or 0) < 0.5:
                alts.append(f"Replace Step {idx} reaction candidate (current retro score={slot.e_retro or 0:.2f})")
            if slot.is_enzymatic() and (slot.e_enzyme or 0) < 0.3:
                alts.append(f"Try different enzyme for Step {idx}")
            if slot.T is not None and idx > 0:
                prev = board.slots[idx - 1]
                if prev.T is not None and abs(slot.T - prev.T) > 20:
                    alts.append(f"Adjust Step {idx} temperature closer to Step {idx-1} ({prev.T:.0f}°C)")
        if not alts:
            alts.append("Route looks reasonable; no obvious improvements")
        return alts

    def _minimal_relaxation(self, board: CascadeBoard, compiled: CompiledConstraints) -> str | None:
        if compiled.hard_satisfied(board):
            return None
        suggestions = []
        for mask in compiled.hard_masks:
            if mask.slot_index is not None and mask.slot_index < board.n_steps:
                val = getattr(board.slots[mask.slot_index], mask.field, None)
                if val is not None and mask.allowed_values and val not in mask.allowed_values:
                    suggestions.append(
                        f"Relax Step {mask.slot_index} {mask.field}: "
                        f"current={val}, required={mask.allowed_values}"
                    )
        return "; ".join(suggestions) if suggestions else "Unable to determine minimal relaxation"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def cascadeboard_plan(
    target: str,
    constraints: dict | None = None,
    objective: str = "balanced",
    n_steps: int | None = None,
    **kwargs,
) -> list[RouteResult]:
    """One-line planning interface."""
    checkpoint = kwargs.pop("model_checkpoint", None)
    device = kwargs.pop("device", "cpu")
    motif_path = kwargs.pop("motif_path", "results/shared/cascade_motifs.json")
    model = load_cascadeboard_model(checkpoint, device=device) if checkpoint else None

    import json
    from pathlib import Path
    motif_memory = None
    if Path(motif_path).exists():
        motif_memory = json.loads(Path(motif_path).read_text())

    energy_api = EnergyAPI(motif_memory=motif_memory)
    planner = CascadeBoardPlanner(energy_api=energy_api, model=model, device=device)
    return planner.plan(target, constraints, objective, n_steps, **kwargs)
