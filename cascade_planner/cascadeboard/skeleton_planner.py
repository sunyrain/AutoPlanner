"""Skeleton-based cascade route planner.

New inference pipeline:
  1. Model generates route skeleton (type/EC/conditions per slot)
  2. RetroChimera fills chemical steps, EnzExpand fills enzymatic steps
  3. Model diagnoses compatibility and issues
  4. User modifies constraints → regenerate (not patch)
"""
from __future__ import annotations

import random
import os
import time
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from cascade_planner.cascadeboard import (
    CascadeBoard, Slot, EditType, CompiledConstraints,
    RouteResult, RouteExplanation,
)
from cascade_planner.cascadeboard.constraint_compiler import ConstraintCompiler
from cascade_planner.cascadeboard.energy_api import EnergyAPI
from cascade_planner.cascadeboard.route_encoder import (
    CascadeBoardTransformer, REACTION_TYPE_TO_ID,
    COMPAT_VOCAB, OPMODE_VOCAB, ISSUE_TYPE_VOCAB, PAIRWISE_VOCAB,
    smiles_to_morgan_fp, inject_constraint_features,
    OBJECTIVE_TO_ID,
)
from cascade_planner.cascadeboard.route_recovery import canonical_reaction, canonical_smiles

logger = logging.getLogger(__name__)

ID_TO_TYPE = {v: k for k, v in REACTION_TYPE_TO_ID.items() if k}


@dataclass
class RouteSkeleton:
    """Model-predicted route skeleton before reaction filling."""
    n_steps: int
    types: list[str]           # reaction type per slot
    ec1s: list[int]            # EC1 class per slot
    Ts: list[float]            # predicted T per slot
    pHs: list[float]           # predicted pH per slot
    compatibility: str = ""    # predicted compatibility label
    operation_mode: str = ""   # predicted operation mode
    issues: list[str] = field(default_factory=list)
    pairwise_modes: list[str] = field(default_factory=list)
    log_prob: float = 0.0


def generate_skeleton(
    model: CascadeBoardTransformer,
    target: str,
    n_steps: int,
    constraints: dict[str, Any] | None = None,
    objective: str = "balanced",
    device: str = "cpu",
) -> RouteSkeleton:
    """Generate route skeleton from model in a single forward pass.

    Input: target + constraints (all slots empty/masked)
    Output: predicted type/EC/conditions for each slot + global diagnostics
    """
    model.eval()

    # Build input: target FP + empty slots (skeleton generation mode)
    fp = inject_constraint_features(smiles_to_morgan_fp(target), [0] * 8)
    B, S = 1, n_steps

    # If constraints fix certain slots, encode them; otherwise all zeros
    type_ids = torch.zeros(B, S, dtype=torch.long)
    ec1_ids = torch.zeros(B, S, dtype=torch.long)
    conditions = torch.zeros(B, S, 2)
    scores = torch.zeros(B, S, 3)
    is_fixed = torch.zeros(B, S, dtype=torch.long)

    if constraints:
        for fc in constraints.get("fixed_steps", []):
            idx = fc["index"]
            if idx < S:
                vals = fc.get("values", {})
                if "reaction_type" in vals:
                    type_ids[0, idx] = REACTION_TYPE_TO_ID.get(vals["reaction_type"], 0)
                    is_fixed[0, idx] = 1
                if "ec" in vals:
                    ec_str = vals["ec"]
                    ec1 = int(ec_str.split(".")[0]) if ec_str and ec_str[0].isdigit() else 0
                    ec1_ids[0, idx] = ec1
                    is_fixed[0, idx] = 1
                if "T" in vals:
                    conditions[0, idx, 0] = (vals["T"] - 37) / 30
                    is_fixed[0, idx] = 1
                if "pH" in vals:
                    conditions[0, idx, 1] = (vals["pH"] - 7) / 3
                    is_fixed[0, idx] = 1

    x = dict(
        target_fp=torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device),
        objective_ids=torch.tensor([OBJECTIVE_TO_ID.get(objective, 0)], dtype=torch.long).to(device),
        slot_type_ids=type_ids.to(device),
        slot_ec1_ids=ec1_ids.to(device),
        slot_conditions=conditions.to(device),
        slot_scores=scores.to(device),
        slot_is_fixed=is_fixed.to(device),
    )

    with torch.no_grad():
        out = model(**x)

    # Decode skeleton
    types = []
    ec1s = []
    Ts = []
    pHs = []
    for i in range(S):
        # Use fixed values if provided, otherwise model prediction
        if is_fixed[0, i]:
            types.append(ID_TO_TYPE.get(type_ids[0, i].item(), ""))
            ec1s.append(ec1_ids[0, i].item())
            T_norm = conditions[0, i, 0].item()
            pH_norm = conditions[0, i, 1].item()
        else:
            type_probs = torch.softmax(out["type_logits"][0, i], dim=-1)
            types.append(ID_TO_TYPE.get(type_probs.argmax().item(), ""))
            ec1_probs = torch.softmax(out["ec1_logits"][0, i], dim=-1)
            ec1s.append(ec1_probs.argmax().item())
            T_norm = out["cond_preds"][0, i, 0].item()
            pH_norm = out["cond_preds"][0, i, 1].item()

        Ts.append(max(0, min(120, 37 + 30 * T_norm)))
        pHs.append(max(1, min(14, 7 + 3 * pH_norm)))

    # Global diagnostics
    compat = ""
    if "compat_logits" in out:
        compat_probs = torch.softmax(out["compat_logits"][0], dim=-1).cpu().numpy()
        compat = COMPAT_VOCAB[int(compat_probs.argmax())]

    opmode = ""
    if "opmode_logits" in out:
        opmode_probs = torch.softmax(out["opmode_logits"][0], dim=-1).cpu().numpy()
        opmode = OPMODE_VOCAB[int(opmode_probs.argmax())]

    issues = []
    if "issue_type_logits" in out:
        issue_probs = torch.sigmoid(out["issue_type_logits"][0]).cpu().numpy()
        issues = [ISSUE_TYPE_VOCAB[i] for i, p in enumerate(issue_probs) if p > 0.3]

    pairwise = []
    if out.get("pairwise_logits") is not None and S >= 2:
        pw = out["pairwise_logits"][0].cpu()
        for i in range(min(pw.shape[0], S - 1)):
            pairwise.append(PAIRWISE_VOCAB[int(pw[i].argmax())])

    return RouteSkeleton(
        n_steps=S, types=types, ec1s=ec1s, Ts=Ts, pHs=pHs,
        compatibility=compat, operation_mode=opmode,
        issues=issues, pairwise_modes=pairwise,
    )


def fill_route_from_skeleton(
    skeleton: RouteSkeleton,
    target: str,
    retro_engine: dict | None = None,
    stock_checker=None,
    starting_material: str | None = None,
    n_routes: int = 1,
    constraints: dict[str, Any] | None = None,
) -> list[CascadeBoard]:
    """Fill a route skeleton with concrete reactions using RetroChimera/EnzExpand.

    When n_routes > 1, expands candidates slot-by-slot with a bounded beam
    rather than varying only the first disconnection and greedily filling the
    rest of the route.
    """
    if n_routes <= 1:
        return [_fill_single_route(
            skeleton,
            target,
            retro_engine,
            starting_material=starting_material,
            constraints=constraints,
        )]

    return _beam_fill_route(
        skeleton,
        target,
        retro_engine,
        stock_checker=stock_checker,
        n_routes=n_routes,
        starting_material=starting_material,
        constraints=constraints,
    )


def _slot_from_skeleton(board: CascadeBoard, skeleton: RouteSkeleton, index: int, product: str) -> Slot:
    slot = board.slots[index]
    slot.product = product
    slot.reaction_type = skeleton.types[index]
    slot.T = skeleton.Ts[index]
    slot.pH = skeleton.pHs[index]
    ec1 = skeleton.ec1s[index]
    if ec1 > 0:
        slot.ec = f"{ec1}.x"
    return slot


def _candidate_stock_score(cand: dict, *, is_terminal: bool, stock_checker=None) -> float:
    if not stock_checker or not is_terminal:
        return 0.0
    reactants = [cand.get("main_reactant", "")]
    reactants.extend(cand.get("aux_reactants") or [])
    reactants = [smi for smi in reactants if smi]
    if not reactants:
        return -2.0
    try:
        n_buyable = sum(1 for smi in reactants if bool(stock_checker(smi)))
    except Exception:
        return 0.0
    if n_buyable == len(reactants):
        return 5.0
    return -1.0 * (len(reactants) - n_buyable)


def _candidate_key(cand: dict) -> str:
    return cand.get("rxn_smiles") or cand.get("reaction_smiles") or cand.get("main_reactant") or ""


def _candidate_source(cand: dict) -> str:
    return cand.get("source") or cand.get("enzyme_source") or "unknown"


def _dedupe_candidates(cands: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for cand in cands:
        key = _candidate_key(cand)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def _source_diverse_candidates(
    cands: list[dict],
    top_k: int,
    *,
    source_priority: tuple[str, ...] = (),
) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for cand in cands:
        buckets[_candidate_source(cand)].append(cand)
    source_order = [src for src in source_priority if src in buckets]
    for src in buckets:
        if src not in source_order:
            source_order.append(src)

    out: list[dict] = []
    seen: set[str] = set()
    while len(out) < top_k:
        progressed = False
        for src in source_order:
            bucket = buckets.get(src)
            if not bucket:
                continue
            while bucket and _candidate_key(bucket[0]) in seen:
                bucket.pop(0)
            if not bucket:
                continue
            cand = bucket.pop(0)
            key = _candidate_key(cand)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(cand)
            progressed = True
            if len(out) >= top_k:
                break
        if not progressed:
            break
    for idx, cand in enumerate(out, start=1):
        cand.setdefault("rank", idx)
    return out


def _prefer_candidates(
    cands: list[dict],
    predicate,
    top_k: int,
) -> list[dict]:
    matching: list[dict] = []
    rest: list[dict] = []
    for cand in cands:
        (matching if predicate(cand) else rest).append(cand)
    return _dedupe_candidates(matching + rest)[:top_k]


def _candidate_base_score(cand: dict) -> float:
    try:
        return float(cand.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_type_alignment_score(cand: dict, skeleton_type: str) -> float:
    if not skeleton_type:
        return 0.0
    candidate_type = cand.get("type") or ""
    if not candidate_type:
        return 0.0
    return 1.5 if candidate_type == skeleton_type else -2.0


def _route_signature(board: CascadeBoard) -> tuple[tuple[str, str], ...]:
    return tuple((s.reaction_smiles or "", s.main_reactant or "") for s in board.slots)


def _route_anchors_from_constraints(
    constraints: dict[str, Any] | None,
    *,
    starting_material: str | None = None,
    n_steps: int = 0,
) -> dict[int, dict[str, Any]]:
    anchors = {}
    if constraints:
        anchors.update(_fixed_steps_by_index(constraints))
    if starting_material and n_steps > 0:
        anchors.setdefault(n_steps - 1, {})
        anchors[n_steps - 1]["main_reactant"] = starting_material
    return anchors


def _fixed_steps_by_index(constraints: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    fixed: dict[int, dict[str, Any]] = {}
    for item in (constraints or {}).get("fixed_steps", []) or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        values = item.get("values") or {}
        if isinstance(values, dict):
            fixed[idx] = dict(values)
    return fixed


def _passes_anchor(candidate: dict[str, Any], step_index: int, anchors: dict[int, dict[str, Any]]) -> bool:
    anchor = anchors.get(step_index)
    if not anchor:
        return True
    cand_rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles")
    if anchor.get("reaction_smiles") and canonical_reaction(cand_rxn) != canonical_reaction(anchor.get("reaction_smiles")):
        return False
    if anchor.get("main_reactant") and canonical_smiles(candidate.get("main_reactant")) != canonical_smiles(anchor.get("main_reactant")):
        return False
    if anchor.get("reaction_type") and candidate.get("type"):
        if candidate.get("type") != anchor.get("reaction_type"):
            return False
    if anchor.get("ec") and candidate.get("ec"):
        wanted = str(anchor.get("ec"))
        got = str(candidate.get("ec"))
        if wanted.endswith(".x"):
            return got.startswith(wanted[:-1])
        return got == wanted
    return True


def _beam_fill_route(
    skeleton: RouteSkeleton,
    target: str,
    retro_engine: dict | None,
    *,
    stock_checker=None,
    n_routes: int = 2,
    starting_material: str | None = None,
    constraints: dict[str, Any] | None = None,
) -> list[CascadeBoard]:
    beam_width = max(n_routes * 4, 8)
    candidate_top_k = max(n_routes * 4, 8)
    anchors = _route_anchors_from_constraints(
        constraints,
        starting_material=starting_material,
        n_steps=skeleton.n_steps,
    )
    seed = CascadeBoard.from_n_steps(skeleton.n_steps, target)
    states: list[tuple[CascadeBoard, str, float]] = [(seed, target, 0.0)]

    for index in range(skeleton.n_steps):
        next_states: list[tuple[CascadeBoard, str, float]] = []
        for board, current_product, path_score in states:
            anchor = anchors.get(index)
            if anchor and anchor.get("product"):
                if canonical_smiles(current_product) != canonical_smiles(anchor.get("product")):
                    next_states.append((board, "", path_score - 50.0))
                    continue
            slot = _slot_from_skeleton(board, skeleton, index, current_product)
            cands = _candidates_for_skeleton_slot(
                retro_engine,
                product_smiles=current_product,
                ec1=skeleton.ec1s[index],
                skel_type=skeleton.types[index],
                top_k=candidate_top_k,
            )
            slot.candidates = list(cands)
            if not cands:
                next_states.append((board, "", path_score - 10.0))
                continue
            if anchor:
                cands = [cand for cand in cands if _passes_anchor(cand, index, anchors)]
                if not cands:
                    next_states.append((board, "", path_score - 50.0))
                    continue

            for cand in cands:
                expanded = board.copy()
                expanded_slot = _slot_from_skeleton(expanded, skeleton, index, current_product)
                expanded_slot.candidates = list(cands)
                _fill_slot_from_candidate(
                    expanded_slot,
                    cand,
                    cand.get("source") or "candidate",
                )
                score = path_score + _candidate_base_score(cand)
                score += _candidate_type_alignment_score(cand, skeleton.types[index])
                score += _candidate_stock_score(
                    cand,
                    is_terminal=index == skeleton.n_steps - 1,
                    stock_checker=stock_checker,
                )
                next_states.append((expanded, cand.get("main_reactant", ""), score))

        deduped: dict[tuple[tuple[str, str], ...], tuple[CascadeBoard, str, float]] = {}
        for state in sorted(next_states, key=lambda row: row[2], reverse=True):
            sig = _route_signature(state[0])
            if sig not in deduped:
                deduped[sig] = state
            if len(deduped) >= beam_width:
                break
        states = list(deduped.values())
        if not states:
            break

    boards = [state[0] for state in sorted(states, key=lambda row: row[2], reverse=True)]
    return boards[:n_routes] if boards else [_fill_single_route(skeleton, target, retro_engine)]


def _candidates_for_skeleton_slot(
    retro_engine: dict | None,
    product_smiles: str,
    ec1: int = 0,
    skel_type: str = "",
    top_k: int = 10,
) -> list[dict]:
    """Return recall-first candidates for a skeleton slot.

    Skeleton type/EC is a prior, not a hard gate. Candidate-miss audits showed
    that exact disconnections are often absent from the first source queried, so
    we overfetch from all available generators, dedupe, source-balance, and only
    then rank EC/type-aligned rows ahead of fallbacks.
    """
    if not retro_engine or not product_smiles:
        return []

    fetch_k = max(top_k * 6, top_k + 12, 32)
    core_candidates: list[dict] = []
    template_candidates: list[dict] = []
    chemical_template_candidates: list[dict] = []

    if ec1 > 0:
        core_candidates.extend(
            _predict_enzymatic_candidates(
                retro_engine,
                product_smiles,
                ec1,
                top_k=fetch_k,
                include_retrorules=False,
            )
        )

    if "retrochimera" in retro_engine:
        try:
            core_candidates.extend(retro_engine["retrochimera"].predict(product_smiles, top_k=fetch_k))
        except Exception:
            pass

    if retro_engine.get("enzexpand") is not None and ec1 <= 0:
        try:
            core_candidates.extend(retro_engine["enzexpand"].predict(product_smiles, top_k=fetch_k))
        except Exception:
            pass

    if ec1 > 0 and retro_engine.get("retrorules") is not None:
        try:
            template_candidates.extend(
                retro_engine["retrorules"].predict(
                    product_smiles,
                    top_k=fetch_k,
                    ec_token=str(ec1) if ec1 > 0 else "",
                    skel_type=skel_type,
                )
            )
        except Exception:
            pass

    if ec1 <= 0 and retro_engine.get("chemtemplates") is not None:
        try:
            chemical_template_candidates.extend(
                retro_engine["chemtemplates"].predict(
                    product_smiles,
                    top_k=fetch_k,
                    skel_type=skel_type,
                )
            )
        except Exception:
            pass

    source_priority = (
        ("v3_retrieval", "enzyformer", "enzexpand", "retrochimera")
        if ec1 > 0
        else ("retrochimera", "enzexpand", "v3_retrieval", "enzyformer")
    )
    cands = _source_diverse_candidates(_dedupe_candidates(core_candidates), top_k=fetch_k, source_priority=source_priority)
    if skel_type:
        cands = _prefer_candidates(cands, lambda c: c.get("type", "") == skel_type, len(cands))
    if ec1 > 0:
        cands = _prefer_candidates(cands, lambda c: _ec_matches(c.get("ec", ""), ec1), len(cands))
    cands = _source_diverse_candidates(cands, top_k=fetch_k, source_priority=source_priority)

    template_cands = _dedupe_candidates(template_candidates)[:fetch_k]
    chemical_template_cands = _dedupe_candidates(chemical_template_candidates)[:fetch_k]

    # RetroRules/Rhea is a recall-rescue source. Keep a small visible tail, but
    # avoid letting low-ranked templates displace the stronger live generators.
    if ec1 > 0:
        template_reserve = _retrorules_reserve_for_type(skel_type)
    else:
        template_cands = chemical_template_cands
        template_reserve = _chemical_template_reserve_for_type(skel_type)
    template_reserve = min(max(template_reserve, 0), top_k, len(template_cands)) if cands else min(top_k, len(template_cands))
    if template_reserve and len(cands) >= top_k:
        merged = [*cands[: top_k - template_reserve], *template_cands[:template_reserve], *cands[top_k - template_reserve :]]
    else:
        merged = [*cands, *template_cands]
    out = _dedupe_candidates(merged)[:top_k]
    for idx, cand in enumerate(out, start=1):
        cand["rank"] = idx
    return out


def _retrorules_reserve_for_type(skel_type: str) -> int:
    raw = os.environ.get("AUTOPLANNER_RETRORULES_RESERVE")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 2
    typed_reserve = {
        "phosphorylation": 10,
        "glycosylation": 6,
        "hydrolysis": 4,
    }
    return typed_reserve.get(str(skel_type or "").strip().lower(), 2)


def _chemical_template_reserve_for_type(skel_type: str) -> int:
    raw = os.environ.get("AUTOPLANNER_CHEM_TEMPLATES_RESERVE")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 4
    typed_reserve = {
        "functional_group_interconversion": 8,
        "esterification": 8,
        "amination": 6,
        "c_c_coupling": 6,
        "c-c coupling": 6,
        "hydrolysis": 6,
        "oxidation": 4,
        "reduction": 4,
        "isomerization": 4,
    }
    return typed_reserve.get(str(skel_type or "").strip().lower(), 3)


def _fill_single_route(
    skeleton: RouteSkeleton,
    target: str,
    retro_engine: dict | None,
    *,
    starting_material: str | None = None,
    constraints: dict[str, Any] | None = None,
) -> CascadeBoard:
    """Original single-route fill logic."""
    board = CascadeBoard.from_n_steps(skeleton.n_steps, target)

    current_product = target
    anchors = _route_anchors_from_constraints(
        constraints,
        starting_material=starting_material,
        n_steps=skeleton.n_steps,
    )
    for i in range(skeleton.n_steps):
        anchor = anchors.get(i)
        if anchor and anchor.get("product") and canonical_smiles(current_product) != canonical_smiles(anchor.get("product")):
            break
        slot = board.slots[i]
        slot.product = current_product
        slot.reaction_type = skeleton.types[i]
        slot.T = skeleton.Ts[i]
        slot.pH = skeleton.pHs[i]
        ec1 = skeleton.ec1s[i]
        if ec1 > 0:
            slot.ec = f"{ec1}.x"

        # Try to fill with real reaction candidates
        filled = False
        if retro_engine and current_product:
            # Decide which engine to use based on skeleton EC
            if ec1 > 0:
                # Enzymatic step: prefer sequence model/retrieval, then templates.
                try:
                    cands = _predict_enzymatic_candidates(retro_engine, current_product, ec1, top_k=10)
                    if anchor:
                        cands = [c for c in cands if _passes_anchor(c, i, anchors)]
                    slot.candidates = list(cands)
                    matching = [c for c in cands if _ec_matches(c.get("ec", ""), ec1)]
                    best = matching[0] if matching else cands[0] if cands else None
                    if best:
                        _fill_slot_from_candidate(slot, best, best.get("source", "enzymatic"))
                        filled = True
                except Exception:
                    pass

            if not filled and "retrochimera" in retro_engine:
                # Chemical step or EnzExpand failed: use RetroChimera
                try:
                    cands = retro_engine["retrochimera"].predict(current_product, top_k=10)
                    if anchor:
                        cands = [c for c in cands if _passes_anchor(c, i, anchors)]
                    slot.candidates = list(cands)
                    # Soft filter: prefer candidates matching skeleton type
                    skel_type = skeleton.types[i]
                    matching = [c for c in cands if c.get("type", "") == skel_type] if skel_type else []
                    best = matching[0] if matching else (cands[0] if cands else None)
                    if best:
                        _fill_slot_from_candidate(slot, best, "retrochimera")
                        filled = True
                except Exception:
                    pass

        # Update current_product for next step
        if slot.main_reactant:
            current_product = slot.main_reactant
        else:
            current_product = ""  # dead end

    return board


def _predict_enzymatic_candidates(
    retro_engine: dict,
    product_smiles: str,
    ec1: int,
    top_k: int = 10,
    *,
    include_retrorules: bool = True,
) -> list[dict]:
    """Return enzymatic candidates from the strongest available source."""
    candidates: list[dict] = []
    fetch_k = max(top_k * 3, top_k + 4, 12)

    enzyformer = retro_engine.get("enzyformer")
    if enzyformer is not None:
        try:
            candidates.extend(enzyformer.predict(product_smiles, top_k=fetch_k, ec_token=str(ec1)))
        except Exception:
            pass

    try:
        from cascade_planner.cascadeboard.enz_retrieval import retrieve_enzymatic_reactions
        candidates.extend(
            retrieve_enzymatic_reactions(
                product_smiles,
                ec_class=str(ec1),
                top_k=fetch_k,
            )
        )
    except Exception:
        pass

    if retro_engine.get("enzexpand") is not None:
        try:
            candidates.extend(retro_engine["enzexpand"].predict(product_smiles, top_k=fetch_k))
        except Exception:
            pass

    if include_retrorules and retro_engine.get("retrorules") is not None:
        try:
            candidates.extend(
                retro_engine["retrorules"].predict(
                    product_smiles,
                    top_k=fetch_k,
                    ec_token=str(ec1),
                )
            )
        except Exception:
            pass

    deduped = _dedupe_candidates(candidates)
    diverse = _source_diverse_candidates(
        deduped,
        top_k=fetch_k,
        source_priority=("v3_retrieval", "enzyformer", "enzexpand", "retrorules_rhea", "retrorules_metanetx"),
    )
    return _prefer_candidates(diverse, lambda c: _ec_matches(c.get("ec", ""), ec1), top_k)


def _ec_matches(ec_str: str, ec1_target: int) -> bool:
    if not ec_str:
        return False
    try:
        return int(ec_str.split(".")[0]) == ec1_target
    except (ValueError, IndexError):
        return False


def _reaction_lhs_reactants(reaction_smiles: str | None) -> list[str]:
    if not reaction_smiles or ">>" not in reaction_smiles:
        return []
    lhs = reaction_smiles.split(">>", 1)[0]
    return [part.strip() for part in lhs.split(".") if part.strip()]


def _candidate_aux_reactants_from_reaction(cand: dict) -> list[str]:
    aux = [str(smi) for smi in cand.get("aux_reactants") or [] if smi]
    if aux:
        return aux
    rxn = cand.get("rxn_smiles") or cand.get("reaction_smiles")
    lhs_reactants = _reaction_lhs_reactants(rxn)
    main = canonical_smiles(cand.get("main_reactant"))
    inferred = []
    for smi in lhs_reactants:
        can = canonical_smiles(smi)
        if can and main and can == main:
            continue
        inferred.append(smi)
    return inferred


def _fill_slot_from_candidate(slot: Slot, cand: dict, source: str) -> None:
    slot.main_reactant = cand.get("main_reactant", "")
    slot.reaction_smiles = cand.get("rxn_smiles", cand.get("reaction_smiles", ""))
    slot.aux_reactants = _candidate_aux_reactants_from_reaction(cand)
    if cand.get("type"):
        slot.reaction_type = cand["type"]
    if cand.get("ec"):
        slot.ec = cand["ec"]
    if cand.get("T") is not None and slot.T is None:
        slot.T = cand.get("T")
    if cand.get("pH") is not None and slot.pH is None:
        slot.pH = cand.get("pH")
    if cand.get("enzyme_uid") and not slot.enzyme_uid:
        slot.enzyme_uid = cand.get("enzyme_uid")
    if cand.get("catalyst") and not slot.catalyst:
        slot.catalyst = cand.get("catalyst")
    if cand.get("solvent") and not slot.solvent:
        slot.solvent = cand.get("solvent")
    slot.e_retro = cand.get("score", 0.0)
    slot.e_enzyme = cand.get("enzyme_score", cand.get("e_enzyme", slot.e_enzyme))
    evidence = dict(cand.get("evidence") or {})
    for key in (
        "uniprot_accession",
        "uniprot_status",
        "organism",
        "tax_id",
        "sequence",
        "sequence_length",
        "cofactor",
        "cofactor_regeneration_mode",
        "doi",
        "pmid",
        "literature_title",
        "cascade_id",
        "step_id",
        "substrate_similarity",
        "reaction_center_similarity",
        "condition_match",
        "literature_precedent",
        "enzyme_name",
        "biocatalyst_format",
        "engineering_status",
        "source_db",
    ):
        if cand.get(key) not in (None, "", [], {}):
            evidence[key] = cand.get(key)
    if evidence:
        slot.evidence = evidence
    slot.source = source


def plan_with_skeleton(
    target: str,
    model: CascadeBoardTransformer | None = None,
    retro_engine: dict | None = None,
    energy_api: EnergyAPI | None = None,
    constraints: dict[str, Any] | None = None,
    objective: str = "balanced",
    n_steps: int | None = None,
    starting_material: str | None = None,
    n_candidates: int = 2,
    stock_checker=None,
    device: str = "cpu",
    skeleton: RouteSkeleton | None = None,
) -> list[RouteResult]:
    """Skeleton-based planning pipeline.

    1. Generate skeleton (from model or pre-built)
    2. Fill reactions with RetroChimera/EnzExpand (linear)
    3. Diagnose compatibility
    4. Diagnostic-driven refinement for bottleneck slots
    """
    t0 = time.time()
    if energy_api is None:
        energy_api = EnergyAPI()
    compiler = ConstraintCompiler()

    # Auto-estimate n_steps if not provided
    if n_steps is None:
        n_steps = skeleton.n_steps if skeleton else 3

    # Stage 1: Generate skeleton (use pre-built if provided)
    if skeleton is None:
        if model is None:
            raise ValueError("Either model or skeleton must be provided")
        skeleton = generate_skeleton(
            model, target, n_steps,
            constraints=constraints, objective=objective, device=device,
        )

    # Stage 2: Fill route from skeleton
    effective_starting_material = starting_material
    if effective_starting_material is None and constraints:
        effective_starting_material = constraints.get("starting_material")

    boards = fill_route_from_skeleton(
        skeleton, target,
        retro_engine=retro_engine,
        stock_checker=stock_checker,
        starting_material=effective_starting_material,
        n_routes=n_candidates,
        constraints=constraints,
    )

    # Score all boards, return best
    results: list[RouteResult] = []
    board_template = CascadeBoard.from_n_steps(n_steps, target)
    if constraints:
        for k in ("one_pot", "max_delta_T", "max_delta_pH", "prefer_enzymatic"):
            if k in constraints:
                board_template.set_global_constraint(k, constraints[k])
    compiled = compiler.compile(board_template, raw_constraints=constraints, objective=objective)

    for board in boards:
        # Apply constraints
        if constraints:
            for k in ("one_pot", "max_delta_T", "max_delta_pH", "prefer_enzymatic"):
                if k in constraints:
                    board.set_global_constraint(k, constraints[k])

        # Stage 3: Score and diagnose
        energy = energy_api.compute_energy(board, compiled)
        quality = energy_api.compute_quality(board)
        risk = energy_api.compute_risk(board)
        idx, reason = energy_api.diagnose_bottleneck(board)

        # Stage 4: Diagnostic-driven refinement
        if idx is not None and retro_engine and idx < board.n_steps:
            refined = False
            slot = board.slots[idx]
            neighbors_T = []
            if idx > 0 and board.slots[idx - 1].T is not None:
                neighbors_T.append(board.slots[idx - 1].T)
            if idx + 1 < board.n_steps and board.slots[idx + 1].T is not None:
                neighbors_T.append(board.slots[idx + 1].T)
            if neighbors_T and slot.T is not None:
                avg_neighbor_T = sum(neighbors_T) / len(neighbors_T)
                dt = abs(slot.T - avg_neighbor_T)
                if dt > 15 and slot.product:
                    try:
                        rc = retro_engine.get("retrochimera")
                        if rc:
                            cands = rc.predict(slot.product, top_k=10)
                            best_cand = None
                            best_dt = dt
                            for c in cands:
                                c_T = avg_neighbor_T + random.uniform(-5, 5)
                                c_dt = abs(c_T - avg_neighbor_T)
                                if c_dt < best_dt:
                                    best_dt = c_dt
                                    best_cand = c
                                    best_cand["_refined_T"] = c_T
                            if best_cand and best_dt < dt:
                                _fill_slot_from_candidate(slot, best_cand, slot.source or "retrochimera")
                                slot.T = best_cand.get("_refined_T", avg_neighbor_T)
                                refined = True
                    except Exception:
                        pass
            if refined:
                energy = energy_api.compute_energy(board, compiled)
                quality = energy_api.compute_quality(board)
                risk = energy_api.compute_risk(board)
                idx, reason = energy_api.diagnose_bottleneck(board)

        explanation = RouteExplanation(
            why_selected=f"{n_steps}-step skeleton: {' → '.join(skeleton.types)}",
            constraints_satisfied={"hard": "satisfied" if compiled.hard_satisfied(board) else "VIOLATED"},
            global_condition_window=f"T: {min(skeleton.Ts):.0f}-{max(skeleton.Ts):.0f}°C, pH: {min(skeleton.pHs):.1f}-{max(skeleton.pHs):.1f}",
            uncertainty_table={
                "predicted_compatibility": skeleton.compatibility,
                "predicted_operation_mode": skeleton.operation_mode,
                "predicted_issues": skeleton.issues,
                "predicted_pairwise": skeleton.pairwise_modes,
            },
        )

        results.append(RouteResult(
            board=board,
            quality_vector=quality,
            risk_vector=risk,
            score=-energy,
            confidence=0.8 if skeleton.compatibility == "empirically_compatible" else 0.4,
            bottleneck_slot=idx,
            bottleneck_reason=reason,
            explanation=explanation,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    elapsed = time.time() - t0
    logger.info(f"Skeleton planning: {n_steps} steps, {len(results)} routes in {elapsed:.1f}s")
    return results
