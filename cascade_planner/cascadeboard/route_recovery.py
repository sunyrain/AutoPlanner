"""Exact and edit-distance route recovery metrics for benchmark artifacts."""
from __future__ import annotations

from collections import Counter
from typing import Any

from rdkit import Chem


def canonical_smiles(smi: str | None) -> str:
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi.strip()
    return Chem.MolToSmiles(mol)


def canonical_side(side: str) -> tuple[str, ...]:
    parts = [canonical_smiles(x.strip()) for x in (side or "").split(".")]
    return tuple(sorted(x for x in parts if x))


def canonical_reaction(rxn_smiles: str | None) -> str:
    if not rxn_smiles or ">>" not in rxn_smiles:
        return ""
    lhs, rhs = rxn_smiles.split(">>", 1)
    return ".".join(canonical_side(lhs)) + ">>" + ".".join(canonical_side(rhs))


def reaction_reactants(rxn_smiles: str | None) -> set[str]:
    if not rxn_smiles or ">>" not in rxn_smiles:
        return set()
    lhs = rxn_smiles.split(">>", 1)[0]
    return set(canonical_side(lhs))


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i]
        for j, bj in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ai == bj else 1),
            ))
        prev = cur
    return prev[-1]


def _best_ordered_fraction(pred: list[str], gt: list[str]) -> float | None:
    if not gt:
        return None
    if len(pred) != len(gt):
        return 0.0
    forward = sum(1 for p, g in zip(pred, gt) if p == g) / len(gt)
    reverse = sum(1 for p, g in zip(pred, list(reversed(gt))) if p == g) / len(gt)
    return max(forward, reverse)


def _best_edit_distance(pred: list[str], gt: list[str]) -> int | None:
    if not gt:
        return None
    return min(_levenshtein(pred, gt), _levenshtein(pred, list(reversed(gt))))


def gt_reaction_keys(entry: dict[str, Any]) -> list[str]:
    return [
        canonical_reaction(step.get("rxn_smiles"))
        for step in entry.get("gt_route", [])
        if canonical_reaction(step.get("rxn_smiles"))
    ]


def route_reaction_keys(route: dict[str, Any]) -> list[str]:
    return [
        canonical_reaction(step.get("reaction_smiles"))
        for step in route.get("steps", [])
        if canonical_reaction(step.get("reaction_smiles"))
    ]


def gt_reactants(entry: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for step in entry.get("gt_route", []):
        out.update(reaction_reactants(step.get("rxn_smiles")))
    return out


def route_reactants(route: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for step in route.get("steps", []):
        if step.get("main_reactant"):
            out.add(canonical_smiles(step.get("main_reactant")))
        for smi in step.get("aux_reactants") or []:
            out.add(canonical_smiles(smi))
        out.update(reaction_reactants(step.get("reaction_smiles")))
    return {x for x in out if x}


def route_candidate_reaction_keys(route: dict[str, Any]) -> list[str]:
    keys = []
    for step in route.get("steps", []):
        pool = (step.get("candidate_pool") or {}).get("top_candidates") or []
        for cand in pool:
            key = canonical_reaction(cand.get("reaction_smiles"))
            if key:
                keys.append(key)
    return keys


def route_candidate_reactants(route: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for step in route.get("steps", []):
        pool = (step.get("candidate_pool") or {}).get("top_candidates") or []
        for cand in pool:
            out.update(candidate_reactants(cand))
    return {x for x in out if x}


def candidate_reactants(candidate: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    if candidate.get("main_reactant"):
        out.add(canonical_smiles(candidate.get("main_reactant")))
    for smi in candidate.get("aux_reactants") or []:
        out.add(canonical_smiles(smi))
    out.update(reaction_reactants(candidate.get("reaction_smiles")))
    return {x for x in out if x}


def route_candidate_best_ranks(
    route: dict[str, Any],
    gt_rxns: list[str],
    gt_reactant_set: set[str],
) -> tuple[int | None, int | None]:
    gt_rxn_set = {x for x in gt_rxns if x}
    best_rxn_rank = None
    best_reactant_rank = None
    for step in route.get("steps", []):
        pool = (step.get("candidate_pool") or {}).get("top_candidates") or []
        for rank, cand in enumerate(pool, 1):
            cand_rxn = canonical_reaction(cand.get("reaction_smiles"))
            if cand_rxn and cand_rxn in gt_rxn_set:
                best_rxn_rank = rank if best_rxn_rank is None else min(best_rxn_rank, rank)
            if gt_reactant_set and (candidate_reactants(cand) & gt_reactant_set):
                best_reactant_rank = (
                    rank if best_reactant_rank is None else min(best_reactant_rank, rank)
                )
    return best_rxn_rank, best_reactant_rank


def recovery_bottleneck_labels(recovery: dict[str, Any]) -> list[str]:
    """Classify whether recovery failed in generation, selection, or composition.

    The labels are based only on exported route/candidate evidence. A
    candidate-pool miss means the benchmark artifact did not expose the exact
    GT reaction in any top candidate pool; it should be treated as the planner's
    observed generator frontier, not as a proof that the upstream model could
    never produce that reaction at a larger beam.
    """
    if not recovery:
        return ["no_recovery_metrics"]
    if recovery.get("exact_route_reaction_match_any"):
        return ["recovered_exact_route"]

    candidate_exact = bool(recovery.get("candidate_exact_reaction_in_pool"))
    selected_exact = bool(recovery.get("exact_reaction_in_route_pool"))
    candidate_reactant = bool(recovery.get("candidate_gt_reactant_in_pool"))
    selected_reactant = bool(recovery.get("gt_reactant_in_route_pool"))

    labels: list[str] = []
    if not candidate_exact:
        if candidate_reactant:
            labels.append("candidate_generator_reaction_detail_miss")
        else:
            labels.append("candidate_generator_reactant_miss")
    elif not selected_exact:
        labels.append("selector_missed_exact_candidate")
    elif selected_exact:
        labels.append("route_composition_or_order_miss")

    if candidate_reactant and not selected_reactant:
        labels.append("selector_missed_gt_reactant_candidate")
    return labels or ["unclassified_recovery_miss"]


def primary_recovery_bottleneck(recovery: dict[str, Any]) -> str:
    labels = recovery_bottleneck_labels(recovery)
    priority = [
        "recovered_exact_route",
        "candidate_generator_reactant_miss",
        "candidate_generator_reaction_detail_miss",
        "selector_missed_exact_candidate",
        "route_composition_or_order_miss",
        "selector_missed_gt_reactant_candidate",
        "unclassified_recovery_miss",
        "no_recovery_metrics",
    ]
    for label in priority:
        if label in labels:
            return label
    return labels[0] if labels else "unclassified_recovery_miss"


def route_recovery_metrics(route: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    pred_rxns = route_reaction_keys(route)
    cand_rxns = route_candidate_reaction_keys(route)
    gt_rxns = gt_reaction_keys(entry)
    pred_types = [step.get("reaction_type") or "" for step in route.get("steps", [])]
    gt_types = [step.get("transformation") or "" for step in entry.get("gt_route", [])]
    gt_rxn_counter = Counter(gt_rxns)
    pred_rxn_counter = Counter(pred_rxns)
    cand_rxn_counter = Counter(cand_rxns)
    exact_reaction_hits = sum((gt_rxn_counter & pred_rxn_counter).values())
    candidate_reaction_hits = sum((gt_rxn_counter & cand_rxn_counter).values())
    gt_reactant_set = gt_reactants(entry)
    pred_reactant_set = route_reactants(route)
    cand_reactant_set = route_candidate_reactants(route)
    gt_reactant_hits = sorted(gt_reactant_set & pred_reactant_set)
    candidate_gt_reactant_hits = sorted(gt_reactant_set & cand_reactant_set)
    candidate_rxn_rank, candidate_reactant_rank = route_candidate_best_ranks(
        route,
        gt_rxns,
        gt_reactant_set,
    )

    ordered_rxn_fraction = _best_ordered_fraction(pred_rxns, gt_rxns)
    ordered_type_fraction = _best_ordered_fraction(pred_types, gt_types)
    reaction_edit_distance = _best_edit_distance(pred_rxns, gt_rxns)
    type_edit_distance = _best_edit_distance(pred_types, gt_types)

    return {
        "gt_n_reactions": len(gt_rxns),
        "pred_n_reactions": len(pred_rxns),
        "exact_reaction_hits": exact_reaction_hits,
        "exact_reaction_fraction": exact_reaction_hits / len(gt_rxns) if gt_rxns else None,
        "exact_route_reaction_match": bool(gt_rxns and ordered_rxn_fraction == 1.0),
        "candidate_exact_reaction_hits": candidate_reaction_hits,
        "candidate_exact_reaction_fraction": candidate_reaction_hits / len(gt_rxns) if gt_rxns else None,
        "candidate_exact_reaction_hit": candidate_reaction_hits > 0,
        "candidate_exact_reaction_best_rank": candidate_rxn_rank,
        "ordered_reaction_match_fraction": ordered_rxn_fraction,
        "ordered_type_match_fraction": ordered_type_fraction,
        "reaction_edit_distance": reaction_edit_distance,
        "type_edit_distance": type_edit_distance,
        "gt_reactant_hits": gt_reactant_hits,
        "gt_reactant_hit": bool(gt_reactant_hits),
        "gt_reactant_hit_count": len(gt_reactant_hits),
        "candidate_gt_reactant_hits": candidate_gt_reactant_hits,
        "candidate_gt_reactant_hit": bool(candidate_gt_reactant_hits),
        "candidate_gt_reactant_hit_count": len(candidate_gt_reactant_hits),
        "candidate_gt_reactant_best_rank": candidate_reactant_rank,
    }


def target_recovery_metrics(routes: list[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any]:
    per_route = [route_recovery_metrics(route, entry) for route in routes]

    def first_rank(predicate) -> int | None:
        for idx, row in enumerate(per_route, 1):
            if predicate(row):
                return idx
        return None

    exact_rxn_rank = first_rank(lambda row: (row.get("exact_reaction_hits") or 0) > 0)
    exact_route_rank = first_rank(lambda row: bool(row.get("exact_route_reaction_match")))
    gt_reactant_rank = first_rank(lambda row: bool(row.get("gt_reactant_hit")))
    candidate_rxn_rank = first_rank(lambda row: bool(row.get("candidate_exact_reaction_hit")))
    candidate_reactant_rank = first_rank(lambda row: bool(row.get("candidate_gt_reactant_hit")))
    edit_values = [row["reaction_edit_distance"] for row in per_route if row["reaction_edit_distance"] is not None]
    type_edit_values = [row["type_edit_distance"] for row in per_route if row["type_edit_distance"] is not None]
    frac_values = [row["exact_reaction_fraction"] for row in per_route if row["exact_reaction_fraction"] is not None]
    candidate_rxn_candidate_ranks = [
        row["candidate_exact_reaction_best_rank"]
        for row in per_route
        if row["candidate_exact_reaction_best_rank"] is not None
    ]
    candidate_reactant_candidate_ranks = [
        row["candidate_gt_reactant_best_rank"]
        for row in per_route
        if row["candidate_gt_reactant_best_rank"] is not None
    ]

    result = {
        "exact_reaction_in_route_pool": exact_rxn_rank is not None,
        "exact_reaction_first_rank": exact_rxn_rank,
        "exact_route_reaction_match_any": exact_route_rank is not None,
        "exact_route_reaction_first_rank": exact_route_rank,
        "gt_reactant_in_route_pool": gt_reactant_rank is not None,
        "gt_reactant_first_rank": gt_reactant_rank,
        "candidate_exact_reaction_in_pool": candidate_rxn_rank is not None,
        "candidate_exact_reaction_first_route_rank": candidate_rxn_rank,
        "candidate_exact_reaction_first_rank": candidate_rxn_rank,
        "candidate_exact_reaction_best_candidate_rank": (
            min(candidate_rxn_candidate_ranks) if candidate_rxn_candidate_ranks else None
        ),
        "candidate_gt_reactant_in_pool": candidate_reactant_rank is not None,
        "candidate_gt_reactant_first_route_rank": candidate_reactant_rank,
        "candidate_gt_reactant_first_rank": candidate_reactant_rank,
        "candidate_gt_reactant_best_candidate_rank": (
            min(candidate_reactant_candidate_ranks) if candidate_reactant_candidate_ranks else None
        ),
        "best_exact_reaction_fraction": max(frac_values) if frac_values else None,
        "best_reaction_edit_distance": min(edit_values) if edit_values else None,
        "best_type_edit_distance": min(type_edit_values) if type_edit_values else None,
        "per_route": per_route,
    }
    result["recovery_bottleneck_labels"] = recovery_bottleneck_labels(result)
    result["recovery_bottleneck"] = primary_recovery_bottleneck(result)
    return result
