"""Assemble independently produced one-step consensuses into an advisory route graph."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.routes.overlay import build_route_hypergraph_v2_overlay


RDLogger.DisableLog("rdApp.*")

EXPANSION_SCHEMA = "route_consensus_expansion.v1"
GRAPH_SCHEMA = "route_consensus_graph.v1"
MOLECULE_SCHEMA = "route_consensus_molecule.v1"
STEP_SCHEMA = "route_consensus_step.v1"
ROUTE_SCHEMA = "route_consensus_route_hypothesis.v1"


def make_route_consensus_expansion(
    consensus: Mapping[str, Any],
    *,
    requested_product_smiles: str,
    consensus_ref: str = "",
    agent_run_ref: str = "",
    depth: int = 0,
) -> dict[str, Any]:
    product = _canonical_smiles(requested_product_smiles)
    return {
        "schema_version": EXPANSION_SCHEMA,
        "expansion_id": _stable_id("expansion", product, consensus_ref, agent_run_ref),
        "requested_product_smiles": product or str(requested_product_smiles or ""),
        "product_node_id": _molecule_id(product) if product else "",
        "depth": max(0, int(depth)),
        "consensus_ref": str(consensus_ref or ""),
        "agent_run_ref": str(agent_run_ref or ""),
        "route_consensus": dict(consensus),
        "semantics": _advisory_semantics(),
    }


def validate_route_consensus_expansion(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["expansion_not_object"]
    reasons: list[str] = []
    if payload.get("schema_version") != EXPANSION_SCHEMA:
        reasons.append("invalid_expansion_schema")
    requested = _canonical_smiles(payload.get("requested_product_smiles"))
    if not requested:
        reasons.append("invalid_expansion_product_smiles")
    consensus = payload.get("route_consensus")
    if not isinstance(consensus, Mapping) or consensus.get("schema_version") != "route_consensus.v1":
        reasons.append("invalid_expansion_consensus")
        return sorted(set(reasons))
    consensus_target = _canonical_smiles(consensus.get("target_smiles"))
    if not consensus_target or (requested and consensus_target != requested):
        reasons.append("expansion_consensus_target_mismatch")
    for index, proposal in enumerate(consensus.get("proposals") or []):
        if not isinstance(proposal, Mapping):
            reasons.append(f"expansion_proposal:{index}:not_object")
            continue
        if proposal.get("schema_version") != "route_consensus_proposal.v1":
            reasons.append(f"expansion_proposal:{index}:invalid_schema")
        product = _canonical_smiles(proposal.get("product_smiles"))
        precursors = [_canonical_smiles(value) for value in proposal.get("precursor_smiles") or []]
        if not product or product != requested:
            reasons.append(f"expansion_proposal:{index}:product_mismatch")
        if not precursors or any(not value for value in precursors):
            reasons.append(f"expansion_proposal:{index}:invalid_precursors")
        if proposal.get("no_solved_claim") is not True or proposal.get("not_parent_route_proof") is not True:
            reasons.append(f"expansion_proposal:{index}:unsafe_semantics")
    return sorted(set(reasons))


def assemble_route_consensus_graph(
    expansions: Iterable[Mapping[str, Any]],
    *,
    case_id: str,
    target_smiles: str,
    max_depth: int = 4,
    max_route_hypotheses: int = 24,
    max_graph_steps: int = 256,
) -> dict[str, Any]:
    max_depth = max(1, int(max_depth))
    max_route_hypotheses = max(1, int(max_route_hypotheses))
    max_graph_steps = max(1, int(max_graph_steps))
    target = _canonical_smiles(target_smiles)
    root_node_id = _molecule_id(target) if target else ""
    accepted_expansions: list[dict[str, Any]] = []
    rejected_inputs: list[dict[str, Any]] = []
    step_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    truncated_steps = False

    for index, raw in enumerate(expansions):
        expansion = dict(raw) if isinstance(raw, Mapping) else {}
        reasons = validate_route_consensus_expansion(expansion)
        if reasons:
            rejected_inputs.append({"index": index, "reasons": reasons})
            continue
        accepted_expansions.append(_expansion_summary(expansion))
        for proposal in (expansion.get("route_consensus") or {}).get("proposals") or []:
            if len(step_groups) >= max_graph_steps:
                truncated_steps = True
                break
            proposal_dict = dict(proposal)
            signature = _step_signature(
                proposal_dict.get("product_smiles"),
                proposal_dict.get("precursor_smiles") or [],
            )
            if signature:
                step_groups[signature].append((proposal_dict, expansion))

    all_steps = [_merge_step(signature, rows) for signature, rows in sorted(step_groups.items())]
    steps_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in all_steps:
        steps_by_product[str(step["product_node_id"])].append(step)
    for rows in steps_by_product.values():
        rows.sort(key=lambda row: (-float(row.get("rank_score") or 0.0), str(row["step_id"])))

    reachable_nodes, reachable_steps, min_depth = _reachable_subgraph(
        root_node_id,
        steps_by_product,
        max_depth=max_depth,
    )
    steps = [step for step in all_steps if step["step_id"] in reachable_steps]
    nodes = _graph_nodes(reachable_nodes, steps, min_depth=min_depth, expanded_products=set(steps_by_product))
    conflicts = _graph_conflicts(steps)
    conflict_ids_by_step: dict[str, list[str]] = defaultdict(list)
    for conflict in conflicts:
        conflict_ids_by_step[str(conflict.get("step_id") or "")].append(str(conflict["conflict_id"]))
    for step in steps:
        step["conflict_ids"] = conflict_ids_by_step.get(str(step["step_id"]), [])
    alternatives = _alternative_sets(steps)
    cycles: list[dict[str, Any]] = []
    hypotheses, routes_truncated = _enumerate_routes(
        root_node_id,
        steps_by_product,
        nodes={row["node_id"]: row for row in nodes},
        max_depth=max_depth,
        max_routes=max_route_hypotheses,
        cycles=cycles,
        conflict_ids_by_step=conflict_ids_by_step,
    )

    graph = {
        "schema_version": GRAPH_SCHEMA,
        "case_id": str(case_id or ""),
        "target_smiles": target or str(target_smiles or ""),
        "root_node_id": root_node_id,
        "has_hypotheses": bool(hypotheses),
        "limits": {
            "max_depth": max_depth,
            "max_route_hypotheses": max_route_hypotheses,
            "max_graph_steps": max_graph_steps,
        },
        "nodes": sorted(nodes, key=lambda row: (int(row.get("min_depth") or 0), str(row["node_id"]))),
        "steps": sorted(steps, key=lambda row: str(row["step_id"])),
        "alternative_sets": alternatives,
        "conflicts": conflicts,
        "cycles": _dedupe_dicts(cycles, key="cycle_id"),
        "route_hypotheses": hypotheses,
        "input_expansions": accepted_expansions,
        "rejected_inputs": rejected_inputs,
        "truncation": {
            "graph_steps_truncated": truncated_steps,
            "route_hypotheses_truncated": routes_truncated,
        },
        "semantics": {
            **_advisory_semantics(),
            "direction": "retrosynthetic_product_to_precursor_hyperedge",
            "solved": False,
            "executable": False,
            "not_parent_route_proof": True,
        },
    }
    overlay = build_route_hypergraph_v2_overlay(graph)
    graph["v2_overlay"] = overlay
    # Kept at the graph boundary as a convenient read-only index for v1
    # consumers; the content-addressed records live in ``v2_overlay``.
    graph["route_neighborhoods"] = list(overlay.get("route_neighborhoods") or [])
    return graph


def select_route_consensus_frontier(
    graph: Mapping[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    nodes = {str(row.get("node_id") or ""): dict(row) for row in graph.get("nodes") or [] if isinstance(row, Mapping)}
    parent_steps: dict[str, list[str]] = defaultdict(list)
    step_scores: dict[str, float] = {}
    for step in graph.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("step_id") or "")
        step_scores[step_id] = float(step.get("rank_score") or 0.0)
        for node_id in step.get("precursor_node_ids") or []:
            parent_steps[str(node_id)].append(step_id)
    frontier: dict[str, dict[str, Any]] = {}
    max_depth = int((graph.get("limits") or {}).get("max_depth") or 1)
    for route in graph.get("route_hypotheses") or []:
        if not isinstance(route, Mapping):
            continue
        for row in route.get("frontier") or []:
            if not isinstance(row, Mapping) or row.get("reason") != "unexpanded":
                continue
            node_id = str(row.get("node_id") or "")
            node = nodes.get(node_id) or {}
            depth = int(row.get("depth") or node.get("min_depth") or 0)
            if not node_id or depth >= max_depth:
                continue
            refs = sorted(set(parent_steps.get(node_id) or []))
            priority = max((step_scores.get(step_id, 0.0) for step_id in refs), default=0.0)
            frontier[node_id] = {
                "node_id": node_id,
                "target_smiles": str(node.get("smiles") or ""),
                "depth": depth,
                "parent_step_ids": refs,
                "reason": "unexpanded",
                "priority_score": round(priority, 4),
            }
    rows = sorted(frontier.values(), key=lambda row: (-float(row["priority_score"]), int(row["depth"]), row["node_id"]))
    return rows[: max(0, int(limit))]


def _merge_step(signature: str, rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    proposals = [row[0] for row in rows]
    expansions = [row[1] for row in rows]
    first = proposals[0]
    product = _canonical_smiles(first.get("product_smiles"))
    precursors = sorted({_canonical_smiles(value) for value in first.get("precursor_smiles") or []})
    source_records = _dedupe_records(
        record
        for proposal in proposals
        for record in proposal.get("source_records") or []
        if isinstance(record, Mapping)
    )
    conditions = _dedupe_text(value for proposal in proposals for value in proposal.get("conditions") or [])
    catalysts = _dedupe_text(value for proposal in proposals for value in proposal.get("catalysts") or [])
    enzymes = _dedupe_text(value for proposal in proposals for value in proposal.get("enzymes") or [])
    condition_support = _dedupe_records(
        record
        for proposal in proposals
        for record in proposal.get("condition_support") or []
        if isinstance(record, Mapping)
    )
    condition_conflicts = _dedupe_records(
        record
        for proposal in proposals
        for record in proposal.get("condition_conflicts") or []
        if isinstance(record, Mapping)
    )
    return {
        "schema_version": STEP_SCHEMA,
        "step_id": _stable_id("step", signature),
        "signature": signature,
        "product_node_id": _molecule_id(product),
        "precursor_node_ids": [_molecule_id(value) for value in precursors],
        "product_smiles": product,
        "precursor_smiles": precursors,
        "proposal_ids": _dedupe_text(str(row.get("consensus_id") or "") for row in proposals),
        "origin_expansion_ids": _dedupe_text(str(row.get("expansion_id") or "") for row in expansions),
        "reaction_family": str(max(proposals, key=lambda row: float(row.get("rank_score") or 0.0)).get("reaction_family") or "unspecified"),
        "reaction_families": _dedupe_text(value for row in proposals for value in row.get("reaction_families") or []),
        "rationales": _dedupe_text(value for row in proposals for value in row.get("rationales") or []),
        "source_channels": _dedupe_text(value for row in proposals for value in row.get("source_channels") or []),
        "source_records": source_records,
        "independent_support_groups": _dedupe_text(value for row in proposals for value in row.get("independent_support_groups") or []),
        "source_refs": _dedupe_text(value for row in proposals for value in row.get("source_refs") or []),
        "evidence_refs": _dedupe_text(value for row in proposals for value in row.get("evidence_refs") or []),
        "conditions": conditions,
        "catalysts": catalysts,
        "enzymes": enzymes,
        "condition_support": condition_support,
        "condition_conflicts": condition_conflicts,
        "conflict_ids": [],
        "rank_score": round(max(float(row.get("rank_score") or 0.0) for row in proposals), 4),
        "confidence": str(max(proposals, key=lambda row: float(row.get("rank_score") or 0.0)).get("confidence") or "low"),
        "limitations": _dedupe_text(value for row in proposals for value in row.get("limitations") or []),
        "required_validation": _dedupe_text(value for row in proposals for value in row.get("required_validation") or []),
        "advisory_only": True,
        "solved": False,
        "executable": False,
        "not_parent_route_proof": True,
    }


def _reachable_subgraph(
    root_node_id: str,
    steps_by_product: Mapping[str, list[dict[str, Any]]],
    *,
    max_depth: int,
) -> tuple[set[str], set[str], dict[str, int]]:
    nodes = {root_node_id} if root_node_id else set()
    step_ids: set[str] = set()
    min_depth = {root_node_id: 0} if root_node_id else {}
    queue: deque[tuple[str, int]] = deque([(root_node_id, 0)] if root_node_id else [])
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for step in steps_by_product.get(node_id) or []:
            step_ids.add(str(step["step_id"]))
            for precursor_id in step.get("precursor_node_ids") or []:
                precursor_id = str(precursor_id)
                nodes.add(precursor_id)
                next_depth = depth + 1
                if next_depth < min_depth.get(precursor_id, 1_000_000):
                    min_depth[precursor_id] = next_depth
                    queue.append((precursor_id, next_depth))
    return nodes, step_ids, min_depth


def _graph_nodes(
    reachable_nodes: set[str],
    steps: list[dict[str, Any]],
    *,
    min_depth: Mapping[str, int],
    expanded_products: set[str],
) -> list[dict[str, Any]]:
    smiles_by_id: dict[str, str] = {}
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        product_id = str(step["product_node_id"])
        smiles_by_id[product_id] = str(step["product_smiles"])
        outgoing[product_id].append(str(step["step_id"]))
        for node_id, smiles in zip(step.get("precursor_node_ids") or [], step.get("precursor_smiles") or []):
            node_id = str(node_id)
            smiles_by_id[node_id] = str(smiles)
            incoming[node_id].append(str(step["step_id"]))
    return [
        {
            "schema_version": MOLECULE_SCHEMA,
            "node_id": node_id,
            "smiles": smiles_by_id.get(node_id, ""),
            "canonical_isomeric_smiles": smiles_by_id.get(node_id, ""),
            "min_depth": int(min_depth.get(node_id, 0)),
            "expansion_status": "expanded" if node_id in expanded_products else "unexpanded",
            "outgoing_step_ids": sorted(outgoing.get(node_id) or []),
            "incoming_step_ids": sorted(incoming.get(node_id) or []),
        }
        for node_id in reachable_nodes
    ]


def _enumerate_routes(
    root_node_id: str,
    steps_by_product: Mapping[str, list[dict[str, Any]]],
    *,
    nodes: Mapping[str, dict[str, Any]],
    max_depth: int,
    max_routes: int,
    cycles: list[dict[str, Any]],
    conflict_ids_by_step: Mapping[str, list[str]],
) -> tuple[list[dict[str, Any]], bool]:
    truncated = False

    def expand(node_id: str, depth: int, ancestry: tuple[str, ...]) -> list[dict[str, Any]]:
        nonlocal truncated
        if depth >= max_depth:
            return [_empty_plan(node_id, depth, "depth_limit")]
        choices = steps_by_product.get(node_id) or []
        if not choices:
            return [_empty_plan(node_id, depth, "unexpanded")]
        plans: list[dict[str, Any]] = []
        for step in choices:
            precursor_ids = [str(value) for value in step.get("precursor_node_ids") or []]
            repeated = [value for value in precursor_ids if value in ancestry or value == node_id]
            if repeated:
                cycle = {
                    "cycle_id": _stable_id("cycle", step["step_id"], *sorted(repeated)),
                    "step_id": str(step["step_id"]),
                    "ancestry_node_ids": list(ancestry) + [node_id],
                    "repeated_node_ids": sorted(set(repeated)),
                }
                cycles.append(cycle)
                continue
            child_options = [expand(value, depth + 1, (*ancestry, node_id)) for value in precursor_ids]
            for combination in itertools.product(*child_options):
                plan = {
                    "steps": [str(step["step_id"])],
                    "nodes": [node_id, *precursor_ids],
                    "frontier": [],
                    "cycle_cut_step_ids": [],
                    "score_values": [float(step.get("rank_score") or 0.0)],
                }
                for child in combination:
                    plan["steps"].extend(child["steps"])
                    plan["nodes"].extend(child["nodes"])
                    plan["frontier"].extend(child["frontier"])
                    plan["cycle_cut_step_ids"].extend(child["cycle_cut_step_ids"])
                    plan["score_values"].extend(child["score_values"])
                plans.append(plan)
                if len(plans) >= max_routes:
                    truncated = True
                    return plans
        if not plans:
            return [_empty_plan(node_id, depth, "cycle_cut")]
        return plans

    raw_plans = expand(root_node_id, 0, ()) if root_node_id else []
    step_by_id = {str(step["step_id"]): step for rows in steps_by_product.values() for step in rows}
    routes: list[dict[str, Any]] = []
    for plan in raw_plans[:max_routes]:
        retro_steps = _dedupe_ordered_text(plan["steps"])
        forward_steps = list(reversed(retro_steps))
        dependencies = _forward_dependencies(retro_steps, step_by_id)
        conflict_ids = _dedupe_text(
            conflict_id
            for step_id in retro_steps
            for conflict_id in conflict_ids_by_step.get(step_id) or []
        )
        score_values = [max(1e-6, float(value)) for value in plan["score_values"]]
        score = math.prod(score_values) ** (1.0 / len(score_values)) if score_values else 0.0
        direct_ref_steps = sum(
            1
            for step_id in retro_steps
            if (step_by_id.get(step_id) or {}).get("source_refs")
            or (step_by_id.get(step_id) or {}).get("evidence_refs")
        )
        route_id = _stable_id("route", *retro_steps, json.dumps(plan["frontier"], sort_keys=True))
        routes.append(
            {
                "schema_version": ROUTE_SCHEMA,
                "route_id": route_id,
                "root_node_id": root_node_id,
                "retrosynthetic_step_ids": retro_steps,
                "forward_step_ids": forward_steps,
                "forward_dependencies": dependencies,
                "node_ids": _dedupe_ordered_text(plan["nodes"]),
                "max_depth_reached": max((int(row.get("depth") or 0) for row in plan["frontier"]), default=len(retro_steps)),
                "frontier": _dedupe_dicts(plan["frontier"], key="node_id"),
                "cycle_cut_step_ids": _dedupe_text(plan["cycle_cut_step_ids"]),
                "conflict_ids": conflict_ids,
                "source_coverage": {
                    "step_count": len(retro_steps),
                    "steps_with_direct_refs": direct_ref_steps,
                    "conflicted_steps": sum(1 for step_id in retro_steps if conflict_ids_by_step.get(step_id)),
                },
                "rank_score": round(score, 4),
                "advisory_only": True,
                "solved": False,
                "executable": False,
                "not_parent_route_proof": True,
            }
        )
    routes.sort(key=lambda row: (-float(row["rank_score"]), str(row["route_id"])))
    return routes, truncated or len(raw_plans) > max_routes


def _empty_plan(node_id: str, depth: int, reason: str) -> dict[str, Any]:
    return {
        "steps": [],
        "nodes": [node_id],
        "frontier": [{"node_id": node_id, "depth": depth, "reason": reason}],
        "cycle_cut_step_ids": [],
        "score_values": [],
    }


def _forward_dependencies(step_ids: list[str], step_by_id: Mapping[str, dict[str, Any]]) -> list[dict[str, str]]:
    dependencies: list[dict[str, str]] = []
    for consumer_id in step_ids:
        consumer = step_by_id.get(consumer_id) or {}
        for precursor_id in consumer.get("precursor_node_ids") or []:
            producer = next(
                (
                    producer_id
                    for producer_id in step_ids
                    if producer_id != consumer_id
                    and str((step_by_id.get(producer_id) or {}).get("product_node_id") or "") == str(precursor_id)
                ),
                "",
            )
            if producer:
                dependencies.append(
                    {
                        "producer_step_id": producer,
                        "consumer_step_id": consumer_id,
                        "via_node_id": str(precursor_id),
                    }
                )
    return _dedupe_dicts(dependencies, key=lambda row: f"{row['producer_step_id']}|{row['consumer_step_id']}|{row['via_node_id']}")


def _alternative_sets(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        grouped[str(step["product_node_id"])].append(str(step["step_id"]))
    return [
        {
            "alternative_set_id": _stable_id("alternative", product_id),
            "product_node_id": product_id,
            "step_ids": sorted(step_ids),
            "kind": "competing_disconnections",
            "is_contradiction": False,
            "requires_selection": True,
        }
        for product_id, step_ids in sorted(grouped.items())
        if len(step_ids) > 1
    ]


def _graph_conflicts(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for step in steps:
        step_id = str(step["step_id"])
        for raw in step.get("condition_conflicts") or []:
            if not isinstance(raw, Mapping):
                continue
            field = str(raw.get("field") or "condition")
            values = _dedupe_text(raw.get("values") or [])
            conflicts.append(
                {
                    "conflict_id": _stable_id("conflict", step_id, field, *values),
                    "scope": "step",
                    "step_id": step_id,
                    "kind": "attribute_disagreement",
                    "field": field,
                    "values": values,
                    "requires_review": True,
                }
            )
        families = _dedupe_text(step.get("reaction_families") or [])
        if len(families) > 1:
            conflicts.append(
                {
                    "conflict_id": _stable_id("conflict", step_id, "reaction_family", *families),
                    "scope": "step",
                    "step_id": step_id,
                    "kind": "attribute_disagreement",
                    "field": "reaction_family",
                    "values": families,
                    "requires_review": True,
                }
            )
    return _dedupe_dicts(conflicts, key="conflict_id")


def _expansion_summary(expansion: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EXPANSION_SCHEMA,
        "expansion_id": str(expansion.get("expansion_id") or ""),
        "requested_product_smiles": str(expansion.get("requested_product_smiles") or ""),
        "product_node_id": str(expansion.get("product_node_id") or ""),
        "depth": int(expansion.get("depth") or 0),
        "consensus_ref": str(expansion.get("consensus_ref") or ""),
        "agent_run_ref": str(expansion.get("agent_run_ref") or ""),
    }


def _step_signature(product: Any, precursors: Iterable[Any]) -> str:
    product_value = _canonical_smiles(product)
    precursor_values = sorted({_canonical_smiles(value) for value in precursors if _canonical_smiles(value)})
    if not product_value or not precursor_values:
        return ""
    return f"{product_value}<-{'.'.join(precursor_values)}"


def _molecule_id(smiles: str) -> str:
    return _stable_id("mol", f"smiles:{smiles}")


def _canonical_smiles(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _stable_id(prefix: str, *parts: Any) -> str:
    text = "|".join(str(part or "") for part in parts)
    return f"{prefix}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _dedupe_text(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _dedupe_ordered_text(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _dedupe_records(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for value in values:
        row = dict(value)
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _dedupe_dicts(
    values: Iterable[Mapping[str, Any]],
    *,
    key: str | Any,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for value in values:
        row = dict(value)
        identity = str(row.get(key) or "") if isinstance(key, str) else str(key(row))
        if not identity or identity in seen:
            continue
        seen.add(identity)
        rows.append(row)
    return rows


def _advisory_semantics() -> dict[str, Any]:
    return {
        "advisory_only": True,
        "no_solved_claim": True,
        "deterministic_parent_proof_required": True,
    }
