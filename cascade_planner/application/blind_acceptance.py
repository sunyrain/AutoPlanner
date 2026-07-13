"""Compile explicit B0-B5 gates for a target-only retrosynthesis run."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.canonical_identity import (
    molecule_identity,
    reaction_edge_identity,
)
from cascade_planner.application.proof_policy import stock_boundary_matches
from cascade_planner.application.reaction_proof_versions import (
    active_reaction_proofs,
    compile_reaction_proof_version_audit,
)


BLIND_ACCEPTANCE_REPORT_SCHEMA = "blind_retrosynthesis_acceptance_report.v1"


def compile_blind_acceptance_report(
    *,
    preflight: Mapping[str, Any],
    director_outcomes: Iterable[Mapping[str, Any]],
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure route generation, evidence, stock, and acceptance independently."""

    outcomes = [dict(value) for value in director_outcomes if isinstance(value, Mapping)]
    skeletons = _expected_skeletons(outcomes, graph=graph)
    minimum_routes = int(
        dict(dict(preflight.get("case") or {}).get("acceptance") or {}).get(
            "minimum_complete_routes"
        )
        or 2
    )
    minimum_sources = int(
        dict(portfolio.get("proof_policy") or {}).get("minimum_independent_source_groups")
        or 1
    )
    stock_boundary = str(
        dict(portfolio.get("proof_policy") or {}).get("stock_boundary")
        or "procurement"
    )
    edges = dict(graph.get("edges") or {})
    molecules = dict(graph.get("molecules") or {})
    observations = dict(graph.get("stock_observations") or {})
    route_rows: list[dict[str, Any]] = []
    for skeleton in skeletons:
        generated_edge_ids = list(skeleton["edge_ids"])
        edge_ids, stock_pruned_edge_ids = _target_reachable_edges_until_stock_boundary(
            generated_edge_ids,
            edges=edges,
            target_molecule_id=str(graph.get("target_molecule_id") or ""),
            molecules=molecules,
            observations=observations,
            required_boundary=stock_boundary,
        )
        materialized = bool(edge_ids) and all(edge_id in edges for edge_id in edge_ids)
        validated = materialized and all(_edge_validated(edges[edge_id]) for edge_id in edge_ids)
        evidence_closed = validated and all(
            len(
                {
                    str(value)
                    for value in edges[edge_id].get("independent_source_groups") or []
                    if str(value)
                }
            )
            >= minimum_sources
            and not _edge_has_unresolved_source_conflict(
                edges[edge_id],
                graph=graph,
            )
            for edge_id in edge_ids
        )
        leaf_ids = _leaf_molecule_ids(
            skeleton,
            edge_ids=edge_ids,
            edges=edges,
        )
        stock_closed = bool(leaf_ids) and all(
            _leaf_stock_closed(
                molecule_id,
                molecules=molecules,
                observations=observations,
                required_boundary=stock_boundary,
            )
            for molecule_id in leaf_ids
        )
        route_rows.append(
            {
                **skeleton,
                "generated_edge_ids": sorted(set(generated_edge_ids)),
                "edge_ids": sorted(set(edge_ids)),
                "pruned_at_stock_boundary_edge_ids": stock_pruned_edge_ids,
                "materialized": materialized,
                "reaction_validated": validated,
                "evidence_closed": evidence_closed,
                "stock_closed": stock_closed,
                "leaf_molecule_ids": leaf_ids,
            }
        )
    distinct_rows = _distinct_edge_sets(route_rows)
    selected_routes = [
        dict(value)
        for value in portfolio.get("selected_routes") or []
        if isinstance(value, Mapping)
    ]
    canonical_evidence_closed = sum(
        bool(route.get("edge_ids"))
        and all(
            _edge_validated(dict(edges.get(str(edge_id)) or {}))
            and len(
                {
                    str(value)
                    for value in dict(edges.get(str(edge_id)) or {}).get(
                        "independent_source_groups"
                    )
                    or []
                    if str(value)
                }
            )
            >= minimum_sources
            and not _edge_has_unresolved_source_conflict(
                dict(edges.get(str(edge_id)) or {}),
                graph=graph,
            )
            for edge_id in route.get("edge_ids") or []
        )
        for route in selected_routes
    )
    canonical_stock_closed = sum(
        bool(route.get("leaf_molecule_ids"))
        and route.get("all_leaves_stock_closed") is True
        for route in selected_routes
    )
    counts = {
        "target_rooted_distinct_skeletons": len(distinct_rows),
        "materialized_skeletons": sum(row["materialized"] for row in distinct_rows),
        "reaction_validated_skeletons": sum(
            row["reaction_validated"] for row in distinct_rows
        ),
        "evidence_closed_skeletons": canonical_evidence_closed,
        "stock_closed_skeletons": canonical_stock_closed,
        "canonical_evidence_closed_routes": canonical_evidence_closed,
        "canonical_stock_closed_routes": canonical_stock_closed,
    }
    gates = {
        "B0_blind_input": preflight.get("accepted") is True,
        "B1_global_multi_route": counts["target_rooted_distinct_skeletons"] >= minimum_routes,
        "B2_host_validated_routes": counts["reaction_validated_skeletons"] >= minimum_routes,
        "B3_exact_multi_source": counts["evidence_closed_skeletons"] >= minimum_routes,
        "B4_stock_boundary": counts["stock_closed_skeletons"] >= minimum_routes,
        "B5_configured_portfolio_acceptance": portfolio.get("accepted") is True,
    }
    ordered = list(gates)
    contiguous = "none"
    for key in ordered:
        if gates[key] is not True:
            break
        contiguous = key.split("_", 1)[0]
    report = {
        "schema_version": BLIND_ACCEPTANCE_REPORT_SCHEMA,
        "gates": gates,
        "highest_contiguous_gate": contiguous,
        "counts": counts,
        "minimum_routes": minimum_routes,
        "minimum_independent_source_groups": minimum_sources,
        "stock_boundary": stock_boundary,
        "routes": distinct_rows,
        "canonical_portfolio_accepted": portfolio.get("accepted") is True,
        "canonical_closeout": dict(portfolio.get("closeout") or {}),
        "false_closure_claim_count": 0,
        "reaction_proof_version_audit": compile_reaction_proof_version_audit(
            graph
        ),
        "semantics": {
            "gates_are_independent_measurements": True,
            "B2_is_not_evidence_grade": True,
            "B3_is_not_stock_grade": True,
            "B5_uses_configured_acceptance_policy": True,
            "only_canonical_portfolio_can_accept_campaign": True,
        },
    }
    report["content_sha256"] = _digest(report)
    return report


def _expected_skeletons(
    outcomes: list[dict[str, Any]],
    *,
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], dict[str, Any]] = {}
    edges = dict(graph.get("edges") or {})
    for outcome in outcomes:
        if outcome.get("status") != "accepted" or not isinstance(outcome.get("plan"), Mapping):
            continue
        plan = dict(outcome["plan"])
        family_targets = {
            str(row.get("route_family_id") or ""): str(row.get("target_smiles") or "")
            for row in plan.get("route_families") or []
            if isinstance(row, Mapping)
        }
        accepted_ids = {
            str(row.get("proposal_id") or "")
            for row in outcome.get("proposal_audits") or []
            if isinstance(row, Mapping) and row.get("accepted") is True
        }
        for raw in plan.get("multi_step_skeletons") or []:
            if not isinstance(raw, Mapping):
                continue
            skeleton = dict(raw)
            steps = [
                dict(step)
                for step in skeleton.get("steps") or []
                if isinstance(step, Mapping)
                and str(step.get("step_id") or "") in accepted_ids
            ]
            rejected_step_ids = sorted(
                str(step.get("step_id") or "")
                for step in skeleton.get("steps") or []
                if isinstance(step, Mapping)
                and str(step.get("step_id") or "") not in accepted_ids
            )
            family_id = str(skeleton.get("route_family_id") or "")
            if not steps or not _target_rooted_connected_steps(
                steps,
                target_smiles=family_targets.get(family_id, ""),
            ):
                continue
            edge_ids: list[str] = []
            edge_replacements: list[dict[str, str]] = []
            for step in steps:
                original_edge_id, audit = reaction_edge_identity(
                    step.get("product_smiles"),
                    step.get("precursor_smiles") or [],
                )
                if not original_edge_id or audit.get("accepted") is not True:
                    edge_ids = []
                    break
                edge_id = _resolved_edge_id(original_edge_id, edges=edges)
                edge_ids.append(edge_id)
                if edge_id != original_edge_id:
                    edge_replacements.append(
                        {
                            "original_edge_id": original_edge_id,
                            "replacement_edge_id": edge_id,
                            "reason": "host_product_grounded_repair",
                        }
                    )
            if not edge_ids:
                continue
            pruned_after_replacement: list[str] = []
            if edge_replacements:
                edge_ids, pruned_after_replacement = (
                    _target_reachable_edges_after_repair(
                        edge_ids,
                        edges=edges,
                        target_smiles=family_targets.get(family_id, ""),
                    )
                )
                if not edge_ids:
                    continue
            identity = (
                family_id,
                str(skeleton.get("skeleton_id") or ""),
            )
            values[identity] = {
                "plan_id": str(plan.get("plan_id") or ""),
                "mode": str(plan.get("mode") or ""),
                "route_family_id": identity[0],
                "skeleton_id": identity[1],
                "edge_ids": sorted(set(edge_ids)),
                "steps": steps,
                "pruned_rejected_tail_step_ids": rejected_step_ids,
                "pruned_after_replacement_edge_ids": pruned_after_replacement,
                "edge_replacements": edge_replacements,
            }
    return [values[key] for key in sorted(values)]


def _target_rooted_connected_steps(
    steps: list[dict[str, Any]],
    *,
    target_smiles: str,
) -> bool:
    target_id = molecule_identity(target_smiles)[0]
    products = [molecule_identity(step.get("product_smiles"))[0] for step in steps]
    precursor_ids = {
        molecule_identity(value)[0]
        for step in steps
        for value in step.get("precursor_smiles") or []
    }
    if not target_id or products.count(target_id) != 1 or target_id in precursor_ids:
        return False
    return all(
        product_id == target_id or product_id in precursor_ids
        for product_id in products
        if product_id
    )


def _resolved_edge_id(
    original_edge_id: str,
    *,
    edges: Mapping[str, Any],
) -> str:
    original = dict(edges.get(original_edge_id) or {})
    if original and _edge_validated(original):
        return original_edge_id
    repairs = sorted(
        str(edge_id)
        for edge_id, edge in edges.items()
        if any(
            isinstance(origin, Mapping)
            and origin.get("origin_kind") == "host_product_grounded_repair"
            and str(origin.get("origin_ref") or "") == original_edge_id
            for origin in dict(edge or {}).get("origin_records") or []
        )
    )
    if repairs:
        return sorted(
            repairs,
            key=lambda edge_id: (
                _edge_validated(dict(edges.get(edge_id) or {})),
                edge_id,
            ),
            reverse=True,
        )[0]
    return original_edge_id


def _edge_validated(edge: Mapping[str, Any]) -> bool:
    return any(
        isinstance(proof, Mapping) and proof.get("accepted") is True
        for proof in active_reaction_proofs(edge.get("reaction_proofs") or [])
    )


def _edge_has_unresolved_source_conflict(
    edge: Mapping[str, Any],
    *,
    graph: Mapping[str, Any],
) -> bool:
    edge_digest = str(edge.get("edge_digest") or "")
    record_ids = {str(value) for value in edge.get("exact_record_ids") or []}
    for raw in dict(graph.get("conflicts") or {}).values():
        if not isinstance(raw, Mapping) or raw.get("status") == "resolved":
            continue
        subject = str(raw.get("subject_id") or "")
        conflict_records = {str(value) for value in raw.get("record_ids") or []}
        if (edge_digest and edge_digest in subject) or record_ids & conflict_records:
            return True
    return False


def _target_reachable_edges_after_repair(
    edge_ids: list[str],
    *,
    edges: Mapping[str, Any],
    target_smiles: str,
) -> tuple[list[str], list[str]]:
    """Drop only tails made obsolete by an explicitly bound host repair."""

    target_id = molecule_identity(target_smiles)[0]
    if not target_id:
        return [], list(edge_ids)
    frontier = {target_id}
    remaining = list(edge_ids)
    selected: list[str] = []
    while remaining:
        match_index = next(
            (
                index
                for index, edge_id in enumerate(remaining)
                if str(dict(edges.get(edge_id) or {}).get("product_molecule_id") or "")
                in frontier
            ),
            None,
        )
        if match_index is None:
            break
        edge_id = remaining.pop(match_index)
        edge = dict(edges.get(edge_id) or {})
        product_id = str(edge.get("product_molecule_id") or "")
        frontier.discard(product_id)
        frontier.update(
            str(value)
            for value in edge.get("precursor_molecule_ids") or []
            if str(value)
        )
        selected.append(edge_id)
    return selected, sorted(remaining)


def _target_reachable_edges_until_stock_boundary(
    edge_ids: list[str],
    *,
    edges: Mapping[str, Any],
    target_molecule_id: str,
    molecules: Mapping[str, Any],
    observations: Mapping[str, Any],
    required_boundary: str,
) -> tuple[list[str], list[str]]:
    """Prune upstream chemistry only where a host-audited stock cut exists."""

    if not edge_ids:
        return [], []
    target_id = target_molecule_id
    if not target_id:
        products = {
            str(dict(edges.get(edge_id) or {}).get("product_molecule_id") or "")
            for edge_id in edge_ids
        }
        precursors = {
            str(value)
            for edge_id in edge_ids
            for value in dict(edges.get(edge_id) or {}).get("precursor_molecule_ids") or []
            if str(value)
        }
        roots = sorted(products - precursors)
        target_id = roots[0] if len(roots) == 1 else ""
    if not target_id:
        return list(edge_ids), []
    by_product: dict[str, list[str]] = {}
    for edge_id in edge_ids:
        product_id = str(dict(edges.get(edge_id) or {}).get("product_molecule_id") or "")
        if product_id:
            by_product.setdefault(product_id, []).append(edge_id)
    frontier = [target_id]
    visited: set[str] = set()
    selected: list[str] = []
    while frontier:
        molecule_id = frontier.pop(0)
        if molecule_id in visited:
            continue
        visited.add(molecule_id)
        if molecule_id != target_id and _leaf_stock_closed(
            molecule_id,
            molecules=molecules,
            observations=observations,
            required_boundary=required_boundary,
        ):
            continue
        for edge_id in sorted(by_product.get(molecule_id, [])):
            selected.append(edge_id)
            frontier.extend(
                str(value)
                for value in dict(edges.get(edge_id) or {}).get("precursor_molecule_ids") or []
                if str(value)
            )
    selected_set = set(selected)
    return selected, sorted(set(edge_ids) - selected_set)


def _leaf_molecule_ids(
    skeleton: Mapping[str, Any],
    *,
    edge_ids: Iterable[str] = (),
    edges: Mapping[str, Any] | None = None,
) -> list[str]:
    graph_edges = dict(edges or {})
    selected_edge_ids = list(edge_ids)
    if graph_edges and selected_edge_ids:
        products = {
            str(dict(graph_edges.get(edge_id) or {}).get("product_molecule_id") or "")
            for edge_id in selected_edge_ids
        }
        precursors = {
            str(value)
            for edge_id in selected_edge_ids
            for value in dict(graph_edges.get(edge_id) or {}).get(
                "precursor_molecule_ids"
            )
            or []
            if str(value)
        }
        return sorted(precursors - products)
    products = {
        molecule_id
        for step in skeleton.get("steps") or []
        if (molecule_id := molecule_identity(step.get("product_smiles"))[0])
    }
    precursors = {
        molecule_id
        for step in skeleton.get("steps") or []
        for value in step.get("precursor_smiles") or []
        if (molecule_id := molecule_identity(value)[0])
    }
    return sorted(precursors - products)


def _leaf_stock_closed(
    molecule_id: str,
    *,
    molecules: Mapping[str, Any],
    observations: Mapping[str, Any],
    required_boundary: str,
) -> bool:
    molecule = dict(molecules.get(molecule_id) or {})
    observation = dict(
        observations.get(str(molecule.get("active_stock_observation_id") or "")) or {}
    )
    return bool(
        observation.get("accepted") is True
        and stock_boundary_matches(observation, required=required_boundary)
    )


def _distinct_edge_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(sorted(str(value) for value in row.get("edge_ids") or []))
        if key and key not in values:
            values[key] = row
    return [values[key] for key in sorted(values)]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["BLIND_ACCEPTANCE_REPORT_SCHEMA", "compile_blind_acceptance_report"]
