"""Proof-aware, bounded read model for the retrosynthesis workbench.

The workbench never consumes the exploratory hypergraph directly.  This module
projects one canonical graph revision and one proof portfolio into a compact
read model, then computes entity deltas between revisions.  It owns no
scientific state and cannot promote a hypothesis or proof level.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping
from cascade_planner.application.product_profiles import (
    PRODUCT_PROFILE_ORDER,
)
from cascade_planner.application.route_workbench_fact_rows import (
    edge_row as _edge_row,
    molecule_row as _molecule_row,
)
from cascade_planner.application.route_workbench_closure import (
    declared_program_closure as _declared_program_closure,
)
from cascade_planner.application.route_workbench_inspectors import (
    PROOF_VECTOR_SCHEMA as PROOF_VECTOR_SCHEMA,
    edge_inspector as _edge_inspector,
    molecule_inspector as _molecule_inspector,
    route_inspector as _route_inspector,
)
from cascade_planner.application.route_workbench_planned_routes import (
    planned_route_rows as _planned_route_rows,
)
from cascade_planner.application.route_workbench_route_rows import (
    PROOF_VISUALS,
    closure_profile as _closure_profile,
    route_row as _route_row,
)
from cascade_planner.application.route_innovations import merge_route_innovations
ROUTE_WORKBENCH_SCHEMA = "retrosynthesis_route_workbench.v1"
ROUTE_WORKBENCH_DELTA_SCHEMA = "retrosynthesis_route_workbench_delta.v1"
MAX_VISIBLE_ROUTES = 5
MAX_VISIBLE_HYPOTHESES = 48
class RouteWorkbenchProjectionError(ValueError):
    """The graph/portfolio pair cannot form one authoritative UI revision."""
def compile_route_workbench(
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    *,
    campaign_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a canonical graph and proof portfolio into a bounded read model."""

    _validate_bindings(graph, portfolio)
    graph_revision = int(graph.get("revision") or 0)
    molecules = dict(graph.get("molecules") or {})
    edges = dict(graph.get("edges") or {})
    edge_proofs = dict(portfolio.get("edge_proofs") or {})
    leaf_proofs = dict(portfolio.get("leaf_proofs") or {})
    proof_policy = dict(portfolio.get("proof_policy") or {})
    stock_boundary = str(proof_policy.get("stock_boundary") or "unknown")
    selected = [
        dict(value)
        for value in portfolio.get("selected_routes") or []
        if isinstance(value, Mapping)
    ][:MAX_VISIBLE_ROUTES]
    planned_routes = _planned_route_rows(
        graph,
        selected_routes=selected,
    )
    route_closure = _declared_program_closure(graph)
    replacement_routes, replacement_records = _replacement_rows(
        portfolio,
        selected_routes=selected,
    )

    selected_edge_ids = {
        str(edge_id)
        for route in [*selected, *replacement_routes.values()]
        for edge_id in route.get("edge_ids") or []
        if str(edge_id) in edges
    }
    selected_edge_ids.update(
        str(step.get("edge_id") or "")
        for route in planned_routes.values()
        for step in route.get("steps") or []
        if str(step.get("edge_id") or "") in edges
    )
    selected_molecule_ids = {
        str(molecule_id)
        for edge_id in selected_edge_ids
        for molecule_id in (
            edges[edge_id].get("product_molecule_id"),
            *(edges[edge_id].get("precursor_molecule_ids") or []),
        )
        if str(molecule_id) in molecules
    }
    target_id = str(graph.get("target_molecule_id") or "")
    if target_id in molecules:
        selected_molecule_ids.add(target_id)

    molecule_rows = {
        molecule_id: _molecule_row(
            molecule_id,
            molecules[molecule_id],
            graph=graph,
            leaf_proof=leaf_proofs.get(molecule_id),
            target_id=target_id,
        )
        for molecule_id in sorted(selected_molecule_ids)
    }
    edge_rows = {
        edge_id: _edge_row(
            edge_id,
            edges[edge_id],
            proof=dict(edge_proofs.get(edge_id) or {}),
            graph=graph,
        )
        for edge_id in sorted(selected_edge_ids)
    }
    deficit_rows = [
        _copy_json(value)
        for value in portfolio.get("deficits") or []
        if isinstance(value, Mapping)
    ]
    route_rows = {
        str(route["route_id"]): _route_row(
            route,
            edge_rows=edge_rows,
            deficits=deficit_rows,
            stock_boundary=stock_boundary,
        )
        for route in selected
        if str(route.get("route_id") or "")
    }
    replacement_route_rows = {
        route_id: {
            **_route_row(
                route,
                edge_rows=edge_rows,
                deficits=deficit_rows,
                stock_boundary=stock_boundary,
            ),
            "listed": False,
            "replacement_projection": True,
            "underlying_route_id": str(route.get("underlying_route_id") or ""),
            "base_route_id": str(route.get("base_route_id") or ""),
        }
        for route_id, route in replacement_routes.items()
    }
    hypotheses = _hypothesis_rows(graph)
    modules = _module_rows(portfolio, selected_route_ids=set(route_rows))
    shared_intermediates = _shared_intermediate_rows(
        portfolio,
        molecules=molecules,
    )
    views = _views(route_rows, hypotheses)
    profile_counts = {
        profile: sum(
            bool(dict(route.get("acceptance_profiles") or {}).get(profile))
            for route in route_rows.values()
        )
        for profile in PRODUCT_PROFILE_ORDER
    }
    achieved_profile = next(
        (
            profile
            for profile in reversed(PRODUCT_PROFILE_ORDER)
            if profile_counts[profile] > 0
        ),
        "unresolved",
    )
    layout = _stable_layout(
        target_id=target_id,
        route_rows=route_rows,
        edge_rows=edge_rows,
    )
    payload = {
        "schema_version": ROUTE_WORKBENCH_SCHEMA,
        "run_id": str(graph.get("run_id") or ""),
        "target": {
            "molecule_id": target_id,
            "name": str(graph.get("target_name") or ""),
            "canonical_smiles": str(
                dict(molecules.get(target_id) or {}).get("canonical_smiles") or ""
            ),
        },
        "revision": {
            "graph": graph_revision,
            "evidence": int(portfolio.get("evidence_revision") or graph_revision),
            "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
            "portfolio_sha256": str(portfolio.get("content_sha256") or ""),
        },
        "portfolio": {
            "route_ids": list(route_rows),
            "default_route_id": next(iter(route_rows), ""),
            "route_count": len(route_rows),
            "accepted": portfolio.get("accepted") is True,
            "stock_boundary": stock_boundary,
            "closure_profile": _closure_profile(
                accepted=portfolio.get("accepted") is True,
                stock_boundary=stock_boundary,
            ),
            "achieved_profile": achieved_profile,
            "acceptance_profile_counts": profile_counts,
            "process_ready": profile_counts["process_ready"] > 0,
            "closeout": _copy_json(portfolio.get("closeout") or {}),
            "metrics": _copy_json(portfolio.get("metrics") or {}),
            "display_limit": MAX_VISIBLE_ROUTES,
        },
        "campaign_summary": _campaign_summary(campaign_summary),
        "route_closure": route_closure,
        "views": views,
        "routes": route_rows,
        "planned_routes": planned_routes,
        "replacement_routes": replacement_route_rows,
        "replacement_validation": {
            "schema_version": "route_replacement_validation.v1",
            "candidate_count": len(replacement_records),
            "validated_count": sum(
                row.get("accepted") is True for row in replacement_records
            ),
            "records": replacement_records,
            "semantics": {
                "full_candidate_route_was_restitched": True,
                "module_patch_alone_never_grants_acceptance": True,
            },
        },
        "molecules": molecule_rows,
        "edges": edge_rows,
        "hypotheses": hypotheses,
        "modules": modules,
        "shared_intermediates": shared_intermediates,
        "layout": layout,
        "inspectors": {
            "routes": {
                route_id: _route_inspector(route, edge_rows=edge_rows)
                for route_id, route in route_rows.items()
            },
            "edges": {
                edge_id: _edge_inspector(edge_id, graph=graph, proof=edge_proofs.get(edge_id))
                for edge_id in edge_rows
            },
            "molecules": {
                molecule_id: _molecule_inspector(molecule_id, graph=graph)
                for molecule_id in molecule_rows
            },
            "rejections": _copy_json(dict(graph.get("delta") or {}).get("rejected") or []),
            "conflicts": {
                str(key): _copy_json(value)
                for key, value in sorted(dict(graph.get("conflicts") or {}).items())
            },
        },
        "proof_visuals": {str(key): value for key, value in PROOF_VISUALS.items()},
        "semantics": {
            "read_model_only": True,
            "canonical_graph_is_authority": True,
            "proof_color_never_grants_proof": True,
            "default_is_bounded_portfolio": True,
            "shared_intermediates_are_not_duplicated": True,
            "hypotheses_are_not_routes": True,
            "planned_routes_are_advisory_only": True,
            "admission_rejected_steps_never_grant_proof": True,
            "replacement_preview_requires_complete_restitched_route": True,
            "aggregate_counts_never_grant_completion": True,
            "campaign_gates_are_measurements_only": True,
            "configured_boundary_closure_is_not_process_readiness": True,
            "benchmark_search_is_exploration_only": True,
            "declared_route_graph_closure_is_a_separate_axis": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def compile_route_workbench_delta(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a revision-bound delta; consumers need not replace the graph."""

    if current.get("schema_version") != ROUTE_WORKBENCH_SCHEMA:
        raise RouteWorkbenchProjectionError("route_workbench_current_schema_invalid")
    previous_row = dict(previous or {})
    if previous_row and previous_row.get("schema_version") != ROUTE_WORKBENCH_SCHEMA:
        raise RouteWorkbenchProjectionError("route_workbench_previous_schema_invalid")
    if previous_row and previous_row.get("run_id") != current.get("run_id"):
        raise RouteWorkbenchProjectionError("route_workbench_delta_run_mismatch")

    collections = (
        "routes",
        "planned_routes",
        "replacement_routes",
        "molecules",
        "edges",
        "hypotheses",
        "modules",
    )
    upserts: dict[str, dict[str, Any]] = {}
    removals: dict[str, list[str]] = {}
    for name in collections:
        before = dict(previous_row.get(name) or {})
        after = dict(current.get(name) or {})
        changed = {
            key: _copy_json(value)
            for key, value in after.items()
            if key not in before or _digest(before[key]) != _digest(value)
        }
        removed = sorted(set(before) - set(after))
        if changed:
            upserts[name] = changed
        if removed:
            removals[name] = removed
    metadata_changed = not previous_row or any(
        _digest(previous_row.get(name)) != _digest(current.get(name))
        for name in (
            "target",
            "revision",
            "portfolio",
            "campaign_summary",
            "route_closure",
            "views",
            "shared_intermediates",
            "layout",
            "inspectors",
            "replacement_validation",
        )
    )
    payload = {
        "schema_version": ROUTE_WORKBENCH_DELTA_SCHEMA,
        "run_id": str(current.get("run_id") or ""),
        "from_graph_revision": int(
            dict(previous_row.get("revision") or {}).get("graph") or 0
        ),
        "to_graph_revision": int(dict(current.get("revision") or {}).get("graph") or 0),
        "base_sha256": str(previous_row.get("content_sha256") or ""),
        "result_sha256": str(current.get("content_sha256") or ""),
        "upserts": upserts,
        "removals": removals,
        "metadata": (
            {
                name: _copy_json(current.get(name))
                for name in (
                    "target",
                    "revision",
                    "portfolio",
                    "campaign_summary",
                    "route_closure",
                    "views",
                    "shared_intermediates",
                    "layout",
                    "inspectors",
                    "replacement_validation",
                )
            }
            if metadata_changed
            else {}
        ),
        "empty": not upserts and not removals and not metadata_changed,
        "semantics": {
            "apply_only_to_matching_base_sha256": True,
            "entity_upserts_preserve_selection_when_ids_survive": True,
            "full_snapshot_is_recovery_fallback": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def _campaign_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Bound reporting metadata without letting the UI infer scientific state."""

    source = dict(value or {})
    gates = {
        str(key): item is True
        for key, item in dict(source.get("gates") or {}).items()
        if str(key).startswith("B")
    }
    return {
        "schema_version": "retrosynthesis_campaign_summary.v1",
        "available": bool(source),
        "gates": gates,
        "highest_contiguous_gate": str(
            source.get("highest_contiguous_gate") or "none"
        ),
        "counts": _copy_json(source.get("counts") or {}),
        "resource_envelope": _copy_json(source.get("resource_envelope") or {}),
        "model_cost": _copy_json(source.get("model_cost") or {}),
        "stop_decision": _copy_json(source.get("stop_decision") or {}),
        "claim": _copy_json(source.get("claim") or {}),
        "current_disposition": _copy_json(
            source.get("current_disposition") or {}
        ),
        "semantics": {
            "measurement_only": True,
            "independent_gates_may_pass_after_a_contiguous_gap": True,
            "branch_counts_never_grant_completion": True,
        },
    }


def _validate_bindings(graph: Mapping[str, Any], portfolio: Mapping[str, Any]) -> None:
    if graph.get("schema_version") != "canonical_retrosynthesis_hypergraph.v1":
        raise RouteWorkbenchProjectionError("route_workbench_graph_schema_invalid")
    if portfolio.get("schema_version") != "proof_stitched_route_portfolio.v1":
        raise RouteWorkbenchProjectionError("route_workbench_portfolio_schema_invalid")
    if int(portfolio.get("graph_revision") or 0) != int(graph.get("revision") or 0):
        raise RouteWorkbenchProjectionError("route_workbench_graph_revision_mismatch")
    bound = str(portfolio.get("graph_scientific_sha256") or "")
    actual = str(graph.get("scientific_sha256") or "")
    if bound and actual and bound != actual:
        raise RouteWorkbenchProjectionError("route_workbench_graph_digest_mismatch")


def _hypothesis_rows(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = [
        (str(key), dict(value))
        for key, value in dict(graph.get("hypotheses") or {}).items()
        if isinstance(value, Mapping) and value.get("status") != "materialized"
    ]
    values.sort(
        key=lambda item: (
            -float(item[1].get("route_diversity_gain") or 0.0),
            item[0],
        )
    )
    return {
        hypothesis_id: {
            "hypothesis_id": hypothesis_id,
            "product_smiles": str(value.get("product_smiles") or ""),
            "precursor_smiles": list(value.get("precursor_smiles") or []),
            "route_family_ids": list(value.get("route_family_ids") or []),
            "status": str(value.get("status") or "hypothesis"),
            "admission_accepted": value.get("admission_accepted") is True,
            "admission_reasons": list(value.get("admission_reasons") or []),
            "proof_level": 0,
            "proof_color": PROOF_VISUALS[0]["color"],
            "origin_kinds": sorted(
                {
                    str(row.get("origin_kind") or "")
                    for row in value.get("origin_records") or []
                    if isinstance(row, Mapping) and str(row.get("origin_kind") or "")
                }
            ),
            "origin_records": _copy_json(value.get("origin_records") or []),
            "route_innovations": _copy_json(
                merge_route_innovations(
                    (), value.get("route_innovations") or []
                )
            ),
            "innovation_kinds": sorted(
                {
                    str(row.get("kind") or "")
                    for row in merge_route_innovations(
                        (), value.get("route_innovations") or []
                    )
                    if isinstance(row, Mapping) and str(row.get("kind") or "")
                }
            ),
            "badges": sorted(
                {
                    "proposal",
                    "unmaterialized",
                    *(
                        ["admission-rejected"]
                        if value.get("status") == "admission_rejected"
                        else []
                    ),
                    *(
                        f"innovation:{row.get('kind')}"
                        for row in merge_route_innovations(
                            (), value.get("route_innovations") or []
                        )
                        if isinstance(row, Mapping) and row.get("kind")
                    ),
                }
            ),
        }
        for hypothesis_id, value in values[:MAX_VISIBLE_HYPOTHESES]
    }


def _module_rows(
    portfolio: Mapping[str, Any],
    *,
    selected_route_ids: set[str],
) -> dict[str, dict[str, Any]]:
    selected_families = {
        str(value.get("route_family_id") or "")
        for value in portfolio.get("selected_routes") or []
        if isinstance(value, Mapping) and str(value.get("route_id") or "") in selected_route_ids
    }
    rows: dict[str, dict[str, Any]] = {}
    for value in portfolio.get("route_modules") or []:
        if not isinstance(value, Mapping) or not str(value.get("module_id") or ""):
            continue
        family_ids = {
            str(item) for item in value.get("route_family_ids") or [] if str(item)
        }
        if not family_ids and str(value.get("route_family_id") or ""):
            family_ids.add(str(value["route_family_id"]))
        if selected_families.isdisjoint(family_ids):
            continue
        rows[str(value["module_id"])] = _copy_json(value)
    return rows


def _replacement_rows(
    portfolio: Mapping[str, Any],
    *,
    selected_routes: list[dict[str, Any]],
    limit: int = 24,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Project only full proof-stitched alternatives for interactive preview."""

    modules = {
        str(row.get("module_id") or ""): dict(row)
        for row in portfolio.get("route_modules") or []
        if isinstance(row, Mapping) and str(row.get("module_id") or "")
    }
    candidates = [
        dict(row)
        for row in portfolio.get("route_candidates") or []
        if isinstance(row, Mapping) and str(row.get("route_id") or "")
    ]
    projected: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for base in selected_routes:
        base_route_id = str(base.get("route_id") or "")
        base_family_id = str(base.get("route_family_id") or "")
        selections = {
            str(key): str(value)
            for key, value in dict(base.get("module_selections") or {}).items()
            if str(key) and str(value)
        }
        for module_id, current_edge_id in sorted(selections.items()):
            module = modules.get(module_id, {})
            module_family_ids = {
                str(value)
                for value in module.get("route_family_ids") or []
                if str(value)
            }
            if not module_family_ids and str(module.get("route_family_id") or ""):
                module_family_ids.add(str(module["route_family_id"]))
            if base_family_id not in module_family_ids:
                continue
            for alternative in module.get("alternatives") or []:
                if not isinstance(alternative, Mapping):
                    continue
                replacement_edge_id = str(alternative.get("edge_id") or "")
                if not replacement_edge_id or replacement_edge_id == current_edge_id:
                    continue
                matches = [
                    candidate
                    for candidate in candidates
                    if str(candidate.get("route_family_id") or "")
                    in module_family_ids
                    and str(
                        dict(candidate.get("module_selections") or {}).get(module_id)
                        or ""
                    )
                    == replacement_edge_id
                ]
                matches.sort(
                    key=lambda row: (
                        row.get("complete") is not True,
                        row.get("all_edges_proven") is not True,
                        row.get("all_leaves_stock_closed") is not True,
                        float(row.get("risk_score") or 0.0),
                        str(row.get("route_id") or ""),
                    )
                )
                candidate = matches[0] if matches else {}
                accepted = bool(
                    candidate
                    and candidate.get("complete") is True
                    and candidate.get("all_edges_proven") is True
                    and candidate.get("all_leaves_stock_closed") is True
                )
                key = {
                    "base_route_id": base_route_id,
                    "module_id": module_id,
                    "replacement_edge_id": replacement_edge_id,
                }
                replacement_id = "replacement:" + _digest(key)[:24]
                replacement_route_id = "replacement-route:" + _digest(
                    {**key, "candidate": candidate.get("route_id")}
                )[:24]
                reasons: list[str] = []
                if not candidate:
                    reasons.append("full_restitched_route_missing")
                elif candidate.get("complete") is not True:
                    reasons.append("replacement_route_not_boundary_closed")
                if candidate and candidate.get("all_edges_proven") is not True:
                    reasons.append("replacement_route_reaction_proof_incomplete")
                if candidate and candidate.get("all_leaves_stock_closed") is not True:
                    reasons.append("replacement_route_stock_closure_incomplete")
                if accepted:
                    projected[replacement_route_id] = {
                        **candidate,
                        "route_id": replacement_route_id,
                        "underlying_route_id": str(candidate.get("route_id") or ""),
                        "base_route_id": base_route_id,
                        "replacement_id": replacement_id,
                    }
                records.append(
                    {
                        "replacement_id": replacement_id,
                        "base_route_id": base_route_id,
                        "base_edge_id": current_edge_id,
                        "module_id": module_id,
                        "replacement_edge_id": replacement_edge_id,
                        "underlying_route_id": str(candidate.get("route_id") or ""),
                        "replacement_route_id": replacement_route_id if accepted else "",
                        "accepted": accepted,
                        "reasons": sorted(set(reasons)),
                    }
                )
                if len(records) >= limit:
                    return projected, records
    return projected, records


def _shared_intermediate_rows(
    portfolio: Mapping[str, Any],
    *,
    molecules: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    shared = dict(dict(portfolio.get("metrics") or {}).get("shared_intermediates") or {})
    return {
        molecule_id: {
            "molecule_id": molecule_id,
            "canonical_smiles": str(
                dict(molecules.get(molecule_id) or {}).get("canonical_smiles") or ""
            ),
            "route_ids": sorted(str(value) for value in route_ids),
            "render_once": True,
        }
        for molecule_id, route_ids in sorted(shared.items())
        if molecule_id in molecules
    }


def _views(
    routes: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    stages = {
        "hypotheses": {"label": "Disconnection hypotheses", "route_ids": []},
        "expanded": {"label": "Expanded routes", "route_ids": []},
        "reaction_validated": {"label": "Reaction-validated routes", "route_ids": []},
        "literature_grounded": {"label": "Literature-grounded routes", "route_ids": []},
        "stock_closed": {"label": "Configured-boundary-closed routes", "route_ids": []},
        "condition_complete": {"label": "Condition-complete routes", "route_ids": []},
        "procurement_closed": {"label": "Procurement-closed routes", "route_ids": []},
        "process_ready": {"label": "Process-ready routes", "route_ids": []},
    }
    stages["hypotheses"]["hypothesis_ids"] = list(hypotheses)
    for route_id, route in routes.items():
        stage = str(route.get("stage") or "expanded")
        if stage == "stock_closed":
            stages["stock_closed"]["route_ids"].append(route_id)
        if route.get("reaction_validated") is True:
            stages["reaction_validated"]["route_ids"].append(route_id)
        if route.get("literature_grounded") is True:
            stages["literature_grounded"]["route_ids"].append(route_id)
        if route.get("condition_complete") is True:
            stages["condition_complete"]["route_ids"].append(route_id)
        if route.get("procurement_closed") is True:
            stages["procurement_closed"]["route_ids"].append(route_id)
        if route.get("process_ready") is True:
            stages["process_ready"]["route_ids"].append(route_id)
        stages["expanded"]["route_ids"].append(route_id)
    for value in stages.values():
        value["count"] = len(value.get("route_ids") or value.get("hypothesis_ids") or [])
    return stages


def _stable_layout(
    *,
    target_id: str,
    route_rows: Mapping[str, Mapping[str, Any]],
    edge_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    adjacency = {
        str(edge["product_molecule_id"]): [
            str(value) for value in edge.get("precursor_molecule_ids") or []
        ]
        for edge in edge_rows.values()
    }
    layers: dict[str, int] = {target_id: 0} if target_id else {}
    queue = [target_id] if target_id else []
    while queue:
        product = queue.pop(0)
        next_layer = layers[product] + 1
        for precursor in adjacency.get(product, []):
            if next_layer > layers.get(precursor, -1):
                layers[precursor] = next_layer
                queue.append(precursor)
    nodes = [
        {"molecule_id": molecule_id, "layer": layer, "order": order}
        for layer in sorted(set(layers.values()))
        for order, molecule_id in enumerate(
            sorted(value for value, current in layers.items() if current == layer)
        )
    ]
    return {
        "algorithm": "canonical_layer_order.v1",
        "orientation": "target_to_precursors",
        "nodes": nodes,
        "route_edge_sets": {
            route_id: list(route.get("edge_ids") or [])
            for route_id, route in route_rows.items()
        },
        "stable_ids": True,
        "heavy_layout_required": len(nodes) > 120,
    }


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


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
__all__ = [
    "MAX_VISIBLE_ROUTES",
    "PROOF_VISUALS",
    "PROOF_VECTOR_SCHEMA",
    "ROUTE_WORKBENCH_DELTA_SCHEMA",
    "ROUTE_WORKBENCH_SCHEMA",
    "RouteWorkbenchProjectionError",
    "compile_route_workbench",
    "compile_route_workbench_delta",
]
