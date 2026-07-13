"""Weakest-link proof stitching and compact Pareto route portfolios for V4."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.deficit_frontier import (
    compile_selected_route_deficits,
)
from cascade_planner.application.proof_policy import (
    ProofPolicy,
    stitch_edge_proof,
    stitch_leaf_stock_proof,
    validate_canonical_graph_entities,
)
from cascade_planner.application.run_kernel import Deficit, RunKernel
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)


PROOF_PORTFOLIO_SCHEMA = "proof_stitched_route_portfolio.v1"
PROOF_ROUTE_SCHEMA = "proof_stitched_route.v1"
ROUTE_MODULE_SCHEMA = "route_replacement_module.v1"
CLOSEOUT_SCHEMA = "retrosynthesis_closeout.v1"


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    minimum_routes_to_show: int = 2
    maximum_routes_to_show: int = 5
    maximum_variants_per_family: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_routes_to_show <= self.maximum_routes_to_show <= 12:
            raise ValueError("portfolio route display limits are invalid")
        if not 1 <= self.maximum_variants_per_family <= 2048:
            raise ValueError("portfolio family variant limit is invalid")


@dataclass(frozen=True, slots=True)
class _Subroute:
    edge_ids: frozenset[str]
    leaf_ids: frozenset[str]
    module_selections: tuple[tuple[str, str], ...]


def compile_proof_portfolio(
    graph: Mapping[str, Any],
    *,
    acceptance_spec: RetrosynthesisAcceptanceSpec | None = None,
    config: PortfolioConfig | None = None,
    budget_exhausted: bool = False,
) -> dict[str, Any]:
    acceptance = acceptance_spec or RetrosynthesisAcceptanceSpec()
    active = config or PortfolioConfig()
    policy = ProofPolicy.from_acceptance(acceptance)
    graph_reasons = validate_canonical_graph_entities(graph)
    edge_proofs = {
        str(edge_id): stitch_edge_proof(graph, str(edge_id), policy=policy)
        for edge_id in sorted(dict(graph.get("edges") or {}))
    }
    leaf_proof_cache: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    route_modules: list[dict[str, Any]] = []
    for family_id, raw in sorted(dict(graph.get("route_families") or {}).items()):
        if (
            not isinstance(raw, Mapping)
            or raw.get("selected") is False
            or raw.get("status") == "dominated"
        ):
            continue
        family = dict(raw)
        variants, modules = _enumerate_family_variants(
            graph,
            family_id=str(family_id),
            family=family,
            policy=policy,
            edge_proofs=edge_proofs,
            leaf_proof_cache=leaf_proof_cache,
            limit=active.maximum_variants_per_family,
        )
        route_modules.extend(modules)
        if not variants:
            variants = [_Subroute(frozenset(), frozenset(), ())]
        for variant in variants:
            candidates.append(
                _route_candidate(
                    graph,
                    family_id=str(family_id),
                    family=family,
                    variant=variant,
                    edge_proofs=edge_proofs,
                    leaf_proof_cache=leaf_proof_cache,
                    policy=policy,
                )
            )

    candidates = _deduplicate_edge_sets(candidates)
    pareto = _pareto_front(candidates)
    pareto_ids = {str(value["route_id"]) for value in pareto}
    candidates = [
        _with_content_digest({**value, "pareto_optimal": value["route_id"] in pareto_ids})
        for value in candidates
    ]
    selected = _select_portfolio(
        candidates,
        minimum_count=active.minimum_routes_to_show,
        maximum_count=active.maximum_routes_to_show,
        require_distinct_edge_sets=acceptance.require_distinct_edge_sets,
    )
    selected_ids = {str(value["route_id"]) for value in selected}
    candidates = [
        _with_content_digest(
            {**value, "selected": str(value["route_id"]) in selected_ids}
        )
        for value in candidates
    ]
    selected = [
        next(value for value in candidates if value["route_id"] == selected_value["route_id"])
        for selected_value in selected
    ]
    deficits = compile_selected_route_deficits(
        selected,
        edge_proofs=edge_proofs,
        acceptance_spec=acceptance,
    )
    metrics = _portfolio_metrics(selected, graph=graph)
    closeout = _closeout(
        selected,
        deficits=deficits,
        metrics=metrics,
        acceptance=acceptance,
        graph_reasons=graph_reasons,
        budget_exhausted=budget_exhausted,
    )
    portfolio_leaf_ids = {
        str(molecule_id)
        for route in candidates
        for molecule_id in route.get("leaf_molecule_ids") or []
    }
    payload = {
        "schema_version": PROOF_PORTFOLIO_SCHEMA,
        "graph_revision": int(graph.get("revision") or 0),
        "graph_scientific_sha256": str(graph.get("scientific_sha256") or ""),
        "proof_policy": policy.to_dict(),
        "edge_proofs": edge_proofs,
        "leaf_proofs": {
            molecule_id: leaf_proof_cache[molecule_id]
            for molecule_id in sorted(portfolio_leaf_ids)
        },
        "route_candidates": sorted(candidates, key=lambda row: row["route_id"]),
        "selected_routes": selected,
        "route_modules": _deduplicate_records(route_modules, key="module_id"),
        "deficits": deficits,
        "metrics": metrics,
        "closeout": closeout,
        "accepted": closeout["accepted"],
        "semantics": {
            "weakest_link_route_acceptance": True,
            "small_pareto_portfolio": True,
            "shared_intermediates_have_canonical_identity": True,
            "module_replacement_does_not_duplicate_route_graph": True,
            "aggregate_counts_never_grant_completion": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def publish_proof_portfolio(
    kernel: RunKernel,
    graph: Mapping[str, Any],
    *,
    idempotency_key: str,
    config: PortfolioConfig | None = None,
    budget_exhausted: bool = False,
) -> dict[str, Any]:
    if int(graph.get("revision") or 0) != kernel.state.graph_revision:
        raise ValueError("proof_portfolio_graph_revision_stale")
    portfolio = compile_proof_portfolio(
        graph,
        acceptance_spec=kernel.spec.acceptance,
        config=config,
        budget_exhausted=budget_exhausted,
    )
    ref = kernel.artifacts.put_json(
        portfolio,
        logical_name="proof_stitched_route_portfolio.json",
        producer="autoplanner.proof_portfolio",
    )
    run_digest = hashlib.sha256(kernel.spec.run_id.encode("utf-8")).hexdigest()
    kernel.artifacts.write_pointer(
        f"p/{run_digest[:24]}/latest",
        ref,
        metadata={
            "run_id": kernel.spec.run_id,
            "graph_revision": graph.get("revision"),
            "accepted": portfolio["accepted"],
        },
    )
    kernel.index.index_artifact(
        run_id=kernel.spec.run_id,
        artifact_id="proof_stitched_route_portfolio",
        ref=ref,
        revision=int(graph.get("revision") or 0),
        authority_scope="proof_stitched_route_portfolio",
    )
    kernel_deficits = [
        Deficit(
            deficit_id=str(row["deficit_id"]),
            kind=str(row["kind"]),
            source_revision=int(graph.get("revision") or 0),
            priority=float(row["priority"]),
            deterministic=row.get("deterministic") is True,
            model_allowed=row.get("model_allowed") is True,
            entity_refs=tuple(str(value) for value in row.get("entity_ids") or []),
            reasons=tuple(
                str(value)
                for value in (
                    dict(row.get("metadata") or {}).get("reasons")
                    or [row.get("reason")]
                )
                if str(value or "")
            ),
            metadata={
                "route_ids": list(
                    dict(row.get("metadata") or {}).get("route_ids") or []
                ),
                "portfolio_sha256": portfolio["content_sha256"],
            },
        )
        for row in portfolio["deficits"]
    ]
    kernel.replace_deficits(
        kernel_deficits,
        source_revision=int(graph.get("revision") or 0),
        idempotency_key=f"portfolio:deficits:{idempotency_key}",
    )
    acceptance_report = {
        "schema_version": "proof_portfolio_acceptance_report.v1",
        "graph_revision": int(graph.get("revision") or 0),
        "portfolio_sha256": portfolio["content_sha256"],
        "accepted": portfolio["accepted"],
        "decision": portfolio["closeout"]["decision"],
        "complete_route_count": portfolio["closeout"]["complete_route_count"],
        "selected_route_count": len(portfolio["selected_routes"]),
        "molecule_count": len(graph.get("molecules") or {}),
        "hyperedge_count": len(graph.get("edges") or {}),
        "deficit_count": len(portfolio["deficits"]),
        "reasons": list(portfolio["closeout"]["reasons"]),
    }
    acceptance_report["content_sha256"] = _digest(acceptance_report)
    kernel.record_acceptance(
        acceptance_report,
        idempotency_key=f"portfolio:acceptance:{idempotency_key}",
    )
    return {
        "portfolio": portfolio,
        "portfolio_ref": ref.to_dict(),
        "acceptance_report": acceptance_report,
    }


def validate_module_replacement(
    portfolio: Mapping[str, Any],
    *,
    route_id: str,
    module_id: str,
    replacement_edge_id: str,
) -> dict[str, Any]:
    module = next(
        (
            dict(value)
            for value in portfolio.get("route_modules") or []
            if value.get("module_id") == module_id
        ),
        {},
    )
    route = next(
        (
            dict(value)
            for value in portfolio.get("route_candidates") or []
            if value.get("route_id") == route_id
        ),
        {},
    )
    alternatives = {
        str(value.get("edge_id") or "")
        for value in module.get("alternatives") or []
    }
    current = str(dict(route.get("module_selections") or {}).get(module_id) or "")
    reasons: list[str] = []
    if not module:
        reasons.append("replacement_module_missing")
    if not route:
        reasons.append("replacement_route_missing")
    if replacement_edge_id not in alternatives:
        reasons.append("replacement_edge_not_in_module")
    if replacement_edge_id == current:
        reasons.append("replacement_edge_is_already_selected")
    if module and route and not current:
        reasons.append("replacement_module_not_used_by_route")
    if module and route and (
        module.get("route_family_id") != route.get("route_family_id")
    ):
        reasons.append("replacement_module_route_family_mismatch")
    patch = {
        "schema_version": "route_module_replacement_patch.v1",
        "route_id": route_id,
        "module_id": module_id,
        "product_molecule_id": str(module.get("product_molecule_id") or ""),
        "remove_edge_id": current,
        "add_edge_id": replacement_edge_id,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "patch_reuses_canonical_subgraph": True,
            "patch_does_not_duplicate_entire_route": True,
            "replacement_requires_reproof": True,
        },
    }
    patch["content_sha256"] = _digest(patch)
    return patch


def _enumerate_family_variants(
    graph: Mapping[str, Any],
    *,
    family_id: str,
    family: Mapping[str, Any],
    policy: ProofPolicy,
    edge_proofs: Mapping[str, Mapping[str, Any]],
    leaf_proof_cache: dict[str, dict[str, Any]],
    limit: int,
) -> tuple[list[_Subroute], list[dict[str, Any]]]:
    allowed = {
        str(value)
        for value in family.get("edge_ids") or []
        if str(value) in dict(graph.get("edges") or {})
    }
    outgoing: dict[str, list[str]] = {}
    for edge_id in sorted(allowed):
        edge = graph["edges"][edge_id]
        outgoing.setdefault(str(edge["product_molecule_id"]), []).append(edge_id)
    modules: list[dict[str, Any]] = []
    for molecule_id, edge_ids in sorted(outgoing.items()):
        if len(edge_ids) < 2:
            continue
        module = {
            "schema_version": ROUTE_MODULE_SCHEMA,
            "module_id": f"module:{_digest({'family': family_id, 'product': molecule_id})}",
            "route_family_id": family_id,
            "product_molecule_id": molecule_id,
            "alternatives": [
                {
                    "edge_id": edge_id,
                    "proof_level": int(
                        edge_proofs.get(edge_id, {}).get("achieved_level") or 0
                    ),
                    "accepted": edge_proofs.get(edge_id, {}).get("accepted") is True,
                }
                for edge_id in sorted(edge_ids)
            ],
            "semantics": {
                "same_product_replacement_boundary": True,
                "shared_subgraph_not_duplicated": True,
            },
        }
        module["content_sha256"] = _digest(module)
        modules.append(module)
    module_by_product = {
        str(value["product_molecule_id"]): str(value["module_id"])
        for value in modules
    }

    def walk(molecule_id: str, ancestors: frozenset[str]) -> list[_Subroute]:
        if molecule_id in ancestors:
            return []
        options: list[_Subroute] = []
        stock = leaf_proof_cache.setdefault(
            molecule_id,
            stitch_leaf_stock_proof(graph, molecule_id, policy=policy),
        )
        edges = outgoing.get(molecule_id, [])
        if stock["accepted"] is True or not edges:
            options.append(_Subroute(frozenset(), frozenset({molecule_id}), ()))
        for edge_id in edges:
            edge = graph["edges"][edge_id]
            precursor_variants = [
                walk(str(precursor_id), ancestors | {molecule_id})
                for precursor_id in edge.get("precursor_molecule_ids") or []
            ]
            if any(not values for values in precursor_variants):
                continue
            for combination in product(*precursor_variants):
                edges_used = {edge_id}
                leaves: set[str] = set()
                selections: dict[str, str] = {}
                for subroute in combination:
                    edges_used.update(subroute.edge_ids)
                    leaves.update(subroute.leaf_ids)
                    selections.update(dict(subroute.module_selections))
                if molecule_id in module_by_product:
                    selections[module_by_product[molecule_id]] = edge_id
                options.append(
                    _Subroute(
                        frozenset(edges_used),
                        frozenset(leaves),
                        tuple(sorted(selections.items())),
                    )
                )
                if len(options) >= limit:
                    break
            if len(options) >= limit:
                break
        deduped = {
            (value.edge_ids, value.leaf_ids, value.module_selections): value
            for value in options
        }
        return list(deduped.values())[:limit]

    root = str(graph.get("target_molecule_id") or "")
    variants = walk(root, frozenset()) if root else []
    return variants[:limit], modules


def _route_candidate(
    graph: Mapping[str, Any],
    *,
    family_id: str,
    family: Mapping[str, Any],
    variant: _Subroute,
    edge_proofs: Mapping[str, Mapping[str, Any]],
    leaf_proof_cache: dict[str, dict[str, Any]],
    policy: ProofPolicy,
) -> dict[str, Any]:
    edge_ids = sorted(variant.edge_ids)
    leaf_ids = sorted(variant.leaf_ids)
    proofs = [dict(edge_proofs[edge_id]) for edge_id in edge_ids]
    leaves = [
        leaf_proof_cache.setdefault(
            molecule_id,
            stitch_leaf_stock_proof(graph, molecule_id, policy=policy),
        )
        for molecule_id in leaf_ids
    ]
    source_groups = sorted(
        {
            str(group)
            for proof in proofs
            for group in proof.get("independent_source_groups") or []
            if str(group)
        }
    )
    conflicts = sorted(
        {
            str(conflict_id)
            for proof in proofs
            for conflict_id in proof.get("conflict_ids") or []
            if str(conflict_id)
        }
    )
    min_proof = min((int(value["achieved_level"]) for value in proofs), default=0)
    unproven_edge_ids = sorted(
        str(value["edge_id"])
        for value in proofs
        if value.get("accepted") is not True
    )
    stock_rate = sum(value["accepted"] is True for value in leaves) / max(1, len(leaves))
    open_leaf_molecule_ids = sorted(
        str(value["molecule_id"])
        for value in leaves
        if value.get("accepted") is not True
    )
    source_met = len(source_groups) >= policy.minimum_independent_source_groups
    complete = bool(edge_ids) and all(value["accepted"] is True for value in proofs)
    if policy.require_stock_for_every_selected_leaf:
        complete = complete and bool(leaves) and stock_rate == 1.0
    complete = complete and source_met and not conflicts
    root_edges = sorted(
        edge_id
        for edge_id in edge_ids
        if str(graph["edges"][edge_id]["product_molecule_id"])
        == str(graph.get("target_molecule_id") or "")
    )
    precursor_frequency: dict[str, int] = {}
    for edge_id in edge_ids:
        for molecule_id in graph["edges"][edge_id]["precursor_molecule_ids"]:
            precursor_frequency[str(molecule_id)] = precursor_frequency.get(str(molecule_id), 0) + 1
    convergence = sum(value > 1 for value in precursor_frequency.values()) / max(
        1, len(precursor_frequency)
    )
    risk = (
        0.35 * (1.0 - min_proof / 4.0)
        + 0.25 * (1.0 - stock_rate)
        + 0.20 * (not source_met)
        + 0.15 * bool(conflicts)
        + 0.05 * min(1.0, len(edge_ids) / 12.0)
    )
    identity = {
        "route_family_id": family_id,
        "edge_ids": edge_ids,
        "leaf_molecule_ids": leaf_ids,
    }
    row = {
        "schema_version": PROOF_ROUTE_SCHEMA,
        "route_id": f"route:{_digest(identity)}",
        "route_family_id": family_id,
        "strategy": str(family.get("strategy") or ""),
        "edge_ids": edge_ids,
        "leaf_molecule_ids": leaf_ids,
        "root_edge_ids": root_edges,
        "module_selections": dict(variant.module_selections),
        "minimum_edge_proof_level": min_proof,
        "all_edges_proven": bool(proofs) and all(value["accepted"] for value in proofs),
        "unproven_edge_ids": unproven_edge_ids,
        "stock_closure_rate": round(stock_rate, 6),
        "all_leaves_stock_closed": bool(leaves) and stock_rate == 1.0,
        "open_leaf_molecule_ids": open_leaf_molecule_ids,
        "independent_source_groups": source_groups,
        "source_independence_met": source_met,
        "conflict_ids": conflicts,
        "length": len(edge_ids),
        "convergence_score": round(convergence, 6),
        "risk_score": round(float(risk), 6),
        "complete": complete,
        "selected": False,
        "semantics": {
            "weakest_edge_controls_route": True,
            "every_leaf_requires_stock_observation": True,
            "counts_do_not_override_boolean_proofs": True,
        },
    }
    return _with_content_digest(row)


def _deduplicate_edge_sets(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_edges: dict[tuple[str, ...], dict[str, Any]] = {}
    for value in candidates:
        row = dict(value)
        key = tuple(row.get("edge_ids") or [])
        current = by_edges.get(key)
        if current is None or _candidate_sort_key(row) < _candidate_sort_key(current):
            if current:
                row["equivalent_route_family_ids"] = sorted(
                    {
                        str(current["route_family_id"]),
                        *(current.get("equivalent_route_family_ids") or []),
                    }
                )
            by_edges[key] = _with_content_digest(row)
        elif current:
            current["equivalent_route_family_ids"] = sorted(
                {
                    str(row["route_family_id"]),
                    *(current.get("equivalent_route_family_ids") or []),
                }
            )
            by_edges[key] = _with_content_digest(current)
    return list(by_edges.values())


def _pareto_front(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(value) for value in candidates]
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        vector = _objective_vector(row)
        if any(
            _dominates(_objective_vector(other), vector)
            for other_index, other in enumerate(rows)
            if other_index != index
        ):
            continue
        out.append(row)
    return sorted(out, key=_candidate_sort_key)


def _select_portfolio(
    candidates: Iterable[Mapping[str, Any]],
    *,
    minimum_count: int,
    maximum_count: int,
    require_distinct_edge_sets: bool,
) -> list[dict[str, Any]]:
    remaining = sorted((dict(value) for value in candidates), key=_candidate_sort_key)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < maximum_count:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in remaining:
            if require_distinct_edge_sets and any(
                set(row.get("edge_ids") or []) == set(value.get("edge_ids") or [])
                for value in selected
            ):
                continue
            diversity = (
                min(_route_distance(row, value) for value in selected)
                if selected
                else 1.0
            )
            new_strategy = not any(
                row.get("root_edge_ids") == value.get("root_edge_ids")
                for value in selected
            )
            utility = (
                _candidate_utility(row)
                + 35.0 * diversity
                + 8.0 * new_strategy
                + 5.0 * (row.get("pareto_optimal") is True)
            )
            scored.append((-utility, str(row["route_id"]), row))
        if not scored:
            break
        chosen = min(scored)[2]
        selected.append(chosen)
        remaining = [row for row in remaining if row["route_id"] != chosen["route_id"]]
        if len(selected) >= minimum_count and not remaining:
            break
    return selected


def _portfolio_metrics(
    selected: Iterable[Mapping[str, Any]],
    *,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    routes = [dict(value) for value in selected]
    pairwise: list[dict[str, Any]] = []
    for index, left in enumerate(routes):
        for right in routes[index + 1 :]:
            pairwise.append(
                {
                    "left_route_id": left["route_id"],
                    "right_route_id": right["route_id"],
                    "edge_set_distance": round(_route_distance(left, right), 6),
                    "strategic_disconnection_distinct": (
                        left.get("root_edge_ids") != right.get("root_edge_ids")
                    ),
                }
            )
    edge_routes: dict[str, list[str]] = {}
    molecule_routes: dict[str, list[str]] = {}
    for route in routes:
        for edge_id in route.get("edge_ids") or []:
            edge_routes.setdefault(str(edge_id), []).append(str(route["route_id"]))
            edge = dict(graph.get("edges") or {}).get(str(edge_id)) or {}
            for molecule_id in [
                edge.get("product_molecule_id"),
                *(edge.get("precursor_molecule_ids") or []),
            ]:
                molecule_routes.setdefault(str(molecule_id), []).append(str(route["route_id"]))
    return {
        "selected_route_count": len(routes),
        "complete_route_count": sum(value.get("complete") is True for value in routes),
        "distinct_edge_set_count": len(
            {tuple(value.get("edge_ids") or []) for value in routes}
        ),
        "distinct_complete_edge_set_count": len(
            {
                tuple(value.get("edge_ids") or [])
                for value in routes
                if value.get("complete") is True
            }
        ),
        "strategic_disconnection_count": len(
            {tuple(value.get("root_edge_ids") or []) for value in routes}
        ),
        "complete_strategic_disconnection_count": len(
            {
                tuple(value.get("root_edge_ids") or [])
                for value in routes
                if value.get("complete") is True
            }
        ),
        "pairwise_diversity": pairwise,
        "shared_bottleneck_edges": {
            edge_id: sorted(route_ids)
            for edge_id, route_ids in sorted(edge_routes.items())
            if len(set(route_ids)) > 1
        },
        "shared_intermediates": {
            molecule_id: sorted(set(route_ids))
            for molecule_id, route_ids in sorted(molecule_routes.items())
            if molecule_id and len(set(route_ids)) > 1
        },
        "minimum_selected_proof_level": min(
            (int(value.get("minimum_edge_proof_level") or 0) for value in routes),
            default=0,
        ),
        "mean_length": round(
            sum(int(value.get("length") or 0) for value in routes) / max(1, len(routes)),
            6,
        ),
        "mean_risk": round(
            sum(float(value.get("risk_score") or 0.0) for value in routes)
            / max(1, len(routes)),
            6,
        ),
    }


def _closeout(
    selected: Iterable[Mapping[str, Any]],
    *,
    deficits: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    acceptance: RetrosynthesisAcceptanceSpec,
    graph_reasons: Iterable[str],
    budget_exhausted: bool,
) -> dict[str, Any]:
    routes = [dict(value) for value in selected]
    reasons = list(graph_reasons)
    complete = sum(value.get("complete") is True for value in routes)
    if complete < acceptance.minimum_complete_routes:
        reasons.append("minimum_complete_route_count_not_met")
    if acceptance.require_distinct_edge_sets and int(
        metrics.get("distinct_complete_edge_set_count") or 0
    ) < acceptance.minimum_complete_routes:
        reasons.append("distinct_complete_edge_sets_not_met")
    if any(value.get("all_edges_proven") is not True for value in routes if value.get("complete")):
        reasons.append("selected_complete_route_has_unproven_edge")
    if any(
        value.get("all_leaves_stock_closed") is not True
        for value in routes
        if value.get("complete")
    ):
        reasons.append("selected_complete_route_has_open_leaf")
    accepted = not reasons and complete >= acceptance.minimum_complete_routes
    if graph_reasons:
        decision = "invalid"
    elif accepted:
        decision = "accepted"
    elif budget_exhausted:
        decision = "budget_exhausted"
    else:
        decision = "unresolved"
    row = {
        "schema_version": CLOSEOUT_SCHEMA,
        "decision": decision,
        "accepted": accepted,
        "complete_route_count": complete,
        "selected_route_count": len(routes),
        "deficit_count": len(deficits),
        "reasons": sorted(set(reasons)),
        "semantics": {
            "only_boolean_proof_and_stock_closure_can_accept": True,
            "counts_are_diagnostics_not_authority": True,
            "budget_exhaustion_never_means_success": True,
        },
    }
    row["content_sha256"] = _digest(row)
    return row


def _candidate_utility(value: Mapping[str, Any]) -> float:
    return (
        150.0 * (value.get("complete") is True)
        + 18.0 * int(value.get("minimum_edge_proof_level") or 0)
        + 28.0 * float(value.get("stock_closure_rate") or 0.0)
        + 7.0 * min(3, len(value.get("independent_source_groups") or []))
        + 8.0 * float(value.get("convergence_score") or 0.0)
        - 35.0 * float(value.get("risk_score") or 0.0)
        - 1.5 * int(value.get("length") or 0)
    )


def _candidate_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -(value.get("complete") is True),
        -_candidate_utility(value),
        float(value.get("risk_score") or 0.0),
        int(value.get("length") or 0),
        str(value.get("route_id") or ""),
    )


def _objective_vector(value: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(value.get("complete") is True),
        float(value.get("minimum_edge_proof_level") or 0) / 4.0,
        float(value.get("stock_closure_rate") or 0.0),
        min(1.0, len(value.get("independent_source_groups") or []) / 3.0),
        1.0 - min(1.0, float(value.get("risk_score") or 0.0)),
        1.0 / (1.0 + float(value.get("length") or 0.0)),
    )


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _route_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_edges = set(left.get("edge_ids") or [])
    right_edges = set(right.get("edge_ids") or [])
    union = left_edges | right_edges
    return 1.0 if not union else 1.0 - len(left_edges & right_edges) / len(union)


def _deduplicate_records(
    values: Iterable[Mapping[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    rows = {str(value.get(key) or ""): dict(value) for value in values}
    return [rows[value] for value in sorted(rows) if value]


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


def _with_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = _digest(row)
    return row
