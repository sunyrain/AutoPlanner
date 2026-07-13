"""Weakest-link proof stitching and compact Pareto route portfolios for V4."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cascade_planner.application.deficit_frontier import (
    compile_selected_route_deficits,
)
from cascade_planner.application.frontier_runtime import publish_frontier_items
from cascade_planner.application.portfolio_selection import (
    CLOSEOUT_SCHEMA as CLOSEOUT_SCHEMA,
    closeout,
    deduplicate_edge_sets,
    deduplicate_records,
    pareto_front,
    portfolio_metrics,
    select_portfolio,
)
from cascade_planner.application.proof_policy import (
    ProofPolicy,
    stitch_edge_proof,
    validate_canonical_graph_entities,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
)
from cascade_planner.application.route_variants import (
    PROOF_ROUTE_SCHEMA as PROOF_ROUTE_SCHEMA,
    ROUTE_MODULE_SCHEMA as ROUTE_MODULE_SCHEMA,
    PortfolioConfig as PortfolioConfig,
    RouteSubroute,
    build_route_candidate,
    enumerate_family_variants,
    with_content_digest,
)
from cascade_planner.application.run_kernel import RunKernel


PROOF_PORTFOLIO_SCHEMA = "proof_stitched_route_portfolio.v1"


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
        variants, modules = enumerate_family_variants(
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
            variants = [RouteSubroute(frozenset(), frozenset(), ())]
        candidates.extend(
            build_route_candidate(
                graph,
                family_id=str(family_id),
                family=family,
                variant=variant,
                edge_proofs=edge_proofs,
                leaf_proof_cache=leaf_proof_cache,
                policy=policy,
            )
            for variant in variants
        )

    candidates = deduplicate_edge_sets(candidates)
    pareto_ids = {str(value["route_id"]) for value in pareto_front(candidates)}
    candidates = [
        with_content_digest(
            {**value, "pareto_optimal": value["route_id"] in pareto_ids}
        )
        for value in candidates
    ]
    selected = select_portfolio(
        candidates,
        minimum_count=active.minimum_routes_to_show,
        maximum_count=active.maximum_routes_to_show,
        require_distinct_edge_sets=acceptance.require_distinct_edge_sets,
    )
    selected_ids = {str(value["route_id"]) for value in selected}
    candidates = [
        with_content_digest(
            {**value, "selected": str(value["route_id"]) in selected_ids}
        )
        for value in candidates
    ]
    candidates_by_id = {str(value["route_id"]): value for value in candidates}
    selected = [candidates_by_id[str(value["route_id"])] for value in selected]
    deficits = compile_selected_route_deficits(
        selected,
        edge_proofs=edge_proofs,
        acceptance_spec=acceptance,
    )
    metrics = portfolio_metrics(selected, graph=graph)
    closeout_record = closeout(
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
        "route_modules": deduplicate_records(route_modules, key="module_id"),
        "deficits": deficits,
        "metrics": metrics,
        "closeout": closeout_record,
        "accepted": closeout_record["accepted"],
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
    publish_frontier_items(
        kernel,
        portfolio["deficits"],
        source_revision=int(graph.get("revision") or 0),
        idempotency_key=f"portfolio:deficits:{idempotency_key}",
        projection_sha256=portfolio["content_sha256"],
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
        str(value.get("edge_id") or "") for value in module.get("alternatives") or []
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
    if module and route and module.get("route_family_id") != route.get(
        "route_family_id"
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
    "CLOSEOUT_SCHEMA",
    "PROOF_PORTFOLIO_SCHEMA",
    "PROOF_ROUTE_SCHEMA",
    "ROUTE_MODULE_SCHEMA",
    "PortfolioConfig",
    "compile_proof_portfolio",
    "publish_proof_portfolio",
    "validate_module_replacement",
]
