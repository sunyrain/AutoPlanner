"""Pure scientific compilation helpers for a reviewed case dossier."""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from rdkit import Chem

from cascade_planner.application.retrosynthesis_workers import normalize_source_binding
from cascade_planner.orchestration.global_campaign_director import GlobalCampaignPlan
from cascade_planner.providers.stock import (
    canonicalize_stock_snapshot,
    stock_snapshot_sha256,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate

from .case_dossier_contract import CaseDossierError, json_copy
from .replay_contract import digest


def compile_routes(
    values: Any,
    target_smiles: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    routes: list[dict[str, Any]] = []
    edges: dict[str, dict[str, Any]] = {}
    all_leaves: set[str] = set()
    for route_index, raw_route in enumerate(values or [], start=1):
        if not isinstance(raw_route, Mapping) or raw_route.get("selected") is False:
            continue
        route = dict(raw_route)
        route_id = str(route.get("route_family_id") or f"route:{route_index}")
        products: set[str] = set()
        steps: list[dict[str, Any]] = []
        for step_index, raw_step in enumerate(route.get("steps") or [], start=1):
            if not isinstance(raw_step, Mapping):
                raise CaseDossierError("case_dossier_step_invalid")
            step = dict(raw_step)
            audit = audit_retrosynthetic_candidate(
                step.get("product_smiles"), step.get("reactant_smiles") or []
            )
            if audit.get("accepted") is not True:
                raise CaseDossierError(
                    "case_dossier_step_admission_failed:"
                    + route_id
                    + ":"
                    + ",".join(audit.get("reasons") or [])
                )
            product = str(audit["product_smiles"])
            reactants = list(audit["precursor_smiles_multiset"])
            if product in products:
                raise CaseDossierError("case_dossier_route_product_expanded_twice")
            products.add(product)
            edge_digest = str(audit["edge_digest"])
            mapped = str(step.get("mapped_reaction_smiles") or "")
            existing = edges.get(edge_digest)
            if existing and mapped and existing["mapped_reaction_smiles"] not in {"", mapped}:
                raise CaseDossierError("case_dossier_edge_mapping_conflict")
            edges.setdefault(
                edge_digest,
                {
                    "product_smiles": product,
                    "reactant_smiles": reactants,
                    "reaction_smiles": ".".join(reactants) + ">>" + product,
                    "mapped_reaction_smiles": mapped,
                },
            )
            if mapped and not edges[edge_digest]["mapped_reaction_smiles"]:
                edges[edge_digest]["mapped_reaction_smiles"] = mapped
            steps.append(
                {
                    "step_id": str(step.get("step_id") or f"{route_id}:{step_index}"),
                    "product_smiles": product,
                    "precursor_smiles": reactants,
                    "transformation_hypothesis": str(
                        step.get("transformation_hypothesis") or "exact source step"
                    ),
                    "source_refs": sorted(
                        {str(value) for value in step.get("source_refs") or [] if str(value)}
                    ),
                }
            )
        if not steps or target_smiles not in products:
            raise CaseDossierError("case_dossier_route_not_rooted_at_target")
        reachable = {target_smiles}
        remaining = list(steps)
        while remaining:
            consumed = [step for step in remaining if step["product_smiles"] in reachable]
            if not consumed:
                break
            for step in consumed:
                reachable.update(step["precursor_smiles"])
                remaining.remove(step)
        if remaining:
            raise CaseDossierError("case_dossier_route_contains_disconnected_step")
        route_leaves = {
            precursor
            for step in steps
            for precursor in step["precursor_smiles"]
            if precursor not in products
        }
        all_leaves.update(route_leaves)
        routes.append(
            {
                "route_family_id": route_id,
                "label": str(route.get("label") or route_id),
                "strategic_disconnection": str(
                    route.get("strategic_disconnection") or route.get("label") or route_id
                ),
                "steps": steps,
                "edge_digests": sorted(
                    {
                        str(
                            audit_retrosynthetic_candidate(
                                step["product_smiles"], step["precursor_smiles"]
                            )["edge_digest"]
                        )
                        for step in steps
                    }
                ),
                "leaf_smiles": sorted(route_leaves),
            }
        )
    if not routes:
        raise CaseDossierError("case_dossier_selected_routes_missing")
    return routes, edges, all_leaves


def source_coverage(values: Any) -> tuple[set[str], set[str]]:
    edge_digests: set[str] = set()
    source_groups: set[str] = set()
    for source in values or []:
        if not isinstance(source, Mapping):
            raise CaseDossierError("case_dossier_source_invalid")
        binding = normalize_source_binding(dict(source.get("binding") or {}))
        if binding.get("usable_for_extraction") is not True:
            raise CaseDossierError("case_dossier_source_invalid")
        source_groups.add(str(binding["independence_group"]))
        for row in source.get("rows") or []:
            if not isinstance(row, Mapping) or row.get("relation_type") != "exact":
                raise CaseDossierError("case_dossier_source_row_invalid")
            audit = audit_retrosynthetic_candidate(
                row.get("product_smiles"), row.get("reactant_smiles") or []
            )
            if audit.get("accepted") is not True:
                raise CaseDossierError("case_dossier_source_row_invalid")
            edge_digests.add(str(audit["edge_digest"]))
    return edge_digests, source_groups


def compile_inventory(value: Any, leaves: set[str]) -> dict[str, Any]:
    inventory = json_copy(value)
    artifact = dict(inventory.get("artifact") or {})
    offers: list[dict[str, Any]] = []
    for raw_offer in artifact.get("offers") or []:
        canonical = canonicalize_stock_snapshot(raw_offer)
        canonical["snapshot_sha256"] = stock_snapshot_sha256(canonical)
        offers.append(canonical)
    artifact["offers"] = offers
    inventory["artifact"] = artifact
    closed = {
        str(offer.get("canonical_smiles") or "")
        for offer in offers
        if offer.get("available") is True
    }
    missing = sorted(leaves - closed)
    if missing:
        raise CaseDossierError("case_dossier_stock_leaf_missing:" + ",".join(missing))
    return inventory


def global_plan(
    dossier: Mapping[str, Any],
    routes: list[dict[str, Any]],
    target_smiles: str,
) -> dict[str, Any]:
    case_id = str(dossier["case_id"])
    route_membership: dict[str, set[str]] = {}
    for route in routes:
        for step in route["steps"]:
            route_membership.setdefault(step["product_smiles"], set()).add(
                route["route_family_id"]
            )
            for precursor in step["precursor_smiles"]:
                route_membership.setdefault(precursor, set()).add(
                    route["route_family_id"]
                )
    shared = tuple(
        {
            "canonical_smiles": smiles,
            "route_family_ids": sorted(families),
            "role": "shared_target" if smiles == target_smiles else "shared_intermediate",
        }
        for smiles, families in sorted(route_membership.items())
        if len(families) > 1
    )
    plan = GlobalCampaignPlan(
        plan_id=f"plan:{case_id}",
        run_id=case_id,
        mode="initial_architecture",
        context_sha256=str(dossier.get("content_sha256") or digest(dossier)),
        graph_revision=0,
        route_families=tuple(
            {
                "route_family_id": route["route_family_id"],
                "label": route["label"],
                "selected": True,
                "strategic_disconnection": route["strategic_disconnection"],
            }
            for route in routes
        ),
        multi_step_skeletons=tuple(
            {
                "skeleton_id": f"skeleton:{route['route_family_id']}",
                "route_family_id": route["route_family_id"],
                "target_smiles": target_smiles,
                "steps": route["steps"],
            }
            for route in routes
        ),
        strategic_disconnections=tuple(
            {
                "route_family_id": route["route_family_id"],
                "description": route["strategic_disconnection"],
            }
            for route in routes
        ),
        shared_intermediates=shared,
        critical_unknowns=(),
        source_plan=tuple(
            {
                "source_ref": str(source.get("binding", {}).get("source_ref") or ""),
                "purpose": "exact route replay",
            }
            for source in dossier["sources"]
        ),
        fallback_strategies=tuple(dict(row) for row in dossier.get("fallback_strategies") or []),
        frontier_priorities=tuple(
            {"route_family_id": route["route_family_id"], "priority": index}
            for index, route in enumerate(routes, start=1)
        ),
        pivot_conditions=(),
        stop_conditions=(
            {"kind": "acceptance_contract_satisfied"},
            {"kind": "hard_budget_exhausted"},
        ),
        portfolio_rationale=str(
            dossier.get("portfolio_rationale")
            or "Preserve every source-bound route family and shared canonical edge."
        ),
        limitations=tuple(str(value) for value in dossier.get("limitations") or []),
    )
    return plan.to_dict()


def dataclass_dict(value: Any) -> dict[str, Any]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name != "schema_version"
    }


def canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule is not None
        else ""
    )


__all__ = [
    "canonical_smiles",
    "compile_inventory",
    "compile_routes",
    "dataclass_dict",
    "global_plan",
    "source_coverage",
]
