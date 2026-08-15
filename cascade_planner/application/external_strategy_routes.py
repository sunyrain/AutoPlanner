"""Provider-neutral admission of strategic route bundles.

External planners may be useful strategy generators, but their self-reported
``solved`` or ``feasible`` fields are not host proof.  This module compiles a
small interchange contract into the existing global-plan ingestion path and
audits the remaining strategy-to-experiment gap after admission.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from rdkit import Chem, RDLogger

from cascade_planner.application.canonical_identity import hypothesis_identity
from cascade_planner.application.condition_predictions import (
    normalize_condition_predictions,
)
from cascade_planner.application.reactionjson_replay import (
    ReactionJsonReplayError,
    replay_reactionjson,
)

RDLogger.DisableLog("rdApp.*")
EXTERNAL_STRATEGY_ROUTE_BUNDLE_SCHEMA = "external_strategy_route_bundle.v1"
EXTERNAL_STRATEGY_ROUTE_IMPORT_SCHEMA = "external_strategy_route_import.v1"


class ExternalStrategyRouteError(ValueError):
    """The external route bundle cannot enter the canonical graph safely."""


def compile_external_strategy_route_bundle(
    value: Mapping[str, Any], *, expected_target_smiles: str
) -> dict[str, Any]:
    """Compile a provider-neutral route bundle into a weak global plan."""

    raw = _json_mapping(value, reason="external_strategy_bundle_mapping_required")
    schema = str(raw.get("schema_version") or EXTERNAL_STRATEGY_ROUTE_BUNDLE_SCHEMA)
    if schema != EXTERNAL_STRATEGY_ROUTE_BUNDLE_SCHEMA:
        raise ExternalStrategyRouteError("external_strategy_bundle_schema_invalid")
    provider = _bounded_text(raw.get("provider") or "external", 120).lower()
    if not provider:
        raise ExternalStrategyRouteError("external_strategy_provider_required")
    target = _canonical_smiles(raw.get("target_smiles"))
    expected = _canonical_smiles(expected_target_smiles)
    if not target or not expected:
        raise ExternalStrategyRouteError("external_strategy_target_invalid")
    if target != expected:
        raise ExternalStrategyRouteError("external_strategy_target_mismatch")
    routes = raw.get("routes")
    if not isinstance(routes, list) or not routes or len(routes) > 64:
        raise ExternalStrategyRouteError("external_strategy_routes_count_invalid")

    source_payload_sha256 = _digest(raw)
    origin_ref = f"external_strategy:{source_payload_sha256}"
    compiled_routes = [
        _compile_route(
            route,
            provider=provider,
            target_smiles=target,
            route_index=index,
        )
        for index, route in enumerate(routes, start=1)
    ]
    aliases = [row["route_family"]["route_family_id"] for row in compiled_routes]
    if len(set(aliases)) != len(aliases):
        raise ExternalStrategyRouteError("external_strategy_route_identity_duplicate")
    plan_core = {
        "schema_version": "global_campaign_plan.v1",
        "plan_id": f"external-plan:{source_payload_sha256[:24]}",
        "mode": "external_strategy_import",
        "route_families": [row["route_family"] for row in compiled_routes],
        "multi_step_skeletons": [row["skeleton"] for row in compiled_routes],
        "frontier_priorities": [],
    }
    receipt = {
        "schema_version": EXTERNAL_STRATEGY_ROUTE_IMPORT_SCHEMA,
        "provider": provider,
        "origin_ref": origin_ref,
        "target_smiles": target,
        "source_payload_sha256": source_payload_sha256,
        "route_count": len(compiled_routes),
        "step_count": sum(len(row["steps"]) for row in compiled_routes),
        "routes": [
            {
                "external_route_id": row["external_route_id"],
                "route_family_alias": row["route_family"]["route_family_id"],
                "skeleton_id": row["skeleton"]["skeleton_id"],
                "hypothesis_ids": [step["hypothesis_id"] for step in row["steps"]],
                "edge_ids": [step["edge_id"] for step in row["steps"]],
                "connectivity_valid": True,
            }
            for row in compiled_routes
        ],
        "authority": {
            "scope": "external_strategy_advisory_only",
            "self_reported_solved_grants_proof": False,
            "self_reported_feasibility_grants_validation": False,
            "raw_condition_text_grants_condition_completeness": False,
        },
    }
    receipt["content_sha256"] = _digest(receipt)
    return {"global_plan": plan_core, "receipt": receipt, "origin_ref": origin_ref}


def _compile_route(
    value: Any, *, provider: str, target_smiles: str, route_index: int
) -> dict[str, Any]:
    route = _json_mapping(value, reason="external_strategy_route_mapping_required")
    steps = route.get("steps")
    if not isinstance(steps, list) or not steps or len(steps) > 128:
        raise ExternalStrategyRouteError("external_strategy_steps_count_invalid")
    route_claims = {
        key: route[key]
        for key in ("solved", "productive", "feasibility", "route_score")
        if key in route
    }
    compiled_steps = [
        _compile_step(
            step,
            provider=provider,
            route_claims=route_claims,
            route_index=route_index,
            step_index=index,
            default_direction=str(route.get("reaction_direction") or "forward"),
        )
        for index, step in enumerate(steps, start=1)
    ]
    ordered = _connected_order(compiled_steps, target_smiles=target_smiles)
    external_id = _bounded_text(
        route.get("route_id") or route.get("id") or f"route-{route_index}", 160
    )
    route_digest = _digest(
        {
            "provider": provider,
            "external_route_id": external_id,
            "steps": [step["identity"] for step in ordered],
        }
    )
    family_alias = f"external-route:{route_digest[:24]}"
    family = {
        "route_family_id": family_alias,
        "family_key": family_alias,
        "title": _bounded_text(route.get("name") or external_id, 240),
        "strategy": _bounded_text(
            route.get("strategy") or route.get("summary") or "external strategy route",
            1200,
        ),
        "target_smiles": target_smiles,
        "selected": True,
    }
    skeleton_steps = []
    for index, step in enumerate(ordered, start=1):
        step_id = (
            f"external-step:{route_digest[:12]}:{index:03d}:{step['identity'][:10]}"
        )
        skeleton_steps.append(
            {
                "step_id": step_id,
                "product_smiles": step["product_smiles"],
                "precursor_smiles": step["precursor_smiles"],
                "transformation_hypothesis": step["transformation_hypothesis"],
                "strategic_role": step["strategic_role"],
                "required_validation": [
                    "host_reaction_validation",
                    "exact_source_binding",
                    "condition_completeness",
                ],
                "hypothesis_only": True,
                "condition_predictions": step["condition_predictions"],
                "provider_reaction_metadata": step["provider_reaction_metadata"],
            }
        )
    skeleton = {
        "skeleton_id": f"external-skeleton:{route_digest[:24]}",
        "route_family_id": family_alias,
        "summary": _bounded_text(route.get("summary") or family["strategy"], 1200),
        "steps": skeleton_steps,
    }
    return {
        "external_route_id": external_id,
        "route_family": family,
        "skeleton": skeleton,
        "steps": ordered,
    }


def _compile_step(
    value: Any,
    *,
    provider: str,
    route_claims: Mapping[str, Any],
    route_index: int,
    step_index: int,
    default_direction: str,
) -> dict[str, Any]:
    step = _json_mapping(value, reason="external_strategy_step_mapping_required")
    operations = _reactionjson_operations(step)
    has_explicit = bool(
        step.get("product_smiles") or step.get("mapped_product_smiles")
    ) and isinstance(step.get("precursor_smiles"), list)
    reaction_text = step.get("reaction_smiles") or step.get("rxn_smiles")
    reaction_product: Any = ""
    reaction_precursors: list[str] = []
    if reaction_text:
        reaction_product, reaction_precursors = _reaction_sides(
            reaction_text,
            direction=str(step.get("reaction_direction") or default_direction),
        )
    replay_audit: dict[str, Any] = {}
    if operations is not None:
        mapped_product = (
            step.get("mapped_product_smiles")
            or step.get("product_smiles")
            or reaction_product
        )
        expected = (
            step.get("precursor_smiles")
            if has_explicit
            else reaction_precursors or None
        )
        try:
            replay_audit = replay_reactionjson(
                mapped_product_smiles=str(mapped_product or ""),
                operations=operations,
                expected_precursor_smiles=expected,
            )
        except ReactionJsonReplayError as exc:
            raise ExternalStrategyRouteError(str(exc)) from exc
        product = replay_audit["mapped_product_smiles"]
        precursors = replay_audit["precursor_smiles"]
    elif has_explicit:
        product = step.get("product_smiles") or step.get("mapped_product_smiles")
        precursors = step["precursor_smiles"]
    else:
        product, precursors = reaction_product, reaction_precursors
    canonical_product = _canonical_smiles(product)
    canonical_precursors = [_canonical_smiles(value) for value in precursors]
    if (
        not canonical_product
        or not canonical_precursors
        or not all(canonical_precursors)
    ):
        raise ExternalStrategyRouteError("external_strategy_step_structure_invalid")
    if (
        reaction_text
        and (has_explicit or operations is not None)
        and (
            _canonical_smiles(reaction_product) != canonical_product
            or sorted(_canonical_smiles(v) for v in reaction_precursors)
            != sorted(canonical_precursors)
        )
    ):
        raise ExternalStrategyRouteError("external_strategy_step_structure_conflict")
    hypothesis_id, audit = hypothesis_identity(canonical_product, canonical_precursors)
    if not hypothesis_id:
        raise ExternalStrategyRouteError("external_strategy_step_identity_invalid")
    predictions = normalize_condition_predictions(
        step.get("condition_predictions")
        or step.get("structured_conditions")
        or (
            step.get("conditions")
            if isinstance(step.get("conditions"), Mapping)
            else []
        ),
        default_model=provider,
        producer=f"external_strategy:{provider}",
    )
    raw_conditions = step.get("conditions")
    metadata = {
        "provider": provider,
        "external_step_id": _bounded_text(
            step.get("step_id") or step.get("idx") or f"{route_index}:{step_index}",
            160,
        ),
        "source_reaction_smiles": _bounded_text(
            step.get("reaction_smiles") or step.get("rxn_smiles") or "", 5000
        ),
        "raw_condition_text": (
            _bounded_text(raw_conditions, 5000)
            if isinstance(raw_conditions, str)
            else ""
        ),
        "external_route_claims": dict(route_claims),
        "external_step_claims": {
            key: step[key]
            for key in (
                "assessment",
                "critic_verdict",
                "critic_reason",
                "main_risk",
                "is_key",
                "reaxys_close_count",
                "reaxys_close_url",
                "reaxys_related_count",
                "reaxys_related_url",
            )
            if key in step
        },
        "authority_scope": "external_strategy_advisory_only",
        "not_reaction_proof": True,
        "not_exact_source_evidence": True,
        "not_condition_completeness_proof": True,
    }
    if replay_audit:
        metadata["reactionjson_replay_audit"] = replay_audit
    return {
        "product_smiles": str(audit.get("product_smiles") or canonical_product),
        "precursor_smiles": [
            str(value)
            for value in audit.get("precursor_smiles_multiset") or canonical_precursors
        ],
        "hypothesis_id": hypothesis_id,
        "edge_id": f"edge:{audit['edge_digest']}",
        "identity": str(audit["edge_digest"]),
        "transformation_hypothesis": _bounded_text(
            step.get("class")
            or step.get("strategy")
            or step.get("description")
            or "external transformation hypothesis",
            1200,
        ),
        "strategic_role": _bounded_text(
            step.get("strategic_role") or step.get("why_critical") or "route step", 800
        ),
        "condition_predictions": predictions,
        "provider_reaction_metadata": metadata,
    }


def _connected_order(
    steps: list[dict[str, Any]], *, target_smiles: str
) -> list[dict[str, Any]]:
    by_product: dict[str, dict[str, Any]] = {}
    for step in steps:
        product = str(step["product_smiles"])
        if product in by_product:
            raise ExternalStrategyRouteError("external_strategy_duplicate_product_step")
        by_product[product] = step
    if target_smiles not in by_product:
        raise ExternalStrategyRouteError("external_strategy_target_step_missing")
    ordered: list[dict[str, Any]] = []
    visited: set[str] = set()
    active: set[str] = set()

    def visit(product: str) -> None:
        if product in active:
            raise ExternalStrategyRouteError("external_strategy_route_cycle")
        if product in visited or product not in by_product:
            return
        active.add(product)
        step = by_product[product]
        ordered.append(step)
        for precursor in sorted(set(step["precursor_smiles"])):
            visit(str(precursor))
        active.remove(product)
        visited.add(product)

    visit(target_smiles)
    if len(visited) != len(steps):
        raise ExternalStrategyRouteError("external_strategy_route_disconnected")
    return ordered


def _reaction_sides(value: Any, *, direction: str) -> tuple[str, list[str]]:
    text = str(value or "").strip()
    parts = text.split(">")
    if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
        raise ExternalStrategyRouteError("external_strategy_reaction_smiles_invalid")
    direction = direction.strip().lower()
    if direction not in {"forward", "retro", "retrosynthetic"}:
        raise ExternalStrategyRouteError("external_strategy_reaction_direction_invalid")
    left, right = parts[0].strip(), parts[2].strip()
    product_side, precursor_side = (
        (right, left) if direction == "forward" else (left, right)
    )
    products = [value for value in product_side.split(".") if value]
    precursors = [value for value in precursor_side.split(".") if value]
    if len(products) != 1 or not precursors:
        raise ExternalStrategyRouteError("external_strategy_reaction_arity_invalid")
    return products[0], precursors


def _reactionjson_operations(step: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    value = step.get("reactionjson")
    if isinstance(value, Mapping):
        value = value.get("operations")
    if value is None and "operations" in step:
        value = step.get("operations")
    if value is None:
        return None
    if not isinstance(value, list):
        raise ExternalStrategyRouteError("reactionjson_operations_list_required")
    return value


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _json_mapping(value: Any, *, reason: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalStrategyRouteError(reason)
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalStrategyRouteError("external_strategy_bundle_not_json") from exc


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


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
    "EXTERNAL_STRATEGY_ROUTE_BUNDLE_SCHEMA",
    "EXTERNAL_STRATEGY_ROUTE_IMPORT_SCHEMA",
    "ExternalStrategyRouteError",
    "compile_external_strategy_route_bundle",
]
