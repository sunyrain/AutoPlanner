"""Normalize, fingerprint, and select proposal-only ChemEnzy routes."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from cascade_planner.baselines.chem_enzy_adapter import (
    route_candidates_from_chem_enzy_result,
)
from cascade_planner.interfaces.chemenzy_advisory import (
    normalized_quarantined_routes,
)
from cascade_planner.interfaces.chemenzy_probe_contract import (
    _content_sha256,
    _json_safe_copy,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate


def _normalized_routes(
    value: Mapping[str, Any], *, target_smiles: str
) -> list[dict[str, Any]]:
    routes = value.get("routes")
    if isinstance(routes, list) and all(
        isinstance(route, Mapping) and isinstance(route.get("steps"), list)
        for route in routes
    ):
        raw_routes = [dict(route) for route in routes]
    else:
        raw_routes = [
            route.to_dict()
        for route in route_candidates_from_chem_enzy_result(
            dict(value), target_smiles=target_smiles
        )
        ]
    return [
        _normalize_proposal_route(route, route_index=index)
        for index, route in enumerate(raw_routes, start=1)
    ]


def compile_chemenzy_route_fingerprints(
    value: Mapping[str, Any], *, target_smiles: str
) -> dict[str, Any]:
    """Compile provider-output fingerprints without granting route authority."""

    routes = _normalized_routes(value, target_smiles=target_smiles)
    quarantined = normalized_quarantined_routes(
        value,
        start_index=len(routes) + 1,
        normalizer=_normalize_proposal_route,
    )
    rows = []
    for route, is_quarantined in [
        *((route, False) for route in routes),
        *((route, True) for route in quarantined),
    ]:
        rows.append(
            {
                "route_index": route.get("route_index"),
                "route_trace_id": str(route.get("route_trace_id") or ""),
                "raw_route_sha256": str(route.get("raw_route_sha256") or ""),
                "normalized_route_sha256": str(
                    route.get("normalized_route_sha256") or ""
                ),
                "proposal_eligible": route.get("proposal_eligible") is True,
                "quarantined": is_quarantined,
                "reasons": list(route.get("admission_reasons") or []),
                "step_count": len(route.get("steps") or []),
            }
        )
    return {
        "schema_version": "chemenzy_route_fingerprint_set.v1",
        "target_smiles": str(target_smiles),
        "raw_result_sha256": _content_sha256(value),
        "route_count": len(routes),
        "quarantined_route_count": len(quarantined),
        "routes": rows,
        "semantics": {
            "proposal_only": True,
            "fingerprints_grant_no_route_or_stock_authority": True,
        },
    }


def _normalize_proposal_route(
    route: Mapping[str, Any], *, route_index: int
) -> dict[str, Any]:
    """Translate old and current launcher schemas into proposal-only rows."""

    normalized_steps: list[dict[str, Any]] = []
    admission_reasons: set[str] = set()
    for step_index, raw_step in enumerate(route.get("steps") or [], start=1):
        if not isinstance(raw_step, Mapping):
            admission_reasons.add("invalid_step_payload")
            continue
        step = dict(raw_step)
        product = str(step.get("product_smiles") or step.get("product") or "").strip()
        reactants = _proposal_reactants(step)
        audit = audit_retrosynthetic_candidate(product, reactants)
        if audit.get("accepted") is not True:
            admission_reasons.update(
                str(reason) for reason in audit.get("reasons") or []
            )
        normalized_steps.append(
            {
                "step_index": step_index,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "rxn_smiles": str(
                    step.get("rxn_smiles")
                    or step.get("reaction_smiles")
                    or ""
                ),
                "source_model": str(
                    step.get("source_model")
                    or step.get("model")
                    or "ChemEnzyRetroPlanner"
                ),
                "score": step.get("score", step.get("confidence")),
                "stock_status": dict(step.get("stock_status") or {}),
                "condition_predictions": list(
                    step.get("condition_predictions") or []
                ),
                "enzyme_ec_annotations": [
                    dict(value)
                    for value in step.get("enzyme_ec_annotations") or []
                    if isinstance(value, Mapping)
                ],
                "catalyst_annotations": [
                    dict(value)
                    for value in step.get("catalyst_annotations") or []
                    if isinstance(value, Mapping)
                ],
                "raw_backend_metadata": _json_safe_copy(
                    step.get("raw_backend_metadata") or {}
                ),
                "is_enzymatic": bool(
                    step.get("is_enzymatic")
                    or step.get("enzyme_ec_annotations")
                ),
                "chemical_step_equivalent_count": step.get(
                    "chemical_step_equivalent_count"
                ),
                "replaced_step_ids": list(step.get("replaced_step_ids") or []),
                "selectivity_objective": str(
                    step.get("selectivity_objective") or ""
                ),
                "host_search_admission": {
                    "accepted": audit.get("accepted") is True,
                    "edge_digest": str(audit.get("edge_digest") or ""),
                    "reasons": list(audit.get("reasons") or []),
                    "not_reaction_proof": True,
                },
            }
        )
    if not normalized_steps:
        admission_reasons.add("missing_route_steps")
    normalized = {
        "route_index": route_index,
        "steps": normalized_steps,
        "score": route.get("score", route.get("confidence")),
        "stock_status": dict(route.get("stock_status") or {}),
        "search_time_s": route.get("search_time_s"),
        "route_rank": route.get("route_rank", route_index - 1),
        "raw_backend_metadata": _json_safe_copy(
            route.get("raw_backend_metadata") or {}
        ),
        "proposal_eligible": bool(normalized_steps) and not admission_reasons,
        "admission_reasons": sorted(admission_reasons),
        "backend_route_status": {
            "solved": route.get("solved"),
            "status": route.get("status"),
            "diagnostic_only": True,
        },
        "semantics": {
            "proposal_only": True,
            "host_search_admission_is_not_reaction_proof": True,
            "backend_solved_is_not_admission_authority": True,
        },
    }
    normalized["raw_route_sha256"] = _content_sha256(route)
    normalized["normalized_route_sha256"] = _content_sha256(normalized)
    normalized["route_trace_id"] = (
        f"chemenzy-route:{normalized['raw_route_sha256'][:24]}"
    )
    return normalized


def _select_host_route_portfolio(
    routes: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select a bounded, diverse host portfolio from the provider reserve."""

    remaining = sorted(
        (dict(route) for route in routes),
        key=_route_quality_key,
    )
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < max(1, int(limit)):
        ranked: list[tuple[Any, ...]] = []
        for route in remaining:
            signature = _route_edge_signature(route)
            diversity = (
                min(
                    _signature_distance(signature, _route_edge_signature(other))
                    for other in selected
                )
                if selected
                else 1.0
            )
            root = next(iter(sorted(signature)), "")
            root_novel = bool(
                not selected
                or all(
                    root != next(iter(sorted(_route_edge_signature(other))), "")
                    for other in selected
                )
            )
            ranked.append(
                (
                    -diversity,
                    -int(root_novel),
                    _route_quality_key(route),
                    int(route.get("route_index") or 0),
                    route,
                )
            )
        chosen = min(ranked)[-1]
        selected.append(chosen)
        remaining = [
            route
            for route in remaining
            if route.get("route_index") != chosen.get("route_index")
        ]
    return selected


def _route_selection_features(route: Mapping[str, Any]) -> dict[str, Any]:
    steps = [
        dict(step)
        for step in route.get("steps") or []
        if isinstance(step, Mapping)
    ]
    products = {str(step.get("product_smiles") or "") for step in steps}
    leaves = {
        str(value)
        for step in steps
        for value in step.get("reactant_smiles") or []
        if str(value) and str(value) not in products
    }
    provider_stock = {
        str(smiles): status
        for step in steps
        for smiles, status in dict(step.get("stock_status") or {}).items()
        if str(smiles)
    }
    stock_hint_count = sum(provider_stock.get(smiles) is True for smiles in leaves)
    template_step_count = sum(
        bool(dict(step.get("raw_backend_metadata") or {}).get("template"))
        for step in steps
    )
    reaction_smiles_count = sum(bool(step.get("rxn_smiles")) for step in steps)
    scored = [
        value
        for step in steps
        if (value := _finite_float(step.get("score"))) is not None
    ]
    route_score = _finite_float(route.get("score"))
    return {
        "step_count": len(steps),
        "leaf_count": len(leaves),
        "provider_stock_closed_leaf_hint_count": stock_hint_count,
        "provider_stock_closed_leaf_hint_rate": (
            round(stock_hint_count / len(leaves), 6) if leaves else 0.0
        ),
        "template_step_count": template_step_count,
        "reaction_smiles_step_count": reaction_smiles_count,
        "mean_step_score": (
            round(sum(scored) / len(scored), 8) if scored else None
        ),
        "route_score": route_score,
        "edge_signature": sorted(_route_edge_signature(route)),
        "semantics": {
            "provider_stock_is_non_authoritative_ranking_hint": True,
            "template_presence_is_replayability_hint_not_reaction_proof": True,
        },
    }


def _route_quality_key(route: Mapping[str, Any]) -> tuple[Any, ...]:
    features = _route_selection_features(route)
    mean_score = features["mean_step_score"]
    route_score = features["route_score"]
    return (
        -float(features["provider_stock_closed_leaf_hint_rate"]),
        -int(features["template_step_count"]),
        -int(features["reaction_smiles_step_count"]),
        -(float(route_score) if route_score is not None else -1.0),
        -(float(mean_score) if mean_score is not None else -1.0),
        int(features["step_count"]),
        int(route.get("route_index") or 0),
    )


def _route_edge_signature(route: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(dict(step.get("host_search_admission") or {}).get("edge_digest") or "")
        for step in route.get("steps") or []
        if isinstance(step, Mapping)
        and str(dict(step.get("host_search_admission") or {}).get("edge_digest") or "")
    )


def _signature_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 1.0 if not union else 1.0 - len(left & right) / len(union)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


def _provider_reaction_metadata(step: Mapping[str, Any]) -> dict[str, Any]:
    raw = _json_safe_copy(step.get("raw_backend_metadata") or {})
    payload = {
        "schema_version": "chemenzy_provider_reaction_metadata.v1",
        "rxn_smiles": str(step.get("rxn_smiles") or ""),
        "source_model": str(step.get("source_model") or "ChemEnzyRetroPlanner"),
        "score": _finite_float(step.get("score")),
        "stock_status": _json_safe_copy(step.get("stock_status") or {}),
        "template": _json_safe_copy(raw.get("template")),
        "raw_backend_metadata": raw,
        "host_search_admission": _json_safe_copy(
            step.get("host_search_admission") or {}
        ),
        "semantics": {
            "provider_metadata_is_advisory": True,
            "host_template_replay_required_for_reaction_proof": True,
            "provider_stock_status_is_not_stock_authority": True,
        },
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _chemenzy_transformation_hypothesis(
    step: Mapping[str, Any],
    *,
    provider_metadata: Mapping[str, Any],
) -> str:
    source = str(step.get("source_model") or "ChemEnzyRetroPlanner")
    template = provider_metadata.get("template")
    if template:
        return (
            f"{source} reaction proposal with provider template metadata "
            f"{str(provider_metadata.get('content_sha256') or '')[:12]}"
        )
    if step.get("rxn_smiles"):
        return f"{source} reaction proposal with retained reaction SMILES"
    return f"{source} one-step retrosynthesis proposal"


def _proposal_reactants(step: Mapping[str, Any]) -> list[str]:
    values = step.get("reactant_smiles") or step.get("precursor_smiles")
    if isinstance(values, str):
        values = [part for part in values.split(".") if part]
    if not isinstance(values, (list, tuple)):
        main = str(
            step.get("main_reactant")
            or step.get("main_reactant_smiles")
            or ""
        ).strip()
        auxiliary = step.get("aux_reactants") or step.get("aux_reactant_smiles") or []
        if isinstance(auxiliary, str):
            auxiliary = [part for part in auxiliary.split(".") if part]
        values = ([main] if main else []) + list(auxiliary or [])
    return [str(value).strip() for value in values if str(value).strip()]


__all__ = ["compile_chemenzy_route_fingerprints"]
