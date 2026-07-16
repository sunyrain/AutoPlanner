"""Deterministic multi-profile Pareto optimization over Program route candidates."""

from __future__ import annotations

import json
from typing import Any, Mapping

from cascade_planner.application.pareto import dominates, pareto_layers
from cascade_planner.application.program_route_candidate_contracts import (
    ProgramRouteCandidateError,
    validate_program_route_candidate_set,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_ROUTE_PORTFOLIO_SCHEMA = "program_route_portfolio.v1"
PROGRAM_ROUTE_PORTFOLIO_ORACLE_SCHEMA = "program_route_portfolio_oracle.v1"

OBJECTIVES = (
    ("minimum_proof_level", "maximize"),
    ("physical_operation_count", "minimize"),
    ("reaction_validation_deficit_count", "minimize"),
    ("condition_deficit_count", "minimize"),
    ("specialized_validation_deficit_count", "minimize"),
    ("procurement_deficit_count", "minimize"),
    ("process_deficit_count", "minimize"),
    ("source_deficit_count", "minimize"),
    ("cofactor_system_count", "minimize"),
    ("risk_burden", "minimize"),
    ("risk_data_deficit_count", "minimize"),
)
PROFILES = (
    ("exploration", "exploration_visible"),
    ("shadow_optimizer", "shadow_optimizer"),
    ("experimental_ready", "experimental_ready"),
    ("process_ready", "process_ready"),
)
_SEMANTICS = {
    "read_only_portfolio": True,
    "pareto_layers_have_no_scalar_weighted_best": True,
    "source_kind_is_not_an_objective": True,
    "low_evidence_candidates_remain_visible_in_exploration": True,
    "profile_eligibility_precedes_pareto_comparison": True,
    "unmodeled_objectives_are_reported_not_imputed": True,
    "portfolio_cannot_grant_proof_completion_or_production_authority": True,
    "edge_ids_remain_production_route_authority": True,
}


def optimize_program_route_candidates(candidate_set: Mapping[str, Any]) -> dict[str, Any]:
    """Build Pareto layers for visibility, shadow, experimental, and process profiles."""

    source = _object(candidate_set, "candidate_set")
    reasons = validate_program_route_candidate_set(source)
    if reasons:
        raise ProgramRouteCandidateError(";".join(reasons))
    candidates = dict(source["candidates"])
    objective_names = [name for name, _direction in OBJECTIVES]
    directions = [direction for _name, direction in OBJECTIVES]
    vectors = {
        candidate_id: [float(dict(row["metrics"])[name]) for name in objective_names]
        for candidate_id, row in sorted(candidates.items())
    }
    profiles: dict[str, dict[str, Any]] = {}
    memberships: dict[str, dict[str, int | None]] = {
        candidate_id: {} for candidate_id in candidates
    }
    for profile_name, eligibility_key in PROFILES:
        eligible = sorted(
            candidate_id
            for candidate_id, row in candidates.items()
            if dict(row["eligibility"]).get(eligibility_key) is True
        )
        layers = pareto_layers(
            {candidate_id: vectors[candidate_id] for candidate_id in eligible},
            directions=directions,
        )
        for candidate_id in candidates:
            memberships[candidate_id][profile_name] = next(
                (
                    index
                    for index, layer in enumerate(layers, start=1)
                    if candidate_id in layer
                ),
                None,
            )
        profiles[profile_name] = {
            "eligibility_key": eligibility_key,
            "eligible_candidate_ids": eligible,
            "pareto_front_ids": layers[0] if layers else [],
            "pareto_layers": layers,
            "counts": {
                "eligible": len(eligible),
                "pareto_front": len(layers[0]) if layers else 0,
                "layers": len(layers),
            },
            "empty_reason": "" if eligible else f"no_{profile_name}_candidates",
        }
    exploration_ids = profiles["exploration"]["eligible_candidate_ids"]
    evaluations = {
        candidate_id: {
            "candidate_id": candidate_id,
            "source_kind": str(candidates[candidate_id]["source_kind"]),
            "objective_vector": {
                name: dict(candidates[candidate_id]["metrics"])[name]
                for name in objective_names
            },
            "profile_pareto_layers": memberships[candidate_id],
            "dominates_in_exploration": sorted(
                other_id
                for other_id in exploration_ids
                if other_id != candidate_id
                and dominates(
                    vectors[candidate_id], vectors[other_id], directions=directions
                )
            ),
            "dominated_by_in_exploration": sorted(
                other_id
                for other_id in exploration_ids
                if other_id != candidate_id
                and dominates(
                    vectors[other_id], vectors[candidate_id], directions=directions
                )
            ),
        }
        for candidate_id in sorted(candidates)
    }
    return _with_digest(
        {
            "schema_version": PROGRAM_ROUTE_PORTFOLIO_SCHEMA,
            "run_id": str(source.get("run_id") or ""),
            "route_id": str(source.get("route_id") or ""),
            "source_candidate_set_sha256": str(source["content_sha256"]),
            "objective_definitions": [
                {"metric": name, "direction": direction}
                for name, direction in OBJECTIVES
            ],
            "unmodeled_objectives": list(source.get("unmodeled_objectives") or []),
            "candidate_evaluations": evaluations,
            "profiles": profiles,
            "counts": {
                "candidates": len(candidates),
                "exploration_pareto_front": profiles["exploration"]["counts"][
                    "pareto_front"
                ],
                "shadow_optimizer_pareto_front": profiles["shadow_optimizer"][
                    "counts"
                ]["pareto_front"],
                "experimental_ready_pareto_front": profiles["experimental_ready"][
                    "counts"
                ]["pareto_front"],
                "process_ready_pareto_front": profiles["process_ready"]["counts"][
                    "pareto_front"
                ],
            },
            "semantics": dict(_SEMANTICS),
        }
    )


def program_route_portfolio_oracle(
    candidate_set: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute a portfolio exactly and reject modified fronts or objectives."""

    try:
        expected = optimize_program_route_candidates(candidate_set)
        observed_value = _object(observed, "observed_portfolio")
    except (ProgramRouteCandidateError, TypeError, ValueError) as exc:
        return _oracle(
            False,
            {"inputs_reprojectable": False},
            [f"program_portfolio_inputs_invalid:{type(exc).__name__}"],
            "",
            "",
        )
    material = dict(observed_value)
    observed_digest = str(material.pop("content_sha256", ""))
    checks = {
        "inputs_reprojectable": True,
        "schema_equal": observed_value.get("schema_version")
        == PROGRAM_ROUTE_PORTFOLIO_SCHEMA,
        "content_digest_valid": observed_digest
        == strict_canonical_json_sha256(material),
        "portfolio_equal": observed_value == expected,
        "authority_semantics_equal": observed_value.get("semantics") == _SEMANTICS,
    }
    reasons = [key for key, accepted in checks.items() if accepted is not True]
    return _oracle(
        not reasons,
        checks,
        reasons,
        str(expected["content_sha256"]),
        observed_digest,
    )


def _oracle(
    accepted: bool,
    checks: Mapping[str, bool],
    reasons: list[str],
    expected_digest: str,
    observed_digest: str,
) -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": PROGRAM_ROUTE_PORTFOLIO_ORACLE_SCHEMA,
            "accepted": accepted,
            "checks": dict(checks),
            "reasons": sorted(set(reasons)),
            "expected_portfolio_sha256": expected_digest,
            "observed_portfolio_sha256": observed_digest,
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_cannot_select_a_production_route": True,
            },
        }
    )


def _object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        copied = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ProgramRouteCandidateError(f"program_optimizer_{label}_not_strict_json") from exc
    if not isinstance(copied, dict):
        raise ProgramRouteCandidateError(f"program_optimizer_{label}_not_object")
    return copied


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


__all__ = [
    "OBJECTIVES",
    "PROFILES",
    "PROGRAM_ROUTE_PORTFOLIO_ORACLE_SCHEMA",
    "PROGRAM_ROUTE_PORTFOLIO_SCHEMA",
    "optimize_program_route_candidates",
    "program_route_portfolio_oracle",
]
