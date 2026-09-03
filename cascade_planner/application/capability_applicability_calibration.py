"""Derive exact-boundary applicability and dirty-domain hints from experiment Claims."""

from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.experimental_claim_contracts import (
    ExperimentalClaimError,
    validate_experimental_claim_set,
    with_experimental_claim_digest,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


CAPABILITY_CALIBRATION_SCHEMA = "capability_applicability_calibration.v1"
EXACT_BOUNDARY_APPLICABILITY_SCHEMA = "exact_boundary_applicability.v1"
CAPABILITY_CALIBRATION_ORACLE_SCHEMA = "capability_calibration_oracle.v1"
CALIBRATION_SEMANTICS = {
    "projection_is_read_only": True,
    "calibration_scope_is_exact_boundary_only": True,
    "positive_negative_and_inconclusive_counts_remain_separate": True,
    "dirty_domains_are_recompute_hints_not_mutations": True,
    "capability_catalog_is_not_mutated_or_disabled": True,
    "mechanism_hypotheses_are_not_promoted_to_capabilities": True,
    "projection_cannot_grant_proof_completion_or_acceptance": True,
}


class CapabilityCalibrationError(ValueError):
    """Experimental Claims cannot be safely calibrated."""


def compile_capability_applicability_calibration(
    claim_set: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate only exact matching boundaries and identify changed domains."""

    current = dict(claim_set)
    reasons = validate_experimental_claim_set(current)
    if reasons:
        raise CapabilityCalibrationError(
            "capability_calibration_claim_set_invalid:" + ",".join(reasons)
        )
    previous_rows: dict[str, dict[str, Any]] = {}
    previous_digest = ""
    if previous is not None:
        prior = dict(previous)
        _validate_calibration(prior)
        if prior.get("run_id") != current.get("run_id") or prior.get("route_id") != current.get(
            "route_id"
        ):
            raise CapabilityCalibrationError("capability_calibration_previous_identity_mismatch")
        previous_rows = {
            str(key): dict(value) for key, value in dict(prior.get("calibrations") or {}).items()
        }
        previous_digest = str(prior.get("content_sha256") or "")
    groups: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for raw_claim in dict(current.get("claims") or {}).values():
        claim = dict(raw_claim)
        identity = _scope_identity(claim)
        key = strict_canonical_json_sha256(identity)
        groups.setdefault(key, []).append(claim)
        identities[key] = identity
    calibrations: dict[str, dict[str, Any]] = {}
    for key, claims in sorted(groups.items()):
        row = _calibration_row(identities[key], claims)
        calibrations[row["calibration_id"]] = row
    dirty = _dirty_domain_hints(previous_rows, calibrations)
    payload = {
        "schema_version": CAPABILITY_CALIBRATION_SCHEMA,
        "run_id": str(current.get("run_id") or ""),
        "route_id": str(current.get("route_id") or ""),
        "source_claim_set_sha256": str(current.get("content_sha256") or ""),
        "previous_calibration_sha256": previous_digest,
        "calibrations": calibrations,
        "dirty_domain_hints": dirty,
        "counts": {
            "calibrations": len(calibrations),
            "positive": sum(
                row["applicability_status"] == "positive_exact_boundary_observed"
                for row in calibrations.values()
            ),
            "negative": sum(
                row["applicability_status"] == "negative_exact_boundary_observed"
                for row in calibrations.values()
            ),
            "conflicting": sum(
                row["applicability_status"] == "conflicting_exact_boundary_observations"
                for row in calibrations.values()
            ),
            "inconclusive_only": sum(
                row["applicability_status"] == "inconclusive_exact_boundary_observed"
                for row in calibrations.values()
            ),
            "dirty_domains": len(dirty),
            "catalog_mutations": 0,
        },
        "semantics": dict(CALIBRATION_SEMANTICS),
    }
    result = with_experimental_claim_digest(payload)
    _validate_calibration(result)
    return result


def capability_calibration_oracle(
    claim_set: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompile calibration and compare all scopes and dirty hints."""

    try:
        expected = compile_capability_applicability_calibration(claim_set, previous=previous)
        observed_value = dict(observed)
        _validate_calibration(observed_value)
    except (CapabilityCalibrationError, ExperimentalClaimError, TypeError, ValueError) as exc:
        return _oracle_result(
            False,
            {"inputs_reprojectable": False},
            [f"capability_calibration_inputs_invalid:{type(exc).__name__}"],
            "",
            "",
        )
    checks = {
        "inputs_reprojectable": True,
        "projection_equal": observed_value == expected,
        "authority_semantics_equal": observed_value.get("semantics") == CALIBRATION_SEMANTICS,
    }
    reasons = [key for key, accepted in checks.items() if accepted is not True]
    return _oracle_result(
        not reasons,
        checks,
        reasons,
        str(expected["content_sha256"]),
        str(observed_value.get("content_sha256") or ""),
    )


def _calibration_row(identity: Mapping[str, Any], claims: list[dict[str, Any]]) -> dict[str, Any]:
    polarities = [str(claim.get("polarity") or "") for claim in claims]
    positive = polarities.count("positive")
    negative = polarities.count("negative")
    inconclusive = polarities.count("inconclusive")
    if positive and negative:
        status = "conflicting_exact_boundary_observations"
    elif negative:
        status = "negative_exact_boundary_observed"
    elif positive:
        status = "positive_exact_boundary_observed"
    else:
        status = "inconclusive_exact_boundary_observed"
    calibration_id = "exact-boundary:" + strict_canonical_json_sha256(identity)[:32]
    return with_experimental_claim_digest(
        {
            "schema_version": EXACT_BOUNDARY_APPLICABILITY_SCHEMA,
            "calibration_id": calibration_id,
            "domain": str(identity["domain"]),
            "subject_refs": dict(identity["subject_refs"]),
            "boundary": dict(identity["boundary"]),
            "scope_signature_sha256": str(identity["scope_signature_sha256"]),
            "claim_ids": sorted(str(claim["claim_id"]) for claim in claims),
            "evidence_counts": {
                "positive": positive,
                "negative": negative,
                "inconclusive": inconclusive,
            },
            "interpretation_statuses": sorted(
                {
                    str(claim.get("interpretation_status") or "")
                    for claim in claims
                    if str(claim.get("interpretation_status") or "")
                }
            ),
            "applicability_status": status,
            "generalization_scope": "exact_boundary_only",
            "catalog_mutated": False,
            "capability_disabled": False,
        }
    )


def _scope_identity(claim: Mapping[str, Any]) -> dict[str, Any]:
    context = dict(claim.get("domain_context") or {})
    signature = ""
    if claim.get("domain") == "execution":
        signature = str(context.get("operation_sequence_sha256") or "")
    elif claim.get("domain") == "mechanism":
        signature = str(context.get("mechanism_signature_sha256") or "")
    return {
        "domain": str(claim.get("domain") or ""),
        "subject_refs": dict(claim.get("subject_refs") or {}),
        "boundary": dict(claim.get("boundary") or {}),
        "scope_signature_sha256": signature,
    }


def _dirty_domain_hints(
    previous: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    for calibration_id in sorted(set(previous).union(current)):
        before = previous.get(calibration_id)
        after = current.get(calibration_id)
        if before == after:
            continue
        change = "created" if before is None else "removed" if after is None else "changed"
        hints.append(
            {
                "calibration_id": calibration_id,
                "change_kind": change,
                "reason": "exact_boundary_observation_projection_changed",
                "recompute_scope": "exact_boundary_only",
            }
        )
    return hints


def _validate_calibration(value: Mapping[str, Any]) -> None:
    row = dict(value)
    material = dict(row)
    observed = str(material.pop("content_sha256", ""))
    if (
        row.get("schema_version") != CAPABILITY_CALIBRATION_SCHEMA
        or row.get("semantics") != CALIBRATION_SEMANTICS
        or not observed
        or observed != strict_canonical_json_sha256(material)
        or not isinstance(row.get("calibrations"), dict)
        or not isinstance(row.get("dirty_domain_hints"), list)
    ):
        raise CapabilityCalibrationError("capability_calibration_contract_invalid")
    for calibration_id, raw in row["calibrations"].items():
        item = dict(raw)
        item_material = dict(item)
        item_digest = str(item_material.pop("content_sha256", ""))
        if (
            item.get("schema_version") != EXACT_BOUNDARY_APPLICABILITY_SCHEMA
            or item.get("calibration_id") != calibration_id
            or item.get("generalization_scope") != "exact_boundary_only"
            or item.get("catalog_mutated") is not False
            or item.get("capability_disabled") is not False
            or item_digest != strict_canonical_json_sha256(item_material)
        ):
            raise CapabilityCalibrationError("capability_calibration_row_contract_invalid")


def _oracle_result(
    accepted: bool,
    checks: Mapping[str, bool],
    reasons: list[str],
    expected_digest: str,
    observed_digest: str,
) -> dict[str, Any]:
    return with_experimental_claim_digest(
        {
            "schema_version": CAPABILITY_CALIBRATION_ORACLE_SCHEMA,
            "accepted": accepted,
            "checks": dict(checks),
            "reasons": reasons,
            "expected_calibration_sha256": expected_digest,
            "observed_calibration_sha256": observed_digest,
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_catalog_mutation_or_generalization": True,
            },
        }
    )


__all__ = [
    "CAPABILITY_CALIBRATION_ORACLE_SCHEMA",
    "CAPABILITY_CALIBRATION_SCHEMA",
    "CALIBRATION_SEMANTICS",
    "CapabilityCalibrationError",
    "capability_calibration_oracle",
    "compile_capability_applicability_calibration",
]
