"""Compile domain validation outcomes into one read-only experiment Claim set."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.biocatalytic_program_contracts import (
    BIOCATALYTIC_PROGRAM_ORACLE_SCHEMA,
)
from cascade_planner.application.execution_capability_feedback import (
    CAPABILITY_FEEDBACK_ORACLE_SCHEMA,
)
from cascade_planner.application.experimental_claim_adapters import (
    adapt_experimental_claims,
)
from cascade_planner.application.experimental_claim_contracts import (
    CLAIM_SET_SEMANTICS,
    EXPERIMENTAL_CLAIM_SET_ORACLE_SCHEMA,
    EXPERIMENTAL_CLAIM_SET_SCHEMA,
    ExperimentalClaimError,
    experimental_claim_counts,
    validate_experimental_claim_set,
    with_experimental_claim_digest,
)
from cascade_planner.application.mechanism_experiment_feedback import (
    MECHANISM_FEEDBACK_ORACLE_SCHEMA,
)
from cascade_planner.application.program_innovation_contracts import (
    ProgramInnovationContractError,
    strict_program_innovation_object,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def compile_experimental_claim_set(
    biocatalytic_bundle: Mapping[str, Any],
    biocatalytic_oracle: Mapping[str, Any],
    execution_feedback: Mapping[str, Any],
    execution_oracle: Mapping[str, Any],
    mechanism_feedback: Mapping[str, Any],
    mechanism_oracle: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Normalize exact-boundary observations without promoting their authority."""

    try:
        bundle = strict_program_innovation_object(biocatalytic_bundle, "biocatalytic_bundle")
        execution = strict_program_innovation_object(execution_feedback, "execution_feedback")
        mechanism = strict_program_innovation_object(mechanism_feedback, "mechanism_feedback")
        bio_oracle = strict_program_innovation_object(biocatalytic_oracle, "biocatalytic_oracle")
        exec_oracle = strict_program_innovation_object(execution_oracle, "execution_oracle")
        mech_oracle = strict_program_innovation_object(mechanism_oracle, "mechanism_oracle")
        rows = [strict_program_innovation_object(value, "validation") for value in validations]
    except ProgramInnovationContractError as exc:
        raise ExperimentalClaimError(str(exc)) from exc
    _validate_oracle_binding(
        bio_oracle,
        expected_schema=BIOCATALYTIC_PROGRAM_ORACLE_SCHEMA,
        source_digest=str(bundle.get("content_sha256") or ""),
        expected_key="expected_bundle_sha256",
        observed_key="observed_bundle_sha256",
        label="biocatalytic",
    )
    _validate_oracle_binding(
        exec_oracle,
        expected_schema=CAPABILITY_FEEDBACK_ORACLE_SCHEMA,
        source_digest=str(execution.get("content_sha256") or ""),
        expected_key="expected_feedback_sha256",
        observed_key="observed_feedback_sha256",
        label="execution",
    )
    _validate_oracle_binding(
        mech_oracle,
        expected_schema=MECHANISM_FEEDBACK_ORACLE_SCHEMA,
        source_digest=str(mechanism.get("content_sha256") or ""),
        expected_key="expected_feedback_sha256",
        observed_key="observed_feedback_sha256",
        label="mechanism",
    )
    run_id = str(bundle.get("run_id") or "")
    route_id = str(bundle.get("source_route_id") or "")
    if not run_id or not route_id:
        raise ExperimentalClaimError("experimental_claim_source_identity_missing")
    if {
        str(execution.get("run_id") or ""),
        str(mechanism.get("run_id") or ""),
    } != {run_id} or {
        str(execution.get("route_id") or ""),
        str(mechanism.get("route_id") or ""),
    } != {route_id}:
        raise ExperimentalClaimError("experimental_claim_source_identity_mismatch")
    claims, rejected = adapt_experimental_claims(
        bundle,
        execution,
        mechanism,
        rows,
    )
    result = with_experimental_claim_digest(
        {
            "schema_version": EXPERIMENTAL_CLAIM_SET_SCHEMA,
            "run_id": run_id,
            "route_id": route_id,
            "source_artifacts": {
                "biocatalytic_bundle_sha256": str(bundle["content_sha256"]),
                "biocatalytic_oracle_sha256": str(bio_oracle["content_sha256"]),
                "execution_feedback_sha256": str(execution["content_sha256"]),
                "execution_oracle_sha256": str(exec_oracle["content_sha256"]),
                "mechanism_feedback_sha256": str(mechanism["content_sha256"]),
                "mechanism_oracle_sha256": str(mech_oracle["content_sha256"]),
                "validation_pack_sha256": _validation_pack_sha256(rows),
            },
            "claims": claims,
            "rejected_validations": rejected,
            "counts": experimental_claim_counts(claims, rejected),
            "semantics": dict(CLAIM_SET_SEMANTICS),
        }
    )
    reasons = validate_experimental_claim_set(result)
    if reasons:
        raise ExperimentalClaimError("experimental_claim_set_invalid:" + ",".join(reasons))
    return result


def experimental_claim_set_oracle(
    biocatalytic_bundle: Mapping[str, Any],
    biocatalytic_oracle: Mapping[str, Any],
    execution_feedback: Mapping[str, Any],
    execution_oracle: Mapping[str, Any],
    mechanism_feedback: Mapping[str, Any],
    mechanism_oracle: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    validations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Recompile the complete Claim set and compare it byte-for-byte."""

    try:
        rows = [dict(value) for value in validations]
        expected = compile_experimental_claim_set(
            biocatalytic_bundle,
            biocatalytic_oracle,
            execution_feedback,
            execution_oracle,
            mechanism_feedback,
            mechanism_oracle,
            validations=rows,
        )
        observed_value = strict_program_innovation_object(observed, "experimental_claim_set")
    except (
        ExperimentalClaimError,
        ProgramInnovationContractError,
        TypeError,
        ValueError,
    ) as exc:
        return _oracle_result(
            False,
            {"inputs_reprojectable": False},
            [f"experimental_claim_inputs_invalid:{type(exc).__name__}"],
            "",
            "",
        )
    material = dict(observed_value)
    observed_digest = str(material.pop("content_sha256", ""))
    checks = {
        "inputs_reprojectable": True,
        "schema_equal": observed_value.get("schema_version") == EXPERIMENTAL_CLAIM_SET_SCHEMA,
        "content_digest_valid": observed_digest == strict_canonical_json_sha256(material),
        "projection_equal": observed_value == expected,
        "authority_semantics_equal": observed_value.get("semantics") == CLAIM_SET_SEMANTICS,
    }
    reasons = [key for key, accepted in checks.items() if accepted is not True]
    return _oracle_result(
        not reasons,
        checks,
        reasons,
        str(expected["content_sha256"]),
        observed_digest,
    )


def _validate_oracle_binding(
    oracle: Mapping[str, Any],
    *,
    expected_schema: str,
    source_digest: str,
    expected_key: str,
    observed_key: str,
    label: str,
) -> None:
    material = dict(oracle)
    observed_digest = str(material.pop("content_sha256", ""))
    if (
        oracle.get("schema_version") != expected_schema
        or observed_digest != strict_canonical_json_sha256(material)
        or oracle.get("accepted") is not True
        or oracle.get(expected_key) != source_digest
        or oracle.get(observed_key) != source_digest
    ):
        raise ExperimentalClaimError(f"experimental_claim_{label}_oracle_binding_invalid")


def _validation_pack_sha256(rows: list[dict[str, Any]]) -> str:
    return strict_canonical_json_sha256(
        {
            "schema_version": "experimental_validation_pack.v1",
            "validations": sorted(
                rows,
                key=lambda row: (
                    str(row.get("schema_version") or ""),
                    str(row.get("validation_id") or ""),
                    str(row.get("content_sha256") or ""),
                ),
            ),
        }
    )


def _oracle_result(
    accepted: bool,
    checks: Mapping[str, bool],
    reasons: list[str],
    expected_digest: str,
    observed_digest: str,
) -> dict[str, Any]:
    return with_experimental_claim_digest(
        {
            "schema_version": EXPERIMENTAL_CLAIM_SET_ORACLE_SCHEMA,
            "accepted": accepted,
            "checks": dict(checks),
            "reasons": reasons,
            "expected_claim_set_sha256": expected_digest,
            "observed_claim_set_sha256": observed_digest,
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_scientific_authority": True,
            },
        }
    )


__all__ = ["compile_experimental_claim_set", "experimental_claim_set_oracle"]
