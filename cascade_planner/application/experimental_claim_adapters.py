"""Adapt domain validation feedback into unified experiment claims."""

from __future__ import annotations

from typing import Mapping, Sequence

from cascade_planner.application.biocatalytic_program_contracts import (
    BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
    BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA,
)
from cascade_planner.application.execution_capability_feedback import (
    CAPABILITY_APPLICABILITY_FEEDBACK_SCHEMA,
    CAPABILITY_FEEDBACK_PROJECTION_SCHEMA,
)
from cascade_planner.application.execution_program_validations import (
    EXECUTION_PROGRAM_VALIDATION_SCHEMA,
)
from cascade_planner.application.experimental_claim_contracts import ExperimentalClaimError
from cascade_planner.application.experimental_claim_rows import (
    biocatalytic_claim,
    canonical_strings,
    execution_claim,
    mechanism_claim,
)
from cascade_planner.application.mechanism_experiment_feedback import (
    MECHANISM_EXPERIMENT_FEEDBACK_SCHEMA,
    MECHANISM_FEEDBACK_PROJECTION_SCHEMA,
)
from cascade_planner.application.mechanism_program_validations import (
    MECHANISM_PROGRAM_VALIDATION_SCHEMA,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


def adapt_experimental_claims(
    biocatalytic_bundle: Mapping[str, object],
    execution_feedback: Mapping[str, object],
    mechanism_feedback: Mapping[str, object],
    validations: Sequence[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """Normalize valid observations while retaining rejected validation rows."""

    bundle = _strict_digest_object(
        biocatalytic_bundle,
        expected_schema=BIOCATALYTIC_PROGRAM_BUNDLE_SCHEMA,
        label="biocatalytic_bundle",
    )
    execution = _strict_digest_object(
        execution_feedback,
        expected_schema=CAPABILITY_FEEDBACK_PROJECTION_SCHEMA,
        label="execution_feedback",
    )
    mechanism = _strict_digest_object(
        mechanism_feedback,
        expected_schema=MECHANISM_FEEDBACK_PROJECTION_SCHEMA,
        label="mechanism_feedback",
    )
    validation_index, duplicates = _validation_index(validations)
    claims: dict[str, dict[str, object]] = {}
    rejected: list[dict[str, object]] = list(duplicates)
    domain_claims, domain_rejected = _biocatalytic_claims(bundle, validation_index)
    claims.update(domain_claims)
    rejected.extend(domain_rejected)
    for domain, projection, row_schema, validation_schema in (
        (
            "execution",
            execution,
            CAPABILITY_APPLICABILITY_FEEDBACK_SCHEMA,
            EXECUTION_PROGRAM_VALIDATION_SCHEMA,
        ),
        (
            "mechanism",
            mechanism,
            MECHANISM_EXPERIMENT_FEEDBACK_SCHEMA,
            MECHANISM_PROGRAM_VALIDATION_SCHEMA,
        ),
    ):
        domain_claims, domain_rejected = _feedback_claims(
            domain,
            projection,
            row_schema=row_schema,
            validation_schema=validation_schema,
            validation_index=validation_index,
        )
        if set(claims).intersection(domain_claims):
            raise ExperimentalClaimError("experimental_claim_identity_collision")
        claims.update(domain_claims)
        rejected.extend(domain_rejected)
    return claims, _canonical_rejections(rejected)


def _biocatalytic_claims(
    bundle: Mapping[str, object],
    validation_index: Mapping[tuple[str, str], dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    claims: dict[str, dict[str, object]] = {}
    rejected: list[dict[str, object]] = []
    proposals = dict(bundle.get("program_proposals") or {})
    for program_id, raw_proposal in sorted(proposals.items()):
        proposal = dict(raw_proposal)
        gate = dict(proposal.get("validation_gate") or {})
        for raw_audit in gate.get("audits") or []:
            audit = dict(raw_audit)
            validation_id = str(audit.get("validation_id") or "")
            validation = _bound_validation(
                validation_index,
                BIOCATALYSIS_PROGRAM_VALIDATION_SCHEMA,
                validation_id,
                expected_sha256=str(audit.get("content_sha256") or ""),
                expected_program_id=str(program_id),
            )
            reasons = [str(value) for value in audit.get("reasons") or []]
            if audit.get("accepted") is True:
                claim = biocatalytic_claim(proposal, validation)
                claims[str(claim["claim_id"])] = claim
            else:
                rejected.append(
                    _rejection(
                        "biocatalytic",
                        validation_id,
                        str(program_id),
                        reasons or ["validation_not_accepted"],
                    )
                )
    rejected.extend(
        _rejection(
            "biocatalytic",
            str(validation_id),
            "",
            ["validation_program_unbound"],
        )
        for validation_id in bundle.get("unbound_validation_ids") or []
    )
    return claims, rejected


def _feedback_claims(
    domain: str,
    projection: Mapping[str, object],
    *,
    row_schema: str,
    validation_schema: str,
    validation_index: Mapping[tuple[str, str], dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    claims: dict[str, dict[str, object]] = {}
    for raw_feedback in dict(projection.get("feedback") or {}).values():
        feedback = _strict_digest_object(
            raw_feedback,
            expected_schema=row_schema,
            label=f"{domain}_feedback_row",
        )
        validation_id = str(feedback.get("validation_id") or "")
        validation = _bound_validation(
            validation_index,
            validation_schema,
            validation_id,
            expected_sha256=str(feedback.get("source_validation_sha256") or ""),
            expected_program_id=str(feedback.get("program_id") or ""),
        )
        claim = (
            execution_claim(feedback, validation)
            if domain == "execution"
            else mechanism_claim(feedback, validation)
        )
        claims[str(claim["claim_id"])] = claim
    rejected = [
        _rejection(
            domain,
            str(dict(row).get("validation_id") or ""),
            str(dict(row).get("program_id") or ""),
            [str(value) for value in dict(row).get("reasons") or []],
        )
        for row in projection.get("rejected_validations") or []
    ]
    return claims, rejected


def _validation_index(
    validations: Sequence[dict[str, object]],
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    list[dict[str, object]],
]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    for validation in validations:
        row = dict(validation)
        key = (
            str(row.get("schema_version") or ""),
            str(row.get("validation_id") or ""),
        )
        if key in index:
            duplicates.append(
                _rejection(
                    _domain_for_schema(key[0]),
                    key[1],
                    str(row.get("program_id") or ""),
                    ["validation_identity_duplicate"],
                )
            )
        else:
            index[key] = row
    return index, duplicates


def _bound_validation(
    index: Mapping[tuple[str, str], dict[str, object]],
    schema: str,
    validation_id: str,
    *,
    expected_sha256: str,
    expected_program_id: str,
) -> dict[str, object]:
    row = dict(index.get((schema, validation_id)) or {})
    if not row:
        raise ExperimentalClaimError("experimental_claim_source_validation_missing")
    material = dict(row)
    observed = str(material.pop("content_sha256", ""))
    if (
        observed != expected_sha256
        or observed != strict_canonical_json_sha256(material)
        or row.get("program_id") != expected_program_id
    ):
        raise ExperimentalClaimError("experimental_claim_source_validation_mismatch")
    return row


def _strict_digest_object(
    value: Mapping[str, object], *, expected_schema: str, label: str
) -> dict[str, object]:
    row = dict(value)
    material = dict(row)
    observed = str(material.pop("content_sha256", ""))
    if row.get("schema_version") != expected_schema:
        raise ExperimentalClaimError(f"experimental_claim_{label}_schema_invalid")
    if not observed or observed != strict_canonical_json_sha256(material):
        raise ExperimentalClaimError(f"experimental_claim_{label}_digest_invalid")
    return row


def _domain_for_schema(schema: str) -> str:
    if schema == EXECUTION_PROGRAM_VALIDATION_SCHEMA:
        return "execution"
    if schema == MECHANISM_PROGRAM_VALIDATION_SCHEMA:
        return "mechanism"
    return "biocatalytic"


def _rejection(
    domain: str, validation_id: str, program_id: str, reasons: Sequence[str]
) -> dict[str, object]:
    return {
        "domain": domain,
        "validation_id": validation_id,
        "program_id": program_id,
        "reasons": canonical_strings(reasons) or ["validation_rejected"],
    }


def _canonical_rejections(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    unique = {strict_canonical_json_sha256(dict(row)): dict(row) for row in rows}
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["domain"]),
            str(row["validation_id"]),
            str(row["program_id"]),
            tuple(row["reasons"]),
        ),
    )


__all__ = ["adapt_experimental_claims"]
