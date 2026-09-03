"""Read-only reprojection oracle for Program applicability models."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cascade_planner.application.program_applicability import (
    APPLICABILITY_SEMANTICS,
    compile_program_applicability_model,
)
from cascade_planner.runtime.canonical_json import strict_canonical_json_sha256


PROGRAM_APPLICABILITY_ORACLE_SCHEMA = "program_applicability_model_oracle.v1"


def program_applicability_model_oracle(
    candidate: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproject one applicability model and reject semantic or digest drift."""

    try:
        expected = compile_program_applicability_model(candidate, records)
        observed_value = dict(observed)
        checks = {
            "inputs_reprojectable": True,
            "projection_equal": observed_value == expected,
            "digest_valid": not observed_value or _digest_valid(observed_value),
            "authority_semantics_equal": (
                not observed_value
                or observed_value.get("semantics") == APPLICABILITY_SEMANTICS
            ),
        }
    except (TypeError, ValueError):
        expected = {}
        observed_value = dict(observed) if isinstance(observed, Mapping) else {}
        checks = {"inputs_reprojectable": False}
    reasons = [key for key, accepted in checks.items() if accepted is not True]
    return _with_digest(
        {
            "schema_version": PROGRAM_APPLICABILITY_ORACLE_SCHEMA,
            "accepted": not reasons,
            "checks": checks,
            "reasons": reasons,
            "expected_model_sha256": str(expected.get("content_sha256") or ""),
            "observed_model_sha256": str(observed_value.get("content_sha256") or ""),
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_validation_or_catalog_authority": True,
            },
        }
    )


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = strict_canonical_json_sha256(row)
    return row


def _digest_valid(value: Mapping[str, Any]) -> bool:
    material = dict(value)
    observed = str(material.pop("content_sha256", ""))
    try:
        return bool(observed) and observed == strict_canonical_json_sha256(material)
    except (TypeError, ValueError):
        return False


__all__ = ["program_applicability_model_oracle"]
