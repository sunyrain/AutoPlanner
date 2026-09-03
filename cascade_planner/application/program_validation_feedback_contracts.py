"""Shared binding collection for read-only Program validation feedback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Mapping


class ProgramValidationFeedbackError(ValueError):
    """A feedback projection is not bound to its compiled Program bundle."""


def collect_program_validation_feedback(
    bundle: Mapping[str, Any],
    validations: Sequence[dict[str, Any]],
    *,
    gate_factory: Callable[[Mapping[str, Any], Sequence[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    proposals = dict(bundle.get("program_proposals") or {})
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for program_id, raw_proposal in sorted(proposals.items()):
        proposal = dict(raw_proposal)
        matching = [row for row in validations if row.get("program_id") == program_id]
        gate = gate_factory(proposal, validations)
        required_checks = list(
            dict(proposal.get("validation_plan") or {}).get("required_checks") or []
        )
        expected_plan = {
            "required_checks": required_checks,
            **gate,
            "grants_validation": gate["accepted"] is True,
        }
        if dict(proposal.get("validation_plan") or {}) != expected_plan:
            raise ProgramValidationFeedbackError("program_feedback_validation_binding_mismatch")
        for validation, audit in zip(matching, gate["audits"], strict=True):
            if audit["feedback_eligible"] is True:
                observations.append(
                    {
                        "proposal": proposal,
                        "validation": validation,
                        "audit": audit,
                    }
                )
            else:
                rejected.append(
                    {
                        "validation_id": str(validation.get("validation_id") or ""),
                        "program_id": str(program_id),
                        "reasons": list(audit["reasons"]),
                    }
                )
    unbound_ids = sorted(
        str(row.get("validation_id") or "")
        for row in validations
        if str(row.get("program_id") or "") not in proposals
    )
    if list(bundle.get("unbound_validation_ids") or []) != unbound_ids:
        raise ProgramValidationFeedbackError("program_feedback_unbound_validation_mismatch")
    rejected.extend(
        {
            "validation_id": validation_id,
            "program_id": "",
            "reasons": ["validation_program_unbound"],
        }
        for validation_id in unbound_ids
    )
    return {
        "observations": observations,
        "rejected_validations": sorted(
            rejected,
            key=lambda row: (
                str(row["validation_id"]),
                str(row["program_id"]),
                tuple(row["reasons"]),
            ),
        ),
        "unbound_validation_ids": unbound_ids,
    }


def validation_feedback_polarity(outcome_status: str, *, accepted: bool) -> str:
    if accepted:
        return "positive"
    if outcome_status == "failure":
        return "negative"
    return "inconclusive"


__all__ = [
    "ProgramValidationFeedbackError",
    "collect_program_validation_feedback",
    "validation_feedback_polarity",
]
