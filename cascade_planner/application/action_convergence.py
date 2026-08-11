"""Replayable convergence ledger derived from durable campaign Action records."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


ACTION_CONVERGENCE_LEDGER_SCHEMA = "campaign_action_convergence_ledger.v1"


def compile_action_convergence_ledger(
    executions: Iterable[Mapping[str, Any]],
    *,
    current_graph_revision: int,
) -> dict[str, Any]:
    """Project attempted work and trailing no-gain state without new authority."""

    current_revision = max(0, int(current_graph_revision))
    rows = sorted(
        (dict(value) for value in executions),
        key=lambda row: (
            int(row.get("reservation_sequence") or 0),
            str(row.get("action_execution_id") or ""),
        ),
    )
    attempted_current = {
        str(row.get("action_id") or "")
        for row in rows
        if int(row.get("input_revision") or 0) == current_revision
        and str(row.get("action_id") or "")
    }
    no_gain_bindings: dict[str, str] = {}
    consecutive_no_gain = 0
    classified_count = 0
    unclassified_boundary_count = 0
    revision_discontinuity_count = 0
    last_output_revision: int | None = None
    previous_input_revision: int | None = None
    previous_same_revision_cohort = False
    for row in rows:
        if row.get("settled") is not True:
            continue
        action_id = str(row.get("action_id") or "")
        opportunity_sha256 = str(row.get("opportunity_sha256") or "")
        gained = row.get("gained")
        input_revision = int(row.get("input_revision") or 0)
        same_revision_cohort = row.get("same_revision_cohort") is True
        shared_cohort_revision = bool(
            same_revision_cohort
            and previous_same_revision_cohort
            and previous_input_revision == input_revision
        )
        if (
            last_output_revision is not None
            and input_revision != last_output_revision
            and not shared_cohort_revision
        ):
            consecutive_no_gain = 0
            revision_discontinuity_count += 1
        last_output_revision = int(row.get("output_revision") or 0)
        previous_input_revision = input_revision
        previous_same_revision_cohort = same_revision_cohort
        if not action_id or not opportunity_sha256 or not isinstance(gained, bool):
            consecutive_no_gain = 0
            unclassified_boundary_count += 1
            continue
        classified_count += 1
        if gained:
            consecutive_no_gain = 0
            no_gain_bindings.pop(action_id, None)
            continue
        consecutive_no_gain += 1
        no_gain_bindings[action_id] = opportunity_sha256
    revision_advanced_outside_history = bool(
        last_output_revision is not None
        and last_output_revision != current_revision
    )
    if revision_advanced_outside_history:
        consecutive_no_gain = 0
    binding_rows = [
        {
            "action_id": action_id,
            "opportunity_sha256": no_gain_bindings[action_id],
        }
        for action_id in sorted(no_gain_bindings)
    ]
    result = {
        "schema_version": ACTION_CONVERGENCE_LEDGER_SCHEMA,
        "current_graph_revision": current_revision,
        "execution_record_count": len(rows),
        "settled_execution_count": sum(
            row.get("settled") is True for row in rows
        ),
        "classified_execution_count": classified_count,
        "unclassified_boundary_count": unclassified_boundary_count,
        "revision_discontinuity_count": revision_discontinuity_count,
        "attempted_action_ids_at_current_revision": sorted(attempted_current),
        "no_gain_bindings": binding_rows,
        "consecutive_no_gain": consecutive_no_gain,
        "last_observed_output_revision": (
            last_output_revision if last_output_revision is not None else 0
        ),
        "revision_advanced_outside_history": revision_advanced_outside_history,
        "execution_history_sha256": _digest(rows),
        "semantics": {
            "run_kernel_reservations_and_action_outcomes_are_authority": True,
            "cache_replay_does_not_add_an_attempt": True,
            "unknown_legacy_records_break_the_consecutive_chain": True,
            "graph_revision_discontinuity_breaks_the_consecutive_chain": True,
            "external_graph_progress_resets_only_the_consecutive_count": True,
            "no_gain_binding_requires_exact_opportunity_digest": True,
            "ledger_creates_no_queue_budget_or_scientific_authority": True,
        },
    }
    result["content_sha256"] = _digest(result)
    return result


def no_gain_binding_map(ledger: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact Action/opportunity exclusions from one verified ledger."""

    return {
        str(row.get("action_id") or ""): str(
            row.get("opportunity_sha256") or ""
        )
        for raw in ledger.get("no_gain_bindings") or []
        for row in (dict(raw),)
        if str(row.get("action_id") or "")
        and str(row.get("opportunity_sha256") or "")
    }


def verified_action_convergence_ledger(
    ledger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one digest-valid durable ledger, otherwise fail closed."""

    if not isinstance(ledger, Mapping):
        return {}
    row = dict(ledger)
    content_sha256 = str(row.pop("content_sha256", ""))
    if (
        row.get("schema_version") != ACTION_CONVERGENCE_LEDGER_SCHEMA
        or len(content_sha256) != 64
        or _digest(row) != content_sha256
        or dict(row.get("semantics") or {}).get(
            "run_kernel_reservations_and_action_outcomes_are_authority"
        )
        is not True
    ):
        return {}
    return {**row, "content_sha256": content_sha256}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ACTION_CONVERGENCE_LEDGER_SCHEMA",
    "compile_action_convergence_ledger",
    "no_gain_binding_map",
    "verified_action_convergence_ledger",
]
