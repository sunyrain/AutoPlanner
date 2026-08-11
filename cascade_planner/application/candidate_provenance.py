"""Digest-bound provenance from provider routes to canonical candidates."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.candidate_lifecycle import candidate_lifecycle_export


CANONICAL_CANDIDATE_PROVENANCE_SCHEMA = "canonical_candidate_provenance.v1"
CANDIDATE_PROVENANCE_RECORD_SCHEMA = "canonical_candidate_provenance_record.v1"
PROVIDER_ROUTE_PROVENANCE_RECORD_SCHEMA = "provider_route_provenance_record.v1"
_PROVIDER_LINEAGE_SCHEMA = "chemenzy_route_lineage.v1"


def compile_candidate_provenance(
    lifecycle: Mapping[str, Any],
    *,
    lineage_observations: Iterable[Any] = (),
) -> dict[str, Any]:
    """Bind verified provider route lineage to one canonical lifecycle."""

    lifecycle_row = dict(lifecycle)
    if not _content_digest_valid(lifecycle_row):
        raise ValueError("candidate_provenance_lifecycle_digest_invalid")
    lineage_reports, ignored = _verified_lineage_reports(lineage_observations)
    provider_rows = _provider_rows(lineage_reports)
    lifecycle_records = [
        dict(value)
        for value in lifecycle_row.get("records") or []
        if isinstance(value, Mapping)
    ]
    candidate_records = [
        _candidate_record(record, provider_rows) for record in lifecycle_records
    ]
    provider_records = [
        _provider_record(source, lifecycle_records) for source in provider_rows
    ]
    first_loss_counts = {
        boundary: sum(
            row.get("first_loss_boundary") == boundary for row in provider_records
        )
        for boundary in sorted(
            {
                str(row.get("first_loss_boundary") or "")
                for row in provider_records
            }
            - {""}
        )
    }
    payload = {
        "schema_version": CANONICAL_CANDIDATE_PROVENANCE_SCHEMA,
        "lifecycle_sha256": str(lifecycle_row.get("content_sha256") or ""),
        "graph_revision": int(lifecycle_row.get("graph_revision") or 0),
        "graph_scientific_sha256": str(
            lifecycle_row.get("graph_scientific_sha256") or ""
        ),
        "candidate_record_count": len(candidate_records),
        "provider_route_count": len(provider_records),
        "bound_provider_route_count": sum(
            bool(row.get("candidate_ids")) for row in provider_records
        ),
        "provider_only_route_count": sum(
            not bool(row.get("candidate_ids")) for row in provider_records
        ),
        "ignored_provider_lineage_count": ignored,
        "first_loss_counts": first_loss_counts,
        "candidate_records": candidate_records,
        "provider_route_records": provider_records,
        "semantics": {
            "projection_is_read_only": True,
            "provider_routes_grant_no_canonical_identity": True,
            "unbound_provider_routes_remain_visible": True,
            "proof_evidence_stock_and_acceptance_are_not_inferred": True,
            "first_loss_is_a_deterministic_audit_boundary": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def candidate_provenance_export(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed reviewer projection."""

    provenance = dict(value)
    available = _content_digest_valid(provenance)
    return {
        "available": available,
        "unavailable_reason": (
            "" if available else "candidate_provenance_digest_invalid"
        ),
        "provenance_sha256": str(provenance.get("content_sha256") or ""),
        "provenance": provenance if available else {},
    }


def candidate_review_lineage_records(
    lifecycle: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project both candidate audits into the existing route-lineage component."""

    rows = []
    if lifecycle:
        rows.append({
            "kind": "canonical_candidate_lifecycle",
            **candidate_lifecycle_export(lifecycle),
        })
    if provenance:
        rows.append({
            "kind": "canonical_candidate_provenance",
            **candidate_provenance_export(provenance),
        })
    return rows


def _candidate_record(
    lifecycle: Mapping[str, Any],
    provider_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    row = dict(lifecycle)
    links = [
        dict(source) for source in provider_rows if _lineage_matches(row, source)
    ]
    origins = [
        dict(value)
        for value in row.get("origin_records") or []
        if isinstance(value, Mapping)
    ]
    portfolio = dict(row.get("portfolio") or {})
    record = {
        "schema_version": CANDIDATE_PROVENANCE_RECORD_SCHEMA,
        "candidate_id": str(row.get("candidate_id") or ""),
        "lifecycle_record_sha256": str(row.get("content_sha256") or ""),
        "status": str(row.get("status") or ""),
        "canonical_entity_ids": sorted(
            str(value) for value in row.get("canonical_entity_ids") or [] if str(value)
        ),
        "proposal_origin": {
            "proposal_ids": sorted(
                {
                    str(value.get("proposal_id") or "")
                    for value in origins
                    if str(value.get("proposal_id") or "")
                }
            ),
            "origin_kinds": sorted(
                {
                    str(value.get("origin_kind") or "")
                    for value in origins
                    if str(value.get("origin_kind") or "")
                }
            ),
            "origin_records": _json_value(origins),
        },
        "provider_normalization": {
            "route_trace_ids": _values(links, "route_trace_id"),
            "raw_route_sha256": _values(links, "raw_route_sha256"),
            "normalized_route_sha256": _values(links, "normalized_route_sha256"),
            "lineage_sha256": _values(links, "source_lineage_sha256"),
        },
        "ingestion_rejection": {
            "kind": str(row.get("rejection_kind") or ""),
            "reasons": sorted(str(value) for value in row.get("rejection_reasons") or []),
            "report_sha256": sorted(
                str(value) for value in row.get("ingestion_report_sha256") or []
            ),
        },
        "host_admission": _json_value(row.get("admission") or {}),
        "materialization": _json_value(row.get("materialization") or {}),
        "reaction_validation": _json_value(row.get("validation") or {}),
        "exact_evidence": _json_value(row.get("evidence") or {}),
        "conditions": _json_value(row.get("conditions") or {}),
        "stock_closure": {
            "route_ids": sorted(portfolio.get("stock_closed_route_ids") or [])
        },
        "configured_acceptance": {
            "route_ids": sorted(portfolio.get("accepted_route_ids") or [])
        },
        "semantics": {
            "lifecycle_record_remains_status_authority": True,
            "provider_normalization_is_observed_not_inferred": True,
        },
    }
    return _with_digest(record)


def _provider_record(
    source: Mapping[str, Any],
    lifecycle_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    row = dict(source)
    linked = [
        dict(value)
        for value in lifecycle_records
        if _lineage_matches(value, row)
    ]
    stock_routes = sorted(
        {
            str(route_id)
            for value in linked
            for route_id in dict(value.get("portfolio") or {}).get(
                "stock_closed_route_ids"
            )
            or []
            if str(route_id)
        }
    )
    record = {
        "schema_version": PROVIDER_ROUTE_PROVENANCE_RECORD_SCHEMA,
        "provider_id": "chemenzy",
        "source_lineage_sha256": str(row.get("source_lineage_sha256") or ""),
        "route_trace_id": str(row.get("route_trace_id") or ""),
        "raw_route_sha256": str(row.get("raw_route_sha256") or ""),
        "normalized_route_sha256": str(
            row.get("normalized_route_sha256") or ""
        ),
        "proposal_eligible": row.get("proposal_eligible") is True,
        "host_portfolio_selected": row.get("host_portfolio_selected") is True,
        "preserved_as_advisory": row.get("preserved_as_advisory") is True,
        "provider_disposition": str(row.get("disposition") or ""),
        "final_disposition": str(row.get("final_disposition") or ""),
        "reasons": sorted(str(value) for value in row.get("reasons") or []),
        "canonical_route_family_id": str(
            row.get("canonical_route_family_id") or ""
        ),
        "canonical_hypothesis_ids": sorted(
            str(value) for value in row.get("canonical_hypothesis_ids") or []
        ),
        "canonical_edge_ids": sorted(
            str(value) for value in row.get("canonical_edge_ids") or []
        ),
        "canonical_route_ids": sorted(
            str(value) for value in row.get("canonical_route_ids") or []
        ),
        "stock_closed_route_ids": stock_routes,
        "candidate_ids": sorted(str(value.get("candidate_id") or "") for value in linked),
        "candidate_statuses": sorted(
            {str(value.get("status") or "") for value in linked} - {""}
        ),
    }
    record["first_loss_boundary"] = _first_loss(row, linked, stock_routes)
    return _with_digest(record)


def _first_loss(
    source: Mapping[str, Any],
    linked: Iterable[Mapping[str, Any]],
    stock_routes: list[str],
) -> str:
    rows = [dict(value) for value in linked]
    if not str(source.get("raw_route_sha256") or ""):
        return "raw_proposal_unobserved"
    if not str(source.get("normalized_route_sha256") or ""):
        return "provider_normalization"
    if source.get("proposal_eligible") is not True:
        return "host_quarantine" if source.get("preserved_as_advisory") is True or source.get("quarantined") is True else "host_admission"
    if source.get("host_portfolio_selected") is not True:
        return (
            "host_quarantine"
            if source.get("preserved_as_advisory") is True
            or source.get("quarantined") is True
            else "host_portfolio_selection"
        )
    if not rows:
        return "canonical_ingestion"
    if any(
        dict(value.get("materialization") or {}).get("materialized") is not True
        for value in rows
    ):
        return "canonical_materialization"
    if any(
        dict(value.get("validation") or {}).get("accepted") is not True
        for value in rows
    ):
        return "reaction_validation"
    return "none" if stock_routes else "stock_closure"


def _lineage_matches(
    lifecycle: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> bool:
    canonical_ids = {
        *(str(value) for value in lifecycle.get("canonical_entity_ids") or []),
        str(lifecycle.get("edge_id") or ""),
    } - {""}
    lineage_ids = {
        *(str(value) for value in lineage.get("canonical_edge_ids") or []),
        *(str(value) for value in lineage.get("canonical_hypothesis_ids") or []),
    } - {""}
    proposal_ids = {
        str(dict(value).get("proposal_id") or "")
        for value in lifecycle.get("origin_records") or []
        if isinstance(value, Mapping)
    } - {""}
    step_ids = {
        str(value) for value in lineage.get("step_proposal_ids") or [] if str(value)
    }
    return bool(canonical_ids & lineage_ids or proposal_ids & step_ids)


def _verified_lineage_reports(
    observations: Iterable[Any],
) -> tuple[list[dict[str, Any]], int]:
    reports: dict[str, dict[str, Any]] = {}
    ignored = 0
    for report in _objects_with_schema(observations, _PROVIDER_LINEAGE_SCHEMA):
        if not _content_digest_valid(report):
            ignored += 1
            continue
        reports[str(report["content_sha256"])] = report
    return [reports[key] for key in sorted(reports)], ignored


def _provider_rows(reports: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for report in reports:
        source_sha256 = str(report.get("content_sha256") or "")
        for value in report.get("routes") or []:
            if not isinstance(value, Mapping):
                continue
            row = {**dict(value), "source_lineage_sha256": source_sha256}
            rows[_digest(row)] = row
    return [rows[key] for key in sorted(rows)]


def _objects_with_schema(value: Any, schema: str) -> list[dict[str, Any]]:
    rows = []
    stack = list(value) if isinstance(value, (list, tuple)) else [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            row = dict(current)
            if row.get("schema_version") == schema:
                rows.append(row)
                continue
            stack.extend(row.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return rows


def _values(rows: Iterable[Mapping[str, Any]], key: str) -> list[str]:
    return sorted({str(row.get(key) or "") for row in rows} - {""})


def _content_digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return len(supplied) == 64 and supplied == _digest(row)


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _json_value(value)
    row.pop("content_sha256", None)
    row["content_sha256"] = _digest(row)
    return row


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


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
    "CANONICAL_CANDIDATE_PROVENANCE_SCHEMA",
    "CANDIDATE_PROVENANCE_RECORD_SCHEMA",
    "PROVIDER_ROUTE_PROVENANCE_RECORD_SCHEMA",
    "candidate_provenance_export",
    "candidate_review_lineage_records",
    "compile_candidate_provenance",
]
