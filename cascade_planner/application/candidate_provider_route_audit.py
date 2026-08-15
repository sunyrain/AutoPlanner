"""Per-route provider conservation and first-loss projection."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


PROVIDER_ROUTE_PROVENANCE_RECORD_SCHEMA = "provider_route_provenance_record.v1"


def provider_route_record(
    source: Mapping[str, Any],
    lifecycle_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    row = dict(source)
    linked = [
        dict(value)
        for value in lifecycle_records
        if lineage_matches(value, row)
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
        "normalized_route_sha256": str(row.get("normalized_route_sha256") or ""),
        "proposal_eligible": row.get("proposal_eligible") is True,
        "host_portfolio_selected": row.get("host_portfolio_selected") is True,
        "preserved_as_advisory": row.get("preserved_as_advisory") is True,
        "provider_disposition": str(row.get("disposition") or ""),
        "final_disposition": str(row.get("final_disposition") or ""),
        "reasons": sorted(str(value) for value in row.get("reasons") or []),
        "canonical_route_family_id": str(
            row.get("canonical_route_family_id") or ""
        ),
        "canonical_hypothesis_ids": _values(row, "canonical_hypothesis_ids"),
        "canonical_edge_ids": _values(row, "canonical_edge_ids"),
        "canonical_route_ids": _values(row, "canonical_route_ids"),
        "provider_step_count": _optional_count(row, "provider_step_count"),
        "normalized_step_count": _optional_count(row, "normalized_step_count"),
        "imported_proposal_count": _optional_count(row, "imported_proposal_count"),
        "canonical_bound_step_count": _optional_count(
            row, "canonical_bound_step_count"
        ),
        "topology_conservation_applicable": (
            row.get("topology_conservation_applicable") is True
        ),
        "topology_conservation_accepted": (
            row.get("topology_conservation_accepted")
            if row.get("topology_conservation_applicable") is True
            else None
        ),
        "missing_imported_proposal_ids": _values(
            row, "missing_imported_proposal_ids"
        ),
        "missing_canonical_proposal_ids": _values(
            row, "missing_canonical_proposal_ids"
        ),
        "stock_closed_route_ids": stock_routes,
        "candidate_ids": sorted(
            str(value.get("candidate_id") or "") for value in linked
        ),
        "candidate_statuses": sorted(
            {str(value.get("status") or "") for value in linked} - {""}
        ),
    }
    record["first_loss_boundary"] = _first_loss(row, linked, stock_routes)
    return _with_digest(record)


def lineage_matches(
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
        return (
            "host_quarantine"
            if source.get("preserved_as_advisory") is True
            or source.get("quarantined") is True
            else "host_admission"
        )
    if source.get("host_portfolio_selected") is not True:
        return (
            "host_quarantine"
            if source.get("preserved_as_advisory") is True
            or source.get("quarantined") is True
            else "host_portfolio_selection"
        )
    if (
        source.get("topology_conservation_applicable") is True
        and source.get("topology_conservation_accepted") is not True
    ):
        return "canonical_topology_conservation"
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


def _optional_count(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _values(row: Mapping[str, Any], key: str) -> list[str]:
    return sorted(str(value) for value in row.get(key) or [] if str(value))


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return row


__all__ = [
    "PROVIDER_ROUTE_PROVENANCE_RECORD_SCHEMA",
    "lineage_matches",
    "provider_route_record",
]
