"""Read-only lifecycle projection for canonical retrosynthesis candidates."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from cascade_planner.application.canonical_hypergraph import (
    CANONICAL_INGESTION_REPORT_SCHEMA,
)


CANONICAL_CANDIDATE_LIFECYCLE_SCHEMA = "canonical_candidate_lifecycle.v1"
CANDIDATE_LIFECYCLE_RECORD_SCHEMA = "canonical_candidate_lifecycle_record.v1"
CANDIDATE_LIFECYCLE_STATUSES = (
    "rejected_invalid",
    "quarantined_reviewable",
    "admitted_unproved",
    "validated",
    "accepted",
)
_STATUS_RANK = {
    status: index for index, status in enumerate(CANDIDATE_LIFECYCLE_STATUSES)
}
_CANDIDATE_REJECTION_KINDS = {"hypothesis", "reaction_edge", "route_innovation"}


def compile_candidate_lifecycle(
    graph: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    *,
    ingestion_observations: Iterable[Any] = (),
) -> dict[str, Any]:
    """Merge canonical topology, proof, stock, portfolio, and rejected inputs."""

    graph_revision = int(graph.get("revision") or 0)
    graph_sha256 = str(graph.get("scientific_sha256") or "")
    if not _content_digest_valid(portfolio):
        raise ValueError("candidate_lifecycle_portfolio_digest_invalid")
    if (
        int(portfolio.get("graph_revision") or 0) != graph_revision
        or str(portfolio.get("graph_scientific_sha256") or "") != graph_sha256
    ):
        raise ValueError("candidate_lifecycle_graph_portfolio_mismatch")

    hypotheses = dict(graph.get("hypotheses") or {})
    edges = dict(graph.get("edges") or {})
    edge_proofs = dict(portfolio.get("edge_proofs") or {})
    route_state = _route_state_by_edge(portfolio)
    hypothesis_by_edge: dict[str, tuple[str, dict[str, Any]]] = {}
    for hypothesis_id, value in sorted(hypotheses.items()):
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        edge_digest = str(row.get("edge_digest") or "")
        if edge_digest:
            hypothesis_by_edge[f"edge:{edge_digest}"] = (str(hypothesis_id), row)

    records: list[dict[str, Any]] = []
    for edge_id in sorted(set(edges) | set(hypothesis_by_edge)):
        edge = dict(edges.get(edge_id) or {})
        hypothesis_id, hypothesis = hypothesis_by_edge.get(edge_id, ("", {}))
        proof = dict(edge_proofs.get(edge_id) or {})
        routes = route_state.get(edge_id, _empty_route_state())
        materialized = bool(edge)
        admission_accepted = bool(
            materialized or hypothesis.get("admission_accepted") is True
        )
        validation_accepted = proof.get("accepted") is True
        if routes["accepted_route_ids"]:
            status = "accepted"
            status_reason = "configured_portfolio_acceptance"
        elif validation_accepted:
            status = "validated"
            status_reason = "host_reaction_validation_accepted"
        elif admission_accepted:
            status = "admitted_unproved"
            status_reason = (
                "reaction_proof_open" if materialized else "materialization_pending"
            )
        else:
            status = "quarantined_reviewable"
            status_reason = "canonical_admission_rejected_retained_l0"
        source_groups = sorted(
            {
                str(value)
                for value in [
                    *(edge.get("independent_source_groups") or []),
                    *(proof.get("independent_source_groups") or []),
                ]
                if str(value)
            }
        )
        record = {
            "schema_version": CANDIDATE_LIFECYCLE_RECORD_SCHEMA,
            "candidate_id": hypothesis_id or edge_id,
            "canonical_entity_ids": sorted({edge_id, hypothesis_id} - {""}),
            "edge_id": edge_id,
            "edge_digest": str(
                edge.get("edge_digest") or hypothesis.get("edge_digest") or ""
            ),
            "product_smiles": str(
                edge.get("product_smiles") or hypothesis.get("product_smiles") or ""
            ),
            "precursor_smiles": list(
                edge.get("precursor_smiles")
                or hypothesis.get("precursor_smiles")
                or []
            ),
            "status": status,
            "status_reason": status_reason,
            "admission": {
                "accepted": admission_accepted,
                "reasons": _strings(hypothesis.get("admission_reasons") or []),
                "audit_sha256": str(
                    edge.get("admission_audit_sha256")
                    or hypothesis.get("admission_audit_sha256")
                    or ""
                ),
            },
            "materialization": {"materialized": materialized},
            "validation": {
                "accepted": validation_accepted,
                "achieved_level": int(proof.get("achieved_level") or 0),
                "reasons": _strings(proof.get("reasons") or []),
            },
            "evidence": {
                "exact_record_count": len(edge.get("exact_record_ids") or []),
                "independent_source_groups": source_groups,
            },
            "conditions": {
                "prediction_count": len(edge.get("condition_predictions") or [])
            },
            "portfolio": {
                key: sorted(routes[key])
                for key in (
                    "route_ids",
                    "pareto_route_ids",
                    "selected_route_ids",
                    "complete_route_ids",
                    "stock_closed_route_ids",
                    "accepted_route_ids",
                )
            },
            "route_family_ids": sorted(
                {
                    str(value)
                    for value in [
                        *(edge.get("route_family_ids") or []),
                        *(hypothesis.get("route_family_ids") or []),
                    ]
                    if str(value)
                }
            ),
            "origin_records": _deduplicate_rows(
                [
                    *(edge.get("origin_records") or []),
                    *(hypothesis.get("origin_records") or []),
                ]
            ),
            "semantics": {
                "open_proof_or_stock_axes_do_not_delete_topology": True,
                "portfolio_selection_does_not_create_scientific_facts": True,
            },
        }
        records.append(_with_digest(record))

    rejected_records, ignored_reports = _rejected_candidate_records(
        ingestion_observations
    )
    records.extend(rejected_records)
    records.sort(
        key=lambda row: (
            _STATUS_RANK[str(row["status"])],
            str(row.get("candidate_id") or ""),
            str(row.get("content_sha256") or ""),
        )
    )
    counts = {
        status: sum(row.get("status") == status for row in records)
        for status in CANDIDATE_LIFECYCLE_STATUSES
    }
    payload = {
        "schema_version": CANONICAL_CANDIDATE_LIFECYCLE_SCHEMA,
        "graph_revision": graph_revision,
        "graph_scientific_sha256": graph_sha256,
        "portfolio_sha256": str(portfolio.get("content_sha256") or ""),
        "candidate_count": len(records),
        "canonical_candidate_count": len(records) - counts["rejected_invalid"],
        "status_counts": counts,
        "ignored_ingestion_report_count": ignored_reports,
        "records": records,
        "semantics": {
            "projection_is_read_only": True,
            "canonical_graph_remains_candidate_authority": True,
            "ingestion_rejections_grant_no_canonical_identity": True,
            "accepted_requires_configured_portfolio_acceptance": True,
            "proof_evidence_stock_and_conditions_remain_independent_axes": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def candidate_lifecycle_export(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed reviewer projection without granting authority."""

    lifecycle = dict(value)
    available = _content_digest_valid(lifecycle)
    return {
        "available": available,
        "unavailable_reason": (
            "" if available else "candidate_lifecycle_digest_invalid"
        ),
        "lifecycle_sha256": str(lifecycle.get("content_sha256") or ""),
        "lifecycle": lifecycle if available else {},
    }


def _route_state_by_edge(portfolio: Mapping[str, Any]) -> dict[str, dict[str, set[str]]]:
    states: dict[str, dict[str, set[str]]] = {}
    campaign_accepted = portfolio.get("accepted") is True
    for route in portfolio.get("route_candidates") or []:
        if not isinstance(route, Mapping):
            continue
        route_id = str(route.get("route_id") or "")
        for edge_id in route.get("edge_ids") or []:
            state = states.setdefault(str(edge_id), _empty_route_state())
            state["route_ids"].add(route_id)
            if route.get("pareto_optimal") is True:
                state["pareto_route_ids"].add(route_id)
            if route.get("selected") is True:
                state["selected_route_ids"].add(route_id)
            if route.get("complete") is True:
                state["complete_route_ids"].add(route_id)
            if route.get("all_leaves_stock_closed") is True:
                state["stock_closed_route_ids"].add(route_id)
            if (
                campaign_accepted
                and route.get("selected") is True
                and route.get("complete") is True
            ):
                state["accepted_route_ids"].add(route_id)
    return states


def _empty_route_state() -> dict[str, set[str]]:
    return {
        "route_ids": set(),
        "pareto_route_ids": set(),
        "selected_route_ids": set(),
        "complete_route_ids": set(),
        "stock_closed_route_ids": set(),
        "accepted_route_ids": set(),
    }


def _rejected_candidate_records(
    observations: Iterable[Any],
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, dict[str, Any]] = {}
    ignored = 0
    for report in _objects_with_schema(observations, CANONICAL_INGESTION_REPORT_SCHEMA):
        if not _content_digest_valid(report):
            ignored += 1
            continue
        report_sha256 = str(report.get("content_sha256") or "")
        for value in report.get("rejected") or []:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            if (
                str(row.get("kind") or "") not in _CANDIDATE_REJECTION_KINDS
                or row.get("retained_as_l0") is True
            ):
                continue
            identity = {
                "kind": str(row.get("kind") or ""),
                "proposal_id": str(row.get("proposal_id") or ""),
                "hypothesis_id": str(row.get("hypothesis_id") or ""),
                "reasons": _strings(row.get("reasons") or []),
            }
            key = _digest(identity)
            current = grouped.setdefault(
                key,
                {
                    "schema_version": CANDIDATE_LIFECYCLE_RECORD_SCHEMA,
                    "candidate_id": identity["proposal_id"]
                    or identity["hypothesis_id"]
                    or f"rejected:{key[:24]}",
                    "canonical_entity_ids": [],
                    "edge_id": "",
                    "edge_digest": "",
                    "product_smiles": "",
                    "precursor_smiles": [],
                    "status": "rejected_invalid",
                    "status_reason": "canonical_ingestion_rejected",
                    "rejection_kind": identity["kind"],
                    "rejection_reasons": identity["reasons"],
                    "ingestion_report_sha256": [],
                    "semantics": {
                        "rejected_input_did_not_mutate_canonical_graph": True,
                        "rejected_invalid_is_an_ingestion_disposition": True,
                    },
                },
            )
            current["ingestion_report_sha256"] = sorted(
                {*current["ingestion_report_sha256"], report_sha256} - {""}
            )
    return [_with_digest(grouped[key]) for key in sorted(grouped)], ignored


def _objects_with_schema(value: Any, schema: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _deduplicate_rows(values: Iterable[Any]) -> list[dict[str, Any]]:
    rows = {
        _digest(dict(value)): dict(value)
        for value in values
        if isinstance(value, Mapping)
    }
    return [rows[key] for key in sorted(rows)]


def _strings(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted({str(item) for item in values if str(item).strip()})


def _content_digest_valid(value: Mapping[str, Any]) -> bool:
    row = dict(value)
    supplied = str(row.pop("content_sha256", ""))
    return bool(supplied and supplied == _digest(row))


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    row.pop("content_sha256", None)
    row["content_sha256"] = _digest(row)
    return row


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
    "CANONICAL_CANDIDATE_LIFECYCLE_SCHEMA",
    "CANDIDATE_LIFECYCLE_RECORD_SCHEMA",
    "CANDIDATE_LIFECYCLE_STATUSES",
    "candidate_lifecycle_export",
    "compile_candidate_lifecycle",
]
