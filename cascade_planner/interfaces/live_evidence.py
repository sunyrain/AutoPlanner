"""Bounded connector boundary for primary-source reaction extraction.

The connector is deliberately chemistry-neutral.  It receives the current
canonical edge set and director acquisition tasks, then returns the existing
typed structured-evidence interchange document.  The host still validates
source identity, exact structures, edge digests, independence, and reaction
proof; the connector cannot grant B3 by returning a boolean.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

import requests

from cascade_planner.application.reaction_proof_versions import active_reaction_proofs
from cascade_planner.interfaces.evidence_import import (
    validate_structured_evidence_document,
)


EVIDENCE_ACQUISITION_REQUEST_SCHEMA = "evidence_acquisition_request.v1"
EVIDENCE_CONNECTOR_RECEIPT_SCHEMA = "evidence_connector_receipt.v1"
SOURCE_DISCOVERY_OBSERVATION_SCHEMA = "source_discovery_observation.v1"
EvidenceConnector = Callable[[Mapping[str, Any]], Mapping[str, Any]]
HttpRequester = Callable[..., tuple[int, bytes, Mapping[str, Any]]]


class LiveEvidenceConnectorError(RuntimeError):
    """A configured evidence connector failed its bounded host contract."""


def compose_evidence_connectors(
    *connectors: EvidenceConnector,
    max_sources: int = 16,
) -> EvidenceConnector:
    """Combine independent paper/patent providers behind one typed boundary."""

    active = tuple(connector for connector in connectors if connector is not None)
    if not active:
        raise ValueError("evidence_connector_composition_empty")
    if not 1 <= max_sources <= 64:
        raise ValueError("evidence_connector_composition_limit_invalid")

    def invoke(request: Mapping[str, Any]) -> Mapping[str, Any]:
        documents: list[dict[str, Any]] = []
        discovery_sources: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        failures: list[str] = []

        def run_one(
            indexed: tuple[int, EvidenceConnector],
        ) -> tuple[int, dict[str, Any] | None, str, float]:
            index, connector = indexed
            started = time.monotonic()
            try:
                result = dict(connector(request))
            except (LiveEvidenceConnectorError, OSError, RuntimeError, ValueError) as exc:
                return (
                    index,
                    None,
                    f"connector_{index}:{type(exc).__name__}:{str(exc)[:300]}",
                    round(time.monotonic() - started, 3),
                )
            return index, result, "", round(time.monotonic() - started, 3)

        indexed_connectors = list(enumerate(active, start=1))
        if len(indexed_connectors) == 1:
            completed = [run_one(indexed_connectors[0])]
        else:
            # Patent and paper acquisition are independent I/O-bound providers.
            # Preserve provider order in the merged observation while reducing
            # the critical path from the sum of their timeouts to the slowest
            # provider timeout.
            with ThreadPoolExecutor(
                max_workers=min(4, len(indexed_connectors)),
                thread_name_prefix="autoplanner-evidence",
            ) as executor:
                completed = list(executor.map(run_one, indexed_connectors))
        child_elapsed_s: dict[str, float] = {}
        for index, result, failure, elapsed_s in sorted(completed):
            child_elapsed_s[f"connector_{index}"] = elapsed_s
            if failure:
                failures.append(failure)
                continue
            assert result is not None
            if isinstance(result.get("document"), Mapping):
                documents.extend(
                    dict(row)
                    for row in dict(result["document"]).get("sources") or []
                    if isinstance(row, Mapping)
                )
            if isinstance(result.get("discovery"), Mapping):
                discovery_sources.extend(
                    dict(row)
                    for row in dict(result["discovery"]).get("sources") or []
                    if isinstance(row, Mapping)
                )
            if isinstance(result.get("receipt"), Mapping):
                receipts.append(dict(result["receipt"]))
        if not documents and not discovery_sources:
            raise LiveEvidenceConnectorError(
                "composed_evidence_connectors_unresolved:" + "|".join(failures)
            )
        output: dict[str, Any] = {
            "receipt": {
                "schema_version": EVIDENCE_CONNECTOR_RECEIPT_SCHEMA,
                "provider_id": "autoplanner.composed_evidence",
                "provider_version": "1.0",
                "request_sha256": str(request.get("content_sha256") or ""),
                "child_receipts": receipts,
                "failures": failures,
                "model_invocations": 0,
                "parallel_connector_count": len(active),
                "child_elapsed_s": child_elapsed_s,
                "connector_wall_time_s": max(child_elapsed_s.values(), default=0.0),
                "semantics": {
                    "independent_connectors_run_concurrently": len(active) > 1,
                    "merge_order_is_deterministic": True,
                },
            }
        }
        output["receipt"]["content_sha256"] = _digest(output["receipt"])
        if documents:
            output["document"] = {
                "schema_version": "structured_evidence_import.v1",
                "sources": documents[:128],
            }
        if discovery_sources:
            discovery = {
                "schema_version": SOURCE_DISCOVERY_OBSERVATION_SCHEMA,
                "provider_id": "autoplanner.composed_evidence",
                "request_sha256": str(request.get("content_sha256") or ""),
                "sources": discovery_sources[:max_sources],
                "semantics": {
                    "providers_remain_independently_attributed": True,
                    "composition_grants_no_scientific_authority": True,
                },
            }
            discovery["content_sha256"] = _digest(discovery)
            output["discovery"] = discovery
        return output

    setattr(
        invoke,
        "autoplanner_prefetch_safe",
        all(
            getattr(connector, "autoplanner_prefetch_safe", False) is True
            for connector in active
        ),
    )
    return invoke


@dataclass(frozen=True, slots=True)
class HttpEvidenceConnectorConfig:
    endpoint: str
    provider_id: str
    provider_version: str
    token_env: str = ""
    timeout_s: float = 60.0
    max_response_bytes: int = 4_000_000
    max_sources: int = 64
    max_rows: int = 256

    def __post_init__(self) -> None:
        if not self.provider_id.strip() or not self.provider_version.strip():
            raise ValueError("evidence_connector_identity_missing")
        _validated_endpoint(self.endpoint)
        if self.timeout_s <= 0 or self.max_response_bytes < 1:
            raise ValueError("evidence_connector_transport_limit_invalid")
        if self.max_sources < 1 or self.max_rows < 1:
            raise ValueError("evidence_connector_content_limit_invalid")


def compile_evidence_acquisition_request(
    *,
    run_id: str,
    target_name: str = "",
    target_smiles: str,
    graph: Mapping[str, Any],
    source_frontier: Mapping[str, Any],
    target_identity: Mapping[str, Any] | None = None,
    max_edges: int = 64,
    max_source_tasks: int = 24,
    prefetch_mode: bool = False,
) -> dict[str, Any]:
    """Compile a bounded, answer-free request from current canonical state."""

    if max_edges < 1 or max_source_tasks < 1:
        raise ValueError("evidence_acquisition_request_limit_invalid")
    edges = dict(graph.get("edges") or {})
    selected_ids = _selected_edge_ids(graph)
    ordered_ids = sorted(selected_ids or edges)[:max_edges]
    edge_rows = []
    for edge_id in ordered_ids:
        edge = dict(edges.get(edge_id) or {})
        if not edge:
            continue
        current_proofs = active_reaction_proofs(edge.get("reaction_proofs") or [])
        edge_rows.append(
            {
                "edge_id": str(edge.get("edge_id") or edge_id),
                "edge_digest": str(edge.get("edge_digest") or ""),
                "product_smiles": str(edge.get("product_smiles") or ""),
                "precursor_smiles": [
                    str(value) for value in edge.get("precursor_smiles") or []
                ],
                "current_host_reaction_validated": any(
                    row.get("accepted") is True for row in current_proofs
                ),
                "existing_independent_source_groups": sorted(
                    {
                        str(value)
                        for value in edge.get("independent_source_groups") or []
                        if str(value)
                    }
                ),
            }
        )
    detail = dict(source_frontier.get("detail") or source_frontier)
    task_rows = [
        dict(row)
        for row in detail.get("source_plan") or []
        if isinstance(row, Mapping)
    ]
    task_rows.sort(
        key=lambda row: (
            -float(row.get("priority") or 0.0),
            str(row.get("source_task_id") or ""),
        )
    )
    tasks = [_bounded_source_task(row) for row in task_rows[:max_source_tasks]]
    source_hints = [
        {
            "source_ref": str(row.get("source_ref") or "")[:500],
            "source_kind": str(row.get("source_kind") or "")[:80],
            "title": " ".join(str(row.get("title") or "").split())[:1000],
            "occurrence_count": max(0, int(row.get("occurrence_count") or 0)),
            "target_edge_occurrence_count": max(
                0, int(row.get("target_edge_occurrence_count") or 0)
            ),
            "corroborating_source_ref_count": max(
                0, int(row.get("corroborating_source_ref_count") or 0)
            ),
            "route_skeleton_count": max(
                0, int(row.get("route_skeleton_count") or 0)
            ),
            "affected_step_ids": _bounded_strings(
                row.get("affected_step_ids") or [], 32, 160
            ),
        }
        for row in detail.get("sources") or []
        if isinstance(row, Mapping) and str(row.get("source_ref") or "")
    ][:max_source_tasks]
    identity = dict(target_identity or {})
    identity_hints = {
        "preferred_name": " ".join(
            str(identity.get("preferred_name") or "").split()
        )[:500],
        "synonyms": [
            " ".join(str(value).split())[:500]
            for value in identity.get("synonyms") or []
            if str(value).strip()
        ][:12],
        "patent_ids": [
            str(value).strip()[:100]
            for value in identity.get("patent_ids") or []
            if str(value).strip()
        ][:24],
        "pubmed_ids": [
            str(value).strip()[:100]
            for value in identity.get("pubmed_ids") or []
            if str(value).strip()
        ][:24],
        "resolved_from_input_structure": bool(identity),
    }
    request = {
        "schema_version": EVIDENCE_ACQUISITION_REQUEST_SCHEMA,
        "run_id": str(run_id),
        "target_name": " ".join(str(target_name or "").split())[:500],
        "target_smiles": str(target_smiles),
        "graph_revision": int(graph.get("revision") or 0),
        "edges": edge_rows,
        "source_tasks": tasks,
        "source_hints": source_hints,
        "target_identity": identity_hints,
        "limits": {
            "max_edges": max_edges,
            "max_source_tasks": max_source_tasks,
            "exact_structured_rows_only": True,
            "source_fetch_policy": (
                "html_first_no_pdf" if prefetch_mode else "full_fallback"
            ),
        },
        "semantics": {
            "request_contains_no_dossier_or_replay_pack": True,
            "source_search_result_is_not_exact_evidence": True,
            "connector_cannot_grant_reaction_validation": True,
            "prefetch_mode": bool(prefetch_mode),
        },
    }
    request["content_sha256"] = _digest(request)
    return request


def acquire_structured_evidence(
    request: Mapping[str, Any],
    *,
    connector: EvidenceConnector,
) -> dict[str, Any]:
    """Invoke an injected connector and normalize its fail-closed response."""

    supplied = str(request.get("content_sha256") or "")
    body = {key: value for key, value in request.items() if key != "content_sha256"}
    if request.get("schema_version") != EVIDENCE_ACQUISITION_REQUEST_SCHEMA:
        raise LiveEvidenceConnectorError("evidence_acquisition_request_schema_invalid")
    if supplied != _digest(body):
        raise LiveEvidenceConnectorError("evidence_acquisition_request_digest_invalid")
    result = connector(dict(request))
    if not isinstance(result, Mapping):
        raise LiveEvidenceConnectorError("evidence_connector_result_not_object")
    row = dict(result)
    wrapped = "document" in row or "discovery" in row
    document = (
        validate_structured_evidence_document(row["document"])
        if "document" in row
        else None
    )
    discovery = (
        _validate_source_discovery_observation(row.get("discovery"))
        if "discovery" in row
        else None
    )
    if not wrapped:
        document = validate_structured_evidence_document(row)
    if document is None and discovery is None:
        raise LiveEvidenceConnectorError("evidence_connector_result_empty")
    receipt = dict(row.get("receipt") or {}) if wrapped else {}
    return {
        "document": document,
        "receipt": receipt,
        "discovery": discovery,
        "document_sha256": _digest(document) if document is not None else "",
    }


def _validate_source_discovery_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveEvidenceConnectorError("source_discovery_observation_not_object")
    row = dict(value)
    if row.get("schema_version") != SOURCE_DISCOVERY_OBSERVATION_SCHEMA:
        raise LiveEvidenceConnectorError("source_discovery_observation_schema_invalid")
    sources = [
        dict(item) for item in row.get("sources") or [] if isinstance(item, Mapping)
    ]
    if not sources or len(sources) > 16:
        raise LiveEvidenceConnectorError("source_discovery_observation_source_count_invalid")
    supplied = str(row.pop("content_sha256", "") or "")
    if supplied and supplied != _digest(row):
        raise LiveEvidenceConnectorError("source_discovery_observation_digest_invalid")
    row["sources"] = sources
    encoded = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > 128_000:
        raise LiveEvidenceConnectorError("source_discovery_observation_too_large")
    row["content_sha256"] = supplied or _digest(row)
    return row


def build_http_evidence_connector(
    config: HttpEvidenceConnectorConfig,
    *,
    requester: HttpRequester | None = None,
) -> EvidenceConnector:
    """Create a typed HTTPS/loopback connector with frozen response metadata."""

    request_json = requester or _requests_json

    def invoke(request: Mapping[str, Any]) -> Mapping[str, Any]:
        token = os.environ.get(config.token_env, "") if config.token_env else ""
        if config.token_env and not token:
            raise LiveEvidenceConnectorError("evidence_connector_token_missing")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "AutoPlanner/1.0 structured-evidence-connector",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            status, content, _response_headers = request_json(
                "POST",
                config.endpoint,
                json_body=dict(request),
                headers=headers,
                timeout_s=config.timeout_s,
                max_response_bytes=config.max_response_bytes,
            )
        except LiveEvidenceConnectorError:
            raise
        except (requests.RequestException, OSError, TimeoutError) as exc:
            raise LiveEvidenceConnectorError(
                f"evidence_connector_transport_failed:{type(exc).__name__}"
            ) from exc
        if status != 200:
            raise LiveEvidenceConnectorError(f"evidence_connector_http_status:{status}")
        if len(content) > config.max_response_bytes:
            raise LiveEvidenceConnectorError("evidence_connector_response_too_large")
        try:
            value = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiveEvidenceConnectorError("evidence_connector_response_invalid_json") from exc
        document = validate_structured_evidence_document(value)
        _validate_connector_identity(document, config=config)
        receipt = {
            "schema_version": EVIDENCE_CONNECTOR_RECEIPT_SCHEMA,
            "provider_id": config.provider_id,
            "provider_version": config.provider_version,
            "endpoint": _public_endpoint(config.endpoint),
            "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "response_sha256": hashlib.sha256(content).hexdigest(),
            "response_bytes": len(content),
            "request_sha256": str(request.get("content_sha256") or ""),
            "credential_recorded": False,
        }
        receipt["content_sha256"] = _digest(receipt)
        return {"document": document, "receipt": receipt}

    return invoke


def _validate_connector_identity(
    document: Mapping[str, Any],
    *,
    config: HttpEvidenceConnectorConfig,
) -> None:
    sources = list(document.get("sources") or [])
    if len(sources) > config.max_sources:
        raise LiveEvidenceConnectorError("evidence_connector_source_limit_exceeded")
    row_count = 0
    for index, source in enumerate(sources, start=1):
        extraction = dict(dict(source).get("extraction") or {})
        extractor = dict(extraction.get("extractor") or {})
        row_count += len(extraction.get("rows") or [])
        if (
            extractor.get("producer_kind") != "typed_connector_structured_extraction"
            or extractor.get("producer_id") != config.provider_id
            or extractor.get("version") != config.provider_version
        ):
            raise LiveEvidenceConnectorError(
                f"evidence_connector_extractor_identity_mismatch:{index}"
            )
    if row_count > config.max_rows:
        raise LiveEvidenceConnectorError("evidence_connector_row_limit_exceeded")


def _selected_edge_ids(graph: Mapping[str, Any]) -> set[str]:
    return {
        str(edge_id)
        for route in dict(graph.get("route_families") or {}).values()
        if isinstance(route, Mapping) and route.get("selected") is not False
        for edge_id in route.get("edge_ids") or []
        if str(edge_id)
    }


def _bounded_source_task(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    return {
        "source_task_id": str(row.get("source_task_id") or "")[:160],
        "query": " ".join(str(row.get("query") or "").split())[:1000],
        "priority": max(0.0, min(1.0, float(row.get("priority") or 0.0))),
        "source_types": _bounded_strings(row.get("source_types") or [], 12, 120),
        "source_refs": _bounded_strings(row.get("source_refs") or [], 12, 500),
        "target_claims": _bounded_strings(row.get("target_claims") or [], 16, 500),
        "affected_proposal_ids": _bounded_strings(
            row.get("affected_proposal_ids") or [], 32, 160
        ),
    }


def _bounded_strings(values: Iterable[Any], count: int, width: int) -> list[str]:
    return [" ".join(str(value).split())[:width] for value in list(values)[:count] if str(value)]


def _validated_endpoint(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    loopback = (parsed.hostname or "").lower() in {"127.0.0.1", "::1", "localhost"}
    if (
        parsed.scheme not in {"https", "http"}
        or (parsed.scheme == "http" and not loopback)
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("evidence_connector_endpoint_invalid")
    return parsed.geturl()


def _public_endpoint(value: str) -> str:
    parsed = urlsplit(_validated_endpoint(value))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _requests_json(
    method: str,
    url: str,
    *,
    json_body: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
    max_response_bytes: int,
) -> tuple[int, bytes, Mapping[str, Any]]:
    response = requests.request(
        method,
        url,
        json=dict(json_body),
        headers=dict(headers),
        timeout=max(1.0, float(timeout_s)),
        stream=True,
        allow_redirects=False,
    )
    try:
        status = int(response.status_code)
        response_headers = dict(response.headers)
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            total += len(chunk)
            if total > max_response_bytes:
                raise LiveEvidenceConnectorError(
                    "evidence_connector_response_too_large"
                )
            chunks.append(bytes(chunk))
        return status, b"".join(chunks), response_headers
    finally:
        response.close()


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
    "EVIDENCE_ACQUISITION_REQUEST_SCHEMA",
    "SOURCE_DISCOVERY_OBSERVATION_SCHEMA",
    "EvidenceConnector",
    "HttpEvidenceConnectorConfig",
    "LiveEvidenceConnectorError",
    "acquire_structured_evidence",
    "build_http_evidence_connector",
    "compose_evidence_connectors",
    "compile_evidence_acquisition_request",
]
