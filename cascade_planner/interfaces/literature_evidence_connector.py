"""Execute bounded literature discovery and source materialization."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests

from cascade_planner.harness.local_pdf_proxy import (
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_request_queue_path,
)
from cascade_planner.interfaces.literature_access import (
    authorized_proxy_artifact,
    downloaded_unextractable_source,
    pending_source,
    queue_authorized_pdf_request,
)
from cascade_planner.interfaces.literature_candidates import (
    candidate_source_ref as _candidate_source_ref,
    dedupe_candidates as _dedupe_candidates,
    doi as _doi,
    interleave_candidates as _interleave_candidates,
    queries as _queries,
    request_source_candidates as _request_source_candidates,
    seed_candidates as _seed_candidates,
    target_relevant_candidates as _target_relevant_candidates,
)
from cascade_planner.interfaces.literature_evidence_contract import (
    _compact_discovery_source,
    _digest,
    _route_binding_eligible,
)
from cascade_planner.interfaces.literature_route_binding import (
    bind_materialized_literature_source,
)
from cascade_planner.interfaces.live_evidence import (
    LiveEvidenceConnectorError,
    SOURCE_DISCOVERY_OBSERVATION_SCHEMA,
)


def invoke_builtin_literature_evidence(
    request: Mapping[str, Any],
    *,
    config: Any,
    cache_root: Path,
    proxy_root: Path,
    search: Callable[[str, int], Iterable[Mapping[str, Any]]],
    fetch: Callable[[str, float, int], bytes],
    fetch_authorized: Callable[..., Mapping[str, Any]],
    default_resolvers: Callable[[], tuple[Any, Any]],
    resolver_cache_receipt: Callable[[], Mapping[str, Any] | None],
    structure_resolver: Any,
    candidate_name_resolver: Any,
    materialize_candidate: Callable[..., Mapping[str, Any]],
    provider_id: str,
    provider_version: str,
) -> Mapping[str, Any]:
    request_sha = str(request.get("content_sha256") or "")
    queries = _queries(request, limit=config.max_queries)
    seed_candidates = _dedupe_candidates(_seed_candidates(config))
    request_candidates = _dedupe_candidates(_request_source_candidates(request))
    candidates = list(request_candidates)

    def run_search(query: str) -> tuple[list[dict[str, Any]], str]:
        try:
            return [dict(row) for row in search(query, config.max_sources * 2)], ""
        except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
            return [], f"{type(exc).__name__}:{str(exc)[:300]}"

    if len(queries) <= 1:
        searched = [run_search(query) for query in queries]
    else:
        with ThreadPoolExecutor(
            max_workers=min(config.max_workers, len(queries)),
            thread_name_prefix="autoplanner-paper-search",
        ) as executor:
            searched = list(executor.map(run_search, queries))
    query_candidates = [rows for rows, _failure in searched]
    search_audits = [
        {
            "query": query,
            "candidate_count": len(rows),
            "status": "completed" if not failure else "failed",
            "reason": failure,
        }
        for query, (rows, failure) in zip(queries, searched, strict=True)
    ]
    candidates.extend(_interleave_candidates(query_candidates))
    ranked_candidates = _target_relevant_candidates(
        _dedupe_candidates(candidates),
        target_name=str(request.get("target_name") or ""),
        pinned_source_refs=(_candidate_source_ref(row) for row in request_candidates),
    )
    # Configured seeds are explicit acquisition inputs.  Keep them ahead
    # of ranked discovery/request hints so a busy source frontier cannot
    # silently consume the bounded source budget before the seeds run.
    candidates = _dedupe_candidates([*seed_candidates, *ranked_candidates])[
        : config.max_sources * 3
    ]
    if not candidates:
        raise LiveEvidenceConnectorError("builtin_literature_no_candidates")
    request_dir = cache_root / request_sha[:24]
    request_dir.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    structured_sources: list[dict[str, Any]] = []
    pending_sources: list[dict[str, Any]] = []
    pending_attempts: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    def admit_materialized_source(
        source: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        admitted = dict(source)
        route_binding: dict[str, Any] = {
            "status": "not_needed",
            "model_invocations": 0,
        }
        if _route_binding_eligible(admitted, request=request):
            default_structure, default_names = default_resolvers()
            admitted, structured, route_binding = bind_materialized_literature_source(
                admitted,
                request=request,
                output_dir=(
                    request_dir
                    / hashlib.sha256(
                        str(admitted.get("source_ref") or "").encode()
                    ).hexdigest()[:20]
                ),
                structure_resolver=(structure_resolver or default_structure),
                candidate_name_resolver=(candidate_name_resolver or default_names),
                timeout_s=config.timeout_s,
                provider_version=provider_version,
            )
            if structured:
                structured_sources.append(structured)
        return admitted, {
            "source_ref": admitted["source_ref"],
            "accepted": True,
            "source_artifact_sha256": str(
                admitted.get("source_fulltext_sha256")
                or admitted.get("source_pdf_sha256")
                or ""
            ),
            "acquisition_method": str(admitted.get("acquisition_method") or ""),
            "fulltext_sha256": str(admitted.get("source_fulltext_sha256") or ""),
            "pdf_sha256": str(admitted.get("source_pdf_sha256") or ""),
            "visual_page_count": len(admitted["visual_candidate_pages"]),
            "route_binding": route_binding,
        }

    remaining = iter(candidates)
    while len(sources) + len(pending_sources) < config.max_sources:
        batch = [
            candidate
            for candidate in (
                next(remaining, None)
                for _ in range(
                    min(
                        config.max_workers,
                        config.max_sources - len(sources) - len(pending_sources),
                    )
                )
            )
            if candidate is not None
        ]
        if not batch:
            break

        def materialize(
            candidate: Mapping[str, Any],
        ) -> tuple[dict[str, Any] | None, Exception | None]:
            try:
                return (
                    materialize_candidate(
                        candidate,
                        request=request,
                        output_dir=request_dir,
                        config=config,
                        fetch=fetch,
                        proxy_root=proxy_root,
                    ),
                    None,
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                requests.RequestException,
            ) as exc:
                return None, exc

        if len(batch) == 1:
            materialized = [materialize(batch[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(config.max_workers, len(batch)),
                thread_name_prefix="autoplanner-paper-fetch",
            ) as executor:
                materialized = list(executor.map(materialize, batch))
        for candidate, (source, exc) in zip(batch, materialized, strict=True):
            if len(sources) + len(pending_sources) >= config.max_sources:
                break
            if exc is not None:
                source_ref = _candidate_source_ref(candidate)
                source_doi = _doi(candidate)
                frozen_authorized_artifact = authorized_proxy_artifact(
                    candidate,
                    proxy_root=proxy_root,
                    source_ref=source_ref,
                    doi=source_doi,
                )
                if frozen_authorized_artifact:
                    reason = f"{type(exc).__name__}:{str(exc)[:500]}"
                    audits.append(
                        {
                            "source_ref": source_ref,
                            "accepted": False,
                            "status": "downloaded_unextractable",
                            "reason": reason,
                            "authorized_artifact": {
                                "request_id": str(
                                    frozen_authorized_artifact.get("request_id") or ""
                                ),
                                "artifact_kind": str(
                                    frozen_authorized_artifact.get("artifact_kind") or ""
                                ),
                            },
                        }
                    )
                    pending_sources.append(
                        downloaded_unextractable_source(
                            candidate,
                            source_ref=source_ref,
                            doi=source_doi,
                            reason=reason,
                        )
                    )
                    continue
                queued = (
                    queue_authorized_pdf_request(
                        candidate,
                        request=request,
                        proxy_root=proxy_root,
                        reason=f"direct_pdf_materialization_failed:{type(exc).__name__}",
                        source_ref=source_ref,
                    )
                    if config.queue_restricted_sources
                    else {}
                )
                audits.append(
                    {
                        "source_ref": source_ref,
                        "accepted": False,
                        "status": "queued_for_authorized_browser" if queued else "failed",
                        "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
                        "proxy_request": queued,
                    }
                )
                if queued:
                    pending_attempts.append(
                        {
                            "candidate": dict(candidate),
                            "source_ref": source_ref,
                            "audit_index": len(audits) - 1,
                        }
                    )
                    pending_sources.append(
                        pending_source(
                            candidate,
                            proxy_request=queued,
                            source_ref=source_ref,
                            doi=source_doi,
                        )
                    )
                continue
            assert source is not None
            admitted, audit = admit_materialized_source(source)
            audits.append(audit)
            sources.append(admitted)

    automatic_fetch: dict[str, Any] = {
        "schema_version": "authorized_browser_autofetch.v1",
        "status": "disabled",
        "reason": "automatic_authorized_browser_fetch_disabled",
        "processed_count": 0,
        "downloaded_count": 0,
    }
    if config.auto_fetch_restricted_sources and pending_attempts:
        try:
            automatic_fetch = dict(
                fetch_authorized(
                    proxy_root=proxy_root,
                    case_id=str(request.get("run_id") or request.get("target_name") or ""),
                    source_refs=tuple(
                        str(row["source_ref"]) for row in pending_attempts
                    ),
                    max_items=min(
                        config.auto_fetch_max_items,
                        len(pending_attempts),
                    ),
                    timeout_s=config.auto_fetch_timeout_s,
                    headless=config.auto_fetch_headless,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            automatic_fetch = {
                "schema_version": "authorized_browser_autofetch.v1",
                "status": "failed",
                "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
                "processed_count": 0,
                "downloaded_count": 0,
            }
        materialized_refs: set[str] = set()
        downloaded_request_ids = {
            str(row.get("request_id") or "")
            for row in automatic_fetch.get("results") or []
            if isinstance(row, Mapping) and row.get("status") == "downloaded"
        }
        downloaded_unextractable: dict[str, str] = {}
        for attempt in pending_attempts:
            source_ref = str(attempt["source_ref"])
            source, retry_error = materialize(attempt["candidate"])
            if retry_error is not None or source is None:
                queued_audit = audits[int(attempt["audit_index"])]
                request_id = str(
                    dict(queued_audit.get("proxy_request") or {}).get(
                        "request_id"
                    )
                    or ""
                )
                artifact_downloaded = request_id in downloaded_request_ids
                extraction_reason = (
                    f"{type(retry_error).__name__}:{str(retry_error)[:500]}"
                    if retry_error is not None
                    else "authorized_artifact_not_materialized"
                )
                queued_audit["automatic_fetch"] = {
                    "status": (
                        "downloaded_unextractable"
                        if artifact_downloaded
                        else "unresolved"
                    ),
                    "reason": (
                        extraction_reason
                    ),
                }
                if artifact_downloaded:
                    queued_audit["status"] = "downloaded_unextractable"
                    downloaded_unextractable[source_ref] = extraction_reason
                continue
            admitted, audit = admit_materialized_source(source)
            audit["status"] = "materialized_after_automatic_browser_fetch"
            audits.append(audit)
            queued_audit = audits[int(attempt["audit_index"])]
            queued_audit["status"] = "resolved_by_automatic_browser_fetch"
            queued_audit["resolved_in_same_invocation"] = True
            sources.append(admitted)
            materialized_refs.add(source_ref)
        if materialized_refs:
            pending_sources = [
                row
                for row in pending_sources
                if str(row.get("source_ref") or "") not in materialized_refs
            ]
        if downloaded_unextractable:
            pending_sources = [
                {
                    **row,
                    "acquisition_status": "downloaded_unextractable",
                    "extraction_failure_reason": downloaded_unextractable[
                        str(row.get("source_ref") or "")
                    ],
                    "semantics": {
                        **dict(row.get("semantics") or {}),
                        "metadata_only": False,
                        "resume_after_browser_download": False,
                        "download_completed_but_extraction_failed": True,
                    },
                }
                if str(row.get("source_ref") or "") in downloaded_unextractable
                else row
                for row in pending_sources
            ]
        automatic_fetch["materialized_source_count"] = len(materialized_refs)
        automatic_fetch["downloaded_unextractable_source_count"] = len(
            downloaded_unextractable
        )
    if not sources and not pending_sources:
        raise LiveEvidenceConnectorError("builtin_literature_no_pdf_materialized")
    discovery = {
        "schema_version": SOURCE_DISCOVERY_OBSERVATION_SCHEMA,
        "provider_id": provider_id,
        "request_sha256": request_sha,
        "sources": [
            _compact_discovery_source(row)
            for row in [*sources, *pending_sources]
        ],
        "semantics": {
            "paper_metadata_is_not_route_evidence": True,
            "structured_fulltext_precedes_pdf": True,
            "pdf_bytes_are_content_addressed": True,
            "native_text_selects_pages_before_vision": True,
            "visual_output_requires_host_admission_and_validation": True,
            "queued_sources_are_not_evidence": True,
            "authorized_browser_results_are_consumed_on_resume": True,
            "automatic_browser_results_are_rematerialized_in_same_invocation": True,
        },
    }
    discovery["content_sha256"] = _digest(discovery)
    receipt = {
        "schema_version": "evidence_connector_receipt.v1",
        "provider_id": provider_id,
        "provider_version": provider_version,
        "request_sha256": request_sha,
        "queries": queries,
        "candidate_count": len(candidates),
        "accepted_source_count": len(sources),
        "queued_source_count": sum(
            row.get("acquisition_status") == "queued_for_authorized_browser"
            for row in pending_sources
        ),
        "downloaded_unextractable_source_count": sum(
            row.get("acquisition_status") == "downloaded_unextractable"
            for row in pending_sources
        ),
        "source_lifecycle": {
            "materialized": len(sources),
            "queued_for_authorized_browser": sum(
                row.get("acquisition_status") == "queued_for_authorized_browser"
                for row in pending_sources
            ),
            "downloaded_unextractable": sum(
                row.get("acquisition_status") == "downloaded_unextractable"
                for row in pending_sources
            ),
            "failed": sum(1 for row in audits if row.get("status") == "failed"),
        },
        "authorized_proxy": {
            "output_dir": str(proxy_root),
            "request_queue_path": str(local_pdf_proxy_request_queue_path(proxy_root)),
            "download_manifest_path": str(local_pdf_proxy_download_manifest_path(proxy_root)),
            "credentials_stored": False,
        },
        "automatic_authorized_fetch": automatic_fetch,
        "audits": audits,
        "search_audits": search_audits,
        "parallel_search": len(queries) > 1,
        "parallel_materialization": config.max_workers > 1,
        "model_invocations": 0,
        "resolver_cache": resolver_cache_receipt(),
    }
    receipt["content_sha256"] = _digest(receipt)
    result: dict[str, Any] = {"discovery": discovery, "receipt": receipt}
    if structured_sources:
        result["document"] = {
            "schema_version": "structured_evidence_import.v1",
            "sources": structured_sources,
        }
    return result


__all__ = ["invoke_builtin_literature_evidence"]
