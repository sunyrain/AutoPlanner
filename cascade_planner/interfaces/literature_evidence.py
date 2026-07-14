"""Bounded primary-paper discovery with delegated source materialization."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests

from cascade_planner.harness.local_pdf_proxy import (
    local_pdf_proxy_download_manifest_path,
    local_pdf_proxy_request_queue_path,
)
from cascade_planner.interfaces.literature_access import (
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
)
from cascade_planner.interfaces.literature_materialization import (
    materialize_candidate as _materialize_candidate,
)
from cascade_planner.interfaces.literature_search import crossref_search, fetch_bytes
from cascade_planner.interfaces.live_evidence import (
    LiveEvidenceConnectorError,
    SOURCE_DISCOVERY_OBSERVATION_SCHEMA,
)


BUILTIN_LITERATURE_PROVIDER_ID = "autoplanner.builtin_literature_evidence"
BUILTIN_LITERATURE_PROVIDER_VERSION = "1.1"
PaperSearch = Callable[[str, int], Iterable[Mapping[str, Any]]]
BytesFetcher = Callable[[str, float, int], bytes]


@dataclass(frozen=True, slots=True)
class BuiltinLiteratureEvidenceConfig:
    cache_dir: str | Path
    seed_dois: tuple[str, ...] = ()
    seed_pdfs: tuple[str, ...] = ()
    max_sources: int = 3
    max_queries: int = 4
    max_pdf_bytes: int = 30_000_000
    max_pdf_pages: int = 160
    enable_structured_fulltext_first: bool = True
    max_fulltext_bytes: int = 12_000_000
    max_fulltext_sections: int = 32
    max_visual_pages: int = 6
    max_workers: int = 3
    timeout_s: float = 30.0
    render_zoom: float = 1.6
    authorized_proxy_output_dir: str | Path = ""
    queue_restricted_sources: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_sources <= 8 or not 1 <= self.max_queries <= 8:
            raise ValueError("literature_evidence_search_limit_invalid")
        if self.max_pdf_bytes < 1024 or self.max_pdf_pages < 1:
            raise ValueError("literature_evidence_pdf_limit_invalid")
        if self.max_fulltext_bytes < 10_000:
            raise ValueError("literature_evidence_fulltext_limit_invalid")
        if not 1 <= self.max_fulltext_sections <= 128:
            raise ValueError("literature_evidence_fulltext_section_limit_invalid")
        if not 1 <= self.max_visual_pages <= 12 or self.timeout_s <= 0:
            raise ValueError("literature_evidence_execution_limit_invalid")
        if not 1 <= self.max_workers <= 8:
            raise ValueError("literature_evidence_worker_limit_invalid")


def build_builtin_literature_evidence_connector(
    config: BuiltinLiteratureEvidenceConfig,
    *,
    searcher: PaperSearch | None = None,
    fetcher: BytesFetcher | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build a metadata-first connector using XML before bounded PDF fallback."""
    cache_root = Path(config.cache_dir).expanduser().resolve()
    proxy_root = (
        Path(config.authorized_proxy_output_dir).expanduser().resolve()
        if str(config.authorized_proxy_output_dir).strip()
        else cache_root
    )
    search = searcher or crossref_search
    fetch = fetcher or fetch_bytes

    def invoke(request: Mapping[str, Any]) -> Mapping[str, Any]:
        request_sha = str(request.get("content_sha256") or "")
        queries = _queries(request, limit=config.max_queries)
        candidates = [*_seed_candidates(config), *_request_source_candidates(request)]

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
        candidates = _dedupe_candidates(candidates)[: config.max_sources * 3]
        if not candidates:
            raise LiveEvidenceConnectorError("builtin_literature_no_candidates")
        request_dir = cache_root / request_sha[:24]
        request_dir.mkdir(parents=True, exist_ok=True)
        sources: list[dict[str, Any]] = []
        pending_sources: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
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
                        _materialize_candidate(
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
                    queued = (
                        queue_authorized_pdf_request(
                            candidate,
                            request=request,
                            proxy_root=proxy_root,
                            reason=f"direct_pdf_materialization_failed:{type(exc).__name__}",
                            source_ref=_candidate_source_ref(candidate),
                        )
                        if config.queue_restricted_sources
                        else {}
                    )
                    audits.append(
                        {
                            "source_ref": _candidate_source_ref(candidate),
                            "accepted": False,
                            "status": "queued_for_authorized_browser" if queued else "failed",
                            "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
                            "proxy_request": queued,
                        }
                    )
                    if queued:
                        pending_sources.append(
                            pending_source(
                                candidate,
                                proxy_request=queued,
                                source_ref=_candidate_source_ref(candidate),
                                doi=_doi(candidate),
                            )
                        )
                    continue
                assert source is not None
                audits.append(
                    {
                        "source_ref": source["source_ref"],
                        "accepted": True,
                        "source_artifact_sha256": str(
                            source.get("source_fulltext_sha256")
                            or source.get("source_pdf_sha256")
                            or ""
                        ),
                        "acquisition_method": str(source.get("acquisition_method") or ""),
                        "fulltext_sha256": str(source.get("source_fulltext_sha256") or ""),
                        "pdf_sha256": str(source.get("source_pdf_sha256") or ""),
                        "visual_page_count": len(source["visual_candidate_pages"]),
                    }
                )
                sources.append(source)
        if not sources and not pending_sources:
            raise LiveEvidenceConnectorError("builtin_literature_no_pdf_materialized")
        discovery = {
            "schema_version": SOURCE_DISCOVERY_OBSERVATION_SCHEMA,
            "provider_id": BUILTIN_LITERATURE_PROVIDER_ID,
            "request_sha256": request_sha,
            "sources": [*sources, *pending_sources],
            "semantics": {
                "paper_metadata_is_not_route_evidence": True,
                "structured_fulltext_precedes_pdf": True,
                "pdf_bytes_are_content_addressed": True,
                "native_text_selects_pages_before_vision": True,
                "visual_output_requires_host_admission_and_validation": True,
                "queued_sources_are_not_evidence": True,
                "authorized_browser_results_are_consumed_on_resume": True,
            },
        }
        discovery["content_sha256"] = _digest(discovery)
        receipt = {
            "schema_version": "evidence_connector_receipt.v1",
            "provider_id": BUILTIN_LITERATURE_PROVIDER_ID,
            "provider_version": BUILTIN_LITERATURE_PROVIDER_VERSION,
            "request_sha256": request_sha,
            "queries": queries,
            "candidate_count": len(candidates),
            "accepted_source_count": len(sources),
            "queued_source_count": len(pending_sources),
            "source_lifecycle": {
                "materialized": len(sources),
                "queued_for_authorized_browser": len(pending_sources),
                "failed": sum(1 for row in audits if row.get("status") == "failed"),
            },
            "authorized_proxy": {
                "output_dir": str(proxy_root),
                "request_queue_path": str(local_pdf_proxy_request_queue_path(proxy_root)),
                "download_manifest_path": str(
                    local_pdf_proxy_download_manifest_path(proxy_root)
                ),
                "credentials_stored": False,
            },
            "audits": audits,
            "search_audits": search_audits,
            "parallel_search": len(queries) > 1,
            "parallel_materialization": config.max_workers > 1,
            "model_invocations": 0,
        }
        receipt["content_sha256"] = _digest(receipt)
        return {"discovery": discovery, "receipt": receipt}

    setattr(invoke, "autoplanner_prefetch_safe", True)
    return invoke


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


__all__ = [
    "BUILTIN_LITERATURE_PROVIDER_ID",
    "BUILTIN_LITERATURE_PROVIDER_VERSION",
    "BuiltinLiteratureEvidenceConfig",
    "build_builtin_literature_evidence_connector",
]
