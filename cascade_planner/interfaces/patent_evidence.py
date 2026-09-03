"""Built-in, zero-model, HTML-first patent evidence acquisition.

The connector freezes complete publication HTML before considering PDF.  PDF
text, local OCR, and sparse vision are progressively narrower fallbacks.
Search metadata, snippets, and Codex source hints never become exact rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

import requests

from cascade_planner.harness.deterministic_literature_registry import (
    DEFAULT_OPSIN_BASE_URL,
    DEFAULT_PUBCHEM_BASE_URL,
    PARSER_AUTHORITY_ID,
    CandidateNameResolver,
    StructureResolver,
    build_deterministic_literature_resolvers,
    compile_deterministic_literature_step_registry,
    extract_deterministic_source_document,
    extract_deterministic_structured_source_document,
)
from cascade_planner.harness.deterministic_resolver_cache import (
    DeterministicResolverCache,
)
from cascade_planner.harness.literature_pdf_extraction import (
    extract_literature_pdf_assets,
    rebuild_literature_pdf_page_focus,
)
from cascade_planner.harness.literature_page_selection import (
    select_pdf_page_numbers,
)
from cascade_planner.harness.source_ocr import (
    LocalOcrConfig,
    OcrRunner,
    materialize_local_ocr_companion,
)
from cascade_planner.harness.source_route_extraction import (
    compile_deterministic_source_route_observation,
)
from cascade_planner.interfaces.live_evidence import LiveEvidenceConnectorError
from cascade_planner.interfaces.patent_html_evidence import (
    attempt_primary_patent_html,
)
from cascade_planner.interfaces.patent_source_discovery import (
    evidence_queries,
    fetch_bounded_bytes,
    google_patent_candidate_provider,
    select_independent_candidates,
)


BUILTIN_PATENT_PROVIDER_ID = "autoplanner.builtin_patent_evidence"
BUILTIN_PATENT_PROVIDER_VERSION = "1.7.0"
SOURCE_DISCOVERY_OBSERVATION_SCHEMA = "source_discovery_observation.v1"
PatentCandidateProvider = Callable[
    [Iterable[str]], Iterable[Mapping[str, Any]]
]
BytesFetcher = Callable[[str, float, int], bytes]
RegistryCompiler = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class BuiltinPatentEvidenceConfig:
    cache_dir: str | Path
    seed_publications: tuple[str, ...] = ()
    timeout_s: float = 30.0
    max_search_queries: int = 4
    max_search_pages_per_query: int = 3
    max_patents: int = 3
    max_pdf_bytes: int = 32_000_000
    max_pdf_pages: int = 80
    enable_html_first: bool = True
    max_html_bytes: int = 20_000_000
    max_html_sections: int = 24
    max_html_paragraphs: int = 256
    max_validated_edges: int = 32
    render_zoom: float = 0.75
    enable_local_ocr: bool = True
    max_ocr_pages: int = 12

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("patent_evidence_timeout_invalid")
        if not 1 <= self.max_search_queries <= 12:
            raise ValueError("patent_evidence_query_limit_invalid")
        if not 1 <= self.max_search_pages_per_query <= 10:
            raise ValueError("patent_evidence_search_page_limit_invalid")
        if not 1 <= self.max_patents <= 8:
            raise ValueError("patent_evidence_patent_limit_invalid")
        if self.max_pdf_bytes < 1_000 or self.max_pdf_pages < 1:
            raise ValueError("patent_evidence_pdf_limit_invalid")
        if self.max_html_bytes < 10_000:
            raise ValueError("patent_evidence_html_byte_limit_invalid")
        if not 1 <= self.max_html_sections <= 64:
            raise ValueError("patent_evidence_html_section_limit_invalid")
        if not 8 <= self.max_html_paragraphs <= 1_024:
            raise ValueError("patent_evidence_html_paragraph_limit_invalid")
        if not 1 <= self.max_validated_edges <= 128:
            raise ValueError("patent_evidence_edge_limit_invalid")
        if not 0.5 <= self.render_zoom <= 2.0:
            raise ValueError("patent_evidence_render_zoom_invalid")
        if not 1 <= self.max_ocr_pages <= 80:
            raise ValueError("patent_evidence_ocr_page_limit_invalid")


def build_builtin_patent_evidence_connector(
    config: BuiltinPatentEvidenceConfig,
    *,
    candidate_provider: PatentCandidateProvider | None = None,
    bytes_fetcher: BytesFetcher | None = None,
    html_fetcher: BytesFetcher | None = None,
    registry_compiler: RegistryCompiler | None = None,
    structure_resolver: StructureResolver | None = None,
    candidate_name_resolver: CandidateNameResolver | None = None,
    ocr_runner: OcrRunner | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build a bounded first-party connector for current validated edges."""

    cache_root = Path(config.cache_dir).expanduser().resolve()
    search = candidate_provider or google_patent_candidate_provider(config)
    fetch = bytes_fetcher or fetch_bounded_bytes
    fetch_html = html_fetcher or fetch
    compile_registry = registry_compiler or compile_deterministic_literature_step_registry
    resolver_cache: DeterministicResolverCache | None = None
    default_structure_resolver: StructureResolver | None = None
    default_name_resolver: CandidateNameResolver | None = None

    def default_resolvers() -> tuple[StructureResolver, CandidateNameResolver]:
        nonlocal resolver_cache, default_structure_resolver, default_name_resolver
        if default_structure_resolver is None or default_name_resolver is None:
            resolver_cache = DeterministicResolverCache(
                cache_root / "resolver-cache",
                authority_id=PARSER_AUTHORITY_ID,
                opsin_base_url=DEFAULT_OPSIN_BASE_URL,
                pubchem_base_url=DEFAULT_PUBCHEM_BASE_URL,
            )
            default_structure_resolver, default_name_resolver = (
                build_deterministic_literature_resolvers(
                    timeout_s=config.timeout_s,
                    persistent_cache=resolver_cache,
                )
            )
        return default_structure_resolver, default_name_resolver

    def invoke(request: Mapping[str, Any]) -> Mapping[str, Any]:
        requested_edges = [
            dict(row)
            for row in request.get("edges") or []
            if isinstance(row, Mapping)
        ][: config.max_validated_edges]
        edges = [
            row
            for row in requested_edges
            if row.get("current_host_reaction_validated") is True
        ]
        unvalidated_edge_discovery_only = bool(requested_edges and not edges)
        discovery_only = not edges
        discovery_pdf_allowed = not (
            discovery_only
            and str(dict(request.get("limits") or {}).get("source_fetch_policy") or "")
            == "html_first_no_pdf"
        )
        queries = evidence_queries(request, limit=config.max_search_queries)
        run_cache = cache_root / "runs" / hashlib.sha256(
            str(request.get("run_id") or "anonymous-run").encode("utf-8")
        ).hexdigest()[:24]
        run_cache.mkdir(parents=True, exist_ok=True)
        candidates, candidate_cache_hit = _run_scoped_candidates(
            run_cache,
            queries=queries,
            limit=config.max_patents,
            search=search,
        )
        if not candidates:
            raise LiveEvidenceConnectorError(
                "builtin_patent_evidence_no_patent_candidates"
            )
        resolve_structure = structure_resolver
        resolve_names = candidate_name_resolver
        source_route_resolve_structure = structure_resolver
        if discovery_only:
            # Target-only prefetch is allowed to freeze source bytes and build
            # a source inventory, but it cannot bind or promote a reaction.
            # Exact-row compilation therefore receives authority-free empty
            # resolvers.  Source-route observation uses a lazy real resolver
            # only when a frozen document actually contains procedures.
            resolve_structure = resolve_structure or (lambda _value: "")
            resolve_names = resolve_names or (lambda _value: [])
            if source_route_resolve_structure is None:
                def lazy_source_route_resolver(value: str) -> str:
                    return default_resolvers()[0](value)

                source_route_resolve_structure = lazy_source_route_resolver
        elif resolve_structure is None or resolve_names is None:
            default_structure, default_names = default_resolvers()
            resolve_structure = resolve_structure or default_structure
            resolve_names = resolve_names or default_names
            source_route_resolve_structure = resolve_structure
        source_route_resolve_structure = (
            source_route_resolve_structure or resolve_structure
        )

        anchor_smiles = []
        for value in [
            request.get("target_smiles"),
            *[row.get("product_smiles") for row in requested_edges],
        ]:
            smiles = str(value or "").strip()
            if smiles and smiles not in anchor_smiles:
                anchor_smiles.append(smiles)
        source_target_terms = [
            value
            for value in [str(request.get("target_name") or ""), *queries]
            if str(value).strip()
        ]

        source_dir = run_cache / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        sources: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for candidate in candidates:
            publication = str(candidate.get("publication_number") or "")
            source_byte_audit: dict[str, Any] = {}

            def cached_fetch(
                kind: str,
                upstream: BytesFetcher,
            ) -> BytesFetcher:
                def fetch_one(url: str, timeout_s: float, max_bytes: int) -> bytes:
                    content, audit = _publication_source_bytes(
                        cache_root,
                        publication=publication,
                        source_kind=kind,
                        source_url=url,
                        timeout_s=timeout_s,
                        max_bytes=max_bytes,
                        fetch=upstream,
                    )
                    source_byte_audit[kind] = audit
                    return content

                return fetch_one

            try:
                source, audit = _extract_candidate(
                    candidate,
                    edges=edges,
                    discovery_only=discovery_only,
                    discovery_pdf_allowed=discovery_pdf_allowed,
                    output_dir=source_dir,
                    config=config,
                    fetch=cached_fetch("pdf", fetch),
                    fetch_html=cached_fetch("html", fetch_html),
                    fetch_xml=cached_fetch("xml", fetch_html),
                    compile_registry=compile_registry,
                    resolve_structure=resolve_structure,
                    resolve_names=resolve_names,
                    source_route_resolve_structure=source_route_resolve_structure,
                    anchor_smiles=anchor_smiles,
                    target_name=str(request.get("target_name") or ""),
                    target_smiles=str(request.get("target_smiles") or ""),
                    target_terms=source_target_terms,
                    ocr_runner=ocr_runner,
                )
            except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
                audits.append(
                    {
                        "publication_number": candidate["publication_number"],
                        "accepted": False,
                        "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
                        "source_byte_cache": source_byte_audit,
                    }
                )
                continue
            audit["source_byte_cache"] = source_byte_audit
            audits.append(audit)
            if source:
                sources.append(source)
        resolver_cache_audit = (
            resolver_cache.flush() if resolver_cache is not None else None
        )
        discovery_sources = [
            {
                "source_kind": "patent",
                "source_ref": f"patent:{str(row.get('publication_number') or '')}",
                "publication_number": str(row.get("publication_number") or ""),
                "family_id": str(row.get("family_id") or ""),
                "title": str(row.get("title") or "")[:1000],
                "html_sha256": str(row.get("html_sha256") or ""),
                "xml_sha256": str(row.get("xml_sha256") or ""),
                "structured_text_sha256": str(
                    row.get("structured_text_sha256") or ""
                ),
                "html_audit": dict(row.get("html_audit") or {}),
                "structured_source_document_audit": dict(
                    row.get("structured_source_document_audit") or {}
                ),
                "pdf_sha256": str(row.get("pdf_sha256") or ""),
                "source_byte_cache": dict(row.get("source_byte_cache") or {}),
                "page_count": int(row.get("page_count") or 0),
                "procedure_inventory": list(row.get("procedure_inventory") or [])[:64],
                "source_route_observation": dict(
                    row.get("source_route_observation") or {}
                ),
                "source_route_proposal_count": int(
                    row.get("source_route_proposal_count") or 0
                ),
                "ocr_audit": dict(row.get("ocr_audit") or {}),
                "visual_candidate_pages": list(
                    row.get("visual_candidate_pages") or []
                )[:8],
                "exact_edge_ids": list(row.get("accepted_edge_ids") or [])[:128],
                "exact_row_count": len(row.get("accepted_edge_ids") or []),
                "approved_exact_row_count": int(
                    row.get("approved_exact_row_count") or 0
                ),
                "source_route_exact_row_count": int(
                    row.get("source_route_exact_row_count") or 0
                ),
                "source_route_exact_step_ids": list(
                    row.get("source_route_exact_step_ids") or []
                )[:128],
                "unresolved_edge_count": int(row.get("rejected_edge_count") or 0),
            }
            for row in audits
            if str(
                row.get("structured_text_sha256")
                or row.get("html_sha256")
                or row.get("xml_sha256")
                or row.get("pdf_sha256")
                or ""
            )
        ]
        if not discovery_sources:
            detail = _digest({"audits": audits})[:16]
            raise LiveEvidenceConnectorError(
                f"builtin_patent_evidence_no_source_material:{detail}"
            )
        discovery = {
            "schema_version": SOURCE_DISCOVERY_OBSERVATION_SCHEMA,
            "provider_id": BUILTIN_PATENT_PROVIDER_ID,
            "request_sha256": str(request.get("content_sha256") or ""),
            "sources": discovery_sources,
            "semantics": {
                "source_text_is_untrusted_data": True,
                "discovery_does_not_grant_exact_evidence": True,
                "host_validation_required_for_every_proposed_edge": True,
            },
        }
        discovery["content_sha256"] = _digest(discovery)
        receipt = {
            "schema_version": "evidence_connector_receipt.v1",
            "provider_id": BUILTIN_PATENT_PROVIDER_ID,
            "provider_version": BUILTIN_PATENT_PROVIDER_VERSION,
            "request_sha256": str(request.get("content_sha256") or ""),
            "query_count": len(queries),
            "candidate_count": len(candidates),
            "candidate_cache_hit": candidate_cache_hit,
            "accepted_source_count": len(sources),
            "audits": audits,
            "model_invocations": 0,
            "resolver_cache": resolver_cache_audit,
            "semantics": {
                "search_metadata_is_not_evidence": True,
                "official_epo_xml_then_html_is_attempted_before_pdf": True,
                "xml_html_or_pdf_source_bytes_are_frozen": True,
                "pdf_ocr_and_vision_are_unresolved_only_fallbacks": True,
                "exact_rows_are_deterministically_reconstructed": True,
                "target_only_prefetch_grants_no_evidence_authority": True,
                "unvalidated_edges_fall_back_to_discovery_only": True,
                "unvalidated_edge_discovery_only": (
                    unvalidated_edge_discovery_only
                ),
            },
        }
        receipt["content_sha256"] = _digest(receipt)
        result: dict[str, Any] = {
            "discovery": discovery,
            "receipt": receipt,
        }
        if sources:
            result["document"] = {
                "schema_version": "structured_evidence_import.v1",
                "sources": sources,
            }
        return result

    setattr(invoke, "autoplanner_prefetch_safe", True)
    return invoke


def _run_scoped_candidates(
    run_cache: Path,
    *,
    queries: list[str],
    limit: int,
    search: PatentCandidateProvider,
) -> tuple[list[dict[str, Any]], bool]:
    key = _digest(
        {
            "provider_id": BUILTIN_PATENT_PROVIDER_ID,
            "provider_version": BUILTIN_PATENT_PROVIDER_VERSION,
            "queries": queries,
            "limit": limit,
        }
    )
    path = run_cache / f"candidate-search-{key[:24]}.json"
    cached = _read_json_mapping(path)
    supplied = str(cached.get("content_sha256") or "")
    body = {name: value for name, value in cached.items() if name != "content_sha256"}
    if (
        cached.get("schema_version") == "run_scoped_patent_candidates.v1"
        and cached.get("cache_key") == key
        and isinstance(cached.get("candidates"), list)
        and supplied == _digest(body)
    ):
        return (
            [
                dict(value)
                for value in cached["candidates"][:limit]
                if isinstance(value, Mapping)
            ],
            True,
        )
    candidates = select_independent_candidates(
        search(queries),
        queries=queries,
        limit=limit,
    )
    row = {
        "schema_version": "run_scoped_patent_candidates.v1",
        "cache_key": key,
        "queries": list(queries),
        "candidates": [
            {
                name: value
                for name, value in dict(candidate).items()
                if not str(name).startswith("_")
                and isinstance(value, (str, int, float, bool, type(None)))
            }
            for candidate in candidates
        ],
        "semantics": {
            "cache_is_scoped_to_one_blind_run": True,
            "cache_grants_no_evidence_authority": True,
        },
    }
    row["content_sha256"] = _digest(row)
    _write_json_atomic(path, row)
    return candidates, False


def _publication_source_bytes(
    cache_root: Path,
    *,
    publication: str,
    source_kind: str,
    source_url: str,
    timeout_s: float,
    max_bytes: int,
    fetch: BytesFetcher,
) -> tuple[bytes, dict[str, Any]]:
    """Reuse frozen public source bytes without sharing target-derived state."""

    normalized_publication = "".join(re.findall(r"[A-Za-z0-9]+", publication)).upper()
    if not normalized_publication:
        raise ValueError("patent_source_cache_publication_invalid")
    if source_kind not in {"html", "pdf", "xml"}:
        raise ValueError("patent_source_cache_kind_invalid")
    publication_key = hashlib.sha256(
        normalized_publication.encode("utf-8")
    ).hexdigest()[:24]
    directory = cache_root / "_publication_source_cache" / publication_key
    metadata_path = directory / f"{source_kind}.json"
    metadata = _read_json_mapping(metadata_path)
    cached_path = directory / str(metadata.get("file_name") or "")
    supplied_sha256 = str(metadata.get("content_sha256") or "").lower()
    if (
        metadata.get("schema_version") == "patent_publication_source_cache.v1"
        and str(metadata.get("publication") or "") == normalized_publication
        and str(metadata.get("source_kind") or "") == source_kind
        and supplied_sha256
        and cached_path.parent.resolve() == directory.resolve()
        and cached_path.is_file()
    ):
        try:
            content = cached_path.read_bytes()
        except OSError:
            content = b""
        if (
            0 < len(content) <= max_bytes
            and hashlib.sha256(content).hexdigest() == supplied_sha256
            and (source_kind != "pdf" or content.startswith(b"%PDF-"))
        ):
            return content, {
                "status": "reused",
                "cache_hit": True,
                "publication": normalized_publication,
                "source_kind": source_kind,
                "content_sha256": supplied_sha256,
                "size_bytes": len(content),
                "semantics": {
                    "cache_contains_source_bytes_only": True,
                    "target_derived_extraction_is_not_shared": True,
                    "content_hash_revalidated": True,
                },
            }

    content = fetch(source_url, timeout_s, max_bytes)
    if not content or len(content) > max_bytes:
        raise ValueError("patent_source_cache_content_size_invalid")
    if source_kind == "pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("patent_pdf_signature_invalid")
    content_sha256 = hashlib.sha256(content).hexdigest()
    # Keep the byte cache out of user-visible PDF/HTML materialization scans;
    # the typed metadata, signature check, and hash carry the source identity.
    file_name = f"{source_kind}-{content_sha256[:24]}.source"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / file_name
    if not path.is_file():
        _write_bytes_atomic(path, content)
    row = {
        "schema_version": "patent_publication_source_cache.v1",
        "publication": normalized_publication,
        "source_kind": source_kind,
        "source_url": source_url,
        "file_name": file_name,
        "content_sha256": content_sha256,
        "size_bytes": len(content),
        "semantics": {
            "cache_contains_source_bytes_only": True,
            "target_derived_extraction_is_not_shared": True,
            "content_hash_must_be_revalidated_on_reuse": True,
        },
    }
    _write_json_atomic(metadata_path, row)
    return content, {
        "status": "fetched",
        "cache_hit": False,
        "publication": normalized_publication,
        "source_kind": source_kind,
        "content_sha256": content_sha256,
        "size_bytes": len(content),
        "semantics": dict(row["semantics"]),
    }


def _cached_publication_pdf(
    patent_dir: Path,
    *,
    publication: str,
    max_bytes: int,
) -> Path | None:
    root = patent_dir.resolve()
    for path in sorted(patent_dir.glob(f"{publication}-*.pdf")):
        resolved = path.resolve()
        try:
            size = resolved.stat().st_size
            header = resolved.read_bytes()[:5]
        except OSError:
            continue
        if (
            resolved.parent == root
            and 5 <= size <= max_bytes
            and header == b"%PDF-"
        ):
            return resolved
    return None


def _cached_pdf_manifest(
    path: Path,
    *,
    pdf_sha256: str,
    render_zoom: float,
    page_numbers: Iterable[int],
) -> dict[str, Any]:
    row = _read_json_mapping(path)
    pages = [
        dict(value)
        for value in row.get("rendered_pages") or []
        if isinstance(value, Mapping)
    ]
    if (
        row.get("accepted") is not True
        or str(row.get("source_pdf_sha256") or "").lower() != pdf_sha256
        or not pages
    ):
        return {}
    root = path.parent.resolve()
    expected_pages = sorted({int(value) for value in page_numbers if int(value) > 0})
    rendered_pages = sorted(
        {
            int(page.get("page_number") or 0)
            for page in pages
            if int(page.get("page_number") or 0) > 0
        }
    )
    if rendered_pages != expected_pages:
        return {}
    for page in pages:
        image = Path(str(page.get("image_path") or "")).expanduser().resolve()
        try:
            within_root = image.is_relative_to(root)
        except ValueError:
            within_root = False
        if (
            not within_root
            or not image.is_file()
            or float(page.get("render_zoom") or 0.0) != float(render_zoom)
            or str(page.get("sha256") or "").lower()
            != hashlib.sha256(image.read_bytes()).hexdigest()
        ):
            return {}
    return row


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _extract_candidate(
    candidate: Mapping[str, Any],
    *,
    edges: list[dict[str, Any]],
    discovery_only: bool = False,
    discovery_pdf_allowed: bool = True,
    output_dir: Path,
    config: BuiltinPatentEvidenceConfig,
    fetch: BytesFetcher,
    fetch_html: BytesFetcher,
    fetch_xml: BytesFetcher,
    compile_registry: RegistryCompiler,
    resolve_structure: StructureResolver,
    resolve_names: CandidateNameResolver,
    source_route_resolve_structure: StructureResolver,
    anchor_smiles: Iterable[str],
    target_name: str,
    target_smiles: str,
    target_terms: Iterable[str],
    ocr_runner: OcrRunner | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    publication = str(candidate["publication_number"])
    source_ref = f"patent:{publication}"
    patent_dir = output_dir / publication.lower()
    patent_dir.mkdir(parents=True, exist_ok=True)
    target_name_key = _source_name_identity(target_name)

    def resolve_source_name(name: str) -> str:
        if (
            target_smiles
            and target_name_key
            and _source_name_identity(name) == target_name_key
        ):
            return target_smiles
        return source_route_resolve_structure(name)

    html_attempt = (
        attempt_primary_patent_html(
            candidate,
            edges=edges,
            output_dir=patent_dir,
            timeout_s=config.timeout_s,
            max_html_bytes=config.max_html_bytes,
            max_html_sections=config.max_html_sections,
            max_html_paragraphs=config.max_html_paragraphs,
            fetch=fetch_html,
            fetch_xml=fetch_xml,
            compile_registry=compile_registry,
            resolve_structure=resolve_source_name,
            resolve_names=resolve_names,
            target_terms=target_terms,
        )
        if config.enable_html_first
        else {"status": "disabled", "reason": "primary_patent_html_disabled"}
    )
    html_registry = dict(html_attempt.get("registry_audit") or {})
    rows, accepted_edges = _exact_rows_from_registry(
        html_registry,
        publication=publication,
    )
    html_row_count = len(rows)
    accepted_ids = set(accepted_edges)
    remaining_edges = [
        edge
        for edge in edges
        if str(edge.get("edge_id") or edge.get("edge_digest") or "")
        not in accepted_ids
    ]

    pdf_sha256 = ""
    pdf_cache_hit = False
    page_count = 0
    manifest: dict[str, Any] = {}
    manifest_cache_hit = False
    ocr_audit: dict[str, Any] = {}
    pdf_registry: dict[str, Any] = {}
    source_route_observation: dict[str, Any] = {}
    structured_source_document: dict[str, Any] = {}
    route_metadata: dict[str, dict[str, Any]] = {}
    html_materialization = dict(html_attempt.get("materialization") or {})
    structured_companion = dict(html_materialization.get("companion") or {})
    if (
        str(html_materialization.get("artifact_sha256") or "")
        and structured_companion
    ):
        structured_source_document = (
            extract_deterministic_structured_source_document(
                structured_companion,
                source_ref=source_ref,
                structure_resolver=resolve_source_name,
            )
        )
        source_route_observation = (
            compile_deterministic_source_route_observation(
                structured_source_document,
                structure_resolver=resolve_source_name,
                anchor_smiles=anchor_smiles,
            )
        )
    if remaining_edges or (
        discovery_only
        and discovery_pdf_allowed
        and not str(html_materialization.get("artifact_sha256") or "")
    ):
        pdf_url = str(candidate.get("pdf_url") or "").strip()
        if not pdf_url:
            if not rows:
                raise ValueError("patent_pdf_fallback_url_missing")
        else:
            cached_pdf = _cached_publication_pdf(
                patent_dir,
                publication=publication,
                max_bytes=config.max_pdf_bytes,
            )
            content = (
                cached_pdf.read_bytes()
                if cached_pdf is not None
                else fetch(pdf_url, config.timeout_s, config.max_pdf_bytes)
            )
            pdf_cache_hit = cached_pdf is not None
            if not content.startswith(b"%PDF-"):
                raise ValueError("patent_pdf_signature_invalid")
            pdf_sha256 = hashlib.sha256(content).hexdigest()
            pdf_path = patent_dir / f"{publication}-{pdf_sha256[:16]}.pdf"
            if not pdf_path.is_file():
                _write_bytes_atomic(pdf_path, content)
            page_count = _pdf_page_count(pdf_path)
            if page_count < 1 or page_count > config.max_pdf_pages:
                raise ValueError(f"patent_pdf_page_limit:{page_count}")
            manifest_path = (
                patent_dir / "materialized" / "literature_pdf_structure_evidence.json"
            )
            focus = rebuild_literature_pdf_page_focus(
                pdf_path,
                target_name=str(next(iter(target_terms), "")),
                target_aliases=[str(value) for value in target_terms],
                route_sequence_hint="; ".join(str(value) for value in target_terms),
            )
            selected_page_numbers = select_pdf_page_numbers(
                focus,
                page_count=page_count,
                max_pages=min(config.max_ocr_pages, page_count),
            )
            manifest = _cached_pdf_manifest(
                manifest_path,
                pdf_sha256=pdf_sha256,
                render_zoom=config.render_zoom,
                page_numbers=selected_page_numbers,
            )
            manifest_cache_hit = bool(manifest)
            if not manifest:
                manifest = extract_literature_pdf_assets(
                    pdf_path=pdf_path,
                    output_dir=patent_dir / "materialized",
                    page_numbers=selected_page_numbers,
                    target_name=str(next(iter(target_terms), "")),
                    target_aliases=[str(value) for value in target_terms],
                    route_sequence_hint="; ".join(
                        str(value) for value in target_terms
                    ),
                    render_zoom=config.render_zoom,
                )
            if manifest.get("accepted") is not True or not manifest.get("rendered_pages"):
                raise ValueError("patent_pdf_materialization_failed")
            document_id = f"patent:{publication}"
            manifest.update(
                {
                    "source_ref": source_ref,
                    "source_binding_audit": {
                        "schema_version": "local_pdf_source_binding_audit.v1",
                        "accepted": True,
                        "source_ref": source_ref,
                        "matched_source_count": 1,
                        "matched_document_ids": [document_id],
                        "binding_method": "builtin_patent_publication_and_pdf_hash",
                    },
                }
            )
            _write_json_atomic(manifest_path, manifest)
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            evidence = [
                {
                    "schema_version": "materialized_source_evidence.v1",
                    "document_id": document_id,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                    "source_pdf_path": str(pdf_path),
                    "source_pdf_sha256": pdf_sha256,
                    "page_number": int(page["page_number"]),
                    "image_path": str(page["image_path"]),
                    "image_sha256": str(page["sha256"]),
                    "source_ref": source_ref,
                }
                for page in manifest["rendered_pages"]
            ]
            if config.enable_local_ocr:
                ocr_audit = materialize_local_ocr_companion(
                    pdf_path=pdf_path,
                    source_ref=source_ref,
                    rendered_pages=manifest["rendered_pages"],
                    output_dir=patent_dir / "ocr",
                    target_terms=target_terms,
                    config=LocalOcrConfig(
                        max_pages=min(config.max_ocr_pages, config.max_pdf_pages)
                    ),
                    runner=ocr_runner,
                )
            companions = (
                [dict(ocr_audit["companion"])]
                if isinstance(ocr_audit.get("companion"), Mapping)
                and ocr_audit.get("companion")
                else []
            )
            source_document = extract_deterministic_source_document(
                pdf_path,
                source_ref=source_ref,
                source_pdf_sha256=pdf_sha256,
                structure_resolver=resolve_source_name,
                source_text_companions=companions,
            )
            pdf_source_route_observation = (
                compile_deterministic_source_route_observation(
                    source_document,
                    structure_resolver=resolve_source_name,
                    source_evidence=evidence,
                    anchor_smiles=[
                        *[str(value) for value in anchor_smiles if str(value)],
                        *[
                            str(edge.get("product_smiles") or "")
                            for edge in edges
                            if str(edge.get("product_smiles") or "")
                        ],
                    ],
                )
            )
            if (
                int(pdf_source_route_observation.get("proposal_count") or 0)
                >= int(source_route_observation.get("proposal_count") or 0)
            ):
                source_route_observation = pdf_source_route_observation
            steps = _pdf_registry_steps(
                remaining_edges,
                source_ref=source_ref,
                evidence=evidence,
                companions=companions,
                resolve_names=resolve_names,
            )
            route_proposals = [
                dict(row)
                for row in source_route_observation.get("proposals") or []
                if isinstance(row, Mapping)
            ]
            existing_step_identities = {
                _reaction_identity(
                    row.get("product_smiles"),
                    row.get("reactant_smiles") or row.get("precursor_smiles") or [],
                )
                for row in steps
            }
            route_registry_steps = [
                {
                    "step_id": str(row.get("proposal_id") or row.get("step_id") or ""),
                    "product_name": str(row.get("product_name") or ""),
                    "product_smiles": str(row.get("product_smiles") or ""),
                    "reactant_names": list(row.get("reactant_names") or []),
                    "reactant_smiles": list(row.get("reactant_smiles") or []),
                    "condition_candidate": dict(row.get("condition_candidate") or {}),
                    "source_ref": source_ref,
                    "source_evidence": [dict(value) for value in evidence],
                    "source_text_companions": [dict(value) for value in companions],
                }
                for row in route_proposals
                if _reaction_identity(
                    row.get("product_smiles"),
                    row.get("reactant_smiles") or row.get("precursor_smiles") or [],
                )
                not in existing_step_identities
            ]
            steps.extend(route_registry_steps)
            pdf_registry = dict(
                compile_registry(
                    steps,
                    registry_path=patent_dir / "pdf-deterministic-step-registry.json",
                    structure_resolver=resolve_source_name,
                    candidate_name_resolver=resolve_names,
                    timeout_s=config.timeout_s,
                )
            )
            pdf_rows, pdf_edges = _exact_rows_from_registry(
                pdf_registry,
                publication=publication,
            )
            route_metadata = {
                str(row.get("proposal_id") or row.get("step_id") or ""): row
                for row in route_proposals
            }
            for row in pdf_rows:
                proposal = route_metadata.get(str(row.get("step_id") or ""))
                if not proposal:
                    continue
                row["route_family_id"] = str(
                    proposal.get("route_family_id") or ""
                )
                row["origin_kind"] = "literature_source_route"
            rows.extend(pdf_rows)
            current_edge_ids = {
                str(edge.get("edge_id") or edge.get("edge_digest") or "")
                for edge in remaining_edges
            }
            matched_current_edges = sorted(set(pdf_edges) & current_edge_ids)
            accepted_edges.extend(matched_current_edges)
            accepted_ids.update(matched_current_edges)

    rows = list({str(row["step_id"]): row for row in rows}.values())
    source_route_exact_step_ids = sorted(
        {
            str(row.get("step_id") or "")
            for row in rows
            if str(row.get("step_id") or "") in route_metadata
        }
        - {""}
    )
    accepted_edges = sorted(accepted_ids)
    structured_artifact_kind = str(
        html_attempt.get("source_artifact_kind") or ""
    )
    source = _structured_patent_source(
        candidate,
        source_ref=source_ref,
        rows=rows,
        used_structured_text=html_row_count > 0,
        structured_artifact_kind=structured_artifact_kind,
        used_pdf=len(rows) > html_row_count,
    )
    structured_inventory_audit = _source_document_inventory_audit(
        structured_source_document
    )
    procedure_inventory = _procedure_inventory(
        html_registry,
        pdf_registry,
        structured_inventory_audit,
    )
    return source, {
        "publication_number": publication,
        "family_id": str(candidate.get("family_id") or ""),
        "title": str(candidate.get("title") or publication),
        "html_sha256": (
            str(html_materialization.get("artifact_sha256") or "")
            if structured_artifact_kind == "html"
            else ""
        ),
        "xml_sha256": (
            str(html_materialization.get("artifact_sha256") or "")
            if structured_artifact_kind == "xml"
            else ""
        ),
        "structured_text_sha256": str(
            html_materialization.get("artifact_sha256") or ""
        ),
        "html_audit": _bounded_html_audit(html_attempt),
        "pdf_sha256": pdf_sha256,
        "pdf_cache_hit": pdf_cache_hit,
        "manifest_cache_hit": manifest_cache_hit,
        "page_count": page_count,
        "accepted": bool(rows),
        "accepted_edge_ids": accepted_edges,
        "approved_exact_row_count": len(rows),
        "source_route_exact_row_count": len(source_route_exact_step_ids),
        "source_route_exact_step_ids": source_route_exact_step_ids,
        "rejected_edge_count": max(0, len(edges) - len(accepted_edges)),
        "registry_audit_sha256": _digest(
            {
                "html": str(html_registry.get("content_sha256") or ""),
                "pdf": str(pdf_registry.get("content_sha256") or ""),
            }
        ),
        "procedure_inventory": procedure_inventory,
        "source_route_observation": source_route_observation,
        "source_route_proposal_count": int(
            source_route_observation.get("proposal_count") or 0
        ),
        "structured_source_document_audit": {
            "accepted": structured_source_document.get("accepted") is True,
            "source_artifact_kind": str(
                structured_source_document.get("source_artifact_kind") or ""
            ),
            "procedure_count": int(
                structured_source_document.get("procedure_count") or 0
            ),
            "resolved_procedure_count": int(
                structured_source_document.get("resolved_procedure_count") or 0
            ),
            "reasons": list(structured_source_document.get("reasons") or [])[:16],
        },
        "ocr_audit": _bounded_ocr_audit(ocr_audit),
        "visual_candidate_pages": _visual_candidate_pages(
            manifest,
            procedure_inventory=procedure_inventory,
            ocr_audit=ocr_audit,
            source_route_observation=source_route_observation,
        ),
    }


def _pdf_registry_steps(
    edges: Iterable[Mapping[str, Any]],
    *,
    source_ref: str,
    evidence: Iterable[Mapping[str, Any]],
    companions: Iterable[Mapping[str, Any]],
    resolve_names: CandidateNameResolver,
) -> list[dict[str, Any]]:
    frozen_evidence = [dict(row) for row in evidence]
    frozen_companions = [dict(row) for row in companions]
    steps: list[dict[str, Any]] = []
    for edge in edges:
        product = str(edge.get("product_smiles") or "")
        names = resolve_names(product)
        steps.append(
            {
                "step_id": str(
                    edge.get("edge_id") or edge.get("edge_digest") or ""
                ),
                "product_name": str(names[0]) if names else "",
                "product_smiles": product,
                "reactant_smiles": list(edge.get("precursor_smiles") or []),
                "source_ref": source_ref,
                "source_evidence": frozen_evidence,
                "source_text_companions": frozen_companions,
            }
        )
    return steps


def _reaction_identity(product: Any, reactants: Iterable[Any]) -> str:
    return _digest(
        {
            "product_smiles": str(product or ""),
            "reactant_smiles": sorted(str(value) for value in reactants if str(value)),
        }
    )


def _exact_rows_from_registry(
    audit: Mapping[str, Any],
    *,
    publication: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    edge_ids: list[str] = []
    for record in audit.get("records") or []:
        if not isinstance(record, Mapping) or record.get("accepted") is not True:
            continue
        binding = dict(record.get("binding") or {})
        projection = dict(binding.get("synthesis_projection") or {})
        product = str(projection.get("product_smiles") or "")
        reactants = [
            str(value)
            for value in projection.get("reactant_smiles") or []
            if str(value)
        ]
        edge_id = str(record.get("step_id") or "")
        if not edge_id or not product or not reactants:
            continue
        artifact_kind = str(binding.get("source_artifact_kind") or "pdf")
        source_location = dict(binding.get("source_location") or {})
        parser_audit = dict(binding.get("parser_audit") or {})
        evidence_refs: list[str] = []
        if artifact_kind in {"html", "xml"}:
            start = str(source_location.get("start_element_id") or "")
            end = str(source_location.get("end_element_id") or "")
            location_ref = f"{publication}:{artifact_kind}:{start}-{end}"
            artifact_sha256 = str(
                binding.get("source_artifact_sha256") or ""
            )
            text_sha256 = str(source_location.get("text_sha256") or "")
            if artifact_sha256:
                evidence_refs.append(
                    f"{artifact_kind}_sha256:{artifact_sha256}"
                )
            if text_sha256:
                evidence_refs.append(f"text_sha256:{text_sha256}")
        else:
            page_number = int(binding.get("page_number") or 0)
            location_ref = f"{publication}:page:{page_number}"
            pdf_sha256 = str(
                binding.get("source_pdf_sha256")
                or binding.get("source_artifact_sha256")
                or ""
            )
            image_sha256 = str(binding.get("image_sha256") or "")
            if pdf_sha256:
                evidence_refs.append(f"pdf_sha256:{pdf_sha256}")
            if image_sha256:
                evidence_refs.append(f"image_sha256:{image_sha256}")
        procedure_text_sha256 = str(
            parser_audit.get("procedure_text_sha256") or ""
        )
        if procedure_text_sha256:
            evidence_refs.append(
                f"procedure-text-sha256:{procedure_text_sha256}"
            )
        rows.append(
            {
                "step_id": edge_id,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "relation_type": "exact",
                "location_ref": location_ref,
                "evidence_refs": evidence_refs,
                "conditions": dict(binding.get("source_conditions") or {}),
            }
        )
        edge_ids.append(edge_id)
    return rows, edge_ids


def _structured_patent_source(
    candidate: Mapping[str, Any],
    *,
    source_ref: str,
    rows: list[dict[str, Any]],
    used_structured_text: bool,
    structured_artifact_kind: str,
    used_pdf: bool,
) -> dict[str, Any]:
    if not rows:
        return {}
    if used_structured_text and used_pdf:
        provenance = (
            "builtin_patent_xml_first_with_pdf_fallback"
            if structured_artifact_kind == "xml"
            else "builtin_patent_html_first_with_pdf_fallback"
        )
    elif used_structured_text:
        provenance = (
            "builtin_deterministic_primary_patent_xml"
            if structured_artifact_kind == "xml"
            else "builtin_deterministic_primary_patent_html"
        )
    else:
        provenance = "builtin_deterministic_patent_pdf_extraction"
    publication = str(candidate.get("publication_number") or "")
    return {
        "binding": {
            "source_kind": "patent",
            "source_ref": source_ref,
            "patent_publication": publication,
            "patent_family": str(candidate.get("family_id") or ""),
            "title": str(candidate.get("title") or publication),
            "provenance": provenance,
            "discovered_by": BUILTIN_PATENT_PROVIDER_ID,
        },
        "extraction": {
            "schema_version": "structured_exact_row_extraction.v1",
            "extractor": {
                "producer_kind": "deterministic_structure_parser",
                "producer_id": PARSER_AUTHORITY_ID,
                "version": BUILTIN_PATENT_PROVIDER_VERSION,
            },
            "rows": rows,
        },
    }


def _procedure_inventory(
    *audits: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for audit in audits:
        for document in audit.get("source_procedure_inventory") or []:
            if not isinstance(document, Mapping):
                continue
            artifact_kind = str(document.get("source_artifact_kind") or "pdf")
            artifact_sha256 = str(
                document.get("source_artifact_sha256")
                or document.get("source_pdf_sha256")
                or ""
            )
            for procedure in document.get("procedures") or []:
                if not isinstance(procedure, Mapping):
                    continue
                row = dict(procedure)
                identity = (
                    artifact_kind,
                    artifact_sha256,
                    int(row.get("page_number") or 0),
                    str(row.get("name") or "").casefold(),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(
                    {
                        **row,
                        "source_artifact_kind": artifact_kind,
                        "source_artifact_sha256": artifact_sha256,
                    }
                )
                if len(rows) >= 64:
                    return rows
    return rows


def _source_document_inventory_audit(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt direct structured-document replay to the bounded audit shape."""

    procedures = [
        {
            "label": str(row.get("label") or "")[:80],
            "name": " ".join(str(row.get("name") or "").split())[:1000],
            "page_number": int(row.get("page_number") or 0),
            "procedure_excerpt": " ".join(
                str(row.get("procedure") or "").split()
            )[:800],
        }
        for row in document.get("procedures") or []
        if isinstance(row, Mapping)
    ][:64]
    if not procedures:
        return {}
    return {
        "source_procedure_inventory": [
            {
                "source_ref": str(document.get("source_ref") or ""),
                "source_artifact_kind": str(
                    document.get("source_artifact_kind") or "html"
                ),
                "source_artifact_sha256": str(
                    document.get("source_artifact_sha256") or ""
                ),
                "procedure_count": len(procedures),
                "procedures": procedures,
                "semantics": {
                    "discovery_only": True,
                    "grants_no_exact_reaction_evidence": True,
                },
            }
        ]
    }


def _source_name_identity(value: Any) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split()
    )


def _bounded_html_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value or {})
    materialization = dict(row.get("materialization") or {})
    bounded = {
        key: row.get(key)
        for key in (
            "schema_version",
            "status",
            "reason",
            "accepted_edge_ids",
            "attempted_edge_count",
            "model_invocations",
            "visual_invocations",
        )
        if key in row
    }
    if materialization:
        bounded["materialization"] = {
            key: materialization.get(key)
            for key in (
                "schema_version",
                "status",
                "reasons",
                "artifact_sha256",
                "paragraph_count",
                "selected_paragraph_count",
                "element_count",
                "selected_element_count",
                "section_count",
                "content_sha256",
            )
            if key in materialization
        }
    registry = dict(row.get("registry_audit") or {})
    if registry:
        bounded["registry_content_sha256"] = str(
            registry.get("content_sha256") or ""
        )
    return bounded


def _bounded_ocr_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value or {})
    return {
        key: row.get(key)
        for key in (
            "schema_version",
            "status",
            "reasons",
            "source_pdf_sha256",
            "native_text_page_count",
            "low_text_page_count",
            "selected_page_count",
            "selected_page_numbers",
            "ocr_page_count",
            "failure_count",
            "coverage_truncated",
            "focus_page_numbers",
            "content_sha256",
        )
        if key in row
    }


def _visual_candidate_pages(
    manifest: Mapping[str, Any],
    *,
    procedure_inventory: Iterable[Mapping[str, Any]],
    ocr_audit: Mapping[str, Any],
    source_route_observation: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rendered = {
        int(row.get("page_number") or 0): dict(row)
        for row in manifest.get("rendered_pages") or []
        if isinstance(row, Mapping) and int(row.get("page_number") or 0) > 0
    }
    # Spend visual budget first on deterministic extraction failures: a source
    # heading recovered only from a downstream mention, or a proposed reaction
    # whose parsed reactants cannot supply every product element.  Routine
    # text-complete pages remain bounded fallbacks after those pages.
    ambiguous_pages: list[int] = []
    for raw in dict(source_route_observation or {}).get("proposals") or []:
        proposal = dict(raw) if isinstance(raw, Mapping) else {}
        audit = dict(proposal.get("admission_audit") or {})
        deficits = dict(audit.get("element_deficits") or {})
        ambiguous = (
            str(proposal.get("product_structure_recovery_mode") or "")
            != "source_heading_opsin"
            or any(int(value or 0) > 0 for value in deficits.values())
        )
        location = dict(proposal.get("source_location") or {})
        page_number = int(location.get("page_number") or 0)
        if ambiguous and page_number > 0:
            ambiguous_pages.append(page_number)
    page_numbers = [*ambiguous_pages, *[
        int(row.get("page_number") or 0)
        for row in procedure_inventory
        if int(row.get("page_number") or 0) > 0
    ]]
    page_numbers.extend(
        int(value)
        for value in ocr_audit.get("focus_page_numbers") or []
        if int(value) > 0
    )
    page_numbers.extend(
        int(value)
        for value in manifest.get("focus_page_numbers") or []
        if int(value) > 0
    )
    page_numbers.extend(
        int(row.get("page_number") or 0)
        for row in ocr_audit.get("visual_candidate_pages") or []
        if isinstance(row, Mapping) and int(row.get("page_number") or 0) > 0
    )
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for page_number in page_numbers:
        if page_number in seen or page_number not in rendered:
            continue
        seen.add(page_number)
        page = rendered[page_number]
        rows.append(
            {
                "page_number": page_number,
                "image_path": str(page.get("image_path") or ""),
                "image_sha256": str(page.get("sha256") or ""),
            }
        )
        if len(rows) >= 8:
            break
    return rows

def _pdf_page_count(path: Path) -> int:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pymupdf_unavailable") from exc
    document = fitz.open(str(path))
    try:
        return len(document)
    finally:
        document.close()

def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _digest(value: Mapping[str, Any]) -> str:
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
    "BUILTIN_PATENT_PROVIDER_ID",
    "BUILTIN_PATENT_PROVIDER_VERSION",
    "BuiltinPatentEvidenceConfig",
    "build_builtin_patent_evidence_connector",
]
