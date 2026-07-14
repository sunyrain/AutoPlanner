"""Built-in, zero-model patent evidence acquisition.

The connector searches only for source material.  It freezes patent PDFs,
renders hash-bound page images, and delegates exact chemistry reconstruction
to the existing deterministic literature parser.  Search metadata, snippets,
and Codex source hints never become exact evidence rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote

import requests

from cascade_planner.harness.deterministic_literature_registry import (
    DEFAULT_OPSIN_BASE_URL,
    DEFAULT_PUBCHEM_BASE_URL,
    PARSER_AUTHORITY_ID,
    CandidateNameResolver,
    StructureResolver,
    build_deterministic_literature_resolvers,
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.deterministic_resolver_cache import (
    DeterministicResolverCache,
)
from cascade_planner.harness.literature_pdf_extraction import (
    extract_literature_pdf_assets,
)
from cascade_planner.harness.source_ocr import (
    LocalOcrConfig,
    OcrRunner,
    materialize_local_ocr_companion,
)
from cascade_planner.interfaces.live_evidence import LiveEvidenceConnectorError


BUILTIN_PATENT_PROVIDER_ID = "autoplanner.builtin_patent_evidence"
BUILTIN_PATENT_PROVIDER_VERSION = "1.1.0"
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
    registry_compiler: RegistryCompiler | None = None,
    structure_resolver: StructureResolver | None = None,
    candidate_name_resolver: CandidateNameResolver | None = None,
    ocr_runner: OcrRunner | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build a bounded first-party connector for current validated edges."""

    cache_root = Path(config.cache_dir).expanduser().resolve()
    search = candidate_provider or _google_patent_candidate_provider(config)
    fetch = bytes_fetcher or _fetch_bounded_bytes
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
        edges = [
            dict(row)
            for row in request.get("edges") or []
            if isinstance(row, Mapping)
            and row.get("current_host_reaction_validated") is True
        ][: config.max_validated_edges]
        if not edges:
            raise LiveEvidenceConnectorError(
                "builtin_patent_evidence_no_validated_edges"
            )
        queries = _evidence_queries(request, limit=config.max_search_queries)
        candidates = _select_independent_candidates(
            search(queries),
            queries=queries,
            limit=config.max_patents,
        )
        if not candidates:
            raise LiveEvidenceConnectorError(
                "builtin_patent_evidence_no_patent_candidates"
            )
        resolve_structure = structure_resolver
        resolve_names = candidate_name_resolver
        if resolve_structure is None or resolve_names is None:
            default_structure, default_names = default_resolvers()
            resolve_structure = resolve_structure or default_structure
            resolve_names = resolve_names or default_names

        request_dir = cache_root / str(request.get("content_sha256") or "")[:24]
        request_dir.mkdir(parents=True, exist_ok=True)
        sources: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                source, audit = _extract_candidate(
                    candidate,
                    edges=edges,
                    output_dir=request_dir,
                    config=config,
                    fetch=fetch,
                    compile_registry=compile_registry,
                    resolve_structure=resolve_structure,
                    resolve_names=resolve_names,
                    target_terms=queries,
                    ocr_runner=ocr_runner,
                )
            except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
                audits.append(
                    {
                        "publication_number": candidate["publication_number"],
                        "accepted": False,
                        "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
                    }
                )
                continue
            audits.append(audit)
            if source:
                sources.append(source)
        resolver_cache_audit = (
            resolver_cache.flush() if resolver_cache is not None else None
        )
        discovery_sources = [
            {
                "publication_number": str(row.get("publication_number") or ""),
                "family_id": str(row.get("family_id") or ""),
                "title": str(row.get("title") or "")[:1000],
                "pdf_sha256": str(row.get("pdf_sha256") or ""),
                "page_count": int(row.get("page_count") or 0),
                "procedure_inventory": list(row.get("procedure_inventory") or [])[:64],
                "ocr_audit": dict(row.get("ocr_audit") or {}),
                "visual_candidate_pages": list(
                    row.get("visual_candidate_pages") or []
                )[:8],
                "exact_edge_ids": list(row.get("accepted_edge_ids") or [])[:128],
                "exact_row_count": len(row.get("accepted_edge_ids") or []),
                "unresolved_edge_count": int(row.get("rejected_edge_count") or 0),
            }
            for row in audits
            if str(row.get("pdf_sha256") or "")
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
            "accepted_source_count": len(sources),
            "audits": audits,
            "model_invocations": 0,
            "resolver_cache": resolver_cache_audit,
            "semantics": {
                "search_metadata_is_not_evidence": True,
                "pdf_and_page_hashes_are_frozen": True,
                "exact_rows_are_deterministically_reconstructed": True,
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

    return invoke


def _extract_candidate(
    candidate: Mapping[str, Any],
    *,
    edges: list[dict[str, Any]],
    output_dir: Path,
    config: BuiltinPatentEvidenceConfig,
    fetch: BytesFetcher,
    compile_registry: RegistryCompiler,
    resolve_structure: StructureResolver,
    resolve_names: CandidateNameResolver,
    target_terms: Iterable[str],
    ocr_runner: OcrRunner | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    publication = str(candidate["publication_number"])
    source_ref = f"patent:{publication}"
    patent_dir = output_dir / publication.lower()
    patent_dir.mkdir(parents=True, exist_ok=True)
    content = fetch(
        str(candidate["pdf_url"]),
        config.timeout_s,
        config.max_pdf_bytes,
    )
    if not content.startswith(b"%PDF-"):
        raise ValueError("patent_pdf_signature_invalid")
    pdf_sha256 = hashlib.sha256(content).hexdigest()
    pdf_path = patent_dir / f"{publication}-{pdf_sha256[:16]}.pdf"
    if not pdf_path.is_file():
        _write_bytes_atomic(pdf_path, content)
    page_count = _pdf_page_count(pdf_path)
    if page_count < 1 or page_count > config.max_pdf_pages:
        raise ValueError(f"patent_pdf_page_limit:{page_count}")
    manifest = extract_literature_pdf_assets(
        pdf_path=pdf_path,
        output_dir=patent_dir / "materialized",
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
    manifest_path = patent_dir / "materialized" / "literature_pdf_structure_evidence.json"
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
    ocr_audit: dict[str, Any] = {}
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
    source_text_companions = (
        [dict(ocr_audit["companion"])]
        if isinstance(ocr_audit.get("companion"), Mapping)
        and ocr_audit.get("companion")
        else []
    )
    steps = []
    for edge in edges:
        product = str(edge.get("product_smiles") or "")
        names = resolve_names(product)
        steps.append(
            {
                "step_id": str(edge.get("edge_id") or edge.get("edge_digest") or ""),
                "product_name": str(names[0]) if names else "",
                "product_smiles": product,
                "reactant_smiles": list(edge.get("precursor_smiles") or []),
                "source_ref": source_ref,
                "source_evidence": evidence,
                "source_text_companions": source_text_companions,
            }
        )
    registry_path = patent_dir / "deterministic-step-registry.json"
    audit = dict(
        compile_registry(
            steps,
            registry_path=registry_path,
            structure_resolver=resolve_structure,
            candidate_name_resolver=resolve_names,
            timeout_s=config.timeout_s,
        )
    )
    rows = []
    accepted_edges: list[str] = []
    for record in audit.get("records") or []:
        if not isinstance(record, Mapping) or record.get("accepted") is not True:
            continue
        binding = dict(record.get("binding") or {})
        projection = dict(binding.get("synthesis_projection") or {})
        product = str(projection.get("product_smiles") or "")
        reactants = [str(value) for value in projection.get("reactant_smiles") or []]
        if not product or not reactants:
            continue
        edge_id = str(record.get("step_id") or "")
        rows.append(
            {
                "step_id": edge_id,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "relation_type": "exact",
                "location_ref": f"{publication}:page:{int(binding.get('page_number') or 0)}",
                "evidence_refs": [
                    f"pdf_sha256:{pdf_sha256}",
                    f"image_sha256:{str(binding.get('image_sha256') or '')}",
                ],
            }
        )
        accepted_edges.append(edge_id)
    source = {}
    if rows:
        source = {
            "binding": {
                "source_kind": "patent",
                "source_ref": source_ref,
                "patent_publication": publication,
                "patent_family": str(candidate.get("family_id") or ""),
                "title": str(candidate.get("title") or publication),
                "provenance": "builtin_deterministic_patent_pdf_extraction",
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
    procedure_inventory = [
        dict(procedure)
        for document in audit.get("source_procedure_inventory") or []
        if isinstance(document, Mapping)
        for procedure in document.get("procedures") or []
        if isinstance(procedure, Mapping)
    ][:64]
    return source, {
        "publication_number": publication,
        "family_id": str(candidate.get("family_id") or ""),
        "title": str(candidate.get("title") or publication),
        "pdf_sha256": pdf_sha256,
        "page_count": page_count,
        "accepted": bool(rows),
        "accepted_edge_ids": sorted(accepted_edges),
        "rejected_edge_count": max(0, len(edges) - len(rows)),
        "registry_audit_sha256": str(audit.get("content_sha256") or ""),
        "procedure_inventory": procedure_inventory,
        "ocr_audit": _bounded_ocr_audit(ocr_audit),
        "visual_candidate_pages": _visual_candidate_pages(
            manifest,
            procedure_inventory=procedure_inventory,
            ocr_audit=ocr_audit,
        ),
    }


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
) -> list[dict[str, Any]]:
    rendered = {
        int(row.get("page_number") or 0): dict(row)
        for row in manifest.get("rendered_pages") or []
        if isinstance(row, Mapping) and int(row.get("page_number") or 0) > 0
    }
    page_numbers = [
        int(row.get("page_number") or 0)
        for row in procedure_inventory
        if int(row.get("page_number") or 0) > 0
    ]
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


def _evidence_queries(request: Mapping[str, Any], *, limit: int) -> list[str]:
    values = [str(request.get("target_name") or "").strip()]
    values.extend(
        str(row.get("query") or "").strip()
        for row in request.get("source_tasks") or []
        if isinstance(row, Mapping)
    )
    values.extend(
        str(row.get("source_ref") or "").removeprefix("patent:").strip()
        for row in request.get("source_hints") or []
        if isinstance(row, Mapping)
    )
    out: list[str] = []
    for value in values:
        compact = " ".join(value.split())[:800]
        if compact and compact.casefold() not in {row.casefold() for row in out}:
            out.append(compact)
        if len(out) >= limit:
            break
    return out


def _google_patent_candidate_provider(
    config: BuiltinPatentEvidenceConfig,
) -> PatentCandidateProvider:
    def search(queries: Iterable[str]) -> Iterable[Mapping[str, Any]]:
        rows: list[dict[str, Any]] = []
        all_queries = [*config.seed_publications, *list(queries)]
        for query in all_queries[: config.max_search_queries]:
            explicit = _patent_publications(query)
            rows.extend(_direct_pdf_candidates(query, explicit))
            for publication in explicit:
                direct = _resolve_google_patent_publication(
                    publication,
                    timeout_s=config.timeout_s,
                )
                if direct:
                    rows.append(direct)
            search_values = [*explicit, query]
            for search_value in search_values:
                nested = f"q=({search_value})"
                for page in range(config.max_search_pages_per_query):
                    if page:
                        nested_page = f"{nested}&page={page}"
                    else:
                        nested_page = nested
                    url = (
                        "https://patents.google.com/xhr/query?url="
                        + quote(nested_page, safe="")
                    )
                    response = requests.get(
                        url,
                        headers={"User-Agent": "AutoPlanner/1.0 patent-evidence"},
                        timeout=config.timeout_s,
                    )
                    if response.status_code != 200:
                        continue
                    try:
                        payload = response.json()
                    except requests.JSONDecodeError:
                        continue
                    for cluster in dict(payload.get("results") or {}).get("cluster") or []:
                        for result in dict(cluster).get("result") or []:
                            patent = dict(dict(result).get("patent") or {})
                            publication = _publication(patent.get("publication_number"))
                            pdf = str(patent.get("pdf") or "").strip("/")
                            if not publication or not pdf:
                                continue
                            family_seed = {
                                "priority_date": patent.get("priority_date"),
                                "assignee": patent.get("assignee"),
                                "title": _plain_text(patent.get("title")),
                            }
                            rows.append(
                                {
                                    "publication_number": publication,
                                    "title": _plain_text(patent.get("title")),
                                    "snippet": _plain_text(patent.get("snippet")),
                                    "priority_date": str(patent.get("priority_date") or ""),
                                    "assignee": _plain_text(patent.get("assignee")),
                                    "pdf_url": (
                                        "https://patentimages.storage.googleapis.com/" + pdf
                                    ),
                                    "family_id": "search-family:" + _digest(family_seed)[:24],
                                    "query": query,
                                }
                            )
        return rows

    return search


def _direct_pdf_candidates(
    query: str,
    publications: Iterable[str],
) -> list[dict[str, Any]]:
    urls = re.findall(
        r"https://patentimages\.storage\.googleapis\.com/[^\s;]+?\.pdf",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    publication_rows = list(publications)
    rows: list[dict[str, Any]] = []
    for url in urls:
        publication = next(
            (
                value
                for value in publication_rows
                if value.casefold() in url.casefold()
            ),
            _publication(Path(url).stem),
        )
        if not publication:
            continue
        rows.append(
            {
                "publication_number": publication,
                "title": f"Patent {publication}",
                "snippet": "direct primary patent PDF locator",
                "pdf_url": url,
                "family_id": f"publication:{publication}",
                "query": query,
            }
        )
    return rows


def _resolve_google_patent_publication(
    publication: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    url = f"https://patents.google.com/patent/{publication}/en"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "AutoPlanner/1.0 patent-evidence"},
            timeout=timeout_s,
        )
    except requests.RequestException:
        return {}
    if response.status_code != 200 or len(response.content) > 20_000_000:
        return {}
    text = response.text
    pdf_match = re.search(
        r'<meta\s+name="citation_pdf_url"\s+content="([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    if not pdf_match:
        return {}
    title_match = re.search(
        r'<meta\s+(?:scheme="[^\"]+"\s+)?name="DC\.title"\s+content="([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    priority_match = re.search(
        r'<meta\s+scheme="dateSubmitted"\s+content="([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )
    title = _plain_text(title_match.group(1) if title_match else publication)
    priority = str(priority_match.group(1) if priority_match else "")
    family_seed = {"priority_date": priority, "title": title}
    return {
        "publication_number": publication,
        "title": title,
        "snippet": "resolved from primary patent publication page",
        "priority_date": priority,
        "pdf_url": html.unescape(pdf_match.group(1)),
        "family_id": "publication-family:" + _digest(family_seed)[:24],
        "query": publication,
    }


def _select_independent_candidates(
    values: Iterable[Mapping[str, Any]],
    *,
    queries: Iterable[str],
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = {
        token
        for query in queries
        for token in re.findall(r"[a-z0-9]{4,}", query.casefold())
        if token not in {"synthesis", "process", "preparation", "patent"}
    }
    candidates: list[tuple[int, dict[str, Any]]] = []
    seen_publications: set[str] = set()
    for raw in values:
        row = dict(raw)
        publication = _publication(row.get("publication_number"))
        if not publication or publication in seen_publications:
            continue
        seen_publications.add(publication)
        title = _plain_text(row.get("title")).casefold()
        snippet = _plain_text(row.get("snippet")).casefold()
        matched = sum(token in f"{title} {snippet}" for token in query_tokens)
        process = sum(
            term in title
            for term in ("synthesis", "synthetic", "process", "preparation", "intermediate")
        )
        noise = sum(
            term in title
            for term in ("medical use", "combination", "formulation", "treatment")
        )
        explicit = int(publication.casefold() in {q.casefold() for q in queries})
        score = 100 * explicit + 12 * matched + 8 * process - 8 * noise
        candidates.append((score, {**row, "publication_number": publication}))
    candidates.sort(key=lambda item: (-item[0], item[1]["publication_number"]))
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    for _score, row in candidates:
        family = str(row.get("family_id") or row["publication_number"])
        if family in families:
            continue
        families.add(family)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _fetch_bounded_bytes(url: str, timeout_s: float, limit: int) -> bytes:
    response = requests.get(
        url,
        headers={"User-Agent": "AutoPlanner/1.0 patent-evidence"},
        timeout=timeout_s,
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_content(64 * 1024):
            total += len(chunk)
            if total > limit:
                raise LiveEvidenceConnectorError("patent_pdf_size_limit_exceeded")
            chunks.append(bytes(chunk))
    finally:
        response.close()
    return b"".join(chunks)


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


def _patent_publications(value: str) -> list[str]:
    return sorted(
        {
            _publication(match)
            for match in re.findall(
                r"\b[A-Z]{2}\s*\d{6,}[A-Z]\d?\b",
                str(value or "").upper(),
            )
            if _publication(match)
        }
    )


def _publication(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return text if re.fullmatch(r"[A-Z]{2}\d{6,}[A-Z]\d?", text) else ""


def _plain_text(value: Any) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(str(value or ""))).split())


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
