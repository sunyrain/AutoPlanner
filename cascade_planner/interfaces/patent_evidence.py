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
BUILTIN_PATENT_PROVIDER_VERSION = "1.2.0"
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
        queries = evidence_queries(request, limit=config.max_search_queries)
        candidates = select_independent_candidates(
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
                    fetch_html=fetch_html,
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
                "html_sha256": str(row.get("html_sha256") or ""),
                "html_audit": dict(row.get("html_audit") or {}),
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
            if str(row.get("html_sha256") or row.get("pdf_sha256") or "")
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
                "primary_html_is_attempted_before_pdf": True,
                "html_or_pdf_source_bytes_are_frozen": True,
                "pdf_ocr_and_vision_are_unresolved_only_fallbacks": True,
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
    fetch_html: BytesFetcher,
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
            compile_registry=compile_registry,
            resolve_structure=resolve_structure,
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
    page_count = 0
    manifest: dict[str, Any] = {}
    ocr_audit: dict[str, Any] = {}
    pdf_registry: dict[str, Any] = {}
    if remaining_edges:
        pdf_url = str(candidate.get("pdf_url") or "").strip()
        if not pdf_url:
            if not rows:
                raise ValueError("patent_pdf_fallback_url_missing")
        else:
            content = fetch(pdf_url, config.timeout_s, config.max_pdf_bytes)
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
            manifest_path = (
                patent_dir / "materialized" / "literature_pdf_structure_evidence.json"
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
            steps = _pdf_registry_steps(
                remaining_edges,
                source_ref=source_ref,
                evidence=evidence,
                companions=companions,
                resolve_names=resolve_names,
            )
            pdf_registry = dict(
                compile_registry(
                    steps,
                    registry_path=patent_dir / "pdf-deterministic-step-registry.json",
                    structure_resolver=resolve_structure,
                    candidate_name_resolver=resolve_names,
                    timeout_s=config.timeout_s,
                )
            )
            pdf_rows, pdf_edges = _exact_rows_from_registry(
                pdf_registry,
                publication=publication,
            )
            rows.extend(pdf_rows)
            accepted_edges.extend(pdf_edges)
            accepted_ids.update(pdf_edges)

    rows = list({str(row["step_id"]): row for row in rows}.values())
    accepted_edges = sorted(accepted_ids)
    source = _structured_patent_source(
        candidate,
        source_ref=source_ref,
        rows=rows,
        used_html=html_row_count > 0,
        used_pdf=len(rows) > html_row_count,
    )
    procedure_inventory = _procedure_inventory(html_registry, pdf_registry)
    html_materialization = dict(html_attempt.get("materialization") or {})
    return source, {
        "publication_number": publication,
        "family_id": str(candidate.get("family_id") or ""),
        "title": str(candidate.get("title") or publication),
        "html_sha256": str(html_materialization.get("artifact_sha256") or ""),
        "html_audit": _bounded_html_audit(html_attempt),
        "pdf_sha256": pdf_sha256,
        "page_count": page_count,
        "accepted": bool(rows),
        "accepted_edge_ids": accepted_edges,
        "rejected_edge_count": max(0, len(edges) - len(accepted_edges)),
        "registry_audit_sha256": _digest(
            {
                "html": str(html_registry.get("content_sha256") or ""),
                "pdf": str(pdf_registry.get("content_sha256") or ""),
            }
        ),
        "procedure_inventory": procedure_inventory,
        "ocr_audit": _bounded_ocr_audit(ocr_audit),
        "visual_candidate_pages": _visual_candidate_pages(
            manifest,
            procedure_inventory=procedure_inventory,
            ocr_audit=ocr_audit,
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
        evidence_refs: list[str] = []
        if artifact_kind == "html":
            start = str(source_location.get("start_element_id") or "")
            end = str(source_location.get("end_element_id") or "")
            location_ref = f"{publication}:html:{start}-{end}"
            artifact_sha256 = str(
                binding.get("source_artifact_sha256") or ""
            )
            text_sha256 = str(source_location.get("text_sha256") or "")
            if artifact_sha256:
                evidence_refs.append(f"html_sha256:{artifact_sha256}")
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
        rows.append(
            {
                "step_id": edge_id,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "relation_type": "exact",
                "location_ref": location_ref,
                "evidence_refs": evidence_refs,
            }
        )
        edge_ids.append(edge_id)
    return rows, edge_ids


def _structured_patent_source(
    candidate: Mapping[str, Any],
    *,
    source_ref: str,
    rows: list[dict[str, Any]],
    used_html: bool,
    used_pdf: bool,
) -> dict[str, Any]:
    if not rows:
        return {}
    if used_html and used_pdf:
        provenance = "builtin_patent_html_first_with_pdf_fallback"
    elif used_html:
        provenance = "builtin_deterministic_primary_patent_html"
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
