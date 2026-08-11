"""Bounded primary-paper discovery with delegated source materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.harness.deterministic_literature_registry import (
    DEFAULT_OPSIN_BASE_URL,
    DEFAULT_PUBCHEM_BASE_URL,
    PARSER_AUTHORITY_ID,
    CandidateNameResolver,
    StructureResolver,
    build_deterministic_literature_resolvers,
)
from cascade_planner.harness.deterministic_resolver_cache import (
    DeterministicResolverCache,
)
from cascade_planner.interfaces.literature_access import (
    run_authorized_browser_fetch,
)
from cascade_planner.interfaces.literature_candidates import (
    interleave_candidates as _interleave_candidates,
    request_source_candidates as _request_source_candidates,
)
from cascade_planner.interfaces.literature_evidence_connector import (
    invoke_builtin_literature_evidence,
)
from cascade_planner.interfaces.literature_materialization import (
    materialize_candidate as _materialize_candidate,
)
from cascade_planner.interfaces.literature_search import fetch_bytes, primary_literature_search


BUILTIN_LITERATURE_PROVIDER_ID = "autoplanner.builtin_literature_evidence"
BUILTIN_LITERATURE_PROVIDER_VERSION = "1.8"
PaperSearch = Callable[[str, int], Iterable[Mapping[str, Any]]]
BytesFetcher = Callable[[str, float, int], bytes]
AuthorizedFetcher = Callable[..., Mapping[str, Any]]


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
    enable_repository_browser_fallback: bool = True
    max_fulltext_bytes: int = 12_000_000
    max_fulltext_sections: int = 32
    max_visual_pages: int = 6
    max_workers: int = 3
    timeout_s: float = 30.0
    render_zoom: float = 1.6
    authorized_proxy_output_dir: str | Path = ""
    queue_restricted_sources: bool = True
    auto_fetch_restricted_sources: bool = False
    auto_fetch_timeout_s: float = 180.0
    auto_fetch_max_items: int = 4
    auto_fetch_headless: bool = True

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
        if self.auto_fetch_timeout_s <= 0 or not 1 <= self.auto_fetch_max_items <= 8:
            raise ValueError("literature_evidence_autofetch_limit_invalid")


def build_builtin_literature_evidence_connector(
    config: BuiltinLiteratureEvidenceConfig,
    *,
    searcher: PaperSearch | None = None,
    fetcher: BytesFetcher | None = None,
    authorized_fetcher: AuthorizedFetcher | None = None,
    structure_resolver: StructureResolver | None = None,
    candidate_name_resolver: CandidateNameResolver | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build a metadata-first connector using XML before bounded PDF fallback."""
    cache_root = Path(config.cache_dir).expanduser().resolve()
    proxy_root = (
        Path(config.authorized_proxy_output_dir).expanduser().resolve()
        if str(config.authorized_proxy_output_dir).strip()
        else cache_root
    )
    search = searcher or primary_literature_search
    fetch = fetcher or fetch_bytes
    fetch_authorized = authorized_fetcher or run_authorized_browser_fetch
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

    def resolver_cache_receipt() -> Mapping[str, Any] | None:
        return resolver_cache.flush() if resolver_cache is not None else None

    def invoke(request: Mapping[str, Any]) -> Mapping[str, Any]:
        return invoke_builtin_literature_evidence(
            request,
            config=config,
            cache_root=cache_root,
            proxy_root=proxy_root,
            search=search,
            fetch=fetch,
            fetch_authorized=fetch_authorized,
            default_resolvers=default_resolvers,
            resolver_cache_receipt=resolver_cache_receipt,
            structure_resolver=structure_resolver,
            candidate_name_resolver=candidate_name_resolver,
            materialize_candidate=_materialize_candidate,
            provider_id=BUILTIN_LITERATURE_PROVIDER_ID,
            provider_version=BUILTIN_LITERATURE_PROVIDER_VERSION,
        )

    setattr(invoke, "autoplanner_prefetch_safe", True)
    return invoke


__all__ = [
    "BUILTIN_LITERATURE_PROVIDER_ID",
    "BUILTIN_LITERATURE_PROVIDER_VERSION",
    "BuiltinLiteratureEvidenceConfig",
    "_interleave_candidates",
    "_request_source_candidates",
    "build_builtin_literature_evidence_connector",
]
