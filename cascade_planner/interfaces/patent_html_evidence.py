"""Official structured-text-first exact-row attempt for one patent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from cascade_planner.harness.deterministic_literature_registry import (
    CandidateNameResolver,
    StructureResolver,
)
from cascade_planner.harness.source_html import (
    PatentHtmlConfig,
    materialize_primary_patent_html,
)
from cascade_planner.harness.source_patent_xml import (
    PATENT_XML_MATERIALIZATION_SCHEMA,
    PatentXmlConfig,
    materialize_primary_patent_xml,
)


BytesFetcher = Callable[[str, float, int], bytes]
RegistryCompiler = Callable[..., Mapping[str, Any]]


def attempt_primary_patent_html(
    candidate: Mapping[str, Any],
    *,
    edges: Iterable[Mapping[str, Any]],
    output_dir: Path,
    timeout_s: float,
    max_html_bytes: int,
    max_html_sections: int,
    max_html_paragraphs: int,
    fetch: BytesFetcher,
    fetch_xml: BytesFetcher | None = None,
    compile_registry: RegistryCompiler,
    resolve_structure: StructureResolver,
    resolve_names: CandidateNameResolver,
    target_terms: Iterable[str],
) -> dict[str, Any]:
    publication = str(candidate.get("publication_number") or "")
    source_ref = f"patent:{publication}"
    edge_rows = [dict(row) for row in edges if isinstance(row, Mapping)]
    terms = _source_terms(
        edge_rows,
        target_terms=target_terms,
        resolve_names=resolve_names,
    )
    materialization, source_attempts, reason = _materialize_primary_source(
        candidate,
        publication=publication,
        source_ref=source_ref,
        output_dir=output_dir,
        timeout_s=timeout_s,
        max_bytes=max_html_bytes,
        max_sections=max_html_sections,
        max_elements=max_html_paragraphs,
        fetch_html=fetch,
        fetch_xml=fetch_xml or fetch,
        target_terms=terms,
    )
    if materialization.get("status") != "completed":
        return _attempt(
            "not_available" if not source_attempts else "unresolved",
            reason=reason or "primary_patent_structured_text_unresolved",
            materialization=materialization,
            structured_source_attempts=source_attempts,
        )
    companion = dict(materialization.get("companion") or {})
    steps = _registry_steps(
        edge_rows,
        source_ref=source_ref,
        companion=companion,
        resolve_names=resolve_names,
    )
    try:
        artifact_kind = (
            "xml"
            if materialization.get("schema_version")
            == PATENT_XML_MATERIALIZATION_SCHEMA
            else "html"
        )
        audit = dict(
            compile_registry(
                steps,
                registry_path=(
                    output_dir
                    / artifact_kind
                    / "deterministic-step-registry.json"
                ),
                structure_resolver=resolve_structure,
                candidate_name_resolver=resolve_names,
                timeout_s=timeout_s,
            )
        )
    except Exception as exc:
        return _attempt(
            "failed",
            reason=(
                "primary_patent_structured_text_registry_failed:"
                f"{type(exc).__name__}:{str(exc)[:300]}"
            ),
            materialization=materialization,
            structured_source_attempts=source_attempts,
        )
    accepted = sorted(
        str(row.get("step_id") or "")
        for row in audit.get("records") or []
        if isinstance(row, Mapping) and row.get("accepted") is True
    )
    return _attempt(
        "completed" if len(accepted) == len(steps) else "partial" if accepted else "unresolved",
        materialization=materialization,
        registry_audit=audit,
        accepted_edge_ids=accepted,
        attempted_edge_count=len(steps),
        source_artifact_kind=artifact_kind,
        structured_source_attempts=source_attempts,
    )


def _materialize_primary_source(
    candidate: Mapping[str, Any],
    *,
    publication: str,
    source_ref: str,
    output_dir: Path,
    timeout_s: float,
    max_bytes: int,
    max_sections: int,
    max_elements: int,
    fetch_html: BytesFetcher,
    fetch_xml: BytesFetcher,
    target_terms: Iterable[str],
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    attempts: list[dict[str, str]] = []
    xml_url = str(candidate.get("xml_url") or "").strip()
    html_url = str(candidate.get("html_url") or "").strip()
    if not publication:
        return {}, attempts, "primary_patent_publication_missing"
    if xml_url:
        try:
            prefetched_xml = candidate.get("_primary_xml_bytes")
            if isinstance(prefetched_xml, bytes):
                if len(prefetched_xml) > max_bytes:
                    raise ValueError("primary_patent_xml_prefetch_size_limit_exceeded")
                xml_content = prefetched_xml
            else:
                xml_content = fetch_xml(xml_url, timeout_s, max_bytes)
            materialization = materialize_primary_patent_xml(
                content=xml_content,
                publication=publication,
                source_ref=source_ref,
                source_url=xml_url,
                output_dir=output_dir / "xml",
                target_terms=target_terms,
                config=PatentXmlConfig(
                    max_bytes=max_bytes,
                    max_sections=max_sections,
                    max_selected_elements=max_elements,
                ),
            )
        except Exception as exc:
            attempts.append(
                {
                    "source_artifact_kind": "xml",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
                }
            )
        else:
            attempts.append(
                {
                    "source_artifact_kind": "xml",
                    "status": str(materialization.get("status") or "failed"),
                    "reason": ";".join(materialization.get("reasons") or []),
                }
            )
            if materialization.get("status") == "completed":
                return materialization, attempts, ""
    if html_url:
        try:
            prefetched_html = candidate.get("_primary_html_bytes")
            if isinstance(prefetched_html, bytes):
                if len(prefetched_html) > max_bytes:
                    raise ValueError("primary_patent_html_prefetch_size_limit_exceeded")
                html_content = prefetched_html
            else:
                html_content = fetch_html(html_url, timeout_s, max_bytes)
            materialization = materialize_primary_patent_html(
                content=html_content,
                publication=publication,
                source_ref=source_ref,
                source_url=html_url,
                output_dir=output_dir / "html",
                target_terms=target_terms,
                config=PatentHtmlConfig(
                    max_bytes=max_bytes,
                    max_sections=max_sections,
                    max_selected_paragraphs=max_elements,
                ),
            )
        except Exception as exc:
            attempts.append(
                {
                    "source_artifact_kind": "html",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
                }
            )
        else:
            attempts.append(
                {
                    "source_artifact_kind": "html",
                    "status": str(materialization.get("status") or "failed"),
                    "reason": ";".join(materialization.get("reasons") or []),
                }
            )
            if materialization.get("status") == "completed":
                return materialization, attempts, ""
    reason = (
        "primary_patent_structured_text_url_missing"
        if not attempts
        else "primary_patent_structured_text_materialization_unresolved"
    )
    return {}, attempts, reason


def _registry_steps(
    edges: Iterable[Mapping[str, Any]],
    *,
    source_ref: str,
    companion: Mapping[str, Any],
    resolve_names: CandidateNameResolver,
) -> list[dict[str, Any]]:
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
                "source_text_companions": [dict(companion)],
            }
        )
    return steps


def _source_terms(
    edges: Iterable[Mapping[str, Any]],
    *,
    target_terms: Iterable[str],
    resolve_names: CandidateNameResolver,
) -> list[str]:
    values = [str(value) for value in target_terms if str(value).strip()]
    for edge in edges:
        for smiles in [
            edge.get("product_smiles"),
            *(edge.get("precursor_smiles") or []),
        ]:
            try:
                values.extend(resolve_names(str(smiles or "")))
            except (OSError, RuntimeError, ValueError):
                continue
    result = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())[:1_000]
        key = text.casefold()
        if len(text) >= 3 and key not in seen:
            seen.add(key)
            result.append(text)
        if len(result) >= 128:
            break
    return result


def _attempt(status: str, *, reason: str = "", **values: Any) -> dict[str, Any]:
    return {
        "schema_version": "primary_patent_structured_text_attempt.v1",
        "status": status,
        "reason": reason,
        "model_invocations": 0,
        "visual_invocations": 0,
        **values,
    }


__all__ = ["attempt_primary_patent_html"]
