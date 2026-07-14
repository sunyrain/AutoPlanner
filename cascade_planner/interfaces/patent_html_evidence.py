"""HTML-first exact-row attempt for one primary patent publication."""
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
    compile_registry: RegistryCompiler,
    resolve_structure: StructureResolver,
    resolve_names: CandidateNameResolver,
    target_terms: Iterable[str],
) -> dict[str, Any]:
    publication = str(candidate.get("publication_number") or "")
    source_ref = f"patent:{publication}"
    html_url = str(candidate.get("html_url") or "").strip()
    if not publication or not html_url:
        return _attempt("not_available", reason="primary_patent_html_url_missing")
    prefetched = candidate.get("_primary_html_bytes")
    if isinstance(prefetched, bytes):
        if len(prefetched) > max_html_bytes:
            return _attempt(
                "failed",
                reason="primary_patent_html_prefetch_size_limit_exceeded",
            )
        content = prefetched
    else:
        try:
            content = fetch(html_url, timeout_s, max_html_bytes)
        except Exception as exc:  # external source failure must fall back to PDF
            return _attempt(
                "failed",
                reason=(
                    "primary_patent_html_fetch_failed:"
                    f"{type(exc).__name__}:{str(exc)[:300]}"
                ),
            )
    try:
        materialization = materialize_primary_patent_html(
            content=content,
            publication=publication,
            source_ref=source_ref,
            source_url=html_url,
            output_dir=output_dir / "html",
            target_terms=_source_terms(
                edges,
                target_terms=target_terms,
                resolve_names=resolve_names,
            ),
            config=PatentHtmlConfig(
                max_bytes=max_html_bytes,
                max_sections=max_html_sections,
                max_selected_paragraphs=max_html_paragraphs,
            ),
        )
    except Exception as exc:
        return _attempt(
            "failed",
            reason=(
                "primary_patent_html_materialization_failed:"
                f"{type(exc).__name__}:{str(exc)[:300]}"
            ),
        )
    if materialization.get("status") != "completed":
        return _attempt(
            "unresolved",
            reason="primary_patent_html_materialization_unresolved",
            materialization=materialization,
        )
    companion = dict(materialization.get("companion") or {})
    steps = _registry_steps(
        edges,
        source_ref=source_ref,
        companion=companion,
        resolve_names=resolve_names,
    )
    try:
        audit = dict(
            compile_registry(
                steps,
                registry_path=(
                    output_dir / "html" / "deterministic-step-registry.json"
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
                "primary_patent_html_registry_failed:"
                f"{type(exc).__name__}:{str(exc)[:300]}"
            ),
            materialization=materialization,
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
    )


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
        "schema_version": "primary_patent_html_attempt.v1",
        "status": status,
        "reason": reason,
        "model_invocations": 0,
        "visual_invocations": 0,
        **values,
    }


__all__ = ["attempt_primary_patent_html"]
