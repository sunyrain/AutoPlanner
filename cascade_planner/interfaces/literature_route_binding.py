"""Promote hash-bound paper procedures into replayable route evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from cascade_planner.harness.deterministic_literature_registry import (
    PARSER_AUTHORITY_ID,
    CandidateNameResolver,
    StructureResolver,
    compile_deterministic_literature_step_registry,
    source_parenthetical_name_aliases,
)
from cascade_planner.harness.source_route_extraction import (
    compile_deterministic_source_route_observation,
)
from cascade_planner.harness.source_text_companion import (
    PRIMARY_HTML_AUTHORITY_MODE,
    SOURCE_TEXT_COMPANION_SPEC_SCHEMA,
    STRUCTURED_FULLTEXT_HTML_FORMAT,
)


def bind_materialized_literature_source(
    source: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    output_dir: Path,
    structure_resolver: StructureResolver,
    candidate_name_resolver: CandidateNameResolver,
    timeout_s: float,
    provider_version: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile a paper route DAG and exact rows without model authority."""

    row = dict(source)
    procedures = _procedures(row.get("procedure_inventory") or [])
    path = _fulltext_path(row)
    artifact_sha256 = str(row.get("source_fulltext_sha256") or "").lower()
    edges = [
        dict(value)
        for value in request.get("edges") or []
        if isinstance(value, Mapping)
        and value.get("current_host_reaction_validated") is True
    ]
    if (
        not edges
        or not procedures
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != artifact_sha256
    ):
        return row, {}, _audit("not_needed", reason="validated_edge_or_fulltext_missing")

    document = {
        "source_ref": str(row.get("source_ref") or ""),
        "source_artifact_sha256": artifact_sha256,
        "source_location_kind": "structured_fulltext_section",
        "procedures": procedures,
        "source_name_aliases": source_parenthetical_name_aliases(
            {"text": value.get("procedure") or ""} for value in procedures
        ),
    }
    observation = compile_deterministic_source_route_observation(
        document,
        structure_resolver=structure_resolver,
        anchor_smiles=[
            str(edge.get("product_smiles") or "")
            for edge in edges
            if str(edge.get("product_smiles") or "")
        ],
    )
    proposals = [
        dict(value)
        for value in observation.get("proposals") or []
        if isinstance(value, Mapping)
    ]
    row["source_route_observation"] = observation
    row["source_route_proposal_count"] = len(proposals)
    if not proposals:
        return row, {}, _audit(
            "unresolved",
            proposal_count=0,
            reason="target_connected_source_route_missing",
        )

    companion = _companion(row, path=path, procedures=procedures)
    if not companion:
        return row, {}, _audit(
            "unresolved",
            proposal_count=len(proposals),
            reason="replayable_fulltext_companion_missing",
        )
    steps = [
        {
            "step_id": str(value.get("proposal_id") or value.get("step_id") or ""),
            "product_name": str(value.get("product_name") or ""),
            "product_smiles": str(value.get("product_smiles") or ""),
            "reactant_smiles": list(
                value.get("reactant_smiles") or value.get("precursor_smiles") or []
            ),
            "reactant_names": list(value.get("reactant_names") or []),
            "source_ref": str(row.get("source_ref") or ""),
            "source_text_companions": [companion],
        }
        for value in proposals
    ]
    registry = dict(
        compile_deterministic_literature_step_registry(
            steps,
            registry_path=output_dir / "paper-deterministic-step-registry.json",
            structure_resolver=structure_resolver,
            candidate_name_resolver=candidate_name_resolver,
            timeout_s=timeout_s,
        )
    )
    exact_rows = _exact_rows_from_registry(
        registry,
        source_ref=str(row.get("source_ref") or ""),
    )
    exact_ids = [str(value.get("step_id") or "") for value in exact_rows]
    row.update(
        {
            "exact_edge_ids": exact_ids,
            "exact_row_count": len(exact_rows),
            "source_route_exact_row_count": len(exact_rows),
            "source_route_exact_step_ids": exact_ids,
            "unresolved_edge_count": max(0, len(proposals) - len(exact_rows)),
        }
    )
    structured = (
        {
            "binding": {
                "source_kind": "paper_si",
                "source_ref": str(row.get("source_ref") or ""),
                "doi": str(row.get("doi") or ""),
                "pmid": str(row.get("pmid") or ""),
                "pmc": str(row.get("pmcid") or ""),
                "title": str(row.get("title") or ""),
                "artifact_sha256": artifact_sha256,
                "provenance": "builtin_deterministic_primary_paper_fulltext",
                "discovered_by": "autoplanner.builtin_literature_evidence",
                "acquisition_status": "materialized",
            },
            "extraction": {
                "schema_version": "structured_exact_row_extraction.v1",
                "extractor": {
                    "producer_kind": "deterministic_structure_parser",
                    "producer_id": PARSER_AUTHORITY_ID,
                    "version": provider_version,
                },
                "rows": exact_rows,
            },
        }
        if exact_rows
        else {}
    )
    return row, structured, _audit(
        "completed" if exact_rows else "unresolved",
        proposal_count=len(proposals),
        exact_row_count=len(exact_rows),
        registry_record_count=len(registry.get("records") or []),
    )


def _procedures(values: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        page_number = int(value.get("page_number") or 0)
        excerpt = " ".join(str(value.get("procedure_excerpt") or "").split())
        name = " ".join(str(value.get("name") or value.get("label") or "").split())
        name_key = name.casefold().strip(" .:")
        if (
            page_number <= 0
            or page_number in seen_pages
            or len(excerpt) < 40
            or name_key.startswith(("abstract", "fig", "discussion", "reference"))
            or not _procedure_like(excerpt)
        ):
            continue
        seen_pages.add(page_number)
        rows.append(
            {
                "label": str(value.get("label") or f"section-{page_number}"),
                "name": name,
                "narrative_context": f"{name}. {excerpt}",
                "procedure": excerpt,
                "page_number": page_number,
            }
        )
        if len(rows) >= 64:
            break
    return rows


def _procedure_like(value: str) -> bool:
    text = str(value or "").casefold()
    return any(
        signal in text
        for signal in (
            " was added",
            " were added",
            "reaction mixture",
            "final concentration",
            " was stirred",
            " was incubated",
            " were incubated",
            " was extracted",
            " was purified",
            "yield ",
            " yield",
        )
    )


def _fulltext_path(source: Mapping[str, Any]) -> Path:
    value = str(
        source.get("fulltext_html_path")
        or source.get("fulltext_xml_path")
        or ""
    )
    return Path(value).expanduser().resolve() if value else Path("__missing__")


def _companion(
    source: Mapping[str, Any],
    *,
    path: Path,
    procedures: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = dict(source.get("acquisition_receipt") or {})
    source_url = str(receipt.get("html_url") or "")
    pmcid = str(source.get("pmcid") or receipt.get("pmcid") or "")
    if not source_url or not pmcid or path.suffix.casefold() != ".html":
        return {}
    sections = []
    for row in procedures:
        text = str(row.get("procedure") or "")
        sections.append(
            {
                "page_number": int(row.get("page_number") or 0),
                "label": str(row.get("label") or ""),
                "name": str(row.get("name") or ""),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "schema_version": SOURCE_TEXT_COMPANION_SPEC_SCHEMA,
        "artifact_path": str(path),
        "artifact_sha256": str(source.get("source_fulltext_sha256") or ""),
        "document_identity": pmcid,
        "source_url": source_url,
        "format": STRUCTURED_FULLTEXT_HTML_FORMAT,
        "authority_mode": PRIMARY_HTML_AUTHORITY_MODE,
        "sections": sections,
    }


def _exact_rows_from_registry(
    audit: Mapping[str, Any],
    *,
    source_ref: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in audit.get("records") or []:
        if not isinstance(value, Mapping) or value.get("accepted") is not True:
            continue
        binding = dict(value.get("binding") or {})
        projection = dict(binding.get("synthesis_projection") or {})
        product = str(projection.get("product_smiles") or "")
        reactants = [
            str(item) for item in projection.get("reactant_smiles") or [] if str(item)
        ]
        step_id = str(value.get("step_id") or "")
        location = dict(binding.get("source_location") or {})
        parser_audit = dict(binding.get("parser_audit") or {})
        artifact_sha256 = str(binding.get("source_artifact_sha256") or "")
        text_sha256 = str(location.get("text_sha256") or "")
        procedure_text_sha256 = str(
            parser_audit.get("procedure_text_sha256") or ""
        )
        if not step_id or not product or not reactants:
            continue
        rows.append(
            {
                "step_id": step_id,
                "product_smiles": product,
                "reactant_smiles": reactants,
                "relation_type": "exact",
                "location_ref": (
                    f"{source_ref}:html:{location.get('start_element_id') or ''}"
                ),
                "evidence_refs": [
                    item
                    for item in (
                        f"html_sha256:{artifact_sha256}" if artifact_sha256 else "",
                        f"text_sha256:{text_sha256}" if text_sha256 else "",
                        (
                            f"procedure-text-sha256:{procedure_text_sha256}"
                            if procedure_text_sha256
                            else ""
                        ),
                    )
                    if item
                ],
                "conditions": dict(binding.get("source_conditions") or {}),
            }
        )
    return rows


def _audit(status: str, **values: Any) -> dict[str, Any]:
    return {"status": status, "model_invocations": 0, **values}


__all__ = ["bind_materialized_literature_source"]
