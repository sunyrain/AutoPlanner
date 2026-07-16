"""Bounded query and source-candidate normalization for literature evidence."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from cascade_planner.interfaces.literature_relevance import (
    candidate_source_ref,
    doi,
    target_relevant_candidates,
)


def queries(request: Mapping[str, Any], *, limit: int) -> list[str]:
    values = request_queries(request)
    target = " ".join(str(request.get("target_name") or "").split())
    generic = target.lower() in {"blind target", "target"} or bool(
        re.fullmatch(r"target-[0-9a-f]{8}", target.lower())
    )
    if target and not generic:
        values = [f'"{target}" synthesis', f'"{target}" total synthesis', *values]
    return list(dict.fromkeys(value for value in values if value))[:limit]


def interleave_candidates(
    groups: Iterable[Iterable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = [[dict(value) for value in group] for group in groups]
    return [
        row[index]
        for index in range(max((len(row) for row in rows), default=0))
        for row in rows
        if index < len(row)
    ]


def request_queries(request: Mapping[str, Any]) -> list[str]:
    return [
        " ".join(str(row.get("query") or "").split())[:500]
        for row in request.get("source_tasks") or []
        if isinstance(row, Mapping)
        and any(
            kind in {"paper", "paper_si", "journal", "literature"}
            for kind in row.get("source_types") or []
        )
        and str(row.get("query") or "").strip()
    ]


def request_source_candidates(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direct_refs: list[tuple[str, str]] = []
    for task in request.get("source_tasks") or []:
        if not isinstance(task, Mapping) or not any(
            str(kind).casefold()
            in {
                "paper",
                "paper_si",
                "journal",
                "literature",
                "primary_literature",
            }
            for kind in task.get("source_types") or []
        ):
            continue
        direct_refs.extend((str(raw), "") for raw in task.get("source_refs") or [])
    for hint in request.get("source_hints") or []:
        if not isinstance(hint, Mapping) or str(hint.get("source_kind") or "").casefold() not in {
            "paper",
            "paper_si",
            "journal",
            "literature",
            "primary_literature",
        }:
            continue
        direct_refs.append(
            (str(hint.get("source_ref") or ""), str(hint.get("title") or ""))
        )
    for raw, title in direct_refs:
        value = str(raw).strip()
        lower = value.casefold()
        source_doi = ""
        if lower.startswith("doi:"):
            source_doi = value[4:].strip()
        elif "doi.org/" in lower:
            source_doi = value[lower.index("doi.org/") + len("doi.org/") :].strip()
        elif lower.startswith("10."):
            source_doi = value
        if source_doi:
            rows.append(
                {
                    "doi": source_doi,
                    "title": " ".join(title.split()) or source_doi,
                    "source_ref": f"doi:{source_doi}",
                }
            )
        elif lower.startswith(("https://", "http://")) and ".pdf" in lower:
            rows.append(
                {
                    "pdf_url": value,
                    "title": " ".join(title.split()) or value,
                    "source_ref": value,
                }
            )
    return rows


def seed_candidates(config: Any) -> list[dict[str, Any]]:
    rows = [{"doi": value, "title": value} for value in config.seed_dois if value]
    rows.extend(
        {
            "local_pdf": value,
            "source_ref": f"local_pdf:{Path(value).expanduser().resolve()}",
            "title": Path(value).stem,
        }
        for value in config.seed_pdfs
        if value
    )
    return rows


def pdf_page_count(path: Path) -> int:
    try:
        import fitz  # type: ignore

        with fitz.open(path) as document:
            return int(document.page_count)
    except (ImportError, OSError, RuntimeError, ValueError):
        try:
            from pypdf import PdfReader  # type: ignore

            return len(PdfReader(str(path)).pages)
        except (ImportError, OSError, RuntimeError, ValueError):
            return 0


def dedupe_candidates(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        identity = candidate_source_ref(row) or str(row.get("pdf_url") or "")
        if identity:
            result.setdefault(identity.lower(), row)
    return list(result.values())


__all__ = [
    "candidate_source_ref",
    "dedupe_candidates",
    "doi",
    "interleave_candidates",
    "pdf_page_count",
    "queries",
    "request_queries",
    "request_source_candidates",
    "seed_candidates",
    "target_relevant_candidates",
]
