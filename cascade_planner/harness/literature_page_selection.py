"""Shared, deterministic PDF page and image selection for literature extraction.

The legacy blackboard and the V4 campaign both consume this module.  Selection
uses text-derived focus first, preserves route-context pages, and spends any
remaining budget on document-wide coverage.  It never assigns evidence status.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def select_pdf_page_numbers(
    pdf_evidence: Mapping[str, Any],
    *,
    page_count: int,
    max_pages: int,
    explicit_page_numbers: Sequence[int] = (),
) -> list[int]:
    """Choose bounded PDF pages without falling back to a first-N prefix."""

    page_count = max(0, int(page_count))
    max_pages = max(0, int(max_pages))
    if not page_count or not max_pages:
        return []

    explicit = _valid_unique_pages(explicit_page_numbers, page_count=page_count)
    if explicit:
        return explicit[:max_pages]

    focus = _focus_page_numbers(pdf_evidence, page_count=page_count)
    route_context = []
    for raw in pdf_evidence.get("page_relevance") or []:
        if not isinstance(raw, Mapping):
            continue
        try:
            page = int(raw.get("page_number") or 0)
            route_score = int(raw.get("route_context_score") or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if 1 <= page <= page_count and route_score > 0 and page not in route_context:
            route_context.append(page)

    selected: list[int] = []

    def add(page: int) -> None:
        if 1 <= page <= page_count and page not in selected and len(selected) < max_pages:
            selected.append(page)

    # Keep the strongest textual focus, but reserve coverage slots on long PDFs.
    reserve = min(2, max(0, max_pages - 1)) if page_count > max_pages else 0
    for page in focus[: max_pages - reserve]:
        add(page)
    for page in route_context:
        add(page)

    remaining_focus = [page for page in focus if page not in selected]
    for page in _evenly_spread(remaining_focus, max_pages - len(selected)):
        add(page)

    remaining = [page for page in range(1, page_count + 1) if page not in selected]
    for page in _evenly_spread(remaining, max_pages - len(selected)):
        add(page)
    return selected


def select_pdf_visual_paths(
    pdf_evidence: Mapping[str, Any],
    *,
    explicit_page_numbers: Sequence[int] = (),
    max_images: int,
) -> list[str]:
    """Select rendered pages/scheme crops using the shared focus policy."""

    rendered = _asset_rows(pdf_evidence.get("rendered_pages"))
    crops = _asset_rows(pdf_evidence.get("scheme_crops"))
    pages = sorted({_asset_page(row) for row in [*rendered, *crops] if _asset_page(row)})
    page_count = max(pages, default=0)
    selected_pages = select_pdf_page_numbers(
        pdf_evidence,
        page_count=page_count,
        max_pages=max(0, int(max_images)),
        explicit_page_numbers=explicit_page_numbers,
    )
    by_page: dict[int, list[dict[str, Any]]] = {}
    for row in [*crops, *rendered]:
        by_page.setdefault(_asset_page(row), []).append(row)

    selected: list[str] = []

    def append(row: Mapping[str, Any]) -> None:
        path = str(row.get("image_path") or "").strip()
        if path and path not in selected and (max_images <= 0 or len(selected) < max_images):
            selected.append(path)

    for page in selected_pages:
        rows = by_page.get(page) or []
        if rows:
            append(rows[0])
    for page in selected_pages:
        for row in by_page.get(page) or []:
            append(row)
    for row in by_page.get(0) or []:
        append(row)
    return selected


def _focus_page_numbers(value: Mapping[str, Any], *, page_count: int) -> list[int]:
    values: list[Any] = list(value.get("focus_page_numbers") or [])
    if not values:
        for raw in value.get("page_relevance") or []:
            if not isinstance(raw, Mapping):
                continue
            try:
                if int(raw.get("score") or 0) > 0:
                    values.append(raw.get("page_number"))
            except (TypeError, ValueError, OverflowError):
                continue
    values.extend(
        raw.get("page_number")
        for raw in value.get("compound_text_snippets") or []
        if isinstance(raw, Mapping)
    )
    return _valid_unique_pages(values, page_count=page_count)[:32]


def _valid_unique_pages(values: Sequence[Any], *, page_count: int) -> list[int]:
    result: list[int] = []
    for value in values:
        try:
            page = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 1 <= page <= page_count and page not in result:
            result.append(page)
    return result


def _asset_rows(values: Any) -> list[dict[str, Any]]:
    rows = [
        dict(value)
        for value in values or []
        if isinstance(value, Mapping) and str(value.get("image_path") or "").strip()
    ]
    rows.sort(key=lambda row: (_asset_page(row), str(row.get("image_path") or "")))
    return rows


def _asset_page(row: Mapping[str, Any]) -> int:
    try:
        return max(0, int(row.get("page_number") or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _evenly_spread(values: Sequence[Any], slots: int) -> list[Any]:
    slots = max(0, int(slots))
    if slots <= 0 or not values:
        return []
    if slots >= len(values):
        return list(values)
    indices = [
        min(len(values) - 1, ((2 * index + 1) * len(values)) // (2 * slots))
        for index in range(slots)
    ]
    return [values[index] for index in indices]


__all__ = ["select_pdf_page_numbers", "select_pdf_visual_paths"]
