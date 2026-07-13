"""Local PDF/image evidence extraction for literature structure curation."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LITERATURE_PDF_EXTRACTION_SCHEMA = "literature_pdf_structure_evidence.v1"
PAGE_FOCUS_ALGORITHM_VERSION = "literature_pdf_page_focus.v3"
_MAX_FOCUS_TERMS = 48
_MAX_FOCUS_PAGES = 16
_MAX_FOCUS_SCAN_PAGES = 512
_MAX_PAGE_TEXT_CHARS = 32_000
_MAX_FOCUS_TOTAL_TEXT_CHARS = 4_000_000
_MAX_PAGE_RELEVANCE_ROWS = 160
_MAX_FOCUS_HIT_AUDIT_ROWS = 48

_ROUTE_CONTEXT_TERMS = (
    ("synthesis of", 18),
    ("synthetic route", 18),
    ("preparation of", 15),
    ("general procedure", 14),
    ("experimental procedure", 14),
    ("was prepared", 10),
    ("scheme", 9),
    ("reaction conditions", 8),
    ("reaction mixture", 8),
    ("was added", 8),
    ("was stirred", 7),
    ("afforded", 7),
    ("provided", 6),
    ("was treated", 6),
    ("was concentrated", 6),
    ("yield", 3),
)
_STRONG_ROUTE_CONTEXT_TERMS = {
    "synthesis of",
    "synthetic route",
    "preparation of",
    "general procedure",
    "experimental procedure",
}
_NON_SYNTHETIC_EXPERIMENT_TERMS = (
    "assay",
    "enzyme kinetics",
    "plasma protein binding",
    "microsomal stability",
    "animal study",
    "crystallization of sars",
)

_ROUTE_HINT_STOP_WORDS = {
    "about",
    "and",
    "after",
    "already",
    "before",
    "chain",
    "compound",
    "contiguous",
    "covering",
    "current",
    "derive",
    "exact",
    "extract",
    "from",
    "focus",
    "for",
    "group",
    "groups",
    "infer",
    "intermediate",
    "labels",
    "literature",
    "missing",
    "only",
    "protecting",
    "reaction",
    "route",
    "scheme",
    "sequence",
    "source",
    "step",
    "steps",
    "structure",
    "synthesis",
    "target",
    "the",
    "when",
    "with",
}


def extract_literature_pdf_assets(
    *,
    pdf_path: str | Path | None = None,
    output_dir: str | Path,
    page_numbers: list[int] | None = None,
    render_zoom: float = 2.0,
    image_paths: list[str | Path] | None = None,
    scheme_crops: list[dict[str, Any]] | None = None,
    compound_labels: list[str] | None = None,
    target_name: str = "",
    target_aliases: list[str] | None = None,
    expected_labels: list[str] | None = None,
    route_sequence_hint: str = "",
) -> dict[str, Any]:
    """Render/source-index literature assets without claiming route evidence.

    The output is an evidence manifest for a later vision/curator step. It may
    contain short text snippets and local image paths, but it deliberately does
    not emit SMILES or route steps by itself.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages_dir = out / "pages"
    crops_dir = out / "crops"
    pages_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    source_pdf = Path(pdf_path).expanduser().resolve() if pdf_path else None
    reasons: list[str] = []
    warnings: list[str] = []
    rendered_pages: list[dict[str, Any]] = []
    fulltext_path = ""
    page_texts: list[dict[str, Any]] = []

    if source_pdf:
        if source_pdf.is_file():
            pdf_result = _render_pdf(
                source_pdf=source_pdf,
                pages_dir=pages_dir,
                page_numbers=page_numbers,
                zoom=max(0.5, float(render_zoom or 2.0)),
            )
            rendered_pages = pdf_result["rendered_pages"]
            page_texts = pdf_result["page_texts"]
            warnings.extend(pdf_result["warnings"])
            if page_texts:
                text = "\n\n".join(str(row.get("text") or "") for row in page_texts)
                text_file = out / "fulltext.txt"
                text_file.write_text(text, encoding="utf-8")
                fulltext_path = str(text_file)
        else:
            reasons.append("pdf_path_missing")

    indexed_images = _index_image_paths(image_paths or [])
    crop_rows = _extract_scheme_crops(
        scheme_crops or [],
        rendered_pages=rendered_pages,
        crops_dir=crops_dir,
        warnings=warnings,
    )
    labels = _dedupe_focus_values([*(compound_labels or []), *(expected_labels or [])])
    snippets = _compound_text_snippets(page_texts, labels)
    focus = _build_page_focus(
        page_texts,
        target_name=target_name,
        target_aliases=target_aliases or [],
        expected_labels=labels,
        route_sequence_hint=route_sequence_hint,
        explicit_page_numbers=page_numbers or [],
    )

    manifest = {
        "schema_version": LITERATURE_PDF_EXTRACTION_SCHEMA,
        "accepted": not reasons,
        "status": "completed" if not reasons else "incomplete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pdf_path": str(source_pdf) if source_pdf else "",
        "source_pdf_sha256": _sha256(source_pdf) if source_pdf and source_pdf.is_file() else "",
        "fulltext_path": fulltext_path,
        "rendered_pages": rendered_pages,
        "indexed_images": indexed_images,
        "scheme_crops": crop_rows,
        "compound_text_snippets": snippets,
        "focus_terms": focus["focus_terms"],
        "focus_page_numbers": focus["focus_page_numbers"],
        "page_relevance": focus["page_relevance"],
        "focus_hit_audit": focus["focus_hit_audit"],
        "focus_audit": focus["focus_audit"],
        "summary": {
            "rendered_page_count": len(rendered_pages),
            "indexed_image_count": len(indexed_images),
            "scheme_crop_count": len(crop_rows),
            "compound_text_snippet_count": len(snippets),
            "focus_term_count": len(focus["focus_terms"]),
            "focus_page_count": len(focus["focus_page_numbers"]),
            "focus_hit_page_count": int(focus["focus_audit"]["hit_page_count"]),
        },
        "source_policy": {
            "route_evidence_until_structured_extraction": False,
            "emits_smiles": False,
            "full_text_content_for_local_evidence_only": bool(fulltext_path),
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "warnings": warnings,
        "reasons": reasons,
    }
    path = out / "literature_pdf_structure_evidence.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def rebuild_literature_pdf_page_focus(
    pdf_path: str | Path,
    *,
    target_name: str,
    target_aliases: list[str] | None = None,
    expected_labels: list[str] | None = None,
    route_sequence_hint: str = "",
) -> dict[str, Any]:
    """Re-index text focus without re-rendering an already materialized PDF."""

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".pdf":
        return {
            "focus_terms": [],
            "focus_page_numbers": [],
            "page_relevance": [],
            "focus_hit_audit": [],
            "focus_audit": {
                "schema_version": "literature_pdf_page_focus_audit.v1",
                "algorithm_version": PAGE_FOCUS_ALGORITHM_VERSION,
                "selection_strategy": "source_pdf_missing_fail_soft",
                "relevance_available": False,
                "no_ocr_or_relevance_fabrication": True,
            },
        }
    try:
        import fitz  # type: ignore
    except ImportError:
        return {
            "focus_terms": [],
            "focus_page_numbers": [],
            "page_relevance": [],
            "focus_hit_audit": [],
            "focus_audit": {
                "schema_version": "literature_pdf_page_focus_audit.v1",
                "algorithm_version": PAGE_FOCUS_ALGORITHM_VERSION,
                "selection_strategy": "pymupdf_unavailable_fail_soft",
                "relevance_available": False,
                "no_ocr_or_relevance_fabrication": True,
            },
        }
    document = fitz.open(str(path))
    try:
        page_texts = [
            {
                "page_number": index + 1,
                "text": document[index].get_text("text") or "",
            }
            for index in range(min(len(document), _MAX_FOCUS_SCAN_PAGES))
        ]
    finally:
        document.close()
    return _build_page_focus(
        page_texts,
        target_name=target_name,
        target_aliases=list(target_aliases or []),
        expected_labels=list(expected_labels or []),
        route_sequence_hint=route_sequence_hint,
        explicit_page_numbers=[],
    )


def _build_page_focus(
    page_texts: list[dict[str, Any]],
    *,
    target_name: str,
    target_aliases: list[str],
    expected_labels: list[str],
    route_sequence_hint: str,
    explicit_page_numbers: list[int],
) -> dict[str, Any]:
    """Build a bounded text-only page ranking without inventing OCR hits."""

    term_rows = _page_focus_terms(
        target_name=target_name,
        target_aliases=target_aliases,
        expected_labels=expected_labels,
        route_sequence_hint=route_sequence_hint,
    )
    text_rows = _bounded_page_text_rows(page_texts)
    text_page_count = sum(1 for row in text_rows if row["text"].strip())
    explicit = _positive_unique_ints(explicit_page_numbers)
    relevance_rows: list[dict[str, Any]] = []
    term_hits: dict[str, list[int]] = {str(row["term"]): [] for row in term_rows}
    if text_page_count and term_rows:
        for row in text_rows:
            if not row["text"].strip():
                continue
            normalized_text = _focus_text_key(row["text"])
            matches: list[dict[str, Any]] = []
            score = 0
            route_context_matches: list[dict[str, Any]] = []
            route_context_score = 0
            for route_term, route_weight in _ROUTE_CONTEXT_TERMS:
                occurrences = _bounded_term_occurrences(
                    normalized_text,
                    _focus_text_key(route_term),
                )
                if occurrences <= 0:
                    continue
                weighted_score = int(route_weight) * occurrences
                route_context_score += weighted_score
                route_context_matches.append(
                    {
                        "term": route_term,
                        "occurrences": occurrences,
                        "weighted_score": weighted_score,
                    }
                )
            matched_route_terms = {
                str(match.get("term") or "")
                for match in route_context_matches
            }
            reference_marker_count = sum(
                _bounded_term_occurrences(normalized_text, marker)
                for marker in (
                    "doi",
                    "doi10",
                    "medline",
                    "http",
                    "references",
                )
            )
            non_synthetic_context = any(
                _bounded_term_occurrences(
                    normalized_text,
                    _focus_text_key(term),
                )
                > 0
                for term in _NON_SYNTHETIC_EXPERIMENT_TERMS
            )
            context_suppression_reason = ""
            if reference_marker_count >= 3:
                route_context_score = 0
                route_context_matches = []
                context_suppression_reason = "reference_like_page"
            elif (
                non_synthetic_context
                and not matched_route_terms.intersection(
                    _STRONG_ROUTE_CONTEXT_TERMS
                )
            ):
                route_context_score = 0
                route_context_matches = []
                context_suppression_reason = "non_synthetic_experiment_page"
            score += route_context_score
            for term_row in term_rows:
                term = str(term_row["term"])
                occurrences = _bounded_term_occurrences(normalized_text, _focus_text_key(term))
                if occurrences <= 0:
                    continue
                weighted_score = int(term_row["weight"]) * occurrences
                score += weighted_score
                matches.append(
                    {
                        "term": term,
                        "source": str(term_row["source"]),
                        "occurrences": occurrences,
                        "weighted_score": weighted_score,
                    }
                )
                if len(term_hits[term]) < _MAX_FOCUS_PAGES:
                    term_hits[term].append(int(row["page_number"]))
            relevance_rows.append(
                {
                    "page_number": int(row["page_number"]),
                    "score": score,
                    "matched_term_count": len(matches),
                    "matched_terms": matches[:12],
                    "route_context_score": route_context_score,
                    "route_context_matches": route_context_matches[:8],
                    "route_context_suppression_reason": (
                        context_suppression_reason
                    ),
                    "text_available": True,
                }
            )
    relevance_rows.sort(
        key=lambda row: (
            -int(int(row.get("route_context_score") or 0) > 0),
            -int(row.get("route_context_score") or 0),
            -int(row["score"]),
            int(row["page_number"]),
        )
    )
    relevance_rows = _coverage_ranked_page_relevance(relevance_rows)
    route_window_pages = _route_anchor_page_window(relevance_rows)
    if route_window_pages:
        rows_by_page = {
            int(row.get("page_number") or 0): row for row in relevance_rows
        }
        relevance_rows = [
            rows_by_page[page_number]
            for page_number in route_window_pages
            if page_number in rows_by_page
        ] + [
            row
            for row in relevance_rows
            if int(row.get("page_number") or 0) not in route_window_pages
        ]
    for rank, row in enumerate(relevance_rows, start=1):
        row["rank"] = rank

    all_relevant_pages = [
        int(row["page_number"])
        for row in relevance_rows
        if int(row["score"]) > 0
    ]
    relevant_pages = all_relevant_pages[:_MAX_FOCUS_PAGES]
    if explicit:
        focus_pages = explicit[:_MAX_FOCUS_PAGES]
        strategy = "explicit_page_numbers"
    elif text_page_count:
        focus_pages = relevant_pages
        strategy = "deterministic_text_relevance" if relevant_pages else "text_available_no_focus_hit"
    else:
        focus_pages = []
        strategy = "text_unavailable_fail_soft"

    hit_audit = [
        {
            "term": str(row["term"]),
            "source": str(row["source"]),
            "weight": int(row["weight"]),
            "matched_page_numbers": term_hits.get(str(row["term"]), []),
        }
        for row in term_rows
    ][:_MAX_FOCUS_HIT_AUDIT_ROWS]
    return {
        "focus_terms": [str(row["term"]) for row in term_rows],
        "focus_page_numbers": focus_pages,
        "page_relevance": relevance_rows[:_MAX_PAGE_RELEVANCE_ROWS],
        "focus_hit_audit": hit_audit,
        "focus_audit": {
            "schema_version": "literature_pdf_page_focus_audit.v1",
            "algorithm_version": PAGE_FOCUS_ALGORITHM_VERSION,
            "selection_strategy": strategy,
            "relevance_available": bool(text_page_count and term_rows),
            "text_page_count": text_page_count,
            "scanned_page_count": len(text_rows),
            "source_page_count": len(page_texts),
            "scan_page_limit": _MAX_FOCUS_SCAN_PAGES,
            "page_text_character_limit": _MAX_PAGE_TEXT_CHARS,
            "total_text_character_limit": _MAX_FOCUS_TOTAL_TEXT_CHARS,
            "scanned_text_character_count": sum(len(row["text"]) for row in text_rows),
            "focus_term_limit": _MAX_FOCUS_TERMS,
            "focus_page_limit": _MAX_FOCUS_PAGES,
            "page_relevance_row_limit": _MAX_PAGE_RELEVANCE_ROWS,
            "hit_page_count": len(all_relevant_pages),
            "explicit_page_numbers": explicit[:_MAX_FOCUS_PAGES],
            "scan_truncated": len(page_texts) > len(text_rows),
            "page_relevance_truncated": len(relevance_rows) > _MAX_PAGE_RELEVANCE_ROWS,
            "no_ocr_or_relevance_fabrication": True,
            "route_context_terms": [term for term, _weight in _ROUTE_CONTEXT_TERMS],
            "route_context_precedes_repeated_identity_mentions": True,
            "route_anchor_window_page_numbers": route_window_pages,
            "route_anchor_window_precedes_disconnected_route_pages": True,
        },
    }


def _page_focus_terms(
    *,
    target_name: str,
    target_aliases: list[str],
    expected_labels: list[str],
    route_sequence_hint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(value: Any, *, source: str, weight: int) -> None:
        term = _compact_ws(str(value or ""))[:96]
        key = _focus_text_key(term)
        if not key or key in seen or len(rows) >= _MAX_FOCUS_TERMS:
            return
        if not re.search(r"[a-z0-9]", key):
            return
        seen.add(key)
        rows.append({"term": term, "source": source, "weight": weight})

    add(target_name, source="target_name", weight=10)
    for alias in target_aliases:
        add(alias, source="target_alias", weight=9)
    for label in expected_labels:
        add(label, source="expected_label", weight=8)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}|[A-Za-z]?\d+[A-Za-z]?", route_sequence_hint or ""):
        if token.lower() in _ROUTE_HINT_STOP_WORDS or token.isdigit():
            continue
        add(token, source="route_sequence_hint", weight=3)
    return rows


def _bounded_page_text_rows(page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    remaining_chars = _MAX_FOCUS_TOTAL_TEXT_CHARS
    for raw in page_texts[:_MAX_FOCUS_SCAN_PAGES]:
        if not isinstance(raw, dict) or remaining_chars <= 0:
            break
        try:
            page_number = int(raw.get("page_number") or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if page_number <= 0:
            continue
        text = str(raw.get("text") or "")[: min(_MAX_PAGE_TEXT_CHARS, remaining_chars)]
        remaining_chars -= len(text)
        rows.append({"page_number": page_number, "text": text})
    return rows


def _focus_text_key(value: str) -> str:
    # PDF text extraction often inserts spaces at letter/number boundaries
    # (``C 43`` / ``PF 07321332``).  Collapse only those boundaries so exact
    # uppercase compound designators remain searchable without fuzzy OCR
    # claims.  Doing this after case-folding used to turn ordinary process
    # text such as ``yield 75%`` into ``yield75`` and hid continuation pages.
    raw = re.sub(
        r"\b([A-Z]{1,4})[\s-]+(?=\d)",
        r"\1",
        str(value or ""),
    )
    text = re.sub(r"[^a-z0-9]+", " ", raw.casefold()).strip()
    return text


def _coverage_ranked_page_relevance(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer pages that cover distinct host-supplied target/label terms."""

    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    while remaining:
        scored: list[tuple[int, int, int, int, int, dict[str, Any]]] = []
        for row in remaining:
            gain = 0
            for match in row.get("matched_terms") or []:
                if not isinstance(match, dict):
                    continue
                source = str(match.get("source") or "")
                term = _focus_text_key(str(match.get("term") or ""))
                if (
                    source not in {"target_name", "target_alias", "expected_label"}
                    or not term
                    or term in covered
                ):
                    continue
                occurrences = max(1, int(match.get("occurrences") or 1))
                gain += int(match.get("weighted_score") or 0) // occurrences
            scored.append(
                (
                    int(int(row.get("route_context_score") or 0) > 0),
                    gain,
                    int(row.get("route_context_score") or 0),
                    int(row.get("score") or 0),
                    -int(row.get("page_number") or 0),
                    row,
                )
            )
        _has_context, gain, _context_score, _score, _page, chosen = max(
            scored,
            key=lambda item: item[:5],
        )
        if gain <= 0:
            break
        selected.append(chosen)
        remaining.remove(chosen)
        covered.update(
            _focus_text_key(str(match.get("term") or ""))
            for match in chosen.get("matched_terms") or []
            if isinstance(match, dict)
            and str(match.get("source") or "")
            in {"target_name", "target_alias", "expected_label"}
        )
    remaining.sort(
        key=lambda row: (
            -int(int(row.get("route_context_score") or 0) > 0),
            -int(row.get("route_context_score") or 0),
            -int(row["score"]),
            int(row["page_number"]),
        )
    )
    return [*selected, *remaining]


def _route_anchor_page_window(rows: list[dict[str, Any]]) -> list[int]:
    """Prefer a contiguous procedure run around a source-bound route heading.

    Supporting information commonly puts the target heading at the bottom of
    one page and carries the exact preparations across the following pages.
    Ranking every page independently used to spend the visual budget on many
    unrelated ``Synthesis of Compound N`` headings.  This window is still
    text-only and deterministic: it starts only at a page that contains a
    host-supplied target/alias/label *and* synthesis context, then admits only
    neighbouring pages that themselves contain synthesis/procedure context.
    """

    by_page = {
        int(row.get("page_number") or 0): row
        for row in rows
        if int(row.get("page_number") or 0) > 0
    }
    anchors: list[tuple[int, int, int]] = []
    for page_number, row in by_page.items():
        context_score = int(row.get("route_context_score") or 0)
        if context_score <= 0:
            continue
        identity_score = sum(
            int(match.get("weighted_score") or 0)
            for match in row.get("matched_terms") or []
            if isinstance(match, dict)
            and str(match.get("source") or "")
            in {"target_name", "target_alias", "expected_label"}
        )
        if identity_score <= 0:
            continue
        strong_context = int(
            any(
                str(match.get("term") or "") in _STRONG_ROUTE_CONTEXT_TERMS
                for match in row.get("route_context_matches") or []
                if isinstance(match, dict)
            )
        )
        anchors.append(
            (
                strong_context,
                identity_score + context_score,
                -page_number,
            )
        )
    if not anchors:
        return []
    _strong, _score, negative_anchor = max(anchors)
    anchor = -negative_anchor
    window = [anchor]
    for delta in range(1, 5):
        page_number = anchor + delta
        row = by_page.get(page_number)
        if not row or int(row.get("route_context_score") or 0) <= 0:
            break
        window.append(page_number)
    for delta in range(1, 3):
        page_number = anchor - delta
        row = by_page.get(page_number)
        if not row or int(row.get("route_context_score") or 0) <= 0:
            break
        window.append(page_number)
    return window


def _bounded_term_occurrences(text: str, term: str) -> int:
    if not text or not term:
        return 0
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
    return min(3, sum(1 for _ in pattern.finditer(text)))


def _positive_unique_ints(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return out


def _dedupe_focus_values(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _compact_ws(str(value or ""))
        key = _focus_text_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _render_pdf(
    *,
    source_pdf: Path,
    pages_dir: Path,
    page_numbers: list[int] | None,
    zoom: float,
) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "rendered_pages": [],
            "page_texts": [],
            "warnings": [f"pymupdf_unavailable:{type(exc).__name__}"],
        }

    rendered: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    warnings: list[str] = []
    doc = fitz.open(str(source_pdf))
    try:
        requested = _page_indices(page_numbers, page_count=len(doc))
        matrix = fitz.Matrix(float(zoom), float(zoom))
        for page_index in requested:
            page = doc.load_page(page_index)
            image_path = pages_dir / f"page_{page_index + 1:03d}_z{_zoom_label(zoom)}.png"
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(image_path))
            rendered.append(
                {
                    "page_number": page_index + 1,
                    "image_path": str(image_path),
                    "width_px": int(pix.width),
                    "height_px": int(pix.height),
                    "render_zoom": float(zoom),
                    "sha256": _sha256(image_path),
                }
            )
            text = page.get_text("text") or ""
            texts.append({"page_number": page_index + 1, "text": text})
    finally:
        doc.close()
    return {"rendered_pages": rendered, "page_texts": texts, "warnings": warnings}


def _page_indices(page_numbers: list[int] | None, *, page_count: int) -> list[int]:
    if not page_numbers:
        return list(range(page_count))
    out: list[int] = []
    for value in page_numbers:
        index = int(value) - 1
        if 0 <= index < page_count and index not in out:
            out.append(index)
    return out


def _zoom_label(zoom: float) -> str:
    text = f"{float(zoom):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _index_image_paths(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(paths, start=1):
        path = Path(raw).expanduser().resolve()
        exists = path.is_file()
        rows.append(
            {
                "image_id": f"provided_image_{idx}",
                "image_path": str(path),
                "exists": exists,
                "sha256": _sha256(path) if exists else "",
            }
        )
    return rows


def _extract_scheme_crops(
    crops: list[dict[str, Any]],
    *,
    rendered_pages: list[dict[str, Any]],
    crops_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_by_number = {int(row.get("page_number") or 0): row for row in rendered_pages}
    for idx, crop in enumerate(crops, start=1):
        crop_id = _safe_id(str(crop.get("crop_id") or crop.get("scheme_id") or f"crop_{idx}"))
        source_raw = str(crop.get("source_image_path") or "").strip()
        source_path = Path(source_raw).expanduser() if source_raw else Path()
        if (not source_raw or not source_path.is_file()) and crop.get("page_number"):
            source_path = Path(str((page_by_number.get(int(crop.get("page_number"))) or {}).get("image_path") or ""))
        bbox = crop.get("bbox_px") or crop.get("bbox")
        if not source_path.is_file() or not _valid_bbox(bbox):
            rows.append(
                {
                    "crop_id": crop_id,
                    "source_image_path": str(source_path) if str(source_path) else "",
                    "image_path": "",
                    "page_number": int(crop.get("page_number") or 0),
                    "bbox_px": bbox if isinstance(bbox, list) else [],
                    "status": "not_created",
                    "reason": "source_image_or_bbox_missing",
                }
            )
            continue
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - environment dependent
            warnings.append(f"pillow_unavailable:{type(exc).__name__}")
            rows.append(
                {
                    "crop_id": crop_id,
                    "source_image_path": str(source_path),
                    "image_path": "",
                    "page_number": int(crop.get("page_number") or 0),
                    "bbox_px": bbox,
                    "status": "not_created",
                    "reason": "pillow_unavailable",
                }
            )
            continue
        image = Image.open(source_path)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0 = max(0, min(x0, image.width))
        x1 = max(0, min(x1, image.width))
        y0 = max(0, min(y0, image.height))
        y1 = max(0, min(y1, image.height))
        if x1 <= x0 or y1 <= y0:
            status = "not_created"
            crop_path = ""
            reason = "empty_bbox"
        else:
            crop_path_obj = crops_dir / f"{crop_id}.png"
            image.crop((x0, y0, x1, y1)).save(crop_path_obj)
            crop_path = str(crop_path_obj)
            status = "created"
            reason = ""
        rows.append(
            {
                "crop_id": crop_id,
                "scheme_id": str(crop.get("scheme_id") or ""),
                "source_image_path": str(source_path),
                "image_path": crop_path,
                "page_number": int(crop.get("page_number") or 0),
                "bbox_px": [x0, y0, x1, y1],
                "status": status,
                "reason": reason,
                "sha256": _sha256(Path(crop_path)) if crop_path else "",
                "evidence_refs": [str(item) for item in crop.get("evidence_refs") or [] if str(item).strip()],
            }
        )
    return rows


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(str(item).strip() for item in value)


def _compound_text_snippets(page_texts: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    if not labels:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for row in page_texts:
        page_number = int(row.get("page_number") or 0)
        text = str(row.get("text") or "")
        for label in labels:
            pattern = re.compile(rf"\b(?:compound\s*)?{re.escape(str(label))}\b", flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                start = max(0, match.start() - 180)
                end = min(len(text), match.end() + 260)
                snippet = _compact_ws(text[start:end])
                key = (page_number, str(label), snippet)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "compound_label": str(label),
                        "page_number": page_number,
                        "source_locator": f"page {page_number}",
                        "snippet": snippet,
                    }
                )
                if len(out) >= 200:
                    return out
    return out


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or "item"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
