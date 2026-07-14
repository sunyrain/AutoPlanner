"""Replay hash-bound source text companions for image-only literature PDFs."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from cascade_planner.harness.source_ocr import HASH_BOUND_OCR_FORMAT


SOURCE_TEXT_COMPANION_SPEC_SCHEMA = "trusted_source_text_companion.v1"
SOURCE_TEXT_COMPANION_BINDING_SCHEMA = "source_text_companion_binding.v1"
GOOGLE_PATENTS_HTML_FORMAT = "google_patents_html.v1"
PRIMARY_HTML_AUTHORITY_MODE = "primary_html"
_MAX_COMPANION_BYTES = 100_000_000


def materialize_source_text_companion_pages(
    raw: Mapping[str, Any],
    *,
    source_ref: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...]]:
    """Reconstruct selected source paragraphs and return their proof binding."""

    spec = dict(raw) if isinstance(raw, Mapping) else {}
    reasons: list[str] = []
    expected_source_ref = str(source_ref or "").strip().lower()
    artifact_path = Path(
        str(spec.get("artifact_path") or spec.get("path") or "")
    ).expanduser()
    expected_sha256 = str(spec.get("artifact_sha256") or "").strip().lower()
    document_identity = str(spec.get("document_identity") or "").strip()
    source_url = str(spec.get("source_url") or "").strip()
    format_id = str(spec.get("format") or "").strip()
    if (
        spec.get("schema_version") == SOURCE_TEXT_COMPANION_SPEC_SCHEMA
        and expected_source_ref
        and format_id == HASH_BOUND_OCR_FORMAT
    ):
        return _materialize_hash_bound_ocr_pages(
            spec,
            expected_source_ref=expected_source_ref,
        )
    if spec.get("schema_version") != SOURCE_TEXT_COMPANION_SPEC_SCHEMA:
        reasons.append("source_text_companion_schema_invalid")
    if not expected_source_ref:
        reasons.append("source_text_companion_source_ref_missing")
    if not artifact_path.is_file():
        reasons.append("source_text_companion_artifact_missing")
    if not _is_sha256(expected_sha256):
        reasons.append("source_text_companion_artifact_sha256_invalid")
    if not document_identity:
        reasons.append("source_text_companion_document_identity_missing")
    if not source_url.startswith("https://"):
        reasons.append("source_text_companion_source_url_not_https")
    if format_id != GOOGLE_PATENTS_HTML_FORMAT:
        reasons.append("source_text_companion_format_unsupported")
    if spec.get("authority_mode") == PRIMARY_HTML_AUTHORITY_MODE and not (
        official_google_patent_source(
            document_identity=document_identity,
            source_url=source_url,
            source_ref=expected_source_ref,
        )
    ):
        reasons.append("source_text_companion_primary_html_origin_invalid")
    if reasons:
        return [], {}, tuple(sorted(set(reasons)))

    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError:
        return [], {}, ("source_text_companion_artifact_unreadable",)
    if len(artifact_bytes) > _MAX_COMPANION_BYTES:
        return [], {}, ("source_text_companion_artifact_too_large",)
    if hashlib.sha256(artifact_bytes).hexdigest() != expected_sha256:
        return [], {}, ("source_text_companion_artifact_digest_mismatch",)
    try:
        artifact_text = artifact_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return [], {}, ("source_text_companion_artifact_not_utf8",)
    if _identity_key(document_identity) not in _identity_key(artifact_text):
        return [], {}, ("source_text_companion_document_identity_not_found",)

    try:
        paragraphs = extract_google_patent_paragraphs(artifact_text)
    except (RuntimeError, ValueError):
        return [], {}, ("source_text_companion_html_parse_failed",)
    section_bindings: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    for index, raw_section in enumerate(spec.get("sections") or [], start=1):
        if not isinstance(raw_section, Mapping):
            reasons.append(f"source_text_companion_section_{index}_not_object")
            continue
        section = dict(raw_section)
        page_number = int(section.get("page_number") or 0)
        start_id = str(section.get("start_element_id") or "").strip().lower()
        end_id = str(section.get("end_element_id") or "").strip().lower()
        start_number = _paragraph_number(start_id)
        end_number = _paragraph_number(end_id)
        if (
            page_number <= 0
            or start_number is None
            or end_number is None
            or end_number < start_number
        ):
            reasons.append(f"source_text_companion_section_{index}_range_invalid")
            continue
        selected = [
            paragraphs[f"p{number:04d}"]
            for number in range(start_number, end_number + 1)
            if paragraphs.get(f"p{number:04d}")
        ]
        if (
            not selected
            or f"p{start_number:04d}" not in paragraphs
            or f"p{end_number:04d}" not in paragraphs
        ):
            reasons.append(f"source_text_companion_section_{index}_range_missing")
            continue
        text = "\n\n".join(selected).strip()
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        section_bindings.append(
            {
                "page_number": page_number,
                "start_element_id": f"p{start_number:04d}",
                "end_element_id": f"p{end_number:04d}",
                "text_sha256": text_sha256,
            }
        )
        page_rows.append({"page_number": page_number, "text": text})
    if reasons:
        return [], {}, tuple(sorted(set(reasons)))
    if not section_bindings:
        return [], {}, ("source_text_companion_sections_missing",)

    binding: dict[str, Any] = {
        "schema_version": SOURCE_TEXT_COMPANION_BINDING_SCHEMA,
        "source_ref": expected_source_ref,
        "document_identity": document_identity,
        "source_url": source_url,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": expected_sha256,
        "format": format_id,
        "sections": section_bindings,
    }
    if spec.get("authority_mode") == PRIMARY_HTML_AUTHORITY_MODE:
        binding["authority_mode"] = PRIMARY_HTML_AUTHORITY_MODE
    binding["content_sha256"] = _digest(binding)
    for page in page_rows:
        page["source_text_companion_binding"] = dict(binding)
    return page_rows, binding, ()


def validate_source_text_companion_binding(
    raw: Mapping[str, Any],
    *,
    expected_source_ref: str,
) -> bool:
    """Replay a persisted binding against current-host artifact bytes."""

    binding = dict(raw) if isinstance(raw, Mapping) else {}
    digest_payload = dict(binding)
    recorded_digest = str(digest_payload.pop("content_sha256", ""))
    if (
        binding.get("schema_version") != SOURCE_TEXT_COMPANION_BINDING_SCHEMA
        or recorded_digest != _digest(digest_payload)
        or str(binding.get("source_ref") or "").strip().lower()
        != str(expected_source_ref or "").strip().lower()
    ):
        return False
    if binding.get("format") == HASH_BOUND_OCR_FORMAT:
        spec = {
            "schema_version": SOURCE_TEXT_COMPANION_SPEC_SCHEMA,
            "format": HASH_BOUND_OCR_FORMAT,
            "source_ref": binding.get("source_ref"),
            "source_pdf_path": binding.get("source_pdf_path"),
            "source_pdf_sha256": binding.get("source_pdf_sha256"),
            "pages": [
                {
                    key: row.get(key)
                    for key in (
                        "page_number",
                        "image_path",
                        "image_sha256",
                        "text_path",
                        "text_sha256",
                        "engine_id",
                        "engine_version",
                    )
                }
                for row in binding.get("pages") or []
                if isinstance(row, Mapping)
            ],
        }
    else:
        spec = {
            "schema_version": SOURCE_TEXT_COMPANION_SPEC_SCHEMA,
            "artifact_path": binding.get("artifact_path"),
            "artifact_sha256": binding.get("artifact_sha256"),
            "document_identity": binding.get("document_identity"),
            "source_url": binding.get("source_url"),
            "format": binding.get("format"),
            "authority_mode": binding.get("authority_mode"),
            "sections": [
                {
                    "page_number": row.get("page_number"),
                    "start_element_id": row.get("start_element_id"),
                    "end_element_id": row.get("end_element_id"),
                }
                for row in binding.get("sections") or []
                if isinstance(row, Mapping)
            ],
        }
    _, replayed, reasons = materialize_source_text_companion_pages(
        spec,
        source_ref=expected_source_ref,
    )
    return not reasons and replayed == binding


def primary_html_companion(raw: Mapping[str, Any]) -> bool:
    row = dict(raw) if isinstance(raw, Mapping) else {}
    return bool(
        row.get("format") == GOOGLE_PATENTS_HTML_FORMAT
        and row.get("authority_mode") == PRIMARY_HTML_AUTHORITY_MODE
    )


def official_google_patent_source(
    *,
    document_identity: str,
    source_url: str,
    source_ref: str,
) -> bool:
    identity = str(document_identity or "").strip()
    parsed = urlparse(str(source_url or ""))
    return bool(
        identity
        and str(source_ref or "").strip().casefold()
        == f"patent:{identity}".casefold()
        and parsed.scheme == "https"
        and (parsed.hostname or "").casefold() == "patents.google.com"
        and parsed.path.rstrip("/").casefold()
        == f"/patent/{identity}/en".casefold()
    )


def source_text_companion_location(
    raw: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Return the replay-bound HTML paragraph range for a synthetic text page."""

    binding = dict(raw) if isinstance(raw, Mapping) else {}
    if not primary_html_companion(binding):
        return {}
    matches = [
        dict(row)
        for row in binding.get("sections") or []
        if isinstance(row, Mapping)
        and int(row.get("page_number") or 0) == int(page_number)
    ]
    if len(matches) != 1:
        return {}
    section = matches[0]
    return {
        "kind": "html_paragraph_range",
        "start_element_id": str(section.get("start_element_id") or ""),
        "end_element_id": str(section.get("end_element_id") or ""),
        "text_sha256": str(section.get("text_sha256") or ""),
    }


def source_text_companion_matches_page(
    raw: Mapping[str, Any],
    *,
    page_number: int,
    image_sha256: str,
    source_pdf_sha256: str,
) -> bool:
    """Require OCR text to bind to the exact PDF page image used by a proof."""

    binding = dict(raw) if isinstance(raw, Mapping) else {}
    if binding.get("format") != HASH_BOUND_OCR_FORMAT:
        return True
    if str(binding.get("source_pdf_sha256") or "") != str(source_pdf_sha256 or ""):
        return False
    matches = [
        dict(row)
        for row in binding.get("pages") or []
        if isinstance(row, Mapping) and int(row.get("page_number") or 0) == int(page_number)
    ]
    return len(matches) == 1 and str(matches[0].get("image_sha256") or "") == str(
        image_sha256 or ""
    )


def _materialize_hash_bound_ocr_pages(
    spec: Mapping[str, Any],
    *,
    expected_source_ref: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...]]:
    pdf_path = Path(str(spec.get("source_pdf_path") or "")).expanduser()
    pdf_sha256 = str(spec.get("source_pdf_sha256") or "").strip().lower()
    source_ref = str(spec.get("source_ref") or "").strip().lower()
    reasons: list[str] = []
    if source_ref != expected_source_ref:
        reasons.append("source_ocr_source_ref_mismatch")
    if not pdf_path.is_file():
        reasons.append("source_ocr_pdf_missing")
    if not _is_sha256(pdf_sha256):
        reasons.append("source_ocr_pdf_sha256_invalid")
    pages = [dict(row) for row in spec.get("pages") or [] if isinstance(row, Mapping)]
    if not pages or len(pages) > 80:
        reasons.append("source_ocr_page_count_invalid")
    if reasons:
        return [], {}, tuple(sorted(set(reasons)))
    try:
        if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != pdf_sha256:
            return [], {}, ("source_ocr_pdf_digest_mismatch",)
    except OSError:
        return [], {}, ("source_ocr_pdf_unreadable",)

    normalized_pages: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, page in enumerate(pages, start=1):
        page_number = int(page.get("page_number") or 0)
        image_path = Path(str(page.get("image_path") or "")).expanduser()
        text_path = Path(str(page.get("text_path") or "")).expanduser()
        image_sha256 = str(page.get("image_sha256") or "").strip().lower()
        text_sha256 = str(page.get("text_sha256") or "").strip().lower()
        engine_id = str(page.get("engine_id") or "").strip()
        engine_version = str(page.get("engine_version") or "").strip()
        if (
            page_number <= 0
            or page_number in seen
            or not image_path.is_file()
            or not text_path.is_file()
            or not _is_sha256(image_sha256)
            or not _is_sha256(text_sha256)
            or not engine_id
            or not engine_version
        ):
            reasons.append(f"source_ocr_page_{index}_binding_invalid")
            continue
        seen.add(page_number)
        try:
            image_bytes = image_path.read_bytes()
            text_bytes = text_path.read_bytes()
        except OSError:
            reasons.append(f"source_ocr_page_{index}_artifact_unreadable")
            continue
        if len(text_bytes) > _MAX_COMPANION_BYTES:
            reasons.append(f"source_ocr_page_{index}_text_too_large")
            continue
        if hashlib.sha256(image_bytes).hexdigest() != image_sha256:
            reasons.append(f"source_ocr_page_{index}_image_digest_mismatch")
            continue
        if hashlib.sha256(text_bytes).hexdigest() != text_sha256:
            reasons.append(f"source_ocr_page_{index}_text_digest_mismatch")
            continue
        try:
            text = text_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            reasons.append(f"source_ocr_page_{index}_text_not_utf8")
            continue
        if not text:
            reasons.append(f"source_ocr_page_{index}_text_empty")
            continue
        normalized_pages.append(
            {
                "page_number": page_number,
                "image_path": str(image_path.resolve()),
                "image_sha256": image_sha256,
                "text_path": str(text_path.resolve()),
                "text_sha256": text_sha256,
                "engine_id": engine_id,
                "engine_version": engine_version,
            }
        )
        page_rows.append({"page_number": page_number, "text": text})
    if reasons or not normalized_pages:
        return [], {}, tuple(sorted(set(reasons or ["source_ocr_pages_missing"])))
    normalized_pages.sort(key=lambda row: int(row["page_number"]))
    binding: dict[str, Any] = {
        "schema_version": SOURCE_TEXT_COMPANION_BINDING_SCHEMA,
        "source_ref": expected_source_ref,
        "format": HASH_BOUND_OCR_FORMAT,
        "source_pdf_path": str(pdf_path.resolve()),
        "source_pdf_sha256": pdf_sha256,
        "pages": normalized_pages,
    }
    binding["content_sha256"] = _digest(binding)
    for page in page_rows:
        page["source_text_companion_binding"] = dict(binding)
    return page_rows, binding, ()


class _GooglePatentParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: dict[str, str] = {}
        self._active_id = ""
        self._active_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {str(key): str(value or "") for key, value in attrs}
        if self._active_id:
            self._active_depth += 1
            if tag in {"br", "p", "div"}:
                self._chunks.append("\n")
            return
        element_id = attributes.get("id", "").lower()
        classes = set(attributes.get("class", "").split())
        if (
            tag == "div"
            and re.fullmatch(r"p\d+", element_id)
            and "description-paragraph" in classes
        ):
            self._active_id = element_id
            self._active_depth = 1
            self._chunks = []

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if self._active_id and tag in {"br", "img"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        del tag
        if not self._active_id:
            return
        self._active_depth -= 1
        if self._active_depth > 0:
            return
        text = re.sub(r"\s+", " ", "".join(self._chunks)).strip()
        self.paragraphs[self._active_id] = text
        self._active_id = ""
        self._active_depth = 0
        self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._active_id:
            self._chunks.append(data)


def extract_google_patent_paragraphs(value: str) -> dict[str, str]:
    parser = _GooglePatentParagraphParser()
    parser.feed(str(value or ""))
    parser.close()
    return dict(parser.paragraphs)


def _paragraph_number(value: str) -> int | None:
    match = re.fullmatch(r"p(\d+)", str(value or "").strip().lower())
    return int(match.group(1)) if match else None


def _identity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _is_sha256(value: str) -> bool:
    text = str(value or "")
    return bool(
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
