"""Bounded, hash-bound materialization of primary patent HTML text."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from cascade_planner.harness.source_text_companion import (
    GOOGLE_PATENTS_HTML_FORMAT,
    PRIMARY_HTML_AUTHORITY_MODE,
    extract_google_patent_paragraphs,
    materialize_source_text_companion_pages,
    official_google_patent_source,
)


PATENT_HTML_MATERIALIZATION_SCHEMA = "primary_patent_html_materialization.v1"


@dataclass(frozen=True, slots=True)
class PatentHtmlConfig:
    max_bytes: int = 20_000_000
    max_sections: int = 24
    max_selected_paragraphs: int = 256
    window_before: int = 2
    window_after: int = 10

    def __post_init__(self) -> None:
        if self.max_bytes < 10_000:
            raise ValueError("patent_html_byte_limit_invalid")
        if not 1 <= self.max_sections <= 64:
            raise ValueError("patent_html_section_limit_invalid")
        if not 8 <= self.max_selected_paragraphs <= 1_024:
            raise ValueError("patent_html_paragraph_limit_invalid")
        if not 0 <= self.window_before <= 8 or not 0 <= self.window_after <= 16:
            raise ValueError("patent_html_window_invalid")


def materialize_primary_patent_html(
    *,
    content: bytes,
    publication: str,
    source_ref: str,
    source_url: str,
    output_dir: str | Path,
    target_terms: Iterable[str] = (),
    config: PatentHtmlConfig | None = None,
) -> dict[str, Any]:
    """Freeze full HTML and expose only bounded, replayable procedure windows."""

    active = config or PatentHtmlConfig()
    identity = str(publication or "").strip()
    source_key = str(source_ref or "").strip().lower()
    url = str(source_url or "").strip()
    if (
        not identity
        or not official_google_patent_source(
            document_identity=identity,
            source_url=url,
            source_ref=source_key,
        )
    ):
        return _result("failed", reasons=["patent_html_source_binding_invalid"])
    if len(content) < 100 or len(content) > active.max_bytes:
        return _result("failed", reasons=["patent_html_byte_limit_invalid"])
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return _result("failed", reasons=["patent_html_not_utf8"])
    if (
        "<html" not in text[:10_000].casefold()
        or _identity_key(identity) not in _identity_key(text)
    ):
        return _result("failed", reasons=["patent_html_identity_not_found"])
    try:
        paragraphs = extract_google_patent_paragraphs(text)
    except (RuntimeError, ValueError):
        paragraphs = {}
    if not paragraphs:
        return _result("failed", reasons=["patent_html_description_missing"])
    sections = _select_sections(
        paragraphs,
        target_terms=target_terms,
        config=active,
    )
    if not sections:
        return _result(
            "unresolved",
            reasons=["patent_html_relevant_procedure_windows_missing"],
            paragraph_count=len(paragraphs),
        )

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    artifact = out / f"{identity}-{digest[:16]}.google-patents.html"
    if not artifact.is_file():
        _write_bytes_atomic(artifact, content)
    spec = {
        "schema_version": "trusted_source_text_companion.v1",
        "artifact_path": str(artifact),
        "artifact_sha256": digest,
        "document_identity": identity,
        "source_url": url,
        "format": GOOGLE_PATENTS_HTML_FORMAT,
        "authority_mode": PRIMARY_HTML_AUTHORITY_MODE,
        "sections": sections,
    }
    pages, binding, reasons = materialize_source_text_companion_pages(
        spec,
        source_ref=source_key,
    )
    if reasons or not pages or not binding:
        return _result(
            "failed",
            reasons=reasons or ["patent_html_companion_replay_failed"],
            paragraph_count=len(paragraphs),
        )
    result = _result(
        "completed",
        artifact_path=str(artifact),
        artifact_sha256=digest,
        paragraph_count=len(paragraphs),
        selected_paragraph_count=sum(
            1
            for row in sections
            for number in range(
                _paragraph_number(row["start_element_id"]),
                _paragraph_number(row["end_element_id"]) + 1,
            )
            if f"p{number:04d}" in paragraphs
        ),
        section_count=len(sections),
        sections=sections,
        companion=spec,
    )
    manifest = out / "primary-patent-html-materialization.json"
    _write_json_atomic(manifest, result)
    return {**result, "manifest_path": str(manifest), "manifest_sha256": _sha256(manifest)}


def _select_sections(
    paragraphs: Mapping[str, str],
    *,
    target_terms: Iterable[str],
    config: PatentHtmlConfig,
) -> list[dict[str, Any]]:
    numbered = sorted(
        (
            (_paragraph_number(identifier), identifier, str(text or ""))
            for identifier, text in paragraphs.items()
            if re.fullmatch(r"p\d+", str(identifier).lower())
        ),
        key=lambda row: row[0],
    )
    if not numbered:
        return []
    terms = [
        " ".join(str(value or "").casefold().split())
        for value in target_terms
        if len(" ".join(str(value or "").split())) >= 3
    ][:128]
    signals = {
        "example": 2,
        "preparation of": 6,
        "synthesis of": 8,
        "general procedure": 8,
        "step ": 3,
        "was added": 4,
        "was treated": 4,
        "was stirred": 3,
        "afforded": 4,
        "yield": 2,
    }
    anchors: list[tuple[int, int]] = []
    example_starts = [
        index
        for index, (_number, _identifier, raw_text) in enumerate(numbered)
        if re.match(
            r"^example\s+[a-z0-9]",
            " ".join(raw_text.casefold().split()),
        )
    ]
    for index, (_number, _identifier, raw_text) in enumerate(numbered):
        text = " ".join(raw_text.casefold().split())
        target_score = sum(
            256 * min(2, text.count(term))
            for term in terms
            if term in text
        )
        process_score = sum(weight * min(3, text.count(term)) for term, weight in signals.items())
        score = target_score + process_score
        if target_score or process_score >= 8:
            anchors.append((score, index))
    anchors.sort(key=lambda row: (-row[0], row[1]))
    selected_indices: set[int] = set()
    for _score, index in anchors:
        window = _procedure_window(
            index,
            row_count=len(numbered),
            example_starts=example_starts,
            window_before=config.window_before,
            window_after=config.window_after,
        )
        candidate = selected_indices | window
        if len(candidate) > config.max_selected_paragraphs:
            continue
        if len(_index_ranges(candidate)) > config.max_sections:
            continue
        selected_indices = candidate
    selected = sorted(selected_indices)
    if not selected:
        return []
    ranges = _index_ranges(selected_indices)

    sections: list[dict[str, Any]] = []
    for start, end in ranges:
        start_id = numbered[start][1]
        end_id = numbered[end][1]
        sections.append(
            {
                "page_number": len(sections) + 1,
                "start_element_id": start_id,
                "end_element_id": end_id,
            }
        )
    return sections


def _procedure_window(
    index: int,
    *,
    row_count: int,
    example_starts: list[int],
    window_before: int,
    window_after: int,
) -> set[int]:
    previous = [value for value in example_starts if value <= index]
    if previous:
        start = previous[-1]
        following = [value for value in example_starts if value > start]
        end = (following[0] if following else row_count) - 1
        if index - start <= 48 and end - start + 1 <= 48:
            return set(range(start, end + 1))
    return set(
        range(
            max(0, index - window_before),
            min(row_count, index + window_after + 1),
        )
    )


def _index_ranges(values: Iterable[int]) -> list[tuple[int, int]]:
    selected = sorted(set(values))
    if not selected:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = selected[0]
    for index in selected[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return ranges


def _paragraph_number(value: str) -> int:
    match = re.fullmatch(r"p(\d+)", str(value or "").strip().lower())
    return int(match.group(1)) if match else 0


def _identity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _result(status: str, *, reasons: Iterable[str] = (), **values: Any) -> dict[str, Any]:
    row = {
        "schema_version": PATENT_HTML_MATERIALIZATION_SCHEMA,
        "status": status,
        "reasons": sorted(set(str(value) for value in reasons if str(value))),
        "model_invocations": 0,
        "visual_invocations": 0,
        "semantics": {
            "search_metadata_is_not_source_text": True,
            "full_html_bytes_are_hash_bound": True,
            "selected_paragraphs_are_replayed_from_full_html": True,
            "html_text_requires_independent_structure_reconstruction": True,
        },
        **values,
    }
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return row


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "PATENT_HTML_MATERIALIZATION_SCHEMA",
    "PatentHtmlConfig",
    "materialize_primary_patent_html",
]
