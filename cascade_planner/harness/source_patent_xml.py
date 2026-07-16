"""Bounded, hash-bound materialization of official EPO ST.36 patent XML."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

from cascade_planner.harness.source_text_companion import (
    EPO_ST36_XML_FORMAT,
    PRIMARY_HTML_AUTHORITY_MODE,
    extract_epo_patent_description_blocks,
    materialize_source_text_companion_pages,
    official_epo_patent_source,
)


PATENT_XML_MATERIALIZATION_SCHEMA = "primary_patent_xml_materialization.v1"


@dataclass(frozen=True, slots=True)
class PatentXmlConfig:
    max_bytes: int = 20_000_000
    max_sections: int = 24
    max_selected_elements: int = 256
    window_before: int = 2
    window_after: int = 10

    def __post_init__(self) -> None:
        if self.max_bytes < 10_000:
            raise ValueError("patent_xml_byte_limit_invalid")
        if not 1 <= self.max_sections <= 64:
            raise ValueError("patent_xml_section_limit_invalid")
        if not 8 <= self.max_selected_elements <= 1_024:
            raise ValueError("patent_xml_element_limit_invalid")
        if not 0 <= self.window_before <= 8 or not 0 <= self.window_after <= 16:
            raise ValueError("patent_xml_window_invalid")


def materialize_primary_patent_xml(
    *,
    content: bytes,
    publication: str,
    source_ref: str,
    source_url: str,
    output_dir: str | Path,
    target_terms: Iterable[str] = (),
    config: PatentXmlConfig | None = None,
) -> dict[str, Any]:
    """Freeze official XML and expose bounded, replayable description ranges."""

    active = config or PatentXmlConfig()
    identity = _publication_identity(publication)
    source_key = str(source_ref or "").strip().lower()
    url = str(source_url or "").strip()
    if not identity or not official_epo_patent_source(
        document_identity=identity,
        source_url=url,
        source_ref=source_key,
    ):
        return _result("failed", reasons=["patent_xml_source_binding_invalid"])
    if len(content) < 100 or len(content) > active.max_bytes:
        return _result("failed", reasons=["patent_xml_byte_limit_invalid"])
    try:
        root, blocks = extract_epo_patent_description_blocks(content)
    except (ET.ParseError, RuntimeError, ValueError):
        return _result("failed", reasons=["patent_xml_parse_failed"])
    if _root_publication_identity(root) != identity:
        return _result("failed", reasons=["patent_xml_identity_not_found"])
    sections = _select_sections(
        blocks,
        target_terms=target_terms,
        config=active,
    )
    if not sections:
        return _result(
            "unresolved",
            reasons=["patent_xml_relevant_procedure_windows_missing"],
            element_count=len(blocks),
        )

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    artifact = out / f"{identity}-{digest[:16]}.epo-st36.xml"
    if not artifact.is_file():
        _write_bytes_atomic(artifact, content)
    spec = {
        "schema_version": "trusted_source_text_companion.v1",
        "artifact_path": str(artifact),
        "artifact_sha256": digest,
        "document_identity": identity,
        "source_url": url,
        "format": EPO_ST36_XML_FORMAT,
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
            reasons=reasons or ["patent_xml_companion_replay_failed"],
            element_count=len(blocks),
        )
    selected_ids = {
        str(block["element_id"])
        for section in sections
        for block in _blocks_in_range(
            blocks,
            str(section["start_element_id"]),
            str(section["end_element_id"]),
        )
    }
    result = _result(
        "completed",
        artifact_path=str(artifact),
        artifact_sha256=digest,
        element_count=len(blocks),
        selected_element_count=len(selected_ids),
        section_count=len(sections),
        sections=sections,
        companion=spec,
    )
    manifest = out / "primary-patent-xml-materialization.json"
    _write_json_atomic(manifest, result)
    return {**result, "manifest_path": str(manifest), "manifest_sha256": _sha256(manifest)}


def _select_sections(
    blocks: list[dict[str, str]],
    *,
    target_terms: Iterable[str],
    config: PatentXmlConfig,
) -> list[dict[str, Any]]:
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
        "was added": 4,
        "was treated": 4,
        "was stirred": 3,
        "afforded": 4,
        "yield": 2,
    }
    heading_indices = [
        index for index, row in enumerate(blocks) if row.get("kind") == "heading"
    ]
    anchors: list[tuple[int, int]] = []
    for index, row in enumerate(blocks):
        text = " ".join(str(row.get("text") or "").casefold().split())
        target_score = sum(
            256 * min(2, text.count(term)) for term in terms if term in text
        )
        process_score = sum(
            weight * min(3, text.count(term)) for term, weight in signals.items()
        )
        if target_score or process_score >= 8:
            anchors.append((target_score + process_score, index))
    anchors.sort(key=lambda row: (-row[0], row[1]))
    selected_indices: set[int] = set()
    for _score, index in anchors:
        window = _procedure_window(
            index,
            row_count=len(blocks),
            heading_indices=heading_indices,
            window_before=config.window_before,
            window_after=config.window_after,
        )
        candidate = selected_indices | window
        if len(candidate) > config.max_selected_elements:
            continue
        if len(_index_ranges(candidate)) > config.max_sections:
            continue
        selected_indices = candidate
    if not selected_indices:
        return []
    return [
        {
            "page_number": page_number,
            "start_element_id": str(blocks[start]["element_id"]),
            "end_element_id": str(blocks[end]["element_id"]),
        }
        for page_number, (start, end) in enumerate(
            _index_ranges(selected_indices),
            start=1,
        )
    ]


def _procedure_window(
    index: int,
    *,
    row_count: int,
    heading_indices: list[int],
    window_before: int,
    window_after: int,
) -> set[int]:
    previous = [value for value in heading_indices if value <= index]
    if previous:
        start = previous[-1]
        following = [value for value in heading_indices if value > start]
        end = (following[0] if following else row_count) - 1
        if index - start <= 24 and end - start + 1 <= 48:
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


def _blocks_in_range(
    blocks: list[dict[str, str]],
    start_element_id: str,
    end_element_id: str,
) -> list[dict[str, str]]:
    indices = {
        str(row.get("element_id") or ""): index for index, row in enumerate(blocks)
    }
    start = indices.get(start_element_id, -1)
    end = indices.get(end_element_id, -1)
    return blocks[start : end + 1] if start >= 0 and end >= start else []


def _root_publication_identity(root: ET.Element) -> str:
    return _publication_identity(
        "".join(
            [
                str(root.attrib.get("country") or ""),
                str(root.attrib.get("doc-number") or ""),
                str(root.attrib.get("kind") or ""),
            ]
        )
    )


def _publication_identity(value: Any) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return compact if re.fullmatch(r"EP\d{5,12}(?:A|B)\d", compact) else ""


def _result(status: str, *, reasons: Iterable[str] = (), **values: Any) -> dict[str, Any]:
    row = {
        "schema_version": PATENT_XML_MATERIALIZATION_SCHEMA,
        "status": status,
        "reasons": sorted(set(str(value) for value in reasons if str(value))),
        "model_invocations": 0,
        "visual_invocations": 0,
        "semantics": {
            "epo_publication_server_is_legally_authoritative": True,
            "full_xml_bytes_are_hash_bound": True,
            "selected_elements_are_replayed_from_full_xml": True,
            "xml_text_requires_independent_structure_reconstruction": True,
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
    "PATENT_XML_MATERIALIZATION_SCHEMA",
    "PatentXmlConfig",
    "materialize_primary_patent_xml",
]
