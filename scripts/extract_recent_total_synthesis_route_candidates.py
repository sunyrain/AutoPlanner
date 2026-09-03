#!/usr/bin/env python3
"""Extract target-linked route-evidence leads from article and SI artifacts.

The extractor is deliberately non-admitting. It binds every short passage to
an immutable source hash and a page, section, or archive-entry locator so a
chemist can review it without treating automated text matching as route truth.
"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import zipfile


TRANSFORMATION_RE = re.compile(
    r"\b(?:synthes|reaction|treat|convert|provid|afford|furnish|yield|reduc|"
    r"oxid|cycl|coupl|addition|condensation|deprotect|protect|benzyl|lacton|"
    r"rearrang|dimer|fragment|annulat|olefinat|hydrogenat|hydroly|eliminat|"
    r"epoxid|metathes|cross[- ]?coupl|photochem|radical)\w*\b",
    re.IGNORECASE,
)
SUPPORTED_ARTIFACT_KINDS = {
    "repository_fulltext_xml",
    "repository_main_pdf",
    "open_access_main_pdf",
    "publisher_text_mining_fulltext",
    "authorized_publisher_fulltext_html",
    "authorized_publisher_fulltext_xml",
    "authorized_publisher_main_pdf",
    "authorized_publisher_structured_text",
    "supporting_information",
}
PRIMARY_TARGET_SLOT_CLASSES = {"primary", "primary_candidate"}
ARTIFACT_PRIORITY = {
    "repository_fulltext_xml": 0,
    "authorized_publisher_fulltext_xml": 1,
    "authorized_publisher_fulltext_html": 2,
    "publisher_text_mining_fulltext": 3,
    "authorized_publisher_structured_text": 4,
    "repository_main_pdf": 5,
    "open_access_main_pdf": 6,
    "authorized_publisher_main_pdf": 7,
    "supporting_information": 8,
}
MAX_ARCHIVE_ENTRIES = 128
MAX_ARCHIVE_MEMBER_BYTES = 100_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-slots",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/target_slots.jsonl"),
    )
    parser.add_argument(
        "--source-receipts",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/source_package_receipts.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/route_evidence_candidates.jsonl"),
    )
    parser.add_argument("--max-passages", type=int, default=10)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def local_name(node: ET.Element) -> str:
    return node.tag.split("}")[-1]


def node_text(node: ET.Element) -> str:
    return " ".join("".join(node.itertext()).split())


def normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def target_match(text: str, target_name: str) -> bool:
    target_words = normalized_words(target_name)
    haystack = normalized_words(text)
    if not target_words:
        return False
    width = len(target_words)
    return any(
        haystack[index : index + width] == target_words
        for index in range(len(haystack) - width + 1)
    )


def section_title(section: ET.Element) -> str:
    return next(
        (node_text(child) for child in section if local_name(child) in {"title", "section-title"}),
        "",
    )


def _text_units(text: str) -> list[str]:
    cleaned = text.replace("\r", "\n")
    chunks = [" ".join(chunk.split()) for chunk in re.split(r"\n\s*\n+", cleaned)]
    chunks = [chunk for chunk in chunks if chunk]
    if len(chunks) > 1:
        return chunks
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", " ".join(text.split()))
        if sentence.strip()
    ]
    if len(sentences) <= 4:
        return sentences or chunks
    return [" ".join(sentences[index : index + 4]) for index in range(0, len(sentences), 3)]


def _xml_blocks(payload: bytes, *, container: str = "") -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    blocks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    sections = (node for node in root.iter() if local_name(node).casefold() in {"sec", "section"})
    for section_index, section in enumerate(sections, start=1):
        title = section_title(section)
        paragraph_index = 0
        for child in section.iter():
            if local_name(child).casefold() not in {"p", "para"}:
                continue
            text = node_text(child)
            key = (title, text)
            if not text or key in seen:
                continue
            seen.add(key)
            paragraph_index += 1
            blocks.append(
                {
                    "scope_key": f"xml-section:{section_index}",
                    "section_id": str(section.attrib.get("id") or ""),
                    "section_title": title,
                    "paragraph_index": paragraph_index,
                    "locator": {
                        "type": "xml_section_paragraph",
                        "section_id": str(section.attrib.get("id") or ""),
                        "section_index": section_index,
                        "paragraph_index": paragraph_index,
                        "container": container,
                    },
                    "cross_references": [
                        {
                            "rid": str(node.attrib.get("rid") or ""),
                            "ref_type": str(node.attrib.get("ref-type") or ""),
                        }
                        for node in child.iter()
                        if local_name(node) == "xref" and node.attrib.get("rid")
                    ],
                    "text": text,
                }
            )
    if blocks:
        return blocks
    for index, node in enumerate(root.iter(), start=1):
        if local_name(node).casefold() not in {"p", "para"}:
            continue
        text = node_text(node)
        if text:
            blocks.append(
                {
                    "scope_key": f"xml-document:{container}",
                    "section_id": "",
                    "section_title": "",
                    "paragraph_index": index,
                    "locator": {
                        "type": "xml_paragraph",
                        "paragraph_index": index,
                        "container": container,
                    },
                    "cross_references": [],
                    "text": text,
                }
            )
    return blocks


class _ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.heading = ""
        self._capture_tag = ""
        self._parts: list[str] = []
        self.blocks: list[dict[str, Any]] = []
        self._paragraph_index = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered in {"h1", "h2", "h3", "h4", "h5", "h6", "p"}:
            self._capture_tag = lowered
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_tag:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered != self._capture_tag:
            return
        text = " ".join(" ".join(self._parts).split())
        if lowered.startswith("h"):
            if text:
                self.heading = text
        elif text:
            self._paragraph_index += 1
            self.blocks.append(
                {
                    "scope_key": f"html-section:{self.heading}",
                    "section_id": "",
                    "section_title": self.heading,
                    "paragraph_index": self._paragraph_index,
                    "locator": {
                        "type": "html_section_paragraph",
                        "section_title": self.heading,
                        "paragraph_index": self._paragraph_index,
                    },
                    "cross_references": [],
                    "text": text,
                }
            )
        self._capture_tag = ""
        self._parts = []


def _html_blocks(payload: bytes) -> list[dict[str, Any]]:
    parser = _ArticleHTMLParser()
    parser.feed(payload.decode("utf-8", errors="ignore"))
    return parser.blocks


def _json_blocks(payload: bytes) -> list[dict[str, Any]]:
    document = json.loads(payload.decode("utf-8"))
    blocks: list[dict[str, Any]] = []
    for section_index, section in enumerate(document.get("full_text") or [], start=1):
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "")
        for paragraph_index, text in enumerate(
            _text_units(str(section.get("text") or "")), start=1
        ):
            blocks.append(
                {
                    "scope_key": f"json-section:{section_index}",
                    "section_id": f"structured-section-{section_index}",
                    "section_title": title,
                    "paragraph_index": paragraph_index,
                    "locator": {
                        "type": "structured_section_paragraph",
                        "section_index": section_index,
                        "paragraph_index": paragraph_index,
                    },
                    "cross_references": [],
                    "text": text,
                }
            )
    return blocks


def _pdf_blocks(payload: bytes, *, container: str = "") -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pymupdf_unavailable") from exc
    document = fitz.open(stream=payload, filetype="pdf")
    blocks: list[dict[str, Any]] = []
    try:
        for page_index in range(document.page_count):
            text = document[page_index].get_text("text") or ""
            for paragraph_index, unit in enumerate(_text_units(text), start=1):
                blocks.append(
                    {
                        "scope_key": f"pdf-page:{container}:{page_index + 1}",
                        "section_id": "",
                        "section_title": "",
                        "paragraph_index": paragraph_index,
                        "locator": {
                            "type": "pdf_page_paragraph",
                            "page_number": page_index + 1,
                            "paragraph_index": paragraph_index,
                            "container": container,
                        },
                        "cross_references": [],
                        "text": unit,
                    }
                )
    finally:
        document.close()
    return blocks


def _docx_blocks(payload: bytes, *, container: str = "") -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        xml_payload = archive.read("word/document.xml")
    root = ET.fromstring(xml_payload)
    blocks: list[dict[str, Any]] = []
    paragraph_index = 0
    for paragraph in root.iter():
        if local_name(paragraph).casefold() != "p":
            continue
        text = node_text(paragraph)
        if not text:
            continue
        paragraph_index += 1
        blocks.append(
            {
                "scope_key": f"docx:{container}",
                "section_id": "",
                "section_title": "",
                "paragraph_index": paragraph_index,
                "locator": {
                    "type": "docx_paragraph",
                    "paragraph_index": paragraph_index,
                    "container": container,
                },
                "cross_references": [],
                "text": text,
            }
        )
    return blocks


def _artifact_blocks(
    payload: bytes, suffix: str, *, container: str = "", archive_depth: int = 0
) -> list[dict[str, Any]]:
    lowered = suffix.casefold()
    if lowered in {".xml", ".nxml"}:
        return _xml_blocks(payload, container=container)
    if lowered in {".html", ".htm"}:
        return _html_blocks(payload)
    if lowered == ".json":
        return _json_blocks(payload)
    if lowered == ".pdf":
        return _pdf_blocks(payload, container=container)
    if lowered == ".docx":
        return _docx_blocks(payload, container=container)
    if lowered != ".zip" or archive_depth >= 2:
        return []
    blocks: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        for info in members[:MAX_ARCHIVE_ENTRIES]:
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                continue
            member_suffix = Path(info.filename).suffix.casefold()
            if member_suffix not in {
                ".xml",
                ".nxml",
                ".html",
                ".htm",
                ".json",
                ".pdf",
                ".docx",
                ".zip",
            }:
                continue
            member_container = f"{container}!{info.filename}" if container else info.filename
            blocks.extend(
                _artifact_blocks(
                    archive.read(info),
                    member_suffix,
                    container=member_container,
                    archive_depth=archive_depth + 1,
                )
            )
    return blocks


def _passage_candidates_from_blocks(
    blocks: Iterable[dict[str, Any]], target_name: str, *, max_passages: int
) -> list[dict[str, Any]]:
    materialized = list(blocks)
    target_scopes = {
        str(block.get("scope_key") or "")
        for block in materialized
        if target_match(str(block.get("text") or ""), target_name)
        or target_match(str(block.get("section_title") or ""), target_name)
    }
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for ordinal, block in enumerate(materialized):
        text = str(block.get("text") or "")
        title = str(block.get("section_title") or "")
        mention = target_match(text, target_name)
        title_match = target_match(title, target_name)
        same_scope = str(block.get("scope_key") or "") in target_scopes
        if not mention and not title_match and not same_scope:
            continue
        transformation_signals = len(TRANSFORMATION_RE.findall(text))
        if transformation_signals == 0:
            continue
        score = (
            (8 if mention else 0)
            + (5 if title_match else 0)
            + (2 if same_scope else 0)
            + min(transformation_signals, 6)
        )
        candidates.append(
            (
                score,
                ordinal,
                {
                    "section_id": str(block.get("section_id") or ""),
                    "section_title": title,
                    "paragraph_index": int(block.get("paragraph_index") or 0),
                    "target_mentioned_in_paragraph": mention,
                    "target_mentioned_in_section_title": title_match,
                    "target_mentioned_in_locator_scope": same_scope,
                    "transformation_signal_count": transformation_signals,
                    "cross_references": list(block.get("cross_references") or []),
                    "source_locator": dict(block.get("locator") or {}),
                    "verbatim_text": text,
                },
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [record for _score, _ordinal, record in candidates[:max_passages]]


def passage_candidates(
    payload: bytes, target_name: str, *, max_passages: int
) -> list[dict[str, Any]]:
    """Compatibility entry point for XML callers."""

    return _passage_candidates_from_blocks(
        _xml_blocks(payload), target_name, max_passages=max_passages
    )


def structured_json_passage_candidates(
    payload: bytes, target_name: str, *, max_passages: int
) -> list[dict[str, Any]]:
    return _passage_candidates_from_blocks(
        _json_blocks(payload), target_name, max_passages=max_passages
    )


def html_passage_candidates(
    payload: bytes, target_name: str, *, max_passages: int
) -> list[dict[str, Any]]:
    return _passage_candidates_from_blocks(
        _html_blocks(payload), target_name, max_passages=max_passages
    )


def main() -> int:
    args = parse_args()
    if args.max_passages < 1:
        raise ValueError("--max-passages must be at least 1")
    repo_root = Path(__file__).resolve().parents[1]
    slots = [
        row
        for row in read_jsonl((repo_root / args.target_slots).resolve())
        if row.get("slot_class") in PRIMARY_TARGET_SLOT_CLASSES
        and row.get("target_name")
    ]
    receipts = read_jsonl((repo_root / args.source_receipts).resolve())
    sources_by_paper: dict[str, list[dict[str, Any]]] = {}
    for receipt in receipts:
        supported = [
            dict(artifact)
            for artifact in receipt.get("artifacts") or []
            if str(artifact.get("artifact_kind") or "") in SUPPORTED_ARTIFACT_KINDS
        ]
        supported.sort(
            key=lambda artifact: (
                ARTIFACT_PRIORITY.get(str(artifact.get("artifact_kind") or ""), 99),
                str(artifact.get("cache_path") or ""),
            )
        )
        if supported:
            sources_by_paper[str(receipt["paper_id"])] = supported

    rows: list[dict[str, Any]] = []
    artifact_failures = 0
    for slot in slots:
        artifacts = sources_by_paper.get(str(slot["paper_id"]), [])
        if not artifacts:
            continue
        passages: list[dict[str, Any]] = []
        processed_artifacts: list[dict[str, Any]] = []
        extraction_errors: list[str] = []
        seen_passages: set[str] = set()
        for artifact in artifacts:
            source_path = repo_root / str(artifact["cache_path"])
            payload = source_path.read_bytes()
            source_sha256 = hashlib.sha256(payload).hexdigest()
            if source_sha256 != artifact["sha256"]:
                raise RuntimeError(f"source hash mismatch: {source_path}")
            processed_artifacts.append(
                {
                    "artifact_kind": str(artifact.get("artifact_kind") or ""),
                    "source_artifact_path": str(artifact["cache_path"]),
                    "source_artifact_sha256": source_sha256,
                    "source_url": str(artifact.get("source_url") or ""),
                }
            )
            try:
                blocks = _artifact_blocks(
                    payload,
                    source_path.suffix,
                    container=str(artifact["cache_path"]),
                )
                candidates = _passage_candidates_from_blocks(
                    blocks,
                    str(slot["target_name"]),
                    max_passages=args.max_passages,
                )
            except (ValueError, RuntimeError, ET.ParseError, zipfile.BadZipFile) as exc:
                extraction_errors.append(f"{artifact['cache_path']}:{type(exc).__name__}:{exc}")
                artifact_failures += 1
                continue
            for candidate in candidates:
                digest = hashlib.sha256(
                    str(candidate.get("verbatim_text") or "").encode("utf-8")
                ).hexdigest()
                if digest in seen_passages:
                    continue
                seen_passages.add(digest)
                passages.append(
                    {
                        **candidate,
                        "source_artifact_kind": str(artifact.get("artifact_kind") or ""),
                        "source_artifact_path": str(artifact["cache_path"]),
                        "source_artifact_sha256": source_sha256,
                    }
                )
                if len(passages) >= args.max_passages:
                    break
            if len(passages) >= args.max_passages:
                break

        primary = processed_artifacts[0]
        rows.append(
            {
                "schema_version": "recent_total_synthesis_route_evidence_candidate.v2",
                "target_slot_id": slot["target_slot_id"],
                "paper_id": slot["paper_id"],
                "doi": slot["doi"],
                "target_name": slot["target_name"],
                "source_artifact_path": primary["source_artifact_path"],
                "source_artifact_sha256": primary["source_artifact_sha256"],
                "source_url": primary["source_url"],
                "source_artifacts": processed_artifacts,
                "extraction_method": "deterministic_multiformat_target_context_v2",
                "extraction_status": (
                    "article_or_si_route_passages_found_unverified"
                    if passages
                    else "no_target_linked_route_passage_found"
                ),
                "route_or_key_step_admitted": False,
                "admission_authority": False,
                "supporting_information_required_for_reconstruction": True,
                "evidence_passages": passages,
                "extraction_errors": extraction_errors,
                "required_next_action": (
                    "review article schemes and SI, bind compound identities, and submit independent route reviews"
                ),
            }
        )

    output = (repo_root / args.output).resolve()
    write_jsonl(output, rows)
    print(
        json.dumps(
            {
                "source_package_papers": len(sources_by_paper),
                "target_rows_with_source_package": len(rows),
                "target_rows_with_route_passages": sum(
                    bool(row["evidence_passages"]) for row in rows
                ),
                "artifact_extraction_failures": artifact_failures,
                "admitted_route_records": 0,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
