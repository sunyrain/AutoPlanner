"""Semantic PMC article parsing and bounded procedure selection."""
from __future__ import annotations

from html.parser import HTMLParser
from typing import Iterable

from cascade_planner.interfaces.literature_procedure_fragments import (
    source_procedure_fragments,
)


class PmcArticleParser(HTMLParser):
    """Extract DOI identity and titled paragraphs from PMC article HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.citation_doi = ""
        self.pmcid = ""
        self.sections: list[tuple[str, str]] = []
        self._ignored_depth = 0
        self._capture = ""
        self._parts: list[str] = []
        self._title = ""

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        lowered = tag.casefold()
        attributes = {str(key).casefold(): str(value or "") for key, value in attrs}
        if lowered in {"script", "style"}:
            self._ignored_depth += 1
            return
        if lowered == "meta" and attributes.get("name", "").casefold() in {
            "citation_doi",
            "dc.identifier",
        }:
            value = attributes.get("content", "").strip()
            if value.casefold().startswith("doi:"):
                value = value[4:].strip()
            if value.startswith("10."):
                self.citation_doi = value
        if lowered == "meta" and attributes.get("name", "").casefold() == (
            "citation_pmcid"
        ):
            self.pmcid = attributes.get("content", "").strip().upper()
        if self._ignored_depth:
            return
        if lowered in {"h2", "h3", "h4", "p"}:
            self._capture = lowered
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth or lowered != self._capture:
            return
        text = " ".join("".join(self._parts).split())
        if lowered in {"h2", "h3", "h4"}:
            self._title = text[:1_000]
        elif lowered == "p" and len(text) >= 40:
            self.sections.append((self._title, text[:8_000]))
        self._capture = ""
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture and not self._ignored_depth:
            self._parts.append(data)


def parse_pmc_html(html_bytes: bytes) -> PmcArticleParser:
    parser = PmcArticleParser()
    try:
        parser.feed(html_bytes.decode("utf-8", errors="strict"))
        parser.close()
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("pmc_repository_html_parse_failed") from exc
    return parser


def html_procedure_inventory(
    sections: Iterable[tuple[str, str]],
    *,
    target_terms: Iterable[str],
    source_artifact_sha256: str,
    limit: int,
    source_artifact_kind: str = "pmc_fulltext_html",
) -> list[dict[str, object]]:
    terms = [
        " ".join(str(value).casefold().split())
        for value in target_terms
        if len(" ".join(str(value).split())) >= 3
    ][:64]
    title_signals = (
        "experimental",
        "materials and methods",
        "synthesis",
        "preparation",
        "production",
        "biotransformation",
    )
    process_signals = (
        "was added",
        "was stirred",
        "reaction mixture",
        "yield",
        "purified",
        "incubated",
        "catalyzed",
        "conversion",
    )
    expanded: list[tuple[str, str, str, str]] = []
    for title, body in sections:
        fragments = source_procedure_fragments(title, body)
        expanded.extend(fragments or [(title, body, "", "")])
    if any(label for _title, _body, label, _name in expanded):
        expanded = [row for row in expanded if row[2]]
    ranked: list[tuple[int, int, dict[str, object]]] = []
    for index, (title, body, label, source_name) in enumerate(expanded, start=1):
        normalized = f"{title} {body}".casefold()
        if label:
            # A bounded, source-authored compound heading is stronger than a
            # topic match.  Give every such block the same score so the
            # secondary source index preserves the experimental route order.
            score = 10_000
        else:
            score = 40 * sum(signal in title.casefold() for signal in title_signals)
            score += 20 * sum(term in normalized for term in terms)
            score += 4 * sum(signal in normalized for signal in process_signals)
        if score < 8:
            continue
        ranked.append(
            (
                -score,
                index,
                {
                    "label": label or f"html-section-{index}",
                    "name": source_name
                    or title
                    or f"PMC full-text paragraph {index}",
                    "visual_expected": False,
                    "page_number": index,
                    "procedure_excerpt": body[:4_000],
                    "source_artifact_kind": source_artifact_kind,
                    "source_artifact_sha256": source_artifact_sha256,
                },
            )
        )
    return [row for _score, _index, row in sorted(ranked)[:limit]]


__all__ = [
    "PmcArticleParser",
    "html_procedure_inventory",
    "parse_pmc_html",
    "source_procedure_fragments",
]
