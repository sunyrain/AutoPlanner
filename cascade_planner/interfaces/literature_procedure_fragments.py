"""Split flattened publisher text into explicit compound procedures."""
from __future__ import annotations

import re


_PROCEDURE_START = (
    r"(?:To\s+(?:a|an)\b|A\s+(?:stirred\s+)?(?:solution|suspension|mixture)\b|"
    r"Into\s+(?:a|an)\b)"
)
_EXPLICIT_PROCEDURE_HEADING = re.compile(
    rf"(?i)(?<![A-Za-z0-9])(?:"
    rf"(?P<compound>Compound\s+(?P<compound_label>[A-Za-z]?\d+[A-Za-z]?))"
    rf"|(?P<named>[A-Z][A-Za-z'’-]*"
    rf"(?:\s+[A-Za-z][A-Za-z'’-]*){{0,5}})"
    rf"\s*\(\s*(?P<named_label>[A-Za-z]?\d+[A-Za-z]?)\s*\)"
    rf")\s*\.\s*(?={_PROCEDURE_START})"
)
_NUMBERED_LINE_HEADING = re.compile(
    r"(?im)^[ \t]*\((?P<label>\d{1,3}[A-Za-z]?)\)[ \t]+"
    r"(?P<name>[^\r\n]{3,180}?)[ \t]*(?=\r?$)"
)
_ENTRY_LINE_HEADING = re.compile(
    r"(?im)^[ \t]*(?P<name>[^\r\n]{3,180}?)[ \t]*"
    r"\([ \t]*Entry[ \t]+(?P<label>\d{1,3}[A-Za-z]?)[ \t]*\)"
    r"[ \t]*[.:]?[ \t]*(?P<inline>[^\r\n]*)(?=\r?$)"
)


def source_procedure_fragments(
    title: str,
    body: str,
) -> list[tuple[str, str, str, str]]:
    """Split only at a labelled heading followed by a procedural opening."""

    line_fragments = _line_procedure_fragments(title, body)
    if line_fragments:
        return line_fragments
    text = " ".join(str(body or "").split())
    matches = list(_EXPLICIT_PROCEDURE_HEADING.finditer(text))
    fragments: list[tuple[str, str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        procedure = text[match.end() : end].strip()
        if len(procedure) < 40:
            continue
        compound_name = " ".join(str(match.group("compound") or "").split())
        named_name = " ".join(str(match.group("named") or "").split())
        label = str(
            match.group("compound_label") or match.group("named_label") or ""
        ).upper()
        name = compound_name or (
            f"{named_name} ({label})" if named_name and label else named_name
        )
        fragments.append((str(title or ""), procedure, label, name))
    return fragments


def _line_procedure_fragments(
    title: str,
    body: str,
) -> list[tuple[str, str, str, str]]:
    """Recover patent ``(1) product`` and journal ``product (Entry 1)`` blocks."""

    text = str(body or "")
    headings: list[dict[str, str | int]] = []
    for pattern, kind in (
        (_NUMBERED_LINE_HEADING, "numbered"),
        (_ENTRY_LINE_HEADING, "entry"),
    ):
        for match in pattern.finditer(text):
            name = " ".join(str(match.group("name") or "").split())
            if kind == "numbered" and not _plausible_numbered_product(name):
                continue
            headings.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "label": str(match.group("label") or "").upper(),
                    "name": name,
                    "inline": (
                        str(match.group("inline") or "") if kind == "entry" else ""
                    ),
                }
            )
    headings.sort(key=lambda row: (int(row["start"]), int(row["end"])))
    fragments: list[tuple[str, str, str, str]] = []
    for index, heading in enumerate(headings):
        end = int(headings[index + 1]["start"]) if index + 1 < len(headings) else len(text)
        procedure = " ".join(
            (
                str(heading.get("inline") or "")
                + " "
                + text[int(heading["end"]) : end]
            ).split()
        )
        if len(procedure) < 40 or not _procedure_like(procedure[:1_200]):
            continue
        fragments.append(
            (
                str(title or ""),
                procedure,
                str(heading["label"]),
                str(heading["name"]),
            )
        )
    return fragments


def _plausible_numbered_product(value: str) -> bool:
    """Reject bibliography numbers and prose footnotes before route ranking."""

    name = " ".join(str(value or "").split()).strip(" .")
    lowered = name.casefold()
    if not 3 <= len(name) <= 180 or ";" in name:
        return False
    if re.search(
        r"\b(?:doi|journal|vol|volume|scheme|table|figure|fig)\b|"
        r"\b(?:j|chem|acta|lett|commun)\.[ \t]",
        lowered,
    ):
        return False
    if re.search(
        r"\b(?:can|could|would|should|was|were|is|are|has|have)\b",
        lowered,
    ):
        return False
    return bool(re.search(r"[A-Za-z]", name))


def _procedure_like(value: str) -> bool:
    text = str(value or "").casefold()
    return any(
        signal in text
        for signal in (
            " was added",
            " were added",
            " was stirred",
            "reaction mixture",
            "yield",
            "to a solution",
            "were charged",
        )
    )


__all__ = ["source_procedure_fragments"]
