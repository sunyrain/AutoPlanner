"""Recover compound procedures from publisher text line headings."""

from __future__ import annotations

import re


_NUMBERED_LINE_HEADING = re.compile(
    r"(?im)^[ \t]*\((?P<label>\d{1,3}[A-Za-z]?)\)[ \t]+"
    r"(?P<name>[^\r\n]{3,180}?)[ \t]*(?=\r?$)"
)
_ENTRY_LINE_HEADING = re.compile(
    r"(?im)^[ \t]*(?P<name>[^\r\n]{3,180}?)[ \t]*"
    r"\([ \t]*Entry[ \t]+(?P<label>\d{1,3}[A-Za-z]?)[ \t]*\)"
    r"[ \t]*[.:]?[ \t]*(?P<inline>[^\r\n]*)(?=\r?$)"
)


def line_procedure_fragments(
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


__all__ = ["line_procedure_fragments"]
