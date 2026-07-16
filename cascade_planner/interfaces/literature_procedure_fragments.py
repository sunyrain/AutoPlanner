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


def source_procedure_fragments(
    title: str,
    body: str,
) -> list[tuple[str, str, str, str]]:
    """Split only at a labelled heading followed by a procedural opening."""

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


__all__ = ["source_procedure_fragments"]
