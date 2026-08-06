"""Bounded parsing helpers for source-authored narrative reaction headings."""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_PRODUCT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:chembiosynthesis|biosynthesis|synthesis|preparation|production)"
        r"\s+of\s+(?P<name>.{3,180}?)"
        r"(?=\s+(?:from|using|with|in|by|was|were)\b|[.;,]|$)"
    ),
    re.compile(
        r"(?i)\bconversion\s+of\s+.{3,180}?\s+to\s+(?P<name>.{3,180}?)"
        r"(?=\s+(?:using|with|in|by|was|were)\b|[.;,]|$)"
    ),
)

_REACTANT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:chembiosynthesis|biosynthesis|synthesis|preparation|production)"
        r"\s+of\s+.{3,180}?\s+from\s+(?P<names>.{2,240}?)"
        r"(?=\s+(?:was|were|is|are)\b|\s+(?:using|by|in)\b|[.;]|$)"
    ),
    re.compile(
        r"(?i)\bconversion\s+of\s+(?P<names>.{2,240}?)\s+to\s+.{3,180}?"
        r"(?=\s+(?:was|were|is|are|using|with|by|in)\b|[.;]|$)"
    ),
)


def narrative_product_name_candidates(value: Any) -> list[str]:
    heading = " ".join(str(value or "").split())
    return _dedupe(
        " ".join(str(match.group("name") or "").split()).strip(" ,;.")
        for pattern in _PRODUCT_PATTERNS
        if (match := pattern.search(heading)) is not None
    )[:4]


def narrative_reactant_name_candidates(value: Any) -> list[str]:
    heading = " ".join(str(value or "").split())
    names: list[str] = []
    for pattern in _REACTANT_PATTERNS:
        match = pattern.search(heading)
        if match is None:
            continue
        names.extend(
            " ".join(raw.split()).strip(" ,;.")
            for raw in re.split(
                r"(?i)\s+(?:and|plus|together\s+with)\s+",
                str(match.group("names") or ""),
            )
        )
    return _dedupe(names)[:6]


def source_name_resolution_candidates(
    name: str,
    source_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    """Return only source-declared alias expansions plus the original name."""

    aliases = {
        str(key).casefold(): str(value).strip()
        for key, value in dict(source_aliases or {}).items()
        if str(key).strip() and str(value).strip()
    }
    expanded = aliases.get(name.casefold(), "")
    if not expanded:
        name_key = name.casefold()
        for alias, value in sorted(aliases.items(), key=lambda row: -len(row[0])):
            suffix = name_key[len(alias) :] if name_key.startswith(alias) else ""
            if suffix.startswith((" ", "-")):
                expanded = f"{value}{name[len(alias):]}"
                break
    candidates: list[str] = []
    if expanded and expanded.casefold() != name.casefold():
        candidates.extend(_systematic_alias_candidates(name, expanded))
        candidates.append(expanded)
    candidates.extend(_source_ocr_name_candidates(name))
    candidates.append(name)
    return _dedupe(candidates)


def _systematic_alias_candidates(alias: str, expanded: str) -> list[str]:
    alias_key = re.sub(r"[^a-z0-9]+", "-", alias.casefold()).strip("-")
    name_key = re.sub(r"\s+", " ", expanded.casefold()).strip()
    if (
        alias_key.endswith("s-mmp")
        and (
            (
                "dimethylbutyryl" in name_key
                and "mercaptopropionate" in name_key.replace(" ", "")
            )
            or (
                "dimethylbutanoylthio" in name_key
                and "propionate" in name_key
            )
        )
    ):
        return ["methyl 3-(2,2-dimethylbutanoylthio)propanoate"]
    return []


def _source_ocr_name_candidates(name: str) -> list[str]:
    """Return conservative repairs for common native-PDF font substitutions."""

    value = " ".join(str(name or "").split()).strip(" .,:;")
    if not value:
        return []
    # Journal product headings commonly carry a citation such as ``7a``
    # directly after the name.  It is not part of the chemical identity.
    without_reference = re.sub(
        r"(?i)(?<=[A-Za-z])\d{1,2}[A-Za-z]$",
        "",
        value,
    ).strip()
    candidates = [without_reference] if without_reference != value else []
    # In older embedded fonts ``m`` is often extracted as ``tn`` and lowercase
    # ``l`` as uppercase ``I``.  Keep both repairs as resolver candidates; the
    # source text itself remains unchanged and the structure resolver must
    # still independently reconstruct a valid molecule.
    repaired = re.sub(r"(?<=[a-z])I(?=[a-z])", "l", without_reference)
    repaired = re.sub(r"(?i)(?<=[a-z])tn(?=[a-z])", "m", repaired)
    if repaired and repaired != without_reference:
        candidates.append(repaired)
    # Native PDF line wrapping may insert whitespace after the systematic
    # nomenclature prefix ``cyclo`` (for example ``cyclo pentadiene``).
    joined_prefix = re.sub(r"(?i)\bcyclo\s+(?=[a-z])", "cyclo", repaired)
    if joined_prefix and joined_prefix not in {without_reference, repaired}:
        candidates.append(joined_prefix)
    return candidates


def _dedupe(values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and re.search(r"[A-Za-z]", text) and key not in seen:
            seen.add(key)
            output.append(text)
    return output


__all__ = [
    "narrative_product_name_candidates",
    "narrative_reactant_name_candidates",
    "source_name_resolution_candidates",
]
