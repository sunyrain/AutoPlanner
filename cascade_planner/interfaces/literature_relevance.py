"""Target relevance policy for bounded primary-literature acquisition."""
from __future__ import annotations

import html
import re
from typing import Any, Iterable, Mapping


_ROUTE_TERMS = (
    "synthesis",
    "synthetic",
    "preparation",
    "production",
    "biocatalysis",
    "biotransformation",
    "process",
)
_NOISE_TERMS = (
    "absorption",
    "clinical",
    "channel function",
    "chitosan",
    "cholesterol synthesis",
    "drug delivery",
    "enhances",
    "effects of inhibition",
    "fatty acid",
    "fibroblast",
    "induction of",
    "inhibition of",
    "inhibitor",
    "formulation",
    "linoleic",
    "mucoadhesive",
    "nanoparticle",
    "pharmacokinetic",
    "pharmacodynamic",
    "metabolism",
    "therapy",
    "treatment",
    "hepatocyte",
    "cancer",
    "syndrome",
    "triglyceride",
    "toxicity",
)
_DERIVATIVE_TERMS = (
    "analog",
    "analogue",
    "conjugate",
    "derivative",
    "derivatives",
    "prodrug",
)


def target_relevant_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_name: str,
    pinned_source_refs: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Rank route sources and reject clinical/biological search noise."""

    target = " ".join(str(target_name or "").split()).casefold()
    generic = target in {"", "target", "blind target"} or bool(
        re.fullmatch(r"target-[0-9a-f]{8}", target)
    )
    pinned = {str(value).strip().casefold() for value in pinned_source_refs}
    ranked: list[
        tuple[
            tuple[int, int, int, int, int, int, int, int, str],
            dict[str, Any],
        ]
    ] = []
    for raw in rows:
        row = dict(raw)
        is_pinned = candidate_source_ref(row).casefold() in pinned
        title = html.unescape(" ".join(str(row.get("title") or "").split())).casefold()
        route_terms = sum(term in title for term in _ROUTE_TERMS)
        noise_terms = sum(term in title for term in _NOISE_TERMS)
        derivative_terms = sum(term in title for term in _DERIVATIVE_TERMS)
        exact_route_phrase = any(
            phrase in title
            for phrase in (
                f"synthesis of {target}",
                f"preparation of {target}",
                f"production of {target}",
                f"{target} synthesis",
            )
            if target
        )
        if not is_pinned and not generic:
            if target not in title or route_terms <= 0:
                continue
            if noise_terms and not exact_route_phrase:
                continue
        ranked.append(
            (
                (
                    -int(is_pinned),
                    -int(row.get("target_edge_occurrence_count") or 0),
                    -int(row.get("corroborating_source_ref_count") or 0),
                    -int(row.get("occurrence_count") or 0),
                    -int(row.get("route_skeleton_count") or 0),
                    -int(exact_route_phrase),
                    -(route_terms - noise_terms - derivative_terms),
                    derivative_terms,
                    title,
                ),
                row,
            )
        )
    return [row for _rank, row in sorted(ranked, key=lambda value: value[0])]


def candidate_source_ref(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("source_ref") or "").strip()
    if explicit:
        return explicit
    value = doi(row)
    return f"doi:{value}" if value else ""


def doi(row: Mapping[str, Any]) -> str:
    value = str(row.get("doi") or "").strip()
    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE
    )


__all__ = ["candidate_source_ref", "doi", "target_relevant_candidates"]
