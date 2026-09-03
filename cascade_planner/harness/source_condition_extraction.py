"""Deterministically project operational conditions from an exact procedure."""
from __future__ import annotations

import re
from typing import Any, Iterable
from cascade_planner.harness.source_condition_text import (
    BASES as _BASES,
    DURATION_PATTERN as _DURATION_PATTERN,
    SOLVENTS as _SOLVENTS,
    TEMPERATURE_PATTERN as _TEMPERATURE_PATTERN,
    bounded_sentences as _bounded_sentences,
    contains_term as _contains_term,
    first_match as _first_match,
    name_key as _name_key,
    ordered_matches as _ordered_matches,
    ph_program as _ph_program,
    reaction_duration as _reaction_duration,
    without_analytical_sentences as _without_analytical_sentences,
)


def extract_source_conditions(
    procedure_text: str,
    *,
    source_amount_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Return only fields explicitly recoverable from source procedure text."""

    text = " ".join(str(procedure_text or "").split())
    if not text:
        return {}
    reaction_phase = re.split(
        r"\b(?:after (?:completion|completeness) of the reaction|"
        r"(?:the reaction(?: mixture)?|the mixture|it) was quenched|"
        r"(?:whereupon\s+)?(?:it|the (?:reaction )?mixture) was partitioned|"
        r"the (?:aqueous|organic) layer was extracted|"
        r",\s*(?:whereupon\s+)?(?:the mixture was\s+)?"
        r"(?:extracted|partitioned|washed|filtered)\b|"
        r"the mixture was concentrated|the reaction mixture was concentrated|"
        r"the crude product|the resulting residue)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    analytical_free_phase = _without_analytical_sentences(reaction_phase)
    solvents = [
        solvent
        for solvent, aliases in _SOLVENTS.items()
        if any(_contains_term(analytical_free_phase, alias) for alias in aliases)
    ]
    reagent_names = [
        " ".join(str(value).split())
        for value in source_amount_names
        if str(value).strip()
        and _name_key(str(value)) in _name_key(reaction_phase)
        and not any(
            _contains_term(str(value), alias)
            for aliases in _SOLVENTS.values()
            for alias in aliases
        )
    ][:16]
    conditions: dict[str, Any] = {}
    if reagent_names:
        conditions["reagents"] = reagent_names
    if solvents:
        conditions["solvent"] = solvents
    bases = [
        base
        for base, aliases in _BASES.items()
        if any(_contains_term(reaction_phase, alias) for alias in aliases)
    ]
    if bases:
        conditions["base"] = bases

    temperature_program = _ordered_matches(
        analytical_free_phase, _TEMPERATURE_PATTERN
    )
    if temperature_program:
        conditions["temperature_program"] = temperature_program
        conditions["temperature"] = " → ".join(temperature_program)
    time_program = _ordered_matches(
        analytical_free_phase,
        rf"\b(?:{_DURATION_PATTERN})\b",
    )
    if time_program:
        conditions["time_program"] = time_program
        conditions["time"] = _reaction_duration(
            analytical_free_phase, time_program
        )
    yield_range = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
        r"(\d+(?:\.\d+)?)\s*%\s*(?:isolated\s+)?yield\b",
        text,
        flags=re.IGNORECASE,
    )
    yield_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*%\s*(?:isolated\s+)?yield\b|"
        r"\bin\s+(\d+(?:\.\d+)?)\s*%\s+yield\b|"
        r"\byield(?:ed|ing)?\b[^%]{0,180}?"
        r"\(?\s*(\d+(?:\.\d+)?)\s*%\s*\)?|"
        r"\b(?:afford(?:ed|ing)?|gave|give)\b[^\n]{0,120}?"
        r"\(\s*[^)]{0,80}?(\d+(?:\.\d+)?)\s*%\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if yield_range:
        conditions["yield_percent_range"] = {
            "min": float(yield_range.group(1)),
            "max": float(yield_range.group(2)),
        }
    elif yield_match:
        conditions["yield_percent"] = float(
            yield_match.group(1)
            or yield_match.group(2)
            or yield_match.group(3)
            or yield_match.group(4)
        )
    conversion_range = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
        r"(\d+(?:\.\d+)?)\s*%\s+conversion\b",
        text,
        flags=re.IGNORECASE,
    )
    conversion = re.search(
        r"\b(?:approximately\s+|about\s+|~\s*)?"
        r"(\d+(?:\.\d+)?)\s*%\s+conversion\b",
        text,
        flags=re.IGNORECASE,
    )
    if conversion_range:
        conditions["conversion_percent_range"] = {
            "min": float(conversion_range.group(1)),
            "max": float(conversion_range.group(2)),
        }
    elif conversion:
        conditions["conversion_percent"] = float(conversion.group(1))
    ph_program = _ph_program(reaction_phase)
    if ph_program:
        conditions["ph_program"] = ph_program
        conditions["ph"] = ph_program[-1]
    agitation_program = _ordered_matches(
        reaction_phase,
        r"\b\d+(?:\.\d+)?\s*rpm\b",
    )
    if agitation_program:
        conditions["agitation_program"] = agitation_program
        conditions["agitation"] = agitation_program[-1]
    catalysts = _ordered_matches(
        reaction_phase,
        r"\b(?:variant\s+)?LovD(?:\s+(?:acyltransferase|enzyme|polypeptide))?\b|"
        r"\bE\.\s*coli\b[^.]{0,80}?\b(?:biocatalyst|expressing\s+LovD)\b",
    )
    if catalysts:
        conditions["catalyst"] = catalysts
    atmosphere = _first_match(text, r"\b(?:under|flow of)\s+(?:nitrogen|hydrogen)\b")
    if atmosphere:
        conditions["atmosphere"] = atmosphere
    pressure = _first_match(text, r"\b\d+(?:\.\d+)?\s*(?:psi|bar)\b")
    if pressure:
        conditions["pressure"] = pressure
    scale = _first_match(
        reaction_phase,
        r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg)\s*,\s*"
        r"\d+(?:\.\d+)?\s*(?:mmoles?|mmol|moles?|mol)\b",
    )
    if scale:
        conditions["scale"] = scale

    addition_order = _bounded_sentences(
        reaction_phase,
        terms=("charged", "added", "treated"),
        limit=600,
    )
    if addition_order:
        conditions["addition_order"] = addition_order
    workup = _bounded_sentences(
        text,
        terms=(
            "concentrated",
            "evaporated",
            "extracted",
            "extraction",
            "filtered",
            "quenched",
            "suspended",
            "washed",
        ),
        limit=800,
    )
    if workup:
        conditions["workup"] = workup
    purification = _bounded_sentences(
        _without_analytical_sentences(text),
        terms=("chromatography", "recrystallized", "crystallized", "purified"),
        limit=800,
    )
    if purification:
        conditions["purification"] = purification
    return conditions


__all__ = ["extract_source_conditions"]
