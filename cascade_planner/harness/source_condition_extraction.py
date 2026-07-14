"""Deterministically project operational conditions from an exact procedure."""
from __future__ import annotations

import re
from typing import Any, Iterable


_SOLVENTS = (
    "dimethylformamide",
    "tetrahydrofuran",
    "dichloromethane",
    "ethyl acetate",
    "isopropanol",
    "acetonitrile",
    "methanol",
    "ethanol",
    "toluene",
    "acetone",
    "dioxane",
    "diethyl ether",
    "water",
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
        r"\b(?:the mixture was concentrated|the reaction mixture was concentrated|"
        r"the crude product|the resulting residue)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    solvents = [
        solvent for solvent in _SOLVENTS if _contains_term(reaction_phase, solvent)
    ]
    reagent_names = [
        " ".join(str(value).split())
        for value in source_amount_names
        if str(value).strip()
        and not any(_contains_term(str(value), solvent) for solvent in _SOLVENTS)
    ][:16]
    conditions: dict[str, Any] = {}
    if reagent_names:
        conditions["reagents"] = reagent_names
    if solvents:
        conditions["solvent"] = solvents

    temperature = _first_match(
        text,
        r"(?<!\w)(?:-\s*)?\d+(?:\.\d+)?\s*°?\s*C\b|"
        r"\broom temperature\b|\breflux\b",
    )
    if temperature:
        conditions["temperature"] = temperature
    duration = _first_match(
        text,
        r"\bovernight\b|\b\d+(?:\.\d+)?\s*(?:h|hr|hrs|hours?|min|minutes?)\b",
    )
    if duration:
        conditions["time"] = duration
    yield_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*%\s*(?:isolated\s+)?yield\b|"
        r"\bin\s+(\d+(?:\.\d+)?)\s*%\s+yield\b",
        text,
        flags=re.IGNORECASE,
    )
    if yield_match:
        conditions["yield_percent"] = float(
            yield_match.group(1) or yield_match.group(2)
        )
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
        terms=("concentrated", "washed", "filtered", "extracted", "quenched"),
        limit=800,
    )
    if workup:
        conditions["workup"] = workup
    purification = _bounded_sentences(
        text,
        terms=("chromatography", "recrystallized", "crystallized", "purified"),
        limit=800,
    )
    if purification:
        conditions["purification"] = purification
    return conditions


def _contains_term(value: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", value, flags=re.IGNORECASE))


def _first_match(value: str, pattern: str) -> str:
    match = re.search(pattern, value, flags=re.IGNORECASE)
    return " ".join(match.group(0).split()) if match else ""


def _bounded_sentences(value: str, *, terms: tuple[str, ...], limit: int) -> str:
    rows = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", value)
        if any(term in sentence.casefold() for term in terms)
    ]
    return " ".join(rows)[:limit]


__all__ = ["extract_source_conditions"]
