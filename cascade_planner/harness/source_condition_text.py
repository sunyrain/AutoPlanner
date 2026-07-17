"""Text constants and helpers for deterministic source-condition extraction."""
from __future__ import annotations

import re


SOLVENTS = {
    "dimethylformamide": ("dimethylformamide", "DMF"),
    "dimethyl sulfoxide": ("dimethyl sulfoxide", "DMSO"),
    "tetrahydrofuran": ("tetrahydrofuran", "THF"),
    "dichloromethane": ("dichloromethane", "DCM", "CH2Cl2"),
    "ethyl acetate": ("ethyl acetate", "AcOEt", "EtOAc"),
    "isopropanol": ("isopropanol", "i-PrOH"),
    "isopropyl acetate": ("isopropyl acetate", "i-PrOAc"),
    "acetic acid": ("acetic acid", "AcOH"),
    "acetonitrile": ("acetonitrile", "MeCN"),
    "methanol": ("methanol", "MeOH"),
    "ethanol": ("ethanol", "EtOH"),
    "toluene": ("toluene",),
    "acetone": ("acetone",),
    "dioxane": ("dioxane",),
    "pyridine": ("pyridine",),
    "4-methylpyridine": ("4-methylpyridine",),
    "ethylene glycol": ("ethylene glycol",),
    "diethyl ether": ("diethyl ether", "Et2O"),
    "heptane": ("heptane",),
    "water": ("water",),
}

BASES = {
    "triethylamine": ("triethylamine", "NEt3", "Et3N"),
    "sodium bicarbonate": ("sodium bicarbonate", "NaHCO3"),
    "sodium hydroxide": ("sodium hydroxide", "NaOH"),
    "ammonium hydroxide": ("ammonium hydroxide", "NH4OH"),
    "diisopropylethylamine": (
        "N,N-diisopropylethylamine",
        "diisopropylethylamine",
        "DIPEA",
        "Huenig's base",
        "Hünig's base",
    ),
}

TEMPERATURE_PATTERN = (
    r"(?<!\w)(?:-?\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*"
    r"-?\d+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*"
    r"(?:°|degrees?\s*)?\s*C\b|"
    r"\broom temperature\b|\breflux\b"
)
DURATION_PATTERN = (
    r"overnight|\d+(?:\.\d+)?\s*"
    r"(?:d|day|days|h|hr|hrs|hours?|min|minutes?)"
)


def contains_term(value: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", value, flags=re.IGNORECASE))


def first_match(value: str, pattern: str) -> str:
    match = re.search(pattern, value, flags=re.IGNORECASE)
    return " ".join(match.group(0).split()) if match else ""


def ordered_matches(value: str, pattern: str) -> list[str]:
    matches = [
        " ".join(match.group(0).split())
        for match in re.finditer(pattern, value, flags=re.IGNORECASE)
    ]
    return list(dict.fromkeys(matches))


def reaction_duration(value: str, program: list[str]) -> str:
    completion = re.search(
        rf"(?:conversion|yield|reaction)[^.]{{0,160}}?"
        rf"(?:after|for)\s+(?P<duration>{DURATION_PATTERN})\b",
        value,
        flags=re.IGNORECASE,
    )
    if completion:
        return " ".join(completion.group("duration").split())
    stirring = list(
        re.finditer(
            rf"\b(?:stirred|heated|cooled|allowed\s+to\s+stir)\b"
            rf"[^.]{{0,120}}?\bfor\s+(?P<duration>{DURATION_PATTERN})\b",
            value,
            flags=re.IGNORECASE,
        )
    )
    if stirring:
        return " ".join(stirring[-1].group("duration").split())
    return program[-1]


def without_analytical_sentences(value: str) -> str:
    return " ".join(
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", value)
        if not re.search(
            r"\b(?:HPLC|TLC|LCMS|HRMS|NMR|chromatograph|eluent|"
            r"retention time|Rf)\b|\b(?:1H|13C|19F|31P)\s+NMR\b",
            sentence,
            flags=re.IGNORECASE,
        )
    )


def ph_program(value: str) -> list[str]:
    output: list[str] = []
    number = r"\d+(?:\.\d+)?"
    for sentence in re.split(r"(?<=[.!?])\s+", value):
        if not re.search(r"\bpH\b", sentence, flags=re.IGNORECASE):
            continue
        matches = [
            *re.finditer(
                rf"\bpH\s*(?:=|at)?\s*(?P<value>{number})\b",
                sentence,
                flags=re.IGNORECASE,
            ),
            *re.finditer(
                rf"\b(?:from|to|controlled\s+at|maintained\s+at)\s+"
                rf"(?P<value>{number})\b",
                sentence,
                flags=re.IGNORECASE,
            ),
        ]
        matches.sort(key=lambda match: match.start())
        output.extend(match.group("value") for match in matches)
    return list(dict.fromkeys(output))


def name_key(value: str) -> str:
    return " " + re.sub(r"\s+", " ", str(value).casefold()).strip() + " "


def bounded_sentences(value: str, *, terms: tuple[str, ...], limit: int) -> str:
    rows = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", value)
        if any(term in sentence.casefold() for term in terms)
    ]
    return " ".join(rows)[:limit]


__all__ = [
    "BASES",
    "DURATION_PATTERN",
    "SOLVENTS",
    "TEMPERATURE_PATTERN",
    "bounded_sentences",
    "contains_term",
    "first_match",
    "name_key",
    "ordered_matches",
    "ph_program",
    "reaction_duration",
    "without_analytical_sentences",
]
