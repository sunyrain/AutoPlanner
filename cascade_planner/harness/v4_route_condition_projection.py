"""Condition rows for the V4 route-workbench adapter."""

from __future__ import annotations

from typing import Any, Mapping


def route_conditions(
    inspector: Mapping[str, Any], records: list[Mapping[str, Any]]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    predictions = [
        dict(value)
        for value in inspector.get("condition_predictions") or []
        if isinstance(value, Mapping)
    ]
    return source_conditions(records) or predicted_conditions(predictions), predictions


def source_conditions(records: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    for record in records:
        raw = record.get("conditions")
        if not isinstance(raw, Mapping) or not raw:
            continue
        return [
            {"label": str(key).replace("_", " "), "value": str(value)}
            for key, value in sorted(raw.items())
            if value not in (None, "", [], {})
        ]
    return []


def predicted_conditions(records: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Project the best model proposal without merging ranked alternatives."""

    aliases = (
        ("reagents", ("Reagent", "reagent", "reagents", "reagent_smiles")),
        ("catalyst", ("Catalyst", "catalyst", "catalysts")),
        ("solvent", ("Solvent", "solvent", "solvents", "solvent_smiles")),
        (
            "temperature",
            ("Temperature", "temperature", "temperature_c", "temp_c"),
        ),
        ("time", ("Time", "time", "duration")),
        ("pH", ("pH", "ph", "PH")),
    )
    for record in records:
        if not isinstance(record, Mapping):
            continue
        rows: list[dict[str, str]] = []
        for label, keys in aliases:
            value = next(
                (record.get(key) for key in keys if record.get(key) not in (None, "", [], {})),
                None,
            )
            if value is None:
                continue
            if label == "temperature":
                try:
                    value = f"{float(value):.1f} °C"
                except (TypeError, ValueError):
                    value = str(value)
            elif isinstance(value, (list, tuple)):
                value = ", ".join(str(item) for item in value if str(item))
            rows.append({"label": label, "value": str(value)})
        if rows:
            return rows
    return []


__all__ = ["predicted_conditions", "route_conditions", "source_conditions"]
