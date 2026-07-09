"""Condition-field extraction helpers for verifier-first route checks."""
from __future__ import annotations

from typing import Any


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": (
        "T",
        "Temperature",
        "temperature",
        "temperature_c",
        "temp_c",
        "temperature_celsius",
    ),
    "ph": ("pH", "ph", "PH", "pH_value", "ph_value"),
    "solvent": ("solvent", "Solvent", "solvents", "solvent_smiles"),
    "catalyst": (
        "catalyst",
        "Catalyst",
        "catalysts",
        "reagent",
        "Reagent",
        "reagents",
        "reagent_smiles",
    ),
}

PREDICTION_KEYS = (
    "condition_predictions",
    "predicted_conditions",
    "condition_prediction",
)
NESTED_KEYS = (
    "condition",
    "conditions",
    "step_conditions",
    "reaction_conditions",
    "raw_condition",
)
RAW_CONTAINER_KEYS = ("raw_metadata", "metadata", "raw_backend_metadata")


def condition_predictions(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized condition evidence rows from common route schemas."""
    rows: list[dict[str, Any]] = []
    if isinstance(step, dict):
        rows.append(_direct_condition_row(step))
        for key in PREDICTION_KEYS:
            _append_condition_rows(rows, step.get(key))
        for key in NESTED_KEYS:
            _append_condition_rows(rows, step.get(key))
        for key in RAW_CONTAINER_KEYS:
            raw = step.get(key)
            if isinstance(raw, dict):
                rows.append(_direct_condition_row(raw))
                for nested_key in (*PREDICTION_KEYS, *NESTED_KEYS):
                    _append_condition_rows(rows, raw.get(nested_key))

    normalized = [_normalize_condition_row(row) for row in rows]
    return [row for row in _dedupe_rows(normalized) if row]


def condition_value(step: dict[str, Any], field: str) -> Any:
    """Return the first non-empty normalized value for a condition field."""
    aliases = FIELD_ALIASES[field]
    for row in condition_predictions(step):
        for key in aliases:
            if key in row and row[key] not in (None, "", []):
                value = row[key]
                if isinstance(value, list):
                    return _join_values(value)
                return value
    return None


def condition_summary(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": condition_value(step, "temperature"),
        "pH": condition_value(step, "ph"),
        "solvent": condition_value(step, "solvent"),
        "catalyst_or_reagent": condition_value(step, "catalyst"),
    }


def _append_condition_rows(rows: list[dict[str, Any]], value: Any) -> None:
    if isinstance(value, dict):
        rows.append(value)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                rows.append(item)


def _direct_condition_row(value: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for aliases in FIELD_ALIASES.values():
        for key in aliases:
            if key in value and value[key] not in (None, "", []):
                row[key] = value[key]
    for key in ("catalyst_classes", "cofactors"):
        if key in value and value[key] not in (None, "", []):
            row[key] = value[key]
    return row


def _normalize_condition_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row or {})
    for key in ("solvents", "reagents", "catalysts"):
        if key in normalized and key[:-1] not in normalized:
            normalized[key[:-1]] = normalized[key]
    if "temperature_c" in normalized and "Temperature" not in normalized:
        normalized["Temperature"] = normalized["temperature_c"]
    if "temperature" in normalized and "Temperature" not in normalized:
        normalized["Temperature"] = normalized["temperature"]
    if "ph" in normalized and "pH" not in normalized:
        normalized["pH"] = normalized["ph"]
    if "reagent_smiles" in normalized and "Reagent" not in normalized:
        normalized["Reagent"] = normalized["reagent_smiles"]
    for key in ("Reagent", "reagent", "Catalyst", "catalyst", "Solvent", "solvent"):
        if isinstance(normalized.get(key), list):
            normalized[key] = _join_values(normalized[key])
    if "Reagent" not in normalized:
        for key in ("reagent", "catalyst", "Catalyst"):
            if normalized.get(key) not in (None, "", []):
                normalized["Reagent"] = normalized[key]
                break
    return {key: value for key, value in normalized.items() if value not in (None, "", [])}


def _join_values(values: list[Any]) -> str:
    return ".".join(str(value).strip() for value in values if str(value).strip())


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        key = tuple(sorted((str(k), str(v)) for k, v in row.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
