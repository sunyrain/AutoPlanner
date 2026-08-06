"""Producer-independent normalization for advisory reaction-condition predictions.

Condition predictions are operational suggestions, never reaction or source
proof.  Keeping their normalization outside proposal providers ensures that
Codex, ChemEnzy, templates, literature-derived routes, and manual proposals all
cross the same authority boundary after canonical materialization.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from cascade_planner.application.fact_lifecycle import graph_fact_lifecycle_state


CONDITION_PREDICTION_SCHEMA = "advisory_reaction_condition_prediction.v1"
CONDITION_PREDICTION_RESULT_SCHEMA = "reaction_condition_prediction_result.v1"

_ALIASES = {
    "catalysts": "catalyst",
    "Catalyst": "catalyst",
    "duration": "time",
    "Time": "time",
    "Reagent": "reagents",
    "reagent": "reagents",
    "reagent_smiles": "reagents",
    "Solvent": "solvent",
    "solvents": "solvent",
    "solvent_smiles": "solvent",
    "Temperature": "temperature_c",
    "temperature": "temperature_c",
    "temp_c": "temperature_c",
    "T": "temperature_c",
}
_OPERATIONAL_FIELDS = {
    "addition_order",
    "atmosphere",
    "base",
    "buffer",
    "catalyst",
    "concentration",
    "equivalents",
    "oxidant",
    "ph",
    "pressure",
    "purification",
    "reductant",
    "reagents",
    "scale",
    "solvent",
    "temperature_c",
    "time",
    "workup",
}
_METADATA_FIELDS = {
    "condition_model",
    "condition_prediction_issues",
    "confidence",
    "Confidence",
    "model",
    "rank",
    "Score",
    "score",
}


def normalize_condition_predictions(
    value: Any,
    *,
    max_candidates: int = 2,
    default_model: str = "",
    producer: str = "",
) -> list[dict[str, Any]]:
    """Return at most ``max_candidates`` useful, ranked advisory candidates."""

    rows = _prediction_rows(value)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = _normalize_row(raw, default_model=default_model, producer=producer)
        if not row or not condition_prediction_is_usable(row):
            continue
        identity = _digest(
            {
                key: row[key]
                for key in sorted(_OPERATIONAL_FIELDS)
                if key in row
            }
        )
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(row)
    normalized.sort(
        key=lambda row: (
            int(row.get("rank") or 10_000),
            -_finite_score(row.get("score")),
            _digest(row),
        )
    )
    limit = max(1, min(2, int(max_candidates or 2)))
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(normalized[:limit], start=1):
        out.append({**row, "rank": rank})
    return out


def condition_prediction_is_usable(value: Mapping[str, Any]) -> bool:
    """A usable prediction must contain chemistry, not only score metadata."""

    row = dict(value)
    if not any(row.get(key) not in (None, "", [], {}) for key in _OPERATIONAL_FIELDS):
        return False
    temperature = _number(row.get("temperature_c"))
    if temperature is not None and not -100.0 <= temperature <= 220.0:
        return False
    score = _number(row.get("score"))
    return score is None or score >= 0.0


def edge_has_usable_condition_prediction(edge: Mapping[str, Any]) -> bool:
    return bool(
        normalize_condition_predictions(
            edge.get("condition_predictions")
            or dict(edge.get("metadata") or {}).get("condition_predictions")
            or (),
        )
    )


def edge_has_complete_source_procedure(
    graph: Mapping[str, Any], edge: Mapping[str, Any]
) -> bool:
    procedures = dict(graph.get("procedure_records") or {})
    for record_id in edge.get("procedure_record_ids") or []:
        record = procedures.get(str(record_id))
        if not isinstance(record, Mapping):
            continue
        if graph_fact_lifecycle_state(
            graph, "procedure_record", str(record_id), record
        ).get("active") is not True:
            continue
        if dict(record.get("condition_completeness") or {}).get("complete") is True:
            return True
    return False


def reaction_smiles_for_edge(edge: Mapping[str, Any]) -> str:
    precursors = [
        str(value).strip()
        for value in edge.get("precursor_smiles") or []
        if str(value).strip()
    ]
    product = str(edge.get("product_smiles") or "").strip()
    return ".".join(precursors) + ">>" + product if precursors and product else ""


def predict_conditions_many(
    predictor: Any,
    reaction_smiles: Iterable[str],
    *,
    top_k: int = 2,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Call either a batch, point, or callable predictor without provider coupling."""

    reactions = list(dict.fromkeys(str(value) for value in reaction_smiles if str(value)))
    if not reactions:
        return {}, {}
    limit = max(1, min(2, int(top_k or 2)))
    if hasattr(predictor, "predict_many"):
        try:
            raw = predictor.predict_many(reactions, top_k=limit)
            if isinstance(raw, Mapping):
                return {str(key): value for key, value in raw.items()}, {}
        except Exception as exc:
            batch_error = f"{type(exc).__name__}:{exc}"
        else:
            batch_error = "batch_predictor_returned_non_mapping"
    else:
        batch_error = ""
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for reaction in reactions:
        try:
            if hasattr(predictor, "predict"):
                values[reaction] = predictor.predict(reaction, top_k=limit)
            else:
                try:
                    values[reaction] = predictor(reaction, top_k=limit)
                except TypeError:
                    values[reaction] = predictor(reaction, limit)
        except Exception as exc:
            errors[reaction] = f"{type(exc).__name__}:{exc}"
            if batch_error:
                errors[reaction] = f"batch={batch_error}; point={errors[reaction]}"
    return values, errors


def _prediction_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            return _prediction_rows(value.to_dict(orient="records"))
        except TypeError:
            return _prediction_rows(value.to_dict())
    if isinstance(value, tuple) and len(value) == 2:
        combos, scores = value
        rows: list[dict[str, Any]] = []
        for combo, score in zip(combos or [], scores or []):
            row = _combo_row(combo)
            if row:
                row["score"] = score
                rows.append(row)
        return rows
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, (list, tuple)):
        return [row for item in value if (row := _combo_row(item))]
    return []


def _combo_row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return {
            "temperature_c": value[0],
            "solvent": value[1],
            "reagents": value[2],
            "catalyst": value[3],
        }
    return {}


def _normalize_row(
    value: Mapping[str, Any], *, default_model: str, producer: str
) -> dict[str, Any]:
    raw = dict(value)
    row: dict[str, Any] = {}
    for raw_key, raw_value in raw.items():
        if raw_value in (None, "", [], {}):
            continue
        key = _ALIASES.get(str(raw_key), str(raw_key).lower())
        if key not in _OPERATIONAL_FIELDS and raw_key not in _METADATA_FIELDS:
            continue
        if key in {"temperature_c", "score", "confidence"}:
            number = _number(raw_value)
            if number is not None:
                row[key] = number
            continue
        if key == "reagents" and isinstance(raw_value, str):
            row[key] = [raw_value]
        else:
            row[key] = _json_value(raw_value)
    score = next(
        (_number(raw.get(key)) for key in ("score", "Score", "confidence", "Confidence") if _number(raw.get(key)) is not None),
        None,
    )
    if score is not None:
        row["score"] = score
    model = str(raw.get("condition_model") or raw.get("model") or default_model).strip()
    if model:
        row["condition_model"] = model
    if producer:
        row["prediction_producer"] = str(producer)
    row.update(
        {
            "schema_version": CONDITION_PREDICTION_SCHEMA,
            "authority_scope": "model_predicted_condition",
            "not_reaction_proof": True,
            "not_source_evidence": True,
        }
    )
    return row


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_score(value: Any) -> float:
    number = _number(value)
    return number if number is not None else -1.0


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CONDITION_PREDICTION_RESULT_SCHEMA",
    "CONDITION_PREDICTION_SCHEMA",
    "condition_prediction_is_usable",
    "edge_has_complete_source_procedure",
    "edge_has_usable_condition_prediction",
    "normalize_condition_predictions",
    "predict_conditions_many",
    "reaction_smiles_for_edge",
]
