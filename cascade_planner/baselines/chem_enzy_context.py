"""Context-source formatting for ChemEnzy ONMT cascade adapters."""
from __future__ import annotations

import re
from typing import Any


def build_chem_enzy_context_source(
    *,
    target_smiles: str,
    product_smiles: str,
    state: Any | None = None,
    step_index: int | None = None,
) -> str:
    """Build the pretokenized context source used by context ONMT training."""
    idx = int(step_index if step_index is not None else len(getattr(state, "step_annotations", []) or []))
    stage_id = _stage_for_next_step(state, idx)
    condition = _condition_for_stage(state, stage_id)
    tokens = [
        f"<step_{idx + 1}>",
        f"<stage_{_safe_token(stage_id)}>",
        f"<temp_{_bucket_temperature(_condition_value(condition, 'temperature'))}>",
        f"<ph_{_bucket_ph(_condition_value(condition, 'ph'))}>",
        f"<solv_{_safe_token(_condition_value(condition, 'solvent') or 'unknown')}>",
        f"<ec_{_safe_token(_ec_prefix_for_stage(state, stage_id))}>",
        "<target>",
        *_chem_enzy_smiles_tokens(target_smiles or product_smiles),
        "<product>",
        *_chem_enzy_smiles_tokens(product_smiles),
    ]
    return " ".join(tokens)


def chem_enzy_smiles_tokenize(smiles: str) -> str:
    return " ".join(_chem_enzy_smiles_tokens(smiles))


def _stage_for_next_step(state: Any | None, step_index: int) -> str:
    if state is None:
        return f"stage_{step_index + 1}"
    current = str(getattr(state, "current_stage", "") or "")
    if current:
        return current
    partition = list(getattr(state, "stage_partition", []) or [])
    if step_index < len(partition) and partition[step_index]:
        return str(partition[step_index])
    return f"stage_{step_index + 1}"


def _condition_for_stage(state: Any | None, stage_id: str) -> Any | None:
    if state is None:
        return None
    by_stage = getattr(state, "condition_envelope_by_stage", {}) or {}
    if stage_id in by_stage:
        return by_stage[stage_id]
    steps = list(getattr(state, "step_annotations", []) or [])
    for step in reversed(steps):
        if str(getattr(step, "stage_id", "") or "") == stage_id and getattr(step, "condition", None) is not None:
            return getattr(step, "condition")
    return None


def _ec_prefix_for_stage(state: Any | None, stage_id: str) -> str:
    if state is None:
        return "unknown"
    modules = list((getattr(state, "enzyme_context_by_stage", {}) or {}).get(stage_id) or [])
    for module in reversed(modules):
        ec_numbers = list(getattr(module, "ec_numbers", []) or [])
        if ec_numbers:
            return str(ec_numbers[0]).split(".", 1)[0]
    steps = list(getattr(state, "step_annotations", []) or [])
    for step in reversed(steps):
        if str(getattr(step, "stage_id", "") or "") != stage_id:
            continue
        ec_numbers = list(getattr(step, "ec_numbers", []) or [])
        if ec_numbers:
            return str(ec_numbers[0]).split(".", 1)[0]
    return "unknown"


def _condition_value(condition: Any | None, field: str) -> Any:
    if condition is None:
        return None
    if field == "temperature":
        values = [
            getattr(condition, "temperature_c_min", None),
            getattr(condition, "temperature_c_max", None),
        ]
        numbers = [_safe_float(value) for value in values if _safe_float(value) is not None]
        return sum(numbers) / len(numbers) if numbers else None
    if field == "ph":
        values = [getattr(condition, "ph_min", None), getattr(condition, "ph_max", None)]
        numbers = [_safe_float(value) for value in values if _safe_float(value) is not None]
        return sum(numbers) / len(numbers) if numbers else None
    if field == "solvent":
        solvents = list(getattr(condition, "solvents", []) or [])
        return solvents[0] if solvents else None
    return None


def _bucket_temperature(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "unknown"
    if number < 0:
        return "freezing"
    if number < 20:
        return "cold"
    if number <= 40:
        return "ambient"
    if number <= 70:
        return "warm"
    return "hot"


def _bucket_ph(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "unknown"
    if number < 4:
        return "acidic"
    if number <= 8:
        return "neutral"
    return "basic"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_token(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else "unknown"
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text or "unknown"


def _chem_enzy_smiles_tokens(text: str) -> list[str]:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    compact = str(text or "").replace(" ", "")
    tokens = [token for token in re.compile(pattern).findall(compact)]
    if compact != "".join(tokens):
        raise ValueError(f"SMILES tokenization failed for {text!r}")
    return tokens
