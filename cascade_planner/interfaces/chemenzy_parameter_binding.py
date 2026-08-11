"""Fail-closed binding across ChemEnzy proposal, launcher, and runtime values."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from cascade_planner.interfaces.chemenzy_probe_contract import _content_sha256, _result


def provider_parameter_binding(
    request: Mapping[str, Any],
    *,
    launcher_request: Mapping[str, Any] | None,
    raw_result: Mapping[str, Any] | None,
    runtime_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare proposal, launcher, and effective native-search parameters."""

    limits = dict(request.get("limits") or {})
    launcher = dict(launcher_request or {})
    raw = dict(raw_result or {})
    effective = dict(raw.get("ui_metadata") or {})
    preflight = dict(runtime_preflight or {})
    expected_values = {
        "target_smiles": str(request.get("target_smiles") or ""),
        "search_preset": str(limits.get("search_preset") or "standard"),
        "max_routes": int(limits.get("max_routes") or 0),
        "max_steps": int(limits.get("max_steps") or 0),
        "iterations": int(limits.get("max_iterations") or 0),
        "expansion_topk": int(limits.get("expansion_topk") or 0),
        "timeout_s": float(limits.get("timeout_s") or 0.0),
        "random_seed": int(limits.get("random_seed") or 0),
        "one_step_models": list(limits.get("one_step_models") or []),
        "stock_names": list(limits.get("stock_names") or []),
        "stock_paths": _normalized_paths(limits.get("stock_paths")),
        "condition_prediction": bool(limits.get("enable_condition_prediction", True)),
        "enzyme_assignment": bool(limits.get("enable_enzyme_assignment", True)),
        "enzyme_coverage_sidecar": bool(
            limits.get("enable_enzyme_coverage_sidecar", True)
        ),
        "pandarallel_workers": int(limits.get("pandarallel_workers") or 0),
    }
    launcher_values = {
        "target_smiles": str(launcher.get("target_smiles") or ""),
        "search_preset": str(launcher.get("search_preset") or ""),
        "max_routes": _int_or_none(launcher.get("max_routes")),
        "max_steps": _int_or_none(launcher.get("max_steps")),
        "iterations": _int_or_none(launcher.get("chem_enzy_iterations")),
        "expansion_topk": _int_or_none(launcher.get("chem_enzy_expansion_topk")),
        "timeout_s": _float_or_none(launcher.get("timeout_s")),
        "random_seed": _int_or_none(launcher.get("chemenzy_seed")),
        "one_step_models": list(launcher.get("one_step_models") or []),
        "stock_names": list(launcher.get("stock_names") or []),
        "stock_paths": _normalized_paths(launcher.get("stock_paths")),
        "condition_prediction": launcher.get("enable_condition_prediction"),
        "enzyme_assignment": launcher.get("enable_enzyme_assignment"),
        "enzyme_coverage_sidecar": launcher.get("enable_enzyme_coverage_sidecar"),
        "pandarallel_workers": _int_or_none(launcher.get("pandarallel_workers")),
    }
    effective_values = {
        "target_smiles": str(raw.get("target") or ""),
        "search_preset": str(effective.get("search_preset") or ""),
        "max_routes": _int_or_none(effective.get("max_routes")),
        "max_steps": _int_or_none(effective.get("max_depth")),
        "iterations": _int_or_none(effective.get("iterations")),
        "expansion_topk": _int_or_none(effective.get("expansion_topk")),
        "timeout_s": _float_or_none(effective.get("timeout_s")),
        "random_seed": _int_or_none(effective.get("random_seed")),
        "one_step_models": list(effective.get("one_step_models") or []),
        "stock_names": list(effective.get("stock_names") or []),
        "stock_paths": _normalized_paths(effective.get("stock_paths")),
        "condition_prediction": effective.get("condition_prediction_enabled"),
        "enzyme_assignment": effective.get("enzyme_assignment_enabled"),
        "enzyme_coverage_sidecar": effective.get("enzyme_coverage_sidecar_enabled"),
        "pandarallel_workers": _int_or_none(effective.get("pandarallel_workers")),
    }
    rows = {}
    for name, expected in expected_values.items():
        launcher_value = launcher_values[name]
        effective_value = effective_values[name]
        complete = all(
            _parameter_present(value)
            for value in (expected, launcher_value, effective_value)
        )
        rows[name] = {
            "expected": expected,
            "launcher_request": launcher_value,
            "effective": effective_value,
            "identity_complete": complete,
            "equal": complete
            and expected == launcher_value
            and expected == effective_value,
        }
    identity_complete = bool(rows) and all(
        row["identity_complete"] for row in rows.values()
    )
    binding = {
        "schema_version": "chemenzy_parameter_binding.v1",
        "accepted": identity_complete and all(row["equal"] for row in rows.values()),
        "identity_complete": identity_complete,
        "mismatch_fields": [
            name
            for name, row in rows.items()
            if row["identity_complete"] and not row["equal"]
        ],
        "incomplete_fields": [
            name for name, row in rows.items() if not row["identity_complete"]
        ],
        "fields": rows,
        "runtime_requested_one_step_models": list(
            preflight.get("requested_one_step_models") or []
        ),
        "semantics": {
            "proposal_launcher_and_effective_values_must_match": True,
            "builtin_execution_fails_closed_on_mismatch": True,
            "custom_provider_without_launcher_receipt_is_not_claimed_bound": True,
        },
    }
    binding["content_sha256"] = _content_sha256(binding)
    return binding


def bind_builtin_provider_parameters(
    request: Mapping[str, Any],
    *,
    raw_result: Mapping[str, Any],
    builtin: bool,
    mode: str,
    scope: str,
    limits: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compile the binding and return a fail-closed stage result if required."""

    raw = dict(raw_result)
    binding = provider_parameter_binding(
        request,
        launcher_request=raw.get("launcher_request"),
        raw_result=raw,
        runtime_preflight=raw.get("runtime_preflight") or raw.get("preflight") or {},
    )
    if not (
        builtin
        and raw.get("search_executed") is True
        and binding.get("accepted") is not True
    ):
        return binding, None
    return binding, _result(
        "failed",
        mode=mode,
        scope=scope,
        limits=dict(limits),
        reason="chemenzy_parameter_binding_mismatch",
        provider_parameter_binding=binding,
        runtime_preflight=raw.get("runtime_preflight") or raw.get("preflight") or {},
        runtime_discovery=raw.get("runtime_discovery") or {},
    )


def _normalized_paths(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): str(Path(str(path)).expanduser().resolve())
        for name, path in sorted(value.items(), key=lambda row: str(row[0]))
        if str(name).strip() and str(path).strip()
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parameter_present(value: Any) -> bool:
    return value is not None and value != "" and value != []


__all__ = ["bind_builtin_provider_parameters", "provider_parameter_binding"]
