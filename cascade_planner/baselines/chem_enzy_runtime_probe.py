"""Isolated, model-free ChemEnzy runtime capability probe.

This module is executed by the candidate ChemEnzy interpreter.  It imports
the compatibility layer and vendor API, but deliberately never constructs an
RSPlanner, deserializes a checkpoint, or performs a search.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from cascade_planner.baselines.chem_enzy_adapter import (
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
    ChemEnzyBackendAdapter,
    _materialize_selected_one_step_io_paths,
    _normal_absolute_path,
    _patch_dgl_graphbolt_optional_import,
    _patch_numpy_legacy_aliases,
    _patch_optional_easifa_import,
    _patch_optional_graphviz_import,
    _patch_torchdata_legacy_aliases,
    _patch_torchtext_legacy_aliases,
    _prune_unavailable_one_step_models,
    _resolve_vendor_model_path,
    _vendor_pythonpath,
    _windows_extended_path,
)
from cascade_planner.baselines.route_contract import RouteSearchConfig


CAPABILITY_PROBE_SCHEMA = "chemenzy_runtime_capability_probe.v1"
_PACKAGE_NAMES = (
    "numpy",
    "PyYAML",
    "rdkit",
    "setuptools",
    "torch",
    "torchtext",
    "transformers",
)
_PROBED_STOCKS = (*DEFAULT_STOCKS, "PaRotes_n1-stock")
_MODEL_OVERRIDE_KEYS = {
    "chem_enzy_onmt_model_path",
    "chem_enzy_onmt_tokenizer",
    "onmt_model_path",
    "onmt_tokenizer",
}


def probe_chem_enzy_runtime(
    *,
    vendor_root: Path | str,
    one_step_models: list[str] | None = None,
    stock_names: list[str] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe imports and configured files without loading a planner/model."""

    started = time.monotonic()
    os.environ.setdefault("CHEMENZY_PANDARALLEL_WORKERS", "1")
    root = Path(vendor_root).expanduser().resolve()
    requested_models = list(
        one_step_models
        if one_step_models is not None
        else DEFAULT_ONE_STEP_MODELS
    )
    requested_stocks = list(
        stock_names if stock_names is not None else _PROBED_STOCKS
    )
    overrides = dict(model_overrides or {})
    probe_request = {
        "one_step_models": requested_models,
        "stock_names": requested_stocks,
        "model_overrides": overrides,
    }
    report: dict[str, Any] = {
        "schema_version": CAPABILITY_PROBE_SCHEMA,
        "accepted": False,
        "status": "runtime_unavailable",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "vendor_root": str(root),
        "package_versions": _package_versions(),
        "requested_one_step_models": requested_models,
        "requested_stock_names": requested_stocks,
        "model_override_digest": _stable_digest(overrides),
        "probe_request_digest": _stable_digest(probe_request),
        "selected_one_step_models": [],
        "one_step_model_availability": {},
        "model_path_checks": [],
        "stock_path_checks": [],
        "vendor_imports": {},
        "issues": [],
        "semantics": {
            "planner_constructed": False,
            "checkpoint_deserialized": False,
            "model_loaded": False,
            "search_executed": False,
        },
    }
    try:
        selection_issues = _request_selection_issues(
            requested_models,
            requested_stocks,
            overrides,
        )
        if selection_issues:
            report["issues"].extend(selection_issues)
            return _finish(report, started)
        adapter = ChemEnzyBackendAdapter(vendor_root=root)
        adapter_failures = adapter.preflight()
        if adapter_failures:
            report["issues"] = [
                str(item.category or "chem_enzy_adapter_preflight_failed")
                for item in adapter_failures
            ]
            return _finish(report, started)

        route_config = RouteSearchConfig(
            target_smiles="CCO",
            stock_names=requested_stocks,
            one_step_models=requested_models,
            max_iterations=1,
            max_depth=1,
            expansion_topk=1,
            search_flags={
                "gpu": -1,
                "use_depth_value_fn": False,
                **overrides,
            },
        )
        try:
            vendor_config = adapter._vendor_config(route_config)
        except ValueError as exc:
            message = str(exc)
            report["issues"].append(
                "chem_enzy_selected_stock_config_missing"
                if "selected stock names not found" in message
                else "chem_enzy_vendor_config_selection_invalid"
            )
            report["failure"] = {
                "type": type(exc).__name__,
                "message": message[:1000],
            }
            return _finish(report, started)
        selected, availability = _prune_unavailable_one_step_models(
            requested_models,
            vendor_config=vendor_config,
            vendor_root=adapter.vendor_root,
        )
        report["selected_one_step_models"] = selected
        report["one_step_model_availability"] = availability or {
            "schema_version": "chem_enzy_one_step_model_availability.v1",
            "selected_before": requested_models,
            "selected_after": selected,
            "unavailable": [],
            "action": "all_selected_models_available",
        }
        if bool(report["one_step_model_availability"].get("configuration_error")):
            report["issues"].append(
                "chem_enzy_selected_model_configuration_invalid"
            )
            return _finish(report, started)
        if not selected:
            report["issues"].append("all_selected_one_step_models_unavailable")
            return _finish(report, started)

        materialized = _materialize_selected_one_step_io_paths(
            vendor_config,
            selected,
            vendor_root=adapter.vendor_root,
        )
        report["model_path_checks"] = _selected_model_path_checks(
            materialized,
            selected,
        )
        report["stock_path_checks"] = _stock_path_checks(
            materialized,
            vendor_root=adapter.vendor_root,
            selected_stocks=requested_stocks,
        )
        if not report["model_path_checks"]:
            report["issues"].append("chem_enzy_selected_model_path_checks_empty")
            return _finish(report, started)
        if len(report["stock_path_checks"]) != len(requested_stocks):
            report["issues"].append("chem_enzy_selected_stock_path_checks_incomplete")
            return _finish(report, started)
        if not all(
            row.get("readable") is True
            for row in [
                *report["model_path_checks"],
                *report["stock_path_checks"],
            ]
        ):
            report["issues"].append("chem_enzy_configured_runtime_path_unreadable")
            return _finish(report, started)

        import_started = time.monotonic()
        with _vendor_pythonpath(adapter.vendor_root):
            _patch_numpy_legacy_aliases()
            _patch_torchdata_legacy_aliases()
            _patch_torchtext_legacy_aliases()
            _patch_dgl_graphbolt_optional_import()
            _patch_optional_easifa_import(False)
            _patch_optional_graphviz_import(False)
            api = importlib.import_module("retro_planner.api")
            mol_tree = importlib.import_module(
                "retro_planner.search_frame.mcts_star.mol_tree"
            )
        report["vendor_imports"] = {
            "retro_planner.api": bool(hasattr(api, "RSPlanner")),
            "retro_planner.search_frame.mcts_star.mol_tree": bool(
                hasattr(mol_tree, "MolTree")
            ),
            "elapsed_s": round(time.monotonic() - import_started, 3),
        }
        if not all(
            report["vendor_imports"].get(name) is True
            for name in (
                "retro_planner.api",
                "retro_planner.search_frame.mcts_star.mol_tree",
            )
        ):
            report["issues"].append("chem_enzy_vendor_symbols_missing")
            return _finish(report, started)
        report["accepted"] = True
        report["status"] = "ready"
    except Exception as exc:  # capability boundary must fail closed
        report["issues"].append("chem_enzy_capability_probe_failed")
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
    return _finish(report, started)


def _selected_model_path_checks(
    config: dict[str, Any],
    selected: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_configs = config.get("one_step_model_configs") or {}
    for full_name in selected:
        model_type, model_subname = full_name.split(".", 1)
        model_config = (all_configs.get(model_type) or {}).get(model_subname) or {}
        values: list[tuple[str, Any]] = []
        if model_type == "onmt_models":
            values.extend(
                ("model_path", item)
                for item in model_config.get("model_path") or []
            )
        elif model_type == "graphfp_models":
            values.extend(
                (key, model_config.get(key))
                for key in ("graph_model_dumb", "graph_dataset_root")
            )
        elif model_type == "mlp_models":
            values.extend(
                (key, model_config.get(key))
                for key in ("mlp_templates", "mlp_model_dump")
            )
        elif model_type == "template_relevance":
            values.append(("archive_path", model_config.get("archive_path")))
        for field, raw_path in values:
            path = Path(str(raw_path or ""))
            rows.append(
                {
                    "model": full_name,
                    "field": field,
                    "path": str(_normal_absolute_path(path)),
                    "windows_extended_io": str(path).startswith("\\\\?\\"),
                    **_readability(path),
                }
            )
    return rows


def _stock_path_checks(
    config: dict[str, Any],
    *,
    vendor_root: Path,
    selected_stocks: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stock_config = config.get("stocks") or {}
    for name in selected_stocks:
        raw_path = stock_config.get(name)
        if not raw_path:
            rows.append(
                {
                    "stock": str(name),
                    "path": "",
                    "windows_extended_io": False,
                    "exists": False,
                    "kind": "missing_config",
                    "readable": False,
                }
            )
            continue
        path = _resolve_vendor_model_path(raw_path, vendor_root=vendor_root)
        rows.append(
            {
                "stock": str(name),
                "path": str(_normal_absolute_path(path)),
                "windows_extended_io": str(path).startswith("\\\\?\\"),
                **_readability(path),
            }
        )
    return rows


def _readability(path: Path) -> dict[str, Any]:
    io_path = _windows_extended_path(path)
    if io_path.is_dir():
        try:
            next(io_path.iterdir(), None)
        except OSError:
            return {"exists": True, "kind": "directory", "readable": False}
        return {"exists": True, "kind": "directory", "readable": True}
    if not io_path.is_file():
        return {"exists": False, "kind": "missing", "readable": False}
    try:
        with io_path.open("rb") as handle:
            header = handle.read(4)
        size = io_path.stat().st_size
    except OSError:
        return {"exists": True, "kind": "file", "readable": False}
    return {
        "exists": True,
        "kind": "file",
        "readable": bool(header and size > 0),
        "size_bytes": int(size),
        "header_hex": header.hex(),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def _request_selection_issues(
    one_step_models: Any,
    stock_names: Any,
    model_overrides: Any,
) -> list[str]:
    issues: list[str] = []
    for field, values in (
        ("one_step_models", one_step_models),
        ("stock_names", stock_names),
    ):
        if not isinstance(values, list) or not values:
            issues.append(f"chem_enzy_probe_request_{field}_invalid")
            continue
        if any(not isinstance(item, str) or not item.strip() for item in values):
            issues.append(f"chem_enzy_probe_request_{field}_invalid")
    if not isinstance(model_overrides, dict):
        issues.append("chem_enzy_probe_request_model_overrides_invalid")
    else:
        unknown = sorted(set(model_overrides) - _MODEL_OVERRIDE_KEYS)
        if unknown:
            issues.append(
                "chem_enzy_probe_request_model_override_unknown:"
                + ",".join(unknown)
            )
        path_override = model_overrides.get(
            "chem_enzy_onmt_model_path",
            model_overrides.get("onmt_model_path"),
        )
        if path_override is not None and not (
            isinstance(path_override, str)
            and path_override.strip()
            or isinstance(path_override, list)
            and bool(path_override)
            and all(isinstance(item, str) and item.strip() for item in path_override)
        ):
            issues.append("chem_enzy_probe_request_onmt_model_path_invalid")
        tokenizer = model_overrides.get(
            "chem_enzy_onmt_tokenizer",
            model_overrides.get("onmt_tokenizer"),
        )
        if tokenizer is not None and str(tokenizer).strip().lower() not in {
            "char",
            "token",
        }:
            issues.append("chem_enzy_probe_request_onmt_tokenizer_invalid")
    return issues


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finish(report: dict[str, Any], started: float) -> dict[str, Any]:
    report["issues"] = sorted(set(str(item) for item in report["issues"] if item))
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    if report["issues"]:
        report["accepted"] = False
        report["status"] = "runtime_unavailable"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    try:
        if request_path.stat().st_size > 1_000_000:
            raise ValueError("capability probe request exceeds 1 MB")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("capability probe request must be an object")
        report = probe_chem_enzy_runtime(
            vendor_root=args.vendor_root,
            one_step_models=request.get("one_step_models"),
            stock_names=request.get("stock_names"),
            model_overrides=request.get("model_overrides"),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": CAPABILITY_PROBE_SCHEMA,
            "accepted": False,
            "status": "runtime_unavailable",
            "issues": ["chem_enzy_capability_probe_request_invalid"],
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            },
            "semantics": {
                "planner_constructed": False,
                "checkpoint_deserialized": False,
                "model_loaded": False,
                "search_executed": False,
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report.get("accepted") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
