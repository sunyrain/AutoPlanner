"""Thin adapter for the external ChemEnzyRetroPlanner baseline.

The adapter intentionally keeps ChemEnzyRetroPlanner as an optional vendor
checkout. It does not import the vendor package until a real search is
requested, so AutoPlanner tests and benchmark assembly do not require the heavy
ChemEnzy conda environment.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from cascade_planner.baselines.route_contract import (
    BackendFailure,
    BaselineRunResult,
    RouteCandidate,
    RouteSearchConfig,
    RouteStepCandidate,
)


BACKEND_NAME = "ChemEnzyRetroPlanner"
DEFAULT_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")
DEFAULT_CONFIG_RELATIVE = Path("retro_planner/config/config.yaml")
DEFAULT_STOCKS = ["Zinc_Fix-stock"]
DEFAULT_ONE_STEP_MODELS = [
    "graphfp_models.USPTO-full_remapped",
    "onmt_models.bionav_one_step",
]
DEFAULT_ONMT_MODEL_NAME = "onmt_models.bionav_one_step"
CHEMENZY_ONMT_MODEL_PATH_ENV = "AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH"
_RUNTIME_SEARCH_FLAGS = {
    # Row-derived cascade state changes how a target is searched, but it does
    # not require rebuilding ChemEnzy's stock/model/value-function machinery.
    "cascade_search_context",
}


@dataclass
class ChemEnzyBackendAdapter:
    """Run ChemEnzyRetroPlanner core search and normalize its route output."""

    vendor_root: Path | str = DEFAULT_VENDOR_ROOT
    config_path: Path | str | None = None
    gpu: int = -1
    enable_condition_prediction: bool = False
    enable_enzyme_assignment: bool = False
    enable_easifa: bool = False
    extra_config: dict[str, Any] = field(default_factory=dict)
    onmt_model_path: Path | str | Iterable[Path | str] | None = None

    def __post_init__(self) -> None:
        self.vendor_root = Path(self.vendor_root)
        if self.config_path is None:
            self.config_path = self.vendor_root / DEFAULT_CONFIG_RELATIVE
        else:
            self.config_path = Path(self.config_path)

    def preflight(self) -> list[BackendFailure]:
        """Return setup failures that would prevent a real backend run."""
        failures: list[BackendFailure] = []
        if not self.vendor_root.exists():
            failures.append(
                BackendFailure(
                    category="vendor_missing",
                    message=f"ChemEnzyRetroPlanner checkout not found at {self.vendor_root}",
                    retryable=True,
                    raw_backend_metadata={"vendor_root": str(self.vendor_root)},
                )
            )
        if not self.config_path.exists():
            failures.append(
                BackendFailure(
                    category="config_missing",
                    message=f"ChemEnzyRetroPlanner config not found at {self.config_path}",
                    retryable=True,
                    raw_backend_metadata={"config_path": str(self.config_path)},
                )
            )
        return failures

    def run_target(self, config: RouteSearchConfig, *, dry_run: bool = False) -> BaselineRunResult:
        """Run one target, returning structured failures instead of raising."""
        failures = self.preflight()
        if dry_run:
            metadata = {
                "dry_run": True,
                "vendor_root": str(self.vendor_root),
                "config_path": str(self.config_path),
                "search_config": config.to_dict(),
            }
            return BaselineRunResult(
                target_smiles=config.target_smiles,
                backend=BACKEND_NAME,
                failures=_failures_for_target(failures, config.target_smiles),
                raw_backend_metadata=metadata,
            )
        if failures:
            return BaselineRunResult(
                target_smiles=config.target_smiles,
                backend=BACKEND_NAME,
                failures=_failures_for_target(failures, config.target_smiles),
            )

        try:
            planner = self._build_planner(config)
        except Exception as exc:  # pragma: no cover - depends on optional vendor env
            return BaselineRunResult(
                target_smiles=config.target_smiles,
                backend=BACKEND_NAME,
                failures=[
                    BackendFailure(
                        category="backend_initialization_failed",
                        message=f"{type(exc).__name__}: {exc}",
                        target_smiles=config.target_smiles,
                        retryable=True,
                        raw_backend_metadata={"vendor_root": str(self.vendor_root)},
                    )
                ],
            )

        return self._run_with_planner(planner, config)

    def run_targets(
        self,
        configs: Iterable[RouteSearchConfig],
        *,
        dry_run: bool = False,
        reuse_planner: bool = True,
    ) -> list[BaselineRunResult]:
        """Run many targets, reusing one initialized ChemEnzy planner per shared config."""
        config_list = list(configs)
        if not config_list:
            return []
        if dry_run or not reuse_planner:
            return [self.run_target(config, dry_run=dry_run) for config in config_list]

        failures = self.preflight()
        if failures:
            return [
                BaselineRunResult(
                    target_smiles=config.target_smiles,
                    backend=BACKEND_NAME,
                    failures=_failures_for_target(failures, config.target_smiles),
                )
                for config in config_list
            ]

        grouped: dict[str, list[tuple[int, RouteSearchConfig]]] = {}
        for idx, config in enumerate(config_list):
            grouped.setdefault(_planner_signature(config), []).append((idx, config))

        results: list[BaselineRunResult | None] = [None] * len(config_list)
        for group in grouped.values():
            first_config = group[0][1]
            try:
                planner = self._build_planner(first_config)
            except Exception as exc:  # pragma: no cover - depends on optional vendor env
                for idx, config in group:
                    results[idx] = BaselineRunResult(
                        target_smiles=config.target_smiles,
                        backend=BACKEND_NAME,
                        failures=[
                            BackendFailure(
                                category="backend_initialization_failed",
                                message=f"{type(exc).__name__}: {exc}",
                                target_smiles=config.target_smiles,
                                retryable=True,
                                raw_backend_metadata={"vendor_root": str(self.vendor_root)},
                            )
                        ],
                    )
                continue
            for idx, config in group:
                results[idx] = self._run_with_planner(planner, config)

        return [result for result in results if result is not None]

    def _run_with_planner(self, planner: Any, config: RouteSearchConfig) -> BaselineRunResult:
        annotation_failures: list[BackendFailure] = []
        annotation_metadata: dict[str, Any] = {}
        started = time.monotonic()
        try:
            _apply_runtime_search_flags(planner, config)
            raw_result = planner.plan(config.target_smiles)
        except Exception as exc:  # pragma: no cover - depends on optional vendor env
            return BaselineRunResult(
                target_smiles=config.target_smiles,
                backend=BACKEND_NAME,
                failures=[
                    BackendFailure(
                        category="backend_search_failed",
                        message=f"{type(exc).__name__}: {exc}",
                        target_smiles=config.target_smiles,
                        retryable=True,
                    )
                ],
                raw_backend_metadata={"elapsed_s": round(time.monotonic() - started, 3)},
            )

        elapsed_s = time.monotonic() - started
        if not raw_result:
            return BaselineRunResult(
                target_smiles=config.target_smiles,
                backend=BACKEND_NAME,
                failures=[
                    BackendFailure(
                        category="no_route_found",
                        message="ChemEnzyRetroPlanner returned no successful routes",
                        target_smiles=config.target_smiles,
                        retryable=True,
                    )
                ],
                raw_backend_metadata={"elapsed_s": round(elapsed_s, 3)},
            )

        if self._attributes_enabled():
            annotation_started = time.monotonic()
            try:
                rxn_attributes = planner.predict_rxn_attributes()
                annotation_metadata = _rxn_attribute_summary(rxn_attributes)
            except Exception as exc:  # pragma: no cover - depends on optional vendor env
                annotation_failures.append(
                    BackendFailure(
                        category="backend_annotation_failed",
                        message=f"{type(exc).__name__}: {exc}",
                        target_smiles=config.target_smiles,
                        retryable=True,
                    )
                )
            annotation_metadata["elapsed_s"] = round(time.monotonic() - annotation_started, 3)

        routes = route_candidates_from_chem_enzy_result(raw_result, target_smiles=config.target_smiles)
        for route in routes:
            route.search_time_s = elapsed_s
        expansion_trace = raw_result.get("cascade_expansion_trace") or []
        trace_preview_limit = int(config.search_flags.get("cascade_expansion_trace_preview", 20))
        trace_metadata = {
            "count": len(expansion_trace),
            "preview": expansion_trace[:trace_preview_limit],
        }
        if config.search_flags.get("include_cascade_expansion_trace"):
            trace_metadata["rows"] = expansion_trace
        return BaselineRunResult(
            target_smiles=config.target_smiles,
            backend=BACKEND_NAME,
            routes=routes,
            failures=annotation_failures,
            raw_backend_metadata={
                "elapsed_s": round(elapsed_s, 3),
                "total_elapsed_s": round(time.monotonic() - started, 3),
                "iter": raw_result.get("iter"),
                "first_succ_time": _finite_or_none(raw_result.get("first_succ_time")),
                "rxn_annotation": annotation_metadata,
                "cascade_expansion_trace": trace_metadata,
            },
        )

    def _build_planner(self, search_config: RouteSearchConfig) -> Any:
        vendor_config = self._vendor_config(search_config)
        with _vendor_pythonpath(self.vendor_root):
            _patch_numpy_legacy_aliases()
            _patch_torchdata_legacy_aliases()
            _patch_dgl_graphbolt_optional_import()
            _patch_optional_easifa_import(self.enable_easifa)
            _patch_optional_graphviz_import(bool(search_config.search_flags.get("viz", False)))
            api = importlib.import_module("retro_planner.api")
            planner = api.RSPlanner(vendor_config)
            planner.select_stocks(search_config.stock_names or DEFAULT_STOCKS)
            planner.select_one_step_model(search_config.one_step_models or DEFAULT_ONE_STEP_MODELS)
            if self.enable_condition_prediction:
                planner.select_condition_predictor(str(search_config.search_flags.get("condition_model", "rcr")))
            planner.prepare_plan(
                prepare_easifa=self.enable_easifa,
                prepare_condition_predictor=self.enable_condition_prediction,
                prepare_enzyme_recommander=self.enable_enzyme_assignment,
            )
            return planner

    def _attributes_enabled(self) -> bool:
        return bool(self.enable_condition_prediction or self.enable_enzyme_assignment)

    def _vendor_config(self, search_config: RouteSearchConfig) -> dict[str, Any]:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        selected_stocks = search_config.stock_names or DEFAULT_STOCKS
        config["stocks"] = {
            name: path
            for name, path in (config.get("stocks") or {}).items()
            if name in set(selected_stocks)
        }
        if not config["stocks"]:
            raise ValueError(f"selected stock names not found in ChemEnzy config: {selected_stocks}")
        config["gpu"] = int(search_config.search_flags.get("gpu", self.gpu))
        config["iterations"] = int(search_config.max_iterations)
        config["max_depth"] = int(search_config.max_depth)
        config["expansion_topk"] = int(search_config.expansion_topk)
        config["pred_condition"] = bool(self.enable_condition_prediction)
        config["enzyme_assign"] = bool(self.enable_enzyme_assignment)
        config["organic_enzyme_rxn_classification"] = bool(self.enable_enzyme_assignment)
        config["viz"] = bool(search_config.search_flags.get("viz", False))
        config["keep_search"] = bool(search_config.search_flags.get("keep_search", True))
        config["use_filter"] = bool(search_config.search_flags.get("use_filter", config.get("use_filter", False)))
        config["stock_limit_dict"] = search_config.search_flags.get("stock_limit_dict")
        config["use_depth_value_fn"] = bool(
            search_config.search_flags.get("use_depth_value_fn", config.get("use_depth_value_fn", False))
        )
        if "cascade_search_context" in search_config.search_flags:
            config["cascade_search_context"] = dict(search_config.search_flags["cascade_search_context"] or {})
        if "cascade_cost_model" in search_config.search_flags:
            config["cascade_cost_model"] = dict(search_config.search_flags["cascade_cost_model"] or {})
            _normalize_cost_model_paths(config["cascade_cost_model"])
        if "cascade_source_policy" in search_config.search_flags:
            config["cascade_source_policy"] = dict(search_config.search_flags["cascade_source_policy"] or {})
            _normalize_source_policy_paths(config["cascade_source_policy"])
        if "use_cascade_cost_model" in search_config.search_flags:
            config["use_cascade_cost_model"] = bool(search_config.search_flags["use_cascade_cost_model"])
        elif (config.get("cascade_cost_model") or {}).get("enabled"):
            config["use_cascade_cost_model"] = True
        if "use_cascade_source_policy" in search_config.search_flags:
            config["use_cascade_source_policy"] = bool(search_config.search_flags["use_cascade_source_policy"])
        elif (config.get("cascade_source_policy") or {}).get("enabled"):
            config["use_cascade_source_policy"] = True
        config.update(self.extra_config)
        onmt_model_path = (
            search_config.search_flags.get("chem_enzy_onmt_model_path")
            or search_config.search_flags.get("onmt_model_path")
            or self.onmt_model_path
            or os.environ.get(CHEMENZY_ONMT_MODEL_PATH_ENV)
        )
        if onmt_model_path:
            apply_onmt_model_path_override(config, onmt_model_path)
        return config


def apply_onmt_model_path_override(
    config: dict[str, Any],
    model_path: Path | str | Iterable[Path | str],
    *,
    model_name: str = DEFAULT_ONMT_MODEL_NAME,
) -> dict[str, Any]:
    """Point a configured ChemEnzy ONMT one-step model at trained checkpoint(s)."""
    model_paths = _as_model_path_list(model_path)
    if not model_paths:
        return config
    try:
        model_type, model_subname = model_name.split(".", 1)
    except ValueError as exc:
        raise ValueError(f"invalid ChemEnzy one-step model name: {model_name}") from exc
    one_step_configs = config.setdefault("one_step_model_configs", {})
    if model_type not in one_step_configs or model_subname not in (one_step_configs.get(model_type) or {}):
        raise ValueError(f"cannot override unknown ChemEnzy ONMT model config: {model_name}")
    model_config = dict(one_step_configs[model_type][model_subname] or {})
    model_config["model_path"] = model_paths
    one_step_configs[model_type][model_subname] = model_config
    return config


def _as_model_path_list(model_path: Path | str | Iterable[Path | str]) -> list[str]:
    if isinstance(model_path, (str, os.PathLike)):
        raw = str(model_path)
        return [_absolute_checkpoint_path(token.strip()) for token in raw.split(",") if token.strip()]
    return [_absolute_checkpoint_path(str(item).strip()) for item in model_path if str(item).strip()]


def _absolute_checkpoint_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return str(path)


def route_candidates_from_chem_enzy_result(raw_result: dict[str, Any], *, target_smiles: str) -> list[RouteCandidate]:
    """Convert ChemEnzy `RSPlanner.plan` output to `RouteCandidate` objects."""
    dict_routes = raw_result.get("all_succ_dict_routes") or []
    if not dict_routes and raw_result.get("dict_routes"):
        dict_routes = [raw_result["dict_routes"]]
    routes: list[RouteCandidate] = []
    for idx, dict_route in enumerate(dict_routes):
        steps = _flatten_chem_enzy_dict_route(dict_route)
        score = _route_score_from_steps(steps)
        routes.append(
            RouteCandidate(
                target_smiles=target_smiles or str((dict_route or {}).get("smiles") or ""),
                steps=steps,
                backend=BACKEND_NAME,
                score=score,
                solved=True,
                route_rank=idx,
                search_time_s=_finite_or_none(raw_result.get("time")),
                raw_backend_metadata={
                    "route_index": idx,
                    "route_lens": raw_result.get("route_lens"),
                    "iter": raw_result.get("iter"),
                },
            )
        )
    return routes


def _flatten_chem_enzy_dict_route(route: dict[str, Any]) -> list[RouteStepCandidate]:
    steps: list[RouteStepCandidate] = []

    def walk(mol_node: dict[str, Any]) -> None:
        children = mol_node.get("children") or []
        for reaction_node in children:
            if not isinstance(reaction_node, dict) or reaction_node.get("type") != "reaction":
                continue
            reactant_nodes = [node for node in reaction_node.get("children") or [] if isinstance(node, dict)]
            reactants = [str(node.get("smiles") or "") for node in reactant_nodes if node.get("smiles")]
            product = str(mol_node.get("smiles") or "")
            rxn_smiles = str(reaction_node.get("rxn_smiles") or _reaction_smiles(reactants, product))
            attrs = reaction_node.get("rxn_attribute") or {}
            steps.append(
                RouteStepCandidate(
                    product_smiles=product,
                    reactant_smiles=reactants,
                    rxn_smiles=rxn_smiles,
                    source_model=_source_model(reaction_node),
                    score=_step_score(reaction_node),
                    stock_status={str(node.get("smiles") or ""): node.get("in_stock") for node in reactant_nodes},
                    condition_predictions=_condition_predictions(attrs),
                    enzyme_ec_annotations=_enzyme_annotations(attrs),
                    raw_backend_metadata={
                        "template": reaction_node.get("template"),
                        "cost": reaction_node.get("cost"),
                        "cascade_cost": reaction_node.get("cascade_cost"),
                        "rxn_attribute": attrs,
                    },
                )
            )
            for reactant_node in reactant_nodes:
                walk(reactant_node)

    if isinstance(route, dict):
        walk(route)
    return steps


def _source_model(reaction_node: dict[str, Any]) -> str:
    template = reaction_node.get("template")
    if isinstance(template, dict):
        return str(template.get("model_full_name") or template.get("model_name") or template.get("source") or "")
    if template:
        return str(template)
    return BACKEND_NAME


def _step_score(reaction_node: dict[str, Any]) -> float | None:
    for key in ("score", "confidence", "probability"):
        value = reaction_node.get(key)
        if value is not None:
            return _float_or_none(value)
    cost = reaction_node.get("cost")
    if cost is not None:
        value = _float_or_none(cost)
        if value is not None:
            return math.exp(-value)
    return None


def _condition_predictions(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    condition = attrs.get("condition") if isinstance(attrs, dict) else None
    return _records_from_backend_table(condition)


def _enzyme_annotations(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    enzymatic = _is_enzymatic_reaction(attrs)
    if enzymatic is False:
        return []
    enzyme = attrs.get("enzyme_assign") if isinstance(attrs, dict) else None
    records = _records_from_backend_table(enzyme)
    out = []
    for record in records:
        out.append(
            {
                "rank": record.get("Ranks") or record.get("rank"),
                "ec_number": record.get("EC Number") or record.get("ec_number"),
                "confidence": record.get("Confidence") or record.get("confidence"),
                "raw": record,
            }
        )
    return out


def _is_enzymatic_reaction(attrs: dict[str, Any]) -> bool | None:
    if not isinstance(attrs, dict):
        return None
    rows = _records_from_backend_table(attrs.get("organic_enzyme_rxn_classification"))
    if not rows:
        return None
    for row in rows:
        name = str(
            row.get("Reaction Type")
            or row.get("reaction_type")
            or row.get("type")
            or row.get("class")
            or ""
        ).lower()
        if "enzymatic" in name:
            return True
    return False


def _records_from_backend_table(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return _records_from_backend_table(json.loads(text))
        except json.JSONDecodeError:
            return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        columns = value.get("columns")
        data = value.get("data")
        if isinstance(columns, list) and isinstance(data, list):
            rows = []
            for row in data:
                if isinstance(row, list):
                    rows.append({str(columns[idx]): row[idx] for idx in range(min(len(columns), len(row)))})
            return rows
        if "data" in value and isinstance(value["data"], list):
            return [item for item in value["data"] if isinstance(item, dict)]
        column_rows = _records_from_column_oriented_table(value)
        if column_rows:
            return column_rows
        return [value]
    return []


def _records_from_column_oriented_table(value: dict[str, Any]) -> list[dict[str, Any]]:
    if not value or not all(isinstance(col_values, dict) for col_values in value.values()):
        return []
    row_keys = set()
    for col_values in value.values():
        row_keys.update(str(key) for key in col_values)

    def sort_key(item: str) -> tuple[int, str]:
        try:
            return (int(item), item)
        except ValueError:
            return (10**9, item)

    rows = []
    for row_key in sorted(row_keys, key=sort_key):
        row = {}
        for col_name, col_values in value.items():
            row[str(col_name)] = col_values.get(row_key)
        rows.append(row)
    return rows


def _route_score_from_steps(steps: Iterable[RouteStepCandidate]) -> float | None:
    values = [step.score for step in steps if step.score is not None]
    if not values:
        return None
    score = 1.0
    for value in values:
        score *= float(value)
    return score


def _reaction_smiles(reactants: list[str], product: str) -> str:
    lhs = ".".join(reactant for reactant in reactants if reactant)
    return f"{lhs}>>{product}"


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_or_none(value: Any) -> float | None:
    return _float_or_none(value)


def _planner_signature(config: RouteSearchConfig) -> str:
    search_flags = {
        key: value
        for key, value in dict(config.search_flags or {}).items()
        if key not in _RUNTIME_SEARCH_FLAGS
    }
    payload = {
        "stock_names": list(config.stock_names or DEFAULT_STOCKS),
        "max_iterations": int(config.max_iterations),
        "max_depth": int(config.max_depth),
        "expansion_topk": int(config.expansion_topk),
        "one_step_models": list(config.one_step_models or DEFAULT_ONE_STEP_MODELS),
        "search_flags": search_flags,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _apply_runtime_search_flags(planner: Any, config: RouteSearchConfig) -> None:
    if "cascade_search_context" not in dict(config.search_flags or {}):
        return
    next_context = dict(config.search_flags.get("cascade_search_context") or {})
    current_context = getattr(planner, "cascade_search_context", None)
    if isinstance(current_context, dict):
        current_context.clear()
        current_context.update(next_context)
        context_ref = current_context
    else:
        setattr(planner, "cascade_search_context", next_context)
        context_ref = next_context
    planner_config = getattr(planner, "config", None)
    if isinstance(planner_config, dict):
        planner_config["cascade_search_context"] = context_ref


def _patch_dgl_graphbolt_optional_import() -> None:
    """Let legacy ChemEnzy imports use DGL even when GraphBolt is unavailable.

    DGL 2.x imports ``dgl.graphbolt`` eagerly. Some Torch/DGL wheel combinations
    omit the matching GraphBolt shared object, but ChemEnzy's graph retrosynthesis
    code does not use GraphBolt. Pre-seeding this optional submodule keeps DGL's
    core graph APIs importable without modifying site-packages.
    """
    if os.environ.get("CHEMENZY_ALLOW_REAL_GRAPHBOLT") == "1":
        return
    sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))


def _patch_optional_easifa_import(enable_easifa: bool) -> None:
    """Provide an import-time EASIFA shim for core-search-only ChemEnzy runs."""
    if enable_easifa or os.environ.get("CHEMENZY_REQUIRE_EASIFA") == "1":
        return
    if "easifa.interface.utils" in sys.modules:
        return

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("EASIFA is not available in this ChemEnzy core-search runtime")

    easifa_mod = sys.modules.setdefault("easifa", types.ModuleType("easifa"))
    interface_mod = sys.modules.setdefault("easifa.interface", types.ModuleType("easifa.interface"))
    utils_mod = types.ModuleType("easifa.interface.utils")
    utils_mod.EasIFAInferenceAPI = unavailable
    utils_mod.UniProtParserEC = unavailable
    utils_mod.full_swissprot_checkpoint_path = ""
    utils_mod.get_structure_html_and_active_data = unavailable
    utils_mod.uniprot_csv_path = ""
    utils_mod.pdb_cache_path = ""
    utils_mod.chebi_path = ""
    utils_mod.uniprot_rxn_path = ""
    utils_mod.uniprot_json_path = ""
    setattr(easifa_mod, "interface", interface_mod)
    setattr(interface_mod, "utils", utils_mod)
    sys.modules["easifa.interface.utils"] = utils_mod


def _patch_optional_graphviz_import(enable_viz: bool) -> None:
    """Provide a no-op Graphviz shim when route rendering is disabled."""
    if enable_viz or "graphviz" in sys.modules:
        return

    class _NoOpDigraph:
        source = ""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def attr(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def node(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def edge(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def render(self, *_args: Any, **_kwargs: Any) -> str:
            return ""

    graphviz_mod = types.ModuleType("graphviz")
    graphviz_mod.Digraph = _NoOpDigraph
    sys.modules["graphviz"] = graphviz_mod


def _normalize_source_policy_paths(policy_config: dict[str, Any]) -> None:
    model_path = policy_config.get("source_value_model_path")
    if not model_path:
        return
    path = Path(str(model_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    policy_config["source_value_model_path"] = str(path.resolve())


def _normalize_cost_model_paths(cost_config: dict[str, Any]) -> None:
    model_path = cost_config.get("action_value_model_path")
    if not model_path:
        return
    path = Path(str(model_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    cost_config["action_value_model_path"] = str(path.resolve())


def _failures_for_target(failures: list[BackendFailure], target_smiles: str) -> list[BackendFailure]:
    return [replace(failure, target_smiles=failure.target_smiles or target_smiles) for failure in failures]


def _rxn_attribute_summary(value: Any) -> dict[str, Any]:
    routes = []
    for route_attrs in value or []:
        if not hasattr(route_attrs, "items"):
            continue
        route = []
        for rxn_smiles, attrs in route_attrs.items():
            route.append({
                "rxn_smiles": str(rxn_smiles),
                "attributes": sorted(str(key) for key in getattr(attrs, "keys", lambda: [])()),
            })
        routes.append(route)
    return {
        "n_routes": len(routes),
        "routes_preview": routes[:3],
    }


def _patch_numpy_legacy_aliases() -> None:
    """Patch old vendor dependencies that still reference removed NumPy aliases."""
    try:
        import numpy as np
    except Exception:
        return
    for name, value in {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            has_alias = hasattr(np, name)
        if not has_alias:
            setattr(np, name, value)


def _patch_torchdata_legacy_aliases() -> None:
    """Patch DGL/torchdata imports across newer torch builds.

    DGL 2.x with torchdata 0.7 imports ``DILL_AVAILABLE`` from torch's
    datapipe common module. Torch 2.3 exposes the same check as
    ``dill_available``. Keep this shim scoped to ChemEnzy startup rather than
    editing site-packages.
    """
    try:
        import torch.utils.data.datapipes.utils.common as common
    except Exception:
        return
    if hasattr(common, "DILL_AVAILABLE"):
        return
    dill_available = getattr(common, "dill_available", None)
    try:
        common.DILL_AVAILABLE = bool(dill_available()) if callable(dill_available) else bool(dill_available)
    except Exception:
        common.DILL_AVAILABLE = False


@contextmanager
def _vendor_pythonpath(vendor_root: Path):
    root = vendor_root.resolve()
    retro_root = root / "retro_planner"
    package_roots = [
        retro_root / "packages" / "mlp_retrosyn",
        retro_root / "packages" / "value_function",
        retro_root / "packages" / "rxn_filter",
        retro_root / "packages" / "onmt",
        retro_root / "packages" / "easifa",
        retro_root / "packages" / "graph_retrosyn",
        retro_root / "packages" / "condition_predictor",
        retro_root / "packages" / "organic_enzyme_rxn_classifier",
    ]
    additions = [str(path) for path in [root, retro_root, *package_roots] if path.exists()]
    old_path = list(sys.path)
    old_cwd = Path.cwd()
    try:
        for item in reversed(additions):
            if item not in sys.path:
                sys.path.insert(0, item)
        os.chdir(root)
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def write_baseline_results(results: list[BaselineRunResult], output_path: Path, *, metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "summary": summarize_baseline_results(results),
        "targets": [result.to_dict() for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_baseline_results(results: list[BaselineRunResult]) -> dict[str, Any]:
    n = len(results)
    solved = sum(1 for result in results if result.solved)
    route_counts = [result.route_count for result in results]
    enzymatic = 0
    target_elapsed = []
    solved_target_elapsed = []
    route_weighted_elapsed = []
    failures: dict[str, int] = {}
    for result in results:
        if any(route.enzymatic_step_present for route in result.routes):
            enzymatic += 1
        for failure in result.failures:
            failures[failure.category] = failures.get(failure.category, 0) + 1
        raw_elapsed = None
        if isinstance(result.raw_backend_metadata, dict):
            raw_elapsed = result.raw_backend_metadata.get("elapsed_s")
        if raw_elapsed is not None:
            try:
                raw_elapsed_f = float(raw_elapsed)
            except (TypeError, ValueError):
                raw_elapsed_f = None
            if raw_elapsed_f is not None:
                target_elapsed.append(raw_elapsed_f)
                if result.solved:
                    solved_target_elapsed.append(raw_elapsed_f)
        for route in result.routes:
            if route.search_time_s is not None:
                route_weighted_elapsed.append(float(route.search_time_s))
    return {
        "n_targets": n,
        "solved": solved,
        "solved_rate": solved / n if n else None,
        "total_routes": sum(route_counts),
        "avg_route_count": sum(route_counts) / n if n else None,
        "targets_with_enzymatic_step": enzymatic,
        "avg_search_time_s": sum(target_elapsed) / len(target_elapsed) if target_elapsed else None,
        "avg_solved_search_time_s": (
            sum(solved_target_elapsed) / len(solved_target_elapsed) if solved_target_elapsed else None
        ),
        "total_search_time_s": sum(target_elapsed) if target_elapsed else None,
        "route_weighted_avg_search_time_s": (
            sum(route_weighted_elapsed) / len(route_weighted_elapsed) if route_weighted_elapsed else None
        ),
        "failure_categories": failures,
    }
