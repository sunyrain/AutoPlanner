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
import traceback
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
from cascade_planner.baselines.template_relevance_runtime import missing_template_relevance_models
from cascade_planner.baselines.chem_enzy_native_enzyme_plugin import (
    NativeEnzymeOneStepWrapper,
    NativeEnzymePluginConfig,
    NativeEnzymePluginState,
    native_enzyme_plugin_config_from_flags,
    native_enzyme_plugin_stats,
    reset_native_enzyme_plugin_state,
)
from cascade_planner.baselines.chem_enzy_native_chemical_plugin import (
    NativeChemicalOneStepWrapper,
    NativeChemicalPluginConfig,
    NativeChemicalPluginState,
    native_chemical_plugin_config_from_flags,
    native_chemical_plugin_stats,
    reset_native_chemical_plugin_state,
)
from cascade_planner.baselines.literature_one_step_plugin import (
    LITERATURE_TEMPLATE_PLUGIN_SOURCE,
    LiteratureOneStepPluginConfig,
    LiteratureOneStepPluginState,
    LiteratureTemplateOneStepWrapper,
    PLUGIN_MODEL_FULL_NAME as LITERATURE_PLUGIN_MODEL_FULL_NAME,
    literature_plugin_config_from_flags,
    literature_plugin_stats,
    reset_literature_plugin_state,
)
from cascade_planner.agent.chem_enzy_policy import chem_enzy_policy_trace_from_search_flags


BACKEND_NAME = "ChemEnzyRetroPlanner"
DEFAULT_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")
DEFAULT_CONFIG_RELATIVE = Path("retro_planner/config/config.yaml")
DEFAULT_STOCKS = ["Zinc_Fix-stock"]
DEFAULT_ONE_STEP_MODELS = [
    "graphfp_models.USPTO-full_remapped",
    "onmt_models.bionav_one_step",
    "onmt_models.bionav_native_one_step",
]
DEFAULT_ONMT_MODEL_NAME = "onmt_models.bionav_one_step"
CHEMENZY_ONMT_MODEL_PATH_ENV = "AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH"
CHEMENZY_ONMT_TOKENIZER_ENV = "AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER"
CHEMENZY_ONMT_SOURCE_PREFIX_ENV = "AUTOPLANNER_CHEMENZY_ONMT_SOURCE_PREFIX"
CHEMENZY_ONMT_PRETOKENIZE_MODE_ENV = "AUTOPLANNER_CHEMENZY_ONMT_PRETOKENIZE_MODE"
CHEMENZY_STEP_STRENGTHENING_SCHEMA = "chem_enzy_step_strengthening.v1"
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
        config = chem_enzy_step_strengthened_config(config)
        failures = self.preflight()
        if dry_run:
            policy_trace = chem_enzy_policy_trace_from_search_flags(config.search_flags)
            metadata = {
                "dry_run": True,
                "vendor_root": str(self.vendor_root),
                "config_path": str(self.config_path),
                "search_config": config.to_dict(),
                **({"chem_enzy_policy_trace": policy_trace} if policy_trace is not None else {}),
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
        template_failures = _template_relevance_failures(
            config.one_step_models or DEFAULT_ONE_STEP_MODELS,
            self.vendor_root,
            target_smiles=config.target_smiles,
        )
        if template_failures:
            return BaselineRunResult(
                target_smiles=config.target_smiles,
                backend=BACKEND_NAME,
                failures=template_failures,
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
        config_list = [chem_enzy_step_strengthened_config(config) for config in configs]
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
        results: list[BaselineRunResult | None] = [None] * len(config_list)
        template_failure_by_signature: dict[str, list[BackendFailure]] = {}
        for idx, config in enumerate(config_list):
            signature = _planner_signature(config)
            if signature not in template_failure_by_signature:
                template_failure_by_signature[signature] = _template_relevance_failures(
                    config.one_step_models or DEFAULT_ONE_STEP_MODELS,
                    self.vendor_root,
                    target_smiles=config.target_smiles,
                )
            template_failures = template_failure_by_signature[signature]
            if template_failures:
                results[idx] = BaselineRunResult(
                    target_smiles=config.target_smiles,
                    backend=BACKEND_NAME,
                    failures=[
                        replace(failure, target_smiles=config.target_smiles)
                        for failure in template_failures
                    ],
                )
                continue
            grouped.setdefault(signature, []).append((idx, config))

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
        policy_trace = chem_enzy_policy_trace_from_search_flags(config.search_flags)
        try:
            _apply_runtime_search_flags(planner, config)
            reset_native_enzyme_plugin_state(planner, config.target_smiles)
            reset_native_chemical_plugin_state(planner, config.target_smiles)
            reset_literature_plugin_state(planner, config.target_smiles)
            raw_result = planner.plan(config.target_smiles)
        except Exception as exc:  # pragma: no cover - depends on optional vendor env
            enzyme_plugin_stats = native_enzyme_plugin_stats(planner)
            chemical_plugin_stats = native_chemical_plugin_stats(planner)
            literature_template_plugin_stats = literature_plugin_stats(planner)
            traceback_text = traceback.format_exc(limit=12)
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
                raw_backend_metadata={
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "exception_traceback": traceback_text,
                    **({"chem_enzy_policy_trace": policy_trace} if policy_trace is not None else {}),
                    **({"native_enzyme_plugin": enzyme_plugin_stats} if enzyme_plugin_stats is not None else {}),
                    **({"native_chemical_plugin": chemical_plugin_stats} if chemical_plugin_stats is not None else {}),
                    **({"literature_template_plugin": literature_template_plugin_stats} if literature_template_plugin_stats is not None else {}),
                },
            )

        elapsed_s = time.monotonic() - started
        if not raw_result:
            enzyme_plugin_stats = native_enzyme_plugin_stats(planner)
            chemical_plugin_stats = native_chemical_plugin_stats(planner)
            literature_template_plugin_stats = literature_plugin_stats(planner)
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
                raw_backend_metadata={
                    "elapsed_s": round(elapsed_s, 3),
                    **({"chem_enzy_policy_trace": policy_trace} if policy_trace is not None else {}),
                    **({"native_enzyme_plugin": enzyme_plugin_stats} if enzyme_plugin_stats is not None else {}),
                    **({"native_chemical_plugin": chemical_plugin_stats} if chemical_plugin_stats is not None else {}),
                    **({"literature_template_plugin": literature_template_plugin_stats} if literature_template_plugin_stats is not None else {}),
                },
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
        enzyme_plugin_stats = native_enzyme_plugin_stats(planner)
        chemical_plugin_stats = native_chemical_plugin_stats(planner)
        literature_template_plugin_stats = literature_plugin_stats(planner)
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
                **({"chem_enzy_policy_trace": policy_trace} if policy_trace is not None else {}),
                **({"native_enzyme_plugin": enzyme_plugin_stats} if enzyme_plugin_stats is not None else {}),
                **({"native_chemical_plugin": chemical_plugin_stats} if chemical_plugin_stats is not None else {}),
                **({"literature_template_plugin": literature_template_plugin_stats} if literature_template_plugin_stats is not None else {}),
            },
        )

    def _build_planner(self, search_config: RouteSearchConfig) -> Any:
        search_config = chem_enzy_step_strengthened_config(search_config)
        vendor_config = self._vendor_config(search_config)
        with _vendor_pythonpath(self.vendor_root):
            _patch_numpy_legacy_aliases()
            _patch_torchdata_legacy_aliases()
            _patch_dgl_graphbolt_optional_import()
            _patch_optional_easifa_import(self.enable_easifa)
            _patch_optional_graphviz_import(bool(search_config.search_flags.get("viz", False)))
            api = importlib.import_module("retro_planner.api")
            _patch_onmt_tokenizer(api, str(vendor_config.get("chem_enzy_onmt_tokenizer") or "char"))
            enzyme_plugin_config = native_enzyme_plugin_config_from_flags(search_config.search_flags)
            chemical_plugin_config = native_chemical_plugin_config_from_flags(search_config.search_flags)
            literature_plugin_config = literature_plugin_config_from_flags(search_config.search_flags)
            chemical_plugin_config = _chemical_plugin_config_with_base_model(
                chemical_plugin_config,
                search_config.one_step_models or DEFAULT_ONE_STEP_MODELS,
            )
            plugin_states = _configure_native_autoplanner_plugins(
                api,
                enzyme_config=enzyme_plugin_config,
                chemical_config=chemical_plugin_config,
                literature_config=literature_plugin_config,
            )
            enzyme_plugin_state, chemical_plugin_state, literature_plugin_state = plugin_states
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
            if enzyme_plugin_state is not None:
                planner._autoplanner_native_enzyme_plugin_state = enzyme_plugin_state
            if chemical_plugin_state is not None:
                planner._autoplanner_native_chemical_plugin_state = chemical_plugin_state
            if literature_plugin_state is not None:
                planner._autoplanner_literature_plugin_state = literature_plugin_state
            return planner

    def _attributes_enabled(self) -> bool:
        return bool(self.enable_condition_prediction or self.enable_enzyme_assignment)

    def _vendor_config(self, search_config: RouteSearchConfig) -> dict[str, Any]:
        search_config = chem_enzy_step_strengthened_config(search_config)
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
        if "chem_enzy_search_policy" in search_config.search_flags:
            config["chem_enzy_search_policy"] = dict(search_config.search_flags["chem_enzy_search_policy"] or {})
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
        onmt_tokenizer = (
            search_config.search_flags.get("chem_enzy_onmt_tokenizer")
            or search_config.search_flags.get("onmt_tokenizer")
            or os.environ.get(CHEMENZY_ONMT_TOKENIZER_ENV)
        )
        if onmt_tokenizer:
            apply_onmt_tokenizer_override(config, str(onmt_tokenizer))
        return config


def chem_enzy_step_strengthened_config(config: RouteSearchConfig) -> RouteSearchConfig:
    """Return a copy with the ChemEnzy enzyme-step strengthening preset applied."""
    flags = dict(config.search_flags or {})
    raw = flags.get("chem_enzy_step_strengthening", flags.get("strengthen_chem_enzy_steps"))
    if not _strengthening_enabled(raw):
        return config

    options = raw if isinstance(raw, dict) else {}
    flags["chem_enzy_step_strengthening_enabled"] = True
    flags["chem_enzy_step_strengthening_schema"] = CHEMENZY_STEP_STRENGTHENING_SCHEMA

    plugin = dict(flags.get("native_enzyme_plugin") or flags.get("autoplanner_native_enzyme_plugin") or {})
    _setdefault(plugin, "enabled", True)
    _setdefault(plugin, "top_k", int(options.get("top_k") or 8))
    _setdefault(plugin, "bridge_top_k", int(options.get("bridge_top_k") or 10))
    _setdefault(plugin, "max_ec_contexts", int(options.get("max_ec_contexts") or 3))
    _setdefault(plugin, "require_bridge", True)
    _setdefault(plugin, "require_verifier_pass", True)
    _setdefault(plugin, "enable_sp_v1", True)
    _setdefault(plugin, "sp_v1_hard_gate", True)
    _setdefault(plugin, "require_material_sanity", True)
    _setdefault(plugin, "material_max_heavy_gain", int(options.get("material_max_heavy_gain") or 3))
    _setdefault(plugin, "material_max_carbon_gain", int(options.get("material_max_carbon_gain") or 2))
    _setdefault(plugin, "material_max_hetero_gain", int(options.get("material_max_hetero_gain") or 3))
    _setdefault(plugin, "min_quality_score", options.get("min_quality_score"))
    _setdefault(plugin, "max_added", int(options.get("max_added") or 8))
    _setdefault(plugin, "score_scale", float(options.get("score_scale") or 1.0))
    _setdefault(plugin, "sp_v1_score_bonus", float(options.get("sp_v1_score_bonus") or 0.20))
    _setdefault(plugin, "quality_score_bonus", float(options.get("quality_score_bonus") or 0.18))
    flags["native_enzyme_plugin"] = plugin

    cost_model = dict(flags.get("cascade_cost_model") or {})
    _setdefault(cost_model, "enabled", True)
    weights = dict(_default_enzyme_strengthening_cost_weights())
    weights.update(cost_model.get("weights") or {})
    cost_model["weights"] = weights
    _setdefault(cost_model, "material_max_heavy_gain", int(options.get("material_max_heavy_gain") or 3))
    _setdefault(cost_model, "material_max_carbon_gain", int(options.get("material_max_carbon_gain") or 2))
    _setdefault(cost_model, "material_max_hetero_gain", int(options.get("material_max_hetero_gain") or 3))
    flags["cascade_cost_model"] = cost_model
    flags["use_cascade_cost_model"] = True

    context = dict(flags.get("cascade_search_context") or {})
    _setdefault(context, "context_policy", "chem_enzy_step_strengthening_v1")
    _setdefault(context, "min_enzyme_evidence_confidence", float(options.get("min_enzyme_evidence_confidence") or 0.35))
    active = list(context.get("active_failure_modes") or [])
    if "enzymeevidenceweak" not in {str(item).lower() for item in active}:
        active.append("enzymeevidenceweak")
    context["active_failure_modes"] = active
    flags["cascade_search_context"] = context

    return replace(config, search_flags=flags)


def _configure_native_autoplanner_plugins(
    api_module: Any,
    *,
    enzyme_config: NativeEnzymePluginConfig,
    chemical_config: NativeChemicalPluginConfig,
    literature_config: LiteratureOneStepPluginConfig | None = None,
) -> (
    tuple[NativeEnzymePluginState | None, NativeChemicalPluginState | None]
    | tuple[NativeEnzymePluginState | None, NativeChemicalPluginState | None, LiteratureOneStepPluginState | None]
):
    original = getattr(api_module, "_autoplanner_original_prepare_molstar_planner", None)
    if original is None:
        original = api_module.prepare_molstar_planner
        api_module._autoplanner_original_prepare_molstar_planner = original

    enzyme_state = NativeEnzymePluginState(config=enzyme_config) if enzyme_config.enabled else None
    chemical_state = NativeChemicalPluginState(config=chemical_config) if chemical_config.enabled else None
    literature_state = (
        LiteratureOneStepPluginState(config=literature_config)
        if literature_config is not None and literature_config.enabled
        else None
    )
    if enzyme_state is None and chemical_state is None and literature_state is None:
        api_module.prepare_molstar_planner = original
        return (None, None, None) if literature_config is not None else (None, None)

    def patched_prepare_molstar_planner(*args: Any, **kwargs: Any) -> Any:
        if args:
            one_step = args[0]
            if chemical_state is not None:
                one_step = NativeChemicalOneStepWrapper(one_step, config=chemical_config, state=chemical_state)
            if literature_state is not None:
                one_step = LiteratureTemplateOneStepWrapper(one_step, config=literature_config, state=literature_state)
            if enzyme_state is not None:
                one_step = NativeEnzymeOneStepWrapper(one_step, config=enzyme_config, state=enzyme_state)
            args = (one_step, *args[1:])
        elif "one_step" in kwargs:
            kwargs = dict(kwargs)
            one_step = kwargs["one_step"]
            if chemical_state is not None:
                one_step = NativeChemicalOneStepWrapper(one_step, config=chemical_config, state=chemical_state)
            if literature_state is not None:
                one_step = LiteratureTemplateOneStepWrapper(one_step, config=literature_config, state=literature_state)
            if enzyme_state is not None:
                one_step = NativeEnzymeOneStepWrapper(one_step, config=enzyme_config, state=enzyme_state)
            kwargs["one_step"] = one_step
        return original(*args, **kwargs)

    api_module.prepare_molstar_planner = patched_prepare_molstar_planner
    if literature_config is None:
        return enzyme_state, chemical_state
    return enzyme_state, chemical_state, literature_state


def _chemical_plugin_config_with_base_model(
    config: NativeChemicalPluginConfig,
    one_step_models: Iterable[str],
) -> NativeChemicalPluginConfig:
    if not config.enabled or config.base_model_full_name:
        return config
    names = [str(name) for name in one_step_models or [] if str(name or "")]
    graphfp = [name for name in names if name.startswith("graphfp_models.")]
    if len(graphfp) == 1:
        return replace(config, base_model_full_name=graphfp[0])
    if len(names) == 1:
        return replace(config, base_model_full_name=names[0])
    return config


def _default_enzyme_strengthening_cost_weights() -> dict[str, float]:
    return {
        "weak_enzyme_evidence_penalty": 0.70,
        "active_failure_match_reward": 0.10,
        "material_new_element_penalty": 1.40,
        "material_heavy_gain_penalty": 1.15,
        "material_carbon_gain_penalty": 1.15,
        "material_hetero_gain_penalty": 0.85,
        "enzyme_material_penalty_multiplier": 1.35,
    }


def _strengthening_enabled(raw: Any) -> bool:
    if isinstance(raw, dict):
        return bool(raw.get("enabled", True))
    if isinstance(raw, str):
        return raw.strip().lower() not in {"", "0", "false", "off", "no"}
    return bool(raw)


def _setdefault(mapping: dict[str, Any], key: str, value: Any) -> None:
    if key not in mapping or mapping.get(key) in (None, ""):
        mapping[key] = value


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


def apply_onmt_tokenizer_override(
    config: dict[str, Any],
    tokenizer: str,
    *,
    model_name: str = DEFAULT_ONMT_MODEL_NAME,
) -> dict[str, Any]:
    """Record the tokenizer mode used by the vendored ONMT one-step wrapper."""
    tokenizer = str(tokenizer or "char").strip().lower()
    if tokenizer not in {"char", "token", "pretokenized"}:
        raise ValueError(f"unsupported ChemEnzy ONMT tokenizer: {tokenizer}")
    try:
        model_type, model_subname = model_name.split(".", 1)
    except ValueError as exc:
        raise ValueError(f"invalid ChemEnzy one-step model name: {model_name}") from exc
    one_step_configs = config.setdefault("one_step_model_configs", {})
    if model_type not in one_step_configs or model_subname not in (one_step_configs.get(model_type) or {}):
        raise ValueError(f"cannot override unknown ChemEnzy ONMT model config: {model_name}")
    model_config = dict(one_step_configs[model_type][model_subname] or {})
    model_config["tokenizer"] = tokenizer
    if tokenizer in {"char", "token", "pretokenized"}:
        model_config.pop("source_prefix", None)
        model_config.pop("source_tokenizer", None)
    one_step_configs[model_type][model_subname] = model_config
    config["chem_enzy_onmt_tokenizer"] = tokenizer
    return config


def _patch_onmt_tokenizer(api_module: Any, tokenizer: str) -> None:
    """Patch vendored ONMT preparation to honor a tokenizer mode without editing vendor files."""
    tokenizer = str(tokenizer or "char").strip().lower()
    if tokenizer not in {"char", "token", "pretokenized"}:
        raise ValueError(f"unsupported ChemEnzy ONMT tokenizer: {tokenizer}")
    if tokenizer == "char":
        return
    from retro_planner.common import prepare_utils  # type: ignore
    from onmt.bin.translate import load_model, run, smi_tokenizer  # type: ignore

    original = getattr(prepare_utils, "_autoplanner_original_prepare_onmt_models", None)
    if original is None:
        original = prepare_utils.prepare_onmt_models
        prepare_utils._autoplanner_original_prepare_onmt_models = original

    def prepare_onmt_models_token(
        model_path: Any,
        beam_size: int,
        topk: int,
        device: Any,
        tokenizer: str = tokenizer,
        source_prefix: str | None = None,
        source_tokenizer: str | None = None,
        **_: Any,
    ) -> Any:
        class OnmtRunWrapper:
            def __init__(self, model_path: Any, beam_size: int, topk: int, device: Any) -> None:
                self.opt, self.translator = load_model(
                    model_path=model_path,
                    beam_size=beam_size,
                    topk=topk,
                    device=int(str(device).split(":")[-1]) if str(device) != "cpu" else -1,
                    tokenizer=tokenizer,
                )

            def run(self, target: str, topk: int | None = None) -> dict[str, Any]:
                results = run(self.translator, self.opt, _format_onmt_source_for_tokenizer(target, tokenizer, smi_tokenizer))
                templates = [None for _ in range(len(results.get("scores") or []))]
                results["template"] = templates
                return results

        return OnmtRunWrapper(model_path=model_path, beam_size=beam_size, topk=topk, device=device)

    prepare_utils.prepare_onmt_models = prepare_onmt_models_token
    api_module.prepare_single_step.__globals__["prepare_onmt_models"] = prepare_onmt_models_token


def _format_onmt_source_for_tokenizer(target: str, tokenizer: str, smi_tokenizer_fn: Any) -> str:
    text = str(target or "").strip()
    if tokenizer != "pretokenized":
        return text
    prefix = str(os.environ.get(CHEMENZY_ONMT_SOURCE_PREFIX_ENV) or "").strip()
    if not prefix:
        return text
    mode = str(os.environ.get(CHEMENZY_ONMT_PRETOKENIZE_MODE_ENV) or "char").strip().lower()
    compact = text.replace(" ", "")
    if mode == "token":
        tokenized = smi_tokenizer_fn(compact)
    else:
        tokenized = " ".join(compact)
    return f"{prefix} {tokenized}".strip()


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
            template_payload = reaction_node.get("template")
            enzyme_annotations = _enzyme_annotations(attrs)
            enzyme_annotations.extend(_enzyme_annotations_from_template(template_payload))
            steps.append(
                RouteStepCandidate(
                    product_smiles=product,
                    reactant_smiles=reactants,
                    rxn_smiles=rxn_smiles,
                    source_model=_source_model(reaction_node),
                    score=_step_score(reaction_node),
                    stock_status={str(node.get("smiles") or ""): node.get("in_stock") for node in reactant_nodes},
                    condition_predictions=_condition_predictions(attrs),
                    enzyme_ec_annotations=enzyme_annotations,
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
    cascade_cost = reaction_node.get("cascade_cost")
    if isinstance(cascade_cost, dict) and cascade_cost.get("source_model"):
        return str(cascade_cost.get("source_model") or "")
    template = reaction_node.get("template")
    if isinstance(template, dict):
        model = str(template.get("model_full_name") or template.get("model_name") or "")
        source_model = str(template.get("source_model") or template.get("source") or "")
        if model == LITERATURE_PLUGIN_MODEL_FULL_NAME or source_model == LITERATURE_TEMPLATE_PLUGIN_SOURCE:
            return LITERATURE_TEMPLATE_PLUGIN_SOURCE
        return str(model or source_model or "")
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


def _enzyme_annotations_from_template(template: Any) -> list[dict[str, Any]]:
    if not isinstance(template, dict):
        return []
    source = str(template.get("source") or template.get("model_full_name") or "").lower()
    ec = str(template.get("ec") or "").strip()
    sp_payload = template.get("enzyme_sp_verifier_v1") if isinstance(template.get("enzyme_sp_verifier_v1"), dict) else {}
    evidence = template.get("evidence") if isinstance(template.get("evidence"), dict) else {}
    ec_numbers = []
    for raw in (
        [ec] if ec else [],
        sp_payload.get("ec_numbers") if isinstance(sp_payload, dict) else [],
        evidence.get("ec_numbers") if isinstance(evidence, dict) else [],
    ):
        if isinstance(raw, str):
            ec_numbers.append(raw)
        elif isinstance(raw, list):
            ec_numbers.extend(str(item) for item in raw if str(item or "").strip())
    ec_numbers = list(dict.fromkeys(item for item in ec_numbers if item))
    if not ec_numbers and "enzyme" not in source:
        return []
    confidence = sp_payload.get("score") if isinstance(sp_payload, dict) else None
    return [
        {
            "rank": f"Top-{idx + 1}",
            "ec_number": ec_number,
            "confidence": confidence,
            "raw": {
                "source": template.get("source") or template.get("model_full_name"),
                "enzyme_sp_verifier_v1": sp_payload,
                "autoplanner_native_enzyme_plugin": bool(template.get("autoplanner_native_enzyme_plugin")),
                "autoplanner_native_chemical_plugin": bool(template.get("autoplanner_native_chemical_plugin")),
            },
        }
        for idx, ec_number in enumerate(ec_numbers or [ec or "enzyme_precedent"])
    ]


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


def _template_relevance_failures(
    models: Iterable[str],
    vendor_root: Path,
    *,
    target_smiles: str = "",
) -> list[BackendFailure]:
    missing = missing_template_relevance_models(tuple(models or ()), vendor_root=vendor_root)
    if not missing:
        return []
    return [
        BackendFailure(
            category="template_relevance_model_missing",
            message="missing local template_relevance .mar archive(s): " + ", ".join(missing),
            target_smiles=target_smiles,
            retryable=True,
            raw_backend_metadata={
                "missing_models": missing,
                "vendor_root": str(vendor_root),
            },
        )
    ]


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
