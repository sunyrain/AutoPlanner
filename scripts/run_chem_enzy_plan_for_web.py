"""Run one ChemEnzyRetroPlanner native search and emit web-compatible JSON."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.chem_enzy_adapter import (
    CHEMENZY_WORKER_BOOTSTRAP_ENV,
    CHEMENZY_WORKER_EASIFA_ENV,
    CHEMENZY_WORKER_GRAPHVIZ_ENV,
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
    install_chemenzy_import_compatibility,
)
from cascade_planner.baselines.chem_enzy_budget import (
    budgeted_chemenzy_payload,
    finalize_effective_chemenzy_budget,
    resolution_from_dict,
    resolve_chemenzy_budget,
)
from cascade_planner.baselines.chem_enzy_step_quality import evaluate_enzyme_step_quality
from cascade_planner.baselines.route_contract import RouteCandidate, RouteSearchConfig, RouteStepCandidate
from cascade_planner.baselines.template_relevance_runtime import missing_template_relevance_models
from cascade_planner.agent.chem_enzy_policy import apply_chem_enzy_search_policy
from cascade_planner.cascade_search.enzyme_coverage_sidecar import (
    EnzymeCoverageSidecarConfig,
    build_enzyme_coverage_sidecar,
)
from cascade_planner.cascade_verifier import load_learned_verifier, predict_learned_verifier, verify_cascade_route
from cascade_planner.legacy.guard import LEGACY_RESEARCH_ENV, legacy_research_enabled


RDLogger.DisableLog("rdApp.*")


def _bootstrap_pandarallel_worker_from_environment() -> bool:
    """Replay parent import shims in Windows spawn workers before dill loads."""

    if os.environ.get(CHEMENZY_WORKER_BOOTSTRAP_ENV) != "1":
        return False
    install_chemenzy_import_compatibility(
        enable_easifa=os.environ.get(CHEMENZY_WORKER_EASIFA_ENV) == "1",
        enable_graphviz=os.environ.get(CHEMENZY_WORKER_GRAPHVIZ_ENV) == "1",
    )
    return True


_bootstrap_pandarallel_worker_from_environment()


DEFAULT_LEARNED_VERIFIER_MODEL = Path(
    "results/shared/cascade_verifier_mainline_20260521/learned_verifier_v4_30k_stage_aware.joblib"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ChemEnzy native core search for the AutoPlanner web UI")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    ap.add_argument("--gpu", type=int, default=-1)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    _configure_pandarallel_worker_environment(payload)
    embedded_resolution = payload.get("chem_enzy_budget_resolution")
    if isinstance(embedded_resolution, dict):
        budget_resolution = resolution_from_dict(embedded_resolution)
    else:
        action_kind = str(payload.get("chem_enzy_action_kind") or "native")
        if action_kind not in {"native", "guided", "child"}:
            action_kind = "native"
        policy_payload = payload.get("chem_enzy_search_policy") or payload.get("search_policy")
        budget_resolution = resolve_chemenzy_budget(
            target_smiles=str(payload.get("target_smiles") or ""),
            action_kind=action_kind,
            payload=payload,
            policy=dict(policy_payload or {}) if isinstance(policy_payload, dict) else {},
            authority="operator_explicit",
            attempt_index=max(1, int(payload.get("chem_enzy_attempt_index") or 1)),
            timeout_cap_s=payload.get("timeout_s"),
        )
    payload = budgeted_chemenzy_payload(payload, budget_resolution)
    started = time.monotonic()
    config = _route_config_from_payload(payload, args.gpu)
    budget_resolution = finalize_effective_chemenzy_budget(
        budget_resolution,
        max_depth=config.max_depth,
        max_iterations=config.max_iterations,
        expansion_topk=config.expansion_topk,
    )
    payload["chem_enzy_budget_resolution"] = budget_resolution.to_dict()
    adapter = ChemEnzyBackendAdapter(
        vendor_root=Path(args.vendor_root),
        gpu=args.gpu,
        enable_condition_prediction=bool(payload.get("enable_condition_prediction", False)),
        enable_enzyme_assignment=bool(payload.get("enable_enzyme_assignment", False)),
        enable_easifa=bool(payload.get("enable_easifa", False)),
    )
    result = adapter.run_target(config)
    output = _web_payload_from_result(result, payload, config, time.monotonic() - started, vendor_root=Path(args.vendor_root))
    output["runtime_compatibility"] = {
        "pandarallel_spawn_bootstrap_configured": True,
        "worker_import_shims": [
            "numpy_legacy_aliases",
            "torchdata_legacy_aliases",
            "torchtext_legacy_aliases",
            "dgl_graphbolt_optional_import",
            "optional_easifa_import",
            "optional_graphviz_import",
        ],
    }
    output["chem_enzy_budget_resolution"] = budget_resolution.to_dict()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _configure_pandarallel_worker_environment(payload: dict[str, Any]) -> None:
    """Configure import-time compatibility for later Windows spawn workers."""

    os.environ[CHEMENZY_WORKER_BOOTSTRAP_ENV] = "1"
    os.environ[CHEMENZY_WORKER_EASIFA_ENV] = (
        "1" if bool(payload.get("enable_easifa", False)) else "0"
    )
    os.environ[CHEMENZY_WORKER_GRAPHVIZ_ENV] = (
        "1" if bool(payload.get("viz", False)) else "0"
    )


def _route_config_from_payload(payload: dict[str, Any], gpu: int) -> RouteSearchConfig:
    preset = str(payload.get("search_preset") or "quick").lower()
    max_depth = _as_int(payload.get("max_steps"), 6, lo=1, hi=20)
    if preset == "quick":
        iterations = _as_int(payload.get("chem_enzy_iterations"), 10, lo=1, hi=200)
        expansion_topk = _as_int(payload.get("chem_enzy_expansion_topk"), 50, lo=1, hi=200)
    elif preset == "thorough":
        iterations = _as_int(payload.get("chem_enzy_iterations"), 50, lo=1, hi=500)
        expansion_topk = _as_int(payload.get("chem_enzy_expansion_topk"), 100, lo=1, hi=500)
    else:
        iterations = _as_int(payload.get("chem_enzy_iterations"), 25, lo=1, hi=300)
        expansion_topk = _as_int(payload.get("chem_enzy_expansion_topk"), 75, lo=1, hi=300)
    legacy_hooks_requested = bool(payload.get("enable_legacy_cascade_hooks", False))
    if legacy_hooks_requested and not legacy_research_enabled():
        raise ValueError(
            "legacy cascade hooks are archived/frozen research code; "
            f"set {LEGACY_RESEARCH_ENV}=1 to enable them explicitly"
        )
    legacy_hooks_enabled = legacy_hooks_requested and legacy_research_enabled()
    onmt_tokenizer = str(payload.get("chem_enzy_onmt_tokenizer") or payload.get("onmt_tokenizer") or "char").strip().lower()
    if onmt_tokenizer not in {"char", "token"}:
        raise ValueError(f"unsupported chem_enzy_onmt_tokenizer: {onmt_tokenizer}")
    search_flags = {
        "gpu": gpu,
        "condition_model": payload.get("condition_model", "rcr"),
        "chem_enzy_onmt_tokenizer": onmt_tokenizer,
        "keep_search": True,
        "use_filter": payload.get("use_filter", False),
        "use_depth_value_fn": payload.get("use_depth_value_fn", False),
        "include_cascade_expansion_trace": True,
        "cascade_search_context": {
            "enabled": True,
            "target_smiles": str(payload["target_smiles"]),
            "search_preset": preset,
            "domain": payload.get("domain", "chemoenzymatic"),
        },
        "use_cascade_cost_model": legacy_hooks_enabled,
        "cascade_cost_model": _default_cascade_cost_model() if legacy_hooks_enabled else {"enabled": False},
        "use_cascade_source_policy": legacy_hooks_enabled,
        "cascade_source_policy": _default_cascade_source_policy() if legacy_hooks_enabled else {"enabled": False},
        "legacy_cascade_hooks_requested": legacy_hooks_requested,
        "legacy_cascade_hooks_enabled": legacy_hooks_enabled,
    }
    raw_stock_paths = payload.get("stock_paths")
    if isinstance(raw_stock_paths, dict):
        search_flags["stock_paths"] = {
            str(name): str(Path(str(path)).expanduser().resolve())
            for name, path in raw_stock_paths.items()
            if str(name).strip() and str(path).strip()
        }
    max_output_routes = _optional_positive_int(payload.get("max_routes"), hi=100)
    if max_output_routes is not None:
        # Iteration/depth/top-k remain hard maxima, not work targets.  Stop
        # MCTS once a bounded successful-route reserve exists, then host-audit
        # that reserve before any serial condition/enzyme annotation.  The
        # reserve prevents one verifier rejection from starving the requested
        # output portfolio.
        search_flags["max_output_routes"] = max_output_routes
        search_flags["max_materialized_routes"] = _as_int(
            payload.get("max_materialized_routes"),
            max(16, max_output_routes * 8),
            lo=max_output_routes,
            hi=500,
        )
        search_flags["max_advisory_materialized_routes"] = _as_int(
            payload.get("max_advisory_materialized_routes"),
            max_output_routes,
            lo=0,
            hi=100,
        )
    native_enzyme_plugin = _native_enzyme_plugin_from_payload(payload)
    if native_enzyme_plugin:
        search_flags["native_enzyme_plugin"] = native_enzyme_plugin
    literature_template_plugin = _literature_template_plugin_from_payload(payload)
    if literature_template_plugin:
        search_flags["literature_template_plugin"] = literature_template_plugin
    step_strengthening = _chem_enzy_step_strengthening_from_payload(payload)
    if step_strengthening:
        search_flags["chem_enzy_step_strengthening"] = step_strengthening
    one_step_models = list(payload.get("one_step_models") or DEFAULT_ONE_STEP_MODELS)
    missing_template_models = missing_template_relevance_models(one_step_models)
    if missing_template_models:
        raise ValueError(
            "missing local template_relevance .mar archive(s): "
            + ", ".join(missing_template_models)
        )
    config = RouteSearchConfig(
        target_smiles=str(payload["target_smiles"]),
        stock_names=_stock_names_from_payload(payload),
        max_iterations=iterations,
        max_depth=max_depth,
        expansion_topk=expansion_topk,
        random_seed=_as_int(payload.get("chemenzy_seed"), 0, lo=0, hi=2**32 - 1),
        one_step_models=one_step_models,
        search_flags=search_flags,
    )
    policy_payload = payload.get("chem_enzy_search_policy") or payload.get("search_policy")
    if policy_payload:
        config = apply_chem_enzy_search_policy(config, dict(policy_payload))
    return config


def _chem_enzy_step_strengthening_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(
        payload.get("enable_chem_enzy_step_strengthening")
        or payload.get("chem_enzy_step_strengthening")
        or str(payload.get("search_preset") or "").lower() in {"enzyme_strengthened", "enzyme-strengthened"}
    )
    if not enabled:
        return {}
    return {
        "enabled": True,
        "top_k": _as_int(payload.get("native_enzyme_topk"), 8, lo=1, hi=50),
        "bridge_top_k": _as_int(payload.get("native_enzyme_bridge_topk"), 10, lo=1, hi=50),
        "max_ec_contexts": _as_int(payload.get("native_enzyme_max_ec_contexts"), 3, lo=0, hi=7),
        "max_added": _as_int(payload.get("native_enzyme_max_added"), 8, lo=1, hi=50),
        "sp_v1_score_bonus": float(payload.get("native_enzyme_sp_v1_score_bonus") or 0.20),
        "quality_score_bonus": float(payload.get("native_enzyme_quality_score_bonus") or 0.18),
    }


def _native_enzyme_plugin_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(
        payload.get("enable_native_enzyme_plugin")
        or payload.get("native_enzyme_plugin")
        or str(payload.get("search_preset") or "").lower() in {"native_enzyme", "native-enzyme"}
    )
    if not enabled:
        return {}
    return {
        "enabled": True,
        "top_k": _as_int(payload.get("native_enzyme_topk"), 6, lo=1, hi=50),
        "bridge_top_k": _as_int(payload.get("native_enzyme_bridge_topk"), 8, lo=1, hi=50),
        "max_ec_contexts": _as_int(payload.get("native_enzyme_max_ec_contexts"), 2, lo=0, hi=7),
        "require_bridge": not bool(payload.get("native_enzyme_disable_bridge_gate")),
        "require_verifier_pass": not bool(payload.get("native_enzyme_disable_bridge_verifier")),
        "enable_sp_v1": not bool(payload.get("disable_enzyme_sp_v1")),
        "sp_v1_hard_gate": not bool(payload.get("native_enzyme_disable_sp_v1_hard_gate")),
        "max_added": _as_int(payload.get("native_enzyme_max_added"), 6, lo=1, hi=50),
        "score_scale": float(payload.get("native_enzyme_score_scale") or 1.0),
        "sp_v1_score_bonus": float(payload.get("native_enzyme_sp_v1_score_bonus") or 0.0),
    }


def _literature_template_plugin_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = (
        payload.get("literature_template_plugin")
        if "literature_template_plugin" in payload
        else payload.get("autoplanner_literature_template_plugin")
    )
    if not raw:
        return {}
    if raw is True:
        return {"enabled": True}
    if not isinstance(raw, dict):
        return {"enabled": bool(raw)}
    out = dict(raw)
    out.setdefault("enabled", True)
    return out


def _stock_names_from_payload(payload: dict[str, Any]) -> list[str]:
    explicit = payload.get("stock_names")
    if explicit:
        return list(explicit)
    mode = str(payload.get("stock_mode") or "building-block").strip().lower()
    if mode in {"commercial", "zinc", "zinc_fix", "zinc-fix"}:
        return ["Zinc_Fix-stock"]
    if mode in {"benchmark-n5", "paroutes-n5", "n5"}:
        return ["PaRotes_n5-stock"]
    if mode in {"building-block", "building_block", "strict", "paroutes-n1", "n1"}:
        return ["PaRotes_n1-stock"]
    return list(DEFAULT_STOCKS)


def _default_cascade_cost_model() -> dict[str, Any]:
    model_path = Path("results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/models/cascade_state_action_value_e4.pt")
    config: dict[str, Any] = {
        "enabled": True,
        "weights": {
            "learned_action_value_score_reward": 0.35,
        },
    }
    if model_path.exists():
        config["action_value_model_path"] = str(model_path)
    return config


def _default_cascade_source_policy() -> dict[str, Any]:
    model_path = Path("results/shared/dataset_v4_release/v4_full_training_stage3_fineshard_20260511/models/cascade_source_value_baseline.pt")
    config: dict[str, Any] = {
        "enabled": True,
    }
    if model_path.exists():
        config["source_value_model_path"] = str(model_path)
    return config


def _web_payload_from_result(
    result: Any,
    request_payload: dict[str, Any],
    config: RouteSearchConfig,
    elapsed_s: float,
    vendor_root: Path | str | None = None,
) -> dict[str, Any]:
    learned_verifier, learned_annotation_report = _load_learned_verifier_annotation(request_payload)
    rescue_candidates = _semisynthesis_rescue_candidates(result, request_payload)
    rescue_routes = [
        _web_route(route, index, learned_verifier=learned_verifier)
        for index, route in enumerate(rescue_candidates)
    ]
    native_raw_routes = [
        _web_route(route, index, learned_verifier=learned_verifier)
        for index, route in enumerate(result.routes)
    ]
    raw_routes = _merge_rescue_and_native_routes(rescue_routes, native_raw_routes)
    learned_annotation_report["input_routes"] = len(raw_routes)
    learned_annotation_report["annotated_routes"] = sum(
        1
        for route in raw_routes
        if ((route.get("metrics") or {}).get("learned_cascade_verifier") or {}).get("available")
    )
    verifier_gate = _cascade_verifier_gate_enabled(request_payload)
    routes, quarantined_routes, verifier_gate_report = _apply_cascade_verifier_gate(
        raw_routes,
        enabled=verifier_gate,
    )
    output_limit = _configured_output_route_limit(request_payload, config)
    output_limit_report = {
        "enabled": output_limit is not None,
        "max_routes": output_limit,
        "eligible_before_limit": len(routes),
        "quarantined_before_limit": len(quarantined_routes),
        "eligible_truncated": 0,
        "quarantined_truncated": 0,
    }
    if output_limit is not None:
        output_limit_report["eligible_truncated"] = max(0, len(routes) - output_limit)
        output_limit_report["quarantined_truncated"] = max(
            0, len(quarantined_routes) - output_limit
        )
        routes = routes[:output_limit]
        quarantined_routes = quarantined_routes[:output_limit]
    strict_solved = any(bool((route.get("metrics") or {}).get("route_solved")) for route in routes)
    raw_solved = any(
        bool((route.get("raw_backend_metadata") or {}).get("raw_solved"))
        for route in native_raw_routes
    )
    rescue_report = {
        "enabled": request_payload.get("enable_semisynthesis_rescue") is True,
        "route_count": len(rescue_candidates),
        "rescue_types": sorted(
            {
                str((route.raw_backend_metadata or {}).get("rescue_type") or "")
                for route in rescue_candidates
                if (route.raw_backend_metadata or {}).get("rescue_type")
            }
        ),
    }
    rescue_report["input_native_routes"] = len(native_raw_routes)
    rescue_report["displayed_routes"] = sum(
        1 for route in routes if ((route.get("raw_backend_metadata") or {}).get("rescue_type"))
    )
    status = "solved" if strict_solved else "partial" if routes else "filtered" if raw_routes and verifier_gate else "failed"
    message = (
        "ChemEnzy native core search returned a host-admitted stock-closed route"
        if strict_solved
        else "ChemEnzy returned raw stock-closed routes, but host edge admission rejected every materialized route"
        if raw_solved
        else "AutoPlanner generated semisynthesis anchor routes; the advanced precursor remains an open upstream subgoal"
        if routes and rescue_routes and not native_raw_routes
        else "ChemEnzy native core search returned routes, but terminal reactants are not all in the selected stock"
        if routes
        else "ChemEnzy native core search returned routes, but the rule verifier gate removed all displayed candidates"
        if raw_routes and verifier_gate
        else "ChemEnzy native core search returned no route"
    )
    output = {
        "ok": (not bool(result.failures) or bool(rescue_routes)) and (bool(routes) or strict_solved),
        "raw_solved": raw_solved,
        "materialization_admission_solved": strict_solved,
        "target": result.target_smiles,
        "objective": "chem_enzy_native",
        "constraints": request_payload.get("constraints"),
        "n_results": len(routes),
        "time_s": round(elapsed_s, 3),
        "routes": routes,
        "quarantined_routes": quarantined_routes,
        "route_set_metrics": {
            "diversity": {
                "n_routes": len(routes),
                "unique_full_signatures": len({_route_signature(route) for route in routes}),
            },
            "cascade_verifier_gate": verifier_gate_report,
            "learned_verifier_annotation": learned_annotation_report,
            "semisynthesis_rescue": rescue_report,
            "route_materialization_admission": _route_materialization_admission_summary(
                raw_routes
            ),
            "output_limit": output_limit_report,
        },
        "ui_metadata": {
            "backend": "CascadePlanner",
            "engine": "ChemEnzyRetroPlanner",
            "planner_strategy": "ChemEnzy native multi-step search with AutoPlanner product audit and rule cascade verifier",
            "search_mode": "chem_enzy_native",
            "search_preset": request_payload.get("search_preset", "quick"),
            "stock_mode": request_payload.get("stock_mode", "building-block"),
            "max_depth": config.max_depth,
            "iterations": config.max_iterations,
            "expansion_topk": config.expansion_topk,
            "random_seed": config.random_seed,
            "condition_prediction_enabled": bool(request_payload.get("enable_condition_prediction", False)),
            "enzyme_assignment_enabled": bool(request_payload.get("enable_enzyme_assignment", False)),
            "chem_enzy_step_strengthening_enabled": bool(
                config.search_flags.get("chem_enzy_step_strengthening_enabled")
                or config.search_flags.get("chem_enzy_step_strengthening")
            ),
            "condition_model": request_payload.get("condition_model", "rcr"),
            "chem_enzy_onmt_tokenizer": config.search_flags.get("chem_enzy_onmt_tokenizer", "char"),
            "one_step_models": config.one_step_models,
            "stock_names": config.stock_names,
            "cascade_hooks": {
                "cost_model": bool(config.search_flags.get("use_cascade_cost_model")),
                "source_policy": bool(config.search_flags.get("use_cascade_source_policy")),
                "expansion_trace": bool(config.search_flags.get("include_cascade_expansion_trace")),
                "action_value_model_path": (config.search_flags.get("cascade_cost_model") or {}).get("action_value_model_path"),
                "source_value_model_path": (config.search_flags.get("cascade_source_policy") or {}).get("source_value_model_path"),
                "legacy_hooks_enabled": bool(config.search_flags.get("legacy_cascade_hooks_enabled")),
            },
            "cascade_verifier_gate": verifier_gate_report,
            "learned_verifier_annotation": learned_annotation_report,
            "semisynthesis_rescue": rescue_report,
            "saved_at": None,
        },
        "skeletons": [],
        "depth_attempts": [
            {
                "depth": config.max_depth,
                "elapsed_s": round(elapsed_s, 3),
                "n_skeletons": 0,
                "n_routes": len(routes),
                "planner": "CascadePlanner",
                "engine": "ChemEnzyRetroPlanner",
                "status": status,
                "best": _route_summary(routes[0]) if routes else None,
            }
        ],
        "search_status": {
            "status": status,
            "solved": strict_solved,
            "raw_solved": raw_solved,
            "materialization_admission_solved": strict_solved,
            "raw_solved_is_not_host_solved_authority": True,
            "native_returned_routes": any(
                not ((route.get("raw_backend_metadata") or {}).get("rescue_type"))
                for route in routes
            ),
            "native_raw_returned_routes": bool(native_raw_routes),
            "native_raw_n_routes": len(native_raw_routes),
            "native_search_found_n_routes": int(
                dict(
                    (result.raw_backend_metadata or {}).get(
                        "route_materialization_selection"
                    )
                    or {}
                ).get("raw_route_count")
                or len(native_raw_routes)
            ),
            "semisynthesis_rescue_returned_routes": bool(rescue_routes),
            "semisynthesis_rescue_n_routes": len(rescue_routes),
            "best_depth": config.max_depth,
            "message": message,
        },
        "failure_diagnosis": [failure.category for failure in result.failures],
        "backend_failures": [failure.to_dict() for failure in result.failures],
        "raw_backend_metadata": result.raw_backend_metadata,
    }
    _attach_enzyme_coverage_sidecar(output, request_payload)
    output["failure_analysis"] = _failure_analysis(result, request_payload, config, vendor_root=Path(vendor_root) if vendor_root else None)
    return output


def _attach_enzyme_coverage_sidecar(output: dict[str, Any], request_payload: dict[str, Any]) -> None:
    enabled = bool(
        request_payload.get("enable_enzyme_coverage_sidecar")
        or request_payload.get("enzyme_coverage_sidecar")
        or str(request_payload.get("search_preset") or "").lower() in {"enzyme_coverage", "enzyme-coverage"}
    )
    metadata = output.setdefault("ui_metadata", {})
    metadata["enzyme_coverage_sidecar_enabled"] = enabled
    if not enabled:
        return
    config = EnzymeCoverageSidecarConfig(
        top_k=_as_int(request_payload.get("enzyme_coverage_topk"), 8, lo=1, hi=50),
        bridge_top_k=_as_int(request_payload.get("enzyme_coverage_bridge_topk"), 8, lo=1, hi=50),
        max_ec_contexts=_as_int(request_payload.get("enzyme_coverage_max_ec_contexts"), 2, lo=0, hi=7),
        enable_sp_v1=not bool(request_payload.get("disable_enzyme_sp_v1")),
    )
    sidecar = build_enzyme_coverage_sidecar(
        str(output.get("target") or request_payload.get("target_smiles") or ""),
        config=config,
    )
    output.setdefault("route_set_metrics", {})["enzyme_coverage_sidecar"] = sidecar
    metadata["enzyme_coverage_sidecar"] = {
        "enabled": True,
        "source": sidecar.get("source"),
        "bridge_hit_count": sidecar.get("bridge_hit_count"),
        "candidate_count": sidecar.get("candidate_count"),
        "sp_v1_accepted_count": sidecar.get("sp_v1_accepted_count"),
        "error": sidecar.get("error"),
    }


def _semisynthesis_rescue_candidates(result: Any, request_payload: dict[str, Any]) -> list[RouteCandidate]:
    # Legacy molecule-specific rescue tables are excluded from the generic
    # agentic mainline.  They remain opt-in only for explicit historical
    # replays and can never be activated by a missing/default field.
    if request_payload.get("enable_semisynthesis_rescue") is not True:
        return []
    from cascade_planner.baselines.semisynthesis_rescue import semisynthesis_rescue_routes

    target = str(result.target_smiles or request_payload.get("target_smiles") or "")
    return semisynthesis_rescue_routes(target)


def _merge_rescue_and_native_routes(
    rescue_routes: list[dict[str, Any]],
    native_routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for route in [*rescue_routes, *native_routes]:
        signature = _route_signature(route)
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(route)
    merged = sorted(merged, key=_display_route_sort_key)
    for index, route in enumerate(merged):
        route["route_rank"] = index
    return merged


def _display_route_sort_key(route: dict[str, Any]) -> tuple[int, int, int, int, float]:
    metrics = route.get("metrics") or {}
    verifier = metrics.get("cascade_verifier") or {}
    return (
        0 if metrics.get("source_supported_semisynthesis") else 1,
        0 if metrics.get("route_solved") else 1,
        0 if verifier.get("feasible") else 1,
        0 if metrics.get("semisynthesis_anchor") else 1,
        -float(route.get("score") or 0.0),
    )


def _web_route(route: RouteCandidate, index: int, *, learned_verifier: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = [_web_step(step, idx) for idx, step in enumerate(route.steps)]
    metrics = _route_metrics(route, steps, learned_verifier=learned_verifier)
    rescue_type = (route.raw_backend_metadata or {}).get("rescue_type")
    why_selected = (
        "Generated by AutoPlanner semisynthesis rescue as a late-stage derivatization anchor; upstream access to the advanced precursor remains unresolved."
        if rescue_type
        else "Returned by CascadePlanner using ChemEnzyRetroPlanner as the multi-step search engine."
    )
    return {
        "score": route.score,
        "confidence": 1.0 if route.solved else 0.0,
        "n_steps": len(steps),
        "quality_vector": {},
        "risk_vector": {},
        "constraint_report": {"search_mode": "CascadePlanner", "backend": "CascadePlanner", "engine": route.backend},
        "bottleneck_slot": None,
        "bottleneck_reason": "",
        "global_constraints": {},
        "steps": steps,
        "metrics": metrics,
        "explanation": {
            "why_selected": why_selected,
            "uncertainty_table": {
                "expansions": None,
                "generated_reactions": None,
            },
        },
        "route_rank": index,
        "raw_backend_metadata": route.raw_backend_metadata,
    }


def _cascade_verifier_gate_enabled(request_payload: dict[str, Any]) -> bool:
    return bool(request_payload.get("enable_rule_verifier_gate") or request_payload.get("cascade_verifier_gate"))


def _apply_cascade_verifier_gate(
    routes: list[dict[str, Any]],
    *,
    enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not enabled:
        return routes, [], {
            "enabled": False,
            "input_routes": len(routes),
            "kept_routes": len(routes),
            "dropped_routes": 0,
            "default_stage_mode": "stepwise",
            "dropped": [],
        }

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for route in routes:
        metrics = route.get("metrics") or {}
        report = metrics.get("cascade_verifier") or {}
        if report.get("feasible") is False:
            route["advisory_only"] = True
            route["confidence"] = 0.0
            route["warning_codes"] = sorted(
                {
                    "cascade_verifier_rejected",
                    *(str(value) for value in (report.get("reason_counts") or {})),
                }
            )
            dropped.append(
                {
                    "route_rank": route.get("route_rank"),
                    "n_steps": route.get("n_steps"),
                    "score": route.get("score"),
                    "reason_counts": report.get("reason_counts") or {},
                }
            )
            continue
        kept.append(route)

    for index, route in enumerate(kept):
        route["route_rank"] = index

    return kept, [
        route
        for route in routes
        if route.get("advisory_only") is True
    ], {
        "enabled": True,
        "input_routes": len(routes),
        "kept_routes": len(kept),
        "dropped_routes": len(dropped),
        "default_stage_mode": "stepwise",
        "dropped": dropped[:50],
    }


def _web_step(step: RouteStepCandidate, index: int) -> dict[str, Any]:
    reactants = list(step.reactant_smiles or [])
    main = reactants[0] if reactants else ""
    aux = reactants[1:]
    condition = _top_condition_prediction(step)
    enzyme = _top_enzyme_annotation(step)
    reaction_type = _reaction_type(step)
    ec = str(enzyme.get("ec_number") or "") if enzyme else ""
    catalyst = _condition_value(condition, "Catalyst", "catalyst") or _condition_value(condition, "Reagent", "reagent")
    if ec and not catalyst:
        catalyst = f"EC {ec}"
    temperature = _condition_value(condition, "Temperature", "temperature", "temperature_c")
    ph = _condition_value(condition, "pH", "ph")
    solvent = _condition_value(condition, "Solvent", "solvent")
    condition_score = _safe_float(_condition_value(condition, "Score", "score", "confidence"))
    enzyme_score = _safe_float(enzyme.get("confidence")) if enzyme else None
    condition_notes = _condition_notes(condition, enzyme)
    quality = _enzyme_quality(step)
    return {
        "index": index,
        "product": step.product_smiles,
        "main_reactant": main,
        "aux_reactants": aux,
        "reaction_smiles": step.rxn_smiles,
        "reaction_type": reaction_type,
        "ec": ec,
        "enzyme_uid": enzyme.get("uniprot_id") if enzyme else None,
        "catalyst": catalyst or "",
        "T": _safe_float(temperature),
        "pH": _safe_float(ph),
        "solvent": str(solvent or ""),
        "condition_predictions": list(step.condition_predictions or []),
        "enzyme_ec_annotations": list(step.enzyme_ec_annotations or []),
        "catalyst_annotations": list(step.catalyst_annotations or []),
        "raw_backend_metadata": dict(step.raw_backend_metadata or {}),
        "chemical_step_equivalent_count": (step.raw_backend_metadata or {}).get(
            "chemical_step_equivalent_count"
        ),
        "replaced_step_ids": list(
            (step.raw_backend_metadata or {}).get("replaced_step_ids") or []
        ),
        "selectivity_objective": str(
            (step.raw_backend_metadata or {}).get("selectivity_objective") or ""
        ),
        "enzyme_quality": quality,
        "evidence": {
            "backend": "CascadePlanner",
            "engine": "ChemEnzyRetroPlanner",
            "condition_prediction_available": bool(step.condition_predictions),
            "enzyme_annotation_available": bool(step.enzyme_ec_annotations),
            "enzyme_quality_score": quality.get("quality_score") if quality else None,
            "enzyme_quality_decision": quality.get("decision") if quality else "",
        },
        "source": _display_source(step),
        "scores": {
            "retro": step.score,
            "enzyme": enzyme_score,
            "condition": condition_score,
            "confidence": quality.get("quality_score") if quality else step.score,
        },
        "fixed_fields": [],
        "is_filled": True,
        "is_enzymatic": bool(step.enzyme_ec_annotations),
        "stock_status": dict(step.stock_status or {}),
        "reaction_interpretation": {
            "reaction_class": reaction_type,
            "forward_summary": _reaction_summary(reaction_type, step),
            "reaction_principle": _reaction_principle(reaction_type),
            "likely_added_or_removed": _reactant_change_notes(reactants),
            "catalysis_and_conditions": condition_notes,
            "atom_change": _atom_change_notes(step.rxn_smiles),
        },
        "candidate_pool": {"n_candidates": 0, "top_candidates": []},
    }


def _route_metrics(
    route: RouteCandidate,
    steps: list[dict[str, Any]],
    *,
    learned_verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    terminal_stock_status = _terminal_stock_status(steps)
    route_metadata = route.raw_backend_metadata or {}
    semisynthesis_anchor = bool(route_metadata.get("rescue_type"))
    route_class_hint = str(route_metadata.get("route_class_hint") or "")
    stitched_semisynthesis = route_class_hint == "stitched_semisynthesis_upstream"
    source_supported_semisynthesis = route_class_hint == "source_supported_semisynthesis"
    materialization_admission = dict(
        route_metadata.get("route_materialization_admission") or {}
    )
    raw_backend_solved = bool(route_metadata.get("raw_solved"))
    strict_stock = (
        all(bool(value) for value in terminal_stock_status.values())
        if terminal_stock_status
        else bool(route.solved)
    )
    native_returned_route = bool(route.solved and not semisynthesis_anchor)
    native_raw_returned_route = bool(raw_backend_solved and not semisynthesis_anchor)
    displayed_progressive_route = bool(native_raw_returned_route or semisynthesis_anchor)
    stock_closed = bool((native_returned_route or stitched_semisynthesis or source_supported_semisynthesis) and strict_stock)
    verifier_report = verify_cascade_route(
        {
            "target": route.target_smiles,
            "steps": steps,
            # ChemEnzy route exports are sequential syntheses unless a stage
            # partition is explicitly supplied. Treating every route as
            # one-pot would falsely reject normal multi-step chemistry for
            # having different temperatures, solvents, or pH values.
            "stage_partition": [f"stage_{idx + 1}" for idx, _step in enumerate(steps)],
        },
        target_smiles=route.target_smiles,
        assume_single_stage=False,
    ).to_dict()
    learned_report = _learned_verifier_route_annotation(
        learned_verifier,
        target_smiles=route.target_smiles,
        steps=steps,
    )
    return {
        "professional_solved": stock_closed,
        "diagnostic_solved": bool(displayed_progressive_route and not stock_closed),
        "route_solved": stock_closed,
        "raw_backend_solved": raw_backend_solved,
        "raw_backend_solved_not_proof": bool(raw_backend_solved and not stock_closed),
        "route_outcome": str(route_metadata.get("route_outcome") or ""),
        "route_materialization_admission": materialization_admission,
        "strict_stock_solve": strict_stock,
        "native_returned_route": native_returned_route,
        "native_raw_returned_route": native_raw_returned_route,
        "semisynthesis_anchor": semisynthesis_anchor,
        "stitched_semisynthesis": stitched_semisynthesis,
        "source_supported_semisynthesis": source_supported_semisynthesis,
        "terminal_reactants": list(terminal_stock_status),
        "terminal_stock_status": terminal_stock_status,
        "progressive_route": displayed_progressive_route,
        "filled_route": displayed_progressive_route,
        "n_steps": len(steps),
        "retrosynthesis_progress": {
            "main_chain_reduction": 1.0 if displayed_progressive_route else 0.0,
            "largest_leaf_reduction": 1.0 if displayed_progressive_route else 0.0,
            "progressive_steps": len(steps),
            "progressive_step_fraction": 1.0 if displayed_progressive_route else 0.0,
        },
        "cascade_compatibility": {
            "cascade_compatibility_success": bool(verifier_report.get("feasible")),
            "score": verifier_report.get("score"),
            "issues": sorted((verifier_report.get("reason_counts") or {}).keys()),
            "reason_counts": verifier_report.get("reason_counts") or {},
            "verifier_contract": verifier_report.get("metrics", {}).get("contract"),
        },
        "cascade_verifier": verifier_report,
        "learned_cascade_verifier": learned_report,
        "condition": {"condition_window_success": None},
        "enzyme_evidence": {"enzyme_evidence_coverage": None},
        "operation_transitions": {"operation_score": None, "issues": []},
        "candidate_pool": {
            "steps_with_candidates": 0,
            "total_candidates": 0,
            "candidate_pool_coverage": 0.0,
        },
    }


def _load_learned_verifier_annotation(request_payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    enabled = bool(
        request_payload.get("enable_learned_verifier_annotation")
        or request_payload.get("learned_verifier_annotation")
    )
    raw_model = (
        request_payload.get("learned_verifier_model")
        or request_payload.get("learned_verifier_model_path")
        or str(DEFAULT_LEARNED_VERIFIER_MODEL)
    )
    report: dict[str, Any] = {
        "enabled": enabled,
        "policy": "annotation_only",
        "model_path": str(raw_model) if raw_model else None,
        "model_loaded": False,
        "input_routes": 0,
        "annotated_routes": 0,
    }
    if not enabled:
        return None, report

    model_path = Path(str(raw_model))
    if not model_path.exists():
        report["error"] = "model_not_found"
        return None, report
    try:
        learned = load_learned_verifier(model_path)
    except Exception as exc:  # pragma: no cover - defensive for web robustness
        report["error"] = f"model_load_failed:{type(exc).__name__}"
        return None, report
    report["model_path"] = learned["path"]
    report["model_loaded"] = True
    return learned, report


def _learned_verifier_route_annotation(
    learned_verifier: dict[str, Any] | None,
    *,
    target_smiles: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    if learned_verifier is None:
        return {
            "available": False,
            "policy": "annotation_only",
        }
    cascade = {
        "target": target_smiles,
        "target_smiles": target_smiles,
        "steps": steps,
        "stage_partition": [f"stage_{idx + 1}" for idx, _step in enumerate(steps)],
        "metadata": {},
    }
    payload = predict_learned_verifier(learned_verifier, cascade, target_smiles=target_smiles)
    payload["available"] = True
    payload["policy"] = "annotation_only"
    return payload


def _terminal_stock_status(steps: list[dict[str, Any]]) -> dict[str, bool | None]:
    products = {str(step.get("product") or "") for step in steps if step.get("product")}
    terminal: dict[str, bool | None] = {}
    fallback: dict[str, bool | None] = {}
    for step in steps:
        for smi, ok in (step.get("stock_status") or {}).items():
            text = str(smi or "")
            if not text:
                continue
            fallback.setdefault(text, ok)
            if text not in products:
                terminal[text] = ok
    return terminal or fallback


def _route_summary(route: dict[str, Any]) -> dict[str, Any]:
    metrics = route.get("metrics") or {}
    progress = metrics.get("retrosynthesis_progress") or {}
    return {
        "n_steps": route.get("n_steps"),
        "score": route.get("score"),
        "route_solved": metrics.get("route_solved"),
        "professional_solved": metrics.get("professional_solved"),
        "strict_stock_solve": metrics.get("strict_stock_solve"),
        "main_chain_reduction": progress.get("main_chain_reduction"),
        "largest_leaf_reduction": progress.get("largest_leaf_reduction"),
    }


def _route_materialization_admission_summary(
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    audits = [
        dict((route.get("raw_backend_metadata") or {}).get("route_materialization_admission") or {})
        for route in routes
        if isinstance(route, dict)
        and isinstance(route.get("raw_backend_metadata"), dict)
        and isinstance(
            (route.get("raw_backend_metadata") or {}).get(
                "route_materialization_admission"
            ),
            dict,
        )
    ]
    rejected = [audit for audit in audits if audit.get("accepted") is not True]
    return {
        "schema_version": "chemenzy_web_route_materialization_admission_summary.v1",
        "audited_route_count": len(audits),
        "accepted_route_count": len(audits) - len(rejected),
        "rejected_route_count": len(rejected),
        "rejected_routes": rejected[:50],
        "host_audit_authority": True,
        "raw_solved_is_not_host_solved_authority": True,
    }


def _route_signature(route: dict[str, Any]) -> str:
    return "|".join(step.get("reaction_smiles") or "" for step in route.get("steps") or [])


def _reaction_type(step: RouteStepCandidate) -> str:
    if step.enzyme_ec_annotations:
        return "enzymatic"
    source = str(step.source_model or "")
    if source.startswith("[") or ">>" in source:
        return "template"
    return source or "reaction"


def _display_source(step: RouteStepCandidate) -> str:
    if step.enzyme_ec_annotations:
        return "CascadePlanner enzyme module"
    source = str(step.source_model or "")
    if source.startswith("[") or ">>" in source:
        return "Template proposal"
    if source in {"", "ChemEnzyRetroPlanner"}:
        return "CascadePlanner"
    return source


def _top_condition_prediction(step: RouteStepCandidate) -> dict[str, Any]:
    for row in step.condition_predictions or []:
        if isinstance(row, dict):
            return row
    return {}


def _top_enzyme_annotation(step: RouteStepCandidate) -> dict[str, Any]:
    for row in step.enzyme_ec_annotations or []:
        if isinstance(row, dict):
            return row
    return {}


def _enzyme_quality(step: RouteStepCandidate) -> dict[str, Any]:
    metadata = step.raw_backend_metadata if isinstance(step.raw_backend_metadata, dict) else {}
    template = metadata.get("template") if isinstance(metadata.get("template"), dict) else {}
    quality = template.get("autoplanner_enzyme_quality_v1") if isinstance(template.get("autoplanner_enzyme_quality_v1"), dict) else {}
    if quality:
        out = dict(quality)
        out.setdefault("origin", "template")
        return out
    cascade_cost = metadata.get("cascade_cost") if isinstance(metadata.get("cascade_cost"), dict) else {}
    enzyme_like = bool(step.enzyme_ec_annotations) or _source_is_enzyme_like(step.source_model)
    if not enzyme_like and not cascade_cost:
        return {}
    sp_payload = template.get("enzyme_sp_verifier_v1") if isinstance(template.get("enzyme_sp_verifier_v1"), dict) else {}
    evidence = template.get("evidence") if isinstance(template.get("evidence"), dict) else {}
    if cascade_cost:
        evidence = dict(evidence)
        evidence.setdefault("cascade_cost_available", True)
    ec_numbers = [
        str(row.get("ec_number") or row.get("EC Number") or "")
        for row in step.enzyme_ec_annotations or []
        if isinstance(row, dict) and (row.get("ec_number") or row.get("EC Number"))
    ]
    out = evaluate_enzyme_step_quality(
        product_smiles=step.product_smiles,
        reactants=step.reactant_smiles,
        source_model=step.source_model,
        template={
            "model_full_name": step.source_model,
            "source": step.source_model,
            "evidence": evidence,
            "enzyme_sp_verifier_v1": sp_payload,
        },
        sp_payload=sp_payload,
        ec_numbers=ec_numbers,
    )
    flags = list(out.get("flags") or [])
    flags.append("native_or_posthoc_derived_quality")
    if cascade_cost:
        flags.append("cascade_costed_during_search")
    out.update({
        "origin": "derived_from_selected_step",
        "search_time_costed": bool(cascade_cost),
        "flags": list(dict.fromkeys(flags)),
        "cascade_adjustment": cascade_cost.get("cascade_adjustment"),
    }
    )
    return out


def _source_is_enzyme_like(source_model: str) -> bool:
    text = str(source_model or "").lower()
    return any(token in text for token in ("enzyme", "enzymatic", "bionav", "bkms", "biocatalysis", "ecreact", "ec_"))


def _condition_value(row: dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        low = key.lower()
        if low in lower and lower[low] not in (None, ""):
            return lower[low]
    return None


def _condition_notes(condition: dict[str, Any], enzyme: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    temp = _condition_value(condition, "Temperature", "temperature", "temperature_c")
    ph = _condition_value(condition, "pH", "ph")
    solvent = _condition_value(condition, "Solvent", "solvent")
    reagent = _condition_value(condition, "Reagent", "reagent")
    catalyst = _condition_value(condition, "Catalyst", "catalyst")
    score = _condition_value(condition, "Score", "score", "confidence")
    if temp not in (None, ""):
        notes.append(f"T={temp} C")
    if ph not in (None, ""):
        notes.append(f"pH={ph}")
    if solvent:
        notes.append(f"solvent={solvent}")
    if reagent:
        notes.append(f"reagent={reagent}")
    if catalyst:
        notes.append(f"catalyst={catalyst}")
    if score not in (None, ""):
        notes.append(f"condition_score={score}")
    if enzyme.get("ec_number"):
        notes.append(f"enzyme_ec={enzyme.get('ec_number')}")
    if enzyme.get("confidence") not in (None, ""):
        notes.append(f"enzyme_confidence={enzyme.get('confidence')}")
    return notes


def _reaction_summary(reaction_type: str, step: RouteStepCandidate) -> str:
    cls = str(reaction_type or "unknown reaction")
    reactants = " + ".join(step.reactant_smiles or [])
    if reactants:
        return f"{cls} proposal connects precursor(s) {reactants} to the displayed product."
    return f"{cls} proposal from ChemEnzyRetroPlanner; inspect reaction SMILES before mechanism assignment."


def _reaction_principle(reaction_type: str) -> str:
    key = str(reaction_type or "").lower().replace("_", " ")
    rules = [
        ("enzym", "Predicted enzymatic transformation; EC assignment, if present, is a catalyst-family hypothesis."),
        ("hydrolysis", "Hydrolysis cleaves a labile bond with water or aqueous conditions."),
        ("ester", "Esterification or transesterification forms or exchanges an ester linkage."),
        ("acyl", "Acyl transfer installs or exchanges an acyl group on a nucleophile."),
        ("coupling", "Coupling joins molecular fragments through a newly formed bond."),
        ("c-c", "C-C bond formation links two carbon fragments."),
        ("oxid", "Oxidation raises oxidation state or introduces oxygen-containing functionality."),
        ("reduct", "Reduction lowers oxidation state or adds hydride/hydrogen equivalents."),
        ("amin", "Amination forms or exchanges a C-N bond."),
        ("alkyl", "Alkylation installs an alkyl substituent through substitution or transfer chemistry."),
        ("deprotect", "Deprotection removes a protecting group to reveal a functional handle."),
        ("isomer", "Isomerization rearranges stereochemistry or connectivity without major atom-count change."),
        ("template", "Template-matched retrosynthetic disconnection; mechanism depends on the underlying reaction template."),
    ]
    for token, text in rules:
        if token in key:
            return text
    return "Mechanistic hypothesis only: use reaction SMILES, template/source, and predicted conditions for chemist review."


def _reactant_change_notes(reactants: list[str]) -> list[str]:
    if len(reactants) <= 1:
        return []
    return [f"multiple precursor/coupling partners: {' . '.join(reactants)}"]


def _atom_change_notes(rxn_smiles: str) -> dict[str, Any]:
    if ">>" not in str(rxn_smiles or ""):
        return {"notes": []}
    lhs, rhs = str(rxn_smiles).split(">>", 1)
    reactants = [part for part in lhs.split(".") if part]
    products = [part for part in rhs.split(".") if part]
    reactant_heavy = sum(_heavy_atoms(smi) for smi in reactants)
    product_heavy = sum(_heavy_atoms(smi) for smi in products)
    delta = product_heavy - reactant_heavy
    notes = []
    if delta > 0:
        notes.append(f"product has {delta} more heavy atom(s) than listed reactants")
    elif delta < 0:
        notes.append(f"product has {-delta} fewer heavy atom(s) than listed reactants")
    return {
        "reactant_heavy_atoms": reactant_heavy,
        "product_heavy_atoms": product_heavy,
        "heavy_atom_delta": delta,
        "notes": notes,
    }


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def _failure_analysis(
    result: Any,
    request_payload: dict[str, Any],
    config: RouteSearchConfig,
    *,
    vendor_root: Path | None = None,
) -> dict[str, Any]:
    failures = [failure.to_dict() for failure in result.failures]
    if not failures:
        return {"available": False, "diagnosis": [], "retry_suggestions": []}
    target = str(result.target_smiles or request_payload.get("target_smiles") or "")
    target_heavy = _heavy_atoms(target)
    stock_membership = _target_stock_membership(
        target,
        config.stock_names,
        vendor_root=vendor_root,
        stock_paths=dict(config.search_flags.get("stock_paths") or {}),
    )
    categories = [str(row.get("category") or "") for row in failures]
    diagnosis: list[str] = []
    suggestions: list[str] = []
    if "no_route_found" in categories:
        diagnosis.extend(
            [
                "ChemEnzy returned no stock-closed successful route; product-audit post-filter did not run.",
                "The backend does not expose the failed partial search tree in this Web path, so this is a search-level diagnosis rather than a step-level proof.",
            ]
        )
        if stock_membership.get("target_in_selected_stock"):
            diagnosis.append(
                "Target itself is present in the selected stock, but ChemEnzy excludes the target from stock to avoid a trivial zero-step solution."
            )
            suggestions.append("report this as target_in_stock_but_excluded; it is a purchasability signal, not a synthetic route")
            suggestions.append("use a route-review mode that shows target-in-stock separately from retrosynthesis success")
        if target_heavy >= 38:
            diagnosis.append(f"Target is large for stock-closed search ({target_heavy} heavy atoms); more iterations/depth or broader stock may be needed.")
        if config.max_iterations < 100:
            suggestions.append("increase chem_enzy_iterations to 100-200")
        if config.expansion_topk < 150:
            suggestions.append("increase chem_enzy_expansion_topk to 150-200")
        if config.max_depth < 16:
            suggestions.append("increase max_steps to 16-20")
        suggestions.append("try risk_guarded post-filter only after routes exist; filtering cannot recover no-route cases")
    if any(cat == "backend_initialization_failed" for cat in categories):
        suggestions.append("disable optional condition/enzyme annotation and retry")
    if any(cat == "backend_annotation_failed" for cat in categories):
        diagnosis.append("Routes may be valid, but optional condition/enzyme annotation failed.")
        suggestions.append("rerun without condition/enzyme annotation if route search is the priority")
    return {
        "available": True,
        "target_heavy_atoms": target_heavy or None,
        "failure_categories": categories,
        "diagnosis": diagnosis,
        "retry_suggestions": _dedupe(suggestions),
        "search_config": {
            "preset": request_payload.get("search_preset", "quick"),
            "max_depth": config.max_depth,
            "iterations": config.max_iterations,
            "expansion_topk": config.expansion_topk,
            "condition_prediction_enabled": bool(request_payload.get("enable_condition_prediction", False)),
            "enzyme_assignment_enabled": bool(request_payload.get("enable_enzyme_assignment", False)),
            "chem_enzy_onmt_tokenizer": config.search_flags.get("chem_enzy_onmt_tokenizer", "char"),
            "one_step_models": config.one_step_models,
            "stock_names": config.stock_names,
            "target_stock_membership": stock_membership,
        },
    }


def _target_stock_membership(
    target_smiles: str,
    stock_names: list[str],
    *,
    vendor_root: Path | None,
    stock_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    target_mol = Chem.MolFromSmiles(str(target_smiles or ""))
    if target_mol is None or vendor_root is None:
        return {"available": False, "target_in_selected_stock": False}
    canonical = Chem.MolToSmiles(target_mol, isomericSmiles=True)
    selected_paths = _stock_paths(
        vendor_root,
        stock_names,
        stock_paths=stock_paths,
    )
    hits: list[str] = []
    checked: list[str] = []
    for stock_name, path in selected_paths.items():
        if not path.exists() or not path.is_file():
            continue
        checked.append(stock_name)
        try:
            if _smiles_in_stock_file(canonical, path):
                hits.append(stock_name)
        except OSError:
            continue
    return {
        "available": bool(checked),
        "canonical_target_smiles": canonical,
        "checked_stocks": checked,
        "hit_stocks": hits,
        "target_in_selected_stock": bool(hits),
        "note": "ChemEnzy uses exclude_target=true, so an exact stock hit is not returned as a zero-step route.",
    }


def _stock_paths(
    vendor_root: Path,
    stock_names: list[str],
    *,
    stock_paths: dict[str, str] | None = None,
) -> dict[str, Path]:
    config_path = vendor_root / "retro_planner" / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    stock_cfg = cfg.get("stocks") or {}
    base = vendor_root / "retro_planner"
    selected = set(stock_names or [])
    out: dict[str, Path] = {
        str(name): Path(str(path)).expanduser().resolve()
        for name, path in dict(stock_paths or {}).items()
        if (not selected or str(name) in selected) and str(path).strip()
    }
    for name, rel in stock_cfg.items():
        if selected and name not in selected:
            continue
        if str(name) in out:
            continue
        path = Path(str(rel))
        out[str(name)] = path if path.is_absolute() else base / path
    return out


def _smiles_in_stock_file(canonical: str, path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            first = line.strip().split(",", 1)[0].strip()
            if first == canonical:
                return True
    return False


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_positive_int(value: Any, *, hi: int) -> int | None:
    if value is None or value == "":
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    if out <= 0:
        return None
    return min(out, hi)


def _configured_output_route_limit(
    request_payload: dict[str, Any],
    config: RouteSearchConfig,
) -> int | None:
    configured = dict(config.search_flags or {}).get("max_output_routes")
    if configured is None:
        configured = request_payload.get("max_routes")
    return _optional_positive_int(configured, hi=100)


def _as_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(out, lo), hi)


if __name__ == "__main__":
    main()
