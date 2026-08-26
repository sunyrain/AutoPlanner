"""Compile target-solve requests and optional evidence providers."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from threading import Event
from typing import Any, Mapping

import yaml

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.unified_campaign_spec import TargetConstraints
from cascade_planner.interfaces.target_solver import (
    DEFAULT_TARGET_DIRECTOR_MODEL,
    TargetSolveConfig,
    _is_paper_reach_profile,
    _resolve_execution_config,
)
from cascade_planner.interfaces.aizynthfinder_sidecar import (
    AiZynthFinderSidecarConfig,
)
from cascade_planner.interfaces.live_stock import FrozenBenchmarkStockIndex
from cascade_planner.interfaces.target_runtime_dependencies import (
    SYNTHEX_MATCHED_PROFILE_DEFAULTS,
    TARGET_PROFILE_DEFAULTS,
    inventory_snapshot_builder,
)


def solve_target_request(
    gateway: Any,
    payload: dict[str, Any],
    *,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    max_visual_invocations = _int(payload, "max_visual_invocations", 0)
    execution_profile = str(payload.get("execution_profile") or "standard")
    paper_protocol = _is_paper_reach_profile(execution_profile)
    profile_defaults = TARGET_PROFILE_DEFAULTS.get(execution_profile, TARGET_PROFILE_DEFAULTS["standard"])
    matched = SYNTHEX_MATCHED_PROFILE_DEFAULTS
    default_model_wall_time_s = float(
        matched["max_model_wall_time_s"]
        if paper_protocol
        else profile_defaults["max_model_wall_time_s"]
    )
    evidence_connector = (
        None if paper_protocol else _web_evidence_connector(gateway, payload)
    )
    visual_provider = (
        None
        if paper_protocol
        else _web_visual_provider(gateway, payload, enabled=max_visual_invocations > 0)
    )
    inventory_builder = inventory_snapshot_builder(payload)
    stock_catalog_builder = _request_stock_catalog_builder(
        payload,
        execution_profile=execution_profile,
    )
    return gateway.solve_target(
        target_name=str(payload.get("target_name") or "blind target"),
        target_smiles=str(payload.get("target_smiles") or ""),
        run_id=str(payload.get("run_id") or "") or None,
        resume=_bool(payload, "resume", False),
        evidence_connector=evidence_connector,
        visual_evidence_provider=visual_provider,
        stock_catalog_builder=stock_catalog_builder,
        inventory_snapshot_builder=inventory_builder,
        cancel_event=cancel_event,
        constraints=_target_constraints(payload),
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=_int(payload, "minimum_complete_routes", 2),
            minimum_edge_proof_level=_int(payload, "minimum_edge_proof_level", 2),
            minimum_independent_source_groups=_int(payload, "minimum_source_groups", 2),
            stock_boundary=str(payload.get("stock_boundary") or "benchmark_search"),
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=_int(
                payload,
                "max_model_invocations",
                matched["max_model_invocations"],
            ),
            max_total_input_tokens=_int(
                payload, "max_input_tokens", matched["max_input_tokens"]
            ),
            max_total_output_tokens=_int(
                payload, "max_output_tokens", matched["max_output_tokens"]
            ),
            max_total_wall_time_s=float(
                payload.get("max_model_wall_time_s", default_model_wall_time_s)
            ),
            max_visual_invocations=max_visual_invocations,
            max_accepted_expansions=_int(
                payload, "max_accepted_expansions", matched["max_accepted_expansions"]
            ),
            max_attempt_runs=_int(payload, "max_attempt_runs", matched["max_attempt_runs"]),
            max_prompt_context_bytes=_int(payload, "max_prompt_context_bytes", matched["max_prompt_context_bytes"]),
        ),
        config=_resolve_execution_config(TargetSolveConfig(
            run_scope=str(payload.get("run_scope") or "blind"),
            model=str(payload.get("model") or DEFAULT_TARGET_DIRECTOR_MODEL),
            reasoning_effort=str(payload.get("reasoning_effort") or matched["reasoning_effort"]),
            execution_profile=execution_profile,
            strategy_search_profile=str(payload.get("strategy_search_profile") or matched["strategy_search_profile"]),
            strategy_portfolio_mode=str(
                payload.get("strategy_portfolio_mode") or "auto"
            ),
            strategy_branch_count=_int(
                payload, "strategy_branch_count", matched["strategy_branches"]
            ),
            strategy_branch_workers=_int(
                payload,
                "strategy_branch_workers",
                matched["strategy_branch_workers"],
            ),
            stop_on_first_stock_closed_branch=_bool(
                payload,
                "stop_on_first_stock_closed_branch",
                bool(matched["stop_on_first_stock_closed_branch"]),
            ),
            max_node_expansions_per_branch=_int(payload, "max_node_expansions_per_branch", matched["node_expansions_per_branch"]),
            max_reactionjson_candidates_per_node=_int(
                payload,
                "max_reactionjson_candidates_per_node",
                matched["reactionjson_candidates_per_node"],
            ),
            max_route_local_repair_rounds=_int(payload, "max_route_local_repair_rounds", matched["route_local_repair_rounds"]),
            max_node_prompt_bytes=_int(
                payload, "max_node_prompt_bytes", matched["max_node_prompt_bytes"]
            ),
            max_node_call_timeout_s=float(
                payload.get("max_node_call_timeout_s", matched["node_call_timeout_s"])
            ),
            critic_call_timeout_s=float(
                payload.get("critic_call_timeout_s", matched["critic_call_timeout_s"])
            ),
            require_complete_route_json=_bool(
                payload,
                "require_complete_route_json",
                bool(matched["route_builder_complete_linear_route"]),
            ),
            allow_editor_route_mutations=_bool(
                payload,
                "allow_editor_route_mutations",
                bool(matched["editor_route_mutations"]),
            ),
            objective_mode=str(payload.get("objective_mode") or "scientific_proof"),
            delivery_boundary=str(
                payload.get("delivery_boundary") or "stock_result"
            ),
            use_coordinator=_bool(payload, "use_coordinator", False),
            enable_web_search=_bool(payload, "enable_web_search", True),
            enable_initial_director_web_search=_bool(
                payload,
                "enable_initial_director_web_search",
                execution_profile in {"standard", "proof"},
            ),
            enable_target_identity=_bool(payload, "enable_target_identity", True),
            resolve_named_target_identity=_bool(
                payload, "resolve_named_target_identity", True
            ),
            blind_audit_root=str(payload.get("blind_audit_root") or ""),
            blind_audit_allowed_paths=tuple(
                _string_list(payload.get("blind_audit_allowed_paths"))
            ),
            enable_replan=_bool(payload, "enable_replan", True),
            enable_live_benchmark_stock=_bool(
                payload, "enable_live_benchmark_stock", True
            ),
            enable_builtin_patent_evidence=(
                evidence_connector is None
                and _bool(payload, "enable_auto_patent_evidence", True)
            ),
            enable_patent_self_evolution=_bool(
                payload, "enable_patent_self_evolution", True
            ),
            enable_chemenzy=_bool(payload, "enable_chemenzy", True),
            enable_target_chemenzy_baseline=_bool(
                payload,
                "enable_target_chemenzy_baseline",
                bool(matched["target_chemenzy_baseline"]),
            ),
            enable_guided_chemenzy=_bool(payload, "enable_guided_chemenzy", True),
            aizynthfinder_python_executable=str(
                payload.get("aizynthfinder_python_executable") or ""
            ),
            aizynthfinder_config_path=str(
                payload.get("aizynthfinder_config_path") or ""
            ),
            aizynthfinder_runtime_root=str(
                payload.get("aizynthfinder_runtime_root") or ""
            ),
            aizynthfinder_short_tail_mode=str(
                payload.get("aizynthfinder_short_tail_mode") or "short_tail"
            ),
            native_short_tail_engine=str(
                payload.get("native_short_tail_engine") or "auto"
            ),
            enable_chemenzy_condition_prediction=_bool(
                payload, "enable_chemenzy_condition_prediction", True
            ),
            enable_chemenzy_enzyme_assignment=_bool(
                payload, "enable_chemenzy_enzyme_assignment", True
            ),
            enable_enzyme_coverage_sidecar=_bool(
                payload, "enable_enzyme_coverage_sidecar", True
            ),
            chemenzy_env_prefix=str(payload.get("chemenzy_env_prefix") or ""),
            self_evo_library_path=str(payload.get("self_evo_library_path") or ""),
            max_atom_mapping_reactions=_int(
                payload,
                "max_atom_mapping_reactions",
                matched["max_atom_mapping_reactions"],
            ),
            max_live_stock_molecules=_int(
                payload,
                "max_live_stock_molecules",
                matched["max_stock_molecules"],
            ),
            max_patent_sources=_int(payload, "max_patent_sources", 3),
            max_self_evo_template_candidates=_int(
                payload, "max_self_evo_template_candidates", 12
            ),
            max_total_tasks=_int(
                payload, "max_total_tasks", matched["max_total_tasks"]
            ),
            max_evidence_tasks=_int(payload, "max_evidence_tasks", 64),
            max_stock_tasks=_int(payload, "max_stock_tasks", 128),
            max_validation_tasks=_int(payload, "max_validation_tasks", 128),
            max_program_tasks=_int(payload, "max_program_tasks", 64),
            max_experiment_tasks=_int(payload, "max_experiment_tasks", 32),
            max_run_wall_time_s=float(
                payload.get("max_run_wall_time_s", 7_200.0)
            ),
            provider_route_reserve=_int(
                payload, "provider_route_reserve", 16
            ),
            host_route_portfolio=_int(
                payload, "host_route_portfolio", 16
            ),
            display_route_limit=_int(payload, "display_route_limit", 4),
            max_chemenzy_routes=(
                int(payload["max_chemenzy_routes"])
                if payload.get("max_chemenzy_routes") is not None
                else None
            ),
            max_chemenzy_steps=_int(
                payload, "max_chemenzy_steps", matched["short_tail_steps"]
            ),
            max_chemenzy_iterations=_int(
                payload,
                "max_chemenzy_iterations",
                matched["short_tail_iterations"],
            ),
            chemenzy_expansion_topk=_int(
                payload, "chemenzy_expansion_topk", profile_defaults["topk"]
            ),
            chemenzy_timeout_s=float(
                payload.get("chemenzy_timeout_s", matched["short_tail_timeout_s"])
            ),
            chemenzy_search_preset=str(
                payload.get("chemenzy_search_preset")
                or ("thorough" if execution_profile == "proof" else "standard")
            ),
            chemenzy_seed=_int(payload, "chemenzy_seed", 0),
            chemenzy_pandarallel_workers=_int(
                payload,
                "chemenzy_pandarallel_workers",
                profile_defaults["workers"],
            ),
            max_guided_chemenzy_frontiers=(
                _int(payload, "max_guided_chemenzy_frontiers", 0)
                if payload.get("max_guided_chemenzy_frontiers") not in {None, ""}
                else None
            ),
            max_guided_chemenzy_iterations=_int(
                payload,
                "max_guided_chemenzy_iterations",
                matched["short_tail_iterations"],
            ),
            guided_chemenzy_timeout_s=float(
                payload.get(
                    "guided_chemenzy_timeout_s", matched["short_tail_timeout_s"]
                )
            ),
            max_visual_evidence_pages=_int(payload, "max_visual_evidence_pages", 6),
            minimum_planning_route_steps=_int(
                payload, "minimum_planning_route_steps", 0
            ),
            max_director_output_tokens=_int(
                payload,
                "max_director_output_tokens",
                18_000 if execution_profile == "proof" else 7_000,
            ),
            max_director_wall_time_s=float(
                payload.get(
                    "max_director_wall_time_s",
                    (
                        matched["max_model_wall_time_s"]
                        if paper_protocol
                        else profile_defaults["max_director_wall_time_s"]
                    ),
                )
            ),
        )),
    )


def _request_stock_catalog_builder(
    payload: Mapping[str, Any],
    *,
    execution_profile: str,
) -> FrozenBenchmarkStockIndex | None:
    """Materialize the stock authority required by Strategy Builder.

    Explicit benchmark requests remain fail-closed on a caller-supplied
    content hash. Interactive paper-profile runs bind the stock selected by
    the already-resolved AiZynthFinder configuration and record its measured
    content identity in the campaign stock oracle.
    """

    if str(payload.get("stock_boundary") or "benchmark_search") != "benchmark_search":
        return None
    explicit_path = str(payload.get("benchmark_stock_index") or "").strip()
    explicit_sha256 = str(
        payload.get("benchmark_stock_index_sha256") or ""
    ).strip()
    catalog_name = str(payload.get("benchmark_stock_name") or "").strip()
    if explicit_path:
        if not explicit_sha256:
            raise ValueError("benchmark_stock_index_sha256_required")
        return _cached_frozen_stock_index(
            str(Path(explicit_path).expanduser().resolve()),
            explicit_sha256,
            catalog_name,
        )
    if explicit_sha256 or catalog_name:
        raise ValueError("benchmark_stock_index_path_required")
    if (
        str(payload.get("run_scope") or "blind") != "interactive"
        or not _is_paper_reach_profile(execution_profile)
        or not _bool(payload, "enable_live_benchmark_stock", True)
    ):
        return None

    binding = AiZynthFinderSidecarConfig(
        python_executable=str(
            payload.get("aizynthfinder_python_executable") or ""
        ),
        config_path=str(payload.get("aizynthfinder_config_path") or ""),
        runtime_root=str(payload.get("aizynthfinder_runtime_root") or ""),
        mode=str(payload.get("aizynthfinder_short_tail_mode") or "short_tail"),
    )
    stock_name, stock_path = _strategy_stock_from_aizynthfinder_config(
        binding.resolved_config(),
        runtime_root=binding.resolved_runtime_root(),
    )
    stat = stock_path.stat()
    measured_sha256 = _cached_file_sha256(
        str(stock_path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )
    return _cached_frozen_stock_index(
        str(stock_path),
        measured_sha256,
        stock_name,
    )


def _strategy_stock_from_aizynthfinder_config(
    config_path: Path,
    *,
    runtime_root: Path,
) -> tuple[str, Path]:
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("aizynthfinder_strategy_config_unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("aizynthfinder_strategy_config_invalid")
    raw_stock = value.get("stock")
    if not isinstance(raw_stock, Mapping) or not raw_stock:
        raise ValueError("aizynthfinder_strategy_stock_not_configured")
    named = [
        (str(name), dict(config))
        for name, config in raw_stock.items()
        if isinstance(config, Mapping) and str(config.get("path") or "").strip()
    ]
    preferred = [row for row in named if row[0] == "paper_zinc_emolecules"]
    selected = preferred or named
    if len(selected) != 1:
        raise ValueError("aizynthfinder_strategy_stock_binding_ambiguous")
    name, config = selected[0]
    raw_path = Path(str(config.get("path") or "")).expanduser()
    stock_path = (
        raw_path if raw_path.is_absolute() else Path(runtime_root) / raw_path
    ).resolve()
    if not stock_path.is_file():
        raise ValueError(f"aizynthfinder_strategy_stock_index_missing:{stock_path}")
    return name, stock_path


@lru_cache(maxsize=8)
def _cached_file_sha256(path: str, size: int, mtime_ns: int) -> str:
    resolved = Path(path)
    before = resolved.stat()
    if int(before.st_size) != size or int(before.st_mtime_ns) != mtime_ns:
        raise ValueError("benchmark_stock_index_changed_before_hash")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    after = resolved.stat()
    if int(after.st_size) != size or int(after.st_mtime_ns) != mtime_ns:
        raise ValueError("benchmark_stock_index_changed_during_hash")
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _cached_frozen_stock_index(
    path: str,
    expected_sha256: str,
    catalog_name: str,
) -> FrozenBenchmarkStockIndex:
    return FrozenBenchmarkStockIndex(
        path,
        expected_sha256=expected_sha256,
        catalog_name=catalog_name,
    )


def _web_evidence_connector(gateway: Any, payload: Mapping[str, Any]) -> Any:
    paths = getattr(gateway, "paths", None)
    if paths is None:
        return None
    connectors = []
    if _bool(dict(payload), "enable_auto_patent_evidence", True):
        from cascade_planner.interfaces.patent_evidence import (
            BuiltinPatentEvidenceConfig,
            build_builtin_patent_evidence_connector,
        )

        connectors.append(
            build_builtin_patent_evidence_connector(
                BuiltinPatentEvidenceConfig(
                    cache_dir=paths.external_data_root / "patent-evidence",
                    seed_publications=tuple(
                        _string_list(payload.get("patent_publications"))
                    ),
                    max_patents=_int(dict(payload), "max_patent_sources", 3),
                    max_validated_edges=_int(
                        dict(payload),
                        "max_atom_mapping_reactions",
                        SYNTHEX_MATCHED_PROFILE_DEFAULTS[
                            "max_atom_mapping_reactions"
                        ],
                    ),
                )
            )
        )
    if _bool(dict(payload), "enable_auto_literature_evidence", True):
        from cascade_planner.interfaces.literature_evidence import (
            BuiltinLiteratureEvidenceConfig,
            build_builtin_literature_evidence_connector,
        )

        connectors.append(
            build_builtin_literature_evidence_connector(
                BuiltinLiteratureEvidenceConfig(
                    cache_dir=paths.external_data_root / "literature-evidence",
                    authorized_proxy_output_dir=str(
                        payload.get("authorized_proxy_output_dir") or ""
                    ),
                    seed_dois=tuple(_string_list(payload.get("literature_dois"))),
                    max_sources=_int(dict(payload), "max_literature_sources", 4),
                    max_visual_pages=_int(
                        dict(payload), "max_visual_evidence_pages", 6
                    ),
                    auto_fetch_restricted_sources=_bool(
                        dict(payload),
                        "auto_fetch_restricted_literature",
                        True,
                    ),
                    auto_fetch_timeout_s=float(
                        _int(dict(payload), "literature_browser_timeout_s", 180)
                    ),
                    auto_fetch_max_items=_int(
                        dict(payload),
                        "max_literature_sources",
                        4,
                    ),
                )
            )
        )
    if len(connectors) == 1:
        return connectors[0]
    if connectors:
        from cascade_planner.interfaces.live_evidence import compose_evidence_connectors

        return compose_evidence_connectors(*connectors)
    return None


def _web_visual_provider(
    gateway: Any, payload: Mapping[str, Any], *, enabled: bool
) -> Any:
    paths = getattr(gateway, "paths", None)
    if not enabled or paths is None:
        return None
    from cascade_planner.interfaces.visual_evidence import (
        CodexVisualEvidenceConfig,
        build_codex_visual_evidence_provider,
    )

    return build_codex_visual_evidence_provider(
        CodexVisualEvidenceConfig(
            cache_dir=paths.external_data_root / "visual-evidence",
            model=str(payload.get("model") or DEFAULT_TARGET_DIRECTOR_MODEL),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            max_pages=_int(dict(payload), "max_visual_evidence_pages", 6),
        )
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("expected_string_or_list")


def _target_constraints(payload: Mapping[str, Any]) -> TargetConstraints:
    safety = payload.get("safety_limits") or {}
    if not isinstance(safety, Mapping):
        raise ValueError("safety_limits_must_be_an_object")
    max_route_steps = payload.get("max_route_steps")
    return TargetConstraints(
        forbidden_reagents=tuple(_string_list(payload.get("forbidden_reagents"))),
        max_route_steps=(
            None
            if max_route_steps in {None, ""}
            else _int(payload, "max_route_steps", 0)
        ),
        allowed_execution_domains=tuple(
            _string_list(payload.get("allowed_execution_domains"))
            or TargetConstraints().allowed_execution_domains
        ),
        safety_limits=dict(safety),
        stock_source_ids=tuple(_string_list(payload.get("stock_source_ids"))),
    )


def _int(value: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(value.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}_must_be_an_integer") from exc


def _bool(value: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = value.get(key, default)
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"{key}_must_be_a_boolean")



__all__ = ["solve_target_request"]
