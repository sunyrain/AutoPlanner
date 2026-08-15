"""Compile target-solve requests and optional evidence providers."""
from __future__ import annotations

from typing import Any, Mapping

from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.application.unified_campaign_spec import TargetConstraints
from cascade_planner.interfaces.target_solver import (
    DEFAULT_TARGET_DIRECTOR_MODEL,
    TargetSolveConfig,
)
from cascade_planner.interfaces.target_runtime_dependencies import (
    SYNTHEX_MATCHED_PROFILE_DEFAULTS,
    TARGET_PROFILE_DEFAULTS,
    inventory_snapshot_builder,
)


def solve_target_request(gateway: Any, payload: dict[str, Any]) -> dict[str, Any]:
    max_visual_invocations = _int(payload, "max_visual_invocations", 0)
    execution_profile = str(payload.get("execution_profile") or "standard")
    profile_defaults = TARGET_PROFILE_DEFAULTS.get(execution_profile, TARGET_PROFILE_DEFAULTS["standard"])
    matched = SYNTHEX_MATCHED_PROFILE_DEFAULTS
    evidence_connector = _web_evidence_connector(gateway, payload)
    visual_provider = _web_visual_provider(gateway, payload, enabled=max_visual_invocations > 0)
    inventory_builder = inventory_snapshot_builder(payload)
    return gateway.solve_target(
        target_name=str(payload.get("target_name") or "blind target"),
        target_smiles=str(payload.get("target_smiles") or ""),
        run_id=str(payload.get("run_id") or "") or None,
        resume=_bool(payload, "resume", False),
        evidence_connector=evidence_connector,
        visual_evidence_provider=visual_provider,
        inventory_snapshot_builder=inventory_builder,
        constraints=_target_constraints(payload),
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=_int(payload, "minimum_complete_routes", 2),
            minimum_edge_proof_level=_int(payload, "minimum_edge_proof_level", 2),
            minimum_independent_source_groups=_int(
                payload, "minimum_source_groups", 2
            ),
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
                payload.get(
                    "max_model_wall_time_s",
                    matched["max_model_wall_time_s"],
                )
            ),
            max_visual_invocations=max_visual_invocations,
            max_accepted_expansions=_int(
                payload, "max_accepted_expansions", matched["max_accepted_expansions"]
            ),
            max_attempt_runs=_int(payload, "max_attempt_runs", matched["max_attempt_runs"]),
            max_prompt_context_bytes=_int(
                payload, "max_prompt_context_bytes", matched["max_prompt_context_bytes"]
            ),
        ),
        config=TargetSolveConfig(
            model=str(payload.get("model") or DEFAULT_TARGET_DIRECTOR_MODEL),
            reasoning_effort=str(
                payload.get("reasoning_effort") or matched["reasoning_effort"]
            ),
            execution_profile=execution_profile,
            strategy_search_profile=str(
                payload.get("strategy_search_profile")
                or matched["strategy_search_profile"]
            ),
            strategy_branch_count=_int(
                payload, "strategy_branch_count", matched["strategy_branches"]
            ),
            max_node_expansions_per_branch=_int(
                payload,
                "max_node_expansions_per_branch",
                matched["node_expansions_per_branch"],
            ),
            max_route_local_repair_rounds=_int(
                payload,
                "max_route_local_repair_rounds",
                matched["route_local_repair_rounds"],
            ),
            max_node_prompt_bytes=_int(
                payload, "max_node_prompt_bytes", matched["max_node_prompt_bytes"]
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
                    matched["max_model_wall_time_s"],
                )
            ),
        ),
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
