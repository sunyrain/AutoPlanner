from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from threading import Event, current_thread
import time
from types import SimpleNamespace
from typing import Any

import fitz
import pytest
from PIL import Image, ImageDraw
import cascade_planner.interfaces.target_solver as target_solver_module

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.application.biocatalytic_program_contracts import (
    with_biocatalysis_program_validation_digest,
)
from cascade_planner.application.campaign_actions import CampaignActionKind
from cascade_planner.application.experimental_claim_contracts import (
    CLAIM_SEMANTICS,
    CLAIM_SET_SEMANTICS,
    experimental_claim_counts,
    with_experimental_claim_digest,
)
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisAcceptanceSpec,
    RetrosynthesisRunBudget,
)
from cascade_planner.interfaces.campaign_gateway import CampaignGateway
from cascade_planner.interfaces.target_solver import (
    TargetSolveConfig,
    _campaign_action_handler_results,
    _bind_native_search_budget,
    _bounded_scheduler_exhausted,
    _resolve_execution_config,
    _automatic_continuation_exhausted,
    _current_disposition,
    _director_outcome_allows_replan,
    _director_depth_replan_events,
    _director_topology_replan_events,
    _evidence_observations,
    _first_stock_result_is_terminal_for_run,
    _frontier_builder_extension_enabled,
    _material_replan_events,
    _planning_depth_requirement,
    _paper_matched_primary_projection,
    _program_milestones_from_stages,
    _replan_gain_audit,
    _replan_retention_audit,
    _replan_reasons,
    _replan_signal_gate,
    _search_method_projection,
    _should_retry_chemenzy_timeout,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    SequentialStrategyDirectorRunner,
)
from cascade_planner.orchestration.global_campaign_director import DirectorConfig
from cascade_planner.interfaces.validation_fork import ValidationForkConfig
from cascade_planner.application.canonical_hypergraph import molecule_identity
from cascade_planner.interfaces.patent_evidence import (
    BuiltinPatentEvidenceConfig,
    build_builtin_patent_evidence_connector,
)
from cascade_planner.runtime import AgentResult, AgentSpec, AgentState
from cascade_planner.runtime.paths import RuntimePaths


TARGET = "CCOC(C)=O"


def test_checkpoint_action_projection_keeps_resume_fields_not_full_receipt() -> None:
    large_payload = {"route": "x" * 10_000}
    stages = target_solver_module._deduplicate_stages(
        [
            {
                "stage": "campaign_action_unified_core_01",
                "status": "completed",
                "detail": {
                    "action": {
                        "action_id": "action:host_materialize:1",
                        "execution_id": "campaign-action:1",
                        "kind": "host_materialize",
                        "producer": "host_worker",
                        "resource_class": "deterministic",
                        "input_revision": 4,
                        "metadata": large_payload,
                    },
                    "decision": large_payload,
                    "outcome": {
                        "status": "completed",
                        "output_revision": 5,
                        "material_events": ["route_materialized"],
                        "handler_result": large_payload,
                    },
                    "outcome_ref": {"sha256": "a" * 64},
                    "cache_hit": False,
                },
            }
        ]
    )

    detail = stages[0]["detail"]
    assert detail["action"]["execution_id"] == "campaign-action:1"
    assert detail["outcome"]["output_revision"] == 5
    assert "handler_result" not in detail["outcome"]
    assert "decision" not in detail
    assert detail["outcome_ref"]["sha256"] == "a" * 64


def test_checkpoint_action_projection_retains_feedback_claim_authority() -> None:
    stages = target_solver_module._deduplicate_stages(
        [
            {
                "stage": "campaign_action_unified_core_01",
                "status": "completed",
                "detail": {
                    "action": {
                        "execution_id": "campaign-action:feedback",
                        "kind": "experiment_feedback_ingest",
                    },
                    "outcome": {
                        "status": "completed",
                        "handler_result": {
                            "validation_id": "validation:1",
                            "experimental_claims": {"content_sha256": "claim"},
                            "experimental_claims_oracle": {"accepted": True},
                            "unrelated_large_payload": "x" * 10_000,
                        },
                    },
                },
            }
        ]
    )

    handler = stages[0]["detail"]["outcome"]["handler_result"]
    assert handler["validation_id"] == "validation:1"
    assert handler["experimental_claims"] == {"content_sha256": "claim"}
    assert "unrelated_large_payload" not in handler


def test_final_route_critic_resume_work_tracks_exact_route_digest() -> None:
    graph = {
        "revision": 7,
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCO"},
            "molecule:ethyl": {"canonical_smiles": "CC"},
            "molecule:water": {"canonical_smiles": "O"},
        },
        "route_families": {
            "route:one": {
                "route_family_id": "route:one",
                "aliases": ["codex:sequential:family:1"],
                "edge_ids": ["edge:root"],
                "selected": True,
                "strategy_card": {"strategy_query": "disconnect C-O"},
            }
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "product_smiles": "CCO",
                "precursor_molecule_ids": ["molecule:ethyl", "molecule:water"],
                "precursor_smiles": ["CC", "O"],
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                "reactionjson_audit": {
                    "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                    "mapped_precursor_smiles": [
                        "[CH3:1][CH3:2]",
                        "[OH2:3]",
                    ],
                },
                "origin_records": [
                    {
                        "proposal_id": "step:root",
                        "canonical_route_family_ids": ["route:one"],
                    }
                ],
            }
        },
    }

    pending = target_solver_module._classify_final_route_critic_resume_work(graph)
    assert pending["has_new_work"] is True
    assert pending["route_family_ids"] == ["route:one"]

    route_sha256 = pending["work_items"][0]["route_sha256"]
    graph["route_families"]["route:one"]["chemical_critic"] = {
        "status": "viable",
        "review_state": "complete",
        "reviewed_route_sha256": route_sha256,
    }
    settled = target_solver_module._classify_final_route_critic_resume_work(graph)
    assert settled["has_new_work"] is False

    graph["edges"]["edge:root"]["condition_predictions"] = [{"reagents": ["base"]}]
    stale = target_solver_module._classify_final_route_critic_resume_work(graph)
    assert stale["has_new_work"] is True
    assert stale["work_items"][0]["route_sha256"] != route_sha256


def test_final_route_critic_resume_repairs_only_unsettled_blocking_rejects() -> None:
    graph = {
        "revision": 7,
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCO"},
            "molecule:ethyl": {"canonical_smiles": "CC"},
            "molecule:water": {"canonical_smiles": "O"},
        },
        "route_families": {
            "route:one": {
                "selected": True,
                "edge_ids": ["edge:root"],
                "aliases": ["family:one"],
                "strategy_card": {},
            }
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "precursor_molecule_ids": ["molecule:ethyl", "molecule:water"],
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "reactionjson_audit": {
                    "mapped_product_smiles": "[CH3:1][CH2:2][OH:3]",
                    "mapped_precursor_smiles": ["[CH3:1][CH3:2]", "[OH2:3]"],
                },
                "reaction_operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                "origin_records": [
                    {
                        "proposal_id": "step:root",
                        "canonical_route_family_ids": ["route:one"],
                    }
                ],
            }
        },
    }
    pending = target_solver_module._classify_final_route_critic_resume_work(graph)
    route_sha256 = pending["work_items"][0]["route_sha256"]
    route = graph["route_families"]["route:one"]
    route["chemical_critic"] = {
        "status": "reject",
        "review_state": "complete",
        "reviewed_route_sha256": route_sha256,
        "step_assessments": [{"step_id": "step:root", "verdict": "reject", "blocking": True}],
    }

    repair = target_solver_module._classify_final_route_critic_resume_work(graph)
    assert repair["has_new_work"] is True
    assert repair["work_items"] == [
        {
            "kind": "final_route_repair",
            "route_family_id": "route:one",
            "route_sha256": route_sha256,
            "repair_contract": target_solver_module._FINAL_ROUTE_REPAIR_CONTRACT,
            "reason": "final_route_critic_blocking_reject",
        }
    ]
    assert repair["work_fingerprint"] == target_solver_module._digest(repair["work_items"])

    route["chemical_critic"]["status"] = "uncertain"
    assert (
        target_solver_module._classify_final_route_critic_resume_work(graph)["has_new_work"]
        is False
    )

    route["chemical_critic"]["status"] = "reject"
    route["final_route_repair_attempts"] = [
        {
            "task_id": "route-repair:legacy:1",
            "origin_route_sha256": route_sha256,
            "repair_contract": "revision_bound_transactional_repair.v4",
            "status": "repair_unresolved",
        }
    ]
    assert (
        target_solver_module._classify_final_route_critic_resume_work(graph)["has_new_work"] is True
    )
    route["final_route_repair_attempts"] = [
        {
            "task_id": "route-repair:test:1",
            "origin_route_sha256": route_sha256,
            "repair_contract": target_solver_module._FINAL_ROUTE_REPAIR_CONTRACT,
            "status": "repair_unresolved",
        }
    ]
    assert (
        target_solver_module._classify_final_route_critic_resume_work(graph)["has_new_work"]
        is False
    )


def test_final_route_review_switches_only_after_materialized_candidate_and_recritics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {
        "route_families": {
            "route:old": {
                "selected": True,
                "chemical_critic": {
                    "status": "reject",
                    "step_assessments": [
                        {
                            "step_id": "step:bad",
                            "verdict": "reject",
                            "blocking": True,
                        }
                    ],
                },
            },
            "route:new": {"selected": False},
        }
    }

    class Store:
        def load(self):
            return graph

    service = SimpleNamespace(graph_store=Store())
    reviewed_ids: list[str] = []
    include_unselected: list[bool] = []
    switches: list[tuple[bool, bool]] = []

    def review(_service, **kwargs):
        route_id = tuple(kwargs["route_family_ids"])[0]
        reviewed_ids.append(route_id)
        include_unselected.append(bool(kwargs.get("include_unselected")))
        if route_id == "route:old":
            return [
                {
                    "route_family_id": route_id,
                    "status": "reused",
                    "critic_status": "reject",
                    "route_sha256": "old-sha",
                }
            ]
        return [
            {
                "route_family_id": route_id,
                "status": "completed",
                "critic_status": "viable",
                "route_sha256": "new-sha",
            }
        ]

    monkeypatch.setattr(
        target_solver_module,
        "_run_revision_bound_route_critics",
        review,
    )
    monkeypatch.setattr(
        target_solver_module,
        "compile_revision_bound_route_critic_context",
        lambda *_args, **_kwargs: (
            SimpleNamespace(route_family_id="route:old", route_sha256="old-sha"),
            {},
        ),
    )
    monkeypatch.setattr(
        target_solver_module,
        "_run_final_route_repair_attempt",
        lambda *_args, **_kwargs: {
            "status": "candidate_materialized",
            "candidate_route_family_id": "route:new",
            "repair_attempt": {"task_id": "repair:one"},
        },
    )

    def switch(_service, **kwargs):
        switches.append(
            (
                graph["route_families"]["route:old"]["selected"],
                graph["route_families"]["route:new"]["selected"],
            )
        )
        graph["route_families"]["route:old"]["selected"] = False
        graph["route_families"]["route:new"]["selected"] = True
        return {"changed": True}

    monkeypatch.setattr(target_solver_module, "_switch_final_route_selection", switch)
    monkeypatch.setattr(
        target_solver_module,
        "_persist_final_route_repair_attempt",
        lambda *_args, **_kwargs: {"changed": True},
    )
    monkeypatch.setattr(
        target_solver_module,
        "_audit_stock_stage",
        lambda *_args, **_kwargs: {"status": "completed"},
    )

    results = target_solver_module._run_revision_bound_route_review_loop(
        service,
        director_runner=SimpleNamespace(),
        director_config=DirectorConfig(),
        route_family_ids=("route:old",),
        acceptance=RetrosynthesisAcceptanceSpec(),
        config=TargetSolveConfig(),
        stock_catalog_builder=None,
        inventory_snapshot_builder=None,
    )

    assert reviewed_ids == ["route:old", "route:new"]
    assert include_unselected == [False, True]
    assert switches == [(True, False)]
    assert graph["route_families"]["route:old"]["selected"] is False
    assert graph["route_families"]["route:new"]["selected"] is True
    assert results[0]["repair"]["status"] == "committed"
    assert results[-1]["route_sha256"] == "new-sha"


def test_final_route_review_keeps_original_when_repair_is_not_materialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {
        "route_families": {
            "route:old": {
                "selected": True,
                "chemical_critic": {
                    "status": "reject",
                    "step_assessments": [
                        {
                            "step_id": "step:bad",
                            "verdict": "reject",
                            "blocking": True,
                        }
                    ],
                },
            }
        }
    }

    class Store:
        def load(self):
            return graph

    monkeypatch.setattr(
        target_solver_module,
        "_run_revision_bound_route_critics",
        lambda *_args, **_kwargs: [
            {
                "route_family_id": "route:old",
                "status": "reused",
                "critic_status": "reject",
            }
        ],
    )
    monkeypatch.setattr(
        target_solver_module,
        "compile_revision_bound_route_critic_context",
        lambda *_args, **_kwargs: (
            SimpleNamespace(route_family_id="route:old", route_sha256="old-sha"),
            {},
        ),
    )
    monkeypatch.setattr(
        target_solver_module,
        "_run_final_route_repair_attempt",
        lambda *_args, **_kwargs: {
            "status": "materialization_failed",
            "reason": "host_rejected",
        },
    )
    monkeypatch.setattr(
        target_solver_module,
        "_switch_final_route_selection",
        lambda *_args, **_kwargs: pytest.fail("unmaterialized route was selected"),
    )

    target_solver_module._run_revision_bound_route_review_loop(
        SimpleNamespace(graph_store=Store()),
        director_runner=SimpleNamespace(),
        director_config=DirectorConfig(),
        route_family_ids=("route:old",),
        acceptance=RetrosynthesisAcceptanceSpec(),
        config=TargetSolveConfig(),
        stock_catalog_builder=None,
        inventory_snapshot_builder=None,
    )

    assert graph["route_families"]["route:old"]["selected"] is True


def test_final_route_review_rejects_stock_closure_regression_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {
        "route_families": {
            "route:old": {
                "selected": True,
                "chemical_critic": {
                    "status": "reject",
                    "step_assessments": [
                        {"step_id": "step:bad", "verdict": "reject", "blocking": True}
                    ],
                },
            },
            "route:new": {"selected": False},
        }
    }

    class Store:
        def load(self):
            return graph

    def review(_service, **kwargs):
        route_id = tuple(kwargs["route_family_ids"])[0]
        return [
            {
                "route_family_id": route_id,
                "status": "reused" if route_id == "route:old" else "completed",
                "critic_status": "reject" if route_id == "route:old" else "viable",
                "route_sha256": f"{route_id}:sha",
            }
        ]

    monkeypatch.setattr(target_solver_module, "_run_revision_bound_route_critics", review)
    monkeypatch.setattr(
        target_solver_module,
        "compile_revision_bound_route_critic_context",
        lambda *_args, **_kwargs: (
            SimpleNamespace(route_family_id="route:old", route_sha256="old-sha"),
            {},
        ),
    )
    monkeypatch.setattr(
        target_solver_module,
        "_run_final_route_repair_attempt",
        lambda *_args, **_kwargs: {
            "status": "candidate_materialized",
            "candidate_route_family_id": "route:new",
            "repair_attempt": {"task_id": "repair:one"},
        },
    )
    monkeypatch.setattr(
        target_solver_module,
        "canonical_route_stock_closed",
        lambda _graph, *, route_family_id, **_kwargs: route_family_id == "route:old",
    )
    audited_scopes: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        target_solver_module,
        "_audit_stock_stage",
        lambda *_args, **kwargs: (
            audited_scopes.append(tuple(kwargs["route_family_ids"]))
            or {"status": "completed"}
        ),
    )
    persisted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        target_solver_module,
        "_persist_final_route_repair_attempt",
        lambda *_args, **kwargs: persisted.append(dict(kwargs["attempt"]))
        or {"changed": True},
    )
    monkeypatch.setattr(
        target_solver_module,
        "_switch_final_route_selection",
        lambda *_args, **_kwargs: pytest.fail("open candidate was selected"),
    )

    results = target_solver_module._run_revision_bound_route_review_loop(
        SimpleNamespace(graph_store=Store()),
        director_runner=SimpleNamespace(),
        director_config=DirectorConfig(),
        route_family_ids=("route:old",),
        acceptance=RetrosynthesisAcceptanceSpec(),
        config=TargetSolveConfig(),
        stock_catalog_builder=None,
        inventory_snapshot_builder=None,
    )

    assert audited_scopes == [("route:new",)]
    assert persisted[-1]["status"] == "stock_closure_regressed"
    assert graph["route_families"]["route:old"]["selected"] is True
    assert graph["route_families"]["route:new"]["selected"] is False
    assert results[0]["repair"]["status"] == "stock_closure_regressed"


@pytest.mark.parametrize(
    (
        "delivery_boundary",
        "sequential_strategy",
        "stop_on_first_stock_closed_branch",
        "expected",
    ),
    [
        ("stock_result", False, False, True),
        ("stock_result", True, False, False),
        ("stock_result", True, True, True),
        ("full", True, True, False),
    ],
)
def test_first_stock_result_terminality_is_campaign_scoped(
    delivery_boundary: str,
    sequential_strategy: bool,
    stop_on_first_stock_closed_branch: bool,
    expected: bool,
) -> None:
    assert (
        _first_stock_result_is_terminal_for_run(
            delivery_boundary=delivery_boundary,
            sequential_strategy=sequential_strategy,
            stop_on_first_stock_closed_branch=stop_on_first_stock_closed_branch,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("paper_synthex", True),
        ("paper_matched_reach", True),
        ("self_correcting_sequential", True),
        ("proof", True),
        ("fast", True),
    ],
)
def test_frontier_builder_continuation_is_available_for_every_sequential_profile(
    profile: str,
    expected: bool,
) -> None:
    assert (
        _frontier_builder_extension_enabled(
            execution_profile=profile,
            enable_codex=True,
            sequential_runner=True,
        )
        is expected
    )
    assert (
        _frontier_builder_extension_enabled(
            execution_profile=profile,
            enable_codex=False,
            sequential_runner=True,
        )
        is False
    )
    assert (
        _frontier_builder_extension_enabled(
            execution_profile=profile,
            enable_codex=True,
            sequential_runner=False,
        )
        is False
    )


def _experimental_claim_stage(*, polarity: str, grants: bool) -> dict[str, Any]:
    claim = with_experimental_claim_digest(
        {
            "schema_version": "experimental_observation_claim.v1",
            "claim_id": "claim:test",
            "claim_kind": "program_validation_observation",
            "domain": "execution",
            "polarity": polarity,
            "outcome_status": "completed",
            "interpretation_status": "accepted_observation",
            "program_id": "program:test",
            "subject_refs": {"route_id": "route:test"},
            "boundary": {
                "input_state_ids": ["state:input"],
                "output_state_ids": ["state:output"],
            },
            "source_validation": {
                "schema_version": "execution_validation.v1",
                "validation_id": "validation:test",
                "content_sha256": "a" * 64,
            },
            "evidence_tier": "experimental_exact_boundary",
            "supporting_claim_refs": [],
            "condition_record_ids": [],
            "outcome_metrics": {"conversion": 0.5},
            "grants_domain_validation": grants,
            "generalization_scope": "exact_boundary_only",
            "authority_scope": "experimental_observation_exact_boundary",
            "domain_context": {},
            "semantics": CLAIM_SEMANTICS,
        }
    )
    claims = {claim["claim_id"]: claim}
    claim_set = with_experimental_claim_digest(
        {
            "schema_version": "experimental_observation_claim_set.v1",
            "run_id": "run:test",
            "route_id": "route:test",
            "source_artifacts": {
                "biocatalytic_bundle_sha256": "b" * 64,
                "biocatalytic_oracle_sha256": "c" * 64,
                "execution_feedback_sha256": "d" * 64,
                "execution_oracle_sha256": "e" * 64,
                "mechanism_feedback_sha256": "f" * 64,
                "mechanism_oracle_sha256": "1" * 64,
                "validation_pack_sha256": "2" * 64,
            },
            "claims": claims,
            "rejected_validations": [],
            "counts": experimental_claim_counts(claims, []),
            "semantics": CLAIM_SET_SEMANTICS,
        }
    )
    oracle = with_experimental_claim_digest(
        {
            "schema_version": "experimental_claim_set_oracle.v1",
            "accepted": True,
            "checks": {"inputs_reprojectable": True},
            "reasons": [],
            "expected_claim_set_sha256": claim_set["content_sha256"],
            "observed_claim_set_sha256": claim_set["content_sha256"],
            "semantics": {
                "oracle_is_read_only": True,
                "oracle_grants_no_scientific_authority": True,
            },
        }
    )
    return {
        "stage": "campaign_action_test",
        "detail": {
            "action": {"kind": "experiment_feedback_ingest"},
            "outcome": {
                "status": "completed",
                "handler_result": {
                    "status": "completed",
                    "experimental_claims": claim_set,
                    "experimental_claims_oracle": oracle,
                },
            },
        },
    }


def test_program_milestones_require_a_positive_exact_boundary_claim() -> None:
    positive = _program_milestones_from_stages(
        [_experimental_claim_stage(polarity="positive", grants=True)]
    )
    negative = _program_milestones_from_stages(
        [_experimental_claim_stage(polarity="negative", grants=False)]
    )

    assert positive["program:action:experiment_feedback_ingest"] is True
    assert positive["experiment:positive_exact_boundary_claim"] is True
    assert "experiment:positive_exact_boundary_claim" not in negative


def test_program_milestones_fail_closed_on_tampered_claim_oracle() -> None:
    stage = _experimental_claim_stage(polarity="positive", grants=True)
    stage["detail"]["outcome"]["handler_result"]["experimental_claims_oracle"][
        "observed_claim_set_sha256"
    ] = "0" * 64

    milestones = _program_milestones_from_stages([stage])

    assert "experiment:positive_exact_boundary_claim" not in milestones


@pytest.fixture(autouse=True)
def _deterministic_target_identity(monkeypatch: Any) -> None:
    """Keep target-solver integration coverage offline and deterministic.

    Transport and exact-InChIKey behavior are covered separately in
    test_target_identity.py.  These integration tests exercise orchestration,
    so a live PubChem dependency only adds latency and nondeterminism.
    """

    def resolve(target_smiles: str, **_kwargs: Any) -> dict[str, Any]:
        _molecule_id, canonical = molecule_identity(target_smiles)
        opaque = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        return {
            "schema_version": "target_identity_observation.v1",
            "status": "completed",
            "provider_id": "tests.deterministic_identity",
            "provider_version": "v1",
            "identity": {
                "preferred_name": f"target-{opaque}",
                "canonical_smiles": canonical,
                "resolved_from_input_structure": True,
                "synonyms": [],
                "patent_ids": [],
                "pubmed_ids": [],
            },
            "semantics": {
                "resolved_from_input_structure": True,
                "test_transport_is_offline": True,
            },
        }

    monkeypatch.setattr(target_solver_module, "resolve_target_identity", resolve)


def test_target_solver_keeps_whole_target_chemenzy_as_a_separate_arm() -> None:
    assert TargetSolveConfig().run_scope == "blind"
    assert TargetSolveConfig(run_scope="interactive").run_scope == "interactive"
    assert TargetSolveConfig().enable_target_chemenzy_baseline is False
    assert TargetSolveConfig().chemenzy_seed == 0

    with pytest.raises(ValueError, match="run scope"):
        TargetSolveConfig(run_scope="unknown")


def test_target_solver_rejects_invalid_chemenzy_seed() -> None:
    with pytest.raises(ValueError, match="ChemEnzy seed"):
        TargetSolveConfig(chemenzy_seed=-1)
    with pytest.raises(ValueError, match="ChemEnzy seed"):
        TargetSolveConfig(chemenzy_seed=2**32)


def test_paper_synthex_profile_keeps_route_depth_and_repair_rounds_fixed() -> None:
    config = TargetSolveConfig(execution_profile="paper_synthex")
    assert config.strategy_portfolio_mode == "auto"
    assert config.strategy_branch_count == 3
    assert config.strategy_branch_workers == 3
    assert config.stop_on_first_stock_closed_branch is False
    assert config.max_node_expansions_per_branch == 25
    assert config.max_route_local_repair_rounds == 6
    assert config.require_complete_route_json is False
    assert config.allow_editor_route_mutations is True

    with pytest.raises(ValueError, match="frozen 25-call ceiling"):
        TargetSolveConfig(
            execution_profile="paper_synthex",
            max_node_expansions_per_branch=2,
        )

    smoke = TargetSolveConfig(
        execution_profile="paper_matched_reach",
        max_node_expansions_per_branch=10,
    )
    assert smoke.max_node_expansions_per_branch == 10
    assert _resolve_execution_config(smoke).max_node_expansions_per_branch == 10
    with pytest.raises(ValueError, match="six Critic/Editor"):
        TargetSolveConfig(
            execution_profile="paper_synthex",
            max_route_local_repair_rounds=2,
        )
    # Complete RouteJSON is now a final admission contract.  The node prompt
    # remains one ReactionJSON edit per open leaf; setting the flag explicitly
    # is therefore valid and must not switch to a one-shot route prompt.
    assert (
        TargetSolveConfig(
            execution_profile="paper_synthex",
            require_complete_route_json=True,
        ).require_complete_route_json
        is True
    )
    with pytest.raises(ValueError, match="RouteJSON-aware Critic/Editor"):
        TargetSolveConfig(
            execution_profile="paper_synthex",
            allow_editor_route_mutations=False,
        )
    with pytest.raises(ValueError, match="all three strategy branches"):
        TargetSolveConfig(
            execution_profile="paper_synthex",
            stop_on_first_stock_closed_branch=True,
        )


def test_self_correcting_profile_keeps_five_step_hot_path_and_final_editor_loop() -> None:
    config = TargetSolveConfig(
        execution_profile="self_correcting_sequential",
        max_node_expansions_per_branch=5,
        max_route_local_repair_rounds=6,
        allow_editor_route_mutations=False,
    )
    resolved = _resolve_execution_config(config)

    assert resolved.strategy_search_profile == "synthex_matched"
    assert resolved.strategy_tree_engine == "aizynthfinder_mcts"
    assert resolved.max_node_expansions_per_branch == 5
    assert resolved.max_strategic_milestones_per_branch == 4
    assert resolved.max_route_local_repair_rounds == 6
    assert resolved.allow_editor_route_mutations is False
    assert resolved.execution_profile == "self_correcting_sequential"
    assert resolved.enable_chemenzy is False
    assert resolved.max_validation_tasks == 128
    search_method = _search_method_projection(resolved)
    assert search_method["leaf_continuation_engine"] == "same_llm_route_builder"
    assert search_method["paper_algorithm_equivalent"] is False
    assert search_method["paper_parameter_alignment"]["policy_call_ceiling_per_branch"] is False
    assert "development canary" in search_method["non_equivalence_reason"]

    full_builder = _resolve_execution_config(
        TargetSolveConfig(
            execution_profile="self_correcting_sequential",
            max_node_expansions_per_branch=25,
            max_route_local_repair_rounds=6,
        )
    )
    assert full_builder.max_node_expansions_per_branch == 25
    full_builder_method = _search_method_projection(full_builder)
    assert full_builder_method["paper_algorithm_equivalent"] is False
    assert "route-internal strategy refresh" in full_builder_method["non_equivalence_reason"]

    with pytest.raises(ValueError, match="six Critic/Editor"):
        TargetSolveConfig(
            execution_profile="self_correcting_sequential",
            max_node_expansions_per_branch=5,
            max_route_local_repair_rounds=0,
        )


def test_paper_profile_keeps_open_leaf_continuation_on_the_same_builder() -> None:
    resolved = _resolve_execution_config(TargetSolveConfig(execution_profile="paper_synthex"))

    method = _search_method_projection(resolved)
    assert method["leaf_continuation_engine"] == "same_llm_route_builder"


def test_paper_synthex_resolver_disables_non_reach_work_and_keeps_stock_validation() -> None:
    resolved = _resolve_execution_config(
        TargetSolveConfig(
            execution_profile="paper_synthex",
            enable_web_search=True,
            enable_initial_director_web_search=True,
            enable_patent_self_evolution=True,
            enable_condition_enrichment=True,
            enable_chemenzy_condition_prediction=True,
            enable_chemenzy_enzyme_assignment=True,
            enable_enzyme_coverage_sidecar=True,
            enable_program_review=True,
            enable_program_discovery=True,
            enable_program_validation=True,
            max_evidence_tasks=64,
            max_program_tasks=64,
            max_experiment_tasks=32,
            max_run_wall_time_s=7200.0,
        )
    )

    assert resolved.strategy_search_profile == "synthex_matched"
    assert resolved.strategy_tree_engine == "aizynthfinder_mcts"
    assert resolved.strategy_portfolio_mode == "paper_independent"
    assert resolved.strategy_branch_count == 3
    assert resolved.strategy_branch_workers == 3
    assert resolved.stop_on_first_stock_closed_branch is False
    assert resolved.max_node_expansions_per_branch == 25
    assert resolved.max_route_local_repair_rounds == 6
    assert resolved.require_complete_route_json is True
    assert resolved.allow_editor_route_mutations is True
    assert resolved.enable_web_search is False
    assert resolved.enable_initial_director_web_search is False
    assert resolved.enable_patent_self_evolution is False


def test_paper_synthex_preserves_explicit_enzyme_companion_arm() -> None:
    resolved = _resolve_execution_config(
        TargetSolveConfig(
            execution_profile="paper_synthex",
            strategy_portfolio_mode="enzyme_advantage",
        )
    )

    assert resolved.strategy_portfolio_mode == "enzyme_advantage"
    assert resolved.strategy_branch_count == 3
    assert resolved.strategy_branch_workers == 3
    assert resolved.max_node_expansions_per_branch == 25
    assert resolved.max_route_local_repair_rounds == 6
    assert resolved.enable_condition_enrichment is False
    assert resolved.enable_condition_enrichment is False
    assert resolved.enable_chemenzy_condition_prediction is False
    assert resolved.enable_chemenzy_enzyme_assignment is False
    assert resolved.enable_enzyme_coverage_sidecar is False
    assert resolved.enable_program_review is False
    assert resolved.enable_program_discovery is False
    assert resolved.enable_program_validation is False
    assert resolved.max_evidence_tasks == 0
    assert resolved.max_program_tasks == 0
    assert resolved.max_experiment_tasks == 0
    assert resolved.max_run_wall_time_s == 86_400.0
    assert resolved.enable_chemenzy is True
    assert resolved.max_chemenzy_steps == 6
    assert resolved.max_chemenzy_iterations == 500
    assert resolved.chemenzy_timeout_s == 1200.0


def test_paper_synthex_preserves_explicit_hybrid_companion_arm() -> None:
    resolved = _resolve_execution_config(
        TargetSolveConfig(
            execution_profile="paper_synthex",
            strategy_portfolio_mode="autoplanner_hybrid",
        )
    )

    assert resolved.strategy_portfolio_mode == "autoplanner_hybrid"


def test_paper_synthex_preserves_explicit_chemoenzymatic_fusion_arm() -> None:
    resolved = _resolve_execution_config(
        TargetSolveConfig(
            execution_profile="paper_synthex",
            strategy_portfolio_mode="chemoenzymatic_fusion",
        )
    )

    assert resolved.strategy_portfolio_mode == "chemoenzymatic_fusion"
    assert resolved.strategy_tree_engine == "aizynthfinder_mcts"
    assert resolved.strategy_branch_count == 3
    assert resolved.strategy_branch_workers == 3
    assert resolved.max_node_expansions_per_branch == 25
    assert resolved.enable_condition_enrichment is False


def test_paper_synthex_preserves_explicit_strategy_v2_arm() -> None:
    resolved = _resolve_execution_config(
        TargetSolveConfig(
            execution_profile="paper_synthex",
            strategy_portfolio_mode="autoplanner_strategy_v2",
        )
    )

    assert resolved.strategy_portfolio_mode == "autoplanner_strategy_v2"
    assert resolved.strategy_tree_engine == "aizynthfinder_mcts"
    assert resolved.strategy_branch_count == 3
    assert resolved.max_node_expansions_per_branch == 25


def test_paper_matched_reach_isolated_profile_forces_exact_reach_arm() -> None:
    resolved = _resolve_execution_config(
        TargetSolveConfig(
            execution_profile="paper_matched_reach",
            strategy_portfolio_mode="chemoenzymatic_fusion",
            enable_chemenzy=True,
            enable_condition_enrichment=True,
            enable_enzyme_coverage_sidecar=True,
            enable_replan=True,
            max_validation_tasks=128,
        )
    )

    assert resolved.strategy_portfolio_mode == "paper_independent"
    assert resolved.strategy_branch_count == 3
    assert resolved.strategy_branch_workers == 3
    assert resolved.max_node_expansions_per_branch == 25
    assert resolved.max_route_local_repair_rounds == 6
    assert resolved.require_complete_route_json is True
    assert resolved.enable_chemenzy is False
    assert resolved.enable_condition_enrichment is False
    assert resolved.enable_enzyme_coverage_sidecar is False
    assert resolved.enable_replan is False
    assert resolved.max_validation_tasks == 128


def test_paper_matched_primary_projection_foregrounds_reach_and_real_calls() -> None:
    config = _resolve_execution_config(TargetSolveConfig(execution_profile="paper_matched_reach"))
    route_families = [
        {
            "route_family_id": f"family:{index}",
            "strategy_call_count": 1,
            "route_call_count": 25,
            "critic_call_count": 1,
            "editor_attempt_count": 0,
            "editor_call_count": 0,
            "paper_policy_call_budget": {
                "actual_calls": 25,
            },
            "aizynthfinder_strategy_search": {
                "provider_callback_count": 25,
                "selected_solved": False,
                "selected_open_leaves": 2,
                "host_stop_reason": "route_builder_call_ceiling_reached",
            },
            "shared_model_budget_ledger": {
                "quota": {"output_tokens": 1_000_000},
                "committed": {"output_tokens": 300_000},
                "inflight": {"output_tokens": 0},
                "protected_final_critics": {"output_tokens": 48_000},
            },
            "critic_editor_history": [],
        }
        for index in range(1, 4)
    ]
    skeletons = [
        {
            "route_family_id": f"family:{index}",
            "routejson_replay_complete": True,
            "steps": [{"step_id": f"step:{index}"}],
        }
        for index in range(1, 4)
    ]
    result = _paper_matched_primary_projection(
        config=config,
        paper_equivalent={
            "paper_reach": True,
            "paper_equivalent_solved": False,
            "paper_reached_route_count": 3,
            "paper_equivalent_solved_route_count": 0,
            "stock_comparable_to_synthex": True,
        },
        outcomes=[
            {
                "resource_usage": {
                    "actual_route_builder_policy_calls": 75,
                    "actual_critic_calls": 5,
                    "actual_editor_calls": 0,
                },
                "plan": {
                    "route_families": route_families,
                    "multi_step_skeletons": skeletons,
                },
            }
        ],
        stages=[],
    )

    assert result["paper_reach"] is True
    assert result["paper_equivalent_solved"] is False
    assert result["worker_ledger_available"] is True
    assert result["total_route_builder_policy_invocations"] == 75
    assert result["retained_route_builder_policy_calls"] == 75
    assert result["total_critic_invocations"] == 5
    assert result["retained_critic_calls"] == 3
    assert result["policy_cap_respected"] is True
    assert result["maximum_route_builder_policy_calls"] == 75
    assert result["complete_routejson_branch_count"] == 3
    assert result["leaf_continuation"] == {
        "engine": "same_llm_route_builder",
        "separate_mode": False,
        "one_step_reactionjson_contract": True,
        "host_validation_and_stock_audit_unchanged": True,
    }
    assert result["branches"][0]["builder_stop_reason"] == ("route_builder_call_ceiling_reached")
    assert result["branches"][0]["builder_selected_open_leaves"] == 2
    assert result["branches"][0]["builder_selected_solved"] is False
    assert result["branches"][0]["shared_model_budget_ledger"]["quota"] == {
        "output_tokens": 1_000_000
    }


def test_rejection_taxonomy_derives_cross_layer_diagnostics_without_authority() -> None:
    taxonomy = target_solver_module._compile_rejection_taxonomy(
        outcomes=[
            {
                "plan": {
                    "route_families": [
                        {
                            "route_family_id": "family:1",
                            "key_event_critic_history": [
                                {
                                    "task_id": "critic:key:1",
                                    "focus_step_id": "step:key",
                                    "assessment": {
                                        "step_id": "step:key",
                                        "verdict": "reject",
                                        "blocking_type": "chemoselectivity",
                                    },
                                }
                            ],
                            "chemical_critic": {
                                "critic_task_id": "critic:route:1",
                                "overall_assessment": "reject",
                                "step_assessments": [
                                    {
                                        "step_id": "step:oxygen",
                                        "verdict": "reject",
                                        "blocking_type": "atom_provenance",
                                    }
                                ],
                            },
                            "rejections": [
                                {
                                    "phase": "key_event_critic",
                                    "reason": "key_event_critic_reject",
                                    "focus_step_id": "step:key",
                                    "candidate_id": "candidate:key",
                                },
                                {
                                    "phase": "route_builder_candidate",
                                    "reason": "reactionjson_map_not_found",
                                    "candidate_id": "candidate:map",
                                    "compiler_error": "reactionjson_map_not_found",
                                },
                                {
                                    "phase": "strategy_generator",
                                    "reason": "strategy_portfolio_output_invalid",
                                },
                            ],
                            "editor_rejection_diagnostics": [
                                {
                                    "task_id": "editor:1",
                                    "reason": "route_json_step_replay_failed",
                                    "replay_diagnostic": {
                                        "compiler_error": "reactionjson_bond_missing",
                                        "failed_step_id": "step:editor",
                                    },
                                }
                            ],
                            "path_repair_transactions": [
                                {
                                    "editor_task_id": "editor:2",
                                    "status": "rolled_back_after_recritic",
                                    "reason": "path_repair_recritic_rejected",
                                }
                            ],
                            "materialization_failures": {"CCO": 2},
                            "aizynthfinder_strategy_search": {
                                "host_stop_reason": (
                                    "route_builder_output_token_allocation_exhausted"
                                )
                            },
                        }
                    ]
                }
            }
        ],
        stages=[
            {
                "stage": "reaction_validation",
                "status": "partial",
                "detail": {
                    "rejection_diagnostics": [
                        {
                            "edge_id": "edge:1",
                            "reasons": ["atom_balance_failed"],
                        }
                    ]
                },
            },
            {
                "stage": "aizynthfinder_stock_recovery",
                "status": "completed",
                "detail": {
                    "status": "timeout",
                    "reason": "provider_timeout",
                },
            },
        ],
    )

    assert taxonomy["counts"] == {
        "runtime_or_provider": 2,
        "model_output_contract": 1,
        "host_replay_or_topology": 2,
        "identity_or_materialization": 1,
        "critic_chemistry": 2,
        "editor_transaction": 1,
        "reaction_validation": 1,
    }
    assert taxonomy["event_count"] == 10
    assert taxonomy["semantics"]["report_only"] is True
    assert taxonomy["semantics"]["no_execution_or_admission_authority"] is True
    assert sum(row["reason"] == "key_event_critic_reject" for row in taxonomy["events"]) == 0


def test_action_handler_projection_preserves_failed_outcome_status_and_reasons() -> None:
    results = _campaign_action_handler_results(
        (
            {
                "status": "failed",
                "action": {"kind": "chemenzy_target_expand"},
                "outcome": {
                    "status": "failed",
                    "handler_result": {},
                    "failure_reasons": ["campaign_action_handler_error:RuntimeError:boom"],
                },
            },
        ),
        kind=CampaignActionKind.CHEMENZY_TARGET_EXPAND,
    )

    assert results == [
        {
            "status": "failed",
            "reasons": ["campaign_action_handler_error:RuntimeError:boom"],
        }
    ]


def test_target_solver_reserves_native_search_only_for_target_baseline() -> None:
    broad = RetrosynthesisRunBudget(max_attempt_runs=192)

    inherited = _bind_native_search_budget(
        broad,
        config=TargetSolveConfig(),
    )
    target_baseline = _bind_native_search_budget(
        broad,
        config=TargetSolveConfig(enable_target_chemenzy_baseline=True),
    )
    disabled = _bind_native_search_budget(
        broad,
        config=TargetSolveConfig(enable_chemenzy=False),
    )

    assert inherited.max_native_search_invocations == 0
    assert inherited.min_target_native_search_invocations == 0
    assert inherited.max_frontier_native_search_invocations == 0
    assert target_baseline.max_native_search_invocations == 1
    assert target_baseline.min_target_native_search_invocations == 1
    assert target_baseline.max_frontier_native_search_invocations == 0
    assert target_baseline.allow_frontier_native_search_borrowing is False
    assert disabled.max_native_search_invocations == 0
    assert disabled.min_target_native_search_invocations == 0
    assert disabled.max_frontier_native_search_invocations == 0


def test_replan_signal_gate_rejects_a_route_deficit_without_new_information() -> None:
    gate = _replan_signal_gate(
        {"gates": {"B2_host_validated_routes": False, "B4_stock_boundary": True}},
        material_events=("portfolio_stagnation", "stock_records_added"),
        trigger_reasons=("host_validated_route_deficit",),
    )

    assert gate["accepted"] is False
    assert gate["actionable_material_events"] == []
    assert gate["ignored_material_events"] == [
        "portfolio_stagnation",
        "stock_records_added",
    ]
    assert gate["reasons"] == ["no_new_actionable_host_observation"]


def test_material_replan_events_only_emit_observed_post_plan_changes() -> None:
    assert _material_replan_events({}) == ()
    assert _material_replan_events(
        {
            "accepted_validation_count": 2,
            "rejected_validation_count": 1,
            "material_events": ["guided_provider_proposals_added"],
        }
    ) == (
        "critical_edge_rejected",
        "guided_provider_proposals_added",
        "host_validated_edges_added_after_initial_plan",
    )


def test_replan_signal_gate_accepts_new_validation_or_open_stock_observation() -> None:
    validation = _replan_signal_gate(
        {"gates": {"B2_host_validated_routes": False, "B4_stock_boundary": True}},
        material_events=("critical_edge_rejected",),
        trigger_reasons=("host_validated_route_deficit",),
    )
    stock = _replan_signal_gate(
        {"gates": {"B2_host_validated_routes": False, "B4_stock_boundary": False}},
        material_events=("stock_boundary_changed", "stock_records_added"),
        trigger_reasons=("host_validated_route_deficit",),
    )

    assert validation["accepted"] is True
    assert validation["actionable_material_events"] == ["critical_edge_rejected"]
    assert stock["accepted"] is True
    assert stock["actionable_material_events"] == [
        "stock_boundary_changed",
        "stock_records_added",
    ]


def test_replan_signal_gate_accepts_observed_provider_search_failure() -> None:
    gate = _replan_signal_gate(
        {"gates": {"B1_global_multi_route": False, "B4_stock_boundary": False}},
        material_events=("provider_search_exhausted_without_proposal",),
        trigger_reasons=("provider_search_failure_requires_new_frontier",),
    )

    assert gate["accepted"] is True
    assert gate["actionable_material_events"] == ["provider_search_exhausted_without_proposal"]


def test_replan_retention_audit_requires_graph_union_semantics() -> None:
    before = {
        "molecules": {"m1": {}},
        "edges": {"e1": {}},
        "route_families": {"r1": {}},
    }
    extended = {
        "molecules": {"m1": {}, "m2": {}},
        "edges": {"e1": {}, "e2": {}},
        "route_families": {"r1": {}, "r2": {}},
    }
    replaced = {
        "molecules": {"m2": {}},
        "edges": {"e2": {}},
        "route_families": {"r2": {}},
    }

    accepted = _replan_retention_audit(before, extended)
    rejected = _replan_retention_audit(before, replaced)

    assert accepted["accepted"] is True
    assert accepted["counts"]["edges"] == {
        "before": 1,
        "after": 2,
        "added": 1,
        "missing": 0,
    }
    assert rejected["accepted"] is False
    assert rejected["missing_ids"] == {
        "molecules": ["m1"],
        "edges": ["e1"],
        "route_families": ["r1"],
    }


def test_replan_gain_audit_separates_scientific_delta_from_model_cost() -> None:
    before = {
        "gates": {"B1_global_multi_route": True, "B2_host_validated_routes": False},
        "counts": {"reaction_validated_skeletons": 0, "materialized_skeletons": 2},
    }
    after = {
        "gates": {"B1_global_multi_route": True, "B2_host_validated_routes": True},
        "counts": {"reaction_validated_skeletons": 1, "materialized_skeletons": 3},
    }

    audit = _replan_gain_audit(
        before,
        after,
        model_cost_before={
            "model_invocations": 1,
            "input_tokens": 10_000,
            "output_tokens": 2_000,
            "wall_time_s": 30.0,
        },
        model_cost_after={
            "model_invocations": 2,
            "input_tokens": 26_000,
            "output_tokens": 5_000,
            "wall_time_s": 75.5,
        },
    )

    assert audit["disposition"] == "positive_gain"
    assert audit["gained_gates"] == ["B2_host_validated_routes"]
    assert audit["positive_count_deltas"] == {
        "materialized_skeletons": 1,
        "reaction_validated_skeletons": 1,
    }
    assert audit["model_cost_delta"] == {
        "model_invocations": 1.0,
        "input_tokens": 16_000.0,
        "output_tokens": 3_000.0,
        "wall_time_s": 45.5,
    }
    assert audit["semantics"]["observed_delta_is_not_a_cross_arm_causal_estimate"]


def test_rejected_director_topology_triggers_one_replan_even_after_b2() -> None:
    outcomes = [
        {
            "status": "accepted",
            "proposal_audits": [
                {
                    "accepted": False,
                    "reasons": ["skeleton_contains_disconnected_steps"],
                }
            ],
        }
    ]

    events = _director_topology_replan_events(outcomes)
    reasons = _replan_reasons(
        {"gates": {"B2_host_validated_routes": True}},
        material_events=events,
    )

    assert events == ("director_topology_rejected",)
    assert reasons == ("director_topology_deficit",)


def test_failed_director_contract_triggers_one_bounded_replan() -> None:
    outcomes = [
        {
            "status": "failed",
            "reasons": [
                "GlobalCampaignPlanValidationError",
                "route_families_without_skeletons:RF4",
            ],
        }
    ]

    events = _director_topology_replan_events(outcomes)
    reasons = _replan_reasons(
        {"gates": {"B2_host_validated_routes": False}},
        material_events=events,
    )

    assert events == ("director_contract_rejected",)
    assert _director_outcome_allows_replan(outcomes) is True
    assert reasons == (
        "director_contract_deficit",
        "host_validated_route_deficit",
    )


def test_planning_depth_deficit_replans_but_keeps_short_skeleton_visible() -> None:
    short_steps = [
        {
            "step_id": f"step:{index}",
            "product_smiles": "CCO",
            "precursor_smiles": ["CC"],
        }
        for index in range(9)
    ]
    outcomes = [
        {
            "status": "accepted",
            "plan": {"multi_step_skeletons": [{"skeleton_id": "short", "steps": short_steps}]},
            "proposal_audits": [
                {
                    "skeleton_id": "short",
                    "proposal_id": f"step:{index}",
                    "accepted": True,
                }
                for index in range(9)
            ],
        }
    ]

    depth = _planning_depth_requirement(outcomes, minimum_steps=20)
    events = _director_depth_replan_events(depth)
    reasons = _replan_reasons(
        {"gates": {"B2_host_validated_routes": True}},
        material_events=events,
    )

    assert depth["maximum_host_contract_accepted_steps"] == 9
    assert depth["requirement_met"] is False
    assert depth["observed_skeletons"][0]["skeleton_id"] == "short"
    assert depth["semantics"]["shorter_routes_remain_visible"] is True
    assert events == ("director_depth_deficit",)
    assert reasons == ("planning_depth_deficit",)


def test_planning_depth_accepts_one_complete_long_skeleton_only() -> None:
    def outcome(skeleton_id: str, count: int, *, rejected: int | None = None) -> dict:
        return {
            "status": "accepted",
            "plan": {
                "multi_step_skeletons": [
                    {
                        "skeleton_id": skeleton_id,
                        "steps": [{"step_id": f"{skeleton_id}:{index}"} for index in range(count)],
                    }
                ]
            },
            "proposal_audits": [
                {
                    "skeleton_id": skeleton_id,
                    "proposal_id": f"{skeleton_id}:{index}",
                    "accepted": index != rejected,
                }
                for index in range(count)
            ],
        }

    depth = _planning_depth_requirement(
        [outcome("rejected-long", 21, rejected=4), outcome("accepted-long", 20)],
        minimum_steps=20,
    )

    assert depth["requirement_met"] is True
    assert depth["maximum_host_contract_accepted_steps"] == 20
    assert depth["qualifying_skeleton_ids"] == ["accepted-long"]
    assert _director_depth_replan_events(depth) == ()


def test_evidence_replan_projection_is_bounded_and_chemistry_focused() -> None:
    sources = []
    for index in range(7):
        sources.append(
            {
                "source_ref": f"patent:US{index}",
                "publication_number": f"US{index}",
                "title": f"Source {index}",
                "source_route_proposal_count": 1 if index == 6 else 0,
                "procedure_inventory": [
                    {
                        "label": f"Example {item}",
                        "name": f"Intermediate {item}",
                        "page_number": item,
                        "procedure_excerpt": "A source-authored reaction " * 100,
                    }
                    for item in range(10)
                ],
                "source_route_observation": {
                    "schema_version": "deterministic_source_route_observation.v1",
                    "source_ref": f"patent:US{index}",
                    "proposal_count": 1,
                    "proposals": [
                        {
                            "proposal_id": f"source-step:{index}",
                            "product_smiles": TARGET,
                            "precursor_smiles": ["CCO", "CC(=O)O"],
                            "condition_candidate": {"temperature_c": 25},
                        }
                    ],
                },
            }
        )

    projected = _evidence_observations(
        {
            "discovery": {
                "schema_version": "source_discovery_observation.v1",
                "provider_id": "tests",
                "sources": sources,
            }
        }
    )["source_discovery"]

    assert projected["source_count"] == 7
    assert projected["selected_source_count"] == 3
    assert projected["omitted_source_count"] == 4
    assert projected["sources"][0]["source_ref"] == "patent:US6"
    assert len(projected["sources"][0]["procedure_inventory"]) == 2
    assert len(projected["sources"][0]["procedure_inventory"][0]["procedure_excerpt"]) <= 1_200
    assert (
        projected["sources"][0]["source_route_observation"]["proposals"][0]["product_smiles"]
        == TARGET
    )


def test_chemenzy_timeout_retry_requires_resume_and_larger_window() -> None:
    stages = [
        {
            "stage": "chemenzy_baseline",
            "status": "timeout",
            "detail": {"limits": {"timeout_s": 90.0}},
        }
    ]

    assert _should_retry_chemenzy_timeout(stages, resume=True, requested_timeout_s=300.0)
    assert not _should_retry_chemenzy_timeout(stages, resume=True, requested_timeout_s=90.0)
    assert not _should_retry_chemenzy_timeout(stages, resume=False, requested_timeout_s=300.0)


def _paths(tmp_path: Path) -> RuntimePaths:
    repository = tmp_path / "repository"
    repository.mkdir()
    return RuntimePaths.discover(
        repository_root=repository,
        environ={
            "AUTOPLANNER_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "AUTOPLANNER_RUNS_ROOT": str(tmp_path / "runs"),
            "AUTOPLANNER_ARTIFACT_STORE_ROOT": str(tmp_path / "cas"),
            "AUTOPLANNER_RUN_INDEX_PATH": str(tmp_path / "index" / "runs.sqlite3"),
            "AUTOPLANNER_EXTERNAL_DATA_ROOT": str(tmp_path / "external"),
            "AUTOPLANNER_MODEL_ROOT": str(tmp_path / "models"),
            "AUTOPLANNER_VENDOR_ROOT": str(tmp_path / "vendor"),
        },
    )


def _scanned_patent_pdf(label: str = "") -> bytes:
    image = Image.new("RGB", (1200, 1600), "white")
    ImageDraw.Draw(image).text(
        (50, 80),
        f"Ethyl acetate procedures T1-T3 {label}",
        fill="black",
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=buffer.getvalue())
    value = document.tobytes()
    document.close()
    return value


def _patent_html(publication: str) -> bytes:
    return f"""
    <html><head><meta name="DC.relation" content="{publication}"></head>
    <body>
      <div id="p0001" class="description-paragraph">Example 1</div>
      <div id="p0002" class="description-paragraph">
        Ethyl acetate (T1). Ethanol and acetyl chloride were added. The
        reaction mixture was stirred to afford T1.
      </div>
      <div id="p0003" class="description-paragraph">Example 2</div>
      <div id="p0004" class="description-paragraph">
        Ethyl acetate (T2). Ethanol and acetic acid were added. The reaction
        mixture was stirred to afford T2.
      </div>
      <div id="p0005" class="description-paragraph">Example 3</div>
      <div id="p0006" class="description-paragraph">
        Ethyl acetate (T3). Ethanol and acetyl bromide were added. The
        reaction mixture was stirred to afford T3.
      </div>
    </body></html>
    """.encode()


def _plan(context: Any, mode: str) -> dict[str, Any]:
    families = [
        ("family:chloride", "CC(=O)Cl", "acyl chloride substitution"),
        ("family:acid", "CC(=O)O", "direct esterification"),
        ("family:bromide", "CC(=O)Br", "acyl bromide substitution"),
    ]
    return {
        "schema_version": "global_campaign_plan.v1",
        "plan_id": f"blind-plan:{mode}",
        "run_id": context.run_id,
        "mode": mode,
        "context_sha256": context.content_sha256,
        "graph_revision": context.revision.graph_revision,
        "route_families": [
            {
                "route_family_id": family_id,
                "title": hypothesis,
                "strategy": hypothesis,
                "target_smiles": TARGET,
                "advantages": ["short"],
                "risks": ["selectivity"],
                "diversity_basis": precursor,
            }
            for family_id, precursor, hypothesis in families
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": f"skeleton:{index}",
                "route_family_id": family_id,
                "summary": hypothesis,
                "steps": [
                    {
                        "step_id": f"step:{index}",
                        "product_smiles": TARGET,
                        "precursor_smiles": ["CCO", precursor],
                        "transformation_hypothesis": hypothesis,
                        "strategic_role": "target convergence",
                        "source_hints": [],
                        "required_validation": ["atom mapping", "stock audit"],
                        "hypothesis_only": True,
                        "condition_predictions": [
                            {
                                "reagents": ["acylation catalyst or promoter"],
                                "solvent": "screen appropriate solvent",
                                "temperature_c": 25,
                                "time": "screen",
                                "authority_scope": "model_predicted_condition",
                                "not_reaction_proof": True,
                            }
                        ],
                    }
                ],
            }
            for index, (family_id, precursor, hypothesis) in enumerate(families, start=1)
        ],
        "strategic_disconnections": [],
        "shared_intermediates": [],
        "critical_unknowns": [],
        "source_plan": [],
        "fallback_strategies": [],
        "frontier_priorities": [
            {
                "priority_id": f"priority:{index}",
                "proposal_id": f"step:{index}",
                "priority": 10 - index,
                "rationale": "close complete route",
                "expected_portfolio_gain": "one distinct family",
            }
            for index in (1, 2)
        ],
        "pivot_conditions": [],
        "stop_conditions": [],
        "portfolio_rationale": "Two different target-level acyl donors.",
        "limitations": ["requires host validation"],
    }


def _runner(spec: AgentSpec, context: Any, mode: str, _config: Any) -> AgentResult:
    return AgentResult(
        run_id=spec.run_id,
        agent_id=spec.agent_id,
        parent_agent_id=spec.parent_agent_id,
        attempt=spec.attempt,
        idempotency_key=f"{spec.idempotency_key}:result",
        context_hash=spec.context_hash,
        capabilities=spec.capabilities,
        write_scope=spec.write_scope,
        budget=spec.budget,
        state=AgentState.SUCCEEDED,
        output=_plan(context, mode),
        usage={
            "model_invocations": 1,
            "input_tokens": 1000,
            "output_tokens": 700,
            "wall_time_s": 1.0,
        },
    )


def _failed_runner(
    spec: AgentSpec,
    _context: Any,
    _mode: str,
    _config: Any,
) -> AgentResult:
    return AgentResult(
        run_id=spec.run_id,
        agent_id=spec.agent_id,
        parent_agent_id=spec.parent_agent_id,
        attempt=spec.attempt,
        idempotency_key=f"{spec.idempotency_key}:result",
        context_hash=spec.context_hash,
        capabilities=spec.capabilities,
        write_scope=spec.write_scope,
        budget=spec.budget,
        state=AgentState.FAILED,
        error="provider_unavailable",
        usage={"model_invocations": 1, "wall_time_s": 0.1},
    )


def _mapper(reactions: list[str]) -> list[str]:
    values = []
    for reaction in reactions:
        if "Br" in reaction:
            values.append(
                "[CH3:1][C:2](=[O:3])[Br:4].[CH3:5][CH2:6][OH:7]>>"
                "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
            )
        elif "Cl" in reaction:
            values.append(
                "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][CH2:6][OH:7]>>"
                "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
            )
        else:
            values.append(
                "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>"
                "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
            )
    return values


def _catalog(smiles: list[str], **_: Any) -> dict[str, Any]:
    return {
        "schema_version": "versioned_benchmark_stock_catalog.v1",
        "adapter_version": "tests.generic-catalog.v1",
        "catalog_name": "test-generic-catalog",
        "catalog_version": "2026-07-14",
        "retrieved_at": "2026-07-14T00:00:00Z",
        "members": [
            {
                "canonical_smiles": value,
                "cid": index,
                "vendor_count": 1,
                "vendors": [],
                "source_url": f"https://catalog.invalid/{index}",
                "response_sha256": f"{index:064x}",
            }
            for index, value in enumerate(sorted(set(smiles)), start=1)
        ],
        "misses": [],
    }


def _partial_catalog(smiles: list[str], **_: Any) -> dict[str, Any]:
    catalog = _catalog(smiles)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    catalog["catalog_version"] = timestamp
    catalog["retrieved_at"] = timestamp
    # Catalog membership is immutable across bounded query batches.  Making
    # the miss depend on the current request set causes the same molecule to
    # alternate between hit and miss as materialized route boundaries grow.
    missing = "CCO"
    catalog["members"] = [row for row in catalog["members"] if row["canonical_smiles"] != missing]
    catalog["misses"] = [
        {
            "canonical_smiles": missing,
            "cid": 0,
            "reason": "test_catalog_miss",
        }
    ]
    return catalog


def _evidence_connector(request: Any) -> dict[str, Any]:
    rows = [
        {
            "product_smiles": edge["product_smiles"],
            "reactant_smiles": edge["precursor_smiles"],
            "step_id": f"connector-step:{index}",
            "location_ref": f"Example 4, step {index}",
            "conditions": {"temperature_c": 25},
        }
        for index, edge in enumerate(request["edges"], start=1)
    ]
    return {
        "schema_version": "structured_evidence_import.v1",
        "sources": [
            {
                "binding": {
                    "source_kind": source_kind,
                    "source_ref": source_ref,
                    "title": f"Independent source {index}",
                    "provenance": "typed_connector",
                },
                "extraction": {
                    "schema_version": "structured_exact_row_extraction.v1",
                    "extractor": {
                        "producer_kind": "typed_connector_structured_extraction",
                        "producer_id": "tests.target-evidence",
                        "version": "1.0.0",
                    },
                    "rows": rows,
                },
            }
            for index, (source_kind, source_ref) in enumerate(
                (
                    ("patent", "patent:US1234567A1"),
                    ("paper_si", "doi:10.1000/example.1"),
                ),
                start=1,
            )
        ],
    }


def _discovery_only_connector(request: Any) -> dict[str, Any]:
    return {
        "discovery": {
            "schema_version": "source_discovery_observation.v1",
            "provider_id": "tests.discovery",
            "request_sha256": request["content_sha256"],
            "sources": [
                {
                    "publication_number": "US7654321A1",
                    "family_id": "family:discovery",
                    "title": "Alternative ester preparation",
                    "procedure_inventory": [
                        {
                            "label": "11",
                            "name": "ethyl acetate",
                            "page_number": 4,
                            "procedure_excerpt": (
                                "Ethanol and acetic acid were combined under the source conditions."
                            ),
                        }
                    ],
                    "exact_edge_ids": [],
                    "exact_row_count": 0,
                }
            ],
            "semantics": {
                "source_text_is_untrusted_data": True,
                "discovery_does_not_grant_exact_evidence": True,
            },
        },
        "receipt": {
            "schema_version": "evidence_connector_receipt.v1",
            "provider_id": "tests.discovery",
            "model_invocations": 0,
        },
    }


def _inventory_builder(smiles: list[str], **_: Any) -> dict[str, Any]:
    checked_at = "2026-07-14T00:00:00Z"
    return {
        "schema_version": "versioned_inventory_snapshot.v1",
        "adapter_version": "tests.inventory.v1",
        "inventory_version": "snapshot-2026-07-14",
        "retrieved_at": checked_at,
        "offers": [
            {
                "schema_version": "stock_offer_snapshot.v1",
                "supplier": "Test Supplier",
                "catalog_number": f"SKU-{index}",
                "smiles": value,
                "checked_at": checked_at,
                "available": True,
                "source_url": f"https://supplier.invalid/SKU-{index}",
            }
            for index, value in enumerate(sorted(set(smiles)), start=1)
        ],
    }


def test_target_only_solver_runs_global_plan_validation_stock_and_resume(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    acceptance = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=2,
        minimum_edge_proof_level=2,
        stock_boundary="benchmark_search",
        minimum_independent_source_groups=2,
    )
    budget = RetrosynthesisRunBudget(
        max_model_invocations=2,
        max_total_input_tokens=10_000,
        max_total_output_tokens=5_000,
        max_total_wall_time_s=60,
        max_visual_invocations=0,
        max_accepted_expansions=8,
        max_attempt_runs=16,
    )
    result = gateway.solve_target(
        target_name="opaque blind molecule",
        target_smiles=TARGET,
        run_id="blind-target-e2e",
        acceptance=acceptance,
        budget=budget,
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=True,
            max_program_tasks=7,
            max_experiment_tasks=3,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )

    assert result["preflight"]["accepted"] is True
    assert result["model_cost"]["model_invocations"] == 1
    assert result["accepted_expansion_count"] == 3
    assert result["gates"]["gates"] == {
        "B0_blind_input": True,
        "B1_global_multi_route": True,
        "B2_host_validated_routes": True,
        "B3_exact_multi_source": False,
        "B4_stock_boundary": True,
        "B5_configured_portfolio_acceptance": True,
    }
    assert set(result["quality_state"]["axes"]) == {
        "topology",
        "reaction_validation",
        "exact_evidence",
        "stock",
        "conditions",
        "procurement",
        "program_validation",
        "diversity",
    }
    assert result["quality_state"]["configured_acceptance"] is True
    assert result["quality_state"]["axes"]["exact_evidence"]["state"] == "open"
    assert result["quality_state"]["axes"]["conditions"]["state"] == "open"
    assert result["claim"]["generated_route_portfolio"] is True
    assert result["claim"]["host_validated_route_portfolio"] is True
    assert result["claim"]["exact_multi_source_grade"] is False
    assert result["claim"]["procurement_ready"] is False
    assert result["claim"]["acceptance_profile"] == "exploration_closed"
    assert result["claim"]["achieved_profile"] == "reaction_validated"
    assert result["claim"]["product_profile_counts"]["reaction_validated"] >= 1
    assert result["claim"]["literature_grounded"] is False
    assert result["claim"]["condition_complete"] is False
    assert result["claim"]["process_ready"] is False
    assert result["current_disposition"]["state"] == "accepted"
    resource_envelope = result["resource_envelope"]
    assert resource_envelope["schema_version"] == "target_solve_resource_envelope.v1"
    assert resource_envelope["observed"]["run_wall_time_s"] >= 0
    dimensions = resource_envelope["task_budget"]["dimensions"]
    assert dimensions["program"]["limit"] == 7
    assert dimensions["experiment"]["limit"] == 3
    assert dimensions["validation"]["settled"] >= 1
    assert dimensions["total"]["settled"] >= dimensions["validation"]["settled"]
    assert Path(result["report_path"]).is_file()
    global_stage = next(row for row in result["stages"] if row["stage"] == "global_campaign")
    assert global_stage["detail"]["status"] == "accepted"
    assert global_stage["detail"]["plan"]["multi_step_skeletons"]

    resumed = gateway.solve_target(
        target_name="opaque blind molecule",
        target_smiles=TARGET,
        run_id="blind-target-e2e",
        acceptance=acceptance,
        budget=budget,
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            max_program_tasks=7,
            max_experiment_tasks=3,
        ),
        resume=True,
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )
    assert resumed["model_cost"]["model_invocations"] == 1
    assert resumed["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    assert resumed["trajectory"]["content_sha256"] == result["trajectory"]["content_sha256"]
    assert resumed["trajectory"]["continuity"]["resume_baseline_preserved"] is True
    assert (
        resumed["trajectory"]["time_to_first"]["B4"] == result["trajectory"]["time_to_first"]["B4"]
    )


def test_interactive_solver_allows_a_target_already_present_in_repository(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    (gateway.paths.repository_root / "known-targets.txt").write_text(
        TARGET,
        encoding="utf-8",
    )
    acceptance = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=2,
        minimum_edge_proof_level=2,
        stock_boundary="benchmark_search",
        minimum_independent_source_groups=2,
    )
    budget = RetrosynthesisRunBudget(
        max_model_invocations=2,
        max_total_input_tokens=10_000,
        max_total_output_tokens=5_000,
        max_total_wall_time_s=60,
        max_visual_invocations=0,
        max_accepted_expansions=8,
        max_attempt_runs=16,
    )

    with pytest.raises(ValueError, match="target_material_already_present"):
        gateway.solve_target(
            target_name="known blind molecule",
            target_smiles=TARGET,
            run_id="blind-known-target-e2e",
            acceptance=acceptance,
            budget=budget,
            config=TargetSolveConfig(
                use_coordinator=False,
                enable_web_search=False,
                enable_replan=False,
            ),
            director_runner=_runner,
            atom_mapper=_mapper,
            stock_catalog_builder=_catalog,
        )

    result = gateway.solve_target(
        target_name="known interactive molecule",
        target_smiles=TARGET,
        run_id="interactive-target-e2e",
        acceptance=acceptance,
        budget=budget,
        config=TargetSolveConfig(
            run_scope="interactive",
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )

    assert result["preflight"]["schema_version"] == "interactive_target_preflight.v1"
    assert result["preflight"]["execution_scope"] == "interactive"
    assert result["preflight"]["repository_absence_attested"] is False
    assert result["preflight"]["accepted"] is True
    run_dir = Path(result["report_path"]).parent
    assert (run_dir / ".autoplanner" / "interactive-preflight.json").is_file()
    assert not (run_dir / ".autoplanner" / "blind-preflight.json").exists()


def test_current_disposition_does_not_treat_stale_terminal_as_scientific_success() -> None:
    disposition = _current_disposition(
        kernel_status="completed",
        stop_decision={"decision": "completed", "terminal": True},
        claim={"accepted_under_configured_policy": False},
        gates={"reaction_proof_version_audit": {"requires_revalidation": True}},
    )

    assert disposition["state"] == "terminal_snapshot_requires_revalidation"
    assert disposition["scientifically_accepted"] is False
    assert disposition["requires_revalidation"] is True


def test_completed_checkpoint_noop_resume_is_terminal_unresolved() -> None:
    baseline = {
        "scientific_sha256": "graph-sha",
        "attempt_count": 12,
        "accepted_expansion_count": 8,
        "model_totals": {"model_invocations": 2},
    }

    assert _automatic_continuation_exhausted(
        resumed_completed_checkpoint=True,
        baseline=baseline,
        current=dict(baseline),
        portfolio_accepted=False,
    )
    assert not _automatic_continuation_exhausted(
        resumed_completed_checkpoint=False,
        baseline=baseline,
        current=dict(baseline),
        portfolio_accepted=False,
    )
    assert not _automatic_continuation_exhausted(
        resumed_completed_checkpoint=True,
        baseline=baseline,
        current={**baseline, "attempt_count": 13},
        portfolio_accepted=False,
    )


def test_no_action_scheduler_pass_is_terminal_without_manual_resume() -> None:
    assert _bounded_scheduler_exhausted(
        {"termination": "no_action", "execution_count": 121},
        portfolio_accepted=False,
    )
    assert _bounded_scheduler_exhausted(
        {"termination": "converged_low_marginal_gain"},
        portfolio_accepted=False,
    )
    assert not _bounded_scheduler_exhausted(
        {"termination": "action_limit"},
        portfolio_accepted=False,
    )
    assert not _bounded_scheduler_exhausted(
        {"termination": "no_action"},
        portfolio_accepted=True,
    )


def test_current_disposition_separates_hypotheses_from_validated_routes() -> None:
    common = {
        "kernel_status": "active",
        "stop_decision": {"decision": "continue", "terminal": False},
        "claim": {"accepted_under_configured_policy": False},
    }
    hypotheses = _current_disposition(
        **common,
        gates={
            "gates": {
                "B1_global_multi_route": True,
                "B2_host_validated_routes": False,
                "B3_exact_multi_source": False,
                "B4_stock_boundary": False,
                "B5_configured_portfolio_acceptance": False,
            }
        },
    )
    validated = _current_disposition(
        **common,
        gates={
            "gates": {
                "B1_global_multi_route": True,
                "B2_host_validated_routes": True,
                "B3_exact_multi_source": False,
                "B4_stock_boundary": False,
                "B5_configured_portfolio_acceptance": False,
            }
        },
    )

    assert hypotheses["state"] == "route_hypotheses_available_validation_open"
    assert "host_route_validation_open" in hypotheses["reasons"]
    assert validated["state"] == "routes_validated_proof_open"
    assert "host_route_validation_open" not in validated["reasons"]


def test_target_solver_reports_provider_failure_without_replan_or_false_closure(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="unknown failed provider target",
        target_smiles=TARGET,
        run_id="blind-provider-failure",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=True,
            enable_live_benchmark_stock=False,
        ),
        director_runner=_failed_runner,
    )

    assert result["director_outcomes"] == [
        {
            "schema_version": "global_campaign_director_outcome.v1",
            "status": "failed",
            "invoked": True,
            "cache_hit": False,
            "mode": "initial_architecture",
            "context_sha256": "",
            "plan": None,
            "proposal_audits": [],
            "contract_repairs": [],
            "reasons": [
                "GlobalCampaignDirectorError",
                "director_child_failed:provider_unavailable",
            ],
            "artifact_sha256": "",
            "task_id": "",
            "resource_usage": {},
        }
    ]
    assert result["model_cost"]["model_invocations"] == 1
    assert result["gates"]["gates"]["B0_blind_input"] is True
    assert all(
        value is False for key, value in result["gates"]["gates"].items() if key != "B0_blind_input"
    )
    assert result["claim"]["accepted_under_configured_policy"] is False
    global_stage = next(row for row in result["stages"] if row["stage"] == "global_campaign")
    assert global_stage["detail"]["reasons"][-1].endswith("provider_unavailable")
    assert Path(result["report_path"]).is_file()


def test_target_solver_pauses_on_typed_provider_runtime_failure_without_charge(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))

    def unavailable_runner(
        spec: AgentSpec,
        _context: Any,
        _mode: str,
        _config: Any,
    ) -> AgentResult:
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:provider-error",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.FAILED,
            error="model_provider_unavailable:provider_auth_unavailable",
            usage={"model_invocations": 0, "provider_failure_count": 1},
        )

    config = TargetSolveConfig(
        use_coordinator=False,
        enable_chemenzy=False,
        enable_web_search=False,
        enable_replan=False,
        enable_live_benchmark_stock=False,
    )
    result = gateway.solve_target(
        target_name="typed provider outage",
        target_smiles=TARGET,
        run_id="typed-provider-runtime-outage",
        config=config,
        director_runner=unavailable_runner,
    )

    assert result["stop_decision"]["decision"] == "paused"
    assert result["stop_decision"]["terminal"] is False
    assert result["model_cost"]["model_invocations"] == 0
    assert result["director_outcomes"][0]["status"] == "runtime_unavailable"
    assert result["director_outcomes"][0]["runtime_pause"] is True
    service = gateway._open(result["run_id"], run_dir=Path(result["run_dir"]))
    assert service.kernel.state.status == "paused"
    assert any(
        str(task_id).startswith("campaign-action:")
        for task_id in service.kernel.state.in_flight_tasks
    )
    assert any(
        str(task_id).startswith("director:") for task_id in service.kernel.state.in_flight_tasks
    )
    original_director_task_ids = tuple(
        str(task_id)
        for task_id in service.kernel.state.in_flight_tasks
        if str(task_id).startswith("director:")
    )
    recovered_runner_task_ids: list[str] = []

    def recovered_runner(
        spec: AgentSpec,
        context: Any,
        mode: str,
        director_config: Any,
    ) -> AgentResult:
        recovered_runner_task_ids.append(spec.agent_id)
        return _runner(spec, context, mode, director_config)

    resumed = gateway.solve_target(
        target_name="typed provider outage",
        target_smiles=TARGET,
        run_id="typed-provider-runtime-outage",
        config=config,
        director_runner=recovered_runner,
        resume=True,
    )
    resumed_service = gateway._open(resumed["run_id"], run_dir=Path(resumed["run_dir"]))
    assert resumed["model_cost"]["model_invocations"] == 1
    assert resumed["director_outcomes"][0]["status"] == "accepted"
    assert resumed_service.kernel.state.in_flight_tasks == {}
    assert tuple(recovered_runner_task_ids) == original_director_task_ids
    reservations = resumed_service.kernel.task_reservation_history()
    assert (
        sum(
            str(dict(event.get("payload") or {}).get("task_id") or "") in original_director_task_ids
            for event in reservations
        )
        == 1
    )


def test_target_solver_reports_recoverable_partial_director_usage_and_routes(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))

    def partially_unavailable_runner(
        spec: AgentSpec,
        context: Any,
        mode: str,
        _config: Any,
    ) -> AgentResult:
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:recoverable-provider-error",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.FAILED,
            output=_plan(context, mode),
            error="model_provider_unavailable:provider_service_unavailable",
            usage={
                "model_invocations": 7,
                "input_tokens": 700,
                "output_tokens": 70,
                "provider_failure_count": 1,
                "resume_required_task_ids": ["critic:branch:2"],
                "provider_runtime_failure": {
                    "reason": "provider_service_unavailable",
                    "task_id": "critic:branch:2",
                },
            },
        )

    result = gateway.solve_target(
        target_name="recoverable partial director",
        target_smiles=TARGET,
        run_id="recoverable-partial-director",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_chemenzy=False,
            enable_web_search=False,
            enable_replan=False,
            enable_live_benchmark_stock=False,
        ),
        director_runner=partially_unavailable_runner,
    )

    outcome = result["director_outcomes"][0]
    assert result["stop_decision"]["decision"] == "paused"
    assert result["model_cost"]["model_invocations"] == 7
    assert result["model_cost"]["includes_unsettled_checkpoint_observations"] is True
    assert outcome["status"] == "runtime_unavailable"
    assert outcome["runtime_pause"] is True
    assert outcome["resume_required_task_ids"] == ["critic:branch:2"]
    assert len(outcome["plan"]["multi_step_skeletons"]) == 3
    service = gateway._open(result["run_id"], run_dir=Path(result["run_dir"]))
    assert service.kernel.state.model_totals["model_invocations"] == 0
    director_task_ids = [
        str(task_id)
        for task_id in service.kernel.state.in_flight_tasks
        if str(task_id).startswith("director:")
    ]
    assert len(director_task_ids) == 1
    assert len(service.kernel.task_lifecycle(director_task_ids[0])["checkpoints"]) == 1


def test_initial_director_limits_are_capped_by_run_budget(tmp_path: Path) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    observed: list[Any] = []

    def recording_runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        observed.append(config)
        return _runner(spec, context, mode, config)

    gateway.solve_target(
        target_name="director budget cap",
        target_smiles=TARGET,
        run_id="director-budget-cap",
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=600,
            max_total_wall_time_s=30.0,
            max_accepted_expansions=16,
            max_attempt_runs=32,
        ),
        config=TargetSolveConfig(
            enable_chemenzy=False,
            enable_web_search=False,
            enable_replan=False,
            enable_live_benchmark_stock=False,
        ),
        director_runner=recording_runner,
    )

    assert observed[0].max_output_tokens == 600
    assert observed[0].max_wall_time_s == 30.0


def test_fast_execution_profile_bounds_global_dossier(tmp_path: Path) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    observed: list[Any] = []

    def recording_runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        observed.append(config)
        return _runner(spec, context, mode, config)

    gateway.solve_target(
        target_name="fast profile",
        target_smiles=TARGET,
        run_id="fast-profile",
        config=TargetSolveConfig(
            execution_profile="fast",
            strategy_search_profile="legacy_global",
            enable_chemenzy=False,
            enable_web_search=False,
            enable_replan=False,
            enable_live_benchmark_stock=False,
        ),
        director_runner=recording_runner,
    )

    assert observed[0].minimum_route_families == 2
    assert observed[0].max_route_families == 2
    assert observed[0].max_skeletons == 2
    assert observed[0].max_steps_per_skeleton == 5
    assert observed[0].max_output_tokens == 3_800
    assert observed[0].max_tool_calls == 4


def test_proof_execution_profile_can_represent_long_route_skeletons(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    observed: list[Any] = []

    def recording_runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        observed.append(config)
        return _runner(spec, context, mode, config)

    gateway.solve_target(
        target_name="long route proof profile",
        target_smiles=TARGET,
        run_id="long-route-proof-profile",
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=20_000,
            max_total_output_tokens=20_000,
            max_total_wall_time_s=60.0,
            max_accepted_expansions=16,
            max_attempt_runs=32,
        ),
        config=TargetSolveConfig(
            execution_profile="proof",
            strategy_search_profile="legacy_global",
            enable_chemenzy=False,
            enable_web_search=False,
            enable_replan=False,
            enable_live_benchmark_stock=False,
        ),
        director_runner=recording_runner,
    )

    assert observed[0].max_steps_per_skeleton == 24
    assert observed[0].max_output_tokens == 18_000


def test_target_solver_starts_chemenzy_and_codex_from_one_frozen_revision(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    director_contexts: list[Any] = []
    chemenzy_started = Event()
    codex_started = Event()

    def recording_runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        codex_started.set()
        assert chemenzy_started.wait(timeout=2.0)
        director_contexts.append(context)
        return _runner(spec, context, mode, config)

    def chemenzy_provider(**_kwargs: Any) -> dict[str, Any]:
        chemenzy_started.set()
        assert codex_started.wait(timeout=2.0)
        return {
            "status": "completed",
            "routes": [
                {
                    "solved": True,
                    "steps": [
                        {
                            "product_smiles": TARGET,
                            "reactant_smiles": ["CCO", "CC(=O)Cl"],
                            "source_model": "fixture-chem-enzy",
                            "enzyme_ec_annotations": [
                                {
                                    "ec_number": "3.1.1.-",
                                    "enzyme_class": "acyltransferase",
                                }
                            ],
                            "selectivity_objective": "chemoselective ester formation",
                            "condition_predictions": [
                                {
                                    "temperature_c": 25,
                                    "solvent": "dichloromethane",
                                    "source": "fixture-condition-model",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

    result = gateway.solve_target(
        target_name="ChemEnzy plus Codex target",
        target_smiles=TARGET,
        run_id="blind-chemenzy-codex",
        config=TargetSolveConfig(
            enable_chemenzy=True,
            enable_target_chemenzy_baseline=True,
            enable_web_search=False,
            enable_replan=False,
            enable_live_benchmark_stock=False,
        ),
        director_runner=recording_runner,
        chemenzy_provider=chemenzy_provider,
    )

    stage = next(value for value in result["stages"] if value["stage"] == "chemenzy_baseline")
    assert stage["detail"]["proposal_count"] == 1
    assert stage["detail"]["provider_envelope"]["accepted"] is True
    assert stage["detail"]["provider_envelope"]["provider_kind"] == "proposal"
    assert stage["detail"]["provider_envelope"]["no_solved_claim"] is True
    assert stage["detail"]["provider_envelope"]["normalized_candidate_count"] == 1
    assert stage["detail"]["provider_registration"]["trust"]["trusted"] is True
    lineage = stage["detail"]["route_lineage"]
    assert len(lineage) == 1
    assert lineage[0]["host_portfolio_selected"] is True
    assert len(lineage[0]["raw_route_sha256"]) == 64
    assert len(lineage[0]["normalized_route_sha256"]) == 64
    assert lineage[0]["canonical_route_family_id"]
    assert director_contexts[0].evidence["chemenzy_provider_observation"] == {}
    assert not any(
        "chemenzy" in row.get("origin_kinds", [])
        for row in director_contexts[0].topology["hypotheses"].values()
    )
    anytime_core = next(
        value for value in result["stages"] if value["stage"] == "campaign_anytime_core"
    )["detail"]
    start_cohort = anytime_core["start_cohort"]
    assert start_cohort["status"] == "completed"
    assert start_cohort["max_in_flight_action_count"] == 2
    assert start_cohort["semantics"]["all_actions_bound_to_one_input_revision"] is True
    latency_audit = start_cohort["latency_audit"]
    assert latency_audit["applicable"] is True
    assert latency_audit["accepted"] is True
    assert latency_audit["both_initial_providers_submitted_before_either_completed"] is True
    assert latency_audit["chemenzy_first_proposal"]["nonempty_raw_proposal_observed"] is True
    assert anytime_core["first_result_timing"] == latency_audit["chemenzy_first_proposal"]
    service = gateway._open(result["run_id"], run_dir=Path(result["run_dir"]))
    origins = {
        origin["origin_kind"]
        for edge in service.graph_store.load()["edges"].values()
        for origin in edge["origin_records"]
    }
    assert {"chemenzy", "codex_global_director"} <= origins
    chem_enzy_edge = next(
        edge
        for edge in service.graph_store.load()["edges"].values()
        if any(origin["origin_kind"] == "chemenzy" for origin in edge["origin_records"])
    )
    assert chem_enzy_edge["condition_predictions"][0]["temperature_c"] == 25
    assert (
        chem_enzy_edge["condition_predictions"][0]["authority_scope"] == "model_predicted_condition"
    )
    assert chem_enzy_edge["condition_predictions"][0]["not_reaction_proof"] is True
    enzyme_option = chem_enzy_edge["route_innovations"][0]
    assert enzyme_option["kind"] == "biocatalytic_step"
    assert enzyme_option["enzyme"]["ec_numbers"] == ["3.1.1.-"]
    assert enzyme_option["not_reaction_proof"] is True

    final_lineage = next(
        value for value in result["stages"] if value["stage"] == "chemenzy_route_lineage"
    )["detail"]
    assert final_lineage["route_count"] >= 1
    seed_lineage = next(row for row in final_lineage["routes"] if row["provider_mode"] == "seed")
    assert seed_lineage["canonical_hypothesis_ids"]
    lineage_payload = dict(final_lineage)
    lineage_sha256 = lineage_payload.pop("content_sha256")
    assert (
        lineage_sha256
        == hashlib.sha256(
            json.dumps(
                lineage_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    provenance = result["candidate_provenance"]
    assert provenance["ignored_provider_lineage_count"] == 0
    assert provenance["provider_route_count"] == final_lineage["route_count"]
    provider_route = next(
        row
        for row in provenance["provider_route_records"]
        if row["route_trace_id"] == lineage[0]["route_trace_id"]
    )
    assert provider_route["candidate_ids"]
    assert provider_route["raw_route_sha256"] == lineage[0]["raw_route_sha256"]
    assert provider_route["normalized_route_sha256"] == lineage[0]["normalized_route_sha256"]
    assert any(
        row["provider_normalization"]["route_trace_ids"] for row in provenance["candidate_records"]
    )


def test_stock_result_cancels_default_codex_peer_after_progressive_b4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    codex_started = Event()
    cancel_observations: list[bool] = []

    def cancellable_default_runner(
        spec: AgentSpec,
        context: Any,
        mode: str,
        config: Any,
        *,
        cancel_event: Event | None = None,
    ) -> AgentResult:
        assert cancel_event is not None
        codex_started.set()
        assert cancel_event.wait(timeout=3.0)
        cancel_observations.append(cancel_event.is_set())
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.CANCELLED,
            error="delivery_milestone_reached",
            usage={"model_invocations": 1, "wall_time_s": 0.1},
        )

    monkeypatch.setattr(
        "cascade_planner.interfaces.target_solver.run_codex_cli_director_child",
        cancellable_default_runner,
    )

    def chemenzy_provider(**_kwargs: Any) -> dict[str, Any]:
        assert codex_started.wait(timeout=2.0)
        return {
            "status": "completed",
            "routes": [
                {
                    "steps": [
                        {
                            "product": TARGET,
                            "main_reactant": "CCO",
                            "aux_reactants": ["CC(=O)Cl"],
                            "source_model": "fixture-progressive-chemenzy",
                        }
                    ]
                }
            ],
        }

    result = gateway.solve_target(
        target_name="progressive B4 cancellation",
        target_smiles=TARGET,
        run_id="progressive-b4-cancellation",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=1,
            minimum_edge_proof_level=2,
            require_all_selected_leaves_stock_closed=True,
            stock_boundary="benchmark_search",
            minimum_independent_source_groups=1,
            require_distinct_edge_sets=False,
        ),
        config=TargetSolveConfig(
            delivery_boundary="stock_result",
            strategy_search_profile="legacy_global",
            enable_chemenzy=True,
            enable_target_chemenzy_baseline=True,
            enable_web_search=False,
            enable_replan=False,
            enable_builtin_patent_evidence=False,
        ),
        chemenzy_provider=chemenzy_provider,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )

    assert cancel_observations == [True]
    assert result["gates"]["gates"]["B4_stock_boundary"] is True
    anytime = next(
        stage for stage in result["stages"] if stage["stage"] == "campaign_anytime_core"
    )["detail"]
    assert anytime["termination"] == "milestone_reached"
    codex_execution = next(
        row
        for row in anytime["start_cohort"]["executions"]
        if row["action"]["kind"] == CampaignActionKind.CODEX_GLOBAL_ARCHITECTURE.value
    )
    assert codex_execution["status"] == "cancelled_after_delivery"
    assert codex_execution["outcome"]["failure_type"] == ""
    service = gateway._open(result["run_id"], run_dir=Path(result["run_dir"]))
    assert service.kernel.state.in_flight_tasks == {}


def test_legacy_benchmark_label_does_not_short_circuit_unified_campaign(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    director_modes: list[str] = []
    director_contexts: list[Any] = []

    def recording_director(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        director_modes.append(mode)
        director_contexts.append(context)
        return _runner(spec, context, mode, config)

    def chemenzy_provider(**_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "routes": [
                {
                    "score": score,
                    "steps": [
                        {
                            "product_smiles": TARGET,
                            "reactant_smiles": ["CCO", precursor],
                            "rxn_smiles": f"CCO.{precursor}>>{TARGET}",
                            "source_model": "fixture-chemenzy",
                            "score": score,
                            "stock_status": {"CCO": True, precursor: True},
                            "raw_backend_metadata": {
                                "template": {"template_id": f"fixture:{index}"}
                            },
                        }
                    ],
                }
                for index, (precursor, score) in enumerate(
                    (("CC(=O)Cl", 0.9), ("CC(=O)O", 0.8), ("CC(=O)Br", 0.7)),
                    start=1,
                )
            ],
        }

    result = gateway.solve_target(
        target_name="benchmark seed closure",
        target_smiles=TARGET,
        run_id="benchmark-seed-closure",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=2,
            stock_boundary="benchmark_search",
            minimum_independent_source_groups=2,
        ),
        config=TargetSolveConfig(
            objective_mode="benchmark_search",
            enable_target_chemenzy_baseline=True,
            enable_target_identity=False,
            enable_web_search=False,
            enable_replan=True,
            enable_builtin_patent_evidence=False,
            enable_patent_self_evolution=True,
            provider_route_reserve=16,
            host_route_portfolio=8,
            display_route_limit=4,
        ),
        director_runner=recording_director,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        chemenzy_provider=chemenzy_provider,
    )

    assert director_modes
    assert director_modes[0] == "initial_architecture"
    assert result["model_cost"]["model_invocations"] >= 1
    assert result["gates"]["gates"]["B4_stock_boundary"] is True
    configured_accepted = result["gates"]["gates"]["B5_configured_portfolio_acceptance"]
    assert result["claim"]["objective_achieved"] is configured_accepted
    assert result["claim"]["benchmark_search_completed"] is True
    assert result["claim"]["scientific_proof_accepted"] is configured_accepted
    assert result["claim"]["semantics"]["objective_mode_is_compatibility_metadata_only"] is True
    assert result["current_disposition"]["state"] == (
        "accepted" if configured_accepted else "stock_closed_proof_open"
    )
    stage_names = {stage["stage"] for stage in result["stages"]}
    assert "chemenzy_seed_materialization" in stage_names
    assert "chemenzy_seed_stock" in stage_names
    assert "campaign_milestone" in stage_names
    assert "global_campaign" in stage_names
    assert "evidence_acquisition" in stage_names
    service = gateway._open(result["run_id"], run_dir=Path(result["run_dir"]))
    assert director_contexts[0].evidence["chemenzy_provider_observation"] == {}
    provider_lineage = next(
        stage["detail"] for stage in result["stages"] if stage["stage"] == "chemenzy_route_lineage"
    )
    assert provider_lineage["route_count"] == 3
    assert all(
        row["canonical_hypothesis_ids"] and row["canonical_edge_ids"]
        for row in provider_lineage["routes"]
    )
    assert all(
        row["final_disposition"]
        in {
            "stock_closed",
            "materialized_or_partially_materialized",
            "canonical_edges_present_outside_complete_measured_route",
        }
        for row in provider_lineage["routes"]
    )
    provider_origins = [
        origin
        for edge in service.graph_store.load()["edges"].values()
        for origin in edge.get("origin_records") or []
        if origin.get("origin_kind") == "chemenzy"
    ]
    assert provider_origins
    assert all(origin["provider_reaction_metadata_digest_valid"] for origin in provider_origins)
    assert all(
        len(origin["provider_reaction_metadata_sha256"]) == 64 for origin in provider_origins
    )
    graph = service.graph_store.load()
    provider_edge_ids = {
        edge_id
        for edge_id, edge in graph["edges"].items()
        if any(origin.get("origin_kind") == "chemenzy" for origin in edge["origin_records"])
    }
    validation = next(
        stage["detail"] for stage in result["stages"] if stage["stage"] == "reaction_validation"
    )
    assert set(validation["accepted_edge_ids"]) == provider_edge_ids
    provider_leaf_ids = {
        molecule_id
        for edge_id in provider_edge_ids
        for molecule_id in graph["edges"][edge_id]["precursor_molecule_ids"]
    }
    assert provider_leaf_ids
    assert all(
        graph["molecules"][molecule_id]["active_stock_observation_id"]
        for molecule_id in provider_leaf_ids
    )
    assert all(
        graph["stock_observations"][graph["molecules"][molecule_id]["active_stock_observation_id"]][
            "accepted"
        ]
        is True
        for molecule_id in provider_leaf_ids
    )


def test_b4_milestone_keeps_scientific_actions_on_the_same_trajectory(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))

    def director_without_condition_suggestions(
        spec: AgentSpec, context: Any, mode: str, _config: Any
    ) -> AgentResult:
        plan = _plan(context, mode)
        for skeleton in plan["multi_step_skeletons"]:
            for step in skeleton["steps"]:
                step.pop("condition_predictions", None)
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.SUCCEEDED,
            output=plan,
            usage={
                "model_invocations": 1,
                "input_tokens": 1_000,
                "output_tokens": 700,
                "wall_time_s": 1.0,
            },
        )

    def chemenzy_provider(**_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "routes": [
                {
                    "score": 0.9,
                    "steps": [
                        {
                            "product_smiles": TARGET,
                            "reactant_smiles": ["CCO", "CC(=O)Cl"],
                            "rxn_smiles": f"CCO.CC(=O)Cl>>{TARGET}",
                            "source_model": "fixture-chemenzy",
                            "score": 0.9,
                            "stock_status": {"CCO": True, "CC(=O)Cl": True},
                        }
                    ],
                }
            ],
        }

    def condition_predictor(_reaction_smiles: str, *, top_k: int = 2) -> list[dict[str, Any]]:
        assert top_k == 2
        return [
            {
                "temperature_c": 20,
                "solvent": ["tetrahydrofuran"],
                "reagents": ["base"],
                "score": 0.8,
            }
        ]

    result = gateway.solve_target(
        target_name="post-B4 science trajectory",
        target_smiles=TARGET,
        run_id="post-b4-science-trajectory",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=1,
            minimum_edge_proof_level=3,
            stock_boundary="benchmark_search",
            minimum_independent_source_groups=2,
        ),
        config=TargetSolveConfig(
            enable_target_chemenzy_baseline=True,
            enable_target_identity=False,
            enable_web_search=False,
            enable_builtin_patent_evidence=False,
            enable_patent_self_evolution=False,
            provider_route_reserve=8,
            host_route_portfolio=4,
            display_route_limit=2,
        ),
        director_runner=director_without_condition_suggestions,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        chemenzy_provider=chemenzy_provider,
        evidence_connector=_discovery_only_connector,
        condition_predictor=condition_predictor,
    )

    stages = result["stages"]
    core_actions = [
        (
            index,
            str(dict(stage["detail"]["action"])["kind"]),
        )
        for index, stage in enumerate(stages)
        if str(stage.get("stage") or "").startswith("campaign_action_unified_core_")
    ]
    first_b4_snapshot_index = next(
        index
        for index, stage in enumerate(stages)
        if str(stage.get("stage") or "").startswith("campaign_snapshot_unified_core_")
        and dict(stage["detail"]["milestones"]).get("B4_stock_boundary") is True
    )
    action_kinds = {kind for _, kind in core_actions}
    post_b4_action_kinds = {kind for index, kind in core_actions if index > first_b4_snapshot_index}
    post_b4_condition = next(
        stage
        for index, stage in enumerate(stages)
        if index > first_b4_snapshot_index
        and str(stage.get("stage") or "").startswith("campaign_action_unified_core_")
        and dict(stage["detail"]["action"]).get("kind") == CampaignActionKind.CONDITION_ENRICH.value
    )

    assert result["gates"]["gates"]["B4_stock_boundary"] is True
    assert {
        CampaignActionKind.CODEX_REPLAN.value,
        CampaignActionKind.BIND_EVIDENCE.value,
        CampaignActionKind.CONDITION_ENRICH.value,
        CampaignActionKind.PROGRAM_REVIEW.value,
    } <= action_kinds
    assert {
        CampaignActionKind.CONDITION_ENRICH.value,
        CampaignActionKind.CODEX_REPLAN.value,
        CampaignActionKind.PROGRAM_REVIEW.value,
    } <= post_b4_action_kinds
    assert result["gates"]["gates"]["B3_exact_multi_source"] is False
    assert result["gates"]["gates"]["B5_configured_portfolio_acceptance"] is False
    assert result["model_cost"]["model_invocations"] == 2
    condition_decision = dict(post_b4_condition["detail"]["decision"])
    assert (
        condition_decision["scientific_closure_pressure"]["route_maturity"]
        == "stock_closed_route_portfolio"
    )
    assert (
        dict(condition_decision["selected_action"])["schedule_components"][
            "scientific_closure_pressure_bonus"
        ]
        > 0.0
    )
    assert any(
        stage["stage"] == "replan_retention_audit" and stage["status"] == "accepted"
        for stage in stages
    )
    assert any(
        stage["stage"] == "program_review"
        and stage["detail"]["store"]["status"]["event_count"] == 0
        for stage in stages
    )
    trajectory = result["trajectory"]
    assert trajectory["schema_version"] == "campaign_trajectory.v2"
    assert trajectory["continuity"]["resume_baseline_preserved"] is True
    assert trajectory["time_to_first"]["B4"] is not None
    assert trajectory["time_to_first"]["B4"]["elapsed_wall_time_s"] >= 0.0
    assert all(
        snapshot["schema_version"] == "campaign_anytime_snapshot.v2"
        and snapshot["bindings"]["complete"] is True
        and "tasks" in snapshot["resource_usage"]
        and "native_search" in snapshot["resource_usage"]
        and "candidate_route_count" in snapshot["route_counts"]
        for snapshot in trajectory["snapshots"]
    )
    assert trajectory["snapshots"][-1]["action_counts"]["total"] == len(core_actions)
    assert trajectory["snapshots"][-1]["pareto_archive"]
    final_bindings = trajectory["snapshots"][-1]["bindings"]
    assert final_bindings["code"]["value"]["source_bundle_complete"] is True
    assert len(final_bindings["input"]["value"]["campaign_spec_sha256"]) == 64
    assert len(final_bindings["stock_oracle"]["value"]["reference_sha256"]) == 64
    assert final_bindings["providers"]["value"]["codex"]["model"]
    assert (
        final_bindings["providers"]["value"]["chemenzy"]["runtime"]["provider_envelope"][
            "provider_id"
        ]
        == "autoplanner.chemenzy_proposals"
    )
    workbench_history = gateway.workbench("post-b4-science-trajectory")["snapshot"][
        "campaign_summary"
    ]["trajectory_history"]
    assert workbench_history["available"] is True
    assert workbench_history["trajectory_sha256"] == trajectory["content_sha256"]
    assert workbench_history["resource_curve"] == trajectory["resource_curve"]
    exported = gateway.export(
        "post-b4-science-trajectory",
        output_dir=tmp_path / "post-b4-review-export",
    )
    review_bundle = json.loads(Path(exported["files"]["review_bundle"]).read_text(encoding="utf-8"))
    assert review_bundle["available"] is True
    assert review_bundle["source_report_digest_valid"] is True
    assert review_bundle["report_sha256"] == result["content_sha256"]
    assert review_bundle["components"]["action_trace"]["record_count"] > 0
    assert any(
        record["kind"] == "canonical_pareto_lineage"
        for record in review_bundle["components"]["route_lineage"]["records"]
    )
    resource_export = review_bundle["components"]["resource_curve"]
    assert resource_export["available"] is True
    assert resource_export["trajectory_sha256"] == trajectory["content_sha256"]
    assert resource_export["records"] == trajectory["resource_curve"]


def test_legacy_objective_labels_produce_the_same_campaign_trace(
    tmp_path: Path,
) -> None:
    delays = {"chemenzy": 0.0, "codex": 0.0}
    provider_target_names: list[str] = []
    gateways: dict[str, CampaignGateway] = {}

    def director_runner(spec, context, mode, config):
        time.sleep(delays["codex"])
        return _runner(spec, context, mode, config)

    def chemenzy_provider(**kwargs: Any) -> dict[str, Any]:
        time.sleep(delays["chemenzy"])
        provider_target_names.append(str(kwargs.get("target_name") or ""))
        return {
            "status": "completed",
            "routes": [
                {
                    "score": 0.9,
                    "steps": [
                        {
                            "product_smiles": TARGET,
                            "reactant_smiles": ["CCO", "CC(=O)Cl"],
                            "rxn_smiles": f"CCO.CC(=O)Cl>>{TARGET}",
                            "source_model": "fixture-chemenzy",
                            "score": 0.9,
                        }
                    ],
                }
            ],
        }

    def solve(label: str) -> dict[str, Any]:
        delays.update(
            {"chemenzy": 0.04, "codex": 0.0}
            if label == "benchmark_search"
            else {"chemenzy": 0.0, "codex": 0.04}
        )
        run_root = tmp_path / f"fresh-{label}"
        run_root.mkdir()
        gateway = CampaignGateway(_paths(run_root))
        gateways[label] = gateway
        return gateway.solve_target(
            target_name="display-only compatibility target",
            target_smiles=TARGET,
            run_id="objective-blind-shared",
            config=TargetSolveConfig(
                objective_mode=label,
                enable_target_chemenzy_baseline=True,
                enable_target_identity=False,
                enable_web_search=False,
                enable_replan=False,
                enable_live_benchmark_stock=False,
                enable_builtin_patent_evidence=False,
                enable_patent_self_evolution=False,
                enable_condition_enrichment=False,
                enable_chemenzy_condition_prediction=False,
            ),
            director_runner=director_runner,
            atom_mapper=_mapper,
            stock_catalog_builder=_catalog,
            chemenzy_provider=chemenzy_provider,
        )

    benchmark = solve("benchmark_search")
    scientific = solve("scientific_proof")
    procurement = solve("procurement_delivery")

    for comparison in (scientific, procurement):
        assert [row["stage"] for row in benchmark["stages"]] == [
            row["stage"] for row in comparison["stages"]
        ]
        assert benchmark["gates"]["gates"] == comparison["gates"]["gates"]
        assert benchmark["model_cost"] == comparison["model_cost"]
    benchmark_lineage = next(
        row for row in benchmark["stages"] if row["stage"] == "chemenzy_baseline"
    )["detail"]["route_lineage"]
    for comparison in (scientific, procurement):
        comparison_lineage = next(
            row for row in comparison["stages"] if row["stage"] == "chemenzy_baseline"
        )["detail"]["route_lineage"]
        assert [row["normalized_route_sha256"] for row in benchmark_lineage] == [
            row["normalized_route_sha256"] for row in comparison_lineage
        ]
    assert len(set(provider_target_names)) == 1
    assert provider_target_names[0].startswith("target-")
    assert all("display-only" not in value for value in provider_target_names)

    def action_decision_signature(
        report: dict[str, Any],
    ) -> list[tuple[str, str, str]]:
        def selected_action_id(row: dict[str, Any]) -> str:
            return str(dict(row["detail"].get("decision") or {}).get("selected_action_id") or "")

        return [
            (
                str(dict(row["detail"]["action"]).get("kind") or ""),
                selected_action_id(row),
                str(dict(row["detail"].get("outcome") or {}).get("status") or ""),
            )
            for row in report["stages"]
            if str(row.get("stage") or "").startswith("campaign_action_unified_core_")
        ]

    def action_binding_signature(
        report: dict[str, Any],
    ) -> list[tuple[str, str, str]]:
        return [
            (
                str(dict(row["detail"]["action"]).get("execution_id") or ""),
                str(dict(row["detail"].get("decision") or {}).get("selected_action_id") or ""),
                str(dict(row["detail"].get("outcome") or {}).get("status") or ""),
            )
            for row in report["stages"]
            if str(row.get("stage") or "").startswith("campaign_action_unified_core_")
        ]

    def action_stage_names(report: dict[str, Any]) -> list[str]:
        return [
            str(row.get("stage") or "")
            for row in report["stages"]
            if str(row.get("stage") or "").startswith("campaign_action_unified_core_")
        ]

    def write_json(path: Path, value: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for comparison in (scientific, procurement):
        assert action_decision_signature(benchmark) == action_decision_signature(comparison)
        assert action_binding_signature(benchmark) == action_binding_signature(comparison)
    benchmark_graph = (
        gateways["benchmark_search"]
        ._open(
            "objective-blind-shared",
            run_dir=benchmark["run_dir"],
        )
        .graph_store.load()
    )
    scientific_graph = (
        gateways["scientific_proof"]
        ._open(
            "objective-blind-shared",
            run_dir=scientific["run_dir"],
        )
        .graph_store.load()
    )
    procurement_graph = (
        gateways["procurement_delivery"]
        ._open(
            "objective-blind-shared",
            run_dir=procurement["run_dir"],
        )
        .graph_store.load()
    )
    assert {
        benchmark_graph["scientific_sha256"],
        scientific_graph["scientific_sha256"],
        procurement_graph["scientific_sha256"],
    } == {benchmark_graph["scientific_sha256"]}

    base_run_dir = Path(benchmark["run_dir"])
    legacy_run_dir = tmp_path / "saved-run-with-legacy-objective"
    no_legacy_run_dir = tmp_path / "saved-run-without-legacy-objective"
    shutil.copytree(base_run_dir, legacy_run_dir)
    shutil.copytree(base_run_dir, no_legacy_run_dir)
    base_before_actions = action_decision_signature(benchmark)
    base_before_bindings = action_binding_signature(benchmark)

    benchmark_checkpoint_path = legacy_run_dir / ".autoplanner" / "target-solver-checkpoint.json"
    legacy_checkpoint = json.loads(benchmark_checkpoint_path.read_text(encoding="utf-8"))
    legacy_checkpoint["objective_mode"] = "benchmark_search"
    write_json(benchmark_checkpoint_path, legacy_checkpoint)

    no_legacy_checkpoint_path = no_legacy_run_dir / ".autoplanner" / "target-solver-checkpoint.json"
    no_legacy_checkpoint = json.loads(no_legacy_checkpoint_path.read_text(encoding="utf-8"))
    no_legacy_checkpoint.pop("objective_mode", None)
    if isinstance(no_legacy_checkpoint.get("config"), dict):
        no_legacy_checkpoint["config"].pop("objective_mode", None)
    write_json(no_legacy_checkpoint_path, no_legacy_checkpoint)

    no_legacy_report_path = no_legacy_run_dir / "target-only-solve-report.json"
    no_legacy_report = json.loads(no_legacy_report_path.read_text(encoding="utf-8"))
    no_legacy_report.pop("content_sha256", None)
    if isinstance(no_legacy_report.get("config"), dict):
        no_legacy_report["config"].pop("objective_mode", None)
    if isinstance(no_legacy_report.get("claim"), dict):
        no_legacy_report["claim"].pop("objective_mode", None)
    no_legacy_report["content_sha256"] = hashlib.sha256(
        json.dumps(
            no_legacy_report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    write_json(no_legacy_report_path, no_legacy_report)

    resume_config = TargetSolveConfig(
        objective_mode="procurement_delivery",
        enable_target_chemenzy_baseline=True,
        enable_target_identity=False,
        enable_web_search=False,
        enable_replan=False,
        enable_live_benchmark_stock=False,
        enable_builtin_patent_evidence=False,
        enable_patent_self_evolution=False,
        enable_condition_enrichment=False,
        enable_chemenzy_condition_prediction=False,
    )
    delays.update({"chemenzy": 0.0, "codex": 0.0})
    resume_gateway = gateways["benchmark_search"]
    resumed_benchmark = resume_gateway.solve_target(
        target_name="renamed compatibility view",
        target_smiles=TARGET,
        run_id="objective-blind-shared",
        run_dir=legacy_run_dir,
        resume=True,
        config=resume_config,
        director_runner=director_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        chemenzy_provider=chemenzy_provider,
    )
    delays.update({"chemenzy": 0.0, "codex": 0.0})
    resumed_without_legacy = resume_gateway.solve_target(
        target_name="renamed compatibility view",
        target_smiles=TARGET,
        run_id="objective-blind-shared",
        run_dir=no_legacy_run_dir,
        resume=True,
        config=resume_config,
        director_runner=director_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        chemenzy_provider=chemenzy_provider,
    )

    benchmark_after_actions = action_decision_signature(resumed_benchmark)
    no_legacy_after_actions = action_decision_signature(resumed_without_legacy)
    benchmark_after_bindings = action_binding_signature(resumed_benchmark)
    no_legacy_after_bindings = action_binding_signature(resumed_without_legacy)
    assert benchmark_after_actions[: len(base_before_actions)] == (base_before_actions)
    assert no_legacy_after_actions[: len(base_before_actions)] == (base_before_actions)
    assert benchmark_after_bindings[: len(base_before_bindings)] == (base_before_bindings)
    assert no_legacy_after_bindings[: len(base_before_bindings)] == (base_before_bindings)
    assert benchmark_after_actions == no_legacy_after_actions
    assert benchmark_after_bindings == no_legacy_after_bindings
    resumed_benchmark_graph = resume_gateway._open(
        "objective-blind-shared",
        run_dir=legacy_run_dir,
    ).graph_store.load()
    resumed_no_legacy_graph = resume_gateway._open(
        "objective-blind-shared",
        run_dir=no_legacy_run_dir,
    ).graph_store.load()
    assert (
        resumed_benchmark_graph["scientific_sha256"]
        == (resumed_no_legacy_graph["scientific_sha256"])
    )
    for after in (resumed_benchmark, resumed_without_legacy):
        before_names = action_stage_names(benchmark)
        after_names = action_stage_names(after)
        assert after_names[: len(before_names)] == before_names
        assert len(after_names) == len(set(after_names))

    benchmark_compatibility = next(
        row
        for row in resumed_benchmark["stages"]
        if row["stage"] == "saved_run_objective_compatibility"
    )["detail"]
    no_legacy_compatibility = next(
        row
        for row in resumed_without_legacy["stages"]
        if row["stage"] == "saved_run_objective_compatibility"
    )["detail"]
    assert benchmark_compatibility["legacy_objective_present"] is True
    assert no_legacy_compatibility["legacy_objective_present"] is False
    assert benchmark_compatibility["requested_compatibility_view"] == ("procurement_delivery")
    assert no_legacy_compatibility["requested_compatibility_view"] == ("procurement_delivery")
    assert {row["value"] for row in benchmark_compatibility["legacy_objective_observations"]} >= {
        "benchmark_search"
    }
    assert no_legacy_compatibility["legacy_objective_observations"] == []
    assert (
        benchmark_compatibility["semantics"][
            "resume_uses_current_unified_campaign_state_and_budget"
        ]
        is True
    )


def test_stock_rejected_leaf_continues_with_route_bound_builder_and_materializes(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    builder_contexts: list[Any] = []
    final_critic_contexts: list[Any] = []

    class FixtureSequentialRunner(SequentialStrategyDirectorRunner):
        def __call__(self, spec, context, mode, _config):
            return AgentResult(
                run_id=spec.run_id,
                agent_id=spec.agent_id,
                parent_agent_id=spec.parent_agent_id,
                attempt=spec.attempt,
                idempotency_key=f"{spec.idempotency_key}:result",
                context_hash=spec.context_hash,
                capabilities=spec.capabilities,
                write_scope=spec.write_scope,
                budget=spec.budget,
                state=AgentState.SUCCEEDED,
                output=_plan(context, mode),
                usage={
                    "model_invocations": 1,
                    "input_tokens": 100,
                    "output_tokens": 100,
                    "wall_time_s": 0.01,
                },
            )

        def run_frontier_builder_once(self, spec, *, context, config, prompt=None):
            del config, prompt
            builder_contexts.append(context)
            assert context.selected_product_smiles == "CCO"
            materialized = self.routejson_compiler.compile_step(
                mapped_product_smiles=context.selected_product_mapped,
                operations=({"op": "break_bond", "map_a": 6, "map_b": 7},),
                expected_product_smiles="CCO",
            )
            step_id = f"fixture:frontier:{context.route_family_id}"
            return (
                {
                    "status": "compiled",
                    "step": {
                        "step_id": step_id,
                        "product_smiles": materialized.product_smiles,
                        "precursor_smiles": list(materialized.precursor_smiles),
                        "mapped_product_smiles": materialized.mapped_product_smiles,
                        "mapped_precursor_smiles": list(materialized.mapped_precursor_smiles),
                        "transformation_hypothesis": "C-O disconnection",
                        "strategic_role": "advance a stock-rejected open leaf",
                        "step_role": "supporting",
                        "checkpoint_relation": "preparatory",
                        "source_hints": [],
                        "required_validation": [
                            "structure",
                            "reaction_feasibility",
                        ],
                        "hypothesis_only": True,
                        "condition_predictions": [],
                        "limitations": [],
                        "strategy_card": dict(context.strategy_card),
                        "reaction_operations": [{"op": "break_bond", "map_a": 6, "map_b": 7}],
                        "reaction_edit_digest": "",
                        "reactionjson_audit": dict(materialized.audit),
                        "strategy_id": str(context.strategy_card.get("strategy_id") or ""),
                        "strategy_digest": str(context.strategy_card.get("strategy_digest") or ""),
                        "step_kind": "chemical_reaction",
                        "execution_domain": "chemical",
                        "biocatalytic_step": {},
                        "biocatalytic_design_deficits": [],
                        "strategy_anchor": False,
                        "strategy_milestone_index": 1,
                    },
                    "candidate_id": step_id,
                    "candidate_count": 1,
                    "rejected_candidates": [],
                    "model_output_validation": {
                        "accepted": True,
                        "reasons": [],
                    },
                },
                WorkerRunRecord(
                    run_id=f"{spec.agent_id}:run",
                    task_id=spec.agent_id,
                    case_id="fixture-frontier-builder",
                    status="accepted_draft",
                    output_validation={"accepted": True, "reasons": []},
                    usage={
                        "model_invocations": 1,
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                    elapsed_s=0.01,
                ),
            )

        def run_final_route_critic_once(
            self,
            spec,
            *,
            context,
            config,
            prompt=None,
        ):
            del config, prompt
            final_critic_contexts.append(context)
            return (
                {
                    "schema_version": "chemical_strategy_critique.v1",
                    "status": "viable",
                    "overall_assessment": "viable",
                    "route_overall_evaluation": (
                        "The route is coherent and reaches simple stocked leaves. "
                        "Its main remaining need is focused experimental validation."
                    ),
                    "route_level_risks": ["substrate-scope uncertainty"],
                    "critic_task_id": spec.agent_id,
                    "step_assessments": [
                        {
                            "step_id": str(step.get("step_id") or ""),
                            "verdict": "pass",
                            "blocking": False,
                        }
                        for step in context.steps
                    ],
                },
                WorkerRunRecord(
                    run_id=f"{spec.agent_id}:run",
                    task_id=spec.agent_id,
                    case_id="fixture-final-route-critic",
                    status="accepted_draft",
                    output_validation={"accepted": True, "reasons": []},
                    usage={
                        # Some Codex worker records expose token usage without
                        # an explicit call count. The owning route-Critic
                        # boundary must still account for the entered call.
                        "model_invocations": 0,
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                    elapsed_s=0.01,
                ),
            )

    runner = FixtureSequentialRunner()
    result = gateway.solve_target(
        target_name="frontier builder fallback",
        target_smiles=TARGET,
        run_id="frontier-builder-fallback",
        config=TargetSolveConfig(
            execution_profile="self_correcting_sequential",
            enable_chemenzy=False,
            enable_web_search=False,
            enable_replan=False,
            enable_builtin_patent_evidence=False,
            enable_patent_self_evolution=False,
            enable_target_identity=False,
            strategy_branch_count=3,
            max_node_expansions_per_branch=1,
        ),
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_partial_catalog,
    )

    builder_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "route_builder_continuation"
    )
    assert builder_contexts
    assert len(final_critic_contexts) == len(builder_contexts)
    stage_names = [str(stage.get("stage") or "") for stage in result["stages"]]
    assert stage_names.index("final_route_critic") > stage_names.index("route_builder_continuation")
    assert stage_names.index("final_route_critic") < stage_names.index("closeout")
    final_critic_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "final_route_critic"
    )
    assert all(
        row["branch_id"] == row["branch_index"] + 1
        for row in final_critic_stage["detail"]["results"]
        if row["status"] in {"completed", "reused"}
    )
    assert all(
        row["route_overall_evaluation"].startswith("The route is coherent")
        for row in final_critic_stage["detail"]["results"]
        if row["status"] in {"completed", "reused"}
    )
    assert builder_stage["detail"]["builder_dispositions"]["materialized"] >= 1
    assert all(len(context.route_family_id) > 0 for context in builder_contexts)
    service = gateway._open(result["run_id"], run_dir=Path(result["run_dir"]))
    graph = service.graph_store.load()
    ethanol_id, _ = molecule_identity("CCO")
    ethane_id, _ = molecule_identity("CC")
    water_id, _ = molecule_identity("O")
    open_routes = target_solver_module.target_reachable_route_boundaries(graph)[
        "open_leaf_route_family_ids"
    ]
    assert not open_routes.get(ethanol_id)
    for new_leaf_id in (ethane_id, water_id):
        molecule = graph["molecules"][new_leaf_id]
        observation = graph["stock_observations"][molecule["active_stock_observation_id"]]
        assert observation["accepted"] is True
    builder_origins = [
        origin
        for edge in graph["edges"].values()
        for origin in edge.get("origin_records") or []
        if str(origin.get("origin_ref") or "").startswith("codex:frontier-builder:")
    ]
    assert builder_origins
    for critic_context in final_critic_contexts:
        route = graph["route_families"][critic_context.route_family_id]
        assert route["chemical_critic"]["review_state"] == "complete"
        assert route["chemical_critic"]["review_owner"] == "route_critic_agent"
        assert route["chemical_critic"]["reviewed_route_sha256"] == (critic_context.route_sha256)
        assert route["chemical_critic"]["route_overall_evaluation"].startswith(
            "The route is coherent"
        )
        critic_task_id = route["chemical_critic"]["critic_task_id"]
        settlement = service.kernel.task_lifecycle(critic_task_id)["settlement"]["payload"]
        assert settlement["model_usage"]["model_invocations"] == 1
        assert settlement["model_usage"]["wall_time_s"] > 0
    reviewed_call_count = len(final_critic_contexts)
    reused = target_solver_module._run_revision_bound_route_critics(
        service,
        director_runner=runner,
        director_config=DirectorConfig(),
        route_family_ids=target_solver_module._final_route_critic_family_ids(graph),
    )
    assert len(final_critic_contexts) == reviewed_call_count
    assert reused
    assert all(result["status"] == "reused" for result in reused)


def test_resume_reuses_fresh_negative_stock_audits_without_spending_attempts(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    config = TargetSolveConfig(
        use_coordinator=False,
        enable_web_search=False,
        enable_replan=False,
    )
    first = gateway.solve_target(
        target_name="partial stock blind molecule",
        target_smiles=TARGET,
        run_id="blind-partial-stock",
        config=config,
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_partial_catalog,
    )

    def unexpected_catalog_call(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("fresh negative stock audit should be reused")

    resumed = gateway.solve_target(
        target_name="partial stock blind molecule",
        target_smiles=TARGET,
        run_id="blind-partial-stock",
        config=config,
        resume=True,
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=unexpected_catalog_call,
    )

    assert first["gates"]["gates"]["B4_stock_boundary"] is False
    assert resumed["attempt_count"] == first["attempt_count"]
    stock_stage = next(stage for stage in resumed["stages"] if stage["stage"] == "stock")
    stock_detail = stock_stage["detail"]
    assert stock_detail["status"] == "reused"
    assert stock_detail["remaining_pending_candidate_count"] == 0
    assert stock_detail["miss_count"] > 0
    assert stock_detail["miss_count"] == (
        stock_detail["selected_stock_candidate_count"]
        - stock_detail["stock_closed_candidate_count"]
    )


def test_target_solver_ingests_connector_rows_before_stock_and_closeout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    evidence_prepare_started = Event()
    validation_prepare_started = Event()
    prepared_revisions: dict[str, int] = {}
    prepare_threads: dict[str, str] = {}
    original_evidence_prepare = target_solver_module._prepare_evidence_acquisition
    original_validation_prepare = target_solver_module.prepare_materialized_edge_validation

    def prepare_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
        service = args[0]
        prepared_revisions["evidence"] = int(service.graph_store.load().get("revision") or 0)
        prepare_threads["evidence"] = current_thread().name
        evidence_prepare_started.set()
        assert validation_prepare_started.wait(timeout=2.0)
        return original_evidence_prepare(*args, **kwargs)

    def prepare_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        service = args[0]
        graph = service.graph_store.load()
        opportunity_kinds = {
            str(row.get("kind") or "")
            for row in target_solver_module.compile_action_opportunities(
                dict(graph.get("deficit_frontier") or {})
            ).get("actions")
            or []
        }
        if opportunity_kinds.intersection(
            {
                CampaignActionKind.ACQUIRE_EVIDENCE.value,
                CampaignActionKind.BIND_EVIDENCE.value,
            }
        ):
            prepared_revisions["validation"] = int(graph.get("revision") or 0)
            prepare_threads["validation"] = current_thread().name
            validation_prepare_started.set()
            assert evidence_prepare_started.wait(timeout=2.0)
        return original_validation_prepare(*args, **kwargs)

    monkeypatch.setattr(
        target_solver_module,
        "_prepare_evidence_acquisition",
        prepare_evidence,
    )
    monkeypatch.setattr(
        target_solver_module,
        "prepare_materialized_edge_validation",
        prepare_validation,
    )
    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind evidence target",
        target_smiles=TARGET,
        run_id="blind-target-evidence-e2e",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=3,
            minimum_independent_source_groups=2,
            stock_boundary="benchmark_search",
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=20,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_evidence_connector,
    )

    assert result["model_cost"]["model_invocations"] == 1
    assert result["gates"]["gates"]["B2_host_validated_routes"] is True
    assert result["gates"]["gates"]["B3_exact_multi_source"] is True
    assert result["gates"]["gates"]["B4_stock_boundary"] is True
    assert result["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    evidence_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "evidence_acquisition"
    )
    assert evidence_stage["status"] == "completed"
    assert evidence_stage["detail"]["exact_record_count"] == 6
    lifecycle = result["candidate_lifecycle"]
    final_graph = gateway._open(
        result["run_id"],
        run_dir=Path(result["run_dir"]),
    ).graph_store.load()
    assert lifecycle["graph_revision"] == final_graph["revision"]
    assert lifecycle["graph_scientific_sha256"] == final_graph["scientific_sha256"]
    assert lifecycle["status_counts"]["accepted"] >= 1
    assert any(
        row["status"] == "accepted"
        and row["evidence"]["exact_record_count"] > 0
        and row["portfolio"]["accepted_route_ids"]
        for row in lifecycle["records"]
    )
    assert prepared_revisions["evidence"] > prepared_revisions["validation"]


def test_result_first_target_solver_defers_safe_evidence_prefetch_until_stock(
    tmp_path: Path,
) -> None:
    prefetch_started = Event()
    director_started = Event()
    prefetch_threads: list[str] = []

    def connector(request: Any) -> dict[str, Any]:
        if not request["edges"]:
            prefetch_threads.append(current_thread().name)
            prefetch_started.set()
            assert director_started.wait(timeout=2.0)
            return _discovery_only_connector(request)
        return _evidence_connector(request)

    setattr(connector, "autoplanner_prefetch_safe", True)

    def runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        director_started.set()
        return _runner(spec, context, mode, config)

    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="prefetched evidence target",
        target_smiles=TARGET,
        run_id="prefetched-evidence-overlap",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_target_chemenzy_baseline=True,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    evidence_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "evidence_acquisition"
    )
    prefetch = evidence_stage["detail"]["prefetch"]
    assert prefetch["status"] == "not_started"
    anytime_core = next(
        stage for stage in result["stages"] if stage["stage"] == "campaign_anytime_core"
    )["detail"]
    assert CampaignActionKind.ACQUIRE_EVIDENCE.value not in {
        execution["action"]["kind"] for execution in anytime_core["start_cohort"]["executions"]
    }
    assert not prefetch_threads


def test_target_solver_replans_globally_from_unbound_source_discovery(
    tmp_path: Path,
) -> None:
    observed_modes: list[str] = []

    def runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        observed_modes.append(mode)
        if mode == "event_replan":
            assert (
                context.evidence["source_discovery"]["sources"][0]["publication_number"]
                == "US7654321A1"
            )
            assert "source_material_discovered" in context.delta.material_events
        return _runner(spec, context, mode, config)

    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind discovery target",
        target_smiles=TARGET,
        run_id="blind-target-discovery-replan",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_target_chemenzy_baseline=False,
            enable_web_search=False,
            enable_replan=True,
        ),
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_discovery_only_connector,
    )

    assert observed_modes == ["initial_architecture", "event_replan"]
    assert result["model_cost"]["model_invocations"] == 2
    assert any(
        stage["stage"] == "global_replan_budget_gate"
        and stage["detail"]["trigger_reasons"] == ["evidence_deficit_with_new_source_material"]
        for stage in result["stages"]
    )
    signal_gate = next(
        stage for stage in result["stages"] if stage["stage"] == "global_replan_signal_gate"
    )
    retention = next(
        stage for stage in result["stages"] if stage["stage"] == "replan_retention_audit"
    )
    gain = next(stage for stage in result["stages"] if stage["stage"] == "global_replan_gain_audit")
    assert signal_gate["status"] == "accepted"
    assert signal_gate["detail"]["actionable_material_events"] == ["source_material_discovered"]
    pressure = signal_gate["detail"]["replan_pressure"]
    assert pressure["schema_version"] == "campaign_replan_pressure.v1"
    assert pressure["convergence_ledger_verified"] is True
    assert pressure["derived_material_events"] == []
    replan_action = next(
        dict(stage["detail"]["action"])
        for stage in result["stages"]
        if str(stage.get("stage") or "").startswith("campaign_action_unified_core_")
        and dict(stage["detail"]["action"]).get("kind") == CampaignActionKind.CODEX_REPLAN.value
    )
    assert (
        dict(replan_action["metadata"])["replan_pressure"]["content_sha256"]
        == pressure["content_sha256"]
    )
    assert retention["status"] == "accepted"
    assert retention["detail"]["missing_ids"] == {}
    assert gain["detail"]["model_cost_delta"]["model_invocations"] == 1.0
    assert gain["detail"]["semantics"]["observed_delta_is_not_a_cross_arm_causal_estimate"]
    discovery_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "evidence_acquisition"
    )
    assert discovery_stage["status"] == "discovered_unbound"
    assert discovery_stage["detail"]["exact_record_count"] == 0


@pytest.mark.parametrize(
    ("provider_delay_s", "director_delay_s"),
    ((0.0, 0.03), (0.03, 0.0)),
)
def test_zero_result_provider_search_triggers_one_failure_aware_replan(
    tmp_path: Path,
    provider_delay_s: float,
    director_delay_s: float,
) -> None:
    observed_modes: list[str] = []

    def runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        if mode == "initial_architecture" and director_delay_s:
            time.sleep(director_delay_s)
        observed_modes.append(mode)
        if mode == "event_replan":
            failures = context.evidence["provider_search_failures"]
            assert failures == [
                {
                    "action_kind": "chemenzy_target_expand",
                    "frontier_smiles": TARGET,
                    "target_level_native_search": True,
                    "status": "completed",
                    "provider_invocation_count": 1,
                    "failure_reasons": [],
                }
            ]
            assert "provider_search_exhausted_without_proposal" in context.delta.material_events
        return _runner(spec, context, mode, config)

    def empty_provider(**_kwargs: Any) -> dict[str, Any]:
        if provider_delay_s:
            time.sleep(provider_delay_s)
        return {"status": "completed", "routes": []}

    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="zero-result provider target",
        target_smiles=TARGET,
        run_id="zero-result-provider-replan",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_target_chemenzy_baseline=True,
            enable_web_search=False,
            enable_replan=True,
            enable_builtin_patent_evidence=False,
            enable_condition_enrichment=False,
            enable_live_benchmark_stock=False,
            delivery_boundary="stock_result",
        ),
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_partial_catalog,
        chemenzy_provider=empty_provider,
    )

    assert observed_modes == ["initial_architecture", "event_replan"]
    assert result["model_cost"]["model_invocations"] == 2
    signal_gate = next(
        stage for stage in result["stages"] if stage["stage"] == "global_replan_signal_gate"
    )
    assert signal_gate["status"] == "accepted"
    assert signal_gate["detail"]["actionable_material_events"] == [
        "provider_search_exhausted_without_proposal"
    ]
    budget_gate = next(
        stage for stage in result["stages"] if stage["stage"] == "global_replan_budget_gate"
    )
    assert (
        "provider_search_failure_requires_new_frontier" in budget_gate["detail"]["trigger_reasons"]
    )
    assert sum(stage["stage"] == "global_replan" for stage in result["stages"]) == 1
    settled_replan = next(stage for stage in result["stages"] if stage["stage"] == "global_replan")
    assert settled_replan["status"] != "running"


def test_completed_checkpoint_resume_does_not_repeat_replan_without_new_event(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(target_solver_module, "_MAX_DIRECTOR_OUTCOMES", 2)
    observed_modes: list[str] = []

    def runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        observed_modes.append(mode)
        return _runner(spec, context, mode, config)

    gateway = CampaignGateway(_paths(tmp_path))
    acceptance = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=4,
        minimum_edge_proof_level=2,
        stock_boundary="benchmark_search",
        minimum_independent_source_groups=2,
    )
    initial_budget = RetrosynthesisRunBudget(
        max_model_invocations=3,
        max_total_input_tokens=100_000,
        max_total_output_tokens=40_000,
        max_total_wall_time_s=720.0,
        max_visual_invocations=0,
        max_accepted_expansions=32,
        max_attempt_runs=72,
    )
    extended_budget = RetrosynthesisRunBudget(
        max_model_invocations=4,
        max_total_input_tokens=100_000,
        max_total_output_tokens=40_000,
        max_total_wall_time_s=720.0,
        max_visual_invocations=0,
        max_accepted_expansions=32,
        max_attempt_runs=72,
    )
    config = TargetSolveConfig(
        use_coordinator=False,
        enable_web_search=False,
        enable_chemenzy=False,
        enable_replan=True,
    )

    first = gateway.solve_target(
        target_name="blind repeated replan target",
        target_smiles=TARGET,
        run_id="blind-target-repeated-replan",
        acceptance=acceptance,
        budget=initial_budget,
        config=config,
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_discovery_only_connector,
    )
    monkeypatch.setattr(target_solver_module, "_MAX_DIRECTOR_OUTCOMES", 3)
    resumed = gateway.solve_target(
        target_name="blind repeated replan target",
        target_smiles=TARGET,
        run_id="blind-target-repeated-replan",
        acceptance=acceptance,
        budget=extended_budget,
        config=config,
        resume=True,
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_discovery_only_connector,
    )

    assert first["stop_decision"]["decision"] == "unresolved"
    assert observed_modes == ["initial_architecture", "event_replan"]
    assert resumed["model_cost"]["model_invocations"] == 2
    assert len(resumed["director_outcomes"]) == 2
    assert resumed["stop_decision"]["decision"] == "unresolved"
    extension = next(
        stage for stage in resumed["stages"] if stage["stage"] == "model_budget_extension"
    )
    assert extension["status"] == "accepted"
    assert extension["detail"]["effective_budget"]["max_model_invocations"] == 4
    assert (
        gateway._open("blind-target-repeated-replan").kernel.spec.limits.model.max_model_invocations
        == 4
    )


def test_target_solver_uses_one_budgeted_visual_candidate_in_global_replan(
    tmp_path: Path,
) -> None:
    image = tmp_path / "visual-source-page.png"
    image.write_bytes(b"target-solver-visual-source-page")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    observed_modes: list[str] = []
    visual_calls = 0

    def connector(request: Any) -> dict[str, Any]:
        result = _discovery_only_connector(request)
        source = result["discovery"]["sources"][0]
        source.update(
            {
                "pdf_sha256": "b" * 64,
                "unresolved_edge_count": len(request["edges"]),
                "visual_candidate_pages": [
                    {
                        "page_number": 4,
                        "image_path": str(image),
                        "image_sha256": image_sha256,
                    }
                ],
            }
        )
        return result

    def visual_provider(request: Any) -> dict[str, Any]:
        nonlocal visual_calls
        visual_calls += 1
        edge = request["edges"][0]
        return {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.visual"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "wall_time_s": 0.1,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": edge["product_smiles"],
                        "reactant_smiles": edge["precursor_smiles"],
                        "source_locator": "page 4",
                    }
                ]
            },
        }

    def runner(spec: AgentSpec, context: Any, mode: str, config: Any) -> AgentResult:
        observed_modes.append(mode)
        if mode == "event_replan":
            visual = context.evidence["visual_source_candidates"]
            assert visual["candidate_step_count"] == 1
            assert visual["candidate_steps"][0]["grants_exact_evidence"] is False
            assert "visual_source_candidates_added" in context.delta.material_events
        return _runner(spec, context, mode, config)

    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind visual discovery target",
        target_smiles=TARGET,
        run_id="blind-target-visual-replan",
        budget=RetrosynthesisRunBudget(
            max_model_invocations=3,
            max_total_input_tokens=50_000,
            max_total_output_tokens=14_000,
            max_total_wall_time_s=720,
            max_visual_invocations=1,
            max_accepted_expansions=8,
            max_attempt_runs=20,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=True,
        ),
        director_runner=runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
        visual_evidence_provider=visual_provider,
    )

    assert observed_modes == ["initial_architecture", "event_replan"]
    assert visual_calls == 1
    assert result["model_cost"]["model_invocations"] == 3
    assert result["model_cost"]["visual_invocations"] == 1
    first_evidence = next(
        stage for stage in result["stages"] if stage["stage"] == "evidence_acquisition"
    )
    assert first_evidence["status"] == "structure_bound_unproven"
    assert first_evidence["detail"]["visual_evidence"]["status"] == "completed"
    assert first_evidence["detail"]["exact_record_count"] == 0
    assert first_evidence["detail"]["exact_structure_binding_count"] == 1


def test_target_solver_runs_program_discovery_without_implicit_store_admission(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="program discovery target",
        target_smiles=TARGET,
        run_id="program-discovery-action",
        config=TargetSolveConfig(
            enable_chemenzy=False,
            enable_web_search=False,
            enable_replan=False,
            enable_condition_enrichment=False,
            enable_builtin_patent_evidence=False,
            enable_program_discovery=True,
            enable_program_review=True,
            enable_program_admission=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        program_capabilities=[
            {
                "capability_id": "fixture:generic-program-capability",
                "enzyme": {"classes": ["ketoreductase"]},
                "match": {
                    "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
                    "min_window_steps": 1,
                    "max_window_steps": 4,
                },
                "precedent_refs": ["doi:10.1000/program-fixture"],
            }
        ],
    )

    discovery = next(row for row in result["stages"] if row["stage"] == "program_discovery")[
        "detail"
    ]
    review = next(row for row in result["stages"] if row["stage"] == "program_review")["detail"]
    program_actions = [
        dict(row["detail"])
        for row in result["stages"]
        if str(row.get("stage") or "").startswith("campaign_action_unified_core_")
        and dict(row["detail"].get("action") or {}).get("kind")
        in {
            CampaignActionKind.PROGRAM_DISCOVER.value,
            CampaignActionKind.PROGRAM_REVIEW.value,
        }
    ]
    discovery_action = next(
        row
        for row in program_actions
        if dict(row["action"]).get("kind") == CampaignActionKind.PROGRAM_DISCOVER.value
    )
    review_action = next(
        row
        for row in program_actions
        if dict(row["action"]).get("kind") == CampaignActionKind.PROGRAM_REVIEW.value
    )
    discovery_pressure = dict(discovery_action["action"]["metadata"])[
        "program_opportunity_pressure"
    ]
    review_pressure = dict(review_action["action"]["metadata"])["program_review_pressure"]

    assert discovery["action_execution_count"] >= 1
    assert discovery["semantics"]["target_names_are_not_matching_inputs"] is True
    assert discovery["semantics"]["program_candidates_are_proposal_only"] is True
    assert discovery_pressure["schema_version"] == ("campaign_program_opportunity_pressure.v1")
    assert (
        discovery_pressure["semantics"]["conventional_route_remains_the_explicit_fallback"] is True
    )
    assert review_pressure["schema_version"] == ("campaign_program_review_pressure.v1")
    assert review["store"]["status"]["event_count"] == 0
    assert not any(row["stage"] == "program_admission" for row in result["stages"])


def test_completed_target_resume_ingests_new_feedback_and_rejects_invalid_feedback(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    feedback_target = "CCO"

    def program_runner(spec: AgentSpec, context: Any, mode: str, _config: Any) -> AgentResult:
        precursors = (
            ("family:carbonyl", "CC=O", "carbonyl reduction"),
            ("family:ether", "COC", "ether rearrangement"),
            ("family:epoxide", "C1CO1", "epoxide reduction"),
        )
        plan = {
            "schema_version": "global_campaign_plan.v1",
            "plan_id": f"program-feedback-plan:{mode}",
            "run_id": context.run_id,
            "mode": mode,
            "context_sha256": context.content_sha256,
            "graph_revision": context.revision.graph_revision,
            "route_families": [
                {
                    "route_family_id": family_id,
                    "title": strategy,
                    "strategy": strategy,
                    "target_smiles": feedback_target,
                    "advantages": ["single precursor"],
                    "risks": ["requires validation"],
                    "diversity_basis": precursor,
                }
                for family_id, precursor, strategy in precursors
            ],
            "multi_step_skeletons": [
                {
                    "skeleton_id": f"program-feedback-skeleton:{index}",
                    "route_family_id": family_id,
                    "summary": strategy,
                    "steps": [
                        {
                            "step_id": f"program-feedback-step:{index}",
                            "product_smiles": feedback_target,
                            "precursor_smiles": [precursor],
                            "transformation_hypothesis": strategy,
                            "strategic_role": "target convergence",
                            "source_hints": [],
                            "required_validation": ["atom mapping", "stock audit"],
                            "hypothesis_only": True,
                            "condition_predictions": [
                                {
                                    "reagents": ["screen catalyst"],
                                    "solvent": "water",
                                    "temperature_c": 25,
                                    "time": "screen",
                                    "authority_scope": "model_predicted_condition",
                                    "not_reaction_proof": True,
                                }
                            ],
                        }
                    ],
                }
                for index, (family_id, precursor, strategy) in enumerate(precursors, start=1)
            ],
            "strategic_disconnections": [],
            "shared_intermediates": [],
            "critical_unknowns": [],
            "source_plan": [],
            "fallback_strategies": [],
            "frontier_priorities": [],
            "pivot_conditions": [],
            "stop_conditions": [],
            "portfolio_rationale": "Three target-blind single-precursor families.",
            "limitations": ["requires host validation"],
        }
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.SUCCEEDED,
            output=plan,
            usage={
                "model_invocations": 1,
                "input_tokens": 500,
                "output_tokens": 500,
                "wall_time_s": 0.1,
            },
        )

    def program_mapper(reactions: list[str]) -> list[str]:
        mapped = {
            "CC=O": "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]",
            "COC": "[CH3:1][O:3][CH3:2]>>[CH3:1][CH2:2][OH:3]",
            "C1CO1": "[CH2:1]1[CH2:2][O:3]1>>[CH3:1][CH2:2][OH:3]",
        }
        return [
            next(
                value
                for precursor, value in mapped.items()
                if reaction.startswith(precursor + ">>")
            )
            for reaction in reactions
        ]

    capability = {
        "capability_id": "fixture:generic-program-capability",
        "enzyme": {"classes": ["ketoreductase"]},
        "match": {
            "net_motif_delta": {"carbonyl": -1, "hydroxyl": 1},
            "element_delta": {"C": 0, "O": 0},
            "min_scaffold_similarity": 0.0,
            "max_abs_heavy_atom_delta": 0,
            "min_window_steps": 1,
            "max_window_steps": 1,
            "reject_unlisted_motif_changes": False,
        },
        "selectivity_objective": "Reduce the exact carbonyl boundary to ethanol.",
        "substrate_scope_basis": "test exact-boundary screen",
        "precedent_refs": ["doi:10.1000/program-fixture"],
    }
    acceptance = RetrosynthesisAcceptanceSpec(
        minimum_complete_routes=1,
        minimum_edge_proof_level=2,
        stock_boundary="benchmark_search",
        minimum_independent_source_groups=1,
    )
    config = TargetSolveConfig(
        enable_chemenzy=False,
        enable_web_search=False,
        enable_replan=False,
        enable_condition_enrichment=False,
        enable_builtin_patent_evidence=False,
        enable_program_discovery=True,
        enable_program_review=True,
        enable_program_admission=False,
        enable_program_validation=True,
        enable_experimental_claim_admission=False,
    )
    run_id = "program-feedback-terminal-resume"
    first = gateway.solve_target(
        target_name="program feedback target",
        target_smiles=feedback_target,
        run_id=run_id,
        acceptance=acceptance,
        config=config,
        director_runner=program_runner,
        atom_mapper=program_mapper,
        stock_catalog_builder=_catalog,
        program_capabilities=[capability],
    )
    discovery_results = next(row for row in first["stages"] if row["stage"] == "program_discovery")[
        "detail"
    ]["results"]
    discovery = next(
        value
        for value in discovery_results
        if dict(dict(value.get("program_review") or {}).get("program_bundle") or {}).get(
            "program_proposals"
        )
    )
    discovery_action_pressures = [
        dict(dict(row["detail"]["action"])["metadata"])["program_opportunity_pressure"]
        for row in first["stages"]
        if str(row.get("stage") or "").startswith("campaign_action_unified_core_")
        and dict(row["detail"].get("action") or {}).get("kind")
        == CampaignActionKind.PROGRAM_DISCOVER.value
    ]
    assert any(
        int(pressure["matched_capability_count"]) > 0 and float(pressure["pressure_total"]) > 0.0
        for pressure in discovery_action_pressures
    )
    assert any(
        int(pressure["matched_capability_count"]) == 0 for pressure in discovery_action_pressures
    )
    assert len({str(pressure["content_sha256"]) for pressure in discovery_action_pressures}) >= 2
    service = gateway._open(run_id, run_dir=first["run_dir"])
    current_program_review = service.review_route_program_innovations(
        discovery["route_id"],
        capabilities=[capability],
    )
    proposal = next(iter(current_program_review["program_bundle"]["program_proposals"].values()))
    program_signal = next(
        row
        for row in service.graph_store.load()["action_signals"].values()
        if row.get("kind") == "program_validation"
    )
    work_item = program_signal["metadata"]["work_item"]
    assert program_signal["priority"] == work_item["scheduling"]["action_priority"]
    assert program_signal["score"] == work_item["scheduling"]["action_score"]
    validation = with_biocatalysis_program_validation_digest(
        {
            "schema_version": "biocatalysis_program_validation.v1",
            "validation_id": "validation:terminal-resume:success",
            "program_id": proposal["program_id"],
            "innovation_id": proposal["source_innovation_id"],
            "accepted": True,
            "evidence_tier": "exact_substrate_screen",
            "input_state_ids": proposal["input_state_ids"],
            "output_state_ids": proposal["output_state_ids"],
            "claim_refs": ["claim:terminal-resume:exact-boundary"],
            "condition_record_ids": [],
            "selectivity_assessed": True,
            "cofactor_ledger_closed": True,
            "outcome": {"conversion_fraction": 0.82},
        }
    )
    scientific_before = service.graph_store.load()["scientific_sha256"]

    resumed = gateway.solve_target(
        target_name="program feedback target",
        target_smiles=feedback_target,
        run_id=run_id,
        acceptance=acceptance,
        config=config,
        resume=True,
        director_runner=program_runner,
        atom_mapper=program_mapper,
        stock_catalog_builder=_catalog,
        program_capabilities=[capability],
        program_validation_feedback=(
            {"route_id": discovery["route_id"], "validation": validation},
        ),
    )
    valid_feedback = [
        row["detail"]
        for row in resumed["stages"]
        if row["stage"].startswith("campaign_action_unified_core_")
        and dict(row["detail"].get("action") or {}).get("kind") == "experiment_feedback_ingest"
    ][-1]

    assert first["stop_decision"]["decision"] == "completed"
    assert any(row["stage"] == "terminal_checkpoint_reopened" for row in resumed["stages"])
    assert valid_feedback["status"] == "completed"
    assert (
        valid_feedback["outcome"]["handler_result"]["validation_id"] == validation["validation_id"]
    )
    assert (
        gateway._open(run_id, run_dir=first["run_dir"]).graph_store.load()["scientific_sha256"]
        == scientific_before
    )
    assert (
        gateway.experimental_claim_store(run_id, run_dir=first["run_dir"])["replay"]["event_count"]
        == 0
    )

    invalid_material = {key: value for key, value in validation.items() if key != "content_sha256"}
    invalid_material["validation_id"] = "validation:terminal-resume:invalid"
    invalid_material["input_state_ids"] = ["chemical-state:tampered"]
    invalid = with_biocatalysis_program_validation_digest(invalid_material)
    rejected = gateway.solve_target(
        target_name="program feedback target",
        target_smiles=feedback_target,
        run_id=run_id,
        acceptance=acceptance,
        config=config,
        resume=True,
        director_runner=program_runner,
        atom_mapper=program_mapper,
        stock_catalog_builder=_catalog,
        program_capabilities=[capability],
        program_validation_feedback=({"route_id": discovery["route_id"], "validation": invalid},),
    )
    invalid_feedback = [
        row["detail"]
        for row in rejected["stages"]
        if row["stage"].startswith("campaign_action_unified_core_")
        and dict(row["detail"].get("action") or {}).get("kind") == "experiment_feedback_ingest"
        and row["detail"]["outcome"]["handler_result"].get("validation_id")
        == invalid["validation_id"]
    ][-1]

    assert invalid_feedback["status"] == "failed"
    assert invalid_feedback["outcome"]["handler_result"]["reasons"] == [
        "experiment_feedback_domain_gate_rejected"
    ]
    assert (
        gateway._open(run_id, run_dir=first["run_dir"]).graph_store.load()["scientific_sha256"]
        == scientific_before
    )
    assert (
        gateway.experimental_claim_store(run_id, run_dir=first["run_dir"])["replay"]["event_count"]
        == 0
    )


def test_target_solver_can_close_procurement_from_frozen_supplier_snapshot(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    result = gateway.solve_target(
        target_name="blind procurement target",
        target_smiles=TARGET,
        run_id="blind-target-procurement-e2e",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=2,
            minimum_independent_source_groups=2,
            stock_boundary="procurement",
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=16,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
            max_live_stock_molecules=2,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        inventory_snapshot_builder=_inventory_builder,
    )

    assert result["gates"]["gates"]["B4_stock_boundary"] is True
    assert result["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    assert result["claim"]["procurement_ready"] is True
    stock_stage = next(stage for stage in result["stages"] if stage["stage"] == "stock")
    assert stock_stage["detail"]["audit_batch_limit"] == 2
    assert stock_stage["detail"]["stock_closed_leaf_count"] == 4


def test_validation_fork_replays_global_plan_and_uses_zero_model_calls(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind validation lineage target",
        target_smiles=TARGET,
        run_id="blind-validation-source",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-validation-derived",
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_evidence_connector,
    )

    assert derived["schema_version"] == "target_validation_fork_report.v1"
    assert derived["model_cost"]["model_invocations"] == 0
    assert derived["lineage"]["source_run_id"] == source["run_id"]
    assert derived["lineage"]["source_report_sha256"] == source["content_sha256"]
    assert derived["semantics"]["B0_refers_to_bound_source_campaign"] is True
    assert derived["gates"]["gates"] == {
        "B0_blind_input": True,
        "B1_global_multi_route": True,
        "B2_host_validated_routes": True,
        "B3_exact_multi_source": True,
        "B4_stock_boundary": True,
        "B5_configured_portfolio_acceptance": True,
    }
    assert derived["current_disposition"]["state"] == "accepted"
    assert derived["self_evolution"]["model_invocations"] == 0
    assert any(
        stage.get("learned_template_ids") for stage in derived["self_evolution"]["learning_stages"]
    )
    assert Path(derived["report_path"]).is_file()


def test_validation_fork_runs_guided_chemenzy_after_stock_open_leaf(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind validation guided tail target",
        target_smiles=TARGET,
        run_id="blind-validation-guided-source",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_partial_catalog,
    )
    observed: list[dict[str, Any]] = []

    def provider(**kwargs: Any) -> dict[str, Any]:
        observed.append(dict(kwargs))
        return {
            "status": "completed",
            "routes": [
                {
                    "steps": [
                        {
                            "product_smiles": kwargs["target_smiles"],
                            "reactant_smiles": ["CC=O"],
                            "source_model": "fixture-guided-tail",
                        }
                    ]
                }
            ],
        }

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-validation-guided-derived",
        atom_mapper=_mapper,
        stock_catalog_builder=_partial_catalog,
        chemenzy_provider=provider,
        config=ValidationForkConfig(
            enable_guided_chemenzy=True,
            max_guided_chemenzy_frontiers=1,
            max_guided_chemenzy_iterations=500,
            max_guided_chemenzy_steps=6,
            guided_chemenzy_timeout_s=1_200,
        ),
    )

    guided = next(
        stage for stage in derived["stages"] if stage["stage"] == "chemenzy_guided_frontier"
    )
    assert len(observed) == 1
    assert observed[0]["target_smiles"] == "CCO"
    assert observed[0]["limits"]["max_iterations"] == 500
    assert observed[0]["limits"]["max_steps"] == 6
    assert observed[0]["limits"]["timeout_s"] == 1_200
    assert guided["detail"]["provider_invocation_count"] == 1
    assert guided["detail"]["proposal_count"] == 1
    assert derived["resource_envelope"]["native_search"]["frontier"]["settled"] == 1
    assert derived["model_cost"]["model_invocations"] == 0
    assert any(stage["stage"] == "guided_materialization" for stage in derived["stages"])


def test_validation_fork_can_admit_one_sparse_visual_candidate(
    tmp_path: Path,
) -> None:
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind visual validation target",
        target_smiles=TARGET,
        run_id="blind-visual-validation-source",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
    )
    image = tmp_path / "downloaded-source-page.png"
    image.write_bytes(b"fresh-validation-fork-source-page")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    visual_calls = 0

    def connector(request: Any) -> dict[str, Any]:
        result = _discovery_only_connector(request)
        result["discovery"]["sources"][0].update(
            {
                "pdf_sha256": "c" * 64,
                "unresolved_edge_count": len(request["edges"]),
                "visual_candidate_pages": [
                    {
                        "page_number": 7,
                        "image_path": str(image),
                        "image_sha256": image_sha256,
                    }
                ],
            }
        )
        return result

    def visual_provider(request: Any) -> dict[str, Any]:
        nonlocal visual_calls
        visual_calls += 1
        edge = request["edges"][0]
        return {
            "request_sha256": request["content_sha256"],
            "provider_status": "completed",
            "provider_receipt": {"provider_id": "tests.visual.validation"},
            "usage": {
                "model_invocations": 1,
                "visual_invocations": 1,
                "input_tokens": 120,
                "output_tokens": 60,
                "wall_time_s": 0.1,
            },
            "candidate_chain": {
                "steps": [
                    {
                        "product_smiles": edge["product_smiles"],
                        "reactant_smiles": edge["precursor_smiles"],
                        "source_locator": "page 7",
                    }
                ]
            },
        }

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-visual-validation-derived",
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
        visual_evidence_provider=visual_provider,
        config=ValidationForkConfig(
            max_visual_invocations=1,
            max_visual_evidence_pages=2,
        ),
    )

    evidence = next(
        stage for stage in derived["stages"] if stage["stage"] == "evidence_acquisition"
    )
    assert visual_calls == 1
    assert derived["model_cost"]["model_invocations"] == 1
    assert derived["model_cost"]["visual_invocations"] == 1
    assert evidence["detail"]["visual_evidence"]["status"] == "completed"
    assert (
        evidence["detail"]["visual_evidence"]["observation"]["candidate_steps"][0][
            "grants_exact_evidence"
        ]
        is False
    )
    assert derived["semantics"]["derived_visual_invocation_limit"] == 1


def test_scanned_patent_ocr_closes_blind_route_and_zero_model_validation_fork(
    tmp_path: Path,
) -> None:
    structures = {
        "ethyl acetate": TARGET,
        "ethanol": "CCO",
        "acetyl chloride": "CC(=O)Cl",
        "acetic acid": "CC(=O)O",
        "acetyl bromide": "CC(=O)Br",
    }
    names = {value: [name] for name, value in structures.items()}
    ocr_text = (
        "Ethyl acetate (T1). Ethanol and acetyl chloride were added. The "
        "reaction mixture was stirred to afford T1.\n\n"
        "Ethyl acetate (T2). Ethanol and acetic acid were added. The "
        "reaction mixture was stirred to afford T2.\n\n"
        "Ethyl acetate (T3). Ethanol and acetyl bromide were added. The "
        "reaction mixture was stirred to afford T3."
    )
    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(
            cache_dir=tmp_path / "scanned-patents",
            max_patents=2,
            max_ocr_pages=1,
        ),
        candidate_provider=lambda _queries: [
            {
                "publication_number": publication,
                "family_id": family,
                "title": "Scanned preparation of ethyl acetate",
                "snippet": "primary process source",
                "pdf_url": f"https://source.invalid/{publication}.pdf",
            }
            for publication, family in (
                ("US1234567A1", "family:one"),
                ("WO7654321A1", "family:two"),
            )
        ],
        bytes_fetcher=lambda url, _timeout, _limit: _scanned_patent_pdf(url),
        structure_resolver=lambda value: structures[str(value).casefold()],
        candidate_name_resolver=lambda value: names.get(str(value), []),
        ocr_runner=lambda *_args: {
            "text": ocr_text,
            "engine_id": "tesseract",
            "engine_version": "fixture",
        },
    )
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind scanned patent target",
        target_smiles=TARGET,
        run_id="blind-scanned-patent-source",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=3,
            stock_boundary="benchmark_search",
            minimum_independent_source_groups=2,
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=24,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert source["model_cost"]["model_invocations"] == 1
    assert source["model_cost"]["visual_invocations"] == 0
    assert source["gates"]["gates"]["B3_exact_multi_source"] is True
    assert source["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    evidence = next(stage for stage in source["stages"] if stage["stage"] == "evidence_acquisition")
    assert evidence["detail"]["exact_record_count"] == 6
    assert all(
        row["ocr_audit"]["status"] == "completed"
        for row in evidence["detail"]["discovery"]["sources"]
    )

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-scanned-patent-validation",
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert derived["model_cost"]["model_invocations"] == 0
    assert derived["model_cost"]["visual_invocations"] == 0
    assert derived["gates"]["gates"]["B3_exact_multi_source"] is True
    assert derived["current_disposition"]["state"] == "accepted"


def test_primary_patent_html_closes_blind_portfolio_without_pdf_or_visual_model(
    tmp_path: Path,
) -> None:
    structures = {
        "ethyl acetate": TARGET,
        "ethanol": "CCO",
        "acetyl chloride": "CC(=O)Cl",
        "acetic acid": "CC(=O)O",
        "acetyl bromide": "CC(=O)Br",
    }
    names = {value: [name] for name, value in structures.items()}
    pdf_fetches = 0

    def reject_pdf(_url: str, _timeout: float, _limit: int) -> bytes:
        nonlocal pdf_fetches
        pdf_fetches += 1
        raise AssertionError("fully closed primary HTML must skip PDF")

    def fetch_html(url: str, _timeout: float, _limit: int) -> bytes:
        publication = url.rstrip("/").split("/")[-2]
        return _patent_html(publication)

    connector = build_builtin_patent_evidence_connector(
        BuiltinPatentEvidenceConfig(
            cache_dir=tmp_path / "html-patents",
            max_patents=2,
        ),
        candidate_provider=lambda _queries: [
            {
                "publication_number": publication,
                "family_id": family,
                "title": "HTML preparation of ethyl acetate",
                "snippet": "search metadata only",
                "html_url": (f"https://patents.google.com/patent/{publication}/en"),
                "pdf_url": f"https://source.invalid/{publication}.pdf",
            }
            for publication, family in (
                ("US1234567A1", "family:html-one"),
                ("WO7654321A1", "family:html-two"),
            )
        ],
        bytes_fetcher=reject_pdf,
        html_fetcher=fetch_html,
        structure_resolver=lambda value: structures[str(value).casefold()],
        candidate_name_resolver=lambda value: names.get(str(value), []),
    )
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="blind primary HTML patent target",
        target_smiles=TARGET,
        run_id="blind-primary-html-source",
        acceptance=RetrosynthesisAcceptanceSpec(
            minimum_complete_routes=2,
            minimum_edge_proof_level=3,
            stock_boundary="benchmark_search",
            minimum_independent_source_groups=2,
        ),
        budget=RetrosynthesisRunBudget(
            max_model_invocations=1,
            max_total_input_tokens=10_000,
            max_total_output_tokens=5_000,
            max_total_wall_time_s=60,
            max_visual_invocations=0,
            max_accepted_expansions=8,
            max_attempt_runs=24,
        ),
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert pdf_fetches == 0
    assert source["model_cost"]["model_invocations"] == 1
    assert source["model_cost"]["visual_invocations"] == 0
    assert source["gates"]["gates"]["B3_exact_multi_source"] is True
    assert source["gates"]["gates"]["B5_configured_portfolio_acceptance"] is True
    evidence = next(stage for stage in source["stages"] if stage["stage"] == "evidence_acquisition")
    assert evidence["detail"]["exact_record_count"] == 6
    assert all(
        row["html_sha256"] and not row["pdf_sha256"]
        for row in evidence["detail"]["discovery"]["sources"]
    )
    assert list(tmp_path.rglob("*.pdf")) == []
    assert list(tmp_path.rglob("*.png")) == []

    derived = gateway.fork_target_validation(
        source_run_id=source["run_id"],
        run_id="blind-primary-html-validation",
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=connector,
    )

    assert pdf_fetches == 0
    assert derived["model_cost"]["model_invocations"] == 0
    assert derived["model_cost"]["visual_invocations"] == 0
    assert derived["current_disposition"]["state"] == "accepted"


def test_patent_self_evolution_learns_then_guides_a_new_blind_target(
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "external-memory" / "patent-templates.json"
    gateway = CampaignGateway(_paths(tmp_path))
    source = gateway.solve_target(
        target_name="unseen self evolution source",
        target_smiles=TARGET,
        run_id="blind-self-evo-source",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
            self_evo_library_path=str(library_path),
        ),
        director_runner=_runner,
        atom_mapper=_mapper,
        stock_catalog_builder=_catalog,
        evidence_connector=_evidence_connector,
    )

    learned = {
        template_id
        for stage in source["self_evolution"]["learning_stages"]
        for template_id in stage.get("learned_template_ids") or []
    }
    assert len(learned) == 3
    assert source["self_evolution"]["model_invocations"] == 0

    analogue = "CCCOC(C)=O"
    director_calls = 0

    def analogue_runner(
        spec: AgentSpec,
        context: Any,
        mode: str,
        config: Any,
    ) -> AgentResult:
        nonlocal director_calls
        director_calls += 1
        memory = context.evidence["self_evo_patent_template_memory"]
        assert memory["generation"] >= 1
        assert any(
            candidate["precursor_smiles"] == ["CC(=O)Cl", "CCCO"]
            for candidate in memory["candidates"]
        )
        plan = _plan(context, mode)
        alternative_donors = ("CC(=O)Cl", "CC(=O)N", "CC(=O)S")
        for family, donor in zip(
            plan["route_families"],
            alternative_donors,
            strict=True,
        ):
            family["target_smiles"] = analogue
            family["diversity_basis"] = donor
        for skeleton, donor in zip(
            plan["multi_step_skeletons"],
            alternative_donors,
            strict=True,
        ):
            skeleton["steps"][0]["product_smiles"] = analogue
            skeleton["steps"][0]["precursor_smiles"] = ["CCCO", donor]
        return AgentResult(
            run_id=spec.run_id,
            agent_id=spec.agent_id,
            parent_agent_id=spec.parent_agent_id,
            attempt=spec.attempt,
            idempotency_key=f"{spec.idempotency_key}:result",
            context_hash=spec.context_hash,
            capabilities=spec.capabilities,
            write_scope=spec.write_scope,
            budget=spec.budget,
            state=AgentState.SUCCEEDED,
            output=plan,
            usage={
                "model_invocations": 1,
                "input_tokens": 1000,
                "output_tokens": 700,
                "wall_time_s": 1.0,
            },
        )

    def analogue_mapper(reactions: list[str]) -> list[str]:
        mapped = []
        for reaction in reactions:
            if "CC(=O)Cl" in reaction:
                leaving = "Cl"
            elif "CC(=O)Br" in reaction:
                leaving = "Br"
            elif "CC(=O)O.CCCO" in reaction:
                leaving = "OH"
            else:
                mapped.append("")
                continue
            mapped.append(
                f"[CH3:1][C:2](=[O:3])[{leaving}:4]."
                "[CH3:5][CH2:6][CH2:8][OH:7]>>"
                "[CH3:1][C:2](=[O:3])[O:7][CH2:8][CH2:6][CH3:5]"
            )
        return mapped

    derived = gateway.solve_target(
        target_name="new blind propyl ester",
        target_smiles=analogue,
        run_id="blind-self-evo-derived",
        config=TargetSolveConfig(
            use_coordinator=False,
            enable_web_search=False,
            enable_replan=False,
            self_evo_library_path=str(library_path),
        ),
        director_runner=analogue_runner,
        atom_mapper=analogue_mapper,
        stock_catalog_builder=_catalog,
    )
    graph = gateway._open(
        derived["run_id"],
        run_dir=derived["run_dir"],
    ).graph_store.load()

    assert director_calls == 1
    assert derived["model_cost"]["model_invocations"] == 1
    # The three Codex-selected families consume three unique proposals.  The
    # self-evo memory annotates the matching selected edge but must not spend
    # two more expansions on unselected target-level template applications.
    assert derived["accepted_expansion_count"] == 3
    assert any(
        origin["origin_kind"] == "self_evo_patent_template"
        for edge in graph["edges"].values()
        for origin in edge.get("origin_records") or []
    )
    assert any(
        proof.get("accepted") is True
        for edge in graph["edges"].values()
        if any(
            origin.get("origin_kind") == "self_evo_patent_template"
            for origin in edge.get("origin_records") or []
        )
        for proof in edge.get("reaction_proofs") or []
    )


def test_stage_dedup_preserves_settled_provider_result_over_resume_reuse() -> None:
    from cascade_planner.interfaces.target_solver import _deduplicate_stages

    stages = _deduplicate_stages(
        [
            {
                "stage": "aizynthfinder_guided_frontier",
                "status": "completed",
                "detail": {
                    "provider_invocation_count": 1,
                    "proposal_count": 6,
                    "statistics": {"profiling": {"iterations": 500}},
                },
            },
            {
                "stage": "aizynthfinder_guided_frontier",
                "status": "reused",
                "detail": {
                    "status": "reused",
                    "new_proposal_count": 0,
                    "reused_from_status": "frontier_budget_already_spent",
                },
            },
        ]
    )

    assert len(stages) == 1
    assert stages[0]["status"] == "completed"
    assert stages[0]["detail"]["proposal_count"] == 6
    assert stages[0]["detail"]["statistics"]["profiling"]["iterations"] == 500


def test_terminal_report_does_not_advertise_stale_scheduler_action() -> None:
    from cascade_planner.interfaces.target_solver import _report_next_action

    projected = _report_next_action(
        {
            "selected_action_id": "action:stale",
            "selected_action": {"kind": "native_search_frontier"},
            "candidate_count": 4,
            "eligible_candidate_count": 1,
            "content_sha256": "a" * 64,
        },
        stop_decision={"terminal": True, "decision": "unresolved"},
    )

    assert projected["selected_action_id"] == ""
    assert projected["selected_action"] == {}
    assert projected["candidate_count"] == 0
    assert projected["eligible_candidate_count"] == 0
    assert projected["terminal"] is True
    assert projected["historical_decision_sha256"] == "a" * 64


def test_expansion_accounting_separates_director_and_provider_origins() -> None:
    from cascade_planner.interfaces.target_solver import _compile_expansion_accounting

    lifecycle = {
        "canonical_candidate_count": 2,
        "records": [
            {
                "status": "admitted_unproved",
                "origin_records": [{"origin_kind": "codex_global_director"}],
                "admission": {"accepted": True},
                "materialization": {"materialized": True},
                "validation": {"accepted": False},
                "portfolio": {"accepted_route_ids": []},
            },
            {
                "status": "admitted_unproved",
                "origin_records": [{"origin_kind": "aizynthfinder"}],
                "admission": {"accepted": True},
                "materialization": {"materialized": True},
                "validation": {"accepted": False},
                "portfolio": {"accepted_route_ids": []},
            },
        ],
    }
    accounting = _compile_expansion_accounting(
        lifecycle,
        outcomes=(
            {
                "status": "accepted",
                "plan": {"multi_step_skeletons": [{"steps": [{"step_id": "codex:1"}]}]},
            },
        ),
        kernel_accepted_expansion_count=2,
    )

    assert accounting["director_selected_step_count"] == 1
    assert accounting["by_origin_kind"]["codex_global_director"]["materialized"] == 1
    assert accounting["by_origin_kind"]["aizynthfinder"]["materialized"] == 1


def test_search_method_projection_keeps_aiz_mcts_strategy_and_unified_builder() -> None:
    from cascade_planner.interfaces.target_solver import _search_method_projection

    projection = _search_method_projection(
        _resolve_execution_config(TargetSolveConfig(execution_profile="paper_synthex"))
    )

    assert projection["leaf_continuation_engine"] == "same_llm_route_builder"
    assert projection["strategy_search_engine"] == (
        "AiZynthFinder.MctsSearchTree with host-replayed Codex ReactionJSON policy"
    )
    assert projection["strategy_tree_engine"] == "aizynthfinder_mcts"
    assert projection["strategy_ucb_active"] is True
    assert projection["llm_expansion_policy_inside_aizynthfinder_mcts"] is True
    assert projection["paper_strategy_search_aligned"] is True
    assert projection["paper_algorithm_equivalent"] is False
    assert projection["paper_source_implementation_identical"] is False
