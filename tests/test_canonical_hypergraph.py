from __future__ import annotations

from pathlib import Path

from cascade_planner.application.canonical_hypergraph import (
    CanonicalHypergraphStore,
    CanonicalIngestionBatch,
    canonical_scientific_projection,
    compile_canonical_hypergraph_revision,
    full_recompute_canonical_hypergraph,
    hypothesis_identity,
    molecule_identity,
    reaction_edge_identity,
    route_family_identity,
    source_binding_identity,
    stock_observation_identity,
)
from cascade_planner.application.deficit_frontier import (
    compile_deficit_frontier,
    frontier_scientific_projection,
    target_reachable_route_boundaries,
)
from cascade_planner.application.route_edge_scope import (
    route_family_scoped_edge_ids,
)
from cascade_planner.application.fact_lifecycle import build_fact_lifecycle_event
from cascade_planner.application.proof_policy import ProofPolicy, stitch_edge_proof
from cascade_planner.application.retrosynthesis_run_contract import (
    RetrosynthesisRunBudget,
)
from cascade_planner.application.reaction_proof_versions import (
    CURRENT_REACTION_VALIDATOR_VERSION,
)
from cascade_planner.application.reactionjson_replay import replay_reactionjson
from cascade_planner.application.routejson_compiler import RouteJSONCompiler
from cascade_planner.application.retrosynthesis_workers import (
    build_retrosynthesis_worker_handlers,
    materialization_commands_for_global_plan,
    normalize_source_binding,
)
from cascade_planner.application.run_kernel import RunKernel, RunLimits, RunSpec
from cascade_planner.application.strategy_contract import normalize_strategy_card
from cascade_planner.application.worker_runtime import (
    WorkerBudget,
    WorkerCommand,
    WorkerRuntime,
)


def _kernel(tmp_path: Path) -> RunKernel:
    kernel = RunKernel(
        tmp_path / "runtime",
        tmp_path / "run",
        spec=RunSpec(
            run_id="canonical-graph",
            target_name="ethyl acetate",
            target_smiles="CCOC(C)=O",
            created_at="2026-07-13T00:00:00Z",
            limits=RunLimits(
                model=RetrosynthesisRunBudget(
                    max_model_invocations=0,
                    max_accepted_expansions=32,
                    max_attempt_runs=64,
                ),
                max_total_tasks=64,
            ),
        ),
    )
    kernel.start()
    return kernel


def test_empty_graph_includes_global_route_deficit_and_matches_oracle(
    tmp_path: Path,
) -> None:
    store = CanonicalHypergraphStore(_kernel(tmp_path))
    graph = store.load()

    assert graph["scientific_sha256"] == store.full_recompute_oracle()[
        "scientific_sha256"
    ]
    assert graph["deficit_frontier"]["summary"]["by_kind"]["diversity"] == 1


def test_action_signal_is_canonical_frontier_work_not_scientific_fact(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    initial = store.load()
    target_id = str(initial["target_molecule_id"])
    signal = {
        "signal_id": "event-deficit:program-review:test",
        "deficit_id": "event-deficit:program-review:test",
        "kind": "program_review",
        "status": "open",
        "object_id": target_id,
        "entity_ids": [target_id],
        "route_family_ids": [],
        "dependency_ids": [],
        "deterministic": True,
        "model_allowed": False,
        "reason": "canonical_graph_requires_program_projection_review",
        "score": {
            "expected_portfolio_gain": 0.1,
            "distance_to_closure": 0.1,
            "evidence_gain": 0.1,
            "route_diversity_gain": 0.2,
            "cost_penalty": 0.05,
            "failure_risk_penalty": 0.02,
        },
        "metadata": {"program_review": True},
    }
    opened = store.apply(
        CanonicalIngestionBatch(action_signals=(signal,)),
        idempotency_key="open-program-review-signal",
    )["graph"]

    operational = next(
        row
        for row in opened["deficit_frontier"]["items"]
        if row["deficit_id"] == signal["deficit_id"]
    )
    assert operational["kind"] == "program_review"
    assert operational["metadata"]["operational_signal"] is True
    assert opened["scientific_sha256"] == initial["scientific_sha256"]

    resolved_signal = {
        **dict(opened["action_signals"][signal["signal_id"]]),
        "status": "resolved",
        "resolution": {"status": "completed"},
    }
    resolved_signal.pop("content_sha256", None)
    resolved = store.apply(
        CanonicalIngestionBatch(action_signals=(resolved_signal,)),
        idempotency_key="resolve-program-review-signal",
    )["graph"]

    assert all(
        row["deficit_id"] != signal["deficit_id"]
        for row in resolved["deficit_frontier"]["items"]
    )
    assert resolved["scientific_sha256"] == initial["scientific_sha256"]


def test_evidence_program_and_feedback_signals_reach_canonical_frontier(
    tmp_path: Path,
) -> None:
    store = CanonicalHypergraphStore(_kernel(tmp_path))
    initial = store.load()
    target_id = str(initial["target_molecule_id"])
    signals = (
        {
            "signal_id": "event-deficit:evidence-prefetch:test",
            "kind": "evidence",
            "object_id": target_id,
            "entity_ids": [target_id],
            "route_family_ids": [],
            "dependency_ids": [],
            "deterministic": False,
            "model_allowed": False,
            "reason": "target_source_prefetch_requires_evidence_acquisition",
            "metadata": {
                "target_level_evidence_prefetch": True,
                "evidence_prefetch_request_sha256": "prefetch:test",
            },
        },
        {
            "signal_id": "event-deficit:program-validation:test",
            "kind": "program_validation",
            "object_id": "experimental-work:test",
            "entity_ids": ["program:test"],
            "route_family_ids": ["route:test"],
            "dependency_ids": [],
            "deterministic": True,
            "model_allowed": False,
            "reason": "program_candidate_requires_specialized_validation",
            "metadata": {"program_validation": True},
        },
        {
            "signal_id": "event-deficit:experiment-feedback:test",
            "kind": "experiment_feedback",
            "object_id": "validation:test",
            "entity_ids": ["program:test"],
            "route_family_ids": ["route:test"],
            "dependency_ids": [],
            "deterministic": True,
            "model_allowed": False,
            "reason": "external_program_validation_feedback_available",
            "metadata": {
                "experiment_feedback": True,
                "route_id": "route:test",
                "validation": {"validation_id": "validation:test"},
            },
        },
    )

    result = store.apply(
        CanonicalIngestionBatch(action_signals=signals),
        idempotency_key="program-validation-feedback-signals",
    )
    graph = result["graph"]
    frontier_by_id = {
        row["deficit_id"]: row for row in graph["deficit_frontier"]["items"]
    }

    assert result["rejected"] == []
    assert frontier_by_id[signals[0]["signal_id"]]["kind"] == "evidence"
    assert frontier_by_id[signals[0]["signal_id"]]["metadata"][
        "target_level_evidence_prefetch"
    ] is True
    assert frontier_by_id[signals[1]["signal_id"]]["kind"] == "program_validation"
    assert frontier_by_id[signals[2]["signal_id"]]["kind"] == "experiment_feedback"
    assert graph["scientific_sha256"] == initial["scientific_sha256"]
    assert target_id == graph["target_molecule_id"]


def test_explicit_derived_projection_repair_restores_legacy_frontier(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    graph = CanonicalHypergraphStore(kernel).load()
    legacy_frontier = {**graph["deficit_frontier"], "items": []}
    legacy = {
        **graph,
        "deficit_frontier": legacy_frontier,
        "scientific_sha256": "legacy",
    }

    repaired, report = compile_canonical_hypergraph_revision(
        legacy,
        batch=CanonicalIngestionBatch(recompute_derived=True),
        acceptance_spec=kernel.spec.acceptance,
    )

    assert report["changed"] is True
    assert repaired["deficit_frontier"]["summary"]["by_kind"]["diversity"] == 1
    assert repaired["scientific_sha256"] == full_recompute_canonical_hypergraph(
        repaired,
        acceptance_spec=kernel.spec.acceptance,
    )["scientific_sha256"]


def test_rejected_stock_leaf_becomes_provider_expansion_deficit() -> None:
    frontier = compile_deficit_frontier(
        {
            "scientific_sha256": "fixture",
            "target_molecule_id": "molecule:target",
            "molecules": {
                "molecule:target": {
                    "canonical_smiles": "CCOC(C)=O",
                    "is_leaf": False,
                    "stock_closed": False,
                },
                "molecule:leaf": {
                    "canonical_smiles": "CC(=O)Cl",
                    "is_leaf": True,
                    "stock_closed": False,
                    "active_stock_observation_id": "stock:miss",
                },
            },
            "stock_observations": {
                "stock:miss": {"accepted": False, "reasons": ["catalog_miss"]}
            },
            "route_families": {
                "route:acyl": {
                    "selected": True,
                    "closed": False,
                    "edge_ids": ["edge:root"],
                    "leaf_molecule_ids": ["molecule:leaf"],
                }
            },
            "dependency_index": {
                "routes_by_entity": {"molecule:leaf": ["route:acyl"]}
            },
            "edges": {
                "edge:root": {
                    "edge_id": "edge:root",
                    "product_molecule_id": "molecule:target",
                    "precursor_molecule_ids": ["molecule:leaf"],
                    "status": "materialized",
                    "reaction_proofs": [],
                }
            },
            "hypotheses": {},
            "conflicts": {},
        }
    )

    expansion = next(
        row
        for row in frontier["items"]
        if row["kind"] == "expansion"
        and row["metadata"].get("stock_observation_id") == "stock:miss"
    )
    assert expansion["model_allowed"] is True
    assert expansion["deterministic"] is False
    assert expansion["metadata"]["frontier_smiles"] == "CC(=O)Cl"
    assert expansion["metadata"]["provider_preferences"][0] == "chemenzy"
    assert frontier["summary"]["by_kind"]["stock"] == 0


def test_stock_deficit_requires_current_selected_materialized_leaf_boundary() -> None:
    graph = {
        "scientific_sha256": "fixture",
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {
                "canonical_smiles": "CCO",
                "is_leaf": False,
                "stock_closed": False,
            },
            "molecule:leaf": {
                "canonical_smiles": "CC",
                "is_leaf": True,
                "stock_closed": False,
            },
        },
        "stock_observations": {},
        "route_families": {
            "route:selected": {
                "selected": True,
                "closed": False,
                "edge_ids": [],
                "leaf_molecule_ids": [],
            }
        },
        "dependency_index": {
            "routes_by_entity": {
                "molecule:leaf": ["route:selected"],
                "hypothesis:one": ["route:selected"],
            }
        },
        "edges": {},
        "hypotheses": {
            "hypothesis:one": {
                "status": "frontier_candidate",
            }
        },
        "conflicts": {},
    }

    proposal_only = compile_deficit_frontier(graph)
    assert proposal_only["summary"]["by_kind"]["materialization"] == 1
    assert proposal_only["summary"]["by_kind"]["stock"] == 0

    graph["route_families"]["route:selected"]["leaf_molecule_ids"] = [
        "molecule:leaf"
    ]
    graph["route_families"]["route:selected"]["edge_ids"] = ["edge:root"]
    graph["edges"]["edge:root"] = {
        "edge_id": "edge:root",
        "product_molecule_id": "molecule:target",
        "precursor_molecule_ids": ["molecule:leaf"],
        "status": "materialized",
        "reaction_proofs": [],
    }
    materialized_boundary = compile_deficit_frontier(graph)
    stock = next(
        row for row in materialized_boundary["items"] if row["kind"] == "stock"
    )
    assert stock["object_id"] == "molecule:leaf"
    assert stock["reason"] == "selected_leaf_requires_trusted_stock_audit"


def test_current_negative_reaction_proof_is_not_revalidated_without_new_input() -> None:
    frontier = compile_deficit_frontier(
        {
            "scientific_sha256": "fixture",
            "target_molecule_id": "molecule:target",
            "molecules": {
                "molecule:target": {
                    "canonical_smiles": "CCO",
                    "is_leaf": False,
                    "stock_closed": False,
                },
                "molecule:leaf": {
                    "canonical_smiles": "CC",
                    "is_leaf": True,
                    "stock_closed": False,
                },
            },
            "stock_observations": {},
            "route_families": {
                "route:selected": {
                    "selected": True,
                    "closed": False,
                    "edge_ids": ["edge:one"],
                    "leaf_molecule_ids": ["molecule:leaf"],
                }
            },
            "dependency_index": {
                "routes_by_entity": {
                    "edge:one": ["route:selected"],
                    "molecule:leaf": ["route:selected"],
                }
            },
            "edges": {
                "edge:one": {
                    "edge_id": "edge:one",
                    "status": "materialized",
                    "product_smiles": "CCO",
                    "precursor_smiles": ["CC"],
                    "reaction_proofs": [
                        {
                            "accepted": False,
                            "validator_version": CURRENT_REACTION_VALIDATOR_VERSION,
                            "reasons": ["reaction_edit_budget_exceeded"],
                        }
                    ],
                    "condition_predictions": [
                        {
                            "reagents": ["fixture"],
                            "solvent": "water",
                            "temperature_c": 25,
                            "authority_scope": "model_predicted_condition",
                        }
                    ],
                }
            },
            "hypotheses": {},
            "conflicts": {},
        }
    )

    assert not any(row["kind"] == "validation" for row in frontier["items"])


def test_discovered_source_lifecycle_becomes_evidence_deficit() -> None:
    binding = normalize_source_binding(
        {
            "source_kind": "paper_si",
            "source_ref": "doi:10.1000/restricted",
            "title": "restricted route paper",
            "acquisition_status": "queued_for_authorized_browser",
            "proxy_request_id": "pdfreq-one",
        }
    )
    frontier = compile_deficit_frontier(
        {
            "scientific_sha256": "fixture",
            "target_molecule_id": "molecule:target",
            "molecules": {
                "molecule:target": {
                    "canonical_smiles": "CC",
                    "is_leaf": True,
                    "stock_closed": False,
                }
            },
            "source_bindings": {binding["binding_id"]: binding},
            "exact_records": {},
            "stock_observations": {},
            "route_families": {},
            "dependency_index": {"routes_by_entity": {}},
            "edges": {},
            "hypotheses": {},
            "conflicts": {},
        }
    )

    item = next(
        value
        for value in frontier["items"]
        if value["object_id"] == binding["binding_id"]
    )
    assert item["kind"] == "evidence"
    assert item["reason"] == "source_waiting_authorized_pdf_acquisition"
    assert item["metadata"]["proxy_request_id"] == "pdfreq-one"
    assert item["model_allowed"] is False


def _plan() -> dict:
    return {
        "schema_version": "global_campaign_plan.v1",
        "route_families": [
            {
                "route_family_id": "family:acyl",
                "strategic_disconnection": "acyl substitution",
            }
        ],
        "multi_step_skeletons": [
            {
                "skeleton_id": "skeleton:acyl",
                "route_family_id": "family:acyl",
                "steps": [
                    {
                        "step_id": "step:ester",
                        "product_smiles": "CCOC(C)=O",
                        "precursor_smiles": ["CCO", "CC(=O)Cl"],
                        "transformation_hypothesis": "acyl substitution",
                    }
                ],
            }
        ],
    }


def _strategy_card() -> dict:
    return normalize_strategy_card(
        {
            "scaffold_motif": "ester target assembled from two fragments",
            "key_forward_transformation": "convergent acyl substitution",
            "key_bond_changes": ["form acyl C-O bond"],
            "functional_group_conflicts": [],
            "protection_policy": "avoid protection",
            "stereochemical_plan": "not applicable",
            "convergence_plan": "join alcohol and acyl donor",
            "strategic_step_count": 1,
            "skeleton_change_class": "fragment union",
            "expected_complexity_drop": "high",
            "orthogonality_basis": "acyl C-O construction",
            "strategy_signature": "convergent esterification",
            "execution_domain": "chemical",
        }
    )


def test_strategy_card_survives_plan_hypothesis_materialization_and_route(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    plan = _plan()
    card = _strategy_card()
    plan["route_families"][0]["strategy_card"] = card
    plan["route_families"][0]["chemical_critic"] = {
        "schema_version": "chemical_strategy_critique.v1",
        "status": "viable",
        "overall_assessment": "viable",
        "strategy_adherence": True,
        "step_assessments": [],
        "route_level_risks": [],
        "no_reaction_proof": True,
        "no_source_authority": True,
        "no_solved_claim": True,
    }
    step = plan["multi_step_skeletons"][0]["steps"][0]
    step["strategy_card"] = card
    step["strategy_id"] = card["strategy_id"]
    step["strategy_digest"] = card["strategy_digest"]
    step["strategy_anchor"] = True

    admitted = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="strategy-plan",
    )["graph"]
    route = next(iter(admitted["route_families"].values()))
    hypothesis = next(iter(admitted["hypotheses"].values()))
    assert route["strategy_digest"] == card["strategy_digest"]
    assert hypothesis["strategy_digest"] == card["strategy_digest"]
    assert route["chemical_critic"]["status"] == "viable"

    commands = materialization_commands_for_global_plan(
        plan,
        run_id=kernel.spec.run_id,
        input_revision=kernel.state.graph_revision,
    )
    assert commands[0].payload["strategy_cards"][0]["strategy_digest"] == card[
        "strategy_digest"
    ]
    results = tuple(runtime.execute(command) for command in commands)
    graph = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="strategy-materialized",
    )["graph"]
    edge = next(iter(graph["edges"].values()))
    assert edge["strategy_cards"][0]["strategy_digest"] == card["strategy_digest"]
    assert edge["chemical_strategy_critic"]["semantics"][
        "runs_before_evidence_acquisition"
    ] is True


def test_shared_reaction_keeps_every_route_strategy_through_materialization(
    tmp_path: Path,
) -> None:
    """A shared OR edge must not collapse independent route-policy bindings."""

    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    first_card = _strategy_card()
    second_card = normalize_strategy_card(
        {
            **first_card,
            "convergence_plan": "prepare the alcohol linearly, then use the shared acyl union",
            "orthogonality_basis": "linear alcohol branch before the common esterification",
            "strategy_signature": "linear alcohol route with shared esterification",
        }
    )
    plan = _plan()
    plan["route_families"] = [
        {
            "route_family_id": "family:convergent",
            "strategic_disconnection": "convergent acyl substitution",
            "strategy_card": first_card,
        },
        {
            "route_family_id": "family:linear",
            "strategic_disconnection": "linear alcohol synthesis then acyl substitution",
            "strategy_card": second_card,
        },
    ]
    shared_step = dict(plan["multi_step_skeletons"][0]["steps"][0])
    plan["multi_step_skeletons"] = [
        {
            "skeleton_id": "skeleton:convergent",
            "route_family_id": "family:convergent",
            "steps": [{**shared_step, "step_id": "step:convergent", "strategy_card": first_card}],
        },
        {
            "skeleton_id": "skeleton:linear",
            "route_family_id": "family:linear",
            "steps": [{**shared_step, "step_id": "step:linear", "strategy_card": second_card}],
        },
    ]

    admitted_result = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="shared-reaction-two-strategies-plan",
    )
    admitted = admitted_result["graph"]
    assert admitted_result["rejected"] == []
    assert len(admitted["route_families"]) == 2
    assert len(admitted["hypotheses"]) == 1
    hypothesis = next(iter(admitted["hypotheses"].values()))
    assert len(hypothesis["route_family_ids"]) == 2
    assert {card["strategy_digest"] for card in hypothesis["strategy_cards"]} == {
        first_card["strategy_digest"],
        second_card["strategy_digest"],
    }

    commands = store.frontier_materialization_commands()
    assert len(commands) == 1
    assert {
        card["strategy_digest"] for card in commands[0].payload["strategy_cards"]
    } == {first_card["strategy_digest"], second_card["strategy_digest"]}
    result = runtime.execute(commands[0])
    materialized_result = store.apply(
        CanonicalIngestionBatch(worker_results=(result,)),
        worker_runtime=runtime,
        idempotency_key="shared-reaction-two-strategies-materialized",
    )
    materialized = materialized_result["graph"]
    assert materialized_result["rejected"] == []
    assert len(materialized["edges"]) == 1
    edge = next(iter(materialized["edges"].values()))
    assert len(edge["route_family_ids"]) == 2
    assert {card["strategy_digest"] for card in edge["strategy_cards"]} == {
        first_card["strategy_digest"],
        second_card["strategy_digest"],
    }


def test_frozen_strategy_card_survives_distinct_multistep_reaction_edits(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    plan = _plan()
    card = _strategy_card()
    ester_operations = [{"op": "break_bond", "map_a": 3, "map_b": 4}]
    ethanol_operations = [{"op": "break_bond", "map_a": 2, "map_b": 3}]
    legacy_ester_card = normalize_strategy_card(
        card,
        reaction_operations=ester_operations,
    )
    legacy_ethanol_card = normalize_strategy_card(
        card,
        reaction_operations=ethanol_operations,
    )
    plan["route_families"][0]["strategy_card"] = card
    skeleton = plan["multi_step_skeletons"][0]
    skeleton["steps"] = [
        {
            "step_id": "step:ester-cleavage",
            "product_smiles": "CCOC(C)=O",
            "precursor_smiles": ["CCO", "CC=O"],
            "transformation_hypothesis": "acyl C-O bond construction",
            "strategy_card": legacy_ester_card,
            "strategy_id": card["strategy_id"],
            "strategy_digest": card["strategy_digest"],
            "strategy_anchor": True,
            "reaction_operations": ester_operations,
        },
        {
            "step_id": "step:ethanol-cleavage",
            "product_smiles": "CCO",
            "precursor_smiles": ["CC", "O"],
            "transformation_hypothesis": "C-O bond construction",
            "strategy_card": legacy_ethanol_card,
            "strategy_id": card["strategy_id"],
            "strategy_digest": card["strategy_digest"],
            "strategy_anchor": False,
            "reaction_operations": ethanol_operations,
        },
    ]

    admitted = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="multistep-frozen-strategy-plan",
    )["graph"]

    assert len(admitted["hypotheses"]) == 2
    assert all(
        row["admission_accepted"] is True
        and row["strategy_digest"] == card["strategy_digest"]
        for row in admitted["hypotheses"].values()
    )
    commands = store.frontier_materialization_commands()
    assert len(commands) == 2
    assert all(
        command.payload["strategy_cards"][0]["strategy_digest"]
        == card["strategy_digest"]
        for command in commands
    )
    results = tuple(runtime.execute(command) for command in commands)
    materialized = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="multistep-frozen-strategy-materialized",
    )["graph"]
    assert len(materialized["edges"]) == 2
    assert {
        origin["reaction_edit_digest"]
        for edge in materialized["edges"].values()
        for origin in edge["origin_records"]
    } == {
        command.payload["proposal_refs"][0]["reaction_edit_digest"]
        for command in commands
    }


def test_route_step_cannot_silently_replace_frozen_strategy(tmp_path: Path) -> None:
    store = CanonicalHypergraphStore(_kernel(tmp_path))
    plan = _plan()
    frozen = _strategy_card()
    replacement = normalize_strategy_card(
        {
            **frozen,
            "key_forward_transformation": "late oxidation instead of fragment union",
            "key_bond_changes": ["change C-O bond order"],
            "skeleton_change_class": "redox FGI",
            "strategy_signature": "late oxidation",
        }
    )
    plan["route_families"][0]["strategy_card"] = frozen
    plan["multi_step_skeletons"][0]["steps"][0]["strategy_card"] = replacement

    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="strategy-replacement",
    )["graph"]
    hypothesis = next(iter(graph["hypotheses"].values()))

    assert hypothesis["status"] == "admission_rejected"
    assert "strategy_replacement_conflict" in hypothesis["admission_reasons"]
    assert not store.frontier_materialization_commands()


def test_declared_route_internal_strategy_milestone_is_not_a_replacement(
    tmp_path: Path,
) -> None:
    store = CanonicalHypergraphStore(_kernel(tmp_path))
    plan = _plan()
    root = _strategy_card()
    milestone = normalize_strategy_card(
        {
            **root,
            "key_forward_transformation": "declared downstream annulation",
            "key_bond_changes": ["form downstream C-N bond"],
            "skeleton_change_class": "route-internal annulation milestone",
            "strategy_signature": "declared route-internal milestone",
        }
    )
    family = plan["route_families"][0]
    family["strategy_card"] = root
    family["root_strategy_card"] = root
    family["strategy_milestone_cards"] = [root, milestone]
    step = plan["multi_step_skeletons"][0]["steps"][0]
    step["strategy_card"] = milestone
    step["strategy_id"] = milestone["strategy_id"]
    step["strategy_digest"] = milestone["strategy_digest"]

    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="declared-route-internal-milestone",
    )["graph"]
    hypothesis = next(iter(graph["hypotheses"].values()))
    route = next(iter(graph["route_families"].values()))

    assert hypothesis["status"] == "frontier_candidate"
    assert hypothesis["admission_reasons"] == []
    assert {
        card["strategy_digest"] for card in route["strategy_cards"]
    } == {root["strategy_digest"], milestone["strategy_digest"]}
    assert len(store.frontier_materialization_commands()) == 1


def _command(
    kernel: RunKernel,
    worker_type: str,
    payload: dict,
    *,
    task_kind: str,
    suffix: str,
    artifact_refs: tuple[dict, ...] = (),
) -> WorkerCommand:
    return WorkerCommand(
        command_id=f"{worker_type}:{suffix}",
        run_id=kernel.spec.run_id,
        worker_type=worker_type,
        input_revision=kernel.state.graph_revision,
        idempotency_key=f"{worker_type}:{suffix}",
        payload=payload,
        budget=WorkerBudget(task_kind=task_kind),
        dependency_revisions={
            "graph_revision": kernel.state.graph_revision,
            "evidence_revision": kernel.state.evidence_revision,
        },
        artifact_refs=artifact_refs,
    )


def _apply_proposals(
    kernel: RunKernel,
    store: CanonicalHypergraphStore,
    runtime: WorkerRuntime,
    proposals: tuple[dict, ...],
    *,
    key: str,
) -> dict:
    commands = store.materialization_commands(proposals)
    results = tuple(runtime.execute(command) for command in commands)
    return store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key=key,
    )


def test_all_core_identities_are_canonical_and_order_independent() -> None:
    first_molecule = molecule_identity("OCC")
    second_molecule = molecule_identity("CCO")
    first_edge, first_audit = reaction_edge_identity(
        "CCOC(C)=O", ["CCO", "CC(=O)Cl"]
    )
    second_edge, second_audit = reaction_edge_identity(
        "CCOC(C)=O", ["CC(=O)Cl", "OCC"]
    )
    hypothesis, hypothesis_audit = hypothesis_identity(
        "CCOC(C)=O", ["CCO", "CC(=O)Cl"]
    )

    assert first_molecule == second_molecule
    assert first_edge == second_edge
    assert first_audit["precursor_smiles_multiset"] == second_audit[
        "precursor_smiles_multiset"
    ]
    assert hypothesis.removeprefix("hypothesis:") == first_edge.removeprefix("edge:")
    assert hypothesis_audit["accepted"] is True
    source_a = source_binding_identity(
        {
            "source_kind": "paper_si",
            "source_ref": "doi:10.1000/example",
            "independence_group": "doi:10.1000/example",
        }
    )
    source_b = source_binding_identity(
        {
            "source_kind": "paper_si",
            "source_ref": "doi:10.1000/example",
            "independence_group": "doi:10.1000/example",
            "title": "presentation-only title",
        }
    )
    stock_a = stock_observation_identity(
        {
            "leaf_id": "leaf:1",
            "canonical_smiles": "CCO",
            "inventory_snapshot_set_id": "inventory:1",
            "audited_as_of": "2026-07-13T00:00:00Z",
            "provider_result": {"content_hash": "a" * 64},
        }
    )
    target_id = first_molecule[0]
    route_a = route_family_identity(
        {"route_family_id": "family:a", "strategy": "acyl"},
        target_molecule_id=target_id,
    )
    route_b = route_family_identity(
        {"route_family_id": "family:a", "strategy": "acyl", "name": "label"},
        target_molecule_id=target_id,
    )
    assert source_a == source_b
    assert stock_a.startswith("stock:")
    assert route_a == route_b


def test_global_codex_plan_enters_real_frontier_then_materializes_once(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())

    planned = store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="plan",
    )
    graph = planned["graph"]
    hypothesis = next(iter(graph["hypotheses"].values()))

    assert planned["changed"] is True
    assert hypothesis["status"] == "frontier_candidate"
    assert graph["deficit_frontier"]["summary"]["by_kind"]["materialization"] == 1
    assert kernel.state.graph_revision == 1
    assert kernel.state.accepted_expansion_count == 0

    commands = store.frontier_materialization_commands()
    results = tuple(runtime.execute(command) for command in commands)
    materialized = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialized",
    )
    graph = materialized["graph"]
    edge = next(iter(graph["edges"].values()))
    route = next(iter(graph["route_families"].values()))

    assert len(graph["edges"]) == 1
    assert edge["origin_records"][0]["origin_kind"] == "codex_global_director"
    assert next(iter(graph["hypotheses"].values()))["status"] == "materialized"
    assert route["edge_ids"] == [edge["edge_id"]]
    assert graph["deficit_frontier"]["summary"]["by_kind"]["materialization"] == 0
    assert graph["deficit_frontier"]["summary"]["by_kind"]["validation"] == 1
    assert kernel.state.graph_revision == 2


def test_guided_frontier_materialization_preserves_canonical_parent_family(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    planned = store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="guided-parent-plan",
    )["graph"]
    parent_id = next(iter(planned["route_families"]))

    guided = store.apply(
        CanonicalIngestionBatch(
            hypotheses=(
                {
                    "step_id": "chemenzy:guided:route:1:step:1",
                    "proposal_id": "chemenzy:guided:route:1:step:1",
                    "route_family_id": "chemenzy:guided:route:1",
                    "canonical_route_family_id": parent_id,
                    "product_smiles": "CCO",
                    "precursor_smiles": ["CC=O"],
                    "origin_kind": "chemenzy",
                    "origin_ref": "guided-provider",
                    "transformation_hypothesis": "guided upstream expansion",
                },
            )
        ),
        idempotency_key="guided-parent-hypothesis",
    )["graph"]
    guided_hypothesis = next(
        row
        for row in guided["hypotheses"].values()
        if row["product_smiles"] == "CCO"
    )
    assert guided_hypothesis["route_family_ids"] == [parent_id]

    commands = store.frontier_materialization_commands(
        [guided_hypothesis["hypothesis_id"]]
    )
    proposal_ref = commands[0].payload["proposal_refs"][0]
    assert proposal_ref["route_family_id"] == "chemenzy:guided:route:1"
    assert proposal_ref["canonical_route_family_ids"] == [parent_id]

    materialized = store.apply(
        CanonicalIngestionBatch(
            worker_results=tuple(runtime.execute(command) for command in commands)
        ),
        worker_runtime=runtime,
        idempotency_key="guided-parent-materialized",
    )["graph"]
    guided_edge = next(
        row for row in materialized["edges"].values() if row["product_smiles"] == "CCO"
    )
    parent = materialized["route_families"][parent_id]
    assert guided_edge["route_family_ids"] == [parent_id]
    assert guided_edge["edge_id"] in parent["edge_ids"]


def test_aiz_mapped_atom_contributor_survives_admission_and_materialization(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    planned = store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="mapped-aiz-parent-plan",
    )["graph"]
    parent_id = next(iter(planned["route_families"]))
    product = "C=C[C@H](C)CCC=C(C)C"
    precursors = [
        "CC(C)=CCC[C@@H](C)C=O",
        "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1",
    ]
    mapped_reaction = (
        r"[CH2:1]=[CH:2][C@H:3]([CH3:4])[CH2:5][CH2:6]/[CH:7]="
        r"[C:8](\[CH3:9])[CH3:10]>>[CH3:1][P+:11]([c:12]1[cH:13]"
        r"[cH:14][cH:15][cH:16][cH:17]1)([c:18]1[cH:19][cH:20]"
        r"[cH:21][cH:22][cH:23]1)[c:24]1[cH:25][cH:26][cH:27]"
        r"[cH:28][cH:29]1.[CH:2]([C@H:3]([CH3:4])[CH2:5][CH2:6]/"
        r"[CH:7]=[C:8](\[CH3:9])[CH3:10])=[O:30]"
    )
    proposed = store.apply(
        CanonicalIngestionBatch(
            hypotheses=(
                {
                    "step_id": "aizynthfinder:guided-test:route:1:step:1",
                    "proposal_id": "aizynthfinder:guided-test:route:1:step:1",
                    "route_family_id": "aizynthfinder:guided-test:route:1",
                    "canonical_route_family_id": parent_id,
                    "product_smiles": product,
                    "precursor_smiles": precursors,
                    "origin_kind": "aizynthfinder",
                    "origin_ref": "aizynthfinder:mapped-test",
                    "transformation_hypothesis": "mapped Wittig disconnection",
                    "provider_reaction_metadata": {
                        "provider": "aizynthfinder",
                        "mapped_reaction_smiles": mapped_reaction,
                    },
                },
            )
        ),
        idempotency_key="mapped-aiz-hypothesis",
    )
    assert proposed["rejected"] == []
    hypothesis = next(
        row
        for row in proposed["graph"]["hypotheses"].values()
        if row["product_smiles"] == product
    )

    commands = store.frontier_materialization_commands(
        [hypothesis["hypothesis_id"]]
    )
    assert len(commands) == 1
    worker_result = runtime.execute(commands[0])
    assert worker_result.status == "completed"
    assert worker_result.payload["mapped_reaction_smiles"] == mapped_reaction

    materialized = store.apply(
        CanonicalIngestionBatch(worker_results=(worker_result,)),
        worker_runtime=runtime,
        idempotency_key="mapped-aiz-materialized",
    )
    assert materialized["rejected"] == []
    edge = next(
        row
        for row in materialized["graph"]["edges"].values()
        if row["product_smiles"] == product
    )
    assert edge["precursor_smiles"] == sorted(precursors)
    assert edge["edge_id"] in materialized["graph"]["route_families"][
        parent_id
    ]["edge_ids"]


def test_v36_provider_template_topology_stitches_while_critic_risk_stays_separate(
    tmp_path: Path,
) -> None:
    """Replay the v36 terminal provider edge that the old critic discarded."""

    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    plan = _plan()
    card = _strategy_card()
    plan["route_families"][0]["strategy_card"] = card
    planned = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="v36-provider-parent-plan",
    )["graph"]
    parent_id = next(iter(planned["route_families"]))
    product = (
        "C[C@@H]1CC(=O)CC(Br)(c2ccc(C(O)(C(=O)O)C(C)(O)C(=O)Cl)cc2)O1"
    )
    precursors = [
        "CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C",
        "COC(=O)C(C)(O)c1ccc(Br)cc1",
    ]

    admitted = store.apply(
        CanonicalIngestionBatch(
            hypotheses=(
                {
                    "step_id": "chemenzy:v36:terminal",
                    "canonical_route_family_id": parent_id,
                    "product_smiles": product,
                    "precursor_smiles": precursors,
                    "origin_kind": "chemenzy",
                    "origin_ref": "cached-v36-guided-result",
                    "strategy_card": card,
                    "transformation_hypothesis": "provider short-tail template",
                },
            )
        ),
        idempotency_key="v36-provider-terminal-hypothesis",
    )["graph"]
    hypothesis = next(
        row
        for row in admitted["hypotheses"].values()
        if row["product_smiles"] == product
    )

    assert hypothesis["admission_accepted"] is True
    assert hypothesis["status"] == "frontier_candidate"
    assert hypothesis["chemical_strategy_critic"]["accepted"] is False
    assert "critic_atom_provenance_deficit" in hypothesis[
        "chemical_strategy_critic"
    ]["blocking_reasons"]
    assert hypothesis["admission_semantics"] == {
        "provider_template_topology": True,
        "reaction_credibility_reported_separately": True,
        "critic_is_admission_authority": False,
    }

    commands = store.frontier_materialization_commands(
        [hypothesis["hypothesis_id"]]
    )
    assert len(commands) == 1
    materialized = store.apply(
        CanonicalIngestionBatch(
            worker_results=tuple(runtime.execute(command) for command in commands)
        ),
        worker_runtime=runtime,
        idempotency_key="v36-provider-terminal-materialized",
    )["graph"]
    edge = next(
        row
        for row in materialized["edges"].values()
        if row["product_smiles"] == product
    )
    assert edge["edge_id"] in materialized["route_families"][parent_id]["edge_ids"]
    assert edge["chemical_strategy_critic"]["accepted"] is False


def test_replaying_host_bound_routejson_clears_stale_map_namespace_rejection(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    plan = _plan()
    card = _strategy_card()
    plan["route_families"][0]["strategy_card"] = card
    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="routejson-readmission-parent",
    )["graph"]
    parent_id = next(iter(graph["route_families"]))
    product = (
        "CCOC(=O)C1(O)c2cc(Br)c(C3CC(=O)C[C@@H](C)O3)cc2C(=O)C1(C)O"
    )
    precursor = (
        "CCOC(=O)C1(O)c2cc(Br)c(C=CC(=O)C[C@@H](C)O)cc2C(=O)C1(C)O"
    )
    operations = [
        {"map_a": 19, "map_b": 26, "op": "break_bond"},
        {"delta": 1, "map_a": 19, "map_b": 20, "op": "change_bond_order"},
    ]
    base = {
        "step_id": "codex:v36:map-space",
        "canonical_route_family_id": parent_id,
        "product_smiles": product,
        "precursor_smiles": [precursor],
        "origin_kind": "codex_global_director",
        "strategy_card": card,
        "reaction_operations": operations,
    }
    rejected = store.apply(
        CanonicalIngestionBatch(hypotheses=(base,)),
        idempotency_key="routejson-before-host-map-binding",
    )["graph"]
    hypothesis_id, _ = hypothesis_identity(product, [precursor])
    assert rejected["hypotheses"][hypothesis_id]["admission_accepted"] is False
    assert "critic_reaction_operations_replay_failed" in rejected["hypotheses"][
        hypothesis_id
    ]["admission_reasons"]

    host_audit = replay_reactionjson(
        mapped_product_smiles=(
            "[CH3:1][CH2:2][O:3][C:4](=[O:5])[C:6]1([OH:7])"
            "[c:8]2[cH:9][c:10]([Br:31])[c:11]([CH:19]3[CH2:20]"
            "[C:21](=[O:22])[CH2:23][C@@H:24]([CH3:25])[O:26]3)"
            "[cH:12][c:13]2[C:14](=[O:15])[C:16]1([CH3:17])[OH:18]"
        ),
        operations=operations,
        expected_precursor_smiles=[precursor],
    )
    readmitted = store.apply(
        CanonicalIngestionBatch(
            hypotheses=({**base, "reactionjson_audit": host_audit},)
        ),
        idempotency_key="routejson-after-host-map-binding",
    )["graph"]["hypotheses"][hypothesis_id]

    assert readmitted["admission_accepted"] is True
    assert readmitted["admission_reasons"] == []
    assert readmitted["status"] == "frontier_candidate"
    assert readmitted["admission_history"][0]["admission_accepted"] is False
    assert "critic_reaction_operations_replay_failed" in readmitted[
        "admission_history"
    ][0]["admission_reasons"]


def test_admission_rejected_director_step_is_retained_as_l0_without_work(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    plan = _plan()
    plan["multi_step_skeletons"][0]["steps"] = [
        {
            "step_id": "step:missing-oxygen-source",
            "product_smiles": "CCO",
            "precursor_smiles": ["CC"],
            "transformation_hypothesis": "hydration with omitted oxygen source",
        }
    ]

    result = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="retain-rejected-plan-step",
    )
    graph = result["graph"]
    hypothesis = next(iter(graph["hypotheses"].values()))
    route = next(iter(graph["route_families"].values()))

    assert hypothesis["status"] == "admission_rejected"
    assert hypothesis["admission_accepted"] is False
    assert hypothesis["admission_reasons"] == [
        "element_inventory_not_conserved"
    ]
    assert route["hypothesis_ids"] == [hypothesis["hypothesis_id"]]
    assert route["edge_ids"] == []
    assert store.frontier_materialization_commands() == ()
    retained = next(
        row
        for row in result["rejected"]
        if row.get("hypothesis_id") == hypothesis["hypothesis_id"]
    )
    assert retained["retained_as_l0"] is True


def test_compiled_external_atom_step_materializes_from_the_target_root(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    product = "CCOC(C)=O"
    precursor = "CCO"
    mapped_product = "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH3:6]"
    operations = [{"op": "remove_group", "map_indices": [4, 5, 6]}]
    materialized = RouteJSONCompiler().compile_step(
        mapped_product_smiles=mapped_product,
        operations=operations,
        expected_product_smiles=product,
    )
    reactionjson_audit = dict(materialized.audit)
    row = {
        "step_id": "step:acetylation",
        "product_smiles": product,
        "mapped_product_smiles": mapped_product,
        "precursor_smiles": [precursor],
        "reaction_operations": operations,
        "reactionjson_audit": reactionjson_audit,
        "transformation_hypothesis": "alcohol acetylation",
    }
    _hypothesis_id, admission = hypothesis_identity(
        product,
        [precursor],
        mapped_product_smiles=mapped_product,
        reaction_operations=operations,
        reactionjson_audit=reactionjson_audit,
    )

    graph = store.apply(
        CanonicalIngestionBatch(hypotheses=(row,)),
        idempotency_key="compiled-acetylation-external-source",
    )["graph"]
    hypothesis = next(iter(graph["hypotheses"].values()))

    assert hypothesis["admission_accepted"] is True
    assert hypothesis["admission_reasons"] == []
    assert admission["replayed_external_atom_deficit_bound"] is True
    assert hypothesis["status"] == "frontier_candidate"

    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    commands = store.frontier_materialization_commands()
    assert len(commands) == 1
    results = tuple(runtime.execute(command) for command in commands)
    applied = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialize-compiled-acetylation-external-source",
    )
    materialized_graph = applied["graph"]
    assert materialized_graph["edges"], applied["rejected"]
    edge = next(iter(materialized_graph["edges"].values()))
    assert edge["product_smiles"] == product
    assert edge["precursor_smiles"] == [precursor]


def test_codex_provider_delegation_becomes_one_canonical_expansion_deficit(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    plan = _plan()
    plan["frontier_priorities"] = [
        {
            "priority_id": "priority:chemenzy:ethanol",
            "target_smiles": "OCC",
            "provider_preferences": ["chemenzy"],
            "retron_hints": ["alcohol feedstock alternatives"],
            "priority": 9,
            "rationale": "compare a local upstream module",
        }
    ]

    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="plan-with-provider-frontier",
    )["graph"]
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    results = tuple(
        runtime.execute(command)
        for command in store.frontier_materialization_commands()
    )
    graph = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialize-provider-frontier-parent",
    )["graph"]

    molecule_id, _ = molecule_identity("CCO")
    molecule = graph["molecules"][molecule_id]
    expansion = next(
        item
        for item in graph["deficit_frontier"]["items"]
        if item["kind"] == "expansion" and item["object_id"] == molecule_id
    )
    assert molecule["provider_expansion_requested"] is True
    assert molecule["provider_preferences"] == ["chemenzy"]
    assert expansion["reason"] == "codex_selected_frontier_requires_local_generation"
    assert expansion["metadata"]["frontier_smiles"] == "CCO"
    assert graph["deficit_frontier"]["semantics"][
        "frontier_is_not_scientific_authority"
    ] is True


def test_disconnected_provider_island_never_becomes_short_tail_work(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    plan = _plan()
    plan["multi_step_skeletons"][0]["steps"].append(
        {
            "step_id": "step:island",
            "product_smiles": "CCCO",
            "precursor_smiles": ["CCC=O"],
            "transformation_hypothesis": "disconnected local reduction",
        }
    )
    plan["frontier_priorities"] = [
        {
            "priority_id": "priority:island",
            "target_smiles": "CCCO",
            "provider_preferences": ["chemenzy"],
            "priority": 9,
        }
    ]
    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="plan-with-disconnected-provider-island",
    )["graph"]
    # Materialize only the island edge; its connector to the target is absent.
    island_commands = tuple(
        command
        for command in store.frontier_materialization_commands()
        if dict(command.payload).get("product_smiles") == "CCCO"
    )
    results = tuple(runtime.execute(command) for command in island_commands)
    graph = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialize-disconnected-provider-island",
    )["graph"]
    island_id, _ = molecule_identity("CCCO")

    assert not any(
        item["kind"] == "expansion" and item["object_id"] == island_id
        for item in graph["deficit_frontier"]["items"]
    )


def test_settled_short_tail_attempt_is_not_retried_or_reaudited() -> None:
    graph = {
        "scientific_sha256": "fixture",
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCO", "stock_closed": False},
            "molecule:leaf": {
                "canonical_smiles": "CC",
                "stock_closed": False,
                "active_stock_observation_id": "stock:miss",
            },
        },
        "stock_observations": {"stock:miss": {"accepted": False}},
        "route_families": {
            "route:selected": {
                "selected": True,
                "closed": False,
                "edge_ids": ["edge:root"],
            }
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "precursor_molecule_ids": ["molecule:leaf"],
                "status": "materialized",
                "reaction_proofs": [],
            }
        },
        "hypotheses": {},
        "action_signals": {
            "attempt:leaf": {
                "kind": "expansion",
                "status": "resolved",
                "object_id": "molecule:leaf",
                "metadata": {"guided_provider_attempt": True},
            }
        },
        "dependency_index": {"routes_by_entity": {}},
        "conflicts": {},
    }

    frontier = compile_deficit_frontier(graph)

    assert not any(
        item["object_id"] == "molecule:leaf"
        and item["kind"] in {"expansion", "stock"}
        for item in frontier["items"]
    )


def test_internal_node_provider_group_is_excluded_from_route_traversal() -> None:
    graph = {
        "target_molecule_id": "molecule:target",
        "molecules": {
            "molecule:target": {"canonical_smiles": "CCCC", "stock_closed": False},
            "molecule:internal": {"canonical_smiles": "CCC", "stock_closed": False},
            "molecule:leaf": {"canonical_smiles": "CC", "stock_closed": False},
            "molecule:provider-leaf": {
                "canonical_smiles": "C",
                "stock_closed": False,
            },
        },
        "edges": {
            "edge:root": {
                "edge_id": "edge:root",
                "product_molecule_id": "molecule:target",
                "precursor_molecule_ids": ["molecule:internal"],
                "origin_records": [{"origin_kind": "codex_global_director"}],
            },
            "edge:planned": {
                "edge_id": "edge:planned",
                "product_molecule_id": "molecule:internal",
                "precursor_molecule_ids": ["molecule:leaf"],
                "origin_records": [{"origin_kind": "codex_global_director"}],
            },
            "edge:invalid-tail": {
                "edge_id": "edge:invalid-tail",
                "product_molecule_id": "molecule:internal",
                "precursor_molecule_ids": ["molecule:provider-leaf"],
                "origin_records": [
                    {
                        "origin_kind": "chemenzy",
                        "origin_ref": "chemenzy:guided-internal:route:1:native",
                    }
                ],
            },
        },
        "route_families": {
            "route:one": {
                "selected": True,
                "edge_ids": ["edge:root", "edge:planned", "edge:invalid-tail"],
                "excluded_provider_group_ids": ["chemenzy:guided-internal"],
            }
        },
        "stock_observations": {},
        "hypotheses": {},
        "action_signals": {},
        "dependency_index": {"routes_by_entity": {}},
        "conflicts": {},
        "scientific_sha256": "fixture",
    }

    allowed = route_family_scoped_edge_ids(
        graph,
        family=graph["route_families"]["route:one"],
    )
    boundaries = target_reachable_route_boundaries(graph)

    assert allowed == {"edge:root", "edge:planned"}
    assert boundaries["open_leaf_route_family_ids"] == {
        "molecule:leaf": ("route:one",)
    }


def test_codex_can_delegate_a_non_leaf_shared_intermediate(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    plan = _plan()
    plan["multi_step_skeletons"][0]["steps"].append(
        {
            "step_id": "step:ethanol",
            "product_smiles": "CCO",
            "precursor_smiles": ["CC=O"],
            "transformation_hypothesis": "carbonyl reduction",
        }
    )
    plan["frontier_priorities"] = [
        {
            "priority_id": "priority:chemenzy:acid",
            "target_smiles": "CCO",
            "provider_preferences": ["chemenzy"],
            "retron_hints": ["alternative acid construction"],
            "priority": 9,
            "rationale": "compare upstream modules around a shared node",
        }
    ]

    graph = store.apply(
        CanonicalIngestionBatch(global_plans=(plan,)),
        idempotency_key="plan-with-non-leaf-provider-frontier",
    )["graph"]
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    results = tuple(
        runtime.execute(command)
        for command in store.frontier_materialization_commands()
    )
    graph = store.apply(
        CanonicalIngestionBatch(worker_results=results),
        worker_runtime=runtime,
        idempotency_key="materialize-non-leaf-provider-frontier",
    )["graph"]

    molecule_id, _ = molecule_identity("CCO")
    molecule = graph["molecules"][molecule_id]
    assert molecule["is_leaf"] is False
    expansion = next(
        item
        for item in graph["deficit_frontier"]["items"]
        if item["kind"] == "expansion" and item["object_id"] == molecule_id
    )
    assert expansion["reason"] == "codex_selected_frontier_requires_local_generation"
    assert expansion["metadata"]["frontier_smiles"] == "CCO"


def test_one_ingestion_path_deduplicates_edges_and_preserves_all_origins(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    base = {
        "product_smiles": "CCOC(C)=O",
        "precursor_smiles": ["CCO", "CC(=O)Cl"],
    }
    result = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {**base, "origin_kind": "aizynthfinder", "proposal_id": "aiz:1"},
            {**base, "origin_kind": "chemenzy", "proposal_id": "chem:1"},
            {**base, "origin_kind": "template", "proposal_id": "template:1"},
            {**base, "origin_kind": "manual", "proposal_id": "manual:1"},
        ),
        key="origins",
    )
    edge = next(iter(result["graph"]["edges"].values()))

    assert len(result["graph"]["edges"]) == 1
    assert {row["origin_kind"] for row in edge["origin_records"]} == {
        "aizynthfinder",
        "chemenzy",
        "template",
        "manual",
    }
    assert kernel.state.graph_revision == 1
    assert kernel.state.accepted_expansion_count == 1

    repeated = _apply_proposals(
        kernel,
        store,
        runtime,
        ({**base, "origin_kind": "manual", "proposal_id": "manual:1"},),
        key="same-origin-again",
    )
    assert repeated["changed"] is False
    assert kernel.state.graph_revision == 1


def test_cycle_and_impossible_edges_are_rejected_before_graph_expansion(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    first = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["COC"],
                "origin_kind": "manual",
            },
        ),
        key="forward",
    )
    second = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "COC",
                "precursor_smiles": ["CCO"],
                "origin_kind": "manual",
            },
            {
                "product_smiles": "CCCCCCCCCCCCCCCCCCCC",
                "precursor_smiles": ["C"],
                "origin_kind": "manual",
            },
        ),
        key="rejected",
    )

    assert len(first["graph"]["edges"]) == 1
    assert second["changed"] is False
    assert len(second["graph"]["edges"]) == 1
    reasons = {reason for row in second["rejected"] for reason in row["reasons"]}
    assert "ancestor_or_target_cycle" in reasons
    assert "large_atom_jump" in reasons


def test_graph_aware_materialization_rejection_retires_pending_hypothesis(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["COC"],
                "origin_kind": "manual",
            },
        ),
        key="cycle-parent",
    )
    planned = store.apply(
        CanonicalIngestionBatch(
            hypotheses=(
                {
                    "product_smiles": "COC",
                    "precursor_smiles": ["CCO"],
                    "origin_kind": "aizynthfinder",
                    "proposal_id": "aiz:cycle-return",
                },
            )
        ),
        idempotency_key="cycle-return-hypothesis",
    )["graph"]
    hypothesis_id = next(
        hypothesis_id
        for hypothesis_id, row in planned["hypotheses"].items()
        if row["product_smiles"] == "COC"
    )

    assert planned["hypotheses"][hypothesis_id]["status"] == "frontier_candidate"
    commands = store.frontier_materialization_commands((hypothesis_id,))
    assert len(commands) == 1
    worker_result = runtime.execute(commands[0])
    assert worker_result.status == "rejected"
    assert "ancestor_or_target_cycle" in worker_result.failure_reasons

    rejected = store.apply(
        CanonicalIngestionBatch(worker_results=(worker_result,)),
        worker_runtime=runtime,
        idempotency_key="cycle-return-materialization",
    )
    graph = rejected["graph"]
    hypothesis = graph["hypotheses"][hypothesis_id]

    assert rejected["changed"] is True
    assert hypothesis["status"] == "admission_rejected"
    assert hypothesis["admission_accepted"] is False
    assert "ancestor_or_target_cycle" in hypothesis["admission_reasons"]
    assert hypothesis["materialization_rejection"]["terminal_for_edge_identity"]
    assert hypothesis["admission_history"][-1]["status"] == "frontier_candidate"
    assert all(
        row["object_id"] != hypothesis_id
        for row in graph["deficit_frontier"]["items"]
        if row["kind"] == "materialization"
    )
    assert not store.frontier_materialization_commands((hypothesis_id,))


def test_incremental_projection_equals_full_recompute_oracle(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    first = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
            },
            {
                "product_smiles": "CC=O",
                "precursor_smiles": ["CCO"],
                "origin_kind": "chemenzy",
            },
        ),
        key="initial",
    )
    graph = first["graph"]
    oracle = full_recompute_canonical_hypergraph(
        graph,
        acceptance_spec=kernel.spec.acceptance,
    )

    assert graph["scientific_sha256"] == oracle["scientific_sha256"]
    assert canonical_scientific_projection(graph) == canonical_scientific_projection(
        oracle
    )
    assert graph["delta"]["dirty_entity_count"] <= graph["delta"][
        "total_entity_count"
    ]


def test_worker_facts_merge_order_independently_without_false_route_closure(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="plan-for-facts",
    )
    source = {
        "source_kind": "patent",
        "source_ref": "patent:US2020123456A1",
        "title": "Fixture process",
    }
    binding = normalize_source_binding(source)
    extraction_ref = kernel.artifacts.put_json(
        {
            "schema_version": "structured_exact_row_extraction.v1",
            "source_binding_id": binding["binding_id"],
            "extractor": {
                "producer_kind": "deterministic_structure_parser",
                "producer_id": "tests.fixture",
                "version": "1.0.0",
            },
            "rows": [
                {
                    "product_smiles": "CCOC(C)=O",
                    "reactant_smiles": ["CCO", "CC(=O)Cl"],
                    "location_ref": "Example 1",
                    "evidence_refs": ["procedure-text-sha256:" + "a" * 64],
                    "conditions": {"temperature_c": 20},
                }
            ],
        },
        logical_name="exact.json",
        producer="tests.fixture",
    ).to_dict()
    inventory_ref = kernel.artifacts.put_json(
        {
            "schema_version": "versioned_inventory_snapshot.v1",
            "adapter_version": "tests.inventory.v1",
            "inventory_version": "2026-07-13",
            "retrieved_at": "2026-07-13T00:00:00Z",
            "offers": [
                {
                    "schema_version": "stock_offer_snapshot.v1",
                    "supplier": "fixture",
                    "catalog_number": "ETHANOL",
                    "smiles": "CCO",
                    "checked_at": "2026-07-13T00:00:00Z",
                    "available": True,
                },
                {
                    "schema_version": "stock_offer_snapshot.v1",
                    "supplier": "fixture",
                    "catalog_number": "ACETYL-CHLORIDE",
                    "smiles": "CC(=O)Cl",
                    "checked_at": "2026-07-13T00:00:00Z",
                    "available": True,
                },
            ],
        },
        logical_name="inventory.json",
        producer="tests.inventory",
    ).to_dict()
    runtime = WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
        artifact_authorities={
            extraction_ref["sha256"]: "structured_exact_row_extraction",
            inventory_ref["sha256"]: "inventory_snapshot_set",
        },
    )
    materialized = runtime.execute(
        materialization_commands_for_global_plan(
            _plan(),
            run_id=kernel.spec.run_id,
            input_revision=kernel.state.graph_revision,
            dependency_revisions={
                "graph_revision": kernel.state.graph_revision,
                "evidence_revision": kernel.state.evidence_revision,
            },
        )[0]
    )
    validated = runtime.execute(
        _command(
            kernel,
            "validate_reaction",
            {
                "candidate": materialized.payload,
                "mapped_reaction_smiles": (
                    "[CH3:1][C:2](=[O:3])[Cl:4]."
                    "[CH3:5][CH2:6][OH:7]>>"
                    "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
                ),
            },
            task_kind="validation",
            suffix="ester",
        )
    )
    discovery_batch = runtime.execute_pipeline(
        _command(
            kernel,
            "discover_sources",
            {
                "sources": [
                    {
                        **source,
                        "extraction_artifact_sha256": extraction_ref["sha256"],
                    }
                ]
            },
            task_kind="evidence",
            suffix="patent",
            artifact_refs=(extraction_ref,),
        )
    )
    stock = runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": "CCOC(C)=O",
                "selected_deep_leaves": [
                    {"leaf_id": "leaf:ethanol", "smiles": "CCO"},
                    {"leaf_id": "leaf:acetyl-chloride", "smiles": "CC(=O)Cl"},
                ],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2026-07-13T12:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="leaves",
            artifact_refs=(inventory_ref,),
        )
    )

    assert validated.status == "completed"
    all_results = (
        stock,
        validated,
        *reversed(discovery_batch.results),
        materialized,
    )
    ingested = store.apply(
        CanonicalIngestionBatch(worker_results=all_results),
        worker_runtime=runtime,
        idempotency_key="all-worker-facts",
    )
    graph = ingested["graph"]
    edge = next(iter(graph["edges"].values()))
    route = next(iter(graph["route_families"].values()))

    assert len(edge["reaction_proofs"]) == 1
    assert len(edge["exact_record_ids"]) == 1
    exact = graph["exact_records"][edge["exact_record_ids"][0]]
    assert exact["procedure_authority_scope"] == ""
    assert exact["semantics"]["conditions_are_compatibility_projection_only"] is True
    assert exact["condition_completeness"]["complete"] is False
    assert set(exact["condition_completeness"]["missing_required_groups"]) == {
        "agents",
        "solvent",
        "time",
    }
    assert len(edge["procedure_record_ids"]) == 1
    procedure = graph["procedure_records"][edge["procedure_record_ids"][0]]
    assert procedure["procedure_authority_scope"] == (
        "source_exact_reaction_procedure"
    )
    assert procedure["source_fragment"]["procedure_text_sha256"] == "a" * 64
    assert procedure["condition_completeness"]["missing_required_groups"] == [
        "agents",
        "solvent",
        "time",
    ]
    assert len(edge["independent_source_groups"]) == 1
    assert {row["origin_kind"] for row in edge["origin_records"]} == {
        "codex_global_director",
        "literature",
    }
    assert all(
        graph["molecules"][molecule_id]["stock_closed"] is True
        for molecule_id in route["leaf_molecule_ids"]
    )
    assert route["minimum_proof_level"] == 3
    assert route["independent_source_requirement_met"] is False
    assert route["stock_closure_rate"] == 1.0
    assert route["closed"] is False
    assert graph["deficit_frontier"]["summary"]["by_kind"]["validation"] == 1
    revalidation = next(
        row
        for row in graph["deficit_frontier"]["items"]
        if row["kind"] == "validation"
    )
    assert revalidation["reason"] == "exact_evidence_requires_reaction_revalidation"
    assert revalidation["metadata"]["force_revalidation"] is True
    assert revalidation["metadata"]["exact_record_ids"] == edge["exact_record_ids"]
    assert graph["deficit_frontier"]["summary"]["by_kind"]["stock"] == 0
    assert graph["deficit_frontier"]["summary"]["by_kind"]["evidence"] == 1
    oracle = store.full_recompute_oracle()
    assert graph["scientific_sha256"] == oracle["scientific_sha256"]

    source_id = edge["source_binding_ids"][0]
    source_record = graph["source_bindings"][source_id]
    source_revoke = build_fact_lifecycle_event(
        subject_kind="source_binding",
        subject_id=source_id,
        subject_content_sha256=source_record["content_sha256"],
        action="revoke",
        effective_at="2026-07-15T12:00:00Z",
        reason_codes=["source_retracted"],
    )
    revoked = store.apply(
        CanonicalIngestionBatch(fact_lifecycle_events=(source_revoke,)),
        idempotency_key="revoke-source",
    )
    revoked_graph = revoked["graph"]
    revoked_route = next(iter(revoked_graph["route_families"].values()))
    revoked_proof = stitch_edge_proof(
        revoked_graph,
        edge["edge_id"],
        policy=ProofPolicy.from_acceptance(kernel.spec.acceptance),
    )

    assert revoked["changed"] is True
    assert revoked_proof["reaction_validated"] is True
    assert revoked_proof["exact_source_bound"] is False
    assert revoked_proof["inactive_facts"][0]["status"] == "revoked"
    assert revoked_route["minimum_proof_level"] == 2
    assert revoked_route["closed"] is False
    assert edge["exact_record_ids"][0] in revoked_graph["exact_records"]
    assert source_revoke["event_id"] in revoked_graph["fact_lifecycle_events"]
    assert {
        row["reason"] for row in revoked_graph["deficit_frontier"]["items"]
    } >= {
        "edge_requires_exact_source_binding",
        "source_fact_revoked_requires_replacement",
    }
    duplicate = store.apply(
        CanonicalIngestionBatch(fact_lifecycle_events=(source_revoke,)),
        idempotency_key="revoke-source-duplicate",
    )
    assert duplicate["changed"] is False

    source_restore = build_fact_lifecycle_event(
        subject_kind="source_binding",
        subject_id=source_id,
        subject_content_sha256=source_record["content_sha256"],
        action="restore",
        effective_at="2026-07-15T13:00:00Z",
        reason_codes=["retraction_withdrawn"],
        supersedes_event_id=source_revoke["event_id"],
    )
    restored_graph = store.apply(
        CanonicalIngestionBatch(fact_lifecycle_events=(source_restore,)),
        idempotency_key="restore-source",
    )["graph"]
    restored_proof = stitch_edge_proof(
        restored_graph,
        edge["edge_id"],
        policy=ProofPolicy.from_acceptance(kernel.spec.acceptance),
    )
    assert restored_proof["exact_source_bound"] is True
    assert restored_proof["inactive_fact_count"] == 0

    reaction_proof = edge["reaction_proofs"][0]
    proof_expiry = build_fact_lifecycle_event(
        subject_kind="reaction_proof",
        subject_id=reaction_proof["proof_digest"],
        subject_content_sha256=reaction_proof["proof_digest"],
        action="expire",
        effective_at="2026-07-16T00:00:00Z",
        reason_codes=["validator_authority_expired"],
    )
    proof_expired_graph = store.apply(
        CanonicalIngestionBatch(fact_lifecycle_events=(proof_expiry,)),
        idempotency_key="expire-proof",
    )["graph"]
    proof_expired_route = next(
        iter(proof_expired_graph["route_families"].values())
    )
    assert proof_expired_route["minimum_proof_level"] == 1
    assert (
        proof_expired_graph["deficit_frontier"]["summary"]["by_kind"][
            "validation"
        ]
        == 1
    )

    stock_id = next(
        observation_id
        for molecule_id in proof_expired_route["leaf_molecule_ids"]
        for observation_id in proof_expired_graph["molecules"][molecule_id][
            "stock_observation_ids"
        ]
    )
    stock_record = proof_expired_graph["stock_observations"][stock_id]
    stock_expiry = build_fact_lifecycle_event(
        subject_kind="stock_observation",
        subject_id=stock_id,
        subject_content_sha256=stock_record["content_sha256"],
        action="expire",
        effective_at="2026-07-16T00:05:00Z",
        reason_codes=["supplier_offer_expired"],
    )
    final_graph = store.apply(
        CanonicalIngestionBatch(fact_lifecycle_events=(stock_expiry,)),
        idempotency_key="expire-stock",
    )["graph"]
    final_route = next(iter(final_graph["route_families"].values()))
    assert final_route["stock_closure_rate"] == 0.5
    assert final_route["closed"] is False
    assert final_graph["scientific_sha256"] == store.full_recompute_oracle()[
        "scientific_sha256"
    ]


def test_deficit_frontier_ties_and_incremental_replacement_are_deterministic(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    result = store.apply(
        CanonicalIngestionBatch(global_plans=(_plan(),)),
        idempotency_key="frontier",
    )
    graph = result["graph"]
    full = compile_deficit_frontier(
        graph,
        acceptance_spec=kernel.spec.acceptance,
    )
    hypothesis_id = next(iter(graph["hypotheses"]))
    incremental = compile_deficit_frontier(
        graph,
        acceptance_spec=kernel.spec.acceptance,
        previous_frontier=full,
        dirty_entity_ids={hypothesis_id},
    )

    assert frontier_scientific_projection(full) == frontier_scientific_projection(
        incremental
    )
    assert [row["deficit_id"] for row in full["items"]] == [
        row["deficit_id"] for row in compile_deficit_frontier(
            graph,
            acceptance_spec=kernel.spec.acceptance,
        )["items"]
    ]


def test_different_procurement_boundaries_are_not_marked_dominated(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    store.apply(
        CanonicalIngestionBatch(
            route_families=(
                {
                    "route_family_id": "route:short",
                    "strategic_disconnection": "buy ethanol",
                },
                {
                    "route_family_id": "route:long",
                    "strategic_disconnection": "make ethanol",
                },
            )
        ),
        idempotency_key="routes",
    )
    result = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:short",
            },
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
        ),
        key="route-edges",
    )
    routes = result["graph"]["route_families"]
    short = next(route for route in routes.values() if "route:short" in route["aliases"])
    long = next(route for route in routes.values() if "route:long" in route["aliases"])

    assert set(short["edge_ids"]) < set(long["edge_ids"])
    assert set(short["leaf_molecule_ids"]) != set(long["leaf_molecule_ids"])
    assert long["status"] != "dominated"
    assert "dominated_by_route_family_id" not in long
    assert {short["route_family_id"], long["route_family_id"]} <= {
        row["route_family_id"] for row in result["graph"]["portfolio_ranking"]
    }


def test_same_boundary_edge_superset_remains_scientifically_actionable(
    tmp_path: Path,
) -> None:
    """A projection preference must not erase a topology-valid route family."""

    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    store.apply(
        CanonicalIngestionBatch(
            route_families=(
                {
                    "route_family_id": "route:short",
                    "strategic_disconnection": "direct assembly",
                },
                {
                    "route_family_id": "route:long",
                    "strategic_disconnection": "condition-rich detour",
                },
            )
        ),
        idempotency_key="same-boundary-routes",
    )
    result = _apply_proposals(
        kernel,
        store,
        runtime,
        (
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:short",
            },
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "origin_kind": "template",
                "route_family_id": "route:short",
            },
            {
                "product_smiles": "CCO",
                "precursor_smiles": ["CC", "O"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
            {
                "product_smiles": "CC(=O)Cl",
                "precursor_smiles": ["CC", "O"],
                "origin_kind": "template",
                "route_family_id": "route:short",
            },
            {
                "product_smiles": "CC(=O)Cl",
                "precursor_smiles": ["CC", "O"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
            {
                # An alternative detour reaches an intermediate already present
                # in the same acyclic route.  It leaves the audited terminal
                # boundary unchanged while making the edge set a strict superset.
                "product_smiles": "CCO",
                "precursor_smiles": ["CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:long",
            },
        ),
        key="same-boundary-edges",
    )
    graph = result["graph"]
    routes = graph["route_families"]
    short = next(route for route in routes.values() if "route:short" in route["aliases"])
    long = next(route for route in routes.values() if "route:long" in route["aliases"])

    assert set(short["edge_ids"]) < set(long["edge_ids"])
    assert set(short["leaf_molecule_ids"]) == set(long["leaf_molecule_ids"])
    assert long["status"] != "dominated"
    assert long.get("dominance_advisory", {}).get("non_authoritative") is True
    assert long["route_family_id"] in {
        route_id
        for item in graph["deficit_frontier"]["items"]
        for route_id in item.get("route_family_ids") or []
    }
    assert {short["route_family_id"], long["route_family_id"]} <= {
        row["route_family_id"] for row in graph["portfolio_ranking"]
    }


def test_local_stock_update_recomputes_only_dirty_subgraph_and_matches_oracle(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    proposal_runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    proposals = tuple(
        {
            "product_smiles": "C" * carbon_count + "O",
            "precursor_smiles": ["C" * (carbon_count - 1) + "OC"],
            "origin_kind": "template",
            "proposal_id": f"isomer:{carbon_count}",
        }
        for carbon_count in range(2, 17)
    )
    first = _apply_proposals(
        kernel,
        store,
        proposal_runtime,
        proposals,
        key="large-fixture",
    )
    selected_smiles = "C" * 9 + "OC"
    inventory_ref = kernel.artifacts.put_json(
        {
            "schema_version": "versioned_inventory_snapshot.v1",
            "adapter_version": "tests.inventory.v1",
            "inventory_version": "local-update",
            "retrieved_at": "2026-07-13T00:00:00Z",
            "offers": [
                {
                    "schema_version": "stock_offer_snapshot.v1",
                    "supplier": "fixture",
                    "catalog_number": "LOCAL-LEAF",
                    "smiles": selected_smiles,
                    "checked_at": "2026-07-13T00:00:00Z",
                    "available": True,
                }
            ],
        },
        logical_name="local-inventory.json",
        producer="tests.inventory",
    ).to_dict()
    stock_runtime = WorkerRuntime(
        kernel,
        build_retrosynthesis_worker_handlers(),
        artifact_authorities={inventory_ref["sha256"]: "inventory_snapshot_set"},
    )
    stock_result = stock_runtime.execute(
        _command(
            kernel,
            "audit_deep_leaf_stock",
            {
                "target_smiles": kernel.spec.target_smiles,
                "selected_deep_leaves": [
                    {"leaf_id": "leaf:local", "smiles": selected_smiles}
                ],
                "inventory_artifact_sha256": inventory_ref["sha256"],
                "as_of": "2026-07-13T12:00:00Z",
                "max_age_days": 30,
            },
            task_kind="stock",
            suffix="local-update",
            artifact_refs=(inventory_ref,),
        )
    )
    updated = store.apply(
        CanonicalIngestionBatch(worker_results=(stock_result,)),
        worker_runtime=stock_runtime,
        idempotency_key="local-stock",
    )
    graph = updated["graph"]
    oracle = store.full_recompute_oracle()

    assert len(first["graph"]["edges"]) == 15
    assert graph["delta"]["recomputed_fraction"] < 0.2
    assert graph["scientific_sha256"] == oracle["scientific_sha256"]
    assert canonical_scientific_projection(graph) == canonical_scientific_projection(
        oracle
    )


def test_inventory_snapshot_change_updates_stock_facts_without_rewriting_topology(
    tmp_path: Path,
) -> None:
    kernel = _kernel(tmp_path)
    store = CanonicalHypergraphStore(kernel)
    proposal_runtime = WorkerRuntime(kernel, build_retrosynthesis_worker_handlers())
    _apply_proposals(
        kernel,
        store,
        proposal_runtime,
        (
            {
                "product_smiles": "CCOC(C)=O",
                "precursor_smiles": ["CCO", "CC(=O)Cl"],
                "origin_kind": "template",
                "route_family_id": "route:stock-snapshot",
            },
        ),
        key="stock-snapshot-topology",
    )

    def audit_snapshot(*, available: bool, day: int, suffix: str) -> dict:
        timestamp = f"2026-07-{day:02d}T00:00:00Z"
        inventory_ref = kernel.artifacts.put_json(
            {
                "schema_version": "versioned_inventory_snapshot.v1",
                "adapter_version": "tests.inventory.v1",
                "inventory_version": suffix,
                "retrieved_at": timestamp,
                "offers": [
                    {
                        "schema_version": "stock_offer_snapshot.v1",
                        "supplier": "fixture",
                        "catalog_number": f"ETHANOL-{suffix}",
                        "smiles": "CCO",
                        "checked_at": timestamp,
                        "available": available,
                    }
                ],
            },
            logical_name=f"inventory-{suffix}.json",
            producer="tests.inventory",
        ).to_dict()
        runtime = WorkerRuntime(
            kernel,
            build_retrosynthesis_worker_handlers(),
            artifact_authorities={inventory_ref["sha256"]: "inventory_snapshot_set"},
        )
        result = runtime.execute(
            _command(
                kernel,
                "audit_deep_leaf_stock",
                {
                    "target_smiles": kernel.spec.target_smiles,
                    "selected_deep_leaves": [{"leaf_id": "leaf:ethanol", "smiles": "CCO"}],
                    "inventory_artifact_sha256": inventory_ref["sha256"],
                    "as_of": f"2026-07-{day:02d}T12:00:00Z",
                    "max_age_days": 30,
                },
                task_kind="stock",
                suffix=suffix,
                artifact_refs=(inventory_ref,),
            )
        )
        return store.apply(
            CanonicalIngestionBatch(worker_results=(result,)),
            worker_runtime=runtime,
            idempotency_key=f"stock-snapshot-{suffix}",
        )["graph"]

    available_graph = audit_snapshot(available=True, day=13, suffix="available")
    unavailable_graph = audit_snapshot(available=False, day=14, suffix="unavailable")

    def topology(graph: dict) -> dict:
        return {
            "target_molecule_id": graph["target_molecule_id"],
            "edges": {
                edge_id: {
                    "product_molecule_id": edge["product_molecule_id"],
                    "precursor_molecule_ids": edge["precursor_molecule_ids"],
                }
                for edge_id, edge in graph["edges"].items()
            },
            "route_families": {
                route_id: {
                    "edge_ids": route["edge_ids"],
                    "leaf_molecule_ids": route["leaf_molecule_ids"],
                }
                for route_id, route in graph["route_families"].items()
            },
        }

    ethanol_id = molecule_identity("CCO")[0]
    first_active = available_graph["molecules"][ethanol_id]["active_stock_observation_id"]
    second_active = unavailable_graph["molecules"][ethanol_id]["active_stock_observation_id"]
    assert topology(available_graph) == topology(unavailable_graph)
    assert first_active != second_active
    assert available_graph["stock_observations"][first_active]["accepted"] is True
    assert unavailable_graph["stock_observations"][second_active]["accepted"] is False
    assert available_graph["scientific_sha256"] != unavailable_graph["scientific_sha256"]
    assert unavailable_graph["scientific_sha256"] == store.full_recompute_oracle()[
        "scientific_sha256"
    ]
