from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cascade_planner.agent.codex_worker import (
    WorkerRunRecord,
    _assign_child_roles,
    _parse_codex_jsonl_events,
    _worker_output_json_schema,
)
from cascade_planner.harness.reaction_step_verifier import REACTION_STEP_VERIFIER_VERSION
from cascade_planner.harness.route_verifier import verify_chemenzy_raw_routes
from cascade_planner.harness.codex_edge_verification import verify_codex_consensus_graph
from cascade_planner.orchestration.codex_retrosynthesis import (
    DEFAULT_CHILD_ROLES,
    RetrosynthesisTeamConfig,
    _assemble_canonical_route_consensus_graph,
    _child_report_payload,
    _conservative_child_report_shape_repair,
    _strict_child_report_shape_reasons,
    build_retrosynthesis_coordinator_task,
    campaign_closure_status,
    migrate_legacy_campaign_commits,
    reconcile_codex_campaign_proof_state,
    run_codex_retrosynthesis_team,
    run_codex_retrosynthesis_campaign,
)
from cascade_planner.orchestration.admitted_hyperedges import (
    canonical_graph_step_id,
    canonical_graph_step_signature,
)
from cascade_planner.application.frontier_ledger import exact_edge_signature
from cascade_planner.routes.admission_receipts import (
    make_chemenzy_admission_material,
)
from cascade_planner.routes.admission import audit_retrosynthetic_candidate
from cascade_planner.routes.graph import select_route_consensus_frontier


def proposal_artifact(case_id: str = "case") -> dict:
    return {
        "schema_version": "retrosynthesis_proposal_report_artifact.v1",
        "artifact_id": f"{case_id}:proposal_report",
        "artifact_type": "RetrosynthesisProposalReport",
        "case_id": case_id,
        "source": "codex_cli",
        "input_refs": ["context_snapshot.json"],
        "evidence_refs": [],
        "validation_status": "draft",
        "summary": "child-agent synthesis",
        "payload": {
            "schema_version": "retrosynthesis_proposal_report.v1",
            "case_id": case_id,
            "agent_role": "retrosynthesis_coordinator",
            "target_smiles": "CCO",
            "candidates": [
                {
                    "schema_version": "retrosynthesis_candidate.v1",
                    "candidate_id": "candidate:aldehyde",
                    "product_smiles": "CCO",
                    "precursor_smiles": ["CC=O"],
                    "reaction_family": "carbonyl reduction",
                    "transformation_rationale": "aldehyde precursor",
                    "source_channel": "codex_strategy",
                    "source_refs": [],
                    "evidence_refs": [],
                    "evidence_level": "model_only",
                    "confidence": "medium",
                    "conditions": [],
                    "catalyst": "",
                    "enzyme": "",
                    "limitations": ["model hypothesis"],
                    "required_validation": ["forward_reconstruction"],
                    "no_solved_claim": True,
                    "not_parent_route_proof": True,
                }
            ],
            "evidence_refs": [],
            "limitations": [],
            "no_solved_claim": True,
        },
    }


def child_report_message(case_id: str, role: str, *, with_candidate: bool) -> str:
    payload = dict(proposal_artifact(case_id)["payload"])
    payload["agent_role"] = role
    payload["candidates"] = list(payload["candidates"]) if with_candidate else []
    return json.dumps(payload, sort_keys=True)


def accepted_runner_record(task) -> WorkerRunRecord:
    return WorkerRunRecord(
        run_id="team:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        backend="codex_cli",
        output_artifact=proposal_artifact(task.case_id),
        output_validation={"accepted": True, "reasons": []},
        metadata={
            "session_id": "thread-1",
            "event_summary": {"child_agent_spawn_count": len(task.child_roles)},
            "child_agents": [
                {
                    "agent_id": f"child-{index}",
                    "role": role,
                    "role_binding_method": "explicit_spawn_contract",
                    "wait_call_id": f"wait-{index}",
                    "status": "completed",
                    "arguments": {"role": role},
                    "message": child_report_message(
                        task.case_id,
                        role,
                        with_candidate=role == "target_structure_strategist",
                    ),
                }
                for index, role in enumerate(task.child_roles)
            ],
        },
        usage={"input_tokens": 100, "output_tokens": 50},
    )


def partial_runner_record(
    task,
    *,
    valid_role_count: int = 3,
    self_reported_evidence_level: str = "model_only",
    exit_code: int = 0,
) -> WorkerRunRecord:
    children = []
    for index, role in enumerate(task.child_roles):
        if index < valid_role_count:
            payload = json.loads(
                child_report_message(
                    task.case_id,
                    role,
                    with_candidate=index == 0,
                )
            )
            for candidate in payload["candidates"]:
                candidate["evidence_level"] = self_reported_evidence_level
                candidate["confidence"] = "high"
            status = "completed"
            message = json.dumps(payload, sort_keys=True)
            wait_call_id = f"wait-{index}"
        else:
            status = "running"
            message = "{not-valid-json"
            wait_call_id = f"wait-{index}"
        children.append(
            {
                "agent_id": f"child-{index}",
                "role": role,
                "role_binding_method": "explicit_spawn_contract",
                "wait_call_id": wait_call_id,
                "status": status,
                "arguments": {"role": role},
                "message": message,
            }
        )
    return WorkerRunRecord(
        run_id="team:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="rejected_output",
        backend="codex_cli",
        exit_code=exit_code,
        output_artifact=proposal_artifact(task.case_id),
        output_validation={
            "accepted": False,
            "reasons": ["required_child_agents_not_completed"],
        },
        metadata={
            # This producer summary is deliberately not consumed as spawn
            # authority; the host-observed child_agents records are.
            "event_summary": {"child_agent_spawn_count": len(task.child_roles)},
            "child_agents": children,
        },
    )


def _objective_closure_ledger(
    *,
    benchmark_any: bool = True,
    benchmark_all: bool = True,
    procurement_any: bool = False,
    procurement_all: bool = False,
) -> dict:
    return {
        "root": {"canonical_smiles": "CCO"},
        "summary": {
            "any_benchmark_route_closed": benchmark_any,
            "all_explored_benchmark_closed": benchmark_all,
            "any_procurement_route_closed": procurement_any,
            "all_explored_procurement_closed": procurement_all,
            # These aliases are deliberately contradictory to prove that the
            # objective evaluator never consumes them as policy authority.
            "any_route_closed": True,
            "all_explored_graph_closed": True,
        },
        "molecules": {
            "CCO": {
                "proposal": {"outgoing_edge_signatures": ["edge:1"]},
                "stock": {"closed": False, "boundary_types": []},
            },
            "CC=O": {
                "proposal": {"outgoing_edge_signatures": []},
                "stock": {
                    "closed": True,
                    "procurement_boundary_closed": False,
                    "boundary_types": ["benchmark_stock"],
                },
            },
        },
        "edges": {
            "edge:1": {
                "product_smiles": "CCO",
                "precursor_smiles": ["CC=O"],
                "reaction_proof": {"closed": True},
            }
        },
    }


def test_closure_objective_is_independent_from_generic_graph_aliases() -> None:
    ledger = _objective_closure_ledger()

    benchmark = campaign_closure_status(
        ledger,
        authoritative=True,
        closure_objective="benchmark_search",
        exploration_mode="exhaustive",
    )
    procurement = campaign_closure_status(
        ledger,
        authoritative=True,
        closure_objective="procurement",
        exploration_mode="exhaustive",
    )

    assert benchmark["route_solved"] is True
    assert benchmark["campaign_search_complete"] is True
    assert procurement["route_solved"] is False
    assert procurement["campaign_search_complete"] is False
    assert procurement["all_reaction_edges_closed"] is True
    assert procurement["all_benchmark_leaves_closed"] is True
    assert procurement["all_procurement_leaves_closed"] is False


def test_exhaustive_mode_does_not_promote_one_solved_route_to_completion() -> None:
    ledger = _objective_closure_ledger(benchmark_any=True, benchmark_all=False)

    exhaustive = campaign_closure_status(
        ledger,
        authoritative=True,
        closure_objective="benchmark_search",
        exploration_mode="exhaustive",
    )
    first_solved = campaign_closure_status(
        ledger,
        authoritative=True,
        closure_objective="benchmark_search",
        exploration_mode="first_solved",
    )

    assert exhaustive["route_solved"] is True
    assert exhaustive["campaign_search_complete"] is False
    assert exhaustive["selected_objective_all_explored_closed"] is False
    assert first_solved["route_solved"] is True
    assert first_solved["campaign_search_complete"] is True


def accepted_runner_record_for_target(task, precursor_smiles: str) -> WorkerRunRecord:
    context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
    target = context["target"]["smiles"]
    artifact = proposal_artifact(task.case_id)
    payload = artifact["payload"]
    payload["target_smiles"] = target
    payload["candidates"][0]["product_smiles"] = target
    payload["candidates"][0]["precursor_smiles"] = [precursor_smiles]
    children = []
    for index, role in enumerate(task.child_roles):
        child_payload = dict(payload)
        child_payload["agent_role"] = role
        child_payload["candidates"] = list(payload["candidates"]) if index == 0 else []
        children.append(
            {
                "agent_id": f"child-{index}",
                "role": role,
                "role_binding_method": "explicit_spawn_contract",
                "wait_call_id": f"wait-{index}",
                "status": "completed",
                "arguments": {"role": role},
                "message": json.dumps(child_payload, sort_keys=True),
            }
        )
    return WorkerRunRecord(
        run_id="team:run",
        task_id=task.task_id,
        case_id=task.case_id,
        status="accepted_draft",
        backend="codex_cli",
        output_artifact=artifact,
        output_validation={"accepted": True, "reasons": []},
        metadata={"session_id": "thread-1", "child_agents": children},
    )


def validated_step_proof(
    step: dict,
    *,
    proof_level: str,
    deterministic_transform: bool,
    trusted_precedent: bool,
) -> dict:
    checks = {
        key: True
        for key in (
            "structures_materialized",
            "mapped_reaction_present",
            "mapped_product_matches",
            "mapped_reactants_match",
            "atom_maps_complete",
            "atom_maps_unique",
            "product_atoms_have_reactant_provenance",
            "mapped_elements_preserved",
            "mapped_reactant_components_contribute",
            "scaffold_continuity_plausible",
            "ring_change_plausible",
            "bond_change_present",
            "reaction_edit_budget_plausible",
            "stereochemical_product_matches",
        )
    }
    checks["deterministic_transform_reapplied"] = deterministic_transform
    checks["trusted_precedent_bound"] = trusted_precedent
    proof = {
        "schema_version": "reaction_step_proof.v1",
        "step_id": step["step_id"],
        "step_index": 0,
        "product_smiles": step["product_smiles"],
        "reactant_smiles": step["precursor_smiles"],
        "proof_level": proof_level,
        "accepted": True,
        "checks": checks,
        "reasons": [],
        "mapping_source": "fixture",
        "atom_map_audit": {},
        "bond_change_audit": {},
        "trusted_precedent_binding": {"accepted": trusted_precedent},
        "procurement_binding": {},
        "reaction_digest": "a" * 64,
        "input_digest": "b" * 64,
        "validator_version": REACTION_STEP_VERIFIER_VERSION,
    }
    proof["proof_digest"] = hashlib.sha256(
        json.dumps(
            proof,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return proof


def materialized_reduction_candidate(step: dict) -> dict:
    return {
        "schema_version": "materialized_reaction_candidate.v1",
        "step_id": step["step_id"],
        "product_smiles": step["product_smiles"],
        "reactant_smiles": step["precursor_smiles"],
        "reaction_smiles": "CC=O>>CCO",
        "atom_mapped_reaction_smiles": (
            "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
        ),
        "mapping_source": "test_materialized_candidate",
    }


def current_host_chemenzy_admission_receipts(
    tmp_path: Path,
    *,
    case_id: str,
    product_smiles: str,
    precursor_smiles: list[str],
) -> dict:
    edge_key = exact_edge_signature(product_smiles, precursor_smiles)
    stock_path = tmp_path / f"admission-stock-{edge_key.rsplit(':', 1)[-1][:12]}.csv"
    stock_path.write_text(
        "smiles\n" + "\n".join(precursor_smiles) + "\n",
        encoding="utf-8",
    )
    stock_sha256 = hashlib.sha256(stock_path.read_bytes()).hexdigest()
    report = verify_chemenzy_raw_routes(
        {
            "target": product_smiles,
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": precursor_smiles,
                        "terminal_stock_status": {
                            value: True for value in precursor_smiles
                        },
                    },
                    "steps": [
                        {
                            "index": 0,
                            "product": product_smiles,
                            "reactant_smiles": precursor_smiles,
                            "stock_status": {
                                value: True for value in precursor_smiles
                            },
                        }
                    ],
                }
            ],
            "stock_catalog_context": {
                "effective_stock_names": ["admission-test-stock"],
                "catalog_bindings": [
                    {
                        "name": "admission-test-stock",
                        "path": str(stock_path),
                        "sha256": stock_sha256,
                    }
                ],
            },
        },
        target_smiles=product_smiles,
        case_id=case_id,
    )
    assert report["accepted"] is True
    bank = report["route_proof_bank"]
    material = make_chemenzy_admission_material(
        bank,
        source_entry_id=bank["entries"][0]["proof_id"],
        source_step_index=0,
        artifact_ref="fixture:current-host-chemenzy-bank",
    )
    assert material
    return {edge_key: [material]}


def rehash_content_sha256(payload: dict) -> dict:
    row = dict(payload)
    row.pop("content_sha256", None)
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return row


def test_canonical_graph_capacity_keeps_more_than_256_durable_edges() -> None:
    expansions = []
    for index in range(1, 301):
        precursor = f"C[{index}CH2]O"
        assert audit_retrosynthetic_candidate("CCO", [precursor])[
            "accepted"
        ] is True
        expansions.append(
            {
                "schema_version": "route_consensus_expansion.v1",
                "expansion_id": f"expansion:{index}",
                "requested_product_smiles": "CCO",
                "depth": 0,
                "route_consensus": {
                    "schema_version": "route_consensus.v1",
                    "target_smiles": "CCO",
                    "proposals": [
                        {
                            "schema_version": "route_consensus_proposal.v1",
                            "consensus_id": f"proposal:{index}",
                            "product_smiles": "CCO",
                            "precursor_smiles": [precursor],
                            "rank_score": 0.5,
                            "no_solved_claim": True,
                            "not_parent_route_proof": True,
                        }
                    ],
                },
            }
        )

    graph = _assemble_canonical_route_consensus_graph(
        expansions,
        case_id="large-canonical-union",
        target_smiles="CCO",
        max_depth=2,
    )

    assert len(graph["steps"]) == 300
    assert graph["truncation"]["graph_steps_truncated"] is False
    assert len(select_route_consensus_frontier(graph, limit=400)) == 300


def test_coordinator_task_requires_direct_child_roles(tmp_path) -> None:
    task = build_retrosynthesis_coordinator_task(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        context_ref=str(tmp_path / "context.json"),
        allowed_workdir=tmp_path,
    )
    assert task.agent_mode == "coordinator"
    assert task.child_roles == list(DEFAULT_CHILD_ROLES)
    assert "spawn_agent" in task.allowed_tools
    assert "Directly call spawn_agent" not in task.objective  # objective is chemistry-facing wording
    assert "directly spawn" in task.objective.lower()
    assert "Spawn at most three children at once" in task.objective
    assert "pending_init" in task.objective
    assert "Never emit" in task.objective
    assert "no field\nmay be null" in task.objective
    assert 'confidence="low", catalyst="", enzyme="", and conditions=[]' in task.objective


def test_codex_retrosynthesis_schema_cannot_self_report_validated(tmp_path) -> None:
    task = build_retrosynthesis_coordinator_task(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        context_ref=str(tmp_path / "context.json"),
        allowed_workdir=tmp_path,
    )
    schema = _worker_output_json_schema(task)
    levels = (
        schema["properties"]["payload"]
        ["properties"]["candidates"]
        ["items"]["properties"]["evidence_level"]["enum"]
    )

    assert "validated" not in levels


def test_team_accepts_only_when_all_child_spawns_are_observed(tmp_path) -> None:
    def runner(task):
        return WorkerRunRecord(
            run_id="team:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="accepted_draft",
            backend="codex_cli",
            output_artifact=proposal_artifact(task.case_id),
            output_validation={"accepted": True, "reasons": []},
            metadata={
                "session_id": "thread-1",
                "event_summary": {"child_agent_spawn_count": len(task.child_roles)},
                "child_agents": [
                    {
                        "agent_id": f"child-{index}",
                        "role": role,
                        "role_binding_method": "explicit_spawn_contract",
                        "wait_call_id": f"wait-{index}",
                        "status": "completed",
                        "arguments": {"role": role},
                        "message": child_report_message(
                            task.case_id,
                            role,
                            with_candidate=role == "target_structure_strategist",
                        ),
                    }
                    for index, role in enumerate(task.child_roles)
                ],
            },
            usage={"input_tokens": 100, "output_tokens": 50},
        )

    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        runner=runner,
    )
    assert report["accepted"], report["reasons"]
    assert report["coordinator"]["session_id"] == "thread-1"
    assert len(report["coordinator"]["observed_child_agents"]) == len(DEFAULT_CHILD_ROLES)
    assert report["route_consensus"]["accepted"]
    assert report["blackboard_proposals"][0]["executable"] is False
    assert report["runtime_summary"]["consistent"]
    assert report["runtime_summary"]["last_event_cursor"] > 0
    assert len(report["runtime_summary"]["children"]) == len(DEFAULT_CHILD_ROLES)
    assert {row["state"] for row in report["runtime_summary"]["children"]} == {"succeeded"}
    assert {row["role"] for row in report["child_reports"]} == set(DEFAULT_CHILD_ROLES)
    assert all(row["accepted"] for row in report["child_reports"])
    assert report["route_consensus"]["proposals"][0]["source_records"][0]["report_ref"].endswith(
        "#agent=child-0"
    )
    assert (tmp_path / "codex_retrosynthesis_team" / "runtime_summary.json").is_file()


def test_strict_mode_rejects_three_of_four_valid_child_reports(tmp_path) -> None:
    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(child_acceptance_mode="strict_all"),
        runner=lambda task: partial_runner_record(task, valid_role_count=3),
    )

    assert report["accepted"] is False
    assert report["child_acceptance"]["mode"] == "strict_all"
    assert "required_child_reports_not_valid" in report["reasons"]


def test_partial_mode_accepts_three_of_four_only_as_l0(tmp_path) -> None:
    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            child_acceptance_mode="valid_subset_l0"
        ),
        runner=lambda task: partial_runner_record(
            task,
            valid_role_count=3,
            self_reported_evidence_level="literature_exact",
        ),
    )

    assert report["accepted"], report["reasons"]
    acceptance = report["child_acceptance"]
    assert acceptance["acceptance_tier"] == "valid_subset_l0"
    assert acceptance["derived_valid_child_quorum"] == 2
    assert len(acceptance["valid_child_roles"]) == 3
    assert len(acceptance["degraded_child_roles"]) == 1
    proposal = report["route_consensus"]["proposals"][0]
    assert proposal["validation_tier"] == "L0"
    assert proposal["achieved_proof_level"] == 0
    assert proposal["evidence_level"] == "model_only"
    assert proposal["authority_evidence_level"] == "model_only"
    assert proposal["confidence"] == "low"
    assert proposal["authority_bound"] is False
    assert proposal["no_solved_claim"] is True
    assert all(
        row["authority_evidence_level"] == "model_only"
        and row["authority_confidence"] == "low"
        and row["authority_bound"] is False
        for row in proposal["source_records"]
    )


def test_partial_mode_rejects_below_derived_valid_role_quorum(tmp_path) -> None:
    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            child_acceptance_mode="valid_subset_l0"
        ),
        runner=lambda task: partial_runner_record(task, valid_role_count=1),
    )

    assert report["accepted"] is False
    assert "valid_child_role_quorum_not_met" in report["reasons"]


def test_partial_mode_does_not_trust_coordinator_spawn_summary(tmp_path) -> None:
    def runner(task):
        record = partial_runner_record(task, valid_role_count=3)
        record.metadata["child_agents"].pop()
        record.metadata["event_summary"]["child_agent_spawn_count"] = len(
            task.child_roles
        )
        return record

    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            child_acceptance_mode="valid_subset_l0"
        ),
        runner=runner,
    )

    assert report["accepted"] is False
    assert "explicit_child_spawn_count_mismatch" in report["reasons"]
    assert (
        report["child_acceptance"]["all_required_roles_explicitly_spawned"]
        is False
    )


def test_partial_mode_keeps_nonzero_coordinator_exit_as_hard_failure(tmp_path) -> None:
    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            child_acceptance_mode="valid_subset_l0"
        ),
        runner=lambda task: partial_runner_record(
            task,
            valid_role_count=3,
            exit_code=9,
        ),
    )

    assert report["accepted"] is False
    assert "coordinator_exit_code_nonzero" in report["reasons"]


def test_team_rejects_unobserved_children(tmp_path) -> None:
    def runner(task):
        return WorkerRunRecord(
            run_id="team:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="accepted_draft",
            backend="codex_cli",
            output_artifact=proposal_artifact(task.case_id),
            output_validation={"accepted": True, "reasons": []},
            metadata={"child_agents": []},
        )

    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        runner=runner,
    )
    assert not report["accepted"]
    assert "required_child_agents_not_observed" in report["reasons"]
    assert "required_child_agents_not_succeeded" in report["reasons"]
    assert {row["state"] for row in report["runtime_summary"]["children"]} == {"lost"}


def test_campaign_persists_rejected_team_as_retryable_frontier(tmp_path) -> None:
    def runner(task):
        return WorkerRunRecord(
            run_id="team:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="accepted_draft",
            backend="codex_cli",
            output_artifact=proposal_artifact(task.case_id),
            output_validation={"accepted": True, "reasons": []},
            metadata={"child_agents": []},
        )

    report = run_codex_retrosynthesis_campaign(
        case_id="retryable-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_expansions=1),
        runner=runner,
    )

    jobs = report["campaign"]["frontier_queue"]["jobs"]
    self_job = next(row for row in jobs if row["frontier_smiles"] == "CCO")
    assert self_job["state"] == "retry_wait"
    assert "codex_team_report_rejected" in self_job["failure_reasons"]
    assert report["campaign"]["graph_complete"] is False


def test_campaign_retries_rejected_team_within_attempt_budget(tmp_path) -> None:
    calls = 0

    def flaky_runner(task):
        nonlocal calls
        calls += 1
        record = accepted_runner_record(task)
        if calls == 1:
            record.metadata = {"child_agents": []}
        return record

    report = run_codex_retrosynthesis_campaign(
        case_id="retry-then-accept-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=2,
            frontier_retry_base_seconds=0.01,
            frontier_retry_max_seconds=0.01,
            frontier_retry_wait_seconds=0.5,
        ),
        runner=flaky_runner,
    )

    root_job = next(
        row
        for row in report["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CCO"
    )
    assert calls == 2
    assert root_job["attempt"] == 2
    assert root_job["state"] == "succeeded"
    assert report["route_expansion_count"] == 1
    assert report["campaign"]["attempt_run_count"] == 2
    assert report["campaign"]["unique_frontier_run_count"] == 1


def test_campaign_attempt_ledger_starts_before_agent_and_projects_budget(tmp_path) -> None:
    observed_started: dict = {}

    def runner(task):
        attempt_root = tmp_path / "codex_retrosynthesis_team" / "campaign_attempts"
        started_paths = list(attempt_root.glob("*/started.json"))
        terminal_paths = list(attempt_root.glob("*/terminal.json"))
        assert len(started_paths) == 1
        assert terminal_paths == []
        observed_started.update(json.loads(started_paths[0].read_text(encoding="utf-8")))
        return accepted_runner_record(task)

    report = run_codex_retrosynthesis_campaign(
        case_id="attempt-ledger-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            max_attempt_runs=4,
        ),
        runner=runner,
    )

    attempt_root = tmp_path / "codex_retrosynthesis_team" / "campaign_attempts"
    terminal = json.loads(next(attempt_root.glob("*/terminal.json")).read_text(encoding="utf-8"))
    campaign_state = json.loads(
        (tmp_path / "codex_retrosynthesis_team" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )
    commit = json.loads(
        next(
            (tmp_path / "codex_retrosynthesis_team" / "campaign_commits").glob("*.json")
        ).read_text(encoding="utf-8")
    )

    assert observed_started["event"] == "started"
    assert terminal["event"] == "terminal"
    assert terminal["started_event_sha256"] == observed_started["content_sha256"]
    assert report["campaign"]["attempt_run_count"] == 1
    assert report["campaign"]["remaining_attempt_runs"] == 3
    assert campaign_state["attempt_budget"]["started_attempt_count"] == 1
    assert campaign_state["attempt_budget"]["terminal_attempt_count"] == 1
    assert commit["summary"]["proposal_expansion_recorded"] is True


def test_campaign_wide_attempt_cap_survives_new_invocation(tmp_path) -> None:
    calls = 0

    def rejected_runner(task):
        nonlocal calls
        calls += 1
        record = accepted_runner_record(task)
        record.metadata = {"child_agents": []}
        return record

    config = RetrosynthesisTeamConfig(
        max_depth=1,
        max_expansions=2,
        max_attempt_runs=1,
        max_attempt_runs_per_invocation=1,
        frontier_retry_base_seconds=0.0,
        frontier_retry_max_seconds=0.0,
        frontier_retry_wait_seconds=0.0,
    )
    first = run_codex_retrosynthesis_campaign(
        case_id="durable-attempt-cap-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=rejected_runner,
    )
    second = run_codex_retrosynthesis_campaign(
        case_id="durable-attempt-cap-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=lambda _: (_ for _ in ()).throw(
            AssertionError("durable campaign attempt cap must prevent another Agent call")
        ),
    )

    assert calls == 1
    assert first["campaign"]["attempt_run_count"] == 1
    assert second["campaign"]["attempt_run_count"] == 1
    assert second["campaign"]["remaining_attempt_runs"] == 0
    assert second["campaign"]["stop_reason"] == "campaign_attempt_run_budget_exhausted"


def test_campaign_policy_rejects_depth_or_stock_authority_change(tmp_path) -> None:
    first = run_codex_retrosynthesis_campaign(
        case_id="immutable-policy-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )
    policy_path = tmp_path / "codex_retrosynthesis_team" / "campaign_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["max_depth"] == 1
    assert policy["closure_objective"] == "benchmark_search"
    assert policy["exploration_mode"] == "exhaustive"
    assert policy["required_reaction_proof_level"] == 2
    assert policy["proposal_agent_policy"]["coordinator_contract_version"]
    assert policy["proposal_agent_policy"]["child_roles"] == list(
        DEFAULT_CHILD_ROLES
    )
    assert policy["stock_authority_binding"]["provider_descriptor"]["provider_id"]
    assert first["campaign"]["campaign_policy_sha256"] == policy["content_sha256"]

    with pytest.raises(ValueError, match="campaign policy mismatch"):
        run_codex_retrosynthesis_campaign(
            case_id="immutable-policy-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(max_depth=2, max_expansions=1),
            runner=lambda _: (_ for _ in ()).throw(
                AssertionError("policy mismatch must fail before Agent work")
            ),
        )

    for incompatible in (
        RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            closure_objective="procurement",
        ),
        RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            exploration_mode="first_solved",
        ),
    ):
        with pytest.raises(ValueError, match="campaign policy mismatch"):
            run_codex_retrosynthesis_campaign(
                case_id="immutable-policy-case",
                target_name="ethanol",
                target_smiles="CCO",
                run_dir=tmp_path,
                repository_root=tmp_path,
                config=incompatible,
                runner=lambda _: (_ for _ in ()).throw(
                    AssertionError("closure policy mismatch must precede Agent work")
                ),
            )

    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture",
        "catalog_number": "NEW-1",
        "smiles": "CC=O",
        "checked_at": "2026-07-11T00:00:00Z",
        "available": True,
    }
    with pytest.raises(ValueError, match="campaign policy mismatch"):
        run_codex_retrosynthesis_campaign(
            case_id="immutable-policy-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(
                max_depth=1,
                max_expansions=1,
                stock_snapshots={"CC=O": snapshot},
            ),
            runner=lambda _: (_ for _ in ()).throw(
                AssertionError("stock authority change must fail before Agent work")
            ),
        )


def test_campaign_policy_binds_child_acceptance_mode_roles_and_quorum(
    tmp_path,
) -> None:
    strict = run_codex_retrosynthesis_campaign(
        case_id="child-policy-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            child_acceptance_mode="strict_all",
        ),
        runner=accepted_runner_record,
    )
    proposal_policy = strict["campaign"]["campaign_policy"][
        "proposal_agent_policy"
    ]
    assert proposal_policy["child_acceptance_mode"] == "strict_all"
    assert proposal_policy["child_roles"] == list(DEFAULT_CHILD_ROLES)
    assert proposal_policy["derived_valid_child_quorum"] == 2
    assert proposal_policy["child_acceptance_contract_version"].endswith(".v1")
    assert proposal_policy["coordinator_contract_version"].endswith(".v3")

    with pytest.raises(ValueError, match="campaign policy mismatch"):
        run_codex_retrosynthesis_campaign(
            case_id="child-policy-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(
                max_depth=1,
                max_expansions=1,
                child_acceptance_mode="valid_subset_l0",
            ),
            runner=accepted_runner_record,
        )


def test_partial_campaign_commits_only_capped_l0_consensus(tmp_path) -> None:
    report = run_codex_retrosynthesis_campaign(
        case_id="partial-commit-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            child_acceptance_mode="valid_subset_l0",
        ),
        runner=lambda task: partial_runner_record(
            task,
            valid_role_count=3,
            self_reported_evidence_level="validated",
        ),
    )

    assert report["campaign"]["accepted_expansion_count"] == 1
    assert report["route_expansion_count"] == 1
    expansion = report["route_consensus_expansions"][0]
    proposal = expansion["route_consensus"]["proposals"][0]
    assert proposal["validation_tier"] == "L0"
    assert proposal["achieved_proof_level"] == 0
    assert proposal["authority_evidence_level"] == "model_only"
    assert proposal["confidence"] == "low"
    assert proposal["no_solved_claim"] is True
    assert not any(
        key in proposal
        for key in ("accepted", "validated", "proof", "proof_level", "solved")
    )

    restarted = run_codex_retrosynthesis_campaign(
        case_id="partial-commit-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            child_acceptance_mode="valid_subset_l0",
        ),
        runner=lambda task: pytest.fail("durable capped commit must replay"),
    )
    assert restarted["campaign"]["accepted_expansion_count"] == 1


def test_campaign_authority_lock_prevents_concurrent_accepted_budget_overspend(
    tmp_path,
) -> None:
    calls = 0
    calls_lock = threading.Lock()

    def runner(task):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.15)
        return accepted_runner_record(task)

    config = RetrosynthesisTeamConfig(
        max_depth=1,
        max_expansions=1,
        max_attempt_runs=2,
        max_expansions_per_invocation=1,
        max_attempt_runs_per_invocation=1,
        campaign_authority_lock_timeout_s=5.0,
    )

    def invoke():
        return run_codex_retrosynthesis_campaign(
            case_id="concurrent-authority-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=config,
            runner=runner,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = [future.result() for future in [pool.submit(invoke), pool.submit(invoke)]]

    assert calls == 1
    assert all(
        report["campaign"]["accepted_expansion_count"] == 1
        for report in reports
    )
    commit_files = list(
        (tmp_path / "codex_retrosynthesis_team" / "campaign_commits").glob(
            "*.json"
        )
    )
    assert len(commit_files) == 1


def test_reconciliation_uses_same_configurable_campaign_authority_lock(
    tmp_path,
) -> None:
    runner_entered = threading.Event()
    release_runner = threading.Event()

    def runner(task):
        runner_entered.set()
        assert release_runner.wait(2.0)
        return accepted_runner_record(task)

    campaign_config = RetrosynthesisTeamConfig(
        max_depth=1,
        max_expansions=1,
        campaign_authority_lock_timeout_s=5.0,
    )

    def invoke_campaign():
        return run_codex_retrosynthesis_campaign(
            case_id="shared-authority-lock-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=campaign_config,
            runner=runner,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(invoke_campaign)
        assert runner_entered.wait(1.0)
        with pytest.raises(TimeoutError, match="campaign authority lock timeout"):
            reconcile_codex_campaign_proof_state(
                graph={
                    "schema_version": "route_consensus_graph.v1",
                    "case_id": "shared-authority-lock-case",
                    "target_smiles": "CCO",
                    "limits": {"max_depth": 1},
                },
                run_dir=tmp_path,
                case_id="shared-authority-lock-case",
                campaign_config=RetrosynthesisTeamConfig(
                    max_depth=1,
                    max_expansions=1,
                    campaign_authority_lock_timeout_s=0.05,
                ),
            )
        release_runner.set()
        report = future.result(timeout=3.0)

    assert report["campaign"]["accepted_expansion_count"] == 1


def test_campaign_budget_extensions_are_append_only_and_cannot_shrink(tmp_path) -> None:
    first = run_codex_retrosynthesis_campaign(
        case_id="monotonic-budget-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=2,
            max_expansions=1,
            max_attempt_runs=3,
        ),
        runner=accepted_runner_record,
    )
    second = run_codex_retrosynthesis_campaign(
        case_id="monotonic-budget-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=2,
            max_expansions=2,
            max_attempt_runs=6,
        ),
        runner=lambda _: (_ for _ in ()).throw(
            AssertionError("unverified child frontier must remain blocked")
        ),
    )
    event_root = tmp_path / "codex_retrosynthesis_team" / "campaign_budget_events"
    events = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(event_root.glob("*.json"))
    ]
    state = json.loads(
        (tmp_path / "codex_retrosynthesis_team" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )

    assert first["campaign"]["campaign_budget"]["event_count"] == 1
    assert second["campaign"]["campaign_budget"]["event_count"] == 2
    assert [row["event"] for row in events] == ["initialized", "monotonic_extension"]
    assert events[1]["previous_event_sha256"] == events[0]["content_sha256"]
    assert events[1]["max_expansions"] == 2
    assert events[1]["max_attempt_runs"] == 6
    assert state["campaign_budget_projection"]["max_expansions"] == 2
    assert state["campaign_budget_projection"]["event_count"] == 2

    with pytest.raises(ValueError, match="budgets cannot shrink"):
        run_codex_retrosynthesis_campaign(
            case_id="monotonic-budget-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(
                max_depth=2,
                max_expansions=1,
                max_attempt_runs=3,
            ),
            runner=lambda _: (_ for _ in ()).throw(
                AssertionError("budget shrink must fail before Agent work")
            ),
        )


def test_failed_attempt_does_not_consume_accepted_budget_and_pending_queue_resumes(
    tmp_path,
) -> None:
    calls = 0

    def runner(task):
        nonlocal calls
        calls += 1
        if calls == 1:
            rejected = accepted_runner_record_for_target(task, "CC=O")
            rejected.metadata = {"child_agents": []}
            return rejected
        context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
        target = context["target"]["smiles"]
        return accepted_runner_record_for_target(
            task,
            "CC=O" if target == "CCO" else "CC(O)O",
        )

    config = RetrosynthesisTeamConfig(
        max_depth=3,
        max_expansions=2,
        max_expansions_per_invocation=1,
        max_attempt_runs_per_invocation=2,
        frontier_retry_base_seconds=0.01,
        frontier_retry_max_seconds=0.01,
        frontier_retry_wait_seconds=0.5,
    )
    first = run_codex_retrosynthesis_campaign(
        case_id="resume-budget-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=runner,
    )

    assert calls == 2
    assert first["campaign"]["attempt_run_count"] == 2
    assert first["campaign"]["accepted_expansion_count"] == 1
    assert first["campaign"]["stop_reason"] == "invocation_accepted_expansion_cap_reached"
    pending = [
        row
        for row in first["campaign"]["frontier_queue"]["jobs"]
        if row["state"] == "pending"
    ]
    assert pending and pending[0]["frontier_smiles"] == "CC=O"
    root_step = first["route_consensus_graph"]["steps"][0]
    config.reaction_proofs = {
        root_step["step_id"]: materialized_reduction_candidate(root_step)
    }

    second = run_codex_retrosynthesis_campaign(
        case_id="resume-budget-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        blackboard_context={"current_belief": {"revision": 2}},
        literature_sources=[{"doi": "10.1000/new-evidence"}],
        runner=runner,
    )

    assert calls == 3
    # Agent-call accounting is restored from the immutable attempt ledger;
    # the rejected first call remains campaign-wide attempt budget.
    assert second["campaign"]["attempt_run_count"] == 3
    assert second["campaign"]["accepted_expansion_count"] == 2
    assert second["campaign"]["invocation_attempt_run_count"] == 1
    assert second["campaign"]["stop_reason"] == "campaign_accepted_expansion_budget_exhausted"
    child_context = next(
        Path(tmp_path / "codex_retrosynthesis_frontiers").glob(
            "d1-*/codex_retrosynthesis_team/context_snapshot.json"
        )
    )
    context_payload = json.loads(child_context.read_text(encoding="utf-8"))
    assert context_payload["blackboard"]["current_belief"]["revision"] == 2
    assert context_payload["literature_sources"][0]["doi"] == "10.1000/new-evidence"


def test_campaign_stock_audits_all_reachable_alternatives_when_routes_truncate(
    tmp_path,
) -> None:
    leaves = ["C" * length + "O" for length in range(3, 33)]

    def runner(task):
        context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
        target = str(context["target"]["smiles"])
        assert target == "CCO"
        precursors = leaves
        artifact = proposal_artifact(task.case_id)
        payload = artifact["payload"]
        payload["target_smiles"] = target
        base = dict(payload["candidates"][0])
        candidates = []
        for index, precursor in enumerate(precursors, start=1):
            candidate = dict(base)
            candidate["candidate_id"] = f"candidate:alternative:{index}"
            candidate["product_smiles"] = target
            candidate["precursor_smiles"] = [precursor]
            candidates.append(candidate)
        payload["candidates"] = candidates
        children = []
        for index, role in enumerate(task.child_roles):
            child_payload = dict(payload)
            child_payload["agent_role"] = role
            child_payload["candidates"] = (
                list(candidates[index * 15 : (index + 1) * 15])
                if index < 2
                else []
            )
            children.append(
                {
                    "agent_id": f"child-{index}",
                    "role": role,
                    "role_binding_method": "explicit_spawn_contract",
                    "wait_call_id": f"wait-{index}",
                    "status": "completed",
                    "arguments": {"role": role},
                    "message": json.dumps(child_payload, sort_keys=True),
                }
            )
        payload["candidates"] = list(candidates[:15])
        return WorkerRunRecord(
            run_id="team:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="accepted_draft",
            backend="codex_cli",
            output_artifact=artifact,
            output_validation={"accepted": True, "reasons": []},
            metadata={"session_id": "thread-1", "child_agents": children},
        )

    report = run_codex_retrosynthesis_campaign(
        case_id="all-reachable-alternatives",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=2,
            max_expansions=1,
        ),
        runner=runner,
    )

    graph = report["route_consensus_graph"]
    assert len(graph["route_hypotheses"]) == 24
    assert graph["truncation"]["route_hypotheses_truncated"] is True
    leaf_jobs = [
        row
        for row in report["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] in set(leaves)
    ]
    assert len(leaf_jobs) == 30
    assert {row["frontier_smiles"] for row in leaf_jobs} == set(leaves)
    assert all(row["metadata"]["stock_audit_preceded_agent_work"] is True for row in leaf_jobs)
    completeness = report["campaign"]["frontier_completeness"]
    assert completeness["terminal_count"] == 30
    assert completeness["complete"] is False


def test_depth_boundary_leaf_is_stock_audited_and_agent_success_is_not_proof(
    tmp_path,
) -> None:
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture",
        "catalog_number": "A-1",
        "smiles": "CC=O",
        "checked_at": "2026-07-11T00:00:00Z",
        "available": True,
    }
    report = run_codex_retrosynthesis_campaign(
        case_id="depth-stock-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
        ),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )

    leaf = next(
        row
        for row in report["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert leaf["state"] == "pending"
    assert leaf["closure_kind"] == ""
    assert leaf["metadata"]["stock_observation_current_closed"] is True
    assert leaf["metadata"]["stock_audit_preceded_agent_work"] is True
    assert report["campaign"]["frontier_completeness"]["stock_closed_count"] == 1
    assert report["campaign"]["accepted_expansion_count"] == 1
    assert report["campaign"]["team_accepted_attempt_count"] == 1
    assert report["campaign"]["proof_closed_attempt_count"] == 0
    assert report["campaign"]["graph_complete"] is False
    assert report["campaign"]["reaction_proof_state"]["summary"]["pending"] == 1


def test_empty_stock_configuration_records_explicit_fail_closed_reason(tmp_path) -> None:
    report = run_codex_retrosynthesis_campaign(
        case_id="empty-stock-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )

    assert report["campaign"]["stock_authority"]["available"] is False
    assert "no_trusted_stock_snapshots_configured" in report["campaign"]["stock_authority"]["reasons"]
    leaf = next(
        row
        for row in report["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert leaf["failure_reasons"] == []
    current_audit = leaf["metadata"]["stock_observations"]["current"][0][
        "provider_result"
    ]
    assert "no_trusted_stock_snapshots_configured" in current_audit["reasons"]


def test_campaign_can_use_explicit_hashed_benchmark_catalog(tmp_path) -> None:
    catalog = tmp_path / "paroutes-n1.csv"
    catalog.write_text("smiles\nCC=O\n", encoding="utf-8")
    digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    report = run_codex_retrosynthesis_campaign(
        case_id="benchmark-stock-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path / "run",
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            benchmark_stock_catalog_artifact=str(catalog),
            benchmark_stock_catalog_sha256=digest,
            benchmark_stock_catalog_name="PaRoutes_n1",
        ),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )

    authority = report["campaign"]["stock_authority"]
    assert authority["source"] == "hashed_benchmark_catalog"
    assert authority["commercial_orderability_claimed"] is False
    policy_binding = report["campaign"]["campaign_policy"][
        "stock_authority_binding"
    ]
    assert policy_binding["catalog_sha256"] == digest
    assert policy_binding["catalog_name"] == "PaRoutes_n1"
    leaf = next(
        row
        for row in report["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert leaf["state"] == "pending"
    assert leaf["closure_kind"] == ""
    assert leaf["achieved_proof_level"] == 0
    assert (
        leaf["metadata"]["stock_boundary_authority"]
        == "benchmark_membership_only"
    )
    assert leaf["metadata"]["stock_audit"]["payload"]["boundary_type"] == "benchmark_stock"


def test_proof_reconciliation_rehydrates_policy_bound_benchmark_provider(
    tmp_path,
) -> None:
    catalog = tmp_path / "rehydrated-stock.csv"
    catalog.write_text("smiles\nCC(O)O\n", encoding="utf-8")
    digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    run_dir = tmp_path / "run"
    first = run_codex_retrosynthesis_campaign(
        case_id="benchmark-rehydration-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=run_dir,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=3,
            max_expansions=1,
            benchmark_stock_catalog_artifact=str(catalog),
            benchmark_stock_catalog_sha256=digest,
            benchmark_stock_catalog_name="rehydration-fixture",
        ),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )
    base = first["route_consensus_graph"]
    nodes = {row["smiles"]: dict(row) for row in base["nodes"]}
    root = nodes["CCO"]
    middle = nodes["CC=O"]
    external_step_id = canonical_graph_step_id("CC=O", ["CC(O)O"])
    external_signature = canonical_graph_step_signature("CC=O", ["CC(O)O"])
    middle["expansion_status"] = "expanded"
    middle["outgoing_step_ids"] = [external_step_id]
    leaf = {
        "schema_version": "route_consensus_molecule.v1",
        "node_id": "rehydrated-leaf:CC(O)O",
        "smiles": "CC(O)O",
        "canonical_isomeric_smiles": "CC(O)O",
        "min_depth": 2,
        "expansion_status": "unexpanded",
        "outgoing_step_ids": [],
        "incoming_step_ids": [external_step_id],
    }
    fused = {
        **base,
        "nodes": [root, middle, leaf],
        "steps": [
            *base["steps"],
            {
                "schema_version": "route_consensus_step.v1",
                "step_id": external_step_id,
                "signature": external_signature,
                "product_node_id": middle["node_id"],
                "precursor_node_ids": [leaf["node_id"]],
                "product_smiles": "CC=O",
                "precursor_smiles": ["CC(O)O"],
                "rank_score": 0.9,
            },
        ],
        "route_hypotheses": [],
    }

    refreshed = reconcile_codex_campaign_proof_state(
        graph=fused,
        run_dir=run_dir,
        case_id="benchmark-rehydration-case",
        external_hyperedge_admission_receipts=(
            current_host_chemenzy_admission_receipts(
                tmp_path,
                case_id="benchmark-rehydration-case",
                product_smiles="CC=O",
                precursor_smiles=["CC(O)O"],
            )
        ),
    )

    stock_authority = refreshed["frontier_sync"]["stock_authority"]
    assert refreshed["frontier_sync"]["enabled"] is True
    assert stock_authority["source"] == "immutable_campaign_policy_rehydration"
    assert stock_authority["available"] is True
    audited_leaf = next(
        row
        for row in refreshed["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC(O)O"
    )
    assert audited_leaf["metadata"]["stock_observation_current_closed"] is True
    assert refreshed["frontier_ledger_authoritative"] is True

    # The immutable digest is replayed at construction time. A catalog edited
    # after the campaign fails before proof or queue state can be mutated.
    from cascade_planner.application.frontier_scheduler import PersistentFrontierQueue

    queue = PersistentFrontierQueue(run_dir / "codex_retrosynthesis_team" / "frontier_queue")
    queue_before = queue.snapshot("benchmark-rehydration-case")
    proof_path = run_dir / "codex_retrosynthesis_team" / "reaction_proof_state.json"
    proof_bytes_before = proof_path.read_bytes()
    catalog.write_text("smiles\nC\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="campaign policy stock provider rehydration failed:.*SHA-256 mismatch",
    ):
        reconcile_codex_campaign_proof_state(
            graph=fused,
            run_dir=run_dir,
            case_id="benchmark-rehydration-case",
        )
    assert queue.snapshot("benchmark-rehydration-case") == queue_before
    assert proof_path.read_bytes() == proof_bytes_before


def test_campaign_runs_benchmark_and_commercial_stock_providers_together(
    tmp_path,
) -> None:
    catalog = tmp_path / "provider-set.csv"
    catalog.write_text("smiles\nCC=O\n", encoding="utf-8")
    digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture",
        "catalog_number": "BOTH-1",
        "smiles": "CC=O",
        "checked_at": "2026-07-11T00:00:00Z",
        "available": True,
    }
    report = run_codex_retrosynthesis_campaign(
        case_id="provider-set-stock-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path / "run",
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
            benchmark_stock_catalog_artifact=str(catalog),
            benchmark_stock_catalog_sha256=digest,
            benchmark_stock_catalog_name="provider-set",
        ),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )

    leaf = next(
        row
        for row in report["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    current = leaf["metadata"]["stock_observations"]["current"]
    assert len(current) == 2
    assert {
        row["provider_result"]["payload"]["boundary_type"] for row in current
    } == {"benchmark_stock", "commercially_orderable"}
    assert report["campaign"]["stock_authority"]["source"] == (
        "benchmark_and_commercial_provider_set"
    )
    policy = report["campaign"]["campaign_policy"]["stock_authority_binding"]
    assert policy["provider_set_binding"]["schema_version"] == (
        "stock_provider_set_binding.v1"
    )
    stock = report["campaign"]["frontier_ledger"]["molecules"]["CC=O"][
        "stock"
    ]
    assert stock["benchmark_membership_closed"] is True
    assert stock["procurement_boundary_closed"] is True


def test_validated_reaction_proof_is_consumed_instead_of_left_open(tmp_path) -> None:
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture",
        "catalog_number": "A-1",
        "smiles": "CC=O",
        "checked_at": "2026-07-11T00:00:00Z",
        "available": True,
    }
    base_config = RetrosynthesisTeamConfig(
        max_depth=1,
        max_expansions=1,
        stock_snapshots={"CC=O": snapshot},
    )
    first = run_codex_retrosynthesis_campaign(
        case_id="proof-consumer-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=base_config,
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )
    step = first["route_consensus_graph"]["steps"][0]
    candidate = materialized_reduction_candidate(step)

    def must_not_run(_):
        raise AssertionError("proof reconciliation must not rerun Codex")

    second = run_codex_retrosynthesis_campaign(
        case_id="proof-consumer-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
            reaction_proofs={step["step_id"]: candidate},
        ),
        runner=must_not_run,
    )

    assert second["campaign"]["reaction_proof_state"]["summary"]["validated"] == 1
    record = second["campaign"]["reaction_proof_state"]["records"][0]
    assert record["proof_authority"] == "current_host_verifier_replay"
    assert record["proof"]["proof_level"] == "L2_reaction_validated"
    assert second["campaign"]["frontier_completeness"]["unresolved_frontiers"] == []
    assert second["campaign"]["graph_complete"] is True
    assert second["campaign"]["stop_reason"] == "graph_proof_complete"


def test_l2_transform_reapplied_edge_verification_report_is_consumed(tmp_path) -> None:
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture",
        "catalog_number": "A-1",
        "smiles": "CC=O",
        "checked_at": "2026-07-11T00:00:00Z",
        "available": True,
    }
    first = run_codex_retrosynthesis_campaign(
        case_id="edge-report-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
        ),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )
    step = first["route_consensus_graph"]["steps"][0]
    stale_invalid = validated_step_proof(
        step,
        proof_level="L2_reaction_validated",
        deterministic_transform=False,
        trusted_precedent=False,
    )
    rejected = run_codex_retrosynthesis_campaign(
        case_id="edge-report-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
            reaction_proofs={step["step_id"]: stale_invalid},
        ),
        runner=lambda _: (_ for _ in ()).throw(
            AssertionError("proof reconciliation must not rerun Codex")
        ),
    )
    rejected_record = rejected["campaign"]["reaction_proof_state"]["records"][0]
    assert rejected_record["status"] == "rejected"
    assert "materialized_reaction_candidate_missing" in rejected_record["validation_reasons"]
    edge_report = verify_codex_consensus_graph(
        first["route_consensus_graph"],
        stock_closed_smiles=["CC=O"],
        atom_mapper=lambda reactions: [
            "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
            for _ in reactions
        ],
    )
    tampered_report = json.loads(json.dumps(edge_report))
    tampered_report["edge_verifications"][0]["materialized_candidate"][
        "atom_mapped_reaction_smiles"
    ] = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    tampered_report = rehash_content_sha256(tampered_report)
    tampered = run_codex_retrosynthesis_campaign(
        case_id="edge-report-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
            reaction_proof_reports=[tampered_report],
        ),
        runner=lambda _: (_ for _ in ()).throw(
            AssertionError("proof reconciliation must not rerun Codex")
        ),
    )
    tampered_record = tampered["campaign"]["reaction_proof_state"]["records"][0]
    assert tampered_record["status"] == "rejected"
    assert tampered_record["proof_authority"] == "none"
    forged_boolean_report = json.loads(json.dumps(edge_report))
    forged_boolean_report["edge_verifications"][0]["step_proof"] = validated_step_proof(
        step,
        proof_level="L2_reaction_validated",
        deterministic_transform=True,
        trusted_precedent=False,
    )
    forged_boolean_report = rehash_content_sha256(forged_boolean_report)
    forged = run_codex_retrosynthesis_campaign(
        case_id="edge-report-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
            reaction_proof_reports=[forged_boolean_report],
        ),
        runner=lambda _: (_ for _ in ()).throw(
            AssertionError("proof reconciliation must not rerun Codex")
        ),
    )
    forged_record = forged["campaign"]["reaction_proof_state"]["records"][0]
    assert forged_record["status"] == "rejected"
    assert any(
        "not_equal_to_current_host_replay" in reason
        for audit in forged_record["replay_options"]
        for reason in audit["reasons"]
    )

    second = run_codex_retrosynthesis_campaign(
        case_id="edge-report-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
            reaction_proof_reports=[edge_report],
        ),
        runner=lambda _: (_ for _ in ()).throw(
            AssertionError("proof reconciliation must not rerun Codex")
        ),
    )

    record = second["campaign"]["reaction_proof_state"]["records"][0]
    assert record["status"] == "validated"
    assert record["proof"]["proof_level"] == "L2_reaction_validated"
    assert second["campaign"]["graph_complete"] is True


def test_public_proof_reconciliation_closes_without_proposal_budget(tmp_path) -> None:
    snapshot = {
        "schema_version": "stock_offer_snapshot.v1",
        "supplier": "fixture",
        "catalog_number": "A-1",
        "smiles": "CC=O",
        "checked_at": "2026-07-11T00:00:00Z",
        "available": True,
    }
    first = run_codex_retrosynthesis_campaign(
        case_id="proof-refresh-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
        ),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )
    graph = first["route_consensus_graph"]
    edge_report = verify_codex_consensus_graph(
        graph,
        stock_closed_smiles=["CC=O"],
        atom_mapper=lambda reactions: [
            "[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]"
            for _ in reactions
        ],
    )
    stock_evidence = [
        row["metadata"]["stock_audit"]
        for row in first["campaign"]["frontier_queue"]["jobs"]
        if row["metadata"].get("stock_observation_current_closed") is True
    ]

    refreshed = reconcile_codex_campaign_proof_state(
        graph=graph,
        run_dir=tmp_path,
        case_id="proof-refresh-case",
        reaction_proof_reports=[edge_report],
        stock_evidence=stock_evidence,
        campaign_config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            stock_snapshots={"CC=O": snapshot},
        ),
    )

    assert refreshed["reaction_proof_state"]["summary"]["validated"] == 1
    assert refreshed["stock_evidence_replay"]["accepted_count"] == 1
    assert refreshed["graph_complete"] is True
    assert refreshed["frontier_ledger_authoritative"] is True
    assert refreshed["proposal_runner_invoked"] is False
    assert refreshed["expansion_budget_consumed"] == 0


def test_public_reconciliation_syncs_new_fused_graph_leaf_into_campaign_queue(
    tmp_path,
) -> None:
    first = run_codex_retrosynthesis_campaign(
        case_id="fused-frontier-sync-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=3, max_expansions=1),
        runner=lambda task: accepted_runner_record_for_target(task, "CC=O"),
    )
    base = first["route_consensus_graph"]
    nodes = {row["smiles"]: dict(row) for row in base["nodes"]}
    root = nodes["CCO"]
    middle = nodes["CC=O"]
    external_step_id = canonical_graph_step_id("CC=O", ["CC(O)O"])
    external_signature = canonical_graph_step_signature("CC=O", ["CC(O)O"])
    middle["expansion_status"] = "expanded"
    middle["outgoing_step_ids"] = [external_step_id]
    leaf = {
        "schema_version": "route_consensus_molecule.v1",
        "node_id": "external-leaf:CC(O)O",
        "smiles": "CC(O)O",
        "canonical_isomeric_smiles": "CC(O)O",
        "min_depth": 2,
        "expansion_status": "unexpanded",
        "outgoing_step_ids": [],
        "incoming_step_ids": [external_step_id],
    }
    fused = {
        **base,
        "nodes": [root, middle, leaf],
        "steps": [
            *base["steps"],
            {
                "schema_version": "route_consensus_step.v1",
                "step_id": external_step_id,
                "signature": external_signature,
                "product_node_id": middle["node_id"],
                "precursor_node_ids": [leaf["node_id"]],
                "product_smiles": "CC=O",
                "precursor_smiles": ["CC(O)O"],
                "rank_score": 0.9,
            },
        ],
        # Deliberately empty: scheduler authority must not depend on bounded
        # route enumeration/presentation rows.
        "route_hypotheses": [],
    }

    refreshed = reconcile_codex_campaign_proof_state(
        graph=fused,
        run_dir=tmp_path,
        case_id="fused-frontier-sync-case",
        campaign_config=RetrosynthesisTeamConfig(
            max_depth=3,
            max_expansions=3,
        ),
        external_hyperedge_admission_receipts=(
            current_host_chemenzy_admission_receipts(
                tmp_path,
                case_id="fused-frontier-sync-case",
                product_smiles="CC=O",
                precursor_smiles=["CC(O)O"],
            )
        ),
    )

    jobs = refreshed["frontier_queue"]["jobs"]
    leaf_job = next(row for row in jobs if row["frontier_smiles"] == "CC(O)O")
    assert leaf_job["state"] == "pending"
    assert refreshed["frontier_sync"]["added_job_count"] == 1
    assert refreshed["proposal_runner_invoked"] is False
    assert refreshed["expansion_budget_consumed"] == 0


def test_external_fused_edge_survives_campaign_restart(tmp_path) -> None:
    calls: list[str] = []

    def root_runner(task):
        context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
        calls.append(context["target"]["smiles"])
        return accepted_runner_record_for_target(task, "CCC=O")

    config = RetrosynthesisTeamConfig(
        max_depth=3,
        max_expansions=1,
        max_expansions_per_invocation=1,
        max_attempt_runs_per_invocation=1,
    )
    first = run_codex_retrosynthesis_campaign(
        case_id="durable-external-edge-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=root_runner,
    )
    base = first["route_consensus_graph"]
    nodes = {row["smiles"]: dict(row) for row in base["nodes"]}
    root = nodes["CCO"]
    middle = nodes["CCC=O"]
    external_step_id = canonical_graph_step_id("CCC=O", ["CCCO"])
    external_signature = canonical_graph_step_signature("CCC=O", ["CCCO"])
    middle["expansion_status"] = "expanded"
    middle["outgoing_step_ids"] = [external_step_id]
    leaf = {
        "schema_version": "route_consensus_molecule.v1",
        "node_id": "external-leaf:CCCO",
        "smiles": "CCCO",
        "canonical_isomeric_smiles": "CCCO",
        "min_depth": 2,
        "expansion_status": "unexpanded",
        "outgoing_step_ids": [],
        "incoming_step_ids": [external_step_id],
    }
    external_step = {
        "schema_version": "route_consensus_step.v1",
        "step_id": external_step_id,
        "signature": external_signature,
        "product_node_id": middle["node_id"],
        "precursor_node_ids": [leaf["node_id"]],
        "product_smiles": "CCC=O",
        "precursor_smiles": ["CCCO"],
        "reaction_family": "carbonyl alcohol redox",
        "source_channels": ["literature_exact"],
        "source_refs": ["fixture:external-fused-edge"],
        "rank_score": 0.9,
    }
    fused = {
        **base,
        "nodes": [root, middle, leaf],
        "steps": [*base["steps"], external_step],
        "route_hypotheses": [],
    }
    materialized_external = {
        "schema_version": "materialized_reaction_candidate.v1",
        "step_id": external_step_id,
        "product_smiles": "CCC=O",
        "reactant_smiles": ["CCCO"],
        "reaction_smiles": "CCCO>>CCC=O",
        "atom_mapped_reaction_smiles": (
            "[CH3:1][CH2:2][CH2:3][OH:4]>>"
            "[CH3:1][CH2:2][CH:3]=[O:4]"
        ),
        "mapping_source": "test_materialized_candidate",
    }
    stock_path = tmp_path / "external-chemenzy-stock.csv"
    stock_path.write_text("smiles\nCCCO\n", encoding="utf-8")
    stock_sha256 = hashlib.sha256(stock_path.read_bytes()).hexdigest()
    chemenzy_report = verify_chemenzy_raw_routes(
        {
            "target": "CCC=O",
            "routes": [
                {
                    "route_rank": 0,
                    "metrics": {
                        "terminal_reactants": ["CCCO"],
                        "terminal_stock_status": {"CCCO": True},
                    },
                    "steps": [
                        {
                            "index": 0,
                            "product": "CCC=O",
                            "reactant_smiles": ["CCCO"],
                            "stock_status": {"CCCO": True},
                        }
                    ],
                }
            ],
            "stock_catalog_context": {
                "effective_stock_names": ["external-test-stock"],
                "catalog_bindings": [
                    {
                        "name": "external-test-stock",
                        "path": str(stock_path),
                        "sha256": stock_sha256,
                    }
                ],
            },
        },
        target_smiles="CCC=O",
        case_id="durable-external-edge-case",
    )
    assert chemenzy_report["accepted"] is True
    bank = chemenzy_report["route_proof_bank"]
    entry_id = bank["entries"][0]["proof_id"]
    admission_material = make_chemenzy_admission_material(
        bank,
        source_entry_id=entry_id,
        source_step_index=0,
        artifact_ref="fixture:current-host-chemenzy-bank",
    )
    admission_receipts = {
        exact_edge_signature("CCC=O", ["CCCO"]): [admission_material]
    }

    reconciled = reconcile_codex_campaign_proof_state(
        graph=fused,
        run_dir=tmp_path,
        case_id="durable-external-edge-case",
        reaction_proofs={external_step_id: materialized_external},
        campaign_config=config,
        external_hyperedge_admission_receipts=admission_receipts,
    )

    assert reconciled["admitted_hyperedge_journal"]["new_event_count"] == 1
    admitted_leaf = next(
        row
        for row in reconciled["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CCCO"
    )
    assert admitted_leaf["metadata"]["proposal_expansion_allowed"] is True
    assert external_step_id in admitted_leaf["metadata"]["parent_step_ids"]

    omitted = reconcile_codex_campaign_proof_state(
        graph=base,
        run_dir=tmp_path,
        case_id="durable-external-edge-case",
        campaign_config=config,
    )
    assert omitted["admitted_hyperedge_journal"]["new_event_count"] == 0
    assert external_step_id in {
        row["step_id"]
        for row in omitted["canonical_route_consensus_graph"]["steps"]
    }
    omitted_leaf = next(
        row
        for row in omitted["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CCCO"
    )
    assert omitted_leaf["metadata"]["proposal_expansion_allowed"] is True
    assert external_step_id in omitted_leaf["metadata"]["parent_step_ids"]

    def must_not_run(_):
        raise AssertionError("durable external edge replay must not rerun Codex")

    restarted = run_codex_retrosynthesis_campaign(
        case_id="durable-external-edge-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=must_not_run,
    )

    assert calls == ["CCO"]
    assert restarted["campaign"]["accepted_expansion_count"] == 1
    assert restarted["campaign"]["admitted_hyperedge_event_count"] == 1
    replayed_step = next(
        row
        for row in restarted["route_consensus_graph"]["steps"]
        if row["step_id"] == external_step_id
    )
    assert replayed_step["product_smiles"] == "CCC=O"
    assert replayed_step["precursor_smiles"] == ["CCCO"]
    replayed_leaf = next(
        row
        for row in restarted["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CCCO"
    )
    assert replayed_leaf["metadata"]["proposal_expansion_allowed"] is True
    assert external_step_id in replayed_leaf["metadata"]["parent_step_ids"]


def test_non_root_frontier_waits_for_host_replayed_l2_parent_proof(tmp_path) -> None:
    calls = 0

    def runner(task):
        nonlocal calls
        calls += 1
        context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
        target = context["target"]["smiles"]
        return accepted_runner_record_for_target(
            task,
            "CC=O" if target == "CCO" else "CC(O)O",
        )

    config = RetrosynthesisTeamConfig(
        max_depth=3,
        max_expansions=2,
        max_expansions_per_invocation=2,
        max_attempt_runs_per_invocation=2,
    )
    first = run_codex_retrosynthesis_campaign(
        case_id="evidence-first-frontier-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=runner,
    )

    leaf = next(
        row
        for row in first["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert calls == 1
    assert leaf["state"] == "pending"
    assert leaf["metadata"]["proposal_expansion_allowed"] is False
    assert (
        leaf["metadata"]["proposal_expansion_gate"]["status"]
        == "blocked_pending_current_host_l2_parent_proof"
    )

    root_step = first["route_consensus_graph"]["steps"][0]
    refreshed = reconcile_codex_campaign_proof_state(
        graph=first["route_consensus_graph"],
        run_dir=tmp_path,
        case_id="evidence-first-frontier-case",
        reaction_proofs={
            root_step["step_id"]: materialized_reduction_candidate(root_step)
        },
    )
    enabled_leaf = next(
        row
        for row in refreshed["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert refreshed["frontier_sync"]["enabled_job_count"] == 1
    assert enabled_leaf["state"] == "pending"
    assert enabled_leaf["metadata"]["proposal_expansion_allowed"] is True
    assert "enabled_by_current_host_l2_parent_proof" in (
        enabled_leaf["metadata"]["proposal_expansion_gate"]["status"]
    )

    second = run_codex_retrosynthesis_campaign(
        case_id="evidence-first-frontier-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=runner,
    )
    assert calls == 2
    assert second["campaign"]["accepted_expansion_count"] == 2


def test_reconciliation_merges_new_parent_before_replaying_succeeded_gate(
    tmp_path,
) -> None:
    def runner(task):
        context = json.loads(Path(task.input_refs[0]).read_text(encoding="utf-8"))
        target = context["target"]["smiles"]
        return accepted_runner_record_for_target(
            task,
            "CC=O" if target == "CCO" else "CC(O)O",
        )

    config = RetrosynthesisTeamConfig(
        max_depth=3,
        max_expansions=2,
        max_expansions_per_invocation=1,
        max_attempt_runs_per_invocation=1,
    )
    first = run_codex_retrosynthesis_campaign(
        case_id="late-parent-binding-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=runner,
    )
    original_graph = first["route_consensus_graph"]
    original_step = dict(original_graph["steps"][0])
    reconcile_codex_campaign_proof_state(
        graph=original_graph,
        run_dir=tmp_path,
        case_id="late-parent-binding-case",
        reaction_proofs={
            original_step["step_id"]: materialized_reduction_candidate(original_step)
        },
    )
    second = run_codex_retrosynthesis_campaign(
        case_id="late-parent-binding-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=config,
        runner=runner,
    )
    expanded_job = next(
        row
        for row in second["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert expanded_job["state"] == "succeeded"
    assert expanded_job["metadata"]["parent_step_ids"] == [original_step["step_id"]]

    # A later fused projection retains the molecule but binds it to a newly
    # discovered inbound edge.  It can temporarily omit the already committed
    # child expansion, exactly as a controller rebuild did in the v5 campaign.
    replacement_step = {**original_step, "step_id": "late-inbound-root-step"}
    fused_nodes = []
    for raw_node in original_graph["nodes"]:
        node = dict(raw_node)
        if original_step["step_id"] in node.get("incoming_step_ids", []):
            node["incoming_step_ids"] = [replacement_step["step_id"]]
        if original_step["step_id"] in node.get("outgoing_step_ids", []):
            node["outgoing_step_ids"] = [replacement_step["step_id"]]
        fused_nodes.append(node)
    fused_graph = {
        **original_graph,
        "nodes": fused_nodes,
        "steps": [replacement_step],
        "route_hypotheses": [],
    }

    refreshed = reconcile_codex_campaign_proof_state(
        graph=fused_graph,
        run_dir=tmp_path,
        case_id="late-parent-binding-case",
        reaction_proofs={
            replacement_step["step_id"]: materialized_reduction_candidate(
                replacement_step
            )
        },
    )

    rebound_job = next(
        row
        for row in refreshed["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert rebound_job["state"] == "succeeded"
    # Caller-controlled display ids cannot create a second parent for the same
    # exact chemistry.  The canonical durable union keeps the queue-fenced
    # Codex step id and treats the caller projection as advisory only.
    assert rebound_job["metadata"]["parent_step_ids"] == [
        original_step["step_id"]
    ]
    assert replacement_step["step_id"] not in {
        row["step_id"]
        for row in refreshed["canonical_route_consensus_graph"]["steps"]
    }
    assert refreshed["accepted"] is True
    assert refreshed["proposal_runner_invoked"] is False
    assert refreshed["expansion_budget_consumed"] == 0


def test_inline_context_survives_missing_file_access_and_ancestor_cycle_is_rejected(
    tmp_path,
) -> None:
    task = build_retrosynthesis_coordinator_task(
        case_id="inline",
        target_name="acetaldehyde",
        target_smiles="CC=O",
        context_ref=str(tmp_path / "unreadable.json"),
        allowed_workdir=tmp_path,
        context_snapshot={
            "target": {"name": "acetaldehyde", "smiles": "CC=O"},
            "blackboard": {
                "frontier_request": {
                    "target_smiles": "CC=O",
                    "ancestor_smiles": ["CCO"],
                    "forbidden_return_smiles": ["CCO", "CC=O"],
                }
            },
            "literature_sources": [{"doi": "10.1000/inline"}],
        },
    )
    assert '"ancestor_target_smiles": [' in task.objective
    assert '"CCO"' in task.objective
    assert '"10.1000/inline"' in task.objective

    config = RetrosynthesisTeamConfig(
        max_depth=3,
        max_expansions=2,
        max_expansions_per_invocation=1,
        max_attempt_runs_per_invocation=1,
    )
    first = run_codex_retrosynthesis_campaign(
        case_id="cycle-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path / "cycle",
        repository_root=tmp_path,
        config=config,
        runner=lambda child_task: accepted_runner_record_for_target(child_task, "CC=O"),
    )
    assert first["route_expansion_count"] == 1
    root_step = first["route_consensus_graph"]["steps"][0]
    config.reaction_proofs = {
        root_step["step_id"]: materialized_reduction_candidate(root_step)
    }

    second = run_codex_retrosynthesis_campaign(
        case_id="cycle-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path / "cycle",
        repository_root=tmp_path,
        config=config,
        runner=lambda child_task: accepted_runner_record_for_target(child_task, "CCO"),
    )
    assert second["route_expansion_count"] == 1
    latest = second["campaign"]["runs"][-1]
    assert latest["team_report_accepted"] is False
    assert any(reason.startswith("proposal_cycle:ancestor_or_target_return") for reason in latest["reasons"])


def test_structurally_impossible_codex_proposal_cannot_consume_accepted_budget(
    tmp_path,
) -> None:
    calls = 0

    def runner(task):
        nonlocal calls
        calls += 1
        return accepted_runner_record_for_target(task, "C")

    report = run_codex_retrosynthesis_campaign(
        case_id="admission-gate-case",
        target_name="large-chain",
        target_smiles="CCCCCCCCCCCCCCCCCC",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=2,
            max_expansions=1,
            max_attempt_runs=1,
            max_expansions_per_invocation=1,
            max_attempt_runs_per_invocation=1,
        ),
        runner=runner,
    )

    assert calls == 1
    assert report["campaign"]["attempt_run_count"] == 1
    assert report["campaign"]["accepted_expansion_count"] == 0
    assert report["route_expansion_count"] == 0
    latest = report["campaign"]["runs"][-1]
    assert latest["team_report_accepted"] is False
    serialized = json.dumps(latest, sort_keys=True)
    assert "element_inventory_not_conserved" in serialized
    assert "large_atom_jump" in serialized


def test_campaign_recovers_succeeded_expansion_from_fenced_commit(tmp_path) -> None:
    first = run_codex_retrosynthesis_campaign(
        case_id="recoverable-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )
    state_path = tmp_path / "codex_retrosynthesis_team" / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    commit_ref = first["campaign"]["frontier_queue"]["jobs"][0]["result_ref"]

    assert state["content_sha256"]
    assert first["route_expansion_count"] == 1
    assert "campaign_commits" in commit_ref
    state_path.unlink()

    def must_not_run(_):
        raise AssertionError("durable expansion should be recovered without rerunning Codex")

    recovered = run_codex_retrosynthesis_campaign(
        case_id="recoverable-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=must_not_run,
    )

    assert recovered["route_expansion_count"] == 1
    assert recovered["campaign"]["recovery_errors"] == []
    assert recovered["campaign"]["runs"][0]["recovered_from_expansion_commit"] is True


def test_campaign_requeues_tampered_expansion_commit(tmp_path) -> None:
    first = run_codex_retrosynthesis_campaign(
        case_id="tampered-commit-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )
    state_path = tmp_path / "codex_retrosynthesis_team" / "campaign_state.json"
    commit_path = next(
        path
        for path in (tmp_path / "codex_retrosynthesis_team" / "campaign_commits").glob("*.json")
    )
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["expansion_sha256"] = "0" * 64
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    state_path.unlink()
    calls = 0

    def rejected_runner(task):
        nonlocal calls
        calls += 1
        record = accepted_runner_record(task)
        record.metadata = {"child_agents": []}
        return record

    recovered = run_codex_retrosynthesis_campaign(
        case_id="tampered-commit-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=rejected_runner,
    )

    assert first["route_expansion_count"] == 1
    assert calls == 1
    assert recovered["route_expansion_count"] == 0
    assert any("digest_invalid" in reason for reason in recovered["campaign"]["recovery_errors"])


def test_campaign_target_identity_is_immutable_for_same_run_dir_and_case(tmp_path) -> None:
    run_codex_retrosynthesis_campaign(
        case_id="identity-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )

    with pytest.raises(ValueError, match="already bound"):
        run_codex_retrosynthesis_campaign(
            case_id="identity-case",
            target_name="ethylamine",
            target_smiles="CCN",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
            runner=lambda _: (_ for _ in ()).throw(
                AssertionError("identity mismatch must fail before Codex")
            ),
        )


def test_campaign_rejects_rehashed_root_report_identity_tamper(tmp_path) -> None:
    run_codex_retrosynthesis_campaign(
        case_id="root-report-identity-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )
    report_path = tmp_path / "codex_retrosynthesis_team" / "team_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["target_smiles"] = "CCN"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="root team report identity mismatch"):
        run_codex_retrosynthesis_campaign(
            case_id="root-report-identity-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
            runner=lambda _: (_ for _ in ()).throw(
                AssertionError("report identity mismatch must fail before Codex")
            ),
        )


def test_campaign_rejects_rehashed_queue_identity_tamper(tmp_path) -> None:
    run_codex_retrosynthesis_campaign(
        case_id="queue-identity-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )
    queue_path = next(
        (tmp_path / "codex_retrosynthesis_team" / "frontier_queue").glob(
            "frontiers-*.json"
        )
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["jobs"][0]["metadata"]["campaign_root_smiles"] = "CCN"
    queue = rehash_content_sha256(queue)
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    with pytest.raises(ValueError, match="queue identity fence mismatch"):
        run_codex_retrosynthesis_campaign(
            case_id="queue-identity-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
            runner=lambda _: (_ for _ in ()).throw(
                AssertionError("queue identity mismatch must fail before Codex")
            ),
        )


def test_rehashed_queue_cannot_self_enable_unproven_child_frontier(tmp_path) -> None:
    first = run_codex_retrosynthesis_campaign(
        case_id="queue-proof-gate-tamper-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=3, max_expansions=2),
        runner=accepted_runner_record,
    )
    child = next(
        row
        for row in first["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CC=O"
    )
    assert child["metadata"]["proposal_expansion_allowed"] is False
    queue_path = next(
        (tmp_path / "codex_retrosynthesis_team" / "frontier_queue").glob(
            "frontiers-*.json"
        )
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    tampered_child = next(
        row for row in queue["jobs"] if row["frontier_smiles"] == "CC=O"
    )
    tampered_child["metadata"]["proposal_expansion_allowed"] = True
    queue = rehash_content_sha256(queue)
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks current host L2 parent proof"):
        run_codex_retrosynthesis_campaign(
            case_id="queue-proof-gate-tamper-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(max_depth=3, max_expansions=2),
            runner=lambda _: (_ for _ in ()).throw(
                AssertionError("forged proof gate must fail before Agent work")
            ),
        )


def test_rehashed_invalid_commit_cannot_leave_cached_budget_consumption(tmp_path) -> None:
    first = run_codex_retrosynthesis_campaign(
        case_id="self-consistent-commit-tamper-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )
    output_dir = tmp_path / "codex_retrosynthesis_team"
    state_before = json.loads((output_dir / "campaign_state.json").read_text(encoding="utf-8"))
    assert len(state_before["expansions"]) == 1
    commit_path = next((output_dir / "campaign_commits").glob("*.json"))
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["campaign_identity_sha256"] = "0" * 64
    commit = rehash_content_sha256(commit)
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    calls = 0

    def rejected_runner(task):
        nonlocal calls
        calls += 1
        record = accepted_runner_record(task)
        record.metadata = {"child_agents": []}
        return record

    recovered = run_codex_retrosynthesis_campaign(
        case_id="self-consistent-commit-tamper-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=rejected_runner,
    )

    assert first["route_expansion_count"] == 1
    assert calls == 1
    assert recovered["campaign"]["accepted_expansion_count"] == 0
    assert recovered["route_expansion_count"] == 0
    assert any(
        "identity_or_digest_invalid" in reason
        for reason in recovered["campaign"]["recovery_errors"]
    )


def test_campaign_renews_lease_during_slow_direct_agent_team(tmp_path, monkeypatch) -> None:
    from cascade_planner.application.frontier_scheduler import PersistentFrontierQueue

    original = PersistentFrontierQueue.heartbeat
    heartbeat_calls = 0

    def observed_heartbeat(self, *args, **kwargs):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PersistentFrontierQueue, "heartbeat", observed_heartbeat)

    def slow_runner(task):
        time.sleep(0.06)
        return accepted_runner_record(task)

    report = run_codex_retrosynthesis_campaign(
        case_id="heartbeat-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(
            max_depth=1,
            max_expansions=1,
            frontier_heartbeat_interval_seconds=0.01,
        ),
        runner=slow_runner,
    )

    assert heartbeat_calls >= 1
    assert report["route_expansion_count"] == 1


def test_parent_completion_failure_never_publishes_child_frontiers(
    tmp_path,
    monkeypatch,
) -> None:
    from cascade_planner.application.frontier_scheduler import (
        FrontierLeaseError,
        PersistentFrontierQueue,
    )

    def reject_completion(self, *args, **kwargs):
        raise FrontierLeaseError("injected fencing loss")

    monkeypatch.setattr(PersistentFrontierQueue, "complete", reject_completion)
    report = run_codex_retrosynthesis_campaign(
        case_id="parent-fencing-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=2, max_expansions=1),
        runner=accepted_runner_record,
    )

    jobs = report["campaign"]["frontier_queue"]["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["frontier_smiles"] == "CCO"
    assert report["route_expansion_count"] == 0
    assert report["campaign"]["runs"][0]["result_quarantined"] is True


def test_prepared_commit_is_adopted_after_crash_without_duplicate_agent_call(
    tmp_path,
    monkeypatch,
) -> None:
    from cascade_planner.application.frontier_scheduler import PersistentFrontierQueue

    original_complete = PersistentFrontierQueue.complete
    injected = False
    calls = 0

    def crash_between_commit_and_complete(self, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            raise KeyboardInterrupt("injected process crash after immutable commit")
        return original_complete(self, *args, **kwargs)

    def observed_runner(task):
        nonlocal calls
        calls += 1
        return accepted_runner_record(task)

    monkeypatch.setattr(
        PersistentFrontierQueue,
        "complete",
        crash_between_commit_and_complete,
    )
    with pytest.raises(KeyboardInterrupt, match="injected process crash"):
        run_codex_retrosynthesis_campaign(
            case_id="prepared-outbox-recovery-case",
            target_name="ethanol",
            target_smiles="CCO",
            run_dir=tmp_path,
            repository_root=tmp_path,
            config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
            runner=observed_runner,
        )
    attempt_root = tmp_path / "codex_retrosynthesis_team" / "campaign_attempts"
    assert len(list(attempt_root.glob("*/started.json"))) == 1
    assert list(attempt_root.glob("*/terminal.json")) == []
    monkeypatch.setattr(PersistentFrontierQueue, "complete", original_complete)
    second = run_codex_retrosynthesis_campaign(
        case_id="prepared-outbox-recovery-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=lambda _: (_ for _ in ()).throw(
            AssertionError("prepared immutable commit must suppress duplicate Agent work")
        ),
    )

    root_job = next(
        row
        for row in second["campaign"]["frontier_queue"]["jobs"]
        if row["frontier_smiles"] == "CCO"
    )
    assert calls == 1
    assert second["route_expansion_count"] == 1
    assert second["campaign"]["attempt_run_count"] == 1
    assert root_job["state"] == "succeeded"
    assert root_job["closure_kind"] == "proposal_expansion"
    assert root_job["metadata"]["prepared_result_recovery"]["adopted"] is True
    assert second["campaign"]["runs"][0]["recovered_from_expansion_commit"] is True


def test_legacy_campaign_result_can_be_migrated_without_model_rerun(tmp_path) -> None:
    from cascade_planner.application.frontier_scheduler import PersistentFrontierQueue

    report = run_codex_retrosynthesis_campaign(
        case_id="legacy-migration-case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        config=RetrosynthesisTeamConfig(max_depth=1, max_expansions=1),
        runner=accepted_runner_record,
    )
    output_dir = tmp_path / "codex_retrosynthesis_team"
    queue = PersistentFrontierQueue(output_dir / "frontier_queue")
    job = queue.list_jobs("legacy-migration-case")[0]
    legacy_ref = str(output_dir / "team_report.json")
    queue.rebind_succeeded_result(
        "legacy-migration-case",
        job.job_id,
        expected_result_ref=job.result_ref,
        result_ref=legacy_ref,
    )
    state_path = output_dir / "campaign_state.json"
    legacy_state = json.loads(state_path.read_text(encoding="utf-8"))
    legacy_state.pop("content_sha256", None)
    state_path.write_text(json.dumps(legacy_state), encoding="utf-8")

    migrated = migrate_legacy_campaign_commits(
        case_id="legacy-migration-case",
        target_smiles="CCO",
        run_dir=tmp_path,
    )
    migrated_job = queue.list_jobs("legacy-migration-case")[0]
    migrated_state = json.loads(state_path.read_text(encoding="utf-8"))

    assert report["route_expansion_count"] == 1
    assert migrated["accepted"] is True
    assert migrated["migrated_job_ids"] == [job.job_id]
    assert "campaign_commits" in migrated_job.result_ref
    assert migrated_state["content_sha256"]


def test_team_rejects_completed_child_with_unstructured_or_wrong_role_report(tmp_path) -> None:
    def runner(task):
        children = []
        for index, role in enumerate(task.child_roles):
            message = child_report_message(task.case_id, role, with_candidate=index == 0)
            if index == 1:
                message = "not-json"
            if index == 2:
                message = child_report_message(task.case_id, "route_evidence_critic", with_candidate=False)
            children.append(
                {
                    "agent_id": f"child-{index}",
                    "role": role,
                    "role_binding_method": "explicit_spawn_contract",
                    "wait_call_id": f"wait-{index}",
                    "status": "completed",
                    "message": message,
                }
            )
        return WorkerRunRecord(
            run_id="team:run",
            task_id=task.task_id,
            case_id=task.case_id,
            status="accepted_draft",
            backend="codex_cli",
            output_artifact=proposal_artifact(task.case_id),
            output_validation={"accepted": True, "reasons": []},
            metadata={"child_agents": children},
        )

    report = run_codex_retrosynthesis_team(
        case_id="case",
        target_name="ethanol",
        target_smiles="CCO",
        run_dir=tmp_path,
        repository_root=tmp_path,
        runner=runner,
    )

    assert not report["accepted"]
    assert "required_child_reports_not_valid" in report["reasons"]
    rejected = [row for row in report["child_reports"] if not row["accepted"]]
    assert any("child_report_json_missing_or_invalid" in row["validation_reasons"] for row in rejected)
    assert any("child_report_role_mismatch" in row["validation_reasons"] for row in rejected)


def test_codex_jsonl_parser_captures_session_spawn_and_usage() -> None:
    text = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-42"}',
            '{"type":"item.completed","item":{"id":"call-1","type":"collab_tool_call","tool":"spawn_agent","sender_thread_id":"thread-42","receiver_thread_ids":["child-1"],"prompt":"scout","agents_states":{"child-1":{"status":"pending_init","message":null}},"status":"completed"}}',
            '{"type":"item.completed","item":{"id":"call-2","type":"collab_tool_call","tool":"wait","sender_thread_id":"thread-42","receiver_thread_ids":["child-1"],"prompt":null,"agents_states":{"child-1":{"status":"completed","message":"finding"}},"status":"completed"}}',
            '{"type":"turn.completed","usage":{"input_tokens":123,"output_tokens":45}}',
        ]
    )
    audit = _parse_codex_jsonl_events(text)
    assert audit["session_id"] == "thread-42"
    assert audit["summary"]["turn_completed"]
    assert audit["summary"]["child_agent_spawn_count"] == 1
    assert audit["summary"]["child_agent_completed_count"] == 1
    assert audit["tool_calls"][0]["tool"] == "spawn_agent"
    assert audit["child_agents"][0]["agent_id"] == "child-1"
    assert audit["child_agents"][0]["status"] == "completed"
    assert audit["child_agents"][0]["message"] == "finding"
    assert audit["usage"]["input_tokens"] == 123


def test_codex_jsonl_parser_does_not_treat_spawn_completion_as_child_success() -> None:
    audit = _parse_codex_jsonl_events(
        '{"type":"item.completed","item":{"id":"call-1","type":"collab_tool_call",'
        '"tool":"spawn_agent","receiver_thread_ids":["child-1"],'
        '"agents_states":{"child-1":{"status":"pending_init","message":null}},'
        '"status":"completed"}}'
    )
    assert audit["summary"]["child_agent_spawn_count"] == 1
    assert audit["summary"]["child_agent_completed_count"] == 0
    assert audit["child_agents"][0]["status"] == "pending_init"


def test_codex_jsonl_parser_ignores_wait_only_and_nested_children() -> None:
    text = "\n".join(
        [
            '{"type":"thread.started","thread_id":"root"}',
            '{"type":"item.completed","item":{"id":"root-spawn","type":"collab_tool_call","tool":"spawn_agent","sender_thread_id":"root","receiver_thread_ids":["child-1"],"prompt":"target structure strategist","agents_states":{"child-1":{"status":"pending_init"}},"status":"completed"}}',
            '{"type":"item.completed","item":{"id":"nested-spawn","type":"collab_tool_call","tool":"spawn_agent","sender_thread_id":"child-1","receiver_thread_ids":["grandchild-1"],"prompt":"helper","agents_states":{"grandchild-1":{"status":"pending_init"}},"status":"completed"}}',
            '{"type":"item.completed","item":{"id":"wait","type":"collab_tool_call","tool":"wait","sender_thread_id":"root","agents_states":{"child-1":{"status":"completed","message":"{}"},"ghost":{"status":"completed","message":"{}"}},"status":"completed"}}',
        ]
    )

    audit = _parse_codex_jsonl_events(text)

    assert [row["agent_id"] for row in audit["child_agents"]] == ["child-1"]
    assert audit["summary"]["child_agent_spawn_count"] == 1
    assert audit["summary"]["orphan_wait_state_count"] == 1


def test_child_roles_are_bound_to_spawn_prompts_not_event_order() -> None:
    children = [
        {"agent_id": "a", "prompt": "AUTOPLANNER_CHILD_ROLE=route_evidence_critic", "arguments": {}},
        {"agent_id": "b", "prompt": "AUTOPLANNER_CHILD_ROLE=target_structure_strategist", "arguments": {}},
    ]

    assigned = _assign_child_roles(
        children,
        roles=["target_structure_strategist", "route_evidence_critic"],
    )

    assert [row["role"] for row in assigned] == ["route_evidence_critic", "target_structure_strategist"]


def test_duplicate_role_prompts_do_not_fake_full_role_coverage() -> None:
    assigned = _assign_child_roles(
        [
            {"agent_id": "a", "prompt": "AUTOPLANNER_CHILD_ROLE=literature_route_scout", "arguments": {}},
            {"agent_id": "b", "prompt": "AUTOPLANNER_CHILD_ROLE=literature_route_scout", "arguments": {}},
        ],
        roles=["literature_route_scout", "route_evidence_critic"],
    )

    assert assigned[0]["role"] == "literature_route_scout"
    assert "role" not in assigned[1]


def test_child_report_parser_rejects_prose_multiple_objects_duplicate_keys_and_nan() -> None:
    valid = child_report_message("case", "target_structure_strategist", with_candidate=True)

    assert _child_report_payload(valid)["agent_role"] == "target_structure_strategist"
    assert _child_report_payload(f"prefix {valid}") == {}
    assert _child_report_payload(f"{valid}\n{valid}") == {}
    assert _child_report_payload('{"schema_version":"retrosynthesis_proposal_report.v1","schema_version":"retrosynthesis_proposal_report.v1"}') == {}
    assert _child_report_payload('{"schema_version":"retrosynthesis_proposal_report.v1","score":NaN}') == {}


def test_child_report_shape_repair_only_applies_conservative_advisory_defaults() -> None:
    payload = json.loads(
        child_report_message("case", "target_structure_strategist", with_candidate=True)
    )
    candidate = payload["candidates"][0]
    original_product = candidate["product_smiles"]
    original_precursors = list(candidate["precursor_smiles"])
    candidate.update(
        {
            "confidence": 0.88,
            "conditions": "ambient temperature",
            "catalyst": None,
            "enzyme": None,
        }
    )

    repaired, repairs = _conservative_child_report_shape_repair(payload)
    repaired_candidate = repaired["candidates"][0]

    assert repaired_candidate["confidence"] == "low"
    assert repaired_candidate["conditions"] == ["ambient temperature"]
    assert repaired_candidate["catalyst"] == ""
    assert repaired_candidate["enzyme"] == ""
    assert repaired_candidate["product_smiles"] == original_product
    assert repaired_candidate["precursor_smiles"] == original_precursors
    assert len(repairs) == 4
    assert _strict_child_report_shape_reasons(repaired) == []

    repaired_candidate["precursor_smiles"] = "CC"
    assert "child_candidate:0:precursor_smiles_not_string_list" in (
        _strict_child_report_shape_reasons(repaired)
    )

    for invalid_candidates in (1, True, 3.14):
        invalid_payload = {**payload, "candidates": invalid_candidates}
        unrepaired, _ = _conservative_child_report_shape_repair(invalid_payload)
        assert unrepaired["candidates"] == invalid_candidates
        assert "child_report_candidates_not_list" in _strict_child_report_shape_reasons(
            unrepaired
        )


def test_child_report_accepts_only_the_optional_fail_closed_parent_guard() -> None:
    payload = json.loads(
        child_report_message("case", "target_structure_strategist", with_candidate=True)
    )

    payload["not_parent_route_proof"] = True
    assert _strict_child_report_shape_reasons(payload) == []

    payload["not_parent_route_proof"] = False
    assert "child_report_parent_route_claim" in _strict_child_report_shape_reasons(
        payload
    )

    payload["not_parent_route_proof"] = True
    payload["unexpected_claim"] = True
    assert "child_report_fields_not_exact" in _strict_child_report_shape_reasons(
        payload
    )
