from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from cascade_planner.application.frontier_ledger import (
    project_frontier_ledger,
)
from cascade_planner.application.frontier_scheduler import (
    FrontierJob,
    FrontierJobState,
    PersistentFrontierQueue,
)
from cascade_planner.harness.agentic_blackboard import initialize_agent_blackboard
from cascade_planner.harness.blackboard_events import (
    append_blackboard_checkpoint,
    rehydrate_blackboard_from_events,
)
from cascade_planner.harness.agentic_blackboard_controller import (
    _blackboard_parent_proof_solved,
    _campaign_projection_closeout_reasons,
    _codex_team_projection_from_reconciliation,
    _consensus_refresh_max_depth,
)
from cascade_planner.harness.preflight import run_preflight
from cascade_planner.harness.schemas import TargetInput
from cascade_planner.harness.tools import ToolExecutionState
from cascade_planner.orchestration.codex_retrosynthesis import (
    CODEX_RETROSYNTHESIS_TEAM_SCHEMA,
    RetrosynthesisTeamConfig,
    _assemble_canonical_route_consensus_graph,
    _bind_campaign_projection,
    _campaign_expansion_case_id,
    _campaign_projection_binding,
    _payload_digest,
    _project_campaign_portfolio,
    _project_campaign_stock_replay,
    _reconcile_reaction_proof_state,
    _write_campaign_projection_bundle,
    _write_expansion_commit,
    validate_codex_campaign_projection_bundle,
)
from cascade_planner.routes.consensus import fuse_route_candidates
from cascade_planner.routes.graph import make_route_consensus_expansion


NOW = "2026-07-12T00:00:00.000000Z"


def test_campaign_policy_depth_wins_when_recovery_graph_is_absent() -> None:
    assert _consensus_refresh_max_depth(
        {},
        codex_campaign_config=RetrosynthesisTeamConfig(max_depth=6),
    ) == 6
    assert _consensus_refresh_max_depth(
        {"limits": {"max_depth": 4}},
        codex_campaign_config=None,
    ) == 4


def test_current_host_reconciliation_rebuilds_minimal_resumable_team() -> None:
    binding = {
        "schema_version": "codex_retrosynthesis_campaign_projection_binding.v1",
        "campaign_revision": 689,
        "accepted_expansion_count": 11,
    }
    reconciliation = {
        "accepted": True,
        "case_id": "nirmatrelvir",
        "target_smiles": "CCO",
        "campaign_identity_sha256": "a" * 64,
        "campaign_policy_sha256": "b" * 64,
        "campaign_policy_ref": "campaign_policy.json",
        "campaign_projection_binding": binding,
        "campaign_projection_bundle": {"accepted": True},
        "campaign_projection_bundle_ref": "campaign_projections/objects/bundle.json",
        "durable_accepted_expansion_count": 11,
        "frontier_queue": {
            "jobs": [
                {
                    "job_id": "frontier:pending",
                    "state": "pending",
                    "metadata": {"proposal_expansion_allowed": True},
                }
            ]
        },
        "queue_state_counts": {"pending": 1},
        "remaining_frontier": [{"target_smiles": "CC"}],
        "reaction_proof_state": {"records": []},
        "open_reaction_proofs": [{"step_id": "step:open"}],
        "frontier_completeness": {"complete": False},
        "campaign_search_complete": False,
        "route_solved": False,
    }
    team = _codex_team_projection_from_reconciliation(
        reconciliation,
        graph={"limits": {"max_depth": 6}, "steps": []},
        config=RetrosynthesisTeamConfig(
            max_depth=6,
            max_expansions=40,
            max_attempt_runs=120,
        ),
    )

    assert team["accepted"] is True
    assert team["accepted_semantics"] == (
        "current_host_campaign_authority_replay_only"
    )
    assert team["blackboard_proposals"] == []
    assert team["route_consensus_expansions"] == []
    assert team["proof_closed"] is False
    assert team["campaign"]["accepted_expansion_count"] == 11
    assert team["campaign"]["max_expansions"] == 40
    assert team["campaign"]["max_attempt_runs"] == 120
    assert team["campaign"]["max_depth"] == 6
    assert team["campaign"]["resumable"] is True
    assert team["semantics"]["mutable_team_report_not_used_for_recovery"] is True


def _candidate(product: str, precursors: list[str], candidate_id: str) -> dict:
    return {
        "schema_version": "retrosynthesis_candidate.v1",
        "candidate_id": candidate_id,
        "product_smiles": product,
        "precursor_smiles": precursors,
        "reaction_family": "fixture disconnection",
        "transformation_rationale": "interruption recovery fixture",
        "source_channel": "codex_strategy",
        "source_refs": [],
        "evidence_refs": [],
        "evidence_level": "model_only",
        "confidence": "medium",
        "conditions": [],
        "catalyst": "",
        "enzyme": "",
        "limitations": ["fixture only"],
        "required_validation": ["forward_reconstruction"],
        "no_solved_claim": True,
        "not_parent_route_proof": True,
    }


def _canonical_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_expired_lease_recovery_is_audited_and_idempotent(tmp_path: Path) -> None:
    queue = PersistentFrontierQueue(tmp_path / "queue")
    identity = "a" * 64
    policy = "b" * 64
    job = FrontierJob(
        run_id="v8-case",
        job_id="frontier:v8-interrupted",
        idempotency_key="v8-interrupted",
        frontier_smiles="CCO",
        frontier_node_id="molecule:target",
        max_attempts=3,
        created_at=NOW,
        updated_at=NOW,
        metadata={
            "campaign_identity_sha256": identity,
            "campaign_policy_sha256": policy,
            "campaign_root_smiles": "CCO",
            "depth": 0,
            "proposal_expansion_allowed": True,
        },
    )
    queue.enqueue(job)
    leased = queue.claim(
        "v8-case",
        worker_id="interrupted-agent",
        lease_seconds=1,
        now=NOW,
    )[0]

    recovered = queue.recover_expired(
        "v8-case",
        retry_base_seconds=0,
        now="2026-07-12T00:00:02.000000Z",
    )
    revision_after_first_recovery = queue.snapshot("v8-case")["revision"]
    duplicate = queue.recover_expired(
        "v8-case",
        retry_base_seconds=0,
        now="2026-07-12T00:00:03.000000Z",
    )

    assert len(recovered) == 1
    assert recovered[0].attempt == leased.attempt == 1
    assert recovered[0].state == FrontierJobState.RETRY_WAIT
    audit = recovered[0].metadata["lease_recovery_audit"]
    assert audit == [
        {
            "schema_version": "frontier_lease_recovery_audit.v1",
            "attempt": 1,
            "lease_owner": "interrupted-agent",
            "lease_token_sha256": hashlib.sha256(
                leased.lease_token.encode("utf-8")
            ).hexdigest(),
            "lease_expires_at": leased.lease_expires_at,
            "recovered_at": "2026-07-12T00:00:02.000000Z",
            "recovered_state": "retry_wait",
            "accepted_expansion_count_delta": 0,
        }
    ]
    assert duplicate == []
    assert queue.snapshot("v8-case")["revision"] == revision_after_first_recovery


def test_event_recovery_retains_only_current_campaign_revision_locator(
    tmp_path: Path,
) -> None:
    target = TargetInput(target_name="revision recovery", target_smiles="CCO")
    preflight = run_preflight(target)
    board = initialize_agent_blackboard(
        target_input=target.to_dict(),
        preflight=preflight,
        max_rounds=1,
    )
    binding = {
        "schema_version": "codex_retrosynthesis_campaign_projection_binding.v1",
        "campaign_revision": 149,
        "campaign_revision_sha256": "c" * 64,
        "campaign_identity_sha256": "a" * 64,
        "campaign_policy_sha256": "b" * 64,
        "frontier_queue_revision": 149,
        "frontier_queue_content_sha256": "d" * 64,
        "accepted_commit_count": 4,
        "accepted_expansion_count": 4,
        "admitted_hyperedge_event_count": 0,
    }
    board["campaign_projection_binding"] = binding
    board["codex_campaign_authority_projection"] = {
        "schema_version": "codex_campaign_authority_projection.v2",
        "accepted": True,
        "campaign_projection_binding": binding,
        "campaign_revision": 149,
        "campaign_revision_sha256": "c" * 64,
    }
    board["codex_agent_team"] = {"accepted": True, "solved": True}
    append_blackboard_checkpoint(tmp_path, board, stage="authority-recovered")

    fresh = initialize_agent_blackboard(
        target_input=target.to_dict(),
        preflight=preflight,
        max_rounds=1,
    )
    recovered, report = rehydrate_blackboard_from_events(fresh, run_dir=tmp_path)

    assert report["recovered"] is True
    assert recovered["campaign_projection_binding"] == binding
    assert recovered["codex_campaign_authority_projection"][
        "campaign_revision"
    ] == 149
    assert "codex_agent_team" not in recovered


def test_four_commit_interruption_rebuilds_one_revision_bundle_and_rejects_drift(
    tmp_path: Path,
) -> None:
    case_id = "v8-four-commit-case"
    target = "CCCCC"
    identity = "a" * 64
    policy = "b" * 64
    run_root = tmp_path / "run"
    authority_root = run_root / "codex_retrosynthesis_team"
    queue = PersistentFrontierQueue(authority_root / "frontier_queue")
    expansions: list[dict] = []
    pairs = [
        ("CCCCC", ["CCCC", "C"]),
        ("CCCC", ["CCC", "C"]),
        ("CCC", ["CC", "C"]),
        ("CC", ["C", "C"]),
    ]
    for index, (product, precursors) in enumerate(pairs, start=1):
        base_job = FrontierJob(
            run_id=case_id,
            job_id=f"frontier:commit-{index}",
            idempotency_key=f"commit-{index}",
            frontier_smiles=product,
            frontier_node_id=f"molecule:commit-{index}",
            max_attempts=3,
            created_at=NOW,
            updated_at=NOW,
            metadata={
                "campaign_identity_sha256": identity,
                "campaign_policy_sha256": policy,
                "campaign_root_smiles": target,
                "depth": 0 if index == 1 else 1,
                "parent_step_ids": [] if index == 1 else [f"step:parent-{index}"],
                "proposal_expansion_allowed": True,
            },
        )
        queue.enqueue(base_job)
        leased = queue.claim(
            case_id,
            worker_id="fixture-agent",
            lease_seconds=60,
            now=NOW,
        )[0]
        expansion_case_id = _campaign_expansion_case_id(case_id, leased)
        consensus = fuse_route_candidates(
            [_candidate(product, precursors, f"candidate-{index}")],
            case_id=expansion_case_id,
            target_smiles=product,
        )
        expansion = make_route_consensus_expansion(
            consensus,
            requested_product_smiles=product,
            depth=int(leased.metadata.get("depth") or 0),
        )
        report_path = run_root / f"team-report-{index}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": CODEX_RETROSYNTHESIS_TEAM_SCHEMA,
                    "accepted": True,
                    "case_id": expansion_case_id,
                    "target_smiles": product,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        summary = {
            "case_id": expansion_case_id,
            "target_smiles": product,
            "depth": int(leased.metadata.get("depth") or 0),
            "accepted": True,
            "team_report_accepted": True,
            "proposal_expansion_recorded": True,
            "frontier_job_id": leased.job_id,
        }
        commit_path = _write_expansion_commit(
            root_output_dir=authority_root,
            case_id=case_id,
            campaign_identity_sha256=identity,
            campaign_target_smiles=target,
            job=leased,
            team_report_ref=report_path,
            expansion=expansion,
            summary=summary,
        )
        queue.complete(
            case_id,
            leased.job_id,
            lease_token=leased.lease_token,
            result_ref=str(commit_path),
            closure_kind="proposal_expansion",
            achieved_proof_level=0,
            now=NOW,
        )
        expansions.append(expansion)

    queue_snapshot = queue.snapshot(case_id)
    binding = _campaign_projection_binding(
        root_output_dir=authority_root,
        case_id=case_id,
        campaign_identity_sha256=identity,
        campaign_policy_sha256=policy,
        campaign_target_smiles=target,
        queue_snapshot=queue_snapshot,
        committed_expansions=expansions,
        admitted_hyperedge_events=[],
    )
    graph = _assemble_canonical_route_consensus_graph(
        expansions,
        case_id=case_id,
        target_smiles=target,
        max_depth=4,
    )
    graph = _bind_campaign_projection(
        graph,
        binding,
        refresh_content_digest=False,
    )
    graph = _project_campaign_portfolio(
        graph,
        binding=binding,
            edge_verification_reports=[],
            stock_provider_results=[],
            trusted_stock_provider_instances={},
            reaction_proof_state={},
        )
    proof_state = _reconcile_reaction_proof_state(
        graph,
        path=authority_root / "reaction_proof_state.json",
        configured_proofs={},
        configured_reports=[],
        campaign_projection_binding=binding,
    )
    ledger = project_frontier_ledger(
        graph,
        queue_snapshot,
        proof_state,
        campaign_policy_sha256=policy,
        campaign_revision=binding["campaign_revision"],
        campaign_revision_sha256=binding["campaign_revision_sha256"],
    )
    stock_replay = _project_campaign_stock_replay(queue_snapshot, binding)
    bundle, bundle_path, _ = _write_campaign_projection_bundle(
        root_output_dir=authority_root,
        binding=binding,
        graph=graph,
        proof_state=proof_state,
        stock_replay=stock_replay,
        frontier_ledger=ledger,
    )
    duplicate, duplicate_path, _ = _write_campaign_projection_bundle(
        root_output_dir=authority_root,
        binding=binding,
        graph=graph,
        proof_state=proof_state,
        stock_replay=stock_replay,
        frontier_ledger=ledger,
    )

    assert binding["accepted_commit_count"] == 4
    assert binding["accepted_expansion_count"] == 4
    assert len(graph["steps"]) == 4
    assert ledger["input_bindings"]["frontier_queue_revision"] == queue_snapshot[
        "revision"
    ]
    assert validate_codex_campaign_projection_bundle(bundle) == []
    selected_projection = bundle["components"]["selected_route_parent_proof"]
    assert selected_projection["proof"]["benchmark_solved"] is False
    assert selected_projection["proof"]["distinct_complete_route_count"] == 0
    assert duplicate == bundle
    assert duplicate_path == bundle_path

    downgrade_attack_board = {
        "campaign_projection_bundle": bundle,
        "parent_route_proof": {
            "schema_version": "stitched_parent_route_proof.v1",
            "accepted": True,
            "solved": True,
            "route_status": "solved",
        },
    }
    assert _blackboard_parent_proof_solved(downgrade_attack_board) is False

    state = ToolExecutionState(
        run_dir=run_root,
        target_input={"target_smiles": target},
        preflight={"case_id": case_id},
    )
    state.artifacts.update(
        {
            "campaign_projection_bundle": bundle,
            "canonical_route_consensus_graph": graph,
            "frontier_ledger": ledger,
            "explored_route_forest": {
                "schema_version": "explored_route_forest.v1",
                "campaign_projection_binding": binding,
            },
        }
    )
    closeout_board = {
        "campaign_projection_binding": binding,
        "campaign_projection_bundle": bundle,
        "canonical_route_consensus_graph": graph,
        "frontier_ledger": ledger,
    }
    assert _campaign_projection_closeout_reasons(
        state=state,
        blackboard=closeout_board,
        binding=binding,
    ) == []

    state.artifacts["explored_route_forest"] = {
        "schema_version": "explored_route_forest.v1",
        "campaign_projection_binding": {
            **binding,
            "campaign_revision": binding["campaign_revision"] - 1,
        },
    }
    assert "campaign_projection_route_forest_revision_mismatch" in (
        _campaign_projection_closeout_reasons(
            state=state,
            blackboard=closeout_board,
            binding=binding,
        )
    )

    drifted = deepcopy(bundle)
    drifted_ledger = drifted["components"]["frontier_ledger"]
    drifted_ledger["input_bindings"]["campaign_revision"] -= 1
    digest_payload = dict(drifted_ledger)
    digest_payload.pop("content_sha256")
    drifted_ledger["content_sha256"] = _canonical_digest(digest_payload)
    drifted["component_sha256"]["frontier_ledger"] = _payload_digest(
        drifted_ledger
    )
    drifted_payload = dict(drifted)
    drifted_payload.pop("content_sha256")
    drifted["content_sha256"] = _canonical_digest(drifted_payload)

    assert "campaign_projection_component_revision_mismatch:frontier_ledger" in (
        validate_codex_campaign_projection_bundle(drifted)
    )

    selected_drift = deepcopy(bundle)
    selected_drift["components"]["selected_route_parent_proof"]["proof"][
        "benchmark_solved"
    ] = True
    selected_drift["component_sha256"]["selected_route_parent_proof"] = (
        _payload_digest(
            selected_drift["components"]["selected_route_parent_proof"]
        )
    )
    selected_drift_payload = dict(selected_drift)
    selected_drift_payload.pop("content_sha256")
    selected_drift["content_sha256"] = _canonical_digest(selected_drift_payload)
    assert (
        "selected_route_parent_proof_projection_full_recompile_mismatch"
        in validate_codex_campaign_projection_bundle(selected_drift)
    )
