from __future__ import annotations

import json
import time

from cascade_planner.agent.codex_worker import (
    WorkerRunRecord,
    _assign_child_roles,
    _parse_codex_jsonl_events,
    _worker_output_json_schema,
)
from cascade_planner.orchestration.codex_retrosynthesis import (
    DEFAULT_CHILD_ROLES,
    RetrosynthesisTeamConfig,
    _child_report_payload,
    _conservative_child_report_shape_repair,
    _strict_child_report_shape_reasons,
    build_retrosynthesis_coordinator_task,
    migrate_legacy_campaign_commits,
    run_codex_retrosynthesis_team,
    run_codex_retrosynthesis_campaign,
)


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
