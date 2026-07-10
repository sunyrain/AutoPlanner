from __future__ import annotations

import json

from cascade_planner.agent.codex_worker import (
    WorkerRunRecord,
    _assign_child_roles,
    _parse_codex_jsonl_events,
    _worker_output_json_schema,
)
from cascade_planner.orchestration.codex_retrosynthesis import (
    DEFAULT_CHILD_ROLES,
    _child_report_payload,
    build_retrosynthesis_coordinator_task,
    run_codex_retrosynthesis_team,
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
